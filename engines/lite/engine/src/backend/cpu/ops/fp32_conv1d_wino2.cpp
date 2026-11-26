// fp32_conv1d_wino2.cpp: blocked Winograd F(4,r), r = 3, 7 and 11, for stride-1 conv1d of any dilation (fp32, NEON and x86 AVX2+FMA).
//
// Why: the Kitten TTS 2 vocoder spends ~85% of its time in k = 3 / 7 / 11 convs with 128 / 256 channels at 800 and 4,800
// frames/s, and the phone CPUs are compute-bound there (one A725 core does 47 GFLOP/s on this kernel, ~peak). F(4,r)
// computes 4 outputs from alpha = r + 3 point products per channel pair: r = 11 needs 14/44 = 0.32 of the direct MACs
// (the F(2,11) it replaces: 12/22 = 0.55), r = 7 needs 10/28 = 0.36 (F(2,7): 0.57).
//
// Accuracy: the evaluation points are 0, +-1, +-2, +-1/2, +-3/2, +-2/3, +-3 (wino2_tables.hpp, from tools/gen_wino_tables.py), not the integers 0, +-1 ..
// +-5 of the old F(2,11): the whole-layer fp32 error is 3.9e-5 (F(4,11)) and 4.2e-6 (F(4,7)) relative, against 7.7e-3 for
// the old F(2,11) (one layer in float32 against a float64 direct conv). The filter transform is done in double and rounded once.
//
// Dilation d: output t = rho + d*q (rho = t mod d) reads input rows rho - pad + d*(q + k), so each residue class rho is an
// ordinary dilation-1 convolution over the rows x[rho - pad + d*j]: the tiles of one class read their alpha rows with
// stride d (rows outside the signal are the zero row), and write their m outputs with stride d.
//
// Blocking: a unit = (class, up to TB tiles). Its transformed input V (alpha x TB x C_in) and GEMM output M
// (alpha x TB x C_out) are thread-private scratch of about 0.5 MB, so they stay in L2; the alpha point GEMMs are the
// engine's own sgemm_f32 on that block. Units are handed out with schedule(dynamic, 1): every unit is computed whole by one
// thread with a fixed summation order, so the output never depends on the thread count or the core speeds, and a slow
// core simply takes fewer units.
#include "engine/dev_env.hpp"
#include "../ops_neon.hpp"
#include "wino2_tables.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))   // MSVC /arch:AVX2 defines __AVX2__ only (the build requires avx2 + fma)
#include <immintrin.h>
#define KT_WINO2_AVX2 1
#endif

namespace {

// Channels handled per vector step: 4 (NEON) or 8 (AVX2). The x86 code is the same arithmetic as the NEON code (the same fused
// multiply-adds in the same order, 8 channels at a time), so it is the same algorithm on both.
#if defined(KT_WINO2_AVX2)
constexpr int kVL = 8;
#else
constexpr int kVL = 4;
#endif

struct WinoTab {
    int m, r, a;
    const float* AT;   // [m][a]
    const float* BT;   // [a][a]
    const double* G;   // [a][r]
    // The evaluation points are symmetric (+x and -x): row -x of BT is row x with the odd columns negated, and column -x of AT is
    // column x with the odd rows negated. So a +-x pair costs one even-column sum and one odd-column sum for both transformed
    // points (V[x] = E + O, V[-x] = E - O), and both outputs' even/odd split in the output transform. Found from the tables.
    int npair = 0, pa[8] = {}, pb[8] = {};  // pair k = points pa[k], pb[k]
    int nsingle = 0, sg[4] = {};             // points with no partner (0 and infinity)
};

WinoTab make_tab(int m, int r, int a, const float* AT, const float* BT, const double* G) {
    WinoTab t{m, r, a, AT, BT, G};
    bool used[16] = {};
    for (int p = 0; p < a; ++p) {
        if (used[p]) continue;
        int q = -1;
        for (int c = p + 1; c < a && q < 0; ++c) {
            if (used[c]) continue;
            bool sym = true;
            for (int j = 0; j < a && sym; ++j) sym = BT[(size_t)c * a + j] == ((j & 1) ? -1.f : 1.f) * BT[(size_t)p * a + j];
            for (int i = 0; i < m && sym; ++i) sym = AT[(size_t)i * a + c] == ((i & 1) ? -1.f : 1.f) * AT[(size_t)i * a + p];
            if (sym) q = c;
        }
        used[p] = true;
        if (q >= 0) { used[q] = true; t.pa[t.npair] = p; t.pb[t.npair] = q; ++t.npair; }
        else t.sg[t.nsingle++] = p;
    }
    return t;
}

const WinoTab* wino_tab(int r) {
    static const WinoTab t3 = make_tab(kW43_M, kW43_R, kW43_A, &kW43_AT[0][0], &kW43_BT[0][0], &kW43_G[0][0]);
    static const WinoTab t7 = make_tab(kW47_M, kW47_R, kW47_A, &kW47_AT[0][0], &kW47_BT[0][0], &kW47_G[0][0]);
    static const WinoTab t11 = make_tab(kW411_M, kW411_R, kW411_A, &kW411_AT[0][0], &kW411_BT[0][0], &kW411_G[0][0]);
    return r == 3 ? &t3 : r == 7 ? &t7 : r == 11 ? &t11 : nullptr;
}

// KT08_WINO2_STATS=1 (diagnostics): total ns per phase over all threads, printed at exit.
struct Stats {
    bool on = false;
    std::atomic<long long> t_in{0}, t_gemm{0}, t_out{0}, units{0};
    Stats() { const char* e = kt_dev_getenv("KT08_WINO2_STATS"); on = e && e[0] == '1'; }
    ~Stats() { if (on) std::fprintf(stderr, "WINO2 units=%lld in=%.1f ms gemm=%.1f ms out=%.1f ms (summed over threads)\n", (long long)units.load(), t_in.load() * 1e-6, t_gemm.load() * 1e-6, t_out.load() * 1e-6); }
};
Stats g_stats;
inline long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int block_tiles_env(int dflt) { static const int v = [] { const char* e = kt_dev_getenv("KT08_WINO2_TB"); return e && *e ? std::atoi(e) : 0; }(); return v > 0 ? v : dflt; }

struct Scratch {
    std::vector<float> v, m, zero;
};

// Tiles per block: V and M together stay near 512 KB (alpha * TB * (C_in + C_out) * 4 bytes), a multiple of 8 (the GEMM's row tile).
int block_tiles(int a, int c_in, int c_out) {
    const int tb = (int)(((size_t)512 * 1024) / ((size_t)4 * a * (c_in + c_out)));
    return std::max(8, std::min(64, tb / 8 * 8));
}

}  // namespace

bool wino2_supported(int kernel_size, int c_in, int c_out) {
#if defined(__ARM_NEON) || defined(KT_WINO2_AVX2)
    return wino_tab(kernel_size) != nullptr && c_in % kVL == 0 && c_out % kVL == 0 && c_in >= 16 && c_out >= 16;
#else
    (void)kernel_size; (void)c_in; (void)c_out;
    return false;
#endif
}

int wino2_points(int kernel_size) {
    const WinoTab* t = wino_tab(kernel_size);
    return t ? t->a : 0;
}

// w: [C_out][r][C_in] (the engine's NHWC filter order). Returns alpha packed [Co_t, C_in, TILE] panels (pack_weights_f32).
void wino2_pack_weights(const float* w, int c_out, int c_in, int r, float** out_panels) {
    const WinoTab* t = wino_tab(r);
    // Each of the alpha panels is computed on its own (own scratch, own output), so they run on different threads at
    // load time: the same double-precision sums in the same order, only on more cores. KT08_LOAD_PAR=0 keeps one thread.
    static const bool par = [] { const char* e = kt_dev_getenv("KT08_LOAD_PAR"); return !(e && e[0] == '0'); }();
    const int a = t->a;
    bool go_par = false;
    int nthr = 1;
#ifdef _OPENMP
    go_par = par && !omp_in_parallel() && omp_get_max_threads() > 1 && a > 1;
    nthr = go_par ? std::min(a, omp_get_max_threads()) : 1;
#endif
    #pragma omp parallel num_threads(nthr) if(go_par)
    {
        std::vector<float> u((size_t)c_out * c_in);
        std::vector<double> acc(c_in);
        #pragma omp for schedule(dynamic, 1)
        for (int p = 0; p < a; ++p) {
            for (int co = 0; co < c_out; ++co) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int k = 0; k < r; ++k) {
                    const double g = t->G[(size_t)p * r + k];
                    const float* wk = w + ((size_t)co * r + k) * c_in;
                    for (int ci = 0; ci < c_in; ++ci) acc[ci] += g * (double)wk[ci];
                }
                for (int ci = 0; ci < c_in; ++ci) u[(size_t)co * c_in + ci] = (float)acc[ci];
            }
            out_panels[p] = pack_weights_f32(u.data(), c_out, c_in);
        }
    }
}

void conv1d_wino2_fp32(const float* x, const float* const* U, const float* bias, bool relu,
                       int T, int T_out, int c_in, int c_out, int pad, int dil, int r, float* y, const float* residual)
{
    const WinoTab* tb = wino_tab(r);
    const int m = tb->m, a = tb->a;
    const int TB = block_tiles_env(block_tiles(a, c_in, c_out));

    struct Unit { int rho, tile0, nt; };
    std::vector<Unit> units;
    for (int rho = 0; rho < dil && rho < T_out; ++rho) {
        const int tp = (T_out - rho + dil - 1) / dil;   // outputs of this class
        const int ntile = (tp + m - 1) / m;
        for (int t0 = 0; t0 < ntile; t0 += TB) units.push_back({rho, t0, std::min(TB, ntile - t0)});
    }
    const int n_units = (int)units.size();

    auto run_unit = [&](const Unit& un, Scratch& sc) {
        const size_t vsz = (size_t)a * TB * c_in, msz = (size_t)a * TB * c_out;
        if (sc.v.size() < vsz) sc.v.resize(vsz);
        if (sc.m.size() < msz) sc.m.resize(msz);
        if (sc.zero.size() < (size_t)c_in) sc.zero.assign(c_in, 0.f);
        float* V = sc.v.data();
        float* Mb = sc.m.data();
        const int nt = un.nt;
        const bool st = g_stats.on;
        long long t0 = st ? now_ns() : 0;

        // ── input transform: V[p][i][:] = sum_j BT[p][j] * row_j ──
        for (int i = 0; i < nt; ++i) {
            const int q0 = (un.tile0 + i) * m;
            const float* rows[14];
            for (int j = 0; j < a; ++j) {
                const long ti = (long)un.rho - pad + (long)dil * (q0 + j);
                rows[j] = (ti >= 0 && ti < T) ? x + (size_t)ti * c_in : sc.zero.data();
            }
            for (int cb = 0; cb < c_in; cb += kVL) {
#ifdef __ARM_NEON
                float32x4_t d[14];
                for (int j = 0; j < a; ++j) d[j] = vld1q_f32(rows[j] + cb);
                float* vo = V + (size_t)i * c_in + cb;               // point p at vo + p * TB * c_in
                const size_t ps = (size_t)TB * c_in;
                for (int k = 0; k < tb->nsingle; ++k) {              // points without a partner: dense row
                    const float* bt = tb->BT + (size_t)tb->sg[k] * a;
                    float32x4_t s = vmulq_n_f32(d[0], bt[0]);
                    for (int j = 1; j < a; ++j) s = vfmaq_n_f32(s, d[j], bt[j]);
                    vst1q_f32(vo + tb->sg[k] * ps, s);
                }
                for (int k = 0; k < tb->npair; ++k) {                // +-x pairs: V[x] = E + O, V[-x] = E - O
                    const float* bt = tb->BT + (size_t)tb->pa[k] * a;
                    float32x4_t e = vmulq_n_f32(d[0], bt[0]), o = vmulq_n_f32(d[1], bt[1]);
                    for (int j = 2; j < a; j += 2) e = vfmaq_n_f32(e, d[j], bt[j]);
                    for (int j = 3; j < a; j += 2) o = vfmaq_n_f32(o, d[j], bt[j]);
                    vst1q_f32(vo + tb->pa[k] * ps, vaddq_f32(e, o));
                    vst1q_f32(vo + tb->pb[k] * ps, vsubq_f32(e, o));
                }
#elif defined(KT_WINO2_AVX2)
                __m256 d[14];
                for (int j = 0; j < a; ++j) d[j] = _mm256_loadu_ps(rows[j] + cb);
                float* vo = V + (size_t)i * c_in + cb;               // point p at vo + p * TB * c_in
                const size_t ps = (size_t)TB * c_in;
                for (int k = 0; k < tb->nsingle; ++k) {              // points without a partner: dense row
                    const float* bt = tb->BT + (size_t)tb->sg[k] * a;
                    __m256 s = _mm256_mul_ps(d[0], _mm256_set1_ps(bt[0]));
                    for (int j = 1; j < a; ++j) s = _mm256_fmadd_ps(d[j], _mm256_set1_ps(bt[j]), s);
                    _mm256_storeu_ps(vo + tb->sg[k] * ps, s);
                }
                for (int k = 0; k < tb->npair; ++k) {                // +-x pairs: V[x] = E + O, V[-x] = E - O
                    const float* bt = tb->BT + (size_t)tb->pa[k] * a;
                    __m256 e = _mm256_mul_ps(d[0], _mm256_set1_ps(bt[0])), o = _mm256_mul_ps(d[1], _mm256_set1_ps(bt[1]));
                    for (int j = 2; j < a; j += 2) e = _mm256_fmadd_ps(d[j], _mm256_set1_ps(bt[j]), e);
                    for (int j = 3; j < a; j += 2) o = _mm256_fmadd_ps(d[j], _mm256_set1_ps(bt[j]), o);
                    _mm256_storeu_ps(vo + tb->pa[k] * ps, _mm256_add_ps(e, o));
                    _mm256_storeu_ps(vo + tb->pb[k] * ps, _mm256_sub_ps(e, o));
                }
#endif
            }
        }

        long long t1 = st ? now_ns() : 0;
        // ── alpha point GEMMs on the block (single-threaded: we are inside the unit loop's team) ──
        for (int p = 0; p < a; ++p)
            sgemm_f32(V + (size_t)p * TB * c_in, U[p], nullptr, Mb + (size_t)p * TB * c_out, false, nt, c_in, c_out, false);

        long long t2 = st ? now_ns() : 0;
        // ── output transform + bias + relu: y[rho + d*(q0 + mm)][:] = sum_p AT[mm][p] * M[p][i][:] ──
        for (int i = 0; i < nt; ++i) {
            const int q0 = (un.tile0 + i) * m;
            for (int cb = 0; cb < c_out; cb += kVL) {
#ifdef __ARM_NEON
                const float* mo = Mb + (size_t)i * c_out + cb;       // point p at mo + p * TB * c_out
                const size_t ps = (size_t)TB * c_out;
                float32x4_t sm[8], df[8], sgv[4];
                for (int k = 0; k < tb->npair; ++k) {
                    const float32x4_t x = vld1q_f32(mo + tb->pa[k] * ps), y = vld1q_f32(mo + tb->pb[k] * ps);
                    sm[k] = vaddq_f32(x, y); df[k] = vsubq_f32(x, y);
                }
                for (int k = 0; k < tb->nsingle; ++k) sgv[k] = vld1q_f32(mo + tb->sg[k] * ps);
                const float32x4_t bv = bias ? vld1q_f32(bias + cb) : vdupq_n_f32(0.f);
                for (int mm = 0; mm < m; ++mm) {
                    const long t = (long)un.rho + (long)dil * (q0 + mm);
                    if (t >= T_out) break;
                    const float* at = tb->AT + (size_t)mm * a;
                    const float32x4_t* pv = (mm & 1) ? df : sm;       // even outputs use x + y, odd ones x - y
                    float32x4_t s = vmulq_n_f32(sgv[0], at[tb->sg[0]]);
                    for (int k = 1; k < tb->nsingle; ++k) s = vfmaq_n_f32(s, sgv[k], at[tb->sg[k]]);
                    for (int k = 0; k < tb->npair; ++k) s = vfmaq_n_f32(s, pv[k], at[tb->pa[k]]);
                    s = vaddq_f32(s, bv);
                    if (relu) s = vmaxq_f32(s, vdupq_n_f32(0.f));
                    if (residual) s = vaddq_f32(s, vld1q_f32(residual + (size_t)t * c_out + cb));   // bias, relu, then + residual (as the K-split GEMM)
                    vst1q_f32(y + (size_t)t * c_out + cb, s);
                }
#elif defined(KT_WINO2_AVX2)
                const float* mo = Mb + (size_t)i * c_out + cb;       // point p at mo + p * TB * c_out
                const size_t ps = (size_t)TB * c_out;
                __m256 sm[8], df[8], sgv[4];
                for (int k = 0; k < tb->npair; ++k) {
                    const __m256 x = _mm256_loadu_ps(mo + tb->pa[k] * ps), y = _mm256_loadu_ps(mo + tb->pb[k] * ps);
                    sm[k] = _mm256_add_ps(x, y); df[k] = _mm256_sub_ps(x, y);
                }
                for (int k = 0; k < tb->nsingle; ++k) sgv[k] = _mm256_loadu_ps(mo + tb->sg[k] * ps);
                const __m256 bv = bias ? _mm256_loadu_ps(bias + cb) : _mm256_setzero_ps();
                for (int mm = 0; mm < m; ++mm) {
                    const long t = (long)un.rho + (long)dil * (q0 + mm);
                    if (t >= T_out) break;
                    const float* at = tb->AT + (size_t)mm * a;
                    const __m256* pv = (mm & 1) ? df : sm;           // even outputs use x + y, odd ones x - y
                    __m256 s = _mm256_mul_ps(sgv[0], _mm256_set1_ps(at[tb->sg[0]]));
                    for (int k = 1; k < tb->nsingle; ++k) s = _mm256_fmadd_ps(sgv[k], _mm256_set1_ps(at[tb->sg[k]]), s);
                    for (int k = 0; k < tb->npair; ++k) s = _mm256_fmadd_ps(pv[k], _mm256_set1_ps(at[tb->pa[k]]), s);
                    s = _mm256_add_ps(s, bv);
                    if (relu) s = _mm256_max_ps(s, _mm256_setzero_ps());
                    if (residual) s = _mm256_add_ps(s, _mm256_loadu_ps(residual + (size_t)t * c_out + cb));   // bias, relu, then + residual
                    _mm256_storeu_ps(y + (size_t)t * c_out + cb, s);
                }
#endif
            }
        }
        if (st) { const long long t3 = now_ns(); g_stats.t_in += t1 - t0; g_stats.t_gemm += t2 - t1; g_stats.t_out += t3 - t2; g_stats.units += 1; }
    };

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    if (nthreads > 1 && !omp_in_parallel() && n_units > 1) {
#pragma omp parallel num_threads(nthreads)
        {
            static thread_local Scratch sc;
#pragma omp for schedule(dynamic, 1)
            for (int u = 0; u < n_units; ++u) run_unit(units[u], sc);
        }
        return;
    }
#endif
    static thread_local Scratch sc_serial;
    for (int u = 0; u < n_units; ++u) run_unit(units[u], sc_serial);
}
