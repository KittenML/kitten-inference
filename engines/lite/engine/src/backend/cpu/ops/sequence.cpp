// sequence.cpp — Sequential / recurrent / strided compute kernels
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../ops_neon.hpp"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ─────────────────────────────────────────────────────────────────
// LSTM recurrence scan (FP32, NEON-vectorised)
//   proj: [T, 4*H]  pre-computed input projections
//   w_hh: [4*H, H]  hidden-to-hidden weights
// ─────────────────────────────────────────────────────────────────

void lstm_scan_fp32(
    const float* proj,
    const float* w_hh,
    const float* b_hh,
    const float* w_hh_packed,
    float*       h_out,
    int T, int H,
    bool reverse)
{
    std::vector<float> h(H, 0.f);
    std::vector<float> c(H, 0.f);
    std::vector<float> gates(4 * H);
    std::vector<float> h_gate(4 * H);

    auto step = [&](int t) {
        gemm_fp32_vec(h.data(), w_hh, b_hh, h_gate.data(), H, 4*H, w_hh_packed, nullptr);

        const float* pt = proj + (size_t)t * 4 * H;
        float* g_  = gates.data();
        float* hg  = h_gate.data();
#ifdef __ARM_NEON
        for (int j = 0; j + 4 <= 4*H; j += 4)
            vst1q_f32(g_ + j, vaddq_f32(vld1q_f32(pt + j), vld1q_f32(hg + j)));
        for (int j = (4*H) & ~3; j < 4*H; ++j) g_[j] = pt[j] + hg[j];
#elif defined(__AVX512F__)
        { int j = 0;
          for (; j + 16 <= 4*H; j += 16)
              _mm512_storeu_ps(g_+j, _mm512_add_ps(_mm512_loadu_ps(pt+j), _mm512_loadu_ps(hg+j)));
          for (; j < 4*H; ++j) g_[j] = pt[j] + hg[j]; }
#elif defined(__AVX2__)
        { int j = 0;
          for (; j + 8 <= 4*H; j += 8)
              _mm256_storeu_ps(g_+j, _mm256_add_ps(_mm256_loadu_ps(pt+j), _mm256_loadu_ps(hg+j)));
          for (; j < 4*H; ++j) g_[j] = pt[j] + hg[j]; }
#else
        for (int i = 0; i < 4*H; ++i) g_[i] = pt[i] + hg[i];
#endif

        float* cv = c.data();
        float* hv = h.data();
#ifdef __ARM_NEON
        int i = 0;
        for (; i + 4 <= H; i += 4) {
            float32x4_t gi = vsigmoidq_f32(vld1q_f32(g_        + i));
            float32x4_t gf = vsigmoidq_f32(vld1q_f32(g_ +   H  + i));
            float32x4_t gg = vtanhq_f32   (vld1q_f32(g_ + 2*H  + i));
            float32x4_t go = vsigmoidq_f32(vld1q_f32(g_ + 3*H  + i));
            float32x4_t cn = vfmaq_f32(vmulq_f32(gi, gg), gf, vld1q_f32(cv + i));
            vst1q_f32(cv + i, cn);
            vst1q_f32(hv + i, vmulq_f32(go, vtanhq_f32(cn)));
        }
        for (; i < H; ++i) {
            float gi = 1.f / (1.f + expf(-g_[i]));
            float gf = 1.f / (1.f + expf(-g_[H + i]));
            float gg = tanhf(g_[2*H + i]);
            float go = 1.f / (1.f + expf(-g_[3*H + i]));
            cv[i] = gf * cv[i] + gi * gg;
            hv[i] = go * tanhf(cv[i]);
        }
#elif defined(__AVX512F__)
        { int i = 0;
          for (; i + 16 <= H; i += 16) {
              __m512 gi = sigmoid512_ps(_mm512_loadu_ps(g_         + i));
              __m512 gf = sigmoid512_ps(_mm512_loadu_ps(g_ +     H + i));
              __m512 gg = tanh512_ps   (_mm512_loadu_ps(g_ + 2 * H + i));
              __m512 go = sigmoid512_ps(_mm512_loadu_ps(g_ + 3 * H + i));
              __m512 cn = _mm512_fmadd_ps(gi, gg, _mm512_mul_ps(gf, _mm512_loadu_ps(cv + i)));
              _mm512_storeu_ps(cv + i, cn);
              _mm512_storeu_ps(hv + i, _mm512_mul_ps(go, tanh512_ps(cn)));
          }
          for (; i < H; ++i) {
              float gi = 1.f / (1.f + expf(-g_[i]));
              float gf = 1.f / (1.f + expf(-g_[H + i]));
              float gg = tanhf(g_[2*H + i]);
              float go = 1.f / (1.f + expf(-g_[3*H + i]));
              cv[i] = gf * cv[i] + gi * gg;
              hv[i] = go * tanhf(cv[i]);
          } }
#elif defined(__AVX2__)
        { int i = 0;
          for (; i + 8 <= H; i += 8) {
              __m256 gi = sigmoid256_ps(_mm256_loadu_ps(g_         + i));
              __m256 gf = sigmoid256_ps(_mm256_loadu_ps(g_ +     H + i));
              __m256 gg = tanh256_ps   (_mm256_loadu_ps(g_ + 2 * H + i));
              __m256 go = sigmoid256_ps(_mm256_loadu_ps(g_ + 3 * H + i));
              __m256 cn = _mm256_fmadd_ps(gi, gg, _mm256_mul_ps(gf, _mm256_loadu_ps(cv + i)));
              _mm256_storeu_ps(cv + i, cn);
              _mm256_storeu_ps(hv + i, _mm256_mul_ps(go, tanh256_ps(cn)));
          }
          for (; i < H; ++i) {
              float gi = 1.f / (1.f + expf(-g_[i]));
              float gf = 1.f / (1.f + expf(-g_[H + i]));
              float gg = tanhf(g_[2*H + i]);
              float go = 1.f / (1.f + expf(-g_[3*H + i]));
              cv[i] = gf * cv[i] + gi * gg;
              hv[i] = go * tanhf(cv[i]);
          } }
#else
        for (int i = 0; i < H; ++i) {
            float gi = 1.f / (1.f + expf(-g_[i]));
            float gf = 1.f / (1.f + expf(-g_[H + i]));
            float gg = tanhf(g_[2*H + i]);
            float go = 1.f / (1.f + expf(-g_[3*H + i]));
            cv[i] = gf * cv[i] + gi * gg;
            hv[i] = go * tanhf(cv[i]);
        }
#endif
        memcpy(h_out + (size_t)t * H, hv, H * sizeof(float));
    };

    if (!reverse) {
        for (int t = 0; t < T; ++t) step(t);
    } else {
        for (int t = T - 1; t >= 0; --t) step(t);
    }
}

// ─────────────────────────────────────────────────────────────────
// ConvTranspose1D grouped scalar fallback
//   weight layout: [C_in, C_out/g, kernel_size]
// ─────────────────────────────────────────────────────────────────

void conv_transpose1d_grouped_fp32(
    const float* in,
    float*       out,
    const float* weight,
    int T, int T_out, int C_in, int C_out,
    int C_in_g, int C_out_g,
    int groups, int stride, int padding, int kernel_size)
{
    // gather form of the scatter loop (was serial, groups outermost: ~60 ms per long text at 16 threads on
    // a 16-core x86 server for the depthwise upsampling pools). Each output row t_out sums the contributions of its inputs t in
    // ascending t, each computed with the same expression (acc = sum over ci, then y += acc) as the scatter did, so
    // every element sees the same sequence of float operations; threads own disjoint output rows.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel() && (size_t)T_out * C_out >= 65536)
#endif
    for (int t_out = 0; t_out < T_out; ++t_out) {
        float* yrow = out + (size_t)t_out * C_out;
        const int num = t_out + padding;          // t * stride + k == num
        const int n0  = num - kernel_size + 1;    // smallest t * stride with k < kernel_size
        int t_lo = n0 <= 0 ? 0 : (n0 + stride - 1) / stride;
        int t_hi = num < 0 ? -1 : num / stride;
        if (t_hi > T - 1) t_hi = T - 1;
        for (int t = t_lo; t <= t_hi; ++t) {
            const int k = num - t * stride;
            for (int g = 0; g < groups; ++g) {
                const int ci_off = g * C_in_g;
                const int co_off = g * C_out_g;
                const float* x = in + (size_t)t * C_in + ci_off;
                float* y = yrow + co_off;
                for (int co = 0; co < C_out_g; ++co) {
                    float acc = 0.f;
                    for (int ci = 0; ci < C_in_g; ++ci)
                        acc += x[ci] * weight[((size_t)(ci_off+ci)*C_out_g + co)*kernel_size + k];
                    y[co] += acc;
                }
            }
        }
    }
}



// ─────────────────────────────────────────────────────────────────
// lstm_scan_pair_fp32: both directions, `per` threads each (see ops_neon.hpp)
//   Thread (dir, gid) owns hidden units [gid*H/per, (gid+1)*H/per) of direction dir. Per step it computes the four gate
//   slices with sgemm_f32 (M = 1, N = H/per, the same packed tiles and fma order as the whole-matrix call), adds the
//   input projection, updates its slice of c and h with the same vector code as lstm_scan_fp32, then waits at the
//   barrier of its direction. h is ping-ponged between two buffers, so one barrier per step is enough.
// ─────────────────────────────────────────────────────────────────
#if defined(__ARM_NEON) && defined(_OPENMP)
namespace {
struct alignas(64) KtStepBarrier {
    std::atomic<int> arrived{0};
    char pad0[60];
    std::atomic<int> gen{0};
    char pad1[60];
};
inline void kt_step_barrier(KtStepBarrier& b, int n) {
    const int g = b.gen.load(std::memory_order_acquire);
    if (b.arrived.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
        b.arrived.store(0, std::memory_order_relaxed);
        b.gen.store(g + 1, std::memory_order_release);
    } else {
        int spins = 0;
        while (b.gen.load(std::memory_order_acquire) == g) {
            if (++spins < 20000) __asm__ __volatile__("yield");
            else std::this_thread::yield();   // a preempted partner: do not burn the core
        }
    }
}
}  // namespace

bool lstm_scan_pair_fp32(
    const float* proj_f, const float* proj_r,
    const float* b_hh_f, const float* b_hh_r,
    const float* w_packed_f, const float* w_packed_r,
    float* h_f, float* h_r,
    int T, int H, int per)
{
    // slices must be whole, even-aligned pairs of 4-row weight tiles (the GEMM walks tiles in pairs from the slice start)
    if (omp_in_parallel() || per < 2 || H % (per * 8) != 0 || !w_packed_f || !w_packed_r || T <= 0) return false;
    const int nthr = 2 * per;
    std::vector<float> hb[2][2], cc[2];
    for (int d = 0; d < 2; ++d) { hb[d][0].assign(H, 0.f); hb[d][1].assign(H, 0.f); cc[d].assign(H, 0.f); }
    KtStepBarrier bar[2];
    #pragma omp parallel num_threads(nthr)
    {
        const int tid = omp_get_thread_num();
        if (omp_get_num_threads() != nthr) {
            // the runtime gave a smaller team: the plain scans on the first threads (or both on thread 0)
            if (tid == 0) lstm_scan_fp32(proj_f, nullptr, b_hh_f, w_packed_f, h_f, T, H, false);
            if (tid == 1 || (tid == 0 && omp_get_num_threads() == 1)) lstm_scan_fp32(proj_r, nullptr, b_hh_r, w_packed_r, h_r, T, H, true);
        } else {
            const int dir = tid / per, gid = tid % per;
            const float* proj = dir ? proj_r : proj_f;
            const float* bh = dir ? b_hh_r : b_hh_f;
            const float* wp = dir ? w_packed_r : w_packed_f;
            float* hout = dir ? h_r : h_f;
            float* c = cc[dir].data();
            const int slice = H / per, i0 = gid * slice;
            std::vector<float> gl((size_t)4 * slice);
            for (int s = 0; s < T; ++s) {
                const int t = dir ? T - 1 - s : s;
                const float* hp = hb[dir][s & 1].data();
                float* hn = hb[dir][(s + 1) & 1].data();
                for (int q = 0; q < 4; ++q)
                    sgemm_f32(hp, wp + (size_t)((q * H + i0) / 4) * H * 4, bh ? bh + q * H + i0 : nullptr, gl.data() + (size_t)q * slice,
                              /*relu=*/false, /*M=*/1, /*K=*/H, /*N=*/slice, /*in_parallel=*/false, nullptr);
                const float* pt = proj + (size_t)t * 4 * H + i0;
                for (int i = 0; i < slice; i += 4) {
                    float32x4_t gi = vsigmoidq_f32(vaddq_f32(vld1q_f32(pt + i),             vld1q_f32(gl.data() + i)));
                    float32x4_t gf = vsigmoidq_f32(vaddq_f32(vld1q_f32(pt + H + i),         vld1q_f32(gl.data() + slice + i)));
                    float32x4_t gg = vtanhq_f32   (vaddq_f32(vld1q_f32(pt + 2 * H + i),     vld1q_f32(gl.data() + 2 * slice + i)));
                    float32x4_t go = vsigmoidq_f32(vaddq_f32(vld1q_f32(pt + 3 * H + i),     vld1q_f32(gl.data() + 3 * slice + i)));
                    float32x4_t cn = vfmaq_f32(vmulq_f32(gi, gg), gf, vld1q_f32(c + i0 + i));
                    vst1q_f32(c + i0 + i, cn);
                    float32x4_t hv = vmulq_f32(go, vtanhq_f32(cn));
                    vst1q_f32(hn + i0 + i, hv);
                    vst1q_f32(hout + (size_t)t * H + i0 + i, hv);
                }
                kt_step_barrier(bar[dir], per);
            }
        }
    }
    return true;
}
#else
bool lstm_scan_pair_fp32(const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, int, int, int)
{
    return false;
}
#endif
