#include "engine/dev_env.hpp"
#include <type_traits>
#include "../ops_neon.hpp"
#include "engine/free_hint.hpp"
#include "profile_internal.hpp"
#include "work_split.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// ──────────────────────────────────────────────────────────────
// FP32 GEMM kernels — weight packing + SGEMM + gemm_fp32.
// Weight packing: [C_out, K] → [C_out_pad/TILE, K, TILE]
// ──────────────────────────────────────────────────────────────

// Tile = number of output channels per co_blk (= SIMD width for floats)
#if defined(__AVX512F__)
static constexpr int TILE = 16;
#elif defined(__AVX2__)
static constexpr int TILE = 8;
#else
static constexpr int TILE = 4;   // NEON / scalar
#endif

// Number of M-rows in main tile
static constexpr int MR = 8;

// ──────────────────────────────────────────────────────────────
// SSE2 tiles for x86-64 builds without AVX2/FMA (x86-64 baseline: Pentium/Celeron up to
// Comet Lake, Atom/Goldmont Celerons, pre-Haswell Core). Before this, such builds ran the scalar fallback loops below.
// TILE = 4 here (one __m128 per packed co_blk). A tile is 4 rows x 8 cols (two co_blks): 8 accumulators, 2 weight
// vectors, 1 broadcast and 1 product register = 12 of the 16 xmm registers. No FMA on these CPUs: each step is
// mulps then addps; per element the sum still runs kp ascending, k ascending, as in the scalar loop.
// KITTEN_FP32_SSE=0 falls back to the scalar loops (A/B switch).
// ──────────────────────────────────────────────────────────────
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__) && defined(__SSE2__)
#include <emmintrin.h>
#define KT_FP32_SSE 1
static bool fp32_sse_enabled() {
    static const bool on = [] { const char* e = kt_dev_getenv("KITTEN_FP32_SSE"); return !(e && e[0] == '0'); }();
    return on;
}
// Rows [m, m+4) x cols [n0, n0+8): B0 / B1 = packed panels of co_blk / co_blk+1 at kp = 0 (stride kp_stride floats
// per kp); A row r of tap kp = A_base + (m + r + kp*dil) * lda.
static inline __attribute__((always_inline)) void sse_rows4_cols8(
    const float* A_base, int lda, const float* B0, const float* B1, size_t kp_stride, int ks, int dil, int K,
    const float* bias, float* C, int ldc, int m, int n0, bool relu)
{
    __m128 c0a = _mm_setzero_ps(), c0b = _mm_setzero_ps(), c1a = _mm_setzero_ps(), c1b = _mm_setzero_ps();
    __m128 c2a = _mm_setzero_ps(), c2b = _mm_setzero_ps(), c3a = _mm_setzero_ps(), c3b = _mm_setzero_ps();
    for (int kp = 0; kp < ks; ++kp) {
        const float* b0 = B0 + (size_t)kp * kp_stride;
        const float* b1 = B1 + (size_t)kp * kp_stride;
        const float* a0 = A_base + (size_t)(m + 0 + kp * dil) * lda;
        const float* a1 = A_base + (size_t)(m + 1 + kp * dil) * lda;
        const float* a2 = A_base + (size_t)(m + 2 + kp * dil) * lda;
        const float* a3 = A_base + (size_t)(m + 3 + kp * dil) * lda;
#pragma GCC unroll 2
        for (int k = 0; k < K; ++k) {
            const __m128 vb0 = _mm_loadu_ps(b0 + (size_t)k * 4), vb1 = _mm_loadu_ps(b1 + (size_t)k * 4);
            __m128 x;
            x = _mm_set1_ps(a0[k]); c0a = _mm_add_ps(c0a, _mm_mul_ps(vb0, x)); c0b = _mm_add_ps(c0b, _mm_mul_ps(vb1, x));
            x = _mm_set1_ps(a1[k]); c1a = _mm_add_ps(c1a, _mm_mul_ps(vb0, x)); c1b = _mm_add_ps(c1b, _mm_mul_ps(vb1, x));
            x = _mm_set1_ps(a2[k]); c2a = _mm_add_ps(c2a, _mm_mul_ps(vb0, x)); c2b = _mm_add_ps(c2b, _mm_mul_ps(vb1, x));
            x = _mm_set1_ps(a3[k]); c3a = _mm_add_ps(c3a, _mm_mul_ps(vb0, x)); c3b = _mm_add_ps(c3b, _mm_mul_ps(vb1, x));
        }
    }
    const __m128 ba = bias ? _mm_loadu_ps(bias + n0) : _mm_setzero_ps();
    const __m128 bb = bias ? _mm_loadu_ps(bias + n0 + 4) : _mm_setzero_ps();
    const __m128 z = _mm_setzero_ps();
    auto st = [&](__m128 va, __m128 vb, int r) __attribute__((always_inline)) {
        va = _mm_add_ps(va, ba); vb = _mm_add_ps(vb, bb);
        if (relu) { va = _mm_max_ps(va, z); vb = _mm_max_ps(vb, z); }
        _mm_storeu_ps(C + (size_t)(m + r) * ldc + n0, va);
        _mm_storeu_ps(C + (size_t)(m + r) * ldc + n0 + 4, vb);
    };
    st(c0a, c0b, 0); st(c1a, c1b, 1); st(c2a, c2b, 2); st(c3a, c3b, 3);
}
// Rows [m, m+8) x cols [n0, n0+4) (one full co_blk): 8 accumulators.
static inline __attribute__((always_inline)) void sse_rows8_cols4(
    const float* A_base, int lda, const float* B0, size_t kp_stride, int ks, int dil, int K,
    const float* bias, float* C, int ldc, int m, int n0, bool relu)
{
    __m128 c0 = _mm_setzero_ps(), c1 = _mm_setzero_ps(), c2 = _mm_setzero_ps(), c3 = _mm_setzero_ps();
    __m128 c4 = _mm_setzero_ps(), c5 = _mm_setzero_ps(), c6 = _mm_setzero_ps(), c7 = _mm_setzero_ps();
    for (int kp = 0; kp < ks; ++kp) {
        const float* b0 = B0 + (size_t)kp * kp_stride;
        const float* a = A_base + (size_t)(m + kp * dil) * lda;
        for (int k = 0; k < K; ++k) {
            const __m128 vb = _mm_loadu_ps(b0 + (size_t)k * 4);
            c0 = _mm_add_ps(c0, _mm_mul_ps(vb, _mm_set1_ps(a[0 * (size_t)lda + k])));
            c1 = _mm_add_ps(c1, _mm_mul_ps(vb, _mm_set1_ps(a[1 * (size_t)lda + k])));
            c2 = _mm_add_ps(c2, _mm_mul_ps(vb, _mm_set1_ps(a[2 * (size_t)lda + k])));
            c3 = _mm_add_ps(c3, _mm_mul_ps(vb, _mm_set1_ps(a[3 * (size_t)lda + k])));
            c4 = _mm_add_ps(c4, _mm_mul_ps(vb, _mm_set1_ps(a[4 * (size_t)lda + k])));
            c5 = _mm_add_ps(c5, _mm_mul_ps(vb, _mm_set1_ps(a[5 * (size_t)lda + k])));
            c6 = _mm_add_ps(c6, _mm_mul_ps(vb, _mm_set1_ps(a[6 * (size_t)lda + k])));
            c7 = _mm_add_ps(c7, _mm_mul_ps(vb, _mm_set1_ps(a[7 * (size_t)lda + k])));
        }
    }
    const __m128 bv = bias ? _mm_loadu_ps(bias + n0) : _mm_setzero_ps();
    const __m128 z = _mm_setzero_ps();
    auto st = [&](__m128 v, int r) __attribute__((always_inline)) {
        v = _mm_add_ps(v, bv);
        if (relu) v = _mm_max_ps(v, z);
        _mm_storeu_ps(C + (size_t)(m + r) * ldc + n0, v);
    };
    st(c0, 0); st(c1, 1); st(c2, 2); st(c3, 3); st(c4, 4); st(c5, 5); st(c6, 6); st(c7, 7);
}
// Every full 8-row tile mi in [mi_s, mi_e) x co_blk in [co_s, co_e): MC-blocked (4 row tiles = 32 rows of A stay in
// cache while the weight panels stream), pairs of full co_blks as 4x8 tiles, a lone full co_blk as 8x4, and a
// partial last co_blk (C_out % 4 != 0) by the scalar loop. Returns nothing; the caller's tail-row code is unchanged.
static void sse_kfused_tiles(const float* A_base, int lda, const float* B, int ks, int dil, int K, int N,
                             const float* bias, float* C, bool relu, int mi_s, int mi_e, int co_s, int co_e)
{
    const size_t kp_stride = (size_t)K * 4;              // floats per (co_blk, kp) panel
    const size_t blk_stride = (size_t)ks * kp_stride;    // floats per co_blk
    int pair_end = co_s;
    while (pair_end + 1 < co_e && (pair_end + 2) * 4 <= N) pair_end += 2;
    constexpr int MC_TILES = 4;
    for (int mc = mi_s; mc < mi_e; mc += MC_TILES) {
        const int mc_end = std::min(mc + MC_TILES, mi_e);
        for (int cb = co_s; cb < pair_end; cb += 2) {
            const float* B0 = B + (size_t)cb * blk_stride;
            const float* B1 = B0 + blk_stride;
            for (int mi = mc; mi < mc_end; ++mi) {
                sse_rows4_cols8(A_base, lda, B0, B1, kp_stride, ks, dil, K, bias, C, N, mi * MR + 0, cb * 4, relu);
                sse_rows4_cols8(A_base, lda, B0, B1, kp_stride, ks, dil, K, bias, C, N, mi * MR + 4, cb * 4, relu);
            }
        }
        for (int cb = pair_end; cb < co_e; ++cb) {
            const int n0 = cb * 4, n_valid = std::min(4, N - n0);
            const float* B0 = B + (size_t)cb * blk_stride;
            for (int mi = mc; mi < mc_end; ++mi) {
                const int m = mi * MR;
                if (n_valid == 4) { sse_rows8_cols4(A_base, lda, B0, kp_stride, ks, dil, K, bias, C, N, m, n0, relu); continue; }
                float acc[MR][4] = {};
                for (int kp = 0; kp < ks; ++kp)
                    for (int r = 0; r < MR; ++r) {
                        const float* ar = A_base + (size_t)(m + r + kp * dil) * lda;
                        const float* bp = B0 + (size_t)kp * kp_stride;
                        for (int k = 0; k < K; ++k)
                            for (int oc = 0; oc < n_valid; ++oc) acc[r][oc] += ar[k] * bp[k * 4 + oc];
                    }
                for (int r = 0; r < MR; ++r)
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = acc[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        C[(size_t)(m + r) * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                    }
            }
        }
    }
}
#endif

// AVX2 K-fused conv runs each 8-row dual tile as two 4-row halves (no register spills); read once,
// KITTEN_FP32_AVX2_H4=0 turns it off. Values are the same either way.
[[maybe_unused]] static bool fp32_avx2_h4_enabled() {
    static const bool on = [] { const char* e = kt_dev_getenv("KITTEN_FP32_AVX2_H4"); return !(e && e[0] == '0'); }();
    return on;
}
// AVX2 K-fused conv / sgemm 4 x 24 triple tile (12 accumulators); read once. Default on (Zen 4 in AVX2
// mode -3 to -6%, Raptor Lake medium -9%), but off on Zen 2: on an EPYC 7K62 it ran 5-10%
// slower with it at 48 threads. KITTEN_FP32_AVX2_T3=0 / =1 forces it (only used with the 4-row halves). Values are the
// same either way.
[[maybe_unused]] static bool fp32_avx2_t3_enabled() {
    static const bool on = [] {
        const char* e = kt_dev_getenv("KITTEN_FP32_AVX2_T3");
        if (e && e[0] == '0') return false;
        if (e && e[0] == '1') return true;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        return __builtin_cpu_is("znver2") == 0;
#else
        return true;
#endif
    }();
    return on;
}
// sgemm one-row tail path with 4 column blocks in flight (LSTM recurrent mat-vec); read once,
// KITTEN_FP32_TAIL4=0 turns it off. Values are the same either way.
[[maybe_unused]] static bool fp32_tail4_enabled() {
    static const bool on = [] { const char* e = kt_dev_getenv("KITTEN_FP32_TAIL4"); return !(e && e[0] == '0'); }();
    return on;
}
// AVX-512 dual N-tile switch for the K-fused conv kernel, read once. Default: on for Intel CPUs (Xeon 6152
// Skylake-SP: -6% wall, bit-identical), off elsewhere (Zen 4 7945HX: no gain, so the single tile stays).
// KITTEN_FP32_NT2=1 / =0 forces it on / off. Values are the same either way.
[[maybe_unused]] static bool fp32_nt2_enabled() {
    static const bool on = [] {
        const char* e = kt_dev_getenv("KITTEN_FP32_NT2");
        if (e && e[0] == '0') return false;
        if (e && e[0] == '1') return true;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        return __builtin_cpu_is("intel") != 0;
#else
        return false;
#endif
    }();
    return on;
}


// ──────────────────────────────────────────────────────────────
// pack_weights_f32
//   w:     [C_out, K]  (K = C_in * kH * kW, NHWC-reordered)
//   →      [Co_t, K, TILE]  where Co_t = ceil(C_out / TILE)
// Zero-padded for C_out not a multiple of TILE.
// ──────────────────────────────────────────────────────────────
float* pack_weights_f32(const float* w, int C_out, int K)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    const size_t sz = (size_t)Co_t * K * TILE;
    float* packed = new float[sz]();   // zero-initialised
    for (int co_blk = 0; co_blk < Co_t; ++co_blk) {
        for (int k = 0; k < K; ++k) {
            for (int oc = 0; oc < TILE; ++oc) {
                const int co = co_blk * TILE + oc;
                if (co < C_out)
                    packed[(size_t)co_blk * K * TILE + k * TILE + oc] = w[co * K + k];
            }
        }
    }
    return packed;
}

void free_packed_f32(float* p)
{
    kitten_free_hint_block(p);  // Apple, low-memory modes only
    delete[] p;
}

size_t packed_f32_elems(int C_out, int K)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    return (size_t)Co_t * K * TILE;
}

// ──────────────────────────────────────────────────────────────
// pack_merged_weights_f32
//   w: [C_out, kernel_size, C_in_g]  (NHWC layout after reorder)
//   → B_merged: [Co_t, kernel_size, C_in_g, TILE]
//   Access: B_merged[(co_blk * kernel_size + kp) * C_in_g * TILE + k * TILE + oc]
// ──────────────────────────────────────────────────────────────
float* pack_merged_weights_f32(const float* w, int C_out, int C_in_g, int kernel_size)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    const size_t sz = (size_t)Co_t * kernel_size * C_in_g * TILE;
    float* packed = new float[sz]();
    for (int co_blk = 0; co_blk < Co_t; ++co_blk) {
        for (int kp = 0; kp < kernel_size; ++kp) {
            float* dst = packed + ((size_t)co_blk * kernel_size + kp) * C_in_g * TILE;
            for (int k = 0; k < C_in_g; ++k) {
                for (int oc = 0; oc < TILE; ++oc) {
                    const int co = co_blk * TILE + oc;
                    if (co < C_out)
                        dst[k * TILE + oc] = w[(size_t)co * kernel_size * C_in_g + (size_t)kp * C_in_g + k];
                }
            }
        }
    }
    return packed;
}

// ──────────────────────────────────────────────────────────────
// sgemm_f32
//   A:        [M, K]          row-major float
//   B_packed: [Co_t, K, TILE] packed floats
//   bias:     [N]  or nullptr
//   C:        [M, N]          row-major float (output)
//   relu:     apply max(0, ·) after bias add
//
// Recursive OMP dispatch: call with in_parallel=false from outside
// a parallel region; the function forks a team and calls itself
// with in_parallel=true inside.  Inside the team each thread
// independently computes its share of output tiles.
//
// Hybrid M/N partition (mirrors INT8 gemm_int8_neon):
//   M >= N  →  M-partition: thread tid owns m-tiles [mi_s, mi_e)
//   M <  N  →  N-partition: thread tid owns co_blks [co_s, co_e)
// ──────────────────────────────────────────────────────────────
#if defined(__APPLE__) && defined(__ARM_NEON) && defined(KITTEN_HAVE_ACCELERATE)
// route large sgemm_f32 calls to Accelerate's cblas_sgemm (fp32 in, fp32 accumulate; on Apple silicon
// Accelerate runs SGEMM on the matrix unit). Off unless KITTEN_ACCEL_SGEMM=1. Thread count is set explicitly
// (BLASSetThreading) from the OpenMP thread count, so "-t 1" means one CPU thread driving the unit.
// KITTEN_ACCEL_LEGACY_CBLAS=1 for deployment targets below iOS 16.4 / macOS 13.3, where the new LAPACK
// interface (ACCELERATE_NEW_LAPACK) is unavailable: the legacy cblas_sgemm is the same SGEMM entry point with int dims.
// BLASSetThreading does not exist there; Accelerate's own threading is then capped by VECLIB_MAXIMUM_THREADS (env).
#if !defined(KITTEN_ACCEL_LEGACY_CBLAS)
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include "../arm_isa.hpp"
static size_t kitten_accel_sgemm_min() {
    static const size_t v = [] {
        const char* e = std::getenv("KITTEN_ACCEL_SGEMM");
#if defined(KITTEN_ACCEL_SGEMM_DEFAULT_ON)
        // iOS builds (CMake KITTEN_ACCELERATE_DEFAULT=ON) use the route by default on Apple A13 and newer, whose
        // Accelerate runs SGEMM on the matrix unit (iPhone SE 2020, mini, 2 threads: RTF 0.135 vs 0.268 without).
        // A13 is the first with the dot-product instructions, so that flag tells them apart. On older chips
        // Accelerate is slower than the NEON kernels (iPhone 7: 0.524 vs 0.424), so they keep NEON.
        // KITTEN_ACCEL_SGEMM=1 / 0 forces it on / off.
        if (e && e[0] == '0') return (size_t)0;
        if ((!e || !e[0]) && kt_arm_isa_hw() < KT_ARM_DOTPROD) return (size_t)0;
#else
        if (!e || !e[0] || e[0] == '0') return (size_t)0;
#endif
        const char* m = kt_dev_getenv("KITTEN_ACCEL_SGEMM_MIN");          // min M*K*N for the Accelerate route
        return (size_t)((m && m[0]) ? std::max(1L, std::atol(m)) : 262144L);
    }();
    return v;
}
// Probes: KITTEN_ACCEL_MT=0 keeps Accelerate single-threaded at any -t; KITTEN_ACCEL_MBLOCK=R splits M into R-row
// GEMMs (bounds Accelerate's internal A packing for long chunks). Defaults: MT follows -t, no blocking.
static int kitten_accel_mblock() {
    static const int v = [] { const char* e = kt_dev_getenv("KITTEN_ACCEL_MBLOCK"); return (e && e[0]) ? std::max(0, std::atoi(e)) : 0; }();
    return v;
}
static void sgemm_f32_accel_threads()
{
    static int threading_set = -1;
    static const bool allow_mt = [] { const char* e = kt_dev_getenv("KITTEN_ACCEL_MT"); return !(e && e[0] == '0'); }();
    const int want = (allow_mt && omp_get_max_threads() > 1) ? 1 : 0;
    if (threading_set != want) {
#if !defined(KITTEN_ACCEL_LEGACY_CBLAS) && \
    ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
     (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000))
        // BLASSetThreading exists from the macOS 15 / iOS 18 SDKs on; older SDKs keep Accelerate's own threading.
        if (__builtin_available(macOS 15.0, iOS 18.0, *))
            BLASSetThreading(want ? BLAS_THREADING_MULTI_THREADED : BLAS_THREADING_SINGLE_THREADED);
#endif
        threading_set = want;
    }
}
// panel -> row-major unpack, blocked over 8 k-rows so each destination cache line (16 floats = 4 panels) is
// filled completely while it is hot. src = [Co_t, Krows, 4] panels with panel stride pstride floats; dst = [Krows, N].
static void accel_unpack_panels(const float* src, size_t pstride, int Krows, int N, float* dst)
{
    const int Co_t = (N + 3) / 4, full = N / 4;
    for (int k0 = 0; k0 < Krows; k0 += 8) {
        const int k1 = std::min(Krows, k0 + 8);
        for (int cb = 0; cb < full; ++cb) {
            const float* sp = src + (size_t)cb * pstride;
            for (int k = k0; k < k1; ++k) vst1q_f32(dst + (size_t)k * N + cb * 4, vld1q_f32(sp + (size_t)k * 4));
        }
        for (int cb = full; cb < Co_t; ++cb) {
            const float* sp = src + (size_t)cb * pstride; const int nv = N - cb * 4;
            for (int k = k0; k < k1; ++k) for (int j = 0; j < nv; ++j) dst[(size_t)k * N + cb * 4 + j] = sp[(size_t)k * 4 + j];
        }
    }
}
static void sgemm_f32_accel(const float* A, const float* B_packed, const float* bias, float* C, bool relu,
                            int M, int K, int N)
{
    sgemm_f32_accel_threads();
    // B_packed is [Co_t, K, 4] (NEON TILE = 4): unpack to row-major [K, N] once per call (K*N, << M*K*N work).
    thread_local std::vector<float> b;
    if (b.size() < (size_t)K * N) b.resize((size_t)K * N);
    accel_unpack_panels(B_packed, (size_t)K * 4, K, N, b.data());
    const int mb = kitten_accel_mblock() > 0 ? kitten_accel_mblock() : M;
    for (int m0 = 0; m0 < M; m0 += mb)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, std::min(mb, M - m0), N, K, 1.0f, A + (size_t)m0 * K, K,
                    b.data(), N, 0.0f, C + (size_t)m0 * N, N);
    if (bias || relu) {
        for (int m = 0; m < M; ++m) {
            float* c = C + (size_t)m * N;
            for (int n = 0; n < N; ++n) { float v = c[n] + (bias ? bias[n] : 0.f); c[n] = relu ? std::max(v, 0.f) : v; }
        }
    }
}
// True when large sgemm_f32 calls go to Accelerate's matrix unit (iOS on A13 and newer, or KITTEN_ACCEL_SGEMM=1).
bool kitten_accel_sgemm_active() { return kitten_accel_sgemm_min() > 0; }
#else
bool kitten_accel_sgemm_active() { return false; }
#endif

#if defined(__ARM_NEON)
// (In-order Cortex-A53/A55 class cores, harmless on out-of-order ones): the 8x8 NEON tile reads eight A rows (row stride K*4 bytes,
// 8 KiB for K = 2048: all eight rows hit the same L1 set) and a weight panel; the measured result on a Kryo silver (A53-class) core was
// 2.6-3.7 GFLOP/s for the BERT GEMMs against 8.9 for the same tile in cache. Fix: for each block of up to 8 row tiles and each K slab of
// up to 512, copy the A slab once to a contiguous [k/4][row][4] layout (one sequential stream, no set aliasing), then run every weight
// panel pair and tile over it, with software prefetch. Every output element still gets the same fused fma chain k = 0..K-1 ascending
// (partial sums go through C as exact fp32), then +bias, then relu: byte-identical to the unblocked kernel.
static bool sgemm_packa_enabled() {
    static const bool on = [] { const char* e = kt_dev_getenv("KITTEN_SGEMM_PACKA"); return !(e && e[0] == '0'); }();
    return on;
}
static inline void tile8x8_packed(const float* ap, const float* bp0, const float* bp1, int kc, float* C, int N, bool first, bool last,
                                  const float* bias, bool relu)
{
    float32x4_t c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5a, c5b, c6a, c6b, c7a, c7b;
    if (first) {
        const float32x4_t z = vdupq_n_f32(0.f);
        c0a = c0b = c1a = c1b = c2a = c2b = c3a = c3b = c4a = c4b = c5a = c5b = c6a = c6b = c7a = c7b = z;
    } else {
        c0a = vld1q_f32(C + 0 * N); c0b = vld1q_f32(C + 0 * N + 4); c1a = vld1q_f32(C + 1 * N); c1b = vld1q_f32(C + 1 * N + 4);
        c2a = vld1q_f32(C + 2 * N); c2b = vld1q_f32(C + 2 * N + 4); c3a = vld1q_f32(C + 3 * N); c3b = vld1q_f32(C + 3 * N + 4);
        c4a = vld1q_f32(C + 4 * N); c4b = vld1q_f32(C + 4 * N + 4); c5a = vld1q_f32(C + 5 * N); c5b = vld1q_f32(C + 5 * N + 4);
        c6a = vld1q_f32(C + 6 * N); c6b = vld1q_f32(C + 6 * N + 4); c7a = vld1q_f32(C + 7 * N); c7b = vld1q_f32(C + 7 * N + 4);
    }
    for (int k = 0; k < kc; k += 4, ap += 32) {
        __builtin_prefetch(ap + 128, 0, 3); __builtin_prefetch(ap + 144, 0, 3);
        __builtin_prefetch(bp0 + k * 4 + 64, 0, 3); __builtin_prefetch(bp1 + k * 4 + 64, 0, 3);
        const float32x4_t ba0 = vld1q_f32(bp0 + k * 4 + 0), ba1 = vld1q_f32(bp0 + k * 4 + 4), ba2 = vld1q_f32(bp0 + k * 4 + 8), ba3 = vld1q_f32(bp0 + k * 4 + 12);
        const float32x4_t bb0 = vld1q_f32(bp1 + k * 4 + 0), bb1 = vld1q_f32(bp1 + k * 4 + 4), bb2 = vld1q_f32(bp1 + k * 4 + 8), bb3 = vld1q_f32(bp1 + k * 4 + 12);
        const float32x4_t av0 = vld1q_f32(ap + 0), av1 = vld1q_f32(ap + 4), av2 = vld1q_f32(ap + 8), av3 = vld1q_f32(ap + 12);
        const float32x4_t av4 = vld1q_f32(ap + 16), av5 = vld1q_f32(ap + 20), av6 = vld1q_f32(ap + 24), av7 = vld1q_f32(ap + 28);
#define KT_PK_ROW(r, A) \
        c##r##a = vfmaq_laneq_f32(c##r##a, ba0, A, 0); c##r##b = vfmaq_laneq_f32(c##r##b, bb0, A, 0); \
        c##r##a = vfmaq_laneq_f32(c##r##a, ba1, A, 1); c##r##b = vfmaq_laneq_f32(c##r##b, bb1, A, 1); \
        c##r##a = vfmaq_laneq_f32(c##r##a, ba2, A, 2); c##r##b = vfmaq_laneq_f32(c##r##b, bb2, A, 2); \
        c##r##a = vfmaq_laneq_f32(c##r##a, ba3, A, 3); c##r##b = vfmaq_laneq_f32(c##r##b, bb3, A, 3);
        KT_PK_ROW(0, av0) KT_PK_ROW(1, av1) KT_PK_ROW(2, av2) KT_PK_ROW(3, av3) KT_PK_ROW(4, av4) KT_PK_ROW(5, av5) KT_PK_ROW(6, av6) KT_PK_ROW(7, av7)
#undef KT_PK_ROW
    }
    if (last) {
        const float32x4_t bva = bias ? vld1q_f32(bias) : vdupq_n_f32(0.f), bvb = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.f);
        const float32x4_t zero_v = vdupq_n_f32(0.f);
#define KT_PK_FIN(r) \
        c##r##a = vaddq_f32(c##r##a, bva); c##r##b = vaddq_f32(c##r##b, bvb); \
        if (relu) { c##r##a = vmaxq_f32(c##r##a, zero_v); c##r##b = vmaxq_f32(c##r##b, zero_v); }
        KT_PK_FIN(0) KT_PK_FIN(1) KT_PK_FIN(2) KT_PK_FIN(3) KT_PK_FIN(4) KT_PK_FIN(5) KT_PK_FIN(6) KT_PK_FIN(7)
#undef KT_PK_FIN
    }
    vst1q_f32(C + 0 * N, c0a); vst1q_f32(C + 0 * N + 4, c0b); vst1q_f32(C + 1 * N, c1a); vst1q_f32(C + 1 * N + 4, c1b);
    vst1q_f32(C + 2 * N, c2a); vst1q_f32(C + 2 * N + 4, c2b); vst1q_f32(C + 3 * N, c3a); vst1q_f32(C + 3 * N + 4, c3b);
    vst1q_f32(C + 4 * N, c4a); vst1q_f32(C + 4 * N + 4, c4b); vst1q_f32(C + 5 * N, c5a); vst1q_f32(C + 5 * N + 4, c5b);
    vst1q_f32(C + 6 * N, c6a); vst1q_f32(C + 6 * N + 4, c6b); vst1q_f32(C + 7 * N, c7a); vst1q_f32(C + 7 * N + 4, c7b);
}

// Dual column panels co_s..dual_end (8 columns each) for row tiles mi_s..mi_e, K % 4 == 0.
static void sgemm_dual_packed_neon(const float* A, const float* B_packed, const float* bias, float* C, bool relu,
                                   int K, int N, int mi_s, int mi_e, int co_s, int dual_end)
{
    constexpr int KCB = 512, MCT = 8;
    thread_local std::vector<float> abuf;
    if (abuf.size() < (size_t)MCT * 8 * KCB + 16) abuf.resize((size_t)MCT * 8 * KCB + 16);
    float* buf = abuf.data();
    for (int mc0 = mi_s; mc0 < mi_e; mc0 += MCT) {
        const int mc1 = std::min(mc0 + MCT, mi_e);
        for (int k0 = 0; k0 < K; k0 += KCB) {
            const int k1 = std::min(k0 + KCB, K), kc = k1 - k0;
            for (int t = mc0; t < mc1; ++t) {
                float* d = buf + (size_t)(t - mc0) * 8 * KCB;
                const float* src = A + (size_t)t * MR * K + k0;
                for (int kk = 0; kk < kc; kk += 4)
                    for (int r = 0; r < 8; ++r) vst1q_f32(d + kk * 8 + r * 4, vld1q_f32(src + (size_t)r * K + kk));
            }
            const bool first = (k0 == 0), last = (k1 == K);
            for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
                const float* bp0 = B_packed + ((size_t)co_blk * K + k0) * 4;
                const float* bp1 = B_packed + ((size_t)(co_blk + 1) * K + k0) * 4;
                const int n0 = co_blk * 4;
                for (int t = mc0; t < mc1; ++t)
                    tile8x8_packed(buf + (size_t)(t - mc0) * 8 * KCB, bp0, bp1, kc, C + (size_t)t * MR * N + n0, N, first, last,
                                   bias ? bias + n0 : nullptr, relu);
            }
        }
    }
}
#endif

void sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool in_parallel,
    StreamHandle /* stream */)
{
    const bool _profile_sgemm = g_ops_profile_enabled && !in_parallel;
    const double _t0_sgemm = _profile_sgemm ? now_ms() : 0.0;
#if defined(__APPLE__) && defined(__ARM_NEON) && defined(KITTEN_HAVE_ACCELERATE)
    if (!in_parallel && !omp_in_parallel()) {
        const size_t amin = kitten_accel_sgemm_min();
        if (amin && M >= 16 && (size_t)M * K * N >= amin) {   // M >= 16 keeps LSTM steps (M=1) on NEON
            sgemm_f32_accel(A, B_packed, bias, C, relu, M, K, N);
            if (_profile_sgemm) g_sgemm_total_ms += now_ms() - _t0_sgemm;
            return;
        }
    }
#endif

    if (!in_parallel) {
#ifdef _OPENMP
        // Only fork a new team if NOT already inside a parallel region.
        // When called inside an existing parallel region (e.g. from the Winograd
        // pos-parallel loop), we compute the full GEMM single-threaded on the
        // calling thread (tid=0, nT=1 override below).
        //
        // Also skip forking for tiny GEMMs (M=1 LSTM scan steps): fork overhead
        // (~5-10 µs) dominates over the compute benefit for < 2M FLOPs.
        if (!omp_in_parallel()) {
            const int nthreads = omp_get_max_threads();
            if (nthreads > 1 && (size_t)M * K * N >= 2000000UL) {
#pragma omp parallel num_threads(nthreads)
                sgemm_f32(A, B_packed, bias, C, relu, M, K, N, /*in_parallel=*/true, nullptr);
                if (_profile_sgemm) g_sgemm_total_ms += now_ms() - _t0_sgemm;
                return;
            }
        }
        // Inside a parallel region or single-threaded → run as tid=0, nT=1
#endif
    }

#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0, nT = 1;
#endif

    const int Co_t    = (N + TILE - 1) / TILE;
    const int m_tiles = M / MR;

    const bool n_part = (nT > 1) && (M < N);
    kt_split_tiles(n_part, m_tiles, Co_t, tid, nT, in_parallel,
                   [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {

#ifdef __ARM_NEON
    // ── NEON path: mi-outer, co_blk-inner (MR=8 rows) ────────────────────
    // column panels outer, row tiles inner. The packed weight panel (K x 8 floats) is read from DRAM once and re-read from
    // cache by every row tile; with the row tiles outer (as before) the whole weight matrix streamed from DRAM once per row tile
    // (M = 66 tokens -> 8x), which made the BERT GEMMs memory-bound on the in-order A53 cores. Same per-element fma order: identical bytes.
    int dual_end_ = co_s;
    while (dual_end_ + 1 < co_e && (dual_end_ + 2) * TILE <= N) dual_end_ += 2;
    const bool packed_dual = sgemm_packa_enabled() && (K % 4 == 0) && K >= 8 && dual_end_ > co_s && mi_e > mi_s;
    if (packed_dual) sgemm_dual_packed_neon(A, B_packed, bias, C, relu, K, N, mi_s, mi_e, co_s, dual_end_);
    for (int co_blk = co_s; !packed_dual && co_blk < dual_end_; co_blk += 2) {
        for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * MR;
        const float* a0 = A + (size_t)(m+0) * K;
        const float* a1 = A + (size_t)(m+1) * K;
        const float* a2 = A + (size_t)(m+2) * K;
        const float* a3 = A + (size_t)(m+3) * K;
        const float* a4 = A + (size_t)(m+4) * K;
        const float* a5 = A + (size_t)(m+5) * K;
        const float* a6 = A + (size_t)(m+6) * K;
        const float* a7 = A + (size_t)(m+7) * K;
        {
            const float32x4_t zero_v = vdupq_n_f32(0.f);
            {
                const float* bp0 = B_packed + (size_t)co_blk * K * 4;
                const float* bp1 = B_packed + (size_t)(co_blk + 1) * K * 4;
                const int n0a = co_blk * 4, n0b = (co_blk + 1) * 4;

                float32x4_t acc0a=vdupq_n_f32(0), acc0b=vdupq_n_f32(0);
                float32x4_t acc1a=vdupq_n_f32(0), acc1b=vdupq_n_f32(0);
                float32x4_t acc2a=vdupq_n_f32(0), acc2b=vdupq_n_f32(0);
                float32x4_t acc3a=vdupq_n_f32(0), acc3b=vdupq_n_f32(0);
                float32x4_t acc4a=vdupq_n_f32(0), acc4b=vdupq_n_f32(0);
                float32x4_t acc5a=vdupq_n_f32(0), acc5b=vdupq_n_f32(0);
                float32x4_t acc6a=vdupq_n_f32(0), acc6b=vdupq_n_f32(0);
                float32x4_t acc7a=vdupq_n_f32(0), acc7b=vdupq_n_f32(0);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    float32x4_t ba0 = vld1q_f32(bp0 + k*4 +  0);
                    float32x4_t ba1 = vld1q_f32(bp0 + k*4 +  4);
                    float32x4_t ba2 = vld1q_f32(bp0 + k*4 +  8);
                    float32x4_t ba3 = vld1q_f32(bp0 + k*4 + 12);
                    float32x4_t bb0 = vld1q_f32(bp1 + k*4 +  0);
                    float32x4_t bb1 = vld1q_f32(bp1 + k*4 +  4);
                    float32x4_t bb2 = vld1q_f32(bp1 + k*4 +  8);
                    float32x4_t bb3 = vld1q_f32(bp1 + k*4 + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    // Row 0
                    acc0a = vfmaq_laneq_f32(acc0a, ba0, av0, 0);
                    acc0a = vfmaq_laneq_f32(acc0a, ba1, av0, 1);
                    acc0a = vfmaq_laneq_f32(acc0a, ba2, av0, 2);
                    acc0a = vfmaq_laneq_f32(acc0a, ba3, av0, 3);
                    acc0b = vfmaq_laneq_f32(acc0b, bb0, av0, 0);
                    acc0b = vfmaq_laneq_f32(acc0b, bb1, av0, 1);
                    acc0b = vfmaq_laneq_f32(acc0b, bb2, av0, 2);
                    acc0b = vfmaq_laneq_f32(acc0b, bb3, av0, 3);
                    // Row 1
                    acc1a = vfmaq_laneq_f32(acc1a, ba0, av1, 0);
                    acc1a = vfmaq_laneq_f32(acc1a, ba1, av1, 1);
                    acc1a = vfmaq_laneq_f32(acc1a, ba2, av1, 2);
                    acc1a = vfmaq_laneq_f32(acc1a, ba3, av1, 3);
                    acc1b = vfmaq_laneq_f32(acc1b, bb0, av1, 0);
                    acc1b = vfmaq_laneq_f32(acc1b, bb1, av1, 1);
                    acc1b = vfmaq_laneq_f32(acc1b, bb2, av1, 2);
                    acc1b = vfmaq_laneq_f32(acc1b, bb3, av1, 3);
                    // Row 2
                    acc2a = vfmaq_laneq_f32(acc2a, ba0, av2, 0);
                    acc2a = vfmaq_laneq_f32(acc2a, ba1, av2, 1);
                    acc2a = vfmaq_laneq_f32(acc2a, ba2, av2, 2);
                    acc2a = vfmaq_laneq_f32(acc2a, ba3, av2, 3);
                    acc2b = vfmaq_laneq_f32(acc2b, bb0, av2, 0);
                    acc2b = vfmaq_laneq_f32(acc2b, bb1, av2, 1);
                    acc2b = vfmaq_laneq_f32(acc2b, bb2, av2, 2);
                    acc2b = vfmaq_laneq_f32(acc2b, bb3, av2, 3);
                    // Row 3
                    acc3a = vfmaq_laneq_f32(acc3a, ba0, av3, 0);
                    acc3a = vfmaq_laneq_f32(acc3a, ba1, av3, 1);
                    acc3a = vfmaq_laneq_f32(acc3a, ba2, av3, 2);
                    acc3a = vfmaq_laneq_f32(acc3a, ba3, av3, 3);
                    acc3b = vfmaq_laneq_f32(acc3b, bb0, av3, 0);
                    acc3b = vfmaq_laneq_f32(acc3b, bb1, av3, 1);
                    acc3b = vfmaq_laneq_f32(acc3b, bb2, av3, 2);
                    acc3b = vfmaq_laneq_f32(acc3b, bb3, av3, 3);
                    // Row 4
                    acc4a = vfmaq_laneq_f32(acc4a, ba0, av4, 0);
                    acc4a = vfmaq_laneq_f32(acc4a, ba1, av4, 1);
                    acc4a = vfmaq_laneq_f32(acc4a, ba2, av4, 2);
                    acc4a = vfmaq_laneq_f32(acc4a, ba3, av4, 3);
                    acc4b = vfmaq_laneq_f32(acc4b, bb0, av4, 0);
                    acc4b = vfmaq_laneq_f32(acc4b, bb1, av4, 1);
                    acc4b = vfmaq_laneq_f32(acc4b, bb2, av4, 2);
                    acc4b = vfmaq_laneq_f32(acc4b, bb3, av4, 3);
                    // Row 5
                    acc5a = vfmaq_laneq_f32(acc5a, ba0, av5, 0);
                    acc5a = vfmaq_laneq_f32(acc5a, ba1, av5, 1);
                    acc5a = vfmaq_laneq_f32(acc5a, ba2, av5, 2);
                    acc5a = vfmaq_laneq_f32(acc5a, ba3, av5, 3);
                    acc5b = vfmaq_laneq_f32(acc5b, bb0, av5, 0);
                    acc5b = vfmaq_laneq_f32(acc5b, bb1, av5, 1);
                    acc5b = vfmaq_laneq_f32(acc5b, bb2, av5, 2);
                    acc5b = vfmaq_laneq_f32(acc5b, bb3, av5, 3);
                    // Row 6
                    acc6a = vfmaq_laneq_f32(acc6a, ba0, av6, 0);
                    acc6a = vfmaq_laneq_f32(acc6a, ba1, av6, 1);
                    acc6a = vfmaq_laneq_f32(acc6a, ba2, av6, 2);
                    acc6a = vfmaq_laneq_f32(acc6a, ba3, av6, 3);
                    acc6b = vfmaq_laneq_f32(acc6b, bb0, av6, 0);
                    acc6b = vfmaq_laneq_f32(acc6b, bb1, av6, 1);
                    acc6b = vfmaq_laneq_f32(acc6b, bb2, av6, 2);
                    acc6b = vfmaq_laneq_f32(acc6b, bb3, av6, 3);
                    // Row 7
                    acc7a = vfmaq_laneq_f32(acc7a, ba0, av7, 0);
                    acc7a = vfmaq_laneq_f32(acc7a, ba1, av7, 1);
                    acc7a = vfmaq_laneq_f32(acc7a, ba2, av7, 2);
                    acc7a = vfmaq_laneq_f32(acc7a, ba3, av7, 3);
                    acc7b = vfmaq_laneq_f32(acc7b, bb0, av7, 0);
                    acc7b = vfmaq_laneq_f32(acc7b, bb1, av7, 1);
                    acc7b = vfmaq_laneq_f32(acc7b, bb2, av7, 2);
                    acc7b = vfmaq_laneq_f32(acc7b, bb3, av7, 3);
                }
                for (; k < K; ++k) {
                    float32x4_t bav = vld1q_f32(bp0 + k * 4);
                    float32x4_t bbv = vld1q_f32(bp1 + k * 4);
                    acc0a = vfmaq_n_f32(acc0a, bav, a0[k]); acc0b = vfmaq_n_f32(acc0b, bbv, a0[k]);
                    acc1a = vfmaq_n_f32(acc1a, bav, a1[k]); acc1b = vfmaq_n_f32(acc1b, bbv, a1[k]);
                    acc2a = vfmaq_n_f32(acc2a, bav, a2[k]); acc2b = vfmaq_n_f32(acc2b, bbv, a2[k]);
                    acc3a = vfmaq_n_f32(acc3a, bav, a3[k]); acc3b = vfmaq_n_f32(acc3b, bbv, a3[k]);
                    acc4a = vfmaq_n_f32(acc4a, bav, a4[k]); acc4b = vfmaq_n_f32(acc4b, bbv, a4[k]);
                    acc5a = vfmaq_n_f32(acc5a, bav, a5[k]); acc5b = vfmaq_n_f32(acc5b, bbv, a5[k]);
                    acc6a = vfmaq_n_f32(acc6a, bav, a6[k]); acc6b = vfmaq_n_f32(acc6b, bbv, a6[k]);
                    acc7a = vfmaq_n_f32(acc7a, bav, a7[k]); acc7b = vfmaq_n_f32(acc7b, bbv, a7[k]);
                }

                float32x4_t bva = bias ? vld1q_f32(bias + n0a) : vdupq_n_f32(0.f);
                float32x4_t bvb = bias ? vld1q_f32(bias + n0b) : vdupq_n_f32(0.f);
                auto sv2 = [&](float32x4_t acca, float32x4_t accb, int row)
                    __attribute__((always_inline)) {
                    acca = vaddq_f32(acca, bva);
                    accb = vaddq_f32(accb, bvb);
                    if (relu) { acca = vmaxq_f32(acca, zero_v); accb = vmaxq_f32(accb, zero_v); }
                    vst1q_f32(C + (size_t)(m + row) * N + n0a, acca);
                    vst1q_f32(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a, acc0b, 0); sv2(acc1a, acc1b, 1);
                sv2(acc2a, acc2b, 2); sv2(acc3a, acc3b, 3);
                sv2(acc4a, acc4b, 4); sv2(acc5a, acc5b, 5);
                sv2(acc6a, acc6b, 6); sv2(acc7a, acc7b, 7);
            }   // end dual body
        }
        }   // mi
    }   // end dual co_blk loop

    // ── NEON: 8 rows × 4 cols (single co_blk, handles odd remainder) ─
    for (int co_blk = dual_end_; co_blk < co_e; ++co_blk) {
        for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * MR;
        const float* a0 = A + (size_t)(m+0) * K;
        const float* a1 = A + (size_t)(m+1) * K;
        const float* a2 = A + (size_t)(m+2) * K;
        const float* a3 = A + (size_t)(m+3) * K;
        const float* a4 = A + (size_t)(m+4) * K;
        const float* a5 = A + (size_t)(m+5) * K;
        const float* a6 = A + (size_t)(m+6) * K;
        const float* a7 = A + (size_t)(m+7) * K;
        {
            const float* bptr   = B_packed + (size_t)co_blk * K * 4;
            const int    n0     = co_blk * 4;
            const int    n_valid = std::min(4, N - n0);

            float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
            float32x4_t acc4 = vdupq_n_f32(0.f), acc5 = vdupq_n_f32(0.f);
            float32x4_t acc6 = vdupq_n_f32(0.f), acc7 = vdupq_n_f32(0.f);

            int k = 0;
            for (; k + 3 < K; k += 4) {
                const float* bp = bptr + k * 4;
                float32x4_t bv0 = vld1q_f32(bp +  0);
                float32x4_t bv1 = vld1q_f32(bp +  4);
                float32x4_t bv2 = vld1q_f32(bp +  8);
                float32x4_t bv3 = vld1q_f32(bp + 12);
                float32x4_t av0 = vld1q_f32(a0 + k);
                float32x4_t av1 = vld1q_f32(a1 + k);
                float32x4_t av2 = vld1q_f32(a2 + k);
                float32x4_t av3 = vld1q_f32(a3 + k);
                float32x4_t av4 = vld1q_f32(a4 + k);
                float32x4_t av5 = vld1q_f32(a5 + k);
                float32x4_t av6 = vld1q_f32(a6 + k);
                float32x4_t av7 = vld1q_f32(a7 + k);
                acc0 = vfmaq_laneq_f32(acc0, bv0, av0, 0);
                acc0 = vfmaq_laneq_f32(acc0, bv1, av0, 1);
                acc0 = vfmaq_laneq_f32(acc0, bv2, av0, 2);
                acc0 = vfmaq_laneq_f32(acc0, bv3, av0, 3);
                acc1 = vfmaq_laneq_f32(acc1, bv0, av1, 0);
                acc1 = vfmaq_laneq_f32(acc1, bv1, av1, 1);
                acc1 = vfmaq_laneq_f32(acc1, bv2, av1, 2);
                acc1 = vfmaq_laneq_f32(acc1, bv3, av1, 3);
                acc2 = vfmaq_laneq_f32(acc2, bv0, av2, 0);
                acc2 = vfmaq_laneq_f32(acc2, bv1, av2, 1);
                acc2 = vfmaq_laneq_f32(acc2, bv2, av2, 2);
                acc2 = vfmaq_laneq_f32(acc2, bv3, av2, 3);
                acc3 = vfmaq_laneq_f32(acc3, bv0, av3, 0);
                acc3 = vfmaq_laneq_f32(acc3, bv1, av3, 1);
                acc3 = vfmaq_laneq_f32(acc3, bv2, av3, 2);
                acc3 = vfmaq_laneq_f32(acc3, bv3, av3, 3);
                acc4 = vfmaq_laneq_f32(acc4, bv0, av4, 0);
                acc4 = vfmaq_laneq_f32(acc4, bv1, av4, 1);
                acc4 = vfmaq_laneq_f32(acc4, bv2, av4, 2);
                acc4 = vfmaq_laneq_f32(acc4, bv3, av4, 3);
                acc5 = vfmaq_laneq_f32(acc5, bv0, av5, 0);
                acc5 = vfmaq_laneq_f32(acc5, bv1, av5, 1);
                acc5 = vfmaq_laneq_f32(acc5, bv2, av5, 2);
                acc5 = vfmaq_laneq_f32(acc5, bv3, av5, 3);
                acc6 = vfmaq_laneq_f32(acc6, bv0, av6, 0);
                acc6 = vfmaq_laneq_f32(acc6, bv1, av6, 1);
                acc6 = vfmaq_laneq_f32(acc6, bv2, av6, 2);
                acc6 = vfmaq_laneq_f32(acc6, bv3, av6, 3);
                acc7 = vfmaq_laneq_f32(acc7, bv0, av7, 0);
                acc7 = vfmaq_laneq_f32(acc7, bv1, av7, 1);
                acc7 = vfmaq_laneq_f32(acc7, bv2, av7, 2);
                acc7 = vfmaq_laneq_f32(acc7, bv3, av7, 3);
            }
            for (; k < K; ++k) {
                float32x4_t bv = vld1q_f32(bptr + k * 4);
                acc0 = vfmaq_n_f32(acc0, bv, a0[k]);
                acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                acc2 = vfmaq_n_f32(acc2, bv, a2[k]);
                acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
                acc4 = vfmaq_n_f32(acc4, bv, a4[k]);
                acc5 = vfmaq_n_f32(acc5, bv, a5[k]);
                acc6 = vfmaq_n_f32(acc6, bv, a6[k]);
                acc7 = vfmaq_n_f32(acc7, bv, a7[k]);
            }

            if (n_valid == 4) {
                float32x4_t bias_v = bias ? vld1q_f32(bias + n0) : vdupq_n_f32(0.f);
                float32x4_t zero_v = vdupq_n_f32(0.f);
                auto sv = [&](float32x4_t acc, int row) __attribute__((always_inline)) {
                    acc = vaddq_f32(acc, bias_v);
                    if (relu) acc = vmaxq_f32(acc, zero_v);
                    vst1q_f32(C + (size_t)(m + row) * N + n0, acc);
                };
                sv(acc0, 0); sv(acc1, 1); sv(acc2, 2); sv(acc3, 3);
                sv(acc4, 4); sv(acc5, 5); sv(acc6, 6); sv(acc7, 7);
            } else {
                alignas(16) float tmp[MR][4];
                vst1q_f32(tmp[0], acc0); vst1q_f32(tmp[1], acc1);
                vst1q_f32(tmp[2], acc2); vst1q_f32(tmp[3], acc3);
                vst1q_f32(tmp[4], acc4); vst1q_f32(tmp[5], acc5);
                vst1q_f32(tmp[6], acc6); vst1q_f32(tmp[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cptr = C + (size_t)(m + r) * N + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cptr[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        }   // end single body
        }   // mi
    }   // end single co_blk loop

    // NEON path fully handled above.

#else
    // ── Non-NEON: co_blk-outer, mi-inner (loop interchange for B cache reuse) ──

    // Determine dual_end: largest co_blk index (aligned to pairs) we can safely
    // process in the dual loop (both co_blks must be fully populated).
    int dual_end = co_s;
    while (dual_end + 1 < co_e && (dual_end + 2) * TILE <= N) dual_end += 2;

#if defined(__AVX512F__)
    // ── AVX-512: M-cache-blocked, co_blk-inner ─────────────────────────────
    // MC=32 m-tiles (256 rows) keeps A panel (~512KB) in L2 per core while
    // sweeping all co_blks; avoids re-fetching A from DRAM per co_blk.
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 16;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 16;
        const int n0a = co_blk * 16, n0b = (co_blk + 1) * 16;
        const __m512 bias_va = bias ? _mm512_loadu_ps(bias + n0a) : _mm512_setzero_ps();
        const __m512 bias_vb = bias ? _mm512_loadu_ps(bias + n0b) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m512 acc0a=_mm512_setzero_ps(), acc0b=_mm512_setzero_ps();
            __m512 acc1a=_mm512_setzero_ps(), acc1b=_mm512_setzero_ps();
            __m512 acc2a=_mm512_setzero_ps(), acc2b=_mm512_setzero_ps();
            __m512 acc3a=_mm512_setzero_ps(), acc3b=_mm512_setzero_ps();
            __m512 acc4a=_mm512_setzero_ps(), acc4b=_mm512_setzero_ps();
            __m512 acc5a=_mm512_setzero_ps(), acc5b=_mm512_setzero_ps();
            __m512 acc6a=_mm512_setzero_ps(), acc6b=_mm512_setzero_ps();
            __m512 acc7a=_mm512_setzero_ps(), acc7b=_mm512_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __m512 bva0=_mm512_loadu_ps(bpa+k*16+ 0), bva1=_mm512_loadu_ps(bpa+k*16+16);
                __m512 bva2=_mm512_loadu_ps(bpa+k*16+32), bva3=_mm512_loadu_ps(bpa+k*16+48);
                __m512 bvb0=_mm512_loadu_ps(bpb+k*16+ 0), bvb1=_mm512_loadu_ps(bpb+k*16+16);
                __m512 bvb2=_mm512_loadu_ps(bpb+k*16+32), bvb3=_mm512_loadu_ps(bpb+k*16+48);
#define FMA2_512(aa,ab,ar,b0,b1,b2,b3) do{ \
    __m512 _a0=_mm512_set1_ps((ar)[k+0]),_a1=_mm512_set1_ps((ar)[k+1]); \
    __m512 _a2=_mm512_set1_ps((ar)[k+2]),_a3=_mm512_set1_ps((ar)[k+3]); \
    aa=_mm512_fmadd_ps(_a0,b0,aa); ab=_mm512_fmadd_ps(_a0,bvb0,ab); \
    aa=_mm512_fmadd_ps(_a1,b1,aa); ab=_mm512_fmadd_ps(_a1,bvb1,ab); \
    aa=_mm512_fmadd_ps(_a2,b2,aa); ab=_mm512_fmadd_ps(_a2,bvb2,ab); \
    aa=_mm512_fmadd_ps(_a3,b3,aa); ab=_mm512_fmadd_ps(_a3,bvb3,ab); }while(0)
                FMA2_512(acc0a,acc0b,a0,bva0,bva1,bva2,bva3);
                FMA2_512(acc1a,acc1b,a1,bva0,bva1,bva2,bva3);
                FMA2_512(acc2a,acc2b,a2,bva0,bva1,bva2,bva3);
                FMA2_512(acc3a,acc3b,a3,bva0,bva1,bva2,bva3);
                FMA2_512(acc4a,acc4b,a4,bva0,bva1,bva2,bva3);
                FMA2_512(acc5a,acc5b,a5,bva0,bva1,bva2,bva3);
                FMA2_512(acc6a,acc6b,a6,bva0,bva1,bva2,bva3);
                FMA2_512(acc7a,acc7b,a7,bva0,bva1,bva2,bva3);
#undef FMA2_512
            }
            for (; k < K; ++k) {
                __m512 bva=_mm512_loadu_ps(bpa+k*16), bvb=_mm512_loadu_ps(bpb+k*16);
                __m512 _a0=_mm512_set1_ps(a0[k]), _a1=_mm512_set1_ps(a1[k]);
                __m512 _a2=_mm512_set1_ps(a2[k]), _a3=_mm512_set1_ps(a3[k]);
                __m512 _a4=_mm512_set1_ps(a4[k]), _a5=_mm512_set1_ps(a5[k]);
                __m512 _a6=_mm512_set1_ps(a6[k]), _a7=_mm512_set1_ps(a7[k]);
                acc0a=_mm512_fmadd_ps(_a0,bva,acc0a); acc0b=_mm512_fmadd_ps(_a0,bvb,acc0b);
                acc1a=_mm512_fmadd_ps(_a1,bva,acc1a); acc1b=_mm512_fmadd_ps(_a1,bvb,acc1b);
                acc2a=_mm512_fmadd_ps(_a2,bva,acc2a); acc2b=_mm512_fmadd_ps(_a2,bvb,acc2b);
                acc3a=_mm512_fmadd_ps(_a3,bva,acc3a); acc3b=_mm512_fmadd_ps(_a3,bvb,acc3b);
                acc4a=_mm512_fmadd_ps(_a4,bva,acc4a); acc4b=_mm512_fmadd_ps(_a4,bvb,acc4b);
                acc5a=_mm512_fmadd_ps(_a5,bva,acc5a); acc5b=_mm512_fmadd_ps(_a5,bvb,acc5b);
                acc6a=_mm512_fmadd_ps(_a6,bva,acc6a); acc6b=_mm512_fmadd_ps(_a6,bvb,acc6b);
                acc7a=_mm512_fmadd_ps(_a7,bva,acc7a); acc7b=_mm512_fmadd_ps(_a7,bvb,acc7b);
            }
            {
                auto sv2 = [&](__m512 acca, __m512 accb, int row) __attribute__((always_inline)) {
                    acca = _mm512_add_ps(acca, bias_va);
                    accb = _mm512_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm512_max_ps(acca, zero_v); accb = _mm512_max_ps(accb, zero_v); }
                    _mm512_storeu_ps(C + (size_t)(m + row) * N + n0a, acca);
                    _mm512_storeu_ps(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
                sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        }   // end mi loop (dual AVX-512)
    }   // end dual co_blk loop (AVX-512)

    // ── AVX-512 single co_blk (8 rows × 16 cols), co_blk-outer mi-inner ──
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * 16;
        const int    n0      = co_blk * 16;
        const int    n_valid = std::min(16, N - n0);
        const __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
        const __m512 bias_v  = (bias && n_valid == 16) ? _mm512_loadu_ps(bias + n0) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            // Dual-accumulator per row to break k-dependency chain
            __m512 acc0a=_mm512_setzero_ps(), acc0b=_mm512_setzero_ps();
            __m512 acc1a=_mm512_setzero_ps(), acc1b=_mm512_setzero_ps();
            __m512 acc2a=_mm512_setzero_ps(), acc2b=_mm512_setzero_ps();
            __m512 acc3a=_mm512_setzero_ps(), acc3b=_mm512_setzero_ps();
            __m512 acc4a=_mm512_setzero_ps(), acc4b=_mm512_setzero_ps();
            __m512 acc5a=_mm512_setzero_ps(), acc5b=_mm512_setzero_ps();
            __m512 acc6a=_mm512_setzero_ps(), acc6b=_mm512_setzero_ps();
            __m512 acc7a=_mm512_setzero_ps(), acc7b=_mm512_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                const float* bp = bptr + k * 16;
                __m512 bv0 = _mm512_loadu_ps(bp +  0);
                __m512 bv1 = _mm512_loadu_ps(bp + 16);
                __m512 bv2 = _mm512_loadu_ps(bp + 32);
                __m512 bv3 = _mm512_loadu_ps(bp + 48);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+0]),bv0,acc0a); acc0b=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+1]),bv1,acc0b);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+2]),bv2,acc0a); acc0b=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+3]),bv3,acc0b);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+0]),bv0,acc1a); acc1b=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+1]),bv1,acc1b);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+2]),bv2,acc1a); acc1b=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+3]),bv3,acc1b);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+0]),bv0,acc2a); acc2b=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+1]),bv1,acc2b);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+2]),bv2,acc2a); acc2b=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+3]),bv3,acc2b);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+0]),bv0,acc3a); acc3b=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+1]),bv1,acc3b);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+2]),bv2,acc3a); acc3b=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+3]),bv3,acc3b);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+0]),bv0,acc4a); acc4b=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+1]),bv1,acc4b);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+2]),bv2,acc4a); acc4b=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+3]),bv3,acc4b);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+0]),bv0,acc5a); acc5b=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+1]),bv1,acc5b);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+2]),bv2,acc5a); acc5b=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+3]),bv3,acc5b);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+0]),bv0,acc6a); acc6b=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+1]),bv1,acc6b);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+2]),bv2,acc6a); acc6b=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+3]),bv3,acc6b);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+0]),bv0,acc7a); acc7b=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+1]),bv1,acc7b);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+2]),bv2,acc7a); acc7b=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+3]),bv3,acc7b);
            }
            for (; k < K; ++k) {
                __m512 bv = _mm512_loadu_ps(bptr + k * 16);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k]),bv,acc0a);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k]),bv,acc1a);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k]),bv,acc2a);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k]),bv,acc3a);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k]),bv,acc4a);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k]),bv,acc5a);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k]),bv,acc6a);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k]),bv,acc7a);
            }
            {
                auto sv16 = [&](__m512 acca, __m512 accb, int row) __attribute__((always_inline)) {
                    __m512 acc = _mm512_add_ps(acca, accb);
                    if (n_valid == 16) {
                        acc = _mm512_add_ps(acc, bias_v);
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_storeu_ps(C + (size_t)(m + row) * N + n0, acc);
                    } else {
                        if (bias) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, bias + n0));
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_mask_storeu_ps(C + (size_t)(m + row) * N + n0, mask, acc);
                    }
                };
                sv16(acc0a,acc0b,0); sv16(acc1a,acc1b,1); sv16(acc2a,acc2b,2); sv16(acc3a,acc3b,3);
                sv16(acc4a,acc4b,4); sv16(acc5a,acc5b,5); sv16(acc6a,acc6b,6); sv16(acc7a,acc7b,7);
            }
        }   // end mi loop (single AVX-512)
    }   // end single co_blk loop (AVX-512)
    } // end MC block (AVX-512)

#elif defined(__AVX2__)
    // ── AVX2: M-cache-blocked, co_blk-inner ────────────────────────────────
    constexpr int MC = 32;
    const bool h4 = fp32_avx2_h4_enabled();  // 4-row halves, no register spills (see conv1d_kfused)
    // full triples of 8-col blocks first (4 x 24 tile, 12 accumulators; see conv1d_kfused), then pairs.
    int tri_end = co_s;
    if (h4 && fp32_avx2_t3_enabled())
        while (tri_end + 2 < co_e && (tri_end + 3) * 8 <= N) tri_end += 3;
    dual_end = tri_end;
    while (dual_end + 1 < co_e && (dual_end + 2) * TILE <= N) dual_end += 2;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < tri_end; co_blk += 3) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 8;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 8;
        const float* bpc = B_packed + (size_t)(co_blk + 2) * K * 8;
        const int n0a = co_blk * 8, n0b = n0a + 8, n0c = n0a + 16;
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
        for (int h = 0; h < 2; ++h) {
            const int m = mi * MR + h * 4;
            const float* r0 = A + (size_t)(m+0) * K;
            const float* r1 = A + (size_t)(m+1) * K;
            const float* r2 = A + (size_t)(m+2) * K;
            const float* r3 = A + (size_t)(m+3) * K;
            __m256 c0a=_mm256_setzero_ps(), c0b=_mm256_setzero_ps(), c0c=_mm256_setzero_ps();
            __m256 c1a=_mm256_setzero_ps(), c1b=_mm256_setzero_ps(), c1c=_mm256_setzero_ps();
            __m256 c2a=_mm256_setzero_ps(), c2b=_mm256_setzero_ps(), c2c=_mm256_setzero_ps();
            __m256 c3a=_mm256_setzero_ps(), c3b=_mm256_setzero_ps(), c3c=_mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                const __m256 ba = _mm256_loadu_ps(bpa + k * 8), bb = _mm256_loadu_ps(bpb + k * 8);
                const __m256 bc = _mm256_loadu_ps(bpc + k * 8);
                __m256 x;
                x = _mm256_set1_ps(r0[k]); c0a = _mm256_fmadd_ps(x, ba, c0a); c0b = _mm256_fmadd_ps(x, bb, c0b); c0c = _mm256_fmadd_ps(x, bc, c0c);
                x = _mm256_set1_ps(r1[k]); c1a = _mm256_fmadd_ps(x, ba, c1a); c1b = _mm256_fmadd_ps(x, bb, c1b); c1c = _mm256_fmadd_ps(x, bc, c1c);
                x = _mm256_set1_ps(r2[k]); c2a = _mm256_fmadd_ps(x, ba, c2a); c2b = _mm256_fmadd_ps(x, bb, c2b); c2c = _mm256_fmadd_ps(x, bc, c2c);
                x = _mm256_set1_ps(r3[k]); c3a = _mm256_fmadd_ps(x, ba, c3a); c3b = _mm256_fmadd_ps(x, bb, c3b); c3c = _mm256_fmadd_ps(x, bc, c3c);
            }
            auto sv3h = [&](__m256 acca, __m256 accb, __m256 accc, int row) __attribute__((always_inline)) {
                if (bias) {
                    acca = _mm256_add_ps(acca, _mm256_loadu_ps(bias + n0a));
                    accb = _mm256_add_ps(accb, _mm256_loadu_ps(bias + n0b));
                    accc = _mm256_add_ps(accc, _mm256_loadu_ps(bias + n0c));
                } else {
                    acca = _mm256_add_ps(acca, zero_v); accb = _mm256_add_ps(accb, zero_v); accc = _mm256_add_ps(accc, zero_v);
                }
                if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); accc = _mm256_max_ps(accc, zero_v); }
                float* cr = C + (size_t)(m + row) * N;
                _mm256_storeu_ps(cr + n0a, acca); _mm256_storeu_ps(cr + n0b, accb); _mm256_storeu_ps(cr + n0c, accc);
            };
            sv3h(c0a,c0b,c0c,0); sv3h(c1a,c1b,c1c,1); sv3h(c2a,c2b,c2c,2); sv3h(c3a,c3b,c3c,3);
        } // h
        } // mi
    } // co_blk triple
    if (h4) for (int co_blk = tri_end; co_blk < dual_end; co_blk += 2) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 8;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 8;
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
        for (int h = 0; h < 2; ++h) {
            const int m = mi * MR + h * 4;
            const float* r0 = A + (size_t)(m+0) * K;
            const float* r1 = A + (size_t)(m+1) * K;
            const float* r2 = A + (size_t)(m+2) * K;
            const float* r3 = A + (size_t)(m+3) * K;
            __m256 c0a=_mm256_setzero_ps(), c0b=_mm256_setzero_ps(), c1a=_mm256_setzero_ps(), c1b=_mm256_setzero_ps();
            __m256 c2a=_mm256_setzero_ps(), c2b=_mm256_setzero_ps(), c3a=_mm256_setzero_ps(), c3b=_mm256_setzero_ps();
#pragma GCC unroll 2
            for (int k = 0; k < K; ++k) {
                const __m256 ba = _mm256_loadu_ps(bpa + k * 8), bb = _mm256_loadu_ps(bpb + k * 8);
                __m256 x;
                x = _mm256_set1_ps(r0[k]); c0a = _mm256_fmadd_ps(x, ba, c0a); c0b = _mm256_fmadd_ps(x, bb, c0b);
                x = _mm256_set1_ps(r1[k]); c1a = _mm256_fmadd_ps(x, ba, c1a); c1b = _mm256_fmadd_ps(x, bb, c1b);
                x = _mm256_set1_ps(r2[k]); c2a = _mm256_fmadd_ps(x, ba, c2a); c2b = _mm256_fmadd_ps(x, bb, c2b);
                x = _mm256_set1_ps(r3[k]); c3a = _mm256_fmadd_ps(x, ba, c3a); c3b = _mm256_fmadd_ps(x, bb, c3b);
            }
            auto sv2h = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                acca = _mm256_add_ps(acca, bias_va);
                accb = _mm256_add_ps(accb, bias_vb);
                if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                _mm256_storeu_ps(C + (size_t)(m + row) * N + n0a, acca);
                _mm256_storeu_ps(C + (size_t)(m + row) * N + n0b, accb);
            };
            sv2h(c0a,c0b,0); sv2h(c1a,c1b,1); sv2h(c2a,c2b,2); sv2h(c3a,c3b,3);
        } // h
        } // mi
    } // co_blk dual h4
    if (!h4) for (int co_blk = tri_end; co_blk < dual_end; co_blk += 2) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 8;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 8;
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m256 acc0a=_mm256_setzero_ps(), acc0b=_mm256_setzero_ps();
            __m256 acc1a=_mm256_setzero_ps(), acc1b=_mm256_setzero_ps();
            __m256 acc2a=_mm256_setzero_ps(), acc2b=_mm256_setzero_ps();
            __m256 acc3a=_mm256_setzero_ps(), acc3b=_mm256_setzero_ps();
            __m256 acc4a=_mm256_setzero_ps(), acc4b=_mm256_setzero_ps();
            __m256 acc5a=_mm256_setzero_ps(), acc5b=_mm256_setzero_ps();
            __m256 acc6a=_mm256_setzero_ps(), acc6b=_mm256_setzero_ps();
            __m256 acc7a=_mm256_setzero_ps(), acc7b=_mm256_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __builtin_prefetch(bpa + (k+8)*8,     0, 0);
                __builtin_prefetch(bpa + (k+8)*8 + 16, 0, 0);
                __builtin_prefetch(bpb + (k+8)*8,     0, 0);
                __builtin_prefetch(bpb + (k+8)*8 + 16, 0, 0);
                __m256 bva0=_mm256_loadu_ps(bpa+k*8+ 0), bva1=_mm256_loadu_ps(bpa+k*8+ 8);
                __m256 bva2=_mm256_loadu_ps(bpa+k*8+16), bva3=_mm256_loadu_ps(bpa+k*8+24);
                __m256 bvb0=_mm256_loadu_ps(bpb+k*8+ 0), bvb1=_mm256_loadu_ps(bpb+k*8+ 8);
                __m256 bvb2=_mm256_loadu_ps(bpb+k*8+16), bvb3=_mm256_loadu_ps(bpb+k*8+24);
#define FMA2_256(aa,ab,ar,b0,b1,b2,b3) do{ \
    __m256 _a0=_mm256_set1_ps((ar)[k+0]),_a1=_mm256_set1_ps((ar)[k+1]); \
    __m256 _a2=_mm256_set1_ps((ar)[k+2]),_a3=_mm256_set1_ps((ar)[k+3]); \
    aa=_mm256_fmadd_ps(_a0,b0,aa); ab=_mm256_fmadd_ps(_a0,bvb0,ab); \
    aa=_mm256_fmadd_ps(_a1,b1,aa); ab=_mm256_fmadd_ps(_a1,bvb1,ab); \
    aa=_mm256_fmadd_ps(_a2,b2,aa); ab=_mm256_fmadd_ps(_a2,bvb2,ab); \
    aa=_mm256_fmadd_ps(_a3,b3,aa); ab=_mm256_fmadd_ps(_a3,bvb3,ab); }while(0)
                FMA2_256(acc0a,acc0b,a0,bva0,bva1,bva2,bva3);
                FMA2_256(acc1a,acc1b,a1,bva0,bva1,bva2,bva3);
                FMA2_256(acc2a,acc2b,a2,bva0,bva1,bva2,bva3);
                FMA2_256(acc3a,acc3b,a3,bva0,bva1,bva2,bva3);
                FMA2_256(acc4a,acc4b,a4,bva0,bva1,bva2,bva3);
                FMA2_256(acc5a,acc5b,a5,bva0,bva1,bva2,bva3);
                FMA2_256(acc6a,acc6b,a6,bva0,bva1,bva2,bva3);
                FMA2_256(acc7a,acc7b,a7,bva0,bva1,bva2,bva3);
#undef FMA2_256
            }
            for (; k < K; ++k) {
                __m256 bva=_mm256_loadu_ps(bpa+k*8), bvb=_mm256_loadu_ps(bpb+k*8);
                __m256 _a0=_mm256_set1_ps(a0[k]), _a1=_mm256_set1_ps(a1[k]);
                __m256 _a2=_mm256_set1_ps(a2[k]), _a3=_mm256_set1_ps(a3[k]);
                __m256 _a4=_mm256_set1_ps(a4[k]), _a5=_mm256_set1_ps(a5[k]);
                __m256 _a6=_mm256_set1_ps(a6[k]), _a7=_mm256_set1_ps(a7[k]);
                acc0a=_mm256_fmadd_ps(_a0,bva,acc0a); acc0b=_mm256_fmadd_ps(_a0,bvb,acc0b);
                acc1a=_mm256_fmadd_ps(_a1,bva,acc1a); acc1b=_mm256_fmadd_ps(_a1,bvb,acc1b);
                acc2a=_mm256_fmadd_ps(_a2,bva,acc2a); acc2b=_mm256_fmadd_ps(_a2,bvb,acc2b);
                acc3a=_mm256_fmadd_ps(_a3,bva,acc3a); acc3b=_mm256_fmadd_ps(_a3,bvb,acc3b);
                acc4a=_mm256_fmadd_ps(_a4,bva,acc4a); acc4b=_mm256_fmadd_ps(_a4,bvb,acc4b);
                acc5a=_mm256_fmadd_ps(_a5,bva,acc5a); acc5b=_mm256_fmadd_ps(_a5,bvb,acc5b);
                acc6a=_mm256_fmadd_ps(_a6,bva,acc6a); acc6b=_mm256_fmadd_ps(_a6,bvb,acc6b);
                acc7a=_mm256_fmadd_ps(_a7,bva,acc7a); acc7b=_mm256_fmadd_ps(_a7,bvb,acc7b);
            }
            {
                auto sv2 = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                    acca = _mm256_add_ps(acca, bias_va);
                    accb = _mm256_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0a, acca);
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
                sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        }   // end mi loop (dual AVX2)
    }   // end dual co_blk loop (AVX2)

    // ── AVX2 single co_blk (8 rows × 8 cols), co_blk-outer mi-inner ──────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * 8;
        const int    n0      = co_blk * 8;
        const int    n_valid = std::min(8, N - n0);
        const __m256 zero_v  = _mm256_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __builtin_prefetch(bptr + (k+8)*8,      0, 0);
                __builtin_prefetch(bptr + (k+8)*8 + 16, 0, 0);
                const float* bp = bptr + k * 8;
                __m256 bv0 = _mm256_loadu_ps(bp +  0);
                __m256 bv1 = _mm256_loadu_ps(bp +  8);
                __m256 bv2 = _mm256_loadu_ps(bp + 16);
                __m256 bv3 = _mm256_loadu_ps(bp + 24);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+0]), bv0, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+1]), bv1, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+2]), bv2, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+3]), bv3, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+0]), bv0, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+1]), bv1, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+2]), bv2, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+3]), bv3, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+0]), bv0, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+1]), bv1, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+2]), bv2, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+3]), bv3, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+0]), bv0, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+1]), bv1, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+2]), bv2, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+3]), bv3, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+0]), bv0, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+1]), bv1, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+2]), bv2, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+3]), bv3, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+0]), bv0, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+1]), bv1, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+2]), bv2, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+3]), bv3, acc5);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+0]), bv0, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+1]), bv1, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+2]), bv2, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+3]), bv3, acc6);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+0]), bv0, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+1]), bv1, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+2]), bv2, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+3]), bv3, acc7);
            }
            for (; k < K; ++k) {
                __m256 bv = _mm256_loadu_ps(bptr + k * 8);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k]), bv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k]), bv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k]), bv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k]), bv, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k]), bv, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k]), bv, acc5);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k]), bv, acc6);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k]), bv, acc7);
            }
            if (n_valid == 8) {
                const __m256 bias_v = bias ? _mm256_loadu_ps(bias + n0) : _mm256_setzero_ps();
                auto sv8 = [&](__m256 acc, int row) __attribute__((always_inline)) {
                    acc = _mm256_add_ps(acc, bias_v);
                    if (relu) acc = _mm256_max_ps(acc, zero_v);
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0, acc);
                };
                sv8(acc0, 0); sv8(acc1, 1); sv8(acc2, 2); sv8(acc3, 3);
                sv8(acc4, 4); sv8(acc5, 5); sv8(acc6, 6); sv8(acc7, 7);
            } else {
                alignas(32) float tmp[MR][8];
                _mm256_store_ps(tmp[0], acc0); _mm256_store_ps(tmp[1], acc1);
                _mm256_store_ps(tmp[2], acc2); _mm256_store_ps(tmp[3], acc3);
                _mm256_store_ps(tmp[4], acc4); _mm256_store_ps(tmp[5], acc5);
                _mm256_store_ps(tmp[6], acc6); _mm256_store_ps(tmp[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cptr = C + (size_t)(m + r) * N + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cptr[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        }   // end mi loop (single AVX2)
    }   // end single co_blk loop (AVX2)
    } // end MC block (AVX2)

#else
#if defined(KT_FP32_SSE)
    if (fp32_sse_enabled()) {
        sse_kfused_tiles(A, K, B_packed, 1, 0, K, N, bias, C, relu, mi_s, mi_e, co_s, co_e);
    } else
#endif
    {
    // ── Scalar fallback: M-cache-blocked, co_blk-inner ────────────────────
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * TILE;
        const int    n0      = co_blk * TILE;
        const int    n_valid = std::min(TILE, N - n0);
        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;
            float acc[MR][TILE] = {};
            for (int k = 0; k < K; ++k) {
                const float* bp = bptr + k * TILE;
                for (int oc = 0; oc < n_valid; ++oc) {
                    acc[0][oc] += a0[k] * bp[oc];
                    acc[1][oc] += a1[k] * bp[oc];
                    acc[2][oc] += a2[k] * bp[oc];
                    acc[3][oc] += a3[k] * bp[oc];
                    acc[4][oc] += a4[k] * bp[oc];
                    acc[5][oc] += a5[k] * bp[oc];
                    acc[6][oc] += a6[k] * bp[oc];
                    acc[7][oc] += a7[k] * bp[oc];
                }
            }
            for (int r = 0; r < MR; ++r) {
                float* cptr = C + (size_t)(m + r) * N + n0;
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                    cptr[oc] = relu ? std::max(0.f, v) : v;
                }
            }
        }
    }
    } // end MC block (scalar)
    }
#endif  // AVX512/AVX2/scalar

#endif  // __ARM_NEON

    // ── Tail rows (M % MR residual, at most MR-1 rows) ────────────────────
    // M-partition: one share (thread 0's) handles the tail over all co_blks.
    // N-partition: every share handles the tail with its co_blk range.
    if (tail) {
        const int co_s2 = co_s;
        const int co_e2 = co_e;
#if defined(__ARM_NEON)
        // 2..7 residual rows (M = 66 tokens leaves 2): all rows share one pass over each weight panel instead of one
        // weight stream per row. Per output element the same k-ascending vfmaq_n_f32 chain, then bias, relu: identical bytes.
        // A single residual row (the LSTM M = 1 mat-vec) keeps the multi-block path below.
        if (M - m_tiles * MR >= 2) {
            const int R = M - m_tiles * MR, m0 = m_tiles * MR;
            auto rows = [&](auto rt) {
                constexpr int Rc = decltype(rt)::value;
                for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                    const float* bptr = B_packed + (size_t)co_blk * K * TILE;
                    const int n0 = co_blk * TILE, n_valid = std::min(TILE, N - n0);
                    float32x4_t acc[Rc];
                    for (int r = 0; r < Rc; ++r) acc[r] = vdupq_n_f32(0.f);
                    const float* ar = A + (size_t)m0 * K;
                    for (int k = 0; k < K; ++k) {
                        const float32x4_t bv = vld1q_f32(bptr + (size_t)k * 4);
                        for (int r = 0; r < Rc; ++r) acc[r] = vfmaq_n_f32(acc[r], bv, ar[(size_t)r * K + k]);
                    }
                    for (int r = 0; r < Rc; ++r) {
                        float32x4_t v = acc[r];
                        if (n_valid == TILE) {
                            if (bias) v = vaddq_f32(v, vld1q_f32(bias + n0));
                            if (relu) v = vmaxq_f32(v, vdupq_n_f32(0.f));
                            vst1q_f32(C + (size_t)(m0 + r) * N + n0, v);
                        } else {
                            alignas(16) float tmp[4];
                            vst1q_f32(tmp, v);
                            for (int oc = 0; oc < n_valid; ++oc) {
                                float t = tmp[oc] + (bias ? bias[n0 + oc] : 0.f);
                                C[(size_t)(m0 + r) * N + n0 + oc] = relu ? std::max(0.f, t) : t;
                            }
                        }
                    }
                }
            };
            switch (R) {
                case 2: rows(std::integral_constant<int, 2>{}); break;
                case 3: rows(std::integral_constant<int, 3>{}); break;
                case 4: rows(std::integral_constant<int, 4>{}); break;
                case 5: rows(std::integral_constant<int, 5>{}); break;
                case 6: rows(std::integral_constant<int, 6>{}); break;
                default: rows(std::integral_constant<int, 7>{}); break;
            }
        } else
#endif
        for (int m = m_tiles * MR; m < M; ++m) {
            const float* a_row = A + (size_t)m * K;
            int co_q = co_s2;
#if !defined(__ARM_NEON) && (defined(__AVX512F__) || defined(__AVX2__))
            // the one-row path (M = 1 in the LSTM recurrent mat-vec, which runs once per time step on the
            // critical path) had one dependent FMA chain per column block: K FMAs x 4-5 cycles latency, so the FMA
            // pipes sat idle most of the time. Run 4 full column blocks at once (4 independent chains). Each block
            // still sums k ascending with the same fmadd, bias add, relu: bit-identical. KITTEN_FP32_TAIL4=0 turns it off.
            if (fp32_tail4_enabled()) for (; co_q + 4 <= co_e2 && (co_q + 4) * TILE <= N; co_q += 4) {
#if defined(__AVX512F__)
                using V = __m512;
#define T4_LD(p) _mm512_loadu_ps(p)
#define T4_FMA(a, b, c) _mm512_fmadd_ps(_mm512_set1_ps(a), b, c)
#define T4_ZERO _mm512_setzero_ps()
#define T4_ADD _mm512_add_ps
#define T4_MAX _mm512_max_ps
#define T4_ST _mm512_storeu_ps
#else
                using V = __m256;
#define T4_LD(p) _mm256_loadu_ps(p)
#define T4_FMA(a, b, c) _mm256_fmadd_ps(_mm256_set1_ps(a), b, c)
#define T4_ZERO _mm256_setzero_ps()
#define T4_ADD _mm256_add_ps
#define T4_MAX _mm256_max_ps
#define T4_ST _mm256_storeu_ps
#endif
                const float* b0 = B_packed + (size_t)(co_q + 0) * K * TILE;
                const float* b1 = B_packed + (size_t)(co_q + 1) * K * TILE;
                const float* b2 = B_packed + (size_t)(co_q + 2) * K * TILE;
                const float* b3 = B_packed + (size_t)(co_q + 3) * K * TILE;
                V a0 = T4_ZERO, a1 = T4_ZERO, a2 = T4_ZERO, a3 = T4_ZERO;
                for (int k = 0; k < K; ++k) {
                    const float x = a_row[k];
                    a0 = T4_FMA(x, T4_LD(b0 + (size_t)k * TILE), a0);
                    a1 = T4_FMA(x, T4_LD(b1 + (size_t)k * TILE), a1);
                    a2 = T4_FMA(x, T4_LD(b2 + (size_t)k * TILE), a2);
                    a3 = T4_FMA(x, T4_LD(b3 + (size_t)k * TILE), a3);
                }
                const int n0 = co_q * TILE;
                if (bias) {
                    a0 = T4_ADD(a0, T4_LD(bias + n0));            a1 = T4_ADD(a1, T4_LD(bias + n0 + TILE));
                    a2 = T4_ADD(a2, T4_LD(bias + n0 + 2 * TILE)); a3 = T4_ADD(a3, T4_LD(bias + n0 + 3 * TILE));
                }
                if (relu) { a0 = T4_MAX(a0, T4_ZERO); a1 = T4_MAX(a1, T4_ZERO); a2 = T4_MAX(a2, T4_ZERO); a3 = T4_MAX(a3, T4_ZERO); }
                float* cp = C + (size_t)m * N + n0;
                T4_ST(cp, a0); T4_ST(cp + TILE, a1); T4_ST(cp + 2 * TILE, a2); T4_ST(cp + 3 * TILE, a3);
#undef T4_LD
#undef T4_FMA
#undef T4_ZERO
#undef T4_ADD
#undef T4_MAX
#undef T4_ST
            }
#endif
#if defined(__ARM_NEON)
            // (NEON twin of the x86 M = 1 path) the M = 1 row (LSTM recurrent mat-vec, once per time step on the
            // critical path) ran one dependent FMA chain per 4-column block: K FMAs back to back, each waiting ~4 cycles
            // for the previous one. Run NB blocks at once (NB independent chains; KITTEN_FP32_TAIL_NB = 8 or 16, default
            // 8). Each block still sums k ascending with the same vfmaq_n_f32, then bias, relu: bit-identical.
            // KITTEN_FP32_TAIL4=0 turns it off.
            if (fp32_tail4_enabled()) {
                static const int NB = [] { const char* e = kt_dev_getenv("KITTEN_FP32_TAIL_NB"); int v = (e && e[0]) ? std::atoi(e) : 8; return v == 16 ? 16 : 8; }();
                auto run_blocks = [&](auto nb_tag) {
                    constexpr int NBc = decltype(nb_tag)::value;
                    for (; co_q + NBc <= co_e2 && (co_q + NBc) * TILE <= N; co_q += NBc) {
                        const float* bp = B_packed + (size_t)co_q * K * TILE;
                        const size_t bs = (size_t)K * TILE;
                        float32x4_t acc[NBc];
                        for (int j = 0; j < NBc; ++j) acc[j] = vdupq_n_f32(0.f);
                        for (int k = 0; k < K; ++k) {
                            const float x = a_row[k];
                            const float* bk = bp + (size_t)k * TILE;
                            for (int j = 0; j < NBc; ++j) acc[j] = vfmaq_n_f32(acc[j], vld1q_f32(bk + j * bs), x);
                        }
                        const int n0 = co_q * TILE;
                        float* cp = C + (size_t)m * N + n0;
                        for (int j = 0; j < NBc; ++j) {
                            float32x4_t v = acc[j];
                            if (bias) v = vaddq_f32(v, vld1q_f32(bias + n0 + j * TILE));
                            if (relu) v = vmaxq_f32(v, vdupq_n_f32(0.f));
                            vst1q_f32(cp + j * TILE, v);
                        }
                    }
                };
                if (NB == 16) run_blocks(std::integral_constant<int, 16>{});
                run_blocks(std::integral_constant<int, 8>{});
            }
#endif
            for (int co_blk = co_q; co_blk < co_e2; ++co_blk) {
                const float* bptr   = B_packed + (size_t)co_blk * K * TILE;
                const int    n0     = co_blk * TILE;
                const int    n_valid = std::min(TILE, N - n0);
#ifdef __ARM_NEON
                float32x4_t acc = vdupq_n_f32(0.f);
                int k = 0;
                for (; k + 3 < K; k += 4) {
                    const float* bp = bptr + k * 4;
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  0), a_row[k+0]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  4), a_row[k+1]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  8), a_row[k+2]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp + 12), a_row[k+3]);
                }
                for (; k < K; ++k)
                    acc = vfmaq_n_f32(acc, vld1q_f32(bptr + k * 4), a_row[k]);
                if (n_valid == TILE) {
                    if (bias) acc = vaddq_f32(acc, vld1q_f32(bias + n0));
                    if (relu) acc = vmaxq_f32(acc, vdupq_n_f32(0.f));
                    vst1q_f32(C + (size_t)m * N + n0, acc);
                } else {
                    alignas(16) float tmp[4];
                    vst1q_f32(tmp, acc);
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[oc] + (bias ? bias[n0 + oc] : 0.f);
                        C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                    }
                }
#elif defined(__AVX512F__)
                {
                    __m512 acc = _mm512_setzero_ps();
                    int k = 0;
                    for (; k + 3 < K; k += 4) {
                        const float* bp = bptr + k * 16;
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+0]), _mm512_loadu_ps(bp +  0), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+1]), _mm512_loadu_ps(bp + 16), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+2]), _mm512_loadu_ps(bp + 32), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+3]), _mm512_loadu_ps(bp + 48), acc);
                    }
                    for (; k < K; ++k)
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k]), _mm512_loadu_ps(bptr + k*16), acc);
                    __mmask16 mask = (n_valid == TILE) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
                    if (bias) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, bias + n0));
                    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                    _mm512_mask_storeu_ps(C + (size_t)m * N + n0, mask, acc);
                }
#elif defined(__AVX2__)
                {
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + 3 < K; k += 4) {
                        const float* bp = bptr + k * 8;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+0]), _mm256_loadu_ps(bp +  0), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+1]), _mm256_loadu_ps(bp +  8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+2]), _mm256_loadu_ps(bp + 16), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+3]), _mm256_loadu_ps(bp + 24), acc);
                    }
                    for (; k < K; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k]), _mm256_loadu_ps(bptr + k*8), acc);
                    if (n_valid == TILE) {
                        if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                        if (relu) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
                        _mm256_storeu_ps(C + (size_t)m * N + n0, acc);
                    } else {
                        alignas(32) float tmp[8];
                        _mm256_store_ps(tmp, acc);
                        for (int oc = 0; oc < n_valid; ++oc) {
                            float v = tmp[oc] + (bias ? bias[n0 + oc] : 0.f);
                            C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                        }
                    }
                }
#else
                float acc[TILE] = {};
                for (int k = 0; k < K; ++k)
                    for (int oc = 0; oc < n_valid; ++oc)
                        acc[oc] += a_row[k] * bptr[k * TILE + oc];
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[oc] + (bias ? bias[n0 + oc] : 0.f);
                    C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                }
#endif
            }
        }
    }
    });
}


// ──────────────────────────────────────────────────────────────
// gemm_fp32
//   input:    [C_in]
//   weight:   [C_out, C_in]
//   bias:     [C_out]
//   output:   [C_out]
//   w_packed: optional pre-packed weights [Co_t, K, TILE] where K = C_in
// ──────────────────────────────────────────────────────────────
void gemm_fp32_vec(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int C_out,
    const float* w_packed,
    StreamHandle /* stream */)
{
    if (w_packed) {
        sgemm_f32(input, w_packed, bias, output,
                  /*relu=*/false, /*M=*/1, /*K=*/C_in, /*N=*/C_out,
                  /*in_parallel=*/false, nullptr);
    } else {
        float* p = pack_weights_f32(weight, C_out, C_in);
        sgemm_f32(input, p, bias, output,
                  /*relu=*/false, /*M=*/1, /*K=*/C_in, /*N=*/C_out,
                  /*in_parallel=*/false, nullptr);
        free_packed_f32(p);
    }
}


// ──────────────────────────────────────────────────────────────
// conv1d_kfused_sgemm_f32
//   Fused K-split conv1d SGEMM: accumulates all kernel_size positions
//   in a single pass, keeping partial sums in SIMD registers.
//   Result: C is written ONCE per output element (not kernel_size times).
//   A_base:   padded input [T_padded, C_in]
//   B_merged: [Co_t, kernel_size, C_in, TILE]  (pack_merged_weights_f32)
//   C:        output [M, C_out]
// ──────────────────────────────────────────────────────────────
#if defined(__APPLE__) && defined(__ARM_NEON) && defined(KITTEN_HAVE_ACCELERATE)
// direct conv1d (all dilations) through cblas_sgemm. B_merged [Co_t, ks, C_in, 4] -> row-major
// [ks*C_in, C_out] (one tap at a time), then ks GEMMs (tap kp reads input rows t + kp*dil; beta = 1 after the first). Epilogue = the NEON kernel's
// order: + bias, relu, then + residual.
static void conv1d_kfused_accel(const float* A_base, const float* B_merged, const float* bias, float* C, bool relu,
                                int M, int C_in, int C_out, int ks, int dil, const float* residual)
{
    // unpack one tap at a time ([C_in, C_out] scratch instead of [ks*C_in, C_out]: ks-fold smaller buffer).
    const size_t Kt = (size_t)ks * C_in;
    thread_local std::vector<float> b;
    if (b.size() < (size_t)C_in * C_out) b.resize((size_t)C_in * C_out);
    sgemm_f32_accel_threads();
    // (cblas rejects lda < K, so the dilation-1 window cannot be passed as one K = ks*C_in GEMM: ks GEMMs for all.)
    for (int kp = 0; kp < ks; ++kp) {
        accel_unpack_panels(B_merged + (size_t)kp * C_in * 4, Kt * 4, C_in, C_out, b.data());
        const int mb = kitten_accel_mblock() > 0 ? kitten_accel_mblock() : M;
        for (int m0 = 0; m0 < M; m0 += mb)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, std::min(mb, M - m0), C_out, C_in, 1.0f,
                        A_base + ((size_t)m0 + (size_t)kp * dil) * C_in, C_in, b.data(), C_out, kp ? 1.0f : 0.0f,
                        C + (size_t)m0 * C_out, C_out);
    }
    if (bias || relu || residual) {
        for (int m = 0; m < M; ++m) {
            float* c = C + (size_t)m * C_out;
            const float* r = residual ? residual + (size_t)m * C_out : nullptr;
            for (int n = 0; n < C_out; ++n) {
                float v = c[n] + (bias ? bias[n] : 0.f);
                if (relu) v = std::max(v, 0.f);
                c[n] = r ? v + r[n] : v;
            }
        }
    }
}
#endif

void conv1d_kfused_sgemm_f32(
    const float*         A_base,
    const float*         B_merged,
    const float*         bias,
    float*               C,
    bool                 relu,
    int M, int C_in, int C_out,
    int kernel_size, int dilation,
    bool in_parallel,
    const float* residual)
{
#if defined(__APPLE__) && defined(__ARM_NEON) && defined(KITTEN_HAVE_ACCELERATE)
    if (!in_parallel && !omp_in_parallel()) {
        const size_t amin = kitten_accel_sgemm_min();
        if (amin && M >= 16 && (size_t)M * C_in * C_out * kernel_size >= amin) {
            conv1d_kfused_accel(A_base, B_merged, bias, C, relu, M, C_in, C_out, kernel_size, dilation, residual);
            return;
        }
    }
#endif
    if (!in_parallel) {
#ifdef _OPENMP
        if (!omp_in_parallel()) {
            const int nthreads = omp_get_max_threads();
            if (nthreads > 1 && (size_t)M * C_in * C_out * kernel_size >= 2000000UL) {
#pragma omp parallel num_threads(nthreads)
                conv1d_kfused_sgemm_f32(A_base, B_merged, bias, C, relu,
                                        M, C_in, C_out, kernel_size, dilation,
                                        /*in_parallel=*/true, residual);
                return;
            }
        }
#endif
    }

#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0, nT = 1;
#endif

    const int Co_t    = (C_out + TILE - 1) / TILE;
    const int m_tiles = M / MR;

    const bool n_part = (nT > 1) && (M < C_out);
    auto body = [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {

    {
#if defined(__AVX512F__)
    // ── AVX-512: MC-blocked, single N-tile (16 cols × 8 rows), kp inner ─────
    // MC blocking: for each A row block (MC_TILES*8 rows), sweep all co_blks.
    // A[MC×C_in] stays in L2 and is reused Co_t times; cuts L3 traffic for A.
    // Each _mm512_set1_ps(scalar) used once → vfmadd231ps {1to16} embedded broadcast.
    constexpr int MC_TILES = 4;  // 4 × 8 = 32 A-rows, ~64KB per kp in L2
    // dual N-tile (8 rows x 32 cols, 16 accumulators). Each A broadcast feeds two FMAs, so the kernel issues
    // 10 loads per 16 FMAs instead of 36 per 32: on cores with 2 load ports and 2 x 512-bit FMA units (Skylake-SP) the
    // single tile is load-bound. Every output element still sums kp then k in the same order with the same fmadd and
    // the same bias/relu epilogue, so the values are bit-identical. KITTEN_FP32_NT2=0 restores the single tile.
    const bool nt2 = fp32_nt2_enabled();
    int dual_end = co_s;
    if (nt2) while (dual_end + 1 < co_e && (dual_end + 2) * 16 <= C_out) dual_end += 2;
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const int n0 = co_blk * 16;
        const __m512 bias_a = bias ? _mm512_loadu_ps(bias + n0) : _mm512_setzero_ps();
        const __m512 bias_b = bias ? _mm512_loadu_ps(bias + n0 + 16) : _mm512_setzero_ps();
        const __m512 zero_v = _mm512_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m512 a0a=_mm512_setzero_ps(), a0b=_mm512_setzero_ps(), a1a=_mm512_setzero_ps(), a1b=_mm512_setzero_ps();
            __m512 a2a=_mm512_setzero_ps(), a2b=_mm512_setzero_ps(), a3a=_mm512_setzero_ps(), a3b=_mm512_setzero_ps();
            __m512 a4a=_mm512_setzero_ps(), a4b=_mm512_setzero_ps(), a5a=_mm512_setzero_ps(), a5b=_mm512_setzero_ps();
            __m512 a6a=_mm512_setzero_ps(), a6b=_mm512_setzero_ps(), a7a=_mm512_setzero_ps(), a7b=_mm512_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bpa = B_merged + ((size_t)co_blk       * kernel_size + kp) * C_in * 16;
                const float* bpb = B_merged + ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 16;
                const int row_off = kp * dilation;
                const float* r0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* r1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* r2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* r3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* r4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* r5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* r6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* r7 = A_base + (size_t)(m+7 + row_off) * C_in;
#pragma GCC unroll 2
                for (int k = 0; k < C_in; ++k) {
                    const __m512 ba = _mm512_loadu_ps(bpa + k * 16), bb = _mm512_loadu_ps(bpb + k * 16);
                    __m512 x;
                    x = _mm512_set1_ps(r0[k]); a0a = _mm512_fmadd_ps(x, ba, a0a); a0b = _mm512_fmadd_ps(x, bb, a0b);
                    x = _mm512_set1_ps(r1[k]); a1a = _mm512_fmadd_ps(x, ba, a1a); a1b = _mm512_fmadd_ps(x, bb, a1b);
                    x = _mm512_set1_ps(r2[k]); a2a = _mm512_fmadd_ps(x, ba, a2a); a2b = _mm512_fmadd_ps(x, bb, a2b);
                    x = _mm512_set1_ps(r3[k]); a3a = _mm512_fmadd_ps(x, ba, a3a); a3b = _mm512_fmadd_ps(x, bb, a3b);
                    x = _mm512_set1_ps(r4[k]); a4a = _mm512_fmadd_ps(x, ba, a4a); a4b = _mm512_fmadd_ps(x, bb, a4b);
                    x = _mm512_set1_ps(r5[k]); a5a = _mm512_fmadd_ps(x, ba, a5a); a5b = _mm512_fmadd_ps(x, bb, a5b);
                    x = _mm512_set1_ps(r6[k]); a6a = _mm512_fmadd_ps(x, ba, a6a); a6b = _mm512_fmadd_ps(x, bb, a6b);
                    x = _mm512_set1_ps(r7[k]); a7a = _mm512_fmadd_ps(x, ba, a7a); a7b = _mm512_fmadd_ps(x, bb, a7b);
                }
            } // kp
            auto sv2 = [&](__m512 va, __m512 vb, int row) __attribute__((always_inline)) {
                va = _mm512_add_ps(va, bias_a); vb = _mm512_add_ps(vb, bias_b);
                if (relu) { va = _mm512_max_ps(va, zero_v); vb = _mm512_max_ps(vb, zero_v); }
                float* cp = C + (size_t)(m + row) * C_out + n0;
                _mm512_storeu_ps(cp, va); _mm512_storeu_ps(cp + 16, vb);
            };
            sv2(a0a,a0b,0); sv2(a1a,a1b,1); sv2(a2a,a2b,2); sv2(a3a,a3b,3);
            sv2(a4a,a4b,4); sv2(a5a,a5b,5); sv2(a6a,a6b,6); sv2(a7a,a7b,7);
        } // mi
    } // co_blk dual
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 16;
        const int n_valid = std::min(16, C_out - n0);
        const __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
        const __m512 bias_v  = (bias && n_valid == 16) ? _mm512_loadu_ps(bias + n0) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m512 acc0=_mm512_setzero_ps(), acc1=_mm512_setzero_ps();
            __m512 acc2=_mm512_setzero_ps(), acc3=_mm512_setzero_ps();
            __m512 acc4=_mm512_setzero_ps(), acc5=_mm512_setzero_ps();
            __m512 acc6=_mm512_setzero_ps(), acc7=_mm512_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off  = ((size_t)co_blk * kernel_size + kp) * C_in * 16;
                const int    row_off = kp * dilation;
                const float* bptr = B_merged + kp_off;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    __m512 bv0=_mm512_loadu_ps(bptr+k*16+ 0), bv1=_mm512_loadu_ps(bptr+k*16+16);
                    __m512 bv2=_mm512_loadu_ps(bptr+k*16+32), bv3=_mm512_loadu_ps(bptr+k*16+48);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+0]),bv0,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+1]),bv1,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+2]),bv2,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+3]),bv3,acc0);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+0]),bv0,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+1]),bv1,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+2]),bv2,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+3]),bv3,acc1);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+0]),bv0,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+1]),bv1,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+2]),bv2,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+3]),bv3,acc2);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+0]),bv0,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+1]),bv1,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+2]),bv2,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+3]),bv3,acc3);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+0]),bv0,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+1]),bv1,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+2]),bv2,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+3]),bv3,acc4);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+0]),bv0,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+1]),bv1,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+2]),bv2,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+3]),bv3,acc5);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+0]),bv0,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+1]),bv1,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+2]),bv2,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+3]),bv3,acc6);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+0]),bv0,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+1]),bv1,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+2]),bv2,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+3]),bv3,acc7);
                }
                for (; k < C_in; ++k) {
                    __m512 bv=_mm512_loadu_ps(bptr+k*16);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k]),bv,acc0);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k]),bv,acc1);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k]),bv,acc2);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k]),bv,acc3);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k]),bv,acc4);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k]),bv,acc5);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k]),bv,acc6);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k]),bv,acc7);
                }
            } // kp
            {
                auto sv = [&](__m512 acc, int row) __attribute__((always_inline)) {
                    if (n_valid == 16) {
                        acc = _mm512_add_ps(acc, bias_v);
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_storeu_ps(C + (size_t)(m + row) * C_out + n0, acc);
                    } else {
                        if (bias) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, bias + n0));
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_mask_storeu_ps(C + (size_t)(m + row) * C_out + n0, mask, acc);
                    }
                };
                sv(acc0,0); sv(acc1,1); sv(acc2,2); sv(acc3,3);
                sv(acc4,4); sv(acc5,5); sv(acc6,6); sv(acc7,7);
            }
        } // mi
    } // co_blk
    } // mc_base
#elif defined(__AVX2__)
    // ── AVX2 dual co_blk (8 rows × 16 cols), MC-blocked ─────────────────────
    // VEX encoding has no embedded-broadcast FMA → dual N-tile remains optimal
    // MC blocking: for each A row block sweep all co_blks → A stays in L2.
    {
    constexpr int MC_TILES = 4;
    const bool h4 = fp32_avx2_h4_enabled();
    // full triples of 8-col blocks first (4 x 24 tile), then pairs, then singles.
    int tri_end = co_s;
    if (h4 && fp32_avx2_t3_enabled())
        while (tri_end + 2 < co_e && (tri_end + 3) * 8 <= C_out) tri_end += 3;
    int dual_end = tri_end;
    while (dual_end + 1 < co_e && (dual_end + 2) * 8 <= C_out) dual_end += 2;
    // the 8 x 16 dual tile needs 16 accumulators + weights + a broadcast, more than the 16 ymm registers,
    // so GCC keeps 13 accumulators on the stack (26 stack loads/stores per 64 FMAs in the hot loop). Run each 8-row
    // tile as two 4 x 16 halves instead (8 accumulators + 2 weights + 1 broadcast = 11 registers, no spills). Every
    // output element still sums kp then k ascending with the same fmadd, bias add and relu: bit-identical.
    // KITTEN_FP32_AVX2_H4=0 restores the 8-row tile.
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    // 4 rows x 24 cols = 12 accumulators + 3 weight vectors + 1 broadcast = 16 ymm registers. 12 independent
    // FMA chains cover Zen 2's 5-cycle x 2-pipe FMA latency (8 chains cover only 80%), and 7 loads feed 12 FMAs instead of
    // 6 per 8. Each output element still sums kp then k ascending with the same fmadd, bias add and relu: bit-identical.
    for (int co_blk = co_s; co_blk < tri_end; co_blk += 3) {
        const int n0a = co_blk * 8, n0b = n0a + 8, n0c = n0a + 16;
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
        for (int h = 0; h < 2; ++h) {
            const int m = mi * MR + h * 4;
            __m256 c0a=_mm256_setzero_ps(), c0b=_mm256_setzero_ps(), c0c=_mm256_setzero_ps();
            __m256 c1a=_mm256_setzero_ps(), c1b=_mm256_setzero_ps(), c1c=_mm256_setzero_ps();
            __m256 c2a=_mm256_setzero_ps(), c2b=_mm256_setzero_ps(), c2c=_mm256_setzero_ps();
            __m256 c3a=_mm256_setzero_ps(), c3b=_mm256_setzero_ps(), c3c=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bpa = B_merged + ((size_t)co_blk       * kernel_size + kp) * C_in * 8;
                const float* bpb = B_merged + ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 8;
                const float* bpc = B_merged + ((size_t)(co_blk + 2) * kernel_size + kp) * C_in * 8;
                const int row_off = kp * dilation;
                const float* r0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* r1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* r2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* r3 = A_base + (size_t)(m+3 + row_off) * C_in;
                for (int k = 0; k < C_in; ++k) {
                    const __m256 ba = _mm256_loadu_ps(bpa + k * 8), bb = _mm256_loadu_ps(bpb + k * 8);
                    const __m256 bc = _mm256_loadu_ps(bpc + k * 8);
                    __m256 x;
                    x = _mm256_set1_ps(r0[k]); c0a = _mm256_fmadd_ps(x, ba, c0a); c0b = _mm256_fmadd_ps(x, bb, c0b); c0c = _mm256_fmadd_ps(x, bc, c0c);
                    x = _mm256_set1_ps(r1[k]); c1a = _mm256_fmadd_ps(x, ba, c1a); c1b = _mm256_fmadd_ps(x, bb, c1b); c1c = _mm256_fmadd_ps(x, bc, c1c);
                    x = _mm256_set1_ps(r2[k]); c2a = _mm256_fmadd_ps(x, ba, c2a); c2b = _mm256_fmadd_ps(x, bb, c2b); c2c = _mm256_fmadd_ps(x, bc, c2c);
                    x = _mm256_set1_ps(r3[k]); c3a = _mm256_fmadd_ps(x, ba, c3a); c3b = _mm256_fmadd_ps(x, bb, c3b); c3c = _mm256_fmadd_ps(x, bc, c3c);
                }
            } // kp
            auto sv3h = [&](__m256 acca, __m256 accb, __m256 accc, int row) __attribute__((always_inline)) {
                if (bias) {
                    acca = _mm256_add_ps(acca, _mm256_loadu_ps(bias + n0a));
                    accb = _mm256_add_ps(accb, _mm256_loadu_ps(bias + n0b));
                    accc = _mm256_add_ps(accc, _mm256_loadu_ps(bias + n0c));
                } else {
                    acca = _mm256_add_ps(acca, zero_v); accb = _mm256_add_ps(accb, zero_v); accc = _mm256_add_ps(accc, zero_v);
                }
                if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); accc = _mm256_max_ps(accc, zero_v); }
                float* cr = C + (size_t)(m + row) * C_out;
                _mm256_storeu_ps(cr + n0a, acca); _mm256_storeu_ps(cr + n0b, accb); _mm256_storeu_ps(cr + n0c, accc);
            };
            sv3h(c0a,c0b,c0c,0); sv3h(c1a,c1b,c1c,1); sv3h(c2a,c2b,c2c,2); sv3h(c3a,c3b,c3c,3);
        } // h
        } // mi
    } // co_blk triple
    if (h4) for (int co_blk = tri_end; co_blk < dual_end; co_blk += 2) {
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
        for (int h = 0; h < 2; ++h) {
            const int m = mi * MR + h * 4;
            __m256 c0a=_mm256_setzero_ps(), c0b=_mm256_setzero_ps(), c1a=_mm256_setzero_ps(), c1b=_mm256_setzero_ps();
            __m256 c2a=_mm256_setzero_ps(), c2b=_mm256_setzero_ps(), c3a=_mm256_setzero_ps(), c3b=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bpa = B_merged + ((size_t)co_blk       * kernel_size + kp) * C_in * 8;
                const float* bpb = B_merged + ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 8;
                const int row_off = kp * dilation;
                const float* r0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* r1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* r2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* r3 = A_base + (size_t)(m+3 + row_off) * C_in;
#pragma GCC unroll 2
                for (int k = 0; k < C_in; ++k) {
                    const __m256 ba = _mm256_loadu_ps(bpa + k * 8), bb = _mm256_loadu_ps(bpb + k * 8);
                    __m256 x;
                    x = _mm256_set1_ps(r0[k]); c0a = _mm256_fmadd_ps(x, ba, c0a); c0b = _mm256_fmadd_ps(x, bb, c0b);
                    x = _mm256_set1_ps(r1[k]); c1a = _mm256_fmadd_ps(x, ba, c1a); c1b = _mm256_fmadd_ps(x, bb, c1b);
                    x = _mm256_set1_ps(r2[k]); c2a = _mm256_fmadd_ps(x, ba, c2a); c2b = _mm256_fmadd_ps(x, bb, c2b);
                    x = _mm256_set1_ps(r3[k]); c3a = _mm256_fmadd_ps(x, ba, c3a); c3b = _mm256_fmadd_ps(x, bb, c3b);
                }
            } // kp
            auto sv2h = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                acca = _mm256_add_ps(acca, bias_va);
                accb = _mm256_add_ps(accb, bias_vb);
                if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0a, acca);
                _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0b, accb);
            };
            sv2h(c0a,c0b,0); sv2h(c1a,c1b,1); sv2h(c2a,c2b,2); sv2h(c3a,c3b,3);
        } // h
        } // mi
    } // co_blk dual h4
    if (!h4) for (int co_blk = tri_end; co_blk < dual_end; co_blk += 2) {
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m256 acc0a=_mm256_setzero_ps(), acc0b=_mm256_setzero_ps();
            __m256 acc1a=_mm256_setzero_ps(), acc1b=_mm256_setzero_ps();
            __m256 acc2a=_mm256_setzero_ps(), acc2b=_mm256_setzero_ps();
            __m256 acc3a=_mm256_setzero_ps(), acc3b=_mm256_setzero_ps();
            __m256 acc4a=_mm256_setzero_ps(), acc4b=_mm256_setzero_ps();
            __m256 acc5a=_mm256_setzero_ps(), acc5b=_mm256_setzero_ps();
            __m256 acc6a=_mm256_setzero_ps(), acc6b=_mm256_setzero_ps();
            __m256 acc7a=_mm256_setzero_ps(), acc7b=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off_a = ((size_t)co_blk       * kernel_size + kp) * C_in * 8;
                const size_t kp_off_b = ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 8;
                const int    row_off  = kp * dilation;
                const float* bpa = B_merged + kp_off_a;
                const float* bpb = B_merged + kp_off_b;
                const float* a0  = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1  = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2  = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3  = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4  = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5  = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6  = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7  = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    __m256 bva0=_mm256_loadu_ps(bpa+k*8+ 0), bva1=_mm256_loadu_ps(bpa+k*8+ 8);
                    __m256 bva2=_mm256_loadu_ps(bpa+k*8+16), bva3=_mm256_loadu_ps(bpa+k*8+24);
                    __m256 bvb0=_mm256_loadu_ps(bpb+k*8+ 0), bvb1=_mm256_loadu_ps(bpb+k*8+ 8);
                    __m256 bvb2=_mm256_loadu_ps(bpb+k*8+16), bvb3=_mm256_loadu_ps(bpb+k*8+24);
#define KF8X2_256(acca,accb,ar) do{ \
    __m256 _a0=_mm256_set1_ps((ar)[k+0]),_a1=_mm256_set1_ps((ar)[k+1]); \
    __m256 _a2=_mm256_set1_ps((ar)[k+2]),_a3=_mm256_set1_ps((ar)[k+3]); \
    acca=_mm256_fmadd_ps(_a0,bva0,acca); accb=_mm256_fmadd_ps(_a0,bvb0,accb); \
    acca=_mm256_fmadd_ps(_a1,bva1,acca); accb=_mm256_fmadd_ps(_a1,bvb1,accb); \
    acca=_mm256_fmadd_ps(_a2,bva2,acca); accb=_mm256_fmadd_ps(_a2,bvb2,accb); \
    acca=_mm256_fmadd_ps(_a3,bva3,acca); accb=_mm256_fmadd_ps(_a3,bvb3,accb); }while(0)
                    KF8X2_256(acc0a,acc0b,a0); KF8X2_256(acc1a,acc1b,a1);
                    KF8X2_256(acc2a,acc2b,a2); KF8X2_256(acc3a,acc3b,a3);
                    KF8X2_256(acc4a,acc4b,a4); KF8X2_256(acc5a,acc5b,a5);
                    KF8X2_256(acc6a,acc6b,a6); KF8X2_256(acc7a,acc7b,a7);
#undef KF8X2_256
                }
                for (; k < C_in; ++k) {
                    __m256 bva=_mm256_loadu_ps(bpa+k*8), bvb=_mm256_loadu_ps(bpb+k*8);
                    __m256 _a0=_mm256_set1_ps(a0[k]), _a1=_mm256_set1_ps(a1[k]);
                    __m256 _a2=_mm256_set1_ps(a2[k]), _a3=_mm256_set1_ps(a3[k]);
                    __m256 _a4=_mm256_set1_ps(a4[k]), _a5=_mm256_set1_ps(a5[k]);
                    __m256 _a6=_mm256_set1_ps(a6[k]), _a7=_mm256_set1_ps(a7[k]);
                    acc0a=_mm256_fmadd_ps(_a0,bva,acc0a); acc0b=_mm256_fmadd_ps(_a0,bvb,acc0b);
                    acc1a=_mm256_fmadd_ps(_a1,bva,acc1a); acc1b=_mm256_fmadd_ps(_a1,bvb,acc1b);
                    acc2a=_mm256_fmadd_ps(_a2,bva,acc2a); acc2b=_mm256_fmadd_ps(_a2,bvb,acc2b);
                    acc3a=_mm256_fmadd_ps(_a3,bva,acc3a); acc3b=_mm256_fmadd_ps(_a3,bvb,acc3b);
                    acc4a=_mm256_fmadd_ps(_a4,bva,acc4a); acc4b=_mm256_fmadd_ps(_a4,bvb,acc4b);
                    acc5a=_mm256_fmadd_ps(_a5,bva,acc5a); acc5b=_mm256_fmadd_ps(_a5,bvb,acc5b);
                    acc6a=_mm256_fmadd_ps(_a6,bva,acc6a); acc6b=_mm256_fmadd_ps(_a6,bvb,acc6b);
                    acc7a=_mm256_fmadd_ps(_a7,bva,acc7a); acc7b=_mm256_fmadd_ps(_a7,bvb,acc7b);
                }
            } // kp
            {
                auto sv2 = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                    acca = _mm256_add_ps(acca, bias_va);
                    accb = _mm256_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0a, acca);
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2);
                sv2(acc3a,acc3b,3); sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5);
                sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        } // mi
    } // co_blk dual
    // ── AVX2 single co_blk remainder ─────────────────────────────────────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 8;
        const int n_valid = std::min(8, C_out - n0);
        const __m256 zero_v = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
            __m256 acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
            __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps();
            __m256 acc6=_mm256_setzero_ps(), acc7=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off = ((size_t)co_blk * kernel_size + kp) * C_in * 8;
                const int    row_off = kp * dilation;
                const float* bptr = B_merged + kp_off;
                const float* a0  = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1  = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2  = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3  = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4  = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5  = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6  = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7  = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    const float* bp = bptr + k * 8;
                    __m256 bv0=_mm256_loadu_ps(bp+ 0), bv1=_mm256_loadu_ps(bp+ 8);
                    __m256 bv2=_mm256_loadu_ps(bp+16), bv3=_mm256_loadu_ps(bp+24);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+0]),bv0,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+1]),bv1,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+2]),bv2,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+3]),bv3,acc0);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+0]),bv0,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+1]),bv1,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+2]),bv2,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+3]),bv3,acc1);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+0]),bv0,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+1]),bv1,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+2]),bv2,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+3]),bv3,acc2);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+0]),bv0,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+1]),bv1,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+2]),bv2,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+3]),bv3,acc3);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+0]),bv0,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+1]),bv1,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+2]),bv2,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+3]),bv3,acc4);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+0]),bv0,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+1]),bv1,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+2]),bv2,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+3]),bv3,acc5);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+0]),bv0,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+1]),bv1,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+2]),bv2,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+3]),bv3,acc6);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+0]),bv0,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+1]),bv1,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+2]),bv2,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+3]),bv3,acc7);
                }
                for (; k < C_in; ++k) {
                    __m256 bv=_mm256_loadu_ps(bptr+k*8);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k]),bv,acc0);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k]),bv,acc1);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k]),bv,acc2);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k]),bv,acc3);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k]),bv,acc4);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k]),bv,acc5);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k]),bv,acc6);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k]),bv,acc7);
                }
            } // kp
            if (n_valid == 8) {
                auto sv8 = [&](__m256 acc, int row) __attribute__((always_inline)) {
                    if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                    if (relu) acc = _mm256_max_ps(acc, zero_v);
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0, acc);
                };
                sv8(acc0,0); sv8(acc1,1); sv8(acc2,2);
                sv8(acc3,3); sv8(acc4,4); sv8(acc5,5);
                sv8(acc6,6); sv8(acc7,7);
            } else {
                alignas(32) float tmp8[MR][8];
                _mm256_store_ps(tmp8[0], acc0); _mm256_store_ps(tmp8[1], acc1);
                _mm256_store_ps(tmp8[2], acc2); _mm256_store_ps(tmp8[3], acc3);
                _mm256_store_ps(tmp8[4], acc4); _mm256_store_ps(tmp8[5], acc5);
                _mm256_store_ps(tmp8[6], acc6); _mm256_store_ps(tmp8[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cp = C + (size_t)(m + r) * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp8[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        } // mi
    } // co_blk single
    } // mc_base
    } // AVX2 dual_end block
#elif defined(__ARM_NEON)
    // ── NEON: MC-blocked, dual co_blk (8 rows × 8 cols), kp inner ────────────
    {
    constexpr int MC_TILES = 4;
    int dual_end = co_s;
    while (dual_end + 1 < co_e && (dual_end + 2) * 4 <= C_out) dual_end += 2;
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    // ── Dual co_blk: 8 rows × 8 cols ─────────────────────────────────────
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const int n0a = co_blk * 4, n0b = (co_blk + 1) * 4;
        const float32x4_t zero_v = vdupq_n_f32(0.f);
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            float32x4_t acc0a=vdupq_n_f32(0), acc0b=vdupq_n_f32(0);
            float32x4_t acc1a=vdupq_n_f32(0), acc1b=vdupq_n_f32(0);
            float32x4_t acc2a=vdupq_n_f32(0), acc2b=vdupq_n_f32(0);
            float32x4_t acc3a=vdupq_n_f32(0), acc3b=vdupq_n_f32(0);
            float32x4_t acc4a=vdupq_n_f32(0), acc4b=vdupq_n_f32(0);
            float32x4_t acc5a=vdupq_n_f32(0), acc5b=vdupq_n_f32(0);
            float32x4_t acc6a=vdupq_n_f32(0), acc6b=vdupq_n_f32(0);
            float32x4_t acc7a=vdupq_n_f32(0), acc7b=vdupq_n_f32(0);
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bp0 = B_merged + ((size_t)co_blk       * kernel_size + kp) * C_in * 4;
                const float* bp1 = B_merged + ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 4;
                const int row_off = kp * dilation;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    float32x4_t ba0 = vld1q_f32(bp0 + k*4 +  0);
                    float32x4_t ba1 = vld1q_f32(bp0 + k*4 +  4);
                    float32x4_t ba2 = vld1q_f32(bp0 + k*4 +  8);
                    float32x4_t ba3 = vld1q_f32(bp0 + k*4 + 12);
                    float32x4_t bb0 = vld1q_f32(bp1 + k*4 +  0);
                    float32x4_t bb1 = vld1q_f32(bp1 + k*4 +  4);
                    float32x4_t bb2 = vld1q_f32(bp1 + k*4 +  8);
                    float32x4_t bb3 = vld1q_f32(bp1 + k*4 + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    acc0a = vfmaq_laneq_f32(acc0a, ba0, av0, 0); acc0b = vfmaq_laneq_f32(acc0b, bb0, av0, 0);
                    acc0a = vfmaq_laneq_f32(acc0a, ba1, av0, 1); acc0b = vfmaq_laneq_f32(acc0b, bb1, av0, 1);
                    acc0a = vfmaq_laneq_f32(acc0a, ba2, av0, 2); acc0b = vfmaq_laneq_f32(acc0b, bb2, av0, 2);
                    acc0a = vfmaq_laneq_f32(acc0a, ba3, av0, 3); acc0b = vfmaq_laneq_f32(acc0b, bb3, av0, 3);
                    acc1a = vfmaq_laneq_f32(acc1a, ba0, av1, 0); acc1b = vfmaq_laneq_f32(acc1b, bb0, av1, 0);
                    acc1a = vfmaq_laneq_f32(acc1a, ba1, av1, 1); acc1b = vfmaq_laneq_f32(acc1b, bb1, av1, 1);
                    acc1a = vfmaq_laneq_f32(acc1a, ba2, av1, 2); acc1b = vfmaq_laneq_f32(acc1b, bb2, av1, 2);
                    acc1a = vfmaq_laneq_f32(acc1a, ba3, av1, 3); acc1b = vfmaq_laneq_f32(acc1b, bb3, av1, 3);
                    acc2a = vfmaq_laneq_f32(acc2a, ba0, av2, 0); acc2b = vfmaq_laneq_f32(acc2b, bb0, av2, 0);
                    acc2a = vfmaq_laneq_f32(acc2a, ba1, av2, 1); acc2b = vfmaq_laneq_f32(acc2b, bb1, av2, 1);
                    acc2a = vfmaq_laneq_f32(acc2a, ba2, av2, 2); acc2b = vfmaq_laneq_f32(acc2b, bb2, av2, 2);
                    acc2a = vfmaq_laneq_f32(acc2a, ba3, av2, 3); acc2b = vfmaq_laneq_f32(acc2b, bb3, av2, 3);
                    acc3a = vfmaq_laneq_f32(acc3a, ba0, av3, 0); acc3b = vfmaq_laneq_f32(acc3b, bb0, av3, 0);
                    acc3a = vfmaq_laneq_f32(acc3a, ba1, av3, 1); acc3b = vfmaq_laneq_f32(acc3b, bb1, av3, 1);
                    acc3a = vfmaq_laneq_f32(acc3a, ba2, av3, 2); acc3b = vfmaq_laneq_f32(acc3b, bb2, av3, 2);
                    acc3a = vfmaq_laneq_f32(acc3a, ba3, av3, 3); acc3b = vfmaq_laneq_f32(acc3b, bb3, av3, 3);
                    acc4a = vfmaq_laneq_f32(acc4a, ba0, av4, 0); acc4b = vfmaq_laneq_f32(acc4b, bb0, av4, 0);
                    acc4a = vfmaq_laneq_f32(acc4a, ba1, av4, 1); acc4b = vfmaq_laneq_f32(acc4b, bb1, av4, 1);
                    acc4a = vfmaq_laneq_f32(acc4a, ba2, av4, 2); acc4b = vfmaq_laneq_f32(acc4b, bb2, av4, 2);
                    acc4a = vfmaq_laneq_f32(acc4a, ba3, av4, 3); acc4b = vfmaq_laneq_f32(acc4b, bb3, av4, 3);
                    acc5a = vfmaq_laneq_f32(acc5a, ba0, av5, 0); acc5b = vfmaq_laneq_f32(acc5b, bb0, av5, 0);
                    acc5a = vfmaq_laneq_f32(acc5a, ba1, av5, 1); acc5b = vfmaq_laneq_f32(acc5b, bb1, av5, 1);
                    acc5a = vfmaq_laneq_f32(acc5a, ba2, av5, 2); acc5b = vfmaq_laneq_f32(acc5b, bb2, av5, 2);
                    acc5a = vfmaq_laneq_f32(acc5a, ba3, av5, 3); acc5b = vfmaq_laneq_f32(acc5b, bb3, av5, 3);
                    acc6a = vfmaq_laneq_f32(acc6a, ba0, av6, 0); acc6b = vfmaq_laneq_f32(acc6b, bb0, av6, 0);
                    acc6a = vfmaq_laneq_f32(acc6a, ba1, av6, 1); acc6b = vfmaq_laneq_f32(acc6b, bb1, av6, 1);
                    acc6a = vfmaq_laneq_f32(acc6a, ba2, av6, 2); acc6b = vfmaq_laneq_f32(acc6b, bb2, av6, 2);
                    acc6a = vfmaq_laneq_f32(acc6a, ba3, av6, 3); acc6b = vfmaq_laneq_f32(acc6b, bb3, av6, 3);
                    acc7a = vfmaq_laneq_f32(acc7a, ba0, av7, 0); acc7b = vfmaq_laneq_f32(acc7b, bb0, av7, 0);
                    acc7a = vfmaq_laneq_f32(acc7a, ba1, av7, 1); acc7b = vfmaq_laneq_f32(acc7b, bb1, av7, 1);
                    acc7a = vfmaq_laneq_f32(acc7a, ba2, av7, 2); acc7b = vfmaq_laneq_f32(acc7b, bb2, av7, 2);
                    acc7a = vfmaq_laneq_f32(acc7a, ba3, av7, 3); acc7b = vfmaq_laneq_f32(acc7b, bb3, av7, 3);
                }
                for (; k < C_in; ++k) {
                    float32x4_t bav = vld1q_f32(bp0 + k * 4);
                    float32x4_t bbv = vld1q_f32(bp1 + k * 4);
                    acc0a = vfmaq_n_f32(acc0a, bav, a0[k]); acc0b = vfmaq_n_f32(acc0b, bbv, a0[k]);
                    acc1a = vfmaq_n_f32(acc1a, bav, a1[k]); acc1b = vfmaq_n_f32(acc1b, bbv, a1[k]);
                    acc2a = vfmaq_n_f32(acc2a, bav, a2[k]); acc2b = vfmaq_n_f32(acc2b, bbv, a2[k]);
                    acc3a = vfmaq_n_f32(acc3a, bav, a3[k]); acc3b = vfmaq_n_f32(acc3b, bbv, a3[k]);
                    acc4a = vfmaq_n_f32(acc4a, bav, a4[k]); acc4b = vfmaq_n_f32(acc4b, bbv, a4[k]);
                    acc5a = vfmaq_n_f32(acc5a, bav, a5[k]); acc5b = vfmaq_n_f32(acc5b, bbv, a5[k]);
                    acc6a = vfmaq_n_f32(acc6a, bav, a6[k]); acc6b = vfmaq_n_f32(acc6b, bbv, a6[k]);
                    acc7a = vfmaq_n_f32(acc7a, bav, a7[k]); acc7b = vfmaq_n_f32(acc7b, bbv, a7[k]);
                }
            } // kp
            float32x4_t bva = bias ? vld1q_f32(bias + n0a) : vdupq_n_f32(0.f);
            float32x4_t bvb = bias ? vld1q_f32(bias + n0b) : vdupq_n_f32(0.f);
            auto sv2 = [&](float32x4_t acca, float32x4_t accb, int row)
                __attribute__((always_inline)) {
                acca = vaddq_f32(acca, bva); accb = vaddq_f32(accb, bvb);
                if (relu) { acca = vmaxq_f32(acca, zero_v); accb = vmaxq_f32(accb, zero_v); }
                vst1q_f32(C + (size_t)(m + row) * C_out + n0a, acca);
                vst1q_f32(C + (size_t)(m + row) * C_out + n0b, accb);
            };
            sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
            sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
        } // mi
    } // co_blk dual
    // ── Single co_blk remainder: 8 rows × 4 cols ──────────────────────────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 4;
        const int n_valid = std::min(4, C_out - n0);
        const float32x4_t zero_v = vdupq_n_f32(0.f);
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            float32x4_t acc0=vdupq_n_f32(0), acc1=vdupq_n_f32(0);
            float32x4_t acc2=vdupq_n_f32(0), acc3=vdupq_n_f32(0);
            float32x4_t acc4=vdupq_n_f32(0), acc5=vdupq_n_f32(0);
            float32x4_t acc6=vdupq_n_f32(0), acc7=vdupq_n_f32(0);
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 4;
                const int row_off = kp * dilation;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    const float* bp = bptr + k * 4;
                    float32x4_t bv0 = vld1q_f32(bp +  0);
                    float32x4_t bv1 = vld1q_f32(bp +  4);
                    float32x4_t bv2 = vld1q_f32(bp +  8);
                    float32x4_t bv3 = vld1q_f32(bp + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    acc0 = vfmaq_laneq_f32(acc0, bv0, av0, 0); acc0 = vfmaq_laneq_f32(acc0, bv1, av0, 1);
                    acc0 = vfmaq_laneq_f32(acc0, bv2, av0, 2); acc0 = vfmaq_laneq_f32(acc0, bv3, av0, 3);
                    acc1 = vfmaq_laneq_f32(acc1, bv0, av1, 0); acc1 = vfmaq_laneq_f32(acc1, bv1, av1, 1);
                    acc1 = vfmaq_laneq_f32(acc1, bv2, av1, 2); acc1 = vfmaq_laneq_f32(acc1, bv3, av1, 3);
                    acc2 = vfmaq_laneq_f32(acc2, bv0, av2, 0); acc2 = vfmaq_laneq_f32(acc2, bv1, av2, 1);
                    acc2 = vfmaq_laneq_f32(acc2, bv2, av2, 2); acc2 = vfmaq_laneq_f32(acc2, bv3, av2, 3);
                    acc3 = vfmaq_laneq_f32(acc3, bv0, av3, 0); acc3 = vfmaq_laneq_f32(acc3, bv1, av3, 1);
                    acc3 = vfmaq_laneq_f32(acc3, bv2, av3, 2); acc3 = vfmaq_laneq_f32(acc3, bv3, av3, 3);
                    acc4 = vfmaq_laneq_f32(acc4, bv0, av4, 0); acc4 = vfmaq_laneq_f32(acc4, bv1, av4, 1);
                    acc4 = vfmaq_laneq_f32(acc4, bv2, av4, 2); acc4 = vfmaq_laneq_f32(acc4, bv3, av4, 3);
                    acc5 = vfmaq_laneq_f32(acc5, bv0, av5, 0); acc5 = vfmaq_laneq_f32(acc5, bv1, av5, 1);
                    acc5 = vfmaq_laneq_f32(acc5, bv2, av5, 2); acc5 = vfmaq_laneq_f32(acc5, bv3, av5, 3);
                    acc6 = vfmaq_laneq_f32(acc6, bv0, av6, 0); acc6 = vfmaq_laneq_f32(acc6, bv1, av6, 1);
                    acc6 = vfmaq_laneq_f32(acc6, bv2, av6, 2); acc6 = vfmaq_laneq_f32(acc6, bv3, av6, 3);
                    acc7 = vfmaq_laneq_f32(acc7, bv0, av7, 0); acc7 = vfmaq_laneq_f32(acc7, bv1, av7, 1);
                    acc7 = vfmaq_laneq_f32(acc7, bv2, av7, 2); acc7 = vfmaq_laneq_f32(acc7, bv3, av7, 3);
                }
                for (; k < C_in; ++k) {
                    float32x4_t bv = vld1q_f32(bptr + k * 4);
                    acc0 = vfmaq_n_f32(acc0, bv, a0[k]); acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                    acc2 = vfmaq_n_f32(acc2, bv, a2[k]); acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
                    acc4 = vfmaq_n_f32(acc4, bv, a4[k]); acc5 = vfmaq_n_f32(acc5, bv, a5[k]);
                    acc6 = vfmaq_n_f32(acc6, bv, a6[k]); acc7 = vfmaq_n_f32(acc7, bv, a7[k]);
                }
            } // kp
            if (n_valid == 4) {
                float32x4_t bias_v = bias ? vld1q_f32(bias + n0) : vdupq_n_f32(0.f);
                auto sv = [&](float32x4_t acc, int row) __attribute__((always_inline)) {
                    acc = vaddq_f32(acc, bias_v);
                    if (relu) acc = vmaxq_f32(acc, zero_v);
                    vst1q_f32(C + (size_t)(m + row) * C_out + n0, acc);
                };
                sv(acc0,0); sv(acc1,1); sv(acc2,2); sv(acc3,3);
                sv(acc4,4); sv(acc5,5); sv(acc6,6); sv(acc7,7);
            } else {
                alignas(16) float tmp[MR][4];
                vst1q_f32(tmp[0],acc0); vst1q_f32(tmp[1],acc1);
                vst1q_f32(tmp[2],acc2); vst1q_f32(tmp[3],acc3);
                vst1q_f32(tmp[4],acc4); vst1q_f32(tmp[5],acc5);
                vst1q_f32(tmp[6],acc6); vst1q_f32(tmp[7],acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cp = C + (size_t)(m + r) * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        } // mi
    } // co_blk single
    } // mc_base
    } // NEON block
#else
#if defined(KT_FP32_SSE)
    if (fp32_sse_enabled()) {
        sse_kfused_tiles(A_base, C_in, B_merged, kernel_size, dilation, C_in, C_out, bias, C, relu, mi_s, mi_e, co_s, co_e);
    } else
#endif
    {
    // ── Scalar fallback ──────────────────────────────────────────────────
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * MR;
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * TILE;
        const int n_valid = std::min(TILE, C_out - n0);
        float acc[MR][TILE] = {};
        for (int kp = 0; kp < kernel_size; ++kp) {
            const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * TILE;
            for (int r = 0; r < MR; ++r) {
                const float* ar = A_base + (size_t)(m + r + (size_t)kp*dilation) * C_in;
                for (int k = 0; k < C_in; ++k)
                    for (int oc = 0; oc < n_valid; ++oc)
                        acc[r][oc] += ar[k] * bptr[k * TILE + oc];
            }
        }
        for (int r = 0; r < MR; ++r) {
            float* cp = C + (size_t)(m + r) * C_out + n0;
            for (int oc = 0; oc < n_valid; ++oc) {
                float v = acc[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                cp[oc] = relu ? std::max(0.f, v) : v;
            }
        }
        } // co_blk
    } // mi
    }
#endif  // AVX512/AVX2/scalar
    } // end block

    // ── Tail rows (M % MR residual) ────────────────────────────────────────
    if (tail) {
        const int co_s2 = co_s;
        const int co_e2 = co_e;
        for (int m = m_tiles * MR; m < M; ++m) {
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int n0      = co_blk * TILE;
                const int n_valid = std::min(TILE, C_out - n0);
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 16;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 16;
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+0]),_mm512_loadu_ps(bp+ 0),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+1]),_mm512_loadu_ps(bp+16),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+2]),_mm512_loadu_ps(bp+32),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+3]),_mm512_loadu_ps(bp+48),acc);
                    }
                    for (; k < C_in; ++k)
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k]),_mm512_loadu_ps(bptr+k*16),acc);
                }
                {
                    __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
                    if (bias) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, bias + n0));
                    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                    _mm512_mask_storeu_ps(C + (size_t)m * C_out + n0, mask, acc);
                }
#elif defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 8;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 8;
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+0]),_mm256_loadu_ps(bp+ 0),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+1]),_mm256_loadu_ps(bp+ 8),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+2]),_mm256_loadu_ps(bp+16),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+3]),_mm256_loadu_ps(bp+24),acc);
                    }
                    for (; k < C_in; ++k)
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k]),_mm256_loadu_ps(bptr+k*8),acc);
                }
                if (n_valid == 8) {
                    if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                    if (relu) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
                    _mm256_storeu_ps(C + (size_t)m * C_out + n0, acc);
                } else {
                    alignas(32) float t8[8]; _mm256_store_ps(t8, acc);
                    float* cp = C + (size_t)m * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = t8[oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
#elif defined(__ARM_NEON)
                {
                float32x4_t acc = vdupq_n_f32(0.f);
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 4;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 4;
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  0), ar[k+0]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  4), ar[k+1]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  8), ar[k+2]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp + 12), ar[k+3]);
                    }
                    for (; k < C_in; ++k)
                        acc = vfmaq_n_f32(acc, vld1q_f32(bptr + k * 4), ar[k]);
                }
                if (n_valid == 4) {
                    if (bias) acc = vaddq_f32(acc, vld1q_f32(bias + n0));
                    if (relu) acc = vmaxq_f32(acc, vdupq_n_f32(0.f));
                    vst1q_f32(C + (size_t)m * C_out + n0, acc);
                } else {
                    alignas(16) float t4[4]; vst1q_f32(t4, acc);
                    float* cp = C + (size_t)m * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = t4[oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
                }
#else
                float acc[TILE] = {};
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * TILE;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    for (int k = 0; k < C_in; ++k)
                        for (int oc = 0; oc < n_valid; ++oc)
                            acc[oc] += ar[k] * bptr[k * TILE + oc];
                }
                float* cp = C + (size_t)m * C_out + n0;
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[oc] + (bias ? bias[n0 + oc] : 0.f);
                    cp[oc] = relu ? std::max(0.f, v) : v;
                }
#endif
            }
        }
    }
    };

    if (!residual) {
        kt_split_tiles(n_part, m_tiles, Co_t, tid, nT, in_parallel, body);
        return;
    }
    // fused residual add. Run the share 4 row tiles (32 rows) at a time and add the residual to those rows
    // while they are still in cache. Every tile is still computed whole by one call (same values); the add is the
    // same a + b as add_fp32 (commutative, so exact whichever operand the graph lists first).
    auto add_rows = [&](int r0, int r1, int c0, int c1) {
        for (int r = r0; r < r1; ++r) {
            float* cp = C + (size_t)r * C_out;
            const float* rp = residual + (size_t)r * C_out;
            for (int c = c0; c < c1; ++c) cp[c] = cp[c] + rp[c];
        }
    };
    kt_split_tiles(n_part, m_tiles, Co_t, tid, nT, in_parallel,
                   [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {
        const int c0 = std::min(C_out, co_s * TILE), c1 = std::min(C_out, co_e * TILE);
        for (int mb = mi_s; mb < mi_e; mb += 4) {
            const int me = std::min(mb + 4, mi_e);
            body(mb, me, co_s, co_e, false);
            add_rows(mb * MR, me * MR, c0, c1);
        }
        if (tail) {
            body(mi_e, mi_e, co_s, co_e, true);
            add_rows(m_tiles * MR, M, c0, c1);
        }
    });
}
