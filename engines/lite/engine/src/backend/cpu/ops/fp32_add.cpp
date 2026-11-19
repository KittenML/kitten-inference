#include "../ops_neon.hpp"
#include <algorithm>
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
// add_fp32 — vectorised element-wise add with optional ReLU
// ──────────────────────────────────────────────────────────────
static void add_fp32_serial(
    const float* in1,
    const float* in2,
    float*       output,
    int N,
    bool relu)
{
#ifdef __ARM_NEON
    const float32x4_t zero_v = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(in1 + i), vld1q_f32(in2 + i));
        if (relu) v = vmaxq_f32(v, zero_v);
        vst1q_f32(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#elif defined(__AVX512F__)
    const __m512 zero_v = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < N; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(in1 + i), _mm512_loadu_ps(in2 + i));
        if (relu) v = _mm512_max_ps(v, zero_v);
        _mm512_storeu_ps(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#elif defined(__AVX2__)
    const __m256 zero_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_loadu_ps(in1 + i), _mm256_loadu_ps(in2 + i));
        if (relu) v = _mm256_max_ps(v, zero_v);
        _mm256_storeu_ps(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#else
    for (int i = 0; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#endif
}

// the generator's residual adds run on [T, C] tensors of millions of elements (T at 4800 frames/s), and the
// serial loop became a fixed cost on the critical path at 8-16 threads (156 ms at 1 thread, 216 ms at 16, long text,
// a 16-core x86 server). Each thread adds one contiguous range (a multiple of 64 elements); every element is computed exactly as
// before, so the output is bit-identical.
void add_fp32(
    const float* in1,
    const float* in2,
    float*       output,
    int N,
    bool relu,
    StreamHandle /* stream */)
{
#ifdef _OPENMP
    const int nt = omp_get_max_threads();
    if (nt > 1 && N >= (1 << 16) && !omp_in_parallel()) {
#pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num(), n = omp_get_num_threads();
            const long blk = (((long)N + n - 1) / n + 63) / 64 * 64;
            const long s = std::min((long)N, (long)tid * blk), e = std::min((long)N, s + blk);
            if (e > s) add_fp32_serial(in1 + s, in2 + s, output + s, (int)(e - s), relu);
        }
        return;
    }
#endif
    add_fp32_serial(in1, in2, output, N, relu);
}
