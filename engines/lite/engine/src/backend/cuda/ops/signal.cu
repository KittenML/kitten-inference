// ops/signal.cu — the Kitten TTS 2 mini/nano/pico signal chain on the GPU: harmonic source (torch step order, with its
// counter-based excitation noise), torch.stft, the normalized iSTFT, the one-sample reflection pad and length
// regulation. Before these, the engine staged each of them through the CPU (a device -> host copy, CPU work, a host ->
// device copy and two synchronisations per layer).
// Float operations follow the CPU code in backend/cpu/ops/signal.cpp one by one, rounded separately (__fmul_rn /
// __fadd_rn: no contraction into fused multiply-adds), so the two agree up to the math library (sinf, cosf, atan2f).
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"
#include <cstdint>
#include <cstring>

#ifndef M_PI  // MSVC does not define it without _USE_MATH_DEFINES
#define M_PI 3.14159265358979323846
#endif

// ── harmonic source ─────────────────────────────────────────────────────────────────────────────────────────────────
// 1. v[h][i]: the linear downsampling (scale s) of rad = frac(f0 * (h + 1) / sr), one value per frame and harmonic.
static __global__ void k_sine_rad_dn(const float* __restrict__ f0, int T, int T_dn, int H, float s, float sr,
                                     float* __restrict__ v)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (i >= T_dn || h >= H) return;
    const float mult = (float)(h + 1);
    float src = __fsub_rn(__fmul_rn(s, __fadd_rn((float)i, 0.5f)), 0.5f);   // area_pixel_compute_source_index
    if (src < 0.f) src = 0.f;
    const int i0 = (int)src;
    const int i1 = i0 + (i0 < T - 1 ? 1 : 0);
    const float l1 = __fsub_rn(src, (float)i0), l0 = __fsub_rn(1.f, l1);
    float r0 = __fdiv_rn(__fmul_rn(f0[i0], mult), sr); r0 = __fsub_rn(r0, floorf(r0));
    float r1 = __fdiv_rn(__fmul_rn(f0[i1], mult), sr); r1 = __fsub_rn(r1, floorf(r1));
    v[(size_t)h * T_dn + i] = __fadd_rn(__fmul_rn(l0, r0), __fmul_rn(l1, r1));
}

// 2. Per harmonic, the running sum in double (one thread each, in order, as the CPU), phase = float(sum) * 2 * pi * s.
//    Thread H hashes F0 (one value per frame) and the length into the noise seed, as the CPU does.
static __global__ void k_sine_cumsum(const float* __restrict__ v, int T_dn, int H, float s, float pi_f,
                                     float* __restrict__ ph_dn, const float* __restrict__ f0, int T, int up,
                                     unsigned long long seed, int hash_f0, unsigned long long* __restrict__ seed_out)
{
    const int h = threadIdx.x;
    if (h < H) {
        const float* vh = v + (size_t)h * T_dn;
        float* pd = ph_dn + (size_t)h * T_dn;
        double cum = 0.0;
        for (int i = 0; i < T_dn; ++i) {
            cum += (double)vh[i];
            float p = (float)cum;
            p = __fmul_rn(p, 2.0f);
            p = __fmul_rn(p, pi_f);
            pd[i] = __fmul_rn(p, s);
        }
    } else if (h == H) {
        unsigned long long sd = seed;
        if (hash_f0) {
            unsigned long long hh = 0xCBF29CE484222325ULL ^ (unsigned long long)T;
            for (int t = 0; t < T; t += up) {
                const unsigned int bits = __float_as_uint(f0[t]);
                hh = (hh ^ bits) * 0x100000001B3ULL;
            }
            sd ^= hh;
        }
        *seed_out = sd;
    }
}

// 3. One thread per pair of output samples (the noise draws one Box-Muller pair per two samples and harmonic).
static __global__ void k_sine_out(const float* __restrict__ f0, const float* __restrict__ ph_dn, int T, int T_dn, int H,
                                  float inv_s, float sine_amp, float thr, float noise_std, float uv_std, int noisy,
                                  float pi_f, const unsigned long long* __restrict__ seed_p, float* __restrict__ out)
{
    const int pr = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = (T + 1) / 2;
    if (pr >= pairs) return;
    for (int q = 0; q < 2; ++q) {
        const int j = 2 * pr + q;
        if (j >= T) break;
        float src = __fsub_rn(__fmul_rn(inv_s, __fadd_rn((float)j, 0.5f)), 0.5f);
        if (src < 0.f) src = 0.f;
        int i0 = (int)src;
        if (i0 > T_dn - 1) i0 = T_dn - 1;
        const int i1 = i0 + (i0 < T_dn - 1 ? 1 : 0);
        const float l1 = __fsub_rn(src, (float)i0), l0 = __fsub_rn(1.f, l1);
        const bool voiced = f0[j] > thr;
        float* row = out + (size_t)j * H;
        for (int h = 0; h < H; ++h) {
            const float* pd = ph_dn + (size_t)h * T_dn;
            const float ph = __fadd_rn(__fmul_rn(l0, pd[i0]), __fmul_rn(l1, pd[i1]));
            const float val = __fmul_rn(sinf(ph), sine_amp);
            row[h] = voiced ? val : 0.0f;
        }
    }
    if (!noisy) return;
    const unsigned long long seed = *seed_p;
    const int j0 = 2 * pr;
    const float a0 = f0[j0] > thr ? noise_std : uv_std;
    const float a1 = (j0 + 1 < T && f0[j0 + 1] > thr) ? noise_std : uv_std;
    for (int h = 0; h < H; ++h) {
        unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(h + 1)
                             + 0xD1B54A32D192ED03ULL * (unsigned long long)(pr + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
        const float u1 = __fmul_rn(__fadd_rn((float)(z >> 40), 0.5f), 1.0f / 16777216.0f);
        const float u2 = __fmul_rn((float)((z >> 8) & 0xFFFFFFull), 1.0f / 16777216.0f);
        const float r = sqrtf(__fmul_rn(-2.0f, logf(u1)));
        const float a = __fmul_rn(__fmul_rn(2.0f, pi_f), u2);
        out[(size_t)j0 * H + h] = __fadd_rn(out[(size_t)j0 * H + h], __fmul_rn(__fmul_rn(a0, r), cosf(a)));
        if (j0 + 1 < T)
            out[(size_t)(j0 + 1) * H + h] = __fadd_rn(out[(size_t)(j0 + 1) * H + h], __fmul_rn(__fmul_rn(a1, r), sinf(a)));
    }
}

void cuda_sine_gen_torch_fp32(const float* f0, float* out, int T, int harmonic_num, int sample_rate,
                              float sine_amp, float voiced_threshold, int upsample_scale,
                              float noise_std, unsigned long long seed, bool hash_f0, StreamHandle /*stream*/)
{
    const int H = harmonic_num + 1;
    const int T_dn = T / upsample_scale;
    if (T <= 0) return;
    if (T_dn <= 0) { CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)T * H * sizeof(float), 0)); return; }
    const float s = (float)upsample_scale;
    const float inv_s = 1.0f / s;
    const float pi_f = (float)M_PI;
    auto mark = g_scratch.save();
    float* v = (float*)g_scratch.alloc((size_t)H * T_dn * sizeof(float));
    float* ph = (float*)g_scratch.alloc((size_t)H * T_dn * sizeof(float));
    auto* sd = (unsigned long long*)g_scratch.alloc(sizeof(unsigned long long));
    k_sine_rad_dn<<<dim3((T_dn + 127) / 128, H), 128>>>(f0, T, T_dn, H, s, (float)sample_rate, v);
    CUDA_CHECK(cudaGetLastError());
    const bool noisy = noise_std >= 0.f;
    k_sine_cumsum<<<1, 32>>>(v, T_dn, H, s, pi_f, ph, f0, T, upsample_scale, seed,
                             (noisy && hash_f0) ? 1 : 0, sd);
    CUDA_CHECK(cudaGetLastError());
    const int pairs = (T + 1) / 2;
    k_sine_out<<<(pairs + 127) / 128, 128>>>(f0, ph, T, T_dn, H, inv_s, sine_amp, voiced_threshold, noise_std,
                                             sine_amp / 3.0f, noisy ? 1 : 0, pi_f, sd, out);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}

// ── torch.stft (center, reflect padding), magnitude and angle per bin ───────────────────────────────────────────────
static __global__ void k_stft_torch(const float* __restrict__ audio, int T_audio, int n_fft, int hop, int T_stft,
                                    const float* __restrict__ wr, const float* __restrict__ wi, float* __restrict__ out)
{
    const int K = n_fft / 2 + 1, pad = n_fft / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_stft * K) return;
    const int frame = idx / K, k = idx - frame * K;
    float re = 0.f, im = 0.f;
    const float* wrk = wr + k * n_fft;
    const float* wik = wi + k * n_fft;
    for (int n = 0; n < n_fft; ++n) {
        const int p = frame * hop + n;           // index into the padded signal
        float x;
        if (p < pad) x = audio[min(pad - p, T_audio - 1)];
        else if (p < pad + T_audio) x = audio[p - pad];
        else x = audio[max(T_audio - 2 - (p - pad - T_audio), 0)];
        re = __fadd_rn(re, __fmul_rn(wrk[n], x));
        im = __fadd_rn(im, __fmul_rn(wik[n], x));
    }
    if (k == 0 || 2 * k == n_fft) im = 0.f;
    out[(size_t)frame * (2 * K) + k] = sqrtf(__fadd_rn(__fmul_rn(re, re), __fmul_rn(im, im)));
    out[(size_t)frame * (2 * K) + K + k] = atan2f(im, re);
}

void cuda_stft_torch_fp32(const float* audio, float* out, int T_audio, int n_fft, int hop_size,
                          const float* w_real, const float* w_imag, StreamHandle /*stream*/)
{
    const int K = n_fft / 2 + 1;
    const int T_stft = T_audio / hop_size + 1;
    const float* d_wr = g_const_cache.get(w_real, K * n_fft);
    const float* d_wi = g_const_cache.get(w_imag, K * n_fft);
    const int n = T_stft * K;
    k_stft_torch<<<(n + 255) / 256, 256>>>(audio, T_audio, n_fft, hop_size, T_stft, d_wr, d_wi, out);
    CUDA_CHECK(cudaGetLastError());
}

// ── iSTFT (overlap-add as a gather, as the CPU) ─────────────────────────────────────────────────────────────────────
constexpr int KT_ISTFT_MAX_NFFT = 64;
static __constant__ float c_istft_win[KT_ISTFT_MAX_NFFT];

static __global__ void k_istft(const float* __restrict__ mag, const float* __restrict__ phase, float* __restrict__ out,
                               int T_frames, int n_fft, int hop, int normalized, float two_pi_inv_nfft, float pi_f,
                               float inv_nfft, int out_len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_len) return;
    const int K = n_fft / 2 + 1, trim = n_fft / 2;
    const int x = i + trim;
    const int f_lo = x - n_fft + 1 <= 0 ? 0 : (x - n_fft + 1 + hop - 1) / hop;
    const int f_hi = min(T_frames - 1, x / hop);
    float acc = 0.f, env = 0.f;
    for (int f = f_lo; f <= f_hi; ++f) {
        const int n = x - f * hop;
        const float* m = mag + (size_t)f * K;
        const float* p = phase + (size_t)f * K;
        float s = __fmul_rn(m[0], cosf(p[0]));
        for (int k = 1; k < K - 1; ++k) {
            const float arg = __fadd_rn(p[k], __fmul_rn(__fmul_rn(two_pi_inv_nfft, (float)k), (float)n));
            const float mk = normalized ? __fmul_rn(2.0f, m[k]) : m[k];
            s = __fadd_rn(s, __fmul_rn(mk, cosf(arg)));
        }
        s = __fadd_rn(s, __fmul_rn(m[K - 1], cosf(__fadd_rn(p[K - 1], __fmul_rn(pi_f, (float)n)))));
        const float w = c_istft_win[n];
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(s, w), inv_nfft));
        env = __fadd_rn(env, __fmul_rn(w, w));
    }
    if (normalized) out[i] = env > 1e-8f ? __fdiv_rn(acc, env) : 0.f;
    else out[i] = acc;
}

void cuda_istft_fp32(const float* mag, const float* phase, float* out, int T_frames, int n_fft, int hop_size,
                     bool normalized, StreamHandle /*stream*/)
{
    if (n_fft > KT_ISTFT_MAX_NFFT || T_frames <= 0) {
        fprintf(stderr, "cuda_istft_fp32: n_fft %d not supported\n", n_fft);
        exit(1);
    }
    static int s_nfft = -1;
    if (s_nfft != n_fft) {   // the Hann window, computed on the host exactly as the CPU op does
        float win[KT_ISTFT_MAX_NFFT];
        for (int i = 0; i < n_fft; ++i)
            win[i] = 0.5f * (1.f - cosf(2.f * (float)M_PI * i / (float)n_fft));
        CUDA_CHECK(cudaMemcpyToSymbol(c_istft_win, win, n_fft * sizeof(float)));
        s_nfft = n_fft;
    }
    const int ola_len = (T_frames - 1) * hop_size + n_fft;
    const int out_len = ola_len - 2 * (n_fft / 2);
    k_istft<<<(out_len + 255) / 256, 256>>>(mag, phase, out, T_frames, n_fft, hop_size, normalized ? 1 : 0,
                                            2.f * (float)M_PI / (float)n_fft, (float)M_PI, 1.f / (float)n_fft, out_len);
    CUDA_CHECK(cudaGetLastError());
}

// ── reflection pad (one sample on the left: out[0] = in[1]) ─────────────────────────────────────────────────────────
void cuda_reflection_pad1d_fp32(const float* in, float* out, int T, int C, StreamHandle /*stream*/)
{
    CUDA_CHECK(cudaMemcpyAsync(out, in + C, (size_t)C * sizeof(float), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(out + C, in, (size_t)T * C * sizeof(float), cudaMemcpyDeviceToDevice, 0));
}

// ── length regulation: out[f] = in[src[f]] (src: the token of each output frame, made on the host) ──────────────────
static __global__ void k_gather_rows(const float* __restrict__ in, const int* __restrict__ src, float* __restrict__ out,
                                     int T_out, int C)
{
    const int f = blockIdx.x;
    if (f >= T_out) return;
    const float* r = in + (size_t)src[f] * C;
    float* o = out + (size_t)f * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = r[c];
}

void cuda_length_regulate_fp32(const float* feat, const int* src_rows_host, float* out, int T_out, int C,
                               StreamHandle /*stream*/)
{
    if (T_out <= 0) return;
    auto mark = g_scratch.save();
    int* d_src = (int*)g_scratch.alloc((size_t)T_out * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(d_src, src_rows_host, (size_t)T_out * sizeof(int), cudaMemcpyHostToDevice, 0));
    k_gather_rows<<<T_out, C >= 256 ? 256 : ((C + 31) / 32) * 32>>>(feat, d_src, out, T_out, C);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}
