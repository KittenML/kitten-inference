// (LOSSY, opt-in KITTEN_FP16_COMPUTE=1): fp16 NEON compute for the K-fused direct conv1d.
// Build flag (read by the build scripts and CMake): KT_ISA_MARCH=armv8.2-a+fp16
// Called only when the CPU reports FEAT_FP16 (HWCAP_ASIMDHP: A55/A75/A76/A78/X1 and later; not A53), see
// kt_arm_has_fp16() in ops/arm_isa.cpp.
//
// What changes numerically (so this is a gated variant, never lossless): weights are rounded to fp16 at load (RNE; the
// same values as the fp16w weights file), each layer's input rows are rounded to fp16, products and partial sums run in
// fp16 (FMLA .8h, 8 lanes: twice the fp32 FMLA rate) over blocks of 32 input channels, and each block's fp16 sum is
// widened and added into fp32 accumulators. Epilogue as the fp32 kernel: + bias, relu, then + residual, in fp32.
// Tile: 4 rows x 16 output channels (8 fp16 + 16 fp32 accumulators + 4 A + 2 B registers).
#if defined(__aarch64__)
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../ops/work_split.hpp"

namespace {
constexpr int KT_H_KB = 32;   // input channels per fp16 partial sum
inline size_t cpad8(int c) { return (size_t)((c + 7) / 8) * 8; }
}

// [C_out, ks*C_in] (tap-major inside a row, as weight_fp32) -> [Co16, ks, C_in8, 16] fp16, zero padded.
uint16_t* kt_pack_kmerged_f16(const float* w, int C_out, int C_in, int ks, size_t* bytes)
{
    const int Co16 = (C_out + 15) / 16; const size_t ci8 = cpad8(C_in), K = (size_t)ks * C_in;
    const size_t n = (size_t)Co16 * ks * ci8 * 16;
    __fp16* p = new __fp16[n];
    std::memset(p, 0, n * sizeof(__fp16));
    for (int cb = 0; cb < Co16; ++cb)
        for (int kp = 0; kp < ks; ++kp)
            for (int ci = 0; ci < C_in; ++ci)
                for (int j = 0; j < 16; ++j) {
                    const int co = cb * 16 + j;
                    if (co < C_out) p[(((size_t)cb * ks + kp) * ci8 + ci) * 16 + j] = (__fp16)w[(size_t)co * K + (size_t)kp * C_in + ci];
                }
    if (bytes) *bytes = n * sizeof(__fp16);
    return reinterpret_cast<uint16_t*>(p);
}

void kt_free_f16(uint16_t* p) { delete[] reinterpret_cast<__fp16*>(p); }

// out[M, C_out] = conv over the padded fp32 input A_base [M + (ks-1)*dil, C_in]; Bh from kt_pack_kmerged_f16.
void conv1d_kfused_hgemm(const float* A_base, const uint16_t* Bh_, const float* bias, float* C, bool relu, int M,
                         int C_in, int C_out, int ks, int dil, const float* residual, int nthreads)
{
    const __fp16* Bh = reinterpret_cast<const __fp16*>(Bh_);
    const size_t ci8 = cpad8(C_in);
    const int rows = M + (ks - 1) * dil;
    static thread_local std::vector<__fp16> ah_buf;
    if (ah_buf.size() < (size_t)rows * ci8 + 8) ah_buf.resize((size_t)rows * ci8 + 8);
    __fp16* ah = ah_buf.data();
    const int Co16 = (C_out + 15) / 16, n_mt = (M + 3) / 4;
    constexpr int RB = 8;                                 // row tiles per task
    const int n_rb = (n_mt + RB - 1) / RB, nT = std::max(1, nthreads);
    int cpc = Co16;
    if (nT > 1 && n_rb < 2 * nT) { const int nc = std::min(Co16, (2 * nT + n_rb - 1) / n_rb); cpc = (Co16 + nc - 1) / nc; }
    const int n_cc = (Co16 + cpc - 1) / cpc, n_tasks = n_rb * n_cc;
    auto conv_row = [&](int r) {
        const float* s = A_base + (size_t)r * C_in; __fp16* d = ah + (size_t)r * ci8; int c = 0;
        for (; c + 8 <= C_in; c += 8) vst1q_f16(d + c, vcombine_f16(vcvt_f16_f32(vld1q_f32(s + c)), vcvt_f16_f32(vld1q_f32(s + c + 4))));
        for (; c < C_in; ++c) d[c] = (__fp16)s[c];
        for (; c < (int)ci8; ++c) d[c] = 0;
    };
    auto task = [&](int id) {
        const int rb = id / n_cc, cc = id % n_cc;
        const int mt0 = rb * RB, mt1 = std::min(n_mt, mt0 + RB), cb0 = cc * cpc, cb1 = std::min(Co16, cb0 + cpc);
        for (int cb = cb0; cb < cb1; ++cb) {
            const int n0 = cb * 16, nv = std::min(16, C_out - n0);
            const __fp16* Bc = Bh + (size_t)cb * ks * ci8 * 16;
            float bz[16] = {0};
            if (bias) std::memcpy(bz, bias + n0, (size_t)nv * sizeof(float));
            for (int mt = mt0; mt < mt1; ++mt) {
                const int m = mt * 4, mr = std::min(4, M - m);
                float32x4_t f00 = vdupq_n_f32(0), f01 = f00, f02 = f00, f03 = f00, f10 = f00, f11 = f00, f12 = f00, f13 = f00;
                float32x4_t f20 = f00, f21 = f00, f22 = f00, f23 = f00, f30 = f00, f31 = f00, f32 = f00, f33 = f00;
                for (int kp = 0; kp < ks; ++kp) {
                    const __fp16* b = Bc + (size_t)kp * ci8 * 16;
                    const int ro = m + kp * dil;
                    const __fp16* a0 = ah + (size_t)(ro + 0) * ci8;
                    const __fp16* a1 = ah + (size_t)(ro + std::min(1, mr - 1)) * ci8;
                    const __fp16* a2 = ah + (size_t)(ro + std::min(2, mr - 1)) * ci8;
                    const __fp16* a3 = ah + (size_t)(ro + std::min(3, mr - 1)) * ci8;
                    for (size_t k0 = 0; k0 < ci8; k0 += KT_H_KB) {
                        const size_t ke = std::min(ci8, k0 + KT_H_KB);
                        float16x8_t h0a = vdupq_n_f16(0), h0b = h0a, h1a = h0a, h1b = h0a, h2a = h0a, h2b = h0a, h3a = h0a, h3b = h0a;
                        for (size_t k = k0; k < ke; k += 8) {
                            const float16x8_t x0 = vld1q_f16(a0 + k), x1 = vld1q_f16(a1 + k), x2 = vld1q_f16(a2 + k), x3 = vld1q_f16(a3 + k);
                            const __fp16* bk = b + k * 16;
#define KT_H_STEP(j)                                                                                                   \
                            {                                                                                          \
                                const float16x8_t ba = vld1q_f16(bk + (j) * 16), bb = vld1q_f16(bk + (j) * 16 + 8);    \
                                h0a = vfmaq_laneq_f16(h0a, ba, x0, j); h0b = vfmaq_laneq_f16(h0b, bb, x0, j);          \
                                h1a = vfmaq_laneq_f16(h1a, ba, x1, j); h1b = vfmaq_laneq_f16(h1b, bb, x1, j);          \
                                h2a = vfmaq_laneq_f16(h2a, ba, x2, j); h2b = vfmaq_laneq_f16(h2b, bb, x2, j);          \
                                h3a = vfmaq_laneq_f16(h3a, ba, x3, j); h3b = vfmaq_laneq_f16(h3b, bb, x3, j);          \
                            }
                            KT_H_STEP(0) KT_H_STEP(1) KT_H_STEP(2) KT_H_STEP(3) KT_H_STEP(4) KT_H_STEP(5) KT_H_STEP(6) KT_H_STEP(7)
#undef KT_H_STEP
                        }
#define KT_H_FLUSH(fa, fb, fc, fd, ha, hb)                                                                             \
                        fa = vaddq_f32(fa, vcvt_f32_f16(vget_low_f16(ha))); fb = vaddq_f32(fb, vcvt_high_f32_f16(ha)); \
                        fc = vaddq_f32(fc, vcvt_f32_f16(vget_low_f16(hb))); fd = vaddq_f32(fd, vcvt_high_f32_f16(hb));
                        KT_H_FLUSH(f00, f01, f02, f03, h0a, h0b)
                        KT_H_FLUSH(f10, f11, f12, f13, h1a, h1b)
                        KT_H_FLUSH(f20, f21, f22, f23, h2a, h2b)
                        KT_H_FLUSH(f30, f31, f32, f33, h3a, h3b)
#undef KT_H_FLUSH
                    }
                }
                float32x4_t acc[4][4] = {{f00, f01, f02, f03}, {f10, f11, f12, f13}, {f20, f21, f22, f23}, {f30, f31, f32, f33}};
                const float32x4_t zero = vdupq_n_f32(0.f);
                for (int r = 0; r < mr; ++r) {
                    float v[16];
                    for (int q = 0; q < 4; ++q) {
                        float32x4_t x = vaddq_f32(acc[r][q], vld1q_f32(bz + 4 * q));
                        if (relu) x = vmaxq_f32(x, zero);
                        vst1q_f32(v + 4 * q, x);
                    }
                    float* o = C + (size_t)(m + r) * C_out + n0;
                    if (residual) {
                        const float* rp = residual + (size_t)(m + r) * C_out + n0;
                        for (int j = 0; j < nv; ++j) o[j] = v[j] + rp[j];
                    } else {
                        std::memcpy(o, v, (size_t)nv * sizeof(float));
                    }
                }
            }
        }
    };
    const int dyn = kt_dynamic_chunks();
#ifdef _OPENMP
    if (nT > 1) {
#pragma omp parallel num_threads(nT)
        {
#pragma omp for schedule(static)
            for (int r = 0; r < rows; ++r) conv_row(r);
            if (dyn > 0) {
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < n_tasks; ++i) task(i);
            } else {
#pragma omp for schedule(static)
                for (int i = 0; i < n_tasks; ++i) task(i);
            }
        }
        return;
    }
#endif
    for (int r = 0; r < rows; ++r) conv_row(r);
    for (int i = 0; i < n_tasks; ++i) task(i);
}
#endif  // __aarch64__
