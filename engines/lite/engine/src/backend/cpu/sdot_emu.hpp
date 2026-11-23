#pragma once
// SDOT for Armv8.0 NEON (CPUs without FEAT_DotProd, e.g. Cortex-A53), exact.
//
// SDOT: acc[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3] for i = 0..3, in int32.
// Here: SMULL/SMULL2 give the 16 int8 x int8 products as int16 (each fits: |p| <= 128*128 = 16384), SADDLP adds
// neighbouring pairs into int32, and ADDP adds neighbouring pairs again, giving the four 4-element sums. Integer
// adds are exact, so the result is the same as SDOT's for every input; only the instruction count differs
// (5 instructions instead of 1).
//
// Included by ops/gemm.cpp only when the compiler has no dot-product target, after <arm_neon.h>, so the int8 kernels
// keep their vdotq_* spelling.
#include <arm_neon.h>

static inline int32x4_t kt_sdot_emu(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));   // products of bytes 0..7
    const int16x8_t hi = vmull_high_s8(a, b);                          // products of bytes 8..15
    const int32x4_t slo = vpaddlq_s16(lo);                             // [p0+p1, p2+p3, p4+p5, p6+p7]
    const int32x4_t shi = vpaddlq_s16(hi);
    return vaddq_s32(acc, vpaddq_s32(slo, shi));                       // [sum 0-3, 4-7, 8-11, 12-15]
}

#ifdef vdotq_s32
#undef vdotq_s32
#endif
#define vdotq_s32(acc, a, b) kt_sdot_emu((acc), (a), (b))

// Lane form: every 4-byte group of a is dotted with the lane-th 4-byte group of b.
#ifdef vdotq_laneq_s32
#undef vdotq_laneq_s32
#endif
#define vdotq_laneq_s32(acc, a, b, lane) \
    kt_sdot_emu((acc), (a), vreinterpretq_s8_s32(vdupq_laneq_s32(vreinterpretq_s32_s8(b), (lane))))
