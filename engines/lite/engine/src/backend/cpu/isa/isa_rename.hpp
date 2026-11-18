// Prelude of a per-level copy of ops/gemm.cpp (see backend/cpu/arm_isa.hpp and backend/cpu/x86_isa.hpp).
// The headers gemm.cpp needs are included first, then gemm.cpp's public names get the level suffix, so that the
// declarations in ops_neon.hpp (included by gemm.cpp after this) and the definitions below carry the suffix.
// Not include-guarded on purpose: each isa/*.cpp includes it once.
#include "engine/free_hint.hpp"
#include "../ops/work_split.hpp"
#include "../ops/profile_internal.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef KT_ISA_SUFFIX
#error "define KT_ISA_SUFFIX before including isa_rename.hpp"
#endif
#if !defined(__aarch64__) && !defined(__x86_64__)
#error "backend/cpu/isa/*.cpp are the AArch64 and x86_64 per-level copies; do not build them for other targets"
#endif
#define KT_ISA_CAT2(a, b) a##_##b
#define KT_ISA_CAT(a, b) KT_ISA_CAT2(a, b)
#define pack_weights_sdot KT_ISA_CAT(pack_weights_sdot, KT_ISA_SUFFIX)
#define free_packed KT_ISA_CAT(free_packed, KT_ISA_SUFFIX)
#define kt_int8_packed_bytes KT_ISA_CAT(kt_int8_packed_bytes, KT_ISA_SUFFIX)
#define gemm_int8 KT_ISA_CAT(gemm_int8, KT_ISA_SUFFIX)
#define gemm_int8_int32 KT_ISA_CAT(gemm_int8_int32, KT_ISA_SUFFIX)
#define gemm_int8_dequant_accum KT_ISA_CAT(gemm_int8_dequant_accum, KT_ISA_SUFFIX)
#define kt_int8_matvec_rows KT_ISA_CAT(kt_int8_matvec_rows, KT_ISA_SUFFIX)
#define conv1d_int8_fused KT_ISA_CAT(conv1d_int8_fused, KT_ISA_SUFFIX)
#define pack_weights_c1i8 KT_ISA_CAT(pack_weights_c1i8, KT_ISA_SUFFIX)
#define kt_c1i8_packed_bytes KT_ISA_CAT(kt_c1i8_packed_bytes, KT_ISA_SUFFIX)
