#pragma once
// Runtime ARM instruction-set level for the int8 kernels.
//
// An Android phone may have plain Armv8.0 NEON (Cortex-A53), the dot-product instructions (SDOT, Armv8.2 "dotprod":
// A55/A76/A78), or also the int8 matrix-multiply instructions (SMMLA, Armv8.6 "i8mm": X2+/A710+). A binary built
// with -march=armv8.2-a+dotprod stops with "Illegal instruction" on an A53, and one built with i8mm on an A78.
//
// Dispatch build (KT_ARM_ISA_DISPATCH): the engine is compiled for armv8-a, and only the int8 GEMM kernels
// (ops/gemm.cpp) are compiled three more times, once per level (backend/cpu/isa/*.cpp), with their public names
// suffixed _v80 / _dp / _i8mm. The plain names (gemm_int8, pack_weights_sdot, ...) become small functions that call
// the variant for kt_arm_isa(), which reads the CPU's feature bits once (getauxval(AT_HWCAP / AT_HWCAP2) on
// Linux/Android). The fp32 path uses only Armv8.0 NEON, so it is the same code on every level.
//
// Every level computes the same int32 sums (dot products of int8 values are exact in int32), so the level never
// changes an int8 model's output; the phone tests compare whole-model output at each level.
//
// The environment variable KITTEN_ARM_ISA=v80|dotprod|i8mm lowers the level (for tests). It never raises it above
// what the CPU reports.
// Only builds that compile those copies (CMake ENABLE_ARM_ISA_DISPATCH=ON, which defines KT_ARM_ISA_DISPATCH_BUILD)
// use the dispatch; every other build keeps one kernel set for the level it is compiled for.
#if defined(KT_ARM_ISA_DISPATCH_BUILD) && defined(__aarch64__) && !defined(__APPLE__) \
    && !defined(__ARM_FEATURE_DOTPROD) && !defined(KT_ISA_SUFFIX)
#define KT_ARM_ISA_DISPATCH 1
#endif

enum KtArmIsa { KT_ARM_V80 = 0, KT_ARM_DOTPROD = 1, KT_ARM_I8MM = 2 };

// Level the int8 kernels use (cached after the first call; thread-safe).
int kt_arm_isa();
// What the CPU reports, before the KITTEN_ARM_ISA cap.
int kt_arm_isa_hw();
const char* kt_arm_isa_name(int isa);
// FEAT_FP16 vector arithmetic (HWCAP_ASIMDHP; A55/A75/A76/A78 and later, not A53). Gate for the opt-in
// LOSSY fp16-compute conv (KITTEN_FP16_COMPUTE=1). KITTEN_ARM_ISA=v80 also turns it off (tests).
bool kt_arm_has_fp16();
