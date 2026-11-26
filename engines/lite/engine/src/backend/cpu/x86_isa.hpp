#pragma once
// Run-time instruction-set level for the int8 kernels on x86_64.
//
// The release wheel is built for AVX2 + FMA, so its int8 kernels have no VNNI: a signed x signed int8 dot product is
// done by widening to int16 and vpmaddwd, about 4x more instructions than vpdpbusd. CPUs with AVX-512 + VNNI (Zen 4
// and later, Ice Lake and later, Sapphire Rapids) run a second copy of the int8 GEMM kernels (ops/gemm.cpp compiled
// with AVX-512 + VNNI, backend/cpu/isa/gemm_int8_x86_avx512.cpp, names suffixed _avx512). Everything else in the
// engine stays the AVX2 build.
//
// The two copies pack weights differently (8 output channels per block on AVX2, 16 on AVX-512), so the level is
// chosen when a weight is packed and stored in the 64 bytes in front of the packed data; the kernels read it from
// there. A model loaded with the other level in the same process therefore keeps working.
//
// Only the Kitten TTS 2 tiny models pick the AVX-512 level (kitten_conv1d_set_tiny_hint, called before the weights
// are packed); every other model keeps the AVX2 code path. KITTEN_X86_ISA=avx2 forces the AVX2 level for all models
// (for tests); the variable never raises the level above what the CPU reports.
//
// Compiled only in the x86_64 dispatch build (CMake ENABLE_X86_ISA_DISPATCH, which defines
// KT_X86_ISA_DISPATCH_BUILD). Builds compiled for AVX-512 themselves (-march=native) keep one kernel set.
#if defined(KT_X86_ISA_DISPATCH_BUILD) && defined(__x86_64__) && !defined(__AVX512BW__) && !defined(KT_ISA_SUFFIX)
#define KT_X86_ISA_DISPATCH 1
#endif

enum KtX86Isa { KT_X86_AVX2 = 0, KT_X86_AVX512_VNNI = 1 };

// The CPU (and the OS) support AVX-512 F/BW/VL/DQ and VNNI. Cached.
bool kt_x86_avx512_vnni_hw();
// Set by InferenceModel (through kitten_conv1d_set_tiny_hint) before it loads weights.
void kt_x86_isa_set_tiny(bool tiny);
// Level the next weight gets packed for.
int kt_x86_pack_isa();
const char* kt_x86_isa_name(int isa);
