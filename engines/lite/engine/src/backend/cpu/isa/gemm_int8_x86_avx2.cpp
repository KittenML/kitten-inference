// Per-level copy of ops/gemm.cpp: the x86_64 baseline (AVX2 + FMA), the code the release wheel is built for.
// Compiled only in the x86_64 dispatch build (see backend/cpu/x86_isa.hpp); the plain names in ops/gemm.cpp call it
// for every model that does not pick the AVX-512 level.
#define KT_ISA_SUFFIX avx2
#include "isa_rename.hpp"
#include "../ops/gemm.cpp"
