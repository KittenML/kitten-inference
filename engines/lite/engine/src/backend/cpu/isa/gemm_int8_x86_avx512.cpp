// Per-level copy of ops/gemm.cpp: x86_64 with AVX-512 (F/BW/VL/DQ) and VNNI (vpdpbusd): Zen 4 and later, Ice Lake and
// later. CMake compiles this file with -mavx2 -mfma -mavx512f/cd/bw/dq/vl/vnni (spelled out, not -march=x86-64-v4: GCC 10
// on manylinux2014 does not know that level). It is only called when kt_x86_avx512_vnni_hw() is true (see
// backend/cpu/x86_isa.hpp).
#define KT_ISA_SUFFIX avx512
#include "isa_rename.hpp"
#include "../ops/gemm.cpp"
