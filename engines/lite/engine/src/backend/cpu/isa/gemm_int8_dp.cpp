// Per-level copy of ops/gemm.cpp: Armv8.2 + dot product (SDOT): A55, A76, A78.
// Build flag (read by the build scripts and CMake): KT_ISA_MARCH=armv8.2-a+dotprod
// Compiled only in the ARM dispatch build (see backend/cpu/arm_isa.hpp); the plain names in ops/gemm.cpp call it
// when kt_arm_isa() picks this level.
#define KT_ISA_SUFFIX dp
#include "isa_rename.hpp"
#include "../ops/gemm.cpp"
