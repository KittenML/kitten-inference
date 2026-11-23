// Per-level copy of ops/gemm.cpp: Armv8.6 + int8 matrix multiply (SMMLA): X2 and newer, A710 and newer.
// Build flag (read by the build scripts and CMake): KT_ISA_MARCH=armv8.6-a+i8mm
// Compiled only in the ARM dispatch build (see backend/cpu/arm_isa.hpp); the plain names in ops/gemm.cpp call it
// when kt_arm_isa() picks this level.
#define KT_ISA_SUFFIX i8mm
#include "isa_rename.hpp"
#include "../ops/gemm.cpp"
