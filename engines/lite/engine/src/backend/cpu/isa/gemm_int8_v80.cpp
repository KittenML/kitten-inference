// Per-level copy of ops/gemm.cpp: Armv8.0 NEON: Cortex-A53 class; SDOT is emulated exactly (backend/cpu/sdot_emu.hpp).
// Build flag (read by the build scripts and CMake): KT_ISA_MARCH=armv8-a
// Compiled only in the ARM dispatch build (see backend/cpu/arm_isa.hpp); the plain names in ops/gemm.cpp call it
// when kt_arm_isa() picks this level.
#define KT_ISA_SUFFIX v80
#include "isa_rename.hpp"
#include "../ops/gemm.cpp"
