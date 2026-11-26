// Runtime ARM instruction-set level (see backend/cpu/arm_isa.hpp).
#include "../arm_isa.hpp"
#include <cstdlib>
#include <cstring>
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
#include <sys/auxv.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// HWCAP bits from the Linux arm64 uapi (asm/hwcap.h); spelled out so old NDK headers are not needed.
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1UL << 10)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif

int kt_arm_isa_hw() {
    static const int level = [] {
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
        const unsigned long hw = getauxval(AT_HWCAP);
        const unsigned long hw2 = getauxval(AT_HWCAP2);
        if (!(hw & HWCAP_ASIMDDP)) return (int)KT_ARM_V80;
        return (hw2 & HWCAP2_I8MM) ? (int)KT_ARM_I8MM : (int)KT_ARM_DOTPROD;
#elif defined(__APPLE__) && defined(__aarch64__)
        int v = 0; size_t n = sizeof v;
        int lvl = KT_ARM_V80;
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, nullptr, 0) == 0 && v) lvl = KT_ARM_DOTPROD;
        v = 0; n = sizeof v;
        if (lvl == KT_ARM_DOTPROD && sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &n, nullptr, 0) == 0 && v)
            lvl = KT_ARM_I8MM;
        return lvl;
#elif defined(__ARM_FEATURE_MATMUL_INT8)
        return (int)KT_ARM_I8MM;
#elif defined(__ARM_FEATURE_DOTPROD)
        return (int)KT_ARM_DOTPROD;
#else
        return (int)KT_ARM_V80;
#endif
    }();
    return level;
}

int kt_arm_isa() {
    static const int level = [] {
        int lvl = kt_arm_isa_hw();
        const char* v = std::getenv("KITTEN_ARM_ISA");
        if (v && v[0]) {
            int cap = lvl;
            if (!std::strcmp(v, "v80") || !std::strcmp(v, "armv8")) cap = KT_ARM_V80;
            else if (!std::strcmp(v, "dotprod") || !std::strcmp(v, "dp")) cap = KT_ARM_DOTPROD;
            else if (!std::strcmp(v, "i8mm")) cap = KT_ARM_I8MM;
            if (cap < lvl) lvl = cap;
        }
        return lvl;
    }();
    return level;
}

const char* kt_arm_isa_name(int isa) {
    switch (isa) {
        case KT_ARM_I8MM: return "i8mm";
        case KT_ARM_DOTPROD: return "dotprod";
        default: return "v80";
    }
}

bool kt_arm_has_fp16() {
    static const bool on = [] {
        if (kt_arm_isa() == KT_ARM_V80) return false;
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDHP) != 0;
#elif defined(__APPLE__) && defined(__aarch64__)
        return true;
#else
        return false;
#endif
    }();
    return on;
}
