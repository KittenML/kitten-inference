// Run-time x86 instruction-set level (see backend/cpu/x86_isa.hpp).
#include "../x86_isa.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>

static std::atomic<bool> g_x86_isa_tiny{false};

bool kt_x86_avx512_vnni_hw()
{
#if defined(KT_X86_ISA_DISPATCH_BUILD) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool ok = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
               __builtin_cpu_supports("avx512vnni");
    }();
    return ok;
#else
    return false;
#endif
}

void kt_x86_isa_set_tiny(bool tiny) { g_x86_isa_tiny.store(tiny); }

int kt_x86_pack_isa()
{
    static const bool capped = [] {
        const char* e = std::getenv("KITTEN_X86_ISA");
        return e && !std::strcmp(e, "avx2");
    }();
    if (capped || !g_x86_isa_tiny.load() || !kt_x86_avx512_vnni_hw()) return KT_X86_AVX2;
    return KT_X86_AVX512_VNNI;
}

const char* kt_x86_isa_name(int isa) { return isa == KT_X86_AVX512_VNNI ? "avx512-vnni" : "avx2"; }
