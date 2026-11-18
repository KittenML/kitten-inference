// Big-core pinning and the big.LITTLE tile split (see cpu_affinity.hpp).
#include "cpu_affinity.hpp"
#include "backend/cpu/ops/work_split.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

static long read_long(const std::string& path) {
    std::ifstream f(path);
    long v = -1;
    if (f) f >> v;
    return f ? v : -1;
}

std::vector<KtCpu> kt_cpus_by_speed() {
    std::vector<KtCpu> out;
#if defined(__linux__) || defined(__ANDROID__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return out;
    const long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    for (int c = 0; c < ncpu && c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        long cap = read_long(base + "/cpu_capacity");
        if (cap <= 0) cap = read_long(base + "/cpufreq/cpuinfo_max_freq");
        if (cap <= 0) cap = 1;
        out.push_back({c, cap});
    }
    std::stable_sort(out.begin(), out.end(), [](const KtCpu& a, const KtCpu& b) { return a.capacity > b.capacity; });
#endif
    return out;
}

std::string KtPinResult::describe() const {
    std::string s = "cpus=";
    for (size_t i = 0; i < cpus.size(); ++i) s += (i ? "," : "") + std::to_string(cpus[i]);
    s += " caps=";
    for (size_t i = 0; i < caps.size(); ++i) s += (i ? "," : "") + std::to_string(caps[i]);
    s += " pinned=" + std::to_string(pinned ? 1 : 0) + " hetero=" + std::to_string(heterogeneous ? 1 : 0) +
         " dyn=" + std::to_string(dynamic_chunks);
    return s;
}

KtPinResult kt_pin_fastest_cores(int n, int dynamic_chunks) {
    KtPinResult r;
    auto cpus = kt_cpus_by_speed();
    if (cpus.empty() || n <= 0) return r;
    n = std::min<int>(n, (int)cpus.size());
    long hi = 0, lo = 0;
    for (int i = 0; i < n; ++i) {
        r.cpus.push_back(cpus[i].id);
        r.caps.push_back(cpus[i].capacity);
    }
    hi = r.caps.front();
    lo = r.caps.back();
    r.heterogeneous = lo * 10 < hi * 9;  // more than 10% apart
#if defined(__linux__) || defined(__ANDROID__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : r.cpus) CPU_SET(c, &set);
    r.pinned = sched_setaffinity(0, sizeof set, &set) == 0;
#endif
    if (r.heterogeneous && n > 1 && dynamic_chunks != 0) {
        r.dynamic_chunks = dynamic_chunks < 0 ? 4 : dynamic_chunks;
        kt_dynamic_chunks_setting().store(r.dynamic_chunks);
    }
    std::vector<std::pair<int, long>> pc;  // report in id order
    for (size_t i = 0; i < r.cpus.size(); ++i) pc.push_back({r.cpus[i], r.caps[i]});
    std::sort(pc.begin(), pc.end());
    for (size_t i = 0; i < pc.size(); ++i) { r.cpus[i] = pc[i].first; r.caps[i] = pc[i].second; }
    return r;
}

std::vector<int> kt_allowed_cpus() {
    std::vector<int> out;
#if defined(__linux__) || defined(__ANDROID__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &allowed)) out.push_back(c);
#endif
    return out;
}

#if defined(__APPLE__)
static int sysctl_int(const char* name) {
    int v = 0;
    size_t n = sizeof v;
    return sysctlbyname(name, &v, &n, nullptr, 0) == 0 ? v : 0;
}
#endif

std::vector<int> kt_fast_core_for_rank(int rank) {
#if !(defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__)))
    (void)rank;
    return {};  // pinning the LSTM scans was measured on Arm Linux / Android only
#else
    if (kt_dynamic_chunks() <= 0) return {};
    static const std::vector<KtCpu> cpus = kt_cpus_by_speed();
    if (cpus.size() < 2 || rank < 0 || rank >= (int)cpus.size() || !kt_speeds_differ(cpus.front().capacity, cpus.back().capacity)) return {};
    return {cpus[(size_t)rank].id};
#endif
}

KtAutoThreads kt_auto_threads_here() {
#if defined(__APPLE__)
    return kt_auto_threads_apple(sysctl_int("hw.perflevel0.physicalcpu"), sysctl_int("hw.physicalcpu"));
#else
    const char* root = std::getenv("KITTEN_TTS08_SYS_ROOT");
    const char* cpuinfo = std::getenv("KITTEN_TTS08_CPUINFO");
    return kt_auto_threads_linux(kt_read_cores(root && *root ? root : "/sys",
                                               cpuinfo && *cpuinfo ? cpuinfo : "/proc/cpuinfo", kt_allowed_cpus()));
#endif
}

KtAffinityScope::KtAffinityScope(const std::vector<int>& cpus) {
#if defined(__linux__) || defined(__ANDROID__)
    if (cpus.empty()) return;
    // Sorted by hand: std::sort on int* links libc++'s prebuilt sort for every scalar type (+60 KB on Android).
    std::vector<int> want(cpus);
    for (size_t i = 1; i < want.size(); ++i)
        for (size_t j = i; j > 0 && want[j - 1] > want[j]; --j) std::swap(want[j - 1], want[j]);
    saved_ = kt_allowed_cpus();
    if (saved_.empty() || saved_ == want) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : want) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    applied_ = sched_setaffinity(0, sizeof set, &set) == 0;
#else
    (void)cpus;
#endif
}

KtAffinityScope::~KtAffinityScope() {
#if defined(__linux__) || defined(__ANDROID__)
    if (!applied_) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : saved_) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof set, &set);
#endif
}
