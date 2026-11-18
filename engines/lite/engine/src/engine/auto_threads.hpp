#pragma once
// Default thread count for phones: what kt08_options.threads = 0 means in the Android and iOS builds of the C library.
// Desktop, server and Python builds never use this (there 0 keeps the OpenMP default).
//
// The engine splits each kernel into equal static shares, so a team that mixes fast and slow cores waits for the
// slowest core at every barrier. Measured with this engine (mini fp32):
//   - big.LITTLE phones: the fast cores win; 2 threads on them is the tested choice (Galaxy A16 2x A76 + 6x A55: RTF
//     0.556 on 2 big cores; Galaxy A73: 0.452 on 2 big cores vs 1.05 on 6 mixed cores).
//   - phones whose cores are all the same design (only Cortex-A53, e.g. LG Stylo 5, Moto G22): in-order cores scale
//     almost linearly, so every core is used, up to 8 (Stylo 5: 1 thread 6.92, 2: 3.40, 4: 1.90, 8: 1.25;
//     Moto G22: 1: 6.69, 2: 3.21, 4: 1.75). The Moto G22 has 4 A53 at 2.3 GHz and 4 at 1.8 GHz: its cores differ in
//     speed but not in design, which is why the design (MIDR part number) is checked, not only the speed.
//
// Inputs come from Linux sysfs (no "topology/core_type" file is read: Linux has none):
//   /sys/devices/system/cpu/cpuN/cpu_capacity              relative speed, 1024 = fastest (most arm64 kernels)
//   /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq  kHz, when cpu_capacity is missing
//   /sys/devices/system/cpu/cpuN/regs/identification/midr_el1  core design (arm64 kernels 4.11+)
//   /proc/cpuinfo "CPU implementer" / "CPU part"           core design, when midr_el1 is missing
// and, on iOS, from sysctl hw.perflevel0.physicalcpu (performance cores) and hw.physicalcpu.
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct KtCoreInfo {
    int id = 0;          // logical CPU number
    long capacity = 0;   // cpu_capacity, else cpuinfo_max_freq (kHz); <= 0: unknown
    unsigned design = 0; // (implementer << 12) | part number from MIDR; 0: unknown
};

struct KtAutoThreads {
    int threads = 0;            // 0: no decision (keep the OpenMP default)
    std::vector<int> cpus;      // cores to keep the work on, fastest first; empty: leave the affinity alone
    bool mixed_speeds = false;  // the chosen cores differ in speed by more than 10% (turns on the dynamic tile split)
    std::string rule;           // "big-cores", "same-design", "same-speed", "single", "apple-performance", "none"
};

constexpr int kKtBigCoreThreads = 2;   // threads on big.LITTLE phones (at most this many fast cores)
constexpr int kKtSameCoreThreads = 8;  // at most this many threads when every core is the same design
constexpr int kKtMixedCoreChunks = 8;  // tile ranges per thread of the dynamic split on mixed-speed cores (tested: 4 < 8)

// The first word of a small text file ("" if it cannot be read). stdio only: this code goes into the phone library.
inline std::string kt_read_word_file(const std::string& path) {
    char buf[64] = {0};
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return std::string();
    const int ok = std::fscanf(f, "%63s", buf);
    std::fclose(f);
    return ok == 1 ? std::string(buf) : std::string();
}

inline long kt_read_long_file(const std::string& path) {
    const std::string s = kt_read_word_file(path);
    if (s.empty()) return -1;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    return end && end != s.c_str() ? v : -1;
}

inline unsigned kt_design_from_midr(unsigned long long midr) {
    const unsigned implementer = (unsigned)((midr >> 24) & 0xff);
    const unsigned part = (unsigned)((midr >> 4) & 0xfff);
    return implementer || part ? (implementer << 12) | part : 0;
}

// "CPU implementer" and "CPU part" per "processor" block of /proc/cpuinfo (arm64 format).
inline std::vector<std::pair<int, unsigned>> kt_designs_from_cpuinfo(const std::string& path) {
    std::vector<std::pair<int, unsigned>> out;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return out;
    char buf[512];
    int cpu = -1;
    unsigned implementer = 0, part = 0;
    auto flush = [&] {
        if (cpu >= 0 && (implementer || part)) out.push_back({cpu, (implementer << 12) | part});
        implementer = part = 0;
    };
    while (std::fgets(buf, sizeof buf, f)) {
        const std::string line(buf, std::strcspn(buf, "\r\n"));
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
        const std::string value = line.substr(colon + 1);
        if (key == "processor") {
            flush();
            cpu = std::atoi(value.c_str());
        } else if (key == "CPU implementer") {
            implementer = (unsigned)std::strtoul(value.c_str(), nullptr, 0) & 0xff;
        } else if (key == "CPU part") {
            part = (unsigned)std::strtoul(value.c_str(), nullptr, 0) & 0xfff;
        }
    }
    flush();
    std::fclose(f);
    return out;
}

// The cores in `allowed`, read from a sysfs tree (sys_root = "/sys" on a device) and a cpuinfo file.
inline std::vector<KtCoreInfo> kt_read_cores(const std::string& sys_root, const std::string& cpuinfo_path,
                                             const std::vector<int>& allowed) {
    std::vector<KtCoreInfo> out;
    std::vector<std::pair<int, unsigned>> from_cpuinfo;
    bool cpuinfo_read = false;
    for (int c : allowed) {
        const std::string base = sys_root + "/devices/system/cpu/cpu" + std::to_string(c);
        KtCoreInfo k;
        k.id = c;
        k.capacity = kt_read_long_file(base + "/cpu_capacity");
        if (k.capacity <= 0) k.capacity = kt_read_long_file(base + "/cpufreq/cpuinfo_max_freq");
        const std::string midr = kt_read_word_file(base + "/regs/identification/midr_el1");  // "0x..."
        if (!midr.empty()) k.design = kt_design_from_midr(std::strtoull(midr.c_str(), nullptr, 0));
        if (!k.design) {
            if (!cpuinfo_read) { from_cpuinfo = kt_designs_from_cpuinfo(cpuinfo_path); cpuinfo_read = true; }
            for (const auto& d : from_cpuinfo) if (d.first == c) k.design = d.second;
        }
        out.push_back(k);
    }
    return out;
}

inline bool kt_speeds_differ(long fast, long slow) { return slow * 10 < fast * 9; }  // more than 10% apart

// Android / Linux rule (see the top of this file).
inline KtAutoThreads kt_auto_threads_linux(std::vector<KtCoreInfo> cores) {
    KtAutoThreads r;
    if (cores.empty()) { r.rule = "none"; return r; }
    for (auto& c : cores) if (c.capacity <= 0) c.capacity = 0;
    std::stable_sort(cores.begin(), cores.end(),
                     [](const KtCoreInfo& a, const KtCoreInfo& b) { return a.capacity > b.capacity; });
    const long fast = cores.front().capacity, slow = cores.back().capacity;
    const bool speeds_known = slow > 0;
    bool designs_known = true, same_design = true;
    for (const auto& c : cores) {
        if (!c.design) designs_known = false;
        else if (c.design != cores.front().design) same_design = false;
    }
    auto take = [&](size_t n, const char* rule) {
        n = std::min(n, cores.size());
        for (size_t i = 0; i < n; ++i) r.cpus.push_back(cores[i].id);
        r.threads = (int)n;
        r.mixed_speeds = n > 1 && kt_speeds_differ(cores.front().capacity, cores[n - 1].capacity);
        r.rule = rule;
    };
    if (cores.size() == 1) { take(1, "single"); return r; }
    if (designs_known && same_design) { take((size_t)kKtSameCoreThreads, "same-design"); return r; }
    if (!speeds_known || !kt_speeds_differ(fast, slow)) { take((size_t)kKtSameCoreThreads, "same-speed"); return r; }
    // big.LITTLE (2 or 3 clusters): the big cores are every core more than 10% faster than the slowest ones.
    size_t big = 0;
    for (const auto& c : cores) if (kt_speeds_differ(c.capacity, slow)) ++big;
    take(std::min<size_t>(big, (size_t)kKtBigCoreThreads), "big-cores");
    return r;
}

// iOS rule: the performance cores, at most 2 (iPhone SE 2020, 2 + 4 cores: RTF 0.122 on 2 threads, 0.155 on 4;
// iPhone 16: 0.065 on 2, 0.076 on 4). No pinning: iOS has no affinity API; its scheduler puts busy threads on P cores.
inline KtAutoThreads kt_auto_threads_apple(int performance_cores, int all_cores) {
    KtAutoThreads r;
    const int n = performance_cores > 0 ? performance_cores : all_cores;
    if (n <= 0) { r.rule = "none"; return r; }
    r.threads = std::min(n, kKtBigCoreThreads);
    r.rule = "apple-performance";
    return r;
}

inline std::string kt_describe_auto_threads(const KtAutoThreads& r) {
    std::string s = "threads=" + std::to_string(r.threads) + " cpus=";
    for (size_t i = 0; i < r.cpus.size(); ++i) s += (i ? "," : "") + std::to_string(r.cpus[i]);
    return s + " mixed=" + (r.mixed_speeds ? "1" : "0") + " rule=" + r.rule;
}
