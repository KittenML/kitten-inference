#pragma once
// Big-core pinning and the big.LITTLE tile split for phones.
//
// Phone CPUs mix fast and slow cores (for example 2x Cortex-A76 + 6x A55 on a Helio G99). Linux reports each core's
// relative speed in /sys/devices/system/cpu/cpuN/cpu_capacity (1024 = the fastest); older kernels only give
// cpufreq/cpuinfo_max_freq. The engine's OpenMP kernels split work into equal static shares, so a team that mixes
// fast and slow cores waits for the slowest core at every barrier.
//
// kt_pin_fastest_cores(n) restricts the calling thread (and every thread it creates later, including the OpenMP
// team) to the n fastest cores. It must be called before the first OpenMP call of the process, because the OpenMP
// runtime reads the affinity mask once, when it starts. If the n cores it picked still differ in speed (n larger
// than the number of big cores), it also turns on the dynamic tile split (ops/work_split.hpp,
// kt_dynamic_chunks_setting), so a slow core takes fewer tiles; the split never changes a value (each tile is still
// computed whole by one thread), only which thread computes it.
#include "auto_threads.hpp"
#include <string>
#include <vector>

struct KtCpu {
    int id;
    long capacity;  // cpu_capacity, or max frequency in kHz when the kernel has no cpu_capacity
};

// Online CPUs the process may run on, fastest first (ties: lower id first).
std::vector<KtCpu> kt_cpus_by_speed();

struct KtPinResult {
    bool pinned = false;           // sched_setaffinity succeeded
    std::vector<int> cpus;         // the cores picked
    bool heterogeneous = false;    // the picked cores differ in capacity by more than 10%
    int dynamic_chunks = 0;        // value given to kt_dynamic_chunks_setting (0: static split)
    std::string describe() const;  // e.g. "cpus=6,7 caps=1024,1024 hetero=0 dyn=0"
    std::vector<long> caps;
};

// dynamic_chunks: tiles per thread for the dynamic split when the picked cores are mixed (-1: 4, the engine's tested
// value; 0: never turn it on).
KtPinResult kt_pin_fastest_cores(int n, int dynamic_chunks = -1);

// The CPUs the calling thread may run on, in id order (Linux/Android; empty elsewhere).
std::vector<int> kt_allowed_cpus();

// The phone default for kt08_options.threads = 0 (auto_threads.hpp), from this device: sysfs and /proc/cpuinfo on
// Android/Linux, sysctl on Apple platforms. Tests may point it at a fake tree with KITTEN_TTS08_SYS_ROOT and
// KITTEN_TTS08_CPUINFO.
KtAutoThreads kt_auto_threads_here();

// On mixed-speed cores with the dynamic split on (kt_dynamic_chunks() > 0): the rank-th fastest core the process may run on, as a one-element
// list for KtAffinityScope; empty otherwise (same-speed cores, rank beyond the cores, off Linux). The list is read once. Used for the work that
// cannot be split over threads (the two LSTM scans), so it runs on the fastest cores instead of wherever the scheduler put the thread.
std::vector<int> kt_fast_core_for_rank(int rank);

// Restricts the calling thread to `cpus` for the lifetime of the scope, then gives it back its previous CPU set, so
// the app's thread is not changed after the call. Threads created inside the scope (the OpenMP team, the first time)
// keep the restricted set. No-op when cpus is empty, when the thread already has exactly that set, or off Linux.
class KtAffinityScope {
public:
    explicit KtAffinityScope(const std::vector<int>& cpus);
    ~KtAffinityScope();
    KtAffinityScope(const KtAffinityScope&) = delete;
    KtAffinityScope& operator=(const KtAffinityScope&) = delete;
    bool applied() const { return applied_; }

private:
    bool applied_ = false;
    std::vector<int> saved_;
};
