// kmp_shim.cpp: a small OpenMP runtime with the libomp (LLVM) ABI, on pthreads.
//
// Why: Apple clang compiles OpenMP pragmas (`-Xclang -fopenmp`) into calls to __kmpc_* functions, but iOS ships no
// OpenMP runtime, so Kitten on iPhone was built serial (1 thread). This file implements the subset of __kmpc_* that
// clang emits for the engine's pragmas (parallel, for static/dynamic, barrier, single, sections, num_threads, if).
//
// Lossless by construction: schedule(static) uses libomp's default unchunked split, "balanced" (thread t gets
// trip/nth iterations, the first trip%nth threads one more; measured: greedy ceil-chunks differ from libomp at 4 and 8
// threads, balanced is bit-identical), so every thread computes exactly the iterations it would under libomp at the
// same thread count, and the engine's per-thread tiling (omp_get_thread_num/omp_get_num_threads) sees the same values.
// Verified bit-identical against libomp on Linux x86 and ARM.
//
// Design: one global pool of worker threads (gtid 1..). A fork publishes a Team and bumps a generation counter;
// workers spin (KMP_SHIM_BLOCKTIME_US, default 200000 = libomp's KMP_BLOCKTIME 200 ms) then sleep on a condvar.
// Nested parallel regions run serialized (libomp default max-active-levels behaviour for this engine, which guards
// nesting with omp_in_parallel anyway). Barrier = sense-reversing counter. Dynamic loops use a ring of 8 shared
// dispatch buffers (as libomp), so consecutive `nowait` loops can overlap safely.
#include "../../../engine/dev_env.hpp"
#include <atomic>
#include <utility>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <limits>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <sched.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

typedef int32_t kmp_int32;
typedef uint32_t kmp_uint32;
typedef int64_t kmp_int64;
typedef uint64_t kmp_uint64;
struct ident_t;
typedef void (*kmpc_micro)(kmp_int32*, kmp_int32*, ...);

namespace {

constexpr int kMaxArgs = 24;
constexpr int kNDisp = 8;
constexpr int kMaxThreads = 256;

enum {   // libomp sched_type values (kmp.h)
    sch_static_chunked = 33, sch_static = 34, sch_dynamic_chunked = 35, sch_guided_chunked = 36, sch_runtime = 37,
    sch_auto = 38, sch_trapezoidal = 39, sch_static_greedy = 40, sch_static_balanced = 41, sch_guided_iterative = 42,
    sch_guided_analytical = 43, sch_static_steal = 44,
    ord_static_chunked = 65, ord_static = 66, ord_dynamic_chunked = 67, ord_guided_chunked = 68, ord_runtime = 69,
    ord_auto = 70,
};
static inline int sched_base(int s) { return s & ~((1 << 29) | (1 << 30) | (1 << 31)); }   // drop (non)monotonic bits

static inline void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}
static inline double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class Pred> static inline void spin_until(Pred p) {
    for (int i = 0; !p(); ++i) { if (i < 4096) cpu_relax(); else sched_yield(); }
}

struct Disp {                          // one shared dynamic-loop descriptor
    std::atomic<int64_t> inst{-1};     // loop instance using this slot (-1 none, -2 being initialised)
    std::atomic<int64_t> next{0};      // next unclaimed iteration index (0..trip)
    std::atomic<int> finished{0};      // threads that saw the end
    int64_t lb = 0, st = 1, trip = 0, chunk = 1;
    int sched = sch_dynamic_chunked;
};

struct Team {
    int nth = 1;
    kmpc_micro fn = nullptr;
    int argc = 0;
    void* args[kMaxArgs];
    bool args_as_one = false;          // __kmpc_fork_call_if passes one pointer
    std::atomic<int> pending{0};       // workers still running the microtask
    std::atomic<int> bar_count{0};
    std::atomic<int> bar_sense{0};
    std::atomic<int64_t> single_claimed{0};
    Disp disp[kNDisp];
};

struct LocalDisp { int64_t lb = 0, st = 1, trip = 0, chunk = 1, next = 0; int sched = 0; bool static_given = false; };

struct Frame {                         // one (possibly serialized) parallel level on this thread
    Team* team = nullptr;              // nullptr for serialized / outermost
    int tid = 0, nth = 1;
    int64_t single_cnt = 0, disp_cnt = 0;
    int local_sense = 0;
    Disp* cur = nullptr;               // shared dispatch slot of the running dynamic loop (team only)
    int64_t cur_inst = -1;
    int cur_nth = 1;
    bool static_disp_done = false;     // static schedule served through dispatch: one chunk per thread
    LocalDisp ld;                      // dynamic loop state when nth == 1
};

struct ThreadState {
    int gtid = -1;
    int push_nt = 0;                   // num_threads clause for the next fork
    int active_levels = 0;             // enclosing parallel regions with nth > 1
    std::vector<Frame> frames;         // frames[0] = implicit outer level
    ThreadState() { frames.reserve(8); frames.emplace_back(); }
};
thread_local ThreadState ts;

std::atomic<int> g_next_gtid{1};       // 0 = the first thread that touches the runtime
std::atomic<int> g_nthreads_var{0};
std::atomic<int> g_dynamic{0};
std::atomic<int> g_max_active_levels{1};
std::mutex g_crit;

int default_threads() {
    if (const char* e = std::getenv("OMP_NUM_THREADS")) { int v = std::atoi(e); if (v > 0) return std::min(v, kMaxThreads); }
    unsigned h = std::thread::hardware_concurrency();
    return h ? (int)std::min<unsigned>(h, kMaxThreads) : 1;
}
int nthreads_var() {
    int v = g_nthreads_var.load(std::memory_order_relaxed);
    if (v <= 0) { v = default_threads(); g_nthreads_var.store(v); }
    return v;
}
int my_gtid() {
    if (ts.gtid < 0) { static std::atomic<bool> first{true}; ts.gtid = first.exchange(false) ? 0 : g_next_gtid.fetch_add(1); }
    return ts.gtid;
}
// libomp's default unchunked static split. KMP_SHIM_STATIC=greedy selects ceil-sized chunks instead.
bool static_balanced() {
    static const bool v = [] { const char* e = kt_dev_getenv("KMP_SHIM_STATIC"); return !(e && std::string(e) == "greedy"); }();
    return v;
}
long blocktime_us() {
    static const long v = [] { const char* e = kt_dev_getenv("KMP_SHIM_BLOCKTIME_US"); return e ? std::atol(e) : 200000L; }();
    return v;
}

// The outlined body is a NON-variadic function void(kmp_int32*, kmp_int32*, void*...) that the compiler only declares
// through the variadic kmpc_micro type. Calling it through that variadic type is wrong on Apple arm64: Darwin's ABI
// passes every variadic argument on the stack, while the callee reads its fixed arguments from x2..x7, so it saw garbage
// (EXC_BAD_ACCESS at 0x7 inside attention_flash_fp32.omp_outlined on iPhone SE 2020 and iPhone 7). Linux AAPCS64 and x86-64 SysV pass variadic and fixed arguments the same way, which hid it there.
// libomp avoids this with an assembly __kmp_invoke_microtask; here each arity casts to the exact non-variadic type.
template <int N, typename... P> struct Micro { using F = typename Micro<N - 1, void*, P...>::F; };
template <typename... P> struct Micro<0, P...> { using F = void (*)(kmp_int32*, kmp_int32*, P...); };
template <int N, size_t... I> static inline void call_n(kmpc_micro fn, kmp_int32* g, kmp_int32* b, void** a, std::index_sequence<I...>) {
    reinterpret_cast<typename Micro<N>::F>(fn)(g, b, a[I]...);
}
template <int N> static inline void call(kmpc_micro fn, kmp_int32* g, kmp_int32* b, void** a) {
    call_n<N>(fn, g, b, a, std::make_index_sequence<N>{});
}

void invoke(kmpc_micro fn, kmp_int32* g, kmp_int32* b, int argc, void** a) {
    switch (argc) {
#define C(n) case n: call<n>(fn, g, b, a); break;
    C(0) C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8) C(9) C(10) C(11) C(12)
    C(13) C(14) C(15) C(16) C(17) C(18) C(19) C(20) C(21) C(22) C(23) C(24)
#undef C
    default: fprintf(stderr, "kmp_shim: microtask with %d args unsupported\n", argc); abort();
    }
}

// ---------------- worker pool ----------------
// Each worker waits on its own generation slot, written only by the master for the workers that join that team, so a
// worker never mistakes a newer team for the one it was woken for.
struct alignas(64) Slot { std::atomic<uint64_t> g{0}; };
struct Pool {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> sleepers{0};
    uint64_t gen = 0;                    // master-only counter
    Team* team = nullptr;
    std::vector<std::thread> workers;    // placeholders (threads are detached); size = worker count
    std::atomic<bool> busy{false};       // a top-level team is running
    Team the_team;
    Slot slot[kMaxThreads];
} g_pool;

void run_member(Team* t, int tid) {
    Frame f; f.team = t; f.tid = tid; f.nth = t->nth;
    ts.frames.push_back(f);
    ts.active_levels++;
    kmp_int32 g = my_gtid(), b = tid;
    if (t->args_as_one) invoke(t->fn, &g, &b, t->args[0] ? 1 : 0, t->args);
    else invoke(t->fn, &g, &b, t->argc, t->args);
    ts.active_levels--;
    ts.frames.pop_back();
}

void worker_main(int widx) {
    ts.gtid = widx + 1;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    std::atomic<uint64_t>& my = g_pool.slot[widx].g;
    uint64_t done = 0;
    for (;;) {
        double t0 = now_us(); int i = 0;
        while (my.load(std::memory_order_acquire) == done) {
            if (++i < 4096) { cpu_relax(); continue; }
            sched_yield();
            if ((i & 255) == 0 && now_us() - t0 > blocktime_us()) {
                std::unique_lock<std::mutex> lk(g_pool.mu);
                g_pool.sleepers.fetch_add(1, std::memory_order_seq_cst);
                g_pool.cv.wait(lk, [&] { return my.load(std::memory_order_seq_cst) != done; });
                g_pool.sleepers.fetch_sub(1, std::memory_order_seq_cst);
                break;
            }
        }
        done = my.load(std::memory_order_acquire);
        Team* t = g_pool.team;
        run_member(t, widx + 1);
        t->pending.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ensure_workers(int n) {   // n = workers needed (team size - 1)
    while ((int)g_pool.workers.size() < n) {
        int w = (int)g_pool.workers.size();
        // 8 MiB stacks (libomp's default worker stack is 4 MiB on 64-bit; the engine keeps small stack buffers).
        pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 8u << 20);
        pthread_t th;
        auto* arg = new int(w);
        pthread_create(&th, &at, [](void* p) -> void* { int w = *(int*)p; delete (int*)p; worker_main(w); return nullptr; }, arg);
        pthread_attr_destroy(&at);
        pthread_detach(th);
        g_pool.workers.emplace_back();   // placeholder (thread detached; vector size = worker count)
    }
}

void run_serialized(kmpc_micro fn, int argc, void** args, bool as_one) {
    Frame f; f.nth = 1; f.tid = 0; f.team = nullptr;
    ts.frames.push_back(f);
    kmp_int32 g = my_gtid(), b = 0;
    if (as_one) invoke(fn, &g, &b, args[0] ? 1 : 0, args); else invoke(fn, &g, &b, argc, args);
    ts.frames.pop_back();
}

void fork(kmpc_micro fn, int argc, void** args, bool as_one) {
    my_gtid();
    int n = ts.push_nt > 0 ? ts.push_nt : nthreads_var();
    ts.push_nt = 0;
    n = std::min(n, kMaxThreads);
    bool expected = false;
    if (n <= 1 || ts.active_levels >= g_max_active_levels.load() || !g_pool.busy.compare_exchange_strong(expected, true)) {
        run_serialized(fn, argc, args, as_one);
        return;
    }
    ensure_workers(n - 1);
    Team* t = &g_pool.the_team;
    t->nth = n; t->fn = fn; t->argc = argc; t->args_as_one = as_one;
    for (int i = 0; i < argc && i < kMaxArgs; ++i) t->args[i] = args[i];
    if (as_one) t->args[0] = args[0];
    t->pending.store(n - 1, std::memory_order_relaxed);
    t->bar_count.store(0, std::memory_order_relaxed);
    t->bar_sense.store(0, std::memory_order_relaxed);
    t->single_claimed.store(0, std::memory_order_relaxed);
    for (auto& d : t->disp) { d.inst.store(-1, std::memory_order_relaxed); d.finished.store(0, std::memory_order_relaxed); }
    g_pool.team = t;
    const uint64_t G = ++g_pool.gen;
    for (int w = 0; w < n - 1; ++w) g_pool.slot[w].g.store(G, std::memory_order_seq_cst);
    if (g_pool.sleepers.load(std::memory_order_seq_cst) > 0) {
        { std::lock_guard<std::mutex> lk(g_pool.mu); }   // a sleeper is either before its predicate check or waiting
        g_pool.cv.notify_all();
    }
    run_member(t, 0);
    spin_until([&] { return t->pending.load(std::memory_order_acquire) == 0; });
    g_pool.busy.store(false, std::memory_order_release);
}

Frame& cur() { return ts.frames.back(); }

void team_barrier() {
    Frame& f = cur();
    if (f.nth <= 1 || !f.team) return;
    Team* t = f.team;
    int s = f.local_sense ^ 1; f.local_sense = s;
    if (t->bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == f.nth) {
        t->bar_count.store(0, std::memory_order_relaxed);
        t->bar_sense.store(s, std::memory_order_release);
    } else {
        spin_until([&] { return t->bar_sense.load(std::memory_order_acquire) == s; });
    }
}

// libomp __kmp_for_static_init: kmp_sch_static (balanced = libomp default, or greedy) and kmp_sch_static_chunked.
template <typename T, typename UT, typename ST>
void static_init(int schedtype, kmp_int32* plast, T* plower, T* pupper, ST* pstride, ST incr, ST chunk) {
    Frame& f = cur();
    const int nth = f.nth, tid = f.tid;
    if (plast) *plast = 0;
    UT trip;
    if (incr == 1) trip = (UT)(*pupper - *plower) + 1;
    else if (incr == -1) trip = (UT)(*plower - *pupper) + 1;
    else if (incr > 0) trip = (UT)(*pupper - *plower) / incr + 1;
    else trip = (UT)(*plower - *pupper) / (UT)(-incr) + 1;
    if ((incr > 0 && *pupper < *plower) || (incr < 0 && *pupper > *plower)) {   // zero-trip loop
        if (plast) *plast = 0;
        *pstride = incr;
        return;
    }
    if (nth == 1) {
        if (plast) *plast = 1;
        *pstride = (incr > 0) ? (*pupper - *plower + 1) : (-(*plower - *pupper + 1));
        return;
    }
    int s = sched_base(schedtype);
    if (s == sch_static || s == sch_static_greedy || s == sch_static_balanced || s == ord_static) {
        if (trip < (UT)nth) {
            if ((UT)tid < trip) *pupper = *plower = *plower + (T)tid * incr;
            else *plower = *pupper + incr;                       // empty range for this thread
            if (plast) *plast = ((UT)tid == trip - 1);
        } else if (s == sch_static_balanced || (s != sch_static_greedy && static_balanced())) {
            UT small = trip / nth, extras = trip % nth;
            *plower += incr * (T)(tid * small + ((UT)tid < extras ? tid : extras));
            *pupper = *plower + (T)small * incr - ((UT)tid < extras ? 0 : incr);
            if (plast) *plast = (tid == nth - 1);
        } else {                                                  // greedy (KMP_SHIM_STATIC=greedy)
            T big = (T)(trip / nth + ((trip % nth) ? 1 : 0)) * incr;
            T old_upper = *pupper;
            *plower += (T)tid * big;
            *pupper = *plower + big - incr;
            if (incr > 0) {
                if (*pupper < *plower) *pupper = std::numeric_limits<T>::max();
                if (plast) *plast = *plower <= old_upper && *pupper > old_upper - incr;
                if (*pupper > old_upper) *pupper = old_upper;
            } else {
                if (*pupper > *plower) *pupper = std::numeric_limits<T>::min();
                if (plast) *plast = *plower >= old_upper && *pupper < old_upper - incr;
                if (*pupper < old_upper) *pupper = old_upper;
            }
        }
        *pstride = (ST)trip;
    } else {                                                      // static chunked
        if (chunk < 1) chunk = 1;
        ST span = chunk * incr;
        *pstride = span * nth;
        *plower = *plower + (T)(span * tid);
        *pupper = *plower + span - incr;
        if (plast) *plast = ((UT)tid == ((trip - 1) / (UT)chunk) % (UT)nth);
    }
}

template <typename T>
void dispatch_init(int schedule, T lb, T ub, T st, T chunk) {
    Frame& f = cur();
    int s = sched_base(schedule);
    int64_t trip;
    if (st > 0) trip = (ub < lb) ? 0 : (int64_t)((ub - lb) / st) + 1;
    else trip = (ub > lb) ? 0 : (int64_t)((lb - ub) / (-st)) + 1;
    f.static_disp_done = false;
    if (f.nth <= 1 || !f.team) {
        f.ld.lb = lb; f.ld.st = st; f.ld.trip = trip; f.ld.chunk = chunk < 1 ? 1 : chunk; f.ld.next = 0; f.ld.sched = s;
        f.cur = nullptr;
        return;
    }
    Team* t = f.team;
    int64_t k = f.disp_cnt++;
    Disp& d = t->disp[k % kNDisp];
    for (int i = 0;; ++i) {
        int64_t c = d.inst.load(std::memory_order_acquire);
        if (c == k) break;
        bool reusable = (c == -1 && k < kNDisp) || (c == k - kNDisp && d.finished.load(std::memory_order_acquire) >= f.nth);
        if (reusable && d.inst.compare_exchange_strong(c, -2, std::memory_order_acq_rel)) {
            d.lb = lb; d.st = st; d.trip = trip; d.chunk = chunk < 1 ? 1 : chunk; d.sched = s;
            d.next.store(0, std::memory_order_relaxed);
            d.finished.store(0, std::memory_order_relaxed);
            d.inst.store(k, std::memory_order_release);
            break;
        }
        if (i < 4096) cpu_relax(); else sched_yield();
    }
    f.cur = &d; f.cur_inst = k; f.cur_nth = f.nth;
}

template <typename T>
int dispatch_next(kmp_int32* plast, T* plb, T* pub, T* pst) {
    Frame& f = cur();
    if (!f.cur) {   // serialized: hand out the whole range once
        LocalDisp& l = f.ld;
        if (l.next >= l.trip) return 0;
        *plb = (T)l.lb; *pub = (T)(l.lb + (l.trip - 1) * l.st); if (pst) *pst = (T)l.st; if (plast) *plast = 1;
        l.next = l.trip;
        return 1;
    }
    Disp& d = *f.cur;
    int s = d.sched;
    if (s == sch_static || s == sch_static_chunked || s == ord_static || s == ord_static_chunked || s == sch_static_greedy ||
        s == sch_static_balanced) {
        // static schedule served through dispatch (e.g. ordered): one greedy chunk per thread, as static_init.
        if (f.static_disp_done || d.trip == 0) goto finish;
        {
            f.static_disp_done = true;
            int64_t lo = 0, hi = d.trip - 1, stride; kmp_int32 last = 0;
            static_init<int64_t, uint64_t, int64_t>(s == sch_static_chunked || s == ord_static_chunked ? sch_static_chunked : sch_static,
                                                    &last, &lo, &hi, &stride, 1, d.chunk);
            if (lo > hi || lo >= d.trip) goto finish;
            if (s == sch_static_chunked || s == ord_static_chunked) {   // static_chunked: claim chunks round robin
                // rare in this engine; fall back to dynamic-style claiming of whole chunks (iteration ownership differs,
                // results are independent of ownership for loops without thread-indexed state)
                f.static_disp_done = false;
                goto dynamic;
            }
            *plb = (T)(d.lb + lo * d.st); *pub = (T)(d.lb + hi * d.st); if (pst) *pst = (T)d.st; if (plast) *plast = last;
            return 1;
        }
    }
dynamic:
    {
        int64_t c = d.chunk;
        if (s == sch_guided_chunked || s == sch_guided_iterative || s == sch_guided_analytical) {
            int64_t rem = d.trip - d.next.load(std::memory_order_relaxed);
            c = std::max<int64_t>(d.chunk, rem / (2 * (int64_t)f.cur_nth));
        }
        int64_t i = d.next.fetch_add(c, std::memory_order_acq_rel);
        if (i < d.trip) {
            int64_t e = std::min(i + c, d.trip);
            *plb = (T)(d.lb + i * d.st); *pub = (T)(d.lb + (e - 1) * d.st); if (pst) *pst = (T)d.st;
            if (plast) *plast = (e == d.trip);
            return 1;
        }
    }
finish:
    d.finished.fetch_add(1, std::memory_order_acq_rel);
    f.cur = nullptr;
    return 0;
}

}  // namespace

extern "C" {

// ---- OpenMP API ----
void omp_set_num_threads(int n) { g_nthreads_var.store(n > 0 ? std::min(n, kMaxThreads) : 1); }
int omp_get_num_threads(void) { return cur().nth; }
int omp_get_max_threads(void) { return nthreads_var(); }
int omp_get_thread_num(void) { return cur().tid; }
int omp_get_num_procs(void) { unsigned h = std::thread::hardware_concurrency(); return h ? (int)h : 1; }
int omp_in_parallel(void) { return ts.active_levels > 0; }
void omp_set_dynamic(int on) { g_dynamic.store(on); }
int omp_get_dynamic(void) { return g_dynamic.load(); }
int omp_get_level(void) { return (int)ts.frames.size() - 1; }
int omp_get_active_level(void) { return ts.active_levels; }
void omp_set_max_active_levels(int n) { g_max_active_levels.store(n < 1 ? 1 : n); }
int omp_get_max_active_levels(void) { return g_max_active_levels.load(); }
int omp_get_thread_limit(void) { return kMaxThreads; }
double omp_get_wtime(void) { return now_us() * 1e-6; }

// ---- compiler entry points ----
kmp_int32 __kmpc_global_thread_num(ident_t*) { return my_gtid(); }
kmp_int32 __kmpc_global_num_threads(ident_t*) { return (kmp_int32)g_pool.workers.size() + 1; }
kmp_int32 __kmpc_bound_thread_num(ident_t*) { return cur().tid; }
kmp_int32 __kmpc_bound_num_threads(ident_t*) { return cur().nth; }
kmp_int32 __kmpc_in_parallel(ident_t*) { return ts.active_levels > 0; }
kmp_int32 __kmpc_ok_to_fork(ident_t*) { return 1; }
void __kmpc_begin(ident_t*, kmp_int32) {}
void __kmpc_end(ident_t*) {}
void __kmpc_flush(ident_t*) { std::atomic_thread_fence(std::memory_order_seq_cst); }
void __kmpc_push_num_threads(ident_t*, kmp_int32, kmp_int32 n) { ts.push_nt = n; }
void __kmpc_push_proc_bind(ident_t*, kmp_int32, int) {}

void __kmpc_fork_call(ident_t*, kmp_int32 argc, kmpc_micro fn, ...) {
    void* a[kMaxArgs];
    if (argc > kMaxArgs) { fprintf(stderr, "kmp_shim: fork with %d args\n", argc); abort(); }
    va_list ap; va_start(ap, fn);
    for (int i = 0; i < argc; ++i) a[i] = va_arg(ap, void*);
    va_end(ap);
    fork(fn, argc, a, false);
}
void __kmpc_fork_call_if(ident_t*, kmp_int32 argc, kmpc_micro fn, kmp_int32 cond, void* args) {
    void* a[kMaxArgs]; a[0] = args; (void)argc;
    if (cond) fork(fn, 1, a, true); else { ts.push_nt = 0; run_serialized(fn, 1, a, true); }
}
void __kmpc_serialized_parallel(ident_t*, kmp_int32) {
    ts.push_nt = 0;
    Frame f; f.nth = 1; f.tid = 0; f.team = nullptr;
    ts.frames.push_back(f);
}
void __kmpc_end_serialized_parallel(ident_t*, kmp_int32) { if (ts.frames.size() > 1) ts.frames.pop_back(); }

void __kmpc_barrier(ident_t*, kmp_int32) { team_barrier(); }

void __kmpc_for_static_init_4(ident_t*, kmp_int32, kmp_int32 s, kmp_int32* pl, kmp_int32* lo, kmp_int32* up, kmp_int32* st, kmp_int32 inc, kmp_int32 ch) {
    static_init<kmp_int32, kmp_uint32, kmp_int32>(s, pl, lo, up, st, inc, ch);
}
void __kmpc_for_static_init_4u(ident_t*, kmp_int32, kmp_int32 s, kmp_int32* pl, kmp_uint32* lo, kmp_uint32* up, kmp_int32* st, kmp_int32 inc, kmp_int32 ch) {
    static_init<kmp_uint32, kmp_uint32, kmp_int32>(s, pl, lo, up, st, inc, ch);
}
void __kmpc_for_static_init_8(ident_t*, kmp_int32, kmp_int32 s, kmp_int32* pl, kmp_int64* lo, kmp_int64* up, kmp_int64* st, kmp_int64 inc, kmp_int64 ch) {
    static_init<kmp_int64, kmp_uint64, kmp_int64>(s, pl, lo, up, st, inc, ch);
}
void __kmpc_for_static_init_8u(ident_t*, kmp_int32, kmp_int32 s, kmp_int32* pl, kmp_uint64* lo, kmp_uint64* up, kmp_int64* st, kmp_int64 inc, kmp_int64 ch) {
    static_init<kmp_uint64, kmp_uint64, kmp_int64>(s, pl, lo, up, st, inc, ch);
}
void __kmpc_for_static_fini(ident_t*, kmp_int32) {}

void __kmpc_dispatch_init_4(ident_t*, kmp_int32, int s, kmp_int32 lb, kmp_int32 ub, kmp_int32 st, kmp_int32 ch) { dispatch_init<kmp_int64>(s, lb, ub, st, ch); }
void __kmpc_dispatch_init_4u(ident_t*, kmp_int32, int s, kmp_uint32 lb, kmp_uint32 ub, kmp_int32 st, kmp_int32 ch) { dispatch_init<kmp_int64>(s, lb, ub, st, ch); }
void __kmpc_dispatch_init_8(ident_t*, kmp_int32, int s, kmp_int64 lb, kmp_int64 ub, kmp_int64 st, kmp_int64 ch) { dispatch_init<kmp_int64>(s, lb, ub, st, ch); }
void __kmpc_dispatch_init_8u(ident_t*, kmp_int32, int s, kmp_uint64 lb, kmp_uint64 ub, kmp_int64 st, kmp_int64 ch) { dispatch_init<kmp_int64>(s, (kmp_int64)lb, (kmp_int64)ub, st, ch); }
int __kmpc_dispatch_next_4(ident_t*, kmp_int32, kmp_int32* pl, kmp_int32* lb, kmp_int32* ub, kmp_int32* st) { return dispatch_next<kmp_int32>(pl, lb, ub, st); }
int __kmpc_dispatch_next_4u(ident_t*, kmp_int32, kmp_int32* pl, kmp_uint32* lb, kmp_uint32* ub, kmp_int32* st) {
    kmp_int32 l, u, s; int r = dispatch_next<kmp_int32>(pl, &l, &u, &s); if (r) { *lb = (kmp_uint32)l; *ub = (kmp_uint32)u; if (st) *st = s; } return r;
}
int __kmpc_dispatch_next_8(ident_t*, kmp_int32, kmp_int32* pl, kmp_int64* lb, kmp_int64* ub, kmp_int64* st) { return dispatch_next<kmp_int64>(pl, lb, ub, st); }
int __kmpc_dispatch_next_8u(ident_t*, kmp_int32, kmp_int32* pl, kmp_uint64* lb, kmp_uint64* ub, kmp_int64* st) {
    kmp_int64 l, u, s; int r = dispatch_next<kmp_int64>(pl, &l, &u, &s); if (r) { *lb = (kmp_uint64)l; *ub = (kmp_uint64)u; if (st) *st = s; } return r;
}
void __kmpc_dispatch_fini_4(ident_t*, kmp_int32) {}
void __kmpc_dispatch_fini_4u(ident_t*, kmp_int32) {}
void __kmpc_dispatch_fini_8(ident_t*, kmp_int32) {}
void __kmpc_dispatch_fini_8u(ident_t*, kmp_int32) {}
void __kmpc_dispatch_deinit(ident_t*, kmp_int32) {}

kmp_int32 __kmpc_single(ident_t*, kmp_int32) {
    Frame& f = cur();
    if (f.nth <= 1 || !f.team) return 1;
    int64_t c = f.single_cnt++;
    int64_t e = c;
    return f.team->single_claimed.compare_exchange_strong(e, c + 1, std::memory_order_acq_rel) ? 1 : 0;
}
void __kmpc_end_single(ident_t*, kmp_int32) {}
kmp_int32 __kmpc_master(ident_t*, kmp_int32) { return cur().tid == 0; }
void __kmpc_end_master(ident_t*, kmp_int32) {}
kmp_int32 __kmpc_masked(ident_t*, kmp_int32, kmp_int32 filter) { return cur().tid == filter; }
void __kmpc_end_masked(ident_t*, kmp_int32) {}
void __kmpc_critical(ident_t*, kmp_int32, void*) { g_crit.lock(); }
void __kmpc_end_critical(ident_t*, kmp_int32, void*) { g_crit.unlock(); }
void __kmpc_critical_with_hint(ident_t*, kmp_int32, void*, uint32_t) { g_crit.lock(); }
// Reductions: critical-section method (return 1 = this thread combines under the lock).
kmp_int32 __kmpc_reduce_nowait(ident_t*, kmp_int32, kmp_int32, size_t, void*, void (*)(void*, void*), void*) { g_crit.lock(); return 1; }
void __kmpc_end_reduce_nowait(ident_t*, kmp_int32, void*) { g_crit.unlock(); }
kmp_int32 __kmpc_reduce(ident_t*, kmp_int32, kmp_int32, size_t, void*, void (*)(void*, void*), void*) { g_crit.lock(); return 1; }
void __kmpc_end_reduce(ident_t*, kmp_int32, void*) { g_crit.unlock(); team_barrier(); }

}  // extern "C"
