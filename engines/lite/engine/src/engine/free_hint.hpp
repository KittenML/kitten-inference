// src/engine/free_hint.hpp — give back the pages of a large block that is about to be freed.
// libmalloc keeps the last 16 freed large blocks cached and dirty ("Malloc Large (empty)" in vmmap, ~104 MiB after load
// with the fp16 store); they stay resident and in phys_footprint (what iOS jetsam counts).
//   KITTEN_FREE_HINT=1: madvise(MADV_FREE_REUSABLE). Tried and dropped: resident memory is unchanged and phys_footprint
//     under-counts afterwards (a reused block is not re-charged: 64 MiB after load with 139 MiB of live weights).
//   KITTEN_FREE_HINT=2: replace the block's whole pages with fresh zero-fill pages (mmap MAP_FIXED|MAP_ANON) just
//     before free(): the old pages are released at once; libmalloc keeps the address range, later use faults in new
//     pages that are charged normally. Values are never affected (the block is dead).
// Apple only (Linux: glibc maps such blocks itself and the malloc trim gives memory back). Default: 2 when the mmap allocator
// (KITTEN_MMAP_ALLOC_MIN) is on, i.e. the Apple low-memory configuration; otherwise 0.
#pragma once
#include "engine/dev_env.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

inline int kitten_free_hint_mode() {
    static const int v = [] {
        if (const char* e = kt_dev_getenv("KITTEN_FREE_HINT")) return e[0] ? std::atoi(e) : 0;
        const char* m = kt_dev_getenv("KITTEN_MMAP_ALLOC_MIN");
        return m && std::strtoull(m, nullptr, 10) > 0 ? 2 : 0;
    }();
    return v;
}

inline void kitten_free_hint(const void* p, size_t n) {
#if defined(__APPLE__) && defined(MADV_FREE_REUSABLE)
    const int mode = (p && n >= ((size_t)256 << 10)) ? kitten_free_hint_mode() : 0;
    if (!mode) return;
    const uintptr_t pg = (uintptr_t)getpagesize();
    const uintptr_t a = ((uintptr_t)p + pg - 1) & ~(pg - 1), b = ((uintptr_t)p + n) & ~(pg - 1);
    if (b <= a) return;
    if (mode == 1) ::madvise((void*)a, (size_t)(b - a), MADV_FREE_REUSABLE);
    else ::mmap((void*)a, (size_t)(b - a), PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
#else
    (void)p; (void)n;
#endif
}

template <class T> inline void kitten_hint_vector(const std::vector<T>& v) { kitten_free_hint(v.data(), v.capacity() * sizeof(T)); }

// For a malloc'd block whose size the caller does not know (packed-weight deleters): malloc_size on Apple.
inline void kitten_free_hint_block(const void* p) {
#if defined(__APPLE__)
    if (p && kitten_free_hint_mode()) kitten_free_hint(p, malloc_size(p));
#else
    (void)p;
#endif
}
// delete[] for packed float buffers, with the hint first.
inline void kitten_delete_floats(void* p) { kitten_free_hint_block(p); delete[] static_cast<float*>(p); }

// hint the listed vectors when the scope ends. Declare it after them so it runs before their destructors.
template <class... T> struct KittenHintAtExit {
    std::tuple<const std::vector<T>&...> v;
    explicit KittenHintAtExit(const std::vector<T>&... x) : v(x...) {}
    ~KittenHintAtExit() { std::apply([](const auto&... x) { (kitten_hint_vector(x), ...); }, v); }
    KittenHintAtExit(const KittenHintAtExit&) = delete;
    KittenHintAtExit& operator=(const KittenHintAtExit&) = delete;
};
