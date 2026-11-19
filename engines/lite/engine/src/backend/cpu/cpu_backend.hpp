// src/backend/cpu/cpu_backend.hpp — CpuAllocator definition, shared by cpu_backend.cpp and model.cpp
#pragma once
#include "engine/dev_env.hpp"
#include "backend/backend.hpp"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

// KITTEN_MMAP_ALLOC_MIN=<bytes> sends every host buffer of at least that size straight to mmap/munmap,
// so a freed activation goes back to the OS at once. macOS libmalloc otherwise keeps freed large blocks cached (counted
// in phys_footprint; MallocLargeCache=0 shows the effect, but an app cannot set libmalloc env vars). Values untouched.
inline size_t kitten_mmap_alloc_min() {
    static const size_t v = [] { const char* e = kt_dev_getenv("KITTEN_MMAP_ALLOC_MIN"); return e ? (size_t)std::strtoull(e, nullptr, 10) : (size_t)0; }();
    return v;
}

struct CpuAllocator : Allocator {
    MemSpace mem_space()  const override { return MemSpace::Host; }
    void* alloc(size_t n) override {
#if defined(__unix__) || defined(__APPLE__)
        const size_t mn = kitten_mmap_alloc_min();
        if (mn && n >= mn) {
            void* p = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (p != MAP_FAILED) { std::lock_guard<std::mutex> g(mu_); mapped_[p] = n; return p; }
        }
#endif
        return ::malloc(n);
    }
    void  free (void* p)  override {
#if defined(__unix__) || defined(__APPLE__)
        if (kitten_mmap_alloc_min() && p) {
            std::lock_guard<std::mutex> g(mu_);
            auto it = mapped_.find(p);
            if (it != mapped_.end()) { ::munmap(p, it->second); mapped_.erase(it); return; }
        }
#endif
        ::free(p);
    }
    void  copy_h2d(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  copy_d2h(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  copy_d2d(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  sync() override {}
private:
    std::mutex mu_;
    std::unordered_map<void*, size_t> mapped_;
};
