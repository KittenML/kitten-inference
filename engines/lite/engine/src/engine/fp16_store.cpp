// src/engine/fp16_store.cpp — see fp16_store.hpp.
#include "engine/dev_env.hpp"
#include "fp16_store.hpp"
#include "load_options.hpp"
#include "layer.hpp"
#include "weight_cache.hpp"
#include "engine/free_hint.hpp"
#include "layers/attention.hpp"
#include "layers/conv1d.hpp"
#include "layers/gemm.hpp"
#include "layers/lstm.hpp"
#include "layers/norm.hpp"
#include "backend/cpu/ops_neon.hpp"  // packed_f32_elems
#include <algorithm>
#include <cstdlib>
#include <cstring>
#if defined(__F16C__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__GLIBC__) || defined(_WIN32)
#include <malloc.h>
#endif

namespace {

// Portable scalar conversions (round to nearest even), used where no hardware conversion is compiled in.
inline uint16_t f2h_scalar(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t)(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
    if (ax >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);  // rounds to >= 65520: inf
    if (ax < 0x38800000u) {                                     // fp16 subnormal or zero
        if (ax < 0x33000000u) return (uint16_t)sign;            // < 2^-25: rounds to 0
        const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - e;                         // 14..24 (value = m * 2^(e-150), unit 2^-24)
        uint32_t r = m >> shift; const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((ax - 0x38000000u) >> 13);                    // rebias 127 -> 15
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t)(sign | r);
}

inline float h2f_scalar(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else { e = 113; while (!(m & 0x400u)) { m <<= 1; --e; } x = sign | (e << 23) | ((m & 0x3ffu) << 13); }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

void narrow(const float* s, uint16_t* d, size_t n) {
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i*)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(s + i), _MM_FROUND_TO_NEAREST_INT));
#endif
    for (; i < n; ++i) d[i] = f2h_scalar(s[i]);
}

void widen_block(const uint16_t* s, float* d, size_t n) {
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(d + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + i))));
#elif defined(__aarch64__)
    for (; i + 4 <= n; i += 4)
        vst1q_f32(d + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s + i))));
#endif
    for (; i < n; ++i) d[i] = h2f_scalar(s[i]);
}

inline size_t align64(size_t b) { return (b + 63) & ~(size_t)63; }

}  // namespace

bool Fp16WeightStore::enabled_from_env() {
    const int o = kt_load_overrides().fp16_store;  // C API option for this load, if set
    if (o >= 0) return o == 1;
    const char* v = kt_dev_getenv("KITTEN_FP16_STORE");
    return v && v[0] && v[0] != '0';
}

// n_true: the element count the kernels actually read. Several layers record Buffer::bytes() with a 4-row padding while
// pack_weights_f32 pads C_out to its TILE (16 on AVX-512), so bytes() can under- (or over-) state the packed size.
bool Fp16WeightStore::try_store(Buffer& b, size_t n_true, int* store_idx) {
    if (!b.valid() || b.where() != MemSpace::Host) return false;
    const size_t n = n_true ? n_true : b.bytes() / 4;
    if (n * 4 < 4096) return false;
#if defined(__GLIBC__)
    if (malloc_usable_size(b.ptr()) < n * 4) { fp32_kept_ += n * 4; ++n_kept_; return false; }  // size mismatch: keep
#endif
    std::unique_ptr<uint16_t[]> h(new uint16_t[n]);
    const float* f = b.as<float>();
    narrow(f, h.get(), n);
    // Exactness: widen back in blocks and compare bits.
    float tmp[1024];
    for (size_t i = 0; i < n; i += 1024) {
        const size_t m = std::min<size_t>(1024, n - i);
        widen_block(h.get() + i, tmp, m);
        if (std::memcmp(tmp, f + i, m * 4) != 0) { fp32_kept_ += n * 4; ++n_kept_; return false; }
    }
    stores_.push_back(Store{std::move(h), n});
    *store_idx = (int)stores_.size() - 1;
    fp16_bytes_ += n * 2; fp32_removed_ += n * 4; ++n_fp16_;
    kitten_free_hint(b.ptr(), n * 4);  // Apple low-memory: the packed fp32 copy leaves the footprint
    b.release();
    return true;
}

void Fp16WeightStore::add(ILayer* layer, Buffer* target, size_t n_true) {
    int idx;
    if (try_store(*target, n_true, &idx)) per_layer_[layer].push_back(Entry{target, idx});
}

static bool class_on(char c) {  // KITTEN_FP16_STORE_ONLY=letters (debug/bisect): c conv1d, t convT, g gemm, l lstm,
    const char* v = kt_dev_getenv("KITTEN_FP16_STORE_ONLY");  // a adain, n adalayernorm, s shared attention/seqgemm
    return !v || !v[0] || std::strchr(v, c);
}

void Fp16WeightStore::compress_layer(ILayer* layer) {
    if (!layer) return;
    if (auto* c = dynamic_cast<Conv1dFp32*>(layer); c && class_on('c')) {
        add(layer, &c->w_packed_f32, packed_f32_elems(c->C_out, c->kernel_size * (c->C_in / std::max(c->groups, 1))));
        for (auto& b : c->w_per_k_packed_) add(layer, &b);
        for (auto& b : c->w_wino_packed_) add(layer, &b);
        add(layer, &c->w_kmerged_packed_);
    } else if (auto* t = dynamic_cast<ConvTranspose1dFp32*>(layer); t && class_on('t')) {
        const int g = std::max(t->groups, 1);
        for (auto& b : t->w_packed_k) add(layer, &b, packed_f32_elems(t->C_out / g, t->C_in / g));
    } else if (auto* g = dynamic_cast<GemmFp32*>(layer); g && class_on('g')) {
        add(layer, &g->w_packed_f32, packed_f32_elems(g->C_out, g->C_in));
    } else if (auto* l = dynamic_cast<LstmFp32*>(layer); l && class_on('l')) {
        const int H4 = 4 * l->hidden_size;
        add(layer, &l->w_ih_packed, packed_f32_elems(H4, l->input_size_));
        add(layer, &l->w_hh_packed, packed_f32_elems(H4, l->hidden_size));
        add(layer, &l->w_ih_r_packed, packed_f32_elems(H4, l->input_size_));
        add(layer, &l->w_hh_r_packed, packed_f32_elems(H4, l->hidden_size));
    } else if (auto* a = dynamic_cast<AdaIn1dFp32*>(layer); a && class_on('a')) {
        add(layer, &a->fc_packed, packed_f32_elems(2 * a->C, a->C_style));
    } else if (auto* a2 = dynamic_cast<AdaLayerNormFp32*>(layer); a2 && class_on('n')) {
        add(layer, &a2->fc_packed, packed_f32_elems(2 * a2->C, a2->C_style));
    }
}

void Fp16WeightStore::compress_shared(const std::vector<std::unique_ptr<ILayer>>& layers) {
    auto shared = [&](ILayer* layer, Buffer& view, std::shared_ptr<SharedPackedWeight>& owner, size_t n_true) {
        if (!view.valid()) return;
        auto it = by_ptr_.find(view.ptr());
        if (it == by_ptr_.end()) {
            if (!owner || owner->buffer.ptr() != view.ptr()) {  // not shared: an ordinary owning buffer
                if (!owner) add(layer, &view, n_true);
                return;
            }
            const void* key = owner->buffer.ptr();
            int idx;
            if (!try_store(owner->buffer, n_true, &idx)) return;
            it = by_ptr_.emplace(key, idx).first;
        }
        view = Buffer();  // non-owning view (no-op deleter); points into the arena during forward
        per_layer_[layer].push_back(Entry{&view, it->second});
    };
    if (!class_on('s')) return;
    for (auto& up : layers) {
        ILayer* layer = up.get();
        if (auto* at = dynamic_cast<AttentionFp32*>(layer)) {
            shared(layer, at->w_qkv_packed, at->w_qkv_packed_owner, packed_f32_elems(3 * at->D, at->D));
            shared(layer, at->w_proj_packed, at->w_proj_packed_owner, packed_f32_elems(at->D, at->D));
        } else if (auto* sg = dynamic_cast<SeqGemmFp32*>(layer)) {
            shared(layer, sg->w_packed_f32, sg->w_packed_f32_owner, packed_f32_elems(sg->C_out, sg->C_in));
        }
    }
}

void Fp16WeightStore::widen(ILayer* layer) {
    auto it = per_layer_.find(layer);
    if (it == per_layer_.end()) return;
    auto& es = it->second;
    size_t total = 0;
    for (auto& e : es) total += align64(stores_[e.store].n * 4);
    if (total > arena_.bytes()) {
        kitten_free_hint(arena_.ptr(), arena_.bytes());  // the old, smaller arena
        arena_.release();
        // posix_memalign / _aligned_malloc: std::aligned_alloc is missing on MSVC and on Android before API 28.
#if defined(_WIN32)
        void* p = _aligned_malloc(align64(total), 64);
        if (!p) throw std::bad_alloc();
        arena_ = Buffer(p, align64(total), MemSpace::Host, [](void* q) { _aligned_free(q); });
#else
        void* p = nullptr;
        if (posix_memalign(&p, 64, align64(total)) != 0 || !p) throw std::bad_alloc();
        arena_ = Buffer(p, align64(total), MemSpace::Host, [](void* q) { std::free(q); });
#endif
    }
    char* base = static_cast<char*>(arena_.ptr());
    size_t off = 0;
    for (auto& e : es) {
        const Store& s = stores_[e.store];
        float* dst = reinterpret_cast<float*>(base + off);
        const long nb = (long)((s.n + 16383) / 16384);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (nb > 1 && !omp_in_parallel())
#endif
        for (long j = 0; j < nb; ++j) {
            const size_t i0 = (size_t)j * 16384, m = std::min<size_t>(16384, s.n - i0);
            widen_block(s.h.get() + i0, dst + i0, m);
        }
        *e.target = Buffer(dst, s.n * 4, MemSpace::Host, nullptr);
        off += align64(s.n * 4);
    }
}

void Fp16WeightStore::unwiden(ILayer* layer) {
    auto it = per_layer_.find(layer);
    if (it == per_layer_.end()) return;
    for (auto& e : it->second) *e.target = Buffer();
}
