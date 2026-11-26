// src/engine/fp16_store.hpp
#pragma once
// optional fp16 storage of the packed fp32 weight buffers (KITTEN_FP16_STORE=1).
//
// A packed buffer is stored in fp16 only if every value in it round-trips exactly (fp32 -> fp16 RNE -> fp32 gives the
// same bits), so the storage itself never changes a weight: with an fp32-trained weights file almost nothing is stored
// in fp16; with a weights file whose values were rounded to fp16 beforehand (the lossy "fp16w" variant) everything is.
// Before a layer's forward its fp16 buffers are widened (exactly) into one shared, grow-only fp32 arena and the layer's
// Buffer members point into it (non-owning); after the forward they are cleared again. So the kernels are unchanged and
// the output is bit-identical to the plain fp32 engine on the same weights file. Cost: one fp16 -> fp32 pass over the
// weights per forward call.
#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

class ILayer;

class Fp16WeightStore {
public:
    static bool enabled_from_env();

    // Owning packed-weight Buffers of one layer: call right after the layer's load_weights (keeps the load peak low).
    void compress_layer(ILayer* layer);
    // Shared (ALBERT) packed weights held through SharedPackedWeight owners: call once after all layers are loaded.
    void compress_shared(const std::vector<std::unique_ptr<ILayer>>& layers);

    void widen(ILayer* layer);    // before the layer's forward
    void unwiden(ILayer* layer);  // after it

    size_t fp16_bytes() const { return fp16_bytes_; }        // fp16 storage held
    size_t fp32_removed() const { return fp32_removed_; }    // fp32 bytes freed
    size_t fp32_kept() const { return fp32_kept_; }          // candidate bytes left in fp32 (not fp16-exact)
    size_t arena_bytes() const { return arena_.bytes(); }
    int buffers_fp16() const { return n_fp16_; }
    int buffers_kept() const { return n_kept_; }

private:
    struct Store { std::unique_ptr<uint16_t[]> h; size_t n = 0; };
    struct Entry { Buffer* target; int store; };
    bool try_store(Buffer& b, size_t n_true, int* store_idx);
    void add(ILayer* layer, Buffer* target, size_t n_true = 0);  // n_true 0: Buffer::bytes() is exact

    std::unordered_map<ILayer*, std::vector<Entry>> per_layer_;
    std::unordered_map<const void*, int> by_ptr_;  // shared owners: original pointer -> store
    std::vector<Store> stores_;
    Buffer arena_;
    size_t fp16_bytes_ = 0, fp32_removed_ = 0, fp32_kept_ = 0;
    int n_fp16_ = 0, n_kept_ = 0;
};
