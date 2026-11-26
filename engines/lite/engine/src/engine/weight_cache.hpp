#pragma once

#include "buffer.hpp"
#include <memory>
#include <string>
#include <unordered_map>

struct SharedPackedWeight {
    explicit SharedPackedWeight(Buffer&& b) : buffer(std::move(b)) {}
    Buffer buffer;
};

struct SharedPackedWeightView {
    std::shared_ptr<SharedPackedWeight> owner;
    Buffer view;
};

inline std::unordered_map<std::string, std::weak_ptr<SharedPackedWeight>>& packed_weight_cache()
{
    static thread_local std::unordered_map<std::string, std::weak_ptr<SharedPackedWeight>> cache;
    return cache;
}

inline void clear_packed_weight_cache()
{
    packed_weight_cache().clear();
}

// true (and the shared view) when a layer with the same key already packed this weight (ALBERT-style shared blocks repeat the
// same tensors 12 times): the caller can skip reading and packing it again; the bytes are the ones the first layer packed.
inline bool find_shared_packed_weight(const std::string& key, SharedPackedWeightView& out)
{
    auto& cache = packed_weight_cache();
    auto it = cache.find(key);
    if (it == cache.end()) return false;
    auto owner = it->second.lock();
    if (!owner) return false;
    out = { owner, Buffer(owner->buffer.ptr(), owner->buffer.bytes(), owner->buffer.where(), [](void*){}) };
    return true;
}

inline SharedPackedWeightView share_host_packed_weight(const std::string& key, Buffer&& buffer)
{
    auto& cache = packed_weight_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (auto owner = it->second.lock()) {
            return {
                owner,
                Buffer(owner->buffer.ptr(), 0, owner->buffer.where(), [](void*){})
            };
        }
    }

    auto owner = std::make_shared<SharedPackedWeight>(std::move(buffer));
    cache[key] = owner;
    return {
        owner,
        Buffer(owner->buffer.ptr(), owner->buffer.bytes(), owner->buffer.where(), [](void*){})
    };
}
