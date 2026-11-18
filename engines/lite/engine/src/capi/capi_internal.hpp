// Helpers shared by the C API's source files (kitten_tts08.cpp, kitten_npu.cpp). Not part of the public API.
#pragma once
#include "kitten_tts08.h"
#include "engine/model.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace kt08_internal {
kt08_status finish_tiny_chunk(const float* wav, size_t n, const float* dur, size_t nd, float speed, int first_chunk,
                              float** out_audio, size_t* out_samples);
std::unordered_map<std::string, ModelIOTensor> forward_named(kt08_model* model,
                                                             const std::unordered_map<std::string, std::vector<float>>& in,
                                                             float speed);
kt08_status set_error(kt08_status s, const std::string& msg);
}  // namespace kt08_internal
