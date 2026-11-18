// Part of kitten-inference (Apache-2.0).
#pragma once
#include <string>
#include <string_view>

namespace kt_tinyen {
// tts_norm.normalize(): expand UTF-8 text to its spoken form (numbers, dates, times, money, units, abbreviations).
std::string tts_normalize(std::string_view text);
}
