// Part of kitten-inference (Apache-2.0).
// English number words for the Kitten TTS 2 English front end. The wording follows British-style English as the
// Python front end speaks it: cardinals ("one thousand, two hundred and thirty-four"), ordinals ("twenty-first"),
// years ("nineteen oh-five") and decimal floats ("three point one four"). Written from the expected output only.
#pragma once
#include <string>
#include <string_view>

namespace kt_tinyen {

// digits: ASCII decimal digits (leading zeros allowed). Numbers of more than 36 digits (19 where the compiler has no
// 128-bit integer) come back as the digits.
std::string num_cardinal(std::string_view digits);
std::string num_cardinal(long long v);
std::string num_ordinal(std::string_view digits);
std::string num_year(std::string_view digits);
// A decimal string "12.50" spoken as the shortest float form: "twelve point five".
std::string num_float(std::string_view dec);

}  // namespace kt_tinyen
