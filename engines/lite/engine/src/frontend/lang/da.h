// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Danish specifics of the espeak-data reader (espeak_lang.cpp): how numbers are read.
#pragma once

#include <functional>
#include <string>

namespace kt_fe {
namespace da {

// Digits (no separators, no leading zero, at most 9 digits) -> phoneme mnemonics from the Danish word list
// (_0.._19, _<n>X tens, _0C hundred, _1C, _0M1 / _1M1 thousand, _0M2 / _1M2 million, _0and "og"). Words are joined
// by "||"; parts of one word by '\x05' (no phoneme name spans them). ph(key) returns the list entry's phonemes, ""
// when it has none.
std::string number(long v, const std::function<std::string(const std::string&)>& ph);

// "$" is not in the Danish word list; the Python package's phonemizer reads it as the Danish word "dollar"
// (measured: [d'VlV] in the list's phoneme names).
constexpr const char* kDollar = "d'VlV";

}  // namespace da
}  // namespace kt_fe
