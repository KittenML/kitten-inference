// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Dutch specifics of the espeak-data text front end (espeak_lang.cpp): number words.
#pragma once

#include <functional>
#include <string>

namespace kt_fe {
namespace nl {

// Digits ("1988") -> phoneme mnemonics of the Dutch number words, built from the word list's number entries
// (ph("_7"), ph("_3X"), ph("_0C"), ph("_0M1"), ...; ph returns "" when an entry is missing). Words are separated by
// "||"; only the first word keeps its primary stress, later stresses become secondary, as the package's
// espeak-ng output reads numbers ("tweeduizend zes" -> two ˈ.. ˌ.. ˌ..).
// flags: kSep = written with thousands dots ("19.500": the part below a thousand keeps its own primary stress);
// kDecimal = digits after a decimal comma (all stresses secondary); kOrdinal = "18e" (achttiende).
enum { kSep = 1, kDecimal = 2, kOrdinal = 4 };
std::string number_words(const std::string& digits, const std::function<std::string(const std::string&)>& ph,
                         int flags = 0);

}  // namespace nl
}  // namespace kt_fe
