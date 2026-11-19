// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Brazilian Portuguese (espeak voice pt-br) parts of the non-English front end that are specific to the language:
// numbers read as words. Original code; the number words themselves come from espeak-ng's pt_list data file
// (_1 .. _19, _2X .. _9X, _1C .. _9C, _0C, _1C0, _0and, _0M1, _1M1, _0M2, _1M2), read at run time.
#pragma once

#include <functional>
#include <string>

namespace kt_fe {
namespace lang_pt {

struct NumberWords {
    std::function<std::string(const std::string&)> ph;       // phonemes of a pt_list key ("" if none)
    std::function<bool(const std::string&)> has;              // the key is in pt_list
    std::function<bool(const std::string&)> stressed_start;   // phonemes mark a stress before their first vowel
};

// A run of ASCII digits -> phoneme mnemonics: "||" separates words, "\x02" separates the stress domains of one word
// (each keeps its own primary stress).
// grouped: written with thousands dots ("19.500"): the group after "mil" is read as a number of its own.
// A number with a leading zero is read digit by digit: see leading_zero() in pt.cpp.
std::string number(const std::string& digits, const NumberWords& w, bool grouped = false);
// The digits one by one, each a word of its own ("2,534": "cinco três quatro").
std::string digits_spaced(const std::string& digits, const NumberWords& w);
// "1º" / "21ª": ordinal words (_1o .. _9o, _1Xo .. _9Xo) with the masculine / feminine ending; 100 and up are read
// as cardinal numbers.
std::string ordinal(const std::string& digits, bool feminine, const NumberWords& w);

}  // namespace lang_pt
}  // namespace kt_fe
