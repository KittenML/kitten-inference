// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Hindi (Devanagari) parts of the espeak-data reader (frontend/espeak_lang.cpp): the letter classes that hi_rules
// uses in its contexts, the stress placement, and numbers in the Indian grouping (thousand, lakh, crore).
// Original code; it reads nothing itself (espeak_lang.cpp passes the data in).
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace kt_fe {
namespace hi {

bool is_devanagari(char32_t c);
// Independent vowel letters and dependent vowel signs (hi_rules: letter groups A and Y).
bool is_vowel(char32_t c);
// Vowel signs and the virama (group B).
bool is_sign_or_virama(char32_t c);
// Consonant letters, including the nukta forms (group C).
bool is_consonant(char32_t c);
// The letter-group test of a rule context for a Devanagari character: 'A', 'B', 'C', 'Y'. Sets *handled = false for
// other group letters (the caller's default applies).
bool in_group(char group, char32_t c, bool* handled);

// One syllable of a word for the stress placement: its vowel and the consonants after it (up to the next vowel or
// the word end; a long consonant such as k: counts twice). The schwas that are dropped later still count here.
struct Syl {
    bool long_vowel = false;
    int coda = 0;
};
// Index of the syllable that takes the primary stress (-1: none).
int primary_stress(const std::vector<Syl>& syl);

// Number words: the digits as the Indian system groups them; `ph(key)` returns the phoneme mnemonics of a list
// entry such as "_45", "_0C", "_0M1" ("" if none). Parts are joined with "||" (word boundaries).
std::string number(const std::u32string& digits, const std::function<std::string(const std::string&)>& ph);

// A word in capital Latin letters inside Hindi text is read by the English rules: spelled out when it has at most
// three letters, no consonant or no vowel, or starts with two consonants that cannot begin an English syllable
// (GMT, DNA, PSTN, RSPCA); read as a word otherwise (NASA). `listed_abbrev`: the English list marks it $abbrev (UK, USA).
bool english_spells(const std::u32string& caps, bool listed_abbrev);

}  // namespace hi
}  // namespace kt_fe
