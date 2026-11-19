// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Korean for the espeak-data reader (espeak_lang.cpp): what the Korean data files need that the shared reader does
// not do for other languages. Original code, written from the Unicode Hangul algorithm and from measured behaviour
// of the Python package's phonemizer (no espeak-ng source code).
#pragma once

#include <string>
#include <vector>

namespace kt_fe {
namespace ko {

// Hangul conjoining jamo (U+1100..U+11FF): the letters espeak's Korean rules (ko_rules) are written in.
bool is_jamo(char32_t c);
// A vowel letter (jungseong, U+1161..U+11A7).
bool is_vowel_jamo(char32_t c);
// Hangul syllables (U+AC00..U+D7A3) -> leading consonant + vowel (+ final consonant) jamo (Unicode's arithmetic
// decomposition). A syllable that starts with the silent consonant ㅇ (U+110B) keeps only its vowel and final, which
// is how the Korean rules and word list spell those syllables. Other characters are copied.
std::u32string decompose(const std::u32string& text);
// A Latin-script word of n letters written in capitals, standing alone (not attached to Hangul), in a clause that has
// other_letters letters that are not capitals (Hangul syllables and lower-case letters): true when it is read as an
// abbreviation, one letter name per word ("REM" in a Korean sentence -> "R E M"). Measured: 2 or 3 capitals, and at
// least 4 other letters in the clause; otherwise the word goes through the English rules like any word.
inline bool is_abbreviation(size_t n, int other_letters) { return (n == 2 || n == 3) && other_letters >= 4; }
// English words inside Korean text (the package reads Latin-script words with the English rules): how many leading
// letters of a lower-case word are said as letter names before the rest is read as a word. 0: none (a normal word);
// -1: the whole word is letter names ("cctv", "dna"). Measured on the package: a word must start with a vowel, one
// consonant or one of a fixed set of consonant groups; otherwise its first letter is named and the rest is tried again
// ("wned" = W + "ned"), and when two letters or fewer are left the whole word is letter names.
int english_spell_prefix(const std::u32string& word);
// Letter names of an n-letter word are stressed in groups (last letter of each group): sizes of the groups.
std::vector<int> english_spell_groups(size_t n);
// A Chinese character (CJK unified ideograph) in Korean text is not read: the package says "Chinese letter" for each
// one (English "Chinese", then these Korean phonemes, no space).
inline bool is_ideograph(char32_t c) { return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0xF900 && c <= 0xFAFF); }
constexpr const char* kIdeographTail = "l'et@";
// Default stress of a Korean word. shape: one letter per sound of the word in order, 'V' vowel, 'C' anything else.
// Returns the index (among the vowels) of the stressed vowel: the first, unless the first syllable is light and the
// second heavy (a vowel is heavy when two or more consonants follow it, or one or more at the end of the word).
int stress_vowel(const std::string& shape);
// A number (ASCII digits) as espeak phoneme mnemonics, built from the word list's number entries (_0 .. _9, _1X ..
// _9X, _0C, _0M0/_1M0 (10^3), _0M1/_1M1 (10^4), _0M2/_1M2 (10^8)) the way the package reads Korean numbers: groups
// of four digits (man, eok), "||" between spoken words and '\x02' where two words are joined without a space but
// keep their own stress. ph(key) returns the entry's mnemonics ("" if none).
template <class PhOf>
std::string number(const std::u32string& digits, const PhOf& ph);

// ----------------------------------------------------------------- implementation of the template
namespace detail {
std::string join_domains(const std::string& mnemonics);   // "_!" pauses inside an entry -> '\x02'
}
template <class PhOf>
std::string number(const std::u32string& digits, const PhOf& ph) {
    auto d = [&](int n) { return ph("_" + std::to_string(n)); };
    auto join = [](std::vector<std::string>& parts, const char* sep) {
        std::string o;
        for (auto& p : parts) { if (p.empty()) continue; if (!o.empty()) o += sep; o += p; }
        return o;
    };
    std::string o;
    size_t z = 0;
    while (digits.size() > 1 && z + 1 < digits.size() && digits[z] == '0') {   // leading zeros are read one by one
        if (!o.empty()) o += '\x02';
        o += d(0);
        z++;
    }
    std::u32string rest = digits.substr(z);
    if (rest.size() > 12) {   // very long digit strings: one digit per word
        std::string s;
        for (char32_t c : rest) { if (!s.empty()) s += "||"; s += d((int)(c - '0')); }
        return detail::join_domains(o.empty() ? s : o + "||" + s);
    }
    long long v = 0;
    for (char32_t c : rest) v = v * 10 + (c - '0');
    auto lt10000 = [&](int n) {
        std::vector<std::string> parts;
        int th = n / 1000, h = (n / 100) % 10, r = n % 100;
        if (th) parts.push_back(th == 1 ? ph("_1M0") : d(th) + "||" + ph("_0M0"));
        if (h) parts.push_back(h == 1 ? ph("_0C") : d(h) + "\x02" + ph("_0C"));
        if (r) {
            if (r < 10) parts.push_back(d(r));
            else {
                std::string t = ph("_" + std::to_string(r / 10) + "X");
                if (r % 10) t += "\x02" + d(r % 10);
                parts.push_back(t);
            }
        }
        return join(parts, "||");
    };
    std::string s;
    if (v == 0) s = d(0);
    else {
        std::vector<std::string> parts;
        long long eok = (v / 100000000LL) % 10000, man = (v / 10000) % 10000, low = v % 10000;
        if (v / 1000000000000LL) parts.push_back(lt10000((int)(v / 1000000000000LL)) + "||" + ph("_0M3"));
        if (eok) parts.push_back(eok == 1 ? ph("_1M2") : lt10000((int)eok) + "||" + ph("_0M2"));
        if (man) parts.push_back(man == 1 ? ph("_1M1") : lt10000((int)man) + "||" + ph("_0M1"));
        if (low) parts.push_back(lt10000((int)low));
        s = join(parts, "||");
    }
    if (!o.empty() && !rest.empty()) o += "||";
    return detail::join_domains(o + s);
}

}  // namespace ko
}  // namespace kt_fe
