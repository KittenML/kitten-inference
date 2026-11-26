// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Russian parts of the espeak-ng data reader (espeak_lang.cpp). Original code: the letter classes are the ones the
// Russian rules file documents in its header, the default stress table was measured on the espeak-ng build the
// Python package uses, and the stress dictionary reader reads
// espeak-ng's ru_listx DATA file (GPL-3.0, downloaded at run time, never bundled).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kt_fe {
namespace ru {

// Letter classes used by the Russian rules (A B C F G H Y). Returns -1 when c is not a Cyrillic letter (the caller
// then uses its Latin classes), else 0/1.
int in_letter_group(char group, char32_t c);

// Default stress when a word is in neither word list: the stressed syllable (1-based) for a word of nsyl syllables
// whose last phoneme is a vowel (end = 0), a stop or affricate (end = 1) or another consonant (end = 2).
int guess_stress(int nsyl, int end);

// The ru_listx word list: "<word> $<n>" lines (stress on syllable n), ~800k words. Stored compactly: one byte per
// Cyrillic letter, sorted, front-coded in blocks of 16 words (~5 MB for the whole list instead of ~130 MB as a hash
// map of strings). The file is a few sorted runs: they are merged while reading it in small pieces, so loading needs
// no more memory than the finished table.
class StressDict {
public:
    // Reads the file; returns false if it cannot be read. Lines that are not "<cyrillic word> $<1-9>" are skipped.
    bool load(const std::string& path);
    // Stress syllable (1-9) of a lower-case word, or 0 when the word is not listed.
    int lookup(const std::u32string& word) const;
    size_t memory_bytes() const { return data_.capacity() + index_.capacity() * sizeof(uint32_t); }
    size_t size() const { return count_; }

private:
    static constexpr int kBlock = 16;
    static constexpr size_t kMaxRuns = 16;   // sorted runs of the file merged while reading; more: sorted in memory
    void append(const std::string& key, const std::string& prev, uint8_t stress);
    std::vector<uint8_t> data_;     // blocks: [full key\0 stress] then 15 x [shared len][suffix len][suffix][stress]
    std::vector<uint32_t> index_;   // offset of each block in data_
    size_t count_ = 0;
};

// A digit string as espeak-ng's Russian word list spells it out (keys _0.._19, _2X.._9X, _0C/_2C.._9C, _1f/_2f,
// _1MA1/_0MA1/_0M1 thousand, _1MA2/_0MA2/_0M2 million, ...): phoneme mnemonics, words separated by "||", tens and
// units written as one word ("\x02" between them: no space, own stress). ph(key) returns the entry's mnemonics.
std::string number_words(const std::u32string& digits, const std::function<std::string(const std::string&)>& ph);

// Byte code of a lower-case Cyrillic letter for StressDict (1..33), 0 for anything else.
uint8_t letter_code(char32_t c);

}  // namespace ru
}  // namespace kt_fe
