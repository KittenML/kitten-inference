// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Japanese text -> phonemes for Kitten TTS 2 mini (phonemizer v2), as the Python package does it
// (tts2_tiny/frontend: misaki's JAG2P on its cutlet path, which uses mojimoji, num2kana and a MeCab tagger with the
// UniDic dictionary from unidic-lite, then phonemize_v2's folds and the model vocabulary). Re-implemented here in
// C++ (no Python, no MeCab): NFKC from generated Unicode tables, the dictionary reader in ja_dict.cpp.
//
// Data (folder passed at load, downloaded by the app with engines/lite/mobile/fetch_phonemizer_data.sh):
//   sys.dic, unk.dic, matrix.bin, char.bin  (UniDic 2.1.2 as packaged by unidic-lite 1.0.8; BSD licence option)
//   ja_words.txt                            (misaki's Japanese word list, Apache-2.0, from this repository)
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace kt_fe {

namespace ja { class JaDictionary; }

// Thrown by phonemize() and chunks() for text the Python package refuses too: it raises an error on a number of
// more than 9 digits and on digits it has no words for (decimal digits of other scripts such as "٣", and digit
// symbols such as "❶").
struct JaTextError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class JaFrontend {
public:
    static std::unique_ptr<JaFrontend> load(const std::string& dir, std::string* err);
    ~JaFrontend();
    // One text -> its phoneme string (misaki JAG2P + phonemize_v2.phonemize_lang), filtered by vocab if given.
    std::string phonemize(const std::string& text_utf8, const std::unordered_set<char32_t>* vocab) const;
    // As the package's Frontend: chunks of at most 350 phonemes, cut at sentence ends, then spaces.
    std::vector<std::string> chunks(const std::string& text_utf8, const std::unordered_set<char32_t>* vocab) const;
    // Debug: the normalised text and the words the tagger returns (surface \t reading \t char type).
    std::string debug_words(const std::string& text_utf8) const;
    size_t memory_bytes() const;

private:
    JaFrontend() = default;
    std::unique_ptr<ja::JaDictionary> dict_;
    std::string words_buf_;                 // ja_words.txt
    std::vector<std::pair<uint32_t, uint32_t>> words_;   // sorted (offset, length) into words_buf_
    size_t max_word_bytes_ = 0;
    bool is_word(const char* s, size_t n) const;
};

}  // namespace kt_fe
