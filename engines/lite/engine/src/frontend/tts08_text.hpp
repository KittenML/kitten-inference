// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// tts08_text.hpp - Kitten TTS 0.8 text front end for the phone library: the Python package's text normalisation
// (src/kitten_inference/tts08/preprocess.py: TextPreprocessor(remove_punctuation=False) then chunk_text) ported to C++,
// and English phonemisation with CEPhonemizer (KittenML's Apache-2.0 reader of espeak-ng's en_rules / en_list).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mini_regex.hpp"

class IPAPhonemizer;

namespace kt08fe {

// TextPreprocessor(remove_punctuation=False) of the Python package. All patterns are compiled once in the constructor.
class Normalizer08 {
public:
    Normalizer08();
    // Normalised text (lower case, numbers/currency/times... in words, one space between words).
    std::u32string normalize(const std::u32string& text) const;
    // chunk_text(text, max_len=400): sentence chunks, each ending in . ! ? , ; or :
    static std::vector<std::u32string> chunks(const std::u32string& normalized, size_t max_len = 400);

    using Repl = std::function<std::string(const Regex::Match&, const std::u32string&)>;
    struct Rule {
        Regex re;
        Repl fn;
        bool strip_after = false;
    };

private:
    std::vector<Rule> rules_;
};

// Phonemiser handle behind kt08_phonemizer: CEPhonemizer (en-us) plus the normaliser.
class Frontend08 {
public:
    // Throws std::runtime_error with a message when a data file cannot be read.
    Frontend08(const std::string& rules_path, const std::string& list_path);
    ~Frontend08();
    // One normalised chunk -> espeak-style IPA (stress marks, punctuation kept), as phonemizer.EspeakBackend gives.
    std::string chunk_to_phonemes(const std::string& chunk_utf8) const;
    const Normalizer08& normalizer() const { return norm_; }

private:
    static const Regex& marks_re();
    std::string piece(const std::string& text) const;
    void clause_end_stress(const std::string& text, std::vector<std::string>& ipa) const;
    static void vowel_context(std::vector<std::string>& ipa);
    Normalizer08 norm_;
    std::unique_ptr<IPAPhonemizer> ph_;
};

}  // namespace kt08fe
