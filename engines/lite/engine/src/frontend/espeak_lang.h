// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// A small text -> IPA phonemizer for the non-English languages of Kitten TTS 2 (mini, nano, pico). It is an
// original implementation (no espeak-ng source code) that reads espeak-ng's published DATA files at run time:
// the language's pronunciation rules (<lang>_rules), its word list (<lang>_list) and the phoneme definitions
// (phsource: "phonemes" and the language's ph_* file). Those files are GPL-3.0 and are never bundled: the app
// downloads them (engines/lite/mobile/fetch_phonemizer_data.sh) and passes their folder.
//
// Everything is parsed once at load; phonemize() keeps its scratch buffers in the object (one call at a time per
// object).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace kt_fe {

class EspeakLang {
public:
    // lang: a voice name as the Python package uses it ("es", "de", "fr-fr", "it", "pt-br", "ru", "hi", ...).
    // data_dir: folder with <lang>_rules, <lang>_list and phsource/ (phonemes + ph_* files).
    // Returns nullptr and sets *err on failure.
    static std::unique_ptr<EspeakLang> load(const std::string& voice, const std::string& data_dir, std::string* err);
    virtual ~EspeakLang() = default;
    // One utterance (no punctuation splitting: that is the caller's job, as phonemizer does it) -> IPA with
    // espeak's stress marks; multi-letter phonemes are joined with tie '^' (exactly as phonemizer's tie='^').
    virtual std::string phonemize_clause(const std::string& text_utf8) = 0;
    virtual size_t memory_bytes() const = 0;
};

// The Python package's whole non-English path: phonemizer's punctuation handling + misaki's EspeakG2P (e2m) +
// phonemize_v1 / phonemize_v2 folds for `lang` ("de", "es", ...). version: 1 or 2.
std::string tiny_lang_phonemize(EspeakLang& eng, const std::string& lang, int version, const std::string& text);
// As the package's Frontend: the phonemes filtered by the model vocabulary (codepoints; may be null) and cut into
// chunks of at most 350 phonemes at sentence ends, then spaces.
std::vector<std::string> tiny_lang_chunks(EspeakLang& eng, const std::string& lang, int version, const std::string& text,
                                          const std::unordered_set<char32_t>* vocab);
// The espeak voice the package uses for a language code ("fr" -> "fr-fr", "pt" -> "pt-br"), or "" if none.
const char* tiny_lang_voice(const std::string& lang);

}  // namespace kt_fe
