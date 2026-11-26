// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Non-English text front end of Kitten TTS 2 mini / nano / pico (kt08_tiny_frontend_*): one EspeakLang per
// language, loaded on first use from espeak_data_dir and parsed once.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "frontend/espeak_lang.h"
#include "frontend/lang/ja.hpp"

namespace kt_fe {

class TinyLangFrontend {
public:
    // version: 1 (nano, pico: phonemizer v1) or 2 (mini: v2). vocab: the model's single-character symbols.
    TinyLangFrontend(int version, std::unordered_set<char32_t> vocab, std::string espeak_data_dir)
        : version_(version), vocab_(std::move(vocab)), dir_(std::move(espeak_data_dir)) {}
    // Languages this build passes on (measured parity, see engines/lite/mobile/README.md).
    static bool supported(const std::string& lang, int version);
    // 0 = ok; 6 (KT08_ERR_UNSUPPORTED) = language not supported; 2 (KT08_ERR_IO) = data files missing or bad.
    int text_to_chunks(const std::string& text, const std::string& lang, std::vector<std::string>& chunks, std::string& err);
    // Reads the single-character keys of a vocab.json (phoneme -> id). Returns false if the file cannot be read.
    static bool read_vocab(const std::string& path, std::unordered_set<char32_t>& out);

private:
    int version_;
    std::unordered_set<char32_t> vocab_;
    std::string dir_;
    std::map<std::string, std::unique_ptr<EspeakLang>> engines_;
    std::unique_ptr<JaFrontend> ja_;   // Japanese (mini): not espeak; its files are in <espeak_data_dir>/ja
};

}  // namespace kt_fe
