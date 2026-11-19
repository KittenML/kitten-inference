// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/tiny_lang_frontend.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace kt_fe {

// Languages whose measured parity passed against the Python package (exact phoneme match, or an audio check on the
// sentences that differ: engines/lite/mobile/README.md). Others return KT08_ERR_UNSUPPORTED.
static const char* const kSupportedV1[] = {"de", "es", "fr", "it", "pt"};   // nano, pico
static const char* const kSupportedV2[] = {"da", "de", "es", "fr", "hi", "it", "ja", "ko", "pt", "sv"};   // mini (sw, tr, fi: rules kept, not enabled: failed a fresh-text check)

bool TinyLangFrontend::supported(const std::string& lang, int version) {
    if (version == 1) { for (auto* l : kSupportedV1) if (lang == l) return true; }
    else { for (auto* l : kSupportedV2) if (lang == l) return true; }
    return false;
}

int TinyLangFrontend::text_to_chunks(const std::string& text, const std::string& lang, std::vector<std::string>& chunks,
                                     std::string& err) {
    chunks.clear();
    if (!supported(lang, version_)) { err = "language not supported by this build: " + lang; return 6; }
    if (dir_.empty()) { err = "this language needs espeak_data_dir at kt08_tiny_frontend_load"; return 6; }
    if (lang == "ja") {   // misaki's Japanese path (frontend/lang/ja.cpp), data in <espeak_data_dir>/ja
        if (!ja_) {
            ja_ = JaFrontend::load(dir_ + "/ja", &err);
            if (!ja_) return 2;
        }
        try {
            chunks = ja_->chunks(text, &vocab_);
        } catch (const JaTextError& e) {   // text the Python package raises on too (lang/ja.hpp)
            chunks.clear();
            err = e.what();
            return 1;   // KT08_ERR_ARGUMENT
        }
        return 0;
    }
    auto it = engines_.find(lang);
    if (it == engines_.end()) {
        auto eng = EspeakLang::load(tiny_lang_voice(lang), dir_, &err);
        if (!eng) return 2;
        it = engines_.emplace(lang, std::move(eng)).first;
    }
    chunks = tiny_lang_chunks(*it->second, lang, version_, text, &vocab_);
    return 0;
}

bool TinyLangFrontend::read_vocab(const std::string& path, std::unordered_set<char32_t>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    size_t i = s.find('{');
    if (i == std::string::npos) return false;
    // {"<key>": <int>, ...}: collect keys that are one code point
    while (i < s.size()) {
        size_t q = s.find('"', i);
        if (q == std::string::npos) break;
        i = q + 1;
        std::u32string key;
        while (i < s.size() && s[i] != '"') {
            unsigned char c = (unsigned char)s[i];
            if (c == '\\' && i + 1 < s.size()) {
                char e = s[i + 1];
                if (e == 'u' && i + 6 <= s.size()) {
                    unsigned x = (unsigned)std::strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16);
                    key += (char32_t)x; i += 6; continue;
                }
                key += (char32_t)(e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e);
                i += 2; continue;
            }
            char32_t cp; int n;
            if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
            else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; } else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
            else { i++; continue; }
            if (i + n > s.size()) break;
            for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
            key += cp; i += n;
        }
        i++;   // closing quote
        size_t colon = s.find_first_not_of(" \t\r\n", i);
        if (colon == std::string::npos || s[colon] != ':') continue;   // a string value, not a key
        if (key.size() == 1) v.insert(key[0]);
        i = colon + 1;
    }
    return !v.empty();
}

}  // namespace kt_fe
