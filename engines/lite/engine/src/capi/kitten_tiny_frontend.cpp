// Part of kitten-inference (Apache-2.0).
// C API of the Kitten TTS 2 mini / nano / pico text front end (kt08_tiny_frontend_*, API version 5): text in, phoneme
// chunks out, on the device. English: src/frontend/tiny_en (data from g2p_dir). Other languages:
// src/frontend/tiny_lang_frontend, which reads espeak-ng's data files from espeak_data_dir (downloaded by the app:
// engines/lite/mobile/fetch_phonemizer_data.sh); languages that are not supported return KT08_ERR_UNSUPPORTED.
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "capi_internal.hpp"
#include "frontend/tiny_en/tiny_en.hpp"
#include "frontend/tiny_lang_frontend.hpp"
#include "kitten_tts08.h"

struct kt08_tiny_frontend {
    std::string phonemizer;
    std::unique_ptr<kt_tinyen::EnglishFrontend> en;
    std::unique_ptr<kt_fe::TinyLangFrontend> other;
    std::mutex mu;  // one call at a time per front end (its caches are not thread-safe)
};

using kt08_internal::set_error;

namespace {
// Text with more than kMaxRun characters (code points) in a row without an ASCII space, tab or line break is refused
// with KT08_ERR_TOO_LONG, as in the Kitten TTS 0.8 text calls: the English normaliser's regular expressions take
// time (and stack) that grows with the longest run, and no word is that long.
constexpr size_t kMaxRun = 1000;
bool has_long_run(const char* s) {
    size_t run = 0;
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == ' ' || (c >= 9 && c <= 13)) run = 0;
        else if ((c & 0xC0) != 0x80 && ++run > kMaxRun) return true;
    }
    return false;
}
}  // namespace

extern "C" {

KT08_API kt08_status kt08_tiny_frontend_load(const char* phonemizer_version, const char* vocab_json_path,
                                             const char* g2p_dir, const char* espeak_data_dir,
                                             kt08_tiny_frontend** out) {
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out is NULL");
    *out = nullptr;
    if (!phonemizer_version || !vocab_json_path) return set_error(KT08_ERR_ARGUMENT, "phonemizer_version and vocab_json_path are required");
    std::string pv = phonemizer_version;
    if (pv != "v1" && pv != "v2") return set_error(KT08_ERR_ARGUMENT, "phonemizer_version must be \"v1\" or \"v2\"");
    try {
        auto fe = std::make_unique<kt08_tiny_frontend>();
        fe->phonemizer = pv;
        if (g2p_dir && *g2p_dir) {
            std::string err;
            fe->en = kt_tinyen::EnglishFrontend::load(g2p_dir, g2p_dir, vocab_json_path, pv, err);
            if (!fe->en) return set_error(KT08_ERR_IO, "English front end: " + err);
        }
        if (espeak_data_dir && *espeak_data_dir) {
            std::unordered_set<char32_t> vocab;
            if (!kt_fe::TinyLangFrontend::read_vocab(vocab_json_path, vocab))
                return set_error(KT08_ERR_IO, std::string("cannot read the vocabulary ") + vocab_json_path);
            fe->other = std::make_unique<kt_fe::TinyLangFrontend>(pv == "v1" ? 1 : 2, std::move(vocab), espeak_data_dir);
        }
        *out = fe.release();
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory loading the text front end");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_IO, e.what());
    } catch (...) {
        return set_error(KT08_ERR_IO, "unknown error loading the text front end");
    }
}

KT08_API void kt08_tiny_frontend_free(kt08_tiny_frontend* fe) { delete fe; }

KT08_API kt08_status kt08_tiny_text_to_phonemes(kt08_tiny_frontend* fe, const char* text_utf8, const char* language,
                                                char** out_chunks) {
    if (!out_chunks) return set_error(KT08_ERR_ARGUMENT, "out_chunks is NULL");
    *out_chunks = nullptr;
    if (!fe || !text_utf8 || !language) return set_error(KT08_ERR_ARGUMENT, "fe, text_utf8 and language are required");
    const bool english = std::strcmp(language, "en") == 0;
    if (english && !fe->en) return set_error(KT08_ERR_UNSUPPORTED, "English needs g2p_dir at kt08_tiny_frontend_load");
    if (!english && !fe->other)
        return set_error(KT08_ERR_UNSUPPORTED,
                         std::string("language ") + language + " needs espeak_data_dir at kt08_tiny_frontend_load");
    if (has_long_run(text_utf8))
        return set_error(KT08_ERR_TOO_LONG, "text has more than 1000 characters without a space; split it or remove that part");
    try {
        std::lock_guard<std::mutex> lock(fe->mu);
        std::vector<std::string> chunks;
        std::string err;
        if (english) {
            if (!fe->en->text_to_chunks(text_utf8, chunks, err)) return set_error(KT08_ERR_RUNTIME, err);
        } else {
            // Supported languages only (kSupportedV1/V2 in tiny_lang_frontend.cpp); others return KT08_ERR_UNSUPPORTED.
            int rc = fe->other->text_to_chunks(text_utf8, language, chunks, err);
            if (rc != 0) return set_error(static_cast<kt08_status>(rc), err);
        }
        size_t n = 1;
        for (auto& c : chunks) n += c.size() + 1;
        char* buf = static_cast<char*>(std::malloc(n));
        if (!buf) return set_error(KT08_ERR_MEMORY, "out of memory");
        size_t p = 0;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i) buf[p++] = '\n';
            std::memcpy(buf + p, chunks[i].data(), chunks[i].size());
            p += chunks[i].size();
        }
        buf[p] = 0;
        *out_chunks = buf;
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_RUNTIME, e.what());
    } catch (...) {
        return set_error(KT08_ERR_RUNTIME, "unknown error in the text front end");
    }
}

}  // extern "C"
