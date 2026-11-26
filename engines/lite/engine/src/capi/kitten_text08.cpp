// C API: English text to phonemes for Kitten TTS 0.8 (kitten_tts08.h, API version 4).
// Normalisation and chunking: src/frontend/tts08_text.cpp (port of the Python package's preprocess.py);
// phonemes: src/frontend/cephonemizer (KittenML's CEPhonemizer, Apache-2.0).
#include "capi_internal.hpp"
#include "frontend/tts08_text.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using kt08_internal::set_error;

struct kt08_phonemizer {
    kt08fe::Frontend08 fe;
    // The last kt08_text_to_phonemes / kt08_chunk_to_phonemes result, so the size query and the real call do the
    // work once.
    std::string last_in, last_out;
    bool last_whole = false;
    kt08_phonemizer(const std::string& r, const std::string& l) : fe(r, l) {}
};

namespace {
const kt08fe::Normalizer08& normalizer() {
    static const kt08fe::Normalizer08 n;  // patterns compiled once, on first use
    return n;
}

kt08_status give(const std::string& s, char* out, size_t capacity, size_t* out_size) {
    *out_size = s.size() + 1;
    if (s.size() + 1 > capacity) {
        if (capacity == 0 && !out) return KT08_OK;  // size query
        return set_error(KT08_ERR_ARGUMENT, "capacity is smaller than *out_size");
    }
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out is NULL");
    std::memcpy(out, s.c_str(), s.size() + 1);
    return KT08_OK;
}

bool to_u32(const char* text, std::u32string& u) {
    bool ok = true;
    u = kt08fe::utf8_to_u32(text, &ok);
    return ok;
}

// The normalisation's patterns (Python's own) backtrack over a run of letters or digits once per start position,
// so their time grows with the square of the longest run. Runs of more than kMaxRun characters without a space
// (never words; e.g. a pasted hash or a long digit string) are refused so one call cannot take seconds on a phone.
constexpr size_t kMaxRun = 1000;
bool has_long_run(const std::u32string& u) {
    size_t run = 0;
    for (char32_t c : u) {
        run = kt08fe::cp_is_space(c) ? 0 : run + 1;
        if (run > kMaxRun) return true;
    }
    return false;
}
kt08_status too_long() {
    return set_error(KT08_ERR_TOO_LONG, "text has more than 1000 characters without a space; split it or remove that part");
}

std::vector<std::u32string> chunk_list(const std::u32string& text) {
    return kt08fe::Normalizer08::chunks(normalizer().normalize(text));
}

std::string chunks_joined(const std::u32string& text) {
    std::string out;
    for (const auto& c : chunk_list(text)) {
        if (!out.empty()) out.push_back('\n');
        out += kt08fe::u32_to_utf8(c);
    }
    return out;
}

template <class F>
kt08_status guarded(F&& f) {
    try {
        return f();
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory in the text front end");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_RUNTIME, std::string("text front end: ") + e.what());
    } catch (...) {
        return set_error(KT08_ERR_RUNTIME, "text front end: unknown error");
    }
}
}  // namespace

extern "C" {

kt08_status kt08_phonemizer_load(const char* en_rules_path, const char* en_list_path, kt08_phonemizer** out) {
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out_phonemizer is NULL");
    *out = nullptr;
    if (!en_rules_path || !en_list_path) return set_error(KT08_ERR_ARGUMENT, "en_rules_path and en_list_path are required");
    try {
        normalizer();
        *out = new kt08_phonemizer(en_rules_path, en_list_path);
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory while loading the phonemizer");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_IO, e.what());
    } catch (...) {
        return set_error(KT08_ERR_IO, "cannot load the phonemizer data");
    }
}

void kt08_phonemizer_free(kt08_phonemizer* ph) { delete ph; }

kt08_status kt08_text_chunks(const char* text, char* out, size_t capacity, size_t* out_size) {
    if (!text || !out_size) return set_error(KT08_ERR_ARGUMENT, "text_utf8 and out_size are required");
    return guarded([&] {
        std::u32string u;
        if (!to_u32(text, u)) return set_error(KT08_ERR_ARGUMENT, "text_utf8 is not valid UTF-8");
        if (has_long_run(u)) return too_long();
        return give(chunks_joined(u), out, capacity, out_size);
    });
}

kt08_status kt08_chunk_to_phonemes(kt08_phonemizer* ph, const char* chunk, char* out, size_t capacity,
                                   size_t* out_size) {
    if (!ph || !chunk || !out_size) return set_error(KT08_ERR_ARGUMENT, "phonemizer, chunk_utf8 and out_size are required");
    return guarded([&] {
        if (ph->last_whole || ph->last_in != chunk) {
            std::u32string u;
            if (!to_u32(chunk, u)) return set_error(KT08_ERR_ARGUMENT, "chunk_utf8 is not valid UTF-8");
            if (has_long_run(u)) return too_long();
            ph->last_out = ph->fe.chunk_to_phonemes(chunk);
            ph->last_in = chunk;
            ph->last_whole = false;
        }
        return give(ph->last_out, out, capacity, out_size);
    });
}

kt08_status kt08_text_to_phonemes(kt08_phonemizer* ph, const char* text, char* out, size_t capacity,
                                  size_t* out_size) {
    if (!ph || !text || !out_size) return set_error(KT08_ERR_ARGUMENT, "phonemizer, text_utf8 and out_size are required");
    return guarded([&] {
        if (!ph->last_whole || ph->last_in != text) {
            std::u32string u;
            if (!to_u32(text, u)) return set_error(KT08_ERR_ARGUMENT, "text_utf8 is not valid UTF-8");
            if (has_long_run(u)) return too_long();
            std::string res;
            for (const auto& c : chunk_list(u)) {
                if (!res.empty()) res.push_back('\n');
                res += ph->fe.chunk_to_phonemes(kt08fe::u32_to_utf8(c));
            }
            ph->last_out = std::move(res);
            ph->last_in = text;
            ph->last_whole = true;
        }
        return give(ph->last_out, out, capacity, out_size);
    });
}

}  // extern "C"
