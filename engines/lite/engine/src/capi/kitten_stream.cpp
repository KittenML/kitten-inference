// kt08_stream: streaming for Kitten TTS 2 mini/nano/pico and Kitten TTS 0.8 in the C library (include/kitten_tts08.h,
// API version 6). The chunks come from the chunker every path shares (src/stream/chunker.hpp, also behind the Python
// package's stream()); this file turns each chunk into phonemes when it is next and synthesises it.
#include "kitten_tts08.h"
#include "capi_internal.hpp"
#include "stream/chunker.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct kt08_stream {
    kt08_model* model = nullptr;
    const kt08_vocab* vocab = nullptr;       // Kitten TTS 2 mini/nano/pico (NULL: Kitten TTS 0.8)
    kt08_tiny_frontend* frontend = nullptr;  // Kitten TTS 2 mini/nano/pico
    kt08_phonemizer* phonemizer = nullptr;   // Kitten TTS 0.8
    std::string language;
    const float* voice = nullptr;
    size_t voice_rows = 0;
    float speed = 1.0f;
    bool trim_first = true;
    std::vector<std::string> chunks;  // text chunks (kitten_stream::pack_sentences)
    size_t next_chunk = 0;
    std::vector<std::string> pieces;  // phonemes of the current chunk (Kitten TTS 2: more than one past 350 phonemes)
    size_t next_piece = 0;
    std::string piece_text;           // the current chunk's text (Kitten TTS 0.8: it picks the voice style row)
    bool started = false;             // a chunk was synthesised
};

namespace {

using kt08_internal::set_error;

kt08_stream_options read_options(const kt08_stream_options* options, kt08_status* st) {
    kt08_stream_options o;
    kt08_stream_options_init(&o);
    *st = KT08_OK;
    if (options) {  // the fields the caller's struct has
        const size_t n = options->struct_size < sizeof o ? options->struct_size : sizeof o;
        if (n < sizeof(uint32_t)) {
            *st = set_error(KT08_ERR_ARGUMENT, "options->struct_size is not set");
            return o;
        }
        std::memcpy(&o, options, n);
        o.struct_size = sizeof o;
    }
    if (o.chunk_chars < 0)
        *st = set_error(KT08_ERR_ARGUMENT,
                        "chunk_chars must be 0 (the model's default) or a positive whole number of characters");
    return o;
}

// chunk_chars for kitten_stream::pack_sentences: 0 in the options is the model's default (as None in Python).
int family_chunk_chars(int chunk_chars, int family_default) {
    return chunk_chars == 0 ? family_default : chunk_chars;
}
static_assert(KT08_STREAM_TINY_CHUNK_CHARS == kitten_stream::kTinyChunkChars, "one default for C and Python");

kt08_status phonemes08(kt08_phonemizer* ph, const std::string& chunk, std::string& out) {
    size_t need = 0;
    kt08_status st = kt08_chunk_to_phonemes(ph, chunk.c_str(), nullptr, 0, &need);
    if (st != KT08_OK && need == 0) return st;
    out.assign(need, '\0');
    st = kt08_chunk_to_phonemes(ph, chunk.c_str(), &out[0], need, &need);
    if (st == KT08_OK) out.resize(need ? need - 1 : 0);
    return st;
}

// The next chunk's phonemes into s->pieces (none for a chunk with nothing to say). On an error the stream stays at
// that chunk, so the next call retries it.
kt08_status load_chunk(kt08_stream* s) {
    s->pieces.clear();
    s->next_piece = 0;
    const std::string& text = s->chunks[s->next_chunk];
    kt08_status st;
    if (s->vocab) {
        st = kt08_internal::tiny_chunk_phonemes(s->frontend, text, s->language, s->pieces);
    } else {
        std::string ph;
        st = phonemes08(s->phonemizer, text, ph);
        if (st == KT08_OK && !ph.empty()) s->pieces.push_back(ph);
    }
    if (st != KT08_OK) {
        s->pieces.clear();
        return st;
    }
    s->piece_text = text;
    ++s->next_chunk;
    return KT08_OK;
}

// Drops the start of *out up to 20 ms before its first sample at or above -50 dBFS (only silence goes).
void trim_lead(float* out, size_t* n_out) {
    const size_t off = kitten_stream::lead_trim_offset(out, *n_out, KT08_SAMPLE_RATE);
    if (off > 0 && off < *n_out) {
        std::memmove(out, out + off, (*n_out - off) * sizeof(float));
        *n_out -= off;
    }
}

kt08_status synth(kt08_stream* s, const std::string& ph, bool first, bool last, float** out, size_t* n_out) {
    size_t n = 0;
    kt08_status st = s->vocab ? kt08_vocab_ids(s->vocab, ph.c_str(), nullptr, 0, &n) : kt08_phonemes_to_ids(ph.c_str(), nullptr, 0, &n);
    if (st != KT08_OK) return st;
    std::vector<int32_t> ids(n);
    st = s->vocab ? kt08_vocab_ids(s->vocab, ph.c_str(), ids.data(), n, &n) : kt08_phonemes_to_ids(ph.c_str(), ids.data(), n, &n);
    if (st != KT08_OK) return st;
    if (s->vocab) {
        // Every chunk keeps the model's start (first_chunk 1: dropping the start pad's frames cut the first sound of
        // some words), then drops its silent start up to 20 ms before the first audible sample, except a first chunk
        // kept whole. A chunk that another follows keeps 30 ms of silence after its last audible sample.
        st = kt08_synthesize_tiny(s->model, ids.data(), n, s->voice, s->voice_rows, kt08_style_row_for_ids(n, s->voice_rows),
                                  s->speed, 1, out, n_out);
        if (st != KT08_OK) return st;
        if (!first || s->trim_first) trim_lead(*out, n_out);
        if (!last) *n_out = kitten_stream::tail_keep(*out, *n_out, KT08_SAMPLE_RATE);
        return st;
    }
    // Kitten TTS 0.8: kt08_synthesize's audio; only the first chunk's silent start goes (later chunks keep theirs: it is
    // the pause between sentences).
    st = kt08_synthesize(s->model, ids.data(), n, s->voice, s->voice_rows,
                         kt08_style_row_for_text(s->piece_text.c_str(), s->voice_rows), s->speed, out, n_out);
    if (st == KT08_OK && first && s->trim_first) trim_lead(*out, n_out);
    return st;
}

kt08_status make(kt08_stream* proto, const std::vector<std::string>& sentences, int chunk_chars, int max_chars,
                 kt08_stream** out) {
    auto s = std::make_unique<kt08_stream>(std::move(*proto));
    // Kitten TTS 0.8 (max_chars set): a sentence cut at spaces gets end punctuation, as the package's chunk_text.
    s->chunks = kitten_stream::pack_sentences(sentences, chunk_chars, max_chars, max_chars > 0);
    *out = s.release();
    return KT08_OK;
}

}  // namespace


extern "C" {

void kt08_stream_options_init(kt08_stream_options* o) {
    if (!o) return;
    std::memset(o, 0, sizeof *o);
    o->struct_size = sizeof *o;
    o->chunk_chars = 0;
    o->trim_first_pause = 1;
}

kt08_status kt08_stream_open(kt08_model* model, const kt08_vocab* vocab, kt08_tiny_frontend* frontend,
                             const char* text_utf8, const char* language, const float* voice, size_t voice_rows,
                             float speed, const kt08_stream_options* options, kt08_stream** out) {
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out is NULL");
    *out = nullptr;
    if (!model || !vocab || !frontend || !text_utf8 || !language || !voice || voice_rows == 0)
        return set_error(KT08_ERR_ARGUMENT, "model, vocab, frontend, text_utf8, language and voice are required");
    if (!(speed >= KT08_TINY_SPEED_MIN && speed <= KT08_TINY_SPEED_MAX))
        return set_error(KT08_ERR_ARGUMENT, "speed must be from 0.25 to 4 (1.0 is normal)");
    kt08_status st;
    const kt08_stream_options o = read_options(options, &st);
    if (st != KT08_OK) return st;
    try {
        std::vector<std::string> sentences;
        st = kt08_internal::tiny_sentences(frontend, text_utf8, language, sentences);
        if (st != KT08_OK) return st;
        kt08_stream s;
        s.model = model;
        s.vocab = vocab;
        s.frontend = frontend;
        s.language = language;
        s.voice = voice;
        s.voice_rows = voice_rows;
        s.speed = speed;
        s.trim_first = o.trim_first_pause != 0;
        return make(&s, sentences, family_chunk_chars(o.chunk_chars, KT08_STREAM_TINY_CHUNK_CHARS), 0, out);
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory");
    }
}

kt08_status kt08_stream_open_tts08(kt08_model* model, kt08_phonemizer* phonemizer, const char* text_utf8,
                                   const float* voice, size_t voice_rows, float speed,
                                   const kt08_stream_options* options, kt08_stream** out) {
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out is NULL");
    *out = nullptr;
    if (!model || !phonemizer || !text_utf8 || !voice || voice_rows == 0)
        return set_error(KT08_ERR_ARGUMENT, "model, phonemizer, text_utf8 and voice are required");
    if (!std::isfinite(speed) || speed <= 0.0f)
        return set_error(KT08_ERR_ARGUMENT, "speed must be a positive finite value");
    kt08_status st;
    const kt08_stream_options o = read_options(options, &st);
    if (st != KT08_OK) return st;
    try {
        size_t need = 0;  // the package's sentences (normalised; at most 400 characters each)
        st = kt08_text_chunks(text_utf8, nullptr, 0, &need);
        if (st != KT08_OK && need == 0) return st;
        std::string joined(need, '\0');
        st = kt08_text_chunks(text_utf8, &joined[0], need, &need);
        if (st != KT08_OK) return st;
        joined.resize(need ? need - 1 : 0);
        std::vector<std::string> sentences;
        for (size_t a = 0; a <= joined.size();) {
            size_t b = joined.find('\n', a);
            if (b == std::string::npos) b = joined.size();
            sentences.push_back(joined.substr(a, b - a));
            a = b + 1;
        }
        kt08_stream s;
        s.model = model;
        s.phonemizer = phonemizer;
        s.voice = voice;
        s.voice_rows = voice_rows;
        s.speed = speed;
        s.trim_first = o.trim_first_pause != 0;
        return make(&s, sentences, family_chunk_chars(o.chunk_chars, kitten_stream::kOneSentence),
                    kitten_stream::kTts08MaxChars, out);
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory");
    }
}

kt08_status kt08_stream_next(kt08_stream* s, float** out_audio, size_t* out_samples, int* out_last) {
    if (!s || !out_audio || !out_samples)
        return set_error(KT08_ERR_ARGUMENT, "stream, out_audio and out_samples are required");
    *out_audio = nullptr;
    *out_samples = 0;
    if (out_last) *out_last = 1;
    try {
        while (s->next_piece >= s->pieces.size()) {
            if (s->next_chunk >= s->chunks.size()) return KT08_OK;  // finished: no audio
            const kt08_status st = load_chunk(s);
            if (st != KT08_OK) return st;
        }
        const std::string& ph = s->pieces[s->next_piece++];
        const bool last = s->next_piece >= s->pieces.size() && s->next_chunk >= s->chunks.size();
        const kt08_status st = synth(s, ph, !s->started, last, out_audio, out_samples);
        if (st != KT08_OK) return st;
        s->started = true;
        if (out_last) *out_last = last ? 1 : 0;
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        kt08_free(*out_audio);
        *out_audio = nullptr;
        *out_samples = 0;
        return set_error(KT08_ERR_MEMORY, "out of memory");
    }
}

kt08_status kt08_stream_chunks(const kt08_stream* s, char** out_chunks) {
    if (!out_chunks) return set_error(KT08_ERR_ARGUMENT, "out_chunks is NULL");
    *out_chunks = nullptr;
    if (!s) return set_error(KT08_ERR_ARGUMENT, "stream is NULL");
    std::string joined;
    for (size_t i = 0; i < s->chunks.size(); ++i) {
        if (i) joined += '\n';
        joined += s->chunks[i];
    }
    char* buf = static_cast<char*>(std::malloc(joined.size() + 1));
    if (!buf) return set_error(KT08_ERR_MEMORY, "out of memory");
    std::memcpy(buf, joined.c_str(), joined.size() + 1);
    *out_chunks = buf;
    return KT08_OK;
}

void kt08_stream_free(kt08_stream* s) {
    delete s;
}

}  // extern "C"
