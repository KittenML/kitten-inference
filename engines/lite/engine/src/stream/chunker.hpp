#pragma once
// Streaming for Kitten TTS 0.8 and Kitten TTS 2 mini/nano/pico: which text goes into which chunk. The same code runs on
// every path: the phone C library (kt08_stream_*, src/capi/kitten_stream.cpp) and the Python package's stream() on the
// CPU, Metal and CUDA (through the _engine module, python/bindings.cpp). Nothing is measured: the same sentences and
// chunk_chars give the same chunks on every machine.
#include <cstddef>
#include <string>
#include <vector>

namespace kitten_stream {

constexpr int kTts08MaxChars = 400;     // Kitten TTS 0.8: most characters per chunk (the package's sentence limit)
constexpr int kTinyChunkChars = 380;    // Kitten TTS 2 mini/nano/pico: stream()'s default chunk_chars (KittenTTS 2's)
constexpr int kOneSentence = 0;         // pack_sentences' chunk_chars for one sentence per chunk (TTS 0.8's default)
constexpr double kTailKeepMs = 30.0;    // Kitten TTS 2: end silence a chunk keeps when another chunk follows

// Sentences (in order, as the model family's own splitter cut them) to chunks:
//   chunk_chars <= 0 (kOneSentence): one sentence per chunk (the default of Kitten TTS 0.8's stream(); users ask for
//   their model's default with chunk_chars None in Python and 0 in the C library, never for this directly);
//   chunk_chars N: whole sentences joined with one space while the chunk has at most N characters (code points); a
//   sentence longer than N is split at spaces into pieces of at most N characters, each its own chunk (a single word
//   longer than N stays whole).
// max_chars > 0 caps N (Kitten TTS 0.8: kTts08MaxChars). Each sentence is stripped of white space at both ends; empty
// ones are skipped. punctuate_cuts: a piece cut from a longer sentence that does not end in . ! ? , ; : gets a ","
// (Kitten TTS 0.8, as its chunk_text's ensure_punctuation). The model's phoneme limit (Kitten TTS 2: 350) is applied
// later, by the front end, to each chunk's phonemes.
std::vector<std::string> pack_sentences(const std::vector<std::string>& sentences, int chunk_chars, int max_chars = 0,
                                        bool punctuate_cuts = false);

// Kitten TTS 2 mini/nano/pico's sentence split (the package's SENT_MULTI: after . ! ? … । and white space, or after
// 。！？). Pieces are returned as cut (not stripped; pack_sentences strips them).
std::vector<std::string> split_sentences(const std::string& text);

// The silence trims of stream() (engine/lead_trim.hpp; only samples below -50 dBFS go):
// samples to drop from a chunk's start (20 ms before its first sample at or above -50 dBFS; 0 when it has none) ...
size_t lead_trim_offset(const float* x, size_t n, int sample_rate = 24000);
// ... and samples to keep: up to keep_ms after its last sample at or above -50 dBFS.
size_t tail_keep(const float* x, size_t n, int sample_rate = 24000, double keep_ms = kTailKeepMs);

}  // namespace kitten_stream
