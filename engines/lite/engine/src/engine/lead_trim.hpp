#pragma once
// Silence trims for streaming (stream() and kt08_stream; only samples below -50 dBFS are dropped).
//
// Every Kitten chunk starts with silence a listener cannot hear (about 0.2-0.3 s for Kitten TTS 2 mini/nano/pico, 0.4 s
// for Kitten TTS 0.8): exact zeros, then noise at -57 to -79 dBFS.
// For the FIRST chunk of a request that lead is pure waiting time, so stream() drops it: kitten_lead_trim_offset()
// returns how many samples to drop (the first sample at or above the threshold, minus a 20 ms pre-roll). Kitten TTS 2
// mini/nano/pico later chunks get the same lead trim (after keeping the model's start, so no word onset is lost), and a
// chunk that another follows keeps only 30 ms after its last sample at or above the threshold (kitten_tail_keep()), so
// with the player's 0.15 s gap the pause at a chunk edge is about generate()'s pause at a sentence end. Kitten TTS 0.8
// later chunks keep their lead: it is the pause between sentences.
//
// Lossless: only samples below the threshold go; every kept sample is unchanged (a trimmed chunk is a contiguous part of
// the untrimmed one) and the model's compute is unchanged. It is an output policy, and callers should label it
// "lead-trimmed".
#include <cmath>
#include <cstddef>

// Index of the first sample with |x| >= 10^(dbfs/20), or n when there is none.
inline size_t kitten_first_speech_sample(const float* x, size_t n, float dbfs = -50.0f) {
    const float thr = std::pow(10.0f, dbfs / 20.0f);
    for (size_t i = 0; i < n; ++i)
        if (std::fabs(x[i]) >= thr) return i;
    return n;
}

// Samples to keep of a chunk whose trailing silence is shortened: up to keep_ms after the last sample at or above the
// threshold (n when the chunk has no such sample, or less than keep_ms after it). Lossless in the same sense: the kept
// samples are a prefix of the chunk.
inline size_t kitten_tail_keep(const float* x, size_t n, int sample_rate = 24000, float keep_ms = 30.0f,
                               float dbfs = -50.0f) {
    const float thr = std::pow(10.0f, dbfs / 20.0f);
    size_t last = n;
    for (size_t i = n; i > 0; --i)
        if (std::fabs(x[i - 1]) >= thr) { last = i - 1; break; }
    if (last >= n) return n;
    const size_t keep = (size_t)std::lround(keep_ms * 1e-3f * (float)sample_rate);
    return last + 1 + keep < n ? last + 1 + keep : n;
}

// Samples to drop from the start of a stream's first chunk: first speech minus preroll_ms (never negative). When the
// chunk has no sample at the threshold, nothing is dropped.
inline size_t kitten_lead_trim_offset(const float* x, size_t n, int sample_rate = 24000, float dbfs = -50.0f,
                                      float preroll_ms = 20.0f) {
    const size_t first = kitten_first_speech_sample(x, n, dbfs);
    if (first >= n) return 0;
    const size_t pre = (size_t)std::lround(preroll_ms * 1e-3f * (float)sample_rate);
    return first > pre ? first - pre : 0;
}
