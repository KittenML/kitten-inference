#pragma once
// Lead trim for the first chunk of a stream.
//
// Every Kitten chunk starts with ~0.4 s that a listener cannot hear: ~230 ms of exact zeros, then noise at -57 to
// -79 dBFS. The first speech sample (|x| >= -50 dBFS) is 394-465 ms in. For the FIRST chunk of a request that lead is
// pure waiting time, so a streaming caller may drop it: kitten_lead_trim_offset() returns how many samples to drop
// (the first sample at or above the threshold, minus a pre-roll). Later chunks keep their lead: it is the pause
// between sentences.
//
// Lossless: the trimmed stream is an exact suffix of the untrimmed one (every kept sample is unchanged, the model's
// compute is unchanged). It is an output policy, and callers should label it "lead-trimmed".
#include <cmath>
#include <cstddef>

// Index of the first sample with |x| >= 10^(dbfs/20), or n when there is none.
inline size_t kitten_first_speech_sample(const float* x, size_t n, float dbfs = -50.0f) {
    const float thr = std::pow(10.0f, dbfs / 20.0f);
    for (size_t i = 0; i < n; ++i)
        if (std::fabs(x[i]) >= thr) return i;
    return n;
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
