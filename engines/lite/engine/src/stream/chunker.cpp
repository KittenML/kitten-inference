#include "stream/chunker.hpp"

#include "engine/lead_trim.hpp"
#include "frontend/tiny_en/uni.hpp"

namespace kitten_stream {

using kt_tinyen::is_space;
using kt_tinyen::u32;

namespace {

u32 to32(const std::string& s) { return kt_tinyen::utf8_to_u32(s); }
std::string to8(const u32& s) { return kt_tinyen::u32_to_utf8(s); }

// ensure_punctuation (tts08/preprocess.py): a "," unless the piece ends in . ! ? , ; :
std::string punctuated(const u32& piece) {
    static const u32 ends = U".!?,;:";
    return to8(piece.empty() || ends.find(piece.back()) != u32::npos ? piece : piece + U",");
}

// A sentence longer than limit: words (split at white space) packed greedily into pieces of at most limit code points.
void split_words(const u32& s, size_t limit, bool punctuate, std::vector<std::string>& out) {
    auto emit = [&](const u32& piece) { out.push_back(punctuate ? punctuated(piece) : to8(piece)); };
    u32 cur;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_space(s[i])) ++i;
        size_t j = i;
        while (j < s.size() && !is_space(s[j])) ++j;
        if (j == i) break;
        const u32 word = s.substr(i, j - i);
        if (!cur.empty() && cur.size() + 1 + word.size() > limit) {
            emit(cur);
            cur.clear();
        }
        if (!cur.empty()) cur += U' ';
        cur += word;
        i = j;
    }
    if (!cur.empty()) emit(cur);
}

}  // namespace

std::vector<std::string> pack_sentences(const std::vector<std::string>& sentences, int chunk_chars, int max_chars,
                                        bool punctuate_cuts) {
    std::vector<std::string> out;
    size_t limit = 0;  // 0: one sentence per chunk
    if (chunk_chars > 0) limit = (size_t)chunk_chars;
    if (max_chars > 0 && limit > (size_t)max_chars) limit = (size_t)max_chars;
    u32 cur;
    for (const std::string& raw : sentences) {
        const u32 s = kt_tinyen::strip(to32(raw));
        if (s.empty()) continue;
        if (limit == 0) {
            out.push_back(to8(s));
            continue;
        }
        if (s.size() > limit) {  // a long sentence: its own pieces, cut at spaces
            if (!cur.empty()) out.push_back(to8(cur));
            cur.clear();
            split_words(s, limit, punctuate_cuts, out);
            continue;
        }
        if (!cur.empty() && cur.size() + 1 + s.size() > limit) {
            out.push_back(to8(cur));
            cur.clear();
        }
        if (!cur.empty()) cur += U' ';
        cur += s;
    }
    if (!cur.empty()) out.push_back(to8(cur));
    return out;
}

std::vector<std::string> split_sentences(const std::string& text) {
    // re.split(r"(?<=[.!?…।])\s+|(?<=[。！？])", text)
    static const u32 ends = U".!?…।", cjk = U"。！？";
    const u32 t = to32(text);
    std::vector<std::string> out;
    size_t st = 0, i = 0;
    while (i < t.size()) {
        if (is_space(t[i]) && i > 0 && ends.find(t[i - 1]) != u32::npos) {
            size_t j = i;
            while (j < t.size() && is_space(t[j])) ++j;
            out.push_back(to8(t.substr(st, i - st)));
            st = i = j;
            continue;
        }
        if (cjk.find(t[i]) != u32::npos) {
            out.push_back(to8(t.substr(st, i + 1 - st)));
            st = i + 1;
        }
        ++i;
    }
    out.push_back(to8(t.substr(st)));
    return out;
}

size_t lead_trim_offset(const float* x, size_t n, int sample_rate) {
    return kitten_lead_trim_offset(x, n, sample_rate);
}

size_t tail_keep(const float* x, size_t n, int sample_rate, double keep_ms) {
    return keep_ms > 0 ? kitten_tail_keep(x, n, sample_rate, (float)keep_ms) : n;
}

}  // namespace kitten_stream
