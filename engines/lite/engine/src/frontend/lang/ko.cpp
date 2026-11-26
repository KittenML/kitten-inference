// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Korean helpers for the espeak-data reader; see ko.h. Original code (Unicode Hangul arithmetic, measured behaviour).
#include "frontend/lang/ko.h"

namespace kt_fe {
namespace ko {

bool is_jamo(char32_t c) { return c >= 0x1100 && c <= 0x11FF; }
bool is_vowel_jamo(char32_t c) { return c >= 0x1161 && c <= 0x11A7; }

std::u32string decompose(const std::u32string& t) {
    // Unicode 15, 3.12 "Conjoining Jamo Behavior": S = SBase + (L * VCount + V) * TCount + T
    const char32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    const int VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = 19 * NCount;
    std::u32string o;
    o.reserve(t.size() * 3);
    for (char32_t c : t) {
        if (c < SBase || c >= SBase + (char32_t)SCount) { o += c; continue; }
        int s = (int)(c - SBase);
        int l = s / NCount, v = (s % NCount) / TCount, tt = s % TCount;
        if (l != 11) o += (char32_t)(LBase + l);   // 11 = ㅇ, silent before a vowel
        o += (char32_t)(VBase + v);
        if (tt) o += (char32_t)(TBase + tt);
    }
    return o;
}

int stress_vowel(const std::string& shape) {
    std::vector<size_t> v;
    for (size_t i = 0; i < shape.size(); i++) if (shape[i] == 'V') v.push_back(i);
    if (v.size() < 2) return 0;
    auto heavy = [&](size_t k) {
        size_t end = k + 1 < v.size() ? v[k + 1] : shape.size();
        size_t cons = end - v[k] - 1;
        return k + 1 < v.size() ? cons >= 2 : cons >= 1;
    };
    return (!heavy(0) && heavy(1)) ? 1 : 0;
}

int english_spell_prefix(const std::u32string& w) {
    // word starts that are read as a word (measured: every pair of consonants + "an", and common three-letter starts)
    static const char* const two[] = {"bh", "bj", "bl", "br", "ch", "cl", "cr", "dj", "dr", "dw", "fl", "fr", "gh", "gl", "gn",
        "gr", "gw", "kh", "kl", "kn", "kr", "kw", "lh", "ll", "mc", "nc", "ng", "nk", "nq", "pf", "ph", "pl", "pr", "ps",
        "pt", "rh", "rw", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sv", "sw", "sz", "th", "tr", "ts", "tw", "vl",
        "vr", "wh", "wr", "zh", "zs", "zw"};
    static const char* const three[] = {"str", "spl", "spr", "scr", "sch", "shr", "thr", "chr", "phr"};
    static const char* const four[] = {"schl", "schm", "schn", "schr", "schw"};   // German names ("Schneider")
    const size_t n = w.size();
    for (char32_t c : w) if (c < 'a' || c > 'z') return 0;
    auto plain = [](char32_t c) { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'; };
    // y is a consonant only at the start of a word before a vowel ("yam"); "Shyam", "Nyiragongo" are read as words
    auto vowel = [&](size_t i) { return plain(w[i]) || (w[i] == 'y' && (i > 0 || i + 1 >= n || !plain(w[i + 1]))); };
    bool any = false;
    for (char32_t c : w) any = any || plain(c);
    if (n <= 2 && !any) return -1;
    for (size_t pos = 0;; pos++) {
        size_t v = pos;
        while (v < n && !vowel(v)) v++;
        if (v == n) return -1;   // no vowel left
        std::string onset(w.begin() + pos, w.begin() + v);
        bool ok = onset.size() <= 1;
        if (onset.size() == 2) for (auto* t : two) ok = ok || onset == t;
        if (onset.size() == 3) for (auto* t : three) ok = ok || onset == t;
        if (onset.size() == 4) for (auto* t : four) ok = ok || onset == t;
        if (ok) return (int)pos;
        if (n - (pos + 1) <= 2) return -1;
    }
}

std::vector<int> english_spell_groups(size_t n) {
    std::vector<int> g;
    while (n > 4) { g.push_back(3); n -= 3; }   // measured: 5 = 3 + 2, 6 = 3 + 3, 7 = 3 + 4
    if (n) g.push_back((int)n);
    return g;
}

namespace detail {
std::string join_domains(const std::string& m) {
    // An entry such as "s'ip_!" ends in a short pause; inside a joined number word ("s'ip_!'o_!") each part keeps its
    // own stress. The pause makes no sound in the phoneme output, so it becomes the reader's "new stress domain, no
    // space" mark.
    std::string o;
    for (size_t i = 0; i < m.size();) {
        if (m.compare(i, 2, "_!") == 0) {
            i += 2;
            bool end = i >= m.size() || m.compare(i, 2, "||") == 0 || m[i] == '\x02';
            if (!end) o += '\x02';
            continue;
        }
        o += m[i++];
    }
    return o;
}
}  // namespace detail

}  // namespace ko
}  // namespace kt_fe
