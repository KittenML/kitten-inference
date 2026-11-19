// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/hi.hpp"

namespace kt_fe {
namespace hi {

bool is_devanagari(char32_t c) { return c >= 0x900 && c <= 0x97F; }

static bool is_vowel_sign(char32_t c) {
    return c == 0x93A || c == 0x93B || (c >= 0x93E && c <= 0x94C) || c == 0x94E || c == 0x94F ||
           (c >= 0x955 && c <= 0x957) || c == 0x962 || c == 0x963;
}
bool is_vowel(char32_t c) {
    return (c >= 0x904 && c <= 0x914) || c == 0x960 || c == 0x961 || (c >= 0x972 && c <= 0x977) || is_vowel_sign(c);
}
bool is_sign_or_virama(char32_t c) { return is_vowel_sign(c) || c == 0x94D; }
bool is_consonant(char32_t c) { return (c >= 0x915 && c <= 0x939) || (c >= 0x958 && c <= 0x95F) || (c >= 0x978 && c <= 0x97F); }

bool in_group(char group, char32_t c, bool* handled) {
    *handled = true;
    switch (group) {
        case 'A': case 'Y': return is_vowel(c);
        case 'B': return is_sign_or_virama(c);
        case 'C': return is_consonant(c);
    }
    *handled = false;
    return false;
}

// Fitted to the reference on FLEURS words. A syllable weighs 1 for a long vowel plus 1 when it is closed (two or more
// consonants before the next vowel); the final syllable is closed by one consonant but weighs 1 less (an open final
// syllable with a long vowel is light). The primary stress goes on the last of the heaviest syllables; when no
// syllable is heavier than light, on the first.
int primary_stress(const std::vector<Syl>& syl) {
    const int n = (int)syl.size();
    if (n == 0) return -1;
    if (n == 1) return 0;
    int best = -1, at = 0;
    for (int k = 0; k < n; k++) {
        const bool last = k + 1 == n;
        int w = (syl[k].long_vowel ? 1 : 0) + (syl[k].coda >= (last ? 1 : 2) ? 1 : 0) - (last ? 1 : 0);
        if (w >= best) { best = w; at = k; }
    }
    return best <= 0 ? 0 : at;
}

// Indian grouping: [N kharab] [N arab] [N crore] [N lakh] [N hazaar] [N-sau] [N], N < 100 from the word list (_0 .. _99). The
// hundreds are one written word with two stresses (ek-sau); the other parts are words of their own.
std::string number(const std::u32string& digits, const std::function<std::string(const std::string&)>& ph) {
    if (digits.empty()) return "";
    if (digits.size() > 1 && digits[0] == '0') {
        // leading zero, measured: up to 3 digits the zeros (joined) then the rest as a number (05, 007, 089);
        // 4 or more digits one by one in pairs, joined (0230)
        auto d = [&](size_t k) { return ph(std::string("_") + (char)digits[k]); };
        const size_t n = digits.size();
        std::string o;
        if (n >= 4) {
            // in pairs (02|30): a pair is one word, the pairs follow without a space (\x02\x02)
            for (size_t k = 0; k < n; k++) { if (k) o += (k % 2) ? "\x02" : "\x02\x02"; o += d(k); }
            return o;
        }
        size_t k = 0;
        while (k + 1 < n && digits[k] == '0') { if (k) o += "\x02"; o += d(k); k++; }
        return o + "||" + number(digits.substr(k), ph);
    }
    if (digits.size() > 14) {   // longer than espeak-ng's named orders (14 digits): digit by digit
        std::string o;
        for (char32_t d : digits) { if (!o.empty()) o += "||"; o += ph(std::string("_") + (char)d); }
        return o;
    }
    unsigned long long v = 0;
    for (char32_t d : digits) v = v * 10 + (unsigned)(d - '0');
    if (v == 0) return ph("_0");
    auto lt100 = [&](unsigned n) { return ph("_" + std::to_string(n)); };
    std::string o;
    auto add = [&](const std::string& w) { if (w.empty()) return; if (!o.empty()) o += "||"; o += w; };
    const unsigned long long scale[] = {100000000000ULL, 1000000000ULL, 10000000ULL, 100000ULL, 1000ULL};
    const char* const name[] = {"_0M5", "_0M4", "_0M3", "_0M2", "_0M1"};   // kharab, arab, crore, lakh, hazaar
    for (int s = 0; s < 5; s++) {
        unsigned long long g = v / scale[s];
        v %= scale[s];
        if (!g) continue;
        if (s == 0 && g >= 100) {
            std::u32string t;
            for (char c : std::to_string(g)) t += (char32_t)c;
            add(number(t, ph));
        } else {
            // an entry without a stress mark (_17 sVtrVh) takes a secondary stress before a scale word
            std::string w = lt100((unsigned)g);
            add(w.find('\'') == std::string::npos && w.find(',') == std::string::npos ? "," + w : w);
        }
        add(ph(name[s]));
    }
    if (v >= 100) { add(lt100((unsigned)(v / 100)) + "\x02" + ph("_0C")); v %= 100; }
    if (v) {
        // an entry without a stress mark (_17 sVtrVh, _68 Vr.sVt.#) after other parts of the number: secondary
        // stress on its last vowel (measured: 1968); alone it is stressed as a word
        std::string w = lt100((unsigned)v);
        if (!o.empty() && w.find('\'') == std::string::npos && w.find(',') == std::string::npos) {
            size_t p = w.find_last_of("aeiouAEIOUV@");
            while (p != std::string::npos && p > 0 && std::string("aeiouAEIOUV@").find(w[p - 1]) != std::string::npos) p--;
            if (p != std::string::npos) w.insert(p, ",");
        }
        add(w);
    }
    return o;
}

bool english_spells(const std::u32string& w, bool listed_abbrev) {
    if (listed_abbrev || w.size() <= 3) return true;   // measured: ADT, AUD, APS, REM, GMT spelled (AD, CEP are not: not modelled)
    auto vowel = [](char32_t c) { return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'; };
    bool any_v = false, any_c = false;
    for (char32_t c : w) (vowel(c) ? any_v : any_c) = true;
    if (!any_v || !any_c) return true;
    if (w.size() >= 2 && !vowel(w[0]) && !vowel(w[1])) {
        static const char* const onset[] = {"BL", "BR", "CL", "CR", "CH", "DR", "FL", "FR", "GL", "GR", "KN", "PH", "PL", "PR",
                                            "SC", "SH", "SK", "SL", "SM", "SN", "SP", "ST", "SW", "TH", "TR", "TW", "WH", "WR"};
        for (auto* o : onset) if (w[0] == (char32_t)o[0] && w[1] == (char32_t)o[1]) return false;
        return true;
    }
    return false;
}

}  // namespace hi
}  // namespace kt_fe
