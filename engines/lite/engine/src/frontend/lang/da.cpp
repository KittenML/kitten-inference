// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Danish number reading, written from the word list's own entries and from how the Python package's espeak-ng reads
// numbers (measured): units come before the tens and are joined to them with "og" in one word (seksogtres),
// "hundrede" is joined to its multiplier (firehundrede) and takes "og" when more follows, a thousand group is its own
// word, "og" stands alone before a final part under 100 after "tusind" / "million(er)", and 1900-1999 are read as
// "nitten hundrede ...".
#include "frontend/lang/da.h"

#include <cstring>

namespace kt_fe {
namespace da {
namespace {

const char kJoin = '\x05';

// "og" inside a number takes a secondary stress when it stands between two unstressed syllables: the word before it
// ends in one (otte, fire, hundrede; not fem, to) and the word after it starts with one (halvtreds, halvfjerds,
// halvfems; not tyve, tres). Measured on the Python package's output.
bool is_vowel_char(char c) { return c != 0 && std::strchr("aeiouyAEIOUVWY&@03", c) != nullptr; }
bool starts_unstressed(const std::string& s) {
    for (char c : s) {
        if (c == '\'') return false;
        if (is_vowel_char(c)) return true;
    }
    return false;
}
bool ends_unstressed(const std::string& s) {
    size_t st = s.rfind('\'');
    int groups = 0; bool in = false;
    for (size_t i = st == std::string::npos ? 0 : st; i < s.size(); i++) {
        bool v = is_vowel_char(s[i]);
        if (v && (!in || s[i] == 'V')) groups++;   // "fire" = f'iV: the V is a syllable of its own
        in = v;
    }
    return st == std::string::npos ? groups >= 1 : groups >= 2;
}
std::string og(const std::string& prev, const std::string& next, const std::function<std::string(const std::string&)>& ph) {
    return (ends_unstressed(prev) && starts_unstressed(next) ? "," : "") + ph("_0and");
}

std::string below100(int n, const std::function<std::string(const std::string&)>& ph) {
    std::string s = ph("_" + std::to_string(n));
    if (!s.empty()) return s;
    int t = n / 10, u = n % 10;
    std::string tens = ph("_" + std::to_string(t) + "X");
    if (u == 0) return tens;
    std::string unit = ph("_" + std::to_string(u));
    return unit + kJoin + og(unit, tens, ph) + kJoin + tens;
}

std::string hundreds(int h, const std::function<std::string(const std::string&)>& ph) {
    if (h == 1) return ph("_1C");
    return ph("_" + std::to_string(h)) + kJoin + ph("_0C");
}

std::string below1000(int n, const std::function<std::string(const std::string&)>& ph) {
    int h = n / 100, r = n % 100;
    if (!h) return below100(r, ph);
    std::string o = hundreds(h, ph);
    if (r) { std::string rest = below100(r, ph); o += kJoin + og(o, rest, ph) + "||" + rest; }
    return o;
}

}  // namespace

std::string number(long v, const std::function<std::string(const std::string&)>& ph) {
    if (v == 0) return ph("_0");
    if (v >= 1900 && v <= 1999) {   // a year: nitten hundrede (og) ...
        std::string o = ph("_19") + kJoin + ph("_0C");
        if (v % 100) { std::string rest = below100((int)(v % 100), ph); o += kJoin + og(o, rest, ph) + "||" + rest; }
        return o;
    }
    std::string o;
    long mil = v / 1000000, th = (v / 1000) % 1000, rest = v % 1000;
    auto add = [&](const std::string& s) { if (!o.empty()) o += "||"; o += s; };
    if (mil) add(mil == 1 ? ph("_1M2") : below1000((int)mil, ph) + "||" + ph("_0M2"));
    if (th) add(th == 1 ? ph("_1M1") : below1000((int)th, ph) + "||" + ph("_0M1"));
    if (rest) {
        if (!o.empty() && rest < 100) add(ph("_0and"));
        add(below1000((int)rest, ph));
    }
    return o;
}

}  // namespace da
}  // namespace kt_fe
