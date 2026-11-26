// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Dutch number words for the espeak-data text front end. Original code; the words themselves come from the
// downloaded nl_list (espeak-ng data, GPL-3.0, not bundled).
#include "frontend/lang/nl.h"

namespace kt_fe {
namespace nl {
namespace {

using Ph = std::function<std::string(const std::string&)>;

// 1..99: a listed form ("_11", "_7"), else unit + "en" + tens ("negenentwintig")
std::string below_100(int n, const Ph& ph) {
    std::string k = ph("_" + std::to_string(n));
    if (!k.empty()) return k;
    int t = n / 10, u = n % 10;
    std::string o;
    if (u) o += ph("_" + std::to_string(u)) + ph("_0and");
    return o + ph("_" + std::to_string(t) + "X");
}
// 1..999: "honderd" / "<n>honderd" written as one word, then the rest as its own word
std::string below_1000(int n, const Ph& ph) {
    int h = n / 100, r = n % 100;
    std::string o;
    if (h) o += (h == 1 ? std::string() : ph("_" + std::to_string(h))) + ph("_0C");
    if (r) { if (h) o += "||"; o += below_100(r, ph); }
    return o;
}
std::string group(long n, const char* key, const Ph& ph) {   // "<n> duizend" / "<n> miljoen"; 1 -> the noun alone
    return (n == 1 ? std::string() : below_1000((int)n, ph) + "||") + ph(key) + "||";
}

}  // namespace

std::string number_words(const std::string& digits, const Ph& ph, int flags) {
    // one primary stress per stretch: the first; the others become secondary
    auto one_primary = [](std::string o, bool none) {
        bool seen = none;
        for (char& c : o) {
            if (c != '\'') continue;
            if (seen) c = ','; else seen = true;
        }
        return o;
    };
    std::string o;
    if (digits.size() > 12 || (digits.size() > 1 && digits[0] == '0')) {   // read digit by digit
        for (char d : digits) { if (!o.empty()) o += "||"; o += ph(std::string("_") + d); }
        return one_primary(o, flags & kDecimal);
    }
    long long v = 0;
    for (char d : digits) v = v * 10 + (d - '0');
    if (v == 0) return one_primary(ph("_0"), flags & kDecimal);
    std::string rest;   // the part below a thousand, when it keeps its own stress
    if (v >= 1900 && v <= 1999 && !(flags & kSep)) {   // years of the 1900s: "negentienhonderd achtentachtig"
        o = below_100((int)(v / 100), ph) + ph("_0C");
        if (v % 100) o += "||" + below_100((int)(v % 100), ph);
    } else {
        long long b = v / 1000000000, m = (v / 1000000) % 1000, t = (v / 1000) % 1000, r = v % 1000;
        if (b) o += group((long)b, "_0M3", ph);
        if (m) o += group((long)m, "_0M2", ph);
        if (t) o += group((long)t, "_0M1", ph);
        if (r && (flags & kSep) && !o.empty()) rest = below_1000((int)r, ph);
        else if (r) o += below_1000((int)r, ph);
        while (o.size() >= 2 && o.compare(o.size() - 2, 2, "||") == 0) o.resize(o.size() - 2);
    }
    if (flags & kOrdinal) {
        std::string& last = rest.empty() ? o : rest;
        std::string listed = v < 100 ? ph("_" + std::to_string(v) + "o") : std::string();
        if (!listed.empty() && rest.empty()) o = listed;
        else last += (v % 100 != 0 && v % 100 < 20) ? ph("_ord") : ph("_ord20");
    }
    o = one_primary(o, flags & kDecimal);
    if (!rest.empty()) o += "||" + one_primary(rest, flags & kDecimal);
    return o;
}

}  // namespace nl
}  // namespace kt_fe
