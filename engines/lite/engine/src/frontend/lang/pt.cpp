// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/pt.hpp"

namespace kt_fe {
namespace lang_pt {
namespace {

std::string key(int n, const char* suffix = "") { return "_" + std::to_string(n) + suffix; }

// "mil" is listed with a trailing pause ("m'iU_"); the words of a number are separated here anyway.
std::string word(const NumberWords& w, const std::string& k) {
    std::string p = w.ph(k);
    while (!p.empty() && p.back() == '_') p.pop_back();
    return p;
}

// 0..99: a listed number, or tens + "e" + units as one word ("vinte e um"), the units with their own stress.
std::string below_100(int n, const NumberWords& w) {
    if (n < 20 || w.has(key(n))) return w.ph(key(n));
    std::string o = w.ph(key(n / 10, "X"));
    if (n % 10) o += "\x02" + w.ph("_0and") + w.ph(key(n % 10));
    return o;
}

// 1..999: "cem"; "cento e" / "duzentos e" + the rest as a separate word. The "e" carries a secondary stress when
// the rest starts without one ("cento e cinquenta"), and none before a stressed start ("cento e vinte").
std::string below_1000(int n, const NumberWords& w) {
    int h = n / 100, r = n % 100;
    if (!h) return below_100(r, w);
    if (h == 1 && !r && w.has("_1C0")) return w.ph("_1C0");
    std::string o = w.has(key(h, "C")) ? w.ph(key(h, "C")) : w.ph(key(h)) + "\x02" + w.ph("_0C");
    if (!r) return o;
    std::string rest = below_100(r, w);
    return o + (w.stressed_start(rest) ? "" : ",") + w.ph("_0and") + "||" + rest;
}

std::string digit_by_digit(const std::string& d, const NumberWords& w, const char* sep = "||") {
    std::string o;
    for (char c : d) {
        if (!o.empty()) o += sep;
        o += w.ph(std::string("_") + c);
    }
    return o;
}

// A number written with a leading zero ("007", "0800", "012345").
// Up to three digits: the zeros as one word, then the rest as a number. Four digits: the digits as one word.
// Five and more: groups of three from the right, each group one word, "ponto" closing every group but the last.
std::string leading_zero(const std::string& d, const NumberWords& w) {
    const char* join = "\x02";
    if (d.size() <= 3) {
        size_t z = 0;
        while (z + 1 < d.size() && d[z] == '0') z++;
        return digit_by_digit(d.substr(0, z), w, join) + "||" + number(d.substr(z), w);
    }
    if (d.size() == 4) return digit_by_digit(d, w, join);
    std::string o;
    size_t first = d.size() % 3 ? d.size() % 3 : 3;
    for (size_t p = 0; p < d.size(); p += p ? 3 : first) {
        size_t len = p ? 3 : first;
        if (p) o += "||";
        o += digit_by_digit(d.substr(p, len), w, join);
        if (p + len < d.size()) o += join + w.ph(".");
    }
    return o;
}

}  // namespace

std::string digits_spaced(const std::string& digits, const NumberWords& w) { return digit_by_digit(digits, w); }

std::string number(const std::string& digits, const NumberWords& w, bool grouped) {
    if (digits.empty()) return "";
    if (digits.size() > 1 && digits[0] == '0') return leading_zero(digits, w);
    if (digits.size() > 9) return digit_by_digit(digits, w);
    long v = std::stol(digits);
    if (v == 0) return w.ph("_0");
    long mil = v / 1000000, th = (v / 1000) % 1000, rest = v % 1000;
    std::string o;
    auto add = [&](const std::string& s) { if (!o.empty()) o += "||"; o += s; };
    if (mil) add(mil == 1 && w.has("_1M2") ? word(w, "_1M2") : below_1000((int)mil, w) + "||" + word(w, "_0M2"));
    if (th) add(th == 1 ? word(w, "_1M1") : below_1000((int)th, w) + "||" + word(w, "_0M1"));
    if (rest) {
        if ((mil || th) && rest < 100) add("%" + w.ph("_0and"));   // unstressed   // "dois mil e cinco"
        add((grouped && (mil || th) ? "\x06" : "") + below_1000((int)rest, w));
    }
    return o;
}

std::string ordinal(const std::string& digits, bool feminine, const NumberWords& w) {
    if (digits.empty() || digits.size() > 2 || digits[0] == '0') return number(digits, w);
    int v = std::stoi(digits), t = v / 10, u = v % 10;
    std::string end = w.ph(feminine ? "_#\xc2\xaa" : "_#\xc2\xba");
    std::string tw = t ? w.ph(key(t, "Xo")) : "", uw = u ? w.ph(key(u, "o")) : "";
    if (end.empty() || (t && tw.empty()) || (u && uw.empty())) return number(digits, w);
    if (!t) return uw + end;
    if (!u) return tw + end;
    // "décimo primeiro": one word, each part with its own stress; the first ending is stressed lightly before
    // an unstressed start
    return tw + (w.stressed_start(uw) ? "" : ",") + end + "\x02" + uw + end;
}

}  // namespace lang_pt
}  // namespace kt_fe
