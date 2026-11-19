// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/sw.h"

namespace kt_fe {

namespace {

struct SwNum {
    const std::function<std::string(const std::string&)>& e;
    const std::function<bool(const std::string&)>& has;

    std::string k(long n, const char* suffix) const { return e("_" + std::to_string(n) + suffix); }
    // a value that is one word: 1..10 or a whole ten
    static bool one_word(long r) { return r <= 10 || r % 10 == 0; }
    // 1..99: "kumi", "ishirini", "ishirini" + (new stress domain) "na" + "moja"
    std::string lt100(long r) const {
        if (r < 10) return k(r, "");
        long t = r / 10, u = r % 10;
        std::string o = k(t, "X");
        if (u) o += "\x02" + e("_0and") + k(u, "");
        return o;
    }
    // "na" + a one-word value: stressed together, written as two words
    std::string na(long r) const { return e("_0and") + "\x0b" + lt100(r); }
    // 1..999
    std::string lt1000(long n) const {
        long h = n / 100, r = n % 100;
        if (!h) return lt100(r);
        std::string o = k(h, "C");
        if (r) o += one_word(r) ? "\x02" + na(r) : "||" + lt100(r);   // "mia moja" + "na" glued, or a new word
        return o;
    }
    // what follows a thousands word: 1..999
    std::string after_thousands(long r) const {
        if (r >= 100) return "||" + lt1000(r);
        return "||" + (one_word(r) ? na(r) : lt100(r));
    }
    // 1..999999
    std::string lt1e6(long n) const {
        long th = n / 1000, r = n % 1000;
        if (!th) return lt1000(r);
        std::string o = th < 10 && has("_" + std::to_string(th) + "M1") ? k(th, "M1") : lt1000(th) + "||" + e("_0M1");
        if (r) o += after_thousands(r);
        return o;
    }
};

}  // namespace

std::string sw_number(const std::string& digits, const std::function<std::string(const std::string&)>& entry,
                      const std::function<bool(const std::string&)>& has) {
    if (digits.empty() || digits.size() > 12 || (digits.size() > 1 && digits[0] == '0')) return "";
    long long v = 0;
    for (char c : digits) { if (c < '0' || c > '9') return ""; v = v * 10 + (c - '0'); }
    SwNum s{entry, has};
    if (v == 0) return entry("_0");
    std::string o;
    long long bil = v / 1000000000LL, below = v % 1000000000LL;
    if (bil) {
        if (bil >= 1000 || !has("_0M3")) return "";
        o = s.lt1000((long)bil) + "||" + entry("_0M3");
        if (!below) return o;
        o += "||";
    }
    long mil = (long)(below / 1000000), rest = (long)(below % 1000000);
    if (mil && mil < 10 && has("_" + std::to_string(mil) + "M2")) {
        o += s.k(mil, "M2");
        if (rest) o += "||" + s.lt1e6(rest);
    } else if (mil) {
        // no million word for this count: the thousands are counted in thousands ("123 elfu 456 elfu 789")
        long th = mil * 1000 + rest / 1000, r = rest % 1000;
        o += s.lt1e6(th) + "||" + entry("_0M1");
        if (r) o += s.after_thousands(r);
    } else {
        o += s.lt1e6(rest);
    }
    return o;
}

}  // namespace kt_fe
