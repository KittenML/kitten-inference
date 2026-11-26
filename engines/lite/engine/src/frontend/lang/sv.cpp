// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/sv.h"

#include <algorithm>

namespace kt_fe {
namespace sv {
namespace {

using Ph = std::function<std::string(const std::string&)>;

std::string key(int n) { return "_" + std::to_string(n); }

std::string below100(int n, const Ph& ph) {
    std::string own = ph(key(n));
    if (!own.empty() || n < 10) return own;
    std::string tens = ph(key(n / 10) + "X");
    if (n % 10 == 0) return tens;
    std::replace(tens.begin(), tens.end(), '\'', ',');   // one word, stress on the units
    return tens + ph(key(n % 10));
}

std::string below1000(int n, const Ph& ph) {
    int h = n / 100, r = n % 100;
    std::string o;
    if (h) o = ph(key(h)) + ph("_0C");                    // "etthundra", "tvåhundra": one word
    if (r) o += (o.empty() ? "" : "||") + below100(r, ph);
    return o;
}

std::string join(const std::string& a, const std::string& b) { return a.empty() ? b : b.empty() ? a : a + "||" + b; }

// the word for 1000^k: tusen, miljoner, biljoner
std::string scale(int k, const Ph& ph) { return ph(k == 1 ? "_0M1" : k == 2 ? "_0M2" : "_0M3"); }

}  // namespace

std::string number(const std::string& digits, const Ph& ph, bool year_group, bool grouped) {
    if (digits.empty()) return "";
    if (year_group && digits.size() >= 7 && digits.size() <= 10 && digits.size() % 3 == 1 && digits[0] != '0') {
        int g = std::stoi(digits.substr(0, 4)), k = (int)(digits.size() - 4) / 3;
        std::string rest = digits.substr(4);
        size_t nz = rest.find_first_not_of('0');
        std::string tail = nz == std::string::npos ? "" : number(rest.substr(nz), ph);
        if (g >= 1900 && g <= 1999)   // read like a year, then the group's word
            return join(join(join(ph("_19") + ph("_0C"), g % 100 ? below100(g % 100, ph) : ""), scale(k, ph)), tail);
        if (g % 1000 == 0 && k + 1 <= 3)   // "2000 300": two millions, then the bare word for the empty group
            return join(join(join(below1000(g / 1000, ph), scale(k + 1, ph)), scale(k, ph)), tail);
    }
    if (digits.size() > 12 || (digits.size() > 1 && digits[0] == '0')) {   // read digit by digit
        std::string o;
        for (char d : digits) o = join(o, ph(std::string("_") + d));
        return o;
    }
    long long v = std::stoll(digits);
    if (v == 0) return ph("_0");
    if (digits.size() == 4 && v >= 1900 && v <= 1999 && !grouped)          // a year: "nittonhundra ..."
        return join(ph("_19") + ph("_0C"), v % 100 ? below100((int)(v % 100), ph) : "");
    std::string o;
    long long bil = v / 1000000000LL, mil = (v / 1000000) % 1000, th = (v / 1000) % 1000, rest = v % 1000;
    if (bil) o = join(o, bil == 1 ? ph("_1M3") : join(below1000((int)bil, ph), ph("_0M3")));
    if (mil) o = join(o, mil == 1 ? ph("_1M2") : join(below1000((int)mil, ph), ph("_0M2")));
    if (th) o = join(o, join(below1000((int)th, ph), ph("_0M1")));      // "ett tusen", "två tusen"
    if (rest) o = join(o, below1000((int)rest, ph));
    return o;
}

}  // namespace sv
}  // namespace kt_fe
