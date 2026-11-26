// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// tts08_text.cpp - see tts08_text.hpp. Each step below names the Python function it ports
// (src/kitten_inference/tts08/preprocess.py); the patterns are the Python ones, matched with Python re semantics.
//
// Known differences from the Python package (documented in engines/lite/mobile/README.md):
//  * no Unicode NFC step (apps pass NFC text; iOS and Android string APIs normally do);
//  * str.lower() and \w use built-in tables for the common scripts, not the full Unicode database;
//  * where the Python code raises (for example "5,%" or a percentage whose float repr is in exponent form), the
//    C++ code leaves that piece of text unchanged instead of failing the whole call.
#include "tts08_text.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <stdexcept>

#include "cephonemizer/phonemizer.h"

namespace kt08fe {

namespace {

using U = std::u32string;

struct PyError {};  // a step where Python would raise: the match is left unchanged

U u(const char* s) { return utf8_to_u32(s); }

const char* const kOnes[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                             "eighteen", "nineteen"};
const char* const kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* const kScale[] = {"", "thousand", "million", "billion", "trillion"};
const char* const kDigitWords[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};

// An integer as Python int(): sign + decimal digits (no leading zeros; "0" for zero).
struct Int {
    bool neg = false;
    std::string d = "0";
};

// int(s) for a string of (Unicode) digits with an optional sign. Throws PyError when Python's int() would.
Int py_int(const U& s) {
    size_t i = 0;
    Int r;
    while (i < s.size() && cp_is_space(s[i])) ++i;
    size_t e = s.size();
    while (e > i && cp_is_space(s[e - 1])) --e;
    if (i < e && (s[i] == '-' || s[i] == '+')) { r.neg = s[i] == '-'; ++i; }
    if (i >= e) throw PyError{};
    std::string d;
    for (size_t k = i; k < e; ++k) {
        int v = cp_digit_value(s[k]);
        if (v < 0) throw PyError{};
        if (d.empty() && v == 0) continue;
        d.push_back(char('0' + v));
    }
    r.d = d.empty() ? "0" : d;
    if (r.d == "0") r.neg = false;
    return r;
}

std::string three_digits(int n) {
    if (n == 0) return "";
    std::string out;
    int h = n / 100, rem = n % 100;
    if (h) out = std::string(kOnes[h]) + " hundred";
    if (rem < 20) {
        if (rem) { if (!out.empty()) out += ' '; out += kOnes[rem]; }
    } else {
        if (!out.empty()) out += ' ';
        out += kTens[rem / 10];
        if (rem % 10) { out += '-'; out += kOnes[rem % 10]; }
    }
    return out;
}

// number_to_words(n)
std::string number_to_words(const Int& n) {
    if (n.d == "0") return "zero";
    if (n.neg) { Int p = n; p.neg = false; return "negative " + number_to_words(p); }
    const std::string& d = n.d;
    if (d.size() <= 4) {
        int v = std::atoi(d.c_str());
        if (v >= 100 && v % 100 == 0 && v % 1000 != 0 && v / 100 < 20) return std::string(kOnes[v / 100]) + " hundred";
    }
    std::vector<std::string> parts;
    size_t end = d.size();
    for (int i = 0; i < 5; ++i) {
        size_t start = end >= 3 ? end - 3 : 0;
        int chunk = std::atoi(d.substr(start, end - start).c_str());
        if (chunk) {
            std::string w = three_digits(chunk);
            parts.push_back(i ? w + " " + kScale[i] : w);
        }
        end = start;
        if (end == 0) break;
        bool rest_zero = true;
        for (size_t k = 0; k < end; ++k) if (d[k] != '0') { rest_zero = false; break; }
        if (rest_zero) break;
    }
    std::string out;
    for (size_t k = parts.size(); k-- > 0;) { if (!out.empty()) out += ' '; out += parts[k]; }
    return out;
}
std::string number_to_words(long long v) {
    Int n;
    n.neg = v < 0;
    n.d = std::to_string(v < 0 ? -(unsigned long long)v : (unsigned long long)v);
    if (n.d == "0") n.neg = false;
    return number_to_words(n);
}

// Python repr(float).
std::string py_float_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
    char buf[64];
    int prec = 1;
    for (; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    std::string s(buf);
    bool neg = s[0] == '-';
    if (neg) s.erase(0, 1);
    size_t epos = s.find('e');
    int exp10 = std::atoi(s.c_str() + epos + 1);
    std::string digits;
    for (size_t k = 0; k < epos; ++k) if (s[k] != '.') digits.push_back(s[k]);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (x == 0.0) out = "0.0";
    else if (exp10 >= -4 && exp10 < 16) {
        if (exp10 >= 0) {
            std::string ip = digits.substr(0, std::min(digits.size(), (size_t)exp10 + 1));
            while (ip.size() < (size_t)exp10 + 1) ip.push_back('0');
            std::string fp = digits.size() > (size_t)exp10 + 1 ? digits.substr(exp10 + 1) : "0";
            out = ip + "." + fp;
        } else {
            out = "0." + std::string((size_t)(-exp10 - 1), '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return neg ? "-" + out : out;
}

std::string to_ascii(const U& s) {  // digits mapped to ASCII; other code points as UTF-8
    std::string o;
    for (char32_t c : s) {
        int v = cp_digit_value(c);
        if (v >= 0) o.push_back(char('0' + v));
        else append_utf8(o, c);
    }
    return o;
}

// float(s) as Python (correctly rounded). Throws PyError when Python's float() would.
double py_float(const U& s) {
    std::string a = to_ascii(s);
    size_t b = 0, e = a.size();
    while (b < e && (a[b] == ' ')) ++b;
    while (e > b && a[e - 1] == ' ') --e;
    a = a.substr(b, e - b);
    if (a.empty()) throw PyError{};
    for (char c : a)
        if (!((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')) throw PyError{};
    char* end = nullptr;
    double v = std::strtod(a.c_str(), &end);
    if (end != a.c_str() + a.size()) throw PyError{};
    return v;
}

// Exact integer value of a finite double (int(float)), as an Int.
Int int_of_double(double v) {
    Int r;
    r.neg = v < 0;
    v = std::fabs(std::trunc(v));
    if (v < 9007199254740992.0) {
        r.d = std::to_string((unsigned long long)v);
    } else {
        int ex;
        double m = std::frexp(v, &ex);  // v = m * 2^ex, 0.5 <= m < 1
        unsigned long long mant = (unsigned long long)std::ldexp(m, 53);
        ex -= 53;
        std::string d = std::to_string(mant);  // decimal string, multiply by 2^ex
        for (int k = 0; k < ex; ++k) {
            int carry = 0;
            for (size_t i = d.size(); i-- > 0;) {
                int x = (d[i] - '0') * 2 + carry;
                d[i] = char('0' + x % 10);
                carry = x / 10;
            }
            if (carry) d.insert(d.begin(), char('0' + carry));
        }
        r.d = d;
    }
    if (r.d == "0") r.neg = false;
    return r;
}

std::string digits_spelled(const U& s) {  // " ".join(_d[c] for c in s)
    std::string o;
    for (char32_t c : s) {
        if (c < '0' || c > '9') throw PyError{};  // the Python dict only has ASCII digits
        if (!o.empty()) o += ' ';
        o += kDigitWords[c - '0'];
    }
    return o;
}

// float_to_words(text) for a string argument.
std::string float_to_words(const U& value) {
    U text = value;
    bool negative = !text.empty() && text[0] == '-';
    if (negative) text.erase(0, 1);
    std::string result;
    size_t dot = text.find(U'.');
    if (dot != U::npos) {
        U ip = text.substr(0, dot), dp = text.substr(dot + 1);
        std::string iw = ip.empty() ? "zero" : number_to_words(py_int(ip));
        std::string dw;
        for (char32_t c : dp) {
            int v = cp_digit_value(c);
            if (v < 0) throw PyError{};
            if (!dw.empty()) dw += ' ';
            dw += kDigitWords[v];
        }
        result = iw + " point " + dw;
    } else {
        result = number_to_words(py_int(text));
    }
    return negative ? "negative " + result : result;
}
std::string float_to_words(double v) { return float_to_words(utf8_to_u32(py_float_repr(v))); }

// _ordinal_suffix(n)
std::string ordinal_words(const Int& n) {
    std::string word = number_to_words(n);
    std::string prefix, last, joiner;
    size_t h = word.rfind('-');
    if (h != std::string::npos) { prefix = word.substr(0, h); last = word.substr(h + 1); joiner = "-"; }
    else {
        size_t sp = word.rfind(' ');
        if (sp != std::string::npos) { prefix = word.substr(0, sp); last = word.substr(sp + 1); joiner = " "; }
        else last = word;
    }
    static const char* const exc[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"},
                                         {"five", "fifth"}, {"six", "sixth"}, {"seven", "seventh"},
                                         {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
    std::string lo;
    bool found = false;
    for (auto& e : exc) if (last == e[0]) { lo = e[1]; found = true; break; }
    if (!found) {
        if (!last.empty() && last.back() == 't') lo = last + "h";
        else if (!last.empty() && last.back() == 'e') lo = last.substr(0, last.size() - 1) + "th";
        else lo = last + "th";
    }
    return prefix.empty() ? lo : prefix + joiner + lo;
}

U strip(const U& s) {
    size_t b = 0, e = s.size();
    while (b < e && cp_is_space(s[b])) ++b;
    while (e > b && cp_is_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

U remove_commas(const U& s) {
    U o;
    for (char32_t c : s) if (c != ',') o.push_back(c);
    return o;
}

bool contains(const U& s, char32_t c) { return s.find(c) != U::npos; }

}  // namespace

// ---- rules ----------------------------------------------------------------------------------------------------------
using Repl = Normalizer08::Repl;

namespace {
// re.sub(pattern, fn, text) into out; false (out untouched) when nothing matched. A replacement that raises (PyError)
// keeps the matched text.
bool sub(const Regex& re, const Repl& fn, const U& s, U& out) {
    Regex::Match m;
    size_t pos = 0;
    bool any = false;
    while (pos <= s.size() && re.search(s, pos, m)) {
        if (!any) { out.clear(); out.reserve(s.size() + 32); any = true; }
        out.append(s, pos, m.start - pos);
        try {
            out += utf8_to_u32(fn(m, s));
        } catch (const PyError&) {
            out.append(s, m.start, m.end - m.start);
        }
        if (m.end == m.start) {
            if (m.end < s.size()) out.push_back(s[m.end]);
            pos = m.end + 1;
        } else pos = m.end;
    }
    if (!any) return false;
    if (pos < s.size()) out.append(s, pos, U::npos);
    return true;
}

void strip_in_place(U& s) {
    size_t e = s.size();
    while (e > 0 && cp_is_space(s[e - 1])) --e;
    s.resize(e);
    size_t b = 0;
    while (b < s.size() && cp_is_space(s[b])) ++b;
    if (b) s.erase(0, b);
}

std::string g(const Regex::Match& m, const U& s, int i) { return u32_to_utf8(m.group(s, i)); }
}  // namespace

Normalizer08::Normalizer08() {
    auto add = [this](const char* pat, bool icase, Repl fn, bool strip_after = false) {
        rules_.push_back(Rule{Regex(pat, icase), std::move(fn), strip_after});
    };
    auto lit = [](const char* text) { return [text](const Regex::Match&, const U&) { return std::string(text); }; };
    auto g1 = [](const char* suffix) {
        return [suffix](const Regex::Match& m, const U& s) { return g(m, s, 1) + suffix; };
    };

    // remove_html_tags, remove_urls, remove_emails
    add(R"re(<[^>]+>)re", false, lit(" "));
    add(R"re(https?://\S+|www\.\S+)re", false, lit(""), true);
    add(R"re(\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b)re", true, lit(""), true);
    // expand_contractions (dict order)
    add(R"re(\bcan't\b)re", true, lit("cannot"));
    add(R"re(\bwon't\b)re", true, lit("will not"));
    add(R"re(\bshan't\b)re", true, lit("shall not"));
    add(R"re(\bain't\b)re", true, lit("is not"));
    add(R"re(\blet's\b)re", true, lit("let us"));
    add(R"re(\b(\w+)n't\b)re", true, g1(" not"));
    add(R"re(\b(\w+)'re\b)re", true, g1(" are"));
    add(R"re(\b(\w+)'ve\b)re", true, g1(" have"));
    add(R"re(\b(\w+)'ll\b)re", true, g1(" will"));
    add(R"re(\b(\w+)'d\b)re", true, g1(" would"));
    add(R"re(\b(\w+)'m\b)re", true, g1(" am"));
    add(R"re(\bit's\b)re", true, lit("it is"));
    // expand_ip_addresses
    add(R"re(\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b)re", false, [](const Regex::Match& m, const U& s) {
        std::string o;
        for (int i = 1; i <= 4; ++i) { if (i > 1) o += " dot "; o += digits_spelled(m.group(s, i)); }
        return o;
    });
    // normalize_leading_decimals
    add(R"re((?<!\d)(-)\.([\d]))re", false,
        [](const Regex::Match& m, const U& s) { return g(m, s, 1) + "0." + g(m, s, 2); });
    add(R"re((?<!\d)\.([\d]))re", false, [](const Regex::Match& m, const U& s) { return "0." + g(m, s, 1); });
    // expand_currency
    add(R"re(([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?(?![a-zA-Z\d]))re", false,
        [](const Regex::Match& m, const U& s) {
            const char32_t sym = s[m.start];
            const char* unit = sym == U'$' ? "dollar" : sym == U'€' ? "euro" : sym == U'£' ? "pound"
                             : sym == U'¥' ? "yen" : sym == U'₹' ? "rupee" : sym == U'₩' ? "won" : "bitcoin";
            U raw = remove_commas(m.group(s, 2));
            if (m.has(3)) {
                char32_t sc = m.group(s, 3)[0];
                const char* scale = sc == 'K' ? "thousand" : sc == 'M' ? "million" : sc == 'B' ? "billion" : "trillion";
                std::string num = contains(raw, U'.') ? float_to_words(raw) : number_to_words(py_int(raw));
                return num + " " + scale + " " + unit + "s";
            }
            std::string result;
            size_t dot = raw.find(U'.');
            if (dot != U::npos) {
                U ip = raw.substr(0, dot), dp = raw.substr(dot + 1);
                U d2 = dp.substr(0, 2);
                while (d2.size() < 2) d2.push_back(U'0');
                Int dv = py_int(d2);
                int dec_val = std::atoi(dv.d.c_str());
                result = number_to_words(py_int(ip)) + " " + unit + "s";
                if (dec_val) result += " and " + number_to_words(dec_val) + " cent" + (dec_val != 1 ? "s" : "");
            } else {
                Int v = py_int(raw);
                result = number_to_words(v) + " " + unit + ((v.d != "1" || v.neg) ? "s" : "");
            }
            return result;
        });
    // expand_percentages
    add(R"re((-?[\d,]+(?:\.\d+)?)\s*%)re", false, [](const Regex::Match& m, const U& s) {
        U raw = remove_commas(m.group(s, 1));
        if (contains(raw, U'.')) return float_to_words(py_float(raw)) + " percent";
        return number_to_words(py_int(raw)) + " percent";
    });
    // expand_scientific_notation
    add(R"re((?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![a-zA-Z\d]))re", false,
        [](const Regex::Match& m, const U& s) {
            U coeff = m.group(s, 1);
            Int ex = py_int(m.group(s, 2));
            std::string cw = contains(coeff, U'.') ? float_to_words(coeff) : number_to_words(py_int(coeff));
            Int a = ex;
            a.neg = false;
            return cw + " times ten to the " + (ex.neg ? "negative " : "") + number_to_words(a);
        });
    // expand_time
    add(R"re(\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b)re", true, [](const Regex::Match& m, const U& s) {
        Int h = py_int(m.group(s, 1));
        int mins = std::atoi(py_int(m.group(s, 2)).d.c_str());
        std::string suffix;
        if (m.has(4)) {
            U a = m.group(s, 4);
            for (auto& c : a) c = cp_lower(c);
            suffix = " " + u32_to_utf8(a);
        }
        std::string hw = number_to_words(h);
        if (mins == 0) return m.has(4) ? hw + suffix : hw + " hundred" + suffix;
        if (mins < 10) return hw + " oh " + number_to_words(mins) + suffix;
        return hw + " " + number_to_words(mins) + suffix;
    });
    // expand_ordinals
    add(R"re(\b(\d+)(st|nd|rd|th)\b)re", true,
        [](const Regex::Match& m, const U& s) { return ordinal_words(py_int(m.group(s, 1))); });
    // expand_units
    add(R"re((\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b)re", true,
        [](const Regex::Match& m, const U& s) {
            static const char* const map[][2] = {
                {"km", "kilometers"}, {"kg", "kilograms"}, {"mg", "milligrams"}, {"ml", "milliliters"},
                {"gb", "gigabytes"}, {"mb", "megabytes"}, {"kb", "kilobytes"}, {"tb", "terabytes"}, {"hz", "hertz"},
                {"khz", "kilohertz"}, {"mhz", "megahertz"}, {"ghz", "gigahertz"}, {"mph", "miles per hour"},
                {"kph", "kilometers per hour"}, {"ms", "milliseconds"}, {"ns", "nanoseconds"},
                {"µs", "microseconds"}, {"°c", "degrees Celsius"}, {"c°", "degrees Celsius"},
                {"°f", "degrees Fahrenheit"}, {"f°", "degrees Fahrenheit"}};
            U raw = m.group(s, 1), un = m.group(s, 2);
            for (auto& c : un) c = cp_lower(c);
            std::string key = u32_to_utf8(un), expanded = g(m, s, 2);
            for (auto& e : map) if (key == e[0]) { expanded = e[1]; break; }
            std::string num = contains(raw, U'.') ? float_to_words(py_float(raw)) : number_to_words(py_int(raw));
            return num + " " + expanded;
        });
    // expand_scale_suffixes
    add(R"re((?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d]))re", false, [](const Regex::Match& m, const U& s) {
        U raw = m.group(s, 1);
        char32_t sc = m.group(s, 2)[0];
        const char* scale = sc == 'K' ? "thousand" : sc == 'M' ? "million" : sc == 'B' ? "billion" : "trillion";
        std::string num = contains(raw, U'.') ? float_to_words(raw) : number_to_words(py_int(raw));
        return num + " " + scale;
    });
    // expand_fractions
    add(R"re(\b(\d+)\s*/\s*(\d+)\b)re", false, [](const Regex::Match& m, const U& s) {
        Int num = py_int(m.group(s, 1)), den = py_int(m.group(s, 2));
        if (den.d == "0") return g(m, s, 0);
        std::string nw = number_to_words(num), dw;
        if (den.d == "2") dw = num.d == "1" ? "half" : "halves";
        else if (den.d == "4") dw = num.d == "1" ? "quarter" : "quarters";
        else { dw = ordinal_words(den); if (num.d != "1") dw += "s"; }
        return nw + " " + dw;
    });
    // expand_decades
    add(R"re(\b(\d{1,3})0s\b)re", false, [](const Regex::Match& m, const U& s) {
        static const char* const dec[] = {"hundreds", "tens", "twenties", "thirties", "forties",
                                          "fifties", "sixties", "seventies", "eighties", "nineties"};
        int base = std::atoi(py_int(m.group(s, 1)).d.c_str());
        std::string w = dec[base % 10];
        if (base < 10) return w;
        return number_to_words(base / 10) + " " + w;
    });
    // expand_phone_numbers (longest first)
    auto join_groups = [](int n) {
        return [n](const Regex::Match& m, const U& s) {
            std::string o;
            for (int i = 1; i <= n; ++i) { if (i > 1) o += ' '; o += digits_spelled(m.group(s, i)); }
            return o;
        };
    };
    add(R"re((?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d))re", false, join_groups(4));
    add(R"re((?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!-\d))re", false, join_groups(3));
    add(R"re((?<!\d-)\b(\d{3})-(\d{4})\b(?!-\d))re", false, join_groups(2));
    // expand_ranges
    add(R"re((?<!\w)(\d+)-(\d+)(?!\w))re", false, [](const Regex::Match& m, const U& s) {
        return number_to_words(py_int(m.group(s, 1))) + " to " + number_to_words(py_int(m.group(s, 2)));
    });
    // expand_model_names
    add(R"re(\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$))re", false,
        [](const Regex::Match& m, const U& s) { return g(m, s, 1) + " " + g(m, s, 2); });
    // replace_numbers(replace_floats=True)
    add(R"re((?<![a-zA-Z])-?[\d,]+(?:\.\d+)?)re", false, [](const Regex::Match& m, const U& s) {
        U raw = remove_commas(m.group(s, 0));
        if (contains(raw, U'.')) return float_to_words(raw);
        double v = py_float(raw);  // int(float(raw)): ValueError/OverflowError keep the text (as in Python)
        if (std::isinf(v) || std::isnan(v)) throw PyError{};
        return number_to_words(int_of_double(v));
    });
}

U Normalizer08::normalize(const U& in) const {
    U text = in;  // normalize_unicode (NFC): not applied, see the file comment
    U tmp;
    for (const Rule& r : rules_) {
        if (sub(r.re, r.fn, text, tmp)) text.swap(tmp);
        if (r.strip_after) strip_in_place(text);
    }
    for (auto& c : text) c = cp_lower(c);  // to_lowercase
    // remove_extra_whitespace: \s+ -> " ", strip
    U out;
    out.reserve(text.size());
    bool in_space = false;
    for (char32_t c : text) {
        if (cp_is_space(c)) { in_space = true; continue; }
        if (in_space && !out.empty()) out.push_back(U' ');
        in_space = false;
        out.push_back(c);
    }
    return out;
}

// ---- chunk_text ----------------------------------------------------------------------------------------------------
namespace {
bool ascii_alpha(char32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

bool is_sentence_boundary(const U& t, size_t i) {
    char32_t ch = t[i];
    if (ch != '.' && ch != '!' && ch != '?') return false;
    if (ch == '.') {
        if (i > 0 && i + 1 < t.size() && cp_is_digit(t[i - 1]) && cp_is_digit(t[i + 1])) return false;
        size_t b = i;
        while (b > 0 && ascii_alpha(t[b - 1])) --b;
        std::string token;
        for (size_t k = b; k < i; ++k) token.push_back((char)cp_lower(t[k]));
        static const char* const abbr[] = {"dr", "prof", "mr", "mrs", "ms", "fig", "figs", "pp", "p", "ch", "sec",
                                           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
                                           "nov", "dec", "al"};
        for (auto a : abbr) if (token == a) return false;
        if ((token == "a" || token == "p") && i + 1 < t.size() && cp_lower(t[i + 1]) == 'm') return false;
        // token == "m" and re.search(r"\b[ap]\.m$", before, re.I)
        if (token == "m" && i >= 3 && t[i - 2] == '.' && (cp_lower(t[i - 3]) == 'a' || cp_lower(t[i - 3]) == 'p') &&
            (i == 3 || !cp_is_word(t[i - 4]))) {
            size_t k = i + 1;
            while (k < t.size() && cp_is_space(t[k])) ++k;
            return k >= t.size() || cp_is_upper(t[k]);
        }
    }
    return i + 1 >= t.size() || cp_is_space(t[i + 1]);
}

U ensure_punctuation(const U& s) {
    U t = strip(s);
    if (t.empty()) return t;
    char32_t l = t.back();
    if (l != '.' && l != '!' && l != '?' && l != ',' && l != ';' && l != ':') t.push_back(U',');
    return t;
}
}  // namespace

std::vector<U> Normalizer08::chunks(const U& text, size_t max_len) {
    std::vector<U> sentences, out;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        if (is_sentence_boundary(text, i)) {
            sentences.push_back(text.substr(start, i + 1 - start));
            start = i + 1;
        }
    }
    if (start < text.size()) sentences.push_back(text.substr(start));
    for (const U& s0 : sentences) {
        U s = strip(s0);
        if (s.empty()) continue;
        if (s.size() <= max_len) { out.push_back(ensure_punctuation(s)); continue; }
        // sentence.split(): runs of whitespace
        U temp;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && cp_is_space(s[i])) ++i;
            if (i >= s.size()) break;
            size_t j = i;
            while (j < s.size() && !cp_is_space(s[j])) ++j;
            U word = s.substr(i, j - i);
            i = j;
            if (temp.size() + word.size() + 1 <= max_len) {
                if (!temp.empty()) temp.push_back(U' ');
                temp += word;
            } else {
                if (!temp.empty()) out.push_back(ensure_punctuation(strip(temp)));
                temp = word;
            }
        }
        if (!temp.empty()) out.push_back(ensure_punctuation(strip(temp)));
    }
    return out;
}

// ---- Frontend08 ------------------------------------------------------------------------------------------------------
Frontend08::Frontend08(const std::string& rules_path, const std::string& list_path) {
    ph_.reset(new IPAPhonemizer(rules_path, list_path, "en-us"));
    if (!ph_->isLoaded()) throw std::runtime_error(ph_->getError().empty() ? "cannot load phonemizer data"
                                                                           : ph_->getError());
    // A wrong, empty or damaged file still parses; check that the pair reads a known sentence as espeak-ng does.
    if (ph_->phonemizeText("hello world, this is a test") != "h\xc9\x99l\xcb\x88o\xca\x8a w\xcb\x88\xc9\x9c\xcb\x90ld \xc3\xb0\xc9\xaas \xc9\xaaz \xc9\x90 t\xcb\x88\xc9\x9bst")
        throw std::runtime_error("en_rules / en_list are not espeak-ng's English data files (see fetch_phonemizer_data.sh): " +
                                 rules_path + ", " + list_path);
}

Frontend08::~Frontend08() = default;

// The Python package phonemises with phonemizer's EspeakBackend(preserve_punctuation=True): the chunk is cut at the
// punctuation marks, each piece is phonemised on its own (one espeak clause), the pieces between quotes or brackets
// are also phonemised together (for context), and the marks are put back (phonemizer/punctuation.py, 3.6.0:
// Punctuation.preserve, inline_groups, restore; backend/espeak/espeak.py: _postprocess_line). Same steps here, with
// CEPhonemizer in place of espeak-ng.
namespace {
const char32_t* const kInlineMarks = U"\"«»“”‘’„‹›()[]{}";

bool is_inline_mark(const std::u32string& m) {
    bool any = false;
    for (char32_t c : m) {
        if (cp_is_space(c)) continue;
        any = true;
        if (!std::char_traits<char32_t>::find(kInlineMarks, std::char_traits<char32_t>::length(kInlineMarks), c))
            return false;
    }
    return any;
}

std::vector<std::string> split_words(const std::string& s) {  // [w for w in s.split(' ') if w]
    std::vector<std::string> w;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(' ', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) w.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return w;
}

std::vector<std::string> split_space(const std::string& s) {  // s.split(' ')
    std::vector<std::string> w;
    size_t i = 0;
    for (;;) {
        size_t j = s.find(' ', i);
        if (j == std::string::npos) { w.push_back(s.substr(i)); break; }
        w.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return w;
}

bool ends_with(const std::string& s, char c) { return !s.empty() && s.back() == c; }
}  // namespace

const Regex& Frontend08::marks_re() {
    static const Regex re(R"re((\s*(?:[;:!?¡¿—…"«»“”(){}\[\]]|(?<![0-9])[,.]|[,.](?![0-9]))+\s*)+)re");
    return re;
}

namespace {
const char* const kPrimary = "\xcb\x88";    // ˈ
const char* const kSecondary = "\xcb\x8c";  // ˌ
bool ipa_vowel_at(const std::string& s, size_t i, size_t* len) {
    static const char* const V[] = {"a", "e", "i", "o", "u", "\xc3\xa6", "\xc9\x91", "\xc9\x90", "\xc9\x92", "\xc9\x94",
                                    "\xc9\x99", "\xc9\x9a", "\xc9\x9b", "\xc9\x9c", "\xc9\x9d", "\xc9\xaa", "\xc9\xa8",
                                    "\xca\x8a", "\xca\x8c", "\xe1\xb5\xbb", "\xc9\xb5", "\xc3\xb8", "\xc5\x93"};
    for (const char* v : V) {
        size_t n = std::strlen(v);
        if (s.compare(i, n, v) == 0) { *len = n; return true; }
    }
    return false;
}
void replace_first(std::string& s, const char* from, const char* to) {
    size_t p = s.find(from);
    if (p != std::string::npos) s.replace(p, std::strlen(from), to);
}
void remove_all(std::string& s, const char* m) {
    size_t p;
    while ((p = s.find(m)) != std::string::npos) s.erase(p, std::strlen(m));
}
}  // namespace

namespace {
bool ends_with_str(const std::string& s, const char* e) {
    size_t n = std::strlen(e);
    return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
}
}  // namespace

// espeak-ng picks the weak forms of "the" (ðə / ðɪ) and "to" (tə / tʊ) from the first *phoneme* of the next word
// ("the honor" ðɪ, "the one" ðə); CEPhonemizer looks at its first letter. Also the en_list phrase "(that it)
// ,Da#tIt" inside a clause (ðˌɐɾɪt), which CEPhonemizer writes ðˌɐtˈɪt ("the tit" to a listener).
void Frontend08::vowel_context(std::vector<std::string>& ipa) {
    const char* const THE_V = "\xc3\xb0\xc9\xaa";  // ðɪ
    const char* const THE_C = "\xc3\xb0\xc9\x99";  // ðə
    const char* const TO_V = "t\xca\x8a";          // tʊ
    const char* const TO_C = "t\xc9\x99";          // tə
    for (size_t k = 0; k + 1 < ipa.size(); ++k) {
        std::string& w = ipa[k];
        if (w == "\xc3\xb0\xcb\x8c\xc9\x90t\xcb\x88\xc9\xaat") {  // ðˌɐtˈɪt
            w = "\xc3\xb0\xcb\x8c\xc9\x90\xc9\xbe\xc9\xaat";      // ðˌɐɾɪt
            continue;
        }
        std::string next = ipa[k + 1];
        while (next.compare(0, 2, kPrimary) == 0 || next.compare(0, 2, kSecondary) == 0) next.erase(0, 2);
        size_t n;
        const bool vowel = !next.empty() && ipa_vowel_at(next, 0, &n);
        if (ends_with_str(w, THE_V) || ends_with_str(w, THE_C)) {
            std::string stem = w.substr(0, w.size() - 4);
            if (stem.find(kPrimary) != std::string::npos || stem.find(kSecondary) != std::string::npos) continue;
            w = stem + (vowel ? THE_V : THE_C);
        } else if (w == TO_V || w == TO_C) {
            std::string bare = next;
            remove_all(bare, kPrimary);
            remove_all(bare, kSecondary);
            w = (vowel || bare == "w\xc9\xaat\xca\x83") ? TO_V : TO_C;  // espeak-ng also says tʊ before "which"
        }
    }
}

// Clause-final stress as espeak-ng 1.52 gives it, which CEPhonemizer (written against a later espeak-ng) does not:
// the last word of a clause takes primary stress when en_list marks it $strend, $strend2 or $u+ ("like that",
// "for this"), and a clause of one word keeps the stress of its en_list entry otherwise ("sir", "and", "but").
void Frontend08::clause_end_stress(const std::string& text, std::vector<std::string>& ipa) const {
    std::vector<std::string> words;
    {
        std::string cur;
        bool letters = false;
        for (size_t i = 0; i <= text.size(); ++i) {
            char c = i < text.size() ? text[i] : ' ';
            if (c == ' ' || c == '\t') {
                if (!cur.empty() && letters) words.push_back(cur);
                cur.clear();
                letters = false;
            } else {
                cur.push_back(c);
                if ((unsigned char)c >= 0x80 || std::isalnum((unsigned char)c)) letters = true;
            }
        }
    }
    if (words.empty() || words.size() != ipa.size()) return;
    const std::string& last = words.back();
    for (char c : last)
        if (!(c >= 'a' && c <= 'z')) return;  // plain lower-case words only
    const int f = ph_->clauseEndFlags(last);
    if (!(f & 16)) return;
    std::string& w = ipa.back();
    if (f & 1) {
        if (w.find(kPrimary) != std::string::npos) return;
        if (w.find(kSecondary) != std::string::npos) { replace_first(w, kSecondary, kPrimary); return; }
        for (size_t i = 0; i < w.size(); ++i) {
            size_t n;
            if (ipa_vowel_at(w, i, &n)) { w.insert(i, kPrimary); return; }
        }
        return;
    }
    if (f & (2 | 8)) {
        if (w.find(kSecondary) == std::string::npos) replace_first(w, kPrimary, kSecondary);
    } else if ((f & 4) && words.size() == 1) {
        remove_all(w, kPrimary);
        remove_all(w, kSecondary);
    }
}

// espeak's output for one piece after EspeakBackend._postprocess_line (separator: phone '', word ' '; strip False).
std::string Frontend08::piece(const std::string& text_in) const {
    // espeak-ng reads a lower-case "i" (the package lower-cases the text) with the stressed letter entry of en_list,
    // not the pronoun entry ("I ... $u+ $verbf") that CEPhonemizer uses for both cases. "eye" gives the same
    // phonemes and has no context flags, as espeak's lower-case "i".
    // espeak-ng also says the names of these symbols (en_list "_&" ...); CEPhonemizer drops them.
    static const std::pair<const char*, const char*> kSymbols[] = {
        {"&", " and "}, {"/", " slash "}, {"*", " asterisk "}, {"+", " plus "}, {"=", " equals "},
        {"%", " percent "}, {"@", " at "}, {"#", " hash "}, {"~", " tilde "}, {"\\", " backslash "},
        {"\xc2\xb0", " degrees "}, {"\xc2\xa7", " section "}, {"\xc2\xa9", " copyright "},
        {"\xe2\x84\xa2", " trade mark "}};
    std::string text;
    text.reserve(text_in.size() + 8);
    for (size_t i = 0; i < text_in.size(); ++i) {
        bool sym = false;
        for (const auto& kv : kSymbols) {
            const size_t n = std::strlen(kv.first);
            if (text_in.compare(i, n, kv.first) == 0) { text += kv.second; i += n - 1; sym = true; break; }
        }
        if (sym) continue;
        auto letter = [&](size_t k) { return k < text_in.size() && ((unsigned char)text_in[k] >= 0x80 || std::isalnum((unsigned char)text_in[k])); };
        if (text_in[i] == '\'' && !(i > 0 && letter(i - 1) && letter(i + 1)))
            text.push_back(' ');  // an apostrophe that is not inside a word: espeak-ng reads the word without it
                                  // ("'tis" = "tis", "come'" = "come", "dogs'" = "dogs"); CEPhonemizer would say
                                  // "tick" or misread the word
        else
            text.push_back(text_in[i]);
    }
    {
        std::string t2;
        t2.reserve(text.size() + 8);
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == 'i' && (i == 0 || text[i - 1] == ' ') && (i + 1 == text.size() || text[i + 1] == ' '))
                t2 += "eye";
            else
                t2.push_back(text[i]);
        }
        text.swap(t2);
    }
    std::string line = ph_->phonemizeText(text);
    for (char& c : line)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    // espeak-ng 1.52 writes en-us "L" as plain l, never the velarised ɫ of CEPhonemizer's table ("isle", "denial").
    for (size_t p; (p = line.find("\xc9\xab")) != std::string::npos;) line.replace(p, 2, "l");
    std::vector<std::string> w = split_words(line);
    if (w.empty()) return "";
    vowel_context(w);
    clause_end_stress(text, w);
    std::string out;
    for (const auto& x : w) { out += x; out += ' '; }
    return out;
}

std::string Frontend08::chunk_to_phonemes(const std::string& chunk_utf8) const {
    const std::u32string line = utf8_to_u32(chunk_utf8);
    // Punctuation.preserve
    std::vector<std::string> chunks;
    struct Part { int chunk; std::u32string mark; };
    std::vector<Part> parts;
    Regex::Match m;
    size_t cursor = 0;
    while (cursor <= line.size() && marks_re().search(line, cursor, m)) {
        if (m.start > cursor) {
            parts.push_back({(int)chunks.size(), U""});
            chunks.push_back(u32_to_utf8(line.substr(cursor, m.start - cursor)));
        }
        parts.push_back({-1, line.substr(m.start, m.end - m.start)});
        cursor = m.end > m.start ? m.end : m.end + 1;
    }
    if (cursor < line.size()) {
        parts.push_back({(int)chunks.size(), U""});
        chunks.push_back(u32_to_utf8(line.substr(cursor)));
    }
    // Punctuation.inline_groups
    std::vector<std::vector<int>> groups;
    {
        std::vector<int> group;
        for (size_t num = 0; num < parts.size(); ++num) {
            if (parts[num].chunk < 0) continue;
            if (!group.empty() && num >= 2 && parts[num - 2].chunk == group.back() && is_inline_mark(parts[num - 1].mark))
                group.push_back(parts[num].chunk);
            else {
                if (group.size() > 1) groups.push_back(group);
                group.assign(1, parts[num].chunk);
            }
        }
        if (group.size() > 1) groups.push_back(group);
    }
    // phonemise every piece, then every group as a whole
    std::vector<std::string> text;
    text.reserve(chunks.size());
    for (const auto& c : chunks) text.push_back(piece(c));
    // Punctuation.restore with the groups phonemised in context (Punctuation._context)
    std::vector<std::string> in_context(chunks.size());
    std::vector<bool> has_context(chunks.size(), false);
    for (const auto& grp : groups) {
        std::string joined;
        for (size_t k = 0; k < grp.size(); ++k) { if (k) joined += ' '; joined += chunks[grp[k]]; }
        const std::string phonemized = piece(joined);
        std::vector<size_t> counts;
        size_t total = 0;
        bool all = true;
        for (int c : grp) { counts.push_back(split_words(text[c]).size()); total += counts.back(); all = all && counts.back(); }
        if (!all || split_words(phonemized).size() != total) continue;
        std::vector<size_t> starts;
        size_t acc = 0;
        for (size_t k = 0; k + 1 < counts.size(); ++k) { acc += counts[k]; starts.push_back(acc); }
        std::vector<std::string> pieces = split_space(phonemized);
        std::vector<size_t> cuts{0};
        size_t nwords = 0;
        for (size_t num = 0; num < pieces.size(); ++num) {
            if (pieces[num].empty()) continue;
            if (std::find(starts.begin(), starts.end(), nwords) != starts.end()) cuts.push_back(num);
            ++nwords;
        }
        cuts.push_back(pieces.size());
        for (size_t k = 0; k + 1 < cuts.size() && k < grp.size(); ++k) {
            std::string s;
            for (size_t q = cuts[k]; q < cuts[k + 1]; ++q) { if (q > cuts[k]) s += ' '; s += pieces[q]; }
            in_context[grp[k]] = s;
            has_context[grp[k]] = true;
        }
    }
    std::string punctuated;
    bool have_mark = false;
    std::string mark;
    for (const auto& p : parts) {
        if (p.chunk < 0) {
            if (ends_with(punctuated, ' ')) punctuated.pop_back();
            mark = u32_to_utf8(p.mark);
            have_mark = true;
            punctuated += mark;
        } else {
            const std::string& c = has_context[p.chunk] ? in_context[p.chunk] : text[p.chunk];
            if (!c.empty()) { punctuated += c; have_mark = false; }
        }
    }
    if (have_mark && !ends_with(mark, ' ')) punctuated += ' ';
    for (char& c : punctuated)
        if (c == '\n' || c == '\r') c = ' ';  // one line per chunk in kt08_text_to_phonemes
    return punctuated;
}

}  // namespace kt08fe
