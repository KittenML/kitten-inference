// Part of kitten-inference (Apache-2.0).
// C++ port of src/kitten_inference/tts2_tiny/frontend/tts_norm.py (text normalisation before the English G2P).
// Same rules, same order. The regular expressions run on UTF-8 bytes with std::regex (ASCII \b, \w, \s: the same as
// Python's for ASCII text). Rules that need a digit (or '$', '@', a URL ...) are skipped when the text has none, so a
// plain sentence costs a few scans. URLs, e-mail addresses, the abbreviation table and the leading minus are scanned
// by hand (no regex recursion on long tokens).
#include "tts_norm.hpp"

#include <cstring>
#include <regex>
#include <string>
#include <vector>

#include "num_words.hpp"
#include "uni.hpp"

namespace kt_tinyen {
namespace {

using std::string;
using rx = std::regex;
constexpr auto ECMA = std::regex::ECMAScript | std::regex::optimize;
constexpr auto ECMAI = std::regex::ECMAScript | std::regex::optimize | std::regex::icase;

template <class F>
string sub(const rx& re, const string& s, F fn) {
    string out;
    auto begin = std::sregex_iterator(s.begin(), s.end(), re), end = std::sregex_iterator();
    size_t last = 0;
    bool any = false;
    for (auto it = begin; it != end; ++it) {
        const std::smatch& m = *it;
        any = true;
        size_t p = (size_t)m.position(0);
        out.append(s, last, p - last);
        out += fn(m, s);
        last = p + (size_t)m.length(0);
    }
    if (!any) return s;
    out.append(s, last, string::npos);
    return out;
}

string replace_str(string s, const string& from, const string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != string::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
    return s;
}

long long to_int(const string& s) {  // int() of a digit string that fits (callers bound the length)
    long long v = 0;
    for (char c : s) if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
    return v;
}
string strip_commas(const string& s) { string o; for (char c : s) if (c != ',') o += c; return o; }
string card(const string& digits) { return num_cardinal(std::string_view(digits)); }
string card(long long v) { return num_cardinal(v); }
string ordinal(const string& digits) { return num_ordinal(digits); }

bool has_digit(const string& s) { for (char c : s) if (c >= '0' && c <= '9') return true; return false; }
bool is_w(unsigned char c) { return c == '_' || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; }
bool is_s(unsigned char c) { return c == ' ' || (c >= 9 && c <= 13); }

// Whitespace runs longer than kMaxSpaceRun keep only their first kMaxSpaceRun characters. The rules only ever see
// "one", "some" or "two or more" spaces and the end result has every run of 2+ as one space, so the output is the
// same, but std::regex (which recurses once per matched character) never sees a long run.
constexpr size_t kMaxSpaceRun = 64;
string cap_space_runs(string s) {
    size_t run = 0, w = 0;
    bool capped = false;
    for (size_t i = 0; i < s.size(); ++i) {
        run = is_s((unsigned char)s[i]) ? run + 1 : 0;
        if (run > kMaxSpaceRun) { capped = true; continue; }
        s[w++] = s[i];
    }
    if (capped) s.resize(w);
    return s;
}

const char* MONTHS[] = {"January", "February", "March", "April", "May", "June", "July",
                        "August", "September", "October", "November", "December"};

string year_words(long long y, const string& digits) {
    if ((1100 <= y && y <= 1999) || (2000 <= y && y <= 2099)) return num_year(std::to_string(y));
    return card(digits);
}

// _keep_final_dot
string keep_final_dot(const std::smatch& m, const string& whole, string out) {
    string g = m.str(0);
    if (g.empty() || g.back() != '.') return out;
    size_t end = (size_t)(m.position(0) + m.length(0));
    string rest = whole.substr(end);
    size_t a = 0;
    while (a < rest.size() && is_s((unsigned char)rest[a])) ++a;
    if (a == rest.size()) return out + ".";
    if (is_s((unsigned char)rest[0])) {
        // rest.strip()[:1].isupper()
        u32 r = utf8_to_u32(rest.substr(a, 4));
        if (!r.empty() && is_upper(r[0])) return out + ".";
    }
    return out;
}

struct Rules {
    rx eg{R"(\be\.\s?g\.)", ECMAI}, ie{R"(\bi\.\s?e\.)", ECMAI}, etc{R"(\betc\.)", ECMAI}, vs{R"(\bvs\.?\b)", ECMAI};
    rx iso{R"(\b(\d{4})-(\d{1,2})-(\d{1,2})\b)", ECMA};
    rx slash{R"(\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b)", ECMA};
    rx month{R"(\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b)", ECMAI};
    rx times{R"(\b(\d{1,2}):(\d{2})(?:\s*([ap]\.?m\.?))?)", ECMAI};
    rx bare_hour{R"(\b(\d{1,2})\s*([ap])\.?m\.?(?![a-z]))", ECMAI};
    rx phones{R"(\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]?\d{0,4}\b)", ECMA};
    rx money{R"(\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?([KkMmBbTt])\b)?)", ECMA};
    rx versions{R"(\b(\d+)\.(\d+)\.(\d+)\b)", ECMA};
    rx fractions{R"(\b(\d{1,3})/(\d{1,3})\b)", ECMA};
    rx magnitudes{R"(\b(\d[\d,]*(?:\.\d+)?)\s*([KMBT])\b)", ECMA};
    rx units{R"(\b(\d[\d,]*(?:\.\d+)?)\s?(mins|secs|lbs|hrs|min|sec|khz|mhz|ghz|mph|kph|km|mi|cm|mm|kg|lb|oz|ml|ft|in|hr|ms|gb|mb|kb|tb|hz|kw|l)\b)", ECMAI};
    rx deg{"((?:-|\xE2\x88\x92)?[\\d.]+)\\s?\xC2\xB0\\s?([CF])\\b", ECMA};
    rx st1{R"(\bSt\.\s+(?=[A-Z]))", ECMA}, st2{R"(\bSt\.(?=\s*$))", ECMA}, st3{R"(\bSt\.(?=\s*["'\)]))", ECMA},
        st4{R"(\bSt\.(?!\s+[A-Z]))", ECMA};
    rx ave{R"(\bAve\.)", ECMA}, rd{R"(\bRd\.)", ECMA}, blvd{R"(\bBlvd\.)", ECMA}, no{R"(\bNo\.\s*(?=\d))", ECMA},
        nos{R"(\bNos\.\s*(?=\d))", ECMA};
    rx romans{R"(\b(chapter|section|part|volume|book|act|phase|appendix|world war|type|class|figure|table|step|stage)\s+([IVX]{1,4})\b)", ECMAI};
    rx ordinals{R"(\b(\d+)(?:st|nd|rd|th)\b)", ECMAI};
    rx ranges{"\\b(\\d+)\\s?(?:-|\xE2\x80\x93|\xE2\x80\x94)\\s?(\\d+)\\b", ECMA};
    rx numbers{R"(\b\d[\d,]*(?:\.\d+)?\b)", ECMA};
    rx punct_space{R"(([.,!?;])(?=[^\s"',.!?;]))", ECMA};
};

const Rules& rules() {
    static const Rules r;
    return r;
}

string decimal_words(const string& tok_in) {
    string tok = strip_commas(tok_in);
    if (!has_digit(tok)) return tok;
    size_t dot = tok.find('.');
    if (dot != string::npos) {
        string a = tok.substr(0, dot), b = tok.substr(dot + 1);
        string out = card(a.empty() ? string("0") : a) + " point";
        for (char d : b) if (d >= '0' && d <= '9') { out += ' '; out += card(string(1, d)); }
        return out;
    }
    return card(tok);
}

// "https?://\S+|www\.\S+" -> " link "
string urls(const string& s) {
    if (s.find("http") == string::npos && s.find("www.") == string::npos) return s;
    string out;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t len = 0;
        if (s.compare(i, 7, "http://") == 0) len = 7;
        else if (s.compare(i, 8, "https://") == 0) len = 8;
        else if (s.compare(i, 4, "www.") == 0) len = 4;
        if (len) {
            size_t j = i + len;
            while (j < n && !is_s((unsigned char)s[j])) ++j;
            if (j > i + len) { out += " link "; i = j; continue; }
        }
        out += s[i++];
    }
    return out;
}

// "\b[\w.+-]+@[\w-]+\.[\w.]+\b" -> " email address "
string emails(const string& s) {
    if (s.find('@') == string::npos) return s;
    static const rx re(R"(\b[\w.+-]+@[\w-]+\.[\w.]+\b)", ECMA);
    // tokens are bounded by whitespace: run the regex per whitespace-free run that holds '@'
    string out;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t j = i;
        while (j < n && !is_s((unsigned char)s[j])) ++j;
        string run = s.substr(i, j - i);
        if (run.find('@') != string::npos) run = std::regex_replace(run, re, " email address ");
        out += run;
        while (j < n && is_s((unsigned char)s[j])) out += s[j++];
        i = j;
    }
    return out;
}

string dates(string s) {
    const Rules& R = rules();
    s = sub(R.iso, s, [](const std::smatch& m, const string&) {
        long long y = to_int(m.str(1)), mo = to_int(m.str(2)), d = to_int(m.str(3));
        if (!(1 <= mo && mo <= 12 && 1 <= d && d <= 31)) return m.str(0);
        return string(MONTHS[mo - 1]) + " " + ordinal(std::to_string(d)) + ", " + year_words(y, m.str(1));
    });
    s = sub(R.slash, s, [](const std::smatch& m, const string&) {
        long long a = to_int(m.str(1)), b = to_int(m.str(2)), y = to_int(m.str(3));
        if (!(1 <= a && a <= 12 && 1 <= b && b <= 31)) return m.str(0);
        if (y < 100) y += y < 50 ? 2000 : 1900;
        return string(MONTHS[a - 1]) + " " + ordinal(std::to_string(b)) + ", " + year_words(y, std::to_string(y));
    });
    s = sub(R.month, s, [](const std::smatch& m, const string&) {
        return m.str(1) + " " + ordinal(std::to_string(to_int(m.str(2)))) + ", " +
               year_words(to_int(m.str(3)), m.str(3));
    });
    return s;
}

string times(const string& s) {
    return sub(rules().times, s, [](const std::smatch& m, const string& whole) {
        long long h = to_int(m.str(1)), mi = to_int(m.str(2));
        string ampm = m[3].matched ? string(m.str(3)) : string();
        for (auto& c : ampm) c = (char)std::tolower((unsigned char)c);
        if (h > 23 || mi > 59) return m.str(0);
        string suffix;
        if (!ampm.empty()) suffix = ampm[0] == 'a' ? " AM" : " PM";
        else if (h == 0) { h = 12; suffix = " AM"; }
        string spoken;
        if (mi == 0) spoken = suffix.empty() ? card(h) + " o'clock" : card(h <= 12 ? h : h - 12);
        else if (mi < 10) spoken = card((h <= 12 || suffix.empty()) ? h : h - 12) + " oh " + card(mi);
        else spoken = card((h <= 12 || suffix.empty()) ? h : h - 12) + " " + card(mi);
        return keep_final_dot(m, whole, spoken + suffix);
    });
}

string bare_hour(const string& s) {
    return sub(rules().bare_hour, s, [](const std::smatch& m, const string& whole) {
        long long h = to_int(m.str(1));
        if (!(1 <= h && h <= 12)) return m.str(0);
        char a = (char)std::tolower((unsigned char)m.str(2)[0]);
        return keep_final_dot(m, whole, card(h) + " " + (a == 'a' ? "AM" : "PM"));
    });
}

string phones(const string& s) {
    return sub(rules().phones, s, [](const std::smatch& m, const string&) {
        string digits;
        for (char c : m.str(0)) if (c >= '0' && c <= '9') digits += c;
        if (!(7 <= digits.size() && digits.size() <= 11)) return m.str(0);
        std::vector<string> groups;
        size_t i = 0;
        if (digits.size() == 11 && digits[0] == '1') { groups.push_back("one"); i = 1; }
        size_t bounds[3][2] = {{i, i + 3}, {i + 3, i + 6}, {i + 6, digits.size()}};
        for (auto& ab : bounds) {
            if (ab[0] < digits.size()) {
                string g;
                for (size_t k = ab[0]; k < std::min(ab[1], digits.size()); ++k) {
                    if (!g.empty()) g += ' ';
                    g += card(string(1, digits[k]));
                }
                groups.push_back(g);
            }
        }
        string out;
        for (auto& g : groups) {
            if (g.empty()) continue;
            if (!out.empty()) out += ", ";
            out += g;
        }
        return out;
    });
}

const char* mag_word(char c) {
    switch (std::tolower((unsigned char)c)) {
        case 'k': return "thousand";
        case 'm': return "million";
        case 'b': return "billion";
        default: return "trillion";
    }
}

string money(const string& s) {
    return sub(rules().money, s, [](const std::smatch& m, const string&) {
        string amt = strip_commas(m.str(1));
        string suf = m[2].matched ? m.str(2) : string();
        if (!suf.empty()) {
            size_t dot = amt.find('.');
            string whole;
            bool integer = true;
            if (dot != string::npos) for (size_t k = dot + 1; k < amt.size(); ++k) if (amt[k] != '0') integer = false;
            string ip = dot == string::npos ? amt : amt.substr(0, dot);
            if (integer) whole = card(ip);
            else {
                whole = card(ip) + " point";
                for (size_t k = dot + 1; k < amt.size(); ++k) { whole += ' '; whole += card(string(1, amt[k])); }
            }
            return whole + " " + mag_word(suf[0]) + " dollars";
        }
        size_t dot = amt.find('.');
        if (dot != string::npos) {
            string d = amt.substr(0, dot), c = (amt.substr(dot + 1) + "00").substr(0, 2);
            long long dv = to_int(d), cv = to_int(c);
            string dollars = card(d) + " dollar" + (dv == 1 ? "" : "s");
            if (cv) return dollars + " and " + card(cv) + " cent" + (cv == 1 ? "" : "s");
            return dollars;
        }
        return card(amt) + " dollar" + (amt.size() < 19 && to_int(amt) == 1 ? "" : "s");
    });
}

string versions(const string& s) {
    return sub(rules().versions, s, [](const std::smatch& m, const string&) {
        return card(m.str(1)) + " point " + card(m.str(2)) + " point " + card(m.str(3));
    });
}

string fractions(const string& s) {
    return sub(rules().fractions, s, [](const std::smatch& m, const string&) {
        long long a = to_int(m.str(1)), b = to_int(m.str(2));
        if (a == 1 && b == 2) return string("a half");
        if (a == 1 && b == 3) return string("a third");
        if (a == 2 && b == 3) return string("two thirds");
        if (a == 1 && b == 4) return string("a quarter");
        if (a == 3 && b == 4) return string("three quarters");
        if (a == 1 && b == 8) return string("an eighth");
        if (b == 0) return m.str(0);
        return card(a) + " " + ordinal(std::to_string(b)) + (a != 1 ? "s" : "");
    });
}

string magnitudes(const string& s) {
    return sub(rules().magnitudes, s, [](const std::smatch& m, const string&) {
        return decimal_words(m.str(1)) + " " + mag_word(m.str(2)[0]);
    });
}

const char* unit_word(const string& u) {
    static const char* T[][2] = {{"km", "kilometres"}, {"mi", "miles"}, {"cm", "centimetres"}, {"mm", "millimetres"},
        {"kg", "kilograms"}, {"lb", "pounds"}, {"lbs", "pounds"}, {"oz", "ounces"}, {"ml", "millilitres"},
        {"l", "litres"}, {"ft", "feet"}, {"in", "inches"}, {"hr", "hours"}, {"hrs", "hours"}, {"min", "minutes"},
        {"mins", "minutes"}, {"sec", "seconds"}, {"secs", "seconds"}, {"ms", "milliseconds"}, {"gb", "gigabytes"},
        {"mb", "megabytes"}, {"kb", "kilobytes"}, {"tb", "terabytes"}, {"hz", "hertz"}, {"khz", "kilohertz"},
        {"mhz", "megahertz"}, {"ghz", "gigahertz"}, {"kw", "kilowatts"}, {"mph", "miles per hour"},
        {"kph", "kilometres per hour"}};
    for (auto& t : T) if (u == t[0]) return t[1];
    return "";
}

string units(string s) {
    const Rules& R = rules();
    s = sub(R.units, s, [](const std::smatch& m, const string&) {
        string n = m.str(1), unit = m.str(2);
        for (auto& c : unit) c = (char)std::tolower((unsigned char)c);
        string val = decimal_words(n);
        string word = unit_word(unit);
        string nc = strip_commas(n);
        if (nc == "1" || nc == "1.0") {
            if (!word.empty() && word.back() == 's') while (!word.empty() && word.back() == 's') word.pop_back();
        }
        return val + " " + word;
    });
    if (s.find("\xC2\xB0") != string::npos) {
        s = sub(R.deg, s, [](const std::smatch& m, const string&) {
            string raw = m.str(1), unit = m.str(2);
            bool neg = false;
            if (!raw.empty() && raw[0] == '-') { neg = true; raw = raw.substr(1); }
            else if (raw.compare(0, 3, "\xE2\x88\x92") == 0) { neg = true; raw = raw.substr(3); }
            string val = decimal_words(raw);
            string scale = (unit == "C" || unit == "c") ? "Celsius" : "Fahrenheit";
            return string(neg ? "minus " : "") + val + " degrees " + scale;
        });
    }
    return s;
}

// "\b{k}\." for the DAYS / MONTHS_ABBR tables (case-sensitive) and ABBREV (any case), in one pass.
string abbrev_table(const string& s) {
    static const char* CS[][2] = {{"Mon", "Monday"}, {"Tue", "Tuesday"}, {"Tues", "Tuesday"}, {"Wed", "Wednesday"},
        {"Weds", "Wednesday"}, {"Thu", "Thursday"}, {"Thur", "Thursday"}, {"Thurs", "Thursday"}, {"Fri", "Friday"},
        {"Sat", "Saturday"}, {"Sun", "Sunday"}, {"Jan", "January"}, {"Feb", "February"}, {"Mar", "March"},
        {"Apr", "April"}, {"Jun", "June"}, {"Jul", "July"}, {"Aug", "August"}, {"Sep", "September"},
        {"Sept", "September"}, {"Oct", "October"}, {"Nov", "November"}, {"Dec", "December"}};
    static const char* CI[][2] = {{"mr", "Mister"}, {"mrs", "Missus"}, {"ms", "Miss"}, {"dr", "Doctor"},
        {"prof", "Professor"}, {"rev", "Reverend"}, {"hon", "Honourable"}, {"ext", "extension"},
        {"approx", "approximately"}, {"vs", "versus"}, {"est", "estimated"}, {"dept", "department"},
        {"univ", "university"}, {"fig", "figure"}, {"eq", "equation"}, {"ref", "reference"}, {"vol", "volume"},
        {"incl", "including"}, {"excl", "excluding"}, {"min", "minimum"}, {"max", "maximum"}, {"avg", "average"},
        {"mt", "mount"}, {"ft", "fort"}, {"appt", "appointment"}, {"pp", "pages"}, {"jr", "Junior"},
        {"sr", "Senior"}, {"inc", "Incorporated"}, {"ltd", "Limited"}, {"corp", "Corporation"},
        {"mins", "minutes"}, {"hrs", "hours"}, {"wks", "weeks"}, {"yrs", "years"}};
    if (s.find('.') == string::npos) return s;
    string out;
    size_t last = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '.') continue;
        size_t a = i;
        while (a > 0 && is_w((unsigned char)s[a - 1])) --a;
        if (a == i || i - a > 6 || a < last) continue;
        string w = s.substr(a, i - a);
        const char* rep = nullptr;
        for (auto& t : CS) if (w == t[0]) { rep = t[1]; break; }
        if (!rep) {
            string wl = w;
            for (auto& c : wl) c = (char)std::tolower((unsigned char)c);
            for (auto& t : CI) if (wl == t[0]) { rep = t[1]; break; }
        }
        if (!rep) continue;
        out.append(s, last, a - last);
        out += rep;
        last = i + 1;
    }
    out.append(s, last, string::npos);
    return out;
}

string abbrev(string s) {
    const Rules& R = rules();
    if (s.find("St.") != string::npos) {
        s = std::regex_replace(s, R.st1, "Saint ");
        s = std::regex_replace(s, R.st2, "Street.");
        s = std::regex_replace(s, R.st3, "Street.");
        s = std::regex_replace(s, R.st4, "Street");
    }
    if (s.find("Ave.") != string::npos) s = std::regex_replace(s, R.ave, "Avenue");
    if (s.find("Rd.") != string::npos) s = std::regex_replace(s, R.rd, "Road");
    if (s.find("Blvd.") != string::npos) s = std::regex_replace(s, R.blvd, "Boulevard");
    if (s.find("No") != string::npos) {
        s = std::regex_replace(s, R.no, "number ");
        s = std::regex_replace(s, R.nos, "numbers ");
    }
    return abbrev_table(s);
}

string romans(const string& s) {
    static const char* ctx[] = {"chapter", "section", "part", "volume", "book", "act", "phase", "appendix",
                                "world war", "type", "class", "figure", "table", "step", "stage"};
    string low = s;
    for (auto& c : low) c = (char)std::tolower((unsigned char)c);
    bool any = false;
    for (auto* c : ctx) if (low.find(c) != string::npos) { any = true; break; }
    if (!any) return s;
    return sub(rules().romans, s, [](const std::smatch& m, const string&) {
        string r = m.str(2);
        for (auto& c : r) c = (char)std::tolower((unsigned char)c);
        static const char* R[] = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"};
        for (int k = 0; k < 12; ++k) if (r == R[k]) return m.str(1) + " " + card(k + 1);
        return m.str(0);
    });
}

string ordinals(const string& s) {
    return sub(rules().ordinals, s, [](const std::smatch& m, const string&) { return ordinal(m.str(1)); });
}

string ranges(const string& s) {
    return sub(rules().ranges, s, [](const std::smatch& m, const string&) {
        return card(m.str(1)) + " to " + card(m.str(2));
    });
}

string numbers(string s) {
    // (?<![\w.])[-−](\d) -> "minus \1"
    string o;
    for (size_t i = 0; i < s.size();) {
        size_t len = 0;
        if (s[i] == '-') len = 1;
        else if (s.compare(i, 3, "\xE2\x88\x92") == 0) len = 3;
        if (len && i + len < s.size() && s[i + len] >= '0' && s[i + len] <= '9') {
            bool prev_ok = i == 0 || !(is_w((unsigned char)s[i - 1]) || s[i - 1] == '.');
            if (prev_ok) { o += "minus "; i += len; continue; }
        }
        o += s[i++];
    }
    s.swap(o);
    return sub(rules().numbers, s, [](const std::smatch& m, const string&) {
        string tok = m.str(0);
        if (tok.find(',') == string::npos && tok.size() == 4 && tok.find('.') == string::npos) {
            long long v = to_int(tok);
            if (1100 <= v && v <= 2099) return year_words(v, tok);
        }
        return decimal_words(tok);
    });
}

// The subset of NFKC that matters for English text input (compatibility characters; precomposed letters are kept).
u32 nfkc_lite(const u32& in) {
    u32 o;
    o.reserve(in.size());
    for (char32_t c : in) {
        if (c == 0xA0 || (c >= 0x2000 && c <= 0x200A) || c == 0x202F || c == 0x205F || c == 0x3000) o += U' ';
        else if (c >= 0xFF01 && c <= 0xFF5E) o += (char32_t)(c - 0xFF01 + 0x21);
        else if (c == 0xB2) o += U'2';
        else if (c == 0xB3) o += U'3';
        else if (c == 0xB9) o += U'1';
        else if (c == 0xAA) o += U'a';
        else if (c == 0xBA) o += U'o';
        else if (c == 0xB5) o += (char32_t)0x3BC;
        else if (c == 0xBC) o += U"1⁄4";
        else if (c == 0xBD) o += U"1⁄2";
        else if (c == 0xBE) o += U"3⁄4";
        else if (c == 0x2024) o += U'.';
        else if (c == 0x2025) o += U"..";
        else if (c == 0x2026) o += U"...";
        else if (c == 0x2011) o += (char32_t)0x2010;
        else if (c == 0x2033) o += U"′′";
        else if (c == 0x2122) o += U"TM";
        else if (c == 0x2120) o += U"SM";
        else if (c == 0x2116) o += U"No";
        else if (c == 0x2103) o += U"°C";
        else if (c == 0x2109) o += U"°F";
        else if (c == 0xFB00) o += U"ff";
        else if (c == 0xFB01) o += U"fi";
        else if (c == 0xFB02) o += U"fl";
        else if (c == 0xFB03) o += U"ffi";
        else if (c == 0xFB04) o += U"ffl";
        else if (c >= 0x2070 && c <= 0x2079 && c != 0x2071 && c != 0x2072 && c != 0x2073) o += (char32_t)('0' + (c - 0x2070));
        else if (c >= 0x2080 && c <= 0x2089) o += (char32_t)('0' + (c - 0x2080));
        else o += c;
    }
    return o;
}

}  // namespace

std::string tts_normalize(std::string_view text) {
    u32 t = nfkc_lite(utf8_to_u32(text));
    replace_all(t, U"’", U"'");
    replace_all(t, U"‘", U"'");
    replace_all(t, U"“", U"\"");
    replace_all(t, U"”", U"\"");
    replace_all(t, U"—", U" - ");
    replace_all(t, U"–", U"-");
    replace_all(t, U"…", U"...");
    string s = cap_space_runs(u32_to_utf8(t));
    const Rules& R = rules();
    s = urls(s);
    s = emails(s);
    {
        string low = s;
        for (auto& c : low) c = (char)std::tolower((unsigned char)c);
        if (low.find("e.") != string::npos) s = std::regex_replace(s, R.eg, "for example,");
        if (low.find("i.") != string::npos) s = std::regex_replace(s, R.ie, "that is,");
        if (low.find("etc.") != string::npos) s = std::regex_replace(s, R.etc, "etcetera");
        if (low.find("vs") != string::npos) s = std::regex_replace(s, R.vs, "versus");
    }
    bool dig = has_digit(s);
    if (dig) {
        s = dates(s);
        s = times(s);
        s = bare_hour(s);
        s = phones(s);
        if (s.find('$') != string::npos) s = money(s);
        s = versions(s);
        s = fractions(s);
        s = magnitudes(s);
        s = units(s);
    } else if (s.find("\xC2\xB0") != string::npos) {
        s = units(s);
    }
    s = abbrev(s);
    s = romans(s);
    if (has_digit(s)) {
        s = ordinals(s);
        s = ranges(s);
    }
    static const char* SYM[][2] = {{"%", " percent "}, {"&", " and "}, {"@", " at "}, {"+", " plus "},
        {"=", " equals "}, {"\xC3\x97", " times "}, {"\xC3\xB7", " divided by "}, {"<", " less than "},
        {">", " greater than "}, {"~", " about "}, {"\xC2\xB0", " degrees "}, {"\xE2\x82\xAC", " euros "},
        {"\xC2\xA3", " pounds "}, {"\xC2\xA5", " yen "}, {"\xC2\xA9", " copyright "},
        {"\xE2\x84\xA2", " trademark "}, {"\xC2\xA7", " section "}};
    for (auto& kv : SYM) if (s.find(kv[0]) != string::npos) s = replace_str(s, kv[0], kv[1]);
    if (has_digit(s) || s.find('-') != string::npos || s.find("\xE2\x88\x92") != string::npos) s = numbers(s);
    for (auto& c : s) if (c == ':') c = ',';
    for (auto& c : s) if (c && std::strchr("_*`^|\\/[]{}<>", c)) c = ' ';
    // \s+([.,!?;]) -> \1, by hand (std::regex recurses once per matched character).
    {
        string t;
        t.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            if (!is_s((unsigned char)s[i])) { t += s[i++]; continue; }
            size_t j = i;
            while (j < s.size() && is_s((unsigned char)s[j])) ++j;
            if (!(j < s.size() && std::strchr(".,!?;", s[j]))) t.append(s, i, j - i);
            i = j;
        }
        s.swap(t);
    }
    s = std::regex_replace(s, R.punct_space, "$1 ");
    // \s{2,} -> " ", then strip()
    string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (is_s((unsigned char)s[i])) {
            size_t j = i;
            while (j < s.size() && is_s((unsigned char)s[j])) ++j;
            if (j - i >= 2) o += ' ';
            else o += s[i];
            i = j;
        } else o += s[i++];
    }
    size_t a = 0, b = o.size();
    while (a < b && is_s((unsigned char)o[a])) ++a;
    while (b > a && is_s((unsigned char)o[b - 1])) --b;
    return o.substr(a, b - a);
}

}  // namespace kt_tinyen
