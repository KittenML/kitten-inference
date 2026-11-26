// Part of kitten-inference (Apache-2.0). See num_words.hpp.
//
// How a cardinal is spoken:
//   - Split the number into groups of three digits from the right (units, thousands, millions, ...).
//   - Each non-zero group is said as "<d> hundred and <rest>" (or just "<d> hundred", or just "<rest>"), where
//     <rest> below 100 is a word up to nineteen or "<tens>-<unit>"; groups above the units add their scale word
//     ("thousand", "million", ...).
//   - Groups are joined with ", ", except that the units group is joined with " and " when it is below 100.
#include "num_words.hpp"

#include <vector>

namespace kt_tinyen {
namespace {

#if defined(__SIZEOF_INT128__)
using uint_big = unsigned __int128;  // up to 36 digits
constexpr size_t kMaxDigits = 36;
#else
using uint_big = unsigned long long;  // compilers without a 128-bit integer (MSVC): up to 19 digits
constexpr size_t kMaxDigits = 19;
#endif

const char* const kUnder20[20] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                  "seventeen", "eighteen", "nineteen"};
const char* const kTensWord[10] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
                                   "ninety"};
// Scale word of group g (g = 0 is the units group). 36 digits need groups 0..11.
const char* const kScale[12] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
                                "sextillion", "septillion", "octillion", "nonillion", "decillion"};

void say_below_100(unsigned n, std::string& out) {
    if (n < 20) { out += kUnder20[n]; return; }
    out += kTensWord[n / 10];
    if (n % 10) { out += '-'; out += kUnder20[n % 10]; }
}

void say_group(unsigned n, std::string& out) {  // 1 <= n <= 999
    unsigned h = n / 100, r = n % 100;
    if (h) {
        out += kUnder20[h];
        out += " hundred";
        if (r) out += " and ";
    }
    if (r) say_below_100(r, out);
}

bool read_digits(std::string_view s, uint_big& value) {
    if (s.empty()) return false;
    size_t first = s.find_first_not_of('0');
    size_t significant = first == std::string_view::npos ? 0 : s.size() - first;
    if (significant > kMaxDigits) return false;
    value = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        value = value * 10 + (unsigned)(c - '0');
    }
    return true;
}

std::string say_cardinal(uint_big v) {
    if (v == 0) return "zero";
    unsigned groups[12] = {};
    int top = -1;
    for (int g = 0; v != 0 && g < 12; ++g, v /= 1000) {
        groups[g] = (unsigned)(v % 1000);
        top = g;
    }
    std::string out;
    bool first = true;
    for (int g = top; g >= 0; --g) {
        unsigned n = groups[g];
        if (!n) continue;
        if (!first) out += (g == 0 && n < 100) ? " and " : ", ";
        first = false;
        say_group(n, out);
        if (g) { out += ' '; out += kScale[g]; }
    }
    return out;
}

// The ordinal of one cardinal word: "one" -> "first", "twenty" -> "twentieth", "hundred" -> "hundredth".
std::string ordinal_word(const std::string& w) {
    if (w == "one") return "first";
    if (w == "two") return "second";
    if (w == "three") return "third";
    if (w == "five") return "fifth";
    if (w == "eight") return "eighth";
    if (w == "nine") return "ninth";
    if (w == "twelve") return "twelfth";
    if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ieth";
    return w + "th";
}

}  // namespace

std::string num_cardinal(std::string_view digits) {
    uint_big v;
    return read_digits(digits, v) ? say_cardinal(v) : std::string(digits);
}

std::string num_cardinal(long long v) {
    if (v < 0) return "minus " + num_cardinal(std::to_string(-v));
    return num_cardinal(std::to_string(v));
}

std::string num_ordinal(std::string_view digits) {
    // Only the last word changes: "one hundred and twenty-one" -> "one hundred and twenty-first".
    std::string c = num_cardinal(digits);
    size_t cut = c.find_last_of(" -");
    size_t start = cut == std::string::npos ? 0 : cut + 1;
    return c.substr(0, start) + ordinal_word(c.substr(start));
}

std::string num_year(std::string_view digits) {
    // A year is said in two halves ("nineteen eighty-four", "nineteen hundred", "nineteen oh-five"), except when the
    // first half is 0 or 100+, or a whole ten with a second half below 10 ("two thousand and five", "one thousand").
    uint_big v;
    if (!read_digits(digits, v)) return std::string(digits);
    uint_big century = v / 100;
    unsigned rest = (unsigned)(v % 100);
    bool whole = century == 0 || century >= 100 || (century % 10 == 0 && rest < 10);
    if (whole) return say_cardinal(v);
    std::string out = say_cardinal(century);
    out += ' ';
    if (rest == 0) out += "hundred";
    else if (rest < 10) { out += "oh-"; out += kUnder20[rest]; }
    else say_below_100(rest, out);
    return out;
}

std::string num_float(std::string_view dec) {
    // The shortest float form of an ordinary decimal is the input with trailing zeros of the fraction dropped
    // (keeping at least one digit); each fraction digit is said on its own.
    size_t dot = dec.find('.');
    std::string_view whole = dot == std::string_view::npos ? dec : dec.substr(0, dot);
    std::string_view frac = dot == std::string_view::npos ? std::string_view() : dec.substr(dot + 1);
    while (frac.size() > 1 && frac.back() == '0') frac.remove_suffix(1);
    std::string out = num_cardinal(whole.empty() ? std::string_view("0") : whole);
    out += " point";
    if (frac.empty()) frac = "0";
    for (char c : frac) { out += ' '; out += kUnder20[c - '0']; }
    return out;
}

}  // namespace kt_tinyen
