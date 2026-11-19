// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// The Python package's Japanese path (tts2_tiny/frontend/misaki/cutlet.py + ja.py + num2kana.py, mojimoji.py,
// phonemize_v2.py, Frontend) in C++. The kana -> IPA table and the number words are the package's own (misaki,
// Apache-2.0; cutlet, MIT; Convert-Numbers-to-Japanese, MIT: see engines/lite/NOTICE).
#include "frontend/lang/ja.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "frontend/lang/ja_dict.hpp"

namespace kt_fe {
namespace {

#include "frontend/lang/ja_unicode_tables.inc"

// ---------------------------------------------------------------------------------------------------------------
// UTF-8 and Unicode helpers
void put(std::string& o, char32_t c) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
}
std::u32string dec(const std::string& s) {
    std::u32string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        char32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; } else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { i++; continue; }   // stray byte: dropped
        if (i + n > s.size()) break;
        bool ok = true;
        for (int k = 1; k < n; k++) {
            const unsigned char d = (unsigned char)s[i + k];
            if ((d & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (d & 0x3F);
        }
        if (!ok || cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) { i++; continue; }
        o += cp; i += n;
    }
    return o;
}
std::string enc(const std::u32string& s) { std::string o; o.reserve(s.size() * 2); for (char32_t c : s) put(o, c); return o; }

template <size_t N>
bool in_ranges(const uint32_t (&r)[N][2], char32_t c) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t m = (lo + hi) / 2;
        if (c < r[m][0]) hi = m;
        else if (c > r[m][1]) lo = m + 1;
        else return true;
    }
    return false;
}
bool py_space(char32_t c) { return in_ranges(kSpaces, c); }
// Python re \d (Unicode category Nd); the digit's value (Nd digits come in runs of ten from zero)
int nd_value(char32_t c) {
    size_t lo = 0, hi = sizeof(kDigits) / sizeof(kDigits[0]);
    while (lo < hi) {
        const size_t m = (lo + hi) / 2;
        if (c < kDigits[m][0]) hi = m;
        else if (c > kDigits[m][1]) lo = m + 1;
        else return (int)((c - kDigits[m][0]) % 10);
    }
    return -1;
}
// Python str.isdigit() for one character (Numeric_Type Decimal or Digit: every Nd digit, and symbols such as ² ① ❶)
bool py_isdigit(char32_t c) { return in_ranges(kIsDigit, c); }
bool py_isdigit(const std::u32string& s) {
    for (char32_t c : s) if (!py_isdigit(c)) return false;
    return !s.empty();
}
int combining_class(char32_t c) {
    size_t lo = 0, hi = sizeof(kCombining) / sizeof(kCombining[0]);
    while (lo < hi) {
        const size_t m = (lo + hi) / 2;
        if (c < kCombining[m][0]) hi = m;
        else if (c > kCombining[m][1]) lo = m + 1;
        else return (int)kCombining[m][2];
    }
    return 0;
}
constexpr char32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
constexpr int LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

bool compose_pair(char32_t a, char32_t b, char32_t* out) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount) {
        *out = SBase + ((a - LBase) * VCount + (b - VBase)) * TCount; return true;
    }
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount) {
        *out = a + (b - TBase); return true;
    }
    size_t lo = 0, hi = sizeof(kCompose) / sizeof(kCompose[0]);
    while (lo < hi) {
        const size_t m = (lo + hi) / 2;
        if (a < kCompose[m][0] || (a == kCompose[m][0] && b < kCompose[m][1])) hi = m;
        else if (a > kCompose[m][0] || b > kCompose[m][1]) lo = m + 1;
        else { *out = kCompose[m][2]; return true; }
    }
    return false;
}

// Unicode NFKC: full compatibility decomposition, canonical ordering, canonical composition.
std::u32string nfkc(const std::u32string& s) {
    std::u32string d;
    d.reserve(s.size() + 8);
    for (char32_t c : s) {
        if (c >= SBase && c < SBase + SCount) {
            const int i = (int)(c - SBase);
            d += (char32_t)(LBase + i / NCount);
            d += (char32_t)(VBase + (i % NCount) / TCount);
            if (i % TCount) d += (char32_t)(TBase + i % TCount);
            continue;
        }
        size_t lo = 0, hi = sizeof(kDecomp) / sizeof(kDecomp[0]);
        bool found = false;
        while (lo < hi) {
            const size_t m = (lo + hi) / 2;
            if (c < kDecomp[m][0]) hi = m;
            else if (c > kDecomp[m][0]) lo = m + 1;
            else { for (uint32_t k = 0; k < kDecomp[m][2]; ++k) d += (char32_t)kDecompData[kDecomp[m][1] + k]; found = true; break; }
        }
        if (!found) d += c;
    }
    // canonical ordering: stable sort of each run of non-starters by combining class
    for (size_t i = 0; i < d.size();) {
        if (combining_class(d[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && combining_class(d[j]) != 0) ++j;
        std::stable_sort(d.begin() + (long)i, d.begin() + (long)j,
                         [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }
    // canonical composition
    std::u32string o;
    o.reserve(d.size());
    long starter = -1;
    int last_cc = 0;
    for (char32_t c : d) {
        const int cc = combining_class(c);
        if (starter >= 0) {
            const bool adjacent = (long)o.size() - 1 == starter;
            const bool blocked = !adjacent && (last_cc == 0 || last_cc >= cc);
            char32_t comp;
            if (!blocked && compose_pair(o[(size_t)starter], c, &comp)) { o[(size_t)starter] = comp; continue; }
        }
        if (cc == 0) starter = (long)o.size();
        o += c;
        last_cc = cc;
    }
    return o;
}

std::u32string py_strip(const std::u32string& s) {
    size_t a = 0, b = s.size();
    while (a < b && py_space(s[a])) a++;
    while (b > a && py_space(s[b - 1])) b--;
    return s.substr(a, b - a);
}
void replace_all(std::u32string& s, const std::u32string& a, const std::u32string& b) {
    if (a.empty()) return;
    size_t p = 0;
    while ((p = s.find(a, p)) != std::u32string::npos) { s.replace(p, a.size(), b); p += b.size(); }
}
// Python's `a in b` for strings (a substring of b; the empty string is in every string)
bool py_in(const std::u32string& a, const std::u32string& b) { return b.find(a) != std::u32string::npos; }

// ---------------------------------------------------------------------------------------------------------------
// num2kana.Convert(digits, 'hiragana') (misaki; from Convert-Numbers-to-Japanese, MIT): the number words, joined.
const char* num_word(const std::string& k) {
    static const std::map<std::string, const char*> m = {
        {"0", "ゼロ"}, {"1", "いち"}, {"2", "に"}, {"3", "さん"}, {"4", "よん"}, {"5", "ご"}, {"6", "ろく"},
        {"7", "なな"}, {"8", "はち"}, {"9", "きゅう"}, {"10", "じゅう"}, {"100", "ひゃく"}, {"1000", "せん"},
        {"10000", "まん"}, {"100000000", "おく"}, {"300", "さんびゃく"}, {"600", "ろっぴゃく"},
        {"800", "はっぴゃく"}, {"3000", "さんぜん"}, {"8000", "はっせん"}, {"01000", "いっせん"}};
    auto it = m.find(k);
    return it == m.end() ? "" : it->second;
}
std::string n_one(const std::string& n) { return num_word(n); }
std::string n_two(const std::string& n) {
    if (n[0] == '0') return n_one(n.substr(1, 1));
    if (n == "10") return num_word("10");
    if (n[0] == '1') return std::string(num_word("10")) + n_one(n.substr(1, 1));
    if (n[1] == '0') return n_one(n.substr(0, 1)) + num_word("10");
    return n_one(n.substr(0, 1)) + num_word("10") + n_one(n.substr(1, 1));
}
std::string n_three(const std::string& n) {
    std::string o;
    if (n[0] == '1') o += num_word("100");
    else if (n[0] == '3') o += num_word("300");
    else if (n[0] == '6') o += num_word("600");
    else if (n[0] == '8') o += num_word("800");
    else { o += n_one(n.substr(0, 1)); o += num_word("100"); }
    if (!(n.substr(1) == "00" && n.size() == 3)) {
        if (n[1] == '0') o += n_one(n.substr(2, 1));
        else o += n_two(n.substr(1));
    }
    return o;
}
std::string n_four(std::string n, bool stand_alone) {
    if (n == "0000") return "";
    while (!n.empty() && n[0] == '0') n = n.substr(1);
    if (n.size() == 1) return n_one(n);
    if (n.size() == 2) return n_two(n);
    if (n.size() == 3) return n_three(n);
    std::string o;
    if (n[0] == '1' && stand_alone) o += num_word("1000");
    else if (n[0] == '1') o += num_word("01000");
    else if (n[0] == '3') o += num_word("3000");
    else if (n[0] == '8') o += num_word("8000");
    else { o += n_one(n.substr(0, 1)); o += num_word("1000"); }
    if (!(n.substr(1) == "000" && n.size() == 4)) {
        if (n[1] == '0') o += n_two(n.substr(2));
        else o += n_three(n.substr(1));
    }
    return o;
}
std::string n_x(const std::string& n) {
    const std::string head = n.substr(0, n.size() - 4);
    std::string o;
    if (head.size() == 1) { o += n_one(head); o += num_word("10000"); }
    else if (head.size() == 2) { o += n_two(n.substr(0, 2)); o += num_word("10000"); }
    else if (head.size() == 3) { o += n_three(n.substr(0, 3)); o += num_word("10000"); }
    else if (head.size() == 4) { o += n_four(n.substr(0, 4), false); o += num_word("10000"); }
    else {
        o += n_one(n.substr(0, 1)); o += num_word("100000000"); o += n_four(n.substr(1, 4), false);
        if (n.substr(1, 4) != "0000") o += num_word("10000");
    }
    o += n_four(n.substr(n.size() - 4), false);
    return o;
}
std::string convert_number(std::string n) {
    // The package's Convert() returns an English message here ("Number length too long, choose less than 10
    // digits"), and cutlet then raises on that message's "10": the text is refused.
    if (n.size() > 9)
        throw JaTextError("Japanese: a number of more than 9 digits cannot be read (the Python package raises an "
                          "error on it too); write it in words or in shorter groups");
    while (n.size() > 1 && n[0] == '0') n = n.substr(1);
    if (n.size() == 1) return n_one(n);
    if (n.size() == 2) return n_two(n);
    if (n.size() == 3) return n_three(n);
    if (n.size() == 4) return n_four(n, true);
    return n_x(n);
}

// ---------------------------------------------------------------------------------------------------------------
// cutlet._normalize_text
std::u32string normalize_text(const std::u32string& in) {
    std::u32string t;
    t.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {   // [〜～](?=\d) -> から
        if ((in[i] == 0x301C || in[i] == 0xFF5E) && i + 1 < in.size() && nd_value(in[i + 1]) >= 0) { t += U"から"; continue; }
        t += in[i];
    }
    static const std::u32string kpe_from = U"ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ", kpe_to = U"クシストヌハヒフヘホムラリルレロ";
    for (auto& c : t) { const size_t k = kpe_from.find(c); if (k != std::u32string::npos) c = kpe_to[k]; }
    t = nfkc(t);
    // mojimoji.zen_to_han(kana=False): full-width ASCII and digits -> half-width
    static const std::u32string zen = U"ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿‘｛｜｝～　＼０１２３４５６７８９";
    static const std::u32string han = U"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[¥]^_`{|}~ \\0123456789";
    for (auto& c : t) { const size_t k = zen.find(c); if (k != std::u32string::npos) c = han[k]; }
    // mojimoji.han_to_zen(digit=False, ascii=False) maps half-width katakana (U+FF61..U+FF9F) to full width; NFKC
    // above has already done that for every one of them, so it changes nothing here.
    // ''.join((' ' + Convert(t)) if t.isdigit() else t for t in re.findall(r'\d+|\D+', text))
    // Each piece is a longest run of \d or of \D characters. Convert() has words for ASCII digits only: a piece
    // that str.isdigit() accepts but Convert() cannot read (a decimal digit of another script, which NFKC keeps,
    // or a run of digit symbols such as "❶") makes the package raise KeyError, so the text is refused here too.
    static const char* const kNoWords = "Japanese: a digit that is not 0-9 (such as ٣ or ❶) cannot be read (the "
                                        "Python package raises an error on it too)";
    std::u32string o;
    o.reserve(t.size() + 16);
    for (size_t i = 0; i < t.size();) {
        const bool nd = nd_value(t[i]) >= 0;
        size_t j = i;
        bool all_digit = true, ascii = true;
        for (; j < t.size() && (nd_value(t[j]) >= 0) == nd; ++j) {
            all_digit = all_digit && py_isdigit(t[j]);
            ascii = ascii && t[j] < 0x80;
        }
        if (!nd) {
            if (all_digit) throw JaTextError(kNoWords);
            o.append(t, i, j - i);
        } else {
            if (!ascii && j - i <= 9) throw JaTextError(kNoWords);
            std::string digits;
            for (size_t k = i; k < j; ++k) digits += (char)('0' + nd_value(t[k]));
            o += U' ';
            o += dec(convert_number(digits));
        }
        i = j;
    }
    return o;
}

// ---------------------------------------------------------------------------------------------------------------
// cutlet's kana -> IPA table (HEPBURN in misaki/cutlet.py)
struct U32Hash { size_t operator()(const std::u32string& s) const { size_t h = 1469598103934665603ull; for (char32_t c : s) { h ^= c; h *= 1099511628211ull; } return h; } };
using Table = std::unordered_map<std::u32string, std::u32string, U32Hash>;
const Table& hepburn() {
    static const Table t = [] {
        static const char* const kv[][2] = {
            {"ぁ", "a"},
            {"あ", "a"},
            {"ぃ", "i"},
            {"い", "i"},
            {"ぅ", "ɯ"},
            {"う", "ɯ"},
            {"ぇ", "e"},
            {"え", "e"},
            {"ぉ", "o"},
            {"お", "o"},
            {"か", "ka"},
            {"が", "ɡa"},
            {"き", "kʲi"},
            {"ぎ", "ɡʲi"},
            {"く", "kɯ"},
            {"ぐ", "ɡɯ"},
            {"け", "ke"},
            {"げ", "ɡe"},
            {"こ", "ko"},
            {"ご", "ɡo"},
            {"さ", "sa"},
            {"ざ", "ʣa"},
            {"し", "ɕi"},
            {"じ", "ʥi"},
            {"す", "sɨ"},
            {"ず", "zɨ"},
            {"せ", "se"},
            {"ぜ", "ʣe"},
            {"そ", "so"},
            {"ぞ", "ʣo"},
            {"た", "ta"},
            {"だ", "da"},
            {"ち", "ʨi"},
            {"ぢ", "ʥi"},
            {"つ", "ʦɨ"},
            {"づ", "zɨ"},
            {"て", "te"},
            {"で", "de"},
            {"と", "to"},
            {"ど", "do"},
            {"な", "na"},
            {"に", "ɲi"},
            {"ぬ", "nɯ"},
            {"ね", "ne"},
            {"の", "no"},
            {"は", "ha"},
            {"ば", "ba"},
            {"ぱ", "pa"},
            {"ひ", "çi"},
            {"び", "bʲi"},
            {"ぴ", "pʲi"},
            {"ふ", "ɸɯ"},
            {"ぶ", "bɯ"},
            {"ぷ", "pɯ"},
            {"へ", "he"},
            {"べ", "be"},
            {"ぺ", "pe"},
            {"ほ", "ho"},
            {"ぼ", "bo"},
            {"ぽ", "po"},
            {"ま", "ma"},
            {"み", "mʲi"},
            {"む", "mɯ"},
            {"め", "me"},
            {"も", "mo"},
            {"ゃ", "ja"},
            {"や", "ja"},
            {"ゅ", "jɯ"},
            {"ゆ", "jɯ"},
            {"ょ", "jo"},
            {"よ", "jo"},
            {"ら", "ɾa"},
            {"り", "ɾʲi"},
            {"る", "ɾɯ"},
            {"れ", "ɾe"},
            {"ろ", "ɾo"},
            {"ゎ", "βa"},
            {"わ", "βa"},
            {"ゐ", "i"},
            {"ゑ", "e"},
            {"を", "o"},
            {"ゔ", "vɯ"},
            {"ゕ", "ka"},
            {"ゖ", "ke"},
            {"ヷ", "va"},
            {"ヸ", "vʲi"},
            {"ヹ", "ve"},
            {"ヺ", "vo"},
            {"いぇ", "je"},
            {"うぃ", "βi"},
            {"うぇ", "βe"},
            {"うぉ", "βo"},
            {"きぇ", "kʲe"},
            {"きゃ", "kʲa"},
            {"きゅ", "kʲɨ"},
            {"きょ", "kʲo"},
            {"ぎゃ", "ɡʲa"},
            {"ぎゅ", "ɡʲɨ"},
            {"ぎょ", "ɡʲo"},
            {"くぁ", "kᵝa"},
            {"くぃ", "kᵝi"},
            {"くぇ", "kᵝe"},
            {"くぉ", "kᵝo"},
            {"ぐぁ", "ɡᵝa"},
            {"ぐぃ", "ɡᵝi"},
            {"ぐぇ", "ɡᵝe"},
            {"ぐぉ", "ɡᵝo"},
            {"しぇ", "ɕe"},
            {"しゃ", "ɕa"},
            {"しゅ", "ɕɨ"},
            {"しょ", "ɕo"},
            {"じぇ", "ʥe"},
            {"じゃ", "ʥa"},
            {"じゅ", "ʥɨ"},
            {"じょ", "ʥo"},
            {"ちぇ", "ʨe"},
            {"ちゃ", "ʨa"},
            {"ちゅ", "ʨɨ"},
            {"ちょ", "ʨo"},
            {"ぢゃ", "ʥa"},
            {"ぢゅ", "ʥɨ"},
            {"ぢょ", "ʥo"},
            {"つぁ", "ʦa"},
            {"つぃ", "ʦʲi"},
            {"つぇ", "ʦe"},
            {"つぉ", "ʦo"},
            {"てぃ", "tʲi"},
            {"てゅ", "tʲɨ"},
            {"でぃ", "dʲi"},
            {"でゅ", "dʲɨ"},
            {"とぅ", "tɯ"},
            {"どぅ", "dɯ"},
            {"にぇ", "ɲe"},
            {"にゃ", "ɲa"},
            {"にゅ", "ɲɨ"},
            {"にょ", "ɲo"},
            {"ひぇ", "çe"},
            {"ひゃ", "ça"},
            {"ひゅ", "çɨ"},
            {"ひょ", "ço"},
            {"びゃ", "bʲa"},
            {"びゅ", "bʲɨ"},
            {"びょ", "bʲo"},
            {"ぴゃ", "pʲa"},
            {"ぴゅ", "pʲɨ"},
            {"ぴょ", "pʲo"},
            {"ふぁ", "ɸa"},
            {"ふぃ", "ɸʲi"},
            {"ふぇ", "ɸe"},
            {"ふぉ", "ɸo"},
            {"ふゅ", "ɸʲɨ"},
            {"ふょ", "ɸʲo"},
            {"みゃ", "mʲa"},
            {"みゅ", "mʲɨ"},
            {"みょ", "mʲo"},
            {"りゃ", "ɾʲa"},
            {"りゅ", "ɾʲɨ"},
            {"りょ", "ɾʲo"},
            {"ゔぁ", "va"},
            {"ゔぃ", "vʲi"},
            {"ゔぇ", "ve"},
            {"ゔぉ", "vo"},
            {"ゔゅ", "bʲɨ"},
            {"ゔょ", "bʲo"},
            {"。", "."},
            {"、", ","},
            {"？", "?"},
            {"！", "!"},
            {"「", "“"},
            {"」", "”"},
            {"『", "“"},
            {"』", "”"},
            {"：", ":"},
            {"；", ";"},
            {"（", "("},
            {"）", ")"},
            {"《", "("},
            {"》", ")"},
            {"【", "["},
            {"】", "]"},
            {"・", " "},
            {"，", ","},
            {"～", "—"},
            {"〜", "—"},
            {"—", "—"},
            {"«", "“"},
            {"»", "”"},
            {"゚", ""},
            {"゙", ""},
        };
        Table m;
        for (auto& e : kv) m[dec(e[0])] = dec(e[1]);
        return m;
    }();
    return t;
}
const std::u32string* table_get(const std::u32string& k) {
    const Table& t = hepburn();
    auto it = t.find(k);
    return it == t.end() ? nullptr : &it->second;
}
const std::u32string kSutegana = U"ゃゅょぁぃぅぇぉ";
const std::u32string kOdori = U"〃々ゝゞヽ";

char32_t add_dakuten(char32_t k) {
    static const std::u32string a = U"かきくけこさしすせそたちつてとはひふへほ", b = U"がぎぐげござじずぜぞだぢづでどばびぶべぼ";
    const size_t i = a.find(k);
    return i == std::u32string::npos ? 0 : b[i];
}

// cutlet._get_single_mapping; pk / nk are 0 when absent
std::u32string single_mapping(char32_t pk, char32_t kk, char32_t nk) {
    if (kOdori.find(kk) != std::u32string::npos) {
        if (kk == U'ゝ' || kk == U'ヽ') return pk ? std::u32string(1, pk) : U"";
        if (kk == U'ゞ') {
            if (!pk) return U"";
            const char32_t vv = add_dakuten(pk);
            if (vv) { const auto* r = table_get(std::u32string(1, vv)); return r ? *r : U""; }
            return U"";
        }
        return U"";
    }
    if (pk) { const auto* r = table_get(std::u32string{pk, kk}); if (r) return *r; }
    if (nk && table_get(std::u32string{kk, nk})) return U"";
    if (nk && kSutegana.find(nk) != std::u32string::npos) {
        if (kk == U'っ') return U"";
        const auto* a = table_get(std::u32string(1, kk));
        const auto* b = table_get(std::u32string(1, nk));
        std::u32string r = a ? *a : U"";   // (the package raises here for a kana it has no entry for)
        if (!r.empty()) r.pop_back();
        return r + (b ? *b : U"");
    }
    if (kSutegana.find(kk) != std::u32string::npos) return U"";
    if (kk == U'ー') return U"ː";
    if (kk == U'っ') return U"ʔ";
    if (kk == U'ん') {
        const auto* t = nk ? table_get(std::u32string(1, nk)) : nullptr;
        if (t && !t->empty()) {
            const char32_t f = (*t)[0];
            if (f == U'm' || f == U'p' || f == U'b') return U"m";
            if (f == U'k' || f == U'ɡ') return U"ŋ";
            if (f == U'ɲ' || f == U'ʨ' || f == U'ʥ') return U"ɲ";
            if (f == U'n' || f == U't' || f == U'd' || f == U'ɾ' || f == U'z') return U"n";
        }
        return U"ɴ";
    }
    const auto* r = table_get(std::u32string(1, kk));
    return r ? *r : U"";
}

// jaconv.kata2hira
std::u32string kata2hira(std::u32string s) {
    for (auto& c : s) if ((c >= 0x30A1 && c <= 0x30F6) || c == 0x30FD || c == 0x30FE) c -= 0x60;
    return s;
}

// the n-th comma-separated field of a feature text (fields may be "quoted"); false if there are fewer fields
bool feature_field(const std::string& f, int n, std::string* out) {
    int k = 0;
    size_t i = 0;
    while (true) {
        std::string v;
        if (i < f.size() && f[i] == '"') {
            ++i;
            while (i < f.size()) {
                if (f[i] == '"') { if (i + 1 < f.size() && f[i + 1] == '"') { v += '"'; i += 2; continue; } ++i; break; }
                v += f[i++];
            }
            while (i < f.size() && f[i] != ',') v += f[i++];
        } else {
            while (i < f.size() && f[i] != ',') v += f[i++];
        }
        if (k == n) { *out = v; return true; }
        if (i >= f.size()) return false;
        ++i;   // the comma
        ++k;
    }
}

struct Word {
    std::u32string surface, hira;
    int char_type;
};
struct Tok {
    std::u32string text;
    bool space;
};

bool is_ascii(const std::u32string& s) { for (char32_t c : s) if (c >= 0x80) return false; return true; }

// cutlet._romaji_word
std::u32string romaji_word(const Word& w) {
    if (is_ascii(w.surface)) return w.surface;
    if (w.char_type == 3) {
        std::u32string o;
        for (char32_t c : w.surface) { const auto* r = table_get(std::u32string(1, c)); if (r) o += *r; else o += c; }
        return o;
    }
    if (w.char_type != 6) return U"";
    std::u32string o;
    const std::u32string& h = w.hira;
    for (size_t i = 0; i < h.size(); ++i)
        o += single_mapping(i > 0 ? h[i - 1] : 0, h[i], i + 1 < h.size() ? h[i + 1] : 0);
    return o;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
JaFrontend::~JaFrontend() = default;

std::unique_ptr<JaFrontend> JaFrontend::load(const std::string& dir, std::string* err) {
    std::unique_ptr<JaFrontend> f(new JaFrontend());
    f->dict_ = ja::JaDictionary::load(dir, err);
    if (!f->dict_) return nullptr;
    const std::string base = dir.empty() || dir.back() == '/' ? dir : dir + "/";
    std::ifstream in(base + "ja_words.txt", std::ios::binary);
    if (!in) { *err = "cannot open " + base + "ja_words.txt"; return nullptr; }
    std::stringstream ss;
    ss << in.rdbuf();
    f->words_buf_ = ss.str();
    const std::string& b = f->words_buf_;
    for (size_t p = 0; p < b.size();) {
        size_t q = b.find('\n', p);
        if (q == std::string::npos) q = b.size();
        size_t a = p, e = q;   // line.strip() (the list is ASCII-spaced)
        while (a < e && (b[a] == ' ' || b[a] == '\t' || b[a] == '\r')) ++a;
        while (e > a && (b[e - 1] == ' ' || b[e - 1] == '\t' || b[e - 1] == '\r')) --e;
        f->words_.push_back({(uint32_t)a, (uint32_t)(e - a)});
        f->max_word_bytes_ = std::max(f->max_word_bytes_, e - a);
        p = q + 1;
    }
    std::sort(f->words_.begin(), f->words_.end(), [&b](const std::pair<uint32_t, uint32_t>& x, const std::pair<uint32_t, uint32_t>& y) {
        return b.compare(x.first, x.second, b, y.first, y.second) < 0;
    });
    if (f->words_.empty()) { *err = "ja_words.txt is empty"; return nullptr; }
    return f;
}

bool JaFrontend::is_word(const char* s, size_t n) const {
    if (n > max_word_bytes_) return false;
    const std::string& b = words_buf_;
    size_t lo = 0, hi = words_.size();
    while (lo < hi) {
        const size_t m = (lo + hi) / 2;
        const int c = b.compare(words_[m].first, words_[m].second, s, n);
        if (c < 0) lo = m + 1;
        else if (c > 0) hi = m;
        else return true;
    }
    return false;
}

size_t JaFrontend::memory_bytes() const { return words_buf_.size() + words_.size() * 8; }

static std::vector<Word> tag_words(const ja::JaDictionary& d, const std::u32string& norm) {
    std::vector<Word> out;
    for (auto& m : d.parse(enc(norm))) {
        Word w;
        w.surface = dec(m.surface);
        std::string r;
        // feature.pron (field 9) or feature.kana (field 17) or the surface; unknown words have 6 fields
        if (!(feature_field(m.feature, 9, &r) && !r.empty()) && !(feature_field(m.feature, 17, &r) && !r.empty())) r = m.surface;
        w.hira = kata2hira(dec(r));
        w.char_type = (m.char_type == 7 || !m.unknown) ? 6 : m.char_type;
        out.push_back(std::move(w));
    }
    return out;
}

std::string JaFrontend::debug_words(const std::string& text) const {
    const std::u32string norm = normalize_text(dec(text));
    std::string o = enc(norm) + "\n";
    for (auto& m : dict_->parse(enc(norm))) o += m.surface + "\t" + m.feature + "\t" + std::to_string(m.char_type) + (m.unknown ? "\tunk\n" : "\n");
    return o;
}

std::string JaFrontend::phonemize(const std::string& text_utf8, const std::unordered_set<char32_t>* vocab) const {
    const std::u32string text = dec(text_utf8);
    std::u32string ps;
    if (!text.empty()) {
        std::vector<Word> words = tag_words(*dict_, normalize_text(text));
        // group runs of words of one character type that together spell a word of the list (longest first)
        std::vector<Word> grouped;
        std::string cat;
        std::vector<size_t> ends;
        for (size_t i = 0; i < words.size();) {
            size_t z = i + 1;
            while (z < words.size() && words[z].char_type == words[i].char_type) ++z;
            cat.clear(); ends.clear();
            for (size_t k = i; k < z && cat.size() <= max_word_bytes_; ++k) { cat += enc(words[k].surface); ends.push_back(cat.size()); }
            size_t j = 0;   // (longer spans than the longest listed word cannot match: not built)
            for (size_t k = i + ends.size(); k > i; --k) if (is_word(cat.data(), ends[k - 1 - i])) { j = k; break; }
            if (!j) { grouped.push_back(words[i]); ++i; continue; }
            Word g{U"", U"", words[i].char_type};
            for (size_t k = i; k < j; ++k) { g.surface += words[k].surface; g.hira += words[k].hira; }
            grouped.push_back(std::move(g));
            i = j;
        }
        // tokens and spacing (cutlet._romaji_tokens)
        std::vector<Tok> out;
        for (auto& w : grouped) {
            // cutlet._romaji_word: `assert not surface.isdigit()` (a digit symbol the tagger made a word of)
            if (py_isdigit(w.surface))
                throw JaTextError("Japanese: a digit symbol (such as ❶) cannot be read (the Python package raises "
                                  "an error on it too)");
            Tok t{romaji_word(w), false};
            if (py_in(w.surface, U"「『«") || py_in(t.text, U"([")) {
                if (!out.empty()) out.back().space = true;
            } else if (py_in(w.surface, U"」』»") || py_in(t.text, U"]).,?!:")) {
                if (!out.empty()) out.back().space = false;
                t.space = true;
            } else if (t.text == U" ") {
                t.space = false;
            } else {
                t.space = true;
            }
            out.push_back(std::move(t));
        }
        std::u32string joined;
        for (auto& t : out) { replace_all(t.text, U"っ", U""); joined += t.text; if (t.space) joined += U' '; }
        // re.sub(r'\s+', ' ', out.strip()), then ( ) -> « »
        const std::u32string st = py_strip(joined);
        std::u32string col;
        for (size_t i = 0; i < st.size(); ++i) {
            if (py_space(st[i])) { col += U' '; while (i + 1 < st.size() && py_space(st[i + 1])) ++i; continue; }
            col += st[i] == U'(' ? U'«' : st[i] == U')' ? U'»' : st[i];
        }
        // re.sub(r'(?<![!",.:;?»—…”]) (?=ʔ)|(?<=ʔ) (?!["«“])', '', ps)
        static const std::u32string before = U"!\",.:;?»—…”", after = U"\"«“";
        for (size_t i = 0; i < col.size(); ++i) {
            if (col[i] == U' ') {
                const char32_t p = i > 0 ? col[i - 1] : 0, n = i + 1 < col.size() ? col[i + 1] : 0;
                const bool a = n == U'ʔ' && (i == 0 || before.find(p) == std::u32string::npos);
                const bool b = p == U'ʔ' && (n == 0 || after.find(n) == std::u32string::npos);
                if (a || b) continue;
            }
            ps += col[i];
        }
    }
    // phonemize_v2.phonemize: strip, FOLD (v1 then v2 additions); phonemize_lang: drop CJK_STRIP leftovers
    ps = py_strip(ps);
    static const std::pair<const char32_t*, const char32_t*> fold[] = {
        {U"̪", U""}, {U"̩", U""}, {U"̍", U""}, {U"̯", U""}, {U"ɭ", U"l"}, {U"ɫ", U"l"},
        {U"ɵ", U"ø"}, {U"ε", U"ɛ"}, {U"\x05", U""}, {U"\x06", U""}, {U"\x07", U""}, {U"\x08", U""},
        {U"{", U""}, {U"}", U""}, {U"[", U""}, {U"]", U""}, {U"¿", U""}, {U"¡", U""}, {U" ", U" "},
        {U" ", U" "}, {U"\t", U" "}, {U"«", U""}, {U"»", U""}, {U"‘", U""}, {U"’", U""}, {U"'", U""},
        {U"~", U""}, {U"·", U""}, {U"#", U""}, {U"g", U"ɡ"}};
    for (auto& kv : fold) replace_all(ps, kv.first, kv.second);
    static const std::u32string cjk_strip = U"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::u32string o;
    o.reserve(ps.size());
    for (char32_t c : ps) {
        if (cjk_strip.find(c) != std::u32string::npos) continue;
        if (vocab && !vocab->count(c)) continue;
        o += c;
    }
    return enc(o);
}

namespace {
// SENT_MULTI = (?<=[.!?…।])\s+|(?<=[。！？])
std::vector<std::u32string> split_sentences(const std::u32string& t) {
    std::vector<std::u32string> out;
    static const std::u32string a = U".!?…।", b = U"。！？";
    size_t start = 0;
    for (size_t i = 0; i < t.size(); i++) {
        if (i > 0 && py_space(t[i]) && a.find(t[i - 1]) != std::u32string::npos) {
            out.push_back(t.substr(start, i - start));
            size_t j = i;
            while (j < t.size() && py_space(t[j])) j++;
            start = j; i = j - 1;
            continue;
        }
        if (b.find(t[i]) != std::u32string::npos) { out.push_back(t.substr(start, i + 1 - start)); start = i + 1; }
    }
    out.push_back(t.substr(start));
    return out;
}
}  // namespace

std::vector<std::string> JaFrontend::chunks(const std::string& text, const std::unordered_set<char32_t>* vocab) const {
    const size_t MAX_PHON = 350;
    const std::string whole = phonemize(text, vocab);
    if (dec(whole).size() <= MAX_PHON) return whole.empty() ? std::vector<std::string>{} : std::vector<std::string>{whole};
    std::vector<std::string> out;
    std::u32string cur;
    for (auto& sent : split_sentences(dec(text))) {
        if (py_strip(sent).empty()) continue;
        std::u32string ps = dec(phonemize(enc(sent), vocab));
        std::vector<std::u32string> wrapped;
        while (ps.size() > MAX_PHON) {
            size_t cut = ps.rfind(U' ', MAX_PHON - 1);
            if (cut == std::u32string::npos || cut == 0) cut = MAX_PHON;
            wrapped.push_back(ps.substr(0, cut));
            ps = py_strip(ps.substr(cut));
        }
        if (!ps.empty()) wrapped.push_back(ps);
        for (auto& w : wrapped) {
            if (!cur.empty() && cur.size() + 1 + w.size() > MAX_PHON) { out.push_back(enc(cur)); cur = w; }
            else cur = py_strip(cur.empty() ? w : cur + U" " + w);
        }
    }
    if (!cur.empty()) out.push_back(enc(cur));
    return out;
}

}  // namespace kt_fe
