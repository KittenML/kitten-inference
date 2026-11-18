// Part of kitten-inference (Apache-2.0).
// Small Unicode helpers for the Kitten TTS 2 English text front end: UTF-8 <-> code points and the few character
// classes / case mappings Python's str methods give the reference front end. Covers ASCII, Latin-1, Latin
// Extended-A, Greek and Cyrillic; other scripts are treated as uncased letters (the reference reports such text as
// unsayable anyway).
#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace kt_tinyen {

using u32 = std::u32string;
using u32v = std::u32string_view;

inline u32 utf8_to_u32(std::string_view s) {
    u32 out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        char32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; len = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + len > n) { out.push_back(0xFFFD); break; }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            unsigned char cc = (unsigned char)s[i + k];
            if ((cc >> 6) != 2) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

inline void append_utf8(std::string& o, char32_t cp) {
    if (cp < 0x80) o.push_back((char)cp);
    else if (cp < 0x800) { o.push_back((char)(0xC0 | (cp >> 6))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) {
        o.push_back((char)(0xE0 | (cp >> 12))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        o.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        o.push_back((char)(0xF0 | (cp >> 18))); o.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

inline std::string u32_to_utf8(u32v s) {
    std::string o;
    o.reserve(s.size() * 2);
    for (char32_t c : s) append_utf8(o, c);
    return o;
}

// ---- case ----
inline char32_t to_lower(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if ((c >= 0xC0 && c <= 0xDE && c != 0xD7)) return c + 32;
    if (c >= 0x100 && c <= 0x17F) {
        if (c == 0x130) return 'i';
        if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c + 1 : c;
        if (c == 0x178) return 0xFF;
        if (c == 0x131 || c == 0x138 || c == 0x149 || c == 0x17F) return c;
        return (c & 1) ? c : c + 1;
    }
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}
inline char32_t to_upper(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') ? c - 32 : c;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c == 0xFF) return 0x178;
    if (c >= 0x100 && c <= 0x17F) {
        if (c == 0x131) return 'I';
        if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c : c - 1;
        if (c == 0x130 || c == 0x138 || c == 0x149 || c == 0x178) return c;
        if (c == 0x17F) return 'S';
        return (c & 1) ? c - 1 : c;
    }
    if (c >= 0x3B1 && c <= 0x3CB && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    return c;
}
inline bool is_upper(char32_t c) { return to_lower(c) != c; }
inline bool is_lower(char32_t c) {
    if (c == 0xDF || c == 0x138 || c == 0x149) return true;
    return to_upper(c) != c;
}
inline bool is_cased(char32_t c) { return is_upper(c) || is_lower(c); }

inline bool is_alpha(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;
    if (c >= 0x250 && c <= 0x2C1) return true;           // IPA, modifier letters
    if (c >= 0x2C6 && c <= 0x2D1) return true;
    if (c >= 0x2E0 && c <= 0x2E4) return true;
    if (c >= 0x370 && c <= 0x3FF) return c != 0x37E && c != 0x387 && c != 0x375 && c != 0x384 && c != 0x385;
    if (c >= 0x400 && c <= 0x52F) return !(c >= 0x482 && c <= 0x489);
    if (c >= 0x5D0 && c <= 0x5EA) return true;
    if (c >= 0x620 && c <= 0x64A) return true;
    if (c >= 0x904 && c <= 0x939) return true;
    if (c >= 0x1D00 && c <= 0x1DBF) return true;          // phonetic extensions (ᵊ ᵻ)
    if (c >= 0x1E00 && c <= 0x1FFF) return true;
    if (c >= 0x3040 && c <= 0x30FF) return c != 0x30FB && c != 0x30A0;
    if (c >= 0x4E00 && c <= 0x9FFF) return true;
    if (c >= 0xAC00 && c <= 0xD7A3) return true;
    return false;
}
inline bool is_ascii_digit(char32_t c) { return c >= '0' && c <= '9'; }
// Python str.isdigit for the characters the front end can meet (ASCII, superscripts, other decimal digits).
inline bool is_digit_py(char32_t c) {
    if (is_ascii_digit(c)) return true;
    if (c == 0xB2 || c == 0xB3 || c == 0xB9) return true;
    if (c >= 0x2070 && c <= 0x2079 && c != 0x2071 && c != 0x2072 && c != 0x2073) return true;
    if (c >= 0x2080 && c <= 0x2089) return true;
    if (c >= 0x660 && c <= 0x669) return true;
    if (c >= 0x966 && c <= 0x96F) return true;
    if (c >= 0xFF10 && c <= 0xFF19) return true;
    return false;
}
inline bool is_space(char32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F || c == 0x85 ||
           c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}
// Python's \w (str patterns): letters, digits, underscore (approximation: also combining marks).
inline bool is_word(char32_t c) { return c == '_' || is_alpha(c) || is_digit_py(c) || (c >= 0x300 && c <= 0x36F); }
inline bool is_combining(char32_t c) {
    return (c >= 0x300 && c <= 0x36F) || (c >= 0x1AB0 && c <= 0x1AFF) || (c >= 0x1DC0 && c <= 0x1DFF) ||
           (c >= 0x20D0 && c <= 0x20FF) || (c >= 0xFE20 && c <= 0xFE2F);
}

inline u32 lower(u32v s) { u32 o(s); for (auto& c : o) c = to_lower(c); return o; }
inline u32 upper(u32v s) { u32 o(s); for (auto& c : o) c = to_upper(c); return o; }
// str.capitalize(): first character upper (title), rest lower.
inline u32 capitalize(u32v s) {
    u32 o = lower(s);
    if (!o.empty()) o[0] = to_upper(o[0]);
    return o;
}
// str.isalpha(): non-empty and every character a letter.
inline bool all_alpha(u32v s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (!is_alpha(c)) return false;
    return true;
}
// str.strip() with Unicode whitespace
inline u32 strip(u32v s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    return u32(s.substr(a, b - a));
}
inline u32 strip_chars(u32v s, u32v chars) {
    size_t a = 0, b = s.size();
    while (a < b && chars.find(s[a]) != u32v::npos) ++a;
    while (b > a && chars.find(s[b - 1]) != u32v::npos) --b;
    return u32(s.substr(a, b - a));
}
inline bool starts_with(u32v s, u32v p) { return s.size() >= p.size() && s.substr(0, p.size()) == p; }
inline bool ends_with(u32v s, u32v p) { return s.size() >= p.size() && s.substr(s.size() - p.size()) == p; }
inline bool ends_with(u32v s, const char* p) {
    size_t n = 0; while (p[n]) ++n;
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) if (s[s.size() - n + i] != (char32_t)(unsigned char)p[i]) return false;
    return true;
}
inline u32 A(const char* ascii_or_utf8) { return utf8_to_u32(ascii_or_utf8); }
inline bool eq(u32v s, const char* p) {
    size_t i = 0;
    for (; p[i]; ++i) { if (i >= s.size() || s[i] != (char32_t)(unsigned char)p[i]) return false; }
    return i == s.size();
}
inline void replace_all(u32& s, u32v from, u32v to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != u32::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
}

}  // namespace kt_tinyen
