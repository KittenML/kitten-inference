// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// The Python package's non-English text -> phonemes path for Kitten TTS 2 (tts2_tiny/frontend), re-implemented:
//   phonemizer (preserve_punctuation=True, with_stress=True, tie='^', language_switch='remove-flags'): the text is
//     cut at punctuation marks, each chunk is phonemized alone (chunks joined only by quotes or brackets also as a
//     whole, used when the word counts agree), and the marks are put back;
//   misaki EspeakG2P: brackets <-> angle quotes around the call, the e2m table, ties and hyphens removed;
//   phonemize_v1 / phonemize_v2: Hindi danda, FOLD tables, Hindi flap rules.
#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontend/espeak_lang.h"

namespace kt_fe {
namespace {

void put(std::string& o, char32_t c) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
}
std::u32string dec(const std::string& s) {
    std::u32string o;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        char32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; } else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { i++; continue; }
        if (i + n > s.size()) break;
        for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        o += cp; i += n;
    }
    return o;
}
std::string enc(const std::u32string& s) { std::string o; for (char32_t c : s) put(o, c); return o; }

void replace_all(std::string& s, const std::string& a, const std::string& b) {
    if (a.empty()) return;
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) { s.replace(p, a.size(), b); p += b.size(); }
}
// Python str.strip() (whitespace as Python defines it, for the characters that occur here)
bool py_space(char32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x1f || c == 0x85 ||
           c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}
std::u32string strip(const std::u32string& s) {
    size_t a = 0, b = s.size();
    while (a < b && py_space(s[a])) a++;
    while (b > a && py_space(s[b - 1])) b--;
    return s.substr(a, b - a);
}
std::string strip(const std::string& s) { return enc(strip(dec(s))); }

const std::u32string kMarks = U";:!?¡¿—…\"«»“”(){}[]";   // + ',' '.' unless between digits
const std::u32string kInline = U"\"«»“”‘’„‹›()[]{}";

bool is_mark(const std::u32string& t, size_t i) {
    char32_t c = t[i];
    if (kMarks.find(c) != std::u32string::npos) return true;
    if (c == ',' || c == '.') {
        bool pd = i > 0 && t[i - 1] >= '0' && t[i - 1] <= '9';
        bool nd = i + 1 < t.size() && t[i + 1] >= '0' && t[i + 1] <= '9';
        return !(pd && nd);
    }
    return false;
}
// regex (\s*(?:mark)+\s*)+ at position i: returns end or i if no match
size_t match_marks(const std::u32string& t, size_t i) {
    size_t end = i;
    size_t j = i;
    while (true) {
        size_t k = j;
        while (k < t.size() && py_space(t[k])) k++;
        if (k >= t.size() || !is_mark(t, k)) break;
        while (k < t.size() && is_mark(t, k)) k++;
        while (k < t.size() && py_space(t[k])) k++;
        end = k; j = k;
    }
    return end;
}
bool is_inline_mark(const std::u32string& m) {
    bool any = false;
    for (char32_t c : m) {
        if (py_space(c)) continue;
        if (kInline.find(c) == std::u32string::npos) return false;
        any = true;
    }
    return any;
}

// espeak output of one chunk as phonemizer's _postprocess_line leaves it (words + trailing ' ', or "")
std::string espeak_line(EspeakLang& eng, const std::u32string& chunk) {
    std::string line = strip(eng.phonemize_clause(enc(chunk)));
    if (line.empty()) return "";
    std::string o;
    size_t p = 0;
    while (p <= line.size()) {
        size_t q = line.find(' ', p);
        if (q == std::string::npos) q = line.size();
        o += strip(line.substr(p, q - p));
        o += ' ';
        p = q + 1;
    }
    return o;
}
std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> w;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(' ', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) w.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return w;
}

std::string phonemizer_call(EspeakLang& eng, const std::u32string& text) {
    // preserve(): parts = chunk index or mark
    std::vector<std::u32string> chunks;
    std::vector<std::pair<int, std::u32string>> parts;   // (chunk index or -1, mark)
    size_t cursor = 0, i = 0;
    while (i < text.size()) {
        size_t e = match_marks(text, i);
        if (e > i) {
            if (i > cursor) { parts.push_back({(int)chunks.size(), U""}); chunks.push_back(text.substr(cursor, i - cursor)); }
            parts.push_back({-1, text.substr(i, e - i)});
            cursor = e; i = e;
        } else i++;
    }
    if (cursor < text.size()) { parts.push_back({(int)chunks.size(), U""}); chunks.push_back(text.substr(cursor)); }
    std::vector<std::string> ph(chunks.size());
    for (size_t k = 0; k < chunks.size(); k++) ph[k] = espeak_line(eng, chunks[k]);
    // inline groups
    std::vector<std::vector<int>> groups;
    {
        std::vector<int> g;
        for (size_t n = 0; n < parts.size(); n++) {
            if (parts[n].first < 0) continue;
            int part = parts[n].first;
            if (!g.empty() && n >= 2 && parts[n - 2].first == g.back() && is_inline_mark(parts[n - 1].second)) g.push_back(part);
            else { if (g.size() > 1) groups.push_back(g); g = {part}; }
        }
        if (g.size() > 1) groups.push_back(g);
    }
    std::vector<std::string> in_ctx(chunks.size());
    std::vector<bool> has_ctx(chunks.size(), false);
    for (auto& g : groups) {
        std::u32string joined;
        for (size_t k = 0; k < g.size(); k++) { if (k) joined += U" "; joined += chunks[g[k]]; }
        std::string gp = espeak_line(eng, joined);
        // _context: word counts must agree
        std::vector<size_t> counts;
        size_t sum = 0; bool ok = true;
        for (int c : g) { size_t n = split_words(ph[c]).size(); counts.push_back(n); sum += n; if (!n) ok = false; }
        if (!ok || split_words(gp).size() != sum) continue;
        // pieces = gp.split(' ')
        std::vector<std::string> pieces;
        { size_t p = 0; while (true) { size_t q = gp.find(' ', p); if (q == std::string::npos) { pieces.push_back(gp.substr(p)); break; } pieces.push_back(gp.substr(p, q - p)); p = q + 1; } }
        std::vector<size_t> starts; size_t acc = 0;
        for (size_t k = 0; k + 1 < counts.size(); k++) { acc += counts[k]; starts.push_back(acc); }
        std::vector<size_t> cuts = {0}; size_t nw = 0;
        for (size_t n = 0; n < pieces.size(); n++) {
            if (!pieces[n].empty()) {
                if (std::find(starts.begin(), starts.end(), nw) != starts.end()) cuts.push_back(n);
                nw++;
            }
        }
        cuts.push_back(pieces.size());
        for (size_t k = 0; k + 1 < cuts.size() && k < g.size(); k++) {
            std::string s;
            for (size_t n = cuts[k]; n < cuts[k + 1]; n++) { if (n > cuts[k]) s += ' '; s += pieces[n]; }
            in_ctx[g[k]] = s; has_ctx[g[k]] = true;
        }
    }
    // restore()
    std::string out;
    bool have_mark = false; std::string mark;
    for (auto& p : parts) {
        if (p.first < 0) {
            if (!out.empty() && out.back() == ' ') out.pop_back();
            std::u32string m = p.second;
            for (auto& c : m) if (c == ' ') c = ' ';
            mark = enc(m); have_mark = true;
            out += mark;
        } else {
            const std::string& c = has_ctx[p.first] ? in_ctx[p.first] : ph[p.first];
            if (!c.empty()) { out += c; have_mark = false; }
        }
    }
    if (have_mark && !(mark.size() && mark.back() == ' ')) out += ' ';
    return out;
}

}  // namespace

std::string tiny_lang_phonemize_one(EspeakLang& eng, const std::string& lang, int version, const std::string& text_in,
                                    const std::unordered_set<char32_t>* vocab) {
    std::string text = text_in;
    if (version == 2 && lang == "hi") { replace_all(text, "\xe0\xa5\xa4", "."); replace_all(text, "\xe0\xa5\xa5", "."); }
    std::string orig_text = text;
    // misaki EspeakG2P
    replace_all(text, "«", "\xe2\x80\x9c"); replace_all(text, "»", "\xe2\x80\x9d");
    replace_all(text, "(", "«"); replace_all(text, ")", "»");
    std::string ps = strip(phonemizer_call(eng, dec(text)));
    static const std::pair<const char*, const char*> e2m[] = {   // sorted(e2m.items())
        {"a^ɪ", "I"}, {"a^ʊ", "W"}, {"d^z", "ʣ"}, {"d^ʒ", "ʤ"}, {"e^ɪ", "A"}, {"o^ʊ", "O"}, {"s^s", "S"},
        {"t^s", "ʦ"}, {"t^ʃ", "ʧ"}, {"ɔ^ɪ", "Y"}, {"ə^ʊ", "Q"}};
    for (auto& kv : e2m) replace_all(ps, kv.first, kv.second);
    replace_all(ps, "^", "");
    replace_all(ps, "-", "");
    replace_all(ps, "«", "("); replace_all(ps, "»", ")");
    // phonemize_v1 / v2: strip + FOLD
    ps = strip(ps);
    static const std::pair<const char*, const char*> fold1[] = {
        {"̪", ""}, {"̩", ""}, {"̍", ""}, {"̯", ""}, {"ɭ", "l"}, {"ɫ", "l"},
        {"ɵ", "ø"}, {"ε", "ɛ"}, {"\x05", ""}, {"\x06", ""}, {"\x07", ""}, {"\x08", ""},
        {"{", ""}, {"}", ""}, {"[", ""}, {"]", ""}, {"¿", ""}, {"¡", ""}, {" ", " "}};
    static const std::pair<const char*, const char*> fold2[] = {
        {" ", " "}, {"\t", " "}, {"«", ""}, {"»", ""}, {"‘", ""}, {"’", ""}, {"'", ""}, {"~", ""},
        {"·", ""}, {"#", ""}, {"g", "ɡ"}};
    for (auto& kv : fold1) replace_all(ps, kv.first, kv.second);
    if (version == 2) for (auto& kv : fold2) replace_all(ps, kv.first, kv.second);
    if (version == 2 && lang == "hi") {
        replace_all(ps, "r.h", "ɽʰ");
        // r\.(?=[^\s.,?!;:…])  -> ɽ
        std::u32string u = dec(ps), o;
        static const std::u32string stop = U".,?!;:…";
        for (size_t i = 0; i < u.size(); i++) {
            if (u[i] == 'r' && i + 2 < u.size() + 0 && u[i + 1] == '.' && i + 2 < u.size() && !py_space(u[i + 2]) &&
                stop.find(u[i + 2]) == std::u32string::npos) { o += U"ɽ"; i++; continue; }
            o += u[i];
        }
        ps = enc(o);
        std::u32string t = dec(orig_text);
        bool flap = false;
        for (size_t i = 0; i < t.size(); i++) {
            if (t[i] == 0x95C || t[i] == 0x95D) flap = true;
            if ((t[i] == 0x921 || t[i] == 0x922) && i + 1 < t.size() && t[i + 1] == 0x93C) flap = true;
        }
        if (flap) replace_all(ps, "r.", "ɽ");
    }
    if (vocab) {
        std::u32string u = dec(ps), o;
        for (char32_t c : u) if (vocab->count(c)) o += c;
        ps = enc(o);
    }
    return ps;
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

// Python: Frontend.__call__ for a non-English language -> chunks of at most MAX_PHON (350) phonemes
std::vector<std::string> tiny_lang_chunks(EspeakLang& eng, const std::string& lang, int version, const std::string& text,
                                          const std::unordered_set<char32_t>* vocab) {
    const size_t MAX_PHON = 350;
    std::string whole = tiny_lang_phonemize_one(eng, lang, version, text, vocab);
    if (dec(whole).size() <= MAX_PHON) return whole.empty() ? std::vector<std::string>{} : std::vector<std::string>{whole};
    std::vector<std::string> chunks;
    std::u32string cur;
    for (auto& sent : split_sentences(dec(text))) {
        if (strip(sent).empty()) continue;
        std::u32string ps = dec(tiny_lang_phonemize_one(eng, lang, version, enc(sent), vocab));
        std::vector<std::u32string> wrapped;
        while (ps.size() > MAX_PHON) {
            size_t cut = ps.rfind(U' ', MAX_PHON - 1);
            if (cut == std::u32string::npos || cut == 0) cut = MAX_PHON;
            wrapped.push_back(ps.substr(0, cut));
            ps = strip(ps.substr(cut));
        }
        if (!ps.empty()) wrapped.push_back(ps);
        for (auto& w : wrapped) {
            if (!cur.empty() && cur.size() + 1 + w.size() > MAX_PHON) { chunks.push_back(enc(cur)); cur = w; }
            else cur = strip(cur.empty() ? w : cur + U" " + w);
        }
    }
    if (!cur.empty()) chunks.push_back(enc(cur));
    return chunks;
}

const char* tiny_lang_voice(const std::string& lang) {
    static const char* const m[][2] = {{"es", "es"}, {"fr", "fr-fr"}, {"it", "it"}, {"pt", "pt-br"}, {"hi", "hi"},
                                       {"de", "de"}, {"nl", "nl"}, {"ru", "ru"}, {"tr", "tr"}, {"sv", "sv"},
                                       {"da", "da"}, {"fi", "fi"}, {"sw", "sw"}, {"ko", "ko"}};
    for (auto& kv : m) if (lang == kv[0]) return kv[1];
    return "";
}

std::string tiny_lang_phonemize(EspeakLang& eng, const std::string& lang, int version, const std::string& text) {
    return tiny_lang_phonemize_one(eng, lang, version, text, nullptr);
}

}  // namespace kt_fe
