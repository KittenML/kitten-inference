// Part of kitten-inference (Apache-2.0). See tiny_en.hpp.
//
// Port of misaki 0.9.4 en.py (Apache-2.0, https://github.com/hexgrad/misaki) as vendored in
// src/kitten_inference/tts2_tiny/frontend/misaki/en.py, plus english.py's Fallback (CMUdict, then the BART) and
// __init__.py's chunking. Function names follow the Python so the two can be read side by side.
#include "tiny_en.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "bart.hpp"
#include "json_lite.hpp"
#include "num_words.hpp"
#include "tts_norm.hpp"

namespace kt_tinyen {
namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Lexicon storage: every key and value in one UTF-8 arena, entries sorted by key (binary search). Parsed once.
// ---------------------------------------------------------------------------------------------------------------
struct LexStore {
    static constexpr uint32_t NUL = 0xFFFFFFFFu;
    struct E { uint32_t ko, kl, vo, vl; int32_t dict; };
    struct DictE { uint32_t to, tl, vo, vl; };
    std::string arena;
    std::vector<E> e;
    std::vector<std::vector<DictE>> dicts;

    uint32_t put(const std::string& s) { uint32_t o = (uint32_t)arena.size(); arena += s; return o; }
    std::string_view key(const E& x) const { return std::string_view(arena).substr(x.ko, x.kl); }

    bool load(const std::string& path, bool allow_dicts, std::string& err) {
        std::string text;
        if (!read_file(path, text)) { err = "cannot read " + path; return false; }
        arena.reserve(text.size());
        json::Reader r(text.data(), text.size());
        if (!r.eat('{')) { err = path + ": not a JSON object"; return false; }
        std::string k, v;
        if (!r.eat('}')) {
            for (;;) {
                if (!r.string(k) || !r.eat(':')) { err = path + ": bad entry"; return false; }
                E x{put(k), (uint32_t)k.size(), NUL, 0, -1};
                if (r.peek('"')) {
                    if (!r.string(v)) { err = path + ": bad value"; return false; }
                    x.vo = put(v); x.vl = (uint32_t)v.size();
                } else if (r.lit("null")) {
                } else if (allow_dicts && r.eat('{')) {
                    std::vector<DictE> d;
                    if (!r.eat('}')) {
                        for (;;) {
                            std::string t;
                            if (!r.string(t) || !r.eat(':')) { err = path + ": bad tag entry"; return false; }
                            DictE de{put(t), (uint32_t)t.size(), NUL, 0};
                            if (r.peek('"')) {
                                if (!r.string(v)) { err = path + ": bad value"; return false; }
                                de.vo = put(v); de.vl = (uint32_t)v.size();
                            } else if (!r.lit("null")) { err = path + ": bad tag value"; return false; }
                            d.push_back(de);
                            if (r.eat(',')) continue;
                            if (r.eat('}')) break;
                            err = path + ": bad tag object"; return false;
                        }
                    }
                    x.dict = (int32_t)dicts.size();
                    dicts.push_back(std::move(d));
                } else { err = path + ": unexpected value"; return false; }
                e.push_back(x);
                if (r.eat(',')) continue;
                if (r.eat('}')) break;
                err = path + ": bad object"; return false;
            }
        }
        std::stable_sort(e.begin(), e.end(), [&](const E& a, const E& b) { return key(a) < key(b); });
        arena.shrink_to_fit();
        return true;
    }
    // last entry with this key (JSON: a repeated key keeps its last value)
    const E* find(std::string_view k) const {
        auto it = std::upper_bound(e.begin(), e.end(), k, [&](std::string_view a, const E& b) { return a < key(b); });
        if (it == e.begin()) return nullptr;
        --it;
        return key(*it) == k ? &*it : nullptr;
    }
    size_t bytes() const {
        size_t n = arena.capacity() + e.capacity() * sizeof(E);
        for (auto& d : dicts) n += d.capacity() * sizeof(DictE) + sizeof(d);
        return n;
    }
};

// A looked-up lexicon value: None, a phoneme string, or a tag -> phonemes table.
struct LexVal {
    const LexStore* st = nullptr;
    const LexStore::E* e = nullptr;
    bool found() const { return e != nullptr; }
    bool is_dict() const { return e && e->dict >= 0; }
    bool is_null() const { return e && e->dict < 0 && e->vo == LexStore::NUL; }
    u32 str() const { return utf8_to_u32(std::string_view(st->arena).substr(e->vo, e->vl)); }
    // dict access: returns false when the tag is absent; *null set when the value is None
    bool dict_get(std::string_view tag, u32& out, bool& null) const {
        for (auto& d : st->dicts[e->dict]) {
            if (std::string_view(st->arena).substr(d.to, d.tl) == tag) {
                null = d.vo == LexStore::NUL;
                if (!null) out = utf8_to_u32(std::string_view(st->arena).substr(d.vo, d.vl));
                return true;
            }
        }
        return false;
    }
};

// Lexicon.grow_dictionary without building the grown table: a lowercase key is also found Capitalised and a
// Capitalised key also lowercase (keys of 2+ characters; the original entry wins).
LexVal grown_get(const LexStore& st, const u32& w) {
    LexVal v{&st, st.find(u32_to_utf8(w))};
    if (v.e) return v;
    if (w.size() < 2) return v;
    u32 lw = lower(w);
    if (w == lw) {
        u32 c = capitalize(w);
        if (c != w) {  // e[k.lower()] from a Capitalised key k == k.lower().capitalize()
            const LexStore::E* x = st.find(u32_to_utf8(c));
            if (x) return LexVal{&st, x};
        }
        return v;
    }
    if (w == capitalize(lw)) {  // e[k.capitalize()] from a lowercase key k (k != k.capitalize())
        const LexStore::E* x = st.find(u32_to_utf8(lw));
        if (x) return LexVal{&st, x};
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------
// misaki constants
// ---------------------------------------------------------------------------------------------------------------
constexpr char32_t PRIMARY = 0x2C8, SECONDARY = 0x2CC;
const u32 VOWELS_S = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const u32 CONSONANTS_S = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const u32 US_TAUS_S = U"AIOWYiuæɑəɛɪɹʊʌ";
const u32 DIPHTHONGS_S = U"AIOQWYʤʧ";
const u32 PUNCTS_S = U";:,.!?—…\"“”";
const u32 NON_QUOTE_PUNCTS_S = U";:,.!?—…";
const u32 SUBTOKEN_JUNKS_S = U"',-._‘’/";

inline bool in(const u32& set, char32_t c) { return set.find(c) != u32::npos; }
inline bool is_vowel(char32_t c) { return in(VOWELS_S, c); }
inline bool has_any_stress(const u32& s) { return s.find(PRIMARY) != u32::npos || s.find(SECONDARY) != u32::npos; }
inline bool has_vowel(const u32& s) { for (char32_t c : s) if (is_vowel(c)) return true; return false; }
inline bool lexicon_ord(char32_t c) { return c == 39 || c == 45 || (c >= 65 && c <= 90) || (c >= 97 && c <= 122); }
inline bool all_lexicon_ords(const u32& w) { for (char32_t c : w) if (!lexicon_ord(c)) return false; return true; }
inline bool is_digits(const u32& w) {  // re.match(r'^[0-9]+$')
    if (w.empty()) return false;
    for (char32_t c : w) if (!is_ascii_digit(c)) return false;
    return true;
}
inline bool is_ordinal_suffix(const u32& s) { return eq(s, "st") || eq(s, "nd") || eq(s, "rd") || eq(s, "th"); }

constexpr double NONE = 1e300;  // Python None (no NaN: safe under any float flags)
inline bool is_none(double x) { return x == NONE; }

u32 restress(const u32& ps) {
    // a single stress mark at index 0 moves to just before the first vowel after it
    u32 out;
    size_t j = 1;
    while (j < ps.size() && !is_vowel(ps[j])) ++j;
    if (j >= ps.size()) return ps;
    out.append(ps, 1, j - 1);
    out += ps[0];
    out.append(ps, j, u32::npos);
    return out;
}

void remove_char(u32& s, char32_t c) { s.erase(std::remove(s.begin(), s.end(), c), s.end()); }
void replace_char(u32& s, char32_t a, char32_t b) { for (auto& c : s) if (c == a) c = b; }

u32 apply_stress(u32 ps, double stress) {
    if (is_none(stress)) return ps;
    bool has_p = ps.find(PRIMARY) != u32::npos;
    if (stress < -1) { remove_char(ps, PRIMARY); remove_char(ps, SECONDARY); return ps; }
    if (stress == -1 || ((stress == 0 || stress == -0.5) && has_p)) {
        remove_char(ps, SECONDARY); replace_char(ps, PRIMARY, SECONDARY); return ps;
    }
    if ((stress == 0 || stress == 0.5 || stress == 1) && !has_any_stress(ps)) {
        if (!has_vowel(ps)) return ps;
        return restress(u32(1, SECONDARY) + ps);
    }
    if (stress >= 1 && !has_p && ps.find(SECONDARY) != u32::npos) { replace_char(ps, SECONDARY, PRIMARY); return ps; }
    if (stress > 1 && !has_any_stress(ps)) {
        if (!has_vowel(ps)) return ps;
        return restress(u32(1, PRIMARY) + ps);
    }
    return ps;
}

int stress_weight(const u32& ps) { int n = 0; for (char32_t c : ps) n += in(DIPHTHONGS_S, c) ? 2 : 1; return n; }

std::string parent_tag(const std::string& tag) {
    if (tag.rfind("VB", 0) == 0) return "VERB";
    if (tag.rfind("NN", 0) == 0) return "NOUN";
    if (tag.rfind("ADV", 0) == 0 || tag.rfind("RB", 0) == 0) return "ADV";
    if (tag.rfind("ADJ", 0) == 0 || tag.rfind("JJ", 0) == 0) return "ADJ";
    return tag;
}

struct Ctx { int future_vowel = -1; /* -1 None, 0 False, 1 True */ bool future_to = false; };

struct Res { bool ok = false; u32 ps; int rating = 0; };
inline Res none() { return Res{}; }
inline Res mk(u32 ps, int r) { return Res{true, std::move(ps), r}; }

// ---------------------------------------------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------------------------------------------
struct MTok {
    u32 text;
    std::string tag;
    u32 ws;
    bool has_ph = false;
    u32 ph;
    bool has_alias = false;
    u32 alias;
    bool is_head = true;
    char32_t currency = 0;
    bool prespace = false;
    int rating = -1;
};

struct Word {  // a single token, or a group of tokens written without spaces between them
    bool is_list = false;
    std::vector<MTok> t;
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
struct EnglishFrontend::Impl {
    LexStore gold, silver, cmu;
    std::unordered_map<std::string, std::vector<std::string>> specials;
    std::unordered_set<char32_t> vocab;   // v1 symbols (English filters with v1, as in training)
    bool v2 = false;
    Bart bart;
    bool bart_loaded = false;
    std::string bart_dir;
    std::unordered_map<char32_t, int> g2i;
    std::unordered_map<int, char32_t> i2p;
    std::unordered_map<std::string, u32> bart_memo;
    std::unordered_map<std::string, unsigned> homographs_nv;  // lowercase word -> tag keys of its gold entry (K_*)
    Timing timing;
    std::vector<int> ids_buf, gen_buf;

    // ======================= spaCy tokenizer (English rules) =======================
    static bool is_quote(char32_t c) {
        static const u32 Q = U"'\"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉〈〉⟦⟧";
        return in(Q, c);
    }
    static bool is_list_punct1(char32_t c) {
        static const u32 P = U"…,:;!?¿؟¡()[]{}<>_#*&。？！，、；：～·।،۔؛٪";
        return in(P, c);
    }
    static bool is_currency1(char32_t c) {
        static const u32 C = U"$£€¥฿₽﷼₴₠₡₢₣₤₥₦₧₨₩₪₫€₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿";
        return in(C, c);
    }
    bool is_special(const u32& s) const { return specials.count(u32_to_utf8(s)) > 0; }

    // length of the prefix spaCy's prefix_search finds at the start of s (0: none)
    static size_t find_prefix(const u32& s) {
        if (s.empty()) return 0;
        char32_t c = s[0];
        if (c == U'§' || c == U'%' || c == U'=' || c == U'—' || c == U'–') return 1;
        if (c == U'+' && !(s.size() > 1 && is_ascii_digit(s[1]))) return 1;
        if (c == U'…') return 1;
        if (is_list_punct1(c)) return 1;
        if (c == U'.' && s.size() > 1 && s[1] == U'.') { size_t n = 0; while (n < s.size() && s[n] == U'.') ++n; return n; }
        if (is_quote(c)) return 1;
        if (is_currency1(c)) return 1;
        if (starts_with(s, U"US$")) return 3;
        if (starts_with(s, U"C$") || starts_with(s, U"A$")) return 2;
        return 0;
    }

    static bool suffix_at(const u32& s, size_t st) {
        size_t n = s.size() - st;
        u32v t(s.data() + st, n);
        char32_t prev = st > 0 ? s[st - 1] : 0;
        if (n == 1) {
            char32_t c = t[0];
            if (is_list_punct1(c) || c == U'…') return true;
            if (is_quote(c)) return true;
            if (c == U'—' || c == U'–') return true;
            if (c == U'+' && is_ascii_digit(prev)) return true;
            if (is_currency1(c) && is_ascii_digit(prev)) return true;
            if (c == U'%' && is_ascii_digit(prev)) return true;
            if (c == U'.') {
                if (st >= 2 && s[st - 2] == U'°' && in(U"FfCcKk", prev)) return true;
                if (is_ascii_digit(prev) || is_lower(prev) || in(U"%²-+|():?", prev) || is_list_punct1(prev) ||
                    is_quote(prev) || prev == U'…')
                    return true;
                if (st >= 2 && is_upper(prev) && is_upper(s[st - 2])) return true;
            }
            if ((c == U'm' || c == U'g' || c == U't' || c == U'T' || c == U'G' || c == U'M' || c == U'K') && is_ascii_digit(prev)) return true;
            return false;
        }
        if (n == 2 && t[0] == U'…' && t[1] == U'…') return true;
        bool dots = n >= 2;
        for (char32_t c : t) if (c != U'.') { dots = false; break; }
        if (dots) return true;
        if (n == 2 && (t[0] == U'\'' || t[0] == U'’') && (t[1] == U's' || t[1] == U'S')) return true;
        if (is_ascii_digit(prev)) {
            static const char* units[] = {"km", "km²", "km³", "m²", "m³", "dm", "dm²", "dm³", "cm", "cm²", "cm³",
                                          "mm", "mm²", "mm³", "ha", "µm", "nm", "yd", "in", "ft", "kg", "mg", "µg",
                                          "lb", "oz", "m/s", "km/h", "kmh", "mph", "hPa", "Pa", "mbar", "mb", "MB",
                                          "kb", "KB", "gb", "GB", "tb", "TB", "US$", "C$", "A$"};
            std::string u8 = u32_to_utf8(t);
            for (auto* u : units) if (u8 == u) return true;
        }
        return false;
    }
    static size_t find_suffix(const u32& s) {
        for (size_t st = 0; st < s.size(); ++st) if (suffix_at(s, st)) return s.size() - st;
        return 0;
    }

    // spaCy English infixes, finditer order
    static void find_infixes(const u32& s, std::vector<std::pair<size_t, size_t>>& out) {
        out.clear();
        size_t i = 0, n = s.size();
        while (i < n) {
            char32_t c = s[i];
            char32_t prev = i > 0 ? s[i - 1] : 0;
            char32_t next = i + 1 < n ? s[i + 1] : 0;
            size_t len = 0;
            if (c == U'.' && next == U'.') { len = 2; while (i + len < n && s[i + len] == U'.') ++len; }
            else if (c == U'…') len = 1;
            else if ((c == U'+' || c == U'-' || c == U'*' || c == U'^') && is_ascii_digit(prev) && (is_ascii_digit(next) || next == U'-')) len = 1;
            else if (c == U'.' && i > 0 && (is_lower(prev) || is_quote(prev)) && (is_upper(next) || is_quote(next)) && next) len = 1;
            else if (c == U',' && i > 0 && is_alpha(prev) && is_alpha(next)) len = 1;
            else if (i > 0 && (is_alpha(prev) || is_ascii_digit(prev))) {
                // (?:-|–|—|--|---|——|~)(?=[alpha])
                static const u32 H[] = {U"-", U"–", U"—", U"--", U"---", U"——", U"~"};
                for (auto& h : H) {
                    if (s.compare(i, h.size(), h) == 0 && i + h.size() < n && is_alpha(s[i + h.size()])) { len = h.size(); break; }
                }
                if (!len && in(U":<>=/", c) && is_alpha(next)) len = 1;
            }
            if (len) { out.push_back({i, i + len}); i += len; }
            else ++i;
        }
    }

    void push_special_or(const u32& s, std::vector<u32>& toks) const {
        auto it = specials.find(u32_to_utf8(s));
        if (it != specials.end()) { for (auto& p : it->second) toks.push_back(utf8_to_u32(p)); return; }
        toks.push_back(s);
    }

    void tokenize_span(u32 str, std::vector<u32>& toks) const {
        if (is_special(str)) { push_special_or(str, toks); return; }
        std::vector<u32> prefixes, suffixes;
        size_t last = (size_t)-1;
        while (!str.empty() && str.size() != last) {
            if (is_special(str)) break;
            last = str.size();
            size_t pre = find_prefix(str);
            u32 prefix, minus_pre;
            if (pre) {
                prefix = str.substr(0, pre);
                minus_pre = str.substr(pre);
                if (!minus_pre.empty() && is_special(minus_pre)) { str = minus_pre; prefixes.push_back(prefix); break; }
            }
            u32 rest = str.substr(pre);
            size_t suf = find_suffix(rest);
            u32 suffix, minus_suf;
            if (suf) {
                suffix = str.substr(str.size() - suf);
                minus_suf = str.substr(0, str.size() - suf);
                if (!minus_suf.empty() && is_special(minus_suf)) { str = minus_suf; suffixes.push_back(suffix); break; }
            }
            if (pre && suf && pre + suf <= str.size()) {
                str = str.substr(pre, str.size() - pre - suf);
                prefixes.push_back(prefix);
                suffixes.push_back(suffix);
            } else if (pre) {
                str = minus_pre;
                prefixes.push_back(prefix);
            } else if (suf) {
                str = minus_suf;
                suffixes.push_back(suffix);
            }
            if (!str.empty() && is_special(str)) break;
        }
        for (auto& p : prefixes) toks.push_back(p);
        if (!str.empty()) {
            if (is_special(str)) push_special_or(str, toks);
            else {
                std::vector<std::pair<size_t, size_t>> m;
                find_infixes(str, m);
                size_t start = 0;
                for (auto& [a, b] : m) {
                    if (a == 0) continue;
                    if (a != start) toks.push_back(str.substr(start, a - start));
                    if (a != b) toks.push_back(str.substr(a, b - a));
                    start = b;
                }
                if (start < str.size()) toks.push_back(str.substr(start));
            }
        }
        for (auto it = suffixes.rbegin(); it != suffixes.rend(); ++it) toks.push_back(*it);
    }

    std::vector<SpTok> spacy_tokenize(const u32& text) const {
        std::vector<SpTok> out;
        size_t i = 0, n = text.size();
        std::vector<u32> toks;
        while (i < n) {
            if (text[i] == U' ') {  // only reached for a leading space or a run: spaCy makes it a token
                size_t j = i;
                while (j < n && text[j] == U' ') ++j;
                if (!out.empty() && !out.back().ws) { out.back().ws = true; ++i; continue; }
                SpTok t; t.text = text.substr(i, j - i - (j < n ? 1 : 0)); t.ws = j < n;
                if (t.text.empty()) { i = j; continue; }
                t.tag = "_SP"; out.push_back(t); i = j; continue;
            }
            size_t j = i;
            while (j < n && !is_space(text[j])) ++j;
            toks.clear();
            tokenize_span(text.substr(i, j - i), toks);
            for (auto& t : toks) out.push_back(SpTok{t, "", false});
            if (j < n && text[j] == U' ') { if (!out.empty()) out.back().ws = true; i = j + 1; }
            else if (j < n) {  // other whitespace: its own token
                size_t k = j;
                while (k < n && is_space(text[k]) && text[k] != U' ') ++k;
                out.push_back(SpTok{text.substr(j, k - j), "_SP", false});
                i = k;
                if (i < n && text[i] == U' ') { out.back().ws = true; ++i; }
            } else i = j;
        }
        return out;
    }

    // ======================= rule tagger (replaces spaCy's statistical tagger) =======================
    static bool in_list(const u32& w, std::initializer_list<const char*> l) {
        for (auto* s : l) if (eq(w, s)) return true;
        return false;
    }
    static bool is_punct_tok(const u32& t) {
        for (char32_t c : t) if (is_alpha(c) || is_ascii_digit(c)) return false;
        return !t.empty();
    }
    // Function-word classes the rules look at (lowercase).
    static bool is_det(const u32& w) { return in_list(w, {"a", "an", "the", "this", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "no", "every", "each", "another", "whose", "what", "which", "one", "two", "three", "first", "last", "next", "own", "same", "such", "'s", "s", "all", "both", "many", "few", "several", "much", "great", "good", "little", "new", "old", "long", "whole", "other"}); }
    static bool is_prep(const u32& w) { return in_list(w, {"of", "for", "with", "on", "at", "from", "into", "about", "over", "under", "after", "before", "through", "between", "against", "during", "without", "within", "upon", "towards", "toward", "among", "across", "behind", "beyond", "beneath", "beside", "besides", "since", "until", "till", "near", "onto", "per", "via", "despite", "except", "around", "along", "inside", "outside", "throughout", "above", "below", "off", "out", "up", "down", "in", "by", "to", "like", "as", "than"}); }
    static bool is_obj(const u32& w) { return in_list(w, {"the", "a", "an", "his", "her", "their", "its", "my", "your", "our", "them", "him", "me", "us", "it", "itself", "himself", "herself", "themselves", "ourselves", "yourself", "myself", "this", "these", "those", "some", "every", "each", "all", "no", "any"}); }
    static bool is_subj(const u32& w) { return in_list(w, {"i", "we", "you", "they", "he", "she", "it", "who", "which"}); }
    static bool is_modal(const u32& w) { return in_list(w, {"to", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "do", "does", "did", "n't", "not", "never", "please", "let", "'ll", "'d", "ca", "wo", "sha"}); }
    static bool is_be(const u32& w) { return in_list(w, {"is", "was", "were", "are", "be", "been", "being", "am", "'m", "'re", "seemed", "seems", "seem", "looked", "felt", "became", "become", "remained", "grew", "get", "got"}); }
    static bool is_have(const u32& w) { return in_list(w, {"have", "has", "had", "having", "'ve"}); }
    static bool is_adv(const u32& w) { return in_list(w, {"only", "just", "never", "also", "then", "already", "ever", "often", "once", "always", "still", "soon", "now", "first", "again", "even", "really", "quickly", "slowly", "suddenly", "finally", "simply", "rather", "almost", "nearly", "hardly", "scarcely", "thus", "so"}); }
    static bool is_function(const u32& w) {
        return is_det(w) || is_prep(w) || is_obj(w) || is_subj(w) || is_modal(w) || is_be(w) || is_have(w) ||
               in_list(w, {"and", "or", "but", "nor", "that", "there", "here", "not", "if", "when", "where", "while", "because", "so", "then", "now", "how", "why"});
    }

    void tag(std::vector<SpTok>& toks) const {
        const size_t n = toks.size();
        auto low = [&](size_t i) { return lower(toks[i].text); };
        auto is_clause_punct = [&](size_t i) {  // . , ! ? ; : and dashes (not quotes or brackets)
            if (i >= n) return true;
            const u32& t = toks[i].text;
            return eq(t, ".") || eq(t, ",") || eq(t, "!") || eq(t, "?") || eq(t, ";") || eq(t, ":") || eq(t, "-") ||
                   eq(t, "—") || eq(t, "--") || eq(t, "...") || eq(t, "…");
        };
        for (size_t i = 0; i < n; ++i) {
            SpTok& t = toks[i];
            if (!t.tag.empty()) continue;
            const u32& x = t.text;
            u32 l = lower(x);
            // sentence start: first token, or after . ! ? (also behind an opening quote)
            size_t pi = i;
            while (pi > 0 && (eq(toks[pi - 1].text, "\"") || eq(toks[pi - 1].text, "'") || eq(toks[pi - 1].text, "“"))) --pi;
            bool sent_start = pi == 0 || (is_punct_tok(toks[pi - 1].text) && in(U".!?", toks[pi - 1].text.back()));
            bool prev_space = i == 0 || toks[i - 1].ws;
            bool next_cp = is_clause_punct(i + 1);
            u32 pl = i > 0 ? low(i - 1) : u32();
            u32 nl = i + 1 < n ? low(i + 1) : u32();
            if (is_punct_tok(x)) {
                char32_t c = x[0];
                if (x.size() == 1 && in(U".!?", c)) t.tag = ".";
                else if (eq(x, ",")) t.tag = ",";
                else if (eq(x, "-") || eq(x, "–") || eq(x, "—") || eq(x, "--")) t.tag = (prev_space && t.ws) ? ":" : "HYPH";
                else if (x.size() >= 2 && x[0] == U'.') t.tag = ":";
                else if (eq(x, ":") || eq(x, ";") || eq(x, "…")) t.tag = ":";
                else if (eq(x, "\"") || eq(x, "“") || eq(x, "”") || eq(x, "`") || eq(x, "``") || eq(x, "''")) {
                    bool opening = c == U'“' || c == U'`' || (c != U'”' && (prev_space || i == 0));
                    t.tag = opening ? "``" : "''";
                } else if (eq(x, "'") || eq(x, "‘") || eq(x, "’")) {
                    if (!prev_space && t.ws && i > 0 && ends_with(toks[i - 1].text, "s")) t.tag = "POS";
                    else t.tag = (c == U'‘' || prev_space || i == 0) ? "``" : "''";
                } else if (eq(x, "(") || eq(x, "[") || eq(x, "{")) t.tag = "-LRB-";
                else if (eq(x, ")") || eq(x, "]") || eq(x, "}")) t.tag = "-RRB-";
                else if (is_currency1(c)) t.tag = "$";
                else if (eq(x, "#")) t.tag = "$";
                else if (eq(x, "&")) t.tag = "CC";
                else if (eq(x, "%")) t.tag = "NN";
                else t.tag = "NFP";
                continue;
            }
            bool digit = false;
            for (char32_t c : x) if (is_ascii_digit(c)) digit = true;
            if (digit) {
                bool num = true;
                for (char32_t c : x) if (!(is_ascii_digit(c) || c == U',' || c == U'.' || c == U'-')) num = false;
                t.tag = num ? "CD" : "NN";
                continue;
            }
            if (in_list(l, {"n't", "not", "never"})) { t.tag = "RB"; continue; }
            if (in_list(l, {"'s"})) { t.tag = (i > 0 && in_list(pl, {"he", "she", "it", "that", "there", "what", "who", "where", "here", "how", "let"})) ? "VBZ" : "POS"; continue; }
            if (in_list(l, {"'m", "'re", "'ve"})) { t.tag = "VBP"; continue; }
            if (in_list(l, {"'ll", "'d", "ca", "wo", "sha"})) { t.tag = "MD"; continue; }
            if (in_list(l, {"the", "a", "an", "this", "these", "those", "every", "each", "another", "no", "some", "any", "all", "both", "either", "neither"})) {
                t.tag = "DT";
                if ((eq(x, "a") || eq(x, "A")) && (i + 1 >= n || in_list(toks[i + 1].text, {".", "!", "?"}))) t.tag = "NN";
                continue;
            }
            if (eq(x, "I") || in_list(l, {"you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "himself", "herself", "itself", "themselves", "yourself", "ourselves"})) { t.tag = "PRP"; continue; }
            if (in_list(l, {"my", "your", "his", "her", "its", "our", "their"})) { t.tag = "PRP$"; continue; }
            if (in_list(l, {"will", "would", "shall", "should", "can", "could", "may", "might", "must"})) { t.tag = "MD"; continue; }
            if (eq(l, "to")) { t.tag = "TO"; continue; }
            if (in_list(l, {"and", "or", "but", "nor"})) { t.tag = "CC"; continue; }
            if (eq(l, "in")) {
                bool particle = next_cp || in_list(nl, {"within", "to", "for", "at", "on", "with", "and", "or", "again", "there", "here", "now", "then", "upon", "among", "between"});
                t.tag = particle ? "RP" : "IN";
                continue;
            }
            if (eq(l, "by")) { t.tag = next_cp ? "RB" : "IN"; continue; }
            if (eq(l, "that")) {
                // demonstrative (DT) before a noun, at the start of a sentence or before punctuation; else IN / WDT
                if (sent_start || eq(nl, "'s")) t.tag = "DT";
                else if (next_cp) t.tag = in_list(pl, {"and", "but", "so", "or"}) ? "IN" : "DT";
                else if (i > 0 && !is_punct_tok(toks[i - 1].text) && is_function(pl) && !is_function(nl)) t.tag = "DT";
                else t.tag = "IN";
                continue;
            }
            if (in_list(l, {"of", "for", "with", "on", "at", "from", "into", "about", "over", "under", "after", "before", "through", "between", "against", "during", "without", "within", "upon", "towards", "toward", "among", "across", "behind", "beyond", "beneath", "beside", "besides", "since", "until", "till", "because", "although", "though", "unless", "whether", "if", "while", "than", "like", "as", "vs", "versus", "near", "onto", "per", "via", "despite", "except", "around", "along", "inside", "outside", "throughout", "above", "below", "off", "out", "up", "down"})) {
                t.tag = "IN";
                continue;
            }
            if (in_list(l, {"am"})) { t.tag = eq(x, "AM") ? "NN" : "VBP"; continue; }
            if (in_list(l, {"is", "was", "has", "does"})) { t.tag = eq(l, "is") || eq(l, "has") || eq(l, "does") ? "VBZ" : "VBD"; continue; }
            if (in_list(l, {"are", "do", "have"})) { t.tag = "VBP"; continue; }
            if (in_list(l, {"were", "did", "had"})) { t.tag = "VBD"; continue; }
            if (in_list(l, {"be"})) { t.tag = "VB"; continue; }
            if (in_list(l, {"been"})) { t.tag = "VBN"; continue; }
            if (in_list(l, {"being"})) { t.tag = "VBG"; continue; }
            if (eq(l, "used")) {
                bool passive = in_list(pl, {"is", "was", "were", "are", "be", "been", "being", "'s", "get", "got"});
                t.tag = passive ? "VBN" : "VBD";
                continue;
            }
            // capitalised words: proper nouns unless at the start of a sentence; ALL-CAPS words of up to 4 letters
            // (acronyms): NNP; longer ALL-CAPS words (headings) are tagged as ordinary words
            bool all_up = x.size() >= 2, any_cased = false;
            for (char32_t c : x) { if (is_lower(c)) all_up = false; if (is_cased(c)) any_cased = true; }
            if (all_up && any_cased && x.size() <= 4) { t.tag = "NNP"; continue; }
            if (!all_up && is_upper(x[0]) && !sent_start) { t.tag = "NNP"; continue; }
            auto hk = homographs_nv.find(u32_to_utf8(l));
            if (hk != homographs_nv.end()) {
                bool prev_nnp = i > 0 && toks[i - 1].tag == "NNP";
                t.tag = tag_homograph(l, hk->second, pl, nl, next_cp, sent_start, prev_nnp);
                continue;
            }
            if (ends_with(l, "ly")) t.tag = "RB";
            else if (ends_with(l, "ing")) t.tag = "VBG";
            else if (ends_with(l, "ed")) t.tag = "VBD";
            else t.tag = ends_with(l, "s") ? "NNS" : "NN";
        }
        (void)low;
    }

    enum : unsigned { K_NOUN = 1, K_VERB = 2, K_ADJ = 4, K_ADV = 8, K_VBD = 16, K_VBN = 32, K_VBP = 64 };
    // POS for a word whose gold entry has tag-dependent readings, from its neighbours.
    static std::string tag_homograph(const u32& l, unsigned keys, const u32& pl, const u32& nl, bool next_cp,
                                     bool sent_start, bool prev_nnp) {
        const bool plural = ends_with(l, "s") && !ends_with(l, "ss");
        const char* noun = plural ? "NNS" : "NN";
        if (is_have(pl)) return "VBN";
        if (in_list(pl, {"very", "so", "too", "quite", "more", "most", "less", "least", "rather", "as"})) return "JJ";
        if (is_be(pl)) {
            if (ends_with(l, "ed") || ends_with(l, "en")) return "VBN";
            if (keys & K_VBN) return "VBN";
            return "JJ";
        }
        if (is_modal(pl)) return "VB";
        if (is_det(pl) || is_prep(pl)) {
            if ((keys & K_ADJ) && !next_cp && !is_function(nl)) return "JJ";
            return noun;
        }
        if (in_list(pl, {"i", "we", "you", "they"})) return (keys & (K_VBD | K_VBP)) ? "VBD" : "VBP";
        if (in_list(pl, {"he", "she", "it", "who", "which", "that"})) return plural ? "VBZ" : ((keys & K_VBD) ? "VBD" : "VBP");
        if (is_adv(pl) || prev_nnp) return (keys & K_VBD) ? "VBD" : (plural ? "VBZ" : "VB");
        if (is_obj(nl) && !is_det(pl)) return plural ? "VBZ" : "VB";
        if (sent_start && is_prep(nl)) return "RB";
        if (sent_start && !next_cp && !is_be(nl) && !eq(nl, "of") && !eq(nl, "'s")) return plural ? "VBZ" : "VB";
        if (next_cp) return "RB";
        if (is_prep(nl) && (keys & K_ADJ)) return "JJ";
        return noun;
    }

    // ======================= misaki Lexicon =======================
    LexVal golds(const u32& w) const { return grown_get(gold, w); }
    LexVal silvers(const u32& w) const { return grown_get(silver, w); }

    Res get_NNP(const u32& word) const {
        u32 ps;
        for (char32_t c : word) {
            if (!is_alpha(c)) continue;
            LexVal v = golds(u32(1, to_upper(c)));
            if (!v.found() || v.is_null() || v.is_dict()) return none();
            ps += v.str();
        }
        ps = apply_stress(ps, 0);
        size_t k = ps.rfind(SECONDARY);
        if (k != u32::npos) ps[k] = PRIMARY;
        return mk(ps, 3);
    }

    Res get_special_case(const u32& word, const std::string& tag, double stress, const Ctx& ctx) const {
        if (tag == "ADD" && (eq(word, ".") || eq(word, "/"))) return lookup(eq(word, ".") ? A("dot") : A("slash"), "", -0.5, &ctx);
        if (eq(word, "%")) return lookup(A("percent"), "", NONE, &ctx);
        if (eq(word, "&")) return lookup(A("and"), "", NONE, &ctx);
        if (eq(word, "+")) return lookup(A("plus"), "", NONE, &ctx);
        if (eq(word, "@")) return lookup(A("at"), "", NONE, &ctx);
        {
            u32 st = strip_chars(word, U".");
            if (st.find(U'.') != u32::npos) {
                u32 nod = word; remove_char(nod, U'.');
                if (all_alpha(nod)) {
                    size_t longest = 0, cur = 0;
                    for (char32_t c : word) { if (c == U'.') { longest = std::max(longest, cur); cur = 0; } else ++cur; }
                    longest = std::max(longest, cur);
                    if (longest < 3) return get_NNP(word);
                }
            }
        }
        if (eq(word, "a") || eq(word, "A")) return mk(tag == "DT" ? A("ɐ") : A("ˈA"), 4);
        if (eq(word, "am") || eq(word, "Am") || eq(word, "AM")) {
            if (tag.rfind("NN", 0) == 0) return get_NNP(word);
            if (ctx.future_vowel == -1 || !eq(word, "am") || (!is_none(stress) && stress > 0)) {
                LexVal v = golds(A("am"));
                if (v.found() && !v.is_dict() && !v.is_null()) return mk(v.str(), 4);
                return none();
            }
            return mk(A("ɐm"), 4);
        }
        if (eq(word, "an") || eq(word, "An") || eq(word, "AN")) {
            if (eq(word, "AN") && tag.rfind("NN", 0) == 0) return get_NNP(word);
            return mk(A("ɐn"), 4);
        }
        if (eq(word, "I") && tag == "PRP") return mk(u32(1, SECONDARY) + U"I", 4);
        if ((eq(word, "by") || eq(word, "By") || eq(word, "BY")) && parent_tag(tag) == "ADV") return mk(A("bˈI"), 4);
        if (eq(word, "to") || eq(word, "To") || (eq(word, "TO") && (tag == "TO" || tag == "IN"))) {
            if (ctx.future_vowel == -1) { LexVal v = golds(A("to")); return v.found() && !v.is_dict() && !v.is_null() ? mk(v.str(), 4) : none(); }
            return mk(ctx.future_vowel ? A("tʊ") : A("tə"), 4);
        }
        if (eq(word, "in") || eq(word, "In") || (eq(word, "IN") && tag != "NNP")) {
            u32 s = (ctx.future_vowel == -1 || tag != "IN") ? u32(1, PRIMARY) : u32();
            return mk(s + U"ɪn", 4);
        }
        if (eq(word, "the") || eq(word, "The") || (eq(word, "THE") && tag == "DT")) return mk(ctx.future_vowel == 1 ? A("ði") : A("ðə"), 4);
        if (tag == "IN") {
            u32 l = lower(word);
            if (eq(l, "vs") || eq(l, "vs.")) return lookup(A("versus"), "", NONE, &ctx);
        }
        if (eq(word, "used") || eq(word, "Used") || eq(word, "USED")) {
            LexVal v = golds(A("used"));
            u32 out; bool null = false;
            if (v.is_dict()) {
                if ((tag == "VBD" || tag == "JJ") && ctx.future_to) { if (v.dict_get("VBD", out, null) && !null) return mk(out, 4); }
                if (v.dict_get("DEFAULT", out, null) && !null) return mk(out, 4);
            }
            return none();
        }
        return none();
    }

    // ctx == nullptr: Python's ctx=None
    Res lookup(u32 word, std::string tag, double stress, const Ctx* ctx) const {
        int is_NNP = -1;
        if (word == upper(word) && !golds(word).found()) {
            word = lower(word);
            is_NNP = tag == "NNP" ? 1 : 0;
        }
        LexVal v = golds(word);
        int rating = 4;
        if (!v.found() && is_NNP != 1) { v = silvers(word); rating = 3; }
        bool ps_none = !v.found() || v.is_null();
        u32 ps;
        if (v.is_dict()) {
            u32 out; bool null = false;
            if (ctx && ctx->future_vowel == -1 && v.dict_get("None", out, null)) tag = "None";
            else if (!v.dict_get(tag, out, null)) tag = parent_tag(tag);
            if (v.dict_get(tag, out, null)) { ps_none = null; ps = out; }
            else { v.dict_get("DEFAULT", out, null); ps_none = null; ps = out; }
        } else if (!ps_none) ps = v.str();
        if (ps_none || (is_NNP == 1 && ps.find(PRIMARY) == u32::npos)) {
            Res r = get_NNP(word);
            if (r.ok) return r;
        }
        if (ps_none) return none();   // Python would fail in apply_stress (or return None for stress None)
        return mk(apply_stress(ps, stress), rating);
    }

    bool is_known(const u32& word, const std::string&) const {
        if (golds(word).found() || eq(word, "%") || eq(word, "&") || eq(word, "+") || eq(word, "@") || silvers(word).found()) return true;
        if (!all_alpha(word) || !all_lexicon_ords(word)) return false;
        if (word.size() == 1) return true;
        if (word == upper(word) && golds(lower(word)).found()) return true;
        u32 r = word.substr(1);
        return r == upper(r);
    }

    static u32 _s(const u32& stem) {
        if (stem.empty()) return stem;
        char32_t l = stem.back();
        if (in(U"ptkfθ", l)) return stem + U"s";
        if (in(U"szʃʒʧʤ", l)) return stem + U"ᵻz";
        return stem + U"z";
    }
    Res stem_s(const u32& word, const std::string& tag, double stress, const Ctx* ctx) const {
        if (word.size() < 3 || !ends_with(word, "s")) return none();
        u32 stem;
        if (!ends_with(word, "ss") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if ((ends_with(word, "'s") || (word.size() > 4 && ends_with(word, "es") && !ends_with(word, "ies"))) && is_known(word.substr(0, word.size() - 2), tag)) stem = word.substr(0, word.size() - 2);
        else if (word.size() > 4 && ends_with(word, "ies") && is_known(word.substr(0, word.size() - 3) + U"y", tag)) stem = word.substr(0, word.size() - 3) + U"y";
        else return none();
        Res r = lookup(stem, tag, stress, ctx);
        if (!r.ok) return none();
        r.ps = _s(r.ps);
        return r;
    }
    static Res _ed_res(Res r) {
        if (!r.ok) return r;
        const u32& stem = r.ps;
        if (stem.empty()) return none();
        char32_t l = stem.back();
        if (in(U"pkfθʃsʧ", l)) r.ps = stem + U"t";
        else if (l == U'd') r.ps = stem + U"ᵻd";
        else if (l != U't') r.ps = stem + U"d";
        else if (stem.size() < 2) r.ps = stem + U"ɪd";
        else if (in(US_TAUS_S, stem[stem.size() - 2])) r.ps = stem.substr(0, stem.size() - 1) + U"ɾᵻd";
        else r.ps = stem + U"ᵻd";
        return r;
    }
    Res stem_ed(const u32& word, const std::string& tag, double stress, const Ctx* ctx) const {
        if (word.size() < 4 || !ends_with(word, "d")) return none();
        u32 stem;
        if (!ends_with(word, "dd") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if (word.size() > 4 && ends_with(word, "ed") && !ends_with(word, "eed") && is_known(word.substr(0, word.size() - 2), tag)) stem = word.substr(0, word.size() - 2);
        else return none();
        return _ed_res(lookup(stem, tag, stress, ctx));
    }
    static Res _ing_res(Res r) {
        if (!r.ok) return r;
        const u32& stem = r.ps;
        if (stem.empty()) return none();
        if (stem.size() > 1 && stem.back() == U't' && in(US_TAUS_S, stem[stem.size() - 2])) r.ps = stem.substr(0, stem.size() - 1) + U"ɾɪŋ";
        else r.ps = stem + U"ɪŋ";
        return r;
    }
    Res stem_ing(const u32& word, const std::string& tag, double stress, const Ctx* ctx) const {
        if (word.size() < 5 || !ends_with(word, "ing")) return none();
        u32 stem;
        u32 base = word.substr(0, word.size() - 3);
        bool dbl = false;
        if (word.size() > 5) {
            // ([bcdgklmnprstvxz])\1ing$|cking$
            size_t n = word.size();
            if (n >= 5 && word[n - 4] == word[n - 5] && in(U"bcdgklmnprstvxz", word[n - 4])) dbl = true;
            if (ends_with(word, "cking")) dbl = true;
        }
        if (word.size() > 5 && is_known(base, tag)) stem = base;
        else if (is_known(base + U"e", tag)) stem = base + U"e";
        else if (word.size() > 5 && dbl && is_known(word.substr(0, word.size() - 4), tag)) stem = word.substr(0, word.size() - 4);
        else return none();
        return _ing_res(lookup(stem, tag, stress, ctx));
    }

    Res get_word(u32 word, const std::string& tag, double stress, const Ctx& ctx) const {
        Res r = get_special_case(word, tag, stress, ctx);
        if (r.ok) return r;
        u32 wl = lower(word);
        u32 noapo = word; remove_char(noapo, U'\'');
        if (word.size() > 1 && all_alpha(noapo) && word != wl && (tag != "NNP" || word.size() > 7) &&
            !golds(word).found() && !silvers(word).found() &&
            (word == upper(word) || word.substr(1) == lower(word.substr(1)))) {
            bool ok = golds(wl).found() || silvers(wl).found();
            if (!ok) { Res a = stem_s(wl, tag, stress, &ctx); ok = a.ok && !a.ps.empty(); }
            if (!ok) { Res a = stem_ed(wl, tag, stress, &ctx); ok = a.ok && !a.ps.empty(); }
            if (!ok) { Res a = stem_ing(wl, tag, stress, &ctx); ok = a.ok && !a.ps.empty(); }
            if (ok) word = wl;
        }
        if (is_known(word, tag)) return lookup(word, tag, stress, &ctx);
        if (ends_with(word, "s'") && is_known(word.substr(0, word.size() - 2) + U"'s", tag)) return lookup(word.substr(0, word.size() - 2) + U"'s", tag, stress, &ctx);
        if (ends_with(word, "'") && is_known(word.substr(0, word.size() - 1), tag)) return lookup(word.substr(0, word.size() - 1), tag, stress, &ctx);
        r = stem_s(word, tag, stress, &ctx);
        if (r.ok) return r;
        r = stem_ed(word, tag, stress, &ctx);
        if (r.ok) return r;
        r = stem_ing(word, tag, is_none(stress) ? 0.5 : stress, &ctx);
        if (r.ok) return r;
        return none();
    }

    static bool is_currency_amount(const u32& word) {
        size_t dots = std::count(word.begin(), word.end(), U'.');
        if (dots == 0) return true;
        if (dots > 1) return false;
        u32 cents = word.substr(word.find(U'.') + 1);
        return cents.size() < 3;   // set(cents) == {0} compares characters with the int 0: never true
    }

    void extend_num(std::vector<Res>& result, const std::string& text_or_digits, bool escape) const {
        std::string words = escape ? text_or_digits : num_cardinal(text_or_digits);
        // re.split(r'[^a-z]+')
        std::string cur;
        auto flush = [&](bool) {
            const std::string& w = cur;
            if (w != "and") result.push_back(lookup(utf8_to_u32(w), "", w == "point" ? -2 : NONE, nullptr));
            cur.clear();
        };
        bool any = false;
        for (char c : words) {
            if (c >= 'a' && c <= 'z') { cur += c; any = true; }
            else { flush(false); }
        }
        flush(true);
        (void)any;
    }

    Res get_number(u32 word, char32_t currency, bool is_head) const {
        u32 suffix;
        {
            size_t k = word.size();
            while (k > 0 && ((word[k - 1] >= U'a' && word[k - 1] <= U'z') || word[k - 1] == U'\'')) --k;
            if (k < word.size()) { suffix = word.substr(k); word = word.substr(0, k); }
        }
        std::vector<Res> result;
        if (starts_with(word, U"-")) { result.push_back(lookup(A("minus"), "", NONE, nullptr)); word = word.substr(1); }
        std::string w8 = u32_to_utf8(word);
        auto is_cur = [&](char32_t c) { return c == U'$' || c == U'£' || c == U'€'; };
        if (is_digits(word) && is_ordinal_suffix(suffix)) {
            extend_num(result, num_ordinal(w8), true);
        } else if (result.empty() && word.size() == 4 && !is_cur(currency) && is_digits(word)) {
            extend_num(result, num_year(w8), true);
        } else if (!is_head && word.find(U'.') == u32::npos) {
            std::string num; for (char c : w8) if (c != ',') num += c;
            if (num.empty()) return none();
            if (num[0] == '0' || num.size() > 3) { for (char c : num) extend_num(result, std::string(1, c), false); }
            else if (num.size() == 3 && !(num[1] == '0' && num[2] == '0')) {
                extend_num(result, num.substr(0, 1), false);
                if (num[1] == '0') { result.push_back(lookup(A("O"), "", -2, nullptr)); extend_num(result, num.substr(2, 1), false); }
                else extend_num(result, num.substr(1), false);
            } else extend_num(result, num, false);
        } else if (std::count(word.begin(), word.end(), U'.') > 1 || !is_head) {
            std::string all; for (char c : w8) if (c != ',') all += c;
            size_t st = 0;
            while (st <= all.size()) {
                size_t e = all.find('.', st);
                if (e == std::string::npos) e = all.size();
                std::string num = all.substr(st, e - st);
                if (!num.empty()) {
                    bool nz = false; for (size_t q = 1; q < num.size(); ++q) if (num[q] != '0') nz = true;
                    if (num[0] == '0' || (num.size() != 2 && nz)) { for (char c : num) extend_num(result, std::string(1, c), false); }
                    else extend_num(result, num, false);
                }
                st = e + 1;
                if (e == all.size()) break;
            }
        } else if (is_cur(currency) && is_currency_amount(word)) {
            const char* units[2];
            if (currency == U'$') { units[0] = "dollar"; units[1] = "cent"; }
            else if (currency == U'£') { units[0] = "pound"; units[1] = "pence"; }
            else { units[0] = "euro"; units[1] = "cent"; }
            std::string all; for (char c : w8) if (c != ',') all += c;
            std::vector<std::pair<long long, const char*>> pairs;
            size_t dot = all.find('.');
            std::string p0 = all.substr(0, dot), p1 = dot == std::string::npos ? "" : all.substr(dot + 1);
            auto toi = [](const std::string& s) { long long v = 0; for (char c : s) v = v * 10 + (c - '0'); return v; };
            if (p0.size() > 15 || p1.size() > 15) return none();
            pairs.push_back({toi(p0), units[0]});
            if (dot != std::string::npos) pairs.push_back({toi(p1), units[1]});
            if (pairs.size() > 1) {
                if (pairs[1].first == 0) pairs.resize(1);
                else if (pairs[0].first == 0) pairs.erase(pairs.begin());
            }
            for (size_t i = 0; i < pairs.size(); ++i) {
                if (i > 0) result.push_back(lookup(A("and"), "", NONE, nullptr));
                extend_num(result, std::to_string(pairs[i].first), false);
                std::string unit = pairs[i].second;
                if (std::llabs(pairs[i].first) != 1 && unit != "pence") result.push_back(stem_s(utf8_to_u32(unit + "s"), "", NONE, nullptr));
                else result.push_back(lookup(utf8_to_u32(unit), "", NONE, nullptr));
            }
        } else {
            std::string text;
            if (is_digits(word)) text = num_cardinal(w8);
            else if (word.find(U'.') == u32::npos) {
                std::string all; for (char c : w8) if (c != ',') all += c;
                text = is_ordinal_suffix(suffix) ? num_ordinal(all) : num_cardinal(all);
            } else {
                std::string all; for (char c : w8) if (c != ',') all += c;
                if (!all.empty() && all[0] == '.') {
                    text = "point";
                    for (size_t q = 1; q < all.size(); ++q) { text += ' '; text += num_cardinal(all.substr(q, 1)); }
                } else text = num_float(all);
            }
            extend_num(result, text, true);
        }
        if (result.empty()) return none();
        u32 ps;
        int rating = 99;
        for (size_t i = 0; i < result.size(); ++i) {
            if (!result[i].ok) return none();
            if (i) ps += U' ';
            ps += result[i].ps;
            rating = std::min(rating, result[i].rating);
        }
        Res r = mk(ps, rating);
        if (eq(suffix, "s") || eq(suffix, "'s")) r.ps = _s(r.ps);
        else if (eq(suffix, "ed") || eq(suffix, "'d")) r = _ed_res(r);
        else if (eq(suffix, "ing")) r = _ing_res(r);
        return r;
    }

    Res append_currency(Res r, char32_t currency) const {
        if (!currency || !r.ok) return r;
        const char* u = currency == U'$' ? "dollars" : currency == U'£' ? "pounds" : currency == U'€' ? "euros" : nullptr;
        if (!u) return r;
        Res c = stem_s(A(u), "", NONE, nullptr);
        if (c.ok && !c.ps.empty()) r.ps += U" " + c.ps;
        return r;
    }

    static bool is_number(u32 word, bool is_head) {
        bool any = false;
        for (char32_t c : word) if (is_ascii_digit(c)) any = true;
        if (!any) return false;
        static const char* sfx[] = {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"};
        for (auto* s : sfx) if (ends_with(word, s)) { word = word.substr(0, word.size() - std::strlen(s)); break; }
        for (size_t i = 0; i < word.size(); ++i) {
            char32_t c = word[i];
            if (!(is_ascii_digit(c) || c == U',' || c == U'.' || (is_head && i == 0 && c == U'-'))) return false;
        }
        return true;
    }

    Res lexicon(const MTok& tk, const Ctx& ctx) const {
        u32 word = tk.has_alias ? tk.alias : tk.text;
        replace_char(word, U'‘', U'\'');
        replace_char(word, U'’', U'\'');
        // NFKC (subset) and numeric_if_needed
        for (auto& c : word) {
            if (c == 0xB2) c = U'2'; else if (c == 0xB3) c = U'3'; else if (c == 0xB9) c = U'1';
            else if (c >= 0xFF10 && c <= 0xFF19) c = U'0' + (c - 0xFF10);
        }
        double stress = word == lower(word) ? NONE : (word == upper(word) ? 2 : 0.5);
        Res r = get_word(word, tk.tag, stress, ctx);
        if (r.ok) return append_currency(r, tk.currency);
        if (is_number(word, tk.is_head)) return get_number(word, tk.currency, tk.is_head);
        return none();
    }

    // ======================= english.py Fallback =======================
    static char32_t translit1(char32_t c, u32& extra) {
        switch (c) {
            case U'ø': return U'o'; case U'Ø': return U'O'; case U'ß': extra = U"s"; return U's';
            case U'æ': extra = U"e"; return U'a'; case U'Æ': extra = U"E"; return U'A';
            case U'œ': extra = U"e"; return U'o'; case U'Œ': extra = U"E"; return U'O';
            case U'ł': return U'l'; case U'Ł': return U'L'; case U'đ': return U'd'; case U'Đ': return U'D';
            case U'’': return U'\'';
            default: return c;
        }
    }
    // NFKD base letter of a precomposed Latin letter (accents dropped, as english.py's unicodedata step)
    static char32_t strip_accent(char32_t c) {
        static const char* lat1 = "AAAAAAACEEEEIIIIDNOOOOO*OUUUUYPsaaaaaaaceeeeiiiidnooooo/ouuuuypy";
        if (c >= 0xC0 && c <= 0xFF) {
            char b = lat1[c - 0xC0];
            if (c == 0xC6 || c == 0xE6 || c == 0xD0 || c == 0xF0 || c == 0xDE || c == 0xFE || c == 0xDF || c == 0xD7 || c == 0xF7 || c == 0xD8 || c == 0xF8) return c;
            return (char32_t)(unsigned char)b;
        }
        if (c >= 0x100 && c <= 0x17F) {
            static const char* extA = "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIiJjJjKkkLlLlLlLlLlNnNnNnnNnOoOoOoOoRrRrRrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZzs";
            char b = extA[c - 0x100];
            if (c == 0x110 || c == 0x111 || c == 0x126 || c == 0x127 || c == 0x131 || c == 0x132 || c == 0x133 || c == 0x138 ||
                c == 0x13F || c == 0x140 || c == 0x141 || c == 0x142 || c == 0x149 || c == 0x14A || c == 0x14B || c == 0x152 ||
                c == 0x153 || c == 0x166 || c == 0x167 || c == 0x17F)
                return c;
            return (char32_t)(unsigned char)b;
        }
        return c;
    }

    u32 bart_word(const u32& word) {
        std::string key = u32_to_utf8(word);
        auto it = bart_memo.find(key);
        if (it != bart_memo.end()) return it->second;
        auto t0 = std::chrono::steady_clock::now();
        u32 out;
        if (!bart_loaded) {
            std::string err;
            if (!bart.load(bart_dir, err)) return u32();
            bart_loaded = true;
        }
        const int step = bart.max_len() - 2;
        for (size_t i = 0; i < word.size(); i += (size_t)step) {
            ids_buf.clear();
            ids_buf.push_back(1);
            for (size_t k = i; k < std::min(word.size(), i + (size_t)step); ++k) {
                auto g = g2i.find(word[k]);
                ids_buf.push_back(g == g2i.end() ? 3 : g->second);
            }
            ids_buf.push_back(2);
            bart.generate(ids_buf, bart.max_len(), gen_buf);
            for (int id : gen_buf) { auto p = i2p.find(id); if (p != i2p.end()) out += p->second; }
        }
        replace_char(out, U'ɾ', U'T');
        replace_char(out, U'ʔ', U't');
        timing.bart_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        timing.bart_words += 1;
        if (bart_memo.size() < 4096) bart_memo.emplace(key, out);
        return out;
    }

    bool cmu_get(const u32& word, u32& out) const {
        u32 w = lower(word);
        const LexStore::E* e = cmu.find(u32_to_utf8(w));
        if (e) {
            if (e->vo == LexStore::NUL) return false;
            out = LexVal{&cmu, e}.str();
            return true;
        }
        if (ends_with(w, "'s")) {
            e = cmu.find(u32_to_utf8(w.substr(0, w.size() - 2)));
            if (e && e->vo != LexStore::NUL) {
                u32 stem = LexVal{&cmu, e}.str();
                if (stem.empty()) return false;   // Python: stem[-1] fails; treat as absent
                char32_t l = stem.back();
                out = stem + (in(U"ptkfθ", l) ? u32(U"s") : in(U"szʃʒʧʤ", l) ? u32(U"ᵻz") : u32(U"z"));
                return true;
            }
        }
        return false;
    }

    u32 fb_word(const u32& word) {
        u32 ps;
        bool have = cmu_get(word, ps);
        if (!have && word.find(U'-') != u32::npos) {
            u32 out;
            size_t st = 0;
            bool first = true;
            while (st <= word.size()) {
                size_t e = word.find(U'-', st);
                if (e == u32::npos) e = word.size();
                if (e > st) {
                    if (!first) out += U' ';
                    out += fb_word(word.substr(st, e - st));
                    first = false;
                }
                if (e == word.size()) break;
                st = e + 1;
            }
            return out;
        }
        if (have && !ps.empty()) return ps;
        return bart_word(word);
    }

    // Fallback.say + __call__: returns false when nothing could be said
    bool fallback(const u32& text, u32& out) {
        u32 t;
        for (char32_t c : text) {
            u32 extra;
            char32_t b = translit1(c, extra);
            t += strip_accent(b);
            t += extra;
        }
        // drop combining marks
        t.erase(std::remove_if(t.begin(), t.end(), is_combining), t.end());
        static const char* DIG[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        std::vector<u32> parts;
        size_t i = 0;
        while (i < t.size()) {
            if (is_digit_py(t[i])) {
                size_t j = i;
                u32 o;
                while (j < t.size() && is_digit_py(t[j])) {
                    int d = is_ascii_digit(t[j]) ? (int)(t[j] - U'0') : 0;
                    if (!o.empty()) o += U' ';
                    o += fb_word(A(DIG[d]));
                    ++j;
                }
                parts.push_back(o);
                i = j;
            } else if (g2i.count(t[i])) {
                size_t j = i;
                while (j < t.size() && g2i.count(t[j]) && !is_digit_py(t[j])) ++j;
                u32 letters = strip_chars(t.substr(i, j - i), U"'-.");
                if (!letters.empty()) parts.push_back(fb_word(letters));
                i = j;
            } else ++i;
        }
        out.clear();
        for (auto& p : parts) {
            if (p.empty()) continue;
            if (!out.empty()) out += U' ';
            out += p;
        }
        return !out.empty();
    }

    // ======================= misaki G2P =======================
    static std::vector<u32> subtokenize(const u32& w) {
        std::vector<u32> out;
        const size_t n = w.size();
        auto apo = [](char32_t c) { return c == U'\'' || c == U'‘' || c == U'’'; };
        size_t i = 0;
        while (i < n) {
            size_t e = 0;
            // ^['‘’]+
            if (i == 0 && apo(w[0])) { e = 0; while (e < n && apo(w[e])) ++e; }
            // \p{Lu}(?=\p{Lu}\p{Ll})
            else if (is_upper(w[i]) && i + 2 < n + 0 && i + 2 <= n - 1 + 1 && i + 1 < n && i + 2 < n && is_upper(w[i + 1]) && is_lower(w[i + 2])) e = i + 1;
            else {
                // (?:^-)?(?:\d?[,.]?\d)+
                size_t p = i;
                if (i == 0 && w[0] == U'-') p = 1;
                auto group = [&](size_t q) -> size_t {  // returns end or 0
                    // options in preference order: d sep d, d d, sep d, d
                    if (q < n && is_ascii_digit(w[q])) {
                        if (q + 2 < n + 0 && q + 1 < n && (w[q + 1] == U',' || w[q + 1] == U'.') && q + 2 < n && is_ascii_digit(w[q + 2])) return q + 3;
                        if (q + 1 < n && is_ascii_digit(w[q + 1])) return q + 2;
                    }
                    if (q + 1 < n && (w[q] == U',' || w[q] == U'.') && is_ascii_digit(w[q + 1])) return q + 2;
                    if (q < n && is_ascii_digit(w[q])) return q + 1;
                    return 0;
                };
                size_t g = group(p);
                if (g) {
                    size_t endp = g;
                    while (true) { size_t g2 = group(endp); if (!g2) break; endp = g2; }
                    e = endp;
                } else if (w[i] == U'-' || w[i] == U'_') {
                    e = i; while (e < n && (w[e] == U'-' || w[e] == U'_')) ++e;
                } else if (apo(w[i]) && i + 1 < n && apo(w[i + 1])) {
                    e = i; while (e < n && apo(w[e])) ++e;
                } else {
                    // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
                    size_t found = 0;
                    for (size_t k = i; k < n && !found; ++k) {
                        if (k > i && !is_alpha(w[k - 1])) break;
                        size_t q = k;
                        while (true) {
                            if (q + 1 < n && is_lower(w[q]) && is_upper(w[q + 1])) { found = q + 1; break; }
                            if (q + 1 < n && apo(w[q]) && is_alpha(w[q + 1])) { q += 2; continue; }
                            break;
                        }
                    }
                    if (found) e = found;
                    else if (is_alpha(w[i])) {
                        // \p{L}+(?:['‘’]\p{L})*
                        e = i; while (e < n && is_alpha(w[e])) ++e;
                        while (e + 1 < n && apo(w[e]) && is_alpha(w[e + 1])) e += 2;
                    } else if (!(w[i] == U'-' || w[i] == U'_' || is_alpha(w[i]) || apo(w[i]) || is_ascii_digit(w[i]))) {
                        e = i + 1;
                    } else if (apo(w[i])) {
                        // ['‘’]+$
                        size_t q = i; while (q < n && apo(w[q])) ++q;
                        if (q == n) e = n;
                    }
                }
            }
            if (e > i) { out.push_back(w.substr(i, e - i)); i = e; }
            else ++i;
        }
        return out;
    }

    static MTok merge_tokens(const std::vector<MTok>& toks, size_t a, size_t b, const u32* unk) {
        MTok m;
        for (size_t i = a; i < b; ++i) { m.text += toks[i].text; if (i + 1 < b) m.text += toks[i].ws; }
        int best = -1; size_t bi = a;
        for (size_t i = a; i < b; ++i) {
            int s = 0;
            for (char32_t c : toks[i].text) s += (c == to_lower(c)) ? 1 : 2;
            if (s > best) { best = s; bi = i; }
        }
        m.tag = toks[bi].tag;
        m.ws = toks[b - 1].ws;
        if (unk) {
            m.has_ph = true;
            for (size_t i = a; i < b; ++i) {
                const MTok& t = toks[i];
                if (t.prespace && !m.ph.empty() && !is_space(m.ph.back()) && t.has_ph && !t.ph.empty()) m.ph += U' ';
                if (t.has_ph) m.ph += t.ph; else m.ph += *unk;
            }
        }
        m.is_head = toks[a].is_head;
        char32_t cur = 0;
        for (size_t i = a; i < b; ++i) if (toks[i].currency > cur) cur = toks[i].currency;
        m.currency = cur;
        m.prespace = toks[a].prespace;
        return m;
    }

    static bool punct_tag(const std::string& t) {
        return t == "." || t == "," || t == "-LRB-" || t == "-RRB-" || t == "``" || t == "\"\"" || t == "''" ||
               t == ":" || t == "$" || t == "#" || t == "NFP";
    }

    std::vector<Word> retokenize(std::vector<MTok>& tokens) const {
        std::vector<Word> words;
        char32_t currency = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            MTok& token = tokens[i];
            std::vector<MTok> tks;
            if (!token.has_alias && !token.has_ph) {
                for (auto& t : subtokenize(token.text)) {
                    MTok x;
                    x.text = t; x.tag = token.tag; x.is_head = true;
                    tks.push_back(x);
                }
            } else tks.push_back(token);
            if (tks.empty()) continue;
            tks.back().ws = token.ws;
            for (size_t j = 0; j < tks.size(); ++j) {
                MTok& tk = tks[j];
                if (tk.has_alias || tk.has_ph) {
                } else if (tk.tag == "$" && (eq(tk.text, "$") || eq(tk.text, "£") || eq(tk.text, "€"))) {
                    currency = tk.text[0]; tk.has_ph = true; tk.ph.clear(); tk.rating = 4;
                } else if (tk.tag == ":" && (eq(tk.text, "-") || eq(tk.text, "–"))) {
                    tk.has_ph = true; tk.ph = U"—"; tk.rating = 3;
                } else if (punct_tag(tk.tag) && !std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { char32_t l = to_lower(c); return l >= 97 && l <= 122; })) {
                    tk.has_ph = true;
                    if (tk.tag == "-LRB-") tk.ph = U"(";
                    else if (tk.tag == "-RRB-") tk.ph = U")";
                    else if (tk.tag == "``") tk.ph = U"“";
                    else if (tk.tag == "\"\"" || tk.tag == "''") tk.ph = U"”";
                    else { tk.ph.clear(); for (char32_t c : tk.text) if (in(PUNCTS_S, c)) tk.ph += c; }
                    tk.rating = 4;
                } else if (currency) {
                    if (tk.tag != "CD") currency = 0;
                    else if (j + 1 == tks.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) tk.currency = currency;
                } else if (j > 0 && j + 1 < tks.size() && eq(tk.text, "2") && !tks[j - 1].text.empty() && !tks[j + 1].text.empty() &&
                           is_alpha(tks[j - 1].text.back()) && is_alpha(tks[j + 1].text[0])) {
                    tk.has_alias = true; tk.alias = U"to";
                }
                if (tk.has_alias || tk.has_ph) words.push_back(Word{false, {tk}});
                else if (!words.empty() && words.back().is_list && words.back().t.back().ws.empty()) {
                    tk.is_head = false;
                    words.back().t.push_back(tk);
                } else if (!tk.ws.empty()) words.push_back(Word{false, {tk}});
                else words.push_back(Word{true, {tk}});
            }
        }
        for (auto& w : words) if (w.is_list && w.t.size() == 1) w.is_list = false;
        return words;
    }

    static Ctx token_context(const Ctx& ctx, const MTok* ps_tok, bool has_ps, const u32& ps, const u32& text, const std::string& tag) {
        Ctx c;
        c.future_vowel = ctx.future_vowel;
        (void)ps_tok;
        if (has_ps && !ps.empty()) {
            for (char32_t ch : ps) {
                bool v = is_vowel(ch), cons = in(CONSONANTS_S, ch), p = in(NON_QUOTE_PUNCTS_S, ch);
                if (v || cons || p) { c.future_vowel = p ? -1 : (v ? 1 : 0); break; }
            }
        }
        c.future_to = eq(text, "to") || eq(text, "To") || (eq(text, "TO") && (tag == "TO" || tag == "IN"));
        return c;
    }

    static void resolve_tokens(std::vector<MTok>& tokens) {
        u32 text;
        for (size_t i = 0; i < tokens.size(); ++i) { text += tokens[i].text; if (i + 1 < tokens.size()) text += tokens[i].ws; }
        bool prespace = text.find(U' ') != u32::npos || text.find(U'/') != u32::npos;
        if (!prespace) {
            int kinds = 0;
            for (char32_t c : text) {
                if (in(SUBTOKEN_JUNKS_S, c)) continue;
                kinds |= is_alpha(c) ? 1 : (is_ascii_digit(c) ? 2 : 4);
            }
            prespace = (kinds & (kinds - 1)) != 0;
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            MTok& tk = tokens[i];
            if (!tk.has_ph) {
                if (i == tokens.size() - 1 && tk.text.size() == 1 && in(NON_QUOTE_PUNCTS_S, tk.text[0])) { tk.has_ph = true; tk.ph = tk.text; tk.rating = 3; }
                else if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return in(SUBTOKEN_JUNKS_S, c); })) { tk.has_ph = true; tk.ph.clear(); tk.rating = 3; }
            } else if (i > 0) tk.prespace = prespace;
        }
        if (prespace) return;
        struct Ix { bool p; int w; size_t i; };
        std::vector<Ix> idx;
        for (size_t i = 0; i < tokens.size(); ++i)
            if (tokens[i].has_ph && !tokens[i].ph.empty())
                idx.push_back({tokens[i].ph.find(PRIMARY) != u32::npos, stress_weight(tokens[i].ph), i});
        if (idx.size() == 2 && tokens[idx[0].i].text.size() == 1) {
            size_t i = idx[1].i;
            tokens[i].ph = apply_stress(tokens[i].ph, -0.5);
            return;
        }
        size_t np = 0;
        for (auto& x : idx) np += x.p;
        if (idx.size() < 2 || np <= (idx.size() + 1) / 2) return;
        std::sort(idx.begin(), idx.end(), [](const Ix& a, const Ix& b) {
            if (a.p != b.p) return a.p < b.p;
            if (a.w != b.w) return a.w < b.w;
            return a.i < b.i;
        });
        size_t half = idx.size() / 2;
        for (size_t k = 0; k < half; ++k) tokens[idx[k].i].ph = apply_stress(tokens[idx[k].i].ph, -0.5);
    }

    // misaki G2P.__call__ (preprocess: no link features survive tts_norm) -> phoneme string (unk = "")
    u32 misaki(const std::vector<SpTok>& sp) {
        std::vector<MTok> tokens;
        tokens.reserve(sp.size());
        for (auto& s : sp) { MTok t; t.text = s.text; t.tag = s.tag; t.ws = s.ws ? u32(U" ") : u32(); tokens.push_back(std::move(t)); }
        std::vector<Word> words = retokenize(tokens);
        Ctx ctx;
        static const u32 unk;
        for (size_t wi = words.size(); wi-- > 0;) {
            Word& W = words[wi];
            if (!W.is_list) {
                MTok& w = W.t[0];
                if (!w.has_ph) {
                    Res r = lexicon(w, ctx);
                    if (r.ok) { w.has_ph = true; w.ph = r.ps; w.rating = r.rating; }
                }
                if (!w.has_ph) {
                    u32 ps;
                    if (fallback(w.text, ps)) { w.has_ph = true; w.ph = ps; w.rating = 2; }
                }
                ctx = token_context(ctx, &w, w.has_ph, w.ph, w.text, w.tag);
                continue;
            }
            std::vector<MTok>& w = W.t;
            size_t left = 0, right = w.size();
            bool should_fallback = false;
            while (left < right) {
                bool blocked = false;
                for (size_t k = left; k < right; ++k) if (w[k].has_alias || w[k].has_ph) blocked = true;
                Res r;
                MTok tk;
                if (!blocked) {
                    tk = merge_tokens(w, left, right, nullptr);
                    r = lexicon(tk, ctx);
                }
                if (r.ok) {
                    w[left].has_ph = true; w[left].ph = r.ps; w[left].rating = r.rating;
                    for (size_t k = left + 1; k < right; ++k) { w[k].has_ph = true; w[k].ph.clear(); }
                    ctx = token_context(ctx, &tk, true, r.ps, tk.text, tk.tag);
                    right = left;
                    left = 0;
                } else if (left + 1 < right) {
                    ++left;
                } else {
                    --right;
                    MTok& t = w[right];
                    if (!t.has_ph) {
                        if (std::all_of(t.text.begin(), t.text.end(), [](char32_t c) { return in(SUBTOKEN_JUNKS_S, c); })) {
                            t.has_ph = true; t.ph.clear(); t.rating = 3;
                        } else { should_fallback = true; break; }
                    }
                    left = 0;
                }
            }
            if (should_fallback) {
                MTok tk = merge_tokens(w, 0, w.size(), nullptr);
                u32 ps;
                bool ok = fallback(tk.text, ps);
                w[0].has_ph = ok; w[0].ph = ok ? ps : u32(); w[0].rating = 2;
                for (size_t k = 1; k < w.size(); ++k) { w[k].has_ph = true; w[k].ph.clear(); }
            } else {
                resolve_tokens(w);
            }
        }
        u32 result;
        for (auto& W : words) {
            MTok m = W.is_list ? merge_tokens(W.t, 0, W.t.size(), &unk) : W.t[0];
            if (m.has_ph) {
                replace_char(m.ph, U'ɾ', U'T');
                replace_char(m.ph, U'ʔ', U't');
                result += m.ph;
            }
            result += m.ws;
        }
        return result;
    }

    // phonemize_v1.phonemize + the vocab filter of EnglishG2P.__call__
    std::string post(const u32& ps_in) const {
        u32 ps = strip(ps_in);
        static const std::pair<u32, u32> FOLD[] = {
            {U"̪", U""}, {U"̩", U""}, {U"̍", U""}, {U"̯", U""}, {U"ɭ", U"l"}, {U"ɫ", U"l"},
            {U"ɵ", U"ø"}, {U"ε", U"ɛ"}, {U"\u0005", U""}, {U"\u0006", U""}, {U"\u0007", U""},
            {U"\u0008", U""}, {U"{", U""}, {U"}", U""}, {U"[", U""}, {U"]", U""}, {U"¿", U""}, {U"¡", U""},
            {U" ", U" "}};
        for (auto& f : FOLD) if (ps.find(f.first) != u32::npos) replace_all(ps, f.first, f.second);
        u32 out;
        for (char32_t c : ps) if (vocab.count(c)) out += c;
        return u32_to_utf8(out);
    }

    std::string g2p_text(const std::string& normalized) {
        u32 t = utf8_to_u32(normalized);
        t = u32(t.begin() + (std::find_if(t.begin(), t.end(), [](char32_t c) { return !is_space(c); }) - t.begin()), t.end());
        std::vector<SpTok> toks = spacy_tokenize(t);
        tag(toks);
        return post(misaki(toks));
    }
};

// ---------------------------------------------------------------------------------------------------------------
EnglishFrontend::~EnglishFrontend() = default;

std::unique_ptr<EnglishFrontend> EnglishFrontend::load(const std::string& g2p_dir, const std::string& data_dir,
                                                       const std::string& vocab_json, const std::string& phonemizer,
                                                       std::string& err) {
    auto fe = std::unique_ptr<EnglishFrontend>(new EnglishFrontend());
    fe->impl_ = std::make_unique<Impl>();
    Impl& I = *fe->impl_;
    if (phonemizer != "v1" && phonemizer != "v2") { err = "phonemizer must be \"v1\" or \"v2\""; return nullptr; }
    I.v2 = phonemizer == "v2";
    const std::string dd = data_dir.empty() ? g2p_dir : data_dir;
    if (!I.gold.load(dd + "/us_gold.json", true, err)) return nullptr;
    if (!I.silver.load(dd + "/us_silver.json", false, err)) return nullptr;
    if (!I.cmu.load(g2p_dir + "/cmu_misaki_us.json", false, err)) return nullptr;
    {
        std::string text;
        if (!read_file(dd + "/spacy_en_exceptions.json", text)) { err = "cannot read " + dd + "/spacy_en_exceptions.json"; return nullptr; }
        json::Value v;
        if (!json::parse(text, v) || v.kind != json::Value::Obj) { err = "bad spacy_en_exceptions.json"; return nullptr; }
        for (auto& kv : v.obj) {
            std::vector<std::string> parts;
            for (auto& p : kv.second.arr) if (p.kind == json::Value::Str) parts.push_back(p.str);
            if (!parts.empty()) I.specials[kv.first] = std::move(parts);
        }
    }
    {
        std::string text;
        if (!read_file(vocab_json, text)) { err = "cannot read " + vocab_json; return nullptr; }
        json::Value v;
        if (!json::parse(text, v) || v.kind != json::Value::Obj) { err = "bad vocab.json"; return nullptr; }
        for (auto& kv : v.obj) {
            u32 k = utf8_to_u32(kv.first);
            if (k.size() == 1 && k[0] != 0x289) I.vocab.insert(k[0]);   // English filters with the v1 set (no U+0289)
        }
    }
    {
        I.bart_dir = g2p_dir + "/g2p_bart_en_us";
        std::string text;
        if (!read_file(I.bart_dir + "/config.json", text)) { err = "cannot read " + I.bart_dir + "/config.json"; return nullptr; }
        json::Value cfg;
        if (!json::parse(text, cfg) || cfg.kind != json::Value::Obj) { err = "bad BART config.json"; return nullptr; }
        const json::Value* g = cfg.get("grapheme_chars");
        const json::Value* p = cfg.get("phoneme_chars");
        if (!g || !p || g->kind != json::Value::Str || p->kind != json::Value::Str) { err = "BART config lacks grapheme_chars/phoneme_chars"; return nullptr; }
        u32 gc = utf8_to_u32(g->str), pc = utf8_to_u32(p->str);
        for (size_t i = 4; i < gc.size(); ++i) I.g2i[gc[i]] = (int)i;
        for (size_t i = 4; i < pc.size(); ++i) I.i2p[(int)i] = pc[i];
        // loaded here, not on first use as in Python: errors show at load, and the first unknown word is not slow
        if (!I.bart.load(I.bart_dir, err)) return nullptr;
        I.bart_loaded = true;
        tts_normalize("warm 1");   // builds the normaliser's regular expressions now, not on the first call
    }
    for (auto& e : I.gold.e) {
        if (e.dict < 0) continue;
        unsigned keys = 0;
        for (auto& d : I.gold.dicts[e.dict]) {
            std::string_view t = std::string_view(I.gold.arena).substr(d.to, d.tl);
            if (t == "NOUN") keys |= Impl::K_NOUN;
            else if (t == "VERB") keys |= Impl::K_VERB;
            else if (t == "ADJ") keys |= Impl::K_ADJ;
            else if (t == "ADV") keys |= Impl::K_ADV;
            else if (t == "VBD") keys |= Impl::K_VBD;
            else if (t == "VBN") keys |= Impl::K_VBN;
            else if (t == "VBP") keys |= Impl::K_VBP;
        }
        if (keys) I.homographs_nv[u32_to_utf8(lower(utf8_to_u32(std::string(I.gold.key(e)))))] |= keys;
    }
    return fe;
}

std::string EnglishFrontend::normalize(std::string_view text) { return tts_normalize(text); }

std::vector<SpTok> EnglishFrontend::tokenize(std::string_view normalized) {
    u32 t = utf8_to_u32(normalized);
    size_t a = 0;
    while (a < t.size() && is_space(t[a])) ++a;
    auto toks = impl_->spacy_tokenize(t.substr(a));
    impl_->tag(toks);
    return toks;
}

std::string EnglishFrontend::g2p_tokens(std::vector<SpTok> tokens) { return impl_->post(impl_->misaki(tokens)); }

const Timing& EnglishFrontend::timing() const { return impl_->timing; }

std::string EnglishFrontend::bart_phonemes(std::string_view word) {
    impl_->bart_memo.clear();
    return u32_to_utf8(impl_->bart_word(utf8_to_u32(word)));
}

void EnglishFrontend::set_bart_fp32(bool on) {
    impl_->bart.set_fp32(on);
    impl_->bart_memo.clear();
}

size_t EnglishFrontend::memory_bytes() const {
    const Impl& I = *impl_;
    return I.gold.bytes() + I.silver.bytes() + I.cmu.bytes() + I.bart.weight_bytes();
}

namespace {
constexpr size_t MAX_PHON = 350;

// Python len() of a UTF-8 string
size_t cps(const std::string& s) { size_t n = 0; for (unsigned char c : s) n += (c & 0xC0) != 0x80; return n; }

// SENT_MULTI = (?<=[.!?…।])\s+|(?<=[。！？]) ; re.split
std::vector<u32> split_sentences(const u32& t) {
    std::vector<u32> out;
    size_t st = 0, i = 0;
    while (i < t.size()) {
        if (is_space(t[i]) && i > 0 && in(U".!?…।", t[i - 1])) {
            size_t j = i;
            while (j < t.size() && is_space(t[j])) ++j;
            out.push_back(t.substr(st, i - st));
            st = i = j;
            continue;
        }
        if (in(U"。！？", t[i])) {
            out.push_back(t.substr(st, i + 1 - st));
            st = i + 1;
        }
        ++i;
    }
    out.push_back(t.substr(st));
    return out;
}

void wrap(u32 ps, std::vector<u32>& out) {
    while (ps.size() > MAX_PHON) {
        size_t cut = ps.rfind(U' ', MAX_PHON - 1);
        if (cut == u32::npos || cut == 0) cut = MAX_PHON;
        out.push_back(ps.substr(0, cut));
        ps = strip(u32v(ps).substr(cut));
    }
    if (!ps.empty()) out.push_back(ps);
}

std::string fold_extra(const std::string& s) {
    u32 t = utf8_to_u32(s);
    u32 o;
    for (char32_t c : t) {
        if (c == 0x2009 || c == U'\t') o += U' ';
        else if (c == U'«' || c == U'»' || c == U'‘' || c == U'’' || c == U'\'' || c == U'~' || c == U'·' || c == U'#') {}
        else if (c == U'g') o += (char32_t)0x261;
        else o += c;
    }
    return u32_to_utf8(o);
}
}  // namespace

bool EnglishFrontend::text_to_chunks(std::string_view text, std::vector<std::string>& chunks, std::string& err) {
    impl_->timing = Timing{};
    return normalized_to_chunks(tts_normalize(text), chunks, err);
}

bool EnglishFrontend::normalized_to_chunks(std::string_view normalized, std::vector<std::string>& chunks,
                                           std::string& err) {
    chunks.clear();
    (void)err;
    const std::string norm(normalized);
    std::string whole = impl_->g2p_text(norm);
    if (cps(whole) <= MAX_PHON) {
        if (!whole.empty()) chunks.push_back(whole);
    } else {
        u32 cur;
        for (auto& sent : split_sentences(utf8_to_u32(norm))) {
            if (strip(sent).empty()) continue;
            u32 ps = utf8_to_u32(impl_->g2p_text(u32_to_utf8(sent)));
            std::vector<u32> pieces;
            wrap(ps, pieces);
            for (auto& p : pieces) {
                if (!cur.empty() && cur.size() + 1 + p.size() > MAX_PHON) {
                    chunks.push_back(u32_to_utf8(cur));
                    cur = p;
                } else {
                    u32 c2 = cur; c2 += U' '; c2 += p;
                    cur = strip(c2);
                }
            }
        }
        if (!cur.empty()) chunks.push_back(u32_to_utf8(cur));
    }
    if (impl_->v2) for (auto& c : chunks) c = fold_extra(c);
    return true;
}

}  // namespace kt_tinyen
