// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Original implementation; reads espeak-ng's data files (GPL-3.0, downloaded at run time, never bundled).
// The ASCII -> IPA table is from KittenML's CEPhonemizer (Apache-2.0).
#include "frontend/espeak_lang.h"
#include "frontend/lang/sw.h"
#include "frontend/lang/pt.hpp"
#include "frontend/lang/sv.h"
#include "frontend/lang/fr.h"
#include "frontend/lang/ko.h"
#include "frontend/lang/fi.h"
#include "frontend/lang/hi.hpp"
#include "frontend/lang/da.h"

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace kt_fe {
namespace {

// ---------------------------------------------------------------- UTF-8
void put_utf8(std::string& o, char32_t c) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
}
std::u32string dec(const std::string& s) {
    std::u32string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        char32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { i++; continue; }
        if (i + n > s.size()) break;
        bool ok = true;
        for (int k = 1; k < n; k++) {
            unsigned char cc = (unsigned char)s[i + k];
            if ((cc >> 6) != 2) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        i += ok ? n : 1;
        if (ok) o += cp;
    }
    return o;
}
std::string enc(const std::u32string& s) { std::string o; for (char32_t c : s) put_utf8(o, c); return o; }

char32_t lower(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x17F) {
        if (c == 0x130) return 'i';
        if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c + 1 : c;
        return (c & 1) ? c : c + 1;
    }
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}
bool is_upper(char32_t c) { return lower(c) != c; }

// Base letter of a Latin letter with a diacritic (lower case), else the letter itself.
char32_t base_letter(char32_t c) {
    static const char* l1 = "aaaaaaaceeeeiiiidnooooo/ouuuuyty";   // 0xE0..0xFF
    if (c >= 0xE0 && c <= 0xFF) { char b = l1[c - 0xE0]; return (b == '/' || b == 't' || b == 'd') ? c : (char32_t)b; }
    if (c >= 0x100 && c <= 0x17F) {
        static const char* la = "aaaaaaccccccccddddeeeeeeeeeegggggggghhhhiiiiiiiiiijjjjkkklllllllllllnnnnnnnnnoooooooorrrrrrssssssssttttttuuuuuuuuuuuuwwyyyzzzzzzs";
        size_t k = c - 0x100;
        if (k < strlen(la)) return (char32_t)la[k];
    }
    return c;
}
bool is_alpha(char32_t c) {
    if (c < 0x80) return (c | 32) >= 'a' && (c | 32) <= 'z';
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;
    if (c >= 0x370 && c <= 0x3FF) return true;
    if (c >= 0x400 && c <= 0x52F) return true;
    if (c >= 0x900 && c <= 0x97F) return !(c >= 0x964 && c <= 0x96F);   // Devanagari, not danda / digits
    if (c >= 0x1E00 && c <= 0x1EFF) return true;
    if (c >= 0xAC00 && c <= 0xD7A3) return true;
    if (ko::is_jamo(c)) return true;   // Korean, after ko::decompose
    return false;
}
bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }

// Swedish (measured against the Python package's espeak-ng 1.52.0, letter by letter, inside the word "ta?at"):
// the plain letter a Latin Extended-B / IPA letter is read as ('.' / '?': none), and the letters that have no name, for
// which the package spells the whole word ("bokstaven" + the letter's code): all of U+1E00..U+1EFF and three modifier
// letters used in transliterated names.
char32_t sv_plain_letter(char32_t c) {
    static const char* t180 = "bbbb??occdddddeeeff.?h?.ccll..n.oooopp.??.sttt.uu?.yyss.ssss???wtttcdddlllnnnaaiioouuuuuuuuuueaaaaaaggggkkoooossjdddgghwnnaaaaooaaaaeeeeiiiioooorrrruuuussttyyhhndoossaaeeooooooooyylntjdc?ccl?ss??b..eejjccrryy";   // U+0180..U+024F
    static const char* t250 = "aaabocddeeeeee?..?.?.....?......?......??.................?............??.?..............???????";   // U+0250..U+02AF
    char b = (c >= 0x180 && c <= 0x24F) ? t180[c - 0x180] : (c >= 0x250 && c <= 0x2AF) ? t250[c - 0x250] : '.';
    if (c == 0xF0) b = 'd'; else if (c == 0xFE) b = 't'; else if (c == 0xDF || c == 0x1E9E) b = 's';
    return (b == '.' || b == '?') ? c : (char32_t)b;
}
bool sv_unnamed_letter(char32_t c) { return (c >= 0x1E00 && c <= 0x1EFF && c != 0x1E9E) || c == 0x2BB || c == 0x2BE || c == 0x2BF; }
char32_t sv_unnamed_lower(char32_t c) { return (c >= 0x1E00 && c <= 0x1EFF && !(c >= 0x1E96 && c <= 0x1E9F) && !(c & 1)) ? c + 1 : c; }
bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xA0 || c == 0x202F || c == 0x2009; }

// ASCII -> IPA (KittenML CEPhonemizer ipa_table.h, Apache-2.0)
const uint32_t kAsciiIpa[96] = {
    0x20, 0x21, 0x22, 0x2b0, 0x24, 0x25, 0xe6, 0x2c8, 0x28, 0x29, 0x27e, 0x2b, 0x2cc, 0x2d, 0x2e, 0x2f,
    0x252, 0x31, 0x32, 0x25c, 0x34, 0x35, 0x36, 0x37, 0x275, 0x39, 0x2d0, 0x2b2, 0x3c, 0x3d, 0x3e, 0x294,
    0x259, 0x251, 0x3b2, 0xe7, 0xf0, 0x25b, 0x46, 0x262, 0x127, 0x26a, 0x25f, 0x4b, 0x26b, 0x271, 0x14b, 0x254,
    0x3a6, 0x263, 0x280, 0x283, 0x3b8, 0x28a, 0x28c, 0x153, 0x3c7, 0xf8, 0x292, 0x32a, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x261, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x303, 0x7f};

std::string strip_ws(const std::string& s) {
    size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------- phoneme tables (phsource)
enum PhType { T_OTHER, T_PAUSE, T_STRESS, T_VOWEL, T_LIQUID, T_NASAL, T_STOP, T_FRIC, T_VIRTUAL };
enum CondWhich { C_THIS, C_PREV, C_PREVW, C_PREV2W, C_NEXT, C_NEXT2, C_NEXTW, C_NEXT2W, C_NEXT3W, C_NEXTVOWEL,
                 C_PREVVOWEL, C_FALSE };
struct Cond { int which = C_FALSE; bool neg = false; bool or_prev = false; std::string attr; };   // or_prev: "OR" before it (Danish)
enum StKind { S_IF, S_CHANGE, S_CH_NOTSTRESSED, S_CH_UNSTRESSED, S_CH_DIMINISHED, S_CH_STRESSED, S_APPEND, S_CALL,
              S_IPA, S_RETURN, S_SOUND };
struct Stmt {
    int kind = S_RETURN;
    std::vector<Cond> conds;
    bool is_or = false;
    std::vector<Stmt> body, els;
    std::string arg;   // phoneme name, ipa string, or table/phoneme for CALL
};
struct PhDef {
    std::string name;
    int table = 0;
    int type = T_OTHER;
    int stress_type = -1;
    bool lng = false;   // long vowel ("lng")
    bool voiced = false, rhotic = false, palatal = false, unstressed = false, nonsyl = false, flag1 = false, flag2 = false;
    bool velar = false;
    bool has_ipa = false;
    std::string ipa;   // final IPA (explicit, or derived from the name)
    std::string starttype, endtype;
    std::string voicingswitch;
    std::vector<Stmt> prog;
};
struct PhTable { std::string name; int parent = -1; std::unordered_map<std::string, int> idx; };

std::string unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) { o += s[++i]; continue; }
        o += s[i];
    }
    return o;
}
std::string ipa_from_spec(const std::string& s) {   // "aU+0303" -> a + U+0303; "NULL" -> ""
    if (s == "NULL") return "";
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == 'U' && i + 6 <= s.size() && s[i + 1] == '+') {
            unsigned v = 0; bool ok = true;
            for (int k = 2; k < 6; k++) {
                char h = s[i + k]; v <<= 4;
                if (h >= '0' && h <= '9') v |= h - '0'; else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
                else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10; else ok = false;
            }
            if (ok) { put_utf8(o, v); i += 5; continue; }
        }
        o += s[i];
    }
    return o;
}
std::string default_ipa(const std::string& name, bool hash_aspirated = false) {
    std::string r;
    bool first = true;
    for (char c : name) {
        unsigned char uc = (unsigned char)c;
        if (c == '/') break;
        if (!first && c >= '0' && c <= '9') continue;
        if (c == '#' && !first && !hash_aspirated) break;   // Hindi: k# -> kʰ
        if (c == '|') continue;
        if (c == '_' && first) break;
        if (uc >= 0x20 && uc < 0x80) put_utf8(r, kAsciiIpa[uc - 0x20]);
        else r += c;
        first = false;
    }
    return r;
}

class PhonemeSet {
public:
    std::vector<PhTable> tables;
    std::vector<PhDef> defs;
    int lang_table = -1;
    size_t max_name = 1;
    bool call_type = false;   // a phoneme defined only as "CALL x" takes x's kind (English 3 = CALL @: a vowel)
    bool hash_aspirated = false;   // '#' after the first letter of a name is aspiration (Hindi tables)
    std::vector<std::string> extra_tables;   // tables outside the language's chain that it imports from (Hindi: consonants)
    bool import_vowels = false;   // German: "import_phoneme <table>/<vowel name>" defines a vowel

    int find_table(const std::string& n) const {
        for (size_t i = 0; i < tables.size(); i++) if (tables[i].name == n) return (int)i;
        return -1;
    }
    int find(const std::string& name, int table) const {
        for (int t = table; t >= 0; t = tables[t].parent) {
            auto it = tables[t].idx.find(name);
            if (it != tables[t].idx.end()) return it->second;
        }
        return -1;
    }
    int find(const std::string& name) const { return find(name, lang_table); }
    // "phoneme 3 / CALL @ / ipa ə" (English): a phoneme that only calls a vowel is that vowel (it counts as a syllable)
    void inherit_called_vowels() {
        for (auto& d : defs) {
            if (d.type != T_OTHER || d.prog.empty() || d.prog[0].kind != S_CALL) continue;
            int j = find(d.prog[0].arg, d.table);
            if (j < 0 || defs[j].type != T_VOWEL) continue;
            d.type = T_VOWEL; d.unstressed = defs[j].unstressed; d.starttype = defs[j].starttype; d.endtype = defs[j].endtype;
        }
    }

    // the definition of a phoneme name that is nearest to the base table (its place in the order of definition)
    int base_def(const std::string& name) const {
        int r = -1;
        for (int t = lang_table; t >= 0; t = tables[t].parent) {
            auto it = tables[t].idx.find(name);
            if (it != tables[t].idx.end()) r = it->second;
        }
        return r;
    }

    std::vector<std::string> import_tables;   // set before load(): other tables to read as well (see load)
    bool load(const std::string& dir, const std::string& table_name, std::string* err) {
        std::string master = read(dir + "/phonemes");
        if (master.empty()) { *err = "cannot read " + dir + "/phonemes"; return false; }
        // Pass 1: which tables are needed (the chain of table_name).
        std::vector<std::pair<std::string, std::string>> decl;   // name, parent
        {
            std::istringstream is(master); std::string line;
            while (std::getline(is, line)) {
                auto t = toks(line);
                if (t.size() >= 2 && t[0] == "phonemetable") decl.push_back({t[1], t.size() > 2 ? t[2] : ""});
            }
        }
        std::vector<std::string> chain;
        for (std::string n = table_name; !n.empty();) {
            chain.push_back(n);
            std::string p;
            for (auto& d : decl) if (d.first == n) p = d.second;
            n = p;
            if (chain.size() > 10) break;
        }
        // tables that the language's own table imports phonemes from (French: "import_phoneme en/I"), with their parents
        for (const std::string& extra : import_tables)
            for (std::string n = extra; !n.empty() && chain.size() < 24;) {
                if (std::find(chain.begin(), chain.end(), n) == chain.end()) chain.push_back(n);
                std::string p;
                for (auto& d : decl) if (d.first == n) p = d.second;
                n = p;
            }
        for (auto& x : extra_tables) chain.push_back(x);
        auto needed = [&](const std::string& n) { return std::find(chain.begin(), chain.end(), n) != chain.end(); };
        // Pass 2: tokens of the needed sections ("" = the stress phonemes before the first table).
        tables.push_back({"", -1, {}});
        int cur = 0;
        std::vector<std::string> tk;
        std::istringstream is(master); std::string line;
        auto flush = [&]() { parse_tokens(tk, cur); tk.clear(); };
        while (std::getline(is, line)) {
            auto t = toks(line);
            if (t.empty()) continue;
            if (t[0] == "phonemetable") {
                flush();
                if (needed(t[1])) {
                    PhTable pt; pt.name = t[1];
                    pt.parent = t.size() > 2 ? find_table(t[2]) : -1;
                    if (pt.parent < 0) pt.parent = 0;
                    tables.push_back(pt);
                    cur = (int)tables.size() - 1;
                } else cur = -1;
                continue;
            }
            if (cur < 0) continue;
            if (t[0] == "include" && t.size() > 1) {
                // Only plain file names inside the phsource folder (no paths out of it).
                if (t[1].find_first_of("/\\") != std::string::npos || t[1].find("..") != std::string::npos) {
                    *err = "bad include name " + t[1]; return false;
                }
                std::string inc = read(dir + "/" + t[1]);
                if (inc.empty()) { *err = "cannot read " + dir + "/" + t[1]; return false; }
                std::istringstream is2(inc); std::string l2;
                while (std::getline(is2, l2)) for (auto& x : toks(l2)) tk.push_back(x);
                continue;
            }
            for (auto& x : t) tk.push_back(x);
        }
        flush();
        lang_table = find_table(table_name);
        if (lang_table < 0) { *err = "no phoneme table " + table_name; return false; }
        if (call_type) {
            for (auto& d : defs) {
                if (d.type != T_OTHER || d.prog.empty() || d.prog[0].kind != S_CALL || d.prog[0].arg.find('/') != std::string::npos) continue;
                int j = find(d.prog[0].arg, d.table);
                if (j >= 0) { d.type = defs[j].type; d.unstressed = defs[j].unstressed; d.nonsyl = defs[j].nonsyl; }
            }
        }
        for (auto& d : defs) {
            max_name = std::max(max_name, d.name.size());
            if (!d.has_ipa) d.ipa = default_ipa(d.name, hash_aspirated);
        }
        return true;
    }

    // A phoneme defined only by a CALL to another (en "phoneme 3: CALL @, ipa ə") takes that one's type and
    // features. Used for the English reader of Hindi text (opt-in: the other languages' outputs stay as measured).
    void inherit_called_types() {
        for (auto& d : defs) {
            if (d.type != T_OTHER || d.prog.empty() || d.prog[0].kind != S_CALL) continue;
            const std::string& a = d.prog[0].arg;
            size_t sl = a.find('/');
            int j = sl == std::string::npos ? find(a, d.table) : find(a.substr(sl + 1), find_table(a.substr(0, sl)));
            if (j < 0 || defs[j].type == T_OTHER) continue;
            const PhDef& c = defs[j];
            d.type = c.type; d.lng = c.lng; d.voiced = c.voiced; d.unstressed = c.unstressed; d.nonsyl = c.nonsyl;
            d.flag1 = c.flag1; d.flag2 = c.flag2; d.palatal = c.palatal; d.rhotic = c.rhotic;
            if (d.starttype.empty()) d.starttype = c.starttype;
            if (d.endtype.empty()) d.endtype = c.endtype;
        }
    }

private:
    static std::string read(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return "";
        std::stringstream ss; ss << f.rdbuf();
        return ss.str();
    }
    static std::vector<std::string> toks(const std::string& line) {
        std::string l = line;
        size_t c = l.find("//");
        if (c != std::string::npos) l = l.substr(0, c);
        std::vector<std::string> o;
        std::istringstream is(l); std::string w;
        while (is >> w) o.push_back(w);
        return o;
    }
    static std::string paren_arg(const std::string& t) {   // "ChangePhoneme(x)" -> x
        size_t a = t.find('('), b = t.rfind(')');
        if (a == std::string::npos) return "";
        if (b == std::string::npos || b < a) b = t.size();
        std::string s = t.substr(a + 1, b - a - 1);
        size_t comma = s.find(',');
        if (comma != std::string::npos) s = s.substr(0, comma);
        return unescape(s);
    }
    static bool parse_cond(const std::string& t, Cond& c) {
        static const std::pair<const char*, int> names[] = {
            {"thisPh", C_THIS}, {"prevPh", C_PREV}, {"prevPhW", C_PREVW}, {"prev2PhW", C_PREV2W}, {"nextPh", C_NEXT},
            {"next2Ph", C_NEXT2}, {"nextPhW", C_NEXTW}, {"next2PhW", C_NEXT2W}, {"next3PhW", C_NEXT3W},
            {"nextVowel", C_NEXTVOWEL}, {"prevVowel", C_PREVVOWEL}};
        size_t a = t.find('(');
        std::string f = a == std::string::npos ? t : t.substr(0, a);
        c.which = C_FALSE;
        for (auto& n : names) if (f == n.first) c.which = n.second;
        c.attr = paren_arg(t);
        return true;
    }
    // Parses statements until one of the stop words; returns the stop word met.
    std::string parse_block(const std::vector<std::string>& tk, size_t& i, std::vector<Stmt>& out, bool single = false) {
        while (i < tk.size()) {
            if (single && out.size() > 0) return "";
            const std::string& t = tk[i];
            if (t == "endphoneme" || t == "ENDIF" || t == "ELSE" || t == "ELIF" || t == "ELSEIF") return t == "ELSEIF" ? "ELIF" : t;
            i++;
            if (t == "IF") {
                Stmt s; s.kind = S_IF;
                parse_if(tk, i, s);
                out.push_back(std::move(s));
            } else if (t == "NextVowelStarts") {
                while (i < tk.size() && tk[i] != "EndSwitch" && tk[i] != "endphoneme") i++;
                if (i < tk.size() && tk[i] == "EndSwitch") i++;
            } else if (t.rfind("ChangePhoneme(", 0) == 0) { Stmt s; s.kind = S_CHANGE; s.arg = paren_arg(t); out.push_back(s); }
            else if (t.rfind("ChangeIfNotStressed(", 0) == 0) { Stmt s; s.kind = S_CH_NOTSTRESSED; s.arg = paren_arg(t); out.push_back(s); }
            else if (t.rfind("ChangeIfUnstressed(", 0) == 0) { Stmt s; s.kind = S_CH_UNSTRESSED; s.arg = paren_arg(t); out.push_back(s); }
            else if (t.rfind("ChangeIfDiminished(", 0) == 0) { Stmt s; s.kind = S_CH_DIMINISHED; s.arg = paren_arg(t); out.push_back(s); }
            else if (t.rfind("ChangeIfStressed(", 0) == 0) { Stmt s; s.kind = S_CH_STRESSED; s.arg = paren_arg(t); out.push_back(s); }
            else if (t.rfind("IfNextVowelAppend(", 0) == 0) { Stmt s; s.kind = S_APPEND; s.arg = paren_arg(t); out.push_back(s); }
            else if (t == "CALL" && i < tk.size()) { Stmt s; s.kind = S_CALL; s.arg = unescape(tk[i++]); out.push_back(s); }
            else if (t == "RETURN") { Stmt s; s.kind = S_RETURN; out.push_back(s); }
            else if (t.rfind("FMT(", 0) == 0 || t.rfind("WAV(", 0) == 0) { Stmt s; s.kind = S_SOUND; out.push_back(s); }
            else if (t == "ipa" && i < tk.size()) { Stmt s; s.kind = S_IPA; s.arg = ipa_from_spec(tk[i++]); out.push_back(s); }
        }
        return "";
    }
    void parse_if(const std::vector<std::string>& tk, size_t& i, Stmt& s) {
        bool neg = false, or_next = false;
        while (i < tk.size() && tk[i] != "THEN") {
            const std::string& t = tk[i++];
            if (t == "NOT") { neg = true; continue; }
            if (t == "AND") { s.is_or = false; or_next = false; continue; }
            if (t == "OR") { s.is_or = true; or_next = true; continue; }
            Cond c; parse_cond(t, c); c.neg = neg; neg = false; c.or_prev = or_next; or_next = false;
            s.conds.push_back(c);
        }
        if (i < tk.size()) i++;   // THEN
        std::string stop = parse_block(tk, i, s.body);
        if (stop == "ELIF") {
            i++;   // ELIF / ELSEIF
            Stmt e; e.kind = S_IF;
            parse_if(tk, i, e);   // consumes the shared ENDIF
            s.els.push_back(std::move(e));
            return;
        }
        if (stop == "ELSE") { i++; stop = parse_block(tk, i, s.els); }
        if (stop == "ENDIF") i++;
    }
    void parse_tokens(const std::vector<std::string>& tk, int table) {
        for (size_t i = 0; i < tk.size();) {
            if (tk[i] != "phoneme" || i + 1 >= tk.size()) { i++; continue; }
            PhDef d; d.name = unescape(tk[i + 1]); d.table = table;
            i += 2;
            // properties and program, until endphoneme
            std::vector<Stmt> prog;
            while (i < tk.size() && tk[i] != "endphoneme") {
                const std::string& t = tk[i];
                if (t == "vwl") { d.type = T_VOWEL; i++; }
                else if (t == "pause") { d.type = T_PAUSE; i++; }
                else if (t == "stress") { d.type = T_STRESS; i++; }
                else if (t == "virtual") { d.type = T_VIRTUAL; i++; }
                else if (t == "liquid" || t == "lat" || t == "apr" || t == "flp" || t == "trl") { if (d.type == T_OTHER) d.type = T_LIQUID; i++; }
                else if (t == "nas") { d.type = T_NASAL; i++; }
                else if (t == "stp" || t == "afr") { if (d.type == T_OTHER) d.type = T_STOP; i++; }
                else if (t == "frc" || t == "sib") { if (d.type == T_OTHER) d.type = T_FRIC; i++; }
                else if (t == "vcd") { d.voiced = true; i++; }
                else if (t == "lng") { d.lng = true; i++; }
                else if (t == "vls") { d.voiced = false; i++; }
                else if (t == "rhotic") { d.rhotic = true; i++; }
                else if (t == "palatal" || t == "pal") { d.palatal = true; i++; }
                else if (t == "vel") { d.velar = true; i++; }
                else if (t == "unstressed") { d.unstressed = true; i++; }
                else if (t == "nsy") { d.nonsyl = true; i++; }
                else if (t == "flag1") { d.flag1 = true; i++; }
                else if (t == "flag2") { d.flag2 = true; i++; }
                else if (t == "stress_type" && i + 1 < tk.size()) { d.stress_type = atoi(tk[i + 1].c_str()); i += 2; }
                else if (t == "starttype" && i + 1 < tk.size()) { d.starttype = unescape(tk[i + 1]); i += 2; }
                else if (t == "endtype" && i + 1 < tk.size()) { d.endtype = unescape(tk[i + 1]); i += 2; }
                else if (t == "voicingswitch" && i + 1 < tk.size()) { d.voicingswitch = unescape(tk[i + 1]); i += 2; }
                else if (t == "ipa" && i + 1 < tk.size() && prog.empty()) { d.ipa = ipa_from_spec(tk[i + 1]); d.has_ipa = true; i += 2; }
                else if (t == "import_phoneme" && i + 1 < tk.size()) {
                    std::string ref = unescape(tk[i + 1]);
                    size_t sl = ref.find('/');
                    if (sl != std::string::npos) {
                        int tb = find_table(ref.substr(0, sl));
                        int j = tb >= 0 ? find(ref.substr(sl + 1), tb) : -1;
                        if (j >= 0) {
                            PhDef c = defs[j];
                            std::string nm = d.name;
                            d = c; d.name = nm; d.table = table;
                            prog = c.prog;
                        }
                    }
                    // "import_phoneme en-us/eI": a table that is not loaded; a vowel name makes it a vowel (German [EI]). Opt-in.
                    {
                        const std::string& src = tk[i + 1];
                        size_t sl2 = src.rfind('/');
                        char c0 = sl2 + 1 < src.size() ? src[sl2 == std::string::npos ? 0 : sl2 + 1] : 0;
                        if (import_vowels && d.type == T_OTHER && c0 && std::strchr("aeiouAEIOU@3", c0)) d.type = T_VOWEL;
                    }
                    i += 2;
                } else if (t == "IF" || t == "CALL" || t == "RETURN" || t == "NextVowelStarts" || t == "ipa" ||
                           t.rfind("FMT(", 0) == 0 || t.rfind("WAV(", 0) == 0 ||
                           t.rfind("Change", 0) == 0 || t.rfind("IfNextVowelAppend", 0) == 0) {
                    std::vector<Stmt> one;
                    std::string stop = parse_block(tk, i, one, true);
                    for (auto& x : one) prog.push_back(std::move(x));
                    if (stop == "endphoneme") break;
                    if (one.empty() && t != "NextVowelStarts") i++;
                } else i++;
            }
            if (i < tk.size()) i++;   // endphoneme
            d.prog = std::move(prog);
            if (d.type == T_VOWEL && d.starttype.empty()) d.starttype = d.endtype;
            defs.push_back(std::move(d));
            tables[table].idx[defs.back().name] = (int)defs.size() - 1;
        }
    }
};

// ---------------------------------------------------------------- rules
struct Rule {
    std::u32string pre, match, post;   // pre is stored reversed (scanned right to left)
    std::string phonemes;
    int cond = 0;                      // +N: only with dictrules N; -N: only without
    int suffix = 0, prefix = 0;        // S<n> / P<n>
    std::string affix_flags;           // letters after S<n> / P<n>
};
struct ListEntry {
    std::string phonemes;   // mnemonics ("" = flags only)
    int stress_pos = 0;     // $N
    bool u = false, u_plus = false, only = false, onlys = false, stem = false, atend = false, atstart = false;
    bool capital = false, allcaps = false, text = false, abbrev = false, dot = false, hasdot = false, sentence = false;
    bool verb = false, noun = false, past = false, strend = false, unstressend = false, combine = false;
    bool nounf = false, verbf = false;
    bool brk = false;       // $brk / $pause: a short pause before the word
    bool verbextend = false;
    int alt = 0;
    bool dbl = false;       // $double: the next word's first consonant is doubled
    bool pause = false;     // $pause: a pause before the word (Finnish: no doubling into it)
    int u_n = 0;            // $u1/$u2/$u3
    int cond = 0;
    int nwords = 1;
    int line = 0;           // position in the list files (later lines are searched first)
};

struct LangOpts {
    std::string table;
    std::string voice;
    std::vector<int> dictrules;
    int stress_rule = 2;          // 1 = first syllable, 2 = penultimate, 3 = final (see apply_stress)
    bool final_spanish = false;   // final stress when the word ends in a consonant other than n / s
    bool secondary_from_start = true;
    bool roman = false;
    bool vowel_pause = false;
    bool keep_first_primary = false;
    bool post_secondary = false;     // German: secondary stress on syllables after the primary
    bool num_swap = false;           // German: units 'and' tens; 'ein' + hundert / tausend
    bool hyphen_join = false;        // hyphenated words are written as one word
    bool italian_alt = false;        // $alt: stressed e/o -> E/O; $alt2: E/O -> e/o
    bool omit_one_hundred = false;
    bool secondary_first_only = false;
    bool dim_non_initial = false;        // German: unstressed syllables after the first are 'diminished'
    bool prefix_word = false;            // a removed prefix counts as its own word for the W conditions   // Portuguese: secondary stress on the first syllable only
    bool final_devoice = false;
    bool number_domains = false;     // number parts keep their own stress (no demotion)
    bool apostrophe_split = false;   // "dell'interno" -> "dell'" + "interno", written as one word
    bool syllable_run = false;       // Swahili: "@@" after the match = at least 2 syllables, scored as one context item
    bool phrase_marks = false;       // Swahili: a multi-word list entry with stress marks gets no default stress
    bool swahili = false;            // Swahili: numbers (lang/sw.cpp), "," groups thousands, "." = _dpt, symbol names, word list flags
    bool syllabic_no_secondary = false;  // Swahili: a syllabic nasal (m-) never takes an automatic secondary stress
    // Turkish (all off for the other languages):
    bool stress_before_unstressed = false;   // default stress: the vowel before the first '%' vowel, else the last
    int max_suffix_depth = 3;                // levels of suffix removal in one word (Turkish: 1)
    bool suffix_once = false;                // Turkish: a suffix rule needs a stem; after one removal (or in a
                                             // word listed with $only, or a suffix after a number) suffix rules
                                             // keep their phonemes without removal; a listed stem (with a vowel)
                                             // makes the word $u / 'given'
    bool tie_later_rule = false;             // equal rule scores: the later rule wins ("ra (_S2" / "ra (_N")
    bool length_mark_transparent = false;    // phoneme conditions look through ':' (length of the previous one)
    bool dotless_i = false;                  // capital I is the capital of dotless i (I -> ı, İ -> i)
    bool colon_repeats_vowel_ipa = false;    // ':' after a vowel with its own 'ipa' repeats that IPA
    bool number_suffix_word = false;         // "1537'de": the suffix after a number is its own word (rules only)
    bool hundreds_domain = false;            // "beşyüz": digit and hundred keep their own stress, one word
    bool spell_clusters = false;             // Turkish: letters before an unpronounceable start are spelled
    bool dot_thousands = false;              // "10.000": the dot before 3 digits groups thousands
    bool stressed_decimal = false;           // the word for the decimal point keeps its stress
    bool caps_word_split = false;            // "SANParks" -> "SAN" "Parks"
    bool caps_spell_by_clause = false;       // 2-3 capitals are spelled only when the clause has >= 4 small letters
                                             // and no more capitals than small letters
    bool y_is_vowel = false;                 // 'y' counts as a vowel for "a word without vowels is spelled"
    bool hyphen_one_word = false;            // hyphenated parts are one word for the phoneme conditions
    bool portuguese = false;         // Brazilian Portuguese (pt-br): see the 'portuguese' branches and lang/pt.cpp
    bool keep_all_primary = false;   // Swedish: every primary mark the rules / list give stays primary
    bool trochaic = false;           // Swedish: one pass, left to right: an unmarked syllable between two unstressed
                                     // ones (or a word edge) takes secondary stress, final syllable included; the
                                     // vowels left unstressed stay 'soft' (see PhItem::soft)
    bool y_vowel = false;            // 'y' is a vowel letter
    bool amp_stressable = false;     // '&' in a rule's left context: the letters before hold a vowel the rules did
                                     // not mark unstressed
    bool sv_numbers = false;         // Swedish numbers (lang/sv.cpp), "40 000" / "1.000" groups, "2,5" decimals
    bool symbol_rules = false;       // a symbol with its own rule ("%", "$", "/") is read as a word
    bool prefix_cut = false;         // after a prefix rule (P<n>) the rest is read as a word of its own
    bool attached_context = false;   // rules file: "itut(_" = match "itut", right context "_" (es/it keep the old read)
    bool final_secondary = false;    // the secondary stresses after the primary may fall on the last syllable
    int class_minus = 0;             // Swedish: a letter class (C, A, ...) after the match scores one point less
    bool sv_english = false;         // the English reader for words inside Swedish text: Swedish stress (first
                                     // syllable, trochaic secondaries) on the English rules' phonemes; "$unstressend"
    bool tie_later = false;          // two rules with the same score: the later one in the file wins
    bool caps_read = false;          // a short word in capitals is spelled only in a clause with at least 4 lower-case
                                     // letters and no more capitals than lower-case letters ("hus NBA" -> nba, "huset
                                     // NBA" -> en be a, "AGES Casting AB" -> ab); "SANParks" -> "SAN" "Parks"
    // French (fr-fr) only; the other languages keep their measured behaviour
    bool prefix_word_flags = false;  // l'un, d'une, qu'il: the word after an elision keeps its list flags ($u+, $nounf)
    bool length_if_stressed = false; // a vowel's length mark (:) is written only on a stressed vowel
    bool letter_by_rules = false;
    bool verb_context = false;       // rule pre-context V: the previous word expects a verb ($verbf)
    bool brk_pause = false;          // $brk / $pause words get a pause before them (blocks liaison)
    bool french_numbers = false;     // French number words (see number_lt1000 / number_word); only the last word of a
                                     // number is stressed; "30 000" (plain space) is one number
    bool keep_word_pauses = false;
    bool multiword_phrase = false;   // list entries of up to three words (see the clause)
    bool syllables_from_match = false;   // rule pre '@': syllables counted from the matched letters
    bool tie_by_phonemes = false;    // with tie_later: equal scores are decided by the rules' phonemes (see tie_wins)
    bool english_ce = false;         // English words (_^_EN) are read by CEPhonemizer in British mode (lang/fr.cpp)
    bool hyphen_word = false;        // hyphenated words are one word for the phoneme programs (demi-heure: i -> j)   // pauses inside a word's phonemes (h = _!aS) stay in the clause    // a letter with a diacritic and no rule group of its own is read as its base letter
    bool korean = false;             // Korean: jamo text, own stress, numbers, Latin words in English (lang/ko.h)
    bool finnish = false;            // Finnish: y is a vowel letter, secondary stress (lang/fi.cpp), doubling ($double, _X1)
    bool indic = false;              // Hindi (Devanagari): letter classes, inherent vowel, stress, numbers (lang/hi.cpp)
    bool en_for_indic = false;       // the English reader of a Hindi reader (Latin-letter words in Hindi text)
    bool german_en = false;    // English reader inside German text: the last syllable takes a secondary stress two or more syllables after the primary
    bool german = false;             // German word-level behaviour (see GermanRules notes in translate_word)
    // Danish (lang/da.cpp sets these; all off for the other languages, whose output stays byte-identical)
    bool da_y_vowel = false;         // the letter y is a vowel
    bool seq_conds = false;          // IF a OR b AND c: evaluated left to right ((a OR b) AND c)
    bool length_repeats_ipa = false; // ':' after a phoneme with its own ipa writes that ipa again, else U+02D0
    bool both_groups = false;        // a one-letter group rule can beat the two-letter group's best (higher score)
    bool all_primaries = false;      // several primary stresses in a word all stay primary
    bool da_trochaic = false;        // secondary stress on alternate free syllables, before and after the primary
    bool phrase_rest_unstressed = false;   // the 2nd word of a two-word list entry gets no default stress
    bool at_word_start = false;      // a context "@letters" before the match is also satisfied by the word start
    bool da_numbers = false;         // numbers read as in Danish (lang/da.cpp); '.' groups thousands
    bool stressable_pre = false;     // '&' before the match: a syllable so far that is not marked unstressed (phonemes so far)
    bool phrase_one_domain = false;  // the words of a multi-word list entry share one stress pattern
    // Rule scores as measured on the reference library with synthetic rules: the letter group C is worth 19, the other
    // letter groups 20; an L group before the match 26 at any distance; a run of n '@' 18 + n, once; '&' 19 at any
    // distance; 'N' 1; a rule of a two-letter group 14 more, and it wins a tie with a one-letter group's rule; of two
    // rules of one group with the same score the one with the greater phoneme string (reference order) is used.
    bool measured_scores = false;
    bool prefix_fresh = false;       // after a prefix rule the rest is read as a new word: the prefix is not seen by the
                                     // context before the match ('@', '&', letters, $p_alt)
    bool first_secondary_blocks = false;   // an explicit secondary stress on the first syllable: no default primary
    bool da_tokens = false;          // Danish word splitting: 1960'erne, 3,50 (digits one by one), SANParks, $ ~ and superscript digits
    bool post_after_ph = false;      // a rule line may give its post-context after the phonemes ("eur) o ?o (+++")
    bool list_ph_after_flags = false;   // a list line may give its phonemes after the flags
    bool sentence_atend = false;     // a $sentence list entry is used for the last word of the clause
    bool stem_unstressed = false;    // $u / $u+ of the stem left after a suffix apply to the whole word
    bool syllabic_blocks = false;    // '-' right after a word's first vowel: the word takes no default stress
};

struct PhItem {   // one phoneme of a clause
    int ph;
    int stress = -1;   // vowels: 0 diminished .. 4 primary
    bool word_start = false, word_end = false;
    int word = 0;
    bool join = false;   // no space before this word in the output
    bool number = false;
    int foreign = -1;    // index into foreign_: print that IPA instead of the phoneme
    bool given = false;  // the word's phonemes come from the word list or a number (isTranslationGiven)
    bool soft = false;   // Swedish: an unstressed vowel no mark made unstressed (ChangeIfUnstressed leaves it)
    int hword = -1;      // French: id shared by hyphen-joined words (phoneme programs' same-word tests)
    bool space_before = false;   // Finnish decimal point: a space inside one stress domain
    bool hyphen = false; // first phoneme of a word written with a hyphen after the previous one
};
const int kSoftUnstressed = -5;   // word_stress() value for such a vowel (trochaic languages only)

class EspeakLangImpl : public EspeakLang {
public:
    PhonemeSet ps;
    LangOpts opt;
    std::unordered_map<std::u32string, std::vector<int>> groups;   // group key -> rule indices
    std::vector<Rule> rules;
    std::vector<std::vector<std::u32string>> lgroups = std::vector<std::vector<std::u32string>>(100);
    std::vector<std::pair<std::u32string, std::u32string>> replace;
    std::unordered_map<std::u32string, std::vector<ListEntry>> list;
    size_t max_group_len = 2;
    int list_lines_ = 0;
    mutable bool cur_expect_noun_ = false;   // Portuguese: the previous word has $nounf (stem lookups)
    mutable int stem_alt_ = 0, stem_stress_pos_ = 0;
    mutable bool stem_u_ = false, stem_u_plus_ = false;   // stem_u_plus_: Danish
    mutable bool stem_given_ = false;   // the stem's phonemes came from the word list
    bool clause_caps_spell_ = true;     // caps_spell_by_clause: this clause spells short capital words
    mutable bool force_target_ = false; // stress_before_unstressed: stress the default vowel even if '%'
    mutable bool stem_uplus_ = false;   // Portuguese: the stem's $u+ / $strend
    mutable int stem_un_ = 0;           // Portuguese: the stem's $u1 / $u2 / $u3
    mutable bool caps_clause_ = false;  // Portuguese: the clause is written mostly in capitals (short words are not spelled)
    mutable int rest_stress_pos_ = 0;
    mutable bool prefix_taken_ = false;
    int clause_lower_ = 0, clause_upper_ = 0;   // lower-case letters and capitals in the clause (caps_read)
    bool not_final_ = false;             // the clause given is a word inside a longer clause of another language
    mutable bool stem_listed_ = false;   // the stem before a suffix came from the word list
    mutable bool amp_ok_ = true;         // '&' in a rule: a vowel that may be stressed was already translated
    mutable bool ctx_next_hyph_ = false;
    mutable bool suffix_taken_ = false;   // French: the next word is joined to this one by a hyphen
    mutable const ListEntry* rest_entry_ = nullptr;   // list entry of the word after a removed prefix
    mutable bool num_after_ = false;                     // Finnish numbers: the part after thousands / millions
    mutable bool cur_capital_ = false;
    mutable bool rest_u_ = false, rest_u_plus_ = false;   // German: $u / $strend of the word after a prefix   // German: a suffix was removed while translating the word   // German: the word being translated starts with a capital
    mutable bool suffix_absorbed_ = false;   // German: a prefix found its rest + the removed suffix in the word list
    mutable bool no_switch_ = false;     // German: the word has a flags-only list entry: stems do not switch language
    mutable std::string switch_lang_;                    // "_^_EN": the word is translated as another language
    mutable bool ignore_switch_ = false;
    std::string data_dir_;
    std::map<std::string, std::unique_ptr<EspeakLang>> other_;   // switched-to languages, loaded on first use
    std::map<std::string, bool> other_failed_;
    EspeakLang* other_lang(const std::string& code) {   // espeak's _^_XX language code -> a loaded reader
        static const char* const m[][2] = {{"EN", "en"}, {"DE", "de"}, {"FR", "fr-fr"}, {"IT", "it"}, {"ES", "es"},
                                           {"PT", "pt-br"}};
        const char* voice = nullptr;
        for (auto& kv : m) if (code == kv[0]) voice = kv[1];
        if (opt.trochaic && code == "EN") voice = "en-x-sv";   // English words inside Swedish (see kVoices)
        if (!voice || opt.voice == voice || depth_guard_) return nullptr;
        auto it = other_.find(voice);
        if (it != other_.end()) return it->second.get();
        if (other_failed_[voice]) return nullptr;
        std::string err;
        auto e = EspeakLang::load(voice, data_dir_, &err);
        if (!e) { other_failed_[voice] = true; return nullptr; }
        auto* ei = static_cast<EspeakLangImpl*>(e.get());
        ei->depth_guard_ = true;   // no further switching from there
        ei->embedded_ = opt.english_ce;
        ei->under_korean_ = opt.korean;
        if (opt.korean) ei->inherit_called_types();
        if (opt.korean) ei->tie_later_ = true;
        if (opt.indic) { ei->opt.en_for_indic = true; ei->ps.inherit_called_types(); ei->opt.stress_rule = 1; }
        if (opt.german) {   // English words inside German text (_^_EN): the English reader as the German voice's switch uses it
            ei->ps.inherit_called_vowels();
            ei->opt.german_en = true;
            ei->opt.stress_rule = 1;   // English words in German text: the first syllable takes the stress unless the rules mark one (Hartlepool)
        }
        return (other_[voice] = std::move(e)).get();
    }
    bool depth_guard_ = false;
    bool caps_spell_ok_ = true;   // Danish: this clause spells short capital words
    mutable bool clause_allcaps_ = false;   // German: every word of the clause is in capitals (then "USA" is read as a word)
    std::string dash_digits_, minus_;   // German: the rules' phonemes for "-" between digits / before a number
    // A phoneme defined only as "CALL <other>" (English 3 = CALL @) is the same kind of sound as the one it calls.
    void inherit_called_types() {
        for (auto& d : ps.defs) {
            if (d.type != T_OTHER || d.prog.empty() || d.prog[0].kind != S_CALL || d.prog[0].arg.find('/') != std::string::npos) continue;
            int p = ps.find(d.prog[0].arg);
            if (p < 0 || &ps.defs[p] == &d) continue;
            const PhDef& o = ps.defs[p];
            d.type = o.type; d.unstressed = o.unstressed; d.nonsyl = o.nonsyl; d.voiced = o.voiced;
            if (d.starttype.empty()) d.starttype = o.starttype;
            if (d.endtype.empty()) d.endtype = o.endtype;
        }
    }
    bool under_korean_ = false;       // this English reader reads the Latin-script words of Korean text
    bool tie_later_ = false;          // equal-score rules: the later one wins (set for English inside Korean)
    mutable bool stem_marks_ = false; // apply_rules: a suffix was removed and the stem's phonemes came from the list
    mutable long hash_pos_ = -1;      // match_rule: the 'e' a rule's '#' marks (English: it becomes the special letter E)
    mutable long accent_miss_ = -1;   // English inside Korean: position of a letter with a diacritic that no rule matched
    mutable bool ko_glue_ = false;    // Korean tokenizer: the next word is glued to the previous one
    bool clause_spells_caps_ = true;
    bool embedded_ = false;        // this reader translates the English words of a French clause (set by that reader)
    bool embedded_final_ = true;   // ... and the word in hand is the last word of that clause
    std::unique_ptr<FrEnglishWords> fr_en_;              // French: English words through CEPhonemizer (first use)
    bool fr_en_failed_ = false;
    std::vector<std::string> foreign_;                   // IPA of switched words in the current clause   // flags of a stem found in the list while applying rules
    int ph_primary = -1, ph_secondary = -1, ph_unstressed = -1, ph_pause = -1, ph_wordend = -1;

    size_t memory_bytes() const override {
        size_t b = 0;
        for (auto& r : rules) b += sizeof(Rule) + (r.pre.size() + r.match.size() + r.post.size()) * 4 + r.phonemes.size();
        for (auto& kv : list) for (auto& e : kv.second) b += sizeof(ListEntry) + kv.first.size() * 4 + e.phonemes.size() + 32;
        for (auto& d : ps.defs) b += sizeof(PhDef) + d.name.size() + d.ipa.size() + d.prog.size() * sizeof(Stmt);
        return b;
    }

    bool cond_ok(int c) const {
        if (c == 0) return true;
        bool has = std::find(opt.dictrules.begin(), opt.dictrules.end(), std::abs(c)) != opt.dictrules.end();
        return c > 0 ? has : !has;
    }

    // ---------- loading
    static std::string read(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return "";
        std::stringstream ss; ss << f.rdbuf();
        return ss.str();
    }
    static std::string strip_comment(const std::string& l) {
        size_t c = l.find("//");
        return c == std::string::npos ? l : l.substr(0, c);
    }
    static int parse_cond_prefix(std::string& l) {   // "?1 ..." / "?!1 ..."
        size_t i = l.find_first_not_of(" \t");
        if (i == std::string::npos || l[i] != '?') return 0;
        bool neg = false; size_t j = i + 1;
        if (j < l.size() && l[j] == '!') { neg = true; j++; }
        int n = 0;
        while (j < l.size() && isdigit((unsigned char)l[j])) { if (n < 100000) n = n * 10 + (l[j] - '0'); j++; }
        l = l.substr(j);
        return neg ? -n : n;
    }
    static std::u32string unoctal(const std::u32string& s) {
        std::u32string o;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '\\' && i + 3 < s.size() + 0 && i + 3 <= s.size() - 0 && is_digit(s[i + 1]) && i + 3 < s.size() + 1) {
                if (i + 3 < s.size() + 1 && is_digit(s[i + 2]) && i + 3 < s.size() && is_digit(s[i + 3])) {
                    o += (char32_t)((s[i + 1] - '0') * 64 + (s[i + 2] - '0') * 8 + (s[i + 3] - '0'));
                    i += 3; continue;
                }
            }
            o += s[i];
        }
        return o;
    }

    bool load_rules(const std::string& path, std::string* err) {
        std::string txt = read(path);
        if (txt.empty()) { *err = "cannot read " + path; return false; }
        std::istringstream is(txt); std::string line;
        std::u32string group; bool in_group = false, in_replace = false;
        while (std::getline(is, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string l = strip_comment(line);
            int cond = parse_cond_prefix(l);
            std::istringstream ws(l); std::vector<std::string> t; std::string w;
            while (ws >> w) t.push_back(w);
            if (t.empty()) continue;
            if (t[0][0] == '.' && t[0].size() > 1) {   // a directive; "." alone is a rule for the full stop
                in_replace = false; in_group = false;
                if (t[0] == ".group") {
                    group = t.size() > 1 ? dec(t[1]) : U"";
                    in_group = true;
                } else if (t[0] == ".replace") {
                    in_replace = true;
                } else if (t[0].size() == 4 && t[0][1] == 'L') {
                    int n = atoi(t[0].c_str() + 2);
                    if (n > 0 && n < 100) for (size_t k = 1; k < t.size(); k++) lgroups[n].push_back(t[k] == "~" ? U"" : dec(t[k]));
                }
                continue;
            }
            if (in_replace) {
                if (t.size() >= 2 && cond_ok(cond)) replace.push_back({dec(t[0]), dec(t[1])});
                continue;
            }
            if (!in_group) continue;
            Rule r; r.cond = cond;
            size_t k = 0;
            std::u32string pre, match, post;
            for (size_t q = 0; q < t.size() && q < 4; q++) {
                if (t[q].back() == ')' && t[q].size() > 1 && t[q][0] != '(') { pre = dec(t[q].substr(0, t[q].size() - 1)); k = q + 1; break; }
                if (t[q] == ")" && q > 0) { pre = dec(t[q - 1]); k = q + 1; break; }
            }
            if (k >= t.size()) continue;
            match = dec(t[k++]);
            if (opt.finnish && post.empty()) {   // "e(lla_": the post context written without a space
                size_t par = match.find(U'(');
                if (par != std::u32string::npos && par > 0) { post = match.substr(par + 1); match = match.substr(0, par); }
            }
            size_t open = match.find('(');   // "itut(_": the right context written without a space
            if (opt.attached_context && open != std::u32string::npos && open > 0) { post = match.substr(open + 1); match = match.substr(0, open); }
            else if (k < t.size() && t[k][0] == '(') { post = dec(t[k].substr(1)); k++; }
            else if (match.size() > 3 && match.find(U'(') != std::u32string::npos && match.find(U'(') > 0 && match.compare(match.size() - 2, 2, U"@P") == 0) {   // "computer(@P": no space before the post context
                size_t lp = match.find(U'(');
                post = match.substr(lp + 1); match = match.substr(0, lp);
            }
            std::string ph = k < t.size() ? t[k] : "";
            if (opt.post_after_ph && post.empty() && k + 1 < t.size() && t[k + 1].size() > 1 && t[k + 1][0] == '(') post = dec(t[k + 1].substr(1));
            r.match = unoctal(match);
            pre = unoctal(pre);
            std::reverse(pre.begin(), pre.end());
            r.pre = pre;
            // trailing S<n>.. / P<n>.. in post
            for (size_t p = 0; p < post.size(); p++) {
                if ((post[p] == 'S' || post[p] == 'P') && p + 1 < post.size() && is_digit(post[p + 1])) {
                    int n = 0; size_t q = p + 1;
                    while (q < post.size() && is_digit(post[q])) n = n * 10 + (post[q++] - '0');
                    if (post[p] == 'S') r.suffix = n; else r.prefix = n;
                    while (q < post.size() && post[q] >= 'a' && post[q] <= 'z') r.affix_flags += (char)post[q++];
                    post = post.substr(0, p);
                    break;
                }
            }
            if (r.prefix == 0 && r.suffix == 0 && post.size() >= 2 && post.back() == 'P' && post[post.size() - 2] == '@') {
                r.prefix = (int)unoctal(match).size();   // "computer(@P": a bare P = the whole matched text is the prefix
                post.pop_back();
            }
            r.post = unoctal(post);
            r.phonemes = ph;
            rules.push_back(r);
            std::u32string key = group;
            groups[key].push_back((int)rules.size() - 1);
        }
        for (auto& g : lgroups) std::stable_sort(g.begin(), g.end(), [](auto& a, auto& b) { return a.size() > b.size(); });
        return true;
    }

    // symbols_only (German "de_emoji"): only one-character keys below U+2190 that the word list does not name yet (currency
    // signs, fractions); their text may be several words ("drei Viertel").
    bool load_list(const std::string& path, std::string* err, bool required, bool symbols_only = false) {
        std::string txt = read(path);
        if (txt.empty()) { if (required) { *err = "cannot read " + path; return false; } return true; }
        std::istringstream is(txt); std::string line;
        bool textmode = false;
        while (std::getline(is, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string l = strip_comment(line);
            int cond = parse_cond_prefix(l);
            size_t a = l.find_first_not_of(" \t");
            if (a == std::string::npos) continue;
            l = l.substr(a);
            if (l.rfind("$textmode", 0) == 0) { textmode = true; continue; }
            if (l.rfind("$phonememode", 0) == 0) { textmode = false; continue; }
            std::string key; std::vector<std::string> t;
            ListEntry e; e.cond = cond;
            if (l[0] == '(') {
                size_t c = l.find(')');
                if (c == std::string::npos) continue;
                key = l.substr(1, c - 1);
                l = l.substr(c + 1);
                std::istringstream ks(key); std::string w; std::string norm; int n = 0;
                while (ks >> w) { if (n++) norm += ' '; norm += w; }
                key = norm; e.nwords = n;
            } else {
                std::istringstream ws(l); ws >> key;
                l = l.substr(l.find(key) + key.size());
            }
            std::istringstream ws(l); std::string w;
            while (ws >> w) t.push_back(w);
            size_t k = 0;
            if (k < t.size() && t[k][0] != '$') e.phonemes = t[k++];
            if (symbols_only) while (k < t.size() && t[k][0] != '$') e.phonemes += " " + t[k++];
            bool any_text = textmode;
            for (; k < t.size(); k++) {
                const std::string& f = t[k];
                if (f.size() == 2 && f[0] == '$' && isdigit((unsigned char)f[1])) e.stress_pos = f[1] - '0';
                else if (f == "$u") e.u = true;
                else if (f == "$u+") { e.u = true; e.u_plus = true; }
                else if (f == "$u1" || f == "$u2" || f == "$u3") { e.u = true; e.u_n = f[2] - '0'; }
                else if (f == "$u1+" || f == "$u2+" || f == "$u3+") { e.u = true; e.u_plus = true; e.u_n = f[2] - '0'; }
                else if (f == "$only") e.only = true;
                else if (f == "$onlys") e.onlys = true;
                else if (f == "$stem") e.stem = true;
                else if (f == "$atend") e.atend = true;
                else if (f == "$atstart") e.atstart = true;
                else if (f == "$capital") e.capital = true;
                else if (f == "$allcaps") e.allcaps = true;
                else if (f == "$text") any_text = true;
                else if (f == "$abbrev") e.abbrev = true;
                else if (f == "$dot") e.dot = true;
                else if (f == "$hasdot") { e.hasdot = true; e.dot = true; }
                else if (f == "$sentence") e.sentence = true;
                else if (f == "$verb") e.verb = true;
                else if (f == "$noun") e.noun = true;
                else if (f == "$past") e.past = true;
                else if (f == "$strend" || f == "$strend2") e.strend = true;
                else if (f == "$unstressend") e.unstressend = true;
                else if (f == "$combine") e.combine = true;
                else if (f == "$nounf") e.nounf = true;
                else if (f == "$verbf") e.verbf = true;
                else if (opt.portuguese && f[0] != '$' && e.phonemes.empty()) e.phonemes = f;   // phonemes after flags
                else if (opt.list_ph_after_flags && f[0] != '$' && e.phonemes.empty()) e.phonemes = f;   // Danish: "sig $atend $sentence sAj"
                else if (f == "$double") e.dbl = true;
                else if (opt.finnish && f == "$pause") e.pause = true;
                else if (f == "$verbextend") e.verbextend = true;
                else if (f == "$brk" || f == "$pause") e.brk = true;
                else if (f == "$alt" || f == "$alt1") e.alt |= 1;
                else if (f.rfind("$alt", 0) == 0 && f.size() == 5) e.alt |= 1 << (f[4] - '1');
            }
            e.text = any_text;
            if (!cond_ok(cond)) continue;
            std::u32string k32 = dec(key);
            if (opt.korean) { k32 = ko::decompose(k32); if (e.text) e.phonemes = enc(ko::decompose(dec(e.phonemes))); }
            for (auto& c : k32) c = lower(c);
            if (symbols_only && (k32.size() != 1 || k32[0] >= 0x2190 || is_alpha(k32[0]) || is_digit(k32[0]) || list.count(k32))) continue;
            e.line = ++list_lines_;
            list[k32].push_back(e);
        }
        return true;
    }

    // ---------- phoneme string -> phoneme indices (with stress phonemes)
    void encode_ph(const std::string& s, std::vector<int>& out) const {
        size_t i = 0;
        while (i < s.size()) {
            int best = -1; size_t bl = 0;
            for (size_t L = std::min(ps.max_name, s.size() - i); L >= 1; L--) {
                int j = ps.find(s.substr(i, L));
                if (j >= 0) { best = j; bl = L; break; }
            }
            if (best < 0) {
                if (s[i] == '\x02') out.push_back(-10); else if (s[i] == '\x03') out.push_back(-11); else if (s[i] == '\x04') out.push_back(-12);
                else if (s[i] == '\x06') out.push_back(-13);
                else if (s[i] == '\x0b') out.push_back(-14);   // Swahili numbers: a space inside one stress domain
                i++; continue;
            }
            out.push_back(best);
            i += bl;
        }
    }

    // ---------- letters
    bool is_vowel_letter(char32_t c) const {
        if (opt.indic && hi::is_devanagari(c)) return hi::is_vowel(c);
        char32_t b = base_letter(lower(c));
        if (b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u') return true;
        if (b == 'y' && opt.portuguese) return true;
        if (b == 'y' && opt.y_vowel) return true;
        if (b == 'y' && opt.german) return true;   // German: y is a vowel letter (Olympia)
        if (opt.da_y_vowel && (c == 'y' || c == 0xFD || c == 0xFF)) return true;   // Danish
        if (c == 0xE6 || c == 0xF8 || c == 0x153) return true;
        if (opt.finnish && b == 'y') return true;
        if (c >= 0x430 && c <= 0x44F) { static const std::u32string v = U"аеёиоуыэюя"; return v.find(c) != std::u32string::npos; }
        if (c == 0x451) return true;
        if (ko::is_vowel_jamo(c)) return true;
        return false;
    }
    bool in_set(char ctx, char32_t c) const {
        if (!is_alpha(c)) return false;
        if (opt.indic && hi::is_devanagari(c)) { bool h; bool r = hi::in_group(ctx, c, &h); if (h) return r; }
        char32_t b = base_letter(lower(c));
        if (ctx == 'E') return under_korean_ && c == 'E';   // English: the special letter a rule's '#' made of an e
        switch (ctx) {
            case 'A': return is_vowel_letter(c);
            case 'C': return !is_vowel_letter(c) && !(opt.korean && ko::is_jamo(c));   // Korean: never (measured)
            case 'B': return b < 128 && strchr(opt.portuguese ? "bcdfgjkmnpqstvxz" : "bcdfgjklmnpqstvxz", (char)b);   // pt_rules: not h l r w y
            case 'H': return b < 128 && strchr("hlmnr", (char)b);
            case 'F': return b < 128 && strchr("cfhkpqstx", (char)b);
            case 'G': return b < 128 && strchr(opt.portuguese ? "bdgjlmnrvwz" : "bdgjlmnrvwyz", (char)b);
            case 'Y': return b < 128 && strchr("eiy", (char)b);
        }
        return false;
    }

    // a vowel letter for the syllable symbols of rules (@ & X): in English words of Korean text also a y that is not
    // the first letter of the word before a vowel
    bool syl_vowel(const std::u32string& w, long q) const {
        if (is_vowel_letter(w[q])) return true;
        if (!under_korean_ || w[q] != 'y') return false;
        bool at_start = q == 0 || is_space(w[q - 1]);
        bool before_vowel = (size_t)q + 1 < w.size() && is_vowel_letter(w[q + 1]);
        return !(at_start && before_vowel);
    }

    // ---------- rule matching. w = " word nextword " with letters lower-cased; pos = index into w.
    struct Seg { size_t pos; bool stressable; };
    struct MatchCtx { bool suffix_removed = false, prefix_removed = false; int alt = 0; long stem_start = -1; long block_before = -1; bool no_suffix = false; bool verb = false, noun = false;
                      const std::vector<Seg>* segs = nullptr; bool more_suffix = false;
                      std::u32string sfx;   // German: the letters of the suffix just removed
                      const std::string* out = nullptr; bool stressable_before = false; mutable int da_amp = -1; };   // Danish ('&')
    // Danish '&': do the phonemes so far hold a syllable that may be stressed (not marked % / %%, not an unstressed vowel)?
    // Danish, measured: of two rules with the same score the reference uses the one whose phoneme string is greater in
    // its phoneme order (the order the phonemes are first named in the tables; stress marks above the others).
    std::vector<int> ph_rank;
    void build_ph_rank() {
        std::unordered_map<std::string, int> r; int n = 0;
        for (auto& d : ps.defs) {
            if (!d.voicingswitch.empty() && !r.count(d.voicingswitch)) r[d.voicingswitch] = n++;
            if (!r.count(d.name)) r[d.name] = n++;
        }
        ph_rank.resize(ps.defs.size());
        for (size_t i = 0; i < ps.defs.size(); i++) ph_rank[i] = r[ps.defs[i].name] + (ps.defs[i].type == T_STRESS ? 100000 : 0);
    }
    std::vector<int> rule_key(const Rule& r) const {
        std::vector<int> ph, k; encode_ph(r.phonemes, ph);
        for (int p : ph) if (p >= 0 && (size_t)p < ph_rank.size()) k.push_back(ph_rank[p]);
        return k;
    }
    // do the phonemes so far hold at least n vowels (syllables)?  (letters that were read as a consonant, like the i of "stier", do not count)
    bool pre_syllables_ok(const std::string& mn, int n) const {
        std::vector<int> ph; encode_ph(mn, ph);
        int c = 0;
        for (int p : ph) if (p >= 0 && ps.defs[p].type == T_VOWEL) c++;
        return c >= n;
    }
    bool has_stressable(const std::string& mn) const {
        std::vector<int> ph; encode_ph(mn, ph);
        int pending = -1;
        for (int p : ph) {
            if (p < 0) continue;
            const PhDef& d = ps.defs[p];
            if (d.type == T_STRESS) { pending = d.stress_type; continue; }
            if (d.type == T_VOWEL && !d.nonsyl) { if (!d.unstressed && !(pending == 0 || pending == 1)) return true; pending = -1; }
        }
        return false;
    }
    // German: does a rule's phoneme string hold a vowel that is not marked unstressed ('%' or an unstressed phoneme)?
    bool stressable(const std::string& ph) const {
        std::vector<int> v; encode_ph(ph, v);
        bool unst = false;
        for (int p : v) {
            if (p < 0) continue;
            const PhDef& d = ps.defs[p];
            if (d.type == T_STRESS) { unst = d.stress_type >= 0 && d.stress_type <= 1; continue; }
            if (d.type == T_VOWEL && !d.nonsyl) { if (!unst && !d.unstressed) return true; unst = false; }
        }
        return false;
    }

    // returns score (<=0: no match)
    int match_rule(const Rule& r, const std::u32string& w, size_t pos, size_t wend, const MatchCtx& mc) const {
        if (!cond_ok(r.cond)) return 0;
        int pts = 1;
        size_t n = r.match.size();
        if (pos + n > wend) return 0;
        for (size_t k = 0; k < n; k++) if (w[pos + k] != r.match[k] && !(w[pos + k] == 'E' && r.match[k] == 'e' && k > 0)) return 0;
        pts += 21 * (int)n;
        // post
        size_t j = pos + n; int dist = -6;
        const std::u32string& po = r.post;
        for (size_t k = 0; k < po.size(); k++) {
            dist = std::min(dist + 6, 19);
            char32_t c = po[k];
            char32_t x = j < w.size() ? w[j] : ' ';
            if (c == '_') { if (!(is_space(x) || x == '-') ) return 0; pts += 21 - dist; j++; continue; }
            if (c == '+') { pts += 20; continue; }
            if (c == '<') { pts -= 20; continue; }
            if (c == 'N') { if (mc.suffix_removed) return 0; if (under_korean_) pts += 1; if (opt.measured_scores) pts += 1; continue; }
            if (c == '#') { if (j > pos + n && w[j - 1] == 'e') hash_pos_ = (long)j - 1; continue; }
            if (opt.portuguese && c == '\\' && k + 1 < po.size()) {   // "\$": that character as written ("R$")
                if (x != po[++k]) return 0;
                pts += 21 - dist; j++; continue;
            }
            if (c == '$') {
                std::u32string word; size_t q = k + 1;
                while (q < po.size() && (po[q] == '_' || (po[q] >= 'a' && po[q] <= 'z') || is_digit(po[q]))) word += po[q++];
                k = q - 1;
                if (word == U"noprefix") { if (mc.prefix_removed) return 0; continue; }
                if (word.rfind(U"w_alt", 0) == 0) {
                    int a = word.size() > 5 ? (int)(word[5] - '0') : 1;
                    if (!(mc.alt & (1 << (a - 1)))) return 0;
                    pts += 23; continue;
                }
                if (word.rfind(U"p_alt", 0) == 0) {   // Danish: the word up to the end of this match is listed with $altN
                    int a = word.size() > 5 ? (int)(word[5] - '0') : 1;
                    size_t q = pos;
                    while (q > 0 && !is_space(w[q - 1]) && !(opt.prefix_fresh && mc.stem_start >= 0 && (long)q <= mc.stem_start)) q--;
                    auto it = list.find(w.substr(q, pos + n - q));
                    bool ok = false;
                    if (it != list.end() && opt.at_word_start) {   // the entry a lookup finds (the last one in the file), not any entry
                        const ListEntry* pe = lookup_entry(w.substr(q, pos + n - q), true, false);
                        if (!pe) pe = lookup_entry(w.substr(q, pos + n - q), false, false);
                        ok = pe && (pe->alt & (1 << (a - 1)));
                    } else
                    if (it != list.end()) for (auto& e : it->second) if (e.alt & (1 << (a - 1))) ok = true;
                    if (!ok) return 0;
                    pts += 23; continue;
                }
                return 0;
            }
            if (c == 'L' && k + 2 < po.size() + 0 && is_digit(po[k + 1])) {
                int g = (po[k + 1] - '0') * 10 + (k + 2 < po.size() ? po[k + 2] - '0' : 0);
                k += 2;
                bool ok = false;
                for (auto& it : lgroups[g]) {
                    if (j + it.size() <= w.size() && w.compare(j, it.size(), it) == 0) { j += it.size(); ok = true; break; }
                }
                if (!ok) return 0;
                pts += (opt.korean ? 21 : 20) - dist;   // Korean: a letter group outranks a letter class (measured)
                continue;
            }
            if (c == '/' && k + 1 < po.size()) { c = po[++k]; if (x != c) return 0; pts += 21 - dist; j++; continue; }
            if (c == '%') { if (j == 0 || x != w[j - 1]) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'K') { if (is_alpha(x) && is_vowel_letter(x)) return 0; pts += (opt.portuguese ? 19 : 20) - dist; j++; continue; }
            if (c == 'D') { if (!is_digit(x)) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'Z') { if (is_alpha(x)) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'X') {   // no vowel until the word end
                for (size_t q = j; q < w.size() && !is_space(w[q]); q++) if (syl_vowel(w, (long)q)) return 0;
                pts += (under_korean_ ? 15 : 19) - dist; continue;
            }
            if (c == '@' && opt.measured_scores) {   // Danish: a run of n '@': n syllables follow; scored once
                int nsy = 1;
                while (k + 1 < po.size() && po[k + 1] == '@') { k++; nsy++; }
                size_t q = j;
                for (int a = 0; a < nsy; a++) {
                    while (q < w.size() && !is_space(w[q]) && !is_vowel_letter(w[q])) q++;
                    if (q >= w.size() || is_space(w[q])) return 0;
                    while (q < w.size() && is_vowel_letter(w[q])) q++;
                }
                j = q; pts += 18 + nsy - dist; continue;
            }
            if (c == '@') {
                int need = 1;
                if (opt.syllable_run) while (k + 1 < po.size() && po[k + 1] == '@') { need++; k++; }
                size_t q = j;
                for (int s = 0; s < need; s++) {
                    while (q < w.size() && !is_space(w[q]) && !(opt.syllables_from_match && w[q] == '-') && !syl_vowel(w, (long)q)) q++;
                    if (q >= w.size() || is_space(w[q]) || (opt.syllables_from_match && w[q] == '-')) return 0;
                    while (q < w.size() && syl_vowel(w, (long)q)) q++;
                }
                j = q; pts += 18 - dist; continue;
            }
            if (c >= 'A' && c <= 'Z') { if (!in_set((char)c, x)) return 0; pts += (opt.measured_scores && c == 'C' ? 19 : 20) - dist - opt.class_minus; j++; continue; }
            if (x != c && !(x == 'E' && c == 'e')) return 0;
            pts += 21 - dist; j++;
        }
        // pre (stored reversed)
        long i = (long)pos - 1; dist = -2;
        int pre_syll = 0;
        long syl_q = (long)pos - 1;
        const std::u32string& pr = r.pre;
        for (size_t k = 0; k < pr.size(); k++) {
            dist = std::min(dist + 2, 19);
            char32_t c = pr[k];
            char32_t x = i >= 0 ? w[i] : ' ';
            if (opt.prefix_fresh && mc.stem_start >= 0 && i < mc.stem_start) x = ' ';   // Danish: the prefix is not context
            if (c == '_') { if (!(is_space(x) || x == '-' || i + 1 == mc.stem_start)) return 0; pts += 4; i--; continue; }
            // reversed: "L01" appears as "10L"; "/x" as "x/"; "%C" as "C%"
            if (is_digit(c) && k + 2 < pr.size() && is_digit(pr[k + 1]) && pr[k + 2] == 'L') {
                int g = (pr[k + 1] - '0') * 10 + (c - '0');
                k += 2;
                bool ok = false;
                for (auto& it : lgroups[g]) {
                    long st = i - (long)it.size() + 1;
                    if (st >= 0 && w.compare((size_t)st, it.size(), it) == 0) { i = st - 1; ok = true; break; }
                    if (it.empty()) { ok = true; break; }
                }
                if (!ok) return 0;
                pts += opt.measured_scores ? 26 : 20 - dist;
                continue;
            }
            if (k + 1 < pr.size() && pr[k + 1] == '/') { k++; if (x != c) return 0; pts += 21 - dist; i--; continue; }
            if (c == '%') { if (i < 0 || (size_t)(i + 1) >= w.size() || x != w[i + 1]) return 0; pts += 21 - dist; i--; continue; }
            if (c == '&' && opt.stressable_pre) {   // Danish: a stressable syllable among the phonemes so far (no letters used)
                if (mc.da_amp < 0) mc.da_amp = (mc.stressable_before || (mc.out && has_stressable(*mc.out))) ? 1 : 0;
                if (!mc.da_amp) return 0;
                pts += 19; continue;
            }
            if (c == '@' && opt.measured_scores) {   // Danish: a run of n '@': n syllables before; scored once
                int nsy = 1;
                while (k + 1 < pr.size() && pr[k + 1] == '@') { k++; nsy++; }
                long q = i, lo = (opt.prefix_fresh && mc.stem_start >= 0) ? mc.stem_start : 0;
                // measured: "@" as the farthest item of a context made of plain letters ("@komi") is also satisfied by the word start
                if (opt.at_word_start && mc.out && !(opt.prefix_fresh && mc.stem_start >= 0) && !pre_syllables_ok(*mc.out, nsy)) return 0;   // the syllables before are counted in the phonemes so far
                bool start_ok = opt.at_word_start && nsy == 1 && k >= 3 && k + 1 == pr.size();
                if (start_ok) for (size_t j = 0; j < k; j++) if (!(pr[j] >= 'a' && pr[j] <= 'z') && pr[j] < 0x80) start_ok = false;
                for (int a = 0; a < nsy; a++) {
                    while (q >= lo && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                    if (q >= lo && is_space(w[q]) && start_ok && !(opt.prefix_fresh && mc.stem_start >= 0)) break;
                    if (q < lo || is_space(w[q])) return 0;
                    while (q >= lo && is_vowel_letter(w[q])) q--;
                }
                i = q; pts += 18 + nsy - dist; continue;
            }
            if (c == '@' && opt.portuguese) {
                // Portuguese: needs that many vowel letters in the word before the match; consumes consonants and one
                // vowel letter (or stops at the word start)
                ++pre_syll;
                int nv = 0;
                for (long q = (long)pos - 1; q >= 0 && !is_space(w[q]); q--) if (is_vowel_letter(w[q])) nv++;
                if (nv < pre_syll) return 0;
                long q = i;
                while (q >= 0 && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                if (q >= 0 && !is_space(w[q])) q--;
                i = q; pts += 18 - dist; continue;
            }
            if (c == '&' && opt.amp_stressable && !amp_ok_) return 0;
            if ((c == '@' || c == '&') && opt.syllables_from_match) {
                // syllables are counted back from the matched letters; letters matched by the other pre items do not count
                long q = syl_q;
                while (q >= 0 && !is_space(w[q]) && w[q] != '-' && !is_vowel_letter(w[q])) q--;
                if (q < 0 || is_space(w[q]) || w[q] == '-') return 0;
                while (q >= 0 && is_vowel_letter(w[q])) q--;
                syl_q = q; pts += 18 - dist; continue;
            }
            if (c == '&' && mc.segs) {   // German: a syllable already read that can take stress
                bool ok = false;
                for (auto& sg : *mc.segs) if ((long)sg.pos <= i && sg.stressable) ok = true;
                if (!ok) return 0;
                pts += 18 - dist; continue;
            }
            if (c == '@' || c == '&') {
                long q = i, lo = mc.block_before >= 0 ? mc.block_before : 0;   // an unstressed prefix is not a syllable here
                while (q >= lo && !is_space(w[q]) && w[q] != '-' && !syl_vowel(w, q)) q--;
                while (q >= lo && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                if (opt.german && q >= 1 && w[q] == 'u' && w[q - 1] == 'q') {   // German: the u of "qu" is no syllable ("quer")
                    q -= 2;
                    while (q >= lo && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                }
                if (opt.german && q >= 0 && w[q] == 'y' && (size_t)(q + 1) < w.size() && w[q + 1] != 'y' && is_vowel_letter(w[q + 1])) {
                    q--;   // German: y before a vowel is a consonant ("yen")
                    while (q >= lo && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                }
                if (q < lo || is_space(w[q]) || w[q] == '-') return 0;
                while (q >= lo && syl_vowel(w, q)) q--;
                i = q; pts += 18 - dist; continue;
            }
            if (c == 'K') { if (is_alpha(x) && is_vowel_letter(x)) return 0; pts += 20 - dist; i--; continue; }
            if (c == 'D') { if (!is_digit(x)) return 0; pts += 21 - dist; i--; continue; }
            if (c == 'Z') { if (is_alpha(x)) return 0; pts += 21 - dist; i--; continue; }
            if (c == 'V') { if (!(opt.verb_context && mc.verb)) return 0; pts += 7; continue; }
            if (c == 'X') {   // no vowel from here back to the word start
                for (long q = i; q >= 0 && !is_space(w[q]); q--) if (syl_vowel(w, q)) return 0;
                pts += (under_korean_ ? 15 : 19) - dist; continue;
            }
            if (c >= 'A' && c <= 'Z') { if (!in_set((char)c, x)) return 0; pts += (opt.measured_scores && c == 'C' ? 19 : 20) - dist; i--; continue; }
            if (x != c && !(x == 'E' && c == 'e')) return 0;
            pts += 21 - dist; i--;
        }
        return pts;
    }

    // The best-scoring rule at w[pos] (as the scan in apply_rules picks it), or nullptr.
    const Rule* best_rule(const std::u32string& w, size_t pos, size_t e, const MatchCtx& mc, int* pts) const {
        const Rule* best = nullptr; int best_pts = 0;
        char32_t c = w[pos];
        auto try_group = [&](const std::u32string& key) {
            auto it = groups.find(key);
            if (it == groups.end()) return;
            for (int ri : it->second) {
                int p = match_rule(rules[ri], w, pos, e, mc);
                if (p > best_pts) { best_pts = p; best = &rules[ri]; }
            }
        };
        if (pos + 1 < e) try_group(std::u32string{c, w[pos + 1]});
        if (!best) {
            try_group(std::u32string{c});
            if (!best) try_group(U"");
            if (!best) { char32_t bl = base_letter(c); if (bl != c) try_group(std::u32string{bl}); }
        }
        *pts = best_pts;
        return best;
    }

    // Translates the letters w[b..e) by rules. Returns mnemonic string.
    std::string apply_rules(const std::u32string& w_in, size_t b, size_t e, const MatchCtx& mc_in, int depth = 0) const {
        std::string out;
        size_t pos = b;
        std::u32string w = w_in;   // English inside Korean text changes letters while translating ('#')
        if (under_korean_) for (auto& ch : w) if (ch == 'E') ch = 'e';   // a stem that is translated again starts afresh
        if (opt.amp_stressable) amp_ok_ = false;
        std::vector<Seg> segs;   // German: which rules so far gave a vowel that can take stress (pre-context '&')
        MatchCtx mc = mc_in;
        mc.out = &out;   // Danish '&'
        if (opt.german) mc.segs = &segs;
        const Rule* pfx = nullptr; size_t pfx_pos = 0; std::string pfx_out;
        auto do_prefix = [&](const Rule* r, size_t p, const std::string& o) -> std::string {
            size_t pend = p + (size_t)r->prefix;
            MatchCtx m2 = mc_in; m2.prefix_removed = true;
            std::string rest;
            std::u32string tail = w.substr(pend, e - pend);
            const ListEntry* le = nullptr;
            bool with_sfx = false;
            if (opt.german && !mc_in.sfx.empty() && (le = lookup_entry(tail + mc_in.sfx, false, true))) {
                // German: prefix + a listed word that was cut by a suffix rule ("be|weg(en)", "ge|word(en)"): the list word stays whole
                tail += mc_in.sfx; with_sfx = true;
            }
            else if (opt.german && !mc_in.sfx.empty() && [&]() {   // no vowel letter ("qu": its u is no vowel here: über|qu|eren)
                        for (size_t ti = 0; ti < tail.size(); ti++)
                            if (is_vowel_letter(tail[ti]) && !(tail[ti] == U'u' && ti > 0 && tail[ti - 1] == U'q')) return false;
                        return true; }()) {
                // German: nothing readable is left after the prefix ("bis|h(er)", "sechs|(er)"): the ending is read with the rest
                tail += mc_in.sfx; with_sfx = true;
            }
            if (!le) le = lookup_entry(tail, false, true);
            if (le) { rest_stress_pos_ = le->stress_pos; stem_alt_ |= le->alt; rest_u_ = le->u; rest_u_plus_ = le->u_plus || le->strend; }
            prefix_taken_ = true;
            if (le && !le->phonemes.empty()) rest = le->phonemes;
            else {
                if (opt.german) {   // German: the part after a prefix is read as a word of its own
                    std::u32string w2 = U" " + (with_sfx ? tail + U" " : w.substr(pend));
                    if (with_sfx) { m2.suffix_removed = false; m2.more_suffix = false; m2.sfx.clear(); }
                    const bool ns = no_switch_;
                    if (le) no_switch_ = true;
                    rest = apply_rules(w2, 1, 1 + (with_sfx ? tail.size() : e - pend), m2, depth + 1);
                    no_switch_ = ns;
                } else {
                m2.stem_start = (long)pend;
                if (opt.post_secondary && r->phonemes.find('\'') == std::string::npos) m2.block_before = (long)pend;
                rest = apply_rules(w, pend, e, m2, depth + 1);
                }
            }
            suffix_absorbed_ = with_sfx;
            return o + r->phonemes + "\x04" + rest;
        };
        while (pos < e) {
            char32_t c = w[pos];
            const Rule* best = nullptr; int best_pts = 0;
            long best_hash = -1;
            mc.da_amp = -1;
            auto try_group = [&](const std::u32string& key) {
                auto it = groups.find(key);
                if (it == groups.end()) return false;
                if (opt.measured_scores) {   // Danish: the best rule of the group, then the group against the best so far
                    const Rule* gb = nullptr; int gp = 0;
                    for (int ri : it->second) {
                        int p = match_rule(rules[ri], w, pos, e, mc);
                        if (p > 0 && key.size() == 2) p += 14;   // a rule of a two-letter group
                        if (p > gp || (p > 0 && p == gp && gb && rule_key(rules[ri]) > rule_key(*gb))) { gp = p; gb = &rules[ri]; }
                    }
                    if (gb && gp > best_pts) { best_pts = gp; best = gb; }
                    return true;
                }
                for (int ri : it->second) {
                    if (opt.suffix_once && rules[ri].suffix > 0 && pos + rules[ri].match.size() <= b + (size_t)rules[ri].suffix) continue;
                    hash_pos_ = -1;
                    int p = match_rule(rules[ri], w, pos, e, mc);
                    // pt: on a tie the rule with the greater match string wins, then the later one
                    if (p > best_pts ||
                        ((opt.tie_later_rule || opt.tie_later || tie_later_) && p > 0 && p == best_pts && (!opt.tie_by_phonemes || !best || tie_wins(rules[ri], *best))) ||
                        (opt.portuguese && p > 0 && p == best_pts && best && rules[ri].match >= best->match)) {
                        best_pts = p; best = &rules[ri]; best_hash = hash_pos_;
                    }
                }
                return true;
            };
            if (pos + 1 < e) try_group(std::u32string{c, w[pos + 1] == 'E' ? U'e' : w[pos + 1]});
            bool had2 = best != nullptr && !opt.both_groups;
            if (!had2) {
                try_group(std::u32string{c});
                if (!best) try_group(U"");
                if (!best) {
                    char32_t bl = base_letter(c);
                    if (bl != c) try_group(std::u32string{bl});
                }
            }
            if (!best && under_korean_ && base_letter(c) != c && accent_miss_ < 0) accent_miss_ = (long)pos;   // see translate_word
            if (!best) {
                if (opt.da_numbers && c == '\'' && pos > b && pos + 1 < e) out += "apostr'Vf";   // Danish: an apostrophe no rule takes is read as its name (I'm, rock'n'roll)
                pos++; continue;
            }
            if (under_korean_ && best_hash >= 0 && (size_t)best_hash < w.size()) w[best_hash] = 'E';
#ifdef KT_FE_TRACE
            fprintf(stderr, "rule @%zu: %s)%s(%s -> %s [%d]\n", pos, enc(std::u32string(best->pre.rbegin(), best->pre.rend())).c_str(), enc(best->match).c_str(), enc(best->post).c_str(), best->phonemes.c_str(), best_pts);
#endif
            if (best->phonemes.rfind("_^_", 0) == 0 && !ignore_switch_) {
                switch_lang_ = "EN";
                if (opt.da_numbers) {   // Danish: the rule names the language (ñ -> _^_ES)
                    std::string code;
                    for (size_t k = 3; k < best->phonemes.size() && k < 5 && std::isalpha((unsigned char)best->phonemes[k]); k++)
                        code += (char)std::toupper((unsigned char)best->phonemes[k]);
                    if (code.size() == 2) switch_lang_ = code;
                }
                return "";
            }
            const bool may_remove = !opt.suffix_once || (depth < opt.max_suffix_depth && !mc.no_suffix);
            // German: after a suffix is removed, another one only if the first allowed it (flag 'm')
            const bool suffix_ok = !(opt.german && mc.suffix_removed && !mc.more_suffix);
            if (best->suffix > 0 && suffix_ok && pos + best->match.size() >= e && may_remove) {
                size_t se = pos + best->match.size() - (size_t)best->suffix;
                if (se > b) {
                    std::u32string stem = w.substr(b, se - b);
                    const ListEntry* sl = opt.prefix_word_flags ? lookup_stem_ctx(stem, mc) : lookup_entry(stem, true, false);
                    size_t ap = stem.rfind('\'');
                    bool via_ap = false;
                    if (!sl && ap != std::u32string::npos) { sl = lookup_entry(stem.substr(ap + 1), true, false); via_ap = sl != nullptr; }
                    if (sl) { stem_alt_ |= sl->alt; if (sl->stress_pos) stem_stress_pos_ = sl->stress_pos; if (sl->u) { stem_u_ = true; stem_u_plus_ = sl->u_plus; }
                               if (sl->u_plus || sl->strend) stem_uplus_ = true;
                               if (sl->u_n) stem_un_ = sl->u_n; }
                    if (sl && !sl->phonemes.empty() && best->affix_flags.find('q') != std::string::npos && depth == 0 && se == pos) {
                        // the stem is in the list with its phonemes (after an elision such as l', the part before it by rules)
                        std::string head = via_ap ? apply_rules(w, b, b + ap + 1, mc, depth + 1) + "\x05" : "";
                        return head + sl->phonemes + "\x05" + best->phonemes;
                    }
                }
            }
            if (best->suffix > 0 && suffix_ok && depth < opt.max_suffix_depth && may_remove && pos + best->match.size() >= e && best->affix_flags.find('q') == std::string::npos) {
                // standard suffix: translate the stem, then add the suffix phonemes
                suffix_taken_ = true;
                size_t slen = (size_t)best->suffix;
                size_t stem_end = pos + best->match.size() - slen;
                if (stem_end > b) {
                    MatchCtx m2 = mc; m2.suffix_removed = true;
                    m2.more_suffix = best->affix_flags.find('m') != std::string::npos;
                    m2.sfx = w.substr(stem_end, slen) + mc.sfx;
                    suffix_taken_ = true;
                    std::u32string stem = w.substr(b, stem_end - b);
                    std::string sp;
                    std::u32string head = w.substr(0, stem_end);
                    if (under_korean_ && stem.size() >= 2) {
                        // documented suffix flags: d = the last letter may have been doubled, i = y may have become i,
                        // e = an e may have been removed (added back after vowel + consonant; the rules sort out the rest)
                        const std::string& fl = best->affix_flags;
                        char32_t z = stem.back(), y = stem[stem.size() - 2];
                        auto ascii_vowel = [](char32_t ch) { return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' || ch == 'y'; };
                        bool cons = z >= 'a' && z <= 'z' && !ascii_vowel(z);
                        const ListEntry* raw = lookup_entry(stem, true, false);
                        if (raw && !raw->phonemes.empty()) { /* the stem as written is a listed word ("edit" + ed) */ }
                        else if (fl.find('d') != std::string::npos && cons && z == y && stem.size() >= 3) { stem.pop_back(); head.pop_back(); }
                        else if (fl.find('i') != std::string::npos && z == 'i') { stem.back() = 'y'; head.back() = 'y'; }
                        else if (fl.find('e') != std::string::npos && cons && z != 'w' && z != 'x' && ascii_vowel(y) && y != 'y') { stem += U'e'; head += U'e'; }
                    }
                    const ListEntry* le = (opt.suffix_once && std::none_of(stem.begin(), stem.end(), [&](char32_t x) { return is_vowel_letter(x); }))
                                              ? nullptr
                                              : (opt.prefix_word_flags ? lookup_stem_ctx(stem, mc) : lookup_entry(stem, true, false));   // Turkish: no lookup of "y" in "yta"
                    if (under_korean_ && best->affix_flags.find('v') != std::string::npos) {   // documented: the verb form
                        auto lit = list.find(stem);
                        if (lit != list.end()) for (auto& en : lit->second) if (en.verb && !en.phonemes.empty()) le = &en;
                    }
                    if (le && opt.portuguese) m2.alt |= le->alt;
                    suffix_absorbed_ = false;
                    if (le && !le->phonemes.empty()) { sp = le->phonemes + (opt.german ? "\x05" : ""); stem_given_ = true; stem_listed_ = true; stem_marks_ = true; }
                    else sp = apply_rules(head + U" ", b, b + stem.size(), m2, depth + 1);
                    if (suffix_absorbed_) { suffix_taken_ = false; return sp; }   // the suffix is part of a listed word (stays set for outer suffixes)
                    // phonemes of the matched part that are not the suffix are dropped (stem re-translated)
                    bool sep = best->affix_flags.find('t') != std::string::npos;
                    return sp + (sep ? "\x02" : "") + best->phonemes;
                }
            }
            if (best->prefix > 0 && depth < 3 && opt.german && !pfx) {
                // German (as espeak-ng behaves): the scan goes on after a prefix; a suffix at the word end is removed
                // first ("erste" = "erst" + e), else the prefix is removed at the end of the scan
                pfx = best; pfx_pos = pos; pfx_out = out;
            } else if (best->prefix > 0 && depth < 3 && opt.german) {
                return do_prefix(best, pos, out);
            } else if (best->prefix > 0 && depth < 3) {
                size_t pend = pos + (size_t)best->prefix;
                MatchCtx m2 = mc; m2.prefix_removed = true;
                if (opt.stressable_pre) m2.stressable_before = !opt.prefix_fresh && (mc.stressable_before || has_stressable(out + best->phonemes));   // Danish
                std::string rest;
                std::u32string tail = w.substr(pend, e - pend);
                const ListEntry* le = lookup_entry(tail, false, true);
                if (opt.prefix_word_flags && best->affix_flags.find('v') != std::string::npos) {
                    // n', s', j' ... (P2v): the rest is read as a verb, so its $verb entries apply
                    auto vit = list.find(tail);
                    if (vit != list.end())
                        for (auto r = vit->second.rbegin(); r != vit->second.rend(); ++r)
                            if (r->verb && !r->text) { le = &*r; break; }
                }
                if (opt.prefix_word_flags && le && le->phonemes.rfind("_^_", 0) == 0) le = nullptr;   // qu'Apple: French rules
                if (opt.prefix_word_flags && best->affix_flags.find('v') != std::string::npos) m2.verb = true;
                if (le) { rest_stress_pos_ = le->stress_pos; stem_alt_ |= le->alt; rest_entry_ = le; }
                prefix_taken_ = true;
                if (le && !le->phonemes.empty()) rest = le->phonemes;
                else if (opt.prefix_word_flags) {
                    // French: the rest is translated as a word of its own (photo|synthèse: no context from the prefix)
                    std::u32string w2 = U" " + w.substr(pend);
                    m2.stem_start = 1;
                    rest = apply_rules(w2, 1, 1 + (e - pend), m2, depth + 1);
                } else {
                    m2.stem_start = (long)pend;
                    if (opt.post_secondary && best->phonemes.find('\'') == std::string::npos) m2.block_before = (long)pend;
                    if (opt.prefix_cut) {   // the rest is read as a word of its own: the prefix letters are not context
                        m2.stem_start = -1;
                        std::u32string w2 = U" " + w.substr(pend);
                        rest = apply_rules(w2, 1, 1 + (e - pend), m2, depth + 1);
                    } else
                    rest = apply_rules(w, pend, e, m2, depth + 1);
                }
                return out + best->phonemes + "\x04" + rest;
            }
            out += best->phonemes;
            out += '\x05';   // phoneme names never span two rules
            if (opt.german) segs.push_back({pos, stressable(best->phonemes)});
            if (opt.amp_stressable && !amp_ok_) {
                std::vector<int> v; encode_ph(best->phonemes, v);
                int pend = -1;
                for (int p : v) {
                    if (stress_of(p) >= 0) { pend = stress_of(p); continue; }
                    if (is_vowel_ph(p) && !ps.defs[p].unstressed && pend != 0 && pend != 1) amp_ok_ = true;
                    if (is_vowel_ph(p)) pend = -1;
                }
            }
            pos += std::max<size_t>(1, best->match.size());
        }
        if (pfx) return do_prefix(pfx, pfx_pos, pfx_out);
        return out;
    }

    // French: which of two rules with the same score wins. The reference tries the rules of a group ordered by their
    // phonemes (no phonemes first, then a language switch, then by the phonemes' order of definition) and keeps the
    // last of the best ones (observed with the reference library; e.g. "uz (z" wins over "uzz", "uy (A" over "g) u (A").
    const std::vector<int>& tie_key(const Rule& r) const {
        auto it = tie_keys_.find(&r);
        if (it != tie_keys_.end()) return it->second;
        std::vector<int> k;
        if (r.phonemes.rfind("_^_", 0) == 0) k.push_back(-1);
        else {
            std::vector<int> ph; encode_ph(r.phonemes, ph);
            for (int p : ph) k.push_back(p < 0 ? 0 : ps.base_def(ps.defs[p].name) + 1);
        }
        return tie_keys_.emplace(&r, std::move(k)).first->second;
    }
    bool tie_wins(const Rule& cand, const Rule& best) const {
        const std::vector<int>& a = tie_key(cand), & b = tie_key(best);
        if (a == b) return &cand > &best;
        return std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end());
    }
    mutable std::unordered_map<const Rule*, std::vector<int>> tie_keys_;
    // French: a stem after a removed suffix, with the $noun / $verb entries the previous word asks for (de points -> point $noun)
    const ListEntry* lookup_stem_ctx(const std::u32string& word, const MatchCtx& mc) const {
        auto it = list.find(word);
        if (it == list.end()) return nullptr;
        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) {
            if (r->atend || r->atstart || r->capital || r->allcaps || r->past || r->hasdot || r->sentence || r->text) continue;
            if (r->only || r->onlys) continue;
            if (r->noun && !mc.noun) continue;
            if (r->verb && !mc.verb) continue;
            return &*r;
        }
        return nullptr;
    }
    const ListEntry* lookup_entry(const std::u32string& word, bool after_suffix, bool after_prefix) const {
        auto it = list.find(word);
        if (it == list.end()) return nullptr;
        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) {
            if (opt.german && (after_suffix || after_prefix)) {
                // German: a stem keeps the capital letter of its word ("Wege" -> "Weg" $capital); under a word that
                // the list names (flags only, e.g. "worden $u") a stem never switches language ("word" _^_EN)
                if (r->capital && !cur_capital_) continue;
                if ((no_switch_ || after_prefix) && r->phonemes.rfind("_^_", 0) == 0) continue;
                if (r->capital && !(r->atend || r->atstart || r->allcaps || r->verb || r->noun || r->past || r->hasdot ||
                                    r->sentence || r->only || r->onlys || r->text)) return &*r;
            }
            if (r->atend || r->atstart || r->capital || r->allcaps || r->verb || r->past || r->hasdot || r->sentence) continue;
            if (r->noun && !(opt.portuguese && cur_expect_noun_)) continue;
            if ((after_suffix || after_prefix) && (r->only || r->onlys)) continue;
            if (!(after_suffix || after_prefix) && r->stem) continue;
            if (r->text) continue;
            return &*r;
        }
        return nullptr;
    }

    // ---------- words of a clause
    struct Word {
        std::u32string text;   // lower case
        bool capital = false, allcaps = false;
        std::vector<int> ph;   // phoneme indices incl. stress phonemes
        int stress_pos = 0;
        bool unstressed = false, u_plus = false;
        int u_n = 0;
        bool join_prev = false;   // hyphenated with the previous word
        bool number = false;
        bool no_stress = false;
        bool number_tail = false;  // Portuguese: a later part of a number (no added secondary stress)
        bool pause_before = false; // Portuguese: '/' or a dash before the word (a pause: no voicing across)
        bool grouped = false;      // Portuguese: the number was written with thousands dots ("19.500")
        char32_t ordinal = 0;      // Portuguese: U+00BA / U+00AA after the digits ("1º", "1ª")
        int fraction = 0;          // Portuguese: digits after a decimal comma (1 = read as a number, 2 = digit by digit)
        bool spelled = false;      // Portuguese: read letter by letter
        bool silent = false;       // Portuguese: seen by the word before it, not spoken
        int alt = 0;
        bool no_default = false;   // a prefix: only the stress its phonemes mark
        bool listed_marks = false; // phonemes from the word list with their own stress marks
        bool expect_noun = false, expect_verb = false;   // the previous word has $nounf / $verbf
        bool nounf = false, verbf = false;
        bool brk = false;          // $brk: a pause before the word (French: no liaison into it)
        bool num_unstressed = false;   // French: a number before a decimal point, read without stress
        bool rest_suffix = false;      // French: prefix + word + suffix (lorsqu'ils): an unstressed word of several syllables
        int extra_syl = 0;             // syllables of a removed prefix, counted for an unstressed word
        bool after_comma = false;      // French: the digits after a decimal comma
        bool digit_groups = false;     // French: a number read digit by digit in groups, each with its own stress
        bool spell_caps = false;       // French: the capitals before a capitalised word ("SANParks"), spelled
        bool num_grouped = false;      // French: a number written with spaces between groups of three ("19 500")
        bool phrase_head = false;      // French: first word of an unstressed two-word list entry ((ceux ci) sY||si $u+)
        int phrase_tail = 0;           // its second word: 1 = $u, 2 = $u+ (stressed at the clause end)
        bool digits_single = false;   // Hindi: digits after a decimal point
        bool marks_only = false;      // Hindi: only the stress marks of the phonemes (no secondary stress added)
        bool symbol = false;          // Hindi: a symbol read with the English list (_/ slaS)
        bool hyphen_next = false;     // Hindi: the next word is joined by a hyphen (its final syllable is not the word's last)
        bool attached = false;        // written right after the previous word (KV62: letters then digits)
        bool phrase = false;       // Danish: a multi-word list entry held in one Word ("||" between its words)
        bool clause_end = false;   // Danish: a dash after this word ends the clause
        bool rest_part = false;    // Danish: the 2nd.. part of a word split at a pause marker (u_!lykke)
        bool no_pre_secondary = false, no_post_secondary = false;   // Hindi number parts (measured, see the parts loop)
        int rest_stress_pos = 0;   // $N of the word after a prefix
        std::string switch_lang;   // set: translate the word with that language (English only)
        std::u32string orig;       // the word as written
        bool digits_each = false;  // digits after a decimal point: read one by one
        bool given = false;        // phonemes from the word list / number words (isTranslationGiven)
        bool rules_only = false;   // not looked up in the word list (a suffix after a number)
        bool digit_by_digit = false;   // the digits after a decimal comma
        bool superscript = false;      // a superscript digit
        bool unstressend = false;  // "$unstressend": no primary stress at the end of a clause
        bool mapped = false;       // a letter was replaced by its plain letter ("Dê" -> "de")
        bool spell_all = false;    // spelled letter by letter (a word with a letter that has no name and no vowel)
        bool symbols = false;      // two or more symbols written together
        bool year_group = false;   // digits whose first group, before a space, had four digits ("1971 621")
        bool listed = false;       // phonemes from the word list
        bool glued_prev = false;   // Korean: written against the previous word with no separator (script change)
        bool spell_letters = false;   // Korean: a short all-capitals word read letter by letter (ko::is_abbreviation)
        bool force_spell = false;  // this one-letter word is a letter name (from spell_letters)
        bool stem_marks = false;   // English inside Korean: its stress marks are those of a listed stem (m,eIk + ing)
        bool dbl_next = false;     // Finnish: the next word's first consonant is doubled
        bool fi_pause_before = false; // $pause
        bool no_next = false;      // Finnish: rules do not see the next word (a mark between, or foreign letters)
        std::string decimal_ph;    // Finnish: the decimal point / comma word, stressed with this number
        bool roman = false;        // German: a roman numeral ("rY:mIS" || number): the first part keeps its own marks
        bool glued_num = false;    // German: letters right after digits ("20er")
        bool suffixed = false;     // German: a suffix was removed (then only the first primary mark stays)
        bool rest_u = false, rest_u_plus = false;   // German: the list flags $u / $strend of the part after a prefix
        std::string num_suffix;    // German: "_#er" phonemes added to the number
    };

    const ListEntry* lookup_ctx(const std::u32string& key, const Word& w, bool atstart, bool atend) const {
        auto it = list.find(key);
        if (it == list.end()) return nullptr;
        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) {
            if (r->atend && !atend) continue;
            if (r->atstart && !atstart) continue;
            if (r->capital && !w.capital) continue;
            if (r->allcaps && !w.allcaps) continue;
            if (r->noun && !w.expect_noun) continue;
            if (r->verb && !w.expect_verb) continue;
            if (r->past || r->hasdot) continue;
            if (r->sentence && !(opt.sentence_atend && atend)) continue;
            if (r->stem) continue;
            return &*r;
        }
        return nullptr;
    }

    // number -> mnemonic using the _N entries (generic: units, tens, hundreds, thousands)
    static std::u32string lkey(const std::string& k) { std::u32string u = dec(k); for (auto& c : u) c = lower(c); return u; }
    std::string ph_of(const std::string& key) const {
        auto it = list.find(lkey(key));
        if (it == list.end()) return "";
        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) if (!r->phonemes.empty()) return r->phonemes;
        return "";
    }
    bool has(const std::string& key) const { return list.count(lkey(key)) > 0; }
    std::string one_prefix(int d) const {   // "ein" before hundert / tausend
        std::string k = "_" + std::to_string(d) + "a";
        return has(k) ? ph_of(k) : ph_of("_" + std::to_string(d));
    }
    std::string number_lt1000_de(int n) const {
        std::string o;
        int h = n / 100, r = n % 100;
        if (h) o += one_prefix(h) + "\x02" + ph_of("_0C");
        if (r) {
            if (h) o += "||";
            std::string k = "_" + std::to_string(r);
            if (has(k)) o += ph_of(k);
            else {
                int t = r / 10, u = r % 10;
                if (u) o += (u == 1 ? one_prefix(1) : ph_of("_" + std::to_string(u))) + "||" + ph_of("_0and");
                o += ph_of("_" + std::to_string(t) + "X");
            }
        }
        return o;
    }
    std::string number_lt1000(int n, bool exact_hundred_ok = true) const {
        if (opt.num_swap) return number_lt1000_de(n);
        std::string o;
        int h = n / 100, r = n % 100;
        if (h) {
            if (r == 0 && h == 1 && has("_1C0")) o += ph_of("_1C0");
            else if (has("_" + std::to_string(h) + "C")) o += ph_of("_" + std::to_string(h) + "C");
            else if (h == 1 && opt.omit_one_hundred) o += ph_of("_0C");
            else o += ph_of("_" + std::to_string(h)) + (opt.number_domains || opt.hundreds_domain ? "\x02" : "") + ph_of("_0C");
            (void)exact_hundred_ok;
        }
        if (r) {
            if (h && !o.empty()) {
                if (opt.finnish && o.back() == '_') o.pop_back();   // "s,ata:_": the pause is the word break here
                o += "||";
            }
            std::string k = "_" + std::to_string(r);
            if (has(k)) o += ph_of(k);
            else if (r < 10) o += ph_of("_" + std::to_string(r));
            else if (opt.french_numbers) {
                // French: tens and units written together, no "and" (21, 31 .. 71 have their own entries);
                // 70-79 = 60 + 10..19, 90-99 = 80 + 10..19
                int t = r / 10, u = r % 10;
                if (t == 7 || t == 9) o += ph_of("_" + std::to_string(t - 1) + "X") + ph_of("_" + std::to_string(10 + u));
                else o += ph_of("_" + std::to_string(t) + "X") + (u ? ph_of("_" + std::to_string(u)) : "");
            } else {
                int t = r / 10, u = r % 10;
                std::string tk = "_" + std::to_string(t) + "X";
                if (opt.finnish && !u && (h || num_after_)) o += ",";   // "sata ˌkaksikymmentä"
                o += ph_of(tk);
                if (u) {
                    std::string a = ph_of("_0and");
                    std::string up = ph_of("_" + std::to_string(u));
                    if (opt.number_domains) {
                        // Italian: settant(a) + otto, vent(i) + uno: the tens lose their final vowel before a vowel
                        size_t f = up.find_first_not_of("',%=");
                        auto isv = [](char ch) { return std::strchr("aeiouAEIOU", ch) != nullptr; };
                        if (f != std::string::npos && isv(up[f]) && !o.empty() && isv(o.back())) o.pop_back();
                        o += "\x02";
                    }
                    o += a;
                    o += up;
                }
            }
        }
        return o;
    }
    std::string number_word(const std::u32string& digits) const {
        if (opt.swahili) {
            std::string r = sw_number(enc(digits), [this](const std::string& k) { return ph_of(k); },
                                      [this](const std::string& k) { return has(k); });
            if (!r.empty()) return r;
        }
        if (opt.french_numbers && digits.size() >= 4 && digits[0] == '0') {
            // French: four digits and more with a leading zero are read digit by digit, in groups of three from the
            // right (four digits: one group); each group is one word
            std::string o;
            size_t first = digits.size() == 4 ? 4 : (digits.size() % 3 ? digits.size() % 3 : 3);
            for (size_t q = 0; q < digits.size(); q++) {
                if (q >= first && (q - first) % 3 == 0) o += "||";
                o += ph_of(std::string("_") + (char)digits[q]);
            }
            return o;
        }
        if (opt.french_numbers && digits.size() > 1 && digits.size() <= 9 && digits[0] == '0') {
            // French: leading zeros are read "zero" joined together, then the rest as its own word ("000" -> zerozero zero)
            size_t k = 0;
            while (k + 1 < digits.size() && digits[k] == '0') k++;
            std::string o;
            for (size_t q = 0; q < k; q++) o += ph_of("_0");
            return o + "||" + number_word(digits.substr(k));
        }
        if (opt.french_numbers) {   // "1000" -> "mil" without an empty last word
            std::string o = number_word_core(digits);
            while (o.size() >= 2 && o.compare(o.size() - 2, 2, "||") == 0) o.resize(o.size() - 2);
            return o;
        }
        return number_word_core(digits);
    }
    std::string number_word_core(const std::u32string& digits) const {
        if (digits.size() > 9) {
            std::string o;
            for (char32_t d : digits) { if (!o.empty()) o += "||"; o += ph_of(std::string("_") + (char)d); }
            return o;
        }
        if (digits.size() > 1 && digits[0] == '0') {
            std::string o;
            for (char32_t d : digits) { if (!o.empty()) o += "||"; o += ph_of(std::string("_") + (char)d); }
            return o;
        }
        long v = 0;
        for (char32_t d : digits) v = v * 10 + (d - '0');
        if (opt.da_numbers) return da::number(v, [this](const std::string& k) { return ph_of(k); });
        if (v == 0) return ph_of("_0");
        std::string o;
        long mil = v / 1000000, th = (v / 1000) % 1000, rest = v % 1000;
        if (mil) {
            if (mil == 1 && has("_1M2")) o += ph_of("_1M2");
            else o += number_lt1000((int)mil) + "||" + ph_of("_0M2");
            o += "||";
        }
        if (th && opt.num_swap) {
            o += (th == 1 ? one_prefix(1) : number_lt1000((int)th)) + "||" + ph_of("_0M1") + "||";
        } else if (th) {
            if (th == 1 && has("_1M1")) o += ph_of("_1M1");
            else if (th == 1) o += ph_of("_0M1");
            else o += number_lt1000((int)th) + "||" + ph_of("_0M1");
            o += "||";
        }
        num_after_ = mil || th;
        if (rest) o += number_lt1000((int)rest);
        num_after_ = false;
        if (opt.finnish) { size_t q; while ((q = o.find("_||")) != std::string::npos) o.erase(q, 1); }
        return o;
    }

    // Swahili: a symbol that neither the word list nor the rules read ("/", "&") takes its name from the English word
    // list (en_list, if downloaded), as espeak-ng does; the phoneme names are then read with this language's table.
    mutable std::unique_ptr<std::unordered_map<char32_t, ListEntry>> en_symbols_;
    const ListEntry* en_symbol(char32_t c) const {
        if (!en_symbols_) {
            en_symbols_.reset(new std::unordered_map<char32_t, ListEntry>());
            EspeakLangImpl en;
            std::string err;
            if (en.load_list(data_dir_ + "/en_list", &err, false))
                for (auto& kv : en.list)
                    if (kv.first.size() == 1 && !is_alpha(kv.first[0]) && !is_digit(kv.first[0]) && !is_space(kv.first[0]))
                        for (auto r = kv.second.rbegin(); r != kv.second.rend(); ++r)
                            if (!r->cond && !r->phonemes.empty()) { en_symbols_->emplace(kv.first[0], *r); break; }
        }
        auto it = en_symbols_->find(c);
        return it == en_symbols_->end() ? nullptr : &it->second;
    }
    bool symbol_rule(char32_t c) const {   // a rule of the default group reads this one symbol ("%" -> asilimia)
        auto g = groups.find(U"");
        if (g == groups.end()) return false;
        for (int ri : g->second) if (rules[ri].match.size() == 1 && rules[ri].match[0] == c && rules[ri].pre.empty() && rules[ri].post.empty()) return true;
        return false;
    }
    // the phonemes mark a stress before their first vowel
    bool stressed_start(const std::string& m) const {
        std::vector<int> v;
        encode_ph(m, v);
        for (int p : v) {
            if (p < 0) continue;
            if (stress_of(p) >= 2) return true;
            if (is_vowel_ph(p)) return false;
        }
        return false;
    }
    std::string pt_number(const std::u32string& digits, bool grouped = false, char32_t ordinal = 0, int fraction = 0) const {
        lang_pt::NumberWords w;
        w.ph = [this](const std::string& k) { return ph_of(k); };
        w.has = [this](const std::string& k) { return has(k); };
        w.stressed_start = [this](const std::string& m) { return stressed_start(m); };
        if (ordinal) return lang_pt::ordinal(enc(digits), ordinal == 0xAA, w);
        if (fraction == 2) return lang_pt::digits_spaced(enc(digits), w);
        return lang_pt::number(enc(digits), w, grouped);
    }

    bool symbol_word(char32_t c) const {   // the rules have their own entry for this one symbol
        auto g = groups.find(U"");
        if (g == groups.end()) return false;
        for (int ri : g->second) if (rules[ri].match.size() == 1 && rules[ri].match[0] == c && rules[ri].pre.empty()) return true;
        return false;
    }
    mutable std::unordered_map<char32_t, bool> known_cache;
    bool known_letter(char32_t c) const {
        auto it = known_cache.find(c);
        if (it != known_cache.end()) return it->second;
        bool k = groups.count(std::u32string{c}) > 0;
        if (!k) { auto g = groups.find(U""); if (g != groups.end()) for (int ri : g->second) if (!rules[ri].match.empty() && rules[ri].match[0] == c) k = true; }
        if (!k) for (auto& kv : groups) if (kv.first.size() == 2 && kv.first[0] == c) k = true;
        if (!k && list.count(std::u32string{c}) && !opt.letter_by_rules && !(opt.da_numbers && c >= 0x100)) k = true;   // Danish: "s-cedilla $accent" entries are letter names only
        known_cache[c] = k;
        return k;
    }
    static int roman_value(const std::u32string& t) {
        static const std::u32string R = U"ivxlcdm";
        static const int V[] = {1, 5, 10, 50, 100, 500, 1000};
        int total = 0, prev = 0;
        for (auto it = t.rbegin(); it != t.rend(); ++it) {
            size_t k = R.find(*it);
            if (k == std::u32string::npos) return -1;
            int v = V[k];
            if (v < prev) total -= v; else { total += v; prev = v; }
        }
        // canonical form only
        std::u32string back; int n = total;
        static const std::pair<int, const char32_t*> tab[] = {{1000, U"m"}, {900, U"cm"}, {500, U"d"}, {400, U"cd"}, {100, U"c"}, {90, U"xc"},
            {50, U"l"}, {40, U"xl"}, {10, U"x"}, {9, U"ix"}, {5, U"v"}, {4, U"iv"}, {1, U"i"}};
        for (auto& p : tab) while (n >= p.first) { back += p.second; n -= p.first; }
        return back == t ? total : -1;
    }
    // Swedish: "à" alone -> "grav a" (the accent's name from the word list, then the plain letter's name)
    std::string accent_letter(char32_t c) const {
        static const struct { const char32_t* letters; const char* key; const char* base; } T[] = {
            {U"àèìòù", "_grv", "aeiou"}, {U"áíóúýćńśź", "_acu", "aiouycnsz"}, {U"âêîôû", "_cir", "aeiou"},
            {U"ãñõ", "_tld", "ano"}, {U"ëïüÿ", "_dia", "eiuy"}, {U"çş", "_ced", "cs"}, {U"øłđ", "_stk", "old"},
            {U"šžčřěň", "_hac", "szcren"}, {U"āēīōū", "_mcn", "aeiou"}, {U"ăğ", "_brv", "ag"},
        };
        for (auto& r : T) {
            std::u32string ls = r.letters;
            size_t k = ls.find(c);
            if (k == std::u32string::npos) continue;
            std::string a = ph_of(r.key);
            if (a.empty()) return "";
            return a + "\x02" + spell(std::u32string{(char32_t)r.base[k]});
        }
        return "";
    }
    // Swedish: a letter without a name is said as "bokstaven" and its code, digit by digit ("1e6f" -> ett e sex f)
    std::vector<std::string> unnamed_units(char32_t c) const {
        std::vector<std::string> u{ph_of("_?A")};
        char buf[12]; std::snprintf(buf, sizeof buf, "%x", (unsigned)sv_unnamed_lower(c));
        for (const char* q = buf; *q; q++)
            u.push_back(is_digit((char32_t)*q) ? ph_of(std::string("_") + *q) : letter_name((char32_t)*q));
        return u;
    }
    std::string letter_name(char32_t c, const std::u32string& nxt = U"") const {
        std::u32string k{c};
        const ListEntry* le = lookup_entry(std::u32string{'_', c}, false, false);
        if (le && !le->phonemes.empty()) return le->phonemes;
        if ((le = lookup_entry(k, false, false)) && !le->phonemes.empty()) return le->phonemes;
        // Portuguese: a single letter is a word like any other for the rules ("R$", "o R é")
        std::u32string w = U" " + k + U" " + nxt + U" ";
        return apply_rules(w, 1, 2, MatchCtx());
    }
    bool letter_u(char32_t c) const {   // Hindi: the entry of the letter's name has $u (य j@ $u: unstressed)
        const ListEntry* le = lookup_entry(std::u32string{'_', c}, false, false);
        if (!(le && !le->phonemes.empty())) le = lookup_entry(std::u32string{c}, false, false);
        return le && le->u;
    }
    std::string spell(const std::u32string& t, const std::u32string& next = U"", const char* join = nullptr) const {
        std::vector<std::string> units;
        for (char32_t c : t) {
            if (opt.caps_read && sv_unnamed_letter(c)) { for (auto& u : unnamed_units(c)) units.push_back(u); continue; }
            std::string nm = letter_name(c, opt.portuguese && t.size() == 1 ? next : U"");
            if (opt.indic && letter_u(c)) nm = "%" + nm;
            units.push_back(nm);
        }
        // Swedish: every third letter keeps its primary stress when at least two follow ("https": h t T p S).
        // Portuguese: five or more letters are read in groups of three, each with its own main stress
        // (a single last letter joins the group before it)
        std::string o;
        for (size_t n = 0; n < units.size(); n++) {
            bool group = (opt.caps_read && n % 3 == 0 && n + 2 <= units.size()) ||
                         (opt.portuguese && t.size() > 4 && n % 3 == 0 && t.size() - n >= 2);
            if (n) o += join ? join : (group ? "\x02" : "\x03");
            o += units[n];
        }
        return o;
    }
    // Hindi: a Devanagari character that no rule starts with (a nukta after a vowel sign or a virama)
    mutable std::unordered_map<char32_t, bool> unruled_cache_;
    bool unruled(char32_t c) const {
        auto it = unruled_cache_.find(c);
        if (it != unruled_cache_.end()) return it->second;
        bool r = !groups.count(std::u32string{c});
        if (r) for (auto& kv : groups) if (!kv.first.empty() && kv.first[0] == c) { r = false; break; }
        if (r) { auto g = groups.find(U""); if (g != groups.end()) for (int ri : g->second) if (!rules[ri].match.empty() && rules[ri].match[0] == c) { r = false; break; } }
        return unruled_cache_[c] = r;
    }

    // Turkish data: a word whose start cannot be pronounced is partly spelled. Observed behaviour of the reference:
    // the letters before an apostrophe are spelled when they hold no vowel ("km'ye"); otherwise, while the word
    // starts with three or more consonants ('y' counts as a vowel; "s" + two consonants is allowed), its first
    // letter is spelled. The rest of the word follows with its own stress, written without a space.
    bool spell_start_letters(Word& wd, const std::u32string& next, bool atend, int depth) const {
        auto isv = [&](char32_t c) { return is_vowel_letter(c) || c == 'y'; };
        const std::u32string& t = wd.text;
        size_t cut = 0; bool rest_rules_only = false;
        size_t ap = t.find('\'');
        if (ap != std::u32string::npos && ap > 1 && std::none_of(t.begin(), t.begin() + ap, isv)) {
            cut = ap; rest_rules_only = true;
        } else {
            while (cut < t.size() && t[cut] != '\'') {   // an apostrophe counts as a consonant here ("g'dir")
                size_t run = 0;
                while (cut + run < t.size() && ((is_alpha(t[cut + run]) && !isv(t[cut + run])) || t[cut + run] == '\'')) run++;
                if (run >= 3 && !(t[cut] == 's' && run == 3)) cut++; else break;
            }
            if (cut < t.size() && t[cut] == '\'') rest_rules_only = true;
        }
        size_t rest = cut;
        while (rest < t.size() && t[rest] == '\'') rest++;
        if (cut == 0 || rest >= t.size()) return false;
        Word w2 = wd; w2.text = t.substr(rest); w2.ph.clear(); w2.rules_only = rest_rules_only; w2.given = false;
        translate_word(w2, next, false, atend, U"", depth + 1);
        if (!w2.switch_lang.empty()) return false;
        encode_ph(spell(t.substr(0, cut)), wd.ph);
        wd.ph.push_back(-10);   // own stress domain, no space
        wd.ph.insert(wd.ph.end(), w2.ph.begin(), w2.ph.end());
        wd.stress_pos = 0; wd.given = false;
        return true;
    }

    // German: how many letters at the start of a word must be spelled before the rest can be read by the rules.
    // A start of two or more consonant letters is readable only if a rule written for a word start ("_) k (nA") matches it.
    // Returns the word length when fewer than three letters would be left (then the whole word is spelled).
    size_t unpronounceable_letters(const std::u32string& t) const {
        size_t n = 0;
        while (n < t.size()) {
            size_t k = n;
            while (k < t.size() && is_alpha(t[k]) && !is_vowel_letter(t[k])) k++;
            if (k - n < 2 || k >= t.size()) break;
            std::u32string w = U" " + t.substr(n) + U" ";
            bool ok = false;
            auto scan = [&](const std::u32string& key) {
                auto it = groups.find(key);
                if (it == groups.end()) return;
                for (int ri : it->second)
                    if (rules[ri].pre == U"_" && match_rule(rules[ri], w, 1, w.size() - 1, MatchCtx()) > 0) { ok = true; return; }
            };
            if (w.size() > 3) scan(std::u32string{w[1], w[2]});
            if (!ok) scan(std::u32string{w[1]});
            if (ok) break;
            n++;
        }
        if (n > 0 && t.size() - n < 3) return t.size();
        return n;
    }

    void translate_word(Word& wd, const std::u32string& next, bool atstart, bool atend, const std::u32string& prev = U"",
                        int depth = 0) const {
        std::string mn;
        if (!sym_override_.empty()) { encode_ph(sym_override_, wd.ph); return; }
        if (opt.indic && !ignore_switch_ && std::any_of(wd.text.begin(), wd.text.end(), [](char32_t c) { return c < 0x80 && is_alpha(c); })) {
            wd.switch_lang = "EN";   // Hindi: a word in Latin letters is read as English
            return;
        }
        if (opt.korean && wd.text == U"\x06") { encode_ph(ko::kIdeographTail, wd.ph); return; }
        if (opt.korean && !wd.text.empty() && !ko::is_jamo(wd.text[0]) && !is_digit(wd.text[0]) && !ignore_switch_) {
            wd.switch_lang = "EN";   // Korean data has no rules for other scripts: espeak reads them as English
            return;
        }
        if (wd.force_spell) { encode_ph(spell(wd.text), wd.ph); return; }
        if (opt.da_numbers && depth == 0) {   // Danish: stem + apostrophe + suffix (FN's, BMW'er, 60'erne): the suffix rules of group "'"
            size_t ap = wd.text.find('\'');
            if (ap != std::u32string::npos && ap > 0) {
                static const char* const sfx[][2] = {{"s", "s"}, {"en", "@-n"}, {"er", "%V"}, {"ere", "VrV"}, {"erne", "%Vn@-"},
                                                      {"ernes", "%Vn@-s"}, {"et", "@-D"}};
                std::string rest = enc(wd.text.substr(ap + 1));
                const ListEntry* stem_le = lookup_entry(wd.text.substr(0, ap), true, false);
                for (auto& kv : sfx) if (rest == kv[0] && !(stem_le && stem_le->abbrev)) {
                    Word sw; sw.text = wd.text.substr(0, ap); sw.orig = sw.text; sw.capital = wd.capital;
                    translate_word(sw, next, atstart, atend, prev, depth + 1);
                    wd.ph = sw.ph; wd.stress_pos = sw.stress_pos; wd.unstressed = sw.unstressed; wd.nounf = sw.nounf; wd.verbf = sw.verbf;
                    encode_ph(kv[1], wd.ph);
                    return;
                }
            }
        }
        if (opt.italian_alt && wd.allcaps && wd.text.size() >= 2) {   // Italian: roman numerals are ordinals (XIII -> tredicesimo)
            int v = roman_value(wd.text);
            std::string k1 = "_" + std::to_string(v) + "ox", k2 = "_" + std::to_string(v) + "o";
            if (v > 1 && v < 100 && (has(k1) || has(k2))) {
                mn = (v <= 10 && has(k1)) ? ph_of(k1) : ph_of(has(k2) ? k2 : k1);
                mn += "o";
                encode_ph(mn, wd.ph);
                return;
            }
        }
        if (opt.da_numbers && wd.allcaps && wd.text.size() >= 2 && wd.text.find_first_not_of(U"ivx") == std::u32string::npos) {
            // Danish: roman numerals up to XIX are ordinals (III -> tredje, XIV -> fjortende)
            int v = roman_value(wd.text);
            if (v >= 2 && v <= 19) {
                std::string key = "_" + std::to_string(v) + "o";
                if (v <= 12 && has(key)) mn = ph_of(key);
                else if (v >= 13) { std::string ds = std::to_string(v); mn = number_word(std::u32string(ds.begin(), ds.end())) + "@-"; }
                if (!mn.empty()) { encode_ph(mn, wd.ph); wd.number = true; return; }
            }
        }
        if (opt.french_numbers && wd.allcaps && !wd.spell_caps && wd.text.size() >= 2 && wd.text.find_first_not_of(U"ivx") == std::u32string::npos && has("_roman")) {
            int v = roman_value(wd.text);
            if (v > 0) {   // "Louis XVI" -> seize romain
                std::string ds = std::to_string(v);
                mn = number_word(std::u32string(ds.begin(), ds.end())) + "||" + ph_of("_roman");
                encode_ph(mn, wd.ph);
                wd.number = true;
                return;
            }
        }
        if (opt.roman && wd.allcaps && wd.text.size() >= 2) {
            int v = roman_value(wd.text);
            if (opt.portuguese) {   // "CC" is a listed abbreviation, not 200
                const ListEntry* ab = lookup_ctx(wd.text, wd, atstart, atend);
                if (ab && ab->abbrev) v = -1;
            }
            if (v > 0 && v < 4000) {
                std::string ds = std::to_string(v);
                mn = opt.portuguese ? pt_number(std::u32string(ds.begin(), ds.end())) : number_word(std::u32string(ds.begin(), ds.end()));
                wd.number = opt.portuguese;
                encode_ph(mn, wd.ph);
                return;
            }
        }
        if (wd.spell_all) { encode_ph(spell(wd.text), wd.ph); return; }
        if (opt.caps_read && wd.orig.size() == 1 && wd.orig[0] >= 0xC0) {
            mn = accent_letter(lower(wd.orig[0]));
            if (!mn.empty()) { encode_ph(mn, wd.ph); wd.listed = true; return; }
        }
        if (opt.german && wd.text.size() >= 2 && !list.count(wd.text)) {
            // German: II, XIV, xl (one case, value 2..49) are read "roemisch" + the number
            bool up = std::all_of(wd.orig.begin(), wd.orig.end(), is_upper);
            bool low = std::none_of(wd.orig.begin(), wd.orig.end(), is_upper);
            int v = (up || low) ? roman_value(wd.text) : -1;
            if (v >= 2 && v <= 49 && has("_roman")) {
                std::string ds = std::to_string(v);
                mn = "," + ph_of("_roman") + "||" + number_word(std::u32string(ds.begin(), ds.end()));
                wd.roman = true;
                encode_ph(mn, wd.ph);
                return;
            }
        }
        const ListEntry* le = (wd.rules_only || (opt.caps_read && wd.mapped)) ? nullptr : lookup_ctx(wd.text, wd, atstart, atend);
        if (!le && opt.swahili && wd.text.size() == 1 && !is_alpha(wd.text[0]) && !is_digit(wd.text[0]) && !symbol_rule(wd.text[0]))
            le = en_symbol(wd.text[0]);
        if (le && ignore_switch_ && le->phonemes.find("_^_") != std::string::npos) le = nullptr;
        if (le && le->text && !le->phonemes.empty() && depth < 4) {   // depth: list entries could point at each other
            // "sounds like" another word
            std::u32string t2 = dec(le->phonemes);
            Word w2; w2.text = t2;
            for (auto& c : w2.text) c = lower(c);
            translate_word(w2, opt.portuguese ? U"" : next, atstart, atend, prev, depth + 1);   // pt: no next-word context
            wd.ph = w2.ph; wd.stress_pos = w2.stress_pos; wd.unstressed = w2.unstressed;
            return;
        }
        if (le) {
            wd.nounf = le->nounf; wd.verbf = le->verbf; wd.brk = le->brk; wd.unstressend = le->unstressend;
            if (opt.prefix_word_flags && le->verb && le->verbextend) wd.verbf = true;   // $verbextend: still a verb next
            wd.stress_pos = le->stress_pos;
            wd.alt = le->alt;
            wd.listed_marks = le->phonemes.find_first_of("',") != std::string::npos;
            wd.unstressed = le->u; wd.u_plus = le->u_plus || le->strend; wd.u_n = le->u_n;
            if (opt.swahili && le->unstressend && atend) wd.unstressed = true;
            wd.unstressend = le->unstressend;
            wd.dbl_next = le->dbl;
            wd.fi_pause_before = le->pause;
            mn = le->phonemes;
            if (opt.da_numbers && mn.size() > 1 && mn.back() == '\'') {   // Danish: a stress mark after the last vowel ("mc m?Ak'") = secondary on it
                mn.pop_back();
                size_t k = mn.find_last_of("aeiouyAEIOUVWY&@03");
                if (k != std::string::npos) { if (k > 0 && mn[k - 1] == '?') k--; mn.insert(k, ","); }
            }
            wd.given = !mn.empty();
            wd.listed = !mn.empty();
        }
        if (mn.empty() && wd.symbols) {   // "//", "++": one word read by the rules
            std::u32string w = U" " + wd.text + U" ";
            mn = apply_rules(w, 1, 1 + wd.text.size(), MatchCtx());
        }
        if (mn.empty()) {
            bool all_digits = !wd.text.empty() && std::all_of(wd.text.begin(), wd.text.end(), is_digit);
            bool no_vowel = !all_digits && std::none_of(wd.text.begin(), wd.text.end(), [&](char32_t c) {
                return is_vowel_letter(c) || ((opt.swahili || opt.y_is_vowel) && c == 'y') ||
                       ((opt.german || opt.german_en) && (c == U'y' || c == U'Y') && wd.text.size() > 2 && !wd.allcaps) ||   // German: "Sky" has the rule '_) sky _^_EN', it is not spelled
                        (opt.indic && hi::is_consonant(c)) ||
                                                                                                      (opt.en_for_indic && (c == 'y' || c == 'Y')); });
            bool spell_caps = wd.allcaps && wd.text.size() <= 3 && (!opt.caps_spell_by_clause || clause_caps_spell_) && (!opt.da_numbers || caps_spell_ok_) &&
                              (!opt.caps_read || (clause_lower_ >= 4 && clause_lower_ >= clause_upper_)) &&
                              (!opt.french_numbers || clause_spells_caps_) &&
                              !(opt.portuguese && caps_clause_) && !(opt.german && clause_allcaps_);
            // French: a word that starts with more than three consonant letters (a first "s" is not counted) is not
            // read as it stands: its first letter is spelled and the rest is tried again (RSPCA -> "R" + spca,
            // Schtroumpf -> "S", "C" + htroumpf; Schneider is read whole)
            // English words of Korean text: how many leading letters are said as letter names (-1: the whole word)
            int en_k = (under_korean_ && !all_digits) ? ((le && le->abbrev) ? -1 : ko::english_spell_prefix(wd.text)) : 0;
            size_t n_spelled = 0;
            size_t unpron = (opt.german && !all_digits && !no_vowel && depth < 4) ? unpronounceable_letters(wd.text) : 0;
            if (opt.french_numbers && !all_digits && !no_vowel && !le && depth == 0) {
                auto too_many = [&](size_t from) {
                    size_t k = from, cnt = 0;
                    while (k < wd.text.size() && is_alpha(wd.text[k]) && !is_vowel_letter(wd.text[k])) { if (!(k == from && wd.text[k] == 's')) cnt++; k++; }
                    return cnt > 3 && k < wd.text.size() && is_vowel_letter(wd.text[k]);
                };
                while (n_spelled < wd.text.size() && too_many(n_spelled)) n_spelled++;
            }
            if (n_spelled) {
                Word w2 = wd; w2.text = wd.text.substr(n_spelled); w2.ph.clear(); w2.allcaps = false;
                translate_word(w2, next, false, atend, U"", depth + 1);
                if (w2.switch_lang.empty() && !w2.ph.empty()) {
                    encode_ph(spell(wd.text.substr(0, n_spelled)), wd.ph);
                    wd.ph.push_back(-10);
                    wd.ph.insert(wd.ph.end(), w2.ph.begin(), w2.ph.end());
                    return;
                }
                switch_lang_.clear();
            }
            if (all_digits && opt.indic && wd.digits_single) {   // Hindi: digits after a decimal point, one by one
                for (char32_t d : wd.text) { if (!mn.empty()) mn += "||"; mn += ph_of(std::string("_") + (char)d); }
                wd.number = true;
            } else if (all_digits && opt.indic) {
                mn = hi::number(wd.text, [this](const std::string& k) { return ph_of(k); });
                wd.number = true;
                wd.listed_marks = true;   // the parts keep the stress of their entries
            }
            else if (all_digits && opt.korean) { mn = ko::number(wd.text, [&](const std::string& k) { return ph_of(k); }); wd.number = true; }
            else if (all_digits && opt.french_numbers && wd.after_comma && wd.text.find_first_not_of(U'0') == std::u32string::npos) {
                for (size_t q = 0; q < wd.text.size(); q++) mn += ph_of("_0");   // "3,00": the zeros in one word
                wd.number = true;
            }
            else if (all_digits && wd.digit_by_digit && wd.text.size() > 2) {   // 2 decimals read as a number
                for (char32_t d : wd.text) { if (!mn.empty()) mn += "||"; mn += ph_of(std::string("_") + (char)d); }
                wd.number = true; wd.given = true;
            }
            else if (all_digits && wd.digits_each) {
                for (char32_t d : wd.text) { if (!mn.empty()) mn += "||"; mn += ph_of(std::string("_") + (char)d); }
                wd.number = true;
            } else if (all_digits) {
                mn = opt.sv_numbers ? sv::number(enc(wd.text), [this](const std::string& k) { return ph_of(k); }, wd.year_group, wd.grouped)
                   : opt.portuguese ? pt_number(wd.text, wd.grouped, wd.ordinal, wd.fraction) : number_word(wd.text);
                wd.number = true; wd.given = opt.suffix_once && !wd.superscript; wd.listed = true;
                if (opt.da_numbers) wd.no_default = true;
                wd.digit_groups = opt.french_numbers && wd.text.size() >= 4 && wd.text[0] == '0';
                if (opt.finnish) wd.listed_marks = true;
                if (opt.german) mn += wd.num_suffix;
            }
            else if (opt.indic && std::any_of(wd.text.begin(), wd.text.end(), [&](char32_t c) {
                         return hi::is_devanagari(c) && unruled(c); }))
                mn = spell(wd.text, U"", "||");   // Hindi: a word with a character no rule reads is spelled, letter by letter
            else if (en_k > 0 && depth < 8) {
                Word w2; w2.text = wd.text.substr(en_k); w2.orig = w2.text;
                translate_word(w2, next, false, atend, U"", depth + 1);
                encode_ph(spell(wd.text.substr(0, en_k)), wd.ph);   // the named letters are one stress group
                wd.ph.push_back(-10);
                wd.ph.insert(wd.ph.end(), w2.ph.begin(), w2.ph.end());
                return;
            }
            else if (en_k < 0) {
                size_t at = 0;
                for (int g : ko::english_spell_groups(wd.text.size())) {
                    if (at) mn += "\x02";
                    mn += spell(wd.text.substr(at, g));
                    at += g;
                }
            }
            else if (!under_korean_ && ((no_vowel && !(opt.swahili && le && !le->abbrev) && !(opt.letter_by_rules && wd.join_prev && ctx_next_hyph_ && wd.text.size() == 1)) ||
                     (le && le->abbrev) || spell_caps || wd.spell_caps || force_spell_))
            {   // (Swahili: a listed word with flags only, "m", is read by the rules; a spelled word drops the list flags)
                if (opt.caps_read && wd.allcaps) { wd.unstressed = false; wd.u_plus = false; wd.u_n = 0; }   // "ER", "VI": letters keep their stress
                mn = spell(wd.text, next);
                if (opt.da_numbers && wd.text.size() == 1 && !next.empty()) {   // Danish: a single letter takes the rules with their context (t-shirt)
                    std::u32string w1 = U" " + prev + U" " + wd.text + U" " + next + U" ";
                    MatchCtx mc1;
                    std::string m1 = apply_rules(w1, prev.size() + 2, prev.size() + 3, mc1);
                    if (!m1.empty()) mn = m1;
                }
                if (opt.da_numbers) { wd.unstressed = false; wd.u_plus = false; wd.u_n = 0; }   // spelled letters: the word's own $u flag does not apply (HER)
                wd.spelled = true;
                if (opt.german && no_vowel && wd.text.size() >= 4 && !ignore_switch_ && !le) {
                    // German: a rule written for a word start can switch to English ("_) ms _^_EN": MSNBC is then spelled with English letter names)
                    std::u32string w = U" " + prev + U" " + wd.text + U" " + next + U" ";
                    size_t b = prev.size() + 2;
                    MatchCtx mc;
                    stem_alt_ = 0; stem_stress_pos_ = 0; stem_u_ = false; rest_stress_pos_ = 0; prefix_taken_ = false;
                    cur_capital_ = wd.capital; no_switch_ = false; suffix_taken_ = false;
                    std::string r = apply_rules(w, b, b + wd.text.size(), mc);
                    cur_capital_ = no_switch_ = false;
                    if (!switch_lang_.empty() || r.find("_^_") != std::string::npos) mn = r;
                    else switch_lang_.clear();
                }
                if (opt.swahili) { wd.unstressed = wd.u_plus = false; wd.u_n = 0; }
                if (opt.portuguese && !(le && le->abbrev)) { wd.unstressed = false; wd.u_plus = false; wd.u_n = 0; wd.stress_pos = 0; }
            }
            else if (unpron >= wd.text.size()) mn = spell(wd.text);
            else if (unpron > 0) {
                // German: a word start that no rule allows ("Bhog", "WLAN") is read letter by letter up to a readable rest
                Word w2; w2.text = wd.text.substr(unpron); w2.orig = w2.text; w2.capital = false;
                translate_word(w2, next, false, atend, U"", depth + 4);
                if (w2.switch_lang.empty()) {
                    encode_ph(spell(wd.text.substr(0, unpron)), wd.ph);
                    wd.ph.push_back(-10);
                    wd.ph.insert(wd.ph.end(), w2.ph.begin(), w2.ph.end());
                    return;
                }
                switch_lang_.clear();
                mn = spell(wd.text);
            }
            else if (opt.spell_clusters && !le && depth < 4 && spell_start_letters(wd, next, atend, depth)) return;
            else {
                std::u32string w = U" " + prev + U" " + wd.text + U" " + next + U" ";
                if (opt.letter_by_rules) {
                    // letters with a diacritic and no rules of their own are read as the base letter (the list was
                    // looked up with the word as written); a hyphen joining words is kept for the rules (-c) e (_)
                    w = U" " + prev + (wd.join_prev ? U"-" : U" ") + wd.text + (ctx_next_hyph_ ? U"-" : U" ") + next + U" ";
                    for (auto& c : w) if (c >= 0x80 && !known_letter(c)) c = base_letter(c);
                }
                size_t b = prev.size() + 2;
                MatchCtx mc;
                if (le) mc.alt = le->alt;
                if (((le && le->only) || wd.rules_only) && opt.suffix_once) mc.no_suffix = true;   // "$only" word / after a number
                cur_expect_noun_ = wd.expect_noun;
                mc.verb = wd.expect_verb; mc.noun = wd.expect_noun;
                stem_alt_ = 0; stem_stress_pos_ = 0; stem_u_ = false; stem_u_plus_ = false; stem_uplus_ = false; stem_un_ = 0; rest_stress_pos_ = 0; prefix_taken_ = false;
                rest_entry_ = nullptr; suffix_taken_ = false;
                cur_capital_ = wd.capital; no_switch_ = le != nullptr; rest_u_ = rest_u_plus_ = false;
                accent_miss_ = -1; stem_marks_ = false;
                stem_given_ = false;
                stem_listed_ = false;
                mn = apply_rules(w, b, b + wd.text.size(), mc);
                cur_capital_ = no_switch_ = false;
                wd.suffixed = suffix_taken_;
                wd.rest_u = rest_u_; wd.rest_u_plus = rest_u_plus_;
                for (int tries = 0; tries < 4 && under_korean_ && accent_miss_ >= 0 && (size_t)accent_miss_ < w.size(); tries++) {
                    // a letter with a diacritic that no rule reads: the word again with its base letter ("Cochamó")
                    w[accent_miss_] = base_letter(w[accent_miss_]);
                    accent_miss_ = -1;
                    stem_alt_ = 0; stem_stress_pos_ = 0; stem_u_ = false; rest_stress_pos_ = 0; prefix_taken_ = false;
                    mn = apply_rules(w, b, b + wd.text.size(), mc);
                }
                if (opt.prefix_word_flags && prefix_taken_ && rest_entry_) {
                    const ListEntry* re = rest_entry_;
                    wd.unstressed = re->u; wd.u_plus = re->u_plus || re->strend; wd.u_n = re->u_n;
                    wd.nounf = re->nounf; wd.verbf = re->verbf || (re->verb && re->verbextend);
                    wd.rest_suffix = suffix_taken_;
                }
                if (stem_listed_) wd.listed = true;
                const bool whole_one = opt.german && prefix_taken_ && !rest_stress_pos_ && stem_stress_pos_ == 1;   // German: the stem "überschrift $1" counts from the word's first syllable
                if (prefix_taken_ && !rest_stress_pos_ && !whole_one) rest_stress_pos_ = stem_stress_pos_;
                if (prefix_taken_ && !whole_one) stem_stress_pos_ = 0;
                wd.rest_stress_pos = rest_stress_pos_;
                if (!(opt.portuguese && le && le->alt)) wd.alt |= stem_alt_;   // pt: the word's own $alt wins
                if (stem_u_ && (opt.post_secondary || opt.italian_alt || opt.suffix_once || opt.portuguese || opt.prefix_word_flags)) wd.unstressed = true;
                if (stem_u_ && opt.stem_unstressed && !le) { wd.unstressed = true; wd.u_plus = stem_u_plus_; }   // Danish
                if (opt.portuguese && stem_u_ && stem_uplus_) wd.u_plus = true;
                if (stem_u_ && stem_uplus_ && opt.prefix_word_flags) wd.u_plus = true;
                if (opt.portuguese && stem_u_ && stem_un_ && !wd.u_n) wd.u_n = stem_un_;
                if (stem_given_ && opt.suffix_once) wd.given = true;
                if (!wd.stress_pos && stem_stress_pos_) wd.stress_pos = stem_stress_pos_;
                // Portuguese: the s of "'s" ("Sainsbury's") is never voiced by the next word
                if (opt.portuguese && wd.text.size() > 2 && wd.text.compare(wd.text.size() - 2, 2, U"'s") == 0 &&
                    mn.size() >= 3 && mn.compare(mn.size() - 3, 3, "s#\x05") == 0) mn.erase(mn.size() - 2, 1);
            }
        }
#ifdef KT_FE_TRACE
        { std::string v; for (char ch : mn) { if (ch >= 1 && ch <= 5) { v += '<'; v += char('0' + ch); v += '>'; } else v += ch; }
          fprintf(stderr, "word %s -> mn %s (stress_pos %d rest %d u %d)\n", enc(wd.text).c_str(), v.c_str(), wd.stress_pos, wd.rest_stress_pos, (int)wd.unstressed); }
#endif
        size_t sw = mn.find("_^_");
        if (sw != std::string::npos) {
            if (!ignore_switch_) {
                std::string code;
                for (size_t k = sw + 3; k < mn.size() && k < sw + 5 && std::isalpha((unsigned char)mn[k]); k++)
                    code += (char)std::toupper((unsigned char)mn[k]);
                switch_lang_ = code;
            }
            mn.clear();
        }
        if (!switch_lang_.empty()) {
            wd.switch_lang = switch_lang_;
            switch_lang_.clear();
            return;
        }
#ifdef KT_FE_TRACE
        fprintf(stderr, "WORD\t%s\t%s\n", enc(wd.text).c_str(), mn.c_str());
#endif
        encode_ph(mn, wd.ph);
        if (opt.finnish) {   // a rule ending in _X1 ("-lle", "-sti", ...) doubles the next word's first consonant
            int x1 = ps.find("_X1");
            if (x1 >= 0 && std::find(wd.ph.begin(), wd.ph.end(), x1) != wd.ph.end()) wd.dbl_next = true;
        }
        if (under_korean_) {
            // English inside Korean: a prefix and the rest of the word are one stress domain ("open" + "ing")
            wd.ph.erase(std::remove(wd.ph.begin(), wd.ph.end(), -12), wd.ph.end());
            wd.stem_marks = stem_marks_;
        }
        if (opt.sv_english) {
            int n = ps.find("n"), k = ps.find("k"), ng = ps.find("N");
            for (size_t i = 0; n >= 0 && k >= 0 && ng >= 0 && i + 1 < wd.ph.size(); i++) if (wd.ph[i] == n && wd.ph[i + 1] == k) wd.ph[i] = ng;
        }
    }

    // ---------- stress
    bool is_vowel_ph(int p) const { return p >= 0 && ps.defs[p].type == T_VOWEL && !ps.defs[p].nonsyl; }
    int stress_of(int p) const { return (p >= 0 && ps.defs[p].type == T_STRESS) ? ps.defs[p].stress_type : -1; }

    // Assign a stress level to each vowel of a word. Returns per-item stress (vowels only), dropping stress phonemes.
    void word_stress(const Word& wd, bool clause_end, std::vector<std::pair<int, int>>& out) const {
        struct V { int ph; int st; bool pct = false; };
        std::vector<V> items;
        int pending = -1;
        bool pending_pct = false;   // Danish
        bool has_primary = false;
        bool no_default = wd.no_default;   // Danish may set it further down; for the other languages it stays wd.no_default
        for (size_t pi = 0; pi < wd.ph.size(); pi++) {
            int p = wd.ph[pi];
            if (opt.syllabic_blocks && p >= 0 && ps.defs[p].name == "-" && !items.empty() && is_vowel_ph(items.back().ph) &&
                std::count_if(items.begin(), items.end(), [&](const V& x) { return is_vowel_ph(x.ph); }) == 1 && items.back().st < 2) {
                no_default = true; items.back().st = 1;   // Danish: the syllabic mark after the first vowel
            }
            int s = stress_of(p);
            if (s >= 0) {
                if (ps.defs[p].name == "=") {   // stress on the previous vowel
                    if (opt.portuguese) {   // ... unless a vowel already has priority stress ('', an accent)
                        bool pri = false;
                        for (auto& it : items) if (it.st >= 5) pri = true;
                        if (pri) continue;
                    }
                    for (auto it = items.rbegin(); it != items.rend(); ++it) if (is_vowel_ph(it->ph) && !(opt.german && ps.defs[it->ph].unstressed)) { if (!(opt.all_primaries && it->st >= 4)) it->st = opt.all_primaries ? 7 : 4; has_primary = true; break; }   // Danish: level 7, resolved below
                    continue;
                }
                pending = s;
                if (opt.all_primaries && ps.defs[p].name == "%") pending_pct = true;
                continue;
            }
            if (ps.defs[p].type == T_PAUSE) {
                // Danish: a first vowel right before a short pause "_!" inside the word (u_!lykke) takes no stress of its own
                if (opt.da_trochaic && ps.defs[p].name == "_!" && !items.empty() && is_vowel_ph(items.back().ph) && items.back().st < 0 &&
                    pi + 1 < wd.ph.size() && std::count_if(items.begin(), items.end(), [&](const V& x) { return is_vowel_ph(x.ph); }) == 1)
                    items.back().st = 1;
                if (opt.keep_word_pauses) items.push_back(V{p, -1});
                continue;
            }
            V v{p, -1};
            if (is_vowel_ph(p) && opt.all_primaries && pending >= 4 && !items.empty() && is_vowel_ph(items.back().ph) &&
                ps.defs[items.back().ph].name == "?i" && ps.defs[p].name == "E" && pi + 1 < wd.ph.size() && wd.ph[pi + 1] >= 0 && ps.defs[wd.ph[pi + 1]].name == ":") {
                // measured (Danish): a primary mark between a vowel and a long vowel ("?i'E:" in karriere) passes on to the next vowel
            } else if (is_vowel_ph(p)) {
                v.st = pending; v.pct = pending_pct;
                if (pending >= 4) has_primary = true;
                pending = -1; pending_pct = false;
            }
            items.push_back(v);
        }
        if (opt.all_primaries && pending >= 4) {   // measured (Danish): a primary mark after the last vowel ("?E'") = secondary stress on that vowel
            for (auto it = items.rbegin(); it != items.rend(); ++it) if (is_vowel_ph(it->ph)) { if (it->st < 2) it->st = 2; break; }
        }
        std::vector<int> vi;
        for (size_t i = 0; i < items.size(); i++) if (is_vowel_ph(items[i].ph)) vi.push_back((int)i);
        int nv = (int)vi.size();
        // demote '' (5) / emphasized to primary 4; extra primaries stay
        if (opt.portuguese) {   // priority stress (an accent) wins over the other primary stresses of the word
            bool pri = false;
            for (auto& it : items) if (it.st >= 5) pri = true;
            if (pri) for (auto& it : items) if (it.st == 4) it.st = -1;
            if (pri) for (auto& it : items) if (it.st >= 5) it.st = 6;   // every accented vowel keeps its primary
        }
        if (!opt.all_primaries) for (auto& it : items) if (it.st > 4 && !(opt.portuguese && it.st == 6)) it.st = 4;
        if (opt.all_primaries) {
            // Danish (measured on the reference): with a priority stress ('') in the word, the plain primaries (') and a
            // '=' stress become secondary; without one, a '=' stress (level 7: on a vowel that had no primary) turns the
            // plain primaries into secondaries; otherwise every primary stays primary.
            bool any5 = false, any_eq = false;
            for (auto& it : items) { if (it.st == 5 || it.st == 6) any5 = true; if (it.st == 7) any_eq = true; }
            if (any5) { for (auto& it : items) if (it.st == 4 || it.st == 7) it.st = 2; }
            else if (any_eq) {   // only the primaries before the last '=' vowel
                size_t last_eq = 0;
                for (size_t k = 0; k < items.size(); k++) if (items[k].st == 7) last_eq = k;
                for (size_t k = 0; k < last_eq; k++) if (items[k].st == 4 || items[k].st == 7) items[k].st = 2;
            }
            for (auto& it : items) if (it.st > 4) it.st = 4;
        } else if (opt.keep_all_primary || opt.indic || opt.german) {   // Hindi: every primary the phonemes mark stays (list entries such as kUld'e:vt'a:)
        } else if (opt.keep_first_primary) {
            bool seen = false;
            for (auto& it : items) if (it.st >= 4) { if (seen) it.st = 2; seen = true; }
        } else if (!(opt.portuguese && wd.listed_marks)) {   // pt: a listed word keeps all its primary stresses
            int last = -1;
            for (size_t k = 0; k < items.size(); k++) if (items[k].st >= 4) last = (int)k;
            if (!(opt.swahili && wd.number))   // Swahili number words keep every listed primary: l'akim'oJa
                for (int k = 0; k < last; k++) if (items[k].st >= 4 && items[k].st != 6) items[k].st = 2;
        }
        for (auto& it : items) if (it.st == 6 && opt.portuguese) it.st = 4;
        bool unstressed_word = wd.unstressed && !(wd.u_plus && clause_end);
        if (wd.u_plus && clause_end && !wd.unstressed) {
            bool any_p = false;
            for (auto& it : items) if (it.st >= 4) any_p = true;
            if (!any_p) for (auto& it : items) if (it.st >= 2) { it.st = 4; break; }
        }
        if (wd.no_stress) {
            for (auto& it : items) out.push_back({it.ph, is_vowel_ph(it.ph) ? 1 : -1});
            return;
        }
        if (wd.marks_only) {   // Hindi decimal point word: only the stress its entry marks
            for (auto& it : items) out.push_back({it.ph, is_vowel_ph(it.ph) ? (it.st >= 0 ? it.st : 1) : -1});
            return;
        }
        int spos = wd.stress_pos;
        if (opt.portuguese && !spos && wd.u_plus && wd.u_n && clause_end) spos = wd.u_n;   // $u1+ at the end
        if (spos > 0 && spos <= nv) {
            for (auto& it : items) if (it.st >= 4) it.st = -1;
            items[vi[spos - 1]].st = 4; has_primary = true;
        }
        if (unstressed_word && nv == 1) for (auto& it : items) if (it.st >= 2) it.st = 1;
        bool any_mark = false;
        for (auto& it : items) if (it.st >= 2) any_mark = true;
        if (under_korean_ && !has_primary && any_mark && !unstressed_word && !wd.no_default && wd.stem_marks) {
            // English inside Korean: a word with secondary stress only (a listed stem such as m,eIk + ing): the first
            // one is its primary
            for (auto& it : items) if (it.st >= 2) { it.st = 4; has_primary = true; break; }
        }
        if (opt.first_secondary_blocks && !wd.rest_part && nv > 0 && !has_primary && (items[vi[0]].st == 2 || items[vi[0]].st == 3)) no_default = true;   // Danish
        bool marks_block = wd.listed_marks && any_mark && !opt.trochaic && !(opt.french_numbers && !unstressed_word && nv >= 2);
        if (!has_primary && nv > 0 && !no_default && !marks_block) {
            int target = default_stress_pos(wd, items.size(), vi, [&](int k) { return items[vi[k]].st; },
                                            [&](int k) { return ps.defs[items[vi[k]].ph].unstressed || (items[vi[k]].st >= 0 && items[vi[k]].st < 2); });
            bool any_free = force_target_;
            force_target_ = false;
            for (int k = 0; k < nv; k++) if (items[vi[k]].st < 0 && !ps.defs[items[vi[k]].ph].unstressed) any_free = true;
            if (!any_free) target = -1;
            if (unstressed_word) {
                if (((nv + wd.extra_syl) > 1 || wd.u_n) && target >= 0 && !(opt.trochaic && any_mark)) {
                    int t = wd.u_n ? std::min(wd.u_n, nv) - 1 : target;
                    if (items[vi[t]].st < 0) items[vi[t]].st = 2;
                    if (opt.indic)
                        for (int k = t + 2; k < nv; k += 2)
                            if (items[vi[k]].st < 0 && items[vi[k - 1]].st < 2 && !ps.defs[items[vi[k]].ph].unstressed) items[vi[k]].st = 2;
                }
            } else if (target >= 0) { items[vi[target]].st = 4; has_primary = true; }
        } else if (unstressed_word && opt.swahili && wd.u_n && nv > 0) {   // $u1: only that syllable keeps a (secondary) stress
            for (auto& it : items) if (it.st >= 2) it.st = -1;
            items[vi[std::min(wd.u_n, nv) - 1]].st = 2;
            has_primary = false;
        } else if (unstressed_word && has_primary) {
            for (auto& it : items) if (it.st == 4) it.st = nv > 1 ? 2 : 1;
            has_primary = false;
        }
        if (opt.sv_english && wd.unstressend && clause_end) for (auto& it : items) if (it.st >= 4) it.st = 2;
        // secondary stress before the primary: alternate from the first syllable, never next to a stressed one
        int prim = -1;
        for (int k = 0; k < nv; k++) if (items[vi[k]].st >= 4) { prim = k; break; }
        if (opt.finnish && nv > 0) {
            std::vector<int> st(nv);
            std::vector<bool> heavy(nv, false);
            for (int k = 0; k < nv; k++) {
                st[k] = items[vi[k]].st;
                if (ps.defs[items[vi[k]].ph].lng) { heavy[k] = true; continue; }
                size_t end = k + 1 < nv ? (size_t)vi[k + 1] : items.size();
                int cons = 0;   // consonants (a length mark counts as one) up to the next vowel
                for (size_t q = (size_t)vi[k] + 1; q < end; q++) {
                    const PhDef& d = ps.defs[items[q].ph];
                    if (d.name == ":" && q == (size_t)vi[k] + 1) { heavy[k] = true; break; }
                    if (d.type != T_VOWEL) cons++;
                }
                if (cons >= 2 || (k + 1 == nv && cons >= 1)) heavy[k] = true;
            }
            lang_fi::secondary_stress(heavy, st);
            for (int k = 0; k < nv; k++) items[vi[k]].st = st[k];
        }
        int ptprim = prim;
        if (opt.portuguese && prim < 0 && unstressed_word)
            for (int k = 0; k < nv; k++) if (items[vi[k]].st >= 2) ptprim = k;
        if (opt.da_trochaic && !unstressed_word && !no_default) {
            // Danish: from the start, a free syllable between two unstressed (or no) neighbours is stressed
            // (primary if the word has none yet, else secondary); this includes the last syllable.
            bool anyp = prim >= 0;
            for (int k = 0; k < nv; k++) {
                int& s = items[vi[k]].st;
                if (s >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                bool pv = k == 0 || items[vi[k - 1]].st < 2, nx = k == nv - 1 || items[vi[k + 1]].st < 2;
                if (pv && nx) { s = anyp ? 2 : 4; anyp = true; }
            }
            if (prim < 0) for (int k = 0; k < nv; k++) if (items[vi[k]].st >= 4) { prim = k; break; }
        } else if (opt.da_trochaic) {
            // Danish words that only take the stress they mark (numbers, 2nd word of a list phrase): nothing more
        } else if (opt.trochaic && !unstressed_word && !wd.no_default) {   // (no_default: a prefix such as "hon-")
            for (int k = 0; k < nv; k++) {
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                bool prev_weak = k == 0 || items[vi[k - 1]].st < 2;
                bool next_weak = k + 1 == nv || items[vi[k + 1]].st < 2;
                if (prev_weak && next_weak) items[vi[k]].st = 2;
            }
        } else if (ptprim > 0 && opt.portuguese) {
            // Portuguese: secondary on every other syllable from the first, not next to the primary (the first
            // syllable of a word of 3+ syllables is the exception); none in the later parts of a number
            for (int k = 0; k < ptprim && !wd.number_tail; k += 2) {
                if (k == ptprim - 1 && !(k == 0 && nv >= 3 && prim >= 0 && !wd.unstressed)) continue;
                if (items[vi[k]].st < 0 && !ps.defs[items[vi[k]].ph].unstressed) items[vi[k]].st = 2;
            }
        } else if (prim > 0 && wd.no_pre_secondary) {
        } else if (prim > 0 && opt.secondary_first_only) {
            if (items[vi[0]].st < 0 && !ps.defs[items[vi[0]].ph].unstressed) items[vi[0]].st = 2;
        } else if (prim > 0 && opt.secondary_from_start) {
            for (int k = 0; k < prim - 1; k++) {
                if (items[vi[k]].st >= 0) continue;
                if (ps.defs[items[vi[k]].ph].unstressed) continue;
                if (opt.syllabic_no_secondary && ps.defs[items[vi[k]].ph].ipa.find("\xcc\xa9") != std::string::npos) continue;
                bool prev_st = k > 0 && items[vi[k - 1]].st >= 2;
                bool next_st = items[vi[k + 1]].st >= 2;
                if (!prev_st && !next_st) items[vi[k]].st = 2;
            }
        }
        if (prim >= 0 && opt.post_secondary && !opt.trochaic) {
            for (int k = prim + 1; k < nv; k++) {
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                if (items[vi[k - 1]].st >= 2) continue;
                if (opt.german && k + 1 < nv && items[vi[k + 1]].st >= 2) continue;   // not right before a marked stress (Informa-tion)
                items[vi[k]].st = 2;
            }
        }
        if (prim >= 0 && opt.german_en && nv - 1 >= prim + 2) {
            int k = nv - 1;
            if (items[vi[k]].st < 0 && !ps.defs[items[vi[k]].ph].unstressed && items[vi[k - 1]].st < 2) items[vi[k]].st = 2;
        }
        if (prim >= 0 && under_korean_) {
            // English words of Korean text (measured): after the primary, every syllable that can take stress and
            // is not next to a stressed one gets secondary stress, the last syllable too
            for (int k = prim + 1; k < nv; k++) {
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                if (items[vi[k - 1]].st >= 2 || (k + 1 < nv && items[vi[k + 1]].st >= 2)) continue;
                items[vi[k]].st = 2;
            }
        } else if (prim >= 0 && opt.secondary_from_start && !opt.post_secondary && !opt.trochaic && !opt.da_trochaic && !wd.no_post_secondary) {
            for (int k = prim + 2; k < (opt.korean ? nv : nv - ((opt.final_secondary || opt.indic) ? 0 : 1)); k += 2) {   // Korean: the last syllable too
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                if (items[vi[k - 1]].st >= 2 || (k + 1 < nv && items[vi[k + 1]].st >= 2)) continue;
                items[vi[k]].st = 2;
            }
        }
        if (opt.all_primaries && nv == 3 && items[vi[2]].pct && items[vi[2]].st == 1 && items[vi[1]].st < 2 &&
            wd.orig.size() > 4 && lower(wd.orig[0]) == 'u' && lower(wd.orig[1]) == 'd')
            items[vi[2]].st = 2;   // measured (Danish): "ud" + a 3-syllable rest with a "%" ending: the last syllable takes a secondary stress (udbydere)
        bool final_open = nv > 0 && vi.back() + 1 == (int)items.size() && !wd.hyphen_next;
        for (int k = 0; k < nv; k++) {
            int& s = items[vi[k]].st;
            if (opt.sv_english && k == 0 && s == 0) s = 1;   // "%%" on a word's first vowel: unstressed, not reduced ("of the")
            if (under_korean_ && s < 0) {
                // English words of Korean text (measured): an unstressed syllable inside the word is 'diminished'
                // (reduced vowel), unless only an unstressed last syllable follows; the first and last are not
                bool last_unstressed = items[vi[nv - 1]].st < 2;
                s = (k > 0 && k < nv - 1 && !(k == nv - 2 && last_unstressed)) ? 0 : 1;
                continue;
            }
            if (s < 0 && opt.trochaic) {
                if (unstressed_word && nv == 1 && !wd.listed) { s = 1; continue; }
                if (wd.listed) { s = kSoftUnstressed; continue; }
            }
            if (opt.trochaic && wd.listed && s >= 0 && s <= 1 && !(unstressed_word && nv == 1)) { s = kSoftUnstressed; continue; }
            if (s < 0 && opt.indic) s = (k + 1 == nv && final_open) ? 0 : 1;   // Hindi: only an unstressed open final syllable is diminished
            // German: an unmarked syllable is 'unstressed' only right after a stressed one, not before a stressed one, and as
            // the last or second-last syllable; else 'diminished' (only [E2] shows it: Rechteck, Bettdecke / Dokument-ieren)
            if (s < 0 && opt.german) s = ((k == 0 || items[vi[k - 1]].st >= 2) && !(k + 1 < nv && items[vi[k + 1]].st >= 2) && k + 2 >= nv) ? 1 : 0;
            if (s < 0 && opt.german_en && prim >= 0 && k > prim && k == nv - 1) s = 1;   // English word in German text: the last syllable keeps its full vowel (Laptop, Teamwork)
            if (s < 0) s = (prim >= 0 && k > prim) ? 0 : (opt.dim_non_initial && k > 0 && items[vi[k - 1]].st >= 2) ? 0 : 1;
        }
        if (opt.sv_english) {   // English inside Swedish: a weak vowel is reduced only right before a stressed syllable
            for (int k = 0; k < nv; k++) {   // ("mångfacetterade": fa -> f6; "network", "desktop": the last vowel stays)
                int& s = items[vi[k]].st;
                if (s == 0 || s == 1) s = (k + 1 < nv && items[vi[k + 1]].st >= 2) ? 0 : 1;
            }
        }
        if ((opt.italian_alt || opt.portuguese) && wd.alt) {
            int top = 0;
            for (auto& it : items) top = std::max(top, it.st);
            for (auto& it : items) {
                if (it.st < 2 || it.st < top) continue;
                const std::string& nm = ps.defs[it.ph].name;
                std::string to;
                if (wd.alt & 1) { if (nm == "e") to = "E"; else if (nm == "o") to = "O"; }
                if (wd.alt & 2) { if (nm == "E") to = "e"; else if (nm == "O") to = "o"; }
                int p = to.empty() ? -1 : ps.find(to);
                if (p >= 0) it.ph = p;
            }
        }
        for (auto& it : items) out.push_back({it.ph, it.st});
    }

    template <class StF, class UnF>
    int default_stress_pos(const Word& wd, size_t, const std::vector<int>& vi, StF st, UnF unst) const {
        int nv = (int)vi.size();
        if (nv == 0) return -1;
        if (opt.korean) {
            std::string shape;
            for (int p : wd.ph) if (p >= 0 && ps.defs[p].type != T_STRESS && ps.defs[p].type != T_PAUSE) shape += is_vowel_ph(p) ? 'V' : 'C';
            return ko::stress_vowel(shape);
        }
        if (under_korean_) {   // English words of Korean text: the first syllable that can take stress (measured)
            for (int k = 0; k < nv; k++) if (st(k) < 0 && !unst(k)) return k;
            return 0;
        }
        int t;
        if (opt.stress_rule == 6) {   // Hindi: weight of each syllable from the word's phonemes (lang/hi.cpp)
            std::vector<hi::Syl> syl;
            bool prev_cons = false;
            for (int p : wd.ph) {
                if (p < 0) continue;
                const PhDef& d = ps.defs[p];
                if (d.type == T_STRESS || d.type == T_PAUSE) continue;
                if (is_vowel_ph(p)) { hi::Syl s; s.long_vowel = d.lng; syl.push_back(s); prev_cons = false; continue; }
                if (syl.empty()) continue;
                if (d.type == T_VIRTUAL) { if (d.name == ":" && prev_cons) syl.back().coda++; continue; }   // k: = a long consonant
                syl.back().coda++; prev_cons = true;
            }
            if ((int)syl.size() != nv) t = 0;
            else t = hi::primary_stress(syl);
        }
        else if (opt.stress_rule == 1) {
            // first syllable; after a marked (secondary) syllable, skip one more: g,e:o:f'y:zIk
            bool marked = false; int skip = 0;
            for (int k = 0; k < nv; k++) {
                if (st(k) >= 2) { marked = true; skip = 1; continue; }
                if (st(k) >= 0 || unst(k)) continue;
                if (marked && skip > 0) { skip--; continue; }
                return k;
            }
            if (marked && opt.german) return -1;   // German: the only free syllable was skipped after the marked one (Photonen): no primary stress
            for (int k = 0; k < nv; k++) if (st(k) < 0 && !unst(k)) return k;
            t = 0;
        }
        else if (opt.stress_rule == 3 && opt.stress_before_unstressed) {
            // skip leading '%' vowels, then the last vowel before the next '%' one; all '%': the first ("drama")
            auto pct = [&](int k) { return st(k) >= 0 && st(k) < 2; };
            int k = 0;
            while (k < nv && pct(k)) k++;
            if (k == nv) { force_target_ = true; return 0; }
            while (k + 1 < nv && !pct(k + 1)) k++;
            return k;
        }
        else if (opt.stress_rule == 3) t = nv - 1;
        else {
            t = nv >= 2 ? nv - 2 : 0;
            if (opt.final_spanish && !wd.ph.empty()) {
                std::vector<int> segs;
                for (int p : wd.ph) if (p >= 0 && ps.defs[p].type != T_STRESS && ps.defs[p].type != T_PAUSE) segs.push_back(p);
                if (!segs.empty() && !is_vowel_ph(segs.back())) {
                    const std::string& nm = ps.defs[segs.back()].name;
                    bool ns = (nm == "n" || nm == "s" || nm == "m" || nm == "N") && segs.size() >= 2 && is_vowel_ph(segs[segs.size() - 2]);
                    if (nm == "s" && segs.size() >= 3 && !ns) {
                        const std::string& p2 = ps.defs[segs[segs.size() - 2]].name;
                        if ((p2 == "n" || p2 == "m" || p2 == "N") && is_vowel_ph(segs[segs.size() - 3])) ns = true;
                    }
                    if (!ns) t = nv - 1;
                }
            }
        }
        // skip vowels that are always unstressed
        for (int k = t; k >= 0; k--) if (!unst(k)) return k;
        for (int k = t + 1; k < nv; k++) if (!unst(k)) return k;
        return t;
    }

    // ---------- phoneme programs (phase 1: phoneme changes)
    const PhItem* rel(const std::vector<PhItem>& L, size_t i, int d, bool same_word) const {
        if (opt.length_mark_transparent) {   // step over ':' items
            long j = (long)i;
            for (int n = 0; n < std::abs(d); n++) {
                j += d > 0 ? 1 : -1;
                while (j >= 0 && j < (long)L.size() && L[j].foreign < 0 && ps.defs[L[j].ph].name == ":") j += d > 0 ? 1 : -1;
            }
            if (j < 0 || j >= (long)L.size()) return nullptr;
            if (same_word && L[j].word != L[i].word) return nullptr;
            return &L[j];
        }
        if (opt.indic && d != 0) {   // Hindi: a length mark (k:) is part of its consonant, not a phoneme of its own
            long j = (long)i;
            for (int left = std::abs(d); left > 0;) {
                j += d > 0 ? 1 : -1;
                if (j < 0 || j >= (long)L.size()) return nullptr;
                if (L[j].foreign < 0 && ps.defs[L[j].ph].type == T_VIRTUAL && ps.defs[L[j].ph].name == ":") continue;
                left--;
            }
            if (same_word && L[j].word != L[i].word) return nullptr;
            return &L[j];
        }
        long j = (long)i + d;
        if (j < 0 || j >= (long)L.size()) return nullptr;
        if (same_word && wid(L[j]) != wid(L[i])) return nullptr;
        return &L[j];
    }
    int wid(const PhItem& x) const { return (opt.hyphen_word && x.hword >= 0) ? x.hword : x.word; }
    bool attr_true(const std::string& a, const std::vector<PhItem>& L, const PhItem* it, bool is_prev, size_t self) const {
        if (!it) return false;
        const PhDef& d = ps.defs[it->ph];
        if (a == "isPause") return d.type == T_PAUSE;
        if (a == "isPause2") return d.type != T_VOWEL && d.type != T_LIQUID;   // approx (not a vowel / liquid)
        if (a == "isVowel") return d.type == T_VOWEL;
        if (a == "isNotVowel") return d.type != T_VOWEL;
        if (a == "isLiquid") return d.type == T_LIQUID;
        if (a == "isNasal") return d.type == T_NASAL;
        if (a == "isVFricative") return d.type == T_FRIC && d.voiced;
        if (a == "isUStop") return d.type == T_STOP && !d.voiced;
        if (a == "isVStop") return d.type == T_STOP && d.voiced;
        if (a == "isPalatal") return d.palatal;
        if (a == "isVelar") return under_korean_ && d.velar;
        if (a == "isRhotic") return d.rhotic;
        if (a == "isFlag1") return d.flag1 && !(is_prev && it->word != L[self].word);
        if (a == "isFlag2") return d.flag2;
        if (a == "isVoiced") return d.type == T_VOWEL || d.voiced || (opt.german && (d.type == T_LIQUID || d.type == T_NASAL));
        if (a == "isWordStart") return it->word_start;
        if (a == "notWordStart") return !it->word_start;
        if (a == "isWordEnd") return it->word_end;
        if (a == "isStressed") return it->stress >= 4;
        if (a == "isNotStressed") return it->stress < 4;
        if (a == "isUnstressed") return it->stress <= 1;
        if (a == "isDiminished") return it->stress == 0;
        if (a == "isMaxStress") return it->stress >= 4;
        if (a == "isTranslationGiven") return it->given;
        if (a == "isFirstVowel" || a == "isSecondVowel" || a == "isFinalVowel" || a == "isAfterStress") {
            size_t idx = (size_t)(it - L.data());
            int w = L[idx].word, n = 0, cnt = 0; long k;
            for (k = (long)idx; k >= 0 && L[k].word == w; k--) {}
            bool after = false; int total = 0;
            for (size_t m = (size_t)(k + 1); m < L.size() && L[m].word == w; m++) {
                if (ps.defs[L[m].ph].type == T_VOWEL) { total++; if (m <= idx) cnt = total; }
                if (m < idx && L[m].stress >= 4) after = true;
            }
            (void)n;
            if (a == "isFirstVowel") return cnt == 1;
            if (a == "isSecondVowel") return cnt == 2;
            if (a == "isFinalVowel") return cnt == total;
            return after;
        }
        if (!a.empty() && a[0] == '#') return (is_prev ? d.endtype : d.starttype) == a;
        (void)self;
        int p = ps.find(a);
        return p >= 0 && ps.defs[p].name == d.name;
    }
    bool eval(const Stmt& s, const std::vector<PhItem>& L, size_t i) const {
        bool acc = !s.is_or;
        for (auto& c : s.conds) {
            const PhItem* it = nullptr; bool prev = false;
            switch (c.which) {
                case C_THIS: it = &L[i]; break;
                case C_PREV: it = rel(L, i, -1, false); prev = true; break;
                case C_PREVW: it = rel(L, i, -1, true); prev = true; break;
                case C_PREV2W: it = rel(L, i, -2, true); prev = true; break;
                case C_NEXT: it = rel(L, i, 1, false); break;
                case C_NEXT2: it = rel(L, i, 2, false); break;
                case C_NEXTW: it = rel(L, i, 1, true); break;
                case C_NEXT2W: it = rel(L, i, 2, true); break;
                case C_NEXT3W: it = rel(L, i, 3, true); break;
                case C_NEXTVOWEL:
                    for (size_t j = i + 1; j < L.size() && wid(L[j]) == wid(L[i]); j++) if (ps.defs[L[j].ph].type == T_VOWEL) { it = &L[j]; break; }
                    break;
                case C_PREVVOWEL:
                    for (long j = (long)i - 1; j >= 0 && wid(L[j]) == wid(L[i]); j--) if (ps.defs[L[j].ph].type == T_VOWEL) { it = &L[j]; prev = true; break; }
                    break;
                default: break;
            }
            bool v = c.which == C_FALSE ? false : attr_true(c.attr, L, it, prev, i);
            if (c.neg) v = !v;
            if (opt.seq_conds) { acc = (&c == &s.conds[0]) ? v : c.or_prev ? (acc || v) : (acc && v); continue; }   // Danish: left to right
            if (s.is_or) acc = acc || v; else acc = acc && v;
        }
        return acc;
    }
    // returns: -2 none, else new phoneme index; append set when IfNextVowelAppend fires; ipa override
    int run(const std::vector<Stmt>& prog, const std::vector<PhItem>& L, size_t i, int& append, std::string* ipa, int depth, bool& ret) const {
        for (auto& s : prog) {
            if (ret) return -2;
            switch (s.kind) {
                case S_IF: {
                    int r = run(eval(s, L, i) ? s.body : s.els, L, i, append, ipa, depth, ret);
                    if (r != -2) return r;
                    break;
                }
                case S_CHANGE: { if (ipa && opt.indic) break; if (s.arg == "NULL") return -3; int p = ps.find(s.arg); if (p >= 0) return p; break; }
                case S_CH_NOTSTRESSED: if (!(ipa && opt.indic) && L[i].stress < 4 && !(L[i].number && opt.final_spanish)) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_UNSTRESSED: if (!(ipa && opt.indic) && L[i].stress <= 1 && !L[i].soft && !(L[i].number && opt.finnish)) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_DIMINISHED: if (!(ipa && opt.indic) && L[i].stress == 0) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_STRESSED: if (!(ipa && opt.indic) && L[i].stress >= 4) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_APPEND: {
                    const PhItem* n = rel(L, i, 1, false);
                    if (n && ps.defs[n->ph].type == T_VOWEL) append = ps.find(s.arg);
                    break;
                }
                case S_CALL: {
                    if (depth > 4) break;
                    size_t sl = s.arg.find('/');
                    if (sl == std::string::npos && !under_korean_) break;
                    int p;
                    if (sl == std::string::npos) p = ps.find(s.arg);   // "CALL @": a phoneme of this language
                    else {
                        int tb = ps.find_table(s.arg.substr(0, sl));
                        p = tb >= 0 ? ps.find(s.arg.substr(sl + 1), tb) : -1;
                    }
                    if (p == L[i].ph) break;
                    if (p >= 0) {
                        if (ipa && ps.defs[p].has_ipa) *ipa = ps.defs[p].ipa;
                        bool same_table = sl == std::string::npos;   // its sound does not end this phoneme's own program
                        bool sub_ret = false;
                        int r = run(ps.defs[p].prog, L, i, append, ipa, depth + 1, (opt.portuguese || opt.indic || ((opt.korean || under_korean_) && same_table)) ? sub_ret : ret);
                        if (r != -2) return r;
                    }
                    break;
                }
                case S_IPA: if (ipa) *ipa = s.arg; break;
                case S_RETURN: ret = true; return -2;
                case S_SOUND: if (ipa) { ret = true; return -2; } break;   // phase 2 ends at the first sound
            }
        }
        return -2;
    }

    // ---------- Korean tokens (opt.korean): returns true when t[i] was handled here.
    template <class Flush>
    bool korean_token(const std::u32string& t, size_t& i, std::u32string& cur, bool& hyph, Flush& flush) const {
        char32_t c = t[i];
        bool cur_digit = !cur.empty() && is_digit(cur.back());
        if (c == ',' && cur_digit && i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) && is_digit(t[i + 3]) &&
            (i + 4 >= t.size() || !is_digit(t[i + 4])))
            return true;   // thousands separator: "1,000"
        if (c == '.' && cur_digit && i + 1 < t.size() && is_digit(t[i + 1])) {   // decimals: one word per digit
            flush();
            size_t j = i + 1;
            for (; j < t.size() && is_digit(t[j]); j++) { cur = std::u32string{t[j]}; flush(); }
            i = j - 1;
            return true;
        }
        if (c < 0x80 && std::strchr("%~+=@#/*", (int)c)) { flush(); cur = std::u32string{c}; flush(); return true; }   // English words
        if (ko::is_ideograph(c)) {   // a Chinese character is named, not read: English "Chinese" + ko::kIdeographTail
            flush();
            cur = U"Chinese"; flush();
            cur = U"\x06"; hyph = true; flush();
            return true;
        }
        if (is_alpha(c) && !cur.empty() && is_alpha(cur.back()) && ko::is_jamo(c) != ko::is_jamo(cur.back())) {
            flush(); hyph = true; ko_glue_ = true;   // Latin <-> Hangul inside one token: two words, no space between
        }
        if (c == '-' && !cur.empty() && i + 1 < t.size() && is_alpha(t[i + 1]) && (is_alpha(cur.back()) || is_digit(cur.back()))) {
            flush(); hyph = true;   // hyphenated words are joined
            return true;
        }
        return false;
    }
    // A symbol by its list name (_/ -> slaS), as a clause of its own; "" if the list has none.
    std::string symbol_ipa(char32_t c) {
        std::u32string k = U"_"; k += c;
        const ListEntry* le = lookup_entry(k, false, false);
        if (!le || le->phonemes.empty()) le = lookup_entry(std::u32string{c}, false, false);   // en_list: "/ slaS"
        if (!le || le->phonemes.empty()) return "";
        sym_override_ = le->phonemes;
        std::string r = phonemize_clause("x");
        sym_override_.clear();
        return strip_ws(r);
    }
    std::string sym_override_;   // symbol_ipa: the one word's phonemes
    bool force_spell_ = false;   // set by a Hindi reader: spell this capitals word

    mutable std::vector<std::vector<int>> scale_ph_;   // Hindi: phonemes of _0M1 .. _0M5 (hazaar, lakh, crore, arab, kharab)
    bool starts_stressed(const std::vector<int>& ph) const {   // a primary stress mark before the first vowel
        for (int p : ph) {
            if (p < 0) continue;
            if (stress_of(p) >= 4) return true;
            if (is_vowel_ph(p)) return false;
        }
        return false;
    }
    bool is_scale_word(const std::vector<int>& ph) const {
        if (scale_ph_.empty())
            for (const char* k : {"_0M1", "_0M2", "_0M3", "_0M4", "_0M5"}) { scale_ph_.emplace_back(); encode_ph(ph_of(k), scale_ph_.back()); }
        for (auto& s : scale_ph_) if (!s.empty() && s == ph) return true;
        return false;
    }

    // ---------- clause
    std::string phonemize_clause(const std::string& text_utf8) override {
        if (opt.french_numbers) {
            // French: an en dash between spaces ends a clause (no liaison over it; the word before it is clause-final)
            const std::string dash = " \xE2\x80\x93 ";
            if (text_utf8.find(dash) != std::string::npos) {
                // a loop, not recursion: a text of thousands of dashes must not overflow a small thread stack
                std::string out;
                size_t from = 0;
                for (;;) {
                    const size_t p = text_utf8.find(dash, from);
                    const std::string piece = text_utf8.substr(from, p == std::string::npos ? std::string::npos : p - from);
                    std::vector<Word> ws;
                    tokenize(dec(piece), ws);
                    const std::string r = render_words(ws);
                    if (!r.empty()) { if (!out.empty()) out += " "; out += r; }
                    if (p == std::string::npos) break;
                    from = p + dash.size();
                }
                return out;
            }
        }
        std::vector<Word> words;
        tokenize(dec(text_utf8), words);
        return render_words(words);
    }

    // text of one clause -> its words
    void tokenize(std::u32string t, std::vector<Word>& words) {
        if (opt.dotless_i) {   // Turkish text: acute accent used as apostrophe, letters from neighbouring alphabets
            for (auto& c : t) {
                if (c == 0xB4) c = '\'';
                else if (c == 0xDF || c == 0x219) c = 's'; else if (c == 0x218) c = 'S';
                else if (c == 0x21B) c = 't'; else if (c == 0x21A) c = 'T'; else if (c == 0x18F) c = 0x259;
            }
        }
        if (opt.caps_spell_by_clause) {
            size_t up = 0, low = 0;
            for (char32_t c : t) if (is_alpha(c)) { if (is_upper(c)) up++; else low++; }
            clause_caps_spell_ = low >= 4 && low >= up;
        }
        if (opt.da_numbers) {   // Danish: a short all-capital word is spelled only in a clause with more than 3 small letters, at least as many small as capital
            size_t lo = 0, up = 0;
            for (char32_t c : t) if (is_alpha(c)) { if (is_upper(c)) up++; else lo++; }
            caps_spell_ok_ = lo > 3 && lo >= up;
        }
        if (opt.caps_read) {
            clause_lower_ = clause_upper_ = 0;
            for (char32_t c : t) if (is_alpha(c) && c < 0x530) (is_upper(c) ? clause_upper_ : clause_lower_)++;   // cased scripts only
        }
        if (opt.french_numbers) {
            // French: a short word in capitals (FBI, PIB) is spelled only when the clause has fewer capital letters than
            // small ones ("le FBI dit" is spelled, "du FBI" is read as a word)
            size_t up = 0, lo = 0;
            for (char32_t c : t) if (is_alpha(c)) { if (lower(c) != c) up++; else lo++; }
            clause_spells_caps_ = up < lo;
        }
        // words: letters / digits runs; other characters break words (apostrophes kept inside words)
        std::u32string cur; bool cap = false, allc = true, hyph = false, frac = false, each_next = false, rules_only = false, by_digit = false, superscript = false;
        bool decimal_seen = false;   // Turkish: a decimal comma was read in this number
        bool da_decimal = false;     // Danish: inside the digits after a decimal comma
        bool pend_pause = false, grouped = false; char32_t ordc = 0;   // Portuguese; Swedish: digits joined across "40 000"
        bool in_quote = false; int pend_frac = 0;                      // Portuguese
        bool after_comma = false, spell_head = false;                  // French
        bool glued = false, decimals = false;   // German: letters right after digits ("20er"); the digits after a decimal comma
        bool singly = false;                    // Swedish: decimals read one by one
        bool year_group = false;                // ... and the first group had four digits ("1971 621")
        size_t sym_end = (size_t)-1;            // index after the last symbol read as a word
        if (opt.portuguese) {
            // a clause written mostly in capitals: its short all-capital words are words, not abbreviations
            int lo = 0, up = 0;
            for (char32_t c : t) if (is_alpha(c)) { if (is_upper(c)) up++; else lo++; }
            caps_clause_ = lo <= 3 || up > lo;
        }
        ko_glue_ = false;
        size_t clause_words = 0;    // Hindi: a danda ends a clause; one with no word before it in its clause is read out
        bool attach_next = false;   // the next word follows the current one without a separator (letters <-> digits)
        size_t num_join = 0;   // Finnish "10 000": digits of cur before the last joined group
        auto flush = [&]() {
            num_join = 0;
            if (cur.empty()) return;
            bool grouped_word = grouped;
            grouped = false;
            if (singly && is_digit(cur[0])) {
                singly = false;
                for (char32_t d : cur) { Word w; w.text = std::u32string{d}; w.orig = w.text; words.push_back(w); }
                cur.clear(); cap = false; allc = true; hyph = false;
                return;
            }
            singly = false;
            std::u32string orig_cur = cur;
            Word w; w.capital = cap; w.allcaps = allc && cur.size() > 1;
            w.pause_before = pend_pause; pend_pause = false;
            w.year_group = year_group && is_digit(cur[0]); year_group = false;
            w.num_grouped = grouped_word;
            w.grouped = grouped_word; grouped_word = false;
            w.after_comma = after_comma; after_comma = false;
            w.spell_caps = spell_head; spell_head = false;
            w.ordinal = ordc; ordc = 0;
            w.fraction = pend_frac; pend_frac = 0;
            w.join_prev = hyph; hyph = false;
            w.rules_only = rules_only; rules_only = false;
            w.digit_by_digit = by_digit; by_digit = false;
            w.superscript = superscript; superscript = false;
            auto plain = [&](char32_t c, bool& mapped) {
                c = (opt.dotless_i && c == 'I') ? (char32_t)0x131 : lower(c);
                // Swedish: letters read as their nearest plain letter ("suður", "Bălți", "Groß", "Lǐ")
                if (opt.caps_read && c >= 0x80 && !known_letter(c)) c = c == 0x163 ? U't' : sv_plain_letter(c);
                if (c >= 0x80 && !known_letter(c) && !opt.letter_by_rules) { char32_t b = base_letter(c); if (b != c) { c = b; mapped = true; } }
                return c;
            };
            if (opt.caps_read && std::any_of(cur.begin(), cur.end(), sv_unnamed_letter)) {
                bool m = false, vowel = false;
                for (char32_t c : cur) if (!sv_unnamed_letter(c) && is_vowel_letter(plain(c, m))) vowel = true;
                if (vowel) {   // "Kuṯay": every letter is a word of its own, the unnamed one "bokstaven" + its code
                    bool first = true;
                    for (char32_t c : cur) {
                        Word lw; lw.join_prev = first && w.join_prev; first = false;
                        if (sv_unnamed_letter(c)) {
                            std::string mn;
                            for (auto& u : unnamed_units(c)) { if (!mn.empty()) mn += "\x02"; mn += u; }
                            lw.text = U"\x02"; encode_ph(mn, lw.ph);
                        } else {
                            lw.orig = std::u32string{c}; lw.text = std::u32string{plain(c, lw.mapped)};
                        }
                        words.push_back(lw);
                    }
                    cur.clear(); cap = false; allc = true;
                    return;
                }
                w.spell_all = true;   // "Kṯ": no vowel, spelled as one word
            }
            for (auto& c : cur) c = plain(c, w.mapped);
            if (opt.finnish) for (char32_t c0 : orig_cur) { char32_t lc = lower(c0); if (lc >= 0x80 && !known_letter(lc)) w.no_next = true; }
            w.glued_prev = ko_glue_; ko_glue_ = false;
            w.glued_num = glued; glued = false;
            for (auto& r : replace) {
                size_t p = 0;
                while ((p = cur.find(r.first, p)) != std::u32string::npos) { cur.replace(p, r.first.size(), r.second); p += r.second.size(); }
            }
            w.text = cur;
            w.orig = orig_cur;
            w.digits_each = each_next && is_digit(cur[0]); each_next = false;
            if (frac && std::all_of(cur.begin(), cur.end(), is_digit)) w.digits_single = true;
            w.attached = attach_next; attach_next = false;
            frac = false;
            words.push_back(w);
            cur.clear(); cap = false; allc = true; grouped = false;
        };
        int other_letters = 0;
        if (opt.korean) {
            for (char32_t c : t) if (is_alpha(c) && !is_upper(c)) other_letters++;   // before the syllables become jamo
            t = ko::decompose(t);
        }
        for (size_t i = 0; i < t.size(); i++) {
            char32_t c = t[i];
            if (!is_digit(c) && c != ',') decimal_seen = false;
            if (decimals && !is_digit(c)) decimals = false;
            if (opt.korean && korean_token(t, i, cur, hyph, flush)) continue;
            bool letter = is_alpha(c) || (c >= 0x300 && c <= 0x36F) || (c >= 0x900 && c <= 0x97F && c != 0x964 && c != 0x965) ||
                          (opt.dotless_i && c == 0x259) ||   // Turkish rules read the schwa letter (Azerbaijani)
                          (opt.caps_read && (c == 0x2BB || c == 0x2BE || c == 0x2BF));
            const int sup = !opt.dotless_i ? 0 : c == 0xB9 ? 1 : c == 0xB2 ? 2 : c == 0xB3 ? 3 : 0;
            if (letter) {
                if (!cur.empty() && is_digit(cur.back())) { flush(); attach_next = true; glued = opt.german; }
                if (!cur.empty() && is_upper(c) && !is_upper(cur.back()) && is_alpha(cur.back())) flush();
                if ((opt.swahili && is_alpha(c) && !is_upper(c) && cur.size() >= 2 && is_upper(cur.back()) && is_upper(cur[cur.size() - 2])) ||
                    (opt.portuguese && !is_upper(c) && is_alpha(c) && cur.size() >= 2 && allc &&
                     std::all_of(cur.begin(), cur.end(), [](char32_t x) { return is_alpha(x) && is_upper(x); }) &&
                     !(c == 's' && (i + 1 >= t.size() || !is_alpha(t[i + 1])))) ||
                    (opt.korean && c < 0x80 && !is_upper(c) && cur.size() >= 2 &&
                     std::all_of(cur.begin(), cur.end(), [](char32_t x) { return x < 0x80 && is_alpha(x) && is_upper(x); }))) {
                    // "SANParks", "ZMapp": the capitals before a capitalised word are a word of their own ("TVs" is not)
                    char32_t last = cur.back();
                    cur.pop_back();
                    flush();
                    cur = std::u32string{last}; cap = true;
                }
                else if (opt.caps_read && cur.size() >= 2 && !is_upper(c) && is_alpha(c) && is_upper(cur.back()) &&
                         is_upper(cur[cur.size() - 2]) && i + 1 < t.size() && is_alpha(t[i + 1]) && !is_upper(t[i + 1])) {   // "SANParks" -> "SAN" "Parks", "ZMapp" -> "Z" "Mapp"
                    char32_t last = cur.back(); cur.pop_back(); flush();
                    cur = std::u32string{last}; cap = true;
                }
                else if (opt.french_numbers && cur.size() >= 2 && !is_upper(c) && i + 1 < t.size() && is_alpha(t[i + 1]) && !is_upper(t[i + 1]) &&
                    std::all_of(cur.begin(), cur.end(), [](char32_t x) { return is_alpha(x) && is_upper(x); })) {
                    // French: "SANParks", "ZMapp": the capitals before the last one are a word of their own, spelled
                    const char32_t last = cur.back();
                    cur.pop_back();
                    spell_head = true;
                    flush();
                    cur = std::u32string{last}; cap = true; allc = true;
                }
                else if (opt.finnish && !is_upper(c) && cur.size() >= 3 && is_upper(cur.back()) && is_upper(cur[cur.size() - 2])) {
                    char32_t last = cur.back();   // "SANParks" -> "SAN" + "Parks"
                    cur.pop_back(); flush();
                    cur = last; cap = true;
                }
                else if (opt.caps_word_split && is_upper(c) && i + 2 < t.size() && is_alpha(t[i + 1]) &&
                         !is_upper(t[i + 1]) && is_alpha(t[i + 2]) && !is_upper(t[i + 2]) && std::all_of(cur.begin(), cur.end(), [](char32_t x) { return is_upper(x); })) flush();
                if (opt.da_tokens && !is_upper(c) && cur.size() >= 2 && allc && is_alpha(cur.back()) && i + 1 < t.size() && is_alpha(t[i + 1]) && !is_upper(t[i + 1])) {
                    // Danish "SANParks", "ZMapp": the capitals before the last one are their own word (not "USAs", "FNs")
                    char32_t last = cur.back(); cur.pop_back();
                    flush(); cur = std::u32string{last}; cap = true; allc = true;
                }
                da_decimal = false;
                if (cur.empty()) cap = is_upper(c);
                if (!is_upper(c)) allc = false;
                cur += c;
            } else if (is_digit(c)) {
                if (!cur.empty() && !is_digit(cur.back())) { flush(); attach_next = true; }
                if (da_decimal && !cur.empty()) flush();   // Danish: the digits after a decimal comma are read one by one
                cur += c;
                if (decimals) flush();   // German: the digits after a decimal comma are read one by one
            } else if (opt.da_tokens && (c == '\'' || c == 0x2019) && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_alpha(t[i + 1])) {
                flush(); da_decimal = false; cur = U"'";   // Danish "1960'erne": the number, then the ending as its own word
            } else if (sup && i + 1 < t.size() && (t[i + 1] == '\'' || t[i + 1] == 0x2019)) {
                flush(); hyph = false;   // "km²'lik": the superscript is not read before a suffix
            } else if (sup) {   // "km²": the digit as a word of its own (not a list translation)
                flush(); hyph = false;
                cur = std::u32string{(char32_t)('0' + sup)};
                superscript = true;
                flush();
            } else if ((c == '\'' || c == 0x2019) && opt.number_suffix_word && !cur.empty() && i + 1 < t.size() &&
                       is_alpha(t[i + 1]) && std::all_of(cur.begin(), cur.end(), is_digit)) {
                flush(); rules_only = true;   // "1537'de": the number, then "de" as a word of its own
            } else if ((c == '\'' || c == 0x2019 || (c == 0x2018 && opt.caps_read)) && !cur.empty() && i + 1 < t.size() && is_alpha(t[i + 1])) {
                if (opt.swahili) {
                    // "ng'ombe", "mng'ao": letters without a vowel before the apostrophe are spelled, the rest is a word
                    // of its own written on (the apostrophe stays on it, so it is not looked up in the word list)
                    bool vowel = std::any_of(cur.begin(), cur.end(), [&](char32_t ch) { return is_vowel_letter(ch); });
                    if (!vowel) { flush(); hyph = true; }
                    cur += '\'';
                    continue;
                }
                cur += '\'';
                if (opt.apostrophe_split) {
                    std::u32string whole = cur;   // keep list entries such as "l'ancora" whole
                    size_t j = i + 1;
                    while (j < t.size() && is_alpha(t[j])) whole += t[j++];
                    for (auto& ch : whole) ch = lower(ch);
                    if (!list.count(whole)) { flush(); hyph = true; }
                }
            } else if (opt.german && c == 0x2018 && !cur.empty() && is_alpha(cur.back()) && i + 1 < t.size() && is_alpha(t[i + 1])) {
                // German: a left quote inside a word is skipped ("Brink‘s" = "Brinks")
            } else if (opt.german && c == 0xBA) {   // "27 ºC": the ordinal sign is read as the letter o
                flush(); hyph = false;
                cur = U"o"; cap = false; allc = false;
                flush();
            } else if (c == '-' && opt.german && i + 1 < t.size() && is_digit(t[i + 1]) &&
                       ((!cur.empty() && is_digit(cur.back()) && !dash_digits_.empty()) ||
                        (cur.empty() && (i == 0 || is_space(t[i - 1])) && !minus_.empty()))) {
                // German: "3-4" -> drei Strich vier; " -5" -> minus fuenf (the rules' "-" between / before digits)
                bool between = !cur.empty();
                flush(); hyph = false;
                Word dw; dw.text = U"\x01"; encode_ph(between ? dash_digits_ : minus_, dw.ph);
                words.push_back(dw);
            } else if (c == '-' && !cur.empty() && i + 1 < t.size() && is_alpha(t[i + 1])) {
                bool one = cur.size() == 1 || opt.hyphen_join || opt.indic;
                flush(); hyph = one;
            } else if (opt.sv_numbers && (c == '.' || is_space(c)) && !cur.empty() && is_digit(cur.back()) &&
                       (cur.size() <= 3 || grouped || (cur.size() == 4 && is_space(c) && std::all_of(cur.begin(), cur.end(), is_digit))) &&
                       i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) &&
                       is_digit(t[i + 3]) && (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                if (!grouped && cur.size() == 4) year_group = true;
                grouped = true;   // "40 000", "1.000": one number
            } else if (opt.sv_numbers && (c == '.' || c == ',') && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() &&
                       is_digit(t[i + 1])) {
                flush();   // decimals: "punkt", then a number after '.', digit by digit after ','
                Word pw; pw.text = U"\x02"; pw.listed = true; encode_ph(ph_of("_dpt"), pw.ph); words.push_back(pw);
                singly = c == ',';
            } else if (opt.sv_numbers && c == '-' && cur.empty() && i + 1 < t.size() && is_digit(t[i + 1]) &&
                       (i == 0 || is_space(t[i - 1]) || t[i - 1] == '(')) {   // "-56": the rules' word for the sign
                Word pw; pw.text = U"\x02"; encode_ph(apply_rules(U"  - 0 ", 2, 3, MatchCtx()), pw.ph);
                if (!pw.ph.empty()) words.push_back(pw);
            } else if (opt.sv_numbers && (c == 0xB2 || c == 0xB3 || c == 0xB9)) {   // "km²": a superscript digit is a number
                flush(); hyph = false;
                cur = std::u32string{c == 0xB2 ? U'2' : c == 0xB3 ? U'3' : U'1'};
                flush();
            } else if (opt.symbol_rules && !is_space(c) && !is_alpha(c) && !is_digit(c) && c != '.' && c != ',' && c != '\'' &&
                       c != '-' && symbol_word(c)) {
                if (sym_end == i && cur.empty() && !words.empty()) {   // "//", "++": one word
                    words.back().text += c; words.back().orig += c; words.back().symbols = true;
                } else {
                    flush(); hyph = false;
                    cur = std::u32string{c};
                    flush();
                }
                sym_end = i + 1;
                if (i + 2 < t.size() && t[i + 1] == '-' && is_alpha(t[i + 2])) { hyph = true; i++; }   // "%-räntan"
            } else if (opt.symbol_rules && (c == '=' || c == '#' || c == '~')) {   // no Swedish word: English "equals", "hash", "tilde"
                flush(); hyph = false;
                Word sw; sw.text = std::u32string{c}; sw.orig = sw.text; sw.switch_lang = "EN";
                words.push_back(sw);
            } else if ((c == ' ' || c == '.') && opt.french_numbers && !cur.empty() && (cur.size() <= 3 || grouped) && is_digit(cur.back()) &&
                       cur[0] != '0' &&
                       std::all_of(cur.begin(), cur.end(), is_digit) && i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) &&
                       is_digit(t[i + 3]) && (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                // French: "30 000" is one number (a plain space between groups of three digits)
                grouped = true;
            } else if (opt.finnish && c == ' ' && !cur.empty() && std::all_of(cur.begin(), cur.end(), is_digit) &&
                       (cur.size() <= 3 || cur.size() == num_join + 3) && i + 3 < t.size() && is_digit(t[i + 1]) &&
                       is_digit(t[i + 2]) && is_digit(t[i + 3]) && (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                num_join = cur.size();   // a space between groups of three digits: one number ("10 000")
            } else if (c == '.' && (opt.num_swap || opt.italian_alt || opt.dot_thousands || opt.portuguese || opt.da_numbers) && !cur.empty() && is_digit(cur.back()) && i + 3 < t.size() + 0 &&
                       i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) && is_digit(t[i + 3]) &&
                       (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                // German thousands separator: "1.000"
                if (opt.portuguese) grouped = true;
            } else if (opt.german && c == ' ' && !cur.empty() && is_digit(cur.back()) && (cur.size() <= 3 || grouped) &&
                       i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) && is_digit(t[i + 3]) &&
                       (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                grouped = true;   // German: "5 000 000" is one number (groups of three digits)
            } else if (opt.portuguese && (c == 0x2018 || (c == '\'' && (cur.empty() || in_quote)))) {
                // a single quotation mark is a break: the words on its two sides do not see each other. An ASCII
                // apostrophe after a word closes a quotation only if one was opened ("as casas' azuis" has none).
                in_quote = c == '\'' && cur.empty();
                flush(); hyph = false;
                pend_pause = true;
            } else if (c == ',' && opt.swahili && !cur.empty() && is_digit(cur.back()) && i + 3 < t.size() &&
                       is_digit(t[i + 1]) && is_digit(t[i + 2]) && is_digit(t[i + 3]) && (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                // Swahili thousands separator: "30,000"
            } else if (opt.indic && c == ',' && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1])) {
                // Hindi: digit group separator (8,00,000 and 800,000)
            } else if (opt.indic && c == '.' && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1])) {
                flush();
                Word pw; pw.text = U"\x02"; pw.marks_only = true; encode_ph(ph_of("_dpt"), pw.ph); words.push_back(pw);
                frac = true;
            } else if ((c == '.' || c == ',') && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1]) &&
                       opt.stressed_decimal && c == ',' && decimal_seen) {
                flush(); decimal_seen = false;   // Turkish "1,351,254": no second decimal word
            } else if (opt.portuguese && c == ',' && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1])) {
                // decimal comma: "vírgula"; zeros right after it are joined to it; then one or two digits are a
                // number (in the stress pattern of "vírgula": its last syllable is stressed lightly before an
                // unstressed start), three or more are read one by one
                flush();
                size_t j = i + 1, z = 0;
                while (j < t.size() && is_digit(t[j])) j++;
                std::u32string frac = t.substr(i + 1, j - i - 1);
                while (z < frac.size() && frac[z] == '0') z++;
                std::u32string rest = frac.substr(z);
                std::string vp = ph_of("_dpt");
                if (!z && rest.size() <= 2 && !stressed_start(pt_number(rest))) {
                    size_t lv = vp.size();
                    while (lv > 0) { std::vector<int> one; encode_ph(vp.substr(lv - 1, 1), one); if (one.size() == 1 && is_vowel_ph(one[0])) break; lv--; }
                    if (lv > 0) vp.insert(lv - 1, ",");
                }
                for (size_t k = 0; k < z; k++) vp += "\x02" + ph_of("_0");
                Word pw; pw.text = U"\x02"; encode_ph(vp, pw.ph); words.push_back(pw);
                if (!rest.empty()) { cur = rest; pend_frac = rest.size() <= 2 ? 1 : 2; flush(); }
                i = j - 1;
            } else if ((c == '.' || c == ',') && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1])) {
                flush();
                if (opt.german && c == ',') decimals = true;
                std::u32string k = U"_"; k += c;
                if (opt.da_numbers && c == ',') { k = U"_dpt"; da_decimal = opt.da_tokens; }   // Danish decimal comma: "komma"
                if (opt.finnish && c == ',') k = U"_dpt";   // decimal comma
                if (opt.swahili && c == '.') { k = U"_dpt"; each_next = true; }   // "6.5": "sita nukta tano"
                if (opt.stressed_decimal && c == ',' && list.count(U"_dpt")) k = U"_dpt";   // the decimal comma's own word
                if (opt.portuguese) k = c == ',' ? U"_dpt" : U"";   // decimal comma: "vírgula"; a dot: nothing
                if (opt.french_numbers && !words.empty()) words.back().num_unstressed = true;
                if (opt.finnish && list.count(k) && !words.empty() && !words.back().text.empty() &&
                    std::all_of(words.back().text.begin(), words.back().text.end(), is_digit)) {
                    words.back().decimal_ph = ph_of(enc(k));   // "kˈymmenˌen piste": one stress domain
                } else if (list.count(k)) {
                    Word pw; pw.text = U"\x02"; encode_ph(ph_of(enc(k)), pw.ph);
                    if (opt.swahili) pw.unstressed = true; else if (opt.finnish) { pw.unstressed = true; pw.number = true; } else pw.no_stress = !(opt.stressed_decimal || opt.portuguese);
                    if (opt.stressed_decimal) pw.given = pw.listed_marks = true;
                    if (opt.da_numbers) {   // Danish "punktum": unstressed, secondary stress on its last syllable
                        pw.no_stress = false; pw.unstressed = true; pw.u_n = 9;
                        pw.listed_marks = ph_of(enc(k)).find_first_of("',") != std::string::npos;
                    }
                    words.push_back(pw);
                    by_digit = opt.stressed_decimal && c == ',';   // Turkish: 3+ decimals after a comma, one by one
                    if (by_digit) {   // leading zeros of the decimals: "sıfır" joined to the comma's word
                        decimal_seen = true;
                        while (i + 1 < t.size() && t[i + 1] == '0') {
                            words.back().ph.push_back(-10);
                            encode_ph(ph_of("_0"), words.back().ph);
                            i++;
                        }
                    }
                }
                if (opt.french_numbers && c == ',' && t[i + 1] == '0') { hyph = true; after_comma = true; }   // "5,0" -> virgule + zero written together
            } else if (opt.da_tokens && (c == '$' || c == '~' || c == 0xB2 || c == 0xB3 || c == 0xB9)) {
                flush(); hyph = false; da_decimal = false;
                if (c == '$') { Word pw; pw.text = U"\x02"; pw.listed_marks = true; encode_ph(da::kDollar, pw.ph); words.push_back(pw); }
                else if (c == '~') { Word pw; pw.text = U"~"; pw.orig = U"~"; pw.switch_lang = "EN"; words.push_back(pw); }   // English name, as the reference
                else { cur = std::u32string{c == 0xB2 ? U'2' : c == 0xB3 ? U'3' : U'1'}; flush(); }           // superscript digit: the number
            } else if (c == 0xB0 && opt.french_numbers && (cur == U"n" || cur == U"N")) {
                // French "n°" (numéro): the sign is not read
            } else if (opt.german && (c == 0xB2 || c == 0xB3)) {   // km² -> km zwei
                flush(); hyph = false;
                cur = c == 0xB2 ? U"2" : U"3";
                flush();
            } else if (opt.portuguese && (c == 0xB2 || c == 0xB3 || c == 0xB9)) {   // "km²": the superscript is a number
                flush(); hyph = false;
                cur = std::u32string{c == 0xB2 ? U'2' : c == 0xB3 ? U'3' : U'1'};
                flush();
            } else if (opt.portuguese && (c == 0xBA || c == 0xAA) && !cur.empty() && is_digit(cur.back())) {
                ordc = c;   // ordinal number
                flush();
            } else if (opt.portuguese && (c == '-' || c == 0x2013 || c == 0x2212 || c == '/')) {
                // "55-65": a hyphen right between digits is "menos"; otherwise '/' and dashes are a pause
                flush(); hyph = false;
                if (c == '-' && i > 0 && is_digit(t[i - 1]) && i + 1 < t.size() && is_digit(t[i + 1])) {
                    Word mw; mw.text = U"\x02"; encode_ph(ph_of("_\xe2\x88\x92"), mw.ph); words.push_back(mw);
                } else pend_pause = true;
            } else if (opt.portuguese && c == 0x200B) {   // a zero-width space separates like a pause
                flush(); hyph = false;
                pend_pause = true;
            } else if (opt.german && !is_space(c) && c != '.' && c != ',' && c != '\'' && c != '-' && list.count(std::u32string{c}) &&
                       list.find(std::u32string{c})->second.back().text && list.find(std::u32string{c})->second.back().phonemes.find(' ') != std::string::npos) {
                flush(); hyph = false;
                const ListEntry& se = list.find(std::u32string{c})->second.back();
                // German: a symbol named by several words ("¾" = "drei Viertel"): each is a word of the clause
                std::u32string words32 = dec(se.phonemes);
                for (char32_t ch : words32) {
                    if (ch == ' ') { flush(); continue; }
                    if (cur.empty()) cap = is_upper(ch);
                    allc = false;
                    cur += ch;
                }
                flush();
            } else if (!is_space(c) && c != '.' && c != ',' && c != '\'' && c != '-' &&
                       !(opt.indic && (c == 0x964 || c == 0x965) && (words.size() > clause_words || !cur.empty())) &&
                       (list.count(std::u32string{c}) || (opt.swahili && (symbol_rule(c) || en_symbol(c))) || (opt.letter_by_rules && known_letter(c)))) {
                // a symbol the word list names ("%", "$", "|"): spoken as a word
                flush(); hyph = false;
                bool after_letter = !words.empty() && words.back().text.size() == 1 && i > 0 && is_alpha(t[i - 1]);
                cur = std::u32string{c};
                flush();
                if (opt.portuguese && i + 1 < t.size() && t[i + 1] == 0xA0 && has("_" + enc(std::u32string{c}))) {
                    // followed by a no-break space the sign is not a word: its character name is read ("US$" +
                    // no-break space + "37": "cifrão"), and nothing after a single letter ("R$": "real")
                    Word& sw = words.back();
                    if (after_letter) sw.silent = true;
                    else {
                        std::string nm = ph_of("_" + enc(std::u32string{c}));
                        nm.erase(std::remove(nm.begin(), nm.end(), ','), nm.end());
                        encode_ph(nm, sw.ph);
                    }
                }
                if (opt.indic && (c == 0x964 || c == 0x965)) clause_words = words.size();
            } else if (opt.indic && c > 0x20 && c < 0x80 && !is_space(c) && std::strchr("/#&@", (char)c)) {
                flush(); hyph = false;   // Hindi: / # & @ are read by their English names (as espeak-ng does)
                Word sw; sw.text = U"\x02"; sw.orig = std::u32string{c}; sw.switch_lang = "EN"; sw.symbol = true;
                words.push_back(sw);
            } else {
                flush(); hyph = false; da_decimal = false;
                // measured (Danish): a dash with a space before it ends the clause (the word before is a last word)
                if (opt.da_numbers && (c == 0x2013 || c == 0x2014) && !words.empty() && i > 0 && is_space(t[i - 1])) words.back().clause_end = true;
                if (opt.finnish && !is_space(c) && !words.empty()) words.back().no_next = true;   // "hyvin – kaikki"
                if (opt.indic && (c == 0x964 || c == 0x965)) clause_words = words.size();
            }
        }
        flush();
        if (opt.korean) {
            for (size_t k = 0; k < words.size(); k++) {
                Word& w = words[k];
                bool glued = w.glued_prev || (k + 1 < words.size() && words[k + 1].glued_prev);
                w.spell_letters = w.allcaps && !glued && ko::is_abbreviation(w.orig.size(), other_letters);
            }
        }
        clause_allcaps_ = false;
        if (opt.german) {
            bool any = false, all = true;
            for (auto& w : words) { if (w.text.empty() || !std::any_of(w.text.begin(), w.text.end(), is_alpha)) continue; any = true; if (!w.allcaps && !(w.capital && w.text.size() == 1)) all = false; }
            clause_allcaps_ = any && all;
            // a short all-capital word is spelled only in a clause with more than three small letters and no fewer small
            // than capital letters ("mit REM und" spelled; "ein SUV," read as a word)
            int lo = 0, up = 0;
            for (char32_t ch : t) { if (!is_alpha(ch)) continue; if (is_upper(ch)) up++; else lo++; }
            if (!(lo > 3 && lo >= up)) clause_allcaps_ = true;
        }
    }

    // the words of one clause -> IPA
    std::string render_words(std::vector<Word>& words) {
        // multi-word list entries (two words)
        std::vector<PhItem> L;
        int word_counter = 0;
        int hchain = -1;
        std::vector<int> mil_ph;
        if (opt.french_numbers) encode_ph(ph_of("_0M1"), mil_ph);
        std::vector<std::vector<int>> big_ph;   // French: million(s), milliard(s)
        if (opt.french_numbers)
            for (const char* k : {"_0M2", "_1M2", "_0M3", "_1M3"})
                if (has(k)) { big_ph.emplace_back(); encode_ph(ph_of(k), big_ph.back()); }
        size_t spell_start = 0;
        foreign_.clear();
        auto is_last = [&](size_t wi) { return (wi + 1 == words.size() && !not_final_ && embedded_final_) || (opt.da_numbers && words[wi].clause_end); };
        bool dbl_pending = false;   // Finnish: the previous word doubles this word's first consonant
        int pause = ps.find("_");
        L.push_back({pause < 0 ? 0 : pause, -1, true, true, -1});
        for (size_t wi = 0; wi < words.size(); wi++) {
            Word& w = words[wi];
            std::u32string next = wi + 1 < words.size() && !w.no_next ? words[wi + 1].text : U"";
            if (opt.portuguese && wi + 1 < words.size() && words[wi + 1].pause_before) next = U"";   // '/' or a dash between
            if (opt.phrase_one_domain && wi + 2 < words.size() && w.text != U"\x01" && words[wi + 1].text != U"\x01" && words[wi + 2].text != U"\x01") {
                // Danish: three-word list entries ("(salt lake city) ...") are one stress domain
                auto it3 = list.find(w.text + U" " + words[wi + 1].text + U" " + words[wi + 2].text);
                if (it3 != list.end() && !it3->second.back().phonemes.empty()) {
                    const ListEntry& e = it3->second.back();
                    w.ph.clear(); encode_ph(e.phonemes, w.ph); w.phrase = true;
                    w.unstressed = e.u; w.u_plus = e.u_plus; w.stress_pos = e.stress_pos;
                    for (int k = 1; k <= 2; k++) { words[wi + k].ph.clear(); words[wi + k].no_default = true; words[wi + k].text = U"\x01"; }
                }
            }
            if (opt.multiword_phrase) {   // French: entries of up to three words; "||" = two words of one phrase
                for (size_t n = 3; n >= 2; n--) {
                    if (wi + n > words.size()) continue;
                    std::u32string key = w.text;
                    for (size_t q = 1; q < n; q++) key += U" " + words[wi + q].text;
                    auto it = list.find(key);
                    if (it == list.end() || it->second.back().phonemes.empty()) continue;
                    const ListEntry& e = it->second.back();
                    if (e.phonemes.rfind("_^_", 0) == 0) {   // (t shirt) _^_en: the words in another language
                        std::string code;
                        for (size_t k = 3; k < e.phonemes.size() && k < 5; k++) code += (char)std::toupper((unsigned char)e.phonemes[k]);
                        w.switch_lang = code;
                        for (size_t q = 1; q < n; q++) { w.orig += U"-" + words[wi + q].orig; words[wi + q].ph.clear(); words[wi + q].text = U"\x01"; }
                        w.text = U"\x01";
                        break;
                    }
                    if (e.text) {   // (chef d'oeuvre) chédeuvre $text: read as that spelling
                        Word w2; w2.text = dec(e.phonemes);
                        for (auto& c : w2.text) c = lower(c);
                        translate_word(w2, U"", wi == 0, wi + n == words.size());
                        w.ph = w2.ph; w.stress_pos = w2.stress_pos; w.unstressed = w2.unstressed;
                        if (w.ph.empty()) continue;
                        for (size_t q = 1; q < n; q++) { words[wi + q].ph.clear(); words[wi + q].text = U"\x01"; }
                        break;
                    }
                    std::string a = e.phonemes, b;
                    size_t bar = a.find("||");
                    if (bar != std::string::npos) { b = a.substr(bar + 2); a = a.substr(0, bar); }
                    w.ph.clear(); encode_ph(a, w.ph);
                    w.unstressed = e.u; w.u_plus = e.u_plus; w.stress_pos = e.stress_pos;
                    w.verbf = e.verbf; w.nounf = e.nounf;
                    for (size_t q = 1; q < n; q++) { words[wi + q].ph.clear(); words[wi + q].text = U"\x01"; }
                    if (!b.empty()) {
                        Word& t2 = words[wi + 1];
                        encode_ph(b, t2.ph);
                        t2.join_prev = false;
                        t2.verbf = e.verbf; t2.nounf = e.nounf;
                        t2.phrase_tail = e.u ? (e.u_plus ? 2 : 1) : 0;
                        w.phrase_head = e.u;
                    }
                    if (n == 3 && !b.empty()) { words[wi + 2].verbf = e.verbf; words[wi + 2].nounf = e.nounf; }
                    break;
                }
            } else if (wi + 1 < words.size() && (!opt.phrase_one_domain || w.ph.empty())) {
                std::u32string two = w.text + U" " + next;
                auto it = list.find(two);
                if (it != list.end()) {
                    const ListEntry* ep = &it->second.back();
                    if (opt.german) {   // "(nicht mehr)" mid-clause vs. its $atend form
                        bool at_end = wi + 2 == words.size();
                        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) if (!r->atend || at_end) { ep = &*r; break; }
                    }
                    const ListEntry& e = *ep;
                    bool single_later = false;
                    if (opt.portuguese) {
                        auto one = list.find(w.text);
                        if (one != list.end()) for (auto& x : one->second) if (x.line > e.line) single_later = true;
                    }
                    if (!e.phonemes.empty() && !single_later) {
                        std::string a = e.phonemes, b;
                        size_t bar = a.find("||");
                        if (bar != std::string::npos) { b = a.substr(bar + 2); a = a.substr(0, bar); }
                        w.ph.clear(); encode_ph(a, w.ph);
                        w.unstressed = e.u; w.u_plus = e.u_plus; w.stress_pos = e.stress_pos;
                        w.given = words[wi + 1].given = true;
                        words[wi + 1].ph.clear();
                        if (opt.phrase_one_domain && !b.empty()) {   // Danish: both words in w, stressed as one word
                            w.ph.clear(); encode_ph(e.phonemes, w.ph); w.phrase = true; b.clear();
                        }
                        if (!b.empty()) encode_ph(b, words[wi + 1].ph);
                        if (opt.phrase_rest_unstressed) words[wi + 1].no_default = true;   // Danish: only the stress it marks
                        if (opt.phrase_marks && e.phonemes.find_first_of("',") != std::string::npos)
                            w.no_default = words[wi + 1].no_default = true;
                        words[wi + 1].text = U"\x01";   // translated
                    } else if (opt.suffix_once && e.stress_pos && words[wi + 1].text != U"\x01") {
                        // Turkish "(ne yaptın) $1": the two words are said as one, with the entry's stress
                        Word m = w; m.text = w.text + next; m.orig = w.orig + words[wi + 1].orig; m.ph.clear();
                        translate_word(m, wi + 2 < words.size() ? words[wi + 2].text : U"", wi == 0, wi + 2 == words.size(),
                                       wi > 0 ? words[wi - 1].text : U"");
                        if (m.switch_lang.empty() && !m.ph.empty()) {
                            m.stress_pos = e.stress_pos; m.unstressed = e.u;
                            w = m;
                            words[wi + 1].ph.clear();
                            words[wi + 1].text = U"\x01";
                        }
                    }
                }
            }
            if (w.text == U"\x02" && !w.symbol) w.text = U"\x01";
            if (opt.german && wi + 1 < words.size() && words[wi + 1].glued_num && !w.text.empty() &&
                std::all_of(w.text.begin(), w.text.end(), is_digit) && has("_#" + enc(words[wi + 1].text))) {
                w.num_suffix = ph_of("_#" + enc(words[wi + 1].text));   // "20er": the number + "_#er"
                words[wi + 1].text = U"\x01";
                next = wi + 2 < words.size() ? words[wi + 2].text : U"";
            }
            if (w.silent) continue;
            if (w.text != U"\x01" || !w.ph.empty()) {
                if (wi > 0) { w.expect_noun = words[wi - 1].nounf; w.expect_verb = words[wi - 1].verbf; }
                // Portuguese: after a break (quotation mark, '/', dash) the word does not see the word before it
                bool no_prev = wi == 0 || (opt.portuguese && w.pause_before);
                ctx_next_hyph_ = opt.letter_by_rules && wi + 1 < words.size() && words[wi + 1].join_prev;
                if (w.ph.empty() && w.text != U"\x01" && w.switch_lang.empty())
                    translate_word(w, next, wi == 0, is_last(wi), no_prev ? U"" : words[wi - 1].text);
                // Portuguese: four digits with a leading zero ("0800") end with a break
                if (opt.portuguese && w.number && w.text.size() == 4 && w.text[0] == '0' && wi + 1 < words.size()) words[wi + 1].pause_before = true;
                // Portuguese: a spelled abbreviation or a foreign word keeps the "a noun follows" state ("o FBI em")
                if (opt.portuguese && w.expect_noun && ((w.spelled && w.text.size() > 1) || !w.switch_lang.empty())) w.nounf = true;
            }
            if (!w.switch_lang.empty()) {
                std::string ipa;
                bool two = false, pair = false;
                size_t run_end = wi + 1;
                if (opt.english_ce && w.switch_lang == "EN") {
                    EspeakLangImpl* en = static_cast<EspeakLangImpl*>(other_lang("EN"));
                    if (en) en->embedded_final_ = wi + 1 == words.size();
                    // two words the English list has as one entry ("of the", "in the"): read together
                    if (en && wi + 1 < words.size() && !words[wi + 1].join_prev && words[wi + 1].text != U"\x01" &&
                        en->list.count(w.text + U" " + words[wi + 1].text)) {
                        en->embedded_final_ = wi + 2 == words.size();
                        ipa = strip_ws(en->phonemize_clause(enc(w.text + U" " + words[wi + 1].text)));
                        pair = !ipa.empty();
                    }
                    if (!pair) {
                        if (!fr_en_ && !fr_en_failed_) { fr_en_ = FrEnglishWords::load(data_dir_); fr_en_failed_ = !fr_en_; }
                        if (fr_en_) ipa = fr_en_->word(enc(w.orig));
                    }
                }
                if (ipa.empty()) {
                if (opt.korean && is_alpha(w.text[0])) {
                    // Korean: neighbouring Latin-script words are one English clause (linking r, unstressed words);
                    // short all-capitals words are letter names, one word per letter (joined after a hyphen)
                    if (EspeakLang* o = other_lang(w.switch_lang)) {
                        auto* en = static_cast<EspeakLangImpl*>(o);
                        while (run_end < words.size() && !words[run_end].glued_prev && !words[run_end].text.empty() &&
                               !ko::is_jamo(words[run_end].text[0]) && is_alpha(words[run_end].text[0]))
                            run_end++;
                        std::vector<Word> ew;
                        for (size_t k = wi; k < run_end; k++) {
                            const Word& rw = words[k];
                            size_t first = ew.size();
                            if (rw.spell_letters || (rw.text.size() == 1 && rw.text[0] < 0x80)) {
                                for (char32_t c : rw.text) {
                                    Word lw; lw.text = std::u32string{c}; lw.orig = lw.text; lw.force_spell = true;
                                    lw.join_prev = rw.join_prev && k > wi;
                                    ew.push_back(lw);
                                }
                            } else {
                                en->tokenize(rw.orig, ew);
                                if (ew.size() > first && k > wi) ew[first].join_prev = rw.join_prev;
                            }
                        }
                        ipa = strip_ws(en->render_words(ew));
                    }
                } else if (opt.indic) {
                if (EspeakLang* o = other_lang(w.switch_lang)) {
                    auto* oi = static_cast<EspeakLangImpl*>(o);
                    bool caps = opt.indic && w.orig.size() >= 2 &&
                                std::all_of(w.orig.begin(), w.orig.end(), [](char32_t c) { return c >= 'A' && c <= 'Z'; });
                    bool spelled = false;
                    if (caps) {
                        std::u32string low = w.orig;
                        for (auto& c : low) c = lower(c);
                        auto le = oi->list.find(low);
                        bool abbrev = le != oi->list.end() && std::any_of(le->second.begin(), le->second.end(), [](const ListEntry& e) { return e.abbrev; });
                        spelled = hi::english_spells(w.orig, abbrev);
                    }
                    if (w.symbol) ipa = oi->symbol_ipa(w.orig[0]);
                    else if (spelled && w.orig.size() <= 3 && [&] { for (size_t k = wi + 1; k < words.size(); k++) if (!words[k].attached) return true; return false; }()) {
                        // Hindi: a short abbreviation with more words after it: each letter a word
                        for (char32_t c : w.orig) { std::string l = strip_ws(o->phonemize_clause(enc(std::u32string{c}))); if (!l.empty()) { if (!ipa.empty()) ipa += ' '; ipa += l; } }
                    } else if (opt.indic && w.orig.size() > 2 && w.orig[0] >= 'A' && w.orig[0] <= 'Z' && w.orig[1] >= 'A' && w.orig[1] <= 'Z' &&
                               std::any_of(w.orig.begin(), w.orig.end(), [](char32_t c) { return c >= 'a' && c <= 'z'; })) {
                        // ZMapp -> Z + Mapp: capitals before a capitalised word are read apart
                        size_t k = 0;
                        while (k < w.orig.size() && w.orig[k] >= 'A' && w.orig[k] <= 'Z') k++;
                        std::string a = strip_ws(o->phonemize_clause(enc(w.orig.substr(0, k - 1))));
                        std::string b = strip_ws(o->phonemize_clause(enc(w.orig.substr(k - 1))));
                        ipa = a.empty() ? b : b.empty() ? a : a + " " + b;
                    } else if (caps) {   // Hindi: capitals read by the English rules, spelled or as a word (hi::english_spells)
                        std::u32string t = w.orig;
                        if (!spelled) for (auto& c : t) c = lower(c);
                        oi->force_spell_ = spelled;
                        ipa = strip_ws(o->phonemize_clause(enc(t)));
                        oi->force_spell_ = false;
                    } else ipa = strip_ws(o->phonemize_clause(enc(w.orig)));
                }
                } else if (EspeakLang* o = other_lang(w.switch_lang)) {
                    std::string src = enc(w.orig);
                    auto* oi = static_cast<EspeakLangImpl*>(o);
                    if (opt.trochaic && wi + 1 < words.size() && words[wi + 1].text != U"\x01" && !words[wi + 1].orig.empty()) {
                        std::u32string key = w.orig + U" " + words[wi + 1].orig;   // "of the": one entry of the English list
                        for (auto& c : key) c = lower(c);
                        if (oi->list.count(key)) { src += " " + enc(words[wi + 1].orig); two = true; }
                    }
                    if (opt.trochaic) oi->not_final_ = wi + (two ? 2 : 1) < words.size();
                    ipa = strip_ws(o->phonemize_clause(src));
                    oi->not_final_ = false;
                }
                }
                if (two && !ipa.empty()) { words[wi + 1].text = U"\x01"; words[wi + 1].ph.clear(); words[wi + 1].switch_lang.clear(); }
                if (pair) { words[wi + 1].ph.clear(); words[wi + 1].text = U"\x01"; words[wi + 1].switch_lang.clear(); }
                if (ipa.empty()) {   // no English data or nothing came out: this language's own rules
                    ignore_switch_ = true;
                    w.switch_lang.clear();
                    translate_word(w, next, wi == 0, is_last(wi), wi > 0 ? words[wi - 1].text : U"");
                    ignore_switch_ = false;
                }
                if (!ipa.empty()) {
                    foreign_.push_back(ipa);
                    PhItem it{pause < 0 ? 0 : pause, -1, true, true, ++word_counter};
                    it.foreign = (int)foreign_.size() - 1;
                    it.join = w.join_prev;
                    L.push_back(it);
                    wi = run_end - 1;
                    continue;
                }
            }
            if (opt.finnish) {
                int len = ps.find(":");
                if (dbl_pending && len >= 0 && !w.fi_pause_before) {
                    // a stop is lengthened at the start of this word (k -> k:); another consonant is repeated at the
                    // end of the previous word
                    for (size_t q = 0; q < w.ph.size(); q++) {
                        int p = w.ph[q];
                        if (p < 0 || ps.defs[p].type == T_STRESS) continue;
                        const PhDef& d = ps.defs[p];
                        if (d.type != T_VOWEL && d.type != T_PAUSE && d.type != T_VIRTUAL) {
                            bool stop = d.type == T_STOP || d.ipa == "k" || d.ipa == "p" || d.ipa == "t" ||
                                        d.ipa == "\xc9\xa1" || d.ipa == "g" || d.ipa == "b" || d.ipa == "d";
                            if (stop) w.ph.insert(w.ph.begin() + (long)q + 1, len);
                            else if (L.size() > 1 && L.back().foreign < 0) {
                                PhItem it = L.back();
                                it.ph = p; it.stress = -1; it.word_start = false;
                                L.push_back(it);
                            }
                        }
                        break;
                    }
                }
                dbl_pending = w.dbl_next;
            }
            // "||" inside a word's phonemes splits it into words; -10 starts a new stress domain without a space
            std::vector<std::vector<int>> parts(1);
            std::vector<int> pjoin(1, w.join_prev ? 1 : 0);
            std::vector<char> pfresh(1, 0);
            int wordend = ps.find("||");
            for (int p : w.ph) {
                if (parts.size() > pfresh.size()) pfresh.resize(parts.size(), 0);
                if (p == -13) { pfresh.back() = 1; continue; }
                if (p == wordend) { parts.emplace_back(); pjoin.push_back(0); }
                else if (p == -10 && opt.indic && parts.size() > 1 && parts.back().empty() && pjoin.back() == 1) pjoin.back() = 4;   // Hindi \x02\x02: a word of its own, no space
                else if (p == -10) { parts.emplace_back(); pjoin.push_back(1); }
                else if (p == -11) { parts.emplace_back(); pjoin.push_back(2); }
                else if (p == -12) { parts.emplace_back(); pjoin.push_back(3); }
                else parts.back().push_back(p);
            }
            long space_at = -1;   // Finnish decimal word: appended to the number's last stress domain
            if (opt.finnish && !w.decimal_ph.empty()) {
                space_at = 0;
                for (int p : parts.back()) if (p >= 0 && ps.defs[p].type != T_STRESS && ps.defs[p].type != T_PAUSE) space_at++;
                std::vector<int> dp; encode_ph(w.decimal_ph, dp);
                parts.back().insert(parts.back().end(), dp.begin(), dp.end());
            }
            size_t grp_start = 0;
            std::vector<std::pair<int, int>> phrase_st; size_t phrase_at = 0;
            if (w.phrase && parts.size() > 1) word_stress(w, wi + 2 >= words.size(), phrase_st);   // Danish: "||" is a pause: dropped
            size_t keep_prim = (size_t)-1;   // Danish: in a spelled word of 5+ letters the 3rd letter keeps its primary stress
            for (size_t pi = 0; pi < parts.size(); pi++) {
                Word pw = w; pw.ph = parts[pi]; pw.rest_part = pi > 0;
                if (w.roman && pi == 0) pw.no_default = true;
                pw.hyphen_next = opt.indic && pi + 1 == parts.size() && wi + 1 < words.size() && words[wi + 1].join_prev;
                if (opt.indic && w.number) {
                    // Hindi numbers (measured): a part after the number's first one gets no secondary stress before its
                    // primary (tiinsau ikatis); a part before hazaar / lakh / arab / kharab, which start stressed, none
                    // after it (unahattar hazaar; but ikahattar karor)
                    pw.no_pre_secondary = pi > 0;
                    pw.no_post_secondary = pi + 1 < parts.size() && is_scale_word(parts[pi + 1]) && starts_stressed(parts[pi + 1]);
                }
                long split_at = -1;   // \x0b: the items from here on are a new written word (same stress domain)
                if (std::find(pw.ph.begin(), pw.ph.end(), -14) != pw.ph.end()) {
                    std::vector<int> keep;
                    long n = 0;
                    for (int p : pw.ph) {
                        if (p == -14) { if (split_at < 0) split_at = n; continue; }   // (only the first one counts)
                        if (p >= 0 && ps.defs[p].type != T_STRESS && ps.defs[p].type != T_PAUSE) n++;
                        keep.push_back(p);
                    }
                    pw.ph.swap(keep);
                }
                if (pi + 1 < parts.size() && pjoin[pi + 1] == 3) { pw.no_default = true; pw.stress_pos = (opt.german && pi == 0 && w.stress_pos == 1) ? 1 : 0; }   // German: a list entry "$1" (gegenstand, unterschrift) puts the primary on the first syllable
                if (pjoin[pi] == 3) pw.stress_pos = w.rest_stress_pos;
                if (opt.german && pjoin[pi] == 3 && pi + 1 == parts.size()) { pw.unstressed |= w.rest_u; pw.u_plus |= (w.rest_u_plus && !w.suffixed); }   // a suffixed rest (Vereine) stays unstressed at the clause end too
                if (pi > 0 && !pjoin[pi]) { pw.stress_pos = 0; }
                pw.number_tail = opt.portuguese && w.number && (pi > 0 || w.fraction) && !(pi < pfresh.size() && pfresh[pi]);
                if (opt.french_numbers && w.number && (pi + 1 < parts.size() || w.num_unstressed)) pw.no_stress = true;
                if (w.phrase_head) pw.no_stress = true;
                if (opt.prefix_word_flags && w.rest_suffix && pjoin[pi] == 3 && pi > 0)
                    for (int p : parts[pi - 1]) if (is_vowel_ph(p)) pw.extra_syl++;
                if (w.num_grouped && pi + 1 < parts.size() && !pw.num_unstressed && parts[pi] == mil_ph) pw.no_stress = false;
                if (opt.french_numbers && w.number && !w.num_unstressed && pi + 1 < parts.size()) {
                    if (w.digit_groups) pw.no_stress = false;
                    else if (!big_ph.empty() && std::find(big_ph.begin(), big_ph.end(), parts[pi]) != big_ph.end()) pw.no_stress = false;
                    else if (parts[pi] == mil_ph && w.text.size() >= 5 && w.text[0] != '0') pw.no_stress = false;
                }
                std::vector<std::pair<int, int>> st;
                if (!phrase_st.empty()) {   // Danish: this word's share of the phrase's pattern
                    size_t cnt = 0;
                    for (int p : parts[pi]) if (p >= 0 && ps.defs[p].type != T_STRESS && ps.defs[p].type != T_PAUSE) cnt++;
                    for (size_t q = 0; q < cnt && phrase_at < phrase_st.size(); q++) st.push_back(phrase_st[phrase_at++]);
                } else
                word_stress(pw, is_last(wi), st);
                if (embedded_ && is_last(wi) && pw.unstressend)
                    for (auto& x : st) if (x.second >= 4) x.second = 2;   // $unstressend: weak at the end of a clause
                if (w.phrase_tail && !(w.phrase_tail == 2 && wi + 1 == words.size()))
                    for (auto& x : st) if (x.second >= 4) x.second = 2;   // the phrase's slight stress
                if (st.empty()) continue;
                if (pw.marks_only) {   // Hindi decimal word: after an unstressed syllable its first syllable takes a secondary stress
                    int last = -1;
                    for (size_t q = L.size(); q-- > 0;) if (L[q].foreign < 0 && is_vowel_ph(L[q].ph)) { last = L[q].stress; break; }
                    if (last >= 0 && last < 2) for (auto& x : st) if (is_vowel_ph(x.first)) { if (x.second < 2) x.second = 2; break; }
                }
                bool prev_primary = false;
                for (size_t q = spell_start; q < L.size(); q++) if (L[q].stress >= 4) prev_primary = true;
                // Portuguese: hyphenated words are one word ("fazê-lo")
                bool hyphen_same_pt = opt.portuguese && pi == 0 && w.join_prev && word_counter > 0;
                // Hindi: hyphenated words are one word for the phoneme rules (each part keeps its own stress)
                if (opt.indic ? ((pi == 0 && !(w.join_prev && word_counter > 0)) || pjoin[pi] == 0 || pjoin[pi] == 4 ||
                                 (pjoin[pi] == 3 && opt.post_secondary && !prev_primary && !(opt.german && pi + 1 < parts.size() && pjoin[pi + 1] == 3 && !parts[pi].empty() && ps.defs[parts[pi][0]].type != T_PAUSE)))
                              : (!hyphen_same_pt && ((pi == 0 && !(opt.hyphen_one_word && w.join_prev && wi > 0)) || (pi > 0 && pjoin[pi] == 0) || (pjoin[pi] == 3 && opt.post_secondary && !prev_primary && !(opt.german && pi + 1 < parts.size() && pjoin[pi + 1] == 3 && !parts[pi].empty() && ps.defs[parts[pi][0]].type != T_PAUSE)))))
                    ++word_counter;
                int widx = word_counter;
                if (opt.brk_pause && pi == 0 && w.brk && L.size() > 1 && wi + 1 < words.size()) {
                    PhItem pz{pause < 0 ? 0 : pause, -1, false, false, word_counter * 1000 + 999};
                    L.push_back(pz);
                }
                if (opt.vowel_pause && (pi == 0 || pjoin[pi] == 0) && L.size() > 1 && !st.empty() && is_vowel_ph(st[0].first)) {
                    PhItem pz{pause < 0 ? 0 : pause, -1, false, false, word_counter * 1000 + 999};
                    L.push_back(pz);
                }
                if (opt.portuguese && pi == 0 && ((!pw.ph.empty() && ps.defs[pw.ph[0]].type == T_PAUSE) ||   // "_vírgula"
                                                  (w.pause_before && L.size() > 1)))
                    L.push_back({pause < 0 ? 0 : pause, -1, true, true, ++word_counter});
                size_t first = L.size();
                // spelled letters: each new letter demotes the previous primary; from five letters on, every third letter keeps its (RSPCA, MSNBC)
                const bool grouped = (opt.german || opt.german_en) && parts.size() >= 5 && pjoin[pi] == 2;
                if (grouped && pi % 3 == 0) grp_start = L.size();
                if (pjoin[pi] == 2 && !opt.finnish && !(grouped && pi % 3 == 0))
                    for (size_t q = grouped ? grp_start : spell_start; q < L.size(); q++) if (L[q].stress >= 4 && q != keep_prim) L[q].stress = 2;
                if (pjoin[pi] == 3 || (pjoin[pi] == 2 && opt.finnish)) {   // Finnish spelling: the first letter keeps it
                    bool had = false;
                    for (size_t q = spell_start; q < L.size(); q++) if (L[q].stress >= 4) had = true;
                    if (had) for (auto& x : st) if (x.second >= 4) x.second = 2;
                }
                if (pjoin[pi] != 2 && pjoin[pi] != 3) spell_start = L.size();
                long nst = 0, nsp = 0;
                for (auto& x : st) {
                    if (ps.defs[x.first].type == T_PAUSE) {
                        if (opt.keep_word_pauses && !(w.join_prev && pi == 0 && L.size() == first && ps.defs[x.first].name != "_!")) {
                            // own word id: the next phoneme still starts the word; a hyphen-joined word's first pause
                            // is linked over (dix-huit)
                            L.push_back(PhItem{x.first, -1, false, false, widx * 1000 + 998});
                            if (L.size() == first + 1) first++;
                        }
                        continue;
                    }
                    if (nst++ == split_at && L.size() > first) {
                        L.back().word_end = true;
                        widx = ++word_counter;
                        PhItem it{x.first, x.second, true, false, widx};
                        it.number = w.number;
                        L.push_back(it);
                        continue;
                    }
                    PhItem it{x.first, x.second, false, false, widx};
                    if (x.second == kSoftUnstressed) { it.stress = 1; it.soft = true; }
                    it.number = w.number;
                    it.given = w.given;
                    if (opt.hyphen_word) {
                        if (pi == 0 && !w.join_prev) hchain = widx;
                        it.hword = hchain;
                    }
                    it.space_before = pi + 1 == parts.size() && nsp++ == space_at;
                    L.push_back(it);
                }
                if (opt.german && w.no_stress && L.size() > first) {
                    // German: the decimal word ("Komma") takes a secondary stress after an unstressed syllable (sieben Komma fuenf)
                    int prev_st = -1;
                    for (size_t q = first; q-- > 1;) if (L[q].foreign < 0 && is_vowel_ph(L[q].ph)) { prev_st = L[q].stress; break; }
                    if (prev_st >= 0 && prev_st < 2)
                        for (size_t q = first; q < L.size(); q++) if (is_vowel_ph(L[q].ph)) { L[q].stress = 2; break; }
                }
                if (opt.da_numbers && pi == 2 && parts.size() >= 5 && pjoin[2] == 2 && pjoin[1] == 2)
                    for (size_t q = first; q < L.size(); q++) if (L[q].stress >= 4) keep_prim = q;
                if (L.size() > first) { L[first].word_start = true; L[first].join = pi > 0 ? pjoin[pi] != 0 : w.join_prev; L[first].hyphen = pi == 0 && w.join_prev; L.back().word_end = true; }
            }
        }
        L.push_back({pause < 0 ? 0 : pause, -1, true, true, 1 << 30});
        // word start / end from the word ids (stress domains of one word are not word boundaries)
        for (size_t i = 1; i + 1 < L.size(); i++) {
            if (L[i].foreign >= 0 || ps.defs[L[i].ph].type == T_PAUSE) continue;
            L[i].word_start = L[i - 1].word != L[i].word;
            L[i].word_end = L[i + 1].word != L[i].word;
        }
        if (opt.final_devoice) {
            // a voiced stop / fricative before a word end, a pause or a voiceless obstruent is devoiced (from the end)
            for (size_t i = L.size(); i-- > 0;) {
                const PhDef& d = ps.defs[L[i].ph];
                if (!d.voiced || d.voicingswitch.empty() || (d.type != T_STOP && d.type != T_FRIC)) continue;
                bool dev = true;   // only voiceless obstruents follow up to the word end
                for (size_t j = i + 1; j < L.size() && L[j].word == L[i].word; j++) {
                    const PhDef& n = ps.defs[L[j].ph];
                    if (!((n.type == T_STOP || n.type == T_FRIC) && !n.voiced)) { dev = false; break; }
                }
                if (!dev) continue;
                int p = ps.find(d.voicingswitch);
                if (p >= 0) L[i].ph = p;
            }
        }

        if (opt.german) {   // German: for the phoneme programs a hyphenated compound is one word ("zwanzig-jaehrige", "Weissmeer-Ostsee")
            for (size_t i = 2; i + 1 < L.size(); i++) {
                if (!L[i].hyphen || L[i].word == L[i - 1].word || L[i].foreign >= 0 || L[i - 1].foreign >= 0 || L[i - 1].word < 0) continue;
                int old = L[i].word, nw = L[i - 1].word;
                for (size_t j = i; j + 1 < L.size() && L[j].word == old; j++) L[j].word = nw;
            }
            for (size_t i = 1; i + 1 < L.size(); i++) {
                if (L[i].foreign >= 0 || ps.defs[L[i].ph].type == T_PAUSE) continue;
                L[i].word_start = L[i - 1].word != L[i].word;
                L[i].word_end = L[i + 1].word != L[i].word;
            }
        }
        // phase 1: phoneme changes
        for (size_t i = 0; i < L.size(); i++) {
            // a phoneme that was changed is not changed again in the reference ("02" -> "0" stays "0": ɒvðə); kept
            // as it was (up to four changes) for the languages measured before
            for (int guard = 0; guard < ((embedded_ || opt.german) ? 1 : 4); guard++) {
                int app = -1; bool ret = false;
                int r = run(ps.defs[L[i].ph].prog, L, i, app, nullptr, 0, ret);
                if (app >= 0) {
                    PhItem x{app, -1, false, false, L[i].word};
                    L.insert(L.begin() + i + 1, x);
                }
                if (r == -3) {
                    // Korean: a word whose first sound is dropped (the linking ; of ㄴ/ㅁ) loses its space
                    if (opt.korean && L[i].word_start && i + 1 < L.size() && L[i + 1].word == L[i].word) L[i + 1].join = true;
                    L.erase(L.begin() + i); i--; break;
                }
                if (r < 0 || r == L[i].ph) break;
                L[i].ph = r;
            }
        }
#ifdef KT_FE_TRACE
        {
            for (auto& x : L) fprintf(stderr, "[%s t%d w%d s%d] ", ps.defs[x.ph].name.c_str(), ps.defs[x.ph].type, x.word, x.stress);
            fprintf(stderr, "\n");
        }
#endif
        // IPA
        std::string out, last_ipa;
        bool last_explicit = false;   // Danish
        int lastw = -1;
        bool spaced = false;   // Korean: a space goes before every word start that is still marked, except the first
        auto space_before = [&](const PhItem& it) {
            if (!opt.korean) return lastw >= 0 && !it.join;
            if (it.join) return false;
            bool r = spaced;
            spaced = true;
            return r;
        };
        for (size_t i = 0; i < L.size(); i++) {
            const PhDef& d = ps.defs[L[i].ph];
            if (L[i].foreign >= 0) {
                if (space_before(L[i])) out += ' ';
                out += foreign_[L[i].foreign];
                lastw = L[i].word;
                continue;
            }
            if (d.type == T_PAUSE || d.type == T_STRESS || (d.type == T_VIRTUAL && d.name != ":")) continue;
            if (L[i].word != lastw) { if (space_before(L[i])) out += ' '; lastw = L[i].word; }
            else if (L[i].space_before) out += ' ';
            if (d.type == T_VOWEL && L[i].stress >= 2) out += L[i].stress >= 4 ? "\xcb\x88" : "\xcb\x8c";
            std::string ipa = d.ipa;
            if (d.name == ":" && i > 0) {
                const PhDef& pd = ps.defs[L[i - 1].ph];
                if (opt.length_repeats_ipa && pd.name == ":") continue;   // Danish "O::" is one long vowel
                if (opt.length_repeats_ipa && !(pd.type == T_NASAL || pd.type == T_LIQUID || pd.type == T_FRIC) && last_explicit) { out += last_ipa; continue; }
                if (pd.type == T_NASAL || pd.type == T_LIQUID || pd.type == T_FRIC ||
                    (opt.colon_repeats_vowel_ipa && pd.type == T_VOWEL && pd.has_ipa)) { out += last_ipa; continue; }
                if (opt.length_if_stressed && pd.type == T_VOWEL && L[i - 1].stress < 4) continue;
                if (opt.indic && (pd.has_ipa || std::any_of(pd.prog.begin(), pd.prog.end(), [](const Stmt& x) { return x.kind == S_IPA; }))) {
                    out += last_ipa; continue;
                }
            }
            int app = -1; bool ret = false;
            std::string cond_ipa = "\x01";
            run(d.prog, L, i, app, &cond_ipa, 0, ret);
            if (cond_ipa != "\x01") ipa = cond_ipa;
            last_explicit = d.has_ipa || cond_ipa != "\x01";
            // tie between the letters of one phoneme (phonemizer tie='^')
            std::u32string u = dec(ipa);
            std::string o;
            for (size_t k = 0; k < u.size(); k++) {
                if (u[k] == U'|') continue;   // "ə|l" (syllabic l of English "apple"): the bar only separates, the tie goes between the letters
                bool mod = (u[k] >= 0x300 && u[k] <= 0x36F) || (u[k] >= 0x2B0 && u[k] <= 0x2FF);
                if (k > 0 && !mod) o += '^';
                put_utf8(o, u[k]);
            }
            out += o;
            last_ipa = o;
        }
        return out;
    }
};

struct VoiceCfg { const char* voice; const char* dict; const char* table; std::vector<int> dictrules; int stress_rule; bool final_es; bool roman = false; bool secondary = true; bool vowel_pause = false; bool keep_first = false; bool german = false; };
const VoiceCfg kVoices[] = {
    {"es", "es", "es", {1}, 2, true, true},
    {"it", "it", "it", {}, 2, false, false, false},
    {"en", "en", "en", {}, 2, false},
    {"de", "de", "de", {}, 1, false, false, true, true, true, true},
    {"fr-fr", "fr", "fr", {1}, 3, false, false, false},
    {"pt-br", "pt", "pt", {2}, 3, false, true},
    {"ru", "ru", "ru", {}, 2, false},
    {"hi", "hi", "hi", {}, 6, false},
    {"sw", "sw", "sw", {}, 2, false},
    {"tr", "tr", "tr", {}, 3, false, false, false},
    {"sv", "sv", "sv", {}, 1, false},
    // English words inside Swedish text: the English data with three settings measured on them ("internet" ->
    // 'Int3n,Et, "kostymdesign" -> k'ostymd,esign); es and it keep the plain "en" reader they were measured with.
    {"en-x-sv", "en", "en", {}, 2, false},
    {"ko", "ko", "ko", {}, 1, false},
    {"fi", "fi", "fi", {}, 1, false, false, false},
    {"da", "da", "da", {}, 1, false},
};

}  // namespace

std::unique_ptr<EspeakLang> EspeakLang::load(const std::string& voice, const std::string& dir, std::string* err) {
    const VoiceCfg* vc = nullptr;
    for (auto& v : kVoices) if (voice == v.voice) vc = &v;
    if (!vc) { *err = "unsupported espeak voice " + voice; return nullptr; }
    auto e = std::make_unique<EspeakLangImpl>();
    e->opt.voice = vc->voice; e->opt.table = vc->table; e->opt.dictrules = vc->dictrules; e->opt.stress_rule = vc->stress_rule;
    e->opt.final_spanish = vc->final_es; e->opt.roman = vc->roman; e->opt.secondary_from_start = vc->secondary;
    e->opt.vowel_pause = vc->vowel_pause; e->opt.keep_first_primary = vc->keep_first;
    e->opt.italian_alt = std::string(vc->voice) == "it";
    e->opt.secondary_first_only = std::string(vc->voice) == "pt-br";
    e->opt.omit_one_hundred = e->opt.number_domains = e->opt.italian_alt;
    e->opt.dim_non_initial = vc->german;
    e->opt.german = vc->german;
    e->opt.post_secondary = vc->german; e->opt.final_devoice = vc->german; e->opt.num_swap = vc->german;
    const bool french = std::string(vc->voice) == "fr-fr";
    e->opt.portuguese = std::string(vc->voice) == "pt-br";
    e->opt.hyphen_join = vc->german || std::string(vc->voice) == "it" || e->opt.portuguese || french;
    e->opt.tie_later = e->opt.prefix_word_flags = e->opt.y_vowel = e->opt.letter_by_rules = e->opt.length_if_stressed = e->opt.verb_context = e->opt.brk_pause = french;
    e->opt.french_numbers = e->opt.keep_word_pauses = e->opt.multiword_phrase = e->opt.syllables_from_match = e->opt.hyphen_word = french;
    if (french) e->opt.omit_one_hundred = true;
    e->opt.english_ce = french;
    e->opt.tie_by_phonemes = french;
    if (std::string(vc->voice) == "sw")
        e->opt.hyphen_join = e->opt.syllable_run = e->opt.phrase_marks = e->opt.swahili = e->opt.syllabic_no_secondary = true;
    if (std::string(vc->voice) == "tr") {
        e->opt.stress_before_unstressed = e->opt.dotless_i = e->opt.colon_repeats_vowel_ipa = true;
        e->opt.number_suffix_word = e->opt.hundreds_domain = e->opt.omit_one_hundred = true;
        e->opt.max_suffix_depth = 1;
        e->opt.spell_clusters = e->opt.dot_thousands = e->opt.stressed_decimal = e->opt.caps_word_split = true;
        e->opt.hyphen_join = e->opt.caps_spell_by_clause = e->opt.y_is_vowel = e->opt.hyphen_one_word = true;
        e->opt.suffix_once = e->opt.tie_later_rule = e->opt.length_mark_transparent = true;
    }
    if (std::string(vc->voice) == "sv") {
        e->opt.keep_all_primary = e->opt.trochaic = e->opt.hyphen_join = e->opt.y_vowel = true;
        e->opt.amp_stressable = e->opt.sv_numbers = e->opt.symbol_rules = e->opt.prefix_cut = true;
        e->opt.attached_context = e->opt.caps_read = true;
        e->opt.tie_later = true; e->opt.class_minus = 1;
    }
    if (std::string(vc->voice) == "en-x-sv") {
        e->opt.keep_first_primary = e->opt.final_secondary = e->ps.call_type = true;
        e->opt.stress_rule = 1; e->opt.trochaic = e->opt.sv_english = true;
    }
    if (std::string(vc->voice) == "da") {   // Danish: every Danish-only behaviour is behind these flags (the other voices never set them)
        e->opt.da_y_vowel = e->opt.seq_conds = e->opt.length_repeats_ipa = e->opt.both_groups = e->opt.measured_scores = e->opt.all_primaries = e->opt.da_trochaic = e->opt.phrase_rest_unstressed = e->opt.hyphen_join = e->opt.da_numbers = e->opt.at_word_start = true;
        e->opt.sentence_atend = e->opt.stem_unstressed = e->opt.syllabic_blocks = e->opt.list_ph_after_flags = true;
        e->opt.stressable_pre = e->opt.phrase_one_domain = e->opt.post_after_ph = e->opt.da_tokens = e->opt.prefix_fresh = e->opt.first_secondary_blocks = true;
    }
    e->opt.korean = std::string(vc->voice) == "ko";
    if (std::string(vc->voice) == "fi") e->opt.finnish = e->opt.hyphen_join = true;
    e->data_dir_ = dir;
    if (french) e->ps.import_tables = {"en"};   // ph_french imports four English phonemes (I, I2, a#, oU)
    e->opt.indic = std::string(vc->voice) == "hi";
    e->data_dir_ = dir;
    e->ps.hash_aspirated = e->opt.indic;
    if (e->opt.indic) e->ps.extra_tables = {"consonants"};
    e->ps.import_vowels = vc->german;
    if (!e->ps.load(dir + "/phsource", vc->table, err)) return nullptr;
    if (e->opt.measured_scores) e->build_ph_rank();   // only Danish orders rules by phoneme rank
    std::string d = vc->dict;
    if (!e->load_rules(dir + "/" + d + "_rules", err)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_list", err, true)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_listx", err, false)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_extra", err, false)) return nullptr;
    if (vc->german && !e->load_list(dir + "/" + d + "_emoji", err, false, true)) return nullptr;   // optional: names of ¥ £ ½ ¾ ...
    if (e->opt.german) {
        auto g = e->groups.find(U"");
        if (g != e->groups.end()) for (int ri : g->second) {
            const Rule& r = e->rules[ri];
            if (r.match != U"-" || r.post != U"_D") continue;
            if (r.pre == U"_D") e->dash_digits_ = r.phonemes;   // "D_) - (_D" (pre is stored reversed)
            else if (r.pre == U"__") e->minus_ = r.phonemes;
        }
    }
    return e;
}

}  // namespace kt_fe
