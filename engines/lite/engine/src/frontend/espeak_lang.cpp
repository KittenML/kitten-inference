// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Original implementation; reads espeak-ng's data files (GPL-3.0, downloaded at run time, never bundled).
// The ASCII -> IPA table is from KittenML's CEPhonemizer (Apache-2.0).
#include "frontend/espeak_lang.h"

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
    return false;
}
bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
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
struct Cond { int which = C_FALSE; bool neg = false; std::string attr; };
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
    bool voiced = false, rhotic = false, palatal = false, unstressed = false, nonsyl = false, flag1 = false, flag2 = false;
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
std::string default_ipa(const std::string& name) {
    std::string r;
    bool first = true;
    for (char c : name) {
        unsigned char uc = (unsigned char)c;
        if (c == '/') break;
        if (!first && c >= '0' && c <= '9') continue;
        if (c == '#' && !first) break;
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
        for (auto& d : defs) {
            max_name = std::max(max_name, d.name.size());
            if (!d.has_ipa) d.ipa = default_ipa(d.name);
        }
        return true;
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
        bool neg = false;
        while (i < tk.size() && tk[i] != "THEN") {
            const std::string& t = tk[i++];
            if (t == "NOT") { neg = true; continue; }
            if (t == "AND") { s.is_or = false; continue; }
            if (t == "OR") { s.is_or = true; continue; }
            Cond c; parse_cond(t, c); c.neg = neg; neg = false;
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
                else if (t == "vls") { d.voiced = false; i++; }
                else if (t == "rhotic") { d.rhotic = true; i++; }
                else if (t == "palatal" || t == "pal") { d.palatal = true; i++; }
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
    int alt = 0;
    int u_n = 0;            // $u1/$u2/$u3
    int cond = 0;
    int nwords = 1;
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
};

struct PhItem {   // one phoneme of a clause
    int ph;
    int stress = -1;   // vowels: 0 diminished .. 4 primary
    bool word_start = false, word_end = false;
    int word = 0;
    bool join = false;   // no space before this word in the output
    bool number = false;
    int foreign = -1;    // index into foreign_: print that IPA instead of the phoneme
};

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
    mutable int stem_alt_ = 0, stem_stress_pos_ = 0;
    mutable bool stem_u_ = false;
    mutable int rest_stress_pos_ = 0;
    mutable bool prefix_taken_ = false;
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
        if (!voice || opt.voice == voice || depth_guard_) return nullptr;
        auto it = other_.find(voice);
        if (it != other_.end()) return it->second.get();
        if (other_failed_[voice]) return nullptr;
        std::string err;
        auto e = EspeakLang::load(voice, data_dir_, &err);
        if (!e) { other_failed_[voice] = true; return nullptr; }
        static_cast<EspeakLangImpl*>(e.get())->depth_guard_ = true;   // no further switching from there
        return (other_[voice] = std::move(e)).get();
    }
    bool depth_guard_ = false;
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
        while (j < l.size() && isdigit((unsigned char)l[j])) n = n * 10 + (l[j++] - '0');
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
            if (t[0][0] == '.') {
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
            if (k < t.size() && t[k][0] == '(') { post = dec(t[k].substr(1)); k++; }
            std::string ph = k < t.size() ? t[k] : "";
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
            r.post = unoctal(post);
            r.phonemes = ph;
            rules.push_back(r);
            std::u32string key = group;
            groups[key].push_back((int)rules.size() - 1);
        }
        for (auto& g : lgroups) std::stable_sort(g.begin(), g.end(), [](auto& a, auto& b) { return a.size() > b.size(); });
        return true;
    }

    bool load_list(const std::string& path, std::string* err, bool required) {
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
                else if (f == "$alt" || f == "$alt1") e.alt |= 1;
                else if (f.rfind("$alt", 0) == 0 && f.size() == 5) e.alt |= 1 << (f[4] - '1');
            }
            e.text = any_text;
            if (!cond_ok(cond)) continue;
            std::u32string k32 = dec(key);
            for (auto& c : k32) c = lower(c);
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
                i++; continue;
            }
            out.push_back(best);
            i += bl;
        }
    }

    // ---------- letters
    bool is_vowel_letter(char32_t c) const {
        char32_t b = base_letter(lower(c));
        if (b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u') return true;
        if (c == 0xE6 || c == 0xF8 || c == 0x153) return true;
        if (c >= 0x430 && c <= 0x44F) { static const std::u32string v = U"аеёиоуыэюя"; return v.find(c) != std::u32string::npos; }
        if (c == 0x451) return true;
        return false;
    }
    bool in_set(char ctx, char32_t c) const {
        if (!is_alpha(c)) return false;
        char32_t b = base_letter(lower(c));
        switch (ctx) {
            case 'A': return is_vowel_letter(c);
            case 'C': return !is_vowel_letter(c);
            case 'B': return b < 128 && strchr("bcdfgjklmnpqstvxz", (char)b);
            case 'H': return b < 128 && strchr("hlmnr", (char)b);
            case 'F': return b < 128 && strchr("cfhkpqstx", (char)b);
            case 'G': return b < 128 && strchr("bdgjlmnrvwyz", (char)b);
            case 'Y': return b < 128 && strchr("eiy", (char)b);
        }
        return false;
    }

    // ---------- rule matching. w = " word nextword " with letters lower-cased; pos = index into w.
    struct MatchCtx { bool suffix_removed = false, prefix_removed = false; int alt = 0; long stem_start = -1; long block_before = -1; };

    // returns score (<=0: no match)
    int match_rule(const Rule& r, const std::u32string& w, size_t pos, size_t wend, const MatchCtx& mc) const {
        if (!cond_ok(r.cond)) return 0;
        int pts = 1;
        size_t n = r.match.size();
        if (pos + n > wend) return 0;
        for (size_t k = 0; k < n; k++) if (w[pos + k] != r.match[k]) return 0;
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
            if (c == 'N') { if (mc.suffix_removed) return 0; continue; }
            if (c == '#') continue;
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
                pts += 20 - dist;
                continue;
            }
            if (c == '/' && k + 1 < po.size()) { c = po[++k]; if (x != c) return 0; pts += 21 - dist; j++; continue; }
            if (c == '%') { if (j == 0 || x != w[j - 1]) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'K') { if (is_alpha(x) && is_vowel_letter(x)) return 0; pts += 20 - dist; j++; continue; }
            if (c == 'D') { if (!is_digit(x)) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'Z') { if (is_alpha(x)) return 0; pts += 21 - dist; j++; continue; }
            if (c == 'X') {   // no vowel until the word end
                for (size_t q = j; q < w.size() && !is_space(w[q]); q++) if (is_vowel_letter(w[q])) return 0;
                pts += 19 - dist; continue;
            }
            if (c == '@') {
                size_t q = j;
                while (q < w.size() && !is_space(w[q]) && !is_vowel_letter(w[q])) q++;
                if (q >= w.size() || is_space(w[q])) return 0;
                while (q < w.size() && is_vowel_letter(w[q])) q++;
                j = q; pts += 18 - dist; continue;
            }
            if (c >= 'A' && c <= 'Z') { if (!in_set((char)c, x)) return 0; pts += 20 - dist; j++; continue; }
            if (x != c) return 0;
            pts += 21 - dist; j++;
        }
        // pre (stored reversed)
        long i = (long)pos - 1; dist = -2;
        const std::u32string& pr = r.pre;
        for (size_t k = 0; k < pr.size(); k++) {
            dist = std::min(dist + 2, 19);
            char32_t c = pr[k];
            char32_t x = i >= 0 ? w[i] : ' ';
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
                pts += 20 - dist;
                continue;
            }
            if (k + 1 < pr.size() && pr[k + 1] == '/') { k++; if (x != c) return 0; pts += 21 - dist; i--; continue; }
            if (c == '%') { if (i < 0 || (size_t)(i + 1) >= w.size() || x != w[i + 1]) return 0; pts += 21 - dist; i--; continue; }
            if (c == '@' || c == '&') {
                long q = i, lo = mc.block_before >= 0 ? mc.block_before : 0;   // an unstressed prefix is not a syllable here
                while (q >= lo && !is_space(w[q]) && !is_vowel_letter(w[q])) q--;
                if (q < lo || is_space(w[q])) return 0;
                while (q >= lo && is_vowel_letter(w[q])) q--;
                i = q; pts += 18 - dist; continue;
            }
            if (c == 'K') { if (is_alpha(x) && is_vowel_letter(x)) return 0; pts += 20 - dist; i--; continue; }
            if (c == 'D') { if (!is_digit(x)) return 0; pts += 21 - dist; i--; continue; }
            if (c == 'Z') { if (is_alpha(x)) return 0; pts += 21 - dist; i--; continue; }
            if (c == 'V') return 0;
            if (c == 'X') {   // no vowel from here back to the word start
                for (long q = i; q >= 0 && !is_space(w[q]); q--) if (is_vowel_letter(w[q])) return 0;
                pts += 19 - dist; continue;
            }
            if (c >= 'A' && c <= 'Z') { if (!in_set((char)c, x)) return 0; pts += 20 - dist; i--; continue; }
            if (x != c) return 0;
            pts += 21 - dist; i--;
        }
        return pts;
    }

    // Translates the letters w[b..e) by rules. Returns mnemonic string.
    std::string apply_rules(const std::u32string& w, size_t b, size_t e, const MatchCtx& mc, int depth = 0) const {
        std::string out;
        size_t pos = b;
        while (pos < e) {
            char32_t c = w[pos];
            const Rule* best = nullptr; int best_pts = 0;
            auto try_group = [&](const std::u32string& key) {
                auto it = groups.find(key);
                if (it == groups.end()) return false;
                for (int ri : it->second) {
                    int p = match_rule(rules[ri], w, pos, e, mc);
                    if (p > best_pts) { best_pts = p; best = &rules[ri]; }
                }
                return true;
            };
            if (pos + 1 < e) try_group(std::u32string{c, w[pos + 1]});
            bool had2 = best != nullptr;
            if (!had2) {
                try_group(std::u32string{c});
                if (!best) try_group(U"");
                if (!best) {
                    char32_t bl = base_letter(c);
                    if (bl != c) try_group(std::u32string{bl});
                }
            }
            if (!best) { pos++; continue; }
            if (best->phonemes.rfind("_^_", 0) == 0 && !ignore_switch_) { switch_lang_ = "EN"; return ""; }
#ifdef KT_FE_TRACE
            fprintf(stderr, "rule @%zu: %s)%s(%s -> %s [%d]\n", pos, enc(std::u32string(best->pre.rbegin(), best->pre.rend())).c_str(), enc(best->match).c_str(), enc(best->post).c_str(), best->phonemes.c_str(), best_pts);
#endif
            if (best->suffix > 0 && pos + best->match.size() >= e) {
                size_t se = pos + best->match.size() - (size_t)best->suffix;
                if (se > b) {
                    std::u32string stem = w.substr(b, se - b);
                    const ListEntry* sl = lookup_entry(stem, true, false);
                    size_t ap = stem.rfind('\'');
                    bool via_ap = false;
                    if (!sl && ap != std::u32string::npos) { sl = lookup_entry(stem.substr(ap + 1), true, false); via_ap = sl != nullptr; }
                    if (sl) { stem_alt_ |= sl->alt; if (sl->stress_pos) stem_stress_pos_ = sl->stress_pos; if (sl->u) stem_u_ = true; }
                    if (sl && !sl->phonemes.empty() && best->affix_flags.find('q') != std::string::npos && depth == 0 && se == pos) {
                        // the stem is in the list with its phonemes (after an elision such as l', the part before it by rules)
                        std::string head = via_ap ? apply_rules(w, b, b + ap + 1, mc, depth + 1) + "\x05" : "";
                        return head + sl->phonemes + "\x05" + best->phonemes;
                    }
                }
            }
            if (best->suffix > 0 && depth < 3 && pos + best->match.size() >= e && best->affix_flags.find('q') == std::string::npos) {
                // standard suffix: translate the stem, then add the suffix phonemes
                size_t slen = (size_t)best->suffix;
                size_t stem_end = pos + best->match.size() - slen;
                if (stem_end > b) {
                    MatchCtx m2 = mc; m2.suffix_removed = true;
                    std::u32string stem = w.substr(b, stem_end - b);
                    std::string sp;
                    const ListEntry* le = lookup_entry(stem, true, false);
                    if (le && !le->phonemes.empty()) sp = le->phonemes;
                    else sp = apply_rules(w.substr(0, stem_end) + U" ", b, stem_end, m2, depth + 1);
                    // phonemes of the matched part that are not the suffix are dropped (stem re-translated)
                    bool sep = best->affix_flags.find('t') != std::string::npos;
                    return sp + (sep ? "\x02" : "") + best->phonemes;
                }
            }
            if (best->prefix > 0 && depth < 3) {
                size_t pend = pos + (size_t)best->prefix;
                MatchCtx m2 = mc; m2.prefix_removed = true;
                std::string rest;
                std::u32string tail = w.substr(pend, e - pend);
                const ListEntry* le = lookup_entry(tail, false, true);
                if (le) { rest_stress_pos_ = le->stress_pos; stem_alt_ |= le->alt; }
                prefix_taken_ = true;
                if (le && !le->phonemes.empty()) rest = le->phonemes;
                else {
                    m2.stem_start = (long)pend;
                    if (opt.post_secondary && best->phonemes.find('\'') == std::string::npos) m2.block_before = (long)pend;
                    rest = apply_rules(w, pend, e, m2, depth + 1);
                }
                return out + best->phonemes + "\x04" + rest;
            }
            out += best->phonemes;
            out += '\x05';   // phoneme names never span two rules
            pos += std::max<size_t>(1, best->match.size());
        }
        return out;
    }

    const ListEntry* lookup_entry(const std::u32string& word, bool after_suffix, bool after_prefix) const {
        auto it = list.find(word);
        if (it == list.end()) return nullptr;
        for (auto r = it->second.rbegin(); r != it->second.rend(); ++r) {
            if (r->atend || r->atstart || r->capital || r->allcaps || r->verb || r->noun || r->past || r->hasdot || r->sentence) continue;
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
        int alt = 0;
        bool no_default = false;   // a prefix: only the stress its phonemes mark
        bool listed_marks = false; // phonemes from the word list with their own stress marks
        bool expect_noun = false, expect_verb = false;   // the previous word has $nounf / $verbf
        bool nounf = false, verbf = false;
        int rest_stress_pos = 0;   // $N of the word after a prefix
        std::string switch_lang;   // set: translate the word with that language (English only)
        std::u32string orig;       // the word as written
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
            if (r->past || r->hasdot || r->sentence) continue;
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
            else o += ph_of("_" + std::to_string(h)) + (opt.number_domains ? "\x02" : "") + ph_of("_0C");
            (void)exact_hundred_ok;
        }
        if (r) {
            if (h && !o.empty()) o += "||";
            std::string k = "_" + std::to_string(r);
            if (has(k)) o += ph_of(k);
            else if (r < 10) o += ph_of("_" + std::to_string(r));
            else {
                int t = r / 10, u = r % 10;
                std::string tk = "_" + std::to_string(t) + "X";
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
        if (rest) o += number_lt1000((int)rest);
        return o;
    }

    std::unordered_map<char32_t, bool> known_cache;
    bool known_letter(char32_t c) {
        auto it = known_cache.find(c);
        if (it != known_cache.end()) return it->second;
        bool k = groups.count(std::u32string{c}) > 0;
        if (!k) { auto g = groups.find(U""); if (g != groups.end()) for (int ri : g->second) if (!rules[ri].match.empty() && rules[ri].match[0] == c) k = true; }
        if (!k) for (auto& kv : groups) if (kv.first.size() == 2 && kv.first[0] == c) k = true;
        if (!k && list.count(std::u32string{c})) k = true;
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
    std::string spell(const std::u32string& t) const {
        std::string o;
        for (char32_t c : t) {
            std::u32string k{c};
            std::string p;
            const ListEntry* le = lookup_entry(std::u32string{'_', c}, false, false);
            if (le && !le->phonemes.empty()) p = le->phonemes;
            else if ((le = lookup_entry(k, false, false)) && !le->phonemes.empty()) p = le->phonemes;
            else { std::u32string w = U" " + k + U"  "; p = apply_rules(w, 1, 2, MatchCtx()); }
            if (!o.empty()) o += "\x03";
            o += p;
        }
        return o;
    }

    void translate_word(Word& wd, const std::u32string& next, bool atstart, bool atend, const std::u32string& prev = U"",
                        int depth = 0) const {
        std::string mn;
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
        if (opt.roman && wd.allcaps && wd.text.size() >= 2) {
            int v = roman_value(wd.text);
            if (v > 0 && v < 4000) {
                std::string ds = std::to_string(v);
                mn = number_word(std::u32string(ds.begin(), ds.end()));
                encode_ph(mn, wd.ph);
                return;
            }
        }
        const ListEntry* le = lookup_ctx(wd.text, wd, atstart, atend);
        if (le && ignore_switch_ && le->phonemes.find("_^_") != std::string::npos) le = nullptr;
        if (le && le->text && !le->phonemes.empty() && depth < 4) {   // depth: list entries could point at each other
            // "sounds like" another word
            std::u32string t2 = dec(le->phonemes);
            Word w2; w2.text = t2;
            for (auto& c : w2.text) c = lower(c);
            translate_word(w2, next, atstart, atend, prev, depth + 1);
            wd.ph = w2.ph; wd.stress_pos = w2.stress_pos; wd.unstressed = w2.unstressed;
            return;
        }
        if (le) {
            wd.nounf = le->nounf; wd.verbf = le->verbf;
            wd.stress_pos = le->stress_pos;
            wd.alt = le->alt;
            wd.listed_marks = le->phonemes.find_first_of("',") != std::string::npos;
            wd.unstressed = le->u; wd.u_plus = le->u_plus || le->strend; wd.u_n = le->u_n;
            mn = le->phonemes;
        }
        if (mn.empty()) {
            bool all_digits = !wd.text.empty() && std::all_of(wd.text.begin(), wd.text.end(), is_digit);
            bool no_vowel = !all_digits && std::none_of(wd.text.begin(), wd.text.end(), [&](char32_t c) { return is_vowel_letter(c); });
            if (all_digits) { mn = number_word(wd.text); wd.number = true; }
            else if (no_vowel || (le && le->abbrev) || (wd.allcaps && wd.text.size() <= 3)) mn = spell(wd.text);
            else {
                std::u32string w = U" " + prev + U" " + wd.text + U" " + next + U" ";
                size_t b = prev.size() + 2;
                MatchCtx mc;
                if (le) mc.alt = le->alt;
                stem_alt_ = 0; stem_stress_pos_ = 0; stem_u_ = false; rest_stress_pos_ = 0; prefix_taken_ = false;
                mn = apply_rules(w, b, b + wd.text.size(), mc);
                if (prefix_taken_ && !rest_stress_pos_) rest_stress_pos_ = stem_stress_pos_;
                if (prefix_taken_) stem_stress_pos_ = 0;
                wd.rest_stress_pos = rest_stress_pos_;
                wd.alt |= stem_alt_;
                if (stem_u_ && (opt.post_secondary || opt.italian_alt)) wd.unstressed = true;
                if (!wd.stress_pos && stem_stress_pos_) wd.stress_pos = stem_stress_pos_;
            }
        }
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
        encode_ph(mn, wd.ph);
    }

    // ---------- stress
    bool is_vowel_ph(int p) const { return p >= 0 && ps.defs[p].type == T_VOWEL && !ps.defs[p].nonsyl; }
    int stress_of(int p) const { return (p >= 0 && ps.defs[p].type == T_STRESS) ? ps.defs[p].stress_type : -1; }

    // Assign a stress level to each vowel of a word. Returns per-item stress (vowels only), dropping stress phonemes.
    void word_stress(const Word& wd, bool clause_end, std::vector<std::pair<int, int>>& out) const {
        struct V { int ph; int st; };
        std::vector<V> items;
        int pending = -1;
        bool has_primary = false;
        for (int p : wd.ph) {
            int s = stress_of(p);
            if (s >= 0) {
                if (ps.defs[p].name == "=") {   // stress on the previous vowel
                    for (auto it = items.rbegin(); it != items.rend(); ++it) if (is_vowel_ph(it->ph)) { it->st = 4; has_primary = true; break; }
                    continue;
                }
                pending = s; continue;
            }
            if (ps.defs[p].type == T_PAUSE) continue;
            V v{p, -1};
            if (is_vowel_ph(p)) {
                v.st = pending;
                if (pending >= 4) has_primary = true;
                pending = -1;
            }
            items.push_back(v);
        }
        std::vector<int> vi;
        for (size_t i = 0; i < items.size(); i++) if (is_vowel_ph(items[i].ph)) vi.push_back((int)i);
        int nv = (int)vi.size();
        // demote '' (5) / emphasized to primary 4; extra primaries stay
        for (auto& it : items) if (it.st > 4) it.st = 4;
        if (opt.keep_first_primary) {
            bool seen = false;
            for (auto& it : items) if (it.st >= 4) { if (seen) it.st = 2; seen = true; }
        } else {
            int last = -1;
            for (size_t k = 0; k < items.size(); k++) if (items[k].st >= 4) last = (int)k;
            for (int k = 0; k < last; k++) if (items[k].st >= 4) items[k].st = 2;
        }
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
        if (wd.stress_pos > 0 && wd.stress_pos <= nv) {
            for (auto& it : items) if (it.st >= 4) it.st = -1;
            items[vi[wd.stress_pos - 1]].st = 4; has_primary = true;
        }
        if (unstressed_word && nv == 1) for (auto& it : items) if (it.st >= 2) it.st = 1;
        bool any_mark = false;
        for (auto& it : items) if (it.st >= 2) any_mark = true;
        if (!has_primary && nv > 0 && !wd.no_default && !(wd.listed_marks && any_mark)) {
            int target = default_stress_pos(wd, items.size(), vi, [&](int k) { return items[vi[k]].st; },
                                            [&](int k) { return ps.defs[items[vi[k]].ph].unstressed || (items[vi[k]].st >= 0 && items[vi[k]].st < 2); });
            bool any_free = false;
            for (int k = 0; k < nv; k++) if (items[vi[k]].st < 0 && !ps.defs[items[vi[k]].ph].unstressed) any_free = true;
            if (!any_free) target = -1;
            if (unstressed_word) {
                if ((nv > 1 || wd.u_n) && target >= 0) {
                    int t = wd.u_n ? std::min(wd.u_n, nv) - 1 : target;
                    if (items[vi[t]].st < 0) items[vi[t]].st = 2;
                }
            } else if (target >= 0) { items[vi[target]].st = 4; has_primary = true; }
        } else if (unstressed_word && has_primary) {
            for (auto& it : items) if (it.st == 4) it.st = nv > 1 ? 2 : 1;
            has_primary = false;
        }
        // secondary stress before the primary: alternate from the first syllable, never next to a stressed one
        int prim = -1;
        for (int k = 0; k < nv; k++) if (items[vi[k]].st >= 4) { prim = k; break; }
        if (prim > 0 && opt.secondary_first_only) {
            if (items[vi[0]].st < 0 && !ps.defs[items[vi[0]].ph].unstressed) items[vi[0]].st = 2;
        } else if (prim > 0 && opt.secondary_from_start) {
            for (int k = 0; k < prim - 1; k++) {
                if (items[vi[k]].st >= 0) continue;
                if (ps.defs[items[vi[k]].ph].unstressed) continue;
                bool prev_st = k > 0 && items[vi[k - 1]].st >= 2;
                bool next_st = items[vi[k + 1]].st >= 2;
                if (!prev_st && !next_st) items[vi[k]].st = 2;
            }
        }
        if (prim >= 0 && opt.post_secondary) {
            for (int k = prim + 1; k < nv; k++) {
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                if (items[vi[k - 1]].st >= 2) continue;
                items[vi[k]].st = 2;
            }
        }
        if (prim >= 0 && opt.secondary_from_start && !opt.post_secondary) {
            for (int k = prim + 2; k < nv - 1; k += 2) {
                if (items[vi[k]].st >= 0 || ps.defs[items[vi[k]].ph].unstressed) continue;
                if (items[vi[k - 1]].st >= 2 || items[vi[k + 1]].st >= 2) continue;
                items[vi[k]].st = 2;
            }
        }
        for (int k = 0; k < nv; k++) {
            int& s = items[vi[k]].st;
            if (s < 0) s = (prim >= 0 && k > prim) ? 0 : (opt.dim_non_initial && k > 0 && items[vi[k - 1]].st >= 2) ? 0 : 1;
        }
        if (opt.italian_alt && wd.alt) {
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
        int t;
        if (opt.stress_rule == 1) {
            // first syllable; after a marked (secondary) syllable, skip one more: g,e:o:f'y:zIk
            bool marked = false; int skip = 0;
            for (int k = 0; k < nv; k++) {
                if (st(k) >= 2) { marked = true; skip = 1; continue; }
                if (st(k) >= 0 || unst(k)) continue;
                if (marked && skip > 0) { skip--; continue; }
                return k;
            }
            for (int k = 0; k < nv; k++) if (st(k) < 0 && !unst(k)) return k;
            t = 0;
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
        long j = (long)i + d;
        if (j < 0 || j >= (long)L.size()) return nullptr;
        if (same_word && L[j].word != L[i].word) return nullptr;
        return &L[j];
    }
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
        if (a == "isPalatal") return d.palatal;
        if (a == "isRhotic") return d.rhotic;
        if (a == "isFlag1") return d.flag1 && !(is_prev && it->word != L[self].word);
        if (a == "isFlag2") return d.flag2;
        if (a == "isVoiced") return d.type == T_VOWEL || d.voiced;
        if (a == "isWordStart") return it->word_start;
        if (a == "notWordStart") return !it->word_start;
        if (a == "isWordEnd") return it->word_end;
        if (a == "isStressed") return it->stress >= 4;
        if (a == "isNotStressed") return it->stress < 4;
        if (a == "isUnstressed") return it->stress <= 1;
        if (a == "isDiminished") return it->stress == 0;
        if (a == "isMaxStress") return it->stress >= 4;
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
                    for (size_t j = i + 1; j < L.size() && L[j].word == L[i].word; j++) if (ps.defs[L[j].ph].type == T_VOWEL) { it = &L[j]; break; }
                    break;
                case C_PREVVOWEL:
                    for (long j = (long)i - 1; j >= 0 && L[j].word == L[i].word; j--) if (ps.defs[L[j].ph].type == T_VOWEL) { it = &L[j]; prev = true; break; }
                    break;
                default: break;
            }
            bool v = c.which == C_FALSE ? false : attr_true(c.attr, L, it, prev, i);
            if (c.neg) v = !v;
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
                case S_CHANGE: { if (s.arg == "NULL") return -3; int p = ps.find(s.arg); if (p >= 0) return p; break; }
                case S_CH_NOTSTRESSED: if (L[i].stress < 4 && !(L[i].number && opt.final_spanish)) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_UNSTRESSED: if (L[i].stress <= 1) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_DIMINISHED: if (L[i].stress == 0) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_CH_STRESSED: if (L[i].stress >= 4) { int p = ps.find(s.arg); if (p >= 0) return p; } break;
                case S_APPEND: {
                    const PhItem* n = rel(L, i, 1, false);
                    if (n && ps.defs[n->ph].type == T_VOWEL) append = ps.find(s.arg);
                    break;
                }
                case S_CALL: {
                    if (depth > 4) break;
                    size_t sl = s.arg.find('/');
                    if (sl == std::string::npos) break;
                    int tb = ps.find_table(s.arg.substr(0, sl));
                    int p = tb >= 0 ? ps.find(s.arg.substr(sl + 1), tb) : -1;
                    if (p >= 0) {
                        if (ipa && ps.defs[p].has_ipa) *ipa = ps.defs[p].ipa;
                        int r = run(ps.defs[p].prog, L, i, append, ipa, depth + 1, ret);
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

    // ---------- clause
    std::string phonemize_clause(const std::string& text_utf8) override {
        std::u32string t = dec(text_utf8);
        // words: letters / digits runs; other characters break words (apostrophes kept inside words)
        std::vector<Word> words;
        std::u32string cur; bool cap = false, allc = true, hyph = false;
        auto flush = [&]() {
            if (cur.empty()) return;
            std::u32string orig_cur = cur;
            Word w; w.capital = cap; w.allcaps = allc && cur.size() > 1;
            w.join_prev = hyph; hyph = false;
            for (auto& c : cur) { c = lower(c); if (c >= 0x80 && !known_letter(c)) c = base_letter(c); }
            for (auto& r : replace) {
                size_t p = 0;
                while ((p = cur.find(r.first, p)) != std::u32string::npos) { cur.replace(p, r.first.size(), r.second); p += r.second.size(); }
            }
            w.text = cur;
            w.orig = orig_cur;
            words.push_back(w);
            cur.clear(); cap = false; allc = true;
        };
        for (size_t i = 0; i < t.size(); i++) {
            char32_t c = t[i];
            bool letter = is_alpha(c) || (c >= 0x300 && c <= 0x36F) || (c >= 0x900 && c <= 0x97F && c != 0x964 && c != 0x965);
            if (letter) {
                if (!cur.empty() && is_digit(cur.back())) flush();
                if (!cur.empty() && is_upper(c) && !is_upper(cur.back()) && is_alpha(cur.back())) flush();
                if (cur.empty()) cap = is_upper(c);
                if (!is_upper(c)) allc = false;
                cur += c;
            } else if (is_digit(c)) {
                if (!cur.empty() && !is_digit(cur.back())) flush();
                cur += c;
            } else if ((c == '\'' || c == 0x2019) && !cur.empty() && i + 1 < t.size() && is_alpha(t[i + 1])) {
                cur += '\'';
                if (opt.apostrophe_split) {
                    std::u32string whole = cur;   // keep list entries such as "l'ancora" whole
                    size_t j = i + 1;
                    while (j < t.size() && is_alpha(t[j])) whole += t[j++];
                    for (auto& ch : whole) ch = lower(ch);
                    if (!list.count(whole)) { flush(); hyph = true; }
                }
            } else if (c == '-' && !cur.empty() && i + 1 < t.size() && is_alpha(t[i + 1])) {
                bool one = cur.size() == 1 || opt.hyphen_join;
                flush(); hyph = one;
            } else if (c == '.' && (opt.num_swap || opt.italian_alt) && !cur.empty() && is_digit(cur.back()) && i + 3 < t.size() + 0 &&
                       i + 3 < t.size() && is_digit(t[i + 1]) && is_digit(t[i + 2]) && is_digit(t[i + 3]) &&
                       (i + 4 >= t.size() || !is_digit(t[i + 4]))) {
                // German thousands separator: "1.000"
            } else if ((c == '.' || c == ',') && !cur.empty() && is_digit(cur.back()) && i + 1 < t.size() && is_digit(t[i + 1])) {
                flush();
                std::u32string k = U"_"; k += c;
                if (list.count(k)) { Word pw; pw.text = U"\x02"; pw.no_stress = true; encode_ph(ph_of(enc(k)), pw.ph); words.push_back(pw); }
            } else if (!is_space(c) && c != '.' && c != ',' && c != '\'' && c != '-' && list.count(std::u32string{c})) {
                // a symbol the word list names ("%", "$", "|"): spoken as a word
                flush(); hyph = false;
                cur = std::u32string{c};
                flush();
            } else { flush(); hyph = false; }
        }
        flush();
        // multi-word list entries (two words)
        std::vector<PhItem> L;
        int word_counter = 0;
        size_t spell_start = 0;
        foreign_.clear();
        int pause = ps.find("_");
        L.push_back({pause < 0 ? 0 : pause, -1, true, true, -1});
        for (size_t wi = 0; wi < words.size(); wi++) {
            Word& w = words[wi];
            std::u32string next = wi + 1 < words.size() ? words[wi + 1].text : U"";
            if (wi + 1 < words.size()) {
                std::u32string two = w.text + U" " + next;
                auto it = list.find(two);
                if (it != list.end()) {
                    const ListEntry& e = it->second.back();
                    if (!e.phonemes.empty()) {
                        std::string a = e.phonemes, b;
                        size_t bar = a.find("||");
                        if (bar != std::string::npos) { b = a.substr(bar + 2); a = a.substr(0, bar); }
                        w.ph.clear(); encode_ph(a, w.ph);
                        w.unstressed = e.u; w.u_plus = e.u_plus; w.stress_pos = e.stress_pos;
                        words[wi + 1].ph.clear();
                        if (!b.empty()) encode_ph(b, words[wi + 1].ph);
                        words[wi + 1].text = U"\x01";   // translated
                    }
                }
            }
            if (w.text == U"\x02") w.text = U"\x01";
            if (w.text != U"\x01" || !w.ph.empty()) {
                if (wi > 0) { w.expect_noun = words[wi - 1].nounf; w.expect_verb = words[wi - 1].verbf; }
                if (w.ph.empty() && w.text != U"\x01") translate_word(w, next, wi == 0, wi + 1 == words.size(), wi > 0 ? words[wi - 1].text : U"");
            }
            if (!w.switch_lang.empty()) {
                std::string ipa;
                if (EspeakLang* o = other_lang(w.switch_lang)) ipa = strip_ws(o->phonemize_clause(enc(w.orig)));
                if (ipa.empty()) {   // no English data or nothing came out: this language's own rules
                    ignore_switch_ = true;
                    w.switch_lang.clear();
                    translate_word(w, next, wi == 0, wi + 1 == words.size(), wi > 0 ? words[wi - 1].text : U"");
                    ignore_switch_ = false;
                }
                if (!ipa.empty()) {
                    foreign_.push_back(ipa);
                    PhItem it{pause < 0 ? 0 : pause, -1, true, true, ++word_counter};
                    it.foreign = (int)foreign_.size() - 1;
                    it.join = w.join_prev;
                    L.push_back(it);
                    continue;
                }
            }
            // "||" inside a word's phonemes splits it into words; -10 starts a new stress domain without a space
            std::vector<std::vector<int>> parts(1);
            std::vector<int> pjoin(1, w.join_prev ? 1 : 0);
            int wordend = ps.find("||");
            for (int p : w.ph) {
                if (p == wordend) { parts.emplace_back(); pjoin.push_back(0); }
                else if (p == -10) { parts.emplace_back(); pjoin.push_back(1); }
                else if (p == -11) { parts.emplace_back(); pjoin.push_back(2); }
                else if (p == -12) { parts.emplace_back(); pjoin.push_back(3); }
                else parts.back().push_back(p);
            }
            for (size_t pi = 0; pi < parts.size(); pi++) {
                Word pw = w; pw.ph = parts[pi];
                if (pi + 1 < parts.size() && pjoin[pi + 1] == 3) { pw.no_default = true; pw.stress_pos = 0; }
                if (pjoin[pi] == 3) pw.stress_pos = w.rest_stress_pos;
                if (pi > 0 && !pjoin[pi]) { pw.stress_pos = 0; }
                std::vector<std::pair<int, int>> st;
                word_stress(pw, wi + 1 == words.size(), st);
                if (st.empty()) continue;
                bool prev_primary = false;
                for (size_t q = spell_start; q < L.size(); q++) if (L[q].stress >= 4) prev_primary = true;
                if (pi == 0 || pjoin[pi] == 0 || (pjoin[pi] == 3 && opt.post_secondary && !prev_primary)) ++word_counter;
                int widx = word_counter;
                if (opt.vowel_pause && (pi == 0 || pjoin[pi] == 0) && L.size() > 1 && !st.empty() && is_vowel_ph(st[0].first)) {
                    PhItem pz{pause < 0 ? 0 : pause, -1, false, false, word_counter * 1000 + 999};
                    L.push_back(pz);
                }
                size_t first = L.size();
                if (pjoin[pi] == 2) for (size_t q = spell_start; q < L.size(); q++) if (L[q].stress >= 4) L[q].stress = 2;
                if (pjoin[pi] == 3) {
                    bool had = false;
                    for (size_t q = spell_start; q < L.size(); q++) if (L[q].stress >= 4) had = true;
                    if (had) for (auto& x : st) if (x.second >= 4) x.second = 2;
                }
                if (pjoin[pi] != 2 && pjoin[pi] != 3) spell_start = L.size();
                for (auto& x : st) {
                    if (ps.defs[x.first].type == T_PAUSE) continue;
                    PhItem it{x.first, x.second, false, false, widx};
                    it.number = w.number;
                    L.push_back(it);
                }
                if (L.size() > first) { L[first].word_start = true; L[first].join = pi > 0 ? pjoin[pi] != 0 : w.join_prev; L.back().word_end = true; }
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
        // phase 1: phoneme changes
        for (size_t i = 0; i < L.size(); i++) {
            for (int guard = 0; guard < 4; guard++) {
                int app = -1; bool ret = false;
                int r = run(ps.defs[L[i].ph].prog, L, i, app, nullptr, 0, ret);
                if (app >= 0) {
                    PhItem x{app, -1, false, false, L[i].word};
                    L.insert(L.begin() + i + 1, x);
                }
                if (r == -3) { L.erase(L.begin() + i); i--; break; }
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
        int lastw = -1;
        for (size_t i = 0; i < L.size(); i++) {
            const PhDef& d = ps.defs[L[i].ph];
            if (L[i].foreign >= 0) {
                if (lastw >= 0 && !L[i].join) out += ' ';
                out += foreign_[L[i].foreign];
                lastw = L[i].word;
                continue;
            }
            if (d.type == T_PAUSE || d.type == T_STRESS || (d.type == T_VIRTUAL && d.name != ":")) continue;
            if (L[i].word != lastw) { if (lastw >= 0 && !L[i].join) out += ' '; lastw = L[i].word; }
            if (d.type == T_VOWEL && L[i].stress >= 2) out += L[i].stress >= 4 ? "\xcb\x88" : "\xcb\x8c";
            std::string ipa = d.ipa;
            if (d.name == ":" && i > 0) {
                const PhDef& pd = ps.defs[L[i - 1].ph];
                if (pd.type == T_NASAL || pd.type == T_LIQUID || pd.type == T_FRIC) { out += last_ipa; continue; }
            }
            int app = -1; bool ret = false;
            std::string cond_ipa = "\x01";
            run(d.prog, L, i, app, &cond_ipa, 0, ret);
            if (cond_ipa != "\x01") ipa = cond_ipa;
            // tie between the letters of one phoneme (phonemizer tie='^')
            std::u32string u = dec(ipa);
            std::string o;
            for (size_t k = 0; k < u.size(); k++) {
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
    {"pt-br", "pt", "pt", {2}, 2, false},
    {"ru", "ru", "ru", {}, 2, false},
    {"hi", "hi", "hi", {}, 2, false},
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
    e->opt.post_secondary = vc->german; e->opt.final_devoice = vc->german; e->opt.num_swap = vc->german;
    e->opt.hyphen_join = vc->german || std::string(vc->voice) == "it";
    e->data_dir_ = dir;
    if (!e->ps.load(dir + "/phsource", vc->table, err)) return nullptr;
    std::string d = vc->dict;
    if (!e->load_rules(dir + "/" + d + "_rules", err)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_list", err, true)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_listx", err, false)) return nullptr;
    if (!e->load_list(dir + "/" + d + "_extra", err, false)) return nullptr;
    return e;
}

}  // namespace kt_fe
