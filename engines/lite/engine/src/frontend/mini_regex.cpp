// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// See mini_regex.hpp.
#include "mini_regex.hpp"

#include <algorithm>
#include <stdexcept>

namespace kt08fe {

// ---- Unicode helpers ------------------------------------------------------------------------------------------------
// Python's str methods use the full Unicode database. A phone library should not carry that, so these cover the
// blocks a text-to-speech app meets in practice (Latin, IPA, Greek, Cyrillic, common Indic/Arabic/CJK letters,
// full-width forms). Outside them a code point is treated as a symbol (not \w), which only matters for text the
// English phonemiser cannot read anyway.

namespace {
struct Range { char32_t a, b; };
// Decimal digit blocks (each starts at its digit zero).
const char32_t kDigitZeros[] = {0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6,
                                0xD66, 0xE50, 0xED0, 0xF20, 0x1040, 0x17E0, 0x1810, 0xFF10};

bool in_ranges(char32_t c, const Range* r, size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < r[mid].a) hi = mid;
        else if (c > r[mid].b) lo = mid + 1;
        else return true;
    }
    return false;
}

// Non-ASCII code points for which str.isalnum() is True (sorted, non-overlapping).
const Range kAlnum[] = {
    {0xAA, 0xAA}, {0xB2, 0xB3}, {0xB5, 0xB5}, {0xB9, 0xBA}, {0xBC, 0xBE}, {0xC0, 0xD6}, {0xD8, 0xF6}, {0xF8, 0x2C1},
    {0x2C6, 0x2D1}, {0x2E0, 0x2E4}, {0x2EC, 0x2EC}, {0x2EE, 0x2EE}, {0x370, 0x374}, {0x376, 0x377}, {0x37A, 0x37D},
    {0x37F, 0x37F}, {0x386, 0x386}, {0x388, 0x38A}, {0x38C, 0x38C}, {0x38E, 0x3A1}, {0x3A3, 0x3F5}, {0x3F7, 0x481},
    {0x48A, 0x52F}, {0x531, 0x556}, {0x559, 0x559}, {0x560, 0x588}, {0x5D0, 0x5EA}, {0x5EF, 0x5F2}, {0x620, 0x64A},
    {0x660, 0x669}, {0x66E, 0x66F}, {0x671, 0x6D3}, {0x6D5, 0x6D5}, {0x6E5, 0x6E6}, {0x6EE, 0x6FC}, {0x6FF, 0x6FF},
    {0x7C0, 0x7EA}, {0x904, 0x939}, {0x93D, 0x93D}, {0x950, 0x950}, {0x958, 0x961}, {0x966, 0x96F}, {0x971, 0x980},
    {0x985, 0x9B9}, {0x9E6, 0x9F1}, {0xA05, 0xA39}, {0xA66, 0xA6F}, {0xA85, 0xAB9}, {0xAE6, 0xAEF}, {0xB05, 0xB39},
    {0xB66, 0xB6F}, {0xB85, 0xBB9}, {0xBE6, 0xBF2}, {0xC05, 0xC39}, {0xC66, 0xC6F}, {0xC85, 0xCB9}, {0xCE6, 0xCEF},
    {0xD05, 0xD3A}, {0xD66, 0xD78}, {0xE01, 0xE30}, {0xE32, 0xE33}, {0xE40, 0xE46}, {0xE50, 0xE59}, {0xE81, 0xEB0},
    {0xED0, 0xED9}, {0xF20, 0xF33}, {0x1000, 0x102A}, {0x1040, 0x1049}, {0x10A0, 0x10FF}, {0x1100, 0x11FF},
    {0x1780, 0x17B3}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x1E00, 0x1FBC}, {0x1FBE, 0x1FBE}, {0x1FC2, 0x1FCC},
    {0x1FD0, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FFC}, {0x2070, 0x2071}, {0x2074, 0x2079}, {0x207F, 0x2089},
    {0x2090, 0x209C}, {0x2102, 0x2102}, {0x2107, 0x2107}, {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D},
    {0x2124, 0x2124}, {0x2126, 0x2126}, {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2139}, {0x2150, 0x2189},
    {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2C00, 0x2CE4}, {0x3005, 0x3007}, {0x3021, 0x3029}, {0x3041, 0x3096},
    {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF}, {0x3105, 0x312F}, {0x3131, 0x318E}, {0x3400, 0x4DBF},
    {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF}, {0xFB00, 0xFB06}, {0xFF10, 0xFF19},
    {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A}, {0xFF66, 0xFFDC},
};
}  // namespace

int cp_digit_value(char32_t c) {
    if (c >= '0' && c <= '9') return int(c - '0');
    if (c < 0x660) return -1;
    for (char32_t z : kDigitZeros)
        if (c >= z && c <= z + 9) return int(c - z);
    return -1;
}
bool cp_is_digit(char32_t c) { return cp_digit_value(c) >= 0; }

bool cp_is_word(char32_t c) {
    if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    return in_ranges(c, kAlnum, sizeof(kAlnum) / sizeof(kAlnum[0]));
}

bool cp_is_space(char32_t c) {
    if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F);
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

char32_t cp_lower(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x137) return c | 1;
    if (c >= 0x139 && c <= 0x148) return (c & 1) ? c + 1 : c;
    if (c >= 0x14A && c <= 0x177) return c | 1;
    if (c == 0x178) return 0xFF;
    if (c >= 0x179 && c <= 0x17E) return (c & 1) ? c + 1 : c;
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c == 0x386) return 0x3AC;
    if (c >= 0x388 && c <= 0x38A) return c + 37;
    if (c == 0x38C) return 0x3CC;
    if (c == 0x38E || c == 0x38F) return c + 63;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    if ((c >= 0x460 && c <= 0x481) || (c >= 0x48A && c <= 0x4BF) || (c >= 0x4D0 && c <= 0x52F)) return c | 1;
    if (c >= 0x1E00 && c <= 0x1E95) return c | 1;
    if (c >= 0x1EA0 && c <= 0x1EFF) return c | 1;
    if (c >= 0xFF21 && c <= 0xFF3A) return c + 32;
    return c;
}

char32_t cp_upper(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') ? c - 32 : c;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c == 0xFF) return 0x178;
    if (c >= 0x100 && c <= 0x137) return c & ~(char32_t)1;
    if (c >= 0x14A && c <= 0x177) return c & ~(char32_t)1;
    if (c >= 0x3B1 && c <= 0x3CB && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    if (c >= 0xFF41 && c <= 0xFF5A) return c - 32;
    return c;
}

bool cp_is_upper(char32_t c) { return cp_lower(c) != c; }
bool cp_is_alpha(char32_t c) { return cp_is_word(c) && c != '_' && !cp_is_digit(c); }

std::u32string utf8_to_u32(const std::string& s, bool* ok) {
    std::u32string out;
    out.reserve(s.size());
    bool good = true;
    for (size_t i = 0; i < s.size();) {
        unsigned char b = (unsigned char)s[i];
        char32_t c;
        int n;
        if (b < 0x80) { c = b; n = 1; }
        else if ((b >> 5) == 6) { c = b & 0x1F; n = 2; }
        else if ((b >> 4) == 14) { c = b & 0x0F; n = 3; }
        else if ((b >> 3) == 30) { c = b & 0x07; n = 4; }
        else { good = false; ++i; out.push_back(0xFFFD); continue; }
        if (i + n > s.size()) { good = false; out.push_back(0xFFFD); break; }
        bool bad = false;
        for (int k = 1; k < n; ++k) {
            unsigned char cb = (unsigned char)s[i + k];
            if ((cb & 0xC0) != 0x80) { bad = true; break; }
            c = (c << 6) | (cb & 0x3F);
        }
        if (bad) { good = false; out.push_back(0xFFFD); ++i; continue; }
        out.push_back(c);
        i += n;
    }
    if (ok) *ok = good;
    return out;
}

void append_utf8(std::string& o, char32_t c) {
    if (c < 0x80) o.push_back((char)c);
    else if (c < 0x800) { o.push_back((char)(0xC0 | (c >> 6))); o.push_back((char)(0x80 | (c & 0x3F))); }
    else if (c < 0x10000) {
        o.push_back((char)(0xE0 | (c >> 12))); o.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
        o.push_back((char)(0x80 | (c & 0x3F)));
    } else {
        o.push_back((char)(0xF0 | (c >> 18))); o.push_back((char)(0x80 | ((c >> 12) & 0x3F)));
        o.push_back((char)(0x80 | ((c >> 6) & 0x3F))); o.push_back((char)(0x80 | (c & 0x3F)));
    }
}

std::string u32_to_utf8(const std::u32string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char32_t c : s) append_utf8(o, c);
    return o;
}

// ---- parser ---------------------------------------------------------------------------------------------------------
namespace {
[[noreturn]] void bad(const char* what) { throw std::runtime_error(std::string("mini_regex: ") + what); }
unsigned short class_bit(char32_t e) {
    switch (e) {
        case 'd': return 1; case 'w': return 2; case 's': return 4;
        case 'D': return 8; case 'W': return 16; case 'S': return 32;
        default: return 0;
    }
}
}  // namespace

Regex::Regex(const std::u32string& p, bool ignore_case) : icase_(ignore_case) {
    size_t i = 0;
    top_alts_ = parse_alts(p, i);
    if (i != p.size()) bad("unbalanced ')'");
    if (ngroups_ > kMaxGroups) bad("too many groups");
}

std::vector<Regex::Seq> Regex::parse_alts(const std::u32string& p, size_t& i) {
    std::vector<Seq> alts(1);
    while (i < p.size() && p[i] != ')') {
        if (p[i] == '|') { alts.emplace_back(); ++i; continue; }
        int ni = parse_atom(p, i);
        if (i < p.size()) {
            char32_t q = p[i];
            int mn = -1, mx = -1;
            if (q == '?') { mn = 0; mx = 1; ++i; }
            else if (q == '*') { mn = 0; mx = 1 << 30; ++i; }
            else if (q == '+') { mn = 1; mx = 1 << 30; ++i; }
            else if (q == '{') {
                size_t j = i + 1; int a = 0, b; bool da = false;
                while (j < p.size() && p[j] >= '0' && p[j] <= '9') { a = a * 10 + int(p[j] - '0'); ++j; da = true; }
                if (!da) bad("bad {");
                if (p[j] == '}') { b = a; }
                else if (p[j] == ',') {
                    ++j;
                    if (p[j] == '}') b = 1 << 30;
                    else { b = 0; while (j < p.size() && p[j] >= '0' && p[j] <= '9') { b = b * 10 + int(p[j] - '0'); ++j; } }
                    if (p[j] != '}') bad("bad {m,n}");
                } else bad("bad {");
                mn = a; mx = b; i = j + 1;
            }
            if (mn >= 0) {
                if (i < p.size() && (p[i] == '?' || p[i] == '+')) bad("lazy/possessive quantifiers unsupported");
                nodes_[ni].qmin = mn;
                nodes_[ni].qmax = mx;
            }
        }
        alts.back().push_back(ni);
    }
    return alts;
}

int Regex::parse_atom(const std::u32string& p, size_t& i) {
    Node n;
    char32_t c = p[i];
    if (c == '(') {
        ++i;
        if (i < p.size() && p[i] == '?') {
            ++i;
            if (p[i] == ':') { n.kind = Node::GROUP; ++i; }
            else if (p[i] == '=' || p[i] == '!') { n.kind = Node::LOOK; n.ahead = true; n.look_neg = p[i] == '!'; ++i; }
            else if (p[i] == '<' && (p[i + 1] == '=' || p[i + 1] == '!')) {
                n.kind = Node::LOOK; n.ahead = false; n.look_neg = p[i + 1] == '!'; i += 2;
            } else bad("unsupported (? group");
        } else {
            n.kind = Node::GROUP;
            n.cap = ngroups_++;
        }
        n.alts = parse_alts(p, i);
        if (i >= p.size() || p[i] != ')') bad("missing ')'");
        ++i;
        if (n.kind == Node::LOOK && !n.ahead) {
            n.width = seq_width(n.alts[0]);
            for (auto& a : n.alts) if (seq_width(a) != n.width) bad("lookbehind must be fixed width");
        }
    } else if (c == '[') {
        n.kind = Node::CLASS;
        ++i;
        if (p[i] == '^') { n.negate = true; ++i; }
        bool first = true;
        while (i < p.size() && (p[i] != ']' || first)) {
            first = false;
            char32_t a = p[i];
            if (a == '\\') {
                char32_t e = p[i + 1];
                i += 2;
                if (unsigned short bit = class_bit(e)) { n.cls |= bit; continue; }
                a = e == 'n' ? U'\n' : e == 't' ? U'\t' : e;
            } else ++i;
            if (i + 1 < p.size() && p[i] == '-' && p[i + 1] != ']') {
                char32_t b = p[i + 1];
                i += 2;
                if (b == '\\') { b = p[i]; ++i; }
                n.ranges.push_back({a, b});
            } else n.ranges.push_back({a, a});
        }
        if (i >= p.size()) bad("missing ]");
        ++i;
    } else if (c == '\\') {
        char32_t e = p[i + 1];
        i += 2;
        if (e == 'b') n.kind = Node::WORDB;
        else if (e == 'B') n.kind = Node::NWORDB;
        else if (unsigned short bit = class_bit(e)) { n.kind = Node::CLASS; n.cls = bit; }
        else { n.kind = Node::CHAR; n.ch = e == 'n' ? U'\n' : e == 't' ? U'\t' : e; }
    } else if (c == '.') { n.kind = Node::ANY; ++i; }
    else if (c == '^') { n.kind = Node::BOL; ++i; }
    else if (c == '$') { n.kind = Node::EOL; ++i; }
    else { n.kind = Node::CHAR; n.ch = c; ++i; }
    nodes_.push_back(std::move(n));
    return (int)nodes_.size() - 1;
}

int Regex::seq_width(const Seq& s) const {
    int w = 0;
    for (int ni : s) {
        const Node& n = nodes_[ni];
        if (n.qmin != n.qmax) bad("variable width in lookbehind");
        switch (n.kind) {
            case Node::CHAR: case Node::CLASS: case Node::ANY: w += n.qmin; break;
            case Node::GROUP: w += n.qmin * seq_width(n.alts[0]); break;
            default: break;
        }
    }
    return w;
}

bool Regex::class_has(const Node& n, char32_t c) const {
    for (auto& r : n.ranges) if (c >= r.first && c <= r.second) return true;
    if (n.cls) {
        if ((n.cls & 1) && cp_is_digit(c)) return true;
        if ((n.cls & 2) && cp_is_word(c)) return true;
        if ((n.cls & 4) && cp_is_space(c)) return true;
        if ((n.cls & 8) && !cp_is_digit(c)) return true;
        if ((n.cls & 16) && !cp_is_word(c)) return true;
        if ((n.cls & 32) && !cp_is_space(c)) return true;
    }
    return false;
}

bool Regex::atom_one(const Node& n, char32_t c) const {
    switch (n.kind) {
        case Node::CHAR:
            return c == n.ch || (icase_ && cp_lower(c) == cp_lower(n.ch));
        case Node::ANY:
            return c != '\n';
        case Node::CLASS: {
            bool in = class_has(n, c);
            if (!in && icase_) {
                char32_t l = cp_lower(c), u = cp_upper(c);
                in = (l != c && class_has(n, l)) || (u != c && class_has(n, u));
            }
            return in != n.negate;
        }
        default:
            return false;
    }
}

// ---- matcher --------------------------------------------------------------------------------------------------------
struct Cont {
    enum K { SEQ, REP, GEND, ACCEPT } k;
    const Regex::Seq* seq = nullptr;
    size_t idx = 0;
    int node = -1, count = 0;
    size_t start = 0;  // REP: where this iteration started; GEND: group start
    size_t want = (size_t)-1;  // ACCEPT: required end (lookbehind)
    const Cont* next = nullptr;
};

struct Matcher {
    const Regex& re;
    const std::u32string& s;
    long* caps;
    size_t endpos = 0;

    bool is_word_at(size_t p) const { return p < s.size() && cp_is_word(s[p]); }

    bool run(const Cont* k, size_t pos) {
        switch (k->k) {
            case Cont::SEQ: return seq(*k->seq, k->idx, pos, k->next);
            case Cont::REP: {
                const auto& n = re.nodes_[k->node];
                if (pos == k->start && k->count > n.qmin) return run(k->next, pos);  // empty iteration: stop looping
                return rep(k->node, k->count, pos, k->next);
            }
            case Cont::GEND: {
                const auto& n = re.nodes_[k->node];
                long os = caps[2 * n.cap], oe = caps[2 * n.cap + 1];
                caps[2 * n.cap] = (long)k->start;
                caps[2 * n.cap + 1] = (long)pos;
                if (run(k->next, pos)) return true;
                caps[2 * n.cap] = os;
                caps[2 * n.cap + 1] = oe;
                return false;
            }
            case Cont::ACCEPT:
                if (k->want != (size_t)-1 && pos != k->want) return false;
                endpos = pos;
                return true;
        }
        return false;
    }

    bool rep(int ni, int count, size_t pos, const Cont* after) {
        const auto& n = re.nodes_[ni];
        if (count < n.qmax) {
            Cont r;
            r.k = Cont::REP; r.node = ni; r.count = count + 1; r.start = pos; r.next = after;
            if (once(ni, pos, &r)) return true;
        }
        if (count >= n.qmin) return run(after, pos);
        return false;
    }

    bool seq(const Regex::Seq& q, size_t i, size_t pos, const Cont* k) {
        if (i == q.size()) return run(k, pos);
        const int ni = q[i];
        const auto& n = re.nodes_[ni];
        const bool simple = n.kind == Regex::Node::CHAR || n.kind == Regex::Node::CLASS || n.kind == Regex::Node::ANY;
        if (simple) {
            size_t cnt = 0;
            const size_t mx = (size_t)n.qmax;
            while (cnt < mx && pos + cnt < s.size() && re.atom_one(n, s[pos + cnt])) ++cnt;
            if (cnt < (size_t)n.qmin) return false;
            for (size_t c = cnt + 1; c-- > (size_t)n.qmin;)
                if (seq(q, i + 1, pos + c, k)) return true;
            return false;
        }
        Cont after;
        after.k = Cont::SEQ; after.seq = &q; after.idx = i + 1; after.next = k;
        if (n.qmin == 1 && n.qmax == 1) return once(ni, pos, &after);
        return rep(ni, 0, pos, &after);
    }

    bool once(int ni, size_t pos, const Cont* k) {
        const auto& n = re.nodes_[ni];
        switch (n.kind) {
            case Regex::Node::CHAR: case Regex::Node::CLASS: case Regex::Node::ANY:
                return pos < s.size() && re.atom_one(n, s[pos]) && run(k, pos + 1);
            case Regex::Node::BOL: return pos == 0 && run(k, pos);
            case Regex::Node::EOL:
                return (pos == s.size() || (pos + 1 == s.size() && s[pos] == '\n')) && run(k, pos);
            case Regex::Node::WORDB: case Regex::Node::NWORDB: {
                bool b = (pos > 0 && is_word_at(pos - 1)) != is_word_at(pos);
                return (b == (n.kind == Regex::Node::WORDB)) && run(k, pos);
            }
            case Regex::Node::GROUP: {
                Cont g;
                if (n.cap >= 0) { g.k = Cont::GEND; g.node = ni; g.start = pos; g.next = k; }
                for (const auto& a : n.alts)
                    if (seq(a, 0, pos, n.cap >= 0 ? &g : k)) return true;
                return false;
            }
            case Regex::Node::LOOK: {
                bool found = false;
                const size_t save_end = endpos;
                if (n.ahead || pos >= (size_t)n.width) {
                    Cont acc;
                    acc.k = Cont::ACCEPT;
                    size_t from = pos;
                    if (!n.ahead) { from = pos - (size_t)n.width; acc.want = pos; }
                    for (const auto& a : n.alts)
                        if (seq(a, 0, from, &acc)) { found = true; break; }
                }
                endpos = save_end;
                return (found != n.look_neg) && run(k, pos);
            }
        }
        return false;
    }
};

bool Regex::search(const std::u32string& s, size_t from, Match& m) const {
    m.caps.fill(-1);
    Matcher mt{*this, s, m.caps.data()};
    Cont acc;
    acc.k = Cont::ACCEPT;
    for (size_t st = from; st <= s.size(); ++st) {
        for (const auto& a : top_alts_) {
            if (mt.seq(a, 0, st, &acc)) {
                m.start = st;
                m.end = mt.endpos;
                return true;
            }
            std::fill(m.caps.begin(), m.caps.end(), -1);
        }
    }
    return false;
}

}  // namespace kt08fe
