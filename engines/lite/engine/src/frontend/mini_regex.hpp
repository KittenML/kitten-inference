// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// mini_regex.hpp - a small backtracking regular-expression matcher with Python `re` semantics for the subset the
// Kitten TTS 0.8 text normaliser uses (tts08_text.cpp). It works on Unicode code points (std::u32string), like
// Python str patterns, so \w, \d, \s, \b and IGNORECASE behave as they do in the Python package.
//
// Supported: literals and escapes (\. \- \$ ...), classes [...] with ranges, negation and \d \w \s \S inside, . (no
// newline), groups (...) and (?:...), alternation |, greedy quantifiers ? * + {m} {m,} {m,n}, anchors ^ $ \b,
// lookahead (?=...) (?!...), fixed-width lookbehind (?<=...) (?<!...). Flag: ignore case.
// Patterns are compiled once (at load); matching allocates nothing once the per-pattern scratch is sized.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kt08fe {

// Python str predicates on one code point (Unicode-aware for the scripts a phone app is likely to meet; see the .cpp).
bool cp_is_word(char32_t c);   // re \w: str.isalnum() or '_'
bool cp_is_digit(char32_t c);  // re \d: Unicode decimal digit (common blocks)
int cp_digit_value(char32_t c);  // 0..9 for cp_is_digit, else -1
bool cp_is_space(char32_t c);  // re \s / str.isspace()
bool cp_is_upper(char32_t c);
bool cp_is_alpha(char32_t c);
char32_t cp_lower(char32_t c);  // simple lower-case mapping (str.lower() for the common blocks)
char32_t cp_upper(char32_t c);

std::u32string utf8_to_u32(const std::string& s, bool* ok = nullptr);
std::string u32_to_utf8(const std::u32string& s);
void append_utf8(std::string& out, char32_t c);

class Regex {
public:
    static constexpr int kMaxGroups = 8;
    struct Match {
        size_t start = 0, end = 0;
        std::array<long, 2 * kMaxGroups> caps{};  // 2 per group: start, end (-1 when the group did not take part)
        bool has(int g) const { return g == 0 || caps[2 * (g - 1)] >= 0; }
        std::u32string group(const std::u32string& s, int g) const {
            if (g == 0) return s.substr(start, end - start);
            long a = caps[2 * (g - 1)], b = caps[2 * (g - 1) + 1];
            return a < 0 ? std::u32string() : s.substr((size_t)a, (size_t)(b - a));
        }
    };

    Regex() = default;
    // Throws std::runtime_error on a syntax error (patterns are fixed in the library, so this is a programming error).
    Regex(const std::u32string& pattern, bool ignore_case = false);
    Regex(const char* pattern_utf8, bool ignore_case = false) : Regex(utf8_to_u32(pattern_utf8), ignore_case) {}

    // First match starting at or after `from` (re.search semantics from that position).
    bool search(const std::u32string& s, size_t from, Match& m) const;

    int groups() const { return ngroups_; }

    using Seq = std::vector<int>;  // node indices

private:
    friend struct Matcher;
    struct Node {
        enum Kind { CHAR, CLASS, ANY, GROUP, BOL, EOL, WORDB, NWORDB, LOOK } kind = CHAR;
        char32_t ch = 0;
        // CLASS
        bool negate = false;
        std::vector<std::pair<char32_t, char32_t>> ranges;
        unsigned short cls = 0;  // bit 1 \d, 2 \w, 4 \s, 8 \D, 16 \W, 32 \S
        // GROUP / LOOK
        std::vector<Seq> alts;
        int cap = -1;  // capture index (0-based) or -1
        bool ahead = true, look_neg = false;
        int width = 0;  // lookbehind width
        // quantifier
        int qmin = 1, qmax = 1;
    };
    std::vector<Node> nodes_;
    std::vector<Seq> top_alts_;
    int ngroups_ = 0;
    bool icase_ = false;

    // parser
    std::vector<Seq> parse_alts(const std::u32string& p, size_t& i);
    int parse_atom(const std::u32string& p, size_t& i);
    int seq_width(const Seq& s) const;
    bool class_has(const Node& n, char32_t c) const;
    bool atom_one(const Node& n, char32_t c) const;
};

}  // namespace kt08fe
