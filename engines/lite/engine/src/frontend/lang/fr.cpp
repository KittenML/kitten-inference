// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/fr.h"

#include <cctype>
#include <fstream>
#include <vector>

#include "frontend/cephonemizer/phonemizer.h"

namespace kt_fe {
namespace {

void replace_all(std::string& s, const std::string& a, const std::string& b) {
    for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) s.replace(p, a.size(), b);
}

bool starts_with_vowel_sound(const std::string& s, size_t p) {
    static const char* const v[] = {"a", "e", "i", "o", "u", "ɪ", "ɛ", "ɐ", "ə", "ʌ", "ɒ", "ʊ", "ɑ", "ɔ", "ɜ", "æ"};
    for (const char* x : v) if (s.compare(p, std::char_traits<char>::length(x), x) == 0) return true;
    return false;
}

// CEPhonemizer's short "ɔ" (American spelling of the vowel of "soft", "Walt") is "ɒ" in the British table.
void replace_bare_open_o(std::string& o) {
    const std::string k = "ɔ";
    for (size_t p = 0; (p = o.find(k, p)) != std::string::npos; p += k.size()) {
        const size_t q = p + k.size();
        if (o.compare(q, 2, "ː") == 0 || o.compare(q, 2, "ɪ") == 0) continue;
        o.replace(p, k.size(), "ɒ");
    }
}

// The English phoneme table (phsource/ph_english: ChangeIfDiminished) weakens some vowels in the weakest syllables
// of a word: a -> ɐ, ɛ -> ɪ (ə before n), ɒ -> ə, ɜː -> ə
// ("Pattavee" pˈatɐvˌiː, "Cuddeback" kˈʌdɪbˌak). CEPhonemizer does not do this in its British mode.
void reduce_diminished(std::string& o) {
    struct Unit { std::string s; bool vowel; bool stressed; };
    static const char* const vowels[] = {"a", "e", "i", "o", "u", "ɪ", "ɛ", "ɐ", "ə", "ʌ", "ɒ", "ʊ", "ɑ", "ɔ", "ɜ", "æ"};
    auto vowel_at = [&](size_t p) -> size_t {
        for (const char* v : vowels) { const size_t n = std::char_traits<char>::length(v); if (o.compare(p, n, v) == 0) return n; }
        return 0;
    };
    std::vector<Unit> u;
    bool mark = false;
    for (size_t p = 0; p < o.size();) {
        if (o.compare(p, 2, "ˈ") == 0 || o.compare(p, 2, "ˌ") == 0) { u.push_back({o.substr(p, 2), false, false}); mark = true; p += 2; continue; }
        size_t n = vowel_at(p);
        if (n) {   // a vowel run (with length marks) is one syllable nucleus
            size_t q = p;
            while (q < o.size()) {
                size_t m = vowel_at(q);
                if (!m && o.compare(q, 2, "ː") == 0) m = 2;
                if (!m) break;
                q += m;
            }
            u.push_back({o.substr(p, q - p), true, mark});
            mark = false; p = q; continue;
        }
        size_t len = 1;
        const unsigned char c = (unsigned char)o[p];
        if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        u.push_back({o.substr(p, len), false, false});
        p += len;
    }
    std::vector<size_t> vi;
    for (size_t i = 0; i < u.size(); i++) if (u[i].vowel) vi.push_back(i);
    // Which syllables are weakened (observed on the reference): never the first or the last one; a syllable without
    // stress after a stressed one or after a plain unstressed one (they alternate), unless only an unstressed last
    // syllable follows ("Netanyahou" nˈɛtɐnɪˌahuː, "Uthappa" ˈʌθapə).
    bool prev_dim = false;
    for (size_t k = 1; k + 1 < vi.size(); k++) {
        Unit& x = u[vi[k]];
        if (x.stressed) { prev_dim = false; continue; }
        const bool only_weak_last = k + 2 == vi.size() && !u[vi[k + 1]].stressed;
        const bool dim = (u[vi[k - 1]].stressed || !prev_dim) && !only_weak_last;
        prev_dim = dim;
        if (!dim) continue;
        const bool n_next = vi[k] + 1 < u.size() && u[vi[k] + 1].s == "n";
        if (x.s == "a") x.s = "ɐ";
        else if (x.s == "ɛ") x.s = n_next ? "ə" : "ɪ";
        else if (x.s == "ɒ") x.s = "ə";
        else if (x.s == "ɜː") x.s = "ə";
    }
    o.clear();
    for (auto& x : u) o += x.s;
}

class Impl : public FrEnglishWords {
public:
    Impl(const std::string& rules, const std::string& list) : ce_(rules, list, "en-gb") {}
    bool ok() const { return ce_.isLoaded(); }

    std::string word(const std::string& w) const override {
        if (w.empty()) return "";
        std::string lower;
        for (unsigned char c : w) {
            if (c >= 0x80) return "";   // é, ï ...: the generic reader has the English rules for those letters
            lower += (char)std::tolower(c);
        }
        // l'overscan: the elided article is spelled as a letter, then the word
        if (lower.size() > 2 && lower[0] == 'l' && lower[1] == '\'') {
            std::string rest = word(w.substr(2));
            return rest.empty() ? "" : "ˈɛl" + rest;
        }
        // words whose stress the English list ties to their place in the clause (of, the, not, be): the generic reader
        // follows that; CEPhonemizer would stress a word standing alone
        if (ce_.clauseEndFlags(lower) & (1 | 2 | 4 | 8)) return "";
        // a word in capitals is read as a word here ("INSEE", "IKEA"), not letter by letter
        bool all_caps = w.size() >= 2;
        for (unsigned char c : w) if (std::isalpha(c) && !std::isupper(c)) all_caps = false;
        std::string s = ce_.phonemizeText(all_caps ? lower : w);
        std::string o;
        for (char c : s) if (c != ' ' && c != '\n' && c != '\t') o += c;
        if (o.empty()) return "";
        // British spelling of the phonemes, as espeak-ng's "en" table prints them
        replace_all(o, "ɚ", "ə");
        replace_all(o, "ᵻ", "ɪ");
        replace_all(o, "r", "ɹ");
        for (const char* d : {"oə", "ɔə"}) {
            const size_t n = std::char_traits<char>::length(d);
            for (size_t p = 0; (p = o.find(d, p)) != std::string::npos;) {
                const bool vowel_next = p + n < o.size() && starts_with_vowel_sound(o, p + n);
                const std::string r = vowel_next ? "ɔːɹ" : "ɔː";
                o.replace(p, n, r);
                p += r.size();
            }
        }
        replace_bare_open_o(o);
        reduce_diminished(o);
        // ties inside diphthongs and affricates (the package asks espeak for tie='^')
        static const char* const tie[][2] = {{"aɪ", "a^ɪ"}, {"aʊ", "a^ʊ"}, {"eɪ", "e^ɪ"}, {"əʊ", "ə^ʊ"}, {"oʊ", "o^ʊ"},
                                             {"ɔɪ", "ɔ^ɪ"}, {"dʒ", "d^ʒ"}, {"tʃ", "t^ʃ"}};
        for (auto& t : tie) replace_all(o, t[0], t[1]);
        return o;
    }

private:
    IPAPhonemizer ce_;
};

}  // namespace

std::unique_ptr<FrEnglishWords> FrEnglishWords::load(const std::string& data_dir) {
    const std::string rules = data_dir + "/en_rules", list = data_dir + "/en_list";
    if (!std::ifstream(rules).good() || !std::ifstream(list).good()) return nullptr;
    auto p = std::make_unique<Impl>(rules, list);
    if (!p->ok()) return nullptr;
    return p;
}

}  // namespace kt_fe
