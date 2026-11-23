// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Russian parts of the espeak-ng data reader (see ru.h). Original implementation.
#include "frontend/lang/ru.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>

namespace kt_fe {
namespace ru {

uint8_t letter_code(char32_t c) {
    if (c >= 0x430 && c <= 0x44F) return (uint8_t)(c - 0x430 + 1);   // а..я -> 1..32
    if (c == 0x451) return 33;                                       // ё
    return 0;
}

int in_letter_group(char group, char32_t c) {
    // As listed in the header of the Russian rules file (ru_rules); Y also holds е and и (they soften the consonant
    // before them, as the rules' "(Y" contexts are used: т (Y -> t;).
    static const char32_t* const kA = U"аеёиоуыэюяәөұүі";
    static const char32_t* const kB = U"ьйчщ";
    static const char32_t* const kC = U"бвгджзйклмнпрстфхцчшщъьқңһ";
    static const char32_t* const kF = U"бвгдзйклмнпрстфхчщь";
    static const char32_t* const kG = U"бвгджз";
    static const char32_t* const kH = U"ъжцш";
    static const char32_t* const kY = U"ьюяёеи";
    bool cyr = (c >= 0x400 && c <= 0x4FF);
    if (!cyr) return -1;
    const char32_t* set = nullptr;
    switch (group) {
        case 'A': set = kA; break;
        case 'B': set = kB; break;
        case 'C': set = kC; break;
        case 'F': set = kF; break;
        case 'G': set = kG; break;
        case 'H': set = kH; break;
        case 'Y': set = kY; break;
        default: return 0;
    }
    for (const char32_t* p = set; *p; p++) if (*p == c) return 1;
    return 0;
}

int guess_stress(int nsyl, int end) {
    // Measured on espeak-ng 1.52.0 (the build the Python package uses) with made-up words of 1..16 syllables.
    static const int kVowel[16] = {1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11};
    static const int kStop[16] = {1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 11};
    static const int kCons[16] = {1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11};
    if (nsyl <= 0) return 0;
    const int* t = end == 0 ? kVowel : end == 1 ? kStop : kCons;
    if (nsyl > 16) return t[15] + (nsyl - 16);
    return t[nsyl - 1];
}

namespace {
// 0..999; feminine: one / two as "одна" / "две" (before тысяча)
std::string below_1000(int n, bool feminine, const std::function<std::string(const std::string&)>& ph) {
    std::string o;
    int h = n / 100, r = n % 100;
    if (h) o += h == 1 ? ph("_0C") : ph("_" + std::to_string(h) + "C");
    if (r) {
        if (!o.empty()) o += "||";
        auto unit = [&](int u) { return (feminine && (u == 1 || u == 2)) ? ph("_" + std::to_string(u) + "f") : ph("_" + std::to_string(u)); };
        if (r < 10) o += unit(r);
        else if (r < 20) o += ph("_" + std::to_string(r));
        else {
            o += ph("_" + std::to_string(r / 10) + "X");
            if (r % 10) o += "\x02" + unit(r % 10);
        }
    }
    return o;
}
}  // namespace

std::string number_words(const std::u32string& digits, const std::function<std::string(const std::string&)>& ph) {
    if (digits.empty()) return "";
    // leading zeros: each zero is said ("07" -> ноль семь; "000" -> нольноль ноль, as espeak-ng reads them)
    size_t z = 0;
    while (z < digits.size() && digits[z] == '0') z++;
    if (z == digits.size()) {
        std::string o;
        for (size_t k = 0; k < z; k++) { if (k) o += (z == 3 && k == 1) ? "\x02" : "||"; o += ph("_0"); }
        return o;
    }
    if (z > 0) {
        std::string o;
        for (size_t k = 0; k < z; k++) { if (k) o += "\x02"; o += ph("_0"); }
        return o + "||" + number_words(digits.substr(z), ph);
    }
    if (digits.size() > 12) {   // longer: digit by digit
        std::string o;
        for (char32_t d : digits) { if (!o.empty()) o += "||"; o += ph(std::string("_") + (char)d); }
        return o;
    }
    long long v = 0;
    for (char32_t d : digits) v = v * 10 + (d - '0');
    std::string o;
    auto add = [&](const std::string& s) { if (s.empty()) return; if (!o.empty()) o += "||"; o += s; };
    static const long long kScale[] = {1000000000LL, 1000000LL, 1000LL};
    for (int si = 0; si < 3; si++) {
        int g = (int)((v / kScale[si]) % 1000);
        if (!g) continue;
        int idx = 3 - si;   // 3 = milliard, 2 = million, 1 = thousand
        bool fem = idx == 1;
        int last2 = g % 100, last = g % 10;
        std::string word;
        if (last2 >= 11 && last2 <= 19) word = ph("_0M" + std::to_string(idx));
        else if (last == 1) word = ph("_1MA" + std::to_string(idx));
        else if (last >= 2 && last <= 4) word = ph("_0MA" + std::to_string(idx));
        else word = ph("_0M" + std::to_string(idx));
        add(below_1000(g, fem, ph));
        add(word);
    }
    add(below_1000((int)(v % 1000), false, ph));
    return o;
}

namespace {

// One line of ru_listx: "<word> $<n>" -> the word as code bytes and n. false: a comment or any other kind of line.
bool parse_line(const char* l, size_t n, std::string& key, uint8_t& stress) {
    size_t a = 0;
    while (a < n && (l[a] == ' ' || l[a] == '\t')) a++;
    size_t b = a;
    while (b < n && l[b] != ' ' && l[b] != '\t' && l[b] != '\r') b++;
    if (b == a || (b - a >= 2 && l[a] == '/' && l[a + 1] == '/')) return false;
    size_t c = b;
    while (c < n && (l[c] == ' ' || l[c] == '\t')) c++;
    if (c + 1 >= n || l[c] != '$' || l[c + 1] < '1' || l[c + 1] > '9') return false;
    if (c + 2 < n && l[c + 2] != ' ' && l[c + 2] != '\t' && l[c + 2] != '\r' && l[c + 2] != '/') return false;
    key.clear();
    for (size_t i = a; i < b; i += 2) {   // 2-byte UTF-8 Cyrillic letters -> one code byte each
        unsigned char c0 = (unsigned char)l[i];
        if ((c0 >> 5) != 6 || i + 1 >= b) return false;
        uint8_t code = letter_code(((char32_t)(c0 & 0x1F) << 6) | ((unsigned char)l[i + 1] & 0x3F));
        if (!code) return false;
        key += (char)code;
    }
    if (key.empty() || key.size() >= 255) return false;
    stress = (uint8_t)(l[c + 1] - '0');
    return true;
}

// The entries of one byte range [begin, end) of the file, read in 64 KB pieces.
class RangeReader {
public:
    RangeReader(const std::string& path, std::streamoff begin, std::streamoff end)
        : f_(path, std::ios::binary), left_(end - begin), next_line_(begin), buf_(1 << 16, '\0') {
        f_.seekg(begin);
    }
    // The next entry and the file offset of its line. false at the end of the range.
    bool next(std::string& key, uint8_t& stress, std::streamoff* line_start = nullptr) {
        const char* l; size_t n;
        for (;;) {
            std::streamoff at = next_line_;
            if (!read_line(l, n)) return false;
            if (!parse_line(l, n, key, stress)) continue;
            if (line_start) *line_start = at;
            return true;
        }
    }

private:
    bool read_line(const char*& l, size_t& n) {
        line_.clear();
        for (;;) {
            if (pos_ == len_) {
                if (left_ <= 0 || !f_) {
                    if (line_.empty()) return false;
                    l = line_.data(); n = line_.size();   // a last line without '\n'
                    next_line_ += (std::streamoff)n;
                    done_tail_ = line_; line_.clear(); l = done_tail_.data();
                    return true;
                }
                std::streamsize want = (std::streamsize)std::min<std::streamoff>(left_, (std::streamoff)buf_.size());
                f_.read(&buf_[0], want);
                len_ = (size_t)f_.gcount(); pos_ = 0;
                left_ -= (std::streamoff)len_;
                if (!len_) left_ = 0;
                continue;
            }
            const char* p = buf_.data() + pos_;
            const char* nl = (const char*)std::memchr(p, '\n', len_ - pos_);
            if (!nl) { line_.append(p, len_ - pos_); pos_ = len_; continue; }
            size_t k = (size_t)(nl - p);
            if (line_.empty()) { l = p; n = k; }
            else { line_.append(p, k); done_tail_.swap(line_); line_.clear(); l = done_tail_.data(); n = done_tail_.size(); }
            pos_ += k + 1;
            next_line_ += (std::streamoff)n + 1;
            return true;
        }
    }
    std::ifstream f_;
    std::streamoff left_, next_line_;
    std::string buf_, line_, done_tail_;
    size_t pos_ = 0, len_ = 0;
};

}  // namespace

void StressDict::append(const std::string& key, const std::string& prev, uint8_t stress) {
    if (count_ % kBlock == 0) {
        index_.push_back((uint32_t)data_.size());
        data_.insert(data_.end(), key.begin(), key.end());
        data_.push_back(0);
    } else {
        size_t sh = 0;
        while (sh < prev.size() && sh < key.size() && prev[sh] == key[sh]) sh++;
        data_.push_back((uint8_t)sh);
        data_.push_back((uint8_t)(key.size() - sh));
        data_.insert(data_.end(), key.begin() + sh, key.end());
    }
    data_.push_back(stress);
    count_++;
}

bool StressDict::load(const std::string& path) {
    // The file is a few sorted runs one after another (two in espeak-ng 1.52.0). Pass 1 finds where each run starts;
    // pass 2 reads the runs side by side and merges them straight into the front-coded table. Nothing but the
    // finished table (~5 MB) and one 64 KB buffer per run is held in memory, instead of the whole 23 MB text plus a
    // row per word. A repeated word keeps its last line, as in espeak-ng ("периодом" $2 then $3 -> 3).
    std::streamoff fsize;
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        fsize = f.tellg();
    }
    data_.clear(); index_.clear(); count_ = 0;
    if (fsize <= 0) return true;
    std::vector<std::streamoff> starts{0};
    {
        RangeReader r(path, 0, fsize);
        std::string key, prev; uint8_t st; std::streamoff at; bool any = false;
        while (r.next(key, st, &at)) {
            if (any && key < prev) starts.push_back(at);
            prev.swap(key); any = true;
        }
    }
    data_.reserve((size_t)fsize / 5);                     // the table is a little under 1/5 of the text
    index_.reserve((size_t)fsize / 28 / kBlock + 16);     // about 28 bytes per line
    std::string prev;
    if (starts.size() <= kMaxRuns) {
        struct Head { std::unique_ptr<RangeReader> r; std::string key; uint8_t stress = 0; bool ok = false; };
        std::vector<Head> heads(starts.size());
        for (size_t i = 0; i < starts.size(); i++) {
            heads[i].r.reset(new RangeReader(path, starts[i], i + 1 < starts.size() ? starts[i + 1] : fsize));
            heads[i].ok = heads[i].r->next(heads[i].key, heads[i].stress);
        }
        std::string k;
        for (;;) {
            int best = -1;
            for (size_t i = 0; i < heads.size(); i++)
                if (heads[i].ok && (best < 0 || heads[i].key < heads[best].key)) best = (int)i;
            if (best < 0) break;
            k = heads[best].key;
            uint8_t stress = 0;
            for (auto& h : heads)   // in file order: the last line of a repeated word wins
                while (h.ok && h.key == k) { stress = h.stress; h.ok = h.r->next(h.key, h.stress); }
            append(k, prev, stress);
            prev.swap(k);
        }
    } else {   // not the layout we know (many runs): sort in memory
        std::vector<std::pair<std::string, uint8_t>> rows;
        RangeReader r(path, 0, fsize);
        std::string key; uint8_t st;
        while (r.next(key, st)) rows.emplace_back(key, st);
        std::stable_sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (size_t i = 0; i < rows.size(); i++) {
            if (i + 1 < rows.size() && rows[i + 1].first == rows[i].first) continue;
            append(rows[i].first, prev, rows[i].second);
            prev = rows[i].first;
        }
    }
    if (data_.capacity() > data_.size() + data_.size() / 4) data_.shrink_to_fit();
    index_.shrink_to_fit();
    return true;
}

int StressDict::lookup(const std::u32string& word) const {
    if (index_.empty() || word.empty() || word.size() >= 255) return 0;
    std::string key;
    for (char32_t ch : word) {
        uint8_t code = letter_code(ch);
        if (!code) return 0;
        key += (char)code;
    }
    // last block whose first key <= key
    size_t lo = 0, hi = index_.size();
    while (hi - lo > 1) {
        size_t mid = (lo + hi) / 2;
        const char* first = (const char*)data_.data() + index_[mid];
        if (std::strcmp(first, key.c_str()) <= 0) lo = mid; else hi = mid;
    }
    size_t p = index_[lo];
    size_t end = lo + 1 < index_.size() ? index_[lo + 1] : data_.size();
    std::string cur((const char*)data_.data() + p);
    p += cur.size() + 1;
    if (cur == key) return data_[p];
    p++;
    while (p < end) {
        size_t sh = data_[p], sl = data_[p + 1];
        p += 2;
        cur.resize(sh);
        cur.append((const char*)data_.data() + p, sl);
        p += sl;
        int st = data_[p++];
        if (cur == key) return st;
        if (cur > key) return 0;
    }
    return 0;
}

}  // namespace ru
}  // namespace kt_fe
