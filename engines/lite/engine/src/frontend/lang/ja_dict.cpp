// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/ja_dict.hpp"

#include <cstdio>
#include <cstring>
#include <limits>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kt_fe {
namespace ja {

namespace {
uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
}  // namespace

// ---------------------------------------------------------------------------------------------------------------
FileView::~FileView() {
#if !defined(_WIN32)
    if (mapped_ && data_) munmap(const_cast<uint8_t*>(data_), size_);
#endif
}

bool FileView::open(const std::string& path, std::string* err) {
#if !defined(_WIN32)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { if (err) *err = "cannot open " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); if (err) *err = "cannot read " + path; return false; }
    size_ = (size_t)st.st_size;
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p != MAP_FAILED) {
        // lookups touch scattered pages: no read-ahead, so only the pages a text needs become resident
        madvise(p, size_, MADV_RANDOM);
        data_ = (const uint8_t*)p; mapped_ = true; return true;
    }
#endif
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); if (err) *err = "cannot read " + path; return false; }
    copy_.resize((size_t)n);
    size_t got = std::fread(copy_.data(), 1, copy_.size(), f);
    std::fclose(f);
    if (got != copy_.size()) { if (err) *err = "cannot read " + path; return false; }
    data_ = copy_.data();
    size_ = copy_.size();
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// sys.dic / unk.dic: a 72-byte header (magic = file size ^ 0xef718f77, version 102, ..., the byte sizes of the
// double array, the token table and the feature text), then those three parts.
bool JaDictionary::Dic::open(const std::string& path, std::string* err) {
    std::string local_err; if (!err) err = &local_err;
    if (!file.open(path, err)) return false;
    const uint8_t* p = file.data();
    const size_t n = file.size();
    if (n < 72) { *err = path + ": too small"; return false; }
    const uint32_t magic = rd32(p), version = rd32(p + 4);
    const uint32_t dsize = rd32(p + 24), tsize = rd32(p + 28), fsize = rd32(p + 32);
    if ((magic ^ 0xef718f77u) != (uint32_t)n || version != 102) { *err = path + ": not a MeCab dictionary (version 102)"; return false; }
    if ((uint64_t)72 + dsize + tsize + fsize != n || dsize % 8 || tsize % 16) { *err = path + ": inconsistent sizes"; return false; }
    units = p + 72;
    n_units = dsize / 8;
    tokens = units + dsize;
    n_tokens = tsize / 16;
    features = (const char*)(tokens + tsize);
    features_size = fsize;
    if (n_units == 0) { *err = path + ": empty index"; return false; }
    return true;
}

// The index is a double array over the UTF-8 bytes of the keys: from a node at base b, byte c leads to unit
// b + c + 1 if that unit's check equals b; a key ends at a node whose unit b (the 0 transition) has check b and a
// negative base, which holds -(value + 1).
template <class F>
void JaDictionary::Dic::prefixes(const uint8_t* s, size_t n, F f) const {
    auto base = [&](size_t i) { return (int32_t)rd32(units + 8 * i); };
    auto check = [&](size_t i) { return rd32(units + 8 * i + 4); };
    int64_t b = base(0);
    for (size_t i = 0;; ++i) {
        if (b < 0 || (size_t)b >= n_units) return;
        const int32_t t = base((size_t)b);
        if (check((size_t)b) == (uint32_t)b && t < 0) f((uint32_t)(-(int64_t)t - 1), i);
        if (i == n) return;
        const int64_t q = b + (int64_t)s[i] + 1;
        if ((size_t)q >= n_units || check((size_t)q) != (uint32_t)b) return;
        b = base((size_t)q);
    }
}

bool JaDictionary::Dic::exact(const std::string& key, uint32_t* value) const {
    bool found = false;
    prefixes((const uint8_t*)key.data(), key.size(), [&](uint32_t v, size_t len) {
        if (len == key.size()) { *value = v; found = true; }
    });
    return found;
}

DicToken JaDictionary::Dic::token(size_t i) const {
    const uint8_t* t = tokens + 16 * i;
    return DicToken{rd16(t), rd16(t + 2), (int16_t)rd16(t + 6), rd32(t + 8)};
}

std::string JaDictionary::Dic::feature(uint32_t off) const {
    if (off >= features_size) return std::string();
    const char* s = features + off;
    const void* z = std::memchr(s, 0, features_size - off);
    return std::string(s, z ? (size_t)((const char*)z - s) : features_size - off);
}

// ---------------------------------------------------------------------------------------------------------------
std::unique_ptr<JaDictionary> JaDictionary::load(const std::string& dir, std::string* err) {
    std::unique_ptr<JaDictionary> d(new JaDictionary());
    const std::string base = dir.empty() || dir.back() == '/' ? dir : dir + "/";
    if (!d->sys_.open(base + "sys.dic", err) || !d->unk_.open(base + "unk.dic", err)) return nullptr;
    // matrix.bin: uint16 left size, uint16 right size, then int16 costs
    if (!d->matrix_file_.open(base + "matrix.bin", err)) return nullptr;
    {
        const uint8_t* p = d->matrix_file_.data();
        const size_t n = d->matrix_file_.size();
        if (n < 4) { *err = "matrix.bin: too small"; return nullptr; }
        d->matrix_left_ = rd16(p);
        d->matrix_right_ = rd16(p + 2);
        if ((uint64_t)4 + 2ull * d->matrix_left_ * d->matrix_right_ != (uint64_t)n) { *err = "matrix.bin: inconsistent size"; return nullptr; }
        d->matrix_ = p + 4;
    }
    // char.bin: uint32 category count, 32-byte category names, then one uint32 per UCS-2 code 0..0xFFFE
    if (!d->char_file_.open(base + "char.bin", err)) return nullptr;
    std::vector<std::string> names;
    {
        const uint8_t* p = d->char_file_.data();
        const size_t n = d->char_file_.size();
        if (n < 4) { *err = "char.bin: too small"; return nullptr; }
        const uint32_t nc = rd32(p);
        if (nc == 0 || nc > 256 || (uint64_t)4 + 32ull * nc + 4ull * 0xFFFF != n) { *err = "char.bin: inconsistent size"; return nullptr; }
        for (uint32_t i = 0; i < nc; ++i) {
            const char* s = (const char*)p + 4 + 32 * i;
            names.emplace_back(s, strnlen(s, 32));
        }
        d->char_table_.assign(65536, 0);
        const uint8_t* t = p + 4 + 32 * nc;
        for (size_t i = 0; i < 0xFFFF; ++i) d->char_table_[i] = rd32(t + 4 * i);
        d->char_table_[0xFFFF] = d->char_table_[0];   // not in the file
    }
    // unknown-word templates per category: the unk.dic key is the category name
    d->unk_tokens_.assign(names.size(), {0, 0});
    for (size_t c = 0; c < names.size(); ++c) {
        uint32_t v = 0;
        if (!d->unk_.exact(names[c], &v)) continue;
        const uint32_t first = v >> 8, count = v & 0xFFu;
        if ((uint64_t)first + count > d->unk_.n_tokens) { *err = "unk.dic: bad token range"; return nullptr; }
        d->unk_tokens_[c] = {first, count};
    }
    if (d->unk_tokens_.empty() || d->unk_tokens_[0].second == 0) { *err = "unk.dic: no DEFAULT category"; return nullptr; }
    d->space_.raw = d->char_table_[0x20];
    return d;
}

size_t JaDictionary::mapped_bytes() const {
    return sys_.file.size() + unk_.file.size() + matrix_file_.size() + char_file_.size();
}

// The category of the character at p (UCS-2 codes index char.bin; 4-byte UTF-8 characters use entry 0, as the
// Python package's tagger does).
JaDictionary::CharInfo JaDictionary::char_info(const uint8_t* p, const uint8_t* end, size_t* len) const {
    const size_t n = (size_t)(end - p);
    uint32_t cp = 0;
    if (n == 0) { *len = 0; cp = 0; }
    else if (p[0] < 0x80) { *len = 1; cp = p[0]; }
    else if (n >= 2 && (p[0] & 0xE0) == 0xC0) { *len = 2; cp = ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu); }
    else if (n >= 3 && (p[0] & 0xF0) == 0xE0) { *len = 3; cp = ((p[0] & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu); }
    else if (n >= 4 && (p[0] & 0xF8) == 0xF0) { *len = 4; cp = 0; }
    else { *len = 1; cp = 0; }
    CharInfo c;
    c.raw = char_table_[cp & 0xFFFFu];
    return c;
}

int JaDictionary::connect(uint16_t r, uint16_t l) const {
    if (r >= matrix_left_ || l >= matrix_right_) return 0x7FFF;   // never for a consistent dictionary
    return (int16_t)rd16(matrix_ + 2 * ((size_t)r + matrix_left_ * (size_t)l));
}

// The candidate words that start at byte pos (after any spaces), in the order the search considers them.
void JaDictionary::lookup(const std::string& text, size_t pos, std::vector<Node>& out) const {
    out.clear();
    const uint8_t* const s = (const uint8_t*)text.data();
    const uint8_t* const e = s + text.size();
    const uint8_t* q = s + pos;
    size_t ml = 0;
    CharInfo ci;
    {   // skip a run of space-category characters (each compared with the one before it)
        CharInfo kind = space_;
        while (q < e) {
            CharInfo c = char_info(q, e, &ml);
            if (!kind.same_kind(c)) break;
            kind = c;   // the run's kind follows its latest character (as the reference tagger: measured)
            q += ml;
        }
    }
    if (q >= e) return;
    ci = char_info(q, e, &ml);
    const uint32_t begin = (uint32_t)(q - s);
    const uint8_t ctype = (uint8_t)ci.default_type();

    sys_.prefixes(q, (size_t)(e - q), [&](uint32_t v, size_t len) {
        if (len == 0) return;
        const size_t first = v >> 8, count = v & 0xFFu;
        for (size_t j = 0; j < count && first + j < sys_.n_tokens; ++j) {
            const DicToken t = sys_.token(first + j);
            out.push_back(Node{begin, begin + (uint32_t)len, t.left_id, t.right_id, t.cost, t.feature, false, ctype});
        }
    });
    if (!out.empty() && !ci.invoke()) return;

    const size_t cat = (size_t)ci.default_type() < unk_tokens_.size() ? (size_t)ci.default_type() : 0;
    auto add = [&](const uint8_t* stop) {
        const auto& r = unk_tokens_[cat];
        for (uint32_t j = 0; j < r.second; ++j) {
            const DicToken t = unk_.token(r.first + j);
            out.push_back(Node{begin, (uint32_t)(stop - s), t.left_id, t.right_id, t.cost, t.feature, true, ctype});
        }
    };
    const uint8_t* const first_end = q + ml;
    const uint8_t* group_end = nullptr;
    if (ci.group()) {   // one word for the whole run of characters of this kind (if not too long)
        const uint8_t* r = first_end;
        CharInfo kind = ci;
        size_t count = 0, l = 0;
        while (r < e) {
            CharInfo c = char_info(r, e, &l);
            if (!kind.same_kind(c)) break;
            kind = c;   // the run's kind follows its latest character (as the reference tagger: measured)
            r += l;
            ++count;
        }
        if (count <= (size_t)kMaxGroup) add(r);
        group_end = r;
    }
    {   // and words of 1 .. length characters of this kind
        const uint8_t* r = first_end;
        for (int i = 1; i <= ci.length(); ++i) {
            if (r > e || r == group_end) break;
            add(r);
            size_t l = 0;
            CharInfo c = char_info(r, e, &l);
            if (!ci.same_kind(c) || l == 0) break;
            r += l;
        }
    }
    if (out.empty()) add(first_end);
}

std::vector<Morph> JaDictionary::parse(const std::string& text) const {
    std::vector<Morph> result;
    const size_t n = text.size();
    if (n == 0) return result;
    std::vector<Node> nodes;
    std::vector<int32_t> end_head(n + 1, -1), end_next;
    nodes.push_back(Node{0, 0, 0, 0, 0, 0, false, 0});   // beginning of sentence
    nodes[0].total = 0;
    end_next.push_back(-1);
    end_head[0] = 0;
    std::vector<Node> cand;
    for (size_t pos = 0; pos < n; ++pos) {
        if (end_head[pos] < 0) continue;
        lookup(text, pos, cand);
        // candidates are considered newest first; words ending at the same place are compared newest first and the
        // first strictly lowest cost wins
        for (size_t k = cand.size(); k-- > 0;) {
            Node r = cand[k];
            int64_t best = std::numeric_limits<int64_t>::max();
            int32_t bi = -1;
            for (int32_t l = end_head[pos]; l >= 0; l = end_next[(size_t)l]) {
                const Node& ln = nodes[(size_t)l];
                const int64_t c = ln.total + connect(ln.right_id, r.left_id) + r.cost;
                if (c < best) { best = c; bi = l; }
            }
            r.total = best;
            r.prev = bi;
            const int32_t id = (int32_t)nodes.size();
            nodes.push_back(r);
            end_next.push_back(end_head[r.end]);
            end_head[r.end] = id;
        }
    }
    // end of sentence: after the last place any word ends
    size_t last = n;
    while (last > 0 && end_head[last] < 0) --last;
    int64_t best = std::numeric_limits<int64_t>::max();
    int32_t bi = -1;
    for (int32_t l = end_head[last]; l >= 0; l = end_next[(size_t)l]) {
        const Node& ln = nodes[(size_t)l];
        const int64_t c = ln.total + connect(ln.right_id, 0);
        if (c < best) { best = c; bi = l; }
    }
    std::vector<int32_t> path;
    for (int32_t i = bi; i > 0; i = nodes[(size_t)i].prev) path.push_back(i);
    for (size_t k = path.size(); k-- > 0;) {
        const Node& nd = nodes[(size_t)path[k]];
        Morph m;
        m.surface = text.substr(nd.begin, nd.end - nd.begin);
        m.feature = nd.unknown ? unk_.feature(nd.feature) : sys_.feature(nd.feature);
        m.char_type = nd.char_type;
        m.unknown = nd.unknown;
        result.push_back(std::move(m));
    }
    return result;
}

}  // namespace ja
}  // namespace kt_fe
