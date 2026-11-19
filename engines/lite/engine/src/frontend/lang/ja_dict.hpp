// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Japanese word segmentation for the Kitten TTS 2 mini text front end: an original reader of a compiled MeCab-format
// dictionary (the UniDic 2.1.2 files that the Python package uses through unidic-lite 1.0.8: sys.dic, unk.dic,
// matrix.bin, char.bin) and a lowest-cost path search over the word lattice, written for this library from the
// published file formats. The dictionary is data: UniDic is offered by the UniDic Consortium under the BSD licence
// (or GPL/LGPL); it is downloaded by the app (engines/lite/mobile/fetch_phonemizer_data.sh), never bundled.
//
// The large files are memory-mapped read only, so the resident memory is only the pages a text touches.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kt_fe {
namespace ja {

// A read-only view of a whole file (memory-mapped where the platform allows it, read into memory otherwise).
class FileView {
public:
    FileView() = default;
    ~FileView();
    FileView(const FileView&) = delete;
    FileView& operator=(const FileView&) = delete;
    bool open(const std::string& path, std::string* err);
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool mapped_ = false;
    std::vector<uint8_t> copy_;
};

// One entry of a dictionary's token table (16 bytes, little-endian in the file).
struct DicToken {
    uint16_t left_id;
    uint16_t right_id;
    int16_t cost;
    uint32_t feature;   // byte offset of the feature text
};

// One morpheme of the best segmentation.
struct Morph {
    std::string surface;     // the text of the word (UTF-8, without the spaces skipped before it)
    std::string feature;     // the dictionary's comma-separated feature text
    int char_type = 0;       // the character category of its first character (char.bin order)
    bool unknown = false;    // not in sys.dic (made from an unk.dic template)
};

class JaDictionary {
public:
    // dir holds sys.dic, unk.dic, matrix.bin and char.bin. Returns nullptr and sets *err on failure.
    static std::unique_ptr<JaDictionary> load(const std::string& dir, std::string* err);
    // UTF-8 text -> the lowest-cost sequence of morphemes.
    std::vector<Morph> parse(const std::string& text) const;
    size_t mapped_bytes() const;

private:
    struct Dic {
        FileView file;
        const uint8_t* units = nullptr;   // double array: (int32 base, uint32 check) pairs
        size_t n_units = 0;
        const uint8_t* tokens = nullptr;
        size_t n_tokens = 0;
        const char* features = nullptr;
        size_t features_size = 0;
        bool open(const std::string& path, std::string* err);
        // Calls f(value, key_length) for every key that is a prefix of s[0, n), shortest first.
        template <class F> void prefixes(const uint8_t* s, size_t n, F f) const;
        bool exact(const std::string& key, uint32_t* value) const;
        DicToken token(size_t i) const;
        std::string feature(uint32_t off) const;
    };
    struct CharInfo {
        uint32_t raw = 0;
        uint32_t types() const { return raw & 0x3FFFFu; }
        int default_type() const { return (int)((raw >> 18) & 0xFFu); }
        int length() const { return (int)((raw >> 26) & 0xFu); }
        bool group() const { return (raw >> 30) & 1u; }
        bool invoke() const { return (raw >> 31) & 1u; }
        bool same_kind(CharInfo o) const { return (types() & o.types()) != 0; }
    };
    struct Node {
        uint32_t begin, end;     // byte range of the surface
        uint16_t left_id, right_id;
        int32_t cost;            // word cost
        uint32_t feature;        // offset into its dictionary's feature text
        bool unknown;
        uint8_t char_type;
        int64_t total = 0;       // best path cost up to and including this node
        int32_t prev = -1;
    };

    CharInfo char_info(const uint8_t* p, const uint8_t* end, size_t* len) const;
    int connect(uint16_t right_id_of_left, uint16_t left_id_of_right) const;
    void lookup(const std::string& text, size_t pos, std::vector<Node>& out) const;

    Dic sys_, unk_;
    FileView matrix_file_, char_file_;
    const uint8_t* matrix_ = nullptr;
    size_t matrix_left_ = 0, matrix_right_ = 0;
    std::vector<uint32_t> char_table_;          // 65536 entries (0xFFFF: as 0)
    std::vector<std::pair<uint32_t, uint32_t>> unk_tokens_;   // per category: (first token, count) in unk.dic
    CharInfo space_;
    static constexpr int kMaxGroup = 24;        // MeCab's default max-grouping-size (unidic's dicrc leaves it unset)
};

}  // namespace ja
}  // namespace kt_fe
