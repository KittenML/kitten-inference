// Part of kitten-inference (Apache-2.0).
// A small, bounds-checked JSON reader for the English text front end's data files (config.json, vocab.json, the
// safetensors header, and the misaki / CMUdict lexicons). Malformed input fails cleanly (returns false), never reads
// past the buffer, and nesting is limited to 64 levels.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kt_tinyen::json {

struct Value {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;
    const Value* get(std::string_view k) const {
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

class Reader {
public:
    Reader(const char* p, size_t n) : p_(p), e_(p + n) {}
    void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    bool peek(char c) { ws(); return p_ < e_ && *p_ == c; }
    bool eat(char c) { ws(); if (p_ < e_ && *p_ == c) { ++p_; return true; } return false; }
    bool at_end() { ws(); return p_ >= e_; }

    bool string(std::string& out) {
        ws();
        if (p_ >= e_ || *p_ != '"') return false;
        ++p_;
        out.clear();
        while (p_ < e_) {
            char c = *p_++;
            if (c == '"') return true;
            if (c != '\\') { out.push_back(c); continue; }
            if (p_ >= e_) return false;
            char x = *p_++;
            switch (x) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        uint32_t lo;
                        if (p_ + 2 <= e_ && p_[0] == '\\' && p_[1] == 'u') {
                            p_ += 2;
                            if (!hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) return false;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else return false;
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool value(Value& v, int depth = 0) {
        if (depth > 64) return false;
        ws();
        if (p_ >= e_) return false;
        char c = *p_;
        if (c == '"') { v.kind = Value::Str; return string(v.str); }
        if (c == '{') {
            ++p_;
            v.kind = Value::Obj;
            if (eat('}')) return true;
            for (;;) {
                std::string k;
                if (!string(k) || !eat(':')) return false;
                v.obj.emplace_back(std::move(k), Value{});
                if (!value(v.obj.back().second, depth + 1)) return false;
                if (eat(',')) continue;
                return eat('}');
            }
        }
        if (c == '[') {
            ++p_;
            v.kind = Value::Arr;
            if (eat(']')) return true;
            for (;;) {
                v.arr.emplace_back();
                if (!value(v.arr.back(), depth + 1)) return false;
                if (eat(',')) continue;
                return eat(']');
            }
        }
        if (lit("null")) { v.kind = Value::Null; return true; }
        if (lit("true")) { v.kind = Value::Bool; v.b = true; return true; }
        if (lit("false")) { v.kind = Value::Bool; v.b = false; return true; }
        // number
        const char* s = p_;
        while (p_ < e_ && (std::strchr("+-.eE", *p_) || (*p_ >= '0' && *p_ <= '9'))) ++p_;
        if (p_ == s || p_ - s > 64) return false;
        std::string tmp(s, p_);
        char* endp = nullptr;
        v.kind = Value::Num;
        v.num = std::strtod(tmp.c_str(), &endp);
        return endp && *endp == 0;
    }

    bool lit(const char* w) {
        size_t n = std::strlen(w);
        if ((size_t)(e_ - p_) >= n && std::memcmp(p_, w, n) == 0) { p_ += n; return true; }
        return false;
    }
    const char* pos() const { return p_; }

private:
    bool hex4(uint32_t& cp) {
        if (e_ - p_ < 4) return false;
        cp = 0;
        for (int i = 0; i < 4; ++i) {
            char h = *p_++;
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= h - '0';
            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
            else return false;
        }
        return true;
    }
    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o.push_back((char)cp);
        else if (cp < 0x800) { o.push_back((char)(0xC0 | (cp >> 6))); o.push_back((char)(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) {
            o.push_back((char)(0xE0 | (cp >> 12))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            o.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            o.push_back((char)(0xF0 | (cp >> 18))); o.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    const char* p_;
    const char* e_;
};

inline bool parse(const std::string& text, Value& out) {
    Reader r(text.data(), text.size());
    return r.value(out) && r.at_end();
}

}  // namespace kt_tinyen::json
