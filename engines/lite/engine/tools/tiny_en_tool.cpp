// Part of kitten-inference (Apache-2.0).
// tiny_en_tool: parity and timing tool for the Kitten TTS 2 English text front end (src/frontend/tiny_en).
//   tiny_en_tool G2P_DIR DATA_DIR VOCAB_JSON v1|v2 [--repeat N] < in.jsonl > out.jsonl
// in.jsonl lines: {"id": "...", "text": "...", "tokens": [[text, tag, ws], ...] (optional: spaCy's tokens of the
// normalised text, from the reference)}. Out: {"id", "norm", "chunks", "us" (warm time of the text -> chunks call,
// mean over N repeats), "bart_ms", "bart_words", "ps_oracle" (G2P with spaCy's tokens and tags),
// "ps_oracle_tok" (spaCy's tokens, rule tags), "tok_same" (our tokens == spaCy's)}. Load time and memory go to
// stderr as one JSON line. --bart-fp64: double-precision BART (default float32). --bart-words FILE: run only the BART on each word of FILE.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "frontend/tiny_en/json_lite.hpp"
#include "frontend/tiny_en/tiny_en.hpp"

using namespace kt_tinyen;

static std::string jesc(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

static long rss_kb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) if (line.rfind(key, 0) == 0) return std::atol(line.c_str() + std::strlen(key));
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: tiny_en_tool G2P_DIR DATA_DIR VOCAB_JSON v1|v2 [--repeat N]\n"); return 2; }
    int repeat = 0;
    bool fp32 = true;
    const char* bart_words = nullptr;
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "--bart-fp64")) fp32 = false;
        if (!std::strcmp(argv[i], "--bart-words") && i + 1 < argc) bart_words = argv[i + 1];
    }
    long rss0 = rss_kb("VmRSS:");
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    auto fe = EnglishFrontend::load(argv[1], argv[2], argv[3], argv[4], err);
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!fe) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    long rss1 = rss_kb("VmRSS:");
    fe->set_bart_fp32(fp32);
    if (bart_words) {  // one word per line -> "word<TAB>phonemes<TAB>us"
        std::ifstream wf(bart_words);
        std::string w;
        while (std::getline(wf, w)) {
            if (w.empty()) continue;
            auto a = std::chrono::steady_clock::now();
            std::string ps = fe->bart_phonemes(w);
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
            std::cout << w << "\t" << ps << "\t" << us << "\n";
        }
        return 0;
    }
    std::fprintf(stderr, "{\"load_ms\": %.2f, \"rss_before_kb\": %ld, \"rss_after_load_kb\": %ld, \"data_bytes\": %zu}\n",
                 load_ms, rss0, rss1, fe->memory_bytes());
    std::string line;
    std::vector<std::string> chunks;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json::Value v;
        if (!json::parse(line, v)) { std::fprintf(stderr, "bad input line\n"); return 1; }
        const json::Value* id = v.get("id");
        const json::Value* text = v.get("text");
        if (!text) continue;
        auto a = std::chrono::steady_clock::now();
        fe->text_to_chunks(text->str, chunks, err);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
        Timing tm = fe->timing();
        const double us_first = us;
        if (repeat > 0) {
            auto b = std::chrono::steady_clock::now();
            for (int r = 0; r < repeat; ++r) fe->text_to_chunks(text->str, chunks, err);
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - b).count() / repeat;
        }
        std::string norm = fe->normalize(text->str);
        std::string o = "{\"id\": " + jesc(id ? (id->kind == json::Value::Str ? id->str : std::to_string((long long)id->num)) : "") +
                        ", \"norm\": " + jesc(norm) + ", \"chunks\": [";
        for (size_t i = 0; i < chunks.size(); ++i) o += (i ? ", " : "") + jesc(chunks[i]);
        char nb[160];
        std::snprintf(nb, sizeof nb, "], \"us\": %.1f, \"us_first\": %.1f, \"bart_ms\": %.3f, \"bart_words\": %d", us, us_first, tm.bart_ms, tm.bart_words);
        o += nb;
        const json::Value* toks = v.get("tokens");
        if (toks && toks->kind == json::Value::Arr) {
            std::vector<SpTok> sp, sp2;
            for (auto& t : toks->arr) {
                if (t.arr.size() < 3) continue;
                SpTok s;
                s.text = utf8_to_u32(t.arr[0].str);
                s.tag = t.arr[1].str;
                s.ws = t.arr[2].kind == json::Value::Bool ? t.arr[2].b : !t.arr[2].str.empty();
                sp.push_back(s);
            }
            o += ", \"ps_oracle\": " + jesc(fe->g2p_tokens(sp));
            // spaCy's tokens with the rule tags
            std::string joined;
            for (auto& s : sp) { joined += u32_to_utf8(s.text); if (s.ws) joined += " "; }
            auto mine = fe->tokenize(joined);
            bool same = mine.size() == sp.size();
            for (size_t i = 0; same && i < sp.size(); ++i) same = mine[i].text == sp[i].text && mine[i].ws == sp[i].ws;
            o += std::string(", \"tok_same\": ") + (same ? "true" : "false");
            if (same) {
                for (size_t i = 0; i < sp.size(); ++i) sp[i].tag = mine[i].tag;
                o += ", \"ps_oracle_tok\": " + jesc(fe->g2p_tokens(sp));
            }
            std::string tl = "[";
            for (size_t i = 0; i < mine.size(); ++i) tl += (i ? ", " : "") + jesc(u32_to_utf8(mine[i].text) + "/" + mine[i].tag);
            o += ", \"tokens_rule\": " + tl + "]";
        }
        o += "}";
        std::cout << o << "\n";
    }
    std::fprintf(stderr, "{\"rss_end_kb\": %ld, \"hwm_kb\": %ld}\n", rss_kb("VmRSS:"), rss_kb("VmHWM:"));
    return 0;
}
