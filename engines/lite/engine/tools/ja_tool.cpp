// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
// Parity and timing tool for the Japanese text front end (mini, phonemizer v2).
//   ja_tool DATA_DIR VOCAB_JSON TEXTS.txt [--words] [--repeat N] [--api] [--maps]
// Prints one line per input line: the phoneme chunks joined by TAB (--words: the normalised text and the tagger's
// words instead); a text the library refuses, as the Python package does, prints "!ERROR <status>". Timing and
// memory go to stderr as JSON. --api goes through the C API (kt08_tiny_*), with DATA_DIR as espeak_data_dir (the
// Japanese files in DATA_DIR/ja). --maps adds one stderr line per memory-mapped dictionary file: its size and how
// much of it is resident in this process (Linux /proc/self/smaps).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "frontend/lang/ja.hpp"
#include "frontend/tiny_lang_frontend.hpp"
#ifdef KT_JA_TOOL_API
#include "kitten_tts08.h"
#endif

static double rss_mib(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(key, 0) == 0) return std::atof(line.c_str() + std::strlen(key)) / 1024.0;
    return -1;
}

// One line per mapped dictionary file (*.dic, *.bin): mapped size and resident part, from /proc/self/smaps.
static void print_maps() {
    std::ifstream f("/proc/self/smaps");
    std::string line, name;
    double size_kb = 0;
    while (std::getline(f, line)) {
        if (line.find('-') != std::string::npos && line.find(' ') != std::string::npos && line.find(':') > line.find(' ')) {
            const size_t slash = line.rfind('/');   // a mapping's header line: "start-end perms offset dev inode path"
            name = slash == std::string::npos ? "" : line.substr(slash + 1);
            if (name.size() < 4 || (name.compare(name.size() - 4, 4, ".dic") && name.compare(name.size() - 4, 4, ".bin"))) name.clear();
        } else if (!name.empty() && line.rfind("Size:", 0) == 0) {
            size_kb = std::atof(line.c_str() + 5);
        } else if (!name.empty() && line.rfind("Rss:", 0) == 0) {
            std::fprintf(stderr, "{\"map\": \"%s\", \"size_mib\": %.2f, \"resident_mib\": %.2f}\n", name.c_str(),
                         size_kb / 1024.0, std::atof(line.c_str() + 4) / 1024.0);
            name.clear();
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ja_tool DATA_DIR VOCAB_JSON TEXTS [--words] [--repeat N] [--api] [--maps]\n"); return 2; }
    bool words = false, api = false, maps = false;
    int repeat = 1;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--words")) words = true;
        else if (!std::strcmp(argv[i], "--api")) api = true;
        else if (!std::strcmp(argv[i], "--maps")) maps = true;
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }
    std::vector<std::string> lines;
    {
        std::ifstream f(argv[3]);
        std::string l;
        while (std::getline(f, l)) lines.push_back(l);
    }
    const double rss0 = rss_mib("VmRSS:");
    auto t0 = std::chrono::steady_clock::now();
    std::unordered_set<char32_t> vocab;
    if (!kt_fe::TinyLangFrontend::read_vocab(argv[2], vocab)) { std::fprintf(stderr, "bad vocab\n"); return 1; }
    std::unique_ptr<kt_fe::JaFrontend> fe;
#ifdef KT_JA_TOOL_API
    kt08_tiny_frontend* cfe = nullptr;
#endif
    if (api) {
#ifdef KT_JA_TOOL_API
        if (kt08_tiny_frontend_load("v2", argv[2], nullptr, argv[1], &cfe) != KT08_OK) { std::fprintf(stderr, "load: %s\n", kt08_last_error()); return 1; }
#else
        std::fprintf(stderr, "built without the C API\n"); return 1;
#endif
    } else {
        std::string err;
        fe = kt_fe::JaFrontend::load(std::string(argv[1]) + "/ja", &err);
        if (!fe) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double rss_load = rss_mib("VmRSS:");
    std::vector<double> ms(lines.size());
    std::string out;
    double first_ms = 0;
    for (int r = 0; r < repeat; ++r) {
        out.clear();
        for (size_t i = 0; i < lines.size(); ++i) {
            auto a = std::chrono::steady_clock::now();
            std::string line;
            if (words) {
                try { line = fe->debug_words(lines[i]); } catch (const kt_fe::JaTextError&) { line = "!ERROR 1\n"; }
            } else if (api) {
#ifdef KT_JA_TOOL_API
                char* c = nullptr;
                const kt08_status st = kt08_tiny_text_to_phonemes(cfe, lines[i].c_str(), "ja", &c);
                if (st == KT08_OK) {
                    line = c;
                    kt08_free(c);
                    std::replace(line.begin(), line.end(), '\n', '\t');
                } else if (st == KT08_ERR_ARGUMENT || st == KT08_ERR_TOO_LONG) {
                    line = "!ERROR " + std::to_string((int)st);   // refused text: reported, the run goes on
                } else {
                    std::fprintf(stderr, "err: %s\n", kt08_last_error());
                    return 1;
                }
#endif
            } else {
                try {
                    auto ch = fe->chunks(lines[i], &vocab);
                    for (size_t k = 0; k < ch.size(); ++k) { if (k) line += '\t'; line += ch[k]; }
                } catch (const kt_fe::JaTextError&) {
                    line = "!ERROR 1";
                }
            }
            ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            if (r == 0 && i == 0) first_ms = ms[i];
            out += line;
            out += '\n';
        }
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double x : s) sum += x;
    auto pct = [&](double p) { return s.empty() ? 0.0 : s[std::min(s.size() - 1, (size_t)(p * (double)s.size()))]; };
    std::fprintf(stderr,
                 "{\"n\": %zu, \"repeat\": %d, \"load_ms\": %.3f, \"first_call_ms\": %.3f, \"mean_ms\": %.4f, \"p50_ms\": %.4f, "
                 "\"p90_ms\": %.4f, \"max_ms\": %.4f, \"rss_start_mib\": %.2f, \"rss_after_load_mib\": %.2f, "
                 "\"rss_after_run_mib\": %.2f, \"hwm_mib\": %.2f, \"rss_anon_mib\": %.2f, \"rss_file_mib\": %.2f}\n",
                 s.size(), repeat, load_ms, first_ms, s.empty() ? 0 : sum / (double)s.size(), pct(0.5), pct(0.9),
                 s.empty() ? 0 : s.back(), rss0, rss_load, rss_mib("VmRSS:"), rss_mib("VmHWM:"), rss_mib("RssAnon:"), rss_mib("RssFile:"));
    if (maps) print_maps();
    return 0;
}
