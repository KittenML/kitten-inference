// Parity and speed tool for the non-English text front end of Kitten TTS 2 (tests only, not installed).
//   tiny_lang_tool ESPEAK_DATA_DIR LANG {1|2} VOCAB_JSON TEXT_FILE [--api] [--repeat N]
// Prints one line per input line: the phoneme chunks separated by TAB. stderr: load time, memory, time per line.
// --api goes through the public C API (kt08_tiny_*), which only allows the languages this build supports.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "frontend/espeak_lang.h"
#include "frontend/tiny_lang_frontend.hpp"
#include "kitten_tts08.h"

static long rss_kib() {
    std::ifstream f("/proc/self/status");
    std::string k;
    while (f >> k) {
        if (k == "VmRSS:") { long v; f >> v; return v; }
        std::getline(f, k);
    }
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: tiny_lang_tool DATA_DIR LANG {1|2} VOCAB_JSON TEXT_FILE [--api] [--repeat N]\n"); return 2; }
    std::string dir = argv[1], lang = argv[2], vocab_path = argv[4];
    int ver = std::atoi(argv[3]);
    bool api = false; int repeat = 1;
    for (int i = 6; i < argc; i++) {
        if (!std::strcmp(argv[i], "--api")) api = true;
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }
    std::vector<std::string> lines;
    { std::ifstream in(argv[5]); std::string l; while (std::getline(in, l)) lines.push_back(l); }
    long rss0 = rss_kib();
    auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<kt_fe::EspeakLang> eng;
    std::unordered_set<char32_t> vocab;
    kt08_tiny_frontend* fe = nullptr;
    if (api) {
        if (kt08_tiny_frontend_load(ver == 1 ? "v1" : "v2", vocab_path.c_str(), nullptr, dir.c_str(), &fe) != KT08_OK) {
            std::fprintf(stderr, "load: %s\n", kt08_last_error()); return 1;
        }
        char* warm = nullptr;   // the language is parsed on its first call
        if (kt08_tiny_text_to_phonemes(fe, "", lang.c_str(), &warm) != KT08_OK) { std::fprintf(stderr, "%s\n", kt08_last_error()); return 1; }
        kt08_free(warm);
    } else {
        std::string err;
        eng = kt_fe::EspeakLang::load(kt_fe::tiny_lang_voice(lang), dir, &err);
        if (!eng) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        kt_fe::TinyLangFrontend::read_vocab(vocab_path, vocab);
    }
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    long rss1 = rss_kib();
    std::vector<std::string> out(lines.size());
    std::vector<double> ms;
    for (int r = 0; r < repeat; r++) {
        for (size_t i = 0; i < lines.size(); i++) {
            auto a = std::chrono::steady_clock::now();
            std::string o;
            if (api) {
                char* s = nullptr;
                if (kt08_tiny_text_to_phonemes(fe, lines[i].c_str(), lang.c_str(), &s) != KT08_OK) { std::fprintf(stderr, "%s\n", kt08_last_error()); return 1; }
                o = s;
                kt08_free(s);
                for (auto& c : o) if (c == '\n') c = '\t';
            } else {
                auto ch = kt_fe::tiny_lang_chunks(*eng, lang, ver, lines[i], &vocab);
                for (size_t k = 0; k < ch.size(); k++) { if (k) o += '\t'; o += ch[k]; }
            }
            if (r == repeat - 1) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            out[i] = o;
        }
    }
    for (auto& o : out) std::printf("%s\n", o.c_str());
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double sum = 0; for (double x : s) sum += x;
    std::fprintf(stderr, "{\"lang\": \"%s\", \"version\": %d, \"api\": %d, \"lines\": %zu, \"load_ms\": %.2f, \"rss_load_mib\": %.2f, "
                 "\"rss_after_mib\": %.2f, \"mean_ms\": %.4f, \"p50_ms\": %.4f, \"p90_ms\": %.4f, \"max_ms\": %.4f}\n",
                 lang.c_str(), ver, api ? 1 : 0, lines.size(), load_ms, (rss1 - rss0) / 1024.0, (rss_kib() - rss0) / 1024.0,
                 sum / std::max<size_t>(1, s.size()), s.empty() ? 0 : s[s.size() / 2], s.empty() ? 0 : s[s.size() * 9 / 10],
                 s.empty() ? 0 : s.back());
    if (fe) kt08_tiny_frontend_free(fe);
    return 0;
}
