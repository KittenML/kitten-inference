/* kt08_text: English text to phonemes with the Kitten TTS 0.8 C library (API version 4), for checks and timing.
 *
 *   kt08_text EN_RULES EN_LIST TEXTS.txt [REPEAT]
 *
 * TEXTS.txt holds one text per line. For each line the program prints "CHUNKS<TAB>PHONEMES", where CHUNKS are the
 * normalised chunks (kt08_text_chunks) and PHONEMES their phonemes (kt08_text_to_phonemes), both with chunks
 * separated by the byte 0x1F. REPEAT=N (default 0) phonemises every line N more times and reports the warm time per
 * line. On stderr: load time, peak memory and timings.
 */
#include "kitten_tts08.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static long status_kb(const char* field) {
    long v = -1;
#if defined(__linux__) || defined(__ANDROID__)
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    const size_t n = strlen(field);
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, field, n) && line[n] == ':') v = atol(line + n + 1);
    if (f) fclose(f);
#else
    (void)field;
#endif
    return v;
}

/* Calls f twice (size query, then the real call) into a growing buffer. */
typedef kt08_status (*text_fn)(kt08_phonemizer*, const char*, char*, size_t, size_t*);
static kt08_status chunks_fn(kt08_phonemizer* ph, const char* t, char* o, size_t c, size_t* n) {
    (void)ph;
    return kt08_text_chunks(t, o, c, n);
}
static kt08_status phonemes_fn(kt08_phonemizer* ph, const char* t, char* o, size_t c, size_t* n) {
    return kt08_text_to_phonemes(ph, t, o, c, n);
}
static char* call(text_fn f, kt08_phonemizer* ph, const char* text, char** buf, size_t* cap) {
    size_t need = 0;
    if (f(ph, text, NULL, 0, &need) != KT08_OK) return NULL;
    if (need > *cap) {
        char* nb = (char*)realloc(*buf, need);
        if (!nb) return NULL;
        *buf = nb;
        *cap = need;
    }
    if (f(ph, text, *buf, *cap, &need) != KT08_OK) return NULL;
    return *buf;
}

static void print_sep(const char* s) {
    for (; *s; ++s) putchar(*s == '\n' ? 0x1F : *s);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s EN_RULES EN_LIST TEXTS.txt [REPEAT]\n", argv[0]);
        return 2;
    }
    const int repeat = argc > 4 ? atoi(argv[4]) : 0;
    const long rss0 = status_kb("VmRSS");
    double t0 = now_s();
    kt08_phonemizer* ph = NULL;
    if (kt08_phonemizer_load(argv[1], argv[2], &ph) != KT08_OK) {
        fprintf(stderr, "load failed: %s\n", kt08_last_error());
        return 1;
    }
    const double load_s = now_s() - t0;
    fprintf(stderr, "LOAD s=%.4f api=%d rss_before_kb=%ld rss_after_kb=%ld rss_peak_kb=%ld\n", load_s,
            kt08_api_version(), rss0, status_kb("VmRSS"), status_kb("VmHWM"));
    FILE* in = fopen(argv[3], "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    static char line[1 << 16];
    char *b1 = NULL, *b2 = NULL;
    size_t c1 = 0, c2 = 0;
    int n = 0, fails = 0;
    double first_s = 0, warm_s = 0, worst_warm = 0;
    while (fgets(line, sizeof line, in)) {
        line[strcspn(line, "\r\n")] = 0;
        double a = now_s();
        const char* ch = call(chunks_fn, ph, line, &b1, &c1);
        const char* phs = call(phonemes_fn, ph, line, &b2, &c2);
        first_s += now_s() - a;
        if (!ch || !phs) {
            fprintf(stderr, "line %d failed: %s\n", n + 1, kt08_last_error());
            printf("\t\n");
            ++fails; ++n;
            continue;
        }
        print_sep(ch);
        putchar('\t');
        print_sep(phs);
        putchar('\n');
        if (repeat > 0) {
            /* warm: the text normalisation and phonemisation (a new text each time, so no cached result) */
            double best = 1e9;
            for (int r = 0; r < repeat; ++r) {
                size_t need = 0;
                kt08_text_to_phonemes(ph, "", NULL, 0, &need); /* clears the one-result cache */
                double s = now_s();
                kt08_text_to_phonemes(ph, line, NULL, 0, &need);
                double e = now_s() - s;
                if (e < best) best = e;
            }
            warm_s += best;
            if (best > worst_warm) worst_warm = best;
        }
        ++n;
    }
    fprintf(stderr, "DONE lines=%d failures=%d first_pass_ms_per_line=%.3f", n, fails, n ? 1e3 * first_s / n : 0.0);
    if (repeat > 0)
        fprintf(stderr, " warm_ms_per_line=%.3f warm_worst_ms=%.3f", n ? 1e3 * warm_s / n : 0.0, 1e3 * worst_warm);
    fprintf(stderr, " rss_peak_kb=%ld rss_now_kb=%ld\n", status_kb("VmHWM"), status_kb("VmRSS"));
    kt08_phonemizer_free(ph);
    free(b1);
    free(b2);
    return fails ? 1 : 0;
}
