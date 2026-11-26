/* kt08_example: speaks phoneme strings with the Kitten TTS 0.8 C library.
 *
 *   kt08_example ARCH.json WEIGHTS.bin VOICE.bin PHONEMES.txt OUT_DIR [THREADS] [SPEED] [MEMORY]
 *
 * PHONEMES.txt holds one chunk per line: "TEXT<TAB>PHONEMES", where PHONEMES is espeak-ng "en-us" output with stress
 * marks and punctuation (as the Python package's phonemizer gives) and TEXT is the chunk's normalised text, which picks
 * the voice style row as in Python. A line without a tab is all phonemes (style row = token count). For line N the
 * program writes OUT_DIR/NNN.wav (32-bit float WAV, 24 kHz) and prints the token count, time and real-time factor.
 * THREADS: 0 (default) lets the library choose. MEMORY: default (the library's default for this platform), latency,
 * low or extreme.
 * Kitten TTS 2 mini, nano and pico: set KT08_EXAMPLE_VOCAB to the model repository's vocab.json. PHONEMES are then the
 * model's own front-end output (Python: KittenTTS(...)._model.phonemize(text, language)), ids come from that vocab,
 * the style row from the token count, and kt08_synthesize_tiny trims and fades each chunk (line 0 is the first chunk).
 * Text in (Kitten TTS 0.8, English): KT08_EXAMPLE_TEXT=DIR, a folder with espeak-ng's en_rules and en_list
 * (fetch_phonemizer_data.sh), without KT08_EXAMPLE_VOCAB. PHONEMES.txt then holds plain text, one or more sentences per
 * line; the library splits it into chunks and phonemises them (kt08_text_chunks, kt08_text_to_phonemes), and each chunk
 * becomes one NNN.wav.
 * Text in (Kitten TTS 2 mini, nano and pico): KT08_EXAMPLE_TEXT=G2P_DIR together with KT08_EXAMPLE_VOCAB makes
 * PHONEMES.txt a file of text, one paragraph per line; the library's front end turns each line into phoneme chunks
 * (kt08_tiny_text_to_phonemes) and the chunks are spoken as above. G2P_DIR is the model repository's g2p/ folder plus
 * the files of export_phone_data.py; KT08_EXAMPLE_PHONEMIZER is the model's "phonemizer" (v1: nano, pico; v2: mini;
 * default v1), KT08_EXAMPLE_LANG the language (default en) and KT08_EXAMPLE_ESPEAK the folder of espeak-ng data files
 * for languages other than English (fetch_phonemizer_data.sh).
 * KT08_EXAMPLE_IDS=1 also prints each line's token ids. KT08_EXAMPLE_REPEAT=N runs every line N more times
 * (timing only; the first pass is the one written). KT08_EXAMPLE_PIN=N pins to the N fastest cores first.
 * Streaming: KT08_EXAMPLE_POLICY=legacy (default: each line one chunk, as above), trim (the same chunks, line 0 without
 * its silent start, cut by this program with kt08_stream's rule: 20 ms before the first sample at or above -50 dBFS)
 * or stream (kt08_stream_open, or kt08_stream_open_tts08 without KT08_EXAMPLE_VOCAB: needs KT08_EXAMPLE_TEXT; the whole
 * input file is one text, its lines joined with spaces; the library cuts it into chunks, phonemises each when it is
 * next and writes OUT_DIR/NNN.wav per chunk it returns; KT08_EXAMPLE_CHUNK_CHARS=N sets kt08_stream_options.chunk_chars).
 * Every policy prints READY lines (time since the request started when each chunk was returned) and a REQUEST line:
 * first audible sound (the first chunk's return time plus its first sample at or above -50 dBFS, as a player starting
 * at once would play it; text -> phonemes is included for stream and excluded for legacy and trim, whose text is
 * phonemised before the request starts) and the playback stalls of a player that plays each chunk when it arrives with
 * 0.15 s between chunks.
 * Qualcomm NPU (Kitten TTS 2 mini/nano/pico, library built with the QNN SDK): KT08_EXAMPLE_NPU=DIR, the folder
 * prepare_npu.py wrote for this phone's chip; WEIGHTS.bin must then be the fp32 weights. Chunks longer than the NPU
 * graphs run on the CPU model. KT08_EXAMPLE_QNN_LIBS=DIR: where Qualcomm's libraries are (default: library path).
 */
#include "kitten_tts08.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void put_u32(FILE* f, unsigned v) {
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    fwrite(b, 1, 4, f);
}

static void put_u16(FILE* f, unsigned v) {
    unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
    fwrite(b, 1, 2, f);
}

/* 32-bit float mono WAV (format 3). The samples are written as the library returns them. */
static int write_wav(const char* path, const float* x, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    const unsigned data = (unsigned)(n * 4);
    fwrite("RIFF", 1, 4, f); put_u32(f, 36 + data); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); put_u32(f, 16); put_u16(f, 3); put_u16(f, 1);
    put_u32(f, KT08_SAMPLE_RATE); put_u32(f, KT08_SAMPLE_RATE * 4); put_u16(f, 4); put_u16(f, 32);
    fwrite("data", 1, 4, f); put_u32(f, data);
    fwrite(x, 4, n, f);  /* little-endian hosts (Android, iOS, x86, ARM Linux) */
    return fclose(f);
}

/* A field of /proc/self/status in kB (VmHWM: peak resident memory; RssAnon: heap and other private memory;
 * RssFile: file pages mapped in, such as the library's code), or -1 where there is no /proc. */
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

static long peak_rss_kb(void) { return status_kb("VmHWM"); }

/* A scratch file for the text modes. tmpfile() fails inside Android apps (no writable temporary folder), so fall back to
 * a file in OUT_DIR, removed at once (it stays usable until closed). */
static FILE* scratch_file(const char* dir) {
    FILE* f = tmpfile();
    if (f) return f;
    char path[4096];
    snprintf(path, sizeof path, "%s/.kt08_example_chunks", dir);
    f = fopen(path, "w+b");
    if (f) remove(path);
    return f;
}

/* Playback of a player that starts each chunk when it is returned, with 0.15 s between chunks (READY / REQUEST). */
typedef struct {
    double t_start, play_end, first_audible_ms;
    int chunks, stalls;
    double stall_ms;
} playback;

static void playback_chunk(playback* p, const float* x, size_t n, double t_ready) {
    const double t = t_ready - p->t_start;
    printf("READY idx=%d t_ms=%.1f samples=%zu\n", p->chunks, t * 1e3, n);
    if (p->chunks == 0) {
        size_t i = 0;
        while (i < n && !(x[i] >= 0.0031622776f || x[i] <= -0.0031622776f)) ++i;  /* -50 dBFS */
        p->first_audible_ms = (t + (double)i / KT08_SAMPLE_RATE) * 1e3;
        p->play_end = t + (double)n / KT08_SAMPLE_RATE;
    } else {
        const double due = p->play_end + 0.15;
        if (t > due) { ++p->stalls; p->stall_ms += (t - due) * 1e3; }
        p->play_end = (t > due ? t : due) + (double)n / KT08_SAMPLE_RATE;
    }
    ++p->chunks;
}

/* CPU time this process has used so far, in seconds (all threads, user + system). */
static double cpu_seconds(void) {
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
    return (double)u.ru_utime.tv_sec + 1e-6 * (double)u.ru_utime.tv_usec + (double)u.ru_stime.tv_sec +
           1e-6 * (double)u.ru_stime.tv_usec;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s ARCH.json WEIGHTS.bin VOICE.bin PHONEMES.txt OUT_DIR [THREADS] [SPEED] [MEMORY]\n", argv[0]);
        return 2;
    }
    const int threads = argc > 6 ? atoi(argv[6]) : 0;
    const float speed = argc > 7 ? (float)atof(argv[7]) : 1.0f;
    const char* mem = argc > 8 ? argv[8] : "default";
    const char* repeat_env = getenv("KT08_EXAMPLE_REPEAT");
    const int repeat = repeat_env ? atoi(repeat_env) : 0;
    const int show_ids = getenv("KT08_EXAMPLE_IDS") != NULL;
    const char* pin = getenv("KT08_EXAMPLE_PIN");
    const char* vocab_path = getenv("KT08_EXAMPLE_VOCAB");
    kt08_vocab* vocab = NULL;
    if (vocab_path && kt08_vocab_load(vocab_path, &vocab) != KT08_OK) {
        fprintf(stderr, "vocab: %s\n", kt08_last_error());
        return 1;
    }

    const char* policy = getenv("KT08_EXAMPLE_POLICY");
    if (!policy || !*policy) policy = "legacy";
    if (strcmp(policy, "legacy") && strcmp(policy, "trim") && strcmp(policy, "stream")) {
        fprintf(stderr, "KT08_EXAMPLE_POLICY must be legacy, trim or stream\n");
        return 2;
    }
    const int trim_first = strcmp(policy, "legacy") != 0;  /* trim: line 0 without its silent start */

    if (pin && atoi(pin) > 0) printf("PIN cores=%d\n", kt08_pin_fastest_cores(atoi(pin)));

    kt08_options opt;
    kt08_options_init(&opt);
    opt.threads = threads;
    if (!strcmp(mem, "latency")) opt.memory_mode = KT08_MEMORY_LATENCY;
    else if (!strcmp(mem, "low")) opt.memory_mode = KT08_MEMORY_LOW;
    else if (!strcmp(mem, "extreme")) opt.memory_mode = KT08_MEMORY_EXTREME_LOW;
    static const char* const mode_names[] = {"latency", "low", "extreme"};
    mem = mode_names[opt.memory_mode];
    opt.fp16_weights = getenv("KT08_EXAMPLE_FP16_WEIGHTS") != NULL;  /* LOSSY options, for tests */
    opt.fp16_compute = getenv("KT08_EXAMPLE_FP16_COMPUTE") != NULL;

    printf("INFO %s\n", kt08_build_info());
    double t0 = now_s();
    kt08_model* model = NULL;
    if (kt08_model_load(argv[1], argv[2], &opt, &model) != KT08_OK) {
        fprintf(stderr, "load failed: %s\n", kt08_last_error());
        return 1;
    }
    printf("LOAD s=%.3f max_tokens=%zu threads=%d memory=%s rss_peak_kb=%ld rss_anon_kb=%ld rss_file_kb=%ld\n",
           now_s() - t0, kt08_model_max_tokens(model), threads, mem, peak_rss_kb(), status_kb("RssAnon"),
           status_kb("RssFile"));

    kt08_npu* npu = NULL;
    const char* npu_dir = getenv("KT08_EXAMPLE_NPU");
    if (npu_dir && *npu_dir) {
        kt08_npu_options no;
        kt08_npu_options_init(&no);
        no.model_dir = npu_dir;
        no.weights_path = argv[2];
        no.qnn_lib_dir = getenv("KT08_EXAMPLE_QNN_LIBS");
        no.cpu_fallback = model;
        no.threads = threads;
        const double tn = now_s();
        if (!vocab) { fprintf(stderr, "KT08_EXAMPLE_NPU needs KT08_EXAMPLE_VOCAB (Kitten TTS 2 models)\n"); return 1; }
        if (kt08_npu_load(&no, &npu) != KT08_OK) {
            fprintf(stderr, "npu load failed: %s\n", kt08_last_error());
            return 1;
        }
        printf("NPU load_s=%.3f max_frames=%zu rss_peak_kb=%ld\n", now_s() - tn, kt08_npu_max_frames(npu), peak_rss_kb());
    }

    float* voice = NULL;
    size_t rows = 0;
    if (kt08_voice_load(argv[3], &voice, &rows) != KT08_OK) {
        fprintf(stderr, "voice: %s\n", kt08_last_error());
        return 1;
    }

    FILE* in = fopen(argv[4], "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[4]); return 1; }
    const char* text_dir = getenv("KT08_EXAMPLE_TEXT");
    const int streaming = !strcmp(policy, "stream");
    if (streaming && !(text_dir && *text_dir)) {
        fprintf(stderr, "KT08_EXAMPLE_POLICY=stream needs KT08_EXAMPLE_TEXT (text in)\n");
        return 2;
    }
    if (text_dir && *text_dir && !vocab && !streaming) {
        /* Text in: every line of the input is text, turned into "CHUNK<TAB>PHONEMES" lines by the library. */
        char rules[4096], list[4096];
        snprintf(rules, sizeof rules, "%s/en_rules", text_dir);
        snprintf(list, sizeof list, "%s/en_list", text_dir);
        const long rss_before = status_kb("VmRSS");
        const double tp = now_s();
        kt08_phonemizer* ph = NULL;
        if (kt08_phonemizer_load(rules, list, &ph) != KT08_OK) {
            fprintf(stderr, "phonemizer: %s\n", kt08_last_error());
            return 1;
        }
        printf("PHONEMIZER load_s=%.3f rss_added_kb=%ld\n", now_s() - tp, status_kb("VmRSS") - rss_before);
        FILE* tmp = scratch_file(argv[5]);
        if (!tmp) { fprintf(stderr, "cannot create a scratch file\n"); return 1; }
        static char text[8192];
        char *chunks = NULL, *phon = NULL;
        size_t n_chunks = 0;
        double text_s = 0;
        while (fgets(text, sizeof text, in)) {
            text[strcspn(text, "\r\n")] = 0;
            size_t nc = 0, np = 0;
            const double tt = now_s();
            if (kt08_text_chunks(text, NULL, 0, &nc) != KT08_OK ||
                kt08_text_to_phonemes(ph, text, NULL, 0, &np) != KT08_OK) {
                fprintf(stderr, "text: %s\n", kt08_last_error());
                return 1;
            }
            chunks = (char*)realloc(chunks, nc);
            phon = (char*)realloc(phon, np);
            if (!chunks || !phon || kt08_text_chunks(text, chunks, nc, &nc) != KT08_OK ||
                kt08_text_to_phonemes(ph, text, phon, np, &np) != KT08_OK) {
                fprintf(stderr, "text: %s\n", kt08_last_error());
                return 1;
            }
            text_s += now_s() - tt;
            /* both strings hold one line per chunk, in the same order */
            for (char *c = chunks, *p = phon; *c || *p;) {
                char* ce = strchr(c, '\n');
                char* pe = strchr(p, '\n');
                if (ce) *ce = 0;
                if (pe) *pe = 0;
                if (*c) { fprintf(tmp, "%s\t%s\n", c, p); ++n_chunks; }
                if (!ce || !pe) break;
                c = ce + 1;
                p = pe + 1;
            }
        }
        printf("TEXT chunks=%zu text_to_phonemes_ms=%.3f\n", n_chunks, 1e3 * text_s);
        free(chunks);
        free(phon);
        kt08_phonemizer_free(ph);
        fclose(in);
        rewind(tmp);
        in = tmp;
    }
    char line[8192];
    const char* text_g2p = getenv("KT08_EXAMPLE_TEXT");
    if (text_g2p && *text_g2p && vocab && !streaming) {
        /* text mode: convert every line to phoneme chunks first, then speak them as phoneme lines */
        const char* phon = getenv("KT08_EXAMPLE_PHONEMIZER");
        const char* lang = getenv("KT08_EXAMPLE_LANG");
        const char* espeak_dir = getenv("KT08_EXAMPLE_ESPEAK");
        kt08_tiny_frontend* fe = NULL;
        const double tf = now_s();
        if (kt08_tiny_frontend_load(phon && *phon ? phon : "v1", vocab_path, text_g2p,
                                   espeak_dir && *espeak_dir ? espeak_dir : NULL, &fe) != KT08_OK) {
            fprintf(stderr, "front end: %s\n", kt08_last_error());
            return 1;
        }
        printf("FRONTEND load_s=%.3f rss_peak_kb=%ld\n", now_s() - tf, peak_rss_kb());
        FILE* chunks = scratch_file(argv[5]);
        if (!chunks) { fprintf(stderr, "cannot create a scratch file\n"); return 1; }
        int n_text = 0;
        while (fgets(line, sizeof line, in)) {
            line[strcspn(line, "\r\n")] = 0;
            char* out = NULL;
            const double t = now_s();
            if (kt08_tiny_text_to_phonemes(fe, line, lang && *lang ? lang : "en", &out) != KT08_OK) {
                fprintf(stderr, "text line %d: %s\n", n_text, kt08_last_error());
                kt08_tiny_frontend_free(fe);
                return 1;
            }
            int n_chunks = 0;
            for (char* c = out; *out && c; ++n_chunks) {
                char* nl = strchr(c, '\n');
                if (nl) *nl = 0;
                fprintf(chunks, "x\t%s\n", c);
                c = nl ? nl + 1 : NULL;
            }
            printf("TEXT idx=%d chunks=%d ms=%.3f\n", n_text++, n_chunks, (now_s() - t) * 1e3);
            kt08_free(out);
        }
        kt08_tiny_frontend_free(fe);
        fclose(in);
        rewind(chunks);
        in = chunks;
    }
    int n_line = 0, failures = 0;
    double audio_s = 0, compute_s = 0;
    playback pb;
    memset(&pb, 0, sizeof pb);
    pb.t_start = now_s();
    if (streaming) {
        /* kt08_stream over the whole input text */
        char* text = (char*)calloc(1, 1);
        size_t len = 0;
        while (text && fgets(line, sizeof line, in)) {
            line[strcspn(line, "\r\n")] = 0;
            const size_t k = strlen(line);
            text = (char*)realloc(text, len + k + 2);
            if (!text) break;
            if (len) text[len++] = ' ';
            memcpy(text + len, line, k + 1);
            len += k;
        }
        fclose(in);
        in = NULL;
        if (!text) { fprintf(stderr, "out of memory\n"); return 1; }
        kt08_stream_options so;
        kt08_stream_options_init(&so);
        const char* cc = getenv("KT08_EXAMPLE_CHUNK_CHARS");
        if (cc && *cc) so.chunk_chars = atoi(cc);
        kt08_phonemizer* ph = NULL;
        kt08_tiny_frontend* fe = NULL;
        const char* lang = getenv("KT08_EXAMPLE_LANG");
        if (vocab) {
            const char* phon = getenv("KT08_EXAMPLE_PHONEMIZER");
            const char* espeak_dir = getenv("KT08_EXAMPLE_ESPEAK");
            if (kt08_tiny_frontend_load(phon && *phon ? phon : "v1", vocab_path, text_dir,
                                       espeak_dir && *espeak_dir ? espeak_dir : NULL, &fe) != KT08_OK) {
                fprintf(stderr, "front end: %s\n", kt08_last_error());
                return 1;
            }
        } else {
            char rules[4096], list[4096];
            snprintf(rules, sizeof rules, "%s/en_rules", text_dir);
            snprintf(list, sizeof list, "%s/en_list", text_dir);
            if (kt08_phonemizer_load(rules, list, &ph) != KT08_OK) {
                fprintf(stderr, "phonemizer: %s\n", kt08_last_error());
                return 1;
            }
        }
        kt08_stream* st = NULL;
        pb.t_start = now_s();
        const kt08_status so_st =
            vocab ? kt08_stream_open(model, vocab, fe, text, lang && *lang ? lang : "en", voice, rows, speed, &so, &st)
                  : kt08_stream_open_tts08(model, ph, text, voice, rows, speed, &so, &st);
        if (so_st != KT08_OK) {
            fprintf(stderr, "stream: %s\n", kt08_last_error());
            return 1;
        }
        char* plan = NULL;
        if (kt08_stream_chunks(st, &plan) == KT08_OK) {
            int k = 0;
            for (char* c = plan; *plan && c; ++k) {
                char* nl = strchr(c, '\n');
                if (nl) *nl = 0;
                printf("PLAN idx=%d chars=%zu text=%s\n", k, strlen(c), c);
                c = nl ? nl + 1 : NULL;
            }
            kt08_free(plan);
        }
        int last = 0;
        while (!last) {
            float* audio = NULL;
            size_t samples = 0;
            const double t = now_s();
            if (kt08_stream_next(st, &audio, &samples, &last) != KT08_OK) {
                fprintf(stderr, "chunk %d: %s\n", n_line, kt08_last_error());
                ++failures;
                break;
            }
            if (!audio) break;  /* nothing (more) to say */
            const double t1 = now_s();
            playback_chunk(&pb, audio, samples, t1);
            const double sec = (double)samples / KT08_SAMPLE_RATE;
            printf("CHUNK pass=0 idx=%d samples=%zu ms=%.1f rtf=%.4f\n", n_line, samples, (t1 - t) * 1e3,
                   sec > 0 ? (t1 - t) / sec : 0.0);
            char path[4096];
            snprintf(path, sizeof path, "%s/%03d.wav", argv[5], n_line);
            if (write_wav(path, audio, samples) != 0) { fprintf(stderr, "cannot write %s\n", path); ++failures; }
            audio_s += sec;
            compute_s += t1 - t;
            kt08_free(audio);
            ++n_line;
        }
        kt08_stream_free(st);
        if (fe) kt08_tiny_frontend_free(fe);
        if (ph) kt08_phonemizer_free(ph);
        free(text);
    }
    while (in && fgets(line, sizeof line, in)) {
        line[strcspn(line, "\r\n")] = 0;
        char* phonemes = line;
        char* tab = strchr(line, '\t');
        if (tab) { *tab = 0; phonemes = tab + 1; }
        size_t n_ids = 0;
        if ((vocab ? kt08_vocab_ids(vocab, phonemes, NULL, 0, &n_ids) : kt08_phonemes_to_ids(phonemes, NULL, 0, &n_ids))
                != KT08_OK) {
            fprintf(stderr, "line %d: %s\n", n_line, kt08_last_error());
            ++failures; ++n_line; continue;
        }
        int32_t* ids = (int32_t*)malloc(n_ids * sizeof(int32_t));
        if (vocab) kt08_vocab_ids(vocab, phonemes, ids, n_ids, &n_ids);
        else kt08_phonemes_to_ids(phonemes, ids, n_ids, &n_ids);
        const size_t style_row = vocab ? kt08_style_row_for_ids(n_ids, rows)
                                       : tab ? kt08_style_row_for_text(line, rows) : n_ids;
        if (show_ids) {
            printf("IDS idx=%d", n_line);
            for (size_t i = 0; i < n_ids; ++i) printf(" %d", ids[i]);
            printf("\n");
        }
        for (int pass = 0; pass <= repeat; ++pass) {
            float* audio = NULL;
            size_t samples = 0;
            const double t = now_s();
            int on_npu = 0;
            const kt08_status st = npu ? kt08_synthesize_tiny_npu(npu, ids, n_ids, voice, rows, style_row, speed,
                                                                  n_line == 0, &audio, &samples, &on_npu)
                                 : vocab ? kt08_synthesize_tiny(model, ids, n_ids, voice, rows, style_row, speed,
                                                                n_line == 0, &audio, &samples)
                                         : kt08_synthesize(model, ids, n_ids, voice, rows, style_row, speed, &audio, &samples);
            if (st == KT08_OK && n_line == 0 && trim_first) {
                /* trim policy: drop the silent start as kt08_stream does for a stream's first chunk */
                size_t i = 0;
                while (i < samples && !(audio[i] >= 0.0031622776f || audio[i] <= -0.0031622776f)) ++i;
                const size_t off = i < samples && i > 480 ? i - 480 : 0;
                if (off) { memmove(audio, audio + off, (samples - off) * sizeof(float)); samples -= off; }
            }
            const double dt = now_s() - t;
            if (st != KT08_OK) {
                fprintf(stderr, "line %d: %s\n", n_line, kt08_last_error());
                ++failures;
                break;
            }
            const double sec = (double)samples / KT08_SAMPLE_RATE;
            printf("CHUNK pass=%d idx=%d tokens=%zu samples=%zu ms=%.1f rtf=%.4f npu=%d\n", pass, n_line, n_ids, samples,
                   dt * 1e3, sec > 0 ? dt / sec : 0.0, on_npu);
            if (pass == 0) {
                playback_chunk(&pb, audio, samples, t + dt);
                char path[4096];
                snprintf(path, sizeof path, "%s/%03d.wav", argv[5], n_line);
                if (write_wav(path, audio, samples) != 0) { fprintf(stderr, "cannot write %s\n", path); ++failures; }
                audio_s += sec;
                compute_s += dt;
            }
            kt08_free(audio);
        }
        free(ids);
        ++n_line;
    }
    if (in) fclose(in);
    printf("REQUEST policy=%s chunks=%d first_audible_ms=%.1f stalls=%d stall_ms=%.1f\n", policy, pb.chunks,
           pb.first_audible_ms, pb.stalls, pb.stall_ms);
    printf("DONE lines=%d failures=%d audio_s=%.2f compute_s=%.2f rtf=%.4f rss_peak_kb=%ld cpu_s=%.2f\n", n_line, failures,
           audio_s, compute_s, audio_s > 0 ? compute_s / audio_s : 0.0, peak_rss_kb(), cpu_seconds());
    kt08_free(voice);
    kt08_vocab_free(vocab);
    kt08_npu_free(npu);
    kt08_model_free(model);
    return failures ? 1 : 0;
}
