/*
 * kitten_tts08.h - C API of the Kitten TTS 0.8 engine, for apps on Android, iOS and other platforms
 * that do not use the Python package.
 *
 * The library turns phonemes into 24 kHz mono float audio. It can also turn text into those phonemes, as the Python
 * package does: English for Kitten TTS 0.8 (API version 4, kt08_phonemizer_* and kt08_text_*), and English, Spanish
 * and Italian for Kitten TTS 2 mini, nano and pico (API version 5, kt08_tiny_frontend_* and kt08_tiny_text_*). The
 * data files these read are downloaded by the app: see engines/lite/mobile/README.md.
 *
 * Thread safety: one kt08_model may be used by one thread at a time. Different models may run at the
 * same time on different threads. Loading is serialised inside the library.
 *
 * All functions return KT08_OK (0) on success. On failure, kt08_last_error() gives a message for the
 * calling thread.
 */
#ifndef KITTEN_TTS08_H
#define KITTEN_TTS08_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(KT08_BUILD_SHARED)
#    define KT08_API __declspec(dllexport)
#  elif defined(KT08_USE_SHARED)
#    define KT08_API __declspec(dllimport)
#  else
#    define KT08_API
#  endif
#else
#  define KT08_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Version of this header. The library reports its own with kt08_api_version(). */
#define KT08_API_VERSION 5  /* 2: Kitten TTS 2 mini, nano and pico (kt08_vocab_*, kt08_synthesize_tiny);
                               * 3: optional Qualcomm NPU path for them (kt08_npu_*);
                               * 4: English text to phonemes for Kitten TTS 0.8 (kt08_phonemizer_*, kt08_text_*);
                               * 5: text front end for Kitten TTS 2 mini, nano and pico (kt08_tiny_frontend_*) */

/* Output sample rate in Hz. */
#define KT08_SAMPLE_RATE 24000

/* Floats per voice style row (128 for the decoder, then 128 for the duration predictor). */
#define KT08_STYLE_DIM 256

typedef enum kt08_status {
    KT08_OK = 0,
    KT08_ERR_ARGUMENT = 1,  /* a NULL pointer, empty input or value out of range */
    KT08_ERR_IO = 2,        /* a file could not be opened or read, or has the wrong format */
    KT08_ERR_TOO_LONG = 3,  /* more phoneme tokens than the model's position table holds: split the text */
    KT08_ERR_RUNTIME = 4,   /* the engine reported an error (message in kt08_last_error) */
    KT08_ERR_MEMORY = 5,    /* out of memory */
    KT08_ERR_UNSUPPORTED = 6 /* this build or this device cannot do it (kt08_npu_load): use the CPU functions */
} kt08_status;

typedef enum kt08_memory_mode {
    KT08_MEMORY_LATENCY = 0,     /* fastest; keeps buffers between calls (default on desktop and server builds) */
    KT08_MEMORY_LOW = 1,         /* frees working buffers after each call; on ARM also skips Winograd copies */
    KT08_MEMORY_EXTREME_LOW = 2  /* also frees per-layer scratch (default on Android and iOS) */
} kt08_memory_mode;

typedef struct kt08_options {
    /* sizeof(kt08_options), set by kt08_options_init. Lets older apps run with newer libraries. */
    uint32_t struct_size;
    /* CPU threads per synthesis call. 0 (default): chosen by the library.
     *   Android: the fast cores, at most 2, on phones with fast and slow cores; every core, up to 8, on phones whose
     *   cores are all the same design (for example only Cortex-A53). Each call then runs on those cores; the calling
     *   thread gets its own CPU set back when the call returns.
     *   iOS: the performance cores, at most 2.
     *   Other platforms: the OpenMP default (OMP_NUM_THREADS, else one per CPU).
     * Any other value is used as given. */
    int32_t threads;
    kt08_memory_mode memory_mode;
    /* LOSSY, off by default (0). 1: store weights in 16-bit floats where that is exact, and widen them per layer.
     * With the stock weights file almost nothing changes; with a weights file rounded by
     * engines/lite/mobile/round_weights_fp16.py it cuts weight memory by about 40% (mini: 297 -> about 180 MiB),
     * and the audio changes (lossy: the weights were rounded). */
    int32_t fp16_weights;
    /* LOSSY, off by default (0). 1: run most convolutions in 16-bit float arithmetic (Android arm64 with
     * FEAT_FP16 only, for example Cortex-A55/A76/A78; ignored elsewhere). About 1.3x faster than low-memory
     * mode on those phones; the audio changes. */
    int32_t fp16_compute;
} kt08_options;

typedef struct kt08_model kt08_model;

/* Fills *options with the defaults: threads 0 (chosen by the library), no lossy options, and the memory mode
 * KT08_MEMORY_EXTREME_LOW on Android and iOS, KT08_MEMORY_LATENCY elsewhere. */
KT08_API void kt08_options_init(kt08_options* options);

/* Loads a model from its architecture file (model_defs/kitten_*_arch.json) and weights file (W1X1 .bin).
 * options may be NULL (defaults). On success *out_model must be freed with kt08_model_free.
 * On Android and Linux the weights file is memory-mapped while loading (elsewhere it is read with stdio) and closed
 * before the call returns; the model keeps its own copy of the weights. */
KT08_API kt08_status kt08_model_load(const char* arch_json_path, const char* weights_path,
                                     const kt08_options* options, kt08_model** out_model);

KT08_API void kt08_model_free(kt08_model* model);

/* Most phoneme tokens (including the start and end tokens) one kt08_synthesize call accepts, or 0 if unknown. */
KT08_API size_t kt08_model_max_tokens(const kt08_model* model);

/* Converts a UTF-8 phoneme string into the model's token ids, exactly as the Python package does: the string is
 * split into words and punctuation marks, joined with single spaces, mapped to symbol ids (unknown symbols are
 * dropped), and wrapped in the start token 0 and the end tokens 10, 0.
 * Writes at most capacity ids. *out_count receives the number needed; if it is larger than capacity, the call
 * returns KT08_ERR_ARGUMENT and writes nothing, so a first call with capacity 0 gives the size. */
KT08_API kt08_status kt08_phonemes_to_ids(const char* phonemes_utf8, int32_t* ids, size_t capacity,
                                          size_t* out_count);

/* Loads a voice file (raw little-endian float32, rows of KT08_STYLE_DIM floats, as the files in voices_kitten_80m/ on the
 * model's Hugging Face page). Free *out_voice with kt08_free. */
KT08_API kt08_status kt08_voice_load(const char* path, float** out_voice, size_t* out_rows);

/* Voice style row the Python package uses for a chunk: the chunk's text length in characters (Unicode code points,
 * after the package's text normalisation), capped at voice_rows - 1. Pass the result to kt08_synthesize. */
KT08_API size_t kt08_style_row_for_text(const char* chunk_text_utf8, size_t voice_rows);

/* Synthesises one chunk (one or a few sentences) from token ids.
 * voice: voice_rows rows of KT08_STYLE_DIM floats; style_row picks the row (capped at voice_rows - 1): use
 * kt08_style_row_for_text() to match the Python package.
 * speed: 1.0 is normal; nano models have a per-voice speed factor the app multiplies in (see the mobile README).
 * On success *out_audio holds *out_samples floats at KT08_SAMPLE_RATE; free it with kt08_free. */
KT08_API kt08_status kt08_synthesize(kt08_model* model, const int32_t* ids, size_t n_ids, const float* voice,
                                     size_t voice_rows, size_t style_row, float speed, float** out_audio,
                                     size_t* out_samples);

/* Frees memory returned by this library (audio, voices). NULL is ignored. */
KT08_API void kt08_free(void* p);

/* Restricts the calling thread, and threads it creates later, to the n fastest CPU cores (Android/Linux; uses
 * cpu_capacity). Call it once, before the first kt08_model_load, from the thread that will run synthesis.
 * On phones with 2 fast and 6 slow cores, n = 2 is usually fastest. Returns the number of cores picked, or 0 when
 * the platform does not support it (iOS, macOS, Windows). Off unless called. */
KT08_API int kt08_pin_fastest_cores(int n);

/* Message for the last failed call on this thread ("" if none). Valid until the next call on this thread. */
KT08_API const char* kt08_last_error(void);

KT08_API int kt08_api_version(void);

/* ---- Kitten TTS 2 mini, nano and pico (KittenML/kitten-tts-2-mini, -nano, -pico), API version 2 ----------------
 * These models load with kt08_model_load like the 0.8 models: convert the official Hugging Face files once with the
 * Python package (python -m kitten_inference.tts2_tiny.convert REPO_DIR OUT_DIR, numpy only), which writes
 * <model>_fp32_arch.json / <model>_fp32.bin (and _int8_), and voices/<language>/<voice>.bin for kt08_voice_load.
 * Token ids come from the repository's vocab.json (kt08_vocab_*), not from kt08_phonemes_to_ids. The phonemes come
 * from the model's own text front end (misaki and espeak-ng, see engines/lite/mobile/README.md). */

typedef struct kt08_vocab kt08_vocab;

/* Loads a repository's vocab.json (phoneme symbol -> id). Free with kt08_vocab_free. */
KT08_API kt08_status kt08_vocab_load(const char* vocab_json_path, kt08_vocab** out_vocab);
KT08_API void kt08_vocab_free(kt08_vocab* vocab);

/* Token ids as the Python package (and kittenml) make them: 0, then one id per phoneme character found in the vocab
 * ('(' and ')' become ',' when the vocab lacks them; other unknown characters are dropped), then 10 and 0.
 * capacity / *out_count work as in kt08_phonemes_to_ids. */
KT08_API kt08_status kt08_vocab_ids(const kt08_vocab* vocab, const char* phonemes_utf8, int32_t* ids, size_t capacity,
                                    size_t* out_count);

/* Voice style row for a chunk of n_ids token ids: min(n_ids - 1, voice_rows - 1). */
KT08_API size_t kt08_style_row_for_ids(size_t n_ids, size_t voice_rows);

/* Synthesises one chunk with a Kitten TTS 2 mini/nano/pico model and finishes it as the Python package does: the end
 * token's frames are cut, and the start pause too unless first_chunk is non-zero; samples are clipped to [-1, 1];
 * 5 ms fades in and out. Between chunks the package puts 0.15 s of silence (3600 samples). The excitation noise is
 * seeded from the chunk's pitch track, so the same input gives the same audio with the same library build and thread
 * count (another build or thread count can change the last bits of the pitch, and so the noise).
 * speed: 1.0 is normal, from KT08_TINY_SPEED_MIN to KT08_TINY_SPEED_MAX (0.25 to 4; outside it the call returns
 * KT08_ERR_ARGUMENT: a very small speed makes one chunk's audio, and memory, grow without bound).
 * Free *out_audio with kt08_free. */
#define KT08_TINY_SPEED_MIN 0.25f
#define KT08_TINY_SPEED_MAX 4.0f
KT08_API kt08_status kt08_synthesize_tiny(kt08_model* model, const int32_t* ids, size_t n_ids, const float* voice,
                                          size_t voice_rows, size_t style_row, float speed, int first_chunk,
                                          float** out_audio, size_t* out_samples);

/* ---- Qualcomm NPU for Kitten TTS 2 mini, nano and pico (optional, API version 3) --------------------------------
 * Runs the decoder and vocoder, most of the work, on the NPU of a Snapdragon phone (Hexagon, through Qualcomm's QNN
 * runtime) and the rest (text encoder, durations, pitch, inverse STFT) on the CPU. Opt-in: nothing changes for apps
 * that do not call these functions. Use it to keep the CPU free; it is not faster than the CPU path on every phone
 * (see engines/lite/mobile/README.md, "Qualcomm NPU").
 * Needs: a library built with the QNN SDK (build_android.sh with KITTEN_QNN_SDK set; other builds return
 * KT08_ERR_UNSUPPORTED); a model folder made for the phone's chip by engines/lite/mobile/npu/prepare_npu.py; and
 * Qualcomm's runtime libraries, which the app ships itself (they are not part of this library).
 * Same thread rules as kt08_model: a kt08_npu is used by one thread at a time. The cpu_fallback model runs inside
 * kt08_synthesize_tiny_npu, so no other thread may use that model meanwhile. */
typedef struct kt08_npu kt08_npu;

typedef struct kt08_npu_options {
    /* sizeof(kt08_npu_options), set by kt08_npu_options_init. */
    uint32_t struct_size;
    /* Folder written by prepare_npu.py for this chip (npu.json, the CPU graphs and the NPU graphs). Required. */
    const char* model_dir;
    /* The model's fp32 weights file (<model>_fp32.bin from the converter), the same file the CPU path loads. Required. */
    const char* weights_path;
    /* Folder holding libQnnHtp.so, libQnnSystem.so and the libQnnHtpV<NN>Stub.so / libQnnHtpV<NN>Skel.so pair for
     * the chip. NULL (default): the normal library search path, for example the app's native library folder. */
    const char* qnn_lib_dir;
    /* Optional CPU model (kt08_model_load of the same model) for chunks longer than the NPU graphs (see
     * kt08_npu_max_frames). NULL (default): such chunks fail with KT08_ERR_TOO_LONG. */
    kt08_model* cpu_fallback;
    /* CPU threads for the CPU parts; 0 (default): chosen by the library, as kt08_options.threads. */
    int32_t threads;
    /* 1 (default): ask the NPU for its highest clock while this object lives (fastest, uses more power).
     * 0: leave the NPU's clock to the system. */
    int32_t high_performance;
} kt08_npu_options;

/* Fills *options with the defaults (all NULL / 0, high_performance 1). */
KT08_API void kt08_npu_options_init(kt08_npu_options* options);

/* Loads the NPU path. Returns KT08_ERR_UNSUPPORTED (and a reason in kt08_last_error) when this library was built
 * without the QNN SDK, Qualcomm's libraries cannot be loaded, or the NPU graphs were made for another chip or QNN
 * version: the app then uses kt08_synthesize_tiny on a CPU model. Free *out with kt08_npu_free. */
KT08_API kt08_status kt08_npu_load(const kt08_npu_options* options, kt08_npu** out);
KT08_API void kt08_npu_free(kt08_npu* npu);

/* Longest chunk the NPU graphs take, in frames of 25 ms (160 = 4 s of audio), or 0. Longer chunks go to
 * options.cpu_fallback. */
KT08_API size_t kt08_npu_max_frames(const kt08_npu* npu);

/* As kt08_synthesize_tiny, with the decoder and vocoder on the NPU. *out_on_npu (may be NULL) is set to 1 when the
 * chunk ran on the NPU and 0 when it went to the CPU fallback. The NPU graphs are fp16 (as prepare_npu.py makes
 * them), so the audio is close to, not identical with, the CPU path's. */
KT08_API kt08_status kt08_synthesize_tiny_npu(kt08_npu* npu, const int32_t* ids, size_t n_ids, const float* voice,
                                              size_t voice_rows, size_t style_row, float speed, int first_chunk,
                                              float** out_audio, size_t* out_samples, int* out_on_npu);

/* ---- Text front end for Kitten TTS 2 mini, nano and pico (API version 5) -------------------------------------
 * Text in, phoneme chunks out, on the device, as the Python package's front end makes them (each chunk then goes to
 * kt08_vocab_ids and kt08_synthesize_tiny). English: text normalisation, misaki's lexicons, CMUdict and the small
 * BART for unknown words, with a rule-based part-of-speech tagger in place of spaCy's. Its data files are read once at
 * load from g2p_dir: the model repository's g2p/ folder (cmu_misaki_us.json, g2p_bart_en_us/) plus us_gold.json,
 * us_silver.json and spacy_en_exceptions.json written by
 * `python -m kitten_inference.tts2_tiny.frontend.export_phone_data DIR`. Languages other than English read espeak-ng's
 * data files (pronunciation rules, word lists and phoneme definitions; GPL-3.0 data, not part of this library) from
 * espeak_data_dir, which the app downloads once (engines/lite/mobile/fetch_phonemizer_data.sh); each language is
 * parsed once, on first use. engines/lite/mobile/README.md, "Text to phonemes", lists the supported languages with
 * their measured match rates; the others return KT08_ERR_UNSUPPORTED.
 * A kt08_tiny_frontend may be shared between threads (calls on one front end run one at a time). */
typedef struct kt08_tiny_frontend kt08_tiny_frontend;

/* phonemizer_version: "v1" (nano, pico) or "v2" (mini), the "phonemizer" field of the model's config.json.
 * vocab_json_path: the repository's vocab.json. g2p_dir: English data (NULL: no English). espeak_data_dir: the
 * folder fetch_phonemizer_data.sh fills, for the other languages (NULL: none). Free *out with
 * kt08_tiny_frontend_free. */
KT08_API kt08_status kt08_tiny_frontend_load(const char* phonemizer_version, const char* vocab_json_path,
                                             const char* g2p_dir, const char* espeak_data_dir,
                                             kt08_tiny_frontend** out);
KT08_API void kt08_tiny_frontend_free(kt08_tiny_frontend* fe);

/* Converts UTF-8 text into phoneme chunks of at most 350 symbols, joined with '\n' in *out_chunks (free it with
 * kt08_free; an empty string when the text has nothing to say). language: "en", or another code of the model
 * ("es", ...). Languages this build does not support return KT08_ERR_UNSUPPORTED. Text with more than 1000
 * characters in a row without a space, tab or line break gives KT08_ERR_TOO_LONG. Italian also reads the en, de, es
 * and fr files that fetch_phonemizer_data.sh fetches (for foreign words in the Italian word list): keep them all. */
KT08_API kt08_status kt08_tiny_text_to_phonemes(kt08_tiny_frontend* fe, const char* text_utf8, const char* language,
                                                char** out_chunks);

/* Build summary, for logs: version, CPU instruction-set level in use, OpenMP runtime. */
KT08_API const char* kt08_build_info(void);

/* ---- English text to phonemes for Kitten TTS 0.8 (API version 4) ----------------------------------------------
 * The same steps as the Python package (kitten_inference.tts08) with normalize=True: its text normalisation (numbers,
 * currency, times, units ... written out in words; contractions expanded; URLs and e-mail addresses removed; lower
 * case), its sentence chunking, then espeak-ng-compatible phonemes ("en-us", stress marks and punctuation kept) from
 * CEPhonemizer, KittenML's Apache-2.0 C++ reader of espeak-ng's English pronunciation rules.
 *
 * Data: the phonemiser reads two files of espeak-ng (GPL-3.0, so they are not part of this library): en_rules and
 * en_list from espeak-ng's dictsource folder at a pinned commit. The app downloads them once
 * (engines/lite/mobile/fetch_phonemizer_data.sh lists the URLs and SHA-256 sums) and passes their paths.
 *
 * Per chunk of kt08_text_chunks the app then calls: kt08_chunk_to_phonemes -> kt08_phonemes_to_ids ->
 * kt08_style_row_for_text(chunk) -> kt08_synthesize. Text goes in as UTF-8 (NFC, as iOS and Android strings normally are).
 *
 * Strings come back in a caller buffer: *out_size receives the bytes needed, including the final NUL. If that is
 * more than capacity, the call returns KT08_ERR_ARGUMENT and writes nothing, so a first call with out = NULL and
 * capacity 0 gives the size (kt08_chunk_to_phonemes and kt08_text_to_phonemes keep the last result in the
 * phonemiser, so the second call with the same text does not redo the work; kt08_text_chunks recomputes).
 * Text with more than 1000 characters in a row without a space gives KT08_ERR_TOO_LONG (the work would grow with
 * the square of that run).
 * Thread safety: a kt08_phonemizer is used by one thread at a time; kt08_text_chunks may run on any thread. */
typedef struct kt08_phonemizer kt08_phonemizer;

/* Loads the phonemiser from espeak-ng's en_rules and en_list files (parsed once, here). Free with
 * kt08_phonemizer_free. KT08_ERR_IO when a file cannot be read. */
KT08_API kt08_status kt08_phonemizer_load(const char* en_rules_path, const char* en_list_path,
                                          kt08_phonemizer** out_phonemizer);
KT08_API void kt08_phonemizer_free(kt08_phonemizer* phonemizer);

/* Normalises text as the Python package does with normalize=True (clean_text=True; the default of generate_to_file)
 * and splits it into the chunks the package synthesises one at a time (sentences, at most 400 characters, each
 * ending in a punctuation mark). Writes the chunks one per line ('\n' between them; none after the last); empty
 * text gives an empty string. Each chunk is also what kt08_style_row_for_text takes. Text that is not normalised
 * (the Python default of generate()) is not supported: its phonemes were not validated against the package. */
KT08_API kt08_status kt08_text_chunks(const char* text_utf8, char* out, size_t capacity, size_t* out_size);

/* Phonemes of one chunk from kt08_text_chunks (the chunk is not normalised again; pass only such chunks). */
KT08_API kt08_status kt08_chunk_to_phonemes(kt08_phonemizer* phonemizer, const char* chunk_utf8, char* out,
                                            size_t capacity, size_t* out_size);

/* Whole text: kt08_text_chunks, then kt08_chunk_to_phonemes for every chunk. Writes one line of phonemes per chunk,
 * in the order of kt08_text_chunks. */
KT08_API kt08_status kt08_text_to_phonemes(kt08_phonemizer* phonemizer, const char* text_utf8, char* out,
                                           size_t capacity, size_t* out_size);

#ifdef __cplusplus
}
#endif

#endif /* KITTEN_TTS08_H */
