# Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico on phones

A C library for Android and iOS apps: the same engine as the Python package, without Python. It runs Kitten TTS 0.8
and Kitten TTS 2 mini, nano and pico ([below](#kitten-tts-2-mini-nano-and-pico)). Give it phonemes, or text, and it
returns 24 kHz mono audio, sample for sample what the Python package makes from the same phonemes with the same build
settings. Text works on the phone in these languages:

| Model | Languages |
|---|---|
| Kitten TTS 0.8 | English |
| Kitten TTS 2 nano, pico | English, Spanish, Italian, Portuguese (Brazil), French, German, Hindi, Russian |
| Kitten TTS 2 mini | those, and Danish, Dutch, Finnish, Japanese, Korean, Swahili, Swedish, Turkish |

Swahili, Turkish, Finnish, Dutch and Russian (and Hindi and Russian on nano and pico) still have some pronunciation
errors on phones and are being corrected, so their output may change; for example, numbers and amounts may be read differently from the desktop
package.

## Build

Android (arm64-v8a, Android 7 or newer), with the Android NDK r26 or newer:

```sh
ANDROID_NDK=$HOME/Android/Sdk/ndk/27.2.12479018 engines/lite/mobile/build_android.sh out/android
```

This gives `out/android/lib/libkitten_tts08.so` (one library for every arm64 phone), `out/android/include/kitten_tts08.h`
and the example program `out/android/bin/kt08_example`.

iOS (iPhone 7 or newer, iOS 15 or newer), on a Mac with Xcode and CMake:

```sh
engines/lite/mobile/build_ios.sh out/ios
```

This gives `out/ios/KittenTTS08.xcframework` (1.9 MB). Link the app with `Accelerate.framework` and the C++
standard library.

## Use

```c
#include "kitten_tts08.h"

kt08_model* model;
kt08_model_load("kitten_fp32_80m_arch.json", "kitten_fp32_80m.bin", NULL, &model);   // NULL: the defaults below

float* voice; size_t rows;
kt08_voice_load("expr-voice-2-f.bin", &voice, &rows);          // Bella

const char* text = "Hello there.";                 // one chunk: one or a few sentences
const char* phonemes = "həlˈoʊ ðˈɛɹ.";             // espeak-ng output for that text
size_t n;
kt08_phonemes_to_ids(phonemes, NULL, 0, &n);
int32_t* ids = malloc(n * sizeof *ids);
kt08_phonemes_to_ids(phonemes, ids, n, &n);

float* audio; size_t samples;
kt08_synthesize(model, ids, n, voice, rows, kt08_style_row_for_text(text, rows), 1.0f, &audio, &samples);
// audio: `samples` floats at 24000 Hz
kt08_free(audio);
```

Every call returns `KT08_OK` or an error code; `kt08_last_error()` says what went wrong. A model serves one thread
at a time; load one model per thread to synthesise in parallel. `engines/lite/mobile/example/kt08_example.c` is a
complete program.

## Files

- Architecture files: `src/kitten_inference/tts08/model_defs/kitten_{fp32,int8}_{80m,40m,15m}_arch.json`
  (80m = mini, 40m = micro, 15m = nano).
- Weights and voices: [KittenML/meownn-models](https://huggingface.co/KittenML/meownn-models), revision
  `5e9bca0cc3a4e2a617f604b1333d5f34536c1eed`: `kitten_fp32_80m.bin` and `voices_kitten_80m/expr-voice-*.bin`.
  Bella = `expr-voice-2-f`, Jasper = `2-m`, Luna = `3-f`, Bruno = `3-m`, Rosie = `4-f`, Hugo = `4-m`, Kiki = `5-f`,
  Leo = `5-m`.
- nano only: multiply the speed by 0.9 for Hugo (`expr-voice-4-m`) and 0.8 for every other voice, as the Python
  package does.
- Text to phonemes: data files the app downloads once (for example on first use, like the weights) and passes to the
  library; none of them is part of the library or of this repository:
  - `engines/lite/mobile/fetch_phonemizer_data.sh DIR [LANGUAGE ...]` (every language by default) downloads espeak-ng
    1.52.0's data files for every language except Kitten TTS 2 English, and checks their SHA-256 sums: 30 to 650 KB
    per language, 23 MB for Russian. With `ja` it also fetches the Japanese dictionary (UniDic, BSD licence: 47 MB
    download, 260 MB unpacked; run it from a checkout of this repository for `ja_words.txt`).
  - Kitten TTS 2 English: the model repository's `g2p/` folder plus the files that
    `python -m kitten_inference.tts2_tiny.frontend.export_phone_data DIR` writes (about 12 MB, Apache-2.0 and MIT).

  espeak-ng's files are GPL-3.0 data. The library contains no espeak-ng code and only reads them; they are downloaded
  at run time and never bundled, as in KittenML's own phone SDKs. Apps that mirror them should keep the same bytes.

## Kitten TTS 2 mini, nano and pico

The same library runs [KittenML/kitten-tts-2-mini](https://huggingface.co/KittenML/kitten-tts-2-mini), `-nano` and
`-pico` (header version 2 or newer).

**1. Convert the official files once, on a computer.** For nano in English (a 70 MB download):

```sh
pip install kitten-inference
hf download KittenML/kitten-tts-2-nano --revision 54526b8d8e45333cd720d3a082c106aacf416064 --include config.json --include vocab.json --include "*_en.safetensors" --include voices_en.npz --local-dir nano
python -m kitten_inference.tts2_tiny.convert nano nano-native
```

`hf` comes with `huggingface_hub`, which kitten-inference installs. The revision is the one the converter was checked
against (`REVISIONS` in `src/kitten_inference/tts2_tiny/model.py` lists it for all three models). For every language,
use `--exclude "*.onnx"` instead of the `--include` options (543 MB for nano; the converter does not use the ONNX
files). For each model it finds, the converter writes `<model>_fp32_arch.json` / `<model>_fp32.bin` (and `_int8_`:
smaller, lossy) and `voices/<language>/<voice>.bin`. nano and pico have one model per language
(`kitten_tts_2_nano_de`, ...); mini has one for all 16 languages.

**2. In the app:** `kt08_model_load` as above, `kt08_voice_load` on a voice file, `kt08_vocab_load` on the
repository's `vocab.json`, then per chunk:

```c
size_t n = 0;
kt08_vocab_ids(vocab, phonemes, NULL, 0, &n);              // size query
int32_t* ids = malloc(n * sizeof *ids);
kt08_vocab_ids(vocab, phonemes, ids, n, &n);
kt08_synthesize_tiny(model, ids, n, voice, rows, kt08_style_row_for_ids(n, rows), 1.0f,
                     /* first_chunk */ i == 0, &audio, &samples);
```

`kt08_synthesize_tiny` trims and fades each chunk as the Python package does; put 0.15 s of silence (3600 samples)
between chunks. A chunk holds at most 350 phonemes. `kt08_example` runs these models when `KT08_EXAMPLE_VOCAB` names
the `vocab.json`.

### Streaming (`kt08_stream`, header version 6)

Text in, audio out chunk by chunk, with the same chunks as the Python package's `stream()` (the same code):

```c
kt08_stream_options so;
kt08_stream_options_init(&so);   // chunk_chars 0: the model's default; trim_first_pause 1
kt08_stream* st = NULL;
kt08_stream_open(model, vocab, frontend, text, "en", voice, rows, 1.0f, &so, &st);   // mini, nano, pico
// Kitten TTS 0.8: kt08_stream_open_tts08(model, phonemizer, text, voice, rows, 1.0f, &so, &st);
int last = 0;
while (!last) {                                      // on a worker thread
    kt08_stream_next(st, &audio, &samples, &last);   // queue for playback at once (skip empty audio)
    kt08_free(audio);
}
kt08_stream_free(st);
```

`chunk_chars`: 0 uses the model's default (one sentence for Kitten TTS 0.8; 380 characters, `KT08_STREAM_TINY_CHUNK_CHARS`,
for mini, nano and pico); N packs whole sentences into chunks of up to N characters (smaller starts sooner); a
negative value is `KT08_ERR_ARGUMENT`. `trim_first_pause` 1 drops the silence before the first chunk's speech. Play mini,
nano and pico chunks with 0.15 s (3600 samples) between them, and Kitten TTS 0.8 chunks back to back. The last call
can return no audio; after an error, calling `kt08_stream_next` again retries the same chunk. `kt08_stream_chunks`
lists the chunks' text. `frontend` comes from `kt08_tiny_frontend_load` and `phonemizer` from `kt08_phonemizer_load`
(below). `kt08_example` streams with `KT08_EXAMPLE_POLICY=stream` and `KT08_EXAMPLE_TEXT` (`KT08_EXAMPLE_CHUNK_CHARS=N`
sets `chunk_chars`).

### Text to phonemes (mini, nano, pico)

`kt08_tiny_frontend_load` and `kt08_tiny_text_to_phonemes` (header version 5) turn text into the phoneme chunks the
model speaks, on the phone, with the steps of the Python package's front ends; symbols missing from `vocab.json` are
dropped and long text is cut at sentence ends into chunks of at most 350 phonemes.

```c
kt08_tiny_frontend* fe = NULL;
kt08_tiny_frontend_load("v1" /* config.json "phonemizer": v1 nano/pico, v2 mini */, "nano/vocab.json",
                        "nano/g2p" /* English data (Files above); NULL: no English */,
                        "espeak-data" /* fetch_phonemizer_data.sh's folder, other languages; NULL: none */, &fe);
char* chunks = NULL;                                       // chunks joined with '\n'
kt08_tiny_text_to_phonemes(fe, "Dr. Smith paid $12.50 on May 3rd.", "en", &chunks);
// or: kt08_tiny_text_to_phonemes(fe, "Su mesa está reservada para esta noche.", "es", &chunks);
// for each line of chunks: kt08_vocab_ids + kt08_synthesize_tiny as above
kt08_free(chunks);
kt08_tiny_frontend_free(fe);
```

English reads the `g2p_dir` folder (see [Files](#files)); a small rule tagger takes the place of spaCy's tagger, so
a few words with two readings ("read", "close") can differ from the package. `kt08_example` speaks text when
`KT08_EXAMPLE_TEXT` names that folder.

Other languages read `espeak_data_dir`, the folder `fetch_phonemizer_data.sh` fills; pass the language code (`"es"`,
`"ja"`, ...) as the language. Each language's files are read once, on first use, and not checked again: keep the
folder unchanged after the script has verified it (a file truncated while mapped can crash the process). Japanese
(mini) memory-maps its dictionary, so it costs disk space rather than app memory.

Errors: a language this build does not support, or that the chosen model was not trained on, returns
`KT08_ERR_UNSUPPORTED` (6); for those, `KittenTTS("KittenML/kitten-tts-2-nano").phonemize(text, language="de")`
returns the chunks on a computer. Text with more than 1000 characters in a row without a space returns
`KT08_ERR_TOO_LONG`. Japanese text with a number of more than 9 digits, or digits other than 0-9 (such as `٣` or
`❶`), returns `KT08_ERR_ARGUMENT`, as the package raises on it.

### Speed on phones

pico and nano are faster than real time on every phone measured (on Cortex-A53-only phones with 1 or 2 threads);
mini is on recent phones (real-time factor 0.21 on a Galaxy S25, 0.11 on an iPhone 13). Peak memory with the defaults:
pico about 50 MiB, nano about 100 MiB, mini about 350 MiB (latency mode: about 600 MiB).

- **mini on Android and on iPhones before the A13: use `KT08_MEMORY_LATENCY` if the app can spend about 600 MiB.**
  The low-memory default makes mini 1.4 to 2 times slower there. On the A13 and newer keep the default.
- **mini on phones with four fast cores** (for example Snapdragon 680): set `threads = 4` and `KT08_MEMORY_LATENCY`.
- **First audio.** The first sound plays when the first chunk is done; send the first sentence as its own chunk to
  start sooner.
- Kitten TTS 2 int8 files are lossy and help only mini on Android; iPhones before the A13 cannot load int8 files.

### Qualcomm NPU (optional)

On Snapdragon phones the library can run the decoder and vocoder, most of the work, on the phone's NPU (Hexagon,
through Qualcomm's QNN runtime). The text encoder, durations, pitch and the final inverse STFT stay on the CPU. It
is off unless the app calls `kt08_npu_load` (header version 3).

With the CPU cores free, the CPU path is faster for all three models. Use the NPU path when speech may take only one
CPU core: it then uses less CPU time for mini and pico. Its audio is close to, not identical with, the CPU path's.
Loading the NPU graphs takes 1 to 4 s, so load it once, early, and set `kt08_npu_options.threads = 1` to save CPU time.

What it needs:

- **Snapdragon 8 Gen 2 or newer** (Hexagon v73 and up: the NPU graphs use 16-bit floats, which older NPUs lack).
  Measured on the Snapdragon 8 Gen 3 and 8 Elite only.
- **The library built with Qualcomm's SDK.** Download the Qualcomm AI Runtime SDK (QAIRT) from Qualcomm and build
  with `KITTEN_QNN_SDK=/path/to/qairt/<version> ANDROID_NDK=... engines/lite/mobile/build_android.sh out/android`.
  Only the SDK's headers are used. Without `KITTEN_QNN_SDK`, `kt08_npu_load` returns `KT08_ERR_UNSUPPORTED`.
- **NPU files for the phone's chip**, made once on a Linux x86-64 computer, with the same SDK version, from a clone
  of this repository and the converted model of step 1 above:

  ```sh
  pip install kitten-inference torch onnx
  python -m kitten_inference.tts2_tiny.convert nano nano-native
  python engines/lite/mobile/npu/prepare_npu.py nano-native kitten_tts_2_nano_en npu-nano-sm8650 \
      --chip SM8650 --qairt /path/to/qairt/<version> --qairt-python /path/to/qairt-python
  ```

  `--qairt-python` is a Python that runs Qualcomm's converter (with its packages installed as Qualcomm's setup guide
  says). `--chip` takes SM8750 (8 Elite), SM8650 (8 Gen 3) or SM8550 (8 Gen 2); other chips with `--dsp-arch` and
  `--soc-model`. The files work only on chips with the same Hexagon version and with the same SDK version. Compiling
  takes minutes to more than an hour per graph length.
- **Qualcomm's runtime libraries in the app**, from the same SDK version: `lib/aarch64-android/libQnnHtp.so`,
  `libQnnSystem.so`, `libQnnHtpV75Stub.so` and `lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` (V79 for the 8 Elite,
  V73 for the 8 Gen 2). Qualcomm's licence allows shipping them inside an app; this project does not distribute them.

```c
kt08_npu_options no;
kt08_npu_options_init(&no);
no.model_dir = npu_dir;                 // prepare_npu.py's folder for this phone's chip
no.weights_path = fp32_weights_path;    // kitten_tts_2_nano_en_fp32.bin, as for the CPU path
no.qnn_lib_dir = native_library_dir;    // where the app's Qualcomm libraries are
no.cpu_fallback = model;                // kt08_model_load of the same model: chunks longer than the NPU graphs
kt08_npu* npu = NULL;
if (kt08_npu_load(&no, &npu) != KT08_OK) {
    // KT08_ERR_UNSUPPORTED: not a supported Snapdragon, or no Qualcomm libraries: use kt08_synthesize_tiny
}
int on_npu = 0;
kt08_synthesize_tiny_npu(npu, ids, n, voice, rows, kt08_style_row_for_ids(n, rows), 1.0f, i == 0,
                         &audio, &samples, &on_npu);
```

`kt08_example` runs the NPU path when `KT08_EXAMPLE_NPU` names the folder (and `KT08_EXAMPLE_QNN_LIBS` the
libraries). Not done yet: Snapdragon laptops (Windows on Arm), 8-bit NPU graphs (faster on the NPU, but lossy),
MediaTek and other NPUs.

## Text to phonemes (Kitten TTS 0.8)

The library turns English text into Kitten TTS 0.8 phonemes the way the Python package does with `normalize=True`
(`clean_text=True`): the package's text normalisation (numbers, money, percentages, times, units, ordinals,
fractions, decades, phone numbers ... in words; contractions expanded; URLs and e-mail addresses dropped; lower
case), its sentence chunking, and phonemes from CEPhonemizer, KittenML's Apache-2.0 C++ reader of espeak-ng's English rules (the one in
KittenML's React Native and Swift SDKs; `engine/src/frontend/cephonemizer/`). No espeak-ng code is in the library.

CEPhonemizer reads two espeak-ng data files, `en_rules` and `en_list` (espeak-ng's, GPL-3.0). They are not part of
this repository or of the library: the app downloads them, for example on first use like the model weights, and
passes their paths. `engines/lite/mobile/fetch_phonemizer_data.sh OUT_DIR` downloads them from espeak-ng's
repository at a pinned commit and checks their SHA-256 sums (161,691 + 102,788 bytes).

```c
kt08_phonemizer* ph;
kt08_phonemizer_load("en_rules", "en_list", &ph);         // parsed once, about 8 ms and 2.4 MiB

const char* text = "Dr. Smith paid $4.99 at 3:30pm. Then he left!";
size_t n;
kt08_text_chunks(text, NULL, 0, &n);                       // size first, then the call (as kt08_phonemes_to_ids)
char* chunks = malloc(n);
kt08_text_chunks(text, chunks, n, &n);  // "dr. smith paid four dollars and ninety-nine cents at three thirty pm.\nthen he left!"
for (char* chunk = strtok(chunks, "\n"); chunk; chunk = strtok(NULL, "\n")) {
    kt08_chunk_to_phonemes(ph, chunk, NULL, 0, &n);
    char* phonemes = malloc(n);
    kt08_chunk_to_phonemes(ph, chunk, phonemes, n, &n);
    // phonemes -> kt08_phonemes_to_ids -> kt08_synthesize, style row kt08_style_row_for_text(chunk, rows), as above
    free(phonemes);
}
free(chunks);
kt08_phonemizer_free(ph);
```

`kt08_text_to_phonemes` does every chunk in one call (one line of phonemes per chunk). `kt08_example` speaks text
when `KT08_EXAMPLE_TEXT` names the folder with the two files (each input line is then text);
`mobile/example/kt08_text.c` prints the chunks and phonemes and times them.

Known differences from the package: no Unicode NFC step (pass NFC text, as iOS and Android strings normally are);
text the package's normalisation would fail on (for example `"5,%"`) is left as written instead of failing the call.
Text that is not normalised (the package's `normalize=False`, the default of `generate()`) is not supported.

## Defaults on phones

You do not need to set any option. On Android and iOS the library picks:

- **Threads: the fast cores.** On phones with fast and slow cores, 2 threads on the fast cores (Android also keeps each
  call on those cores). On phones whose cores are all the same, such as Cortex-A53-only phones, every core, up to 8.
  On iPhones, 2 threads (the performance cores).
- **Memory: `KT08_MEMORY_EXTREME_LOW`.** Working memory is freed after each call: for Kitten TTS 0.8 mini, peak
  memory about 310 MiB instead of 560 MiB, for 16-19% more compute time than `KT08_MEMORY_LATENCY` on the same cores on Android phones (no
  slower on the iPhone SE 2020).

## Options

To choose yourself, fill a `kt08_options` and pass it to `kt08_model_load`:

```c
kt08_options opt;
kt08_options_init(&opt);                  // the defaults above
opt.threads = 4;                          // for example: the Galaxy A36, which has 4 fast cores
opt.memory_mode = KT08_MEMORY_LATENCY;    // fastest, more memory
kt08_model_load("kitten_fp32_80m_arch.json", "kitten_fp32_80m.bin", &opt, &model);
```

| `kt08_options` field | Default | What it does |
|---|---|---|
| `threads` | 0 (chosen by the library) | Threads per call. Phones: see above. Desktop and server: one per CPU (or `OMP_NUM_THREADS`). |
| `memory_mode` | phones: `KT08_MEMORY_EXTREME_LOW`; desktop: `KT08_MEMORY_LATENCY` | `KT08_MEMORY_LATENCY` keeps working memory between calls: fastest, more memory. `KT08_MEMORY_LOW` and `KT08_MEMORY_EXTREME_LOW` free it after each call. |
| `fp16_compute` | 0 | **Lossy.** 16-bit arithmetic for most convolutions, Android only, on CPUs with FEAT_FP16 (Cortex-A55/A76/A78 and newer; other CPUs ignore it): less memory, faster. |
| `fp16_weights` | 0 | **Lossy with the rounded weights file it needs.** Keeps weights in 16 bits; saves memory only with a weights file made by `round_weights_fp16.py`. |

`kt08_pin_fastest_cores(n)`, called once before loading, keeps the calling thread on the `n` fastest cores (Android,
Linux). With the default `threads = 0` you do not need it.

## Phones that are supported, not optimal

- **Phones with only Cortex-A53 cores** (for example LG Stylo 5): supported, not optimal. These cores lack the
  dot-product and 16-bit float instructions: Kitten TTS 0.8 mini runs at about 1.4 times real time there, and Kitten
  TTS 2 mini at 1.35, so speech comes slower than it plays. Use Kitten TTS 2 pico or nano there (0.73 and 0.76).
- **Kitten TTS 0.8 int8 models on phones**: supported, not optimal. They are about twice as fast, but on ARM they lost
  quality (word error rate up by as much as 1 point); use the fp32 models. On iPhones they need an A13 or newer.
- **iPhones older than the A13** (iPhone 8, X, XS and older): supported, not optimal. They lack Apple's matrix unit,
  so the library uses its own NEON code there, about half the speed of Accelerate on newer iPhones.

## Matching the Python package

With the same build settings, thread count and memory mode, the library returns exactly the Python package's samples
(`tests/tts08/test_capi.py` and, for mini, nano and pico, `tests/tts2_tiny/test_capi_tiny.py` check this; the Python
package uses `KT08_MEMORY_LATENCY`). The Android library gives the
same samples on every phone, whatever its instruction set. Between different builds, thread counts or, on ARM, latency
and low-memory modes (they use different convolution methods), the samples differ in rounding only; you will not hear
it, but the files are not byte-identical.

## Not included

- Phone NPUs for Kitten TTS 0.8, and MediaTek or Apple Neural Engine NPUs for any model: not supported. Kitten TTS 2
  mini, nano and pico can use Qualcomm NPUs (see "Qualcomm NPU (optional)" above).
- 32-bit ARM (armeabi-v7a) and x86 Android: not built.
