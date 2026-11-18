# Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico on phones

A C library for Android and iOS apps: the same engine as the Python package, without Python. It runs Kitten TTS 0.8
and Kitten TTS 2 mini, nano and pico ([below](#kitten-tts-2-mini-nano-and-pico)).
Give it phonemes; it returns 24 kHz mono audio, sample for sample what the Python package makes from the same
phonemes with the same build settings. It also turns text into those phonemes on the phone, so an app gives text
and gets audio: English for Kitten TTS 0.8 ([Text to phonemes](#text-to-phonemes-kitten-tts-08)) and English, Spanish
and Italian for Kitten TTS 2 mini, nano and pico ([Text to phonemes](#text-to-phonemes-mini-nano-pico)).

Kitten TTS 0.8 mini, fp32, 12 sentences (122 s of audio), measured with this library on each phone:

| Phone | Threads | Real-time factor |
|---|---|---|
| Galaxy A36 (4 fast cores) | 4 | **0.29** (3.5 s of speech per second) |
| Galaxy A16 (2 fast cores) | 2 | **0.55** |

Kitten TTS 0.8 mini on iPhones, 2 threads, 4 short sentences (11 s of audio): iPhone SE 2020 **0.14**, iPhone 7
**0.43**.

Real-time factor is compute time divided by audio length; below 1 is faster than real time. Peak memory of Kitten
TTS 0.8 mini is about 570 MB for short sentences in latency mode (`KT08_MEMORY_LATENCY`) and grows with the longest
chunk (1.1 GB for this set); the low-memory modes, the default on phones, keep it at about 320 MB for short sentences.

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
  library. None of them is part of the library or of this repository:

| Model and language | Files | Where from | Size | Licence |
|---|---|---|---|---|
| Kitten TTS 0.8, English | `en_rules`, `en_list` | espeak-ng, commit `59eb19938f12e30881c81d86ce4a7de25414c9f4`: `fetch_phonemizer_data.sh DIR en` | 264 KB | GPL-3.0 (data) |
| Kitten TTS 2, English | `cmu_misaki_us.json`, `g2p_bart_en_us/` | the model repository's `g2p/` folder (the same for mini, nano and pico) | 6.4 MB | the model repository's (Apache-2.0); CMUdict's BSD-style licence |
| | `us_gold.json`, `us_silver.json`, `spacy_en_exceptions.json` | `python -m kitten_inference.tts2_tiny.frontend.export_phone_data DIR` | 6.1 MB | Apache-2.0 (misaki), MIT (spaCy) |
| Kitten TTS 2, Spanish | `es_rules`, `es_list`, `phsource/{phonemes,ph_base2,ph_spanish}`, and the English files | espeak-ng, same commit: `fetch_phonemizer_data.sh DIR es` | 330 KB | GPL-3.0 (data) |
| Kitten TTS 2, Italian | `it_rules`, `it_list`, `it_listx`, `phsource/{phonemes,ph_base2,ph_italian}`, and the English, German, Spanish and French files its rules use for foreign words (required: keep them all) | espeak-ng, same commit: `fetch_phonemizer_data.sh DIR it` | 627 KB | GPL-3.0 (data) |

  `fetch_phonemizer_data.sh` downloads the espeak-ng files from
  `https://raw.githubusercontent.com/espeak-ng/espeak-ng/59eb19938f12e30881c81d86ce4a7de25414c9f4/` and checks
  their SHA-256 sums. The library contains no espeak-ng code and reads only these data files, which are downloaded
  at run time and never bundled: the same approach as KittenML's own phone SDKs (KittenTTS-react-native,
  KittenTTS-swift), whose CEPhonemizer this library uses for Kitten TTS 0.8. Apps that mirror the files should keep
  the same bytes.

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

`chunk_chars`: None (0 in C) uses the model's default: one sentence per chunk for Kitten TTS 0.8, 380 characters for
mini, nano and pico. A number N packs whole sentences into chunks of up to N characters; smaller starts sooner, larger
finishes the whole text sooner. (A negative value is `KT08_ERR_ARGUMENT`; a sentence longer than N is cut at spaces;
Kitten TTS 0.8 chunks hold at most 400 characters; a mini/nano/pico chunk past 350 phonemes is spoken in parts cut at
sentence ends, then spaces; 380 is `KT08_STREAM_TINY_CHUNK_CHARS`.) The same text and `chunk_chars`
give the same chunks on every device; nothing is measured. Each chunk becomes phonemes only
when it is next (`frontend` from `kt08_tiny_frontend_load`, `phonemizer` from `kt08_phonemizer_load`, see below).
Silence (never sound) is removed at chunk edges only: the first chunk starts 20 ms before its first sample at or above
-50 dBFS (`trim_first_pause` 0 keeps its silent start); mini, nano and pico chunks are `kt08_synthesize_tiny`'s audio
with `first_chunk` 1 and the same start trim (the start of the first word kept, only the silence before it dropped) and
keep 30 ms of silence after their last audible sample when another chunk follows: play them with 0.15 s (3600 samples) between chunks. Kitten
TTS 0.8 chunks are `kt08_synthesize`'s audio (the first without its silent start): play them back to back. A phone
slower than real time will pause between chunks: use a smaller model (pico or nano rather than mini fp32 on an Oppo
A96) or a larger `chunk_chars` there. The Kitten TTS 0.8 chunks equal the Python package's `stream(normalize=True)`
chunks (its `stream()` does not normalise by default). The last call can return no audio (text, or a last chunk, with
nothing to say); after an error, calling `kt08_stream_next` again retries the same chunk. `kt08_stream_chunks` lists the
chunks' text (for logs). `kt08_example` runs it
with `KT08_EXAMPLE_POLICY=stream` and `KT08_EXAMPLE_TEXT` (`KT08_EXAMPLE_CHUNK_CHARS=N` sets `chunk_chars`; `trim`: the
app's chunks with the first chunk's silent start cut; `legacy`, the default: the app's chunks as given).

### Text to phonemes (mini, nano, pico)

**English is built in (header version 5).** `kt08_tiny_frontend_load` and `kt08_tiny_text_to_phonemes` turn text into
the phoneme chunks the model speaks, on the phone, with the steps of the Python package's English front end
(`src/kitten_inference/tts2_tiny/frontend/english.py`): text normalisation (numbers, dates, times, money, units,
abbreviations, URLs), misaki's US lexicon, CMUdict, and the small BART model for words neither knows; symbols missing
from `vocab.json` are dropped and long text is cut at sentence ends into chunks of at most 350 phonemes.

```c
kt08_tiny_frontend* fe = NULL;
kt08_tiny_frontend_load("v1" /* config.json "phonemizer": v1 nano/pico, v2 mini */, "nano/vocab.json",
                        "nano/g2p" /* English data, see below; NULL: no English */,
                        "espeak-data" /* fetch_phonemizer_data.sh's folder, for es and it; NULL: none */, &fe);
char* chunks = NULL;                                       // chunks joined with '\n'
kt08_tiny_text_to_phonemes(fe, "Dr. Smith paid $12.50 on May 3rd.", "en", &chunks);
// or: kt08_tiny_text_to_phonemes(fe, "Su mesa está reservada para esta noche.", "es", &chunks);
// for each line of chunks: kt08_vocab_ids + kt08_synthesize_tiny as above
kt08_free(chunks);
kt08_tiny_frontend_free(fe);
```

The front end reads its data once, at load, from one folder: the model repository's `g2p/` folder
(`cmu_misaki_us.json` and `g2p_bart_en_us/`, the same files for mini, nano and pico; download them like the weights,
`--include "g2p/*"`) plus three files that `python -m kitten_inference.tts2_tiny.frontend.export_phone_data DIR` writes:
misaki's `us_gold.json` and `us_silver.json` (Apache-2.0) and `spacy_en_exceptions.json`, the tokenizer exceptions
of spaCy's `en_core_web_sm` 3.8.0 (MIT). About 12 MB in all; the app downloads or bundles them (none is GPL).
`kt08_example` speaks text when `KT08_EXAMPLE_TEXT` names that folder.

How close it is to the Python package (measured on an arm64 Linux box, `tests/tts2_tiny/test_frontend_capi_en.py` and
the parity tool `tools/tiny_en_tool.cpp`): the Python front end tags words with spaCy's statistical part-of-speech
tagger, which decides between readings such as "read" (/ɹˈid/ or /ɹˈɛd/), "close", "use" or the stress of "that".
The library does not ship spaCy; a small rule tagger takes its place. Everything else (normalisation, tokenizer,
lexicon, CMUdict, BART, chunking) gives the same phonemes as the package on every sentence when it is given spaCy's
tags. Share of sentences with exactly the package's phonemes (nano; mini and pico give the same rates):

| Text set | Sentences | Exact match |
|---|---|---|
| CMU ARCTIC prompts | 1132 | 98.3 % |
| LibriSpeech-PC test-clean | 2417 | 92.1 % |
| LibriSpeech-PC dev-clean (not used while writing the rules) | 2530 | 92.3 % |
| Numbers, dates, money, abbreviations, URLs (320 lines) | 320 | 97.5 % |

A differing sentence differs in about one phoneme symbol (mostly a stress mark or one homograph). Listening test by
machine on 150 such sentences (nano, English, fp32, this library): Whisper large-v3 word error rate 2.51 % with the
package's phonemes and 2.35 % with the library's (difference -0.16 points, 95 % interval -0.36 to +0.02), UTMOS
4.473 and 4.474: no measurable difference.

Speed and memory, one thread, measured: on a phone's Cortex-A53 core (Oppo A96, 1.9 GHz) a CMU ARCTIC sentence takes
0.3 ms (mean; slowest 0.9 ms), and loading takes 0.6 s and 18 MiB. On a Cortex-A725 core (the Android build on an
arm64 Linux box) a sentence takes 0.1 ms (median; 99 % of LibriSpeech sentences under 2.2 ms) and loading 0.1 s. A word
that is in none of the dictionaries goes through the BART model, about 1.4 ms the first time on the Cortex-A725 (it
is remembered after that); about 1 sentence in 15 of LibriSpeech has one.

**Spanish and Italian** (nano, pico and mini) are built in too. They use the library's own reader of espeak-ng's
data files (no espeak-ng code is linked or copied), then the package's symbol tables (`phonemize_v1.py` for nano and
pico, `phonemize_v2.py` for mini). Pass `espeak_data_dir`, the folder that `fetch_phonemizer_data.sh` fills, to
`kt08_tiny_frontend_load`, and `"es"` or `"it"` as the language. Each language's files are read once, on first use.
Other languages return `KT08_ERR_UNSUPPORTED`; for them, `KittenTTS("KittenML/kitten-tts-2-nano").phonemize(text,
language="de")` returns the phoneme chunks on a computer, so an app can phonemize on a server or ship the chunks of
fixed prompts.

How close it is to the Python package (measured on FLEURS test sentences; the same for nano, pico and mini):

| Language | Sentences | Exact match | Listening test by machine on 150 differing sentences (word error rate change, 95 % interval) |
|---|---|---|---|
| Spanish | 348 | 94.3 % | nano -0.38 points [-0.86, +0.07]; mini -0.81 [-1.22, -0.43] |
| Italian | 346 | 90.8 % | nano +0.07 points [-0.21, +0.36]; mini -0.17 [-0.45, +0.08] |

Not supported yet: German (71 % exact; its listening test does not pass), French (30 %), Portuguese, Russian and
Hindi, mini's Danish, Finnish, Dutch, Swedish, Swahili, Turkish and Korean (not tried), and Japanese.
Speed, one thread, measured: on a phone's Cortex-A53 core (Oppo A96) a FLEURS sentence takes 1.3 ms in Spanish and
1.5 ms in Italian (mean; 90 % under 1.9 and 2.4 ms, slowest 6.5 and 8.3 ms); the first sentence also reads the
language's files (20 ms for Spanish, 93 ms for Italian). On a Cortex-A725 core: 0.2 ms (median). A loaded language
takes 0.5 to 2.6 MiB (up to 5.4 MiB for Italian once its foreign-word data is read). Symbols missing from `vocab.json` are dropped and
text is cut into chunks of at most 350 phonemes, as for English.

Text in, audio out, on a phone (`kt08_example`, Oppo A96, Android 13, the library's defaults, two paragraphs per
model): Kitten TTS 2 nano speaks English at a real-time factor of 0.39 (peak memory 96 MiB) and Spanish at 0.42
(91 MiB); Kitten TTS 0.8 mini at 1.69 (318 MiB), slower than real time on this phone with the defaults. Turning the
text into phonemes took 0.2 to 8.5 ms of that per paragraph (8.5 ms for the first Spanish one, which also reads
Spanish's files).

### Speed on phones

Real-time factor (compute time divided by audio length; below 1 is faster than speech) with the defaults above,
fp32, two English sentences (6 to 8 s of audio), measured on a call after the first one, average of two runs (one
run where marked):

| Phone | pico | nano | mini |
|---|---|---|---|
| iPhone 13 | 0.03 (1 thread) | 0.04 (1 thread) | **0.11** |
| Galaxy S25 | 0.02 | 0.03 | **0.21** |
| Galaxy S23 | 0.06 | 0.11 | 0.77 (latency mode: **0.43**) |
| Galaxy A36 | 0.08 | 0.14 | 0.95 (latency mode: **0.54**) |
| Galaxy A16 | 0.10 | 0.18 | 1.13 (latency mode: **0.72**) |
| Oppo A96 | 0.20 | 0.41 | 3.0 (4 threads, latency mode: **0.95**) |
| Vivo Y22 | 0.22 | 0.40 | 2.9 (4 threads, latency mode: 1.47): not real time |
| iPhone 7 | 0.11 (1 thread) | 0.21 (1 thread) | 1.1, one run (latency mode: **0.56**) * |
| LG Stylo 5 (8x Cortex-A53) | 0.73 (1 thread) | 0.76 (2 threads) | 1.9 (latency mode: 1.35): not real time |

\* One of the four iPhone 7 phones measured was up to 1.45 times slower for mini (2 threads); pico and nano (1 thread)
ran at the same speed on all four.

Peak memory with the defaults, in MiB (1 MiB = 1,048,576 bytes): pico 42 to 56, nano 77 to 101, mini 333 to 378.
mini in latency mode: 586 to 594 MiB on the Android phones and the iPhone 7, 654 MiB on the iPhone 13 and 902 MiB on
the iPhone 16 (on the A13 and newer, latency mode uses a faster convolution method that needs more memory).

- **mini on Android and on iPhones before the A13: use `KT08_MEMORY_LATENCY` if the app can spend about 600 MiB.**
  The low-memory default makes mini 1.4 to 2 times slower, so it is not real time on phones like the Galaxy A16.
  On the A13 and newer keep the default: it is as fast (iPhone 13: 0.11, latency mode 0.12) with less memory (374
  instead of 654 MiB). The same holds on the Galaxy S25 (0.21, latency mode 0.23).
  pico and nano are fast enough with the defaults.
- **mini on phones with four fast cores** (for example the Oppo A96, Snapdragon 680: 4x Cortex-A73 and 4x Cortex-A53):
  the defaults use 2 of the fast cores in low-memory mode, and mini runs at 3.0 there. Set `threads = 4` and
  `KT08_MEMORY_LATENCY` for real time: 0.95 (590 MiB), or 0.98 with the int8 file (271 MiB). Either setting alone
  gives 1.6.
- **First audio.** The first sound plays when the first chunk is done. To start sooner, send the first sentence as
  its own chunk: on the LG Stylo 5 (pico, 1 thread) the first audio of "Got it. Your table for four is booked for
  seven thirty tonight." comes after 0.9 s instead of 2.5 s (load included); on the five other phones measured it
  came 2.4 to 3.6 times sooner (Galaxy S25 0.07 s instead of 0.16 s), and mini's on the Oppo A96 (4 threads,
  latency mode) after 2.4 s instead of 5.0 s. The Python package packs sentences up to 350 phonemes per chunk, so
  the audio then differs from it at that sentence end.
- Kitten TTS 2 int8 files are lossy. On phones they help only mini on Android. With the defaults, at 201 to 217 MiB:
  Galaxy S25 0.08, Galaxy S23 0.29, Moto G54 0.46, Galaxy A16 0.51. On the LG Stylo 5 int8 is as fast as fp32 in
  latency mode (1.32) with half the memory (274 instead of 594 MiB). On the iPhone 13 mini int8 is slower than fp32
  (0.16), and iPhones before the A13 cannot load int8 files.

### Qualcomm NPU (optional)

On Snapdragon phones the library can run the decoder and vocoder, most of the work, on the phone's NPU (Hexagon,
through Qualcomm's QNN runtime). The text encoder, durations, pitch and the final inverse STFT stay on the CPU. It
is off unless the app calls `kt08_npu_load` (header version 3).

**When it helps, and when the CPU is faster.** With the CPU cores free, the CPU path is faster for all three
models on the phones measured: 1.7 to 2.9 times for pico and nano; for mini 2.0 to 2.2 times in latency mode and
1.1 to 1.2 times in the default (low-memory) mode. Use the NPU path when speech may take only one CPU core, for
example while the app does other work: there it uses less CPU time for mini (about 40% less) and pico (6 to 19%
less), about the same for nano (2 to 3% less), and mini is also faster on the Galaxy S25 (0.63 against 0.82). The
NPU path does not make pico or nano faster. For mini it uses more memory than the CPU path (603 to 749 MiB against
388 MiB in the same mode).

Real-time factor (compute time divided by audio length; below 1 is faster than speech), English, 8 sentences
(21 to 26 s of audio), average of the two calls after the first, fp32 CPU path, fp16 NPU graphs of 2 s and 4 s
(longer sentences ran on the CPU fallback: 2 of 8 for pico, 3 of 8 for nano and mini). Each row's CPU and NPU
numbers come from the same test session:

| Phone | Model | CPU path | NPU path | CPU path, 1 thread | NPU path, 1 CPU thread | CPU time per second of audio, 1 thread: CPU / NPU | Peak memory: CPU / NPU |
|---|---|---|---|---|---|---|---|
| Galaxy S25 (Snapdragon 8 Elite) | pico | **0.024** | 0.069 | 0.055 | 0.090 | 0.056 / **0.046** s | 52 / 103 MiB |
| Galaxy S25 (Snapdragon 8 Elite) | nano | **0.039** | 0.085 | 0.134 | 0.171 | 0.134 / 0.130 s | 95 / 234 MiB |
| Galaxy S25 (Snapdragon 8 Elite) | mini | **0.278** (latency mode **0.136**) | 0.300 | 0.817 | **0.632** | 0.826 / **0.504** s | **388** / 749 MiB |
| Galaxy S24 (Snapdragon 8 Gen 3) | pico | **0.044** | 0.089 | 0.062 | 0.105 | 0.064 / **0.060** s | 52 / 104 MiB |
| Galaxy S24 (Snapdragon 8 Gen 3) | nano | **0.082** | 0.141 | 0.122 | 0.178 | 0.123 / 0.122 s | 98 / 238 MiB |
| Galaxy S24 (Snapdragon 8 Gen 3) | mini * | **0.570** (latency mode **0.334**) | 0.681 | 0.869 | 0.851 | 0.870 / **0.540** s | **387** / 603 MiB |

Every row uses the library's defaults (the NPU path's CPU parts and its CPU fallback run in the default memory mode;
`kt08_npu_options` has no memory mode); mini's CPU path is also shown in latency mode (`KT08_MEMORY_LATENCY`). \* The
S24's mini NPU graphs were compiled at a lower optimisation level (`--opt 1`/`2`); on the S25 the default level ran
the 4 s graph 2.1 times faster (0.91 s against 1.94 s).

- **Audio.** The NPU path's audio is as close to the fp32 CPU audio as the CPU's own 16-bit mode, or closer:
  39 to 49 dB SNR and a log-mel distance of 0.07 to 0.16 dB against the CPU path's audio of the same sentence
  (`fp16_compute` on the CPU: 26 to 31 dB, 0.23 to 0.30 dB).
- **First audio.** Loading the NPU graphs takes 0.9 to 4.3 s (mini's), so the first audio comes 1.2 to 6.2 s after
  the app starts loading (6.2 s: mini with one thread), against 0.1 to 1.5 s on the CPU. Load the NPU path once, early.
- **Each sentence costs as much as the shortest graph it fits in**: about 0.2 s for pico and 0.9 s for mini in a
  4 s graph on the S25, however short the sentence.
- **To save CPU time, set `kt08_npu_options.threads = 1`.** With more threads, the idle ones keep spinning while
  the NPU works: on a Galaxy S25, pico's NPU path used 0.50 s of CPU time per second of audio with the default
  threads and 0.05 s with one thread (real-time factor 0.063 and 0.088).

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

  `--qairt-python` is a Python that runs Qualcomm's converter (a version their SDK supports, with its packages
  installed as Qualcomm's setup guide says). The script writes `npu.json`, two small CPU graphs and one NPU graph
  per length: 2 s and 4 s by default, since a sentence costs as much as the shortest graph it fits in (`--frames`
  picks others, in 25 ms frames; an 8 s graph, `320`, was slower than the CPU). `--chip` takes SM8750
  (8 Elite), SM8650 (8 Gen 3) or SM8550 (8 Gen 2); other chips with `--dsp-arch` and `--soc-model`. The files work
  only on chips with the same Hexagon version and with the same SDK version. Compiling is slow: on a 32-core x86
  server running many compiles at once, one length took 4 to 95 minutes for pico (1 s to 8 s), 11 to 75 minutes for
  nano (2 s to 4 s), and more than an hour for mini's 4 s length at the default optimisation level (`--opt 3`;
  lower levels compile faster but ran about 2 times slower on the phone).
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

How close it is to the Python package (espeak-ng 1.52.0 through `phonemizer`), English, measured on Linux arm64
(`tests/tts08/test_capi_text.py` checks the API):

| Text set | Lines | Same chunks | Same phonemes (model input) |
|---|---|---|---|
| CMU ARCTIC prompts | 1,132 | 100% | 95.4% |
| LibriSpeech test-clean (punctuated) | 2,417 | 100% | 86.7% |
| Numbers, money, dates, URLs ... (written for this test) | 320 | 100% | 91.3% |

The differences are mostly stress marks on short words and a few joined word pairs ("in which"). Spoken by Kitten TTS
0.8 mini, the 373 ARCTIC and LibriSpeech lines that differ have Whisper large-v3 word errors of 1.90% with the
package's phonemes and 2.04% with the library's (+0.14 points, 95% interval -0.02 to +0.31); UTMOS is 0.008 lower
on average (4.365 vs 4.356, interval -0.015 to -0.001), well inside the -0.05 limit. Both are inside the limits we
use for lossless changes (+0.5 points, -0.05).

Speed, one thread, measured: on a phone's Cortex-A53 core (Oppo A96, 1.9 GHz) 1.7 ms per ARCTIC sentence (2.9 ms
the first time, slowest 5.0 ms), 0.76 ms on its Cortex-A73 core; on a Cortex-A725 core 0.24 ms per ARCTIC sentence
and 0.59 ms per LibriSpeech line (the longest line, 482 characters, 3.6 ms). Loading takes 8 ms (52 ms on the
Cortex-A53) and 2.4 to 2.9 MiB of memory. Known differences from the
package: no Unicode NFC step (pass NFC text, as iOS and Android strings normally are); text the package's
normalisation would fail on (for example `"5,%"`) is left as written instead of failing the call. Text that is not
normalised (the package's `normalize=False`, the default of `generate()`) is not supported.

## Defaults on phones

You do not need to set any option. On Android and iOS the library picks:

- **Threads: the fast cores.** On phones with fast and slow cores, 2 threads on the fast cores (Android also keeps each
  call on those cores). On phones whose cores are all the same, such as Cortex-A53-only phones, every core, up to 8.
  On iPhones, 2 threads (the performance cores).
- **Memory: `KT08_MEMORY_EXTREME_LOW`.** Working memory is freed after each call: peak memory about 310 MB instead
  of 560 MB, for 16-19% more compute time than `KT08_MEMORY_LATENCY` on the same cores on Android phones (no
  slower on the iPhone SE 2020).

Kitten TTS 0.8 mini, fp32, 4 short sentences (11 s of audio), with the defaults and with the previous defaults
(every core, latency mode). First audio is the compute time of the first sentence, after a first call:

| Phone | Real-time factor | First audio | Peak memory | Previous defaults |
|---|---|---|---|---|
| Galaxy A16 (2 fast cores) | **0.70** | 1.8 s | 309 MB | 0.88, 2.3 s, 563 MB |
| Galaxy A36 (4 fast cores) | **0.58** | 1.5 s | 309 MB | 0.78, 2.2 s, 563 MB |
| iPhone SE 2020 | **0.13** | 0.35 s | 356 MB | 0.14, 0.39 s, 604 MB |

On Android the weights file is memory-mapped while loading and closed afterwards; loading is about 0.3 s faster.

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
| `memory_mode` | phones: `KT08_MEMORY_EXTREME_LOW`; desktop: `KT08_MEMORY_LATENCY` | `KT08_MEMORY_LATENCY` keeps working memory between calls: fastest (A16 0.60, A36 0.49 on the defaults' cores), 560 MB for short sentences and more for long ones. `KT08_MEMORY_LOW` and `KT08_MEMORY_EXTREME_LOW` free it after each call: the same speed and memory for short sentences. |
| `fp16_compute` | 0 | **Lossy.** 16-bit arithmetic for most convolutions, Android only, on CPUs with FEAT_FP16 (Cortex-A55/A76/A78 and newer; other CPUs ignore it). With low-memory mode on the A16: 235 MB and 1.3 times as fast as low-memory fp32. The audio changes slightly. |
| `fp16_weights` | 0 | **Lossy with the rounded weights file it needs.** Keeps weights in 16 bits where that is exact. It saves memory (about 125 MB for mini) only with a weights file made by `round_weights_fp16.py`, and that file changes the audio slightly. |

`kt08_pin_fastest_cores(n)`, called once before loading, keeps the calling thread on the `n` fastest cores (Android,
Linux). With the default `threads = 0` you do not need it.

## Phones that are supported, not optimal

- **Phones with only Cortex-A53 cores** (for example LG Stylo 5): supported, not optimal. These cores lack the
  dot-product and 16-bit float instructions, and Kitten TTS 0.8 mini runs at about 1.4 times real time with the
  default 8 threads (LG Stylo 5, low-memory mode), so speech comes slower than it plays. nano and micro are faster but
  sound worse.
  Kitten TTS 2: pico (1 thread, 0.73) and nano (2 threads, 0.76) are real time there; mini is not (1.35 with 8
  threads in latency mode).
- **Kitten TTS 0.8 int8 models on phones**: supported, not optimal. They are about twice as fast (A16: 0.33, A36:
  0.23 on 2 threads), but on ARM they lost quality in listening and speech-recognition tests (word error rate up by
  as much as 1 point); use the fp32 models. On iPhones they need an A13 or newer. For Kitten TTS 2, see the int8 line
  in [Speed on phones](#speed-on-phones).
- **iPhones older than the A13** (iPhone 8, X, XS and older): supported, not optimal. They lack Apple's matrix
  unit, so the library uses its own NEON code there (iPhone 7: 0.43; forcing Accelerate with
  `KITTEN_ACCEL_SGEMM=1` is slower, 0.52). On A13 and newer, Accelerate is on by default and twice as fast
  (iPhone SE 2020: 0.14 vs 0.27 with `KITTEN_ACCEL_SGEMM=0`).

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
