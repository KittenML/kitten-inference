# Python and command-line API

Install with `pip install kitten-inference`; model weights download on first use. Each model's requirements: [Kitten TTS 2](../engines/tts2/README.md#install-with-pip), [Kitten TTS 0.8](../engines/lite/README.md#platforms), [Kitten TTS 2 mini, nano and pico](../engines/lite/README.md#kitten-tts-2-mini-nano-and-pico).

```python
from kitten_inference import KittenTTS

with KittenTTS("KittenML/kitten-tts-nano-0.8") as tts:
    tts.generate_to_file("Hello!", "hello.wav", voice="Jasper")

with KittenTTS("KittenML/kitten-tts-2") as tts:
    audio = tts.generate("Hello!", voice="Bruno", seed=42)
    tts.save_wav("hello.wav", audio)
```

Kitten TTS 2 mini, nano and pico run on the CPU by default, and on the GPU (Metal) by default on Apple Silicon Macs:

```python
with KittenTTS("KittenML/kitten-tts-2-mini") as tts:
    audio = tts.generate("Hola, ¿qué tal?", language="es")
    audio = tts.generate("Hello there.", voice="Eleanor", speed=1.1)
```

A model is required. TTS 0.8 accepts [nano/micro/mini FP32 and INT8 names](../engines/lite/README.md#models-and-voices), aliases, native Hugging Face repositories, and local native configuration directories. TTS 2 accepts `KittenML/kitten-tts-2` or `kitten-tts-2`. Kitten TTS 2 mini, nano and pico accept `KittenML/kitten-tts-2-mini`, `-nano`, `-pico` (with or without `KittenML/`) or a local copy of one of those repositories. Hardware detection selects a backend for the requested model.

## Common operations

- `KittenTTS(model, backend=None, cache_dir=None, **model_options)` loads the selected model. Backends: TTS 0.8 `cpu`, `metal`; TTS 2 `cuda`, `rocm`; mini, nano and pico `cpu`, `metal` (Apple Silicon) or `cuda` (NVIDIA, Linux x86-64; Windows x64 experimental). The backend is selected automatically when omitted. A backend the model does not run on, `int8` on a GPU backend, or `low_memory=True` on a GPU backend raises `ValueError` before anything is downloaded; an unknown option raises `TypeError`.
- `generate(text, voice=None, **options)` returns a mono float32 NumPy array. `sample_rate` gives its rate: normally 24 kHz, or the rate in a custom TTS 0.8 configuration.
- `stream(text, voice=None, **options)` yields arrays. TTS 0.8, mini, nano and pico yield one text chunk at a time, the same chunks on every backend and in the phone library (`kt08_stream`); TTS 2 yields incremental audio. Close the iterator when stopping early, preferably with `contextlib.closing`. Options for TTS 0.8, mini, nano and pico:
  - `chunk_chars` (`None` or a positive int, default `None`): `None` uses the model's default (one sentence for TTS 0.8, 380 characters for mini, nano and pico); N packs whole sentences into chunks of up to N characters. Other values raise `ValueError`.
  - `trim_lead` (bool, default `True`): drop the silence before the first chunk's speech.
  - Play mini, nano and pico chunks with 0.15 s of silence between them, and TTS 0.8 chunks back to back.
- `generate_to_file(text, output_path, ...)` writes audio using the model's actual sample rate; SoundFile selects the format from the extension. `save_wav(path, audio)` writes PCM16 WAV and accepts an array or a list of chunks.
- `list_voices()` lists available voices (TTS 0.8, mini, nano and pico take voice names in any case). `default_voice()` is `Leo` for the standard TTS 0.8 models and Bruno for TTS 2.
- `capabilities` describes streaming mode, cloning availability, and normalization span support. `close()` is safe to call repeatedly; closed instances reject synthesis. Instances support context managers. On Metal, reuse each model: its GPU weights stay allocated until the process exits ([details](../engines/lite/README.md#models-and-voices)).

Synthesis options: `speed` (TTS 0.8: above 0, with the model's speed priors; mini, nano and pico: 0.25 to 4; otherwise `ValueError`), `language` (a code from `languages()`, default `en`; TTS 0.8 is English only; other codes raise `ValueError`) and `normalize` (TTS 0.8: off for `generate` and `stream`, on for file generation; mini, nano and pico: always on, `False` raises `ValueError`; TTS 2: on by default). Unsupported options raise `TypeError`.

TTS 0.8 constructor options are `precision` (`fp32` or `int8`; default: the model name's) and `low_memory` (default `False`; `True` uses the least memory on the CPU).

## Model-specific operations

Kitten TTS 2 mini, nano and pico: English and Japanese text need Python 3.10 to 3.14, and Japanese is not available on Windows on Arm. Constructor options are `precision` (`fp32`, the default, or `int8`: smaller and faster on most Arm64 CPUs; see the [README](../engines/lite/README.md#kitten-tts-2-mini-nano-and-pico)), `revision` (default: the revision this release was tested with), `low_memory` and `cache_dir`. `list_voices(language=None)` lists voice names; a name that only one language has also selects that language. `phonemize(text, language=None)` returns `generate()`'s chunks as phoneme strings, the input of the C library's `kt08_synthesize_tiny`. CPU threads: set `OMP_NUM_THREADS` before the first model loads (Linux and Windows; macOS uses one core for now and warns when it is above 1).


TTS 0.8 standalone normalization is available through `tts.normalize_text(text, locale="en-US", return_spans=False)` and the root exports `normalize_text`, `normalize_text_result`, `NormalizedSpan`, and `NormalizedTextResult`. This normalizer is not used for TTS 2.

TTS 2 constructor options are `revision` (latest by default), `device` (GPU 0), `memory_profile` (`auto`), and `voice_cache` (automatic). Synthesis options include `seed`, `preset` (`stable` or `expressive`), `temperature`, `top_p`, `top_k`, `min_p`, `max_new_tokens`/`max_tokens`, and `speed`. `generate(stream=True)` renders through the streaming path and returns all audio.

Use `clone_voice(reference_audio_path, transcript, **options)` to create a voice, `save_voice(voice, path)` to save it, and `load_voice(path)` to reuse it. `Voice` is exported from `kitten_inference`. Check `can_clone()` and the [GPU memory and FFmpeg requirements](../engines/tts2/README.md#install-with-pip) before cloning. Saved voices require the same model revision and preparation version. TTS 2 also exposes `revision`, `model_dir`, `build_info()`, and `stats()` for diagnostics.

Public exceptions are exported from `kitten_inference`: `EngineError`, `EngineNotInstalled`, `EngineMismatch`, `PrepareError`, `BundleError`, `AudioError`, and `FFmpegNotFound`.

## Same behaviour on every hardware

Within a model family, a program gets the same options, errors, voices, languages and chunks on every computer: Kitten
TTS 0.8 on the CPU and the Mac GPU, and Kitten TTS 2 mini, nano and pico on the CPU, the Mac GPU and NVIDIA GPUs. The
speed and the exact samples differ (Kitten TTS 0.8 on the Mac GPU: lengths up to 3.5%; see the table below).

- **Options:** both families take `backend`, `cache_dir`, `precision` and `low_memory` when you create the model, and
  `voice`, `speed`, `language` and `normalize` when you make speech (`revision` too for mini, nano and pico). An option
  a model does not have raises `TypeError`; a value it cannot use raises `ValueError`. Both say what to pass instead,
  in the same words on every computer.
- **Checked first:** the backend, precision and memory settings are checked before anything is downloaded. For example
  `backend="cuda"` with Kitten TTS 0.8 fails at once and names the models that do run on NVIDIA GPUs.
- **Speed:** 1.0 is normal and 2.0 twice as fast, on every backend. Kitten TTS 2 mini, nano and pico take 0.25 to 4;
  Kitten TTS 0.8 takes any value above 0, and nano 0.8 multiplies it by each voice's own pace, as kittenml does.
- **Voices and languages:** the same list everywhere; names work in any case. Kitten TTS 0.8 speaks English
  (`languages()` is `['en']`, and `language="en"` is accepted) and its default voice is Leo.
- **Text:** the same text is cut into the same chunks and turned into the same phonemes on every backend.
- **Streaming:** `stream()` cuts a text into the same chunks on every backend and on phones (`kt08_stream`): sentences
  packed up to `chunk_chars` characters, or the model's default ([Common operations](#common-operations)).

What must differ, and why:

| Setting | CPU | Mac GPU (`metal`) | NVIDIA GPU (`cuda`, mini/nano/pico) | Phones (C library) | Why |
|---|---|---|---|---|---|
| Backend when you leave it out | everywhere except as listed | float32 on Apple silicon | only when you ask for it | the CPU (or the NPU when the app asks) | the Mac GPU is faster for float32; on NVIDIA machines the CPU stays the default for now |
| Precision | fp32 by default; int8 when asked (0.8: the `-int8` models) | fp32 only | fp32 only | as the app chooses | the GPU backends have no fast int8 path: int8 on a GPU raises `ValueError` |
| `low_memory=True` | yes | no (`ValueError`) | no (`ValueError`) | the phones' default | it frees CPU working memory after each call; without a backend it picks the CPU |
| CPU threads | Linux and Windows: one per logical CPU (Intel P+E chips: the P-cores); `OMP_NUM_THREADS` sets it | macOS: one core (its build has no OpenMP yet; setting `OMP_NUM_THREADS` warns) | - | 2 fast cores (iPhone: at most 2) | measured per platform |
| Audio, bit for bit | the same on the same machine with 1 thread; with several threads, runs can differ in rounding | 0.8: lengths can differ from the CPU by up to 3.5%; mini/nano/pico: the CPU's length | the CPU's length, not bit-identical | the CPU's (NPU: close, not identical) | each backend has its own kernels |

Phones use the C library, which takes text (its own front ends; `kt08_stream` streams it with the same chunks) or phonemes and follows the same speed rules (mini, nano and pico:
0.25 to 4); the app picks a 0.8 voice by file name (Leo is `expr-voice-5-m`) and applies nano 0.8's voice pace itself
([engines/lite/mobile](../engines/lite/mobile/README.md)).
Kitten TTS 2 (large) runs only on NVIDIA and AMD GPUs and has its own streaming.

## Command line

```sh
kitten-tts --model KittenML/kitten-tts-nano-0.8 --text "Hello." --voice Jasper --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --text "Hello." --voice Bruno --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --prepare-only --with-cloning
```

`python -m kitten_inference` is equivalent. Use `kitten-tts --model MODEL --help` for model-specific options. TTS 0.8 accepts text or a UTF-8 text file, voice listing, backend/cache selection, speed, and normalization. TTS 2 also supports output formats, reports, streaming, repeat/concurrency/thread controls, sampling, voice-cache limits, and cloning. TTS 0.8, mini, nano and pico take `--backend`, `--precision`, `--low-memory`, `--language` and `--normalize`; for mini, nano and pico `--list-voices` prints one line per language (`--language de` lists one). Use `--cache-dir` to choose the cache location and, for TTS 2, `--revision` to pin a model version (mini, nano and pico: `revision=` in Python).

## Source builds

`pip install .` builds the TTS 0.8 extension. Source installs do not include TTS 2 GPU binaries; use the PyPI package for GPU inference, or follow the [native build instructions](../engines/tts2/README.md#build-from-source).
