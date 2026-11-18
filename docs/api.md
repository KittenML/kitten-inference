# Python and command-line API

Install with `pip install kitten-inference`; model weights download on first use. See [installation requirements](../README.md#install) for supported platforms and GPU setup.

```python
from kitten_inference import KittenTTS

with KittenTTS("KittenML/kitten-tts-nano-0.8") as tts:
    tts.generate_to_file("Hello!", "hello.wav", voice="Jasper")

with KittenTTS("KittenML/kitten-tts-2") as tts:
    audio = tts.generate("Hello!", voice="Bruno", seed=42)
    tts.save_wav("hello.wav", audio)
```

Kitten TTS 2 mini, nano and pico run on the CPU by default, also on machines with a GPU (`backend="cuda"` runs float32 on an NVIDIA GPU); on Apple Silicon Macs they run most of the model on the GPU (Metal) by default, for float32 (`backend="cpu"` keeps them on the CPU):

```python
from kitten_inference import KittenTTS

with KittenTTS("KittenML/kitten-tts-2-mini") as tts:
    audio = tts.generate("Hola, ¿qué tal?", language="es")          # the language's first voice
    audio = tts.generate("Hello there.", voice="Eleanor", speed=1.1)
```

A model is required. TTS 0.8 accepts [nano/micro/mini FP32 and INT8 names](../engines/lite/README.md#models-and-voices), aliases, native Hugging Face repositories, and local native configuration directories. TTS 2 accepts `KittenML/kitten-tts-2` or `kitten-tts-2`. Kitten TTS 2 mini, nano and pico accept `KittenML/kitten-tts-2-mini`, `-nano`, `-pico`, the same names without `KittenML/`, and a local copy of one of those repositories. Hardware detection selects a backend for the requested model.

## Common operations

- `KittenTTS(model, backend=None, cache_dir=None, **model_options)` loads the selected model. TTS 0.8 backends are `cpu` and `metal`; TTS 2 backends are `cuda` and `rocm`; Kitten TTS 2 mini, nano and pico use `cpu`, `metal` (Apple Silicon, float32, the default there) or `cuda` (NVIDIA GPUs, float32). The backend is selected automatically when omitted.
- `generate(text, voice=None, **options)` returns a mono float32 NumPy array. `sample_rate` gives its rate: normally 24 kHz, or the rate in a custom TTS 0.8 configuration.
- `stream(text, voice=None, **options)` yields arrays. TTS 0.8 yields text chunks; TTS 2 yields incremental audio. Explicitly close the iterator when stopping early, preferably with `contextlib.closing`.
- `generate_to_file(text, output_path, ...)` writes audio using the model's actual sample rate; SoundFile selects the format from the extension. `save_wav(path, audio)` writes PCM16 WAV and accepts an array or a list of chunks.
- `list_voices()` lists available voices. The default voice is Leo for the standard TTS 0.8 models and Bruno for TTS 2.
- `capabilities` describes streaming mode, cloning availability, and normalization span support. `close()` is safe to call repeatedly; closed instances reject synthesis. Instances support context managers. On Metal, reuse each model: its GPU weights stay allocated until the process exits ([details](../engines/lite/README.md#models-and-voices)).

`speed` and `normalize` are synthesis options for both models. TTS 0.8 applies the model's speed priors; its `generate` and `stream` default to normalization off, while file generation defaults to normalization on. TTS 2 enables normalization by default. Unsupported options raise errors.

## Model-specific operations

Kitten TTS 2 mini, nano and pico come with the plain install; English and Japanese text need Python 3.10 to 3.14, and Japanese is not available on Windows on Arm ([details](../README.md#kitten-tts-2-mini-nano-and-pico-the-cpu-the-mac-gpu-and-nvidia-gpus)). Synthesis options are `language` (a code from `languages()`; default `en`) and `speed`. `list_voices(language=None)` lists voice names; a voice is chosen by name in any case, and a name that only one language has also picks that language. Constructor options are `precision` (`fp32`, the default, or `int8`: smaller, and faster on most Arm64 CPUs; on x86-64 slower on CPUs without AVX-512 VNNI and on Windows, faster on Linux on CPUs with it, where its speech quality was checked only for pico and nano with the AVX2 kernels; lossier than the official int8 files, see the [README](../README.md#kitten-tts-2-mini-nano-and-pico-the-cpu-the-mac-gpu-and-nvidia-gpus)), `revision` (default: the revision this release was tested with), `low_memory` (`True` uses the least memory on the CPU, at some cost in speed; `--low-memory` on the command line) and `cache_dir`. To choose the number of CPU threads, set `OMP_NUM_THREADS` before the first model loads: `OMP_NUM_THREADS=4 kitten-tts ...` in a Linux or macOS shell, `$env:OMP_NUM_THREADS=4` in PowerShell before running `kitten-tts`, or `os.environ["OMP_NUM_THREADS"] = "4"` in Python before `import kitten_inference`. `stream` yields one array per text chunk, and `phonemize(text, language=None)` returns those chunks as phoneme strings (what the C library's `kt08_synthesize_tiny` takes on phones). The same text gives the same audio on the same machine and thread count.

TTS 0.8 standalone normalization is available through `tts.normalize_text(text, locale="en-US", return_spans=False)` and the root exports `normalize_text`, `normalize_text_result`, `NormalizedSpan`, and `NormalizedTextResult`. This normalizer is not used for TTS 2.

TTS 2 constructor options are `revision` (latest by default), `device` (GPU 0), `memory_profile` (`auto`), and `voice_cache` (automatic). Synthesis options include `seed`, `preset` (`stable` or `expressive`), `temperature`, `top_p`, `top_k`, `min_p`, `max_new_tokens`/`max_tokens`, and `speed`. `generate(stream=True)` renders through the streaming path and returns all audio.

Use `clone_voice(reference_audio_path, transcript, **options)` to create a voice, `save_voice(voice, path)` to save it, and `load_voice(path)` to reuse it. `Voice` is exported from `kitten_inference`. Check `can_clone()` and the [GPU memory and FFmpeg requirements](../README.md#kitten-tts-2-nvidia-and-amd-gpus) before cloning. Saved voices require the same model revision and preparation version. TTS 2 also exposes `revision`, `model_dir`, `build_info()`, and `stats()` for diagnostics.

Public exceptions are exported from `kitten_inference`: `EngineError`, `EngineNotInstalled`, `EngineMismatch`, `PrepareError`, `BundleError`, `AudioError`, and `FFmpegNotFound`.

## Command line

```sh
kitten-tts --model KittenML/kitten-tts-nano-0.8 --text "Hello." --voice Jasper --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --text "Hello." --voice Bruno --output hello.wav
kitten-tts --model KittenML/kitten-tts-2 --prepare-only --with-cloning
```

`python -m kitten_inference` is equivalent. Use `kitten-tts --model MODEL --help` for model-specific options. For Kitten TTS 2 mini, nano and pico, `--list-voices` prints one line per language (`--language de` lists one language). TTS 0.8 accepts text or a UTF-8 text file, voice listing, backend/cache selection, speed, and normalization. TTS 2 also supports output formats, reports, streaming, repeat/concurrency/thread controls, sampling, voice-cache limits, and cloning. Every model takes `--cache-dir` to choose its cache location; Kitten TTS 2 also takes `--revision` to pin a model version (from Python, mini, nano and pico take `revision=`).

## Source builds

`pip install .` builds the TTS 0.8 extension. Source installs do not include TTS 2 GPU binaries; use the PyPI package for GPU inference, or follow the [native build instructions](../engines/tts2/README.md#build-from-source).
