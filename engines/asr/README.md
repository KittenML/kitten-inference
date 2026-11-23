# Kitten ASR

Speech to text on NVIDIA GPUs, on CPUs and on Macs, from one class: `KittenASR(model)`. On an NVIDIA GPU one engine
transcribes files and live audio and decodes the audio of many requests together; it is Python code that runs on
PyTorch. On a CPU (x86-64 with AVX2, or 64-bit Arm) and on the GPU of Macs with Apple silicon, a compiled engine built
on llama.cpp's libraries runs the same models with no PyTorch. Both are part of the `kitten-inference` package.

On an RTX 4090 a 15 to 26 s clip takes 0.12 s with tiny and 0.20 s with small-enhanced.

**On a CPU, tiny transcribes 26 times and small-enhanced 20 times faster than real time on 32 threads of a Xeon
Gold 6430 (Sapphire Rapids), 18 and 13 times on the 10 fast cores of an Arm NVIDIA GB10, and 8 and 6 times on a 16-core
Ryzen 9 5900XT** (one clip of 15 to 26 s per call, default settings), using 1.4 to 2.3 GB of memory. Many files in
one call and long recordings are faster still with tiny ([Speed on a CPU](#speed-on-a-cpu)).

## Install

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip
pip install kitten-inference
```

- Python 3.10 or newer.
- On a CPU: Linux on x86-64 with AVX2 (Intel 2013 and later, AMD 2015 and later) or on 64-bit Arm, and Macs with Apple
  silicon. Nothing else to install. On Windows x64 with AVX2, see the [Windows guide](docs/windows.md).
- On an NVIDIA GPU: Linux or Windows x64, a GPU of the Ampere generation or newer (RTX 30 series and later), and
  CUDA-enabled PyTorch. Use `pip install torch` on Linux and follow the [Windows guide](docs/windows.md) on Windows.
- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.

## Use

```python
from kitten_inference import KittenASR

asr = KittenASR("KittenML/kitten-asr-small-enhanced")    # an NVIDIA GPU when it can use one, otherwise the CPU
text = asr.transcribe("meeting.wav")

# A list is transcribed in one call and returns one text per item.
texts = asr.transcribe(["a.wav", "b.wav"])

# Samples in an array need their sample rate. details=True returns a Transcript.
result = asr.transcribe(audio_array, sample_rate=48000, language="English", details=True)
print(result.text, result.language, result.segments)

asr.close()
```

The first use of a model downloads it from Hugging Face; later uses load it from the local cache. A local model
folder works in place of the name.

- Audio: a file path, or a NumPy array of float or int16 samples (1-D, or 2-D as `[samples, channels]`) with
  its `sample_rate` (16000 when left out), or a list of these.
- `language`: the name of the spoken language, for example `"English"`. Without it the model detects the language.
- `details=True`: a `Transcript` with `text`, `language`, `segments` (the text of each piece of the audio with its
  `start` and `end` in seconds) and `raw` (the model's output for each piece, unparsed).
- Audio longer than 30 s is cut into pieces at quiet points, and the texts are joined with a space. Silence, and
  audio shorter than 0.1 s, give an empty text.
- The model sometimes repeats a phrase until it runs out of room (its limit of words for the piece). When a piece
  stops at that limit and its text ends in the same words repeated again and again, one copy is kept. This applies
  on every backend, to files and to live audio. A piece that ends by itself keeps every word, so speech that really
  repeats itself ("Thank you. Thank you.") is not shortened.

### Live audio

```python
with asr.stream() as s:
    for chunk in chunks:
        update = s.push(chunk)
        if update is not None:
            print(update.text)
    final = s.finish()
```

- `push(chunk)` takes mono samples at 16 kHz (float32 or int16), in chunks of any length.
- It returns a `StreamUpdate` when the text was brought up to date, and `None` when the chunk only went into the
  buffer. `committed` is the text that is final, `partial` the newest text, which may still change, and `text` is
  both together.
- `finish()` transcribes what is left and returns the full text.
- `asr.stream(language="English")` names the language. `step` (0.5 by default) is how many seconds of new audio
  bring an update, and `max_window` (20 by default) how many seconds of audio stay open to change before their
  text is final.
- No update is made until the session hears something clearly louder than its background (15 dB above it, or
  louder than -30 dB of full scale), so quiet noise before speech costs nothing and gives no invented words; after a
  window is cut at `max_window`, a window that holds only background noise is not transcribed either.
  `finish()` always transcribes the first window. `asr.stream(speech_gate=False)` turns both off.
- On the CPU, give each live session its own model when several run at once: sessions that share a model take
  turns on its threads.

### Backends

`KittenASR(model, backend=None, precision=None)`. Leave `backend` out and it is picked for you; `asr.capabilities`
says which one runs (`"backend"`, `"precision"`).

| `backend` | Runs on | Needs | Default `precision` | Other precisions |
|---|---|---|---|---|
| left out | an NVIDIA GPU when PyTorch is installed and finds one that can run the model; otherwise the CPU engine (on a Mac with Apple silicon: the Mac GPU) | - | the backend's | the backend's |
| `"cuda"` | an NVIDIA GPU | Linux or [Windows x64](docs/windows.md), PyTorch, a GPU of the Ampere generation or newer | `"bf16"` | `"fp16"`, `"int8"` |
| `"cpu"` | the CPU | Linux x86-64 (AVX2) or arm64, [Windows x64 (AVX2)](docs/windows.md), or a Mac with Apple silicon | `"q8_0"` | `"f16"`, `"bf16"` |
| `"metal"` | the GPU of a Mac with Apple silicon | macOS on Apple silicon | `"q8_0"` | `"f16"`, `"bf16"` |

- A backend that cannot run on the computer, and a precision the backend does not offer, raise an error that says
  what to use instead, before anything is downloaded. A named backend never falls back to another one.
- The CPU engine needs no PyTorch. On first use it converts the model to its own format with NumPy (a few
  seconds) and keeps the result next to the Hugging Face cache (`KITTEN_ASR_CACHE` moves it).
- `threads` (CPU engine): one per physical core on Linux and macOS, counting only fast cores on hybrid CPUs.
  Windows counts available logical CPUs, using the fast cores on Intel hybrid CPUs.
- `batch` (CPU engine): how many pieces of audio are decoded in the same steps when a call holds several (a list of
  files, or a recording longer than 30 s). The default is 8 for tiny and 1 for small-enhanced; `batch=8` turns it on
  for small-enhanced ([Speed on a CPU](#speed-on-a-cpu); no language measured worse, 8 of its 30 not yet settled),
  and `batch=1`
  turns it off. It needs about 1.2 GB more memory (tiny: 2.6 instead of 1.4 GB at the peak), from the first call with
  several pieces until the model is closed. On a Mac the default is 1 (decoding several pieces at once has not been
  checked on Apple hardware). A file's text can differ slightly between transcribing it alone and in a list, because
  the order of additions changes (for tiny, the same error rate on FLEURS).
- With `backend` left out, options that only the other engine knows (such as `threads` and `batch` on a computer
  with an NVIDIA GPU) are left out, so the same code runs on both kinds of computer.
- Everything else is the same on every backend: lists of files, `details=True`, `language`, live audio with
  `stream()`.

### Command line

```sh
kitten-asr --model KittenML/kitten-asr-small-enhanced meeting.wav
kitten-asr --model KittenML/kitten-asr-small-enhanced meeting.wav --language English --json
```

The command prints the text. With several files it prints one line per file, with the file name in front.
`--language NAME` names the spoken language, `--backend` picks `cuda`, `cpu` or `metal`, `--precision` picks the
number format of the weights ([Precision](#precision)), `--cache-dir DIR` sets where downloaded models are kept, and
`--json` prints one JSON object per file instead: `file`, `text`, `language` and `segments`.
`python -m kitten_inference.asr` is the same command.

## Models

| Model | Parameters | Languages |
|---|---|---|
| `KittenML/kitten-asr-tiny` | 0.3B | English, Spanish, Portuguese, Italian, German, French, Russian |
| `KittenML/kitten-asr-small-enhanced` | 0.6B | 30 languages; also writes tags such as emotions and sounds into the text |

A model name is required: one of these, or a local folder with a model of the same family.

## Precision

### Published INT4 and INT8

Small-enhanced accepts `precision="int4"` and `precision="int8"`. Linux and Windows CPU inference caches
the published matrices in packed GGUF. CUDA with Triton reads packed matrices in its matmul kernels.
The audio convolutions, macOS, and CUDA without Triton use 16-bit weights.

For a local copy, keep the configuration and tokenizer at the root and the selected `int4/` or `int8/`
folder beside them. CUDA models without published INT8 use the conversion below. Published INT4 increased
multilingual errors against BF16. [Published files](https://huggingface.co/KittenML/kitten-asr-small-enhanced/tree/main)

### On-device conversion

`KittenASR(model, precision=...)` (or `--precision`) picks the number format of the text decoder's weights. The two
engines name them differently: the CPU engine uses its own format names (`q8_0` is 8-bit weights, `f16` is
`fp16`). The formats in the table are made from the original weights on first load. The table's accuracy results
come from FLEURS (30 languages for small-enhanced, 7 for tiny) and an English test set. A change is measurably worse
when its entire 95% interval is above +0.5 points. "The same" below means the whole interval is within +0.5 points; for
small-enhanced some languages are not settled either way.

| Precision | NVIDIA GPU | CPU and Mac GPU |
|---|---|---|
| 16-bit, as published | `"bf16"`: supported, the default | `"bf16"`: supported; slower than `q8_0` |
| 16-bit floats | `"fp16"`: supported; the same speed as `bf16`; the same accuracy for tiny; for small-enhanced 11 of 30 languages not settled | `"f16"`: supported; for tiny the same accuracy as `bf16`; twice the memory of `q8_0` |
| 8-bit weights | On-load `"int8"`: supported, not optimal: 1.05 to 1.43 times faster for one stream; slower than `bf16` when many streams are decoded together; for small-enhanced 6 of 30 languages not settled (Persian, Finnish, Hungarian, Macedonian, Portuguese, Thai; measured up to 1.4 points worse) | `"q8_0"`: supported, the default: the same accuracy as the 16-bit weights for tiny in its 7 languages, and no language measurably worse for small-enhanced |

- Why the defaults differ: a GPU is fastest with 16-bit weights; on a CPU, writing each word reads every weight
  once, so 8-bit weights are the faster default there. The two 8-bit formats differ (one scale per 128 weights on
  the GPU, per 32 on the CPU).
- Locally rounded INT4 and FP8 on NVIDIA, and locally converted 6-, 5- and 4-bit GGUF formats, are excluded after accuracy tests.
- `encoder_precision` (CPU engine) sets the audio encoder's format: `"f16"` by default. `"q8_0"` is supported, not
  optimal: about 1.2 times faster on Arm and 1 to 2% on x86, but it fails the accuracy rule in Macedonian with
  small-enhanced.

## Speed on a CPU

Default settings, three rounds each; "x real time" is seconds of audio per second of computing.

| Machine (threads) | tiny: 40 files in one call | tiny: an hour-long recording |
|---|---|---|
| Ryzen 9 5900XT (16) | 16.6x real time | 26x |
| Xeon Gold 5418Y, Sapphire Rapids (16) | 33.9x | 46x |
| Arm NVIDIA GB10 (10 fast cores) | 27.5x | 37x |

Decoding several pieces of audio in each step (`batch`, under [Backends](#backends)) makes many files 1.4 to 1.9
times faster with tiny, and 1.5 to 2.4 times with small-enhanced at `batch=8`.

## Attention kernels

The engine keeps the keys and values of all requests in one paged pool on the GPU, and picks the attention kernel
that can read that pool:

| What is installed | Attention kernel | Status |
|---|---|---|
| Linux with a C compiler and the Python headers (Triton is installed with PyTorch) | The package's own Triton kernels | Supported |
| Anything else | Plain PyTorch attention | Supported, not optimal: it copies each sequence's keys on every step and runs more small GPU calls, so it slows down as more audio is decoded at once |

The engine warns once at start when it runs on the second row. On Ubuntu and Debian,
`sudo apt-get update && sudo apt-get install -y gcc python3-dev` installs what the first row needs.

## Platforms

| Platform | Status |
|---|---|
| Linux with an NVIDIA GPU (Ampere or newer), Python 3.10 or newer | Supported |
| Linux x86-64 (AVX2 or newer) and Linux arm64, CPU, glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL 8 or later) | Supported. Older Linux systems get the wheel without the CPU engine, which then says so |
| macOS on Apple silicon | Supported, not optimal: the default is the Mac GPU (`backend="metal"`), which runs but is not measured on a real Mac yet; the CPU (`backend="cpu"`) was measured only in a macOS virtual machine (Apple M2, 4 threads: tiny 14 to 16 times and small-enhanced 7.4 times faster than real time) |
| Windows x64 | CPU and NVIDIA GPU. See [Windows setup](docs/windows.md) |
| Windows ARM64 and Macs with Intel processors | Not supported |
| Phones | Not in this package |

When it loads a model, `KittenASR` checks what the chosen backend needs (Python, the system, PyTorch and the GPU, or
the CPU engine) and says which one is missing. The Python files are in every `kitten-inference` wheel, so
`import kitten_inference.asr` works everywhere and needs no PyTorch.

## Servers

On Linux, `KittenASR(model, process=True)` runs the NVIDIA GPU engine in a process of its own; use it in a server whose own
threads keep Python busy. A server that schedules the audio of many callers itself uses the engine below `KittenASR`:

- `kitten_inference.asr.Engine` decodes the audio of all callers together on one GPU. `decode` transcribes one
  piece of audio and returns a `DecodeResult` (`text`, `token_ids`, `info`), `release` frees what the engine kept
  for a stream, and `snapshot` reports the engine's counters and settings, among them the attention kernel in use
  (`attention`), whether the model math is compiled (`compiled`), the `precision` and the encoder's attention window
  (`encoder_window`).
- `kitten_inference.asr.EngineProcess` has the same methods and runs the engine in a process of its own, so a busy
  server process cannot slow the decoding down.
- Errors: `EngineError`, and under it `PoolExhausted` (no room for the request; it fails and is never returned cut
  short) and `EngineStopped` (the engine was closed or stopped; create a new one).

## License

The CPU engine includes llama.cpp and ggml (MIT, `engines/asr-cpu/llama.cpp/LICENSE`); our changes to them are
marked `kitten:` in the source. The code is licensed under Apache-2.0. Its Triton attention kernels follow kernels of SGLang and LightLLM
(Apache-2.0) and Triton's tutorials (MIT), and two short functions of its loader follow Hugging Face Transformers
(Apache-2.0) and OpenAI Whisper (MIT): see [NOTICE](NOTICE) and [PROVENANCE.md](PROVENANCE.md). Each wheel carries the notice in its
`.dist-info/licenses/asr/` folder.

The model weights are not in the package. They download on first use and keep the licence on their Hugging Face
pages ([tiny](https://huggingface.co/KittenML/kitten-asr-tiny),
[small-enhanced](https://huggingface.co/KittenML/kitten-asr-small-enhanced)). PyTorch (for NVIDIA GPUs) is installed
separately under its own licence.
