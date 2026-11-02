# kitten-inference

Fast text-to-speech with Kitten TTS 2 on NVIDIA and AMD GPUs and Kitten TTS 0.8 on the CPU and the Mac GPU.
Install with pip; no PyTorch needed.

| Model | Runs on | Speed |
|---|---|---|
| Kitten TTS 2 | NVIDIA and AMD GPUs | complete files 5 to 7 times faster than on vLLM or llama.cpp on an RTX PRO 6000 (2.4 times or more on an RTX 3060) |
| Kitten TTS 0.8 | the CPU and the Mac GPU | nano 1.1 to 2.3 times faster than kittenml (ONNX Runtime) |

![Kitten TTS 2 native against vLLM and llama.cpp: time to make a complete file on an RTX PRO 6000, RTX 4090 and RTX 3060](engines/tts2/docs/assets/benchmarks/tts2-complete-files.svg)

![Kitten TTS 0.8 engine against ONNX Runtime on the CPU, 80M int8 model: latency and peak memory on an Android phone (Cortex-A77), a Core i7-12650H laptop and an Apple M2 Pro](engines/tts08/docs/assets/benchmarks/cpu_onnx_summary.svg)

The Kitten TTS 0.8 chart is the 80M int8 model on the CPU only (the table's numbers are nano, on the Mac GPU on Macs).
More numbers: [Kitten TTS 2](engines/tts2/README.md#performance), [Kitten TTS 0.8](engines/tts08/README.md).

## Install

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip
pip install kitten-inference
```

- On Ubuntu and Debian, `python3 -m venv` needs `sudo apt-get update && sudo apt-get install -y python3-venv` first.
- On Windows, run the commands in a Command Prompt (in PowerShell, type `cmd` first), with `py -m venv .venv` and then
  `.venv\Scripts\activate.bat` in place of the first line.

What each model needs (GPU, driver, Python version): [Kitten TTS 2](engines/tts2/README.md#install-with-pip),
[Kitten TTS 0.8](engines/tts08/README.md#platforms).

## Use

```sh
kitten-tts --model KittenML/kitten-tts-2 --text "Hello from Kitten TTS 2." --voice Bruno --output hello.wav
kitten-tts --model KittenML/kitten-tts-nano-0.8 --text "Hello from Kitten TTS." --voice Jasper --output hello.wav
```

```python
from kitten_inference import KittenTTS

tts = KittenTTS("KittenML/kitten-tts-2")
tts.save_wav("hello.wav", tts.generate("Hello from Kitten TTS 2.", voice="Bruno"))
for chunk in tts.stream("Streaming speech, chunk by chunk.", voice="Bruno"):
    ...
```

The first use of a model downloads it from Hugging Face.

## Docs

- [Python API and `kitten-tts` command](docs/api.md): voices, streaming, voice cloning, saved voices, and model
  options.
- [Kitten TTS 2](engines/tts2/README.md): requirements, speed, [supported GPUs](engines/tts2/README.md#supported-gpus),
  GPU memory, building from source, the C API. Also: [Windows](engines/tts2/docs/windows.md),
  [AMD Radeon](engines/tts2/docs/amd-radeon.md), [GPU memory](engines/tts2/docs/memory.md),
  [speed and audio by GPU group](engines/tts2/docs/consumer-gpus.md).
- [Kitten TTS 0.8](engines/tts08/README.md): speed, models and voices, options, platforms, and upgrading from 0.1.x
  (`import model_inference` needs `kitten-inference==0.1.1`).

## License

KittenML's code is Apache-2.0 ([LICENSE](LICENSE)); the package also contains third-party code under other licences
([NOTICE](NOTICE), [engines/tts2/NOTICE](engines/tts2/NOTICE), [engines/tts08/NOTICE](engines/tts08/NOTICE)). The models download on first use and keep their publishers' licences. The Kitten TTS 2 model is
under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not
Apache or MIT: it limits commercial use, training on the outputs and redistribution, so read it before you build on
Kitten TTS 2. The Kitten TTS 0.8 model licences are in [engines/tts08/README.md](engines/tts08/README.md#license).
