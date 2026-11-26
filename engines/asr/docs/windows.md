# Kitten ASR on Windows

Use Windows 10, 11 or Server 2022 x64 and Python 3.12 x64. The CPU backend needs AVX2 and runs without PyTorch. The CUDA backend needs an NVIDIA Ampere GPU or newer and driver 570.65 or newer. Install the x64 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). Allow 16 GB of disk space for the environment and model caches. Windows ARM64 is unsupported. On Windows, run the CUDA engine in your Python process with the default `process=False`.

## Install and transcribe

Open a Command Prompt in a short, writable folder. The first transcription downloads and converts the model. This example uses a [sample WAV file](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav).

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install "kitten-inference>=0.2.1"
curl.exe --fail --location --output speech.wav https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav
kitten-asr --model KittenML/kitten-asr-tiny --backend cpu speech.wav
python -c "from kitten_inference import KittenASR; asr = KittenASR('KittenML/kitten-asr-tiny', backend='cpu'); print(asr.transcribe('speech.wav')); asr.close()"
```

## Use an NVIDIA GPU

In the same Command Prompt, install [PyTorch with CUDA](https://pytorch.org/get-started/previous-versions/#v291).

```bat
nvidia-smi
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
kitten-asr --model KittenML/kitten-asr-tiny --backend cuda speech.wav
```

PyTorch handles attention and int8 weights when Triton is missing. To enable Triton with this PyTorch version, install [triton-windows 3.5](https://github.com/triton-lang/triton-windows).

```bat
python -m pip install "triton-windows>=3.5.0.post13,<3.6"
kitten-asr --model KittenML/kitten-asr-tiny --backend cuda --precision int8 speech.wav
```

Omit `--backend` to use CUDA when available and the CPU otherwise. CUDA accepts `bf16` (default), `fp16` and `int8`. The CPU accepts `q8_0` (default), `f16` and `bf16`. Small-enhanced also accepts its published `int4` and `int8` weights on both backends. Install Triton for packed CUDA inference. [Published precisions](../README.md#published-int4-and-int8)

```bat
kitten-asr --model KittenML/kitten-asr-small-enhanced --backend cpu --precision int4 speech.wav
kitten-asr --model KittenML/kitten-asr-small-enhanced --backend cuda --precision int8 speech.wav
```

## Live audio

Pass 16 kHz mono samples from your recorder to `stream().push()` as they arrive. Save the example as `live.py` next to `speech.wav` to try the same interface with a file.

```python
import soundfile
from kitten_inference import KittenASR

with KittenASR("KittenML/kitten-asr-tiny", backend="cpu") as asr:
    with asr.stream(language="English") as stream:
        for chunk in soundfile.blocks("speech.wav", blocksize=8000, dtype="float32"):
            update = stream.push(chunk)
            if update is not None:
                print(update.text)
        print(stream.finish())
```

```bat
python live.py
```

## Speed

On Windows Server 2022, a 16-second clip ran at 6.3× and 2.7× real time with tiny and small-enhanced on an 8-vCPU Xeon (`q8_0`), and at 105× and 39× on an NVIDIA L4 (`bf16` with Triton). These are medians of three calls.

## Cache and troubleshooting

- Hugging Face keeps downloads under `%USERPROFILE%\.cache\huggingface\hub`. The CPU engine keeps converted models under `%USERPROFILE%\.cache\huggingface\kitten-inference\asr`. Set `HF_HOME` before launching Python to move both, or `KITTEN_ASR_CACHE` to move just the converted files.
- If Windows reports a missing DLL, install the x64 Visual C++ Redistributable and recreate the virtual environment in a short path such as `C:\ki\.venv`. If a cache path is too long, set `HF_HOME` to a shorter writable path.
- If antivirus is scanning a new cache file and Windows reports a sharing violation, wait for the scan and retry. Set `KITTEN_ASR_CACHE` to use a different writable cache directory.
- If PyTorch prints `False` for CUDA, check `nvidia-smi` and install the CUDA wheel shown above. For a CUDA compiler cache path error, set `TORCHINDUCTOR_CACHE_DIR` to a short writable path. If compilation still fails, try `compile=False` in Python. For a CUDA graph capture error, try `cuda_graph=False`.

For a source checkout, run `python -m pip install .` in an **x64 Native Tools Command Prompt for VS 2022** with the C++ build tools installed.
