# Kitten TTS 2 mini, nano and pico on Windows

Use Windows 10, 11 or Server 2022 x64, an AVX2 CPU and Python 3.12 x64 from [python.org](https://www.python.org/downloads/windows/). Install the [Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) if Windows cannot load a DLL. The first use downloads the model. Japanese on mini also downloads a dictionary.

## Install and generate a WAV file

Open a Command Prompt in a short, writable folder.

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install "kitten-inference>=0.2.1"
kitten-tts --model KittenML/kitten-tts-2-mini --backend cpu --precision fp32 --language en --text "Hello from Kitten." --output mini.wav
kitten-tts --model KittenML/kitten-tts-2-nano --backend cpu --precision int8 --language de --text "Guten Morgen." --output nano.wav
kitten-tts --model KittenML/kitten-tts-2-pico --backend cpu --precision int8 --language en --text "Hello from Kitten." --output pico.wav
```

Each model accepts both CPU precisions. See the [model table](../README.md#kitten-tts-2-mini-nano-and-pico) for languages. In Python, stream the audio in chunks.

```bat
python -c "from kitten_inference import KittenTTS; tts = KittenTTS('KittenML/kitten-tts-2-mini', backend='cpu'); print([len(chunk) for chunk in tts.stream('Hello from Kitten. This is streamed speech.', language='en', chunk_chars=24)]); tts.close()"
```

## NVIDIA GPU

Use NVIDIA driver R570 or newer and a wheel that includes `kt08_cuda.dll`. CUDA uses float32.

```bat
nvidia-smi
kitten-tts --model KittenML/kitten-tts-2-mini --backend cuda --language en --text "Hello from Kitten." --output mini-cuda.wav
```

If the package reports a missing CUDA engine, use `--backend cpu`. If Windows cannot load the engine from a deeply nested folder, recreate the virtual environment in a shorter path.
