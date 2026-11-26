# Provenance

This file records where the Kitten ASR code comes from and what it does not contain. The code is in
`src/kitten_inference/asr/` (the Python engine for NVIDIA GPUs, and the Python side of the CPU engine),
`engines/asr-cpu/` (the compiled CPU engine: `src/kasr.cpp` and `include/kasr.h` are KittenML's, `llama.cpp/` is a
copy of [llama.cpp](https://github.com/ggml-org/llama.cpp) at commit 9048bad, MIT, with our changes marked `kitten:`)
and `engines/asr/` (this folder: documents and the build input of the release wheels). Licenses and copyright notices are in [NOTICE](NOTICE).

## Origin of the code

The engine is written by KittenML for this repository and licensed under Apache-2.0 ([LICENSE](LICENSE)), except
the parts in the table below, which follow or are adapted from other projects. It is Python that runs on PyTorch:

- an audio front end (reads a file, mixes to mono, resamples to 16 kHz, computes log-mel features);
- a loader that reads the model files;
- the model itself: an audio encoder and a text decoder;
- a paged key/value cache with the attention kernels that read it;
- lower-precision copies of the decoder's weights (8-bit and 4-bit integers, 8-bit floats), made on the GPU from the
  published 16-bit weights when the model loads, with the Triton kernel that multiplies by them;
- a scheduler that decodes the requests of many callers together on one GPU, in the caller's process or in a
  process of its own.

## Code that follows published work

| Part | Follows | License |
|---|---|---|
| The Triton attention kernels (`_attn.py`: prefill, decode in one pass, decode with the keys cut into pieces and its merge step) | The structure of the Triton attention kernels in [SGLang](https://github.com/sgl-project/sglang), Copyright 2023-2024 SGLang Team, including its two-stage decode (partial results per piece of the keys, then a merge). SGLang's decode kernel is adapted from [LightLLM](https://github.com/ModelTC/lightllm). | Apache-2.0 (both) |
| The same kernels | The fused-attention tutorial of [Triton](https://github.com/triton-lang/triton) | MIT |
| The low-precision matrix product (`_qmm_kernel` in `_quant.py`) | The tiling of the matrix multiplication tutorial of [Triton](https://github.com/triton-lang/triton) (grouped program order, float32 accumulation); the weight formats, the in-loop conversion and the split over the inputs are KittenML's | MIT |
| The Slaney-scale mel filter bank (`_mel_filters` in `_loader.py`) | Adapted from `transformers.audio_utils.mel_filter_bank` of [Hugging Face Transformers](https://github.com/huggingface/transformers), Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors. | Apache-2.0 |
| The sinusoidal position table of the audio encoder (`_sinusoid` in `_loader.py`) | The formula of the `sinusoids` function in [OpenAI Whisper](https://github.com/openai/whisper), Copyright (c) 2022 OpenAI. | MIT |

## Not included

- **Model weights.** They are downloaded from Hugging Face on first use and keep the license on their Hugging Face
  pages.
- **PyTorch.** It is installed separately (`pip install torch`). The package's own attention kernels are compiled by
  Triton when they are used; Triton is not in the package either.
- **Anything compiled for NVIDIA GPUs.** Every release wheel carries the same Python files in
  `kitten_inference/asr/`. The compiled CPU engine is in `kitten_inference/_asr_cpu/` of the wheels for Linux
  (x86-64 and arm64) and Apple silicon Macs; [NOTICE](NOTICE) lists what it is built from.
- **A network server.** `Engine` and `EngineProcess` are the parts a server is built from.
