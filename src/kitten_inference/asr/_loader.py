"""Model loader of the ASR engine: a model folder -> the tensors, tokenizer and prompt text the engine runs on."""
# Needs torch, numpy and tokenizers (huggingface-hub when the model is a repo id). config.json is read as
# plain JSON; the safetensors file(s) are memory-mapped and copied to the device tensor by tensor.
#
# Rules this loader follows (each one can change tokens silently if done differently):
#   - the output head is `thinker.lm_head.weight` when the file has it, else the embedding table (tied models
#     do not store a head);
#   - head_dim comes from the config, never from hidden_size / heads;
#   - stop ids and the audio placeholder id are checked against the tokenizer, not taken from generation
#     settings;
#   - tokenizer.json is used exactly as stored (padding and truncation off, pre-tokenizer rules not rewritten);
#   - decode() drops only the tokens flagged special in tokenizer.json, so <asr_text> and <non_speech> stay;
#   - the mel filter bank is built in float64 (Slaney scale, Slaney norm) and only then cast to float32;
#   - the encoder position rows are computed in float32 (sin block, then cos block) and then cast.
from __future__ import annotations

import json
import math
import os
import struct
import warnings

import numpy as np
import torch

__all__ = ["Weights", "load"]

SR = 16000          # the engine's STFT is fixed: 16 kHz, 400-sample window, hop 160 -> 201 frequency bins
N_FFT = 400
HOP = 160

_AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
_STOP = ("<|im_end|>", "<|endoftext|>")
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# safetensors dtype code -> (numpy dtype the bytes are viewed as, torch dtype to reinterpret them as)
_STORED = {"BF16": ("<i2", torch.bfloat16), "F16": ("<f2", None), "F32": ("<f4", None)}


class Weights:
    """What the engine needs from a checkpoint.

    cfg          dict: H, L, nh, nkv, hd, eps, rope_theta, vocab, enc_d, enc_nh, enc_layers, n_mels,
                 chunk_frames (mel frames per conv chunk), block_frames (mel frames per attention block), audio_id
    embed        [vocab, H]          norm_w [H]          head_w [vocab, H] (the embed tensor itself when tied)
    W            per decoder layer: (ln1 [H], QKV [(nh + 2 nkv) hd, H], q_norm [hd], k_norm [hd], o [H, nh hd],
                 ln2 [H], gate+up [2 I, H], down [H, I])
    conv         [(w, b)] x 3 for F.conv2d(y, w, b, stride=2, padding=1);  conv_out_w [enc_d, C * F], no bias
    enc_pos      [tokens per conv chunk, enc_d] sinusoid rows
    EW           per encoder layer: (ln1 w, b, QKV w, b, out w, b, ln2 w, b, fc1 w, b, fc2 w, b)
    ln_post, proj1, proj2   (w, b) each
    mel_filters  [n_mels, 201] float32
    tok          encode(text) -> ids (no special tokens added), decode(ids) -> str (special tokens dropped),
                 token_to_id(token) -> int or None
    eos          set of stop ids
    prompt       prompt(system="", language=None) -> str with exactly one audio placeholder
    folder       the local model folder the weights came from (None when built by hand)
    """

    def __init__(self, *, cfg, embed, norm_w, head_w, W, conv, conv_out_w, enc_pos, EW, ln_post, proj1, proj2,
                 mel_filters, tok, eos, prompt, folder=None, published_precision=None):
        self.cfg = cfg
        self.embed, self.norm_w, self.head_w, self.W = embed, norm_w, head_w, W
        self.conv, self.conv_out_w, self.enc_pos, self.EW = conv, conv_out_w, enc_pos, EW
        self.ln_post, self.proj1, self.proj2 = ln_post, proj1, proj2
        self.mel_filters = mel_filters
        self.tok, self.eos, self.prompt = tok, eos, prompt
        self.folder = folder
        self.published_precision = published_precision


# ------------------------------------------------------------------ small pieces
def _dtype(dtype):
    d = dtype if isinstance(dtype, torch.dtype) else _DTYPES.get(dtype)
    if d is None or not d.is_floating_point:
        raise ValueError(f"dtype must be a floating torch dtype or one of {sorted(_DTYPES)}, got {dtype!r}")
    return d


def _down3(n: int) -> int:
    """Length after the three stride-2 convolutions (kernel 3, padding 1)."""
    for _ in range(3):
        n = (n - 1) // 2 + 1
    return n


def _prompt_text(system: str = "", language: str | None = None) -> str:
    """The text the checkpoint's chat template renders for [system, user(audio)] with a generation prompt."""
    base = (f"<|im_start|>system\n{system or ''}<|im_end|>\n<|im_start|>user\n{_AUDIO}<|im_end|>\n"
            "<|im_start|>assistant\n")
    return base + (f"language {language}<asr_text>" if language else "")


def _stop_ids(tok, cfg) -> set:
    eos = {tok.token_to_id(t) for t in _STOP}
    if None in eos:
        raise ValueError(f"the tokenizer has no id for one of the stop tokens {_STOP}")
    bad = [i for i in sorted(eos) + [cfg["audio_id"]] if not 0 <= i < cfg["vocab"]]
    if bad:
        raise ValueError(f"token ids {bad} are outside the embedding table ({cfg['vocab']} rows)")
    return eos


def _sinusoid(length: int, d: int) -> torch.Tensor:
    """Encoder position table [length, d], float32, sin block then cos block.

    Formula: the `sinusoids` function in OpenAI Whisper, https://github.com/openai/whisper (MIT, Copyright (c) 2022 OpenAI).
    The table is built at full length in float32 and sliced by the caller, so every element goes through the
    same vectorised sin/cos as the table the checkpoints were trained with.
    """
    inc = np.log(10000) / (d // 2 - 1)
    inv = torch.exp(-inc * torch.arange(d // 2).float())
    t = torch.arange(length)[:, np.newaxis] * inv[np.newaxis, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


def _mel_filters(n_mels: int) -> np.ndarray:
    """Mel filter bank [201, n_mels] in float64: Slaney mel scale, Slaney area normalisation, 0..8000 Hz.

    Adapted from transformers.audio_utils.mel_filter_bank (Hugging Face Transformers, Apache-2.0).
    The order of the float64 operations is kept, so the matrix is bit-identical to that one.
    """
    # Slaney scale: linear (200/3 Hz per mel) below 1 kHz, logarithmic above. Filters span 0..8000 Hz.
    mel_max = 15.0 + np.log(8000.0 / 1000.0) * (27.0 / np.log(6.4))
    mels = np.linspace(0.0, mel_max, n_mels + 2)
    hz = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    hz[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[log_region] - 15.0))
    fft = np.linspace(0, SR // 2, 1 + N_FFT // 2)
    # triangles in Hz between neighbouring centre frequencies
    diff = np.diff(hz)
    slopes = np.expand_dims(hz, 0) - np.expand_dims(fft, 1)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down, up))
    fb *= np.expand_dims(2.0 / (hz[2:n_mels + 2] - hz[:n_mels]), 0)      # constant energy per filter
    return fb


# ------------------------------------------------------------------ files
def _resolve(model, cache_dir=None, precision=None) -> str:
    """Local folder of `model`: the folder itself, or a snapshot of the Hugging Face repo with that id
    (downloaded into `cache_dir`, or the default Hugging Face cache)."""
    model = os.fspath(model)
    if os.path.isdir(model):
        return model
    if os.path.exists(model):
        raise ValueError(f"{model} is a file; expected a model folder or a Hugging Face repo id")
    if model.startswith((".", "/", "~")) or "\\" in model or ":" in model or model.count("/") > 1:
        raise FileNotFoundError(f"model folder not found: {model}")
    # "a/b" where the folder "a" exists is a mistyped local path far more often than a repo id: say so here
    # instead of sending a local folder name to the Hub and getting a "repository not found" answer back
    parent = os.path.dirname(model)
    if parent and os.path.isdir(parent):
        raise FileNotFoundError(f"model folder not found: {model} (read as a local path because the folder "
                                f"{parent!r} exists)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(f"{model!r} is not a local folder, and downloading it needs the huggingface-hub "
                          "package") from e
    from kitten_inference.asr import _published   # absolute: tests load this file on its own
    try:
        return _published.download(snapshot_download, model, cache_dir, precision, required=precision == "int4")
    except Exception as e:
        raise RuntimeError(f"{model!r} is not a local folder and could not be downloaded as a Hugging Face "
                           f"repo id ({type(e).__name__}: {e})") from e


def _json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON ({e})") from e


def _config(folder: str):
    """config.json -> (cfg for the engine, further sizes used only to check tensor shapes).
    model_type and class names are ignored: only the layout matters."""
    path = os.path.join(folder, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no config.json in {folder}")
    c = _json(path)

    def sub(d, key, where):
        v = d.get(key) if isinstance(d, dict) else None
        if not isinstance(v, dict):
            raise ValueError(f"{path}: {where}{key} is missing (this loader expects thinker_config with "
                             "text_config and audio_config)")
        return v

    def num(d, key, where, integer=True):
        v = d.get(key)
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or (integer and isinstance(v, float) and not v.is_integer())):
            raise ValueError(f"{path}: {where}.{key} is missing or not a{' whole' if integer else ''} number "
                             f"(got {v!r})")
        return int(v) if integer else v

    th = sub(c, "thinker_config", "")
    tc, ac = sub(th, "text_config", "thinker_config."), sub(th, "audio_config", "thinker_config.")
    t, a = "thinker_config.text_config", "thinker_config.audio_config"
    cfg = dict(H=num(tc, "hidden_size", t), L=num(tc, "num_hidden_layers", t),
               nh=num(tc, "num_attention_heads", t), nkv=num(tc, "num_key_value_heads", t),
               hd=num(tc, "head_dim", t), eps=float(num(tc, "rms_norm_eps", t, integer=False)),
               rope_theta=num(tc, "rope_theta", t, integer=False), vocab=num(tc, "vocab_size", t),
               enc_d=num(ac, "d_model", a), enc_nh=num(ac, "encoder_attention_heads", a),
               enc_layers=num(ac, "encoder_layers", a), n_mels=num(ac, "num_mel_bins", a),
               chunk_frames=2 * num(ac, "n_window", a), block_frames=num(ac, "n_window_infer", a),
               audio_id=num(th, "audio_token_id", "thinker_config"))
    dims = dict(I=num(tc, "intermediate_size", t), ffn=num(ac, "encoder_ffn_dim", a),
                C=num(ac, "downsample_hidden_size", a), out=num(ac, "output_dim", a),
                max_pos=num(ac, "max_source_positions", a))
    small = [k for k, v in {**cfg, **dims}.items() if k != "audio_id" and v <= 0]
    if small:
        raise ValueError(f"{path}: these sizes must be positive: {small}")
    if dims["out"] != cfg["H"]:
        raise ValueError(f"{path}: audio output_dim {dims['out']} differs from the text hidden_size {cfg['H']}")
    if cfg["enc_d"] % cfg["enc_nh"] or cfg["enc_d"] % 2:
        raise ValueError(f"{path}: audio d_model {cfg['enc_d']} must be even and divisible by "
                         f"encoder_attention_heads {cfg['enc_nh']}")
    if cfg["nh"] % cfg["nkv"]:
        raise ValueError(f"{path}: num_attention_heads {cfg['nh']} is not a multiple of num_key_value_heads "
                         f"{cfg['nkv']}")
    if cfg["block_frames"] % cfg["chunk_frames"]:
        raise ValueError(f"{path}: n_window_infer {cfg['block_frames']} is not a multiple of 2 * n_window "
                         f"{cfg['chunk_frames']}")
    if _down3(cfg["chunk_frames"]) > dims["max_pos"]:
        raise ValueError(f"{path}: max_source_positions {dims['max_pos']} is smaller than one conv chunk")
    return cfg, dims


def _check_frontend(folder: str, n_mels: int):
    """The engine computes the spectrogram itself with fixed settings; refuse a checkpoint that wants others."""
    path = os.path.join(folder, "preprocessor_config.json")
    if not os.path.isfile(path):
        return
    pc = _json(path)
    want = dict(feature_size=n_mels, n_fft=N_FFT, hop_length=HOP, sampling_rate=SR)
    bad = {k: pc[k] for k, v in want.items() if isinstance(pc, dict) and k in pc and pc[k] != v}
    if bad:
        raise ValueError(f"{path} says {bad}, but config.json and the engine's spectrogram need {want}")


def _check_template(folder: str):
    """The prompt text is fixed in _prompt_text(); warn if the checkpoint ships a template that looks different."""
    path = os.path.join(folder, "chat_template.jinja")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pieces = ("'<|im_start|>system\\n'", "'<|im_end|>\\n'", "'<|im_start|>user\\n'", _AUDIO,
              "'<|im_start|>assistant\\n'")
    missing = [p for p in pieces if p not in text]
    if missing:
        warnings.warn(f"{path} does not contain {missing}: the built-in prompt text may not match this "
                      "checkpoint's chat template", stacklevel=3)


def _weight_files(folder: str) -> list:
    one = os.path.join(folder, "model.safetensors")
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(one):
        names = ["model.safetensors"]
    elif os.path.isfile(index):
        wm = _json(index).get("weight_map")
        if not isinstance(wm, dict) or not wm:
            raise ValueError(f"{index} has no weight_map")
        names = sorted(set(wm.values()))
    else:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".safetensors"))
    if not names:
        raise FileNotFoundError(f"no *.safetensors weight file in {folder}")
    paths = [os.path.join(folder, n) for n in names]
    gone = [p for p in paths if not os.path.isfile(p)]
    if gone:
        raise FileNotFoundError(f"weight shard(s) listed in {index} but not on disk: {gone}")
    return paths


class _Files:
    """Tensors of one or several safetensors files, read through read-only memory maps.

    File layout (https://github.com/huggingface/safetensors#format): 8 bytes little-endian header length N,
    N bytes of JSON {name: {dtype, shape, data_offsets: [begin, end]}}, then the data; offsets count from the
    end of the header. Tensors are little-endian, row-major.
    """

    def __init__(self, paths, device, dtype):
        self.device, self.dtype = device, dtype
        self.maps, self.dir, self.used = [], {}, set()
        for p in paths:
            size = os.path.getsize(p)
            with open(p, "rb") as fh:
                head = fh.read(8)
                n = struct.unpack("<Q", head)[0] if len(head) == 8 else -1
                if not 2 <= n <= min(size - 8, 100_000_000):
                    raise ValueError(f"{p} is not a safetensors file (header length field {n}, file size {size})")
                try:
                    meta = json.loads(fh.read(n).decode("utf-8"))
                except ValueError as e:
                    raise ValueError(f"{p}: the safetensors header is not valid JSON ({e})") from e
            if not isinstance(meta, dict):
                raise ValueError(f"{p}: the safetensors header is not a JSON object")
            base = 8 + n
            for name, info in meta.items():
                if name == "__metadata__":
                    continue
                try:
                    code, shape = str(info["dtype"]), tuple(int(x) for x in info["shape"])
                    b0, b1 = (int(x) for x in info["data_offsets"])
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{p}: malformed header entry for tensor {name}") from e
                if not 0 <= b0 <= b1 <= size - base:
                    raise ValueError(f"{p}: tensor {name} lies outside the file (incomplete download?)")
                if name in self.dir:
                    raise ValueError(f"tensor {name} is in more than one weight file")
                self.dir[name] = (len(self.maps), code, shape, base + b0, base + b1)
            # the mapping costs no memory by itself: pages are read when a tensor is copied to the device
            self.maps.append(np.memmap(p, dtype=np.uint8, mode="r"))

    def has(self, name: str) -> bool:
        return name in self.dir

    def _host(self, name: str, shape):
        """CPU tensor over the mapped bytes of one tensor, in its stored dtype. Read-only memory: only copy from it."""
        e = self.dir.get(name)
        if e is None:
            some = ", ".join(sorted(self.dir)[:3])
            raise ValueError(f"tensor {name} is missing from the weight file(s) (they hold {len(self.dir)} "
                             f"tensors, e.g. {some})")
        k, code, shp, b0, b1 = e
        if shp != tuple(shape):
            raise ValueError(f"tensor {name} has shape {list(shp)}, but config.json implies {list(shape)}")
        if code not in _STORED:
            raise ValueError(f"tensor {name} is stored as {code}; this loader reads {sorted(_STORED)}")
        npdt, tdt = _STORED[code]
        if b1 - b0 != math.prod(shp) * np.dtype(npdt).itemsize:
            raise ValueError(f"tensor {name}: {b1 - b0} bytes in the file do not match shape {list(shp)} of {code}")
        # numpy has no bfloat16: BF16 bytes are viewed as int16 and reinterpreted by torch
        raw = np.asarray(self.maps[k][b0:b1]).view(npdt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # torch warns that the array is read-only; it is only read from
            t = torch.from_numpy(raw)
        self.used.add(name)
        return (t.view(tdt) if tdt is not None else t).reshape(shp)

    def _put(self, hosts, rows: int):
        """Stack host tensors along dim 0 straight into one new device tensor, then cast.
        The stored bytes go to the device as they are (a real copy on CPU too, never a view of the mapping)."""
        out = torch.empty((rows,) + tuple(hosts[0].shape[1:]), dtype=hosts[0].dtype, device=self.device)
        r = 0
        for t in hosts:
            out[r:r + t.shape[0]].copy_(t)
            r += t.shape[0]
        return out.to(self.dtype)

    def get(self, name: str, shape):
        t = self._host(name, shape)
        return self._put([t], t.shape[0])

    def cat(self, names, shapes):
        """torch.cat of several tensors along dim 0, built in place on the device: the parts never exist
        there on their own, so nothing is held twice."""
        hosts = [self._host(n, s) for n, s in zip(names, shapes)]
        if len({t.dtype for t in hosts}) != 1:
            raise ValueError(f"tensors {list(names)} are stored with different dtypes")
        return self._put(hosts, sum(t.shape[0] for t in hosts))

    linear = get
    fused_linear = cat

    def unused(self, prefixes) -> list:
        return sorted(n for n in self.dir if n.startswith(prefixes) and n not in self.used)


class _PublishedFiles(_Files):
    """Read packed matrices when CUDA can run them, and dense tensors for the other ops."""

    def __init__(self, folder, precision, device, dtype):
        from ._gguf import SafeTensors

        self.device, self.dtype = device, dtype
        self.tensors = SafeTensors(folder, precision)
        self.used = set()

    def has(self, name):
        return self.tensors.has(name)

    def _host(self, name, shape):
        if not self.has(name):
            raise ValueError(f"tensor {name} is missing from the published weight file")
        found = self.tensors.shape(name)
        if found != tuple(shape):
            raise ValueError(f"tensor {name} has shape {list(found)}, but config.json implies {list(shape)}")
        bits = self.tensors.bf16_bits(name)
        if not bits.flags.writeable:
            bits = bits.copy()
        self.used.add(name)
        return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)

    def linear(self, name, shape):
        return self.fused_linear([name], [shape])

    def fused_linear(self, names, shapes):
        from . import _quant

        def supported(name, shape):
            meta = self.tensors.packed.get(name)
            return (meta is not None and meta[0] == tuple(shape)
                    and meta[2] == (128 if meta[3] == 4 else meta[1])
                    and meta[1] % 64 == 0)

        if self.device.type != "cuda" or _quant.triton is None or not all(
                supported(n, shape) for n, shape in zip(names, shapes)):
            return self.get(names[0], shapes[0]) if len(names) == 1 else self.cat(names, shapes)
        self.used.update(names)
        packed = []
        for suffix in ("q", "scale", "zero"):
            parts = [self.tensors.raw(f"{name}.{suffix}") for name in names]
            # Transpose into the kernel's [input, output] layout without expanding the codes.
            transposed = np.empty((parts[0].shape[1], sum(p.shape[0] for p in parts)), dtype=parts[0].dtype)
            start = 0
            for part in parts:
                transposed[:, start:start + part.shape[0]] = part.T
                start += part.shape[0]
            packed.append(torch.from_numpy(transposed).to(self.device))
        return (*packed, _quant.PUBLISHED_FMT[self.tensors.packed[names[0]][3]])

    def unused(self, prefixes):
        return sorted(n for n in self.tensors.names() - self.used if n.startswith(prefixes))


class _Tok:
    """tokenizer.json exactly as stored: its pre-tokenizer rules are used as written in the file. Rewriting
    them while loading would split free text (mixed-case words, contractions) into other tokens."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no tokenizer.json in {os.path.dirname(path)}")
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)
        self.t.no_padding()         # some checkpoints store a padding block; one text at a time needs none
        self.t.no_truncation()

    def encode(self, text: str) -> list:
        return self.t.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.t.decode([int(i) for i in ids], skip_special_tokens=True)

    def token_to_id(self, token: str):
        return self.t.token_to_id(token)


# ------------------------------------------------------------------ loader
def load(model, device, dtype, cache_dir=None, precision=None) -> Weights:
    """model: local folder or Hugging Face repo id (`cache_dir`: where a repo is downloaded to)."""
    device, dtype = torch.device(device), _dtype(dtype)
    from kitten_inference.asr import _published

    folder = _resolve(model, cache_dir, precision)
    packed = _published.variant(folder, precision, required=precision == "int4")
    cfg, dm = _config(folder)
    _check_frontend(folder, cfg["n_mels"])
    _check_template(folder)

    tok = _Tok(os.path.join(folder, "tokenizer.json"))
    eos = _stop_ids(tok, cfg)
    n_audio = tok.encode(_prompt_text()).count(cfg["audio_id"])
    if n_audio != 1:
        raise ValueError(f"{folder}: the prompt holds {n_audio} tokens with the audio placeholder id "
                         f"{cfg['audio_id']} (expected 1): tokenizer.json and config.json do not belong together")

    H, L, nh, nkv, hd, V = cfg["H"], cfg["L"], cfg["nh"], cfg["nkv"], cfg["hd"], cfg["vocab"]
    d, I, ffn, C = cfg["enc_d"], dm["I"], dm["ffn"], dm["C"]
    f = _PublishedFiles(folder, precision, device, dtype) if packed else _Files(_weight_files(folder), device, dtype)
    T, A = "thinker.model.", "thinker.audio_tower."
    with torch.no_grad():
        embed = f.linear(T + "embed_tokens.weight", (V, H))
        norm_w = f.get(T + "norm.weight", (H,))
        head = "thinker.lm_head.weight"
        head_w = f.linear(head, (V, H)) if f.has(head) else embed

        W = []
        for i in range(L):
            p = f"{T}layers.{i}."
            a, m = p + "self_attn.", p + "mlp."
            W.append((f.get(p + "input_layernorm.weight", (H,)),
                      f.fused_linear([a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"],
                            [(nh * hd, H), (nkv * hd, H), (nkv * hd, H)]),
                      f.get(a + "q_norm.weight", (hd,)), f.get(a + "k_norm.weight", (hd,)),
                      f.linear(a + "o_proj.weight", (H, nh * hd)),
                      f.get(p + "post_attention_layernorm.weight", (H,)),
                      f.fused_linear([m + "gate_proj.weight", m + "up_proj.weight"], [(I, H), (I, H)]),
                      f.linear(m + "down_proj.weight", (H, I))))

        conv = [(f.get(f"{A}conv2d{j}.weight", (C, c_in, 3, 3)), f.get(f"{A}conv2d{j}.bias", (C,)))
                for j, c_in in ((1, 1), (2, C), (3, C))]
        # conv_out sees [.., channels * mel rows after the three convolutions], channel-major
        conv_out_w = f.linear(A + "conv_out.weight", (d, C * _down3(cfg["n_mels"])))

        EW = []
        for i in range(cfg["enc_layers"]):
            p = f"{A}layers.{i}."
            a = p + "self_attn."
            EW.append((f.get(p + "self_attn_layer_norm.weight", (d,)), f.get(p + "self_attn_layer_norm.bias", (d,)),
                       f.fused_linear([a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"], [(d, d)] * 3),
                       f.cat([a + "q_proj.bias", a + "k_proj.bias", a + "v_proj.bias"], [(d,)] * 3),
                       f.linear(a + "out_proj.weight", (d, d)), f.get(a + "out_proj.bias", (d,)),
                       f.get(p + "final_layer_norm.weight", (d,)), f.get(p + "final_layer_norm.bias", (d,)),
                       f.linear(p + "fc1.weight", (ffn, d)), f.get(p + "fc1.bias", (ffn,)),
                       f.linear(p + "fc2.weight", (d, ffn)), f.get(p + "fc2.bias", (d,))))

        ln_post = (f.get(A + "ln_post.weight", (d,)), f.get(A + "ln_post.bias", (d,)))
        proj1 = (f.linear(A + "proj1.weight", (d, d)), f.get(A + "proj1.bias", (d,)))
        proj2 = (f.linear(A + "proj2.weight", (H, d)), f.get(A + "proj2.bias", (H,)))

        # A tensor of the model that nothing above asked for (more layers than the config says, a bias the
        # engine has no place for, ...) means the engine would compute something else than the checkpoint.
        left = f.unused((T, A, "thinker.lm_head."))
        if left:
            more = f" and {len(left) - 6} more" if len(left) > 6 else ""
            raise ValueError(f"{folder}: the weight file(s) hold tensors this engine does not use, so the model "
                             f"layout differs from the one implemented: {left[:6]}{more}")
        del f                                   # drops the file mappings

        rows = _down3(cfg["chunk_frames"])
        enc_pos = _sinusoid(dm["max_pos"], d)[:rows].to(device=device, dtype=dtype, copy=True)
        mel = torch.tensor(np.ascontiguousarray(_mel_filters(cfg["n_mels"]).T), dtype=torch.float32,
                           device=device).contiguous()

    return Weights(cfg=cfg, embed=embed, norm_w=norm_w, head_w=head_w, W=W, conv=conv, conv_out_w=conv_out_w,
                   enc_pos=enc_pos, EW=EW, ln_post=ln_post, proj1=proj1, proj2=proj2, mel_filters=mel,
                   tok=tok, eos=eos, prompt=_prompt_text, folder=folder, published_precision=precision if packed else None)

