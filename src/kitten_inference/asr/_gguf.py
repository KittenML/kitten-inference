"""Converts a Kitten ASR model folder into the two GGUF files the CPU engine reads.

Only numpy and the standard library are used, so the conversion runs on the user's machine on first use without
PyTorch. The files match what llama.cpp's own converter (convert_hf_to_gguf.py, classes Qwen3ASRTextModel and
Qwen3ASRMmprojModel) writes for the same architecture; tests compare the two.

  decoder  the Qwen3 text decoder, architecture "qwen3vl", every matrix in bf16 exactly as published. Lower
           precisions (q8_0, q4_k_m, ...) are made from it by the engine's quantiser (llama.cpp's own).
  encoder  the audio encoder + projector ("mmproj", projector "qwen3a") in f16, bf16 or q8_0.

Published INT4/INT8 matrices can be written as native packed GGUF blocks. The caller supplies a local folder.
"""
from __future__ import annotations

import json
import math
import os
import struct
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

CONVERTER_VERSION = 1

# ggml tensor types and GGUF value types used here
F32, F16, Q4_1, Q8_0, BF16, KASR_Q8 = 0, 1, 3, 8, 30, 44
_TYPE_NAME = {F32: "f32", F16: "f16", Q4_1: "q4_1", Q8_0: "q8_0", BF16: "bf16", KASR_Q8: "kasr_q8"}
U8, I8, U16, I16, U32, I32, F32V, BOOL, STR, ARR, U64, I64, F64 = range(13)
_ALIGN = 32
_QK8_0 = 32

# token types (llama.cpp)
_NORMAL, _UNUSED, _CONTROL, _USER_DEFINED = 1, 5, 3, 4


# ------------------------------------------------------------------ safetensors (numpy)
class SafeTensors:
    """Every tensor of a folder's *.safetensors files, memory-mapped: name -> (dtype, shape, raw array)."""

    _DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}

    def __init__(self, folder: str, precision=None):
        from . import _published

        self.items: Dict[str, Tuple[str, Tuple[int, ...], str, int, int]] = {}
        self.packed = {}
        self.parts = set()
        packed = _published.variant(folder, precision, required=precision in _published.PRECISIONS)
        dtypes = {**self._DT, "U8": np.uint8} if packed else self._DT
        files = [os.path.join(packed, _published.WEIGHTS)] if packed else sorted(
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {folder}")
        for path in files:
            with open(path, "rb") as fh:
                size = os.fstat(fh.fileno()).st_size
                length = fh.read(8)
                n = struct.unpack("<Q", length)[0] if len(length) == 8 else 0
                if not 2 <= n <= min(size - 8, 100_000_000):
                    raise ValueError(f"{path}: invalid safetensors header length {n}")
                header = json.loads(fh.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                code, shape = meta["dtype"], tuple(meta["shape"])
                if code not in dtypes:
                    raise ValueError(f"{path}: tensor {name} has unsupported dtype {code}")
                a, b = meta["data_offsets"]
                if any(type(d) is not int or d < 0 for d in shape):
                    raise ValueError(f"{path}: tensor {name} has an invalid shape {shape}")
                if not 0 <= a <= b <= size - base or b - a != math.prod(shape) * np.dtype(dtypes[code]).itemsize:
                    raise ValueError(f"{path}: tensor {name} has invalid data offsets")
                self.items[name] = (code, shape, path, base + a, b - a)
        if packed:
            self._read_packed(os.path.join(packed, _published.META), int(precision[3:]))

    def _read_packed(self, path, bits):
        meta = _json(path)
        tensors = meta.get("tensors") if isinstance(meta, dict) else None
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f"{path}: expected a nonempty tensors object")
        for name, entry in tensors.items():
            try:
                shape = tuple(entry["shape"])
                cols, group = entry["cols"], entry["group"]
                if (type(cols) is not int or type(group) is not int or type(entry["bits"]) is not int or
                        entry["bits"] != bits or len(shape) < 2 or any(type(d) is not int or d <= 0 for d in shape)
                        or cols != math.prod(shape[1:]) or group <= 0 or cols % group):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{path}: invalid {bits}-bit metadata for tensor {name}") from None
            if name in self.items:
                raise ValueError(f"{path}: tensor {name} is both packed and dense")
            for suffix, spec in (
                ("q", ("U8", (shape[0], (cols * bits + 7) // 8))),
                ("scale", ("F16", (shape[0], cols // group))),
                ("zero", ("F16", (shape[0], cols // group))),
            ):
                component = f"{name}.{suffix}"
                if self.items.get(component, ())[:2] != spec:
                    raise ValueError(f"{path}: tensor {component} must have dtype {spec[0]} and shape {spec[1]}")
                self.parts.add(component)
            self.packed[name] = (shape, cols, group, bits)
        if any(info[0] == "U8" and name not in self.parts for name, info in self.items.items()):
            raise ValueError(f"{path}: weight file has packed tensors without metadata")

    def has(self, name: str) -> bool:
        return name in self.packed or name in self.items

    def shape(self, name: str) -> tuple:
        return self.packed[name][0] if name in self.packed else self.items[name][1]

    def names(self):
        return self.packed.keys() | (self.items.keys() - self.parts)

    def raw(self, name: str) -> np.ndarray:
        dt, shape, path, off, _ = self.items[name]
        dtype = np.uint8 if dt == "U8" else self._DT[dt]
        return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape if shape else (1,))

    def ggml_packed_type(self, name):
        meta = self.packed.get(name)
        if meta is None or len(meta[0]) != 2 or meta[2] % _QK8_0:
            return None
        return Q4_1 if meta[3] == 4 else KASR_Q8

    def ggml_packed(self, name, kind):
        """Reblock original UINT codes and affine parameters for ggml's 32-element blocks."""
        shape, cols, group, _ = self.packed[name]
        q, scale, zero = (self.raw(f"{name}.{part}") for part in ("q", "scale", "zero"))
        per_group = group // _QK8_0
        width = 16 if kind == Q4_1 else 32
        blocks = np.empty((shape[0], cols // group, per_group, 4 + width), dtype=np.uint8)
        blocks[..., :2] = np.ascontiguousarray(scale, dtype="<f2").view(np.uint8).reshape(*scale.shape, 1, 2)
        if kind == Q4_1:
            # GGML Q4_1 uses an additive FP16 minimum and low nibbles for the first half of a block.
            offset = (-zero.astype(np.float32) * scale).astype("<f2")
            values = q.reshape(*blocks.shape[:-1], 16)
            codes = np.empty((*values.shape[:-1], 32), dtype=np.uint8)
            codes[..., ::2], codes[..., 1::2] = values >> 4, values & 15
            blocks[..., 4:] = codes[..., :16] | (codes[..., 16:] << 4)
        else:
            offset = np.ascontiguousarray(zero, dtype="<f2")
            blocks[..., 4:] = q.reshape(*blocks.shape[:-1], 32) ^ 128
        blocks[..., 2:4] = offset.view(np.uint8).reshape(*offset.shape, 1, 2)
        return blocks.tobytes()

    def _unpack(self, name: str) -> np.ndarray:
        shape, cols, group, bits = self.packed[name]
        q, scale, zero = (self.raw(f"{name}.{suffix}") for suffix in ("q", "scale", "zero"))
        out = np.empty((shape[0], cols), dtype=np.uint16)
        rows = max(1, (1 << 20) // cols)
        for start in range(0, shape[0], rows):
            end = min(start + rows, shape[0])
            data = q[start:end]
            if bits == 4:
                unpacked = np.empty((end - start, cols), dtype=np.uint8)
                # The Hub stores the earlier column in the high nibble.
                unpacked[:, ::2] = data >> 4
                unpacked[:, 1::2] = data[:, :cols // 2] & 15
                data = unpacked
            values = data.reshape(end - start, -1, group).astype(np.float32)
            values -= zero[start:end, :, None]
            values *= scale[start:end, :, None]
            out[start:end] = _f32_to_bf16(values.reshape(end - start, cols))
        return out.reshape(shape)

    def f32(self, name: str) -> np.ndarray:
        if name in self.packed or self.items[name][0] == "BF16":
            return (self.bf16_bits(name).astype(np.uint32) << 16).view(np.float32)
        return self.raw(name).astype(np.float32)

    def bf16_bits(self, name: str) -> np.ndarray:
        """The tensor as bf16 bit patterns (uint16)."""
        if name in self.packed:
            return self._unpack(name)
        x = np.asarray(self.raw(name))
        if self.items[name][0] == "BF16":
            return x
        return _f32_to_bf16(x.astype(np.float32))


def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    # round to nearest even, NaN kept quiet, as gguf-py and ggml do
    u = x.astype(np.float32).view(np.uint32)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)
    return np.where(nan, ((u >> 16) | 64).astype(np.uint16), r)


def quantize_q8_0(x: np.ndarray) -> bytes:
    """ggml q8_0: blocks of 32 values, one fp16 scale (max |x| / 127) and 32 int8, as gguf-py writes them."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    blocks = x.reshape(-1, _QK8_0)
    d = np.abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1 / d)
    q = blocks * inv
    q = (np.sign(q) * np.floor(np.abs(q) + 0.5)).astype(np.int8)     # round half away from zero
    out = np.empty((blocks.shape[0], 2 + _QK8_0), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


# ------------------------------------------------------------------ GGUF writer
class _Writer:
    def __init__(self):
        self.kv: List[Tuple[str, int, object, Optional[int]]] = []
        self.tensors: List[Tuple[str, int, List[int], Callable[[], bytes], int]] = []

    def add(self, key: str, vtype: int, value, sub: Optional[int] = None):
        self.kv.append((key, vtype, value, sub))

    def tensor(self, name: str, ttype: int, shape: Tuple[int, ...], data: Callable[[], bytes]):
        """shape in numpy order (outermost first); GGUF stores it innermost first."""
        ne = [int(d) for d in reversed(shape)]
        n = int(np.prod(ne)) if ne else 1
        if ttype == F32:
            nbytes = 4 * n
        elif ttype in (F16, BF16):
            nbytes = 2 * n
        elif ttype in (Q4_1, Q8_0, KASR_Q8):
            nbytes = n // _QK8_0 * {Q4_1: 20, Q8_0: 34, KASR_Q8: 36}[ttype]
        else:
            raise ValueError(ttype)
        self.tensors.append((name, ttype, ne, data, nbytes))

    @staticmethod
    def _str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, vtype: int, value, sub) -> bytes:
        if vtype == STR:
            return self._str(value)
        if vtype == ARR:
            out = [struct.pack("<IQ", sub, len(value))]
            if sub == STR:
                out.extend(self._str(v) for v in value)
            else:
                fmt = {U8: "B", I8: "b", U32: "I", I32: "i", F32V: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}[sub]
                out.append(struct.pack(f"<{len(value)}{fmt}", *value))
            return b"".join(out)
        fmt = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32V: "f", BOOL: "?", U64: "Q",
               I64: "q", F64: "d"}[vtype]
        return struct.pack("<" + fmt, value)

    def write(self, path: str):
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
            for key, vtype, value, sub in self.kv:
                f.write(self._str(key) + struct.pack("<I", vtype) + self._value(vtype, value, sub))
            offset = 0
            for name, ttype, ne, _, nbytes in self.tensors:
                f.write(self._str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
                        + struct.pack("<IQ", ttype, offset))
                offset += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
            f.write(b"\0" * (-f.tell() % _ALIGN))
            for name, ttype, ne, data, nbytes in self.tensors:
                b = data()
                if len(b) != nbytes:
                    raise RuntimeError(f"{name}: {len(b)} bytes, expected {nbytes}")
                f.write(b)
                f.write(b"\0" * (-nbytes % _ALIGN))
        os.replace(tmp, path)


# ------------------------------------------------------------------ model folder
def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _configs(folder: str) -> Tuple[dict, dict, dict]:
    cfg = _json(os.path.join(folder, "config.json"))
    th = cfg.get("thinker_config", cfg)
    tc, ac = th.get("text_config"), th.get("audio_config")
    if not isinstance(tc, dict) or not isinstance(ac, dict):
        raise ValueError(f"{folder}/config.json has no thinker_config.text_config / audio_config")
    return cfg, tc, ac


def _vocab(folder: str, vocab_size: int):
    """tokens, token types and merges, as llama.cpp's converter makes them from the Hugging Face tokenizer."""
    tj = _json(os.path.join(folder, "tokenizer.json"))
    model = tj["model"]
    if model.get("type") != "BPE":
        raise ValueError("expected a byte-level BPE tokenizer")
    vocab = dict(model["vocab"])
    added = {t["content"]: t for t in tj.get("added_tokens", [])}
    for content, t in added.items():
        vocab[content] = t["id"]
    reverse = {i: s for s, i in vocab.items()}
    if max(reverse) >= vocab_size:
        raise ValueError(f"token id {max(reverse)} is outside the vocabulary ({vocab_size})")
    tokens, types = [], []
    for i in range(vocab_size):
        tok = reverse.get(i)
        if tok is None:
            tokens.append(f"[PAD{i}]")
            types.append(_UNUSED)
        elif tok in added:
            special = added[tok].get("special", False) or (tok.startswith("<|") and tok.endswith("|>"))
            if not special:
                tok = tok.replace("▁", " ")
            tokens.append(tok)
            types.append(_CONTROL if special else _USER_DEFINED)
        else:
            tokens.append(tok)
            types.append(_NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", [])]
    return tokens, types, merges, vocab


def _token_id(vocab: dict, value) -> Optional[int]:
    if isinstance(value, dict):
        value = value.get("content")
    return vocab.get(value) if isinstance(value, str) else None


_CHAT_TEMPLATE = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
                  "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
                  "{% endif %}")

_TEXT_NAMES = {
    "input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm", "post_attention_layernorm": "ffn_norm", "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def convert_decoder(folder: str, out: str, name: str = "kitten-asr", *, published=None, packed=False) -> str:
    """The text decoder as a bf16 GGUF (1-D tensors in f32, as llama.cpp's converter writes them)."""
    st = SafeTensors(folder, published)
    _, tc, _ = _configs(folder)
    tcfg = _json(os.path.join(folder, "tokenizer_config.json"))
    n_layer = int(tc["num_hidden_layers"])
    head_dim = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"])
    vocab_size = int(tc.get("vocab_size"))
    rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    sections = list(rope.get("mrope_section") or [])
    if not sections:
        raise ValueError(f"{folder}: text_config.rope_scaling.mrope_section is missing")
    sections = (sections + [0, 0, 0, 0])[:4]
    arch = "qwen3vl"

    w = _Writer()
    w.add("general.architecture", STR, arch)
    w.add("general.type", STR, "model")
    w.add("general.name", STR, name)
    w.add(f"{arch}.block_count", U32, n_layer)
    w.add(f"{arch}.context_length", U32, int(tc["max_position_embeddings"]))
    w.add(f"{arch}.embedding_length", U32, int(tc["hidden_size"]))
    w.add(f"{arch}.feed_forward_length", U32, int(tc["intermediate_size"]))
    w.add(f"{arch}.attention.head_count", U32, int(tc["num_attention_heads"]))
    w.add(f"{arch}.attention.head_count_kv", U32, int(tc["num_key_value_heads"]))
    w.add(f"{arch}.rope.dimension_sections", ARR, sections, I32)
    w.add(f"{arch}.rope.freq_base", F32V, float(rope.get("rope_theta", tc.get("rope_theta", 10000.0))))
    w.add(f"{arch}.attention.layer_norm_rms_epsilon", F32V, float(tc["rms_norm_eps"]))
    w.add(f"{arch}.attention.key_length", U32, head_dim)
    w.add(f"{arch}.attention.value_length", U32, head_dim)
    w.add("general.file_type", U32, 32)                       # LLAMA_FTYPE_MOSTLY_BF16
    w.add(f"{arch}.n_deepstack_layers", U32, 0)
    w.add("general.quantization_version", U32, 2)

    tokens, types, merges, vocab = _vocab(folder, vocab_size)
    w.add("tokenizer.ggml.model", STR, "gpt2")
    w.add("tokenizer.ggml.pre", STR, "qwen2")
    w.add("tokenizer.ggml.tokens", ARR, tokens, STR)
    w.add("tokenizer.ggml.token_type", ARR, types, I32)
    w.add("tokenizer.ggml.merges", ARR, merges, STR)
    im_end = vocab.get("<|im_end|>")
    if im_end is None:
        raise ValueError(f"{folder}: the tokenizer has no <|im_end|> token")
    w.add("tokenizer.ggml.eos_token_id", U32, im_end)
    pad = _token_id(vocab, tcfg.get("pad_token"))
    if pad is not None:
        w.add("tokenizer.ggml.padding_token_id", U32, pad)
    w.add("tokenizer.ggml.bos_token_id", U32, im_end)
    if "add_bos_token" in tcfg:
        w.add("tokenizer.ggml.add_bos_token", BOOL, bool(tcfg["add_bos_token"]))
    if "add_eos_token" in tcfg:
        w.add("tokenizer.ggml.add_eos_token", BOOL, bool(tcfg["add_eos_token"]))
    if "clean_up_tokenization_spaces" in tcfg:
        w.add("tokenizer.ggml.clean_spaces", BOOL, bool(tcfg["clean_up_tokenization_spaces"]))
    norm = (_json(os.path.join(folder, "tokenizer.json")).get("normalizer") or {}).get("type")
    if norm == "NFC":
        w.add("tokenizer.ggml.normalizer.nfc", BOOL, True)
    w.add("tokenizer.chat_template", STR, _CHAT_TEMPLATE)

    T = "thinker.model."
    used = set()

    def bf16(src):
        used.add(src)
        return lambda: np.ascontiguousarray(st.bf16_bits(src)).tobytes()

    def f32(src):
        used.add(src)
        return lambda: np.ascontiguousarray(st.f32(src)).tobytes()

    def put(dst, src):
        if not st.has(src):
            raise ValueError(f"{folder}: tensor {src} is missing")
        shape = st.shape(src)
        kind = st.ggml_packed_type(src) if packed else None
        if kind is not None:
            used.add(src)
            w.tensor(dst, kind, shape, lambda: st.ggml_packed(src, kind))
        elif len(shape) == 1:
            w.tensor(dst, F32, shape, f32(src))
        else:
            w.tensor(dst, BF16, shape, bf16(src))

    put("token_embd.weight", T + "embed_tokens.weight")
    for i in range(n_layer):
        for hf, gg in _TEXT_NAMES.items():
            put(f"blk.{i}.{gg}.weight", f"{T}layers.{i}.{hf}.weight")
    put("output_norm.weight", T + "norm.weight")
    for head in ("thinker.lm_head.weight", "lm_head.weight"):
        if st.has(head):
            put("output.weight", head)
            break
    _sort_like_llama(w)
    w.write(out)
    return out


def _sort_like_llama(w: _Writer):
    # llama.cpp's converter writes token_embd, the blocks in order, then output_norm and output; keep that order
    def key(t):
        name = t[0]
        if name.startswith("blk."):
            return (1, int(name.split(".")[1]))
        return (0 if name == "token_embd.weight" else 2, 0)
    w.tensors.sort(key=key)


def _sinusoids(length: int, channels: int) -> np.ndarray:
    # SinusoidsPositionEmbedding, in float32 as llama.cpp's converter computes it
    inc = np.float32(np.log(10000) / (channels // 2 - 1))
    inv = np.exp(-inc * np.arange(channels // 2, dtype=np.float32)).astype(np.float32)
    t = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)


def convert_encoder(folder: str, out: str, kind: str = "f16", window_frames: Optional[int] = None,
                    name: str = "kitten-asr", *, published=None, packed=False) -> str:
    """The audio encoder and projector ("mmproj") in f16, bf16 or q8_0 (convolutions f16, 1-D tensors f32, as
    llama.cpp's converter does; bf16 keeps the published weights exactly). window_frames: mel frames per encoder pass written into the file
    (None: leave it out, the engine then uses 8 s blocks)."""
    if kind not in ("f16", "bf16", "q8_0"):
        raise ValueError(f"encoder precision must be f16, bf16 or q8_0, not {kind!r}")
    st = SafeTensors(folder, published)
    _, tc, ac = _configs(folder)
    d = int(ac["d_model"])
    n_layer = int(ac["encoder_layers"])
    w = _Writer()
    w.add("general.architecture", STR, "clip")
    w.add("general.type", STR, "mmproj")
    w.add("general.name", STR, name)
    w.add("general.file_type", U32, {"f32": 0, "f16": 1, "q8_0": 7, "bf16": 32}[kind])
    w.add("clip.has_audio_encoder", BOOL, True)
    w.add("clip.audio.projection_dim", U32, int(tc["hidden_size"]))
    w.add("clip.audio.embedding_length", U32, d)
    w.add("clip.audio.feed_forward_length", U32, int(ac["encoder_ffn_dim"]))
    w.add("clip.audio.block_count", U32, n_layer)
    w.add("clip.audio.attention.head_count", U32, int(ac["encoder_attention_heads"]))
    w.add("clip.audio.num_mel_bins", U32, int(ac["num_mel_bins"]))
    w.add("clip.audio.attention.layer_norm_epsilon", F32V, float(ac.get("layer_norm_eps", 1e-5)))
    w.add("clip.audio.projector_type", STR, "qwen3a")
    if window_frames is not None:
        w.add("clip.audio.kitten.window_frames", U32, int(window_frames))
    w.add("general.quantization_version", U32, 2)

    A = "thinker.audio_tower."
    mat_type = {"f16": F16, "f32": F32, "q8_0": Q8_0, "bf16": BF16}[kind]

    def put(dst, src, force=None, shape=None):
        if not st.has(src):
            raise ValueError(f"{folder}: tensor {src} is missing")
        sshape = st.shape(src)
        shape = shape or sshape
        t = force if force is not None else (F32 if len(sshape) == 1 else mat_type)
        native = st.ggml_packed_type(src) if packed else None
        if native is not None:
            w.tensor(dst, native, shape, lambda: st.ggml_packed(src, native))
            return
        if t == Q8_0 and shape[-1] % _QK8_0:
            t = F16
        if t == F32:
            w.tensor(dst, F32, shape, lambda: np.ascontiguousarray(st.f32(src)).reshape(shape).tobytes())
        elif t == F16:
            w.tensor(dst, F16, shape, lambda: st.f32(src).astype(np.float16).reshape(shape).tobytes())
        elif t == BF16:
            w.tensor(dst, BF16, shape, lambda: np.ascontiguousarray(st.bf16_bits(src)).reshape(shape).tobytes())
        else:
            w.tensor(dst, Q8_0, shape, lambda: quantize_q8_0(st.f32(src)))

    pos = _sinusoids(1500, d)                  # generated, not in the checkpoint; kept in f32 as llama.cpp does
    w.tensor("a.position_embd.weight", F32, pos.shape, lambda: pos.tobytes())
    for i in (1, 2, 3):
        put(f"a.conv2d.{i}.weight", f"{A}conv2d{i}.weight", force=F16)
        c = st.shape(f"{A}conv2d{i}.bias")[0]
        put(f"a.conv2d.{i}.bias", f"{A}conv2d{i}.bias", shape=(c, 1, 1))
    put("a.conv_out.weight", f"{A}conv_out.weight", force=F16 if mat_type != F32 else None)
    for i in range(n_layer):
        L = f"{A}layers.{i}."
        for hf, gg in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                       ("self_attn.out_proj", "attn_out"), ("self_attn_layer_norm", "ln1"),
                       ("final_layer_norm", "ln2"), ("fc1", "ffn_up"), ("fc2", "ffn_down")):
            for part in ("weight", "bias"):
                if st.has(f"{L}{hf}.{part}"):
                    put(f"a.blk.{i}.{gg}.{part}", f"{L}{hf}.{part}")
    for part in ("weight", "bias"):
        put(f"a.post_ln.{part}", f"{A}ln_post.{part}")
        put(f"mm.a.mlp.1.{part}", f"{A}proj1.{part}")
        put(f"mm.a.mlp.2.{part}", f"{A}proj2.{part}")
    w.write(out)
    return out


def describe(path: str) -> dict:
    """Header of a GGUF file: {"kv": {key: value (arrays: length + first items)}, "tensors": [(name, type, ne)]}."""
    with open(path, "rb") as f:
        data = f.read(64 << 20)
    o = 4
    _, nt, nkv = struct.unpack_from("<IQQ", data, o)
    o += 20

    def s():
        nonlocal o
        n = struct.unpack_from("<Q", data, o)[0]
        o += 8
        v = data[o:o + n].decode("utf-8", "replace")
        o += n
        return v

    sizes = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32V: "f", BOOL: "?", U64: "Q", I64: "q",
             F64: "d"}

    def val(t):
        nonlocal o
        if t == STR:
            return s()
        if t == ARR:
            sub, n = struct.unpack_from("<IQ", data, o)
            o += 12
            if sub == STR:
                items = [s() for _ in range(n)]
            else:
                fmt = sizes[sub]
                items = list(struct.unpack_from(f"<{n}{fmt}", data, o))
                o += n * struct.calcsize(fmt)
            return {"type": sub, "n": n, "items": items}
        fmt = sizes[t]
        v = struct.unpack_from("<" + fmt, data, o)[0]
        o += struct.calcsize(fmt)
        return v

    kv = {}
    for _ in range(nkv):
        k = s()
        t = struct.unpack_from("<I", data, o)[0]
        o += 4
        kv[k] = (t, val(t))
    tensors = []
    for _ in range(nt):
        name = s()
        nd = struct.unpack_from("<I", data, o)[0]
        o += 4
        ne = list(struct.unpack_from(f"<{nd}Q", data, o))
        o += 8 * nd
        t, off = struct.unpack_from("<IQ", data, o)
        o += 12
        tensors.append((name, t, ne, off))
    return {"kv": kv, "tensors": tensors, "data_start": o + (-o % _ALIGN)}
