"""Converts the official Kitten TTS 2 mini / nano / pico files into the native engine's files.

Input: a model repository as published on Hugging Face (config.json, the float32 ``*.safetensors`` of each model,
``voices_<language>.npz``). Output, per model: an architecture file (``<model>_fp32_arch.json``, and
``<model>_int8_arch.json``), a weights file in the engine's W1X1 format (``<model>_fp32.bin``, ``<model>_int8.bin``)
and, per language, one raw float32 file per voice (``voices/<language>/<key>.bin``, [510, 256] rows) for the C
library. Python and numpy only: no torch, no onnx.

What the conversion does:
  - weight norm is folded: w = g * v / ||v|| (norm over every axis but the first, as torch's weight_norm(dim=0));
  - PyTorch LSTM gates (i, f, g, o) are already in the engine's order; nn.Linear's [out, in] is the engine's layout;
  - ALBERT's shared layer is stored once (every layer node uses the same names);
  - the STFT filters are computed (periodic Hann, as torch.hann_window), not read;
  - the graph follows the config: hidden size, layer count, ALBERT width and heads, decoder type
    ("istftnet": encode/decode 1024 wide, 3 resblocks per upsample stage, softplus magnitude;
     "istftnet_mini_2": encode/decode as wide as the first upsample stage, exp magnitude), upsample rates and kernels.

int8 (``<model>_int8.bin``): per-output-channel symmetric weights, the same layer choice as the TTS 0.8 int8 models:
ALBERT and its projections, every 1-D convolution except the last one (its output feeds exp/softplus and sin),
and the style projections of the adaptive norms (stored int8, computed in float32).
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
import uuid

import numpy as np

FORMAT_VERSION = 1  # bump when the files this module writes change; kept in the converted folder's manifest

# ---- safetensors (numpy only) -----------------------------------------------------------------------------------
_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32,
              "I8": np.int8, "U8": np.uint8}


def load_safetensors(path):
    """{name: float32 array} from a .safetensors file (F32/F16/BF16)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dt = info["dtype"]
            if dt == "BF16":
                a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt in _ST_DTYPES:
                a = np.frombuffer(raw, dtype=_ST_DTYPES[dt])
            else:
                raise ValueError(f"{path}: tensor {name} has unsupported dtype {dt}")
            out[name] = a.reshape(info["shape"]).astype(np.float32)
    return out


# ---- W1X1 writer (the engine's weights format) ------------------------------------------------------------------
def _pack(arr):
    arr = np.ascontiguousarray(arr)
    header = struct.pack("BBH" + "i" * arr.ndim, 1, 0, arr.ndim, *arr.shape)
    return header + arr.tobytes()


class W1X1:
    def __init__(self):
        self.blobs = {}

    def add(self, name, arr, dtype=np.float32):
        if len(name) > 63:
            raise ValueError(f"tensor name too long for W1X1: {name}")
        if name not in self.blobs:
            self.blobs[name] = _pack(np.asarray(arr, dtype=dtype))

    def write(self, path):
        n = len(self.blobs)
        data_base = 12 + n * 80
        idx, off = io.BytesIO(), data_base
        for name, blob in self.blobs.items():
            idx.write(name.encode("ascii").ljust(64, b"\0"))
            idx.write(struct.pack("<QII", off, len(blob), 0))
            off += len(blob)
        tmp = _tmp_name(path)
        with open(tmp, "wb") as f:
            f.write(b"W1X1")
            f.write(struct.pack("<II", 1, n))
            f.write(idx.getvalue())
            for blob in self.blobs.values():
                f.write(blob)
        os.replace(tmp, path)


# ---- architecture ------------------------------------------------------------------------------------------------
def _nd(name, op, inputs, outputs, **kw):
    n = {"name": name, "op": op, "inputs": inputs if isinstance(inputs, list) else [inputs],
         "outputs": outputs if isinstance(outputs, list) else [outputs]}
    n.update(kw)
    return n


def dims(arch):
    """The graph's sizes from one model's config.json "architecture" entry."""
    dec, alb = arch["decoder"], arch["albert"]
    hid = int(arch["hidden_dim"])
    ups_init = int(dec["upsample_initial_channel"])
    if dec["type"] == "istftnet":                 # mini: kitten-tts istftnet.Decoder (fixed 1024-wide decoder)
        enc_ch, dec_hidden, mag = 1024, 1024, "softplus"
    elif dec["type"] == "istftnet_mini_2":        # nano, pico
        enc_ch = dec_hidden = ups_init
        mag = "exp"
    else:
        raise ValueError(f"unsupported decoder type {dec['type']!r}")
    if alb.get("num_hidden_groups", 1) != 1 or alb.get("inner_group_num", 1) != 1:
        raise ValueError("only ALBERT with one shared layer group is supported")
    if alb.get("hidden_act") not in ("gelu_new", "gelu_pytorch_tanh"):
        raise ValueError(f"unsupported ALBERT activation {alb.get('hidden_act')!r} (the engine has tanh GELU)")
    if len(dec["resblock_kernel_sizes"]) != len(dec["resblock_dilation_sizes"]):
        raise ValueError("resblock kernel and dilation lists differ in length")
    return dict(
        hidden=hid, n_layer=int(arch["n_layer"]), n_token=int(arch["n_token"]), style=int(arch["style_dim"]),
        max_dur=int(arch["max_dur"]), te_kernel=int(arch["text_encoder_kernel_size"]),
        bert_emb=int(alb["embedding_size"]), bert_hidden=int(alb["hidden_size"]), bert_ff=int(alb["intermediate_size"]),
        bert_heads=int(alb["num_attention_heads"]), bert_layers=int(alb["num_hidden_layers"]),
        bert_eps=float(alb.get("layer_norm_eps", 1e-12)),
        enc_ch=enc_ch, dec_hidden=dec_hidden, ups_init=ups_init, magnitude=mag,
        ups_rates=[int(x) for x in dec["upsample_rates"]], ups_kernels=[int(x) for x in dec["upsample_kernel_sizes"]],
        rb_kernels=[int(x) for x in dec["resblock_kernel_sizes"]],
        rb_dilations=[[int(y) for y in x] for x in dec["resblock_dilation_sizes"]],
        n_fft=int(dec["gen_istft_n_fft"]), hop=int(dec["gen_istft_hop_size"]),
    )


def _adain(name, x, s, out, C, Cs):
    return _nd(name, "ada_in1d_fp32", [x, s], [out], C=C, C_style=Cs)


def _resblk1d(p, din, dout, s, up, x, Cs):
    """kitten-tts AdainResBlk1d (prosody F0/N blocks, decoder encode/decode)."""
    n = [_adain(f"{p}_n1", x, s, f"{p}_n1t", din, Cs),
         _nd(f"{p}_r1", "leaky_relu_fp32", f"{p}_n1t", f"{p}_r1t", alpha=0.2)]
    c1_in = f"{p}_r1t"
    if up:
        n.append(_nd(f"{p}_pool", "conv_transpose1d_fp32", c1_in, f"{p}_pooled", C_in=din, C_out=din,
                     kernel_size=3, stride=2, padding=1, output_padding=1, groups=din))
        c1_in = f"{p}_pooled"
    n += [_nd(f"{p}_c1", "conv1d_fp32", c1_in, f"{p}_c1t", C_in=din, C_out=dout, kernel_size=3, stride=1, padding=1),
          _adain(f"{p}_n2", f"{p}_c1t", s, f"{p}_n2t", dout, Cs),
          _nd(f"{p}_r2", "leaky_relu_fp32", f"{p}_n2t", f"{p}_r2t", alpha=0.2),
          _nd(f"{p}_c2", "conv1d_fp32", f"{p}_r2t", f"{p}_c2t", C_in=dout, C_out=dout, kernel_size=3, stride=1,
              padding=1)]
    sc = x
    if up:
        n.append(_nd(f"{p}_sc_up", "upsample_nearest1d_fp32", x, f"{p}_sc_up_t", scale_factor=2))
        sc = f"{p}_sc_up_t"
    if din != dout:
        n.append(_nd(f"{p}_sc", "conv1d_fp32", sc, f"{p}_sc_t", C_in=din, C_out=dout, kernel_size=1, stride=1,
                     padding=0))
        sc = f"{p}_sc_t"
    n += [_nd(f"{p}_add", "add_fp32", [f"{p}_c2t", sc], f"{p}_addt"),
          _nd(f"{p}_sc2", "scale_fp32", f"{p}_addt", f"{p}_out", scalar=1.0 / math.sqrt(2))]
    return n, f"{p}_out"


def _resblock1(p, ch, k, dilations, s, x, Cs):
    """iSTFTNet AdaINResBlock1 (generator resblocks and noise blocks): AdaIN, Snake, dilated conv, twice, + skip."""
    n, cur = [], x
    for di, dil in enumerate(dilations):
        ip = f"{p}_d{di}"
        n += [_adain(f"{ip}_n1", cur, s, f"{ip}_n1t", ch, Cs),
              _nd(f"{ip}_sk1", "snake1d_fp32", f"{ip}_n1t", f"{ip}_sk1t"),
              _nd(f"{ip}_c1", "conv1d_fp32", f"{ip}_sk1t", f"{ip}_c1t", C_in=ch, C_out=ch, kernel_size=k, stride=1,
                  padding=(k * dil - dil) // 2, dilation=dil),
              _adain(f"{ip}_n2", f"{ip}_c1t", s, f"{ip}_n2t", ch, Cs),
              _nd(f"{ip}_sk2", "snake1d_fp32", f"{ip}_n2t", f"{ip}_sk2t"),
              _nd(f"{ip}_c2", "conv1d_fp32", f"{ip}_sk2t", f"{ip}_c2t", C_in=ch, C_out=ch, kernel_size=k, stride=1,
                  padding=(k - 1) // 2, dilation=1),
              _nd(f"{ip}_add", "add_fp32", [f"{ip}_c2t", cur], f"{ip}_out")]
        cur = f"{ip}_out"
    return n, cur


def build_arch(arch, model_name):
    """The engine graph (fp32) of one model. Inputs: phoneme_ids [T], style_dec [128], style_pred [128].
    Outputs: t_audio (waveform before any trimming) and t_dur (each token's summed duration bins; the frame count is
    max(1, round(t_dur / speed)))."""
    d = dims(arch)
    H, L, S = d["hidden"], d["n_layer"], d["style"]
    BH = d["bert_hidden"]
    nodes = [_nd("bert_embed", "bert_embeddings_fp32", "phoneme_ids", "t_emb", emb_size=d["bert_emb"]),
             _nd("bert_enc_proj", "seqgemm_fp32", "t_emb", "t_emb_proj", C_in=d["bert_emb"], C_out=BH)]
    cur = "t_emb_proj"
    for li in range(d["bert_layers"]):   # ALBERT: every layer is the same shared layer (same weight names)
        nodes += [_nd("bert_attn", "attention_fp32", cur, f"bl{li}_attn", embed_dim=BH, num_heads=d["bert_heads"]),
                  _nd(f"bl{li}_add1n", "add_fp32", [cur, f"bl{li}_attn"], f"bl{li}_add1"),
                  _nd("bert_attn_ln", "layernorm_fp32", f"bl{li}_add1", f"bl{li}_aln", eps=d["bert_eps"]),
                  _nd("bert_ff1", "seqgemm_fp32", f"bl{li}_aln", f"bl{li}_ff1", C_in=BH, C_out=d["bert_ff"], gelu=True),
                  _nd("bert_ff2", "seqgemm_fp32", f"bl{li}_ff1", f"bl{li}_ff2", C_in=d["bert_ff"], C_out=BH),
                  _nd(f"bl{li}_add2n", "add_fp32", [f"bl{li}_aln", f"bl{li}_ff2"], f"bl{li}_add2"),
                  _nd("bert_full_ln", "layernorm_fp32", f"bl{li}_add2", f"bl{li}_fln", eps=d["bert_eps"])]
        cur = f"bl{li}_fln"
    nodes.append(_nd("bert_enc", "seqgemm_fp32", cur, "t_bert_enc", C_in=BH, C_out=H))

    # text encoder
    nodes.append(_nd("te_embed", "embedding_fp32", "phoneme_ids", "t_te_emb", num_embeddings=d["n_token"],
                     embedding_dim=H))
    te = "t_te_emb"
    for ci in range(L):
        nodes += [_nd(f"te_conv_{ci}", "conv1d_fp32", te, f"t_te_c{ci}", C_in=H, C_out=H, kernel_size=d["te_kernel"],
                      stride=1, padding=(d["te_kernel"] - 1) // 2),
                  _nd(f"te_ln_{ci}", "layernorm_fp32", f"t_te_c{ci}", f"t_te_l{ci}", eps=1e-5),
                  _nd(f"te_relu_{ci}", "leaky_relu_fp32", f"t_te_l{ci}", f"t_te_a{ci}", alpha=0.2)]
        te = f"t_te_a{ci}"
    nodes.append(_nd("te_lstm", "lstm_fp32", te, "t_te", hidden_size=H // 2, bidirectional=True))

    # duration encoder
    nodes.append(_nd("de_cat_init", "concat1d_fp32", ["t_bert_enc", "style_pred"], "t_de_c0", C2=S))
    de = "t_de_c0"
    for li in range(L):
        nodes += [_nd(f"de_lstm_{li}", "lstm_fp32", de, f"t_de_lstm{li}", hidden_size=H // 2, bidirectional=True),
                  _nd(f"de_adaln_{li}", "ada_layer_norm_fp32", [f"t_de_lstm{li}", "style_pred"], f"t_de_aln{li}",
                      C=H, C_style=S),
                  _nd(f"de_cat_{li}", "concat1d_fp32", [f"t_de_aln{li}", "style_pred"], f"t_de_cat{li}", C2=S)]
        de = f"t_de_cat{li}"

    # durations
    nodes += [_nd("pred_lstm", "lstm_fp32", de, "t_pred_lstm", hidden_size=H // 2, bidirectional=True),
              _nd("dur_proj", "seqgemm_fp32", "t_pred_lstm", "t_dur_raw", C_in=H, C_out=d["max_dur"]),
              _nd("dur_sig", "sigmoid_fp32", "t_dur_raw", "t_dur_sig"),
              _nd("dur_sum", "sum_channels_fp32", "t_dur_sig", "t_dur"),
              _nd("len_reg_en", "length_regulate_fp32", [de, "t_dur"], "en"),
              _nd("len_reg_asr", "length_regulate_fp32", ["t_te", "t_dur"], "asr"),
              _nd("shared_lstm", "lstm_fp32", "en", "t_shared", hidden_size=H // 2, bidirectional=True)]

    # F0 and N
    for tag, proj, outname in (("f0b", "f0_proj", "t_f0"), ("nb", "n_proj", "t_n")):
        cur = "t_shared"
        for bi, (din, dout, up) in enumerate(((H, H, False), (H, H // 2, True), (H // 2, H // 2, False))):
            bl, cur = _resblk1d(f"{tag}{bi}", din, dout, "style_pred", up, cur, S)
            nodes += bl
        nodes.append(_nd(proj, "conv1d_fp32", cur, outname, C_in=H // 2, C_out=1, kernel_size=1, stride=1, padding=0))

    # decoder
    nodes += [_nd("dec_f0_conv", "conv1d_fp32", "t_f0", "t_f0d", C_in=1, C_out=1, kernel_size=3, stride=2, padding=1),
              _nd("dec_n_conv", "conv1d_fp32", "t_n", "t_nd", C_in=1, C_out=1, kernel_size=3, stride=2, padding=1),
              _nd("dec_cat_01", "concat1d_fp32", ["asr", "t_f0d"], "t_dec_cat01", C2=1),
              _nd("dec_cat_012", "concat1d_fp32", ["t_dec_cat01", "t_nd"], "t_dec_in", C2=1),
              _nd("asr_res", "conv1d_fp32", "asr", "t_asr_res", C_in=H, C_out=64, kernel_size=1, stride=1, padding=0),
              _nd("dec_ctx01", "concat1d_fp32", ["t_asr_res", "t_f0d"], "t_dec_ctx01", C2=1),
              _nd("dec_ctx012", "concat1d_fp32", ["t_dec_ctx01", "t_nd"], "t_dec_ctx", C2=1)]
    bl, x = _resblk1d("enc", H + 2, d["enc_ch"], "style_dec", False, "t_dec_in", S)
    nodes += bl
    ctx = 66
    widths = [(d["enc_ch"] + ctx, d["dec_hidden"], False)] * 3 + [(d["dec_hidden"] + ctx, d["ups_init"], True)]
    for di, (din, dout, up) in enumerate(widths):
        nodes.append(_nd(f"dec{di}_cat", "concat1d_fp32", [x, "t_dec_ctx"], f"t_dec{di}_in", C2=ctx))
        bl, x = _resblk1d(f"dec{di}", din, dout, "style_dec", up, f"t_dec{di}_in", S)
        nodes += bl

    # generator (iSTFTNet)
    rates, kernels, n_up = d["ups_rates"], d["ups_kernels"], len(d["ups_rates"])
    n_fft, hop = d["n_fft"], d["hop"]
    scale = math.prod(rates) * hop
    nodes += [_nd("gen_f0_up", "upsample_nearest1d_fp32", "t_f0", "t_f0_up", scale_factor=scale),
              _nd("gen_sine", "sine_gen_fp32", "t_f0_up", "t_sines", sample_rate=24000, harmonic_num=8, sine_amp=0.1,
                  voiced_threshold=10.0, noise_std=0.003, upsample_scale=scale, torch_order=True,
                  noise=True, noise_seed=0),
              _nd("gen_l_lin", "seqgemm_fp32", "t_sines", "t_har_lin", C_in=9, C_out=1),
              _nd("gen_l_tanh", "tanh_fp32", "t_har_lin", "t_har_src"),
              _nd("gen_stft", "stft_fp32", "t_har_src", "t_har", n_fft=n_fft, hop_size=hop, torch_stft=True)]
    nk = len(d["rb_kernels"])
    for i in range(n_up):
        cin, cout = d["ups_init"] // 2 ** i, d["ups_init"] // 2 ** (i + 1)
        nodes.append(_nd(f"gen_lr{i}", "leaky_relu_fp32", x, f"t_gen_lr{i}", alpha=0.1))
        if i + 1 < n_up:
            sf = math.prod(rates[i + 1:])
            nk_, ns_, np_ = sf * 2, sf, (sf + 1) // 2
        else:
            nk_, ns_, np_ = 1, 1, 0
        nodes.append(_nd(f"gen_nc{i}", "conv1d_fp32", "t_har", f"t_gen_nc{i}", C_in=n_fft + 2, C_out=cout,
                         kernel_size=nk_, stride=ns_, padding=np_))
        bl, nr = _resblock1(f"gen_nr{i}", cout, 7 if i + 1 < n_up else 11, [1, 3, 5], "style_dec", f"t_gen_nc{i}", S)
        nodes += bl
        nodes.append(_nd(f"gen_up{i}", "conv_transpose1d_fp32", f"t_gen_lr{i}", f"t_gen_up{i}", C_in=cin, C_out=cout,
                         kernel_size=kernels[i], stride=rates[i], padding=(kernels[i] - rates[i]) // 2))
        up = f"t_gen_up{i}"
        if i == n_up - 1:
            nodes.append(_nd(f"gen_pad{i}", "reflection_pad1d_fp32", up, f"t_gen_pad{i}"))
            up = f"t_gen_pad{i}"
        nodes.append(_nd(f"gen_add_n{i}", "add_fp32", [up, nr], f"t_gen_xn{i}"))
        outs = []
        for j, (rk, rd) in enumerate(zip(d["rb_kernels"], d["rb_dilations"])):
            bl, o = _resblock1(f"gen_rb{i}_{j}", cout, rk, rd, "style_dec", f"t_gen_xn{i}", S)
            nodes += bl
            outs.append(o)
        acc = outs[0]
        for j in range(1, nk):   # ((rb0 + rb1) + rb2) / nk, the reference's order
            name = f"t_gen_rb{i}_s" if j == nk - 1 else f"t_gen_rb{i}_s{j}"
            nodes.append(_nd(f"gen_rb{i}_sum{j}", "add_fp32", [acc, outs[j]], name))
            acc = name
        nodes.append(_nd(f"gen_rb{i}_avg", "scale_fp32", acc, f"t_gen_rb{i}", scalar=1.0 / nk))
        x = f"t_gen_rb{i}"
    bins = n_fft // 2 + 1
    final_ch = d["ups_init"] // 2 ** n_up
    nodes += [_nd("gen_final_lr", "leaky_relu_fp32", x, "t_gen_flr", alpha=0.01),
              _nd("gen_post", "conv1d_fp32", "t_gen_flr", "t_gen_post", C_in=final_ch, C_out=n_fft + 2, kernel_size=7,
                  stride=1, padding=3),
              _nd("gen_slice_mag", "slice_channels_fp32", "t_gen_post", "t_gen_mag", ch_start=0, ch_end=bins),
              _nd("gen_slice_phase", "slice_channels_fp32", "t_gen_post", "t_gen_phase", ch_start=bins,
                  ch_end=2 * bins),
              _nd("gen_mag", "softplus_fp32" if d["magnitude"] == "softplus" else "exp_fp32", "t_gen_mag", "t_spec"),
              _nd("gen_sin", "sin_fp32", "t_gen_phase", "t_phase"),
              _nd("gen_istft", "istft_fp32", ["t_spec", "t_phase"], "t_audio", n_fft=n_fft, hop_size=hop,
                  normalized=True)]
    return {"model": model_name,
            "inputs": [{"name": "phoneme_ids", "shape": [1, 0, 1, 1], "dtype": "float32"},
                       {"name": "style_dec", "shape": [1, 1, 1, S], "dtype": "float32"},
                       {"name": "style_pred", "shape": [1, 1, 1, S], "dtype": "float32"}],
            "outputs": [{"name": "t_audio"}, {"name": "t_dur"}],
            "nodes": nodes}


# ---- weights -----------------------------------------------------------------------------------------------------
def _wn(sd, p):
    if p + ".weight_g" not in sd:
        return sd[p + ".weight"]
    g, v = sd[p + ".weight_g"].astype(np.float64), sd[p + ".weight_v"].astype(np.float64)
    norm = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape(-1, *([1] * (v.ndim - 1)))
    return (g * v / np.maximum(norm, 1e-12)).astype(np.float32)


class _Extractor:
    def __init__(self, sd, w):
        self.sd, self.w, self.used = sd, w, set()

    def t(self, key, required=True):
        if key not in self.sd:
            if required:
                raise KeyError(f"weights file has no tensor {key}")
            return None
        self.used.add(key)
        return self.sd[key]

    def conv(self, node, p, weight_norm=True):
        if weight_norm and p + ".weight_g" in self.sd:
            self.used.update({p + ".weight_g", p + ".weight_v"})
            self.w.add(f"{node}.weight", _wn(self.sd, p))
        else:
            self.w.add(f"{node}.weight", self.t(p + ".weight"))
        b = self.t(p + ".bias", required=False)
        if b is not None:
            self.w.add(f"{node}.bias", b)

    def linear(self, node, p):
        self.w.add(f"{node}.weight", self.t(p + ".weight"))
        b = self.t(p + ".bias", required=False)
        if b is not None:
            self.w.add(f"{node}.bias", b)

    def adain(self, node, p):
        self.linear(f"{node}.fc", p + ".fc")
        nw, nb = self.t(p + ".norm.weight", False), self.t(p + ".norm.bias", False)
        if nw is not None and not np.allclose(nw, 1.0):
            self.w.add(f"{node}.norm.weight", nw)
        if nb is not None and not np.allclose(nb, 0.0):
            self.w.add(f"{node}.norm.bias", nb)

    def lstm(self, node, p):
        for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "weight_ih_l0_reverse",
                  "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse"):
            self.w.add(f"{node}.{k}", self.t(f"{p}.{k}"))

    def resblk1d(self, node, p, din, dout, up):
        self.adain(f"{node}_n1", p + ".norm1")
        if up:
            self.conv(f"{node}_pool", p + ".pool")
        self.conv(f"{node}_c1", p + ".conv1")
        self.adain(f"{node}_n2", p + ".norm2")
        self.conv(f"{node}_c2", p + ".conv2")
        if din != dout:
            self.conv(f"{node}_sc", p + ".conv1x1")

    def resblock1(self, node, p, n_dil):
        for di in range(n_dil):
            ip = f"{node}_d{di}"
            self.adain(f"{ip}_n1", f"{p}.adain1.{di}")
            self.w.add(f"{ip}_sk1.alpha", self.t(f"{p}.alpha1.{di}").reshape(-1))
            self.conv(f"{ip}_c1", f"{p}.convs1.{di}")
            self.adain(f"{ip}_n2", f"{p}.adain2.{di}")
            self.w.add(f"{ip}_sk2.alpha", self.t(f"{p}.alpha2.{di}").reshape(-1))
            self.conv(f"{ip}_c2", f"{p}.convs2.{di}")


def stft_filters(n_fft):
    """CustomSTFT / torch.stft analysis filters: periodic Hann (float32, as torch.hann_window) times cos / -sin."""
    n = np.arange(n_fft)
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)).astype(np.float32).astype(np.float64)
    ang = 2.0 * np.pi * np.outer(np.arange(n_fft // 2 + 1), n) / n_fft
    return (np.cos(ang) * win).astype(np.float32), (-np.sin(ang) * win).astype(np.float32)


def extract_weights(sd, arch):
    """W1X1 tensors (fp32) of one model from its safetensors state dict."""
    d = dims(arch)
    w = W1X1()
    e = _Extractor(sd, w)
    H, L = d["hidden"], d["n_layer"]
    for node_key, key in (("word_weight", "embeddings.word_embeddings.weight"),
                          ("pos_weight", "embeddings.position_embeddings.weight"),
                          ("layernorm.weight", "embeddings.LayerNorm.weight"),
                          ("layernorm.bias", "embeddings.LayerNorm.bias")):
        w.add(f"bert_embed.{node_key}", e.t("bert." + key))
    w.add("bert_embed.type_weight", e.t("bert.embeddings.token_type_embeddings.weight")[0:1].reshape(-1))
    e.linear("bert_enc_proj", "bert.encoder.embedding_hidden_mapping_in")
    al = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    q, k, v = (e.t(f"{al}.attention.{n}.weight") for n in ("query", "key", "value"))
    w.add("bert_attn.w_qkv", np.concatenate([q, k, v], axis=0))
    w.add("bert_attn.b_qkv", np.concatenate([e.t(f"{al}.attention.{n}.bias") for n in ("query", "key", "value")]))
    w.add("bert_attn.w_proj", e.t(f"{al}.attention.dense.weight"))
    w.add("bert_attn.b_proj", e.t(f"{al}.attention.dense.bias"))
    e.linear("bert_ff1", f"{al}.ffn")
    e.linear("bert_ff2", f"{al}.ffn_output")
    w.add("bert_attn_ln.gamma", e.t(f"{al}.attention.LayerNorm.weight"))
    w.add("bert_attn_ln.beta", e.t(f"{al}.attention.LayerNorm.bias"))
    w.add("bert_full_ln.gamma", e.t(f"{al}.full_layer_layer_norm.weight"))
    w.add("bert_full_ln.beta", e.t(f"{al}.full_layer_layer_norm.bias"))
    e.linear("bert_enc", "bert_encoder")

    w.add("te_embed.weight", e.t("text_encoder.embedding.weight"))
    for ci in range(L):
        e.conv(f"te_conv_{ci}", f"text_encoder.cnn.{ci}.0")
        w.add(f"te_ln_{ci}.gamma", e.t(f"text_encoder.cnn.{ci}.1.gamma"))
        w.add(f"te_ln_{ci}.beta", e.t(f"text_encoder.cnn.{ci}.1.beta"))
    e.lstm("te_lstm", "text_encoder.lstm")
    for li in range(L):
        e.lstm(f"de_lstm_{li}", f"predictor.text_encoder.lstms.{2 * li}")
        e.linear(f"de_adaln_{li}.fc", f"predictor.text_encoder.lstms.{2 * li + 1}.fc")
    e.lstm("pred_lstm", "predictor.lstm")
    e.linear("dur_proj", "predictor.duration_proj.linear_layer")
    e.lstm("shared_lstm", "predictor.shared")
    for tag, src in (("f0b", "F0"), ("nb", "N")):
        for bi, (din, dout, up) in enumerate(((H, H, False), (H, H // 2, True), (H // 2, H // 2, False))):
            e.resblk1d(f"{tag}{bi}", f"predictor.{src}.{bi}", din, dout, up)
    e.conv("f0_proj", "predictor.F0_proj", weight_norm=False)
    e.conv("n_proj", "predictor.N_proj", weight_norm=False)

    e.conv("dec_f0_conv", "decoder.F0_conv")
    e.conv("dec_n_conv", "decoder.N_conv")
    e.conv("asr_res", "decoder.asr_res.0")
    e.resblk1d("enc", "decoder.encode", H + 2, d["enc_ch"], False)
    widths = [(d["enc_ch"] + 66, d["dec_hidden"], False)] * 3 + [(d["dec_hidden"] + 66, d["ups_init"], True)]
    for di, (din, dout, up) in enumerate(widths):
        e.resblk1d(f"dec{di}", f"decoder.decode.{di}", din, dout, up)

    g = "decoder.generator"
    e.linear("gen_l_lin", f"{g}.m_source.l_linear")
    nk = len(d["rb_kernels"])
    for i in range(len(d["ups_rates"])):
        e.conv(f"gen_nc{i}", f"{g}.noise_convs.{i}", weight_norm=False)
        e.resblock1(f"gen_nr{i}", f"{g}.noise_res.{i}", 3)
        e.conv(f"gen_up{i}", f"{g}.ups.{i}")
        for j in range(nk):
            e.resblock1(f"gen_rb{i}_{j}", f"{g}.resblocks.{i * nk + j}", len(d["rb_dilations"][j]))
    e.conv("gen_post", f"{g}.conv_post")
    wr, wi = stft_filters(d["n_fft"])
    w.add("gen_stft.w_real", wr)
    w.add("gen_stft.w_imag", wi)

    unused = sorted(k for k in sd if k not in e.used and not k.startswith("bert.pooler."))
    if unused:
        raise ValueError(f"weights the graph does not use (unsupported model variant?): {unused[:8]}")
    return w


# ---- int8 ----------------------------------------------------------------------------------------------------------
SEQGEMM_INT8 = {"bert_enc_proj", "bert_ff1", "bert_ff2", "bert_enc"}
CONV1D_KEEP_FP32 = {"gen_post"}


def int8_arch(arch_json, model_name):
    a = json.loads(json.dumps(arch_json))
    for n in a["nodes"]:
        if n["op"] == "seqgemm_fp32" and n["name"] in SEQGEMM_INT8:
            n["op"] = "seqgemm_int8"
        elif n["op"] == "attention_fp32" and n["name"] == "bert_attn":
            n["op"] = "attention_int8"
        elif n["op"] == "conv1d_fp32" and n["name"] not in CONV1D_KEEP_FP32:
            n["op"] = "conv1d_int8"
    a["model"] = model_name
    return a


def _q8(w2d):
    amax = np.maximum(np.abs(w2d).max(axis=1, keepdims=True), 1e-8)
    scales = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(w2d / scales), -128, 127).astype(np.int8)
    return q, scales.reshape(-1)


def int8_weights(fp32, arch_json):
    """W1X1 int8 weights from the fp32 writer, laid out as tools/kitten/extract_weights_kitten_int8.py does for
    TTS 0.8 (per-output-channel symmetric; conv weights reordered to [C_out, K, C_in])."""
    conv = {n["name"] for n in arch_json["nodes"] if n["op"] == "conv1d_fp32"} - CONV1D_KEEP_FP32
    ada = {n["name"] for n in arch_json["nodes"] if n["op"] in ("ada_in1d_fp32", "ada_layer_norm_fp32")}
    out = W1X1()
    for name, blob in fp32.blobs.items():
        ndim = struct.unpack_from("<H", blob, 2)[0]
        shape = struct.unpack_from("<" + "i" * ndim, blob, 4)
        arr = np.frombuffer(blob, dtype=np.float32, offset=4 + 4 * ndim).reshape(shape)
        layer, _, field = name.rpartition(".")
        if (layer in SEQGEMM_INT8 or layer == "dur_proj") and field == "weight":
            q, s = _q8(arr.reshape(arr.shape[0], -1))
            out.add(layer + ".weight_i8", q, np.int8)
            out.add(layer + ".w_scales", s)
        elif layer == "bert_attn" and field in ("w_qkv", "w_proj"):
            q, s = _q8(arr.reshape(arr.shape[0], -1))
            out.add(f"{layer}.{field}_i8", q, np.int8)
            out.add(f"{layer}.{field}_scales", s)
        elif layer in conv and field == "weight":
            w2 = arr.transpose(0, 2, 1).reshape(arr.shape[0], -1) if arr.ndim == 3 else arr.reshape(arr.shape[0], -1)
            q, s = _q8(w2)
            out.add(layer + ".weight_i8", q, np.int8)
            out.add(layer + ".w_scales", s)
        elif layer.endswith(".fc") and layer[:-3] in ada and field == "weight":
            q, s = _q8(arr.reshape(arr.shape[0], -1))
            out.add(layer + ".weight_i8", q, np.int8)
            out.add(layer + ".w_scales", s)
        else:
            out.blobs[name] = blob
    return out


# ---- voices --------------------------------------------------------------------------------------------------------
def voice_packs(npz_path):
    """{voice key: float32 [rows, 256]} from a voices_<language>.npz."""
    with np.load(npz_path) as z:
        return {k: np.ascontiguousarray(z[k], dtype=np.float32).reshape(-1, 256) for k in z.files}


# ---- whole repository -------------------------------------------------------------------------------------------
def convert_model(config, model_key, weights_path, out_dir, int8=True, fp32=True):
    """Writes <model_key>_fp32_arch.json / _fp32.bin (fp32=True) and _int8_* (int8=True) into out_dir. Returns their
    paths. The int8 files are made from the float32 ones in memory, so fp32=False still reads the float32 weights."""
    arch = config["models"][model_key]["architecture"]
    sd = load_safetensors(weights_path)
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    a32 = build_arch(arch, f"{model_key}_fp32")
    w32 = extract_weights(sd, arch)
    if fp32:
        out["fp32_arch"] = os.path.join(out_dir, f"{model_key}_fp32_arch.json")
        out["fp32_weights"] = os.path.join(out_dir, f"{model_key}_fp32.bin")
        _write_json(out["fp32_arch"], a32)
        w32.write(out["fp32_weights"])
    if int8:
        a8 = int8_arch(a32, f"{model_key}_int8")
        out["int8_arch"] = os.path.join(out_dir, f"{model_key}_int8_arch.json")
        out["int8_weights"] = os.path.join(out_dir, f"{model_key}_int8.bin")
        _write_json(out["int8_arch"], a8)
        int8_weights(w32, a32).write(out["int8_weights"])
    return out


def write_voice_bins(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, pack in voice_packs(npz_path).items():
        p = os.path.join(out_dir, key + ".bin")
        tmp = _tmp_name(p)
        pack.astype("<f4").tofile(tmp)
        os.replace(tmp, p)
        paths[key] = p
    return paths


def _tmp_name(path):
    """A temporary name next to path that no other converter uses: several processes, or several threads of one
    process, may convert the same model at once; each writes its own file and renames it into place."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"


def _write_json(path, obj):
    tmp = _tmp_name(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def convert_repo(repo_dir, out_dir, models=None, int8=True, voice_bins=True):
    """Converts every model of a downloaded repository (or the ones named in `models`). Returns the manifest."""
    with open(os.path.join(repo_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if config.get("type") != "KITTEN2_TINY":
        raise ValueError(f"{repo_dir}: config type {config.get('type')!r} is not KITTEN2_TINY")
    manifest = {"format_version": FORMAT_VERSION, "family": config.get("family"), "models": {}, "voices": {},
                "missing": []}
    wanted = set(models) if models else None
    for key, entry in config["models"].items():
        if wanted is not None and key not in wanted:
            continue
        if not os.path.isfile(os.path.join(repo_dir, entry["safetensors"])):   # a partial download
            manifest["missing"].append(entry["safetensors"])
            continue
        manifest["models"][key] = {
            k: os.path.basename(v) for k, v in
            convert_model(config, key, os.path.join(repo_dir, entry["safetensors"]), out_dir, int8).items()}
    if voice_bins:
        for lang, entry in config["languages"].items():
            if wanted is not None and entry["model"] not in wanted:
                continue
            if not os.path.isfile(os.path.join(repo_dir, entry["voices"])):
                manifest["missing"].append(entry["voices"])
                continue
            paths = write_voice_bins(os.path.join(repo_dir, entry["voices"]), os.path.join(out_dir, "voices", lang))
            manifest["voices"][lang] = {k: os.path.relpath(p, out_dir) for k, p in paths.items()}
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Convert a Kitten TTS 2 mini/nano/pico repository for the native engine")
    p.add_argument("repo_dir", help="downloaded repository folder (config.json, *.safetensors, voices_*.npz)")
    p.add_argument("out_dir")
    p.add_argument("--model", action="append", help="model key from config.json (default: all)")
    p.add_argument("--no-int8", action="store_true")
    a = p.parse_args(argv)
    m = convert_repo(a.repo_dir, a.out_dir, a.model, int8=not a.no_int8)
    print(json.dumps(m["models"], indent=1))
    if m["missing"]:
        print("not in the folder (skipped):", ", ".join(m["missing"]))


if __name__ == "__main__":
    main()
