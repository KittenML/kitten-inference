"""Prepares the files of the C library's optional Qualcomm NPU path (kt08_npu_load) for one Kitten TTS 2 mini, nano or
pico model and one Snapdragon chip.

    python engines/lite/mobile/npu/prepare_npu.py CONVERTED_DIR MODEL OUT_DIR --chip SM8650 --qairt QAIRT_SDK

CONVERTED_DIR is the converter's output folder (python -m kitten_inference.tts2_tiny.convert REPO_DIR CONVERTED_DIR),
MODEL a model key in it (the part before _fp32_arch.json). OUT_DIR receives, for kt08_npu_options.model_dir:
  npu.json             what the C library reads: the CPU graphs, the NPU graphs and their lengths
  front.json           CPU graph: text encoder, durations, pitch and energy, harmonic source (the engine's own nodes)
  post.json            CPU graph: magnitude, phase and inverse STFT
  body_<frames>.bin    the decoder and vocoder compiled for the chip's NPU (QNN HTP context binary), one per length
                       (with --share-weights: body.bin, every length in one file with one copy of the weights)
and, for checking, body_<frames>.onnx (the exported graph before compiling). The phone also needs the model's fp32
weights file (<MODEL>_fp32.bin, unchanged), the voices, and Qualcomm's runtime libraries (see the mobile README).

How the decoder is made NPU-friendly (same arithmetic as the engine, checked by --verify):
  * fixed lengths: 80 frames of 25 ms (2 s) and 160 (4 s) by default; a shorter chunk is padded with zeros and 0/1
    masks keep the padding out of every convolution and every mean, so the chunk's frames do not depend on the length;
  * time on the H axis of 4-D tensors [1, C, T, 1]; every 1-D convolution as a 2-D one with a (k, 1) kernel;
  * adaptive instance norm statistics at the two decoder rates computed in a range-safe form (each channel scaled by its
    peak first), which 16-bit arithmetic needs; the depthwise transposed convolution as two plain convolutions
    interleaved (exact);
  * the output kept 4-D [1, 22, T, 1] (a final 3-D squeeze of this odd-length tensor was miscompiled in fp16).
Precision: fp16 (Snapdragon 8 Gen 2 and newer: Hexagon v73 and up). The NPU graphs are compiled ahead of time on this
machine with Qualcomm's QAIRT SDK (qairt-converter, qnn-context-binary-generator); building them needs no phone.

Requirements (this script only, not the phone): kitten-inference, numpy, torch, onnx; onnxruntime for --verify; the
QAIRT SDK (Linux x86-64) for compiling. Compiling takes 4 minutes to more than an hour per length (longer lengths and
bigger models take longer). Chunks longer than the longest length go to the CPU fallback: on the phones measured, an
8 s graph (--frames 80 160 320) was slower than the CPU, so it is not made by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

# Snapdragon phone chips: Hexagon architecture and the QNN SoC model number (QnnTypes.h QNN_SOC_MODEL_*).
CHIPS = {
    "SM8750": ("v79", 69),   # Snapdragon 8 Elite
    "SM8650": ("v75", 57),   # Snapdragon 8 Gen 3
    "SM8550": ("v73", 43),   # Snapdragon 8 Gen 2
    "SM8475": ("v69", 42),   # Snapdragon 8+ Gen 1
    "SM8450": ("v69", 36),   # Snapdragon 8 Gen 1
    "SM7325": ("v68", 35),   # Snapdragon 778G
}

INPUTS = ["asr", "f0", "n", "har", "style", "m40", "m80", "m800", "m4800", "s40", "s80", "s800", "s4800"]
FRONT_OUTPUTS = ["asr", "t_f0", "t_n", "t_har", "t_dur"]
BODY_IN = {"asr": "asr", "t_f0": "f0", "t_n": "n", "t_har": "har", "style_dec": "style"}


# ---- engine files ------------------------------------------------------------------------------------------------
def read_w1x1(path):
    """{name: float32 array} of an engine weights file (fp32 tensors only)."""
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"W1X1":
        raise ValueError(f"{path} is not an engine weights file")
    _, n = struct.unpack_from("<II", data, 4)
    for i in range(n):
        e = 12 + 80 * i
        name = data[e:e + 64].split(b"\0")[0].decode()
        off, size, _ = struct.unpack_from("<QII", data, e + 64)
        code, _, ndim = struct.unpack_from("BBH", data, off)
        shape = struct.unpack_from("i" * ndim, data, off + 4)
        if code != 1:
            continue
        start = off + 4 + 4 * ndim
        out[name] = np.frombuffer(data, np.float32, int(np.prod(shape)) if ndim else 1, start).reshape(shape)
    return out


def split_graph(arch):
    """(front arch, post arch, body nodes) of a converted Kitten TTS 2 mini/nano/pico graph."""
    nodes = arch["nodes"]
    prod = {o: i for i, nd in enumerate(nodes) for o in nd["outputs"]}

    def ancestors(targets):
        need, stack = set(), list(targets)
        while stack:
            t = stack.pop()
            if t in prod and prod[t] not in need:
                need.add(prod[t])
                stack.extend(nodes[prod[t]]["inputs"])
        return need

    for t in FRONT_OUTPUTS + ["t_gen_post", "t_audio"]:
        if t not in prod:
            raise ValueError(f"the graph has no tensor {t}: not a Kitten TTS 2 mini/nano/pico model")
    front_ids = ancestors(FRONT_OUTPUTS)
    body_ids = ancestors(["t_gen_post"]) - front_ids
    have, post = {"t_gen_post"}, []
    for nd in nodes:   # the nodes after t_gen_post, fed only by it or by each other
        if any(i in have for i in nd["inputs"]):
            if not all(i in have for i in nd["inputs"]):
                raise ValueError(f"node {nd['name']} mixes the decoder output with other tensors")
            post.append(nd)
            have.update(nd["outputs"])
    front = dict(arch, nodes=[nodes[i] for i in sorted(front_ids)], outputs=[{"name": t} for t in FRONT_OUTPUTS])
    post_arch = dict(arch, nodes=post, outputs=[{"name": "t_audio"}],
                     inputs=[{"name": "t_gen_post", "shape": [1, 0, 1, 22], "dtype": "float32"}])
    body = [nodes[i] for i in sorted(body_ids)]
    for nd in body:
        for t in nd["inputs"]:
            if t not in prod or prod[t] in front_ids:
                if t not in BODY_IN:
                    raise ValueError(f"decoder node {nd['name']} reads {t}, which the NPU graph does not get")
    return front, post_arch, body


# ---- decoder + vocoder as a fixed-length torch module -------------------------------------------------------------
def make_body(body_nodes, W, frames=None):
    import torch
    import torch.nn.functional as F

    class Body(torch.nn.Module):
        """The engine's decoder nodes, NPU form. frames=None: exact length, no masks (for checks)."""

        def __init__(self):
            super().__init__()
            self.nodes = body_nodes
            self.B = frames
            self.w = {k: torch.from_numpy(np.array(v, np.float32)) for k, v in W.items()
                      if k.split(".")[0] in {nd["name"] for nd in body_nodes}}

        def p(self, name, required=True):
            if name not in self.w:
                if required:
                    raise KeyError(f"weights file has no {name}")
                return None
            return self.w[name]

        def forward(self, asr, f0, n, har, style, m40=None, m80=None, m800=None, m4800=None,
                    s40=None, s80=None, s800=None, s4800=None):
            u = lambda t: None if t is None else t.unsqueeze(-1)
            env = {"asr": u(asr), "t_f0": u(f0), "t_n": u(n), "t_har": u(har), "style_dec": style}
            masked = m40 is not None
            B = int(asr.shape[2])
            M = {}
            if masked:
                M = {B: (u(m40), u(s40)), 2 * B: (u(m80), u(s80)), 20 * B: (u(m800), u(s800)),
                     120 * B + 1: (u(m4800), u(s4800))}

            def mk(x):
                m = M.get(int(x.shape[2]))
                return x if m is None else x * m[0]

            for nd in self.nodes:
                op, name = nd["op"], nd["name"]
                xs = [env[i] for i in nd["inputs"]]
                x = xs[0]
                if op == "conv1d_fp32":
                    k = int(nd["kernel_size"])
                    w = self.p(name + ".weight").unsqueeze(-1)
                    y = F.conv2d(mk(x) if k > 1 else x, w, self.p(name + ".bias", False),
                                 stride=(int(nd.get("stride", 1)), 1), padding=(int(nd.get("padding", 0)), 0),
                                 dilation=(int(nd.get("dilation", 1)), 1), groups=int(nd.get("groups", 1)))
                elif op == "conv_transpose1d_fp32":
                    k, s, pd = int(nd["kernel_size"]), int(nd["stride"]), int(nd.get("padding", 0))
                    op_, g = int(nd.get("output_padding", 0)), int(nd.get("groups", 1))
                    w, b = self.p(name + ".weight"), self.p(name + ".bias", False)
                    if g == int(nd["C_in"]) and g > 1 and (k, s, pd, op_) == (3, 2, 1, 1):
                        # depthwise, exact polyphase: y[2t] = w1 x[t] + b; y[2t+1] = w2 x[t] + w0 x[t+1] + b
                        x = mk(x)
                        wv = w.reshape(-1, 3)
                        w0, w1, w2 = (wv[:, j].reshape(1, -1, 1, 1) for j in range(3))
                        bb = 0 if b is None else b.reshape(1, -1, 1, 1)
                        xn = torch.cat([x[:, :, 1:], torch.zeros_like(x[:, :, :1])], dim=2)
                        y = torch.cat([x * w1 + bb, x * w2 + xn * w0 + bb], dim=3).reshape(
                            x.shape[0], x.shape[1], 2 * x.shape[2], 1)
                    else:
                        y = F.conv_transpose2d(mk(x), w.unsqueeze(-1), b, stride=(s, 1), padding=(pd, 0),
                                               output_padding=(op_, 0), groups=g)
                elif op == "ada_in1d_fp32":
                    y = self.adain(name, x, xs[1], M, masked)
                elif op == "leaky_relu_fp32":
                    y = F.leaky_relu(x, float(nd.get("alpha", 0.01)))
                elif op == "concat1d_fp32":
                    y = torch.cat(xs, dim=1)
                elif op == "upsample_nearest1d_fp32":
                    if int(nd.get("scale_factor", 2)) != 2:
                        raise ValueError(f"{name}: only x2 nearest upsampling is supported in the decoder")
                    y = torch.cat([x, x], dim=3).reshape(x.shape[0], x.shape[1], 2 * x.shape[2], 1)
                elif op == "add_fp32":
                    y = xs[0] + xs[1]
                elif op == "scale_fp32":
                    y = x * float(nd["scalar"])
                elif op == "snake1d_fp32":
                    a = self.p(name + ".alpha").reshape(1, -1, 1, 1)
                    ia = (1.0 / a).clone()   # a constant, not a Reciprocal node
                    sn = torch.sin(a * x)
                    y = x + ia * (sn * sn)
                elif op == "reflection_pad1d_fp32":
                    y = torch.cat([x[:, :, 1:2], x], dim=2)
                else:
                    raise ValueError(f"decoder op {op} ({name}) has no NPU form here")
                env[nd["outputs"][0]] = y
            return env["t_gen_post"]   # [1, 22, T, 1]: kept 4-D

        def adain(self, name, x, s, M, masked):
            C = x.shape[1]
            h = F.linear(s, self.p(name + ".fc.weight"), self.p(name + ".fc.bias", False))
            gamma, beta = h[:, :C].reshape(1, C, 1, 1), h[:, C:].reshape(1, C, 1, 1)
            nw, nb = self.p(name + ".norm.weight", False), self.p(name + ".norm.bias", False)
            eps = 1e-5
            T = int(x.shape[2])
            m, sc = M.get(T, (None, None)) if masked else (None, None)
            B = int(self.B) if self.B else None
            if B is not None and T <= 2 * B:
                # Range-safe statistics: per-channel peak first, so 16-bit sums of squares cannot overflow or vanish.
                xm = x if m is None else x * m
                mean = xm.mean(2, keepdim=True) if sc is None else xm.mean(2, keepdim=True) * sc
                d = x - mean
                dm = d if m is None else d * m
                r = 1.0 / (dm.abs().amax(2, keepdim=True) + 1e-3)
                ds = dm * r
                vs = (ds * ds).mean(2, keepdim=True)
                if sc is not None:
                    vs = vs * sc
                er = r * (eps ** 0.5)
                xn = (d * r) / torch.sqrt(vs + er * er)
            else:
                xm = x if m is None else x * m
                mean = xm.mean(2, keepdim=True) if sc is None else xm.mean(2, keepdim=True) * sc
                d = x - mean
                dm = d if m is None else d * m
                var = (dm * dm).mean(2, keepdim=True)
                if sc is not None:
                    var = var * sc
                xn = d / torch.sqrt(var + eps)
            if nw is not None:
                xn = xn * nw.reshape(1, -1, 1, 1)
            if nb is not None:
                xn = xn + nb.reshape(1, -1, 1, 1)
            return (1 + gamma) * xn + beta

    return Body().eval()


def bucket_inputs(asr, f0, n, har, style, B):
    """Exact-length tensors in engine layout (asr [L, H], f0/n [2L], har [120L+1, 22], style [128]) -> the padded
    bucket inputs (numpy, the NPU graph's names and shapes)."""
    L, H = asr.shape
    if L > B:
        raise ValueError(f"chunk of {L} frames does not fit {B}")
    o = {"asr": np.zeros((1, H, B), np.float32), "f0": np.zeros((1, 1, 2 * B), np.float32),
         "n": np.zeros((1, 1, 2 * B), np.float32), "har": np.zeros((1, 22, 120 * B + 1), np.float32),
         "style": np.asarray(style, np.float32).reshape(1, -1)}
    o["asr"][0, :, :L] = asr.T
    o["f0"][0, 0, :2 * L] = f0
    o["n"][0, 0, :2 * L] = n
    o["har"][0, :, :120 * L + 1] = har.T
    for tag, lp, lv in (("40", B, L), ("80", 2 * B, 2 * L), ("800", 20 * B, 20 * L),
                        ("4800", 120 * B + 1, 120 * L + 1)):
        m = np.zeros((1, 1, lp), np.float32)
        m[..., :lv] = 1
        o["m" + tag] = m
        o["s" + tag] = np.full((1, 1, 1), lp / lv, np.float32)
    return o


def export_onnx(body_nodes, W, hidden, B, path):
    import torch
    body = make_body(body_nodes, W, B)
    ex = bucket_inputs(np.zeros((B // 2, hidden), np.float32), np.zeros(B, np.float32), np.zeros(B, np.float32),
                       np.zeros((60 * B + 1, 22), np.float32), np.zeros(128, np.float32), B)
    args = tuple(torch.from_numpy(ex[k]) for k in INPUTS)
    with torch.no_grad():
        torch.onnx.export(body, args, path, input_names=INPUTS, output_names=["post"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
    return path


# ---- checks against the engine ----------------------------------------------------------------------------------
def snr_db(ref, x):
    ref, x = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30)))


def verify(conv_dir, model, out_dir, frames, phoneme_ids, voice_path):
    """Runs the engine on one chunk and compares: the front/post CPU graphs with the full graph, and the NPU-form
    decoder (torch, and the exported ONNX in onnxruntime) with the engine's decoder output. Prints SNRs in dB."""
    import torch
    import onnxruntime as ort
    from kitten_inference.tts08 import cpp_model
    eng = cpp_model._import_engine()
    arch_path = os.path.join(conv_dir, f"{model}_fp32_arch.json")
    wpath = os.path.join(conv_dir, f"{model}_fp32.bin")
    arch = json.load(open(arch_path))
    probe = dict(arch, outputs=[{"name": t} for t in FRONT_OUTPUTS + ["t_gen_post", "t_audio"]])
    probe_path = os.path.join(out_dir, "probe_arch.json")
    json.dump(probe, open(probe_path, "w"))
    voice = np.fromfile(voice_path, "<f4").reshape(-1, 256)
    ids = np.asarray(phoneme_ids, np.float32)
    style = voice[min(len(ids) - 1, len(voice) - 1)]
    feed = {"phoneme_ids": ids, "style_dec": style[:128].copy(), "style_pred": style[128:].copy()}
    full = eng.InferenceModel(probe_path, wpath).forward(feed)
    front = eng.InferenceModel(os.path.join(out_dir, "front.json"), wpath).forward(feed)
    post = eng.InferenceModel(os.path.join(out_dir, "post.json"), wpath).forward(
        {"t_gen_post": np.asarray(full["t_gen_post"], np.float32)})
    res = {"front_" + t: snr_db(full[t], front[t]) for t in FRONT_OUTPUTS if t != "t_dur"}
    res["front_dur_equal"] = bool(np.array_equal(np.asarray(full["t_dur"]), np.asarray(front["t_dur"])))
    res["post_audio"] = snr_db(full["t_audio"], post["t_audio"])
    W = read_w1x1(wpath)
    H = int(W["asr_res.weight"].shape[1])
    asr = np.asarray(full["asr"], np.float32).reshape(-1, H)
    L = asr.shape[0]
    f0 = np.asarray(full["t_f0"], np.float32).ravel()
    nn = np.asarray(full["t_n"], np.float32).ravel()
    har = np.asarray(full["t_har"], np.float32).reshape(-1, 22)
    ref = np.asarray(full["t_gen_post"], np.float32).reshape(-1, 22)
    if f0.size != 2 * L or har.shape[0] != 120 * L + 1 or ref.shape[0] != 120 * L + 1:
        raise ValueError(f"unexpected engine tensor sizes: L={L} f0={f0.size} har={har.shape} post={ref.shape}")
    _, _, body_nodes = split_graph(arch)
    with torch.no_grad():
        ex = make_body(body_nodes, W, None)(torch.from_numpy(asr.T[None].copy()), torch.from_numpy(f0[None, None]),
                                            torch.from_numpy(nn[None, None]), torch.from_numpy(har.T[None].copy()),
                                            torch.from_numpy(style[None, :128].copy()))
    res["body_exact_torch"] = snr_db(ref, ex[0, :, :, 0].numpy().T)
    res["frames"] = int(L)
    for B in frames:
        if L > B:
            continue
        inp = bucket_inputs(asr, f0, nn, har, style[:128], B)
        with torch.no_grad():
            y = make_body(body_nodes, W, B)(*[torch.from_numpy(inp[k]) for k in INPUTS])[0, :, :, 0].numpy().T
        res[f"body_B{B}_torch"] = snr_db(ref, y[:120 * L + 1])
        onnx_path = os.path.join(out_dir, f"body_{B}.onnx")
        if os.path.exists(onnx_path):
            s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            y = s.run(None, inp)[0][0, :, :, 0].T
            res[f"body_B{B}_onnx"] = snr_db(ref, y[:120 * L + 1])
    os.remove(probe_path)
    return res


# ---- QNN compile ----------------------------------------------------------------------------------------------
def compile_qnn(onnx_paths, out_bin, qairt, chip, opt_level, workdir, log, qairt_python, share_weights):
    """ONNX graphs -> one QNN HTP context binary per graph, or (share_weights) one context holding every graph with
    one copy of the weights. Returns {graph name: context file} and the largest spill-fill buffer size."""
    arch_v, soc = chip
    env = dict(os.environ)
    bin_dir = os.path.join(qairt, "bin", "x86_64-linux-clang")
    lib_dir = os.path.join(qairt, "lib", "x86_64-linux-clang")
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = os.path.join(qairt, "lib", "python") + os.pathsep + env.get("PYTHONPATH", "")

    def run(cmd):
        t = time.time()
        with open(log, "a") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            f.write(f"# exit {r.returncode} after {time.time() - t:.0f} s\n")
        if r.returncode:
            raise RuntimeError(f"{os.path.basename(cmd[0])} failed (exit {r.returncode}); see {log}")

    dlcs = []
    for onnx_path in onnx_paths:
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        dlcs.append(os.path.join(workdir, stem + ".dlc"))
        run([qairt_python, os.path.join(bin_dir, "qairt-converter"), "--input_network", onnx_path,
             "--output_path", dlcs[-1]])
    groups = [dlcs] if share_weights else [[d] for d in dlcs]
    out, sf = {}, 0
    for group in groups:
        names = [os.path.splitext(os.path.basename(d))[0] for d in group]
        stem = os.path.splitext(os.path.basename(out_bin))[0] if share_weights else names[0]
        ext = os.path.join(workdir, stem + "_htp.json")
        cfg = os.path.join(workdir, stem + "_cfg.json")
        htp = {"graphs": [{"graph_names": names, "vtcm_mb": 0, "O": opt_level, "fp16_relaxed_precision": 1}],
               "devices": [{"dsp_arch": arch_v, "soc_model": soc}]}
        if share_weights:
            htp["context"] = {"weight_sharing_enabled": True}
        json.dump(htp, open(ext, "w"))
        json.dump({"backend_extensions": {"shared_library_path": os.path.join(lib_dir, "libQnnHtpNetRunExtensions.so"),
                                          "config_file_path": ext}}, open(cfg, "w"))
        run([os.path.join(bin_dir, "qnn-context-binary-generator"), "--backend", os.path.join(lib_dir, "libQnnHtp.so"),
             "--model", os.path.join(lib_dir, "libQnnModelDlc.so"), "--dlc_path", ",".join(group), "--binary_file", stem,
             "--output_dir", workdir, "--config_file", cfg, "--log_level", "warn"])
        target = os.path.join(os.path.dirname(out_bin), stem + ".bin")
        shutil.move(os.path.join(workdir, stem + ".bin"), target)
        info = os.path.join(workdir, stem + "_info.json")
        run([os.path.join(bin_dir, "qnn-context-binary-utility"), "--context_binary", target, "--json_file", info])
        sf = max(sf, spill_fill_bytes(json.load(open(info))))
        for nm in names:
            out[nm] = os.path.basename(target)
    return out, sf


def spill_fill_bytes(info):
    """The context's spill-fill buffer size from qnn-context-binary-utility's JSON (0 if not found)."""
    best = 0
    stack = [info]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            for k, x in v.items():
                if "spill" in k.lower() and isinstance(x, (int, float)):
                    best = max(best, int(x))
                else:
                    stack.append(x)
        elif isinstance(v, list):
            stack.extend(v)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("converted_dir", help="the converter's output folder")
    p.add_argument("model", help="model key in it (the part before _fp32_arch.json)")
    p.add_argument("out_dir")
    p.add_argument("--chip", help="Snapdragon chip: " + ", ".join(f"{k} ({v[0]})" for k, v in CHIPS.items()))
    p.add_argument("--dsp-arch", help="Hexagon architecture (v73, v75, v79, ...) for a chip not in the list")
    p.add_argument("--soc-model", type=int, help="QNN SoC model number for a chip not in the list (QnnTypes.h)")
    p.add_argument("--qairt", help="QAIRT SDK folder (Linux x86-64); without it only the ONNX graphs are written")
    p.add_argument("--frames", type=int, nargs="+", default=[80, 160],
                   help="fixed lengths in 25 ms frames (default 80 160 = 2 s and 4 s)")
    p.add_argument("--opt", type=int, default=3, help="QNN graph optimisation level (default 3; 1 compiles faster)")
    p.add_argument("--share-weights", action="store_true",
                   help="put every length in one NPU file that holds the weights once (slower to compile; on pico it ran "
                        "at the same speed and did not lower peak memory)")
    p.add_argument("--qairt-python", default="python3",
                   help="the Python that runs QAIRT's converter (a version QAIRT supports, with its requirements; "
                        "default: python3)")
    p.add_argument("--verify-ids", help="comma-separated token ids of one chunk: compare with the engine (needs "
                                        "onnxruntime) using --verify-voice")
    p.add_argument("--verify-voice", help="a voice .bin from the converter (voices/<language>/<voice>.bin)")
    a = p.parse_args(argv)

    arch_path = os.path.join(a.converted_dir, f"{a.model}_fp32_arch.json")
    wpath = os.path.join(a.converted_dir, f"{a.model}_fp32.bin")
    for f in (arch_path, wpath):
        if not os.path.isfile(f):
            p.error(f"{f} not found (run python -m kitten_inference.tts2_tiny.convert first)")
    chip = None
    if a.qairt:
        if a.chip:
            if a.chip.upper() not in CHIPS:
                p.error(f"unknown chip {a.chip}; give --dsp-arch and --soc-model")
            chip = CHIPS[a.chip.upper()]
        if a.dsp_arch or a.soc_model:
            if not (a.dsp_arch and a.soc_model):
                p.error("--dsp-arch and --soc-model go together")
            chip = (a.dsp_arch, a.soc_model)
        if chip is None:
            p.error("--qairt needs --chip (or --dsp-arch and --soc-model)")
        if int(chip[0].lstrip("vV")) < 73:
            p.error(f"Hexagon {chip[0]} has no fp16 arithmetic; the NPU path needs v73 or newer (Snapdragon 8 Gen 2+)")
    os.makedirs(a.out_dir, exist_ok=True)
    arch = json.load(open(arch_path))
    front, post, body_nodes = split_graph(arch)
    json.dump(front, open(os.path.join(a.out_dir, "front.json"), "w"), indent=1)
    json.dump(post, open(os.path.join(a.out_dir, "post.json"), "w"), indent=1)
    W = read_w1x1(wpath)
    hidden = int(W["asr_res.weight"].shape[1])
    frames = sorted(set(a.frames))
    for B in frames:
        if B < 8 or B > 1000:
            p.error("--frames must be from 8 to 1000")
        export_onnx(body_nodes, W, hidden, B, os.path.join(a.out_dir, f"body_{B}.onnx"))
        print(f"exported body_{B}.onnx ({B} frames = {B / 40:.1f} s)", flush=True)
    if a.verify_ids:
        if not a.verify_voice:
            p.error("--verify-ids needs --verify-voice")
        ids = [int(x) for x in a.verify_ids.split(",")]
        print(json.dumps(verify(a.converted_dir, a.model, a.out_dir, frames, ids, a.verify_voice), indent=1), flush=True)
    if not a.qairt:
        print("no --qairt: ONNX graphs only (nothing to load on a phone yet)")
        return
    work = os.path.join(a.out_dir, "work")
    os.makedirs(work, exist_ok=True)
    log = os.path.join(a.out_dir, "compile.log")
    t = time.time()
    files, sf = compile_qnn([os.path.join(a.out_dir, f"body_{B}.onnx") for B in frames],
                            os.path.join(a.out_dir, "body.bin"), a.qairt, chip, a.opt, work, log, a.qairt_python,
                            a.share_weights)
    graphs = [dict({"frames": B, "context": files[f"body_{B}"]}, **({"graph": f"body_{B}"} if a.share_weights else {}))
              for B in frames]
    print(f"compiled {', '.join(sorted(set(files.values())))} for {chip[0]} in {time.time() - t:.0f} s", flush=True)
    manifest = {"format_version": 1, "model": a.model, "precision": "fp16", "dsp_arch": chip[0], "soc_model": chip[1],
                "front": "front.json", "post": "post.json", "graphs": graphs, "spill_fill_bytes": sf}
    json.dump(manifest, open(os.path.join(a.out_dir, "npu.json"), "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", os.path.join(a.out_dir, "npu.json"))


if __name__ == "__main__":
    main()
