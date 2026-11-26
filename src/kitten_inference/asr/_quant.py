"""Triton kernels for locally converted and published packed ASR weights.

Precisions offered (the `precision` option of KittenASR / Engine / Core):

    bf16   the published weights as they are (default)
    fp16   the same weights in float16 (activations float16 too)
    int8   8-bit integer weights, 16-bit activations (W8A16); one scale per group of 128 inputs

Experimental kernels used by the numerical tests:

    int4   4-bit integer weights, 16-bit activations (W4A16); one scale and one offset per group of 128 inputs;
           the output head stays at 8 bits
    fp8    8-bit float weights and activations (W8A8, e4m3): one scale per output for the weights, one scale
           per row of activations computed on the fly; compute capability 8.9 or newer

On-load quantization changes the decoder linear layers and output head.
The audio encoder, the embeddings, the norms and the key/value cache stay in 16 bits. Weights are converted
when the model loads, on the GPU, from the bf16 tensors of the official files (round to nearest, no calibration
data): nothing is written to disk and nothing is downloaded beyond the official files.

Layout of a converted weight W [N outputs, K inputs] (y = x @ W.t()):

    int8   q [K, N] int8          s [K/128, N] fp16                 w = q * s
    int4   q [K/2, N] uint8       s, z [K/128, N] fp16              w = q * s + z,  q in 0..15
           in every block of 64 inputs, inputs 0..31 are the low nibbles and 32..63 the high nibbles
    fp8    q [K, N] float8_e4m3fn s [N] fp32                        w = q * s

Published INT4 has adjacent inputs in the high and low nibbles, with 128 inputs per scale and zero point.
Published INT8 has one scale and zero point per output. These kernels convert a tile inside matmul.
Packed published matrices cover the audio encoder, decoder, and output head. Embedding lookup converts
only the selected rows. The CUDA loader uses dense weights when Triton is unavailable.

The matrix product is one Triton kernel per layer (`_qmm_kernel`), registered as the PyTorch operator
kitten_asr::qmm so that torch.compile and CUDA graphs see it as one opaque call. Without Triton the same
operator dequantises the weight with plain PyTorch on every call (correct, slow).
"""
from __future__ import annotations

import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception:                 # no Triton: the PyTorch fallback below is used
    triton = tl = None
if sys.platform == "win32" and triton is not None:
    from . import _attn
    if not _attn.probe("triton")[0]:
        triton = tl = None

from ._precision import GPU as PRECISIONS, NOT_OFFERED_GPU as NOT_OFFERED  # noqa: E402  (the reasons, written once)
GROUP = 128                       # inputs per scale (int8, int4)
_PACK = 64                        # int4: inputs per packing block (low nibbles = first half)
FMT = {"int8": 1, "int4": 2, "fp8": 3}
PUBLISHED_FMT = {4: 4, 8: 5}
FP8_MAX = 448.0                   # largest finite float8_e4m3fn


def act_dtype(precision: str) -> torch.dtype:
    return torch.float16 if precision == "fp16" else torch.bfloat16


def check(precision, device) -> None:
    """ValueError for an unknown or not offered precision, RuntimeError when this GPU cannot run it."""
    if precision in NOT_OFFERED:
        raise ValueError("precision %r is not offered: %s. Use one of %s."
                         % (precision, NOT_OFFERED[precision], ", ".join(PRECISIONS)))
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of %s, got %r" % (", ".join(PRECISIONS), precision))
    if precision == "fp8":
        dev = torch.device(device)
        cap = torch.cuda.get_device_capability(dev)
        if cap < (8, 9):
            raise RuntimeError("precision 'fp8' needs an NVIDIA GPU with compute capability 8.9 or newer (Ada, "
                               "Hopper, Blackwell); %s has %d.%d. Use 'int8' for 8-bit weights on this GPU."
                               % (torch.cuda.get_device_name(dev), cap[0], cap[1]))
        if triton is None:
            raise RuntimeError("precision 'fp8' needs Triton (it comes with PyTorch on Linux)")


# ------------------------------------------------------------------ conversion (on the GPU, at load)
def quantize(w: torch.Tensor, precision: str):
    """W [N, K] (16-bit, on the GPU) -> (q, s, z, fmt) for `linear`. z is None except for int4."""
    N, K = w.shape
    if K % GROUP:
        raise ValueError("a weight with %d inputs cannot be grouped by %d" % (K, GROUP))
    wt = w.t().float()                                                   # [K, N]
    if precision == "int8":
        g = wt.view(K // GROUP, GROUP, N)
        s = (g.abs().amax(1) / 127.0).clamp_min(1e-10).half()           # [K/G, N]
        q = torch.round(g / s.float()[:, None]).clamp_(-127, 127).to(torch.int8).view(K, N)
        return q.contiguous(), s.contiguous(), None, FMT["int8"]
    if precision == "int4":
        g = wt.view(K // GROUP, GROUP, N)
        lo, hi = g.amin(1), g.amax(1)
        s = ((hi - lo) / 15.0).clamp_min(1e-10).half()
        z = lo.half()
        q = torch.round((g - z.float()[:, None]) / s.float()[:, None]).clamp_(0, 15).to(torch.uint8)
        q = q.view(K // _PACK, 2, _PACK // 2, N)                         # per 64-input block: two halves
        packed = q[:, 0] | (q[:, 1] << 4)                                # [K/64, 32, N]
        return packed.reshape(K // 2, N).contiguous(), s.contiguous(), z.contiguous(), FMT["int4"]
    if precision == "fp8":
        s = (wt.abs().amax(0) / FP8_MAX).clamp_min(1e-12)                # [N] fp32
        q = (wt / s[None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q.contiguous(), s.contiguous(), None, FMT["fp8"]
    raise ValueError(precision)


def dequantize(q, s, z, fmt: int, dtype) -> torch.Tensor:
    """The weight W [N, K] a converted tensor stands for (tests and the fallback)."""
    if fmt == FMT["int8"]:
        K, N = q.shape
        w = q.float().view(K // GROUP, GROUP, N) * s.float()[:, None]
    elif fmt == FMT["int4"]:
        K2, N = q.shape
        K = 2 * K2
        p = q.view(K // _PACK, 1, _PACK // 2, N)
        u = torch.cat([p & 15, p >> 4], 1).float().view(K // GROUP, GROUP, N)
        w = u * s.float()[:, None] + z.float()[:, None]
    elif fmt == FMT["fp8"]:
        K, N = q.shape
        w = q.float() * s[None]
    elif fmt in PUBLISHED_FMT.values():
        return _published_values(q, s, z, fmt, dtype).t()
    else:
        raise ValueError(fmt)
    return w.reshape(K, N).t().to(dtype)


def _published_values(q, scale, zero, fmt, dtype):
    if fmt == PUBLISHED_FMT[4]:
        q = torch.stack((q >> 4, q & 15), dim=1).flatten(0, 1)
    group = q.shape[0] // scale.shape[0]
    values = q.reshape(scale.shape[0], group, q.shape[1]).float()
    return ((values - zero.float()[:, None]) * scale.float()[:, None]).reshape(q.shape).to(dtype)


def embedding(weight, ids, dtype):
    if isinstance(weight, torch.Tensor):
        return weight[ids]
    q, scale, zero, fmt = weight
    flat = ids.flatten()
    rows = _published_values(q.index_select(1, flat), scale.index_select(1, flat),
                             zero.index_select(1, flat), fmt, dtype)
    return rows.t().reshape(*ids.shape, rows.shape[0])


def linear_bias(x, weight, bias):
    if isinstance(weight, torch.Tensor):
        return torch.nn.functional.linear(x, weight, bias)
    y = mm(x.flatten(0, -2), weight).reshape(*x.shape[:-1], weight[0].shape[1])
    return y if bias is None else y + bias


# ------------------------------------------------------------------ the kernel
# Tiling follows the Triton tutorial 03-matrix-multiplication.py (https://github.com/triton-lang/triton, MIT):
# grouped program order, tl.dot with float32 accumulation. The weight is loaded in its stored format and
# converted inside the K loop; the per-group scale multiplies the partial product of each group (exact: the
# integer codes are whole numbers, which bf16 and fp16 hold exactly up to 256).
if triton is not None:

    @triton.jit
    def _qmm_kernel(A, Q, S, Z, C, M, N, K,
                    s_am, s_ak, s_qk, s_qn, s_sk, s_cm, s_cs,
                    FMT: tl.constexpr, G: tl.constexpr, SPLIT: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        # SPLIT > 1: program (tile, j) sums inputs j*K/SPLIT .. (j+1)*K/SPLIT and writes float32 partial sums
        # to C[j] ([SPLIT, M, N]); the caller adds them up (inside the next fused kernel when compiled).
        pid = tl.program_id(0)
        pk = tl.program_id(1)
        KS = K // SPLIT
        k_lo = pk * KS
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        per_group = GROUP_M * num_n
        first_m = (pid // per_group) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pm = first_m + (pid % per_group) % size_m
        pn = (pid % per_group) // size_m
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mmask = rm < M
        nmask = rn < N
        a_row = A + rm.to(tl.int64)[:, None] * s_am
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if FMT == 3:
            # fp8 W8A8: one scale per activation row, from the row's largest magnitude (a first pass over K)
            amax = tl.zeros((BM,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(a_row + rk[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                amax = tl.maximum(amax, tl.max(tl.abs(a.to(tl.float32)), 1))
            a_scale = tl.maximum(amax, 1e-12) / 448.0
            a_inv = 1.0 / a_scale
            for k0 in range(k_lo, k_lo + KS, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(a_row + rk[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                a8 = (a.to(tl.float32) * a_inv[:, None]).to(tl.float8e4nv)
                w = tl.load(Q + rk[:, None] * s_qk + rn[None, :], mask=nmask[None, :], other=0.0)
                acc += tl.dot(a8, w)
            ws = tl.load(S + rn, mask=nmask, other=0.0)
            acc = acc * a_scale[:, None] * ws[None, :]
        elif FMT == 4 or FMT == 5:
            for k0 in range(k_lo, k_lo + KS, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(a_row + rk[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                packed_k = rk // 2 if FMT == 4 else rk
                q = tl.load(Q + packed_k[:, None] * s_qk + rn[None, :], mask=nmask[None, :], other=0)
                if FMT == 4:
                    q = tl.where((rk % 2)[:, None] == 0, q >> 4, q & 15)
                g = k0 // G
                scale = tl.load(S + g * s_sk + rn, mask=nmask, other=0.0).to(tl.float32)
                zero = tl.load(Z + g * s_sk + rn, mask=nmask, other=0.0).to(tl.float32)
                w = ((q.to(tl.float32) - zero[None, :]) * scale[None, :]).to(a.dtype)
                acc = tl.dot(a, w, acc)
        elif FMT == 2:
            HK: tl.constexpr = BK // 2
            rh = tl.arange(0, HK)
            for k0 in range(k_lo, k_lo + KS, BK):
                a_lo = tl.load(a_row + (k0 + rh)[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                a_hi = tl.load(a_row + (k0 + HK + rh)[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                p = tl.load(Q + (k0 // 2 + rh)[:, None] * s_qk + rn[None, :], mask=nmask[None, :], other=0)
                lo = (p & 15).to(a_lo.dtype)
                hi = (p >> 4).to(a_lo.dtype)
                part = tl.dot(a_lo, lo) + tl.dot(a_hi, hi)
                g = k0 // G
                s = tl.load(S + g * s_sk + rn, mask=nmask, other=0.0).to(tl.float32)
                z = tl.load(Z + g * s_sk + rn, mask=nmask, other=0.0).to(tl.float32)
                a_sum = tl.sum(a_lo.to(tl.float32), 1) + tl.sum(a_hi.to(tl.float32), 1)
                acc += part * s[None, :] + a_sum[:, None] * z[None, :]
        else:
            for k0 in range(k_lo, k_lo + KS, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(a_row + rk[None, :] * s_ak, mask=mmask[:, None], other=0.0)
                q = tl.load(Q + rk[:, None] * s_qk + rn[None, :], mask=nmask[None, :], other=0).to(a.dtype)
                s = tl.load(S + (k0 // G) * s_sk + rn, mask=nmask, other=0.0).to(tl.float32)
                acc += tl.dot(a, q) * s[None, :]
        c = acc.to(C.dtype.element_ty)
        tl.store(C + pk * s_cs + rm.to(tl.int64)[:, None] * s_cm + rn[None, :], c, mask=mmask[:, None] & nmask[None, :])


# Launch settings: (BM, BN, BK, warps, stages, SPLIT) by format, by rows M (decode steps have M <= 16) and by layer
# shape (N, K). TUNED holds the settings measured best on an RTX 4090 for the decoder shapes of the models;
# other shapes use the defaults of their row bucket.
DEFAULT = {16: (16, 64, 64, 4, 2, 4), 64: (16, 64, 128, 4, 3, 2), 0: (64, 256, 64, 8, 3, 1)}
# (format, row bucket, N, K): settings. Bucket 16 was tuned at M = 1 (one stream), 64 at M = 64, 0 at M = 512.
TUNED = {
    (1, 0, 512, 2048): (64, 64, 64, 8, 3, 1),
    (1, 0, 512, 3072): (64, 64, 64, 8, 3, 1),
    (1, 0, 1024, 2048): (64, 64, 64, 8, 3, 1),
    (1, 0, 1024, 3072): (64, 64, 64, 8, 3, 1),
    (1, 0, 2048, 2048): (64, 64, 64, 8, 3, 1),
    (1, 0, 2048, 6144): (128, 64, 64, 8, 3, 1),
    (1, 0, 4096, 512): (64, 256, 64, 8, 3, 1),
    (1, 0, 4096, 1024): (64, 256, 64, 8, 3, 1),
    (1, 0, 4096, 2048): (64, 256, 64, 8, 3, 1),
    (1, 0, 6144, 512): (64, 64, 64, 8, 3, 1),
    (1, 0, 6144, 1024): (64, 64, 64, 8, 3, 1),
    (1, 0, 12288, 2048): (64, 256, 64, 8, 3, 1),
    (1, 0, 151936, 512): (64, 256, 64, 8, 3, 1),
    (1, 0, 151936, 1024): (64, 256, 64, 8, 3, 1),
    (1, 0, 151936, 2048): (64, 256, 64, 8, 3, 1),
    (1, 16, 512, 2048): (16, 32, 64, 4, 4, 8),
    (1, 16, 512, 3072): (16, 32, 64, 4, 4, 8),
    (1, 16, 1024, 2048): (16, 32, 128, 4, 2, 8),
    (1, 16, 1024, 3072): (16, 32, 128, 4, 2, 8),
    (1, 16, 2048, 2048): (16, 32, 64, 4, 2, 8),
    (1, 16, 2048, 6144): (16, 32, 64, 4, 2, 8),
    (1, 16, 4096, 512): (16, 32, 64, 4, 4, 1),
    (1, 16, 4096, 1024): (16, 32, 128, 4, 4, 1),
    (1, 16, 4096, 2048): (16, 32, 128, 4, 4, 1),
    (1, 16, 6144, 512): (16, 32, 64, 4, 4, 1),
    (1, 16, 6144, 1024): (16, 32, 64, 4, 4, 1),
    (1, 16, 12288, 2048): (16, 64, 64, 4, 4, 1),
    (1, 16, 151936, 512): (16, 64, 64, 4, 2, 4),
    (1, 16, 151936, 1024): (16, 64, 64, 4, 2, 8),
    (1, 16, 151936, 2048): (16, 64, 128, 4, 4, 8),
    (1, 64, 512, 2048): (16, 64, 128, 4, 3, 2),
    (1, 64, 512, 3072): (16, 64, 128, 4, 3, 2),
    (1, 64, 1024, 2048): (16, 64, 128, 4, 3, 2),
    (1, 64, 1024, 3072): (16, 64, 128, 4, 3, 2),
    (1, 64, 2048, 2048): (16, 64, 64, 4, 3, 2),
    (1, 64, 2048, 6144): (16, 64, 128, 4, 3, 2),
    (1, 64, 4096, 512): (16, 64, 64, 4, 3, 1),
    (1, 64, 4096, 1024): (16, 64, 64, 4, 3, 1),
    (1, 64, 4096, 2048): (16, 64, 64, 4, 3, 1),
    (1, 64, 6144, 512): (64, 64, 128, 4, 3, 1),
    (1, 64, 6144, 1024): (64, 64, 128, 4, 3, 1),
    (1, 64, 12288, 2048): (64, 128, 64, 4, 3, 2),
    (1, 64, 151936, 512): (64, 128, 64, 4, 3, 1),
    (1, 64, 151936, 1024): (64, 128, 64, 4, 3, 1),
    (1, 64, 151936, 2048): (64, 128, 64, 4, 3, 1),
    (2, 0, 512, 2048): (64, 128, 64, 8, 3, 1),
    (2, 0, 512, 3072): (64, 128, 64, 8, 3, 1),
    (2, 0, 1024, 2048): (64, 128, 64, 8, 3, 1),
    (2, 0, 1024, 3072): (64, 128, 64, 8, 3, 1),
    (2, 0, 2048, 2048): (64, 128, 64, 8, 3, 1),
    (2, 0, 2048, 6144): (64, 128, 64, 8, 3, 1),
    (2, 0, 4096, 512): (64, 256, 64, 8, 3, 1),
    (2, 0, 4096, 1024): (64, 256, 64, 8, 3, 1),
    (2, 0, 4096, 2048): (64, 256, 64, 8, 3, 1),
    (2, 0, 6144, 512): (64, 128, 64, 8, 3, 1),
    (2, 0, 6144, 1024): (64, 128, 64, 8, 3, 1),
    (2, 0, 12288, 2048): (64, 256, 64, 8, 3, 1),
    (2, 0, 151936, 512): (64, 256, 64, 8, 3, 1),
    (2, 0, 151936, 1024): (64, 256, 64, 8, 3, 1),
    (2, 0, 151936, 2048): (64, 256, 64, 8, 3, 1),
    (2, 16, 512, 2048): (16, 32, 64, 4, 4, 8),
    (2, 16, 512, 3072): (16, 32, 64, 4, 4, 8),
    (2, 16, 1024, 2048): (16, 64, 64, 4, 4, 8),
    (2, 16, 1024, 3072): (16, 64, 64, 4, 4, 8),
    (2, 16, 2048, 2048): (16, 64, 64, 4, 4, 8),
    (2, 16, 2048, 6144): (16, 64, 64, 4, 4, 8),
    (2, 16, 4096, 512): (16, 64, 64, 4, 2, 4),
    (2, 16, 4096, 1024): (16, 64, 64, 4, 2, 8),
    (2, 16, 4096, 2048): (16, 64, 64, 4, 2, 8),
    (2, 16, 6144, 512): (16, 64, 64, 4, 2, 4),
    (2, 16, 6144, 1024): (16, 64, 64, 4, 2, 8),
    (2, 16, 12288, 2048): (16, 64, 64, 4, 2, 4),
    (2, 16, 151936, 512): (16, 64, 64, 4, 2, 1),
    (2, 16, 151936, 1024): (16, 64, 64, 4, 2, 4),
    (2, 16, 151936, 2048): (16, 64, 64, 4, 2, 4),
    (2, 64, 512, 2048): (16, 64, 64, 4, 3, 2),
    (2, 64, 512, 3072): (16, 64, 64, 4, 3, 2),
    (2, 64, 1024, 2048): (16, 64, 64, 4, 3, 2),
    (2, 64, 1024, 3072): (16, 64, 64, 4, 3, 2),
    (2, 64, 2048, 2048): (16, 64, 64, 4, 3, 2),
    (2, 64, 2048, 6144): (16, 64, 64, 4, 3, 2),
    (2, 64, 4096, 512): (16, 64, 64, 4, 3, 1),
    (2, 64, 4096, 1024): (16, 64, 64, 4, 3, 1),
    (2, 64, 4096, 2048): (64, 64, 64, 4, 3, 2),
    (2, 64, 6144, 512): (16, 64, 64, 4, 3, 1),
    (2, 64, 6144, 1024): (16, 64, 64, 4, 3, 1),
    (2, 64, 12288, 2048): (64, 64, 64, 4, 3, 2),
    (2, 64, 151936, 512): (64, 128, 64, 4, 3, 1),
    (2, 64, 151936, 1024): (64, 128, 64, 4, 3, 1),
    (2, 64, 151936, 2048): (64, 128, 64, 4, 3, 1),
    (3, 0, 512, 2048): (64, 64, 64, 8, 3, 1),
    (3, 0, 512, 3072): (64, 64, 64, 8, 3, 1),
    (3, 0, 1024, 2048): (64, 64, 64, 8, 3, 1),
    (3, 0, 1024, 3072): (64, 64, 64, 8, 3, 1),
    (3, 0, 2048, 2048): (64, 128, 64, 8, 3, 1),
    (3, 0, 2048, 6144): (64, 128, 64, 8, 3, 1),
    (3, 0, 4096, 512): (64, 128, 64, 8, 3, 1),
    (3, 0, 4096, 1024): (64, 128, 64, 8, 3, 1),
    (3, 0, 4096, 2048): (64, 128, 64, 8, 3, 1),
    (3, 0, 6144, 512): (128, 256, 64, 8, 3, 1),
    (3, 0, 6144, 1024): (128, 256, 64, 8, 3, 1),
    (3, 0, 12288, 2048): (128, 128, 64, 8, 3, 1),
    (3, 0, 151936, 512): (128, 256, 64, 8, 3, 1),
    (3, 0, 151936, 1024): (128, 256, 64, 8, 3, 1),
    (3, 0, 151936, 2048): (128, 256, 64, 8, 3, 1),
    (3, 16, 512, 2048): (16, 32, 128, 4, 2, 8),
    (3, 16, 512, 3072): (16, 32, 128, 4, 2, 8),
    (3, 16, 1024, 2048): (16, 64, 128, 4, 4, 8),
    (3, 16, 1024, 3072): (16, 64, 128, 4, 2, 8),
    (3, 16, 2048, 2048): (16, 64, 128, 4, 2, 4),
    (3, 16, 2048, 6144): (16, 64, 128, 4, 2, 4),
    (3, 16, 4096, 512): (16, 32, 128, 4, 4, 1),
    (3, 16, 4096, 1024): (16, 32, 128, 4, 4, 1),
    (3, 16, 4096, 2048): (16, 32, 128, 4, 4, 1),
    (3, 16, 6144, 512): (16, 64, 128, 4, 4, 1),
    (3, 16, 6144, 1024): (16, 64, 128, 4, 4, 1),
    (3, 16, 12288, 2048): (16, 64, 128, 4, 2, 1),
    (3, 16, 151936, 512): (16, 64, 128, 4, 2, 4),
    (3, 16, 151936, 1024): (16, 64, 128, 4, 2, 8),
    (3, 16, 151936, 2048): (16, 64, 128, 4, 4, 4),
    (3, 64, 512, 2048): (16, 64, 128, 4, 3, 2),
    (3, 64, 512, 3072): (16, 64, 128, 4, 3, 2),
    (3, 64, 1024, 2048): (16, 64, 128, 4, 3, 2),
    (3, 64, 1024, 3072): (16, 64, 128, 4, 3, 2),
    (3, 64, 2048, 2048): (16, 64, 128, 4, 3, 1),
    (3, 64, 2048, 6144): (16, 128, 128, 4, 3, 2),
    (3, 64, 4096, 512): (16, 128, 128, 4, 3, 1),
    (3, 64, 4096, 1024): (16, 128, 128, 4, 3, 1),
    (3, 64, 4096, 2048): (16, 128, 128, 4, 3, 1),
    (3, 64, 6144, 512): (64, 64, 128, 4, 3, 1),
    (3, 64, 6144, 1024): (64, 64, 128, 4, 3, 1),
    (3, 64, 12288, 2048): (64, 128, 64, 4, 3, 1),
    (3, 64, 151936, 512): (64, 128, 64, 4, 3, 1),
    (3, 64, 151936, 1024): (64, 128, 128, 4, 3, 1),
    (3, 64, 151936, 2048): (64, 128, 128, 4, 3, 1),
}


def _bucket(M: int) -> int:
    return 16 if M <= 16 else 64 if M <= 64 else 0


def config(fmt: int, M: int, N: int, K: int):
    BM, BN, BK, warps, stages, split = TUNED.get(({4: 2, 5: 1}.get(fmt, fmt), _bucket(M), N, K)) or DEFAULT[_bucket(M)]
    if fmt == FMT["int4"]:
        BK = _PACK                    # the kernel unpacks one 64-input block per step
    elif fmt == FMT["int8"]:
        BK = min(BK, GROUP)           # one scale per step
    elif fmt == PUBLISHED_FMT[8] and K % BK:
        BK = 64                       # published INT8 also accepts input widths divisible by only 64
    while split > 1 and (K % split or (K // split) % BK):
        split //= 2
    return BM, BN, BK, warps, stages, split


def _launch(x, q, s, z, fmt: int, cfg=None):
    M, K = x.shape
    N = q.shape[1]
    BM, BN, BK, warps, stages, split = cfg or config(fmt, M, N, K)
    if split > 1:
        c = torch.empty(split, M, N, device=x.device, dtype=torch.float32)
    else:
        c = torch.empty(M, N, device=x.device, dtype=x.dtype)
    if M == 0:
        return c
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), split)
    _qmm_kernel[grid](x, q, s, z if z is not None else s, c, M, N, K,
                      x.stride(0), x.stride(1), q.stride(0), q.stride(1), s.stride(0), N, M * N,
                      FMT=fmt, G=K if fmt == PUBLISHED_FMT[8] else GROUP, SPLIT=split, BM=BM, BN=BN, BK=BK, GROUP_M=8,
                      num_warps=warps, num_stages=stages)
    return c


def splits(fmt: int, M: int, N: int, K: int) -> int:
    """How many float32 partial sums the kernel writes for this call (1: the [M, N] result itself)."""
    if triton is None:
        return 1
    return config(fmt, M, N, K)[-1]


@torch.library.custom_op("kitten_asr::qmm", mutates_args=())
def qmm(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, fmt: int, split: int) -> torch.Tensor:
    """x [M, K] @ W.t() for a converted weight (q, s, z, fmt); z is an empty tensor when unused. With split > 1
    the result is [split, M, N] float32 partial sums (their sum is the product)."""
    x = x.contiguous()
    zz = z if z.numel() else None
    M, K = x.shape
    N = q.shape[1]
    if triton is None or not x.is_cuda:
        y = x @ dequantize(q, s, zz, fmt, x.dtype).t()
        return y if split == 1 else torch.cat([y.float()[None], y.new_zeros(split - 1, M, N, dtype=torch.float32)])
    cfg = list(config(fmt, M, N, K))
    cfg[-1] = split
    return _launch(x, q, s, zz, fmt, tuple(cfg))


@qmm.register_fake
def _(x, q, s, z, fmt, split):
    if split > 1:
        return x.new_empty(split, x.shape[0], q.shape[1], dtype=torch.float32)
    return x.new_empty(x.shape[0], q.shape[1])


class Linear:
    """One linear layer of the decoder: a 16-bit weight [N, K], or a converted one."""
    __slots__ = ("w", "q", "s", "z", "fmt", "N", "K")

    def __init__(self, w, precision="bf16"):
        if isinstance(w, tuple):
            self.q, self.s, self.z, self.fmt = w
            self.K, self.N = self.q.shape
            if self.fmt == PUBLISHED_FMT[4]:
                self.K *= 2
            self.w = None
            return
        self.N, self.K = w.shape
        if precision in FMT:
            self.q, self.s, z, self.fmt = quantize(w, precision)
            self.z = z if z is not None else torch.empty(0, device=w.device, dtype=torch.float16)
            self.w = None
        else:
            self.w, self.q, self.s, self.z, self.fmt = w, None, None, None, 0

    def args(self):
        """What the compiled functions take for this layer: the 16-bit weight tensor itself, or (q, s, z, fmt).
        A 16-bit layer is passed as a bare tensor so that torch.compile builds exactly the graph it built before
        lower precisions existed: decode steps and prefill passes then keep giving the same numbers for the
        same row, which draft verification of live text relies on."""
        return self.w if self.fmt == 0 else (self.q, self.s, self.z, self.fmt)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.q, self.s, self.z) if t is not None)


def mm(x, w):
    """x [M, K] @ W.t() for a layer given as Linear.args(): a tensor, or (q, s, z, fmt) with fmt a constant."""
    if isinstance(w, torch.Tensor):
        return x @ w.t()
    q, s, z, fmt = w
    N, K = q.shape[1], x.shape[1]
    M = x.shape[0]                  # symbolic under torch.compile: each comparison below becomes a guard
    rows = 16 if M <= 16 else 64 if M <= 64 else 1 << 20
    split = splits(fmt, rows, N, K)                    # decode steps: float32 partial sums over K
    if split > 1:
        return torch.ops.kitten_asr.qmm(x, q, s, z, fmt, split).sum(0).to(x.dtype)
    return torch.ops.kitten_asr.qmm(x, q, s, z, fmt, 1)
