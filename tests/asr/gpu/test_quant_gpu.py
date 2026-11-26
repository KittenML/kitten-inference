"""The low-precision layers on an NVIDIA GPU, without a model: conversion error, and the Triton kernel against the
same arithmetic done with plain PyTorch, for the row counts of one decode step, a batch of streams and a prefill."""
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA GPU", allow_module_level=True)

from kitten_inference.asr import _core, _quant  # noqa: E402

DEV = "cuda"
SHAPES = [(4096, 512), (512, 2048), (2048, 6144), (12288, 2048)]       # (outputs N, inputs K)
ROWS = [1, 3, 16, 17, 64, 100, 700]
FP8 = torch.cuda.get_device_capability() >= (8, 9)


def _weight(n, k, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, device=DEV, generator=g) * 0.02
    w[:, 5] *= 20                                       # one large input channel, as real layers have
    return w.to(torch.bfloat16)


@pytest.mark.parametrize("precision, bound", [("int8", 0.5 / 127), ("int4", 0.5 / 15)])
def test_conversion_error_is_at_most_half_a_step(precision, bound):
    w = _weight(256, 512)
    q, s, z, fmt = _quant.quantize(w, precision)
    back = _quant.dequantize(q, s, z, fmt, torch.float32)
    g = w.float().t().reshape(512 // _quant.GROUP, _quant.GROUP, 256)
    span = g.abs().amax(1) * 2 if precision == "int8" else g.amax(1) - g.amin(1)
    err = (back.t() - w.float().t()).reshape_as(g).abs().amax(1)
    assert torch.all(err <= span * bound * 1.01 + 1e-6)


def test_fp8_conversion_keeps_the_scale():
    if not FP8:
        pytest.skip("fp8 needs compute capability 8.9")
    w = _weight(256, 512)
    q, s, z, fmt = _quant.quantize(w, "fp8")
    back = _quant.dequantize(q, s, z, fmt, torch.float32)
    assert z is None and s.shape == (256,)
    rel = (back - w.float()).abs().amax() / w.float().abs().amax()
    assert rel < 0.07                                   # e4m3 has 3 mantissa bits: 1/16 relative step


def _reference(x, q, s, z, fmt):
    if fmt == _quant.FMT["fp8"]:
        xs = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / _quant.FP8_MAX
        x8 = (x.float() / xs).to(torch.float8_e4m3fn).float()
        return ((x8 @ q.float()) * xs * s[None]).to(x.dtype)
    return (x.float() @ _quant.dequantize(q, s, z, fmt, torch.float32).t()).to(x.dtype)


@pytest.mark.parametrize("precision", ["int8", "int4", "fp8"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kernel_matches_plain_pytorch(precision, shape, dtype):
    if precision == "fp8" and not FP8:
        pytest.skip("fp8 needs compute capability 8.9")
    n, k = shape
    q, s, z, fmt = _quant.quantize(_weight(n, k), precision)
    zz = z if z is not None else torch.empty(0, device=DEV, dtype=torch.float16)
    for m in ROWS:
        x = torch.randn(m, k, device=DEV).to(dtype)
        want = _reference(x, q, s, z, fmt).float()
        scale = want.abs().amax().item()
        for split in (1, 2, 4) if m <= 16 else (1,):
            got = torch.ops.kitten_asr.qmm(x, q, s, zz, fmt, split)
            got = (got.sum(0) if split > 1 else got).float()
            err = (got - want).abs().amax().item()
            assert err <= 0.02 * scale + 1e-3, (precision, shape, m, split, err, scale)


@pytest.mark.parametrize("bits, rows, k", [(4, 1, 896), (4, 73, 896), (8, 1, 896), (8, 73, 896), (8, 33, 192)])
def test_published_affine_kernel_matches_expanded_weights(bits, rows, k):
    if _quant.triton is None:
        pytest.skip("published packed CUDA weights require Triton")
    n = 131
    group = 128 if bits == 4 else k
    rng = torch.Generator(device=DEV).manual_seed(48)
    q = torch.randint(1 << bits, (k, n), dtype=torch.uint8, device=DEV, generator=rng)
    scale = (torch.rand(k // group, n, device=DEV, generator=rng) / 32).half()
    zero = (torch.rand(k // group, n, device=DEV, generator=rng) * (1 << bits)).half()
    dense = ((q.float() - zero.repeat_interleave(group, dim=0).float()) *
             scale.repeat_interleave(group, dim=0).float()).t().to(torch.bfloat16).contiguous()
    packed = ((q[::2] << 4) | q[1::2]).contiguous() if bits == 4 else q
    weight = packed, scale, zero, _quant.PUBLISHED_FMT[bits]
    x = torch.randn(rows, k, device=DEV, dtype=torch.bfloat16, generator=rng)
    want = x @ dense.t()
    got = _quant.mm(x, weight)
    assert (got.float() - want.float()).abs().max() <= 0.009 * want.abs().max()
    ids = torch.tensor([[0, 127], [130, 5]], device=DEV)
    assert torch.equal(_quant.embedding(weight, ids, x.dtype), dense[ids])


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
def test_compiled_and_graphed_layer(precision):
    """The operator inside torch.compile(fullgraph) and a CUDA graph gives the eager result."""
    lin = _quant.Linear(_weight(1024, 512), precision)
    args = lin.args()
    f = torch.compile(lambda x: _quant.mm(x, args) * 2, fullgraph=True, dynamic=True,
                      options=_core._compile_options())
    x = torch.randn(4, 512, device=DEV, dtype=torch.bfloat16)
    want = _quant.mm(x, args) * 2
    assert torch.equal(f(x), want)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = f(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    assert torch.equal(out, _quant.mm(x, args) * 2)


def test_only_measured_precisions_are_offered():
    for p in ("bf16", "fp16", "int8", "int4"):
        _quant.check(p, DEV)
    with pytest.raises(ValueError, match="not offered"):
        _quant.check("fp8", DEV)
    with pytest.raises(ValueError, match="precision"):
        _quant.check("int2", DEV)


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_GPU_MODEL"), reason="KITTEN_ASR_GPU_MODEL is not set")
@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_model_at_each_precision_transcribes(precision):
    from kitten_inference.asr import KittenASR
    wav, text = os.environ.get("KITTEN_ASR_GPU_WAV"), os.environ.get("KITTEN_ASR_GPU_TEXT")
    if not (wav and text):
        pytest.skip("KITTEN_ASR_GPU_WAV and KITTEN_ASR_GPU_TEXT are not both set")
    with KittenASR(os.environ["KITTEN_ASR_GPU_MODEL"], backend="cuda", precision=precision,
                   cache_dir=os.environ.get("KITTEN_ASR_GPU_CACHE") or None) as asr:
        assert asr.capabilities["precision"] == precision
        from .test_engine_gpu import check_words, words
        check_words(asr.transcribe(wav), words(text))      # the same word error limit as the bf16 tests
