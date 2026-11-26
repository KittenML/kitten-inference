"""The CPU engine (backend="cpu"): the numpy GGUF converter, the q8_0 encoder format, CPU selection, the backend
switch, and (when the engine library and a public model folder are present) one transcription end to end.

The end-to-end test needs KITTEN_ASR_TEST_MODEL=<local folder of KittenML/kitten-asr-tiny>; without it it skips.
"""
import json
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kitten_inference.asr import _cpu, _deps, _gguf, _published


def _bf16_bytes(x):
    return _gguf._f32_to_bf16(np.asarray(x, np.float32)).tobytes()


def _write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    stored = {"F32": np.float32, "F16": np.float16, "U8": np.uint8}
    for name, (dtype, arr) in tensors.items():
        raw = _bf16_bytes(arr) if dtype == "BF16" else np.asarray(arr, stored[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(np.shape(arr)), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@pytest.mark.parametrize("bits,group,packed,scale,zero,want", [
    (4, 5, [[0x6c, 0xb2, 0xf0]], [[0.25]], [[8]], [-0.5, 1, 0.75, -1.5, 1.75]),
    (8, 5, [[30, 40, 20, 10, 0]], [[0.125]], [[30]], [0, 1.25, -1.25, -2.5, -3.75]),
])
def test_published_checkpoint_expands_asymmetric_weights(tmp_path, bits, group, packed, scale, zero, want):
    precision = f"int{bits}"
    folder = tmp_path / precision
    folder.mkdir()
    name, shape = "conv.weight", (1, 1, 1, 5)
    metadata = {"tensors": {name: {"bits": bits, "group": group, "cols": 5, "shape": shape}}}
    (folder / _published.META).write_text(json.dumps(metadata))
    _write_safetensors(folder / _published.WEIGHTS, {
        name + ".q": ("U8", packed),
        name + ".scale": ("F16", scale),
        name + ".zero": ("F16", zero),
        "bias": ("BF16", [1, 2]),
    })
    weights = _gguf.SafeTensors(str(tmp_path), precision)
    assert weights.names() == {name, "bias"}
    assert weights.shape(name) == shape
    np.testing.assert_array_equal(weights.f32(name), np.asarray(want).reshape(shape))
    np.testing.assert_array_equal(weights.f32("bias"), [1, 2])


def test_q8_0_matches_ggml_layout():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    b = _gguf.quantize_q8_0(x)
    assert len(b) == 4 * 2 * 34                       # 8 blocks of (fp16 scale + 32 int8)
    blocks = np.frombuffer(b, np.uint8).reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    back = (q * d[:, None]).reshape(x.shape)
    assert np.max(np.abs(back - x)) <= np.max(np.abs(x)) / 127 * 0.51 + 1e-3
    assert np.all(np.abs(q).max(axis=1) == 127)       # the largest value of every block maps to +-127


def test_q8_0_zero_block():
    b = _gguf.quantize_q8_0(np.zeros(32, np.float32))
    assert b == bytes(34)


def test_bf16_round_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5, np.inf], np.float32)   # 1 + 2^-8 ties to even (down)
    bits = _gguf._f32_to_bf16(x)
    back = (bits.astype(np.uint32) << 16).view(np.float32)
    assert back[0] == 1.0 and back[1] == 1.0 and back[2] == 1.015625 and back[3] == -2.5 and np.isinf(back[4])


@pytest.mark.parametrize("bits", [4, 8])
def test_published_reblocking_preserves_codes(tmp_path, bits):
    name = "a.weight"
    folder = tmp_path / f"int{bits}"
    folder.mkdir()
    codes = (np.arange(64).reshape(1, 64) % (1 << bits)).astype(np.uint8)
    raw = (codes[:, ::2] << 4) | codes[:, 1::2] if bits == 4 else codes
    scales = np.array([[0.25, 0.125]], dtype=np.float16)
    zeros = np.array([[7, 11] if bits == 4 else [61, 101]], dtype=np.float16)
    info = {"tensors": {name: {"shape": [1, 64], "bits": bits, "cols": 64, "group": 32}}}
    (folder / _published.META).write_text(json.dumps(info))
    _write_safetensors(folder / _published.WEIGHTS, {
        name + ".q": ("U8", raw), name + ".scale": ("F16", scales), name + ".zero": ("F16", zeros),
    })
    tensors = _gguf.SafeTensors(str(tmp_path), f"int{bits}")
    kind = tensors.ggml_packed_type(name)
    assert kind == (_gguf.Q4_1 if bits == 4 else _gguf.KASR_Q8)
    blocks = np.frombuffer(tensors.ggml_packed(name, kind), dtype=np.uint8).reshape(2, -1)
    params = blocks[:, :4].copy().view("<f2")
    assert np.array_equal(params[:, 0], scales[0])
    if bits == 4:
        assert np.array_equal(params[:, 1], (-scales * zeros)[0])
        decoded = np.concatenate((blocks[:, 4:] & 15, blocks[:, 4:] >> 4), axis=1)
    else:
        assert np.array_equal(params[:, 1], zeros[0])
        decoded = blocks[:, 4:] ^ 128
    assert np.array_equal(decoded.reshape(1, 64), codes)



def test_safetensors_reader_and_gguf_writer(tmp_path):
    w = np.arange(64, dtype=np.float32).reshape(2, 32) / 7
    _write_safetensors(tmp_path / "model.safetensors", {"a.weight": ("BF16", w), "b.bias": ("F32", np.ones(3))})
    st = _gguf.SafeTensors(str(tmp_path))
    assert st.items["a.weight"][1] == (2, 32)
    assert np.allclose(st.f32("a.weight"), w, rtol=1e-2)
    assert st.bf16_bits("a.weight").dtype == np.uint16
    out = tmp_path / "x.gguf"
    wr = _gguf._Writer()
    wr.add("general.architecture", _gguf.STR, "clip")
    wr.add("k.u32", _gguf.U32, 7)
    wr.add("k.arr", _gguf.ARR, [1, 2, 3], _gguf.I32)
    wr.add("k.strs", _gguf.ARR, ["a", "bc"], _gguf.STR)
    wr.tensor("a.weight", _gguf.BF16, (2, 32), lambda: st.bf16_bits("a.weight").tobytes())
    wr.tensor("a.q8", _gguf.Q8_0, (2, 32), lambda: _gguf.quantize_q8_0(st.f32("a.weight")))
    wr.tensor("b.bias", _gguf.F32, (3,), lambda: st.f32("b.bias").tobytes())
    wr.write(str(out))
    d = _gguf.describe(str(out))
    assert d["kv"]["k.u32"] == (_gguf.U32, 7)
    assert d["kv"]["k.arr"][1]["items"] == [1, 2, 3]
    assert d["kv"]["k.strs"][1]["items"] == ["a", "bc"]
    names = {t[0]: t for t in d["tensors"]}
    assert names["a.weight"][1] == _gguf.BF16 and names["a.weight"][2] == [32, 2]      # GGUF: innermost first
    assert names["a.q8"][1] == _gguf.Q8_0
    assert all(t[3] % 32 == 0 for t in d["tensors"])                                   # aligned offsets
    with open(out, "rb") as f:
        f.seek(d["data_start"] + names["b.bias"][3])
        assert np.array_equal(np.frombuffer(f.read(12), np.float32), np.ones(3, np.float32))
    assert not (tmp_path / "x.gguf.part").exists()


def test_vocab_types(tmp_path):
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
           "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": True},
                            {"id": 4, "content": "<asr_text>", "special": False}]}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok))
    tokens, types, merges, vocab = _gguf._vocab(str(tmp_path), 6)
    assert tokens == ["a", "b", "ab", "<|im_end|>", "<asr_text>", "[PAD5]"]
    assert types == [1, 1, 1, 3, 4, 5]                 # normal, control, user-defined, unused
    assert merges == ["a b"] and vocab["<|im_end|>"] == 3


def test_encoder_precision_checked(tmp_path):
    with pytest.raises(ValueError, match="f16, bf16 or q8_0"):
        _gguf.convert_encoder(str(tmp_path), str(tmp_path / "e.gguf"), "q4_0")


def test_prepare_checks_precision(tmp_path):
    with pytest.raises(ValueError, match="precision must be one of"):
        _cpu.prepare(str(tmp_path), "int3")
    with pytest.raises(ValueError, match="encoder_precision must be one of"):
        _cpu.prepare(str(tmp_path), "q8_0", "q4_0")


class _QuantizeFails:
    """kasr_quantize that fails; `finish` first does what a second worker finishing the same model does."""

    def __init__(self, finish):
        self.finish = finish

    def kasr_quantize(self, src, dst, kind, threads):
        if self.finish:
            dec = dst.decode().split(".gguf", 1)[0] + ".gguf"
            with open(dec, "wb") as f:
                f.write(b"q")
            os.remove(src)
        return 0

    def kasr_last_error(self):
        return b"cannot open the bf16 file"


def _fake_conversion(monkeypatch, tmp_path, lib):
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(_cpu, "_window_for", lambda folder: 0)
    monkeypatch.setattr(_gguf, "convert_encoder", lambda folder, out, kind, window_frames=None: open(out, "wb").close())
    monkeypatch.setattr(_gguf, "convert_decoder", lambda folder, out: open(out, "wb").close())
    monkeypatch.setattr(_cpu, "_lib", lambda: lib)
    model = tmp_path / "model"
    model.mkdir()
    return str(model)


def test_prepare_uses_the_file_another_worker_finished(monkeypatch, tmp_path):
    files = _cpu.prepare(_fake_conversion(monkeypatch, tmp_path, _QuantizeFails(finish=True)), "q8_0")
    assert files["decoder"].endswith("decoder-q8_0.gguf") and os.path.isfile(files["decoder"])
    assert sorted(os.listdir(os.path.dirname(files["decoder"]))) == ["decoder-q8_0.gguf", "encoder-f16-w0.gguf"]


def test_an_interrupted_conversion_leaves_no_temp_file(monkeypatch, tmp_path):
    model = _fake_conversion(monkeypatch, tmp_path, _QuantizeFails(finish=False))

    def interrupted(folder, out):
        open(out, "wb").close()
        raise KeyboardInterrupt
    monkeypatch.setattr(_gguf, "convert_decoder", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _cpu.prepare(model, "q8_0")
    left = [f for _, _, fs in os.walk(tmp_path / "cache") for f in fs]
    assert left == ["encoder-f16-w0.gguf"]


def test_prepare_raises_when_quantising_really_fails(monkeypatch, tmp_path):
    with pytest.raises(_cpu.EngineError, match="cannot open the bf16 file"):
        _cpu.prepare(_fake_conversion(monkeypatch, tmp_path, _QuantizeFails(finish=False)), "q8_0")


def test_fast_cpus_never_empty():
    cpus = _cpu._fast_cpus()
    assert cpus and _cpu.default_threads() >= 1
    if sys.platform != "darwin":
        limit = _cpu._cgroup_cpus()
        assert _cpu.default_threads() == (min(len(cpus), limit) if limit else len(cpus))


def test_container_cpu_limit_caps_the_default_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(_cpu, "sys", SimpleNamespace(platform="linux"))
    v2 = tmp_path / "v2"
    (v2 / "app").mkdir(parents=True)
    (tmp_path / "cg").write_text("0::/app\n")
    (v2 / "app" / "cpu.max").write_text("150000 100000\n")          # 1.5 CPUs: 2 threads
    assert _cpu._cgroup_cpus(str(v2), str(tmp_path / "cg")) == 2
    (v2 / "app" / "cpu.max").write_text("max 100000\n")             # no limit
    assert _cpu._cgroup_cpus(str(v2), str(tmp_path / "cg")) is None
    v1 = tmp_path / "v1" / "cpu,cpuacct"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("200000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert _cpu._cgroup_cpus(str(tmp_path / "v1"), str(tmp_path / "none")) == 2
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert _cpu._cgroup_cpus(str(tmp_path / "v1"), str(tmp_path / "none")) is None
    monkeypatch.setattr(_cpu, "_fast_cpus", lambda: list(range(64)))
    monkeypatch.setattr(_cpu, "_cgroup_cpus", lambda: 2)
    assert _cpu.default_threads() == 2


@pytest.mark.skipif(sys.platform != "darwin", reason="platform: macOS only")
def test_mac_threads_are_performance_cores():
    perf = _cpu._sysctl_int("hw.perflevel0.physicalcpu") or _cpu._sysctl_int("hw.physicalcpu")
    if _cpu._sysctl_int("kern.hv_vmm_present") == 1 and _cpu._sysctl_int("hw.perflevel1.physicalcpu") is None:
        perf = max(1, perf // 2)                          # virtual machine: see test_mac_threads_virtual_machine
    assert perf and _cpu.default_threads() == perf <= (os.cpu_count() or perf)


def test_mac_threads_virtual_machine(monkeypatch):
    real = {"hw.perflevel0.physicalcpu": 4, "hw.perflevel1.physicalcpu": 4, "kern.hv_vmm_present": 0}
    vm = {"hw.perflevel0.physicalcpu": 8, "kern.hv_vmm_present": 1}   # every virtual CPU reported as a performance core
    monkeypatch.setattr(_cpu, "_sysctl_int", lambda name: real.get(name))
    assert _cpu._mac_threads() == 4
    monkeypatch.setattr(_cpu, "_sysctl_int", lambda name: vm.get(name))
    assert _cpu._mac_threads() == 4


def test_gpu_default(monkeypatch):
    monkeypatch.setattr(_cpu.sys, "platform", "linux")
    assert _cpu.gpu_default() is False
    monkeypatch.setattr(_cpu.sys, "platform", "darwin")
    monkeypatch.setattr(_cpu.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(_cpu, "_sysctl_int", lambda name: {"kern.hv_vmm_present": 0}.get(name))
    assert _cpu.gpu_default() is _cpu._MAC_GPU_DEFAULT
    # a macOS virtual machine: the paravirtual GPU has no simdgroup reduction, so ggml runs the matrix products on
    # the CPU anyway (M2 virtual machine: Metal 1.8x vs CPU 5.9x real time for tiny); gpu=None means the CPU there
    monkeypatch.setattr(_cpu, "_sysctl_int", lambda name: {"kern.hv_vmm_present": 1}.get(name))
    assert _cpu.gpu_default() is False
    monkeypatch.setattr(_cpu, "_sysctl_int", lambda name: {"kern.hv_vmm_present": 0}.get(name))
    monkeypatch.setenv("KITTEN_ASR_GPU", "0")
    assert _cpu.gpu_default() is False
    monkeypatch.delenv("KITTEN_ASR_GPU")
    monkeypatch.setattr(_cpu.platform, "machine", lambda: "x86_64")
    assert _cpu.gpu_default() is False


def test_cache_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path))
    assert _cpu.cache_root() == str(tmp_path)


def test_missing_library_message(monkeypatch, tmp_path):
    monkeypatch.setenv(_cpu.LIB_ENV, str(tmp_path / "nothing.so"))
    ok, why = _cpu.available()
    assert not ok and "nothing.so" in why
    with pytest.raises(ImportError, match="on the CPU"):
        _deps.require("asr-cpu")


def test_backend_choice_checked():
    from kitten_inference.asr._api import KittenASR
    with pytest.raises(ValueError, match="backend must be"):
        KittenASR("KittenML/kitten-asr-tiny", backend="tpu")


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_TEST_MODEL"), reason="model: KITTEN_ASR_TEST_MODEL is not set")
def test_transcribe_end_to_end(tmp_path, monkeypatch):
    ok, why = _cpu.available()
    if not ok:
        pytest.skip(f"platform: {why}")
    cache = tmp_path
    if sys.platform == "win32":
        cache = cache / ("caché-" + "x" * 135) / ("音声-" + "y" * 135)
    monkeypatch.setenv(_cpu.CACHE_ENV, str(cache))
    from kitten_inference import KittenASR
    had_torch = "torch" in sys.modules                  # true when GPU tests ran earlier in the same session
    asr = KittenASR(os.environ["KITTEN_ASR_TEST_MODEL"], backend="cpu", threads=2)
    try:
        assert asr.capabilities["backend"] == "cpu" and asr.capabilities["precision"] == "q8_0"
        assert ("torch" in sys.modules) == had_torch    # the CPU engine does not import PyTorch
        snap = asr._engine.snapshot()
        assert snap["native"]["n_threads"] == 2 and snap["native"]["threadpool"] == 1
        assert asr.transcribe(np.zeros(16000, np.float32)) == ""            # silence is never sent to the model
        t = 16000 * 2
        tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(t) / 16000)).astype(np.float32)
        out = asr.transcribe(tone, details=True)
        assert isinstance(out.text, str) and len(out.segments) == 1
        r = asr._engine.decode(tone, asr._engine.default_prompt("English"), max_new_tokens=8)
        again = asr._engine.decode(tone, asr._engine.default_prompt("English"), draft_ids=r.token_ids,
                                   max_new_tokens=8)
        assert again.token_ids == r.token_ids                                 # a draft never changes the result
        assert again.info["accepted_tokens"] == len([t for t in r.token_ids])  # and it is accepted in full
    finally:
        asr.close()


@pytest.mark.parametrize("d_model, batch", [(640, 8), (896, 1), (1024, 1)])
def test_batch_default_follows_the_accuracy_check(tmp_path, d_model, batch):
    (tmp_path / "config.json").write_text(json.dumps({"thinker_config": {"audio_config": {"d_model": d_model}}}))
    assert _cpu.batch_default(str(tmp_path)) == batch


def test_batch_option_checked():
    for bad in (0, -1, 1.5, True, 257):
        with pytest.raises(ValueError, match="batch must be"):
            _cpu.CpuEngine("KittenML/kitten-asr-tiny", batch=bad, files={"decoder": "x", "encoder": "y"})


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_TEST_MODEL"), reason="model: KITTEN_ASR_TEST_MODEL is not set")
def test_transcribe_many_windows_together(tmp_path, monkeypatch):
    ok, why = _cpu.available()
    if not ok:
        pytest.skip(f"platform: {why}")
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path))
    from kitten_inference import KittenASR
    sr = 16000
    tones = [(0.1 * np.sin(2 * np.pi * f * np.arange(int(sr * d)) / sr)).astype(np.float32)
             for f, d in ((440, 2.0), (330, 1.0), (550, 3.0))]
    asr = KittenASR(os.environ["KITTEN_ASR_TEST_MODEL"], backend="cpu", threads=2, gpu=False, batch=4)
    try:
        assert asr.capabilities["batching"] is True
        if not hasattr(asr._engine._lib, "kasr_transcribe_batch"):
            pytest.skip("this engine library has no kasr_transcribe_batch")
        many = asr.transcribe(tones + [np.zeros(sr, np.float32)], details=True)
        assert len(many) == 4 and many[3].text == "" and all(len(t.segments) == 1 for t in many)
        assert asr._engine.stats["batch_calls"] == 1                         # one engine call for the list
        res = asr._engine.decode_batch(tones, asr._engine.default_prompt("English"), [8, 8, 8])
        assert [len(r.token_ids) <= 8 for r in res] == [True] * 3
        one = asr._engine.decode(tones[0], asr._engine.default_prompt("English"), max_new_tokens=8)
        assert asr._engine.decode_batch(tones[:1], asr._engine.default_prompt("English"), [8])[0].token_ids == \
            one.token_ids                                                     # one window: the one-at-a-time path
        prompt = asr._engine.default_prompt("English")
        asr._engine._want(1)            # a caller that keeps waiting: every batch call still starts one window
        try:
            assert len(asr._engine.decode_batch(tones[:2], prompt, [8, 8])) == 2
        finally:
            asr._engine._want(-1)
        long = np.concatenate([tones[2]] * 12)                                # 36 s: two windows, one call
        calls = asr._engine.stats["batch_calls"]
        assert len(asr.transcribe(long, details=True).segments) == 2
        assert asr._engine.stats["batch_calls"] == calls + 1
    finally:
        asr.close()


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_TEST_MODEL"), reason="model: KITTEN_ASR_TEST_MODEL is not set")
def test_stream_cache_gives_the_same_text(tmp_path, monkeypatch):
    """A growing window decoded with a stream key (the decoder state of its prompt head and of its complete 8 s
    audio blocks is kept) gives exactly the text of decoding every window from scratch."""
    ok, why = _cpu.available()
    if not ok:
        pytest.skip(f"platform: {why}")
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path))
    from kitten_inference import KittenASR
    asr = KittenASR(os.environ["KITTEN_ASR_TEST_MODEL"], backend="cpu", threads=2, gpu=False, audio_window=800)
    try:
        eng = asr._engine
        prompt = eng.default_prompt("English")
        rng = np.random.default_rng(0)
        x = (0.1 * np.sin(2 * np.pi * 330 * np.arange(16000 * 18) / 16000) +
             0.01 * rng.standard_normal(16000 * 18)).astype(np.float32)
        for n in (16000 * 5, 16000 * 9, 16000 * 17, 16000 * 18):
            keyed = eng.decode(x[:n], prompt, stream_key="s:0", max_new_tokens=16)
            fresh = eng.decode(x[:n], prompt, max_new_tokens=16)
            assert keyed.token_ids == fresh.token_ids
            if n == 16000 * 18:
                # the fresh decode in between dropped the kept state: only the next keyed call starts over
                assert keyed.info["cached_positions"] == 0
        a = eng.decode(x[:16000 * 17], prompt, stream_key="s:1", max_new_tokens=16)
        b = eng.decode(x, prompt, stream_key="s:1", max_new_tokens=16)
        assert a.info["cached_positions"] == 0 and b.info["cached_blocks"] == 2      # two complete 8 s blocks reused
        assert b.token_ids == eng.decode(x, prompt, max_new_tokens=16).token_ids
        eng.release("s:1")
    finally:
        asr.close()


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_TEST_MODEL"), reason="model: KITTEN_ASR_TEST_MODEL is not set")
@pytest.mark.skipif(sys.platform != "darwin", reason="platform: the Mac GPU (Metal) is only built on macOS")
def test_transcribe_metal_matches_cpu(tmp_path, monkeypatch):
    ok, why = _cpu.available()
    if not ok:
        pytest.skip(f"platform: {why}")
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path))
    from kitten_inference import KittenASR
    t = 16000 * 3
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(t) / 16000)).astype(np.float32)
    texts = {}
    for gpu in (False, True):
        asr = KittenASR(os.environ["KITTEN_ASR_TEST_MODEL"], backend="metal" if gpu else "cpu")
        try:
            native = asr._engine.snapshot()["native"]
            assert native["use_gpu"] == int(gpu) and asr.capabilities["backend"] == ("metal" if gpu else "cpu")
            if gpu:
                # ggml names its Metal devices MTL0, MTL1, ...; the description is the GPU's name
                assert any(d["name"].startswith("MTL") or "metal" in (d["name"] + d["description"]).lower()
                           for d in native["devices"])
                assert native["threadpool"] == 0
                assert native["repack"] == 0          # weights stay on the GPU (no CPU_REPACK buffer)
            texts[gpu] = asr.transcribe(tone)
        finally:
            asr.close()
    assert isinstance(texts[True], str) and isinstance(texts[False], str)


def _engine_without_library(lib, batch=4):
    """A CpuEngine whose library is a stand-in (no model, no compiled engine)."""
    import threading
    eng = object.__new__(_cpu.CpuEngine)
    eng._lock, eng._ctx, eng._closed, eng.batch, eng._lib = threading.Lock(), 1, False, batch, lib
    eng.stats = dict(batch_calls=0, requests=0, generated_tokens=0, steps=0, mel_ms=0.0, encode_ms=0.0,
                     prefill_ms=0.0, decode_ms=0.0)
    return eng


class _TurnLib:
    """Stands in for libkasr's kasr_batch_yield / kasr_transcribe_batch."""

    def __init__(self, stop_after=None):
        self.want, self.calls, self.stop_after = 0, [], stop_after

    def kasr_batch_yield(self, ctx, delta):
        self.want += delta
        return self.want

    def kasr_last_error(self):
        return b""

    def kasr_transcribe_batch(self, ctx, m, ptrs, lens, head, tail, mx, slots, res):
        self.calls.append(m)
        for i in range(m):
            if len(self.calls) == 1 and self.stop_after is not None and i >= self.stop_after:
                res[i].finish = -1            # not started: another caller asked for a turn
            else:
                res[i].text, res[i].finish = b"w%d" % lens[i], 0
        return 1


def test_a_batch_call_cut_short_for_another_caller_sends_the_rest_again():
    lib = _TurnLib(stop_after=2)
    eng = _engine_without_library(lib)
    xs = [np.zeros(8000 + 100 * i, np.float32) for i in range(5)]
    out = eng.decode_batch(xs, "head" + _cpu._AUDIO + "tail", [8] * 5)
    assert lib.calls == [5, 3]                                   # 2 done in the first call, the 3 others after
    assert [r.text for r in out] == ["w%d" % len(x) for x in xs]  # in the given order
    assert eng.stats["batch_calls"] == 2 and lib.want == 0


def test_a_caller_waiting_for_the_engine_is_counted_until_it_has_its_turn():
    lib = _TurnLib()
    eng = _engine_without_library(lib)
    with eng._turn():
        assert lib.want == 0 and eng._lock.locked()
    assert lib.want == 0 and not eng._lock.locked()


def test_ctrl_c_during_a_batch_call_asks_it_to_stop_and_is_raised_after():
    import _thread
    import threading
    import time
    lib = _TurnLib()
    eng = _engine_without_library(lib)
    seen = []

    def long_call():
        t_end = time.monotonic() + 10
        while lib.want == 0 and time.monotonic() < t_end:    # the library finishes the windows in flight
            time.sleep(0.01)
        seen.append(lib.want)
        return 1

    import signal
    old = signal.signal(signal.SIGINT, signal.default_int_handler)  # a background job may run with SIGINT ignored
    try:
        threading.Timer(0.3, _thread.interrupt_main).start()
        with pytest.raises(KeyboardInterrupt):
            eng._call_interruptibly(long_call)
    finally:
        signal.signal(signal.SIGINT, old)
    assert seen == [1] and lib.want == 0
