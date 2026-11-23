"""KittenASR(model, backend=None, precision=None): which engine a backend name starts, what the automatic choice
picks, the precisions each backend offers, and the words of the errors. The engines are stand-ins: nothing here
needs PyTorch, a GPU or the CPU engine library."""
import types

import pytest

from kitten_inference.asr import _api, _cpu, _deps, _precision

MODEL = "KittenML/kitten-asr-tiny"


@pytest.fixture
def cpu_engines(fake_engine, monkeypatch):
    """KittenASR on the CPU gets a stand-in for CpuEngine; returns the list of those made (model, options)."""
    made = []

    class FakeCpuEngine(fake_engine.type):
        batch = 1

        def __init__(self, model, **options):
            super().__init__(model, **options)
            self.params = {"use_gpu": int(bool(options.get("gpu")))}
            made.append(self)

        def snapshot(self):
            return {"attention": None, "precision": self.options.get("precision")}

    monkeypatch.setattr(_cpu, "CpuEngine", FakeCpuEngine)
    return made


def no_nvidia_gpu(monkeypatch, calls):
    def require(feature="asr", device=None):
        calls.append(feature)
        if feature != "asr-cpu":
            raise RuntimeError("Kitten ASR with backend='cuda' needs an NVIDIA GPU. PyTorch found no CUDA device here.")
    monkeypatch.setattr(_deps, "require", require)


def test_left_out_backend_takes_the_nvidia_gpu_when_it_can_be_used(fake_engine, cpu_engines):
    asr = _api.KittenASR(MODEL)
    assert fake_engine.required == [("asr", None)] and cpu_engines == []
    assert fake_engine.engines[0].options == {"cache_dir": None, "precision": "bf16"}
    assert asr.capabilities["backend"] == "cuda" and asr.capabilities["precision"] == "bf16"
    asr.close()


def test_left_out_backend_falls_back_to_the_cpu(fake_engine, cpu_engines, monkeypatch):
    calls = []
    no_nvidia_gpu(monkeypatch, calls)
    asr = _api.KittenASR(MODEL)
    assert calls == ["asr", "asr-cpu"]
    (engine,) = cpu_engines
    assert engine.options == {"cache_dir": None, "precision": "q8_0"}     # gpu is the engine's own choice (Mac)
    caps = asr.capabilities
    assert (caps["backend"], caps["device"], caps["precision"], caps["batching"]) == ("cpu", "cpu", "q8_0", False)
    asr.close()


def test_backend_cpu_is_the_cpu_only(fake_engine, cpu_engines):
    asr = _api.KittenASR(MODEL, backend="CPU ", precision="f16", threads=4)
    (engine,) = cpu_engines
    assert engine.options == {"cache_dir": None, "precision": "f16", "threads": 4, "gpu": False}
    assert fake_engine.required == [("asr-cpu", None)] and asr.capabilities["backend"] == "cpu"
    asr.close()


def test_backend_cuda_never_falls_back(fake_engine, cpu_engines, monkeypatch):
    calls = []
    no_nvidia_gpu(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="needs an NVIDIA GPU"):
        _api.KittenASR(MODEL, backend="cuda")
    assert calls == ["asr"] and cpu_engines == [] and fake_engine.engines == []


def test_backend_metal_is_for_apple_silicon(fake_engine, cpu_engines, monkeypatch):
    monkeypatch.setattr(_api, "sys", types.SimpleNamespace(platform="linux"))
    with pytest.raises(RuntimeError, match="backend='metal' is the GPU of a Mac with Apple silicon"):
        _api.KittenASR(MODEL, backend="metal")
    assert cpu_engines == []
    monkeypatch.setattr(_api, "sys", types.SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(_api.platform, "machine", lambda: "arm64")
    with pytest.raises(ValueError, match="backend='metal' runs on the Mac GPU"):
        _api.KittenASR(MODEL, backend="metal", gpu=False)
    asr = _api.KittenASR(MODEL, backend="metal")
    assert cpu_engines[0].options == {"cache_dir": None, "precision": "q8_0", "gpu": True}
    assert asr.capabilities["backend"] == "metal" and asr.capabilities["device"] == "metal"
    asr.close()


@pytest.mark.parametrize("backend", ["auto", "gpu", "", 3, True, b"cpu"])
def test_unknown_backend(fake_engine, cpu_engines, backend):
    with pytest.raises(ValueError, match="backend must be 'cuda'"):
        _api.KittenASR(MODEL, backend=backend)
    assert fake_engine.engines == [] and cpu_engines == [] and fake_engine.required == []


@pytest.mark.parametrize("backend, precision, words", [
    ("cuda", "q8_0", "is for the CPU and the Mac GPU"),
    ("cuda", "fp8", "is not offered: 8-bit float"),
    ("cuda", "fp32", "precision must be one of bf16, fp16, int8, int4 on an NVIDIA GPU"),
    ("cpu", "fp16", r"is for NVIDIA GPUs \(backend='cuda'\)"),
    ("cpu", "q4_k_m", "is not offered: 4-bit weights"),
    ("cpu", "q6_k", "is not offered: 6-bit weights"),
    ("cpu", "f32", "is not offered"),
    ("cpu", 8, "precision must be one of q8_0, f16, bf16, int8, int4 on the CPU"),
])
def test_a_precision_the_backend_does_not_offer_fails_before_anything_loads(fake_engine, cpu_engines, backend,
                                                                           precision, words):
    with pytest.raises(ValueError, match=words):
        _api.KittenASR(MODEL, backend=backend, precision=precision)
    assert fake_engine.engines == [] and cpu_engines == [] and fake_engine.required == []


def test_precision_is_checked_for_the_backend_that_was_picked(fake_engine, cpu_engines, monkeypatch):
    no_nvidia_gpu(monkeypatch, [])
    with pytest.raises(ValueError, match=r"precision 'fp16' is for NVIDIA GPUs .*Use one of q8_0, f16, bf16"):
        _api.KittenASR(MODEL, precision="fp16")
    assert cpu_engines == []


def test_process_is_for_the_gpu_engine(fake_engine, cpu_engines):
    with pytest.raises(ValueError, match="process=True is for the NVIDIA GPU engine"):
        _api.KittenASR(MODEL, backend="cpu", process=True)
    assert cpu_engines == [] and fake_engine.required == []


def test_no_backend_at_all_names_both_problems(fake_engine, monkeypatch):
    def require(feature="asr", device=None):
        if feature == "asr-cpu":
            raise ImportError("Kitten ASR on the CPU: this build has no CPU engine.")
        raise ImportError("Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch")
    monkeypatch.setattr(_deps, "require", require)
    with pytest.raises(ImportError, match="no CPU engine. An NVIDIA GPU cannot be used here either: .*pip install torch"):
        _api.KittenASR(MODEL)


def test_offered_precisions_and_defaults():
    assert _precision.offered("cuda") == ("bf16", "fp16", "int8", "int4") and _precision.check("cuda", None) == "bf16"
    assert _precision.offered("cpu") == _precision.offered("metal") == ("q8_0", "f16", "bf16", "int8", "int4")
    assert _precision.check("metal", None) == "q8_0" and _cpu.DECODER_PRECISIONS == _precision.CPU
    assert not set(_precision.NOT_OFFERED_CPU) & set(_precision.CPU)
    assert not set(_precision.NOT_OFFERED_GPU) & set(_precision.GPU)


def test_a_run_away_loop_is_cut_on_every_backend(fake_engine, cpu_engines):
    import numpy as np
    loop = "The meeting starts now. " + "a bit of " * 30
    fake_engine.reply = lambda call: loop.strip()
    x = np.full(16000, 0.1, np.float32)
    on_gpu = _api.KittenASR(MODEL, backend="cuda")
    on_cpu = _api.KittenASR(MODEL, backend="cpu")
    cpu_engines[0].reply = fake_engine.reply
    for model in (on_gpu, on_cpu):          # the decode used up its token budget: a run-away loop
        model._engine.finish = lambda call, text: "length"
    assert on_gpu.transcribe(x) == on_cpu.transcribe(x) == "The meeting starts now. a bit of"
    for model in (on_gpu, on_cpu):          # the same words from a decode that ended by itself are kept
        model._engine.finish = lambda call, text: "stop"
    assert on_gpu.transcribe(x) == on_cpu.transcribe(x) == loop.strip()
    on_gpu.close()
    on_cpu.close()


def test_left_out_backend_leaves_out_the_other_engines_options(fake_engine, cpu_engines, monkeypatch):
    asr = _api.KittenASR(MODEL, threads=4, batch=2, max_batch=8)             # an NVIDIA GPU is found
    assert fake_engine.engines[0].options == {"cache_dir": None, "precision": "bf16", "max_batch": 8}
    asr.close()
    no_nvidia_gpu(monkeypatch, [])
    asr = _api.KittenASR(MODEL, threads=4, batch=2, max_batch=8)             # no GPU: the CPU engine
    assert cpu_engines[0].options == {"cache_dir": None, "precision": "q8_0", "threads": 4, "batch": 2}
    asr.close()
    asr = _api.KittenASR(MODEL, backend="cpu", threads=4, max_batch=8)      # a named backend gets every option
    assert cpu_engines[1].options == {"cache_dir": None, "precision": "q8_0", "threads": 4, "max_batch": 8,
                                      "gpu": False}
    asr.close()


def test_named_metal_never_runs_on_the_cpu_silently(fake_engine, cpu_engines, monkeypatch):
    monkeypatch.setattr(_api, "sys", types.SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(_api.platform, "machine", lambda: "arm64")
    cls = _cpu.CpuEngine

    class NoGpuFound(cls):
        def __init__(self, model, **options):
            super().__init__(model, **options)
            self.params = {"use_gpu": 0}          # the engine found no GPU device and runs on the CPU

    monkeypatch.setattr(_cpu, "CpuEngine", NoGpuFound)
    with pytest.raises(RuntimeError, match="no Mac GPU could be used here"):
        _api.KittenASR(MODEL, backend="metal")
    assert cpu_engines[-1].closed == 1


def test_cpu_file_windows_go_to_the_engine_in_one_call(fake_engine, cpu_engines, monkeypatch):
    """Every window of a long file goes to the CPU engine's decode_batch in one call, longest first; the engine
    itself lets a live session in between windows (CpuEngine._turn)."""
    import numpy as np
    no_nvidia_gpu(monkeypatch, [])
    calls = []

    def decode_batch(self, audios, prompt, budgets):
        from kitten_inference.asr._types import DecodeResult
        calls.append([len(a) for a in audios])
        return [DecodeResult("language English<asr_text>w%d" % len(a), [], {"finish_reason": "stop"}) for a in audios]

    asr = _api.KittenASR(MODEL)
    engine = asr._engine
    engine.batch = 2
    engine.decode_batch = types.MethodType(decode_batch, engine)
    x = np.full(16000 * 290, 0.1, np.float32)
    n = len(_api._windows(x))
    text = asr.transcribe(x)
    assert len(calls) == 1 and len(calls[0]) == n >= 10 and calls[0] == sorted(calls[0], reverse=True)
    assert text.count("w") == n
    asr.close()
