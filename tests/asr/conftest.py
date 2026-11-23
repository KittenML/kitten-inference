"""Shared pieces of the speech recognition tests. None of the unit tests needs PyTorch or a GPU: the engine is
replaced by FakeEngine, and PyTorch by a small stand-in module where a test needs "PyTorch is installed"."""
import contextlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path

import pytest

GPU_DIR = Path(__file__).parent / "gpu"
# Unit tests (not tests/asr/gpu) that were skipped: with one of these variables set to 1 (CI sets the second for
# the whole run) they are a failure, except tests for another operating system (skip reason "platform: ...") and
# end-to-end tests that need a downloaded model (skip reason "model: ...", run when KITTEN_ASR_TEST_MODEL is set).
STRICT = ("KITTEN_ASR_STRICT_SKIPS", "KT2N_STRICT_SKIPS")
EXPECTED_SKIPS = ("platform:", "model:")
_skipped = []
_gpu_tests = set()  # node ids of the tests in tests/asr/gpu (filled at collection)


def pytest_runtest_logreport(report):
    if not report.skipped or report.nodeid in _gpu_tests:
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if not str(reason).replace("Skipped: ", "").startswith(EXPECTED_SKIPS):
        _skipped.append(f"{report.nodeid} ({reason})")


def pytest_sessionfinish(session, exitstatus):
    if _skipped and any(os.environ.get(name) == "1" for name in STRICT):
        print(f"\n{STRICT[0]}: {len(_skipped)} unit test(s) skipped: {', '.join(_skipped)}", file=sys.stderr)
        session.exitstatus = 1


def pytest_collection_modifyitems(config, items):
    # tests/asr/gpu skips by itself unless a model is named (see its conftest); those skips are expected.
    _gpu_tests.update(item.nodeid for item in items if GPU_DIR in Path(str(item.fspath)).parents)


class FakeEngine:
    """Stands in for kitten_inference.asr.Engine and EngineProcess: the same calls, no model. Every call is
    recorded. `reply(call)` makes the model's output for one decode; it may block or raise. `finish(call, text)`
    is the decode's finish_reason ("stop" by default; "length" when the decode used up its token budget)."""

    def __init__(self, model, **options):
        self.model, self.options = model, options
        self.calls, self.released, self.prompts = [], [], []
        self.log = []                 # ("decode", stream_key) and ("release", stream_key), in the order they came
        self.snapshots = self.closed = self.active = self.max_active = 0
        self.reply = lambda call: "language English<asr_text>n%d" % len(call.audio)
        self.finish = lambda call, text: "stop"
        self._lock = threading.Lock()

    def _check(self):
        if self.closed:
            from kitten_inference.asr._types import EngineStopped
            raise EngineStopped("kitten-asr: the engine is closed")

    def default_prompt(self, language=None):
        self._check()
        self.prompts.append(language)
        return "<prompt:%s>" % language

    def decode(self, audio, prompt, *, draft_ids=None, stream_key=None, max_new_tokens=256, priority=0,
               timeout_s=30.0, on_tokens=None):
        import numpy as np

        from kitten_inference.asr._types import DecodeResult
        self._check()
        with self._lock:
            call = types.SimpleNamespace(
                index=len(self.calls), audio=np.array(audio, copy=True), prompt=prompt,
                draft_ids=None if draft_ids is None else list(draft_ids), stream_key=stream_key,
                max_new_tokens=max_new_tokens, priority=priority, timeout_s=timeout_s, on_tokens=on_tokens)
            call.token_ids = [1000 + call.index, len(call.audio)]      # different for every call
            self.calls.append(call)
            self.log.append(("decode", stream_key))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            text = self.reply(call)
        finally:
            with self._lock:
                self.active -= 1
        return DecodeResult(text, list(call.token_ids), {"finish_reason": self.finish(call, text), "backend": "fake"})

    def release(self, stream_key):
        if self.closed:
            return
        with self._lock:
            self.released.append(stream_key)
            self.log.append(("release", stream_key))

    def snapshot(self):
        self._check()
        self.snapshots += 1
        return {"attention": "torch", "compiled": False, "page": 16, "precision": "bf16", "system_prompt": "auto:no-file"}

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_engine(monkeypatch):
    """KittenASR(...) gets a FakeEngine instead of the real engine, and the check for Python, PyTorch and a GPU
    only records how it was called. Returns: engines (every FakeEngine made, each with kind "thread" or
    "process"), required ((feature, device) of every check), type (FakeEngine), and reply, which, when set,
    becomes the reply of the engines made afterwards."""
    from kitten_inference.asr import _deps, _proc
    made = types.SimpleNamespace(engines=[], required=[], reply=None, type=FakeEngine)

    def build(kind):
        def make(model, **options):
            engine = FakeEngine(model, **options)
            engine.kind = kind
            if made.reply is not None:
                engine.reply = made.reply
            made.engines.append(engine)
            return engine
        return make

    module = types.ModuleType("kitten_inference.asr._engine")       # the real one imports torch
    module.Engine = build("thread")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(_proc, "EngineProcess", build("process"))
    monkeypatch.setattr(_deps, "require", lambda feature="asr", device=None: made.required.append((feature, device)))
    return made


@pytest.fixture
def asr(fake_engine):
    """A KittenASR on a FakeEngine (the `engine` fixture)."""
    from kitten_inference.asr._api import KittenASR
    model = KittenASR("KittenML/kitten-asr-tiny")
    yield model
    model.close()


@pytest.fixture
def engine(asr, fake_engine):
    return fake_engine.engines[-1]


class _Sys:
    """The sys module with another version_info and platform."""

    def __init__(self, version, platform):
        self.version_info, self.platform = tuple(version) + (0, "final", 0), platform

    def __getattr__(self, name):
        return getattr(sys, name)


@pytest.fixture
def machine(monkeypatch):
    """machine(python=(3, 12), platform="linux"): what the dependency check sees as sys.version_info and
    sys.platform, whatever runs the tests (the unit tests also run on Python 3.8 and on Windows)."""
    from kitten_inference.asr import _deps

    def choose(python=(3, 12), platform="linux"):
        monkeypatch.setattr(_deps, "sys", _Sys(python, platform))
    return choose


@pytest.fixture
def packages(monkeypatch):
    """packages(torch=False, tokenizers=True): importlib.util.find_spec says that those top-level modules are
    (not) installed, whatever is installed here. find_spec imports nothing."""
    def choose(**present):
        real = importlib.util.find_spec

        def find_spec(name, package=None):
            top = name.partition(".")[0]
            if top in present:
                return importlib.machinery.ModuleSpec(top, None) if present[top] else None
            return real(name, package)
        monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return choose


@pytest.fixture
def fake_torch(monkeypatch):
    """fake_torch(cuda=..., built=..., gpus=...) puts a stand-in "torch" module into sys.modules and returns it.
    It has what the dependency check and the loader's module level read, and nothing that computes:
      cuda     torch.cuda.is_available()
      built    torch.version.cuda (None = a build without CUDA)
      gpus     (name, compute capability) of every device torch.cuda sees
    """
    def install(cuda=True, built="12.4", gpus=(("Fake RTX", (8, 6)),), current=0):
        torch = types.ModuleType("torch")
        torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)

        class device:
            def __init__(self, spec):
                kind, _, index = str(spec).partition(":")
                self.type, self.index = kind, (int(index) if index else None)

            def __str__(self):
                return self.type if self.index is None else "%s:%d" % (self.type, self.index)

        class dtype:
            def __init__(self, name, floating=True):
                self.name, self.is_floating_point = name, floating

            def __repr__(self):
                return "torch." + self.name

        torch.device, torch.dtype, torch.Tensor = device, dtype, type("Tensor", (), {})
        torch.bfloat16, torch.float16, torch.float32 = dtype("bfloat16"), dtype("float16"), dtype("float32")
        torch.int64 = dtype("int64", floating=False)
        torch.version = types.SimpleNamespace(cuda=built)
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: cuda, current_device=lambda: current, device_count=lambda: len(gpus),
            get_device_capability=lambda index: gpus[index][1], get_device_name=lambda index: gpus[index][0])
        torch.from_numpy = lambda array: array          # the loader's host tensors stay numpy arrays
        torch.no_grad = contextlib.nullcontext
        monkeypatch.setitem(sys.modules, "torch", torch)
        return torch
    return install
