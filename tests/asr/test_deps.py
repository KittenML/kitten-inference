"""What the engine asks of the machine, and the words it uses when something is missing: an old Python, a system
other than Linux or Windows x64, no PyTorch (with the pip command), no NVIDIA GPU, an old GPU, no tokenizers package. PyTorch is
a stand-in module."""
import sys
import types

import pytest

from kitten_inference.asr import _deps

ON_CPU = "Leave out backend (or pass backend='cpu') to run on the CPU."
GPU = ("Kitten ASR with backend='cuda' needs an NVIDIA GPU of the Ampere generation or newer (compute capability 8.0 or "
       "higher).")
NO_TORCH = "Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch"
NO_TOKENIZERS = "Kitten ASR needs the tokenizers package, which is not installed. Install it: pip install tokenizers"
CUDA_PLATFORMS = f"Kitten ASR with backend='cuda' needs Linux or Windows x64. {ON_CPU}"


@pytest.fixture
def ready(machine, packages, fake_torch, monkeypatch):
    """A machine where everything is in place; a test then takes one thing away. Returns the stand-in torch."""
    machine()
    packages(torch=True, tokenizers=True)
    monkeypatch.setitem(sys.modules, "tokenizers", types.ModuleType("tokenizers"))
    return fake_torch()


def test_constants():
    assert _deps.NAME == "Kitten ASR" and _deps.MIN_PYTHON == (3, 10) and _deps.MIN_CAPABILITY == (8, 0)
    assert _deps.FEATURES == ("asr", "asr-process", "asr-cpu")


def test_everything_in_place(ready):
    assert _deps.require() is None and _deps.require("asr", device="cuda:0") is None
    assert _deps.require("asr-process") is None
    assert _deps.require_torch() is ready


@pytest.mark.parametrize("call", [lambda: _deps.require(), lambda: _deps.require("asr"),
                                  lambda: _deps.require("asr-process"), lambda: _deps.require_torch()])
@pytest.mark.parametrize("python", [(3, 8), (3, 9)])
def test_old_python(ready, machine, packages, call, python):
    machine(python=python)
    packages(torch=False, tokenizers=False)             # the Python version is said first
    with pytest.raises(ImportError) as e:
        call()
    assert str(e.value) == f"Kitten ASR needs Python 3.10 or newer (this is Python {python[0]}.{python[1]})."


def test_python_3_10_is_enough(ready, machine):
    machine(python=(3, 10))
    _deps.require("asr")


@pytest.mark.parametrize("call", [lambda: _deps.require("asr"), lambda: _deps.require("asr-process"),
                                  lambda: _deps.require_torch()])
def test_torch_missing_gives_the_pip_command(ready, packages, call):
    packages(torch=False, tokenizers=True)
    with pytest.raises(ImportError) as e:
        call()
    assert str(e.value) == NO_TORCH and str(e.value).endswith("pip install torch")


def test_torch_installed_but_broken(ready, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)       # "import torch" fails although the package is found
    with pytest.raises(ImportError, match=r"Kitten ASR could not import PyTorch \(.*\)\. "
                                          r"Reinstall it: pip install torch$") as e:
        _deps.require("asr")
    assert isinstance(e.value.__cause__, ImportError)


@pytest.mark.parametrize("built, why", [(None, "This PyTorch build has no CUDA support"),
                                        ("12.4", "PyTorch found no CUDA device here")])
def test_no_cuda(ready, fake_torch, packages, built, why):
    fake_torch(cuda=False, built=built)
    packages(torch=True, tokenizers=False)              # the GPU is said before the tokenizers package
    with pytest.raises(RuntimeError) as e:
        _deps.require("asr")
    assert str(e.value) == f"{GPU} {why}. {ON_CPU}"
    assert not isinstance(e.value, ImportError)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_device_must_be_cuda(ready, device):
    with pytest.raises(RuntimeError) as e:
        _deps.require("asr", device=device)
    assert str(e.value) == f"{GPU} Device {device!r} is not supported."


def test_gpu_older_than_ampere(ready, fake_torch):
    fake_torch(gpus=(("Tesla T4", (7, 5)), ("Fake RTX", (8, 0))))
    with pytest.raises(RuntimeError) as e:
        _deps.require("asr")
    assert str(e.value) == f"{GPU} Tesla T4 has compute capability 7.5, which is not supported."
    _deps.require("asr", device="cuda:1")                # the device that was asked for is the one checked
    fake_torch(gpus=(("Tesla T4", (7, 5)), ("Fake RTX", (8, 0))), current=1)
    _deps.require("asr")                                 # "cuda" is the current device


def test_device_number_that_does_not_exist(ready):
    with pytest.raises(RuntimeError) as e:
        _deps.require("asr", device="cuda:1")
    assert str(e.value) == "Kitten ASR: there is no device cuda:1 (PyTorch sees 1)."


def test_tokenizers_missing(ready, packages):
    packages(torch=True, tokenizers=False)
    for feature in ("asr", "asr-process"):              # the CPU engine ("asr-cpu") needs no tokenizers
        with pytest.raises(ImportError) as e:
            _deps.require(feature)
        assert str(e.value) == NO_TOKENIZERS


def test_tokenizers_installed_but_broken(ready, monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(ImportError, match=r"could not import the tokenizers package \(.*\)\. "
                                          r"Reinstall it: pip install tokenizers$"):
        _deps.require("asr")
    _deps.require("asr-process")                         # only looks for the package


def test_process_side_imports_nothing_and_skips_the_gpu(ready, fake_torch, monkeypatch):
    fake_torch(cuda=False, built=None)                   # no GPU on this side: not looked at
    monkeypatch.setitem(sys.modules, "torch", None)      # any "import torch" would fail
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    assert _deps.require("asr-process") is None


@pytest.mark.parametrize("call", [lambda: _deps.require(), lambda: _deps.require("asr"),
                                  lambda: _deps.require("asr-process")])
@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_unsupported_cuda_platform(ready, machine, packages, monkeypatch, call, platform):
    machine(platform=platform)
    monkeypatch.setattr(_deps.platform, "machine", lambda: "ARM64")
    with pytest.raises(RuntimeError) as e:
        call()
    assert str(e.value) == CUDA_PLATFORMS and not isinstance(e.value, ImportError)
    packages(torch=False, tokenizers=False)              # the system is said before the packages
    with pytest.raises(RuntimeError) as e:
        call()
    assert str(e.value) == CUDA_PLATFORMS


def test_windows_x64_cuda_runs_in_process(ready, machine, monkeypatch):
    machine(platform="win32")
    monkeypatch.setattr(_deps.platform, "machine", lambda: "AMD64")
    _deps.require("asr")
    with pytest.raises(RuntimeError, match="On Windows use backend='cuda' with process=False"):
        _deps.require("asr-process")


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_old_python_is_said_before_the_system(ready, machine, platform):
    machine(python=(3, 9), platform=platform)
    for feature in _deps.FEATURES:
        with pytest.raises(ImportError) as e:
            _deps.require(feature)
        assert str(e.value) == "Kitten ASR needs Python 3.10 or newer (this is Python 3.9)."


@pytest.mark.parametrize("feature", ["tts", "", "ASR", None])
def test_unknown_feature(ready, feature):
    with pytest.raises(ValueError, match="unknown feature .*; choose one of asr, asr-process"):
        _deps.require(feature)
