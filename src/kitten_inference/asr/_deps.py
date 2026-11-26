"""Checks what a speech recognition backend needs before it starts. The NVIDIA GPU engine: Python, PyTorch,
an NVIDIA GPU and tokenizers. The CPU engine: Python and the engine library this package ships."""
from __future__ import annotations

import importlib.util
import platform
import sys

NAME = "Kitten ASR"
MIN_PYTHON = (3, 10)
MIN_CAPABILITY = (8, 0)              # CUDA compute capability of the Ampere generation
FEATURES = ("asr", "asr-process", "asr-cpu")
_ON_CPU = "Leave out backend (or pass backend='cpu') to run on the CPU."
_GPU = (f"{NAME} with backend='cuda' needs an NVIDIA GPU of the Ampere generation or newer (compute capability 8.0 or "
        "higher).")
_CUDA_SYSTEM = f"{NAME} with backend='cuda' needs Linux or Windows x64. {_ON_CPU}"
_WINDOWS_PROCESS = f"{NAME} with process=True needs Linux. On Windows use backend='cuda' with process=False."


def windows_x64() -> bool:
    return platform.machine().lower() in ("amd64", "x86_64") and sys.maxsize > 2**32


def _installed(module: str) -> bool:
    """Whether `module` can be found, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _require_python() -> None:
    version = tuple(sys.version_info[:2])
    if version < MIN_PYTHON:
        raise ImportError(f"{NAME} needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer "
                          f"(this is Python {version[0]}.{version[1]}).")


def _require_cuda_system(feature: str) -> None:
    if sys.platform.startswith("linux"):
        return
    if sys.platform == "win32" and windows_x64():
        if feature == "asr-process":
            raise RuntimeError(_WINDOWS_PROCESS)
        return
    raise RuntimeError(_CUDA_SYSTEM)


def _require_torch(load: bool = True):
    """The torch module (None with load=False, which only looks for it)."""
    if not _installed("torch"):
        raise ImportError(f"{NAME} needs PyTorch, which is not installed. Install it: pip install torch")
    if not load:
        return None
    # Triton (PyTorch's GPU kernel compiler) warns on every compile when this is unset; 1 is its default behaviour,
    # and the setting every GPU number of this engine was measured with. A value the user set is kept.
    import os
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise ImportError(f"{NAME} could not import PyTorch ({type(exc).__name__}: {exc}). "
                          "Reinstall it: pip install torch") from exc
    return torch


def _require_gpu(torch, device=None) -> None:
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        raise RuntimeError(f"{_GPU} Device {str(dev)!r} is not supported.")
    if not torch.cuda.is_available():
        built = getattr(torch.version, "cuda", None)
        why = "PyTorch found no CUDA device here" if built else "This PyTorch build has no CUDA support"
        if sys.platform == "win32":
            why += ". Check the NVIDIA driver and install a CUDA build of PyTorch"
        raise RuntimeError(f"{_GPU} {why}. {_ON_CPU}")
    index = torch.cuda.current_device() if dev.index is None else dev.index
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f"{NAME}: there is no device cuda:{index} (PyTorch sees {count}).")
    capability = tuple(torch.cuda.get_device_capability(index))
    if capability < MIN_CAPABILITY:
        raise RuntimeError(f"{_GPU} {torch.cuda.get_device_name(index)} has compute capability "
                           f"{capability[0]}.{capability[1]}, which is not supported.")


def _require_tokenizers(load: bool = True) -> None:
    message = f"{NAME} needs the tokenizers package, which is not installed. Install it: pip install tokenizers"
    if not _installed("tokenizers"):
        raise ImportError(message)
    if load:
        try:
            import tokenizers  # noqa: F401
        except ImportError as exc:
            raise ImportError(f"{NAME} could not import the tokenizers package ({type(exc).__name__}: {exc}). "
                              "Reinstall it: pip install tokenizers") from exc


def require_torch():
    """Raises ImportError, saying what to do, when PyTorch cannot be imported here; returns the torch module.
    Checks the Python version first."""
    _require_python()
    return _require_torch()


def require(feature: str = "asr", device=None) -> None:
    """Raises, saying what to do, when `feature` cannot work here.

    "asr" (the engine) checks in this order: Python 3.10 or newer (ImportError), the system is Linux or Windows x64
    (RuntimeError), PyTorch can be imported (ImportError), `device` (default "cuda") is an NVIDIA GPU of the
    Ampere generation or newer (RuntimeError), the tokenizers package can be imported (ImportError).

    "asr-cpu" (backend="cpu") checks Python 3.10 or newer and that this package has the CPU engine library;
    it needs neither PyTorch nor a GPU.

    "asr-process" is the calling side of the Linux engine process. It checks Python and the platform,
    and that PyTorch and tokenizers are installed. It imports neither and does not look at the GPU: the
    engine process runs the "asr" checks itself and reports what it finds.
    """
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; choose one of {', '.join(FEATURES)}")
    _require_python()
    if feature == "asr-cpu":
        from . import _cpu
        ok, why = _cpu.available()
        if not ok:
            raise ImportError(f"{NAME} on the CPU: {why}. The CPU engine comes with the kitten-inference packages for "
                              "Linux (x86-64 and arm64), Windows x64 and Macs with Apple silicon, on Python 3.10 or newer; "
                              "there, reinstall it: pip install --force-reinstall kitten-inference")
        return
    _require_cuda_system(feature)
    if feature == "asr-process":
        _require_torch(load=False)
        _require_tokenizers(load=False)
        return
    torch = _require_torch()
    _require_gpu(torch, device)
    _require_tokenizers()


def attention_report() -> dict:
    """{kind: (usable here, reason)} for the attention kinds "triton", "torch" and "flash", for diagnostics.
    Needs PyTorch, but no GPU."""
    require_torch()
    from . import _attn
    return _attn.available()

