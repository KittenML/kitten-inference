"""Kitten speech recognition: :class:`kitten_inference.KittenASR` and the batching engine behind it."""

__version__ = "0.2.1"

import importlib as _importlib

# Importing this package needs no third-party module: every name below is loaded on first use. The backend says
# what is missing (Python 3.10; for NVIDIA GPUs Linux, PyTorch and the GPU; for the CPU its engine library) when a
# model is loaded, not when a name is imported.
_LAZY = {"KittenASR": "_api", "Transcript": "_api", "Segment": "_api", "StreamSession": "_api",
         "StreamUpdate": "_api", "Engine": "_engine", "EngineProcess": "_proc", "DecodeResult": "_types",
         "EngineError": "_types", "PoolExhausted": "_types", "EngineStopped": "_types"}

__all__ = ["KittenASR", "Transcript", "Segment", "StreamSession", "StreamUpdate", "Engine", "EngineProcess",
           "DecodeResult", "EngineError", "PoolExhausted", "EngineStopped", "__version__"]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value
