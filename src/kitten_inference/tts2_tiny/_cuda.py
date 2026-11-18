"""backend="cuda" for Kitten TTS 2 mini, nano and pico: the engine built for NVIDIA GPUs.

The library (libkt08_cuda.so, kt08_cuda.dll on Windows) ships in kitten_inference/_tts2_cuda/lib/ next to the Kitten
TTS 2 engine, in the Linux x86-64 and Windows x64 packages. It has the CUDA runtime and its own matrix kernels built
in, so it needs only an NVIDIA driver (R570 or newer). It is loaded only when a model asks for backend="cuda". Its C
API (engines/lite/engine/src/capi/kitten_graph.cpp) is the Python extension's InferenceModel: forward() takes and
returns float32 arrays by name, set_speed() sets the duration speed.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
from pathlib import Path

import numpy as np

API_VERSION = 1
LIB_NAME = "kt08_cuda.dll" if sys.platform == "win32" else "libkt08_cuda.so"
# A development build of the library can be used instead of the packaged one.
LIB_ENV = "KITTEN_INFERENCE_CUDA_LIB"

# The library has one set of GPU buffers per process: one call at a time, from any model or thread. Re-entrant: the
# garbage collector may free an old model (__del__) while this thread holds the lock.
_LOCK = threading.RLock()
_engine = None


def library_path() -> Path:
    override = os.environ.get(LIB_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "_tts2_cuda" / "lib" / LIB_NAME


def _supported_system() -> bool:
    machine = platform.machine().lower()
    return ((sys.platform.startswith("linux") and machine in ("x86_64", "amd64"))
            or (sys.platform == "win32" and machine in ("amd64", "x86_64")))


def engine():
    """The CUDA engine (loaded once), or ValueError saying why backend="cuda" cannot run here."""
    global _engine
    with _LOCK:
        if _engine is None:
            _engine = _CudaEngine(_load())
        return _engine


def _load():
    path = library_path()
    if not path.is_file():
        if not _supported_system():
            raise ValueError(f"backend='cuda' runs on Linux x86-64 and Windows x64 with an NVIDIA GPU, not on "
                             f"{platform.system()} {platform.machine()}. Pass backend='cpu' to run on the CPU.")
        raise ValueError(f"backend='cuda' is not available: this kitten-inference package has no NVIDIA GPU library "
                         f"({path}). Reinstall it with: pip install --force-reinstall \"kitten-inference\", or pass "
                         "backend='cpu' to run on the CPU.")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ValueError(f"backend='cuda' is not available: the NVIDIA GPU library could not be loaded ({exc}). "
                         "Pass backend='cpu' to run on the CPU.") from None
    c_char_p, c_int, c_void_p, i64 = ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64
    for name, res, args in (
            ("kt08g_api_version", c_int, []),
            ("kt08g_last_error", c_char_p, []),
            ("kt08g_init", c_int, [c_char_p]),
            ("kt08g_device", c_char_p, []),
            ("kt08g_load", c_void_p, [c_char_p, c_char_p]),
            ("kt08g_free", None, [c_void_p]),
            ("kt08g_set_speed", c_int, [c_void_p, ctypes.c_float]),
            ("kt08g_forward", c_int, [c_void_p, c_int, ctypes.POINTER(c_char_p), ctypes.POINTER(c_void_p),
                                      ctypes.POINTER(i64)]),
            ("kt08g_output_count", c_int, [c_void_p]),
            ("kt08g_output_name", c_char_p, [c_void_p, c_int]),
            ("kt08g_output_data", ctypes.POINTER(ctypes.c_float), [c_void_p, c_int, ctypes.POINTER(i64)])):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    if lib.kt08g_api_version() != API_VERSION:
        raise ValueError(f"backend='cuda' is not available: {path} does not match this kitten-inference version. "
                         "Reinstall it with: pip install --force-reinstall \"kitten-inference\"")
    if lib.kt08g_init(b"cuda") != 0:
        reason = lib.kt08g_last_error().decode("utf-8", "replace")
        raise ValueError(f"backend='cuda' is not available: {reason}. It needs an NVIDIA GPU and NVIDIA driver R570 "
                         "or newer. Pass backend='cpu' to run on the CPU.")
    return lib


def _check(lib, rc):
    if rc != 0:
        raise RuntimeError(lib.kt08g_last_error().decode("utf-8", "replace"))


class _CudaEngine:
    """The part of the Python extension's interface that KittenTTS2Tiny uses: InferenceModel(arch, weights)."""

    def __init__(self, lib):
        self._lib = lib
        self.device = lib.kt08g_device().decode("utf-8", "replace")

    def InferenceModel(self, arch_path, weights_path):  # noqa: N802 (the extension's name)
        return _CudaModel(self._lib, arch_path, weights_path)


class _CudaModel:
    def __init__(self, lib, arch_path, weights_path):
        self._lib = lib
        with _LOCK:
            self._handle = lib.kt08g_load(os.fsencode(arch_path), os.fsencode(weights_path))
            if not self._handle:
                raise RuntimeError(lib.kt08g_last_error().decode("utf-8", "replace"))

    def set_speed(self, speed):
        with _LOCK:
            _check(self._lib, self._lib.kt08g_set_speed(self._handle, float(speed)))

    def forward(self, inputs):
        """{name: array} -> {name: 1-D float32 array}, as the extension's dict forward."""
        arrays = [(name.encode("utf-8"), np.ascontiguousarray(value, dtype=np.float32))
                  for name, value in inputs.items()]
        n = len(arrays)
        names = (ctypes.c_char_p * n)(*[a[0] for a in arrays])
        data = (ctypes.c_void_p * n)(*[a[1].ctypes.data for a in arrays])
        sizes = (ctypes.c_int64 * n)(*[a[1].size for a in arrays])
        lib, out = self._lib, {}
        with _LOCK:
            _check(lib, lib.kt08g_forward(self._handle, n, names, data, sizes))
            for i in range(lib.kt08g_output_count(self._handle)):
                size = ctypes.c_int64()
                ptr = lib.kt08g_output_data(self._handle, i, ctypes.byref(size))
                name = lib.kt08g_output_name(self._handle, i).decode("utf-8")
                out[name] = np.ctypeslib.as_array(ptr, shape=(size.value,)).copy() if size.value else \
                    np.zeros(0, dtype=np.float32)
        return out

    def __del__(self):
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            try:
                with _LOCK:
                    self._lib.kt08g_free(handle)
            except Exception:  # interpreter shutdown
                pass
