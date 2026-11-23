"""backend="cpu": speech recognition on the CPU (and the Mac GPU) with the engine in kitten_inference/_asr_cpu.

The engine is libkasr (engines/asr-cpu): llama.cpp's libraries (ggml, libllama, libmtmd) with a small C API, loaded
with ctypes. It needs no PyTorch and no tokenizers package: the tokenizer travels inside the model files.

On first use a model folder (the published bf16 safetensors) is converted with numpy into two GGUF files, the decoder
in the chosen precision and the audio encoder, cached under ``<Hugging Face cache home>/kitten-inference/asr``
(``KITTEN_ASR_CACHE`` overrides it). Later loads read the cache. Nothing is uploaded or re-hosted.

Precisions of the decoder: "q8_0" (8-bit weights, one scale per 32 weights; the default), "f16", and "bf16" (as
published). The audio encoder: "f16" (default), "q8_0", "bf16". Lower precisions lost accuracy in our tests and are
not offered (kitten_inference/asr/_precision.py says why for each).
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
from typing import Optional

import numpy as np

from . import _deps, _precision, _published, _system
from ._types import DecodeResult, EngineError, EngineStopped

API_VERSION = 1
LIB_ENV = "KITTEN_ASR_CPU_LIB"        # a development build of the library instead of the packaged one
CACHE_ENV = "KITTEN_ASR_CACHE"
DECODER_PRECISIONS = _precision.CPU
ENCODER_PRECISIONS = _precision.ENCODER_CPU
_AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
_STOP = ("<|im_end|>", "<|endoftext|>")
_MIN_SAMPLES = 8000                     # shorter audio is zero-padded to 0.5 s, as the GPU engine does
_WHOLE_CLIP_BELOW = 896                 # encoders narrower than this (tiny) attend over the whole clip
# One-token decode steps are memory-bound and end every op with a barrier across all threads: past 16 threads the
# barriers cost more than the extra bandwidth brings (Xeon Gold 6430, 32 cores: 16 threads 4.5 vs 32 threads
# 4.8 ms per step on tiny, 5.9 vs 6.2 on small-enhanced). The encoder and the prompt keep every thread.
_DECODE_THREADS_CAP = 16
# Windows of file audio decoded together (KittenASR.transcribe with a list, or one file longer than 30 s): each decode
# step reads the decoder's weights once for all of them instead of once per window. Batching changes the summation
# order inside attention, so a few transcripts change slightly; it is on by default only where the accuracy rule
# proved it: models with an audio encoder narrower than _WHOLE_CLIP_BELOW (tiny: every language it supports on the
# full FLEURS split, and the English set). Wider models (small-enhanced and up) opt in with batch=8; batched vs one
# window at a time showed no language worse for them, but 8 of 30 languages are not yet proven at the +0.5 margin.
_BATCH_DEFAULT = 8
# Apple Silicon: gpu=None uses Metal, except inside a macOS virtual machine. A VM's GPU is Apple's paravirtual device
# (GPU family Apple5, no simdgroup reduction, no simdgroup matrix multiply), on which ggml-metal does not support
# MUL_MAT, RMS_NORM, SOFT_MAX or FLASH_ATTN: every matrix product runs on the CPU with a CPU<->GPU handoff around each
# one. Measured in a macOS virtual machine (M2, 8 virtual CPUs): CPU 4 threads about 14-16x real time for tiny on
# English clips and 7.4x for small-enhanced (q8_0), Metal 1.8x / 1.4x. A bare-metal Mac (GPU family Apple7+) is not
# measured yet.
_MAC_GPU_DEFAULT = True


class _Params(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("n_threads", "n_threads_batch", "n_ctx", "n_batch", "flash_attn",
                                              "audio_window", "use_gpu", "kv_type", "warmup", "repack")]


class _Result(ctypes.Structure):
    _fields_ = [("tokens", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_int32), ("text", ctypes.c_char_p),
                ("n_prompt", ctypes.c_int32), ("n_audio_tokens", ctypes.c_int32), ("n_draft", ctypes.c_int32),
                ("n_accepted", ctypes.c_int32), ("finish", ctypes.c_int32), ("mel_ms", ctypes.c_double),
                ("encode_ms", ctypes.c_double), ("prefill_ms", ctypes.c_double), ("draft_ms", ctypes.c_double),
                ("decode_ms", ctypes.c_double), ("n_steps", ctypes.c_int32)]


class _StreamResult(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("cached_positions", "cached_blocks", "resyncs", "resync_tokens",
                                              "resync_accepted")]


def _stream_id(key: str) -> int:
    """A nonzero 63-bit number for a stream key (the engine keeps state for one stream id at a time)."""
    return (int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little") >> 1) or 1


_LIB = None
_LIB_LOCK = threading.Lock()
_DLL_DIR = None  # Keep the search directory available to dynamically loaded ggml backends.


def _native_path(path: str) -> str:
    if sys.platform != "win32":
        return path
    path = os.path.abspath(path)
    if len(path) < 200 or path.startswith(("\\\\?\\", "\\\\.\\")):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def library_path() -> str:
    override = os.environ.get(LIB_ENV)
    if override:
        return override
    name = {"win32": "kasr.dll", "darwin": "libkasr.dylib"}.get(sys.platform, "libkasr.so")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_asr_cpu", "lib", name)


def available() -> tuple:
    """(True, "") when the CPU engine library is in this package, else (False, why)."""
    if sys.platform == "win32" and not _deps.windows_x64():
        return False, "only Windows x64 is supported"
    path = _native_path(library_path())
    if not os.path.isfile(path):
        return False, (f"this kitten-inference package has no speech recognition CPU engine for "
                       f"{platform.system()} {platform.machine()} ({path} is missing)")
    return True, ""


def _lib():
    global _LIB, _DLL_DIR
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        ok, why = available()
        if not ok:
            raise ImportError(f"Kitten ASR on the CPU is not available: {why}")
        path = _native_path(library_path())
        dll_dir = None
        try:
            if sys.platform == "win32":
                dll_dir = os.add_dll_directory(os.path.dirname(path))
            lib = ctypes.CDLL(path)
        except OSError as exc:
            if dll_dir is not None:
                dll_dir.close()
            if sys.platform == "win32":
                raise ImportError(f"Kitten ASR on the CPU: the engine library could not be loaded ({exc}). Install the "
                                  "Microsoft Visual C++ Redistributable (x64) and check that the engine DLLs are together.") from None
            raise ImportError(f"Kitten ASR on the CPU: the engine library could not be loaded ({exc})") from None
        _DLL_DIR = dll_dir
        P, c = ctypes.POINTER, ctypes
        for name, res, args in (
                ("kasr_api_version", c.c_int32, []),
                ("kasr_last_error", c.c_char_p, []),
                ("kasr_backend_init", c.c_int32, [c.c_char_p, c.c_char_p]),
                ("kasr_info", c.c_char_p, [c.c_void_p]),
                ("kasr_default_params", _Params, []),
                ("kasr_load", c.c_void_p, [c.c_char_p, c.c_char_p, P(_Params)]),
                ("kasr_free", None, [c.c_void_p]),
                ("kasr_set_threads", c.c_int32, [c.c_void_p, c.c_int32, c.c_int32]),
                ("kasr_transcribe", c.c_int32, [c.c_void_p, P(c.c_float), c.c_int32, c.c_char_p, c.c_char_p,
                                                 P(c.c_int32), c.c_int32, c.c_int32, P(_Result)]),
                ("kasr_detokenize", c.c_char_p, [c.c_void_p, P(c.c_int32), c.c_int32]),
                ("kasr_quantize", c.c_int32, [c.c_char_p, c.c_char_p, c.c_char_p, c.c_int32])):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        try:   # many windows at once (newer libraries); without it, windows run one at a time
            fn = lib.kasr_transcribe_batch
            fn.restype = c.c_int32
            fn.argtypes = [c.c_void_p, c.c_int32, P(P(c.c_float)), P(c.c_int32), c.c_char_p, c.c_char_p, P(c.c_int32),
                           c.c_int32, P(_Result)]
        except AttributeError:
            pass
        # live audio (optional: a library without them transcribes every window from scratch)
        for name, res, args in (
                ("kasr_transcribe_stream", c.c_int32, [c.c_void_p, P(c.c_float), c.c_int32, c.c_char_p, c.c_char_p,
                                                        P(c.c_int32), c.c_int32, c.c_int32, c.c_int64,
                                                        P(_StreamResult), P(_Result)]),
                ("kasr_stream_release", None, [c.c_void_p, c.c_int64]),
                ("kasr_batch_yield", c.c_int32, [c.c_void_p, c.c_int32])):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.restype, fn.argtypes = res, args
        if lib.kasr_api_version() != API_VERSION:
            raise ImportError(f"Kitten ASR on the CPU: engine library API {lib.kasr_api_version()}, expected "
                              f"{API_VERSION}; reinstall kitten-inference")
        variant = os.environ.get("KITTEN_ASR_CPU_VARIANT") or None   # for measurements: force one CPU variant
        if not lib.kasr_backend_init(None, variant.encode() if variant else None):
            raise ImportError(f"Kitten ASR on the CPU: {_err(lib)}")
        _LIB = lib
        return lib


def _err(lib) -> str:
    msg = lib.kasr_last_error()
    return (msg or b"unknown error").decode("utf-8", "replace").strip()


# ------------------------------------------------------------------ CPU topology
def _sysctl_int(name: str) -> Optional[int]:
    """A whole-number sysctl value on macOS (None when the name does not exist, e.g. in a virtual machine)."""
    try:
        libc = ctypes.CDLL(None)
        v, n = ctypes.c_int64(0), ctypes.c_size_t(8)
        if libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(n), None, ctypes.c_size_t(0)) != 0:
            return None
        return int(v.value) if n.value in (4, 8) else None
    except (OSError, AttributeError):
        return None


def _mac_threads() -> int:
    """Apple Silicon: one thread per performance core. The efficiency cores are several times slower, and every op
    ends with a barrier across all threads, so a thread on a slow core holds the others back.
    A macOS virtual machine reports every virtual CPU as a performance core (no hw.perflevel1), although they run on
    the host's performance and efficiency cores and share them with the host: there, half of the CPUs."""
    perf = _sysctl_int("hw.perflevel0.physicalcpu")
    if perf and perf > 0 and _sysctl_int("hw.perflevel1.physicalcpu") is None and _sysctl_int("kern.hv_vmm_present") == 1:
        return max(1, perf // 2)
    for name in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        n = _sysctl_int(name)
        if n and n > 0:
            return n
    return os.cpu_count() or 1


def _fast_cpus() -> list:
    """CPUs this process may use, one per physical core, and only the fastest cores on a CPU with big and little
    cores (Linux). Elsewhere: every CPU the system reports."""
    if sys.platform == "win32":
        return _windows_fast_cpus()
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))
    root = "/sys/devices/system/cpu"
    seen, picked, speed = set(), [], {}
    for cpu in allowed:
        try:
            with open(f"{root}/cpu{cpu}/topology/core_id") as f:
                core = f.read().strip()
            with open(f"{root}/cpu{cpu}/topology/physical_package_id") as f:
                pkg = f.read().strip()
        except OSError:
            return allowed
        if (pkg, core) in seen:
            continue
        seen.add((pkg, core))
        picked.append(cpu)
        try:
            with open(f"{root}/cpu{cpu}/cpufreq/cpuinfo_max_freq") as f:
                speed[cpu] = int(f.read())
        except (OSError, ValueError):
            pass
    if len(speed) == len(picked) and speed:
        top = max(speed.values())
        picked = [c for c in picked if speed[c] >= 0.85 * top]
    return picked or allowed


def _windows_fast_cpus() -> list:
    from ctypes import wintypes
    from ..tts08 import _hybrid

    count = os.cpu_count() or 1
    allowed = list(range(count))
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        current = kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        affinity = kernel32.GetProcessAffinityMask
        mask = ctypes.c_size_t
        affinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(mask), ctypes.POINTER(mask)]
        affinity.restype = wintypes.BOOL
        process, system = mask(), mask()
        if affinity(current(), ctypes.byref(process), ctypes.byref(system)) and process.value:
            if count <= 64 or process.value != system.value:
                allowed = [i for i in range(64) if process.value & (1 << i)]
        if count <= 64 and _hybrid._is_intel():
            cpu_sets = [s for s in (_hybrid._windows_cpu_sets() or []) if s[1] in allowed]
            if _hybrid.windows_performance_threads(cpu_sets) is not None:
                fastest = max(s[3] for s in cpu_sets)
                return [s[1] for s in cpu_sets if s[3] == fastest]
    except (AttributeError, OSError):
        pass
    return allowed


def _cgroup_cpus(root: str = "/sys/fs/cgroup", proc: str = "/proc/self/cgroup") -> Optional[int]:
    """The CPU limit of this process's container (Linux cgroup v2 cpu.max, or v1 cfs quota/period), rounded up;
    None without a limit. sched_getaffinity does not show such a limit: a container limited to 2 CPUs on a 64-core
    host sees 64 CPUs, and 32 threads there would wait on each other at every step."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        quota, period = None, None
        try:
            with open(proc) as f:
                rel = next((ln.strip().split(":", 2)[2] for ln in f if ln.startswith("0::")), "/")
        except (OSError, IndexError):
            rel = "/"
        for path in (f"{root}{rel.rstrip('/')}/cpu.max", f"{root}/cpu.max"):
            try:
                with open(path) as f:
                    q, per = f.read().split()[:2]
                if q != "max":
                    quota, period = int(q), int(per)
                break
            except (OSError, ValueError):
                continue
        if quota is None:
            for d in (f"{root}/cpu,cpuacct", f"{root}/cpu"):
                try:
                    with open(f"{d}/cpu.cfs_quota_us") as f:
                        q = int(f.read())
                    with open(f"{d}/cpu.cfs_period_us") as f:
                        per = int(f.read())
                except (OSError, ValueError):
                    continue
                if q > 0 and per > 0:
                    quota, period = q, per
                break
        if quota and period and quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except Exception:        # never let a strange /sys stop the engine from starting
        return None
    return None


def default_threads() -> int:
    if sys.platform == "darwin":
        return max(1, _mac_threads())
    n = len(_fast_cpus())
    limit = _cgroup_cpus()
    return max(1, min(n, limit) if limit else n)


def gpu_default() -> bool:
    """Whether gpu=None means the Mac GPU: on Apple Silicon (Metal), unless KITTEN_ASR_GPU=0 or the Mac is a virtual
    machine (kern.hv_vmm_present = 1), whose paravirtual GPU runs the matrix products on the CPU (see above)."""
    if os.environ.get("KITTEN_ASR_GPU", "").strip() == "0":
        return False
    if not (sys.platform == "darwin" and platform.machine() == "arm64" and _MAC_GPU_DEFAULT):
        return False
    return _sysctl_int("kern.hv_vmm_present") != 1


# ------------------------------------------------------------------ model files
def _resolve(model, cache_dir=None, precision=None) -> str:
    model = os.fspath(model)
    if os.path.isdir(model):
        return model
    if os.path.exists(model):
        raise ValueError(f"{model} is a file; expected a model folder or a Hugging Face repo id")
    if model.startswith((".", "/", "~")) or "\\" in model or ":" in model or model.count("/") > 1:
        raise FileNotFoundError(f"model folder not found: {model}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(f"{model!r} is not a local folder, and downloading it needs huggingface-hub") from e
    try:
        return _published.download(snapshot_download, model, cache_dir, precision, required=True)
    except Exception as e:
        raise RuntimeError(f"{model!r} is not a local folder and could not be downloaded as a Hugging Face "
                           f"repo id ({type(e).__name__}: {e})") from e


def cache_root() -> str:
    root = os.environ.get(CACHE_ENV)
    if root:
        return root
    try:
        from huggingface_hub import constants
        home = constants.HF_HOME
    except Exception:
        home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "kitten-inference", "asr")


def _folder_key(folder: str, packed=None, native=False) -> str:
    h = hashlib.sha256()
    h.update(f"converter {_gguf_version()}\n".encode())
    for f in sorted(os.listdir(folder)):
        if f.endswith((".safetensors", ".json")):
            st = os.stat(os.path.join(folder, f))
            h.update(f"{f} {st.st_size} {int(st.st_mtime)}\n".encode())
    if packed:
        if native:
            h.update(b"native-packed-v1\n")
        h.update(os.path.basename(packed).encode())
        for name in (_published.META, _published.WEIGHTS):
            st = os.stat(os.path.join(packed, name))
            h.update(f"{name} {st.st_size} {st.st_mtime_ns}\n".encode())
    h.update(os.path.realpath(folder).encode())
    return h.hexdigest()[:16]


def _gguf_version() -> int:
    from . import _gguf
    return _gguf.CONVERTER_VERSION


def _window_for(folder: str) -> int:
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    d = int(cfg.get("thinker_config", cfg)["audio_config"]["d_model"])
    return 0 if d < _WHOLE_CLIP_BELOW else 800


def batch_default(folder: str) -> int:
    """Windows decoded together by default for this model folder: _BATCH_DEFAULT for tiny-class models, else 1."""
    return _BATCH_DEFAULT if _window_for(folder) == 0 else 1


def _write(tmp: str, final: str, make) -> None:
    """make(tmp), then rename it to final. A failed or interrupted make (Ctrl+C, a full disk) removes tmp: its name
    holds the process id, so nothing would reuse or clean it later."""
    try:
        make(tmp)
        _publish(tmp, final)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def _publish(tmp: str, final: str) -> None:
    deadline = time.monotonic() + 5
    while True:
        try:
            os.replace(tmp, final)
            return
        except OSError as exc:
            if sys.platform != "win32" or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            if os.path.isfile(final):  # Another process published the same file and may already have it open.
                return
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def prepare(folder: str, precision: str = "q8_0", encoder_precision: str = "f16", threads: Optional[int] = None,
            log=None) -> dict:
    """The GGUF files of `folder` at these precisions, converted on first use. Returns {"decoder", "encoder",
    "convert_s", "cached"}."""
    from . import _gguf
    _precision.check("cpu", precision)
    if encoder_precision not in ENCODER_PRECISIONS:
        raise ValueError(f"encoder_precision must be one of {', '.join(ENCODER_PRECISIONS)}, "
                         f"not {encoder_precision!r}")
    folder = _native_path(folder)
    packed = _published.variant(folder, precision, required=True)
    native = bool(packed and sys.platform != "darwin" and hasattr(_lib(), "kasr_has_packed_weights"))
    conversion_options = {"published": precision, "packed": native} if packed else {}
    out = _native_path(os.path.join(cache_root(), _folder_key(folder, packed, native)))
    os.makedirs(out, exist_ok=True)
    window = _window_for(folder)
    dec = os.path.join(out, f"decoder-{precision}.gguf")
    enc = os.path.join(out, f"encoder-{encoder_precision}-w{window}.gguf")
    t0 = time.perf_counter()
    cached = os.path.isfile(dec) and os.path.isfile(enc)
    tag = f"{os.getpid()}-{threading.get_ident()}"
    if not os.path.isfile(enc):
        if log:
            log(f"converting the audio encoder to {encoder_precision}")
        _write(f"{enc}.{tag}.tmp", enc, lambda t: _gguf.convert_encoder(folder, t, encoder_precision,
                                                                      window_frames=window, **conversion_options))
    if packed and not os.path.isfile(dec):
        if log:
            log(f"converting the published {precision} decoder")
        _write(f"{dec}.{tag}.tmp", dec, lambda t: _gguf.convert_decoder(folder, t, **conversion_options))
    if not packed and not os.path.isfile(dec):
        bf16 = os.path.join(out, "decoder-bf16.gguf")
        keep = os.environ.get("KITTEN_ASR_KEEP_BF16") == "1"
        # A Windows bf16 load must be able to open its persistent file while another process quantises.
        private = sys.platform == "win32" and not keep and precision != "bf16" and not os.path.isfile(bf16)
        source = f"{bf16}.{tag}.tmp" if private else bf16
        try:
            if not os.path.isfile(source):
                if log:
                    log("converting the decoder")
                if private:
                    _gguf.convert_decoder(folder, source)
                else:
                    _write(f"{bf16}.{tag}.tmp", bf16, lambda t: _gguf.convert_decoder(folder, t))
            if precision != "bf16" and not os.path.isfile(dec):
                if log:
                    log(f"quantising the decoder to {precision}")
                lib = _lib()

                def quantize(tmp):
                    if not lib.kasr_quantize(source.encode(), tmp.encode(), precision.encode(), int(threads or 0)):
                        raise EngineError(f"kitten-asr: {_err(lib)}")

                try:
                    _write(f"{dec}.{tag}.tmp", dec, quantize)
                except EngineError:
                    if not os.path.isfile(dec):
                        raise   # Another worker may have finished the same conversion.
                if sys.platform != "win32" and not keep:
                    with contextlib.suppress(OSError):
                        os.remove(bf16)
        finally:
            if private:
                with contextlib.suppress(OSError):
                    os.remove(source)
    return {"decoder": dec, "encoder": enc, "convert_s": round(time.perf_counter() - t0, 3), "cached": cached,
            "window_frames": window}


# ------------------------------------------------------------------ engine
class CpuEngine:
    """The CPU engine behind KittenASR(backend="cpu"): the same calls as the GPU engine (default_prompt, decode,
    release, snapshot, close) for one window at a time. Calls from several threads run one after another; each
    uses all of the engine's threads.

    precision          decoder precision: "q8_0" (default), "f16", "bf16", or published "int4" or "int8"
    encoder_precision  audio encoder precision (ENCODER_PRECISIONS), default "f16". "q8_0" is faster (about 1.2x on
                       Arm, 1-2% on x86) but costs accuracy: +0.1 to +0.3 WER points on FLEURS, and it fails the
                       accuracy rule in Macedonian with small-enhanced
    threads            threads for the audio encoder, the prompt and draft checks; default: one per fast physical core
    decode_threads     threads for the one-token decode steps; default: threads, at most 16
    audio_window       mel frames per encoder pass (800 = 8 s blocks, 0 = whole clip); default: the model's
    flash_attn         None (auto), True or False
    system_prompt      "auto" (default): the system message the folder's asr_config.json names (empty
                       without it); "none": always empty. See _system.py
    stream_cache       live audio: keep the decoder state of a window's prompt and complete 8 s audio blocks between
                       updates (exact: reused only when bit-identical); default True
    gpu                True: use the Mac GPU (Metal) when the library and the machine have one; None (default):
                       Metal on Apple Silicon (the CPU in a macOS virtual machine), the CPU elsewhere; False:
                       the CPU only
    batch              windows of file audio decoded together by decode_batch: 1.2-1.9x faster for many files or
                       long files. Default 8 for tiny on the CPU, 1 (one window at a time) for wider models until
                       their accuracy check is complete and on the Mac GPU (not measured there); batch=8 turns it on
    """

    name = "kitten-asr-cpu"

    def __init__(self, model, cache_dir=None, precision: str = "q8_0", encoder_precision: str = "f16",
                 threads: Optional[int] = None, decode_threads: Optional[int] = None,
                 audio_window: Optional[int] = None, flash_attn: Optional[bool] = None, gpu: Optional[bool] = None,
                 kv_type: Optional[int] = None, n_ctx: int = 2048, n_batch: int = 512, repack: bool = True,
                 warmup: bool = True, files: Optional[dict] = None, batch: Optional[int] = None,
                 stream_cache: bool = True, system_prompt: str = "auto", **unknown):
        if unknown:
            raise TypeError(f"unknown option(s) for the CPU engine: {', '.join(sorted(unknown))}")
        system_mode = _system.check_mode(system_prompt)
        if batch is not None and (isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 256):
            raise ValueError(f"batch must be a whole number from 1 to 256, not {batch!r}")
        lib = _lib()
        self._lib = lib
        self._lock = threading.Lock()
        self._ctx = None
        self._closed = False
        self._stream_fn = getattr(lib, "kasr_transcribe_stream", None) if stream_cache else None
        t0 = time.perf_counter()
        folder = _resolve(model, cache_dir, precision)
        self._system, self.system_prompt_source = _system.resolve(folder, system_mode)
        if files is None:
            files = prepare(folder, precision, encoder_precision, threads)
        self.files = dict(files)
        p = lib.kasr_default_params()
        all_threads = int(threads or default_threads())
        p.n_threads = int(decode_threads or min(all_threads, _DECODE_THREADS_CAP))
        p.n_threads_batch = all_threads
        p.n_ctx, p.n_batch = int(n_ctx), int(n_batch)
        p.flash_attn = -1 if flash_attn is None else int(bool(flash_attn))
        p.audio_window = -1 if audio_window is None else int(audio_window)
        p.use_gpu = int(gpu_default() if gpu is None else bool(gpu))
        p.kv_type = int(kv_type or 0)
        p.warmup = int(bool(warmup))
        p.repack = int(bool(repack))
        self.params = {k: getattr(p, k) for k, _ in _Params._fields_}
        ctx = lib.kasr_load(self.files["decoder"].encode(), self.files["encoder"].encode(), ctypes.byref(p))
        if not ctx:
            raise EngineError(f"kitten-asr: {_err(lib)}")
        self._ctx = ctx
        self.load_s = round(time.perf_counter() - t0, 3)
        self.precision, self.encoder_precision = precision, encoder_precision
        try:                # what the engine really runs on: it uses the CPU when it finds no GPU device
            native = json.loads(lib.kasr_info(ctx).decode("utf-8", "replace"))
            self.params["use_gpu"] = int(native.get("use_gpu", p.use_gpu))
        except (ValueError, AttributeError):
            pass
        if batch is None:   # the Mac (GPU or CPU) was not measured with batching: one window at a time there
            batch = 1 if (self.params["use_gpu"] or sys.platform == "darwin") else batch_default(folder)
        self.batch = int(batch) if hasattr(lib, "kasr_transcribe_batch") else 1
        self.stats = dict(batch_calls=0, requests=0, generated_tokens=0, draft_tokens=0, accepted_tokens=0, mel_ms=0.0,
                          encode_ms=0.0, prefill_ms=0.0, draft_ms=0.0, decode_ms=0.0, steps=0, cached_positions=0,
                          cached_blocks=0)

    # ------------------------------------------------------------ the GPU engine's calls
    def default_prompt(self, language: Optional[str] = None) -> str:
        self._check()
        base = (f"<|im_start|>system\n{self._system}<|im_end|>\n<|im_start|>user\n{_AUDIO}<|im_end|>\n"
                "<|im_start|>assistant\n")
        return base + (f"language {language}<asr_text>" if language else "")

    def decode(self, audio: np.ndarray, prompt: str, *, draft_ids=None, stream_key=None, max_new_tokens: int = 256,
               priority: int = 0, timeout_s: float = 30.0, on_tokens=None) -> DecodeResult:
        """Greedy transcription of one window (see the GPU engine's decode). stream_key, priority and timeout_s are
        accepted for the same calls; windows run one at a time."""
        if prompt.count(_AUDIO) != 1:
            raise ValueError("prompt must contain exactly one audio placeholder")
        head, tail = prompt.split(_AUDIO)
        x = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if len(x) < _MIN_SAMPLES:
            x = np.concatenate([x, np.zeros(_MIN_SAMPLES - len(x), np.float32)])
        draft = [int(t) for t in (draft_ids or [])]
        d_arr = (ctypes.c_int32 * len(draft))(*draft) if draft else None
        res, sres = _Result(), _StreamResult()
        with self._turn():
            self._check()
            args = (self._ctx, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), head.encode("utf-8"),
                    tail.encode("utf-8"), d_arr, len(draft), int(max_new_tokens))
            if stream_key is not None and self._stream_fn is not None:
                ok = self._stream_fn(*args, _stream_id(str(stream_key)), ctypes.byref(sres), ctypes.byref(res))
            else:
                ok = self._lib.kasr_transcribe(*args, ctypes.byref(res))
            if not ok:
                raise EngineError(f"kitten-asr: {_err(self._lib)}")
            ids = [res.tokens[i] for i in range(res.n_tokens)]
            text = (res.text or b"").decode("utf-8", "replace")
        info = dict(finish_reason=("stop", "length", "length")[res.finish], prompt_tokens=res.n_prompt,
                    audio_tokens=res.n_audio_tokens, draft_tokens=res.n_draft, accepted_tokens=res.n_accepted,
                    steps=res.n_steps, mel_ms=res.mel_ms, encode_ms=res.encode_ms, prefill_ms=res.prefill_ms,
                    draft_ms=res.draft_ms, decode_ms=res.decode_ms, cached_positions=sres.cached_positions,
                    cached_blocks=sres.cached_blocks, resyncs=sres.resyncs, resync_tokens=sres.resync_tokens,
                    resync_accepted=sres.resync_accepted)
        s = self.stats
        s["requests"] += 1
        s["generated_tokens"] += len(ids)
        s["draft_tokens"] += res.n_draft
        s["accepted_tokens"] += res.n_accepted
        s["steps"] += res.n_steps
        s["cached_positions"] += sres.cached_positions
        s["cached_blocks"] += sres.cached_blocks
        for k in ("mel_ms", "encode_ms", "prefill_ms", "draft_ms", "decode_ms"):
            s[k] += info[k]
        if on_tokens is not None:
            on_tokens(ids)
        return DecodeResult(text, ids, info)

    def decode_batch(self, audios, prompt: str, max_new_tokens) -> list:
        """Greedy transcription of several windows with the same prompt, up to `batch` of them decoded together.
        Returns one DecodeResult per window, in order. With batch=1 (or one window) each goes through decode."""
        if prompt.count(_AUDIO) != 1:
            raise ValueError("prompt must contain exactly one audio placeholder")
        budgets = [int(m) for m in max_new_tokens]
        if len(budgets) != len(audios):
            raise ValueError("one max_new_tokens per window")
        if self.batch <= 1 or len(audios) <= 1:
            # one window: the one-token kernels are as fast as the batched ones and its text stays as before
            return [self.decode(a, prompt, max_new_tokens=m) for a, m in zip(audios, budgets)]
        head, tail = prompt.split(_AUDIO)
        xs = []
        for a in audios:
            x = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
            if len(x) < _MIN_SAMPLES:
                x = np.concatenate([x, np.zeros(_MIN_SAMPLES - len(x), np.float32)])
            xs.append(x)
        n = len(xs)
        fp = ctypes.POINTER(ctypes.c_float)
        out = [None] * n
        todo = list(range(n))
        calls = 0
        while todo:
            m = len(todo)
            ptrs = (fp * m)(*[xs[k].ctypes.data_as(fp) for k in todo])
            lens = (ctypes.c_int32 * m)(*[len(xs[k]) for k in todo])
            mx = (ctypes.c_int32 * m)(*[budgets[k] for k in todo])
            res = (_Result * m)()
            if calls:
                self._let_others_in()
            with self._lock:
                self._check()
                ok = self._call_interruptibly(self._lib.kasr_transcribe_batch, self._ctx, m, ptrs, lens,
                                              head.encode("utf-8"), tail.encode("utf-8"), mx, int(self.batch), res)
                if not ok:
                    raise EngineError(f"kitten-asr: {_err(self._lib)}")
                left = []
                for k, r in zip(todo, res):
                    if r.finish < 0:          # not started: another caller had asked for a turn
                        left.append(k)
                        continue
                    out[k] = ([r.tokens[i] for i in range(r.n_tokens)], (r.text or b"").decode("utf-8", "replace"),
                              dict(finish_reason=("stop", "length", "length")[r.finish], prompt_tokens=r.n_prompt,
                                   audio_tokens=r.n_audio_tokens, draft_tokens=0, accepted_tokens=0,
                                   steps=r.n_steps, mel_ms=r.mel_ms, encode_ms=r.encode_ms,
                                   prefill_ms=r.prefill_ms, draft_ms=0.0, decode_ms=r.decode_ms))
            calls += 1
            todo = left
        s = self.stats
        s["batch_calls"] += calls
        results = []
        for ids, text, info in out:
            s["requests"] += 1
            s["generated_tokens"] += len(ids)
            s["steps"] += info["steps"]
            for k in ("mel_ms", "encode_ms", "prefill_ms", "decode_ms"):
                s[k] += info[k]
            results.append(DecodeResult(text, ids, info))
        return results

    # ------------------------------------------------------------ turns between callers
    # One call runs at a time (self._lock). A long kasr_transcribe_batch call (every window of an hour-long file) would
    # make a live session wait for all of it, so a caller that wants the engine first tells the library
    # (kasr_batch_yield +1): the batch call then starts no new window after its first, finishes the windows in flight and returns, the
    # caller gets its turn, and decode_batch sends the windows that were not started again.
    def _want(self, delta: int) -> int:
        fn = getattr(self._lib, "kasr_batch_yield", None)
        return fn(self._ctx, delta) if fn is not None and self._ctx else 0

    @contextlib.contextmanager
    def _turn(self):
        self._want(1)
        try:
            self._lock.acquire()
        finally:
            self._want(-1)
        try:
            yield
        finally:
            self._lock.release()

    def _let_others_in(self) -> None:
        """Before the next batch call: wait (briefly) until every caller that asked for a turn has taken it."""
        t_end = time.monotonic() + 5.0
        while self._want(0) > 0 and time.monotonic() < t_end:
            time.sleep(0.0005)

    def _call_interruptibly(self, fn, *args):
        """fn(*args) in a helper thread, waiting in this one: Ctrl+C (KeyboardInterrupt) then asks the batch to stop
        starting windows, waits for the windows in flight (seconds, not the whole file) and is raised."""
        box, finished = {}, threading.Event()

        def run():
            try:
                box["ok"] = fn(*args)
            except BaseException as exc:          # noqa: BLE001 (handed to the caller)
                box["exc"] = exc
            finally:
                finished.set()
        threading.Thread(target=run, name="kitten-asr-cpu-batch", daemon=True).start()
        interrupted = None
        while True:      # an Event, not Thread.join: an interrupted join can report a live thread as finished
            try:
                if finished.wait(0.2):
                    break
            except BaseException as exc:          # KeyboardInterrupt and the like
                if interrupted is None:
                    interrupted = exc
                    self._want(1)
        if interrupted is not None:
            self._want(-1)
            raise interrupted
        if "exc" in box:
            raise box["exc"]
        return box["ok"]

    def release(self, stream_key: str) -> None:
        """Drops the decoder state kept for `stream_key` (live audio)."""
        fn = getattr(self._lib, "kasr_stream_release", None)
        if fn is None or stream_key is None:
            return
        with self._turn():
            if self._ctx:
                fn(self._ctx, _stream_id(str(stream_key)))

    def set_threads(self, decode_threads: int, threads: Optional[int] = None) -> None:
        """Threads of the decode steps (and of the prompt, with `threads`) can change at any time, up to the counts
        the engine was loaded with; the audio encoder keeps its count."""
        with self._turn():
            self._check()
            if not self._lib.kasr_set_threads(self._ctx, int(decode_threads), int(threads or 0)):
                raise ValueError(_err(self._lib))

    def snapshot(self) -> dict:
        self._check()
        with self._turn():
            info = json.loads(self._lib.kasr_info(self._ctx).decode("utf-8", "replace"))
        return dict(self.stats, engine=self.name, precision=self.precision,
                    encoder_precision=self.encoder_precision, files=self.files, params=self.params, batch=self.batch,
                    stream_cache=self._stream_fn is not None,
                    system_prompt=self.system_prompt_source, system_prompt_chars=len(self._system),
                    load_s=self.load_s, native=info, attention="ggml")

    def close(self) -> None:
        with self._lock:
            if self._ctx:
                self._lib.kasr_free(self._ctx)
            self._ctx = None
            self._closed = True

    def _check(self):
        if self._closed or not self._ctx:
            raise EngineStopped("kitten-asr: the CPU engine is closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
