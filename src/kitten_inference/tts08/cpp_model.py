import contextlib
import json
import os
import platform
import struct
import sys
import threading

import espeakng_loader
import numpy as np
import phonemizer
import soundfile as sf
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from . import _espeak_path, _hybrid
from .preprocess import TextPreprocessor, chunk_text, normalize_text

_HERE = os.path.dirname(os.path.abspath(__file__))
# Windows: the Microsoft Visual C++ runtime files that espeak-ng.dll (from espeakng_loader) imports by these names.
# Python ships only some of them and a fresh Windows has none; phonemizer then reports "espeak not installed".
MSVC_RUNTIME = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")  # in load order
_msvc_loaded = []  # handles of the copies loaded from this package (kept for the life of the process)


def _load_msvc_runtime(folder=os.path.join(_HERE, "_msvc"), windll=None):
    """Loads this package's copy of each MSVC_RUNTIME file (the Windows wheels carry one in _msvc/) that Windows
    cannot load by name. An installed runtime is always used first: it can be newer, and other modules in the
    process share it. Returns the names loaded from ``folder``."""
    if windll is None:
        if sys.platform != "win32":
            return []
        import ctypes
        windll = ctypes.WinDLL
    loaded = []
    for name in MSVC_RUNTIME:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            windll(name)  # already loaded, or found next to python.exe, in System32 or an added DLL folder
            continue
        except OSError:
            pass
        try:
            _msvc_loaded.append(windll(path))
            loaded.append(name)
        except OSError:
            pass  # espeak's own error follows; the README names the redistributable
    return loaded


_load_msvc_runtime()
EspeakWrapper.set_library(espeakng_loader.get_library_path())


def _espeak_version():
    """espeakng_loader's version, which names the data copy (asked only when a copy is needed: 0.2.4 has no
    __version__, and the distribution lookup scans sys.path)."""
    version = getattr(espeakng_loader, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _dist_version
            version = _dist_version("espeakng_loader")  # a third-party distribution: its name never changes
        except Exception:
            version = "unknown"
    return version


_espeak_env_written = None  # the ESPEAK_DATA_PATH value this module set last


def _set_espeak_data_path(at_import=False):
    """Points espeak-ng at its data folder before it loads. A long path (Linux, macOS) or, on Windows, one with
    characters outside ASCII would make espeak-ng end the Python process; such a folder is replaced by a short copy
    (kitten_inference/tts08/_espeak_path.py), or a RuntimeError says what to do. After the import, a value someone
    else set is left alone."""
    global _espeak_env_written
    if not at_import and os.environ.get("ESPEAK_DATA_PATH") not in (None, _espeak_env_written):
        return
    path = _espeak_path.usable_data_path(espeakng_loader.get_data_path(), _espeak_version)
    os.environ["ESPEAK_DATA_PATH"] = _espeak_env_written = path


try:  # at import too, for code that starts phonemizer's espeak backend itself
    _set_espeak_data_path(at_import=True)
except (_espeak_path.EspeakPathError, OSError):
    # KittenTTS() raises the error with the fix.
    os.environ["ESPEAK_DATA_PATH"] = _espeak_env_written = espeakng_loader.get_data_path()


# Windows does not load a DLL whose full path has 260 characters or more, even with long paths enabled (CI: pip
# installed the engine at 273 characters on a runner with long paths on, and it did not load).
WINDOWS_MAX_PATH = 260


def _longest_engine_path(here=_HERE):
    """The longest path among the engine and the libraries bundled for it (Windows wheels keep them in a
    *.libs folder next to kitten_inference), or 0."""
    longest = 0
    site = os.path.dirname(os.path.dirname(here))
    try:
        for name in os.listdir(here):
            if name.startswith("_engine") or name == "_msvc":
                longest = max(longest, len(os.path.join(here, name)))
        msvc = os.path.join(here, "_msvc")
        if os.path.isdir(msvc):
            longest = max([longest] + [len(os.path.join(msvc, n)) for n in os.listdir(msvc)])
        for libs in os.listdir(site):
            if libs.endswith(".libs") and "tts08" in libs:
                folder = os.path.join(site, libs)
                longest = max([longest] + [len(os.path.join(folder, n)) for n in os.listdir(folder)])
    except OSError:
        pass
    return longest

SAMPLE_RATE = 24000
STYLE_DIM = 256
# The reference ONNX path (kittenml 0.9.6) ends the ids with "…" (10) then the pad (0), and drops the last
# 5000 samples of every chunk. Both are copied here so the two paths give the same audio length.
END_TOKEN = 10
TAIL_TRIM_SAMPLES = 5000

# low_memory=True: the C library's KT08_MEMORY_EXTREME_LOW (the phones' default), as Kitten TTS 2 mini/nano/pico use.
LOW_MEMORY_MODE = "extreme-low-memory"
DEFAULT_VOICE = "expr-voice-5-m"
DEFAULT_NATIVE_VOICE = "expr-voice-2-f"

DEFAULT_VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    return re.findall(r"\w+|[^\w\s]", text)


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {symbols[i]: i for i in range(len(symbols))}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


SKIP_CPU_CHECK = "KITTEN_TTS08_SKIP_CPU_CHECK"
_CPU_FEATURES = {
    "avx2": "AVX2",
    "fma": "FMA",
    "asimddp": "the ARMv8.2 dot-product instructions (asimddp)",
}
_CPU_HINT = {
    "avx2": "Intel CPUs have them since 2013, AMD CPUs since 2015",
    "asimddp": "a Raspberry Pi 5 has them, a Raspberry Pi 4 does not",
}


GLIBC_FLOOR = (2, 29)


def _glibc_version():
    """(major, minor) of the GNU C library, or None when this is not glibc or it cannot be told."""
    if not (sys.platform.startswith("linux") or sys.platform == "android"):
        return None
    try:
        name = os.confstr("CS_GNU_LIBC_VERSION")  # e.g. "glibc 2.31"
    except (ValueError, OSError, AttributeError):
        return None
    if not name or not name.startswith("glibc"):
        return None
    try:
        major, minor = name.split()[1].split(".")[:2]
        return (int(major), int(minor))
    except (IndexError, ValueError):
        return None


def _check_glibc(version=None):
    """espeak-ng (from espeakng_loader) needs glibc 2.29 or newer on Linux; an older glibc otherwise fails later
    with a confusing "espeak not installed on your system". Refuse it here with a clear message instead. pip can
    install the wheel on an older glibc because espeakng_loader's x86_64 wheel is tagged manylinux_2_17."""
    v = _glibc_version() if version is None else version
    if v is not None and v < GLIBC_FLOOR:
        raise RuntimeError(
            f"This system's glibc is {v[0]}.{v[1]}, older than the {GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]} that "
            "Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico need (the espeak-ng library they use for text). "
            "On glibc older than 2.29 "
            "(for example RHEL/Rocky 8, Debian 10, Ubuntu 18.04 or Amazon Linux 2) it would otherwise fail with "
            "\"espeak not installed on your system\". Use a Linux with glibc 2.29 or newer."
        )


def _pos_weight_rows(weights_path, name=b"bert_embed.pos_weight"):
    """The number of position rows in the text encoder's position table (bert_embed.pos_weight), read from the W1X1
    weights file, or None when it cannot be told. That is the most phoneme tokens (ids) one chunk may have; past it
    the engine reports a weights-size mismatch, which misleads users into re-downloading an intact model."""
    try:
        with open(weights_path, "rb") as f:
            if f.read(4) != b"W1X1":
                return None
            _version, count = struct.unpack("<II", f.read(8))
            offset = None
            for _ in range(count):
                entry = f.read(80)
                if len(entry) < 80:
                    return None
                nm = entry[:64].split(b"\0", 1)[0]
                if nm == name or (offset is None and nm.endswith(b".pos_weight")):
                    offset = struct.unpack("<QII", entry[64:80])[0]
                    if nm == name:
                        break
            if offset is None:
                return None
            f.seek(offset)
            head = f.read(4)
            if len(head) < 4:
                return None
            _dtype, _flags, ndim = struct.unpack("<BBH", head)
            if ndim < 1:
                return None
            rows = struct.unpack("<i", f.read(4))[0]
            return rows if rows > 0 else None
    except (OSError, struct.error):
        return None


def _cpu_requirements(path=os.path.join(_HERE, "_cpu_requirements.txt")):
    """The instruction sets the engine was compiled for, as the build wrote them ("native": the build machine's own
    CPU, which cannot be checked here). Empty when the file is missing (a build without it)."""
    try:
        with open(path, encoding="ascii") as f:
            return f.read().split()
    except OSError:
        return []


def _cpuinfo_features(text, machine):
    """The feature names /proc/cpuinfo lists for the first CPU ("flags" on x86, "Features" on Arm)."""
    key = "features" if machine in ("aarch64", "arm64", "armv8l") else "flags"
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip().lower() == key:
            return set(value.split())
    return None


def _cpu_features():
    """The features of this CPU that the engine's builds can need, or None when they cannot be told."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") or sys.platform == "android":
        if machine in ("aarch64", "arm64"):
            try:  # the kernel's view, also inside containers: getauxval(AT_HWCAP) & HWCAP_ASIMDDP
                import ctypes
                libc = ctypes.CDLL(None)
                libc.getauxval.restype = ctypes.c_ulong
                libc.getauxval.argtypes = [ctypes.c_ulong]
                return {"asimddp"} if libc.getauxval(16) & (1 << 20) else set()
            except (OSError, AttributeError):
                pass
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                return _cpuinfo_features(f.read(), machine)
        except OSError:
            return None
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        import ctypes
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        if present(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE (every AVX2 CPU also has FMA)
            return {"avx2", "fma"}
        # Windows versions that do not know these feature numbers answer "no" for all of them; SSE4.2 (38) is on
        # every x86-64 CPU that runs a current Windows, so "no" for it means this Windows cannot tell.
        return set() if present(38) else None
    if sys.platform == "darwin" and machine == "x86_64":
        import ctypes
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.c_void_p, ctypes.c_size_t]
        have = set()
        for feature, key in (("avx2", b"hw.optional.avx2_0"), ("fma", b"hw.optional.fma")):
            value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
            if libc.sysctlbyname(key, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
                return None
            if value.value:
                have.add(feature)
        return have
    return None


def _check_cpu(requirements=None, features=None):
    """Refuses to load the engine on a CPU without the instructions it was compiled for: it would stop the whole
    Python process with "Illegal instruction" instead of raising an error."""
    need = [f for f in (_cpu_requirements() if requirements is None else requirements) if f != "native"]
    if not need or os.environ.get(SKIP_CPU_CHECK):
        return
    have = _cpu_features() if features is None else features
    if have is None:
        return
    lacking = [f for f in need if f not in have]
    if lacking:
        hint = next((_CPU_HINT[f] for f in lacking if f in _CPU_HINT), "")
        raise RuntimeError(
            f"This CPU does not have {' and '.join(_CPU_FEATURES.get(f, f) for f in lacking)}, which the native "
            f"engine of Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico is built for{f' ({hint})' if hint else ''}. "
            f"Loading it would stop Python with \"Illegal instruction\". If this check is wrong, set {SKIP_CPU_CHECK}=1."
        )


_cpu_setup_lock = threading.Lock()
_cpu_setup_done = False


def _set_up_cpu(engine):
    """Once per process, before any model runs: on an Intel CPU with performance and efficiency cores, default to one
    thread per P-core logical CPU and let the faster cores take more of the work (kitten_inference/tts08/_hybrid.py).
    OMP_NUM_THREADS and set_num_threads() still choose the thread count."""
    global _cpu_setup_done
    with _cpu_setup_lock:
        if _cpu_setup_done:
            return
        _cpu_setup_done = True
        _warn_if_threads_ignored(engine)
        perf_threads = _hybrid.performance_threads()
        if not perf_threads:
            return
        if hasattr(engine, "set_dynamic_split"):
            engine.set_dynamic_split(_hybrid.DYNAMIC_SPLIT)
        threads = _hybrid.default_threads(perf_threads)
        if threads and hasattr(engine, "set_default_threads"):
            engine.set_default_threads(threads)


def _warn_if_threads_ignored(engine):
    """A build without OpenMP (today's macOS packages) runs on one CPU thread and ignores OMP_NUM_THREADS: say so
    once, rather than run slower than asked without a word."""
    requested = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    try:
        requested = int(requested)
    except ValueError:
        return
    max_threads = getattr(engine, "max_threads", None)
    # With OpenMP, max_threads() is OMP_NUM_THREADS here (before any default is set); without it, always 1.
    if requested > 1 and max_threads is not None and max_threads() == 1:
        import warnings
        warnings.warn(f"OMP_NUM_THREADS={requested} has no effect: this build of the native engine (Kitten TTS 0.8 "
                      "and Kitten TTS 2 mini, nano and pico) has no OpenMP and runs on one CPU thread. Today's macOS "
                      "packages are built this way.", RuntimeWarning, stacklevel=3)


def check_metal():
    """ValueError unless backend='metal' can run here (Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico: one
    message, before any download)."""
    engine = _import_engine()
    try:
        with _ENGINE_GATE.use(engine, "metal"):
            pass
    except ValueError:
        # set_backend() fails both on a computer without a Metal GPU and in a build without the Metal backend (for
        # example one compiled without Xcode's Metal compiler): name both.
        raise ValueError("backend='metal' is not available: this build of kitten-inference has no Metal backend, or "
                         "this computer has no Metal GPU (Apple silicon Macs have one). Pass backend='cpu' to run on "
                         "the CPU.") from None


def _import_engine():
    _check_cpu()
    try:
        from . import _engine
    except ImportError as exc:
        longest = _longest_engine_path() if sys.platform == "win32" else 0
        if longest >= WINDOWS_MAX_PATH:
            raise ImportError(
                f"This folder path is too long for Windows: the native engine's files have paths of up to {longest} "
                f"characters, and Windows loads them only from shorter ones ({WINDOWS_MAX_PATH - 1} at most). "
                "Use a shorter folder for the virtual environment (for example C:\\venvs\\kitten)."
            ) from exc
        raise ImportError(
            "The native engine (kitten_inference.tts08._engine) is missing or does not match "
            'this Python and platform. Reinstall it with: pip install --force-reinstall "kitten-inference"'
        ) from exc
    _set_up_cpu(_engine)
    return _engine


class _EngineGate:
    """Lets engine calls run at the same time only when that is safe.

    The engine has one process-wide backend, which a model's kernels use while it loads and runs, and the
    Metal backend records all GPU work into one shared command buffer. So CPU calls may overlap (one model
    per thread), while a Metal call, or a call on another backend than the running ones, waits until the
    engine is idle. Each call sets its own model's backend first.

    Calls get in in the order they arrive: a call waits while an earlier one is still waiting. So a Metal call
    is not passed by CPU calls that come after it, and CPU calls are not held back for ever by Metal calls that
    keep queuing (two threads generating on Metal always have one call waiting); CPU calls that arrive one
    after another still run together.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._running = 0
        self._exclusive = False
        self._backend = None
        self._next_ticket = 0  # handed to each arriving call
        self._serving = 0  # the earliest ticket still waiting
        self._abandoned = set()  # tickets whose wait was interrupted

    def _must_wait(self, backend, exclusive):
        if self._running == 0:
            return False
        return exclusive or self._exclusive or self._backend != backend

    def _advance(self):
        self._serving += 1
        while self._serving in self._abandoned:
            self._abandoned.discard(self._serving)
            self._serving += 1
        self._cond.notify_all()

    @contextlib.contextmanager
    def use(self, engine, backend):
        exclusive = backend != "cpu"
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            try:
                while ticket != self._serving or self._must_wait(backend, exclusive):
                    self._cond.wait()
            except BaseException:  # e.g. KeyboardInterrupt while waiting: give up the place in the queue
                if ticket == self._serving:
                    self._advance()
                else:
                    self._abandoned.add(ticket)
                raise
            try:
                if not self._running:
                    if not engine.set_backend(backend):
                        raise ValueError(
                            f"Backend {backend!r} is not available in this build of Kitten TTS 0.8."
                        )
                    self._backend = backend
                    self._exclusive = exclusive
                self._running += 1
            finally:
                self._advance()
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                self._cond.notify_all()


_ENGINE_GATE = _EngineGate()


def _has_int8_layers(arch_path):
    # int8 models use *_int8 layers (conv1d_int8, seqgemm_int8, attention_int8).
    with open(arch_path, encoding="utf-8") as f:
        nodes = json.load(f).get("nodes", [])
    return any(str(node.get("op", "")).endswith("_int8") for node in nodes)


def _default_backend(engine, arch_path):
    # Apple Silicon: Metal when a Metal device is available, except for int8 models, which run several times
    # faster on the CPU than on Metal. Everywhere else: the CPU.
    if sys.platform == "darwin" and platform.machine() == "arm64" and not _has_int8_layers(arch_path):
        try:
            with _ENGINE_GATE.use(engine, "metal"):
                return "metal"
        except ValueError:
            pass
    return "cpu"


def _coerce_voice_pack(values):
    pack = np.asarray(values, dtype=np.float32)
    if pack.size % STYLE_DIM != 0:
        raise ValueError(
            f"Voice style data has {pack.size} floats, which is not divisible by {STYLE_DIM}."
        )
    return pack.reshape(-1, STYLE_DIM)


def _ids_from_phonemes(text_cleaner, phonemes):
    """espeak phonemes -> the model's token ids (0, symbols, END_TOKEN, 0), as the reference ONNX path."""
    phonemes = basic_english_tokenize(phonemes)
    phonemes = " ".join(phonemes)
    tokens = text_cleaner(phonemes)
    tokens.insert(0, 0)
    tokens.append(END_TOKEN)
    tokens.append(0)
    return np.asarray(tokens, dtype=np.float32)


def _load_voice_bin(path):
    return _coerce_voice_pack(np.fromfile(path, dtype=np.float32))


class KittenTTS_1_Cpp:
    def __init__(
        self,
        arch_path,
        weights_path,
        voices_path=None,
        voice_paths=None,
        speed_priors=None,
        voice_aliases=None,
        backend=None,
        style_index="token_count",
        sample_rate=SAMPLE_RATE,
        precision=None,
        low_memory=False,
    ):
        """Initialize KittenTTS with the native engine. precision: None (the graph's own) or the precision the graph
        must have ("fp32"/"int8"). low_memory=True: the least memory, on the CPU (the C library's
        KT08_MEMORY_EXTREME_LOW, as Kitten TTS 2 mini/nano/pico's low_memory)."""
        _check_glibc()  # before espeak-ng is loaded below, so an old glibc fails with a clear message
        self.arch_path = arch_path
        self.weights_path = weights_path
        self.sample_rate = sample_rate
        self.style_index = style_index
        # Most phoneme tokens one chunk may have (the model's position table); past it the engine reports a
        # weights-size mismatch, so check it in Python and say what is really wrong.
        self._max_tokens = _pos_weight_rows(weights_path)

        aliases = dict(DEFAULT_VOICE_ALIASES)
        aliases.update(voice_aliases or {})
        self.voice_aliases = aliases
        self.speed_priors = speed_priors or {}
        self.preprocessor = TextPreprocessor(remove_punctuation=False)
        _set_espeak_data_path()  # before espeak-ng loads (EspeakBackend below); RuntimeError when it cannot
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        # One call at a time per model: the engine model's forward reuses its own buffers and speed setting, and
        # espeak (this model's phonemizer) is not thread-safe. Different models still run at the same time.
        self._lock = threading.Lock()

        self._voices_npz = None
        self._voice_packs = {}
        self._load_voices(voices_path, voice_paths or {})
        if self._voices_npz is None and not self._voice_packs:
            raise ValueError(
                "Native KittenTTS config must provide voices as a .npz file or voice_files."
            )

        self.available_voices = sorted(self._available_voice_names())
        self.all_voice_names = [
            alias for alias, target in self.voice_aliases.items()
            if target in self.available_voices
        ] or self.available_voices

        from . import _family
        backend = _family.check_backend("tts08", backend)
        graph_precision = "int8" if _has_int8_layers(arch_path) else "fp32"
        if precision is not None and precision != graph_precision:
            raise ValueError(f"precision={precision!r}, but this model is the {graph_precision} model")
        _family.check_gpu_rules("tts08", backend, graph_precision, low_memory)
        self.precision, self.low_memory = graph_precision, bool(low_memory)
        self._mi = _import_engine()
        self.backend = backend or ("cpu" if low_memory else _default_backend(self._mi, arch_path))
        # The engine allocates and uploads the weights for the backend that is set while the model loads.
        with _ENGINE_GATE.use(self._mi, self.backend):
            self.session = (self._mi.InferenceModel(arch_path, weights_path, LOW_MEMORY_MODE) if low_memory
                            else self._mi.InferenceModel(arch_path, weights_path))

    def _load_voices(self, voices_path, voice_paths):
        if voices_path:
            if str(voices_path).endswith(".npz"):
                self._voices_npz = np.load(voices_path)
            else:
                self._voice_packs[DEFAULT_NATIVE_VOICE] = _load_voice_bin(voices_path)

        for voice, path in voice_paths.items():
            self._voice_packs[voice] = _load_voice_bin(path)

    def _available_voice_names(self):
        names = set(self._voice_packs.keys())
        if self._voices_npz is not None:
            names.update(self._voices_npz.files)
        return names

    def _resolve_voice(self, voice):
        if voice not in self.voice_aliases and voice not in self.available_voices:
            # Any case, as Kitten TTS 2 mini/nano/pico: 'bella' is Bella.
            want = str(voice).casefold()
            voice = next((n for n in list(self.voice_aliases) + self.available_voices if n.casefold() == want), voice)
        voice = self.voice_aliases.get(voice, voice)
        if voice not in self.available_voices:
            choices = sorted(set(self.all_voice_names) | set(self.available_voices))
            raise ValueError(f"Voice {voice!r} not available. Choose from: {choices}")
        return voice

    def _phoneme_ids(self, text):
        return _ids_from_phonemes(self.text_cleaner, self.phonemizer.phonemize([text])[0])

    def _ids_from_phonemes(self, phonemes):
        return _ids_from_phonemes(self.text_cleaner, phonemes)

    def _style_row(self, text, n_ids, rows):
        if rows <= 0:
            return 0
        if self.style_index == "text_length":
            return min(len(text), rows - 1)
        if self.style_index == "phoneme_count":
            return min(max(0, n_ids - 2), rows - 1)
        if self.style_index != "token_count":
            raise ValueError(
                "style_index must be one of: 'token_count', 'phoneme_count', 'text_length'."
            )
        return min(max(0, n_ids), rows - 1)

    def _style_for_voice(self, voice, text, n_ids):
        if voice in self._voice_packs:
            pack = self._voice_packs[voice]
        elif self._voices_npz is not None and voice in self._voices_npz:
            pack = _coerce_voice_pack(self._voices_npz[voice])
        else:
            raise ValueError(f"Voice {voice!r} has no style data.")

        row = self._style_row(text, n_ids, pack.shape[0])
        style = pack[row]
        return style[:128].copy(), style[128:].copy()

    def _effective_speed(self, voice, speed):
        return float(speed) * float(self.speed_priors.get(voice, 1.0))

    def _set_native_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be a positive finite value")

        set_speed = getattr(self.session, "set_speed", None)
        if set_speed is None:
            if np.isclose(speed, 1.0):
                return
            raise RuntimeError("This build of the native engine does not support speed control.")
        set_speed(speed)

    def normalize_text(self, text, locale="en-US", return_spans=False):
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True):
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        if not out_chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(out_chunks, axis=-1)

    def generate_stream(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True, trim_lead=True, chunk_chars=None):
        """Yield one float32 array per chunk, as it is produced; play them back to back.

        chunk_chars (default None, the model's default): one sentence per chunk, as generate(); N: whole sentences
        packed into chunks of at most N characters (at most 400; a longer sentence is cut at spaces, each part ending
        with a comma as the package's chunk_text does). The same text gives the same chunks on every machine (the C
        library's kt08_stream runs the same chunker).
        trim_lead (default True): drop the first chunk's inaudible lead (exact zeros and noise under -50 dBFS,
        about 0.4 s), keeping a 20 ms pre-roll, so speech starts sooner. Every kept sample is unchanged; later chunks
        keep their lead, which is the pause between sentences. With chunk_chars=None the stream's audio is generate()'s,
        chunk by chunk, apart from that trim.
        """
        from . import _family, _stream
        chunk_chars = _family.check_chunk_chars(chunk_chars)
        trim = _stream.check_trim_lead(trim_lead)
        if clean_text:
            text = self.preprocessor(text)
        sentences = chunk_text(text)
        st = self._mi.stream
        voice = self._resolve_voice(voice)
        chunks = st.pack_sentences(sentences, chunk_chars or st.ONE_SENTENCE, st.TTS08_MAX_CHARS, True)

        def phonemize(chunk):
            with self._lock:
                ph = self.phonemizer.phonemize([chunk])[0]
            return [ph] if ph else []

        def synth(ph, chunk, first, _last):
            audio = self._synth_phonemes(ph, chunk, voice, speed)   # the chunk's text picks the style row
            if first and trim:
                audio = audio[st.lead_trim_offset(audio, self.sample_rate):]
            return audio

        yield from _stream.run(chunks, phonemize, synth)

    def stream_chunks(self, text, clean_text=True, chunk_chars=None):
        """The text chunks stream() speaks for this text (before phonemes), the same on every machine."""
        from . import _family
        chunk_chars = _family.check_chunk_chars(chunk_chars)
        st = self._mi.stream
        return st.pack_sentences(chunk_text(self.preprocessor(text) if clean_text else text),
                                 chunk_chars or st.ONE_SENTENCE, st.TTS08_MAX_CHARS, True)

    def _synth_phonemes(self, phonemes, text, voice, speed):
        """generate_single_chunk with the phonemes already made (text picks the style row on text_length models)."""
        effective_speed = self._effective_speed(voice, speed)
        with self._lock:
            ids = self._ids_from_phonemes(phonemes)
            if self._max_tokens is not None and len(ids) > self._max_tokens:
                raise ValueError(
                    f"this text becomes {len(ids)} phoneme tokens in one chunk, more than the model's limit of "
                    f"{self._max_tokens}. Use shorter sentences (chunks split on sentence breaks), or "
                    "clean_text=True so numbers and symbols are normalized rather than spelled out at length."
                )
            style_dec, style_pred = self._style_for_voice(voice, text, len(ids))
            with _ENGINE_GATE.use(self._mi, self.backend):
                self._set_native_speed(effective_speed)
                outputs = self.session.forward(
                    {
                        "phoneme_ids": ids,
                        "style_dec": style_dec,
                        "style_pred": style_pred,
                    }
                )
        audio = np.asarray(outputs["t_audio"], dtype=np.float32).reshape(-1)
        return audio[:-TAIL_TRIM_SAMPLES]

    def generate_single_chunk(self, text, voice=DEFAULT_VOICE, speed=1.0):
        voice = self._resolve_voice(voice)
        effective_speed = self._effective_speed(voice, speed)

        with self._lock:
            ids = self._phoneme_ids(text)
            if self._max_tokens is not None and len(ids) > self._max_tokens:
                raise ValueError(
                    f"this text becomes {len(ids)} phoneme tokens in one chunk, more than the model's limit of "
                    f"{self._max_tokens}. Use shorter sentences (chunks split on sentence breaks), or "
                    "clean_text=True so numbers and symbols are normalized rather than spelled out at length."
                )
            style_dec, style_pred = self._style_for_voice(voice, text, len(ids))
            with _ENGINE_GATE.use(self._mi, self.backend):
                self._set_native_speed(effective_speed)
                outputs = self.session.forward(
                    {
                        "phoneme_ids": ids,
                        "style_dec": style_dec,
                        "style_pred": style_pred,
                    }
                )
        audio = np.asarray(outputs["t_audio"], dtype=np.float32).reshape(-1)
        return audio[:-TAIL_TRIM_SAMPLES]

    def generate_to_file(
        self,
        text,
        output_path,
        voice=DEFAULT_VOICE,
        speed=1.0,
        sample_rate=SAMPLE_RATE,
        clean_text=True,
    ):
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")
