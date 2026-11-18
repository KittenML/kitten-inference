"""The C library for apps (engines/lite/engine/include/kitten_tts08.h).

Source-only checks always run. The library checks need a shared build of it:
    cmake -S engines/lite/engine -B build/capi -DKITTEN_BUILD_C_LIBRARY=ON -DBUILD_SHARED_LIBS=ON \
          -DKITTEN_BUILD_PYTHON_MODULE=OFF -DENABLE_NATIVE_ARCH=OFF && cmake --build build/capi
    KITTEN_TTS08_CAPI_LIB=build/capi/libkitten_tts08.so python -m pytest tests/tts08/test_capi.py
With KITTEN_TTS08_CAPI_MODEL_DIR (a folder with kitten_fp32_15m.bin and voices_kitten_15m/, e.g. a snapshot of
KittenML/meownn-models) the synthesis checks also run; KITTEN_TTS08_CAPI_SAME_BUILD=1 (library built with the same CMake
flags as the installed package, including -DKITTEN_C_LIBRARY_LTO=ON like the wheel's extension) makes them require
byte-identical audio with the Python package.

The phone defaults (threads 0 = the fast cores, low-memory mode) are compiled only into Android and iOS builds; on a
desktop build, KITTEN_TTS08_PHONE_DEFAULTS=1 turns them on for these tests, with a fake CPU tree given by
KITTEN_TTS08_SYS_ROOT and KITTEN_TTS08_CPUINFO. The core-selection rules are also compiled on their own with the system
C++ compiler and run against fake big.LITTLE, all-equal and three-cluster trees.
"""
import contextlib
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines" / "lite" / "engine"
CAPI_SRC = ENGINE / "src" / "capi" / "kitten_tts08.cpp"

# Core designs (MIDR implementer 0x41 = Arm, part numbers).
A53, A55, A75, A76, A78, A510, A710, X2 = 0xD03, 0xD05, 0xD0A, 0xD0B, 0xD41, 0xD46, 0xD47, 0xD48


def fake_cpu_tree(root, cores, midr_files=True, cpuinfo=True, capacity_files=True):
    """Writes a sysfs-like tree and a /proc/cpuinfo for cores = [(cpu, capacity_or_khz, part)]. A capacity above 100000 is
    written as cpufreq/cpuinfo_max_freq (kernels without cpu_capacity). Also plants a misleading topology/core_type
    file, which the engine must not read (Linux has no such file)."""
    root = Path(root)
    lines = []
    for cpu, cap, part in cores:
        d = root / "devices" / "system" / "cpu" / ("cpu%d" % cpu)
        (d / "topology").mkdir(parents=True, exist_ok=True)
        (d / "topology" / "core_type").write_text("performance\n" if cap < 600 else "efficiency\n")
        if cap and capacity_files:
            if cap > 100000:
                (d / "cpufreq").mkdir(exist_ok=True)
                (d / "cpufreq" / "cpuinfo_max_freq").write_text("%d\n" % cap)
            else:
                (d / "cpu_capacity").write_text("%d\n" % cap)
        if part and midr_files:
            (d / "regs" / "identification").mkdir(parents=True, exist_ok=True)
            (d / "regs" / "identification" / "midr_el1").write_text("0x00000000%02x1f%03x4\n" % (0x41, part))
        lines += ["processor\t: %d" % cpu, "BogoMIPS\t: 52.00", "CPU implementer\t: 0x41", "CPU architecture: 8",
                  "CPU variant\t: 0x1", "CPU part\t: 0x%03x" % part, "CPU revision\t: 4", ""]
    info = root / "cpuinfo"
    info.write_text("\n".join(lines) + "\n" if cpuinfo else "")
    return str(root), str(info)


AUTO_DRIVER = r"""
#include "engine/auto_threads.hpp"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "apple") {
        std::printf("%s\n", kt_describe_auto_threads(kt_auto_threads_apple(std::atoi(argv[2]), std::atoi(argv[3]))).c_str());
        return 0;
    }
    std::vector<int> allowed;
    for (const char* p = argv[3]; *p;) { char* e; allowed.push_back((int)std::strtol(p, &e, 10)); p = *e ? e + 1 : e; }
    std::printf("%s\n", kt_describe_auto_threads(kt_auto_threads_linux(kt_read_cores(argv[1], argv[2], allowed))).c_str());
    return 0;
}
"""


@contextlib.contextmanager
def env(**values):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def captured_stderr():
    """What the C library writes to stderr (file descriptor 2) inside the block."""
    with tempfile.TemporaryFile() as f:
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(f.fileno(), 2)
        out = []
        try:
            yield out
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            f.seek(0)
            out.append(f.read().decode(errors="replace"))
GEN = ROOT / "engines" / "lite" / "mobile" / "gen_symbols.py"

# Phoneme strings with what Python's tokenizer does to them: stress marks, length marks, punctuation runs, combining
# marks (U+0329), arrows, digits, symbols outside the table, repeated and odd white space.
PHONEME_CASES = [
    "həlˈoʊ wˈɜːld!",
    "ðɪs ɪz ɐ tˈɛst, wɪð pˈʌŋktʃuːˌeɪʃən; ænd “kwoʊts”…",
    "  lˈiːdɪŋ   ænd tɹˈeɪlɪŋ spˈeɪsᵻz  ",
    "n̩ ɹ̩ m̩ ↓↑→↗↘ ˈ ˌ ː ʰ",
    "1 2 3 tuː θɹˈiː _ ˈæ",
    "ab\tcd\nef gh ij",
    "¿ ¡ « » — … \" ' ; : , . ! ?",
    "αβγ дж ʒ ʤ ʧ ç ø œ ɶ",
    "日本 ñ é ü ★ ☃ 😀",
    "",
]


def python_ids(phonemes):
    """The Python package's mapping (cpp_model.py: basic_english_tokenize, TextCleaner, start/end tokens)."""
    from kitten_inference.tts08.cpp_model import TextCleaner, basic_english_tokenize

    tokens = TextCleaner()(" ".join(basic_english_tokenize(phonemes)))
    return [0] + tokens + [10, 0]


class AutoThreadRules(unittest.TestCase):
    """The phone default for threads = 0 (engine/auto_threads.hpp), compiled on its own against fake CPU trees."""

    @classmethod
    def setUpClass(cls):
        cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cxx or not (ENGINE / "src" / "engine" / "auto_threads.hpp").is_file():
            raise unittest.SkipTest("needs a C++ compiler and a source checkout")
        cls.tmp = tempfile.TemporaryDirectory()
        src = Path(cls.tmp.name) / "driver.cpp"
        src.write_text(AUTO_DRIVER)
        cls.exe = str(Path(cls.tmp.name) / "driver")
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", str(ENGINE / "src"), str(src), "-o", cls.exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(r.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def plan(self, cores, allowed=None, **kw):
        with tempfile.TemporaryDirectory() as d:
            root, info = fake_cpu_tree(d, cores, **kw)
            allowed = allowed if allowed is not None else [c for c, _, _ in cores]
            out = subprocess.run([self.exe, root, info, ",".join(map(str, allowed))], capture_output=True, text=True,
                                 check=True).stdout.strip()
        return dict(kv.split("=", 1) for kv in out.split())

    def test_big_little_two_fast_cores(self):  # Galaxy A16: 6x A55 + 2x A76
        p = self.plan([(c, 348, A55) for c in range(6)] + [(6, 1024, A76), (7, 1024, A76)])
        self.assertEqual(p, {"threads": "2", "cpus": "6,7", "mixed": "0", "rule": "big-cores"})

    def test_big_little_four_fast_cores(self):  # Galaxy A36: 4x A55 + 4x A78: at most 2 threads
        p = self.plan([(c, 405, A55) for c in range(4)] + [(c, 1024, A78) for c in range(4, 8)])
        self.assertEqual(p, {"threads": "2", "cpus": "4,5", "mixed": "0", "rule": "big-cores"})

    def test_three_clusters(self):  # 1 prime + 3 big + 4 little: the prime core and one big core, dynamic split
        p = self.plan([(c, 325, A510) for c in range(4)] + [(c, 870, A710) for c in range(4, 7)] + [(7, 1024, X2)])
        self.assertEqual(p, {"threads": "2", "cpus": "7,4", "mixed": "1", "rule": "big-cores"})

    def test_three_clusters_close_prime(self):  # Galaxy A73: prime 1024 and big 1003 count as one fast cluster
        p = self.plan([(c, 397, A55) for c in range(4)] + [(c, 1003, A78) for c in range(4, 7)] + [(7, 1024, A78)])
        self.assertEqual(p, {"threads": "2", "cpus": "7,4", "mixed": "0", "rule": "big-cores"})

    def test_only_allowed_cores(self):  # an app or OS cpuset that leaves out a core
        p = self.plan([(c, 397, A55) for c in range(4)] + [(c, 1003, A78) for c in range(4, 7)] + [(7, 1024, A78)],
                      allowed=[0, 1, 2, 3, 5, 6])
        self.assertEqual(p, {"threads": "2", "cpus": "5,6", "mixed": "0", "rule": "big-cores"})

    def test_frequency_only(self):  # Pixel 3a: no cpu_capacity; 6x A55 at 1.7 GHz + 2x A75 at 2.0 GHz
        p = self.plan([(c, 1708800, A55) for c in range(6)] + [(6, 1996800, A75), (7, 1996800, A75)])
        self.assertEqual(p, {"threads": "2", "cpus": "6,7", "mixed": "0", "rule": "big-cores"})

    def test_all_equal_cores(self):  # LG Stylo 5: 8x A53, one speed: every core
        p = self.plan([(c, 1024, A53) for c in range(8)])
        self.assertEqual(p, {"threads": "8", "cpus": "0,1,2,3,4,5,6,7", "mixed": "0", "rule": "same-design"})

    def test_same_design_two_speeds(self):  # Moto G22: 4x A53 2.3 GHz + 4x A53 1.8 GHz, design from /proc/cpuinfo only
        p = self.plan([(c, 1024, A53) for c in range(4)] + [(c, 801, A53) for c in range(4, 8)], midr_files=False)
        self.assertEqual(p, {"threads": "8", "cpus": "0,1,2,3,4,5,6,7", "mixed": "1", "rule": "same-design"})

    def test_same_design_capped_at_8(self):
        p = self.plan([(c, 1024, A55) for c in range(12)])
        self.assertEqual((p["threads"], p["rule"]), ("8", "same-design"))

    def test_nothing_known(self):  # no capacity, no design: same speed, every core up to 8
        p = self.plan([(c, 0, 0) for c in range(4)], midr_files=False, cpuinfo=False)
        self.assertEqual(p, {"threads": "4", "cpus": "0,1,2,3", "mixed": "0", "rule": "same-speed"})

    def test_mixed_designs_unknown_speed(self):  # designs differ but no speed: cannot tell the fast ones, use all
        p = self.plan([(0, 0, A55), (1, 0, A55), (2, 0, A76), (3, 0, A76)])
        self.assertEqual((p["threads"], p["rule"]), ("4", "same-speed"))

    def test_single_core(self):
        self.assertEqual(self.plan([(3, 1024, A53)])["threads"], "1")
        self.assertEqual(self.plan([], allowed=[])["threads"], "0")

    def test_apple(self):
        def apple(perf, total):
            out = subprocess.run([self.exe, "apple", str(perf), str(total)], capture_output=True, text=True, check=True)
            return out.stdout.split()[0]
        self.assertEqual(apple(2, 6), "threads=2")   # A13: 2 performance + 4 efficiency cores
        self.assertEqual(apple(4, 10), "threads=2")  # M1 iPad
        self.assertEqual(apple(0, 2), "threads=2")   # A10: no perflevel sysctl
        self.assertEqual(apple(0, 1), "threads=1")
        self.assertEqual(apple(0, 0), "threads=0")   # nothing known: OpenMP default


class SourceTests(unittest.TestCase):
    def test_phone_defaults_only_in_phone_builds(self):
        src = CAPI_SRC.read_text(encoding="utf-8")
        self.assertIn("#if defined(__ANDROID__) || (defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE)\n"
                      "#define KT08_PHONE_BUILD 1", src)
        self.assertIn("options->memory_mode = phone_defaults() ? kPhoneMemoryMode : KT08_MEMORY_LATENCY;", src)
        # The Python package never goes through the C API, so its defaults cannot change.
        bindings = (ENGINE / "python" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertNotIn("auto_threads", bindings)
        self.assertNotIn("phone_defaults", bindings)

    def test_symbol_table_matches_python(self):
        if not GEN.is_file():
            self.skipTest("not a source checkout")
        r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_c_library_is_off_by_default_and_outside_the_wheel(self):
        cmake = (ENGINE / "CMakeLists.txt").read_text(encoding="utf-8")
        for option in ("KITTEN_BUILD_C_LIBRARY", "ENABLE_ARM_ISA_DISPATCH", "KITTEN_ACCELERATE", "KITTEN_ACCELERATE_DEFAULT"):
            self.assertRegex(cmake, r'option\(%s "[^"]*" OFF\)' % option)
        # The wheels install only the "python" component; the C library and example install as "c_library".
        for block in re.findall(r"install\(TARGETS kitten_tts08.*?\)\n|install\(TARGETS kt08_example.*?\)\n", cmake, re.S):
            self.assertNotIn("COMPONENT python", block)
            self.assertIn("COMPONENT c_library", block)
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('install.components = ["python"]', pyproject)
        self.assertIn('wheel.packages = ["src/kitten_inference"]', pyproject)
        self.assertNotIn("KITTEN_BUILD_C_LIBRARY", pyproject)


class Options(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("threads", ctypes.c_int32), ("memory_mode", ctypes.c_int),
                ("fp16_weights", ctypes.c_int32), ("fp16_compute", ctypes.c_int32)]


LATENCY, LOW, EXTREME_LOW = 0, 1, 2


@unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_LIB"), "set KITTEN_TTS08_CAPI_LIB to a shared build of the C library")
class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib = ctypes.CDLL(os.environ["KITTEN_TTS08_CAPI_LIB"])
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        lib.kt08_api_version.restype = ctypes.c_int
        lib.kt08_last_error.restype = ctypes.c_char_p
        lib.kt08_build_info.restype = ctypes.c_char_p
        lib.kt08_phonemes_to_ids.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, c_size_p]
        lib.kt08_style_row_for_text.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.kt08_style_row_for_text.restype = ctypes.c_size_t
        lib.kt08_voice_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), c_size_p]
        lib.kt08_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.kt08_model_max_tokens.argtypes = [ctypes.c_void_p]
        lib.kt08_model_max_tokens.restype = ctypes.c_size_t
        lib.kt08_model_free.argtypes = [ctypes.c_void_p]
        lib.kt08_synthesize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), c_size_p]
        lib.kt08_free.argtypes = [ctypes.c_void_p]
        lib.kt08_options_init.argtypes = [ctypes.POINTER(Options)]
        cls.lib = lib

    def options(self, **fields):
        opt = Options()
        self.lib.kt08_options_init(ctypes.byref(opt))
        for k, v in fields.items():
            setattr(opt, k, v)
        return opt

    def nano_paths(self):
        from kitten_inference.tts08 import _engine

        d = Path(os.environ["KITTEN_TTS08_CAPI_MODEL_DIR"])
        arch = Path(_engine.__file__).parent / "model_defs" / "kitten_fp32_15m_arch.json"
        return arch, d / "kitten_fp32_15m.bin", d / "voices_kitten_15m" / "expr-voice-2-f.bin"

    def synth_bytes(self, opt, texts=(("Hello there.", "həlˈoʊ ðˈɛɹ."), ("A test, please!", "ɐ tˈɛst, plˈiːz!"))):
        """Loads nano fp32 with these options and returns the audio bytes of each text."""
        arch, weights, voice_path = self.nano_paths()
        model = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_model_load(str(arch).encode(), str(weights).encode(),
                                                  None if opt is None else ctypes.byref(opt), ctypes.byref(model)),
                         0, self.lib.kt08_last_error())
        voice = ctypes.POINTER(ctypes.c_float)()
        rows = ctypes.c_size_t()
        out_all = []
        try:
            self.assertEqual(self.lib.kt08_voice_load(str(voice_path).encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
            for text, phonemes in texts:
                ids = self.ids(phonemes)
                arr = (ctypes.c_int32 * len(ids))(*ids)
                out = ctypes.POINTER(ctypes.c_float)()
                n = ctypes.c_size_t()
                row = self.lib.kt08_style_row_for_text(text.encode(), rows.value)
                self.assertEqual(self.lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, row, 1.0,
                                                          ctypes.byref(out), ctypes.byref(n)), 0, self.lib.kt08_last_error())
                out_all.append(ctypes.string_at(out, n.value * 4))
                self.lib.kt08_free(out)
            self.lib.kt08_free(voice)
        finally:
            self.lib.kt08_model_free(model)
        self.assertTrue(all(len(b) > 4000 for b in out_all))
        return out_all

    def test_desktop_defaults_unchanged(self):
        with env(KITTEN_TTS08_PHONE_DEFAULTS=None):
            opt = self.options()
        self.assertEqual((opt.struct_size, opt.threads, opt.memory_mode, opt.fp16_weights, opt.fp16_compute),
                         (ctypes.sizeof(Options), 0, LATENCY, 0, 0))

    def test_phone_default_memory_mode(self):
        with env(KITTEN_TTS08_PHONE_DEFAULTS="1"):
            opt = self.options()
        self.assertEqual((opt.threads, opt.memory_mode), (0, EXTREME_LOW))
        with env(KITTEN_TTS08_PHONE_DEFAULTS="0"):
            self.assertEqual(self.options().memory_mode, LATENCY)

    @unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_MODEL_DIR"), "set KITTEN_TTS08_CAPI_MODEL_DIR for synthesis")
    def test_extreme_speed_is_refused_and_the_model_still_works(self):
        # A speed far below any useful value would stretch one token past 1000 frames (and overflow the float-to-int
        # frame count at 1e-30): the call fails with an error instead, and the next call gives the normal audio.
        before = self.synth_bytes(self.options())
        arch, weights, voice_path = self.nano_paths()
        model = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_model_load(str(arch).encode(), str(weights).encode(), None, ctypes.byref(model)), 0)
        voice, rows = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
        try:
            self.assertEqual(self.lib.kt08_voice_load(str(voice_path).encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
            ids = self.ids("həlˈoʊ ðˈɛɹ.")
            arr = (ctypes.c_int32 * len(ids))(*ids)
            out, n = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
            row = self.lib.kt08_style_row_for_text(b"Hello there.", rows.value)
            for speed in (1e-30, 1e-4):
                self.assertEqual(self.lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, row, speed,
                                                          ctypes.byref(out), ctypes.byref(n)), 4, speed)  # KT08_ERR_RUNTIME
                self.assertFalse(out)
            self.assertIn(b"speed", self.lib.kt08_last_error())
            self.assertEqual(self.lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, row, 1.0,
                                                      ctypes.byref(out), ctypes.byref(n)), 0, self.lib.kt08_last_error())
            self.assertEqual(ctypes.string_at(out, n.value * 4), before[0])
            self.lib.kt08_free(out)
            self.lib.kt08_free(voice)
        finally:
            self.lib.kt08_model_free(model)

    @unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_MODEL_DIR"), "set KITTEN_TTS08_CAPI_MODEL_DIR for synthesis")
    def test_mmap_and_fread_give_the_same_audio(self):
        for threads in (1, 2):
            with self.subTest(threads=threads):
                with env(KITTEN_WEIGHTS_MMAP=None):
                    mapped = self.synth_bytes(self.options(threads=threads))
                with env(KITTEN_WEIGHTS_MMAP="0"):
                    read = self.synth_bytes(self.options(threads=threads))
                self.assertEqual(mapped, read)

    @unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_MODEL_DIR"), "set KITTEN_TTS08_CAPI_MODEL_DIR for synthesis")
    def test_memory_modes_give_the_same_audio(self):
        for threads in (1, 2):
            with self.subTest(threads=threads):
                low = self.synth_bytes(self.options(threads=threads, memory_mode=LOW))
                self.assertEqual(self.synth_bytes(self.options(threads=threads, memory_mode=EXTREME_LOW)), low)
                latency = self.synth_bytes(self.options(threads=threads, memory_mode=LATENCY))
                if platform.machine().lower() in ("x86_64", "amd64"):
                    # x86 uses direct convolution in every mode. On ARM, latency mode uses Winograd convolution (faster,
                    # 131 MiB more memory), which rounds differently: the same audio within float rounding.
                    self.assertEqual(latency, low)

    @unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_MODEL_DIR"), "set KITTEN_TTS08_CAPI_MODEL_DIR for synthesis")
    def test_phone_auto_threads_choose_the_fast_cores(self):
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        if len(allowed) < 4:
            self.skipTest("needs 4 or more CPUs")
        fast = allowed[-2:]
        cores = [(c, 1024 if c in fast else 400, A76 if c in fast else A55) for c in allowed]
        with tempfile.TemporaryDirectory() as d:
            root, info = fake_cpu_tree(d, cores)
            with env(KITTEN_TTS08_PHONE_DEFAULTS="1", KITTEN_TTS08_SYS_ROOT=root, KITTEN_TTS08_CPUINFO=info,
                     KITTEN_TTS08_VERBOSE="1"):
                with captured_stderr() as err:
                    auto = self.synth_bytes(self.options())
                self.assertIn("kt08: auto threads=2 cpus=%d,%d mixed=0 rule=big-cores" % (fast[0], fast[1]), err[0])
                # The caller's thread keeps its own CPU set after the calls.
                self.assertEqual(sorted(os.sched_getaffinity(0)), allowed)
                # An explicit thread count is used as given, without choosing cores.
                with captured_stderr() as err:
                    explicit = self.synth_bytes(self.options(threads=2))
                self.assertNotIn("kt08: auto", err[0])
        # Lossless: the same audio as 2 threads chosen by hand, in the same memory mode.
        self.assertEqual(auto, explicit)

    def ids(self, phonemes):
        raw = phonemes.encode("utf-8")
        n = ctypes.c_size_t()
        self.assertEqual(self.lib.kt08_phonemes_to_ids(raw, None, 0, ctypes.byref(n)), 0)
        buf = (ctypes.c_int32 * n.value)()
        self.assertEqual(self.lib.kt08_phonemes_to_ids(raw, buf, n.value, ctypes.byref(n)), 0)
        return list(buf)

    def test_version_and_build_info(self):
        self.assertEqual(self.lib.kt08_api_version(), 5)  # 2: Kitten TTS 2 mini, nano and pico; 3: their NPU path; 4: text to phonemes (0.8); 5: text front end (mini, nano, pico)
        self.assertIn(b"kitten_tts08", self.lib.kt08_build_info())

    def test_phoneme_ids_match_python(self):
        for case in PHONEME_CASES:
            with self.subTest(case=case):
                self.assertEqual(self.ids(case), python_ids(case))

    def test_phoneme_ids_errors(self):
        n = ctypes.c_size_t()
        self.assertEqual(self.lib.kt08_phonemes_to_ids(b"\xff\xfe", None, 0, ctypes.byref(n)), 1)  # not UTF-8
        self.assertTrue(self.lib.kt08_last_error())
        self.assertEqual(self.lib.kt08_phonemes_to_ids(b"\xc0\xaf", None, 0, ctypes.byref(n)), 1)  # overlong
        small = (ctypes.c_int32 * 2)()
        self.assertEqual(self.lib.kt08_phonemes_to_ids("həlˈoʊ".encode(), small, 2, ctypes.byref(n)), 1)
        self.assertEqual(n.value, len(python_ids("həlˈoʊ")))

    def test_style_row_is_python_text_length(self):
        for text, rows in (("Hello there.", 400), ("naïve café 日本", 400), ("x" * 900, 400), ("", 400), ("abc", 1)):
            self.assertEqual(self.lib.kt08_style_row_for_text(text.encode(), rows), min(len(text), rows - 1))

    def test_voice_load_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.bin"
            bad.write_bytes(b"\0" * 1000)
            out = ctypes.POINTER(ctypes.c_float)()
            rows = ctypes.c_size_t()
            self.assertEqual(self.lib.kt08_voice_load(str(bad).encode(), ctypes.byref(out), ctypes.byref(rows)), 2)
            self.assertEqual(self.lib.kt08_voice_load(b"/no/such/voice.bin", ctypes.byref(out), ctypes.byref(rows)), 2)

    def test_load_errors(self):
        model = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_model_load(b"/no/such/arch.json", b"/no/such/w.bin", None, ctypes.byref(model)), 2)
        self.assertFalse(model.value)

    @unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_MODEL_DIR"), "set KITTEN_TTS08_CAPI_MODEL_DIR for synthesis")
    def test_synthesis_matches_python_engine(self):
        import numpy as np
        from kitten_inference.tts08 import _engine
        from kitten_inference.tts08.cpp_model import TAIL_TRIM_SAMPLES

        d = Path(os.environ["KITTEN_TTS08_CAPI_MODEL_DIR"])
        arch = Path(_engine.__file__).parent / "model_defs" / "kitten_fp32_15m_arch.json"
        weights, voice_path = d / "kitten_fp32_15m.bin", d / "voices_kitten_15m" / "expr-voice-2-f.bin"
        model = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_model_load(str(arch).encode(), str(weights).encode(), None, ctypes.byref(model)),
                         0, self.lib.kt08_last_error())
        try:
            self.assertGreater(self.lib.kt08_model_max_tokens(model), 100)
            voice = ctypes.POINTER(ctypes.c_float)()
            rows = ctypes.c_size_t()
            self.assertEqual(self.lib.kt08_voice_load(str(voice_path).encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
            pack = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 256)
            self.assertEqual(rows.value, pack.shape[0])
            ref = _engine.InferenceModel(str(arch), str(weights))
            for text, phonemes in (("Hello there.", "həlˈoʊ ðˈɛɹ."), ("A test, please!", "ɐ tˈɛst, plˈiːz!")):
                ids = self.ids(phonemes)
                row = self.lib.kt08_style_row_for_text(text.encode(), rows.value)
                arr = (ctypes.c_int32 * len(ids))(*ids)
                out = ctypes.POINTER(ctypes.c_float)()
                n = ctypes.c_size_t()
                self.assertEqual(self.lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, row, 1.0,
                                                          ctypes.byref(out), ctypes.byref(n)), 0, self.lib.kt08_last_error())
                audio = np.ctypeslib.as_array(out, shape=(n.value,)).copy()
                self.lib.kt08_free(out)
                ref.set_speed(1.0)
                style = pack[row]
                expect = np.asarray(ref.forward({"phoneme_ids": np.asarray(ids, np.float32), "style_dec": style[:128].copy(),
                                                 "style_pred": style[128:].copy()})["t_audio"], np.float32)[:-TAIL_TRIM_SAMPLES]
                self.assertEqual(audio.size, expect.size)
                if os.environ.get("KITTEN_TTS08_CAPI_SAME_BUILD") == "1":
                    self.assertEqual(audio.tobytes(), expect.tobytes())
                else:
                    self.assertTrue(np.isfinite(audio).all())
                # Too many tokens: a clear error, no crash.
                long_ids = (ctypes.c_int32 * 5000)()
                self.assertEqual(self.lib.kt08_synthesize(model, long_ids, 5000, voice, rows.value, 0, 1.0,
                                                          ctypes.byref(out), ctypes.byref(n)), 3)
            self.lib.kt08_free(voice)
        finally:
            self.lib.kt08_model_free(model)


if __name__ == "__main__":
    unittest.main()
