"""English text to phonemes in the C library for apps (kitten_tts08.h, API version 4: kt08_phonemizer_*, kt08_text_*).

Source-only checks always run (licence rules: CEPhonemizer's header and NOTICE entry, no espeak-ng data file in the
repository, pinned download script). The library checks need a shared build of the C library:
    cmake -S engines/lite/engine -B build/capi -DKITTEN_BUILD_C_LIBRARY=ON -DBUILD_SHARED_LIBS=ON \
          -DKITTEN_BUILD_PYTHON_MODULE=OFF -DENABLE_NATIVE_ARCH=OFF && cmake --build build/capi
    KITTEN_TTS08_CAPI_LIB=build/capi/libkitten_tts08.so python -m pytest tests/tts08/test_capi_text.py
The phonemiser checks also need espeak-ng's en_rules and en_list (engines/lite/mobile/fetch_phonemizer_data.sh DIR):
KITTEN_TTS08_PHONEMIZER_DATA=DIR. The comparison with the Python package's phonemes runs when phonemizer and
espeakng_loader are installed (they are dependencies of the package).
"""
import ctypes
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "engines" / "lite"
ENGINE = LITE / "engine"
CEPH = ENGINE / "src" / "frontend" / "cephonemizer"
HEADER = ENGINE / "include" / "kitten_tts08.h"
FETCH = LITE / "mobile" / "fetch_phonemizer_data.sh"
PINNED = "59eb19938f12e30881c81d86ce4a7de25414c9f4"

# Text that the package's normalisation changes in many ways (numbers, money, times, units, contractions, URLs, ...).
NORMALISATION_CASES = [
    "Hello there.", "", "   ", "There are 1200 students and 42 teachers.",
    "The project costs $1,000,000 and took 365 days.", "Temperature dropped to -5 degrees overnight.",
    "Pi is approximately 3.14159.", "Add .5 teaspoons of salt and .25 cup of milk.", "A -.05 correction was applied.",
    "Learning rate is 1e-4, weight decay 1e-5.", "We trained a 7B parameter model and a 13B variant.",
    "The salary was $85K per year.", "Rent is €1,200 per month.", "The book is £9.99.", "Inflation rose by 3.5%.",
    "She finished 1st, he came 2nd, I was 3rd.", "Add 3/4 cup of sugar.", "The meeting starts at 3:30pm.",
    "Departure at 14:00.", "The 80s and the 1990s.", "Read pages 10-20.", "GPT-3.5 and Python-3.10.",
    "The trail is 42km long; water boils at 100°C.", "<b>Hello</b> World! It's a great day.",
    "Visit https://example.com or email hello@example.com.", "I don't know, won't you help? They've left.",
    "Connect to 192.168.1.1 on port 8080.", "Call us at 555-1234 or 1-800-555-0199.",
    "Dr. Smith met Mr. Jones at 10 a.m. on Jan. 5th. Then they left!", "Version v2.0.1 shipped.",
    "naïve café, Ångström and 日本語 text.", "word " * 150, "Is it 3.0 or 3.00? 50%-60% agreed.",
    "\"Quoted,\" she said — (with brackets) and ‘curly’ quotes…", "It's 72°F; the 1st of May.",
]

# Sentences whose phonemes the front end gives exactly as the Python package does (espeak-ng 1.52.0).
PHONEME_CASES = [
    "Hello there.", "For the twentieth time that evening the two men shook hands.",
    "There are 1200 students and 42 teachers.", "A coffee costs $4.99 here.", "The trail is 42km long.",
    "Author of the danger trail, Philip Steels, etc.", "I think I will go home now.",
    "The CPU runs at 3.6GHz.", "Rates rose 5% & fell 2/3 of a point.",
]


class SourceTests(unittest.TestCase):
    def test_header_version_and_functions(self):
        h = HEADER.read_text(encoding="utf-8")
        self.assertGreaterEqual(int(re.search(r"#define KT08_API_VERSION (\d+)", h).group(1)), 4)  # 4: this API; later versions add to it
        for name in ("kt08_phonemizer_load", "kt08_phonemizer_free", "kt08_text_chunks", "kt08_chunk_to_phonemes",
                     "kt08_text_to_phonemes"):
            self.assertRegex(h, r"KT08_API [^;]*\b%s\(" % name)
        # append-only: the version 4 block comes after every older declaration
        self.assertGreater(h.index("kt08_phonemizer_load"), h.index("kt08_build_info"))

    def test_cephonemizer_keeps_its_licence_header(self):
        for f in ("phonemizer.cpp", "phonemizer.h", "rule_parser.h", "ipa_table.h"):
            self.assertTrue((CEPH / f).read_text(encoding="utf-8").startswith("// Copyright 2024 - Apache 2.0 License"), f)
        notice = (LITE / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("CEPhonemizer", notice)
        self.assertIn("KittenTTS-react-native", notice)

    def test_no_espeak_data_or_code_in_the_repository(self):
        try:
            files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
        data = [f for f in files if re.search(r"(^|/)[a-z]{2,3}_(rules|list|listx|extra|emoji|dict)$", f)]
        self.assertEqual(data, [], "espeak-ng data files must be downloaded by apps, not committed")
        for f in CEPH.iterdir():
            text = f.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("GNU General Public License", text, f)
            self.assertNotRegex(text, r"#include\s*[<\"](espeak|speak_lib)", f)

    def test_fetch_script_pins_the_data(self):
        s = FETCH.read_text(encoding="utf-8")
        self.assertIn(PINNED, s)
        for name, digest in (("en_rules", "8e75e9341ea735cc514b29a7d3a95c6c241c1cc176ad43e5699b8f7f66ab3194"),
                             ("en_list", "24eb79018ed6253c10682096de672ce9265c1fe15c3e19e7f754d57a0fcd9790")):
            self.assertRegex(s, r"dictsource/" + name + " " + digest)
        self.assertIn("dictsource", s)
        self.assertTrue(os.access(FETCH, os.X_OK))

    def test_front_end_is_only_in_the_c_library(self):
        cmake = (ENGINE / "CMakeLists.txt").read_text(encoding="utf-8")
        block = re.search(r"add_library\(kitten_tts08\b.*?\)", cmake, re.S).group(0)
        for src in ("src/capi/kitten_text08.cpp", "src/frontend/tts08_text.cpp", "src/frontend/mini_regex.cpp",
                    "src/frontend/cephonemizer/phonemizer.cpp"):
            self.assertIn(src, block)
            self.assertEqual(cmake.count(src), 1, src)


def python_chunks(text):
    from kitten_inference.tts08.preprocess import TextPreprocessor, chunk_text

    return chunk_text(TextPreprocessor(remove_punctuation=False)(text))


@unittest.skipUnless(os.environ.get("KITTEN_TTS08_CAPI_LIB"), "set KITTEN_TTS08_CAPI_LIB to a shared build of the C library")
class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib = ctypes.CDLL(os.environ["KITTEN_TTS08_CAPI_LIB"])
        sz = ctypes.POINTER(ctypes.c_size_t)
        lib.kt08_api_version.restype = ctypes.c_int
        lib.kt08_last_error.restype = ctypes.c_char_p
        lib.kt08_phonemizer_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.kt08_phonemizer_free.argtypes = [ctypes.c_void_p]
        lib.kt08_text_chunks.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, sz]
        lib.kt08_chunk_to_phonemes.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, sz]
        lib.kt08_text_to_phonemes.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, sz]
        lib.kt08_phonemes_to_ids.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, sz]
        cls.lib = lib
        cls.ph = ctypes.c_void_p()
        data = os.environ.get("KITTEN_TTS08_PHONEMIZER_DATA")
        if data:
            st = lib.kt08_phonemizer_load(os.path.join(data, "en_rules").encode(), os.path.join(data, "en_list").encode(),
                                          ctypes.byref(cls.ph))
            assert st == 0, lib.kt08_last_error()

    @classmethod
    def tearDownClass(cls):
        if cls.ph:
            cls.lib.kt08_phonemizer_free(cls.ph)

    def call(self, fn, *args):
        """Size query, then the call; returns the text."""
        n = ctypes.c_size_t()
        self.assertEqual(fn(*args, None, 0, ctypes.byref(n)), 0, self.lib.kt08_last_error())
        buf = ctypes.create_string_buffer(n.value)
        self.assertEqual(fn(*args, buf, n.value, ctypes.byref(n)), 0, self.lib.kt08_last_error())
        self.assertEqual(len(buf.value) + 1, n.value)
        return buf.value.decode("utf-8")

    def need_data(self):
        if not self.ph:
            self.skipTest("set KITTEN_TTS08_PHONEMIZER_DATA to a folder with en_rules and en_list")

    def test_version(self):
        self.assertGreaterEqual(self.lib.kt08_api_version(), 4)

    def test_chunks_match_python(self):
        for text in NORMALISATION_CASES:
            with self.subTest(text=text[:60]):
                got = self.call(self.lib.kt08_text_chunks, text.encode())
                self.assertEqual(got.split("\n") if got else [], python_chunks(text))

    def test_size_query_and_errors(self):
        n = ctypes.c_size_t()
        self.assertEqual(self.lib.kt08_text_chunks(b"\xff\xfe", None, 0, ctypes.byref(n)), 1)  # not UTF-8
        self.assertTrue(self.lib.kt08_last_error())
        self.assertEqual(self.lib.kt08_text_chunks(None, None, 0, ctypes.byref(n)), 1)
        self.assertEqual(self.lib.kt08_text_chunks(b"Hi.", None, 0, None), 1)
        self.assertEqual(self.lib.kt08_text_chunks(b"Hello there.", None, 0, ctypes.byref(n)), 0)
        small = ctypes.create_string_buffer(4)
        self.assertEqual(self.lib.kt08_text_chunks(b"Hello there.", small, 4, ctypes.byref(n)), 1)  # too small
        self.assertEqual(n.value, len("hello there.") + 1)

    def test_long_runs_are_refused(self):
        n = ctypes.c_size_t()
        self.assertEqual(self.lib.kt08_text_chunks(("1" * 1001).encode(), None, 0, ctypes.byref(n)), 3)  # TOO_LONG
        self.assertEqual(self.lib.kt08_text_chunks((("a" * 1000 + " ") * 3).encode(), None, 0, ctypes.byref(n)), 0)

    def test_load_errors(self):
        ph = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_phonemizer_load(b"/no/such/en_rules", b"/no/such/en_list", ctypes.byref(ph)), 2)
        self.assertFalse(ph.value)
        self.assertIn(b"/no/such", self.lib.kt08_last_error())
        self.assertEqual(self.lib.kt08_phonemizer_load(None, None, ctypes.byref(ph)), 1)
        self.lib.kt08_phonemizer_free(None)
        import tempfile

        with tempfile.TemporaryDirectory() as d:  # files that parse but are not espeak-ng's English data
            for name, text in (("en_rules", ".group a\n a  a\n"), ("en_list", "hello h@loU\n")):
                Path(d, name).write_text(text)
            self.assertEqual(self.lib.kt08_phonemizer_load(os.path.join(d, "en_rules").encode(),
                                                           os.path.join(d, "en_list").encode(), ctypes.byref(ph)), 2)
            self.assertIn(b"fetch_phonemizer_data", self.lib.kt08_last_error())

    def test_one_line_of_phonemes_per_chunk(self):
        self.need_data()
        for text in NORMALISATION_CASES:
            with self.subTest(text=text[:60]):
                chunks = self.call(self.lib.kt08_text_chunks, text.encode())
                phon = self.call(self.lib.kt08_text_to_phonemes, self.ph, text.encode())
                self.assertEqual(len(phon.split("\n")), len(chunks.split("\n")))
                for c, p in zip(chunks.split("\n") if chunks else [], phon.split("\n")):
                    self.assertEqual(self.call(self.lib.kt08_chunk_to_phonemes, self.ph, c.encode()), p)
                    n = ctypes.c_size_t()
                    self.assertEqual(self.lib.kt08_phonemes_to_ids(p.encode(), None, 0, ctypes.byref(n)), 0)

    def test_phonemes_match_python_package(self):
        self.need_data()
        try:
            from kitten_inference.tts08 import cpp_model

            cpp_model._set_espeak_data_path()
            import phonemizer

            be = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        except Exception as e:  # pragma: no cover - environment without espeak-ng
            self.skipTest("phonemizer/espeak-ng not available: %s" % e)
        for text in PHONEME_CASES:
            with self.subTest(text=text):
                want = [be.phonemize([c])[0] for c in python_chunks(text)]
                self.assertEqual(self.call(self.lib.kt08_text_to_phonemes, self.ph, text.encode()).split("\n"), want)


if __name__ == "__main__":
    unittest.main()
