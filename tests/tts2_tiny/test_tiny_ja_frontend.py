"""Japanese text front end of Kitten TTS 2 mini in the C library (kt08_tiny_frontend_*, language "ja").

Source-only checks always run. The library check needs a shared build of the C library, the Japanese data
(engines/lite/mobile/fetch_phonemizer_data.sh OUT ja -> OUT/ja/) and a downloaded mini repository for its vocab.json:
    KITTEN_TTS08_CAPI_LIB=build/libkitten_tts08.so KITTEN_ESPEAK_DATA=phonemizer-data \
    KITTEN_TTS2_MINI_REPO=<kitten-tts-2-mini folder> python -m pytest tests/tts2_tiny/test_tiny_ja_frontend.py
It compares the C library with the Python package (misaki's cutlet path: fugashi + UniDic from unidic-lite).
"""
import ctypes
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FETCH = ROOT / "engines" / "lite" / "mobile" / "fetch_phonemizer_data.sh"
JA_SRC = ROOT / "engines" / "lite" / "engine" / "src" / "frontend" / "lang"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
DATA = os.environ.get("KITTEN_ESPEAK_DATA")
REPO = os.environ.get("KITTEN_TTS2_MINI_REPO")
JA = [
    "技術決定論のほとんどの解釈は、2つの一般論を共有しています。",
    "場にふさわしい敬意を払い、尊厳と礼節をもって立ち振る舞ってください。",
    "今日は東京で12月３日にｶﾞｷﾞAIのテストをしました。「ありがとう」😀 エクササイズ",
    "1000円、3000人、123456789、〜5、ヴァイオリンとウィキペディア（Wikipedia）。",
    # several sentences: more than 350 phonemes, cut into chunks at sentence ends
    "場にふさわしい敬意を払い、尊厳と礼節をもって立ち振る舞ってください。" * 6,
]


class SourceTest(unittest.TestCase):
    def test_fetch_script_pins_dictionary(self):
        s = FETCH.read_text(encoding="utf-8")
        self.assertIn("unidic-lite-1.0.8.tar.gz", s)
        self.assertIn("db9d4572d9fdd4d00a97949d4b0741ec480ee05a7e7e2e32f547500dae27b245", s)
        for f in ("sys.dic", "unk.dic", "matrix.bin", "char.bin"):
            self.assertRegex(s, re.escape(f) + r" [0-9a-f]{64}")

    def test_no_dictionary_in_repo(self):
        # UniDic is downloaded by the app (as the Python package downloads it), never committed
        names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        bad = [n for n in names if re.search(r"(^|/)(sys\.dic|unk\.dic|matrix\.bin|char\.bin|matrix\.def)$", n)]
        self.assertEqual(bad, [])

    def test_own_code(self):
        for p in list(JA_SRC.glob("ja*.cpp")) + list(JA_SRC.glob("ja*.hpp")):
            text = p.read_text(encoding="utf-8")
            self.assertIn("Apache License", text)
            self.assertNotIn("mecab.h", text)


@unittest.skipUnless(LIB and DATA and REPO, "needs KITTEN_TTS08_CAPI_LIB, KITTEN_ESPEAK_DATA and KITTEN_TTS2_MINI_REPO")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib = ctypes.CDLL(LIB)
        P = ctypes.c_void_p
        lib.kt08_tiny_frontend_load.argtypes = [ctypes.c_char_p] * 4 + [ctypes.POINTER(P)]
        lib.kt08_tiny_frontend_free.argtypes = [P]
        lib.kt08_tiny_text_to_phonemes.argtypes = [P, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.kt08_free.argtypes = [ctypes.c_void_p]
        lib.kt08_last_error.restype = ctypes.c_char_p
        cls.lib = lib

    def _fe(self, version):
        fe = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_frontend_load(version.encode(), os.path.join(REPO, "vocab.json").encode(), None,
                                              DATA.encode(), ctypes.byref(fe))
        self.assertEqual(rc, 0, self.lib.kt08_last_error())
        return fe

    def _run(self, fe, text):
        out = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_text_to_phonemes(fe, text.encode(), b"ja", ctypes.byref(out))
        if rc:
            return rc, None
        s = ctypes.string_at(out.value).decode()
        self.lib.kt08_free(out)
        return 0, (s.split("\n") if s else [])

    def test_matches_package(self):
        from kitten_inference.tts2_tiny.frontend import Frontend
        vocab = json.load(open(os.path.join(REPO, "vocab.json"), encoding="utf-8"))
        py = Frontend("v2", vocab, None)
        fe = self._fe("v2")
        try:
            for text in JA:
                rc, chunks = self._run(fe, text)
                self.assertEqual(rc, 0, self.lib.kt08_last_error())
                self.assertEqual(chunks, py(text, "ja")[0], text)
            self.assertGreater(len(self._run(fe, JA[-1])[1]), 1)
            # a paragraph of more than 1000 characters without a space: accepted (runs end at 。、！？ in Japanese)
            long_text = JA[1] * 30
            rc, chunks = self._run(fe, long_text)
            self.assertEqual(rc, 0, self.lib.kt08_last_error())
            self.assertEqual(chunks, py(long_text, "ja")[0])
        finally:
            self.lib.kt08_tiny_frontend_free(fe)

    def test_refuses_what_the_package_refuses(self):
        # The package raises on a number of more than 9 digits (num2kana has no words for it) and on digits that
        # are not 0-9; the library returns KT08_ERR_ARGUMENT (1) for the same text instead of saying something else.
        from kitten_inference.tts2_tiny.frontend import Frontend
        vocab = json.load(open(os.path.join(REPO, "vocab.json"), encoding="utf-8"))
        py = Frontend("v2", vocab, None)
        fe = self._fe("v2")
        try:
            for text in ["電話は09012345678です。", "1234567890", "٣人", "❶番", "❶"]:
                with self.assertRaises((AssertionError, KeyError), msg=text):
                    py(text, "ja")
                rc, _ = self._run(fe, text)
                self.assertEqual(rc, 1, text)
                self.assertIn(b"Python package", self.lib.kt08_last_error())
            for text in ["123456789円", "²と①と１２３", "1,234,567,890"]:   # still read, as in the package
                rc, chunks = self._run(fe, text)
                self.assertEqual(rc, 0, self.lib.kt08_last_error())
                self.assertEqual(chunks, py(text, "ja")[0], text)
        finally:
            self.lib.kt08_tiny_frontend_free(fe)

    def test_nano_pico_have_no_japanese(self):
        fe = self._fe("v1")
        try:
            rc, _ = self._run(fe, "こんにちは。")
            self.assertEqual(rc, 6)   # KT08_ERR_UNSUPPORTED: Japanese is a mini (v2) language
        finally:
            self.lib.kt08_tiny_frontend_free(fe)


if __name__ == "__main__":
    unittest.main()
