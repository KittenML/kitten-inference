"""Text front end of Kitten TTS 2 for the languages other than English, in the C library (kt08_tiny_frontend_*).

Source-only checks always run. The library check needs a shared build of the C library, the espeak-ng data files
(engines/lite/mobile/fetch_phonemizer_data.sh) and a downloaded model repository for its vocab.json:
    KITTEN_TTS08_CAPI_LIB=build/libkitten_tts08.so KITTEN_ESPEAK_DATA=phonemizer-data \
    KITTEN_TTS2_TINY_REPO=<kitten-tts-2-nano folder> python -m pytest tests/tts2_tiny/test_tiny_lang_frontend.py
It compares the C library with the Python package (espeak-ng through phonemizer) on Spanish and Italian sentences.
"""
import ctypes
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engines" / "lite" / "engine" / "include" / "kitten_tts08.h"
FETCH = ROOT / "engines" / "lite" / "mobile" / "fetch_phonemizer_data.sh"
FRONTEND = ROOT / "engines" / "lite" / "engine" / "src" / "frontend"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
DATA = os.environ.get("KITTEN_ESPEAK_DATA")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
ES = [
    "Esto parece tener sentido, ya que en la Tierra no se percibe su movimiento, ¿cierto?",
    "Las escenas se proyectan en las pirámides y todas ellas son iluminadas.",
    "Durante esta fase de la historia de Europa, la iglesia católica, que había acumulado riqueza y poder, fue cuestionada.",
    "Contiene 78 recomendaciones, de las cuales la número uno es la adopción de una nueva iniciativa.",
]
IT = [
    "L'incidente è avvenuto in alta montagna e si ritiene sia stato causato da fuoco nemico.",
    "Per la discesa a terra non sarà necessario un visto separato (a partire dal 2009) utilizzando le escursioni di bordo.",
    "Secondo una stima dell'USACE, 6 pollici di pioggia potrebbero rompere gli argini danneggiati in precedenza.",
]


class SourceTest(unittest.TestCase):
    def test_header_declares_front_end(self):
        h = HEADER.read_text(encoding="utf-8")
        for name in ("kt08_tiny_frontend_load", "kt08_tiny_frontend_free", "kt08_tiny_text_to_phonemes"):
            self.assertIn(name + "(", h)

    def test_fetch_script_pins_data(self):
        s = FETCH.read_text(encoding="utf-8")
        self.assertIn("COMMIT=59eb19938f12e30881c81d86ce4a7de25414c9f4", s)
        for f in ("es_rules", "es_list", "phsource/phonemes", "phsource/ph_spanish"):
            self.assertRegex(s, re.escape(f) + r" [0-9a-f]{64}")

    def test_no_espeak_data_in_repo(self):
        # espeak-ng's data files are GPL-3.0: downloaded by the app, never committed
        names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        bad = [n for n in names if re.search(r"(^|/)([a-z]{2,3}_(rules|list|listx|extra)|ph_[a-z_]+|phonemes)$", n)]
        self.assertEqual(bad, [])

    def test_front_end_has_no_espeak_code(self):
        for p in FRONTEND.glob("*.cpp"):
            text = p.read_text(encoding="utf-8")
            self.assertIn("Apache License", text)
            self.assertNotIn("espeak_ng.h", text)


@unittest.skipUnless(LIB and DATA and REPO, "needs KITTEN_TTS08_CAPI_LIB, KITTEN_ESPEAK_DATA and KITTEN_TTS2_TINY_REPO")
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

    def _run(self, fe, text, lang):
        out = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_text_to_phonemes(fe, text.encode(), lang.encode(), ctypes.byref(out))
        if rc:
            return rc, None
        s = ctypes.string_at(out.value).decode()
        self.lib.kt08_free(out)
        return 0, (s.split("\n") if s else [])

    def test_matches_package(self):
        from kitten_inference.tts2_tiny.frontend import Frontend
        vocab = json.load(open(os.path.join(REPO, "vocab.json"), encoding="utf-8"))
        for version in ("v1", "v2"):
            py = Frontend(version, vocab, None)
            fe = self._fe(version)
            try:
                for lang, texts in (("es", ES), ("it", IT)):
                    for text in texts:
                        rc, chunks = self._run(fe, text, lang)
                        self.assertEqual(rc, 0, self.lib.kt08_last_error())
                        self.assertEqual(chunks, py(text, lang)[0], text)
            finally:
                self.lib.kt08_tiny_frontend_free(fe)

    def test_unsupported_language(self):
        fe = self._fe("v1")
        try:
            for lang in ("fr", "de", "pt"):
                rc, _ = self._run(fe, "Bonjour.", lang)
                self.assertEqual(rc, 6)   # KT08_ERR_UNSUPPORTED: not passing yet
        finally:
            self.lib.kt08_tiny_frontend_free(fe)


if __name__ == "__main__":
    unittest.main()
