"""The C library's English text front end for Kitten TTS 2 mini/nano/pico (kt08_tiny_frontend_*, API version 5).

Source-only checks always run. The library checks need a shared build of the C library and the front end's data:
    KITTEN_TTS08_CAPI_LIB=build/libkitten_tts08.so \\
    KITTEN_TTS2_TINY_REPO=<downloaded KittenML/kitten-tts-2-nano or -mini folder> \\
    KITTEN_TINY_FE_DATA=<folder from python -m kitten_inference.tts2_tiny.frontend.export_phone_data> \\
    python -m pytest tests/tts2_tiny/test_frontend_capi_en.py
They compare the C library with the Python package's Frontend(text, "en") on sentences whose phonemes do not depend
on spaCy's part-of-speech tags (the C library uses a rule tagger instead; see engines/lite/mobile/README.md).
"""
import ctypes
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engines" / "lite" / "engine" / "include" / "kitten_tts08.h"
SRC = ROOT / "engines" / "lite" / "engine" / "src" / "frontend" / "tiny_en"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
DATA = os.environ.get("KITTEN_TINY_FE_DATA")
NEW = ("kt08_tiny_frontend_load", "kt08_tiny_frontend_free", "kt08_tiny_text_to_phonemes")

# Exact match expected: numbers, dates, money, abbreviations, unknown names (CMUdict and the BART), long text (chunks).
SENTENCES = [
    "Hello world, this is a test of the phone front end.",
    "It cost $5.99 at the store on 2026-10-13 at 3:45pm.",
    "Dr. Smith lives on St. Paul St. with 1,234 books & 7 cats.",
    "Kevin and Diego met Zbigniew Brzezinski in Szczecin.",
    "NASA's budget rose 12% to $24.5B in Q3 2025.",
    "Call 555-0123 ext. 42, or visit https://example.com today.",
    "The temperature is -3°C; the flight leaves at 07:05.",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 12),
    "",
]


class SourceTest(unittest.TestCase):
    def test_header(self):
        h = HEADER.read_text()
        self.assertIn("#define KT08_API_VERSION 5", h)
        for name in NEW:
            self.assertIn(name + "(", h)

    def test_no_gpl_data_or_spacy_model_in_the_front_end(self):
        names = {p.name for p in SRC.rglob("*")}
        for bad in ("en_rules", "en_list", "en_dict", "us_gold.json", "us_silver.json", "cmu_misaki_us.json"):
            self.assertNotIn(bad, names)

    def test_no_new_dependencies(self):
        for p in SRC.glob("*.[ch]pp"):
            for line in p.read_text().splitlines():
                if line.startswith("#include <"):
                    inc = line.split("<")[1].split(">")[0]
                    self.assertNotIn("/", inc, f"{p.name}: {line}")   # standard headers only


def _lib():
    lib = ctypes.CDLL(LIB)
    P = ctypes.c_void_p
    lib.kt08_tiny_frontend_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                            ctypes.POINTER(P)]
    lib.kt08_tiny_frontend_load.restype = ctypes.c_int
    lib.kt08_tiny_frontend_free.argtypes = [P]
    lib.kt08_tiny_text_to_phonemes.argtypes = [P, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(P)]
    lib.kt08_tiny_text_to_phonemes.restype = ctypes.c_int
    lib.kt08_free.argtypes = [P]
    lib.kt08_last_error.restype = ctypes.c_char_p
    lib.kt08_api_version.restype = ctypes.c_int
    return lib


@unittest.skipUnless(LIB and REPO and DATA, "KITTEN_TTS08_CAPI_LIB, KITTEN_TTS2_TINY_REPO and KITTEN_TINY_FE_DATA are needed")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = _lib()
        cfg = json.loads((Path(REPO) / "config.json").read_text())
        cls.phon = cfg.get("phonemizer", "v1")
        cls.tmp = tempfile.mkdtemp()
        g2p = Path(cls.tmp) / "g2p"
        shutil.copytree(Path(REPO) / cfg.get("g2p_dir", "g2p"), g2p)
        for f in Path(DATA).iterdir():
            shutil.copy(f, g2p / f.name)
        cls.g2p = str(g2p)
        cls.fe = ctypes.c_void_p()
        rc = cls.lib.kt08_tiny_frontend_load(cls.phon.encode(), str(Path(REPO) / "vocab.json").encode(),
                                             cls.g2p.encode(), None, ctypes.byref(cls.fe))
        assert rc == 0, cls.lib.kt08_last_error()
        from kitten_inference.tts2_tiny.frontend import Frontend
        cls.py = Frontend(cls.phon, json.loads((Path(REPO) / "vocab.json").read_text()), str(Path(REPO) / "g2p"))

    @classmethod
    def tearDownClass(cls):
        cls.lib.kt08_tiny_frontend_free(cls.fe)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _c(self, text, lang="en"):
        out = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_text_to_phonemes(self.fe, text.encode(), lang.encode(), ctypes.byref(out))
        if rc != 0:
            return rc, None
        s = ctypes.string_at(out).decode()
        self.lib.kt08_free(out)
        return rc, (s.split("\n") if s else [])

    def test_version(self):
        self.assertEqual(self.lib.kt08_api_version(), 5)

    def test_matches_python(self):
        for text in SENTENCES:
            rc, chunks = self._c(text)
            self.assertEqual(rc, 0, self.lib.kt08_last_error())
            self.assertEqual(chunks, self.py(text, "en")[0], text)

    def test_long_text_is_chunked(self):
        rc, chunks = self._c(SENTENCES[7])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 350 for c in chunks))

    def test_errors(self):
        self.assertEqual(self._c("Hallo Welt", "de")[0], 6)          # KT08_ERR_UNSUPPORTED in this build
        out = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_tiny_text_to_phonemes(self.fe, None, b"en", ctypes.byref(out)), 1)
        bad = ctypes.c_void_p()
        self.assertEqual(self.lib.kt08_tiny_frontend_load(b"v1", b"/nonexistent/vocab.json", self.g2p.encode(), None,
                                                          ctypes.byref(bad)), 2)
        self.assertFalse(bad.value)

    def test_malformed_lexicon_fails_cleanly(self):
        d = Path(self.tmp) / "bad"
        shutil.copytree(self.g2p, d)
        (d / "us_gold.json").write_text('{"a": "x", "b": {"DEFAULT": ')
        bad = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_frontend_load(self.phon.encode(), str(Path(REPO) / "vocab.json").encode(),
                                              str(d).encode(), None, ctypes.byref(bad))
        self.assertEqual(rc, 2)
        self.assertFalse(bad.value)
        (d / "us_gold.json").write_bytes(b"\xff\xfe" + os.urandom(4096))
        rc = self.lib.kt08_tiny_frontend_load(self.phon.encode(), str(Path(REPO) / "vocab.json").encode(),
                                              str(d).encode(), None, ctypes.byref(bad))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
