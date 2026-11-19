"""The C library's Kitten TTS 2 mini/nano/pico functions (kt08_vocab_*, kt08_synthesize_tiny, kt08_stream_*).

Source-only checks always run. The library checks need a shared build of the C library (see tests/tts08/test_capi.py):
    KITTEN_TTS08_CAPI_LIB=build/capi/libkitten_tts08.so python -m pytest tests/tts2_tiny/test_capi_tiny.py
With KITTEN_TTS2_TINY_REPO (a downloaded KittenML/kitten-tts-2-nano or -pico folder) the synthesis check also runs and
compares the C library with the Python package: byte-identical when KITTEN_TTS08_CAPI_SAME_BUILD=1 (same CMake flags
as the package's extension), else within 1e-4.
"""
import ctypes
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engines" / "lite" / "engine" / "include" / "kitten_tts08.h"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
NEW = ("kt08_vocab_load", "kt08_vocab_free", "kt08_vocab_ids", "kt08_style_row_for_ids", "kt08_synthesize_tiny",
       "kt08_stream_options_init", "kt08_stream_open", "kt08_stream_next", "kt08_stream_chunks", "kt08_stream_free",
       "kt08_stream_open_tts08")


class HeaderTest(unittest.TestCase):
    def test_declares_tiny_functions(self):
        h = HEADER.read_text(encoding="utf-8")
        self.assertIn("#define KT08_API_VERSION 6", h)   # 6: streaming
        for name in NEW:
            self.assertIn(name + "(", h)


def _lib():
    lib = ctypes.CDLL(LIB)
    P, S, I32, F = ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
    lib.kt08_vocab_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(P)]
    lib.kt08_vocab_ids.argtypes = [P, ctypes.c_char_p, I32, S, ctypes.POINTER(S)]
    lib.kt08_vocab_free.argtypes = [P]
    lib.kt08_style_row_for_ids.argtypes = [S, S]
    lib.kt08_style_row_for_ids.restype = S
    lib.kt08_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, P, ctypes.POINTER(P)]
    lib.kt08_model_free.argtypes = [P]
    lib.kt08_voice_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(F), ctypes.POINTER(S)]
    lib.kt08_synthesize_tiny.argtypes = [P, I32, S, F, S, S, ctypes.c_float, ctypes.c_int, ctypes.POINTER(F),
                                         ctypes.POINTER(S)]
    lib.kt08_free.argtypes = [P]
    lib.kt08_synthesize_tiny.restype = ctypes.c_int
    lib.kt08_last_error.restype = ctypes.c_char_p
    lib.kt08_api_version.restype = ctypes.c_int
    return lib


def _ids(lib, vocab, phonemes):
    n = ctypes.c_size_t(0)
    lib.kt08_vocab_ids(vocab, phonemes.encode(), None, 0, ctypes.byref(n))
    buf = (ctypes.c_int32 * n.value)()
    assert lib.kt08_vocab_ids(vocab, phonemes.encode(), buf, n.value, ctypes.byref(n)) == 0
    return list(buf)


@unittest.skipUnless(LIB, "KITTEN_TTS08_CAPI_LIB is not set")
class VocabTest(unittest.TestCase):
    def test_ids_match_python(self):
        from kitten_inference.tts2_tiny.model import END_TOKEN
        lib = _lib()
        self.assertEqual(lib.kt08_api_version(), 6)
        vocab = {";": 1, ",": 3, " ": 16, "a": 43, "b": 44, "ə": 83, "ˈ": 156, "ʃ": 131, "…": 10}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "vocab.json")
            Path(p).write_text(json.dumps(vocab, ensure_ascii=True))     # \\u escapes must decode too
            v = ctypes.c_void_p()
            self.assertEqual(lib.kt08_vocab_load(p.encode(), ctypes.byref(v)), 0, lib.kt08_last_error())
            for text in ("ab ˈʃə", "a(b)z ;", "", "ʃʃʃ…"):
                want = [0] + [vocab[c] if c in vocab else vocab[","] for c in text if c in vocab or c in "()"]
                self.assertEqual(_ids(lib, v, text), want + [END_TOKEN, 0], text)
            lib.kt08_vocab_free(v)
        self.assertEqual(lib.kt08_style_row_for_ids(12, 510), 11)
        self.assertEqual(lib.kt08_style_row_for_ids(900, 510), 509)
        self.assertEqual(lib.kt08_style_row_for_ids(0, 5), 0)
        self.assertEqual(lib.kt08_style_row_for_ids(1, 0), 0)

    def test_malformed_vocab_is_refused(self):
        lib = _lib()
        bad = ["", "{", '{"a":', '{"a":1', '{"ab":1}', '{"\\ud800":1}', '{"\\ud800\\u0041":1}', '{"\\udc00":1}',
               '{"\\u12":1}', '{"\\uZZZZ":1}', '{"a":99999999999999999999}', '{"a":-5}', '{"a":4294967296}', "[1,2]",
               '{"\\', "{}"]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "vocab.json")
            for text in bad:
                Path(p).write_text(text)
                v = ctypes.c_void_p()
                self.assertEqual(lib.kt08_vocab_load(p.encode(), ctypes.byref(v)), 2, text)   # KT08_ERR_IO
                self.assertIn(b"not a vocab.json", lib.kt08_last_error(), text)
                self.assertFalse(v.value, text)
            Path(p).write_text('{"\\ud83d\\ude00": 7, "a": 1}')                      # a UTF-16 pair is one character
            v = ctypes.c_void_p()
            self.assertEqual(lib.kt08_vocab_load(p.encode(), ctypes.byref(v)), 0, lib.kt08_last_error())
            self.assertEqual(_ids(lib, v, "\U0001F600a"), [0, 7, 1, 10, 0])
            lib.kt08_vocab_free(v)

    def test_argument_errors(self):
        lib = _lib()
        vocab = {"a": 43, "b": 44, ",": 3}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "vocab.json")
            Path(p).write_text(json.dumps(vocab))
            self.assertEqual(lib.kt08_vocab_load(None, ctypes.byref(ctypes.c_void_p())), 1)
            self.assertEqual(lib.kt08_vocab_load(p.encode(), None), 1)
            self.assertEqual(lib.kt08_vocab_load(os.path.join(d, "missing.json").encode(),
                                                 ctypes.byref(ctypes.c_void_p())), 2)
            v = ctypes.c_void_p()
            self.assertEqual(lib.kt08_vocab_load(p.encode(), ctypes.byref(v)), 0)
            n = ctypes.c_size_t(0)
            buf = (ctypes.c_int32 * 2)()
            self.assertEqual(lib.kt08_vocab_ids(None, b"a", buf, 2, ctypes.byref(n)), 1)
            self.assertEqual(lib.kt08_vocab_ids(v, b"ab", buf, 2, ctypes.byref(n)), 1)       # 5 ids, capacity 2
            self.assertEqual(n.value, 5)
            self.assertEqual(lib.kt08_vocab_ids(v, b"ab", None, 10, ctypes.byref(n)), 1)
            self.assertEqual(lib.kt08_vocab_ids(v, b"\xff\xfe", buf, 2, ctypes.byref(n)), 1)  # not UTF-8
            self.assertEqual(_ids(lib, v, ""), [0, 10, 0])
            lib.kt08_vocab_free(v)
        lib.kt08_vocab_free(None)
        out, n = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
        ids = (ctypes.c_int32 * 3)(0, 10, 0)
        voice = (ctypes.c_float * 256)()
        self.assertEqual(lib.kt08_synthesize_tiny(None, ids, 3, voice, 1, 0, 1.0, 1, ctypes.byref(out), ctypes.byref(n)), 1)
        self.assertEqual(lib.kt08_synthesize_tiny(None, ids, 3, voice, 1, 0, 1.0, 1, None, None), 1)


@unittest.skipUnless(LIB and REPO, "KITTEN_TTS08_CAPI_LIB and KITTEN_TTS2_TINY_REPO are needed")
class SynthesisTest(unittest.TestCase):
    def test_same_audio_as_python_package(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        lib = _lib()
        tts = KittenTTS2Tiny(REPO)
        lang = tts.default_language
        key = tts.config["languages"][lang]["model"]
        arch, weights = tts._converted(key)
        model = ctypes.c_void_p()
        self.assertEqual(lib.kt08_model_load(arch.encode(), weights.encode(), None, ctypes.byref(model)), 0,
                         lib.kt08_last_error())
        v = ctypes.c_void_p()
        self.assertEqual(lib.kt08_vocab_load(os.path.join(REPO, "vocab.json").encode(), ctypes.byref(v)), 0)
        _, vkey = tts.resolve_voice(None, lang)
        with tempfile.TemporaryDirectory() as d:
            vp = os.path.join(d, vkey + ".bin")
            tts._pack(lang, vkey).astype("<f4").tofile(vp)
            voice, rows = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
            self.assertEqual(lib.kt08_voice_load(vp.encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
        same_build = os.environ.get("KITTEN_TTS08_CAPI_SAME_BUILD") == "1"
        for i, ps in enumerate(tts.phonemize("Hello there. This is a test of the C library, with two chunks.", lang)):
            ids = _ids(lib, v, ps)
            self.assertEqual(ids, tts._tokens(ps))
            for speed, first in ((1.0, 1), (1.3, 0)):
                arr = (ctypes.c_int32 * len(ids))(*ids)
                out, n = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
                row = lib.kt08_style_row_for_ids(len(ids), rows.value)
                self.assertEqual(lib.kt08_synthesize_tiny(model, arr, len(ids), voice, rows.value, row, speed, first,
                                                          ctypes.byref(out), ctypes.byref(n)), 0, lib.kt08_last_error())
                c = np.ctypeslib.as_array(out, shape=(n.value,)).copy()
                lib.kt08_free(out)
                py = tts._chunk(lang, vkey, ps, speed, keep_start=bool(first))
                self.assertEqual(len(c), len(py))
                if same_build:
                    self.assertTrue(np.array_equal(c, py))
                else:
                    self.assertLess(float(np.max(np.abs(c - py))), 1e-4)
        # Error paths on a loaded model: speed outside 0.25 to 4, ids outside the vocabulary, too many tokens, no voice.
        ids = _ids(lib, v, tts.phonemize("Hello there.", lang)[0])
        arr = (ctypes.c_int32 * len(ids))(*ids)
        out, n = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
        for speed in (0.0, -1.0, float("nan"), float("inf"), 0.2, 4.5, 1e-30):
            self.assertEqual(lib.kt08_synthesize_tiny(model, arr, len(ids), voice, rows.value, 0, speed, 1,
                                                      ctypes.byref(out), ctypes.byref(n)), 1, speed)
            self.assertFalse(out)
        for speed in (0.25, 4.0):
            self.assertEqual(lib.kt08_synthesize_tiny(model, arr, len(ids), voice, rows.value, 0, speed, 1,
                                                      ctypes.byref(out), ctypes.byref(n)), 0, lib.kt08_last_error())
            self.assertGreater(n.value, 0)
            lib.kt08_free(out)
        # kt08_synthesize (the TTS 0.8 call) applies the same speed range to these models.
        lib.kt08_synthesize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_size_t)]
        lib.kt08_synthesize.restype = ctypes.c_int
        for speed in (0.2, 4.5, 1e-30):
            self.assertEqual(lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, 0, speed, ctypes.byref(out),
                                                 ctypes.byref(n)), 1, speed)
            self.assertFalse(out)
        self.assertEqual(lib.kt08_synthesize(model, arr, len(ids), voice, rows.value, 0, 1.0, ctypes.byref(out),
                                             ctypes.byref(n)), 0, lib.kt08_last_error())
        self.assertGreater(n.value, 0)
        lib.kt08_free(out)
        bad = (ctypes.c_int32 * 5)(0, 100000, -7, 10, 0)
        self.assertNotEqual(lib.kt08_synthesize_tiny(model, bad, 5, voice, rows.value, 4, 1.0, 1, ctypes.byref(out),
                                                     ctypes.byref(n)), 0)
        self.assertFalse(out)
        lib.kt08_model_max_tokens.argtypes = [ctypes.c_void_p]
        lib.kt08_model_max_tokens.restype = ctypes.c_size_t
        limit = lib.kt08_model_max_tokens(model)
        self.assertGreater(limit, 0)
        long_ids = (ctypes.c_int32 * (limit + 1))(*([0] + [ids[1]] * (limit - 2) + [10, 0]))
        self.assertEqual(lib.kt08_synthesize_tiny(model, long_ids, limit + 1, voice, rows.value, limit, 1.0, 1,
                                                  ctypes.byref(out), ctypes.byref(n)), 3)      # KT08_ERR_TOO_LONG
        self.assertEqual(lib.kt08_synthesize_tiny(model, arr, len(ids), None, rows.value, 0, 1.0, 1, ctypes.byref(out),
                                                  ctypes.byref(n)), 1)
        lib.kt08_free(voice)
        lib.kt08_vocab_free(v)
        lib.kt08_model_free(model)

    def test_memory_mode_out_of_range_is_refused(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny

        class Options(ctypes.Structure):
            _fields_ = [("struct_size", ctypes.c_uint32), ("threads", ctypes.c_int32), ("memory_mode", ctypes.c_int32),
                        ("fp16_weights", ctypes.c_int32), ("fp16_compute", ctypes.c_int32)]
        lib = _lib()
        lib.kt08_options_init.argtypes = [ctypes.POINTER(Options)]
        tts = KittenTTS2Tiny(REPO)
        arch, weights = tts._converted(tts.config["languages"][tts.default_language]["model"])
        for mode in (7, -1):
            opt = Options()
            lib.kt08_options_init(ctypes.byref(opt))
            opt.memory_mode = mode
            model = ctypes.c_void_p()
            self.assertEqual(lib.kt08_model_load(arch.encode(), weights.encode(), ctypes.byref(opt), ctypes.byref(model)),
                             1, mode)
            self.assertFalse(model.value)


if __name__ == "__main__":
    unittest.main()
