"""The optional Qualcomm NPU path of the C library (kt08_npu_*) and its preparation script.

Always: the header declares the functions; prepare_npu.py splits each model's graph into the CPU front, the decoder
that goes to the NPU and the CPU post part, with no node lost or shared (needs a converted model: see below).
With KITTEN_TTS08_CAPI_LIB (a shared build of the C library): without the QNN SDK, or without Qualcomm's libraries,
kt08_npu_load returns KT08_ERR_UNSUPPORTED and a reason, so an app can fall back to the CPU.
With KITTEN_TTS2_TINY_REPO (a downloaded KittenML/kitten-tts-2-nano or -pico folder) and torch + onnxruntime installed:
the decoder in its NPU form (fixed length, masks) matches the engine's decoder output (SNR above 90 dB).
"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engines" / "lite" / "engine" / "include" / "kitten_tts08.h"
PREP = ROOT / "engines" / "lite" / "mobile" / "npu" / "prepare_npu.py"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
KT08_ERR_UNSUPPORTED = 6


class HeaderTest(unittest.TestCase):
    def test_declares_npu_functions(self):
        h = HEADER.read_text()
        for name in ("kt08_npu_options_init", "kt08_npu_load", "kt08_npu_free", "kt08_npu_max_frames",
                     "kt08_synthesize_tiny_npu"):
            self.assertIn(name + "(", h)
        self.assertIn("KT08_ERR_UNSUPPORTED = 6", h)


class NpuOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("model_dir", ctypes.c_char_p), ("weights_path", ctypes.c_char_p),
                ("qnn_lib_dir", ctypes.c_char_p), ("cpu_fallback", ctypes.c_void_p), ("threads", ctypes.c_int32),
                ("high_performance", ctypes.c_int32)]


@unittest.skipUnless(LIB, "KITTEN_TTS08_CAPI_LIB is not set")
class LibraryTest(unittest.TestCase):
    def test_unsupported_is_reported(self):
        lib = ctypes.CDLL(LIB)
        lib.kt08_npu_options_init.argtypes = [ctypes.POINTER(NpuOptions)]
        lib.kt08_npu_load.argtypes = [ctypes.POINTER(NpuOptions), ctypes.POINTER(ctypes.c_void_p)]
        lib.kt08_npu_max_frames.argtypes = [ctypes.c_void_p]
        lib.kt08_npu_max_frames.restype = ctypes.c_size_t
        lib.kt08_npu_free.argtypes = [ctypes.c_void_p]
        lib.kt08_last_error.restype = ctypes.c_char_p
        o = NpuOptions()
        lib.kt08_npu_options_init(ctypes.byref(o))
        self.assertEqual(o.struct_size, ctypes.sizeof(NpuOptions))
        self.assertEqual(o.high_performance, 1)
        out = ctypes.c_void_p(1)
        # model_dir missing: KT08_ERR_ARGUMENT, or KT08_ERR_UNSUPPORTED from a build without the NPU path
        self.assertIn(lib.kt08_npu_load(ctypes.byref(o), ctypes.byref(out)), (1, KT08_ERR_UNSUPPORTED))
        self.assertIsNone(out.value)
        with tempfile.TemporaryDirectory() as d:
            # A folder with a manifest and empty CPU graphs: a build without QNN, or with no Qualcomm libraries in
            # qnn_lib_dir, must refuse with KT08_ERR_UNSUPPORTED (or an I/O error for the fake graphs).
            Path(d, "npu.json").write_text(json.dumps({"graphs": [{"frames": 160, "context": "body_160.bin"}]}))
            o.model_dir, o.weights_path, o.qnn_lib_dir = d.encode(), str(Path(d, "w.bin")).encode(), d.encode()
            rc = lib.kt08_npu_load(ctypes.byref(o), ctypes.byref(out))
            self.assertIn(rc, (KT08_ERR_UNSUPPORTED, 2), lib.kt08_last_error())
            self.assertIsNone(out.value)
            self.assertTrue(lib.kt08_last_error())
        self.assertEqual(lib.kt08_npu_max_frames(None), 0)
        lib.kt08_npu_free(None)


@unittest.skipUnless(REPO, "KITTEN_TTS2_TINY_REPO is not set")
class PrepareTest(unittest.TestCase):
    def test_split_and_decoder_form(self):
        try:
            import onnxruntime  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch and onnxruntime are needed")
        from kitten_inference.tts2_tiny import convert
        sys.path.insert(0, str(PREP.parent))
        import prepare_npu
        cfg = json.loads(Path(REPO, "config.json").read_text())
        key = cfg["languages"]["en"]["model"]
        with tempfile.TemporaryDirectory() as d:
            conv = os.path.join(d, "conv")
            convert.convert_repo(REPO, conv, models=[key], int8=False)
            arch = json.loads(Path(conv, f"{key}_fp32_arch.json").read_text())
            front, post, body = prepare_npu.split_graph(arch)
            names = [n["name"] for n in front["nodes"]] + [n["name"] for n in body] + [n["name"] for n in post["nodes"]]
            self.assertEqual(sorted(names), sorted(n["name"] for n in arch["nodes"]))
            out = os.path.join(d, "npu")
            ids = "0,50,156,86,16,102,43,4,10,0"
            r = subprocess.run([sys.executable, str(PREP), conv, key, out, "--frames", "160",
                                "--verify-ids", ids, "--verify-voice",
                                str(next(Path(conv, "voices", "en").glob("*.bin")))],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            res = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
            self.assertTrue(res["front_dur_equal"])
            self.assertGreater(res["post_audio"], 150)
            self.assertGreater(res["body_B160_torch"], 90)
            self.assertGreater(res["body_B160_onnx"], 90)
