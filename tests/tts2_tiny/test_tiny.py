"""Kitten TTS 2 mini / nano / pico: model routing, the converter's graphs and weights layout, and (with the official
files and the engine) synthesis.

Always run: routing, graph structure for the three published architectures (tests/tts2_tiny/arch_configs.json, the
"architecture" blocks of the models' config.json), int8 layout, W1X1 round trip.
With KITTEN_TTS2_TINY_REPO (a downloaded KittenML/kitten-tts-2-mini, -nano or -pico folder): conversion of every model in
it uses every weight, and the engine synthesises finite, reproducible audio.
"""
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kitten_inference._api import model_family
from kitten_inference.tts2_tiny import convert
from kitten_inference.tts2_tiny.model import REPOS, REVISIONS, repo_id

HERE = Path(__file__).resolve().parent
ARCHS = json.loads((HERE / "arch_configs.json").read_text())
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")


class RoutingTest(unittest.TestCase):
    def test_ids_route_to_the_cpu_engine(self):
        for name in ("KittenML/kitten-tts-2-mini", "kittenml/kitten-tts-2-nano", "kitten-tts-2-pico",
                     "KITTEN-TTS-2-NANO", "KittenML/kitten-tts-2-mini/"):
            self.assertEqual(model_family(name), "tts2_tiny", name)
            self.assertIn(repo_id(name), REVISIONS)
        self.assertEqual(model_family("KittenML/kitten-tts-2"), "tts2")
        self.assertEqual(model_family("kitten-tts-2"), "tts2")
        self.assertEqual(model_family("KittenML/kitten-tts-mini-0.8"), "tts08")
        self.assertEqual(model_family("kitten-tts-2-large"), "tts08")   # not one of the three: the 0.8 loader decides
        self.assertEqual(set(REPOS.values()), set(REVISIONS))

    def test_local_folder_by_config_type(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"type": "KITTEN2_TINY"}))
            self.assertEqual(model_family(d), "tts2_tiny")
            Path(d, "config.json").write_text(json.dumps({"type": "CPP1"}))
            self.assertEqual(model_family(d), "tts08")

    def test_gpu_and_unknown_backends_and_int8_on_metal_are_refused(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        for gpu in ("rocm", "amd_gpu"):
            with self.assertRaisesRegex(ValueError, "NVIDIA GPUs and Apple silicon GPUs"):
                KittenTTS2Tiny("KittenML/kitten-tts-2-pico", backend=gpu)
        with self.assertRaisesRegex(ValueError, "must be 'cpu', 'metal' or 'cuda'"):
            KittenTTS2Tiny("KittenML/kitten-tts-2-pico", backend="vulkan")
        for gpu in ("metal", "cuda"):
            with self.assertRaisesRegex(ValueError, "float32"):
                KittenTTS2Tiny("KittenML/kitten-tts-2-pico", backend=gpu, precision="int8")
            with self.assertRaisesRegex(ValueError, "low_memory=True is for the CPU"):
                KittenTTS2Tiny("KittenML/kitten-tts-2-pico", backend=gpu, low_memory=True)

    def test_cuda_without_its_library_is_a_clear_error(self):
        from unittest import mock
        from kitten_inference.tts2_tiny import KittenTTS2Tiny, _cuda
        with mock.patch.dict(os.environ, {_cuda.LIB_ENV: os.path.join(tempfile.gettempdir(), "no-such-kt08_cuda")}), \
                mock.patch.object(_cuda, "_engine", None):
            with self.assertRaisesRegex(ValueError, "backend='cuda'.*backend='cpu'"):
                KittenTTS2Tiny("KittenML/kitten-tts-2-pico", backend="cuda")

class FrontEndInstallTest(unittest.TestCase):
    """The base install brings the front end's packages where the model says they install, and the error names the
    Python version elsewhere."""

    def test_markers_match_the_model(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        from packaging.requirements import Requirement
        from kitten_inference.tts2_tiny.model import FRONT_END_PYTHONS
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.is_file():
            self.skipTest("no pyproject.toml (installed tests)")
        deps = [Requirement(r) for r in tomllib.loads(pyproject.read_text())["project"]["dependencies"]]
        from kitten_inference.tts2_tiny.model import FRONT_END_MODULES
        tiny = [r for r in deps if r.name in FRONT_END_MODULES]
        self.assertEqual(sorted(r.name for r in tiny), sorted(FRONT_END_MODULES))
        self.assertFalse({r.name for r in deps} & {"misaki", "num2words", "pyopenjtalk", "mojimoji", "unidic-lite"})
        for minor in range(8, 17):
            for system, machine in (("linux", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64"),
                                    ("win32", "ARM64"), ("darwin", "arm64"), ("android", "aarch64")):
                env = dict(python_version=f"3.{minor}", sys_platform=system, platform_machine=machine, extra="")
                want = FRONT_END_PYTHONS[0] <= (3, minor) <= FRONT_END_PYTHONS[1]
                for r in tiny:
                    no_wheel = r.name == "fugashi" and machine == "ARM64"     # fugashi: no Windows on Arm package
                    self.assertEqual(r.marker.evaluate(env), want and not no_wheel, (r, env))

    def test_error_names_the_python(self):
        from kitten_inference.tts2_tiny.model import front_end_unavailable
        msg = front_end_unavailable("spacy", "en", version=(3, 15))
        self.assertIn("not supported on Python 3.15", msg)
        self.assertIn("3.10 to 3.14", msg)
        self.assertNotIn("[", msg)       # no install extra
        msg = front_end_unavailable("spacy", "en", version=(3, 12))
        self.assertIn("pip install --force-reinstall kitten-inference", msg)
        msg = front_end_unavailable("fugashi", "ja", version=(3, 12), system="Windows", machine="ARM64")
        self.assertIn("not supported on Windows on Arm", msg)


class VendoredFrontEndTest(unittest.TestCase):
    """The vendored parts of the English and Japanese front end: the mojimoji port, num2words' English converter and the
    unidic-lite download (the Japanese dictionary)."""

    def test_mojimoji_port(self):
        from kitten_inference.tts2_tiny.frontend import mojimoji
        self.assertEqual(mojimoji.zen_to_han("ＡＢＣ１２３アガパ", kana=False), "ABC123アガパ")
        self.assertEqual(mojimoji.zen_to_han("アガパ"), "ｱｶﾞﾊﾟ")
        self.assertEqual(mojimoji.han_to_zen("ｱｶﾞﾊﾟABC123", digit=False, ascii=False), "アガパABC123")
        self.assertEqual(mojimoji.han_to_zen("ﾞｶﾞ", ascii=False, digit=False), "゛ガ")

    def test_num2words_english(self):
        from kitten_inference.tts2_tiny.frontend.num2words import num2words
        self.assertEqual(num2words(1234), "one thousand, two hundred and thirty-four")
        self.assertEqual(num2words(21, to="ordinal"), "twenty-first")
        self.assertEqual(num2words(1999, to="year"), "nineteen ninety-nine")
        self.assertEqual(num2words(3.5), "three point five")

    def test_unidic_unpack_keeps_only_the_dictionary_folder(self):
        import io
        import tarfile
        from kitten_inference.tts2_tiny.frontend import unidic
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "a.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for name in (unidic._PREFIX + "sys.dic", unidic._PREFIX + "mecabrc", unidic._PREFIX + "../evil",
                             unidic._PREFIX + "sub/x", "unidic-lite-1.0.8/setup.py", "/etc/evil"):
                    info = tarfile.TarInfo(name)
                    info.size = 2
                    tar.addfile(info, io.BytesIO(b"ok"))
            out = os.path.join(tmp, "dicdir")
            unidic._unpack(archive, out)
            self.assertEqual(sorted(os.listdir(out)), ["mecabrc", "sys.dic"])
            self.assertFalse(os.path.exists(os.path.join(tmp, "evil")))

    def test_unidic_download_checks_the_hash(self):
        from unittest import mock
        from kitten_inference.tts2_tiny.frontend import unidic
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "not-the-archive.tar.gz"
            src.write_bytes(b"something else")
            with mock.patch.object(unidic, "URL", src.as_uri()):
                with self.assertRaises(OSError):
                    unidic._download(os.path.join(tmp, "out"))

    def test_unidic_failed_download_explains(self):
        from unittest import mock
        from kitten_inference.tts2_tiny.frontend import unidic
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(unidic, "_installed", return_value=None), \
                mock.patch.object(unidic, "cache_root", return_value=tmp), \
                mock.patch.object(unidic, "_download", side_effect=OSError("no network")):
            with self.assertRaises(RuntimeError) as cm:
                unidic.dictionary_dir()
            self.assertIn("unidic-lite", str(cm.exception))
            self.assertEqual(os.listdir(tmp), [])     # no partial files left


class SpacyModelTest(unittest.TestCase):
    """spaCy's small English model ships inside the package and is loaded from there: no download and no pip."""

    def test_the_model_ships_with_the_package(self):
        folder = Path(convert.__file__).resolve().parent / "frontend" / "en_core_web_sm"
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["lang"], meta["name"], meta["version"], meta["license"]),
                         ("en", "core_web_sm", "3.8.0", "MIT"))
        for part in ("config.cfg", "tokenizer", "tok2vec/model", "tagger/model", "vocab/strings.json", "LICENSE"):
            self.assertTrue((folder / part).is_file(), part)

    def test_loads_without_download_or_pip(self):
        try:
            import spacy.cli  # noqa: F401
        except ImportError:
            self.skipTest("spaCy is not installed on this Python")
        import subprocess
        from unittest import mock
        from kitten_inference.tts2_tiny.frontend.misaki import en
        refuse = AssertionError("the English front end must not download or install anything")
        with mock.patch("spacy.cli.download", side_effect=refuse), \
                mock.patch.object(subprocess, "Popen", side_effect=refuse):
            g2p = en.G2P(trf=False, british=False, unk="")
            phonemes, _ = g2p("Hello world.")
        packaged = Path(convert.__file__).resolve().parent / "frontend" / "en_core_web_sm"
        self.assertEqual(Path(en.SPACY_MODEL_DIR), packaged)
        self.assertEqual(g2p.nlp.meta["version"], "3.8.0")
        self.assertEqual(g2p.nlp.pipe_names, ["tok2vec", "tagger"])
        self.assertTrue(phonemes.strip())


class GraphTest(unittest.TestCase):
    def nodes(self, fam):
        return convert.build_arch(ARCHS[fam], fam)["nodes"]

    def test_outputs_and_inputs(self):
        for fam in ARCHS:
            a = convert.build_arch(ARCHS[fam], fam)
            self.assertEqual([o["name"] for o in a["outputs"]], ["t_audio", "t_dur"])
            self.assertEqual([i["name"] for i in a["inputs"]], ["phoneme_ids", "style_dec", "style_pred"])
            produced = {o for n in a["nodes"] for o in n["outputs"]}
            for n in a["nodes"]:      # every input is produced earlier or is a graph input
                for i in n["inputs"]:
                    self.assertTrue(i in produced or i in ("phoneme_ids", "style_dec", "style_pred"), (fam, i))

    def test_magnitude_and_resblocks(self):
        mini, nano, pico = (self.nodes(f) for f in ("mini", "nano", "pico"))
        self.assertEqual([n["op"] for n in mini if n["name"] == "gen_mag"], ["softplus_fp32"])
        self.assertEqual([n["op"] for n in nano if n["name"] == "gen_mag"], ["exp_fp32"])
        rb = lambda ns: sorted({n["name"].split("_d")[0] for n in ns if n["name"].startswith("gen_rb") and "_d" in n["name"]})
        self.assertEqual(len(rb(mini)), 6)      # 2 upsample stages x kernels 3/7/11
        self.assertEqual(len(rb(nano)), 4)      # 2 x kernels 3/3
        k = {n["name"]: n["kernel_size"] for n in mini if n["name"].endswith("_d0_c1") and n["name"].startswith("gen_rb0")}
        self.assertEqual(sorted(k.values()), [3, 7, 11])

    def test_widths(self):
        get = lambda ns, name, key: next(n[key] for n in ns if n["name"] == name)
        mini, nano, pico = (self.nodes(f) for f in ("mini", "nano", "pico"))
        self.assertEqual(get(mini, "enc_c1", "C_out"), 1024)
        self.assertEqual(get(mini, "dec3_c1", "C_out"), 512)
        self.assertEqual(get(nano, "enc_c1", "C_out"), 192)
        self.assertEqual(get(pico, "enc_c1", "C_out"), 128)
        self.assertEqual(get(pico, "bert_attn", "num_heads"), 6)
        self.assertEqual(get(pico, "bert_attn", "embed_dim"), 384)
        self.assertEqual(get(pico, "bert_ff1", "C_out"), 1024)
        self.assertEqual(sum(1 for n in pico if n["op"] == "lstm_fp32" and n["name"].startswith("de_lstm")), 2)

    def test_reference_signal_options(self):
        for fam in ARCHS:
            ns = self.nodes(fam)
            sine = next(n for n in ns if n["op"] == "sine_gen_fp32")
            self.assertTrue(sine["torch_order"] and sine["noise"])
            self.assertEqual(sine["upsample_scale"], 300)
            self.assertTrue(next(n for n in ns if n["op"] == "stft_fp32")["torch_stft"])
            self.assertTrue(next(n for n in ns if n["op"] == "istft_fp32")["normalized"])

    def test_split_for_metal(self):
        # Metal runs the graph in two parts: text and durations on the CPU, the rest on Metal.
        from kitten_inference.tts2_tiny.model import split_for_metal
        for fam in ARCHS:
            a = convert.build_arch(ARCHS[fam], fam)
            hid, style = int(ARCHS[fam]["hidden_dim"]), int(ARCHS[fam]["style_dim"])
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x_fp32_arch.json")
                Path(path).write_text(json.dumps(a))
                tp, dp = split_for_metal(path)
                text, dec = json.loads(Path(tp).read_text()), json.loads(Path(dp).read_text())
            # the CPU part ends after the F0 and energy predictor; every node that sets the frame counts or F0 is in it
            self.assertEqual(text["nodes"][-1]["name"], "dec_n_conv")
            self.assertTrue({"dur_sum", "len_reg_en", "len_reg_asr", "shared_lstm", "dec_f0_conv"}
                            <= {n["name"] for n in text["nodes"]})
            self.assertEqual(text["nodes"] + dec["nodes"], a["nodes"])
            self.assertEqual([o["name"] for o in dec["outputs"]], ["t_audio"])
            self.assertEqual({o["name"] for o in text["outputs"]}, {"asr", "t_f0", "t_f0d", "t_nd", "t_dur"})
            widths = {i["name"]: i["shape"][-1] for i in dec["inputs"]}
            self.assertEqual(widths, {"style_dec": style, "asr": hid, "t_f0": 1, "t_f0d": 1,
                                      "t_nd": 1}, fam)
            made = {o for n in dec["nodes"] for o in n["outputs"]}
            for n in dec["nodes"]:
                for i in n["inputs"]:
                    self.assertTrue(i in made or i in widths, (fam, i))

    def test_int8_graph(self):
        a = convert.int8_arch(convert.build_arch(ARCHS["nano"], "x"), "y")
        ops = {n["name"]: n["op"] for n in a["nodes"]}
        self.assertEqual(ops["gen_post"], "conv1d_fp32")
        self.assertEqual(ops["enc_c1"], "conv1d_int8")
        self.assertEqual(ops["bert_ff1"], "seqgemm_int8")
        self.assertEqual(ops["bert_attn"], "attention_int8")
        self.assertEqual(ops["dur_proj"], "seqgemm_fp32")


class WeightsFormatTest(unittest.TestCase):
    def test_w1x1_round_trip_and_int8_layout(self):
        w = convert.W1X1()
        rng = np.random.default_rng(0)
        conv = rng.standard_normal((8, 4, 3)).astype(np.float32)
        w.add("enc_c1.weight", conv)
        w.add("enc_c1.bias", np.ones(8, np.float32))
        arch = {"nodes": [{"name": "enc_c1", "op": "conv1d_fp32"}]}
        q = convert.int8_weights(w, arch)
        self.assertEqual(set(q.blobs), {"enc_c1.weight_i8", "enc_c1.w_scales", "enc_c1.bias"})
        blob = q.blobs["enc_c1.weight_i8"]
        ndim = struct.unpack_from("<H", blob, 2)[0]
        shape = struct.unpack_from("<%di" % ndim, blob, 4)
        self.assertEqual(shape, (8, 12))
        i8 = np.frombuffer(blob, np.int8, offset=4 + 4 * ndim).reshape(shape)
        sc = np.frombuffer(q.blobs["enc_c1.w_scales"], np.float32, offset=8)
        back = (i8 * sc[:, None]).reshape(8, 3, 4).transpose(0, 2, 1)     # [C_out, K, C_in] -> [C_out, C_in, K]
        self.assertLess(float(np.max(np.abs(back - conv))), float(sc.max()) * 0.51)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.bin")
            w.write(p)
            raw = Path(p).read_bytes()
            self.assertEqual(raw[:4], b"W1X1")
            self.assertEqual(struct.unpack_from("<II", raw, 4), (1, 2))

    def test_weight_norm_folding(self):
        v = np.random.default_rng(1).standard_normal((5, 3, 2)).astype(np.float32)
        g = np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1)
        w = convert._wn({"x.weight_g": g, "x.weight_v": v}, "x")
        norms = np.sqrt((w.reshape(5, -1) ** 2).sum(1))
        np.testing.assert_allclose(norms, g.reshape(-1), rtol=1e-5)


    def test_temporary_names_differ_per_thread(self):
        import threading
        names = []
        threads = [threading.Thread(target=lambda: names.append(convert._tmp_name("/x/w.bin"))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(names)), 8)
        self.assertTrue(all(n.startswith("/x/w.bin.") and n.endswith(".tmp") for n in names))


class SpeedTest(unittest.TestCase):
    def test_speed_range(self):
        from kitten_inference.tts2_tiny.model import check_speed
        for ok in (0.25, 1, 1.0, "2", 4.0):
            self.assertEqual(check_speed(ok), float(ok))
        for bad in (0, -1, 0.2, 4.01, 100, float("nan"), float("inf"), None, "fast"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_speed(bad)

    def test_generate_to_file_has_no_sample_rate_option(self):
        import inspect
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        self.assertNotIn("sample_rate", inspect.signature(KittenTTS2Tiny.generate_to_file).parameters)


class ConverterCommandTest(unittest.TestCase):
    def test_runs_as_a_module_without_runpy_warning(self):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m", "kitten_inference.tts2_tiny.convert", "--help"],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("repo_dir", r.stdout)


def _present(repo):
    """The models (and their languages) whose weights file is in the folder (a partial download is fine)."""
    cfg = json.loads(Path(repo, "config.json").read_text())
    models = [k for k, e in cfg["models"].items() if Path(repo, e["safetensors"]).is_file()]
    langs = [l for l, e in cfg["languages"].items() if e["model"] in models and Path(repo, e["voices"]).is_file()]
    return models, langs


@unittest.skipUnless(REPO, "KITTEN_TTS2_TINY_REPO is not set")
class RepoTest(unittest.TestCase):
    def test_convert_every_model(self):
        with tempfile.TemporaryDirectory() as d:
            m = convert.convert_repo(REPO, d, models=_present(REPO)[0], voice_bins=False)
            self.assertTrue(m["models"])
            for files in m["models"].values():
                for f in files.values():
                    self.assertGreater(os.path.getsize(os.path.join(d, f)), 0)

    def test_synthesis_is_finite_and_reproducible(self):
        from kitten_inference import KittenTTS
        tts = KittenTTS(REPO)
        self.assertEqual(tts.family, "tts2_tiny")
        for lang in _present(REPO)[1][:2]:
            a = tts.generate("Hello there, this is a short test.", language=lang)
            b = tts.generate("Hello there, this is a short test.", language=lang)
            self.assertTrue(np.isfinite(a).all())
            self.assertGreater(len(a), 24000 * 0.5)
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(tts.generate("Hello.", speed=2.0)) < len(tts.generate("Hello.", speed=0.7)), True)

    def test_low_memory_matches_the_default_mode(self):
        from kitten_inference import KittenTTS
        lang = _present(REPO)[1][0]
        text = "Hello there, this is a short test of the low memory mode."
        a = KittenTTS(REPO, backend="cpu").generate(text, language=lang)
        low = KittenTTS(REPO, low_memory=True)
        b = low.generate(text, language=lang)
        self.assertEqual(low._model.backend, "cpu")
        self.assertEqual(len(a), len(b))
        snr = 10 * np.log10(np.sum(a.astype(np.float64) ** 2) / max(np.sum((a.astype(np.float64) - b) ** 2), 1e-30))
        self.assertGreater(snr, 20.0)   # a different convolution method: rounding differences only (pico x86-64: 31 dB)
        self.assertTrue(np.array_equal(b, low.generate(text, language=lang)))

    def test_edge_inputs(self):
        from kitten_inference import KittenTTS
        tts = KittenTTS(REPO)
        lang = _present(REPO)[1][0]
        self.assertEqual(len(tts.generate("", language=lang)), 0)
        self.assertEqual(len(tts.generate("   ", language=lang)), 0)
        for speed in (0.0, 0.01, 5.0, float("nan")):
            with self.assertRaises(ValueError):
                tts.generate("Hello.", language=lang, speed=speed)
        long_text = "The quick brown fox jumps over the lazy dog near the river. " * 60
        chunks = tts.phonemize(long_text, language=lang)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(c) <= 350 for c in chunks), max(map(len, chunks)))
        audio = tts.generate(long_text, language=lang)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(len(audio), 24000 * 60)

    def test_metal_split_matches_the_whole_graph_on_the_cpu(self):
        # The two graphs Metal uses, both run on the CPU, give the whole graph's audio bit for bit.
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        from kitten_inference.tts2_tiny.model import split_for_metal
        tts = KittenTTS2Tiny(REPO, backend="cpu")
        lang = _present(REPO)[1][0]
        tokens = tts._tokens(tts.phonemize("Hello there, this is a short test.", language=lang)[0])
        lang_, key = tts.resolve_voice(None, lang)
        style = tts._pack(lang_, key)[min(len(tokens) - 1, 509)][None]
        arch, weights = tts._converted(tts.config["languages"][lang]["model"])
        paths = split_for_metal(arch)
        with tts._cm._ENGINE_GATE.use(tts._mi, "cpu"):
            split = tuple(tts._mi.InferenceModel(p, weights) for p in paths)
        dec_inputs = frozenset(i["name"] for i in json.loads(Path(paths[1]).read_text())["inputs"])
        tts._sessions[tts.config["languages"][lang]["model"]] = split + (dec_inputs, "cpu")
        for speed in (1.0, 1.3):
            wav, dur = KittenTTS2Tiny(REPO, backend="cpu")._run(lang, tokens, style, speed)
            wav2, dur2 = tts._run(lang, tokens, style, speed)
            self.assertTrue(np.array_equal(dur, dur2), speed)
            self.assertTrue(np.array_equal(wav, wav2), speed)

    @unittest.skipUnless(__import__("sys").platform == "darwin", "Metal: macOS only")
    def test_metal_has_the_cpu_durations(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        cpu, metal = KittenTTS2Tiny(REPO, backend="cpu"), KittenTTS2Tiny(REPO, backend="metal")
        lang = _present(REPO)[1][0]
        for text in ("Hello there, this is a short test.", "The quick brown fox jumps over the lazy dog."):
            a, b = cpu.generate(text, language=lang), metal.generate(text, language=lang)
            self.assertEqual(len(a), len(b))
            snr = 10 * np.log10(np.sum(a.astype(np.float64) ** 2) / max(np.sum((a.astype(np.float64) - b) ** 2), 1e-30))
            self.assertGreater(snr, 40.0)   # F0 comes from the CPU part, so the excitation noise is the CPU's too
        # float32 defaults to Metal on Apple silicon, int8 to the CPU
        default, int8 = KittenTTS2Tiny(REPO), KittenTTS2Tiny(REPO, precision="int8")
        default.generate("Hello.", language=lang)
        int8.generate("Hello.", language=lang)
        if __import__("platform").machine() == "arm64":
            self.assertEqual(default.backend, "metal")
        self.assertEqual(int8.backend, "cpu")

    def test_cli_lists_voices_per_language(self):
        import contextlib
        import io
        from kitten_inference._cli import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--model", REPO, "--list-voices"]), 0)
        lines = out.getvalue().splitlines()
        from kitten_inference import KittenTTS
        langs = KittenTTS(REPO).languages()
        self.assertEqual([line.split(":")[0] for line in lines], langs)

    def test_list_voices_without_a_language_names_each_voice_once(self):
        from kitten_inference import KittenTTS
        tts = KittenTTS(REPO)
        names = tts.list_voices()
        self.assertEqual(len(names), len(set(names)))
        per_language = [n for lang in tts.languages() for n in tts.list_voices(language=lang)]
        self.assertEqual(names, list(dict.fromkeys(per_language)))

    def test_phonemize_returns_the_chunks_the_model_speaks(self):
        from kitten_inference import KittenTTS
        tts = KittenTTS(REPO)
        lang = _present(REPO)[1][-1]
        chunks = tts.phonemize("Hello there. This is a test.", language=lang)
        self.assertTrue(chunks and all(isinstance(c, str) and c for c in chunks))
        self.assertEqual(chunks, tts.phonemize("Hello there. This is a test.", language=lang))


if __name__ == "__main__":
    unittest.main()
