"""The model loader on tiny written checkpoints, without PyTorch: reading config.json, listing the weight files,
the safetensors header, where a model name leads, and what is said about a tensor that is missing or has another
shape. The loader file is run with a stand-in "torch" module, so nothing here computes with tensors: loading real
weights is a GPU test."""
import importlib.util
import json
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import kitten_inference.asr as asr

T = "thinker.model."


@pytest.fixture
def loader(fake_torch):
    """_loader.py as a module of its own, bound to the stand-in torch (the real module is not touched)."""
    fake_torch()
    path = Path(asr.__file__).with_name("_loader.py")
    spec = importlib.util.spec_from_file_location("kitten_asr_loader_under_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(folder, text=None, audio=None, thinker=None, drop=()):
    """A small valid config.json; text / audio / thinker replace values, drop removes "section.key" entries."""
    config = {"model_type": "any name", "thinker_config": {
        "audio_token_id": 7,
        "text_config": dict(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=4, rms_norm_eps=1e-6, rope_theta=1000000, vocab_size=32, intermediate_size=16),
        "audio_config": dict(d_model=8, encoder_attention_heads=2, encoder_layers=1, num_mel_bins=128, n_window=50,
                             n_window_infer=800, encoder_ffn_dim=16, downsample_hidden_size=4, output_dim=8,
                             max_source_positions=1500)}}
    th = config["thinker_config"]
    th["text_config"].update(text or {})
    th["audio_config"].update(audio or {})
    th.update(thinker or {})
    for entry in drop:
        section, _, key = entry.partition(".")
        del (th if section == "thinker" else th[section])[key]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def write_weights(path, tensors, metadata=None):
    """A safetensors file: 8 bytes header length, the JSON header, then the data. tensors: {name: (dtype code,
    shape, bytes)}."""
    header, data = {}, b""
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (code, shape, raw) in tensors.items():
        header[name] = {"dtype": code, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    head = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(head)) + head + data)
    return str(path)


class Tok:
    """A tokenizer for load(): the two stop tokens and one audio placeholder per prompt."""
    ids = {"<|im_end|>": 1, "<|endoftext|>": 2}
    placeholders = 1

    def __init__(self, path):
        assert os.path.basename(path) == "tokenizer.json"

    def token_to_id(self, token):
        return self.ids.get(token)

    def encode(self, text):
        return [3] + [7] * (self.placeholders * text.count("<|audio_pad|>")) + [4]


# ---------------------------------------------------------------- small pieces
def test_prompt_text(loader):
    base = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")
    assert loader._prompt_text() == base and base.count("<|audio_pad|>") == 1
    assert loader._prompt_text("", "English") == base + "language English<asr_text>"
    assert loader._prompt_text("Be brief.").startswith("<|im_start|>system\nBe brief.<|im_end|>\n")


def test_dtype_names(loader):
    torch = loader.torch
    assert loader._dtype("bf16") is torch.bfloat16 and loader._dtype("fp16") is torch.float16
    assert loader._dtype(torch.float32) is torch.float32
    for bad in ("int8", None, torch.int64):
        with pytest.raises(ValueError, match=r"dtype must be a floating torch dtype or one of "
                                             r"\['bf16', 'fp16', 'fp32'\]"):
            loader._dtype(bad)


def test_conv_lengths_and_mel_filters(loader):
    assert [loader._down3(n) for n in (1, 2, 100, 128)] == [1, 1, 13, 16]
    filters = loader._mel_filters(128)
    assert filters.shape == (201, 128) and filters.dtype == np.float64
    assert np.isfinite(filters).all() and filters.min() >= 0.0 and filters.max() > 0.0


def test_stop_ids(loader):
    cfg = {"audio_id": 7, "vocab": 32}
    assert loader._stop_ids(Tok("tokenizer.json"), cfg) == {1, 2}
    missing = Tok("tokenizer.json")
    missing.ids = {"<|im_end|>": 1}
    with pytest.raises(ValueError, match="the tokenizer has no id for one of the stop tokens"):
        loader._stop_ids(missing, cfg)
    outside = Tok("tokenizer.json")
    outside.ids = {"<|im_end|>": 1, "<|endoftext|>": 40}
    with pytest.raises(ValueError, match=r"token ids \[40\] are outside the embedding table \(32 rows\)"):
        loader._stop_ids(outside, cfg)
    with pytest.raises(ValueError, match=r"token ids \[32\] are outside the embedding table"):
        loader._stop_ids(Tok("tokenizer.json"), {"audio_id": 32, "vocab": 32})


def test_tokenizer_file_is_used_as_stored(loader, tmp_path, monkeypatch):
    calls = []

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            calls.append(("from_file", path))
            return cls()

        def no_padding(self):
            calls.append("no_padding")

        def no_truncation(self):
            calls.append("no_truncation")

        def encode(self, text, add_special_tokens=True):
            calls.append(("encode", text, add_special_tokens))
            return types.SimpleNamespace(ids=[5, 6])

        def decode(self, ids, skip_special_tokens=False):
            calls.append(("decode", ids, skip_special_tokens))
            return "text"

        def token_to_id(self, token):
            return {"<|im_end|>": 1}.get(token)

    module = types.ModuleType("tokenizers")
    module.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    path = tmp_path / "tokenizer.json"
    path.write_text("{}")
    tok = loader._Tok(str(path))
    assert calls == [("from_file", str(path)), "no_padding", "no_truncation"]
    assert tok.encode("hi") == [5, 6] and calls[-1] == ("encode", "hi", False)      # no special tokens added
    assert tok.decode((np.int64(5), 6)) == "text" and calls[-1] == ("decode", [5, 6], True)
    assert all(type(i) is int for i in calls[-1][1])
    assert tok.token_to_id("<|im_end|>") == 1 and tok.token_to_id("nope") is None
    with pytest.raises(FileNotFoundError, match="no tokenizer.json in"):
        loader._Tok(str(tmp_path / "other" / "tokenizer.json"))


# ---------------------------------------------------------------- config.json
def test_config_is_read_by_layout_not_by_name(loader, tmp_path):
    cfg, dims = loader._config(str(write_config(tmp_path)))
    assert cfg == dict(H=8, L=1, nh=2, nkv=1, hd=4, eps=1e-6, rope_theta=1000000, vocab=32, enc_d=8, enc_nh=2,
                       enc_layers=1, n_mels=128, chunk_frames=100, block_frames=800, audio_id=7)
    assert dims == dict(I=16, ffn=16, C=4, out=8, max_pos=1500)
    assert type(cfg["eps"]) is float and all(type(cfg[k]) is int for k in ("H", "hd", "vocab", "audio_id"))
    cfg, _ = loader._config(str(write_config(tmp_path, text={"hidden_size": 8.0, "head_dim": 6})))
    assert cfg["H"] == 8 and type(cfg["H"]) is int and cfg["hd"] == 6       # head_dim is read, not derived


@pytest.mark.parametrize("changes, message", [
    (dict(text={"hidden_size": "8"}), r"thinker_config\.text_config\.hidden_size is missing or not a whole number "
                                      r"\(got '8'\)"),
    (dict(text={"hidden_size": 8.5}), r"text_config\.hidden_size is missing or not a whole number \(got 8\.5\)"),
    (dict(text={"hidden_size": True}), r"text_config\.hidden_size is missing or not a whole number \(got True\)"),
    (dict(drop=["text_config.head_dim"]), r"text_config\.head_dim is missing or not a whole number \(got None\)"),
    (dict(drop=["text_config.rms_norm_eps"]), r"text_config\.rms_norm_eps is missing or not a number"),
    (dict(drop=["audio_config.n_window"]), r"audio_config\.n_window is missing or not a whole number"),
    (dict(drop=["thinker.audio_token_id"]), r"thinker_config\.audio_token_id is missing or not a whole number"),
    (dict(drop=["thinker.text_config"]), r"thinker_config\.text_config is missing"),
    (dict(thinker={"audio_config": [1, 2]}), r"thinker_config\.audio_config is missing"),
    (dict(text={"num_hidden_layers": 0}), r"these sizes must be positive: \['L'\]"),
    (dict(audio={"output_dim": 16}), "audio output_dim 16 differs from the text hidden_size 8"),
    (dict(audio={"d_model": 9}), "audio d_model 9 must be even and divisible by encoder_attention_heads 2"),
    (dict(text={"num_attention_heads": 3, "num_key_value_heads": 2}),
     "num_attention_heads 3 is not a multiple of num_key_value_heads 2"),
    (dict(audio={"n_window_infer": 750}), r"n_window_infer 750 is not a multiple of 2 \* n_window 100"),
    (dict(audio={"max_source_positions": 5}), "max_source_positions 5 is smaller than one conv chunk"),
])
def test_config_that_is_refused(loader, tmp_path, changes, message):
    folder = write_config(tmp_path, **changes)
    with pytest.raises(ValueError, match=message) as e:
        loader._config(str(folder))
    assert "config.json" in str(e.value)                 # the message names the file


def test_config_missing_or_broken(loader, tmp_path):
    with pytest.raises(FileNotFoundError, match="no config.json in"):
        loader._config(str(tmp_path))
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"config\.json is not valid JSON"):
        loader._config(str(tmp_path))
    (tmp_path / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="thinker_config is missing"):
        loader._config(str(tmp_path))


def test_spectrogram_settings_must_be_the_engines(loader, tmp_path):
    loader._check_frontend(str(tmp_path), 128)           # no file: nothing to check
    path = tmp_path / "preprocessor_config.json"
    path.write_text(json.dumps({"feature_size": 128, "n_fft": 400, "hop_length": 160, "sampling_rate": 16000,
                                "padding_value": 0.0}))
    loader._check_frontend(str(tmp_path), 128)
    path.write_text(json.dumps({"feature_size": 80, "hop_length": 160}))
    with pytest.raises(ValueError, match=r"preprocessor_config\.json says \{'feature_size': 80\}, but config\.json "
                                         r"and the engine's spectrogram need"):
        loader._check_frontend(str(tmp_path), 128)


# ---------------------------------------------------------------- where a model name leads
def test_a_folder_is_used_as_it_is(loader, tmp_path):
    assert loader._resolve(str(tmp_path)) == str(tmp_path) and loader._resolve(tmp_path) == str(tmp_path)


def test_local_looking_names_are_not_sent_to_the_hub(loader, tmp_path, monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: pytest.fail("a local path was sent to the Hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"x")
    (tmp_path / "models").mkdir()
    with pytest.raises(ValueError, match="is a file; expected a model folder or a Hugging Face repo id"):
        loader._resolve("weights.bin")
    for name in ("./tiny", "../tiny", "~/tiny", "a/b/c", str(tmp_path / "tiny"), "C:\\models\\tiny"):
        with pytest.raises(FileNotFoundError, match="model folder not found: "):
            loader._resolve(name)
    with pytest.raises(FileNotFoundError, match=r"model folder not found: models/tiny \(read as a local path "
                                                r"because the folder 'models' exists\)"):
        loader._resolve("models/tiny")


def test_a_repo_id_is_downloaded_into_the_cache_dir(loader, tmp_path, monkeypatch):
    asked = []

    def snapshot_download(repo, allow_patterns=None, cache_dir=None):
        asked.append((repo, list(allow_patterns), cache_dir))
        return str(tmp_path / "snapshot")
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.chdir(tmp_path)
    assert loader._resolve("KittenML/kitten-asr-tiny", "cache") == str(tmp_path / "snapshot")
    (repo, patterns, cache_dir), = asked
    assert repo == "KittenML/kitten-asr-tiny" and cache_dir == "cache"
    assert {"config.json", "tokenizer.json", "model*.safetensors", "model.safetensors.index.json"} <= set(patterns)
    assert loader._resolve("KittenML/kitten-asr-tiny") and asked[-1][2] is None

    def offline(repo, allow_patterns=None, cache_dir=None):
        raise OSError("offline")
    hub.snapshot_download = offline
    with pytest.raises(RuntimeError, match=r"'KittenML/kitten-asr-tiny' is not a local folder and could not be "
                                           r"downloaded as a Hugging Face repo id \(OSError: offline\)"):
        loader._resolve("KittenML/kitten-asr-tiny")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)        # not installed
    with pytest.raises(ImportError, match="is not a local folder, and downloading it needs the huggingface-hub"):
        loader._resolve("KittenML/kitten-asr-tiny")


# ---------------------------------------------------------------- weight files
def test_weight_file_listing(loader, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"no \*\.safetensors weight file in"):
        loader._weight_files(str(tmp_path))
    for name in ("b.safetensors", "a.safetensors", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert loader._weight_files(str(tmp_path)) == [str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")]

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "model-00002-of-00002.safetensors",
                                                "y": "model-00001-of-00002.safetensors",
                                                "z": "model-00001-of-00002.safetensors"}}))
    with pytest.raises(FileNotFoundError, match=r"weight shard\(s\) listed in .*index\.json but not on disk"):
        loader._weight_files(str(tmp_path))
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (tmp_path / name).write_bytes(b"")
    assert loader._weight_files(str(tmp_path)) == [str(tmp_path / "model-00001-of-00002.safetensors"),
                                                   str(tmp_path / "model-00002-of-00002.safetensors")]
    index.write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match=r"index\.json has no weight_map"):
        loader._weight_files(str(tmp_path))

    (tmp_path / "model.safetensors").write_bytes(b"")                 # the single file wins over everything
    assert loader._weight_files(str(tmp_path)) == [str(tmp_path / "model.safetensors")]


def test_safetensors_header_and_tensor_bytes(loader, tmp_path):
    a = np.arange(6, dtype="<f4").reshape(2, 3)
    b = np.array([1.5, -2.0, 0.25, 8.0], dtype="<f2")
    path = write_weights(tmp_path / "model.safetensors", {"a": ("F32", a.shape, a.tobytes()),
                                                          "b": ("F16", b.shape, b.tobytes()),
                                                          "c": ("I64", (1,), bytes(8))},
                         metadata={"format": "pt"})
    files = loader._Files([path], "cuda", "bf16")
    assert files.has("a") and files.has("b") and not files.has("__metadata__") and not files.has("d")
    assert np.array_equal(files._host("a", (2, 3)), a) and np.array_equal(files._host("b", [4]), b)
    assert files.used == {"a", "b"} and files.unused(("a", "b", "c")) == ["c"] and files.unused(("x",)) == []

    with pytest.raises(ValueError, match=r"tensor w is missing from the weight file\(s\) \(they hold 3 tensors, "
                                         r"e\.g\. a, b, c\)"):
        files._host("w", (2, 3))
    with pytest.raises(ValueError, match=r"tensor a has shape \[2, 3\], but config\.json implies \[3, 2\]"):
        files._host("a", (3, 2))
    with pytest.raises(ValueError, match=r"tensor c is stored as I64; this loader reads \['BF16', 'F16', 'F32'\]"):
        files._host("c", (1,))


def test_tensors_of_several_files(loader, tmp_path):
    one = np.ones(2, dtype="<f4")
    first = write_weights(tmp_path / "1.safetensors", {"a": ("F32", (2,), one.tobytes())})
    second = write_weights(tmp_path / "2.safetensors", {"b": ("F32", (2,), (2 * one).tobytes())})
    files = loader._Files([first, second], "cuda", "bf16")
    assert files._host("a", (2,)).tolist() == [1.0, 1.0] and files._host("b", (2,)).tolist() == [2.0, 2.0]
    again = write_weights(tmp_path / "3.safetensors", {"a": ("F32", (2,), one.tobytes())})
    with pytest.raises(ValueError, match="tensor a is in more than one weight file"):
        loader._Files([first, again], "cuda", "bf16")


def _raw_file(tmp_path, content):
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    return str(path)


def _framed(header, data=b""):
    """`header` behind its 8-byte length, as in a safetensors file."""
    return struct.pack("<Q", len(header)) + header + data


@pytest.mark.parametrize("content, message", [
    (b"abcd", r"is not a safetensors file \(header length field -1, file size 4\)"),
    (struct.pack("<Q", 1000) + b"{}", r"is not a safetensors file \(header length field 1000, file size 10\)"),
    (struct.pack("<Q", 0), "is not a safetensors file"),
    (_framed(b"not json!"), "the safetensors header is not valid JSON"),
    (_framed(b"\xff\xfe"), "the safetensors header is not valid JSON"),
    (_framed(b"[1, 2]"), "the safetensors header is not a JSON object"),
    (_framed(b'{"a": 5}'), "malformed header entry for tensor a"),
    (_framed(b'{"a": {"dtype": "F32", "shape": [2]}}'), "malformed header entry for tensor a"),
    (_framed(b'{"a": {"dtype": "F32", "shape": ["x"], "data_offsets": [0, 8]}}', bytes(8)),
     "malformed header entry for tensor a"),
])
def test_files_that_are_not_safetensors(loader, tmp_path, content, message):
    with pytest.raises(ValueError, match=message):
        loader._Files([_raw_file(tmp_path, content)], "cuda", "bf16")


def test_tensor_data_that_does_not_fit(loader, tmp_path):
    header = json.dumps({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 100]}}).encode()
    cut_short = _raw_file(tmp_path, _framed(header, bytes(8)))
    with pytest.raises(ValueError, match=r"tensor a lies outside the file \(incomplete download\?\)"):
        loader._Files([cut_short], "cuda", "bf16")
    path = write_weights(tmp_path / "other.safetensors", {"a": ("F32", (2, 3), bytes(20))})
    files = loader._Files([path], "cuda", "bf16")
    with pytest.raises(ValueError, match=r"tensor a: 20 bytes in the file do not match shape \[2, 3\] of F32"):
        files._host("a", (2, 3))


# ---------------------------------------------------------------- load(), up to the first tensor
@pytest.fixture
def checkpoint(loader, tmp_path, monkeypatch):
    """A model folder with a valid config and tokenizer; a test writes the weights. Returns (folder, Tok)."""
    monkeypatch.setattr(loader, "_Tok", Tok)
    monkeypatch.setattr(Tok, "ids", dict(Tok.ids))
    monkeypatch.setattr(Tok, "placeholders", 1)
    folder = write_config(tmp_path / "model")
    (folder / "tokenizer.json").write_text("{}")
    return folder


def test_load_names_a_missing_tensor(loader, checkpoint):
    write_weights(checkpoint / "model.safetensors", {"other.weight": ("F32", (2,), bytes(8))})
    with pytest.raises(ValueError, match=r"tensor thinker\.model\.embed_tokens\.weight is missing from the weight "
                                         r"file\(s\) \(they hold 1 tensors, e\.g\. other\.weight\)"):
        loader.load(str(checkpoint), "cuda", "bf16")


def test_load_names_a_tensor_of_another_shape(loader, checkpoint):
    write_weights(checkpoint / "model.safetensors", {T + "embed_tokens.weight": ("F32", (4, 8), bytes(4 * 8 * 4))})
    with pytest.raises(ValueError, match=r"tensor thinker\.model\.embed_tokens\.weight has shape \[4, 8\], but "
                                         r"config\.json implies \[32, 8\]"):
        loader.load(str(checkpoint), "cuda", "bf16")


def test_load_checks_tokenizer_and_config_together(loader, checkpoint, monkeypatch):
    write_weights(checkpoint / "model.safetensors", {})
    monkeypatch.setattr(Tok, "placeholders", 0)
    with pytest.raises(ValueError, match=r"the prompt holds 0 tokens with the audio placeholder id 7 \(expected "
                                         r"1\): tokenizer\.json and config\.json do not belong together"):
        loader.load(str(checkpoint), "cuda", "bf16")
    monkeypatch.setattr(Tok, "placeholders", 1)
    monkeypatch.setattr(Tok, "ids", {"<|im_end|>": 1})
    with pytest.raises(ValueError, match="the tokenizer has no id for one of the stop tokens"):
        loader.load(str(checkpoint), "cuda", "bf16")


def test_load_refuses_before_reading_weights(loader, checkpoint):
    with pytest.raises(ValueError, match="dtype must be a floating torch dtype"):
        loader.load(str(checkpoint), "cuda", "int8")
    with pytest.raises(FileNotFoundError, match=r"no \*\.safetensors weight file in"):
        loader.load(str(checkpoint), "cuda", "bf16")
    (checkpoint / "preprocessor_config.json").write_text(json.dumps({"n_fft": 512}))
    with pytest.raises(ValueError, match=r"says \{'n_fft': 512\}"):
        loader.load(str(checkpoint), "cuda", "bf16")
    with pytest.raises(FileNotFoundError, match="model folder not found"):
        loader.load(str(checkpoint / "missing" / "deeper" / "x"), "cuda", "bf16")
