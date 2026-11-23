"""The system message of the decode prompt: "auto" follows the model folder's asr_config.json (empty
without it, as before), "none" turns it off. Unit tests run anywhere; the CPU engine test needs
KITTEN_ASR_TEST_MODEL (see test_cpu.py)."""
import json
import os

import numpy as np
import pytest

from kitten_inference.asr import _cli, _cpu, _system

TEXT = "Transcribe the audio. Keep the speaker's words."


def _folder(tmp_path, content=None, raw=None):
    if content is not None:
        (tmp_path / _system.SYSTEM_PROMPT_FILE).write_text(json.dumps(content), encoding="utf-8")
    elif raw is not None:
        (tmp_path / _system.SYSTEM_PROMPT_FILE).write_text(raw, encoding="utf-8")
    return str(tmp_path)


def test_a_folder_without_the_file_gets_an_empty_system_message(tmp_path):
    assert _system.resolve(_folder(tmp_path)) == ("", "auto:no-file")
    assert _system.resolve(_folder(tmp_path), None) == ("", "auto:no-file")


def test_auto_reads_the_checkpoint_s_own_text(tmp_path):
    folder = _folder(tmp_path, {_system.SYSTEM_PROMPT_KEY: TEXT, "other": 1})
    assert _system.resolve(folder) == (TEXT, "auto:asr_config.json")
    assert _system.resolve(folder, " AUTO ") == (TEXT, "auto:asr_config.json")


def test_none_turns_it_off(tmp_path):
    folder = _folder(tmp_path, {_system.SYSTEM_PROMPT_KEY: TEXT})
    assert _system.resolve(folder, "none") == ("", "none")


def test_an_empty_string_in_the_file_is_kept(tmp_path):
    assert _system.resolve(_folder(tmp_path, {_system.SYSTEM_PROMPT_KEY: ""})) == (
        "", "auto:asr_config.json")


@pytest.mark.parametrize("content, raw", [
    ({"no_such_key": TEXT}, None), ({_system.SYSTEM_PROMPT_KEY: 3}, None), ([TEXT], None), (None, "{not json"),
])
def test_a_bad_file_is_refused_instead_of_guessed(tmp_path, content, raw):
    folder = _folder(tmp_path, content, raw)
    with pytest.raises(ValueError, match="asr_config.json"):
        _system.resolve(folder)
    assert _system.resolve(folder, "none") == ("", "none")        # the way out the error names


@pytest.mark.parametrize("bad", ["", "off", "custom text", 1, True])
def test_unknown_modes_are_refused(bad):
    with pytest.raises(ValueError, match="system_prompt must be one of auto, none"):
        _system.check_mode(bad)


def test_kittenasr_checks_the_mode_before_loading(fake_engine):
    from kitten_inference import KittenASR
    with pytest.raises(ValueError, match="system_prompt must be one of"):
        KittenASR("KittenML/kitten-asr-tiny", system_prompt="off")
    assert fake_engine.engines == []


def test_kittenasr_passes_none_on_and_leaves_auto_to_the_engine(fake_engine):
    from kitten_inference import KittenASR
    KittenASR("KittenML/kitten-asr-tiny").close()
    KittenASR("KittenML/kitten-asr-tiny", system_prompt="none").close()
    assert "system_prompt" not in fake_engine.engines[0].options
    assert fake_engine.engines[1].options["system_prompt"] == "none"


def test_cli_option(fake_engine, tmp_path, capsys):
    import soundfile as sf
    wav = str(tmp_path / "a.wav")
    sf.write(wav, np.zeros(16000, np.float32) + 0.01, 16000)
    assert _cli.main(["--model", "KittenML/kitten-asr-tiny", "--system-prompt", "none", wav]) == 0
    assert fake_engine.engines[0].options["system_prompt"] == "none"
    assert _cli.main(["--model", "KittenML/kitten-asr-tiny", wav]) == 0
    assert "system_prompt" not in fake_engine.engines[1].options
    capsys.readouterr()


def test_cpu_engine_prompt_carries_the_system_message():
    engine = object.__new__(_cpu.CpuEngine)          # only the prompt: no library, no model
    engine._closed, engine._ctx, engine._system = False, object(), TEXT
    p = engine.default_prompt("English")
    assert p.startswith(f"<|im_start|>system\n{TEXT}<|im_end|>\n<|im_start|>user\n")
    assert p.endswith("<|im_start|>assistant\nlanguage English<asr_text>")
    engine._system = ""
    assert engine.default_prompt().startswith("<|im_start|>system\n<|im_end|>\n")


def test_gpu_loader_prompt_carries_the_system_message():
    try:
        import torch  # noqa: F401  the GPU engine's loader needs PyTorch
    except ImportError:
        pytest.skip("platform: PyTorch is not installed (the GPU engine's loader needs it)")
    from kitten_inference.asr import _loader
    p = _loader._prompt_text(TEXT, "English")
    assert p.startswith(f"<|im_start|>system\n{TEXT}<|im_end|>\n") and p.endswith("language English<asr_text>")


def _linked_copy(src, dst, content):
    """The model folder `src` as links in `dst`, with a asr_config.json added (or none)."""
    os.makedirs(dst)
    for name in os.listdir(src):
        if name != _system.SYSTEM_PROMPT_FILE:
            os.symlink(os.path.join(os.path.abspath(src), name), os.path.join(dst, name))
    if content is not None:
        with open(os.path.join(dst, _system.SYSTEM_PROMPT_FILE), "w", encoding="utf-8") as f:
            json.dump(content, f)
    return dst


@pytest.mark.skipif(not os.environ.get("KITTEN_ASR_TEST_MODEL"), reason="model: KITTEN_ASR_TEST_MODEL is not set")
def test_cpu_engine_end_to_end(tmp_path, monkeypatch):
    ok, why = _cpu.available()
    if not ok:
        pytest.skip(f"platform: {why}")
    monkeypatch.setenv(_cpu.CACHE_ENV, str(tmp_path / "cache"))
    from kitten_inference import KittenASR
    folder = _linked_copy(os.environ["KITTEN_ASR_TEST_MODEL"], str(tmp_path / "m"), {_system.SYSTEM_PROMPT_KEY: TEXT})
    for mode, want, source in (("auto", TEXT, "auto:asr_config.json"), ("none", "", "none")):
        with KittenASR(folder, backend="cpu", threads=2, system_prompt=mode) as asr:
            assert asr.capabilities["system_prompt"] == source
            assert asr._engine.default_prompt().startswith(f"<|im_start|>system\n{want}<|im_end|>")
            tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)).astype(np.float32)
            assert isinstance(asr.transcribe(tone), str)


def test_both_engines_download_the_system_prompt_file():
    from kitten_inference.asr import _system
    assert _system.SYSTEM_PROMPT_FILE in _system.HUB_FILES
    src = os.path.join(os.path.dirname(_system.__file__))
    for name in ("_cpu.py", "_loader.py"):
        with open(os.path.join(src, name), encoding="utf-8") as f:
            assert "_published.download(" in f.read(), name


def test_published_download_keeps_configuration_and_fetches_only_selected_weights(tmp_path):
    from kitten_inference.asr import _published, _system

    calls = []

    def snapshot_download(repo, allow_patterns, cache_dir):
        calls.append(allow_patterns)
        for name in allow_patterns:
            if name.startswith("int4/"):
                file = tmp_path / name
                file.parent.mkdir(exist_ok=True)
                file.touch()
        return str(tmp_path)

    assert _published.download(snapshot_download, "example/model", None, "int4", required=True) == str(tmp_path)
    (patterns,) = calls
    assert _system.SYSTEM_PROMPT_FILE in patterns and "tokenizer.json" in patterns
    assert {p for p in patterns if p.endswith(".safetensors")} == {"int4/quantized.safetensors"}
