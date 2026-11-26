"""kitten-asr (python -m kitten_inference.asr): what it prints, its exit codes and its one-line errors. The engine is a fake."""
import json
import subprocess
import sys

import numpy as np
import pytest

import kitten_inference
from kitten_inference.asr import _cli
from kitten_inference.asr._types import PoolExhausted

MODEL = "KittenML/kitten-asr-tiny"
SR = 16000
ERR = "kitten-asr: "      # an error is one line under the command's name, like argparse's own

VERSION_ONLY = """
import runpy, sys
sys.argv = ["kitten_inference.asr", "--version"]
try:
    runpy.run_module("kitten_inference.asr", run_name="__main__")
except SystemExit as e:
    print("exit", e.code, sorted(m for m in ("numpy", "torch", "soundfile") if m in sys.modules))
"""


@pytest.fixture
def wav(tmp_path):
    """wav("a.wav", seconds) writes a 16 kHz recording of a tone and returns its path."""
    def write(name="speech.wav", seconds=1.0):
        import soundfile as sf
        samples = 0.1 * np.sin(2 * np.pi * 220.0 * np.arange(int(seconds * SR)) / SR)
        path = tmp_path / name
        sf.write(str(path), samples.astype(np.float32), SR)
        return str(path)
    return write


def test_version(capsys):
    assert _cli.main(["--version"]) == 0
    assert capsys.readouterr().out == kitten_inference.__version__ + "\n"


def test_version_through_the_module_needs_no_library():
    out = subprocess.run([sys.executable, "-c", VERSION_ONLY], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [kitten_inference.__version__, "exit 0 []"]


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_needs_no_model_or_engine(capsys, flag):
    assert _cli.main([flag]) == 0
    out = capsys.readouterr().out
    for option in ("--model", "--language", "--backend", "--precision", "--cache-dir", "--json", "kitten-asr"):
        assert option in out


def test_no_arguments_prints_help_and_returns_2(capsys):
    assert _cli.main([]) == 2
    assert "--model" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["speech.wav"],                                                   # no model
    ["--model", MODEL],                                               # no audio
    ["--mod", MODEL, "speech.wav"],                                   # abbreviations are not options
    ["--model", MODEL, "--lang", "English", "speech.wav"],
    ["--model", MODEL, "--js", "speech.wav"],
    ["--model", MODEL, "--cache", "models", "speech.wav"],
    ["--model", MODEL, "--version", "speech.wav"],                    # --version is a command of its own
    ["--model", MODEL, "--no-such-option", "speech.wav"],
])
def test_usage_errors_return_2(fake_engine, capsys, argv):
    with pytest.raises(SystemExit) as e:
        _cli.main(argv)
    assert e.value.code == 2
    assert "error:" in capsys.readouterr().err and fake_engine.engines == []


def test_one_file_prints_its_text(fake_engine, wav, capsys):
    assert _cli.main(["--model", MODEL, wav()]) == 0
    assert capsys.readouterr() == ("n16000\n", "")
    engine = fake_engine.engines[0]
    assert engine.model == MODEL and engine.options == {"cache_dir": None, "precision": "bf16"} and engine.prompts == [None]
    assert engine.calls[0].priority == 10 and engine.closed == 1


def test_several_files_print_their_names(fake_engine, wav, capsys):
    a, b = wav("a.wav", 1.0), wav("b.wav", 2.5)
    assert _cli.main(["--model", MODEL, a, b]) == 0
    assert capsys.readouterr() == (f"{a}: n16000\n{b}: n40000\n", "")
    assert len(fake_engine.engines) == 1                 # one model for all files


def test_json_language_and_cache_dir(fake_engine, wav, capsys, tmp_path):
    a, b = wav("a.wav", 1.0), wav("b.wav", 2.5)
    cache = str(tmp_path / "models")
    assert _cli.main(["--model", MODEL, "--json", "--language", "english", "--cache-dir", cache, a, b]) == 0
    out, err = capsys.readouterr()
    assert err == "" and [json.loads(line) for line in out.splitlines()] == [
        {"file": a, "text": "n16000", "language": "English",
         "segments": [{"start": 0.0, "end": 1.0, "text": "n16000"}]},
        {"file": b, "text": "n40000", "language": "English",
         "segments": [{"start": 0.0, "end": 2.5, "text": "n40000"}]}]
    engine = fake_engine.engines[0]
    assert engine.options == {"cache_dir": cache, "precision": "bf16"} and engine.prompts == ["English"]


def test_json_of_one_silent_file(fake_engine, tmp_path, capsys):
    import soundfile as sf
    path = str(tmp_path / "silence.wav")
    sf.write(path, np.zeros(SR, np.float32), SR)
    assert _cli.main(["--model", MODEL, "--json", path]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "file": path, "text": "", "language": None, "segments": [{"start": 0.0, "end": 1.0, "text": ""}]}


def test_missing_file_is_said_before_the_model_loads(fake_engine, wav, capsys, tmp_path):
    missing = str(tmp_path / "nothing.wav")
    assert _cli.main(["--model", MODEL, wav(), missing]) == 1
    assert capsys.readouterr() == ("", f"{ERR}audio file not found: {missing}\n")
    assert fake_engine.engines == [] and fake_engine.required == []


@pytest.mark.parametrize("error", [ValueError("boom"), RuntimeError("boom"), ImportError("boom"), OSError("boom")])
def test_errors_are_one_line_and_exit_1(fake_engine, wav, capsys, error):
    def reply(call):
        raise error
    fake_engine.reply = reply
    assert _cli.main(["--model", MODEL, wav()]) == 1
    assert capsys.readouterr() == ("", ERR + "boom\n")
    assert fake_engine.engines[0].closed == 1            # the model is closed on the way out


def test_an_engine_error_is_said_under_the_command_name(fake_engine, wav, capsys):
    def reply(call):
        raise PoolExhausted("kitten-asr: KV pool exhausted")
    fake_engine.reply = reply
    assert _cli.main(["--model", MODEL, wav()]) == 1
    # the engine's own name in front of its message is dropped: the line names one program, not two
    assert capsys.readouterr() == ("", ERR + "KV pool exhausted\n")
    assert fake_engine.engines[0].closed == 1 and _cli.PROG + ": " == ERR


def test_missing_pytorch_is_one_line(machine, packages, wav, capsys):
    machine()
    packages(torch=False)
    assert _cli.main(["--model", MODEL, "--backend", "cuda", wav()]) == 1
    assert capsys.readouterr() == (
        "", ERR + "Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch\n")


def test_ctrl_c_returns_130(fake_engine, wav, capsys):
    def reply(call):
        raise KeyboardInterrupt
    fake_engine.reply = reply
    assert _cli.main(["--model", MODEL, wav()]) == 130
    assert capsys.readouterr() == ("", ERR + "cancelled\n")
    assert fake_engine.engines[0].closed == 1


def test_backend_and_precision_go_to_the_model(fake_engine, wav, capsys):
    assert _cli.main(["--model", MODEL, "--backend", "cuda", "--precision", "int8", wav()]) == 0
    assert fake_engine.engines[0].options == {"cache_dir": None, "precision": "int8"}
    capsys.readouterr()
    # a precision of the other backend is one line of error, before any model loads
    assert _cli.main(["--model", MODEL, "--backend", "cuda", "--precision", "q8_0", wav()]) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith(ERR + "precision 'q8_0' is for the CPU and the Mac GPU") and err.count("\n") == 1
    assert len(fake_engine.engines) == 1
