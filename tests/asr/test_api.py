"""KittenASR without a GPU. The engine is a fake that records what it is asked: how the model's output is read,
where long audio is cut, what transcribe() and a streaming session send to the engine, and what a closed model
says."""
import re
import threading
import time

import numpy as np
import pytest

from kitten_inference.asr import _api, _audio
from kitten_inference.asr._types import PoolExhausted

SR = 16000
MODEL = "KittenML/kitten-asr-tiny"
STREAM_KEY = re.compile(r"[0-9a-f]{12}:\d+\Z")


def tone(seconds, level=0.1, hz=220.0):
    n = int(round(seconds * SR))
    return (level * np.sin(2 * np.pi * hz * np.arange(n) / SR)).astype(np.float32)


def with_gaps(x, *starts):
    """`x` with 300 ms of silence from each of `starts` on."""
    x = x.copy()
    for start in starts:
        x[start:start + 4800] = 0.0
    return x


# ---------------------------------------------------------------- the model's output
@pytest.mark.parametrize("raw, forced, expected", [
    ("language English<asr_text>hello", None, ("English", "hello")),
    ("hello", None, (None, "hello")),                                  # no prefix: all of it is the transcript
    ("hello", "German", ("German", "hello")),
    ("", None, (None, "")),
    ("", "German", ("German", "")),
    ("  language Chinese <asr_text>  ni hao \n", None, ("Chinese", "ni hao")),
    ("LANGUAGE French<asr_text>bonjour", None, ("French", "bonjour")),
    ("language None<asr_text>", None, (None, "")),                     # the model heard no speech
    ("language None<asr_text>", "German", ("German", "")),
    ("<asr_text>hello", "German", ("German", "hello")),
    ("language<asr_text>hello", None, (None, "hello")),
    ("language English<asr_text>hello", "German", ("English", "hello")),
    ("language English<asr_text>one <non_speech> two<asr_text>three", None,
     ("English", "one <non_speech> two<asr_text>three")),              # tags stay; only the first one splits
])
def test_parse_output(raw, forced, expected):
    assert _api.parse_output(raw, forced) == expected


def test_parse_output_reads_what_the_prompt_writes():
    # the prompt of a forced language ends with the same words the model writes by itself
    assert _api.parse_output(_api._LANGUAGE_PREFIX + "English" + _api._TEXT_TAG + "hi") == ("English", "hi")
    assert (_api._LANGUAGE_PREFIX, _api._TEXT_TAG) == ("language ", "<asr_text>")


# ---------------------------------------------------------------- windows
@pytest.mark.parametrize("raw, expected", [
    ("language English<asr_text>hello world", "language English<asr_text>hello world"),
    ("no no no no", "no no no no"),                                      # short repeats are speech, not a loop
    ("language English<asr_text>it is" + " a bit of" * 150 + " a bit", "language English<asr_text>it is a bit of a bit"),
    ("language Chinese<asr_text>你好" + "的" * 60, "language Chinese<asr_text>你好的"),
    ("we will we will we will we will rock you", "we will we will we will we will rock you"),   # loop not at the end
    ("", ""),
])
def test_trim_loop(raw, expected):
    assert _api._trim_loop(raw) == expected


def test_a_run_away_window_keeps_one_copy_of_its_loop():
    job = [(0, SR, _done("language English<asr_text>Thank you." + " Thank you." * 40, "length"))]
    assert _api.KittenASR._collect(job, None, False) == "Thank you. Thank you."   # the text, then one copy


@pytest.mark.parametrize("said", [
    "He said: Thank you. Thank you. Thank you. Thank you.",
    "I said no, no, no, no, no, no, no, no, no, no, no, no",
    "Ha" + " ha" * 21,
    "go team " * 5,
])
def test_repeated_speech_in_a_window_that_ended_by_itself_is_kept(said):
    # the same text from a decode that used up its token budget is a run-away loop, and is cut
    job = [(0, SR, _done("language English<asr_text>" + said, "stop"))]
    assert _api.KittenASR._collect(job, None, False) == said.strip()
    job = [(0, SR, _done("language English<asr_text>" + said, "length"))]
    assert len(_api.KittenASR._collect(job, None, False)) < len(said.strip())


def _done(text, finish="stop"):
    from concurrent.futures import Future

    from kitten_inference.asr._types import DecodeResult
    f = Future()
    f.set_result(DecodeResult(text, [], {"finish_reason": finish}))
    return f


_LOOP = "language English<asr_text>Hello there. " + "and then we went " * 20


def _looping(engine):
    engine.reply = lambda call: _LOOP
    engine.finish = lambda call, text: "length"


def test_live_audio_trims_a_run_away_loop_like_a_file(fake_engine):
    """The same looping window gives the same text from transcribe() and from a live session (partial and final)."""
    fake_engine.reply = lambda call: _LOOP
    model = _api.KittenASR(MODEL)
    try:
        _looping(model._engine)
        x = tone(3.0)
        file_text = model.transcribe(x)
        assert file_text == "Hello there. and then we went"
        session = model.stream(step=1.0)
        update = session.push(x)
        assert update is not None and update.partial == file_text
        assert session.finish() == file_text
        live_calls = [c for c in model._engine.calls if c.stream_key is not None]
        assert live_calls and len(live_calls[-1].token_ids) == 2      # the draft is the model's own (untrimmed) ids
    finally:
        model.close()


def test_live_audio_keeps_repeated_speech_when_the_window_ended_by_itself(fake_engine):
    said = "He said: Thank you. Thank you. Thank you. Thank you."
    fake_engine.reply = lambda call: "language English<asr_text>" + said
    model = _api.KittenASR(MODEL)
    try:
        session = model.stream(step=1.0)
        session.push(tone(3.0))
        assert session.finish() == said == model.transcribe(tone(3.0))
    finally:
        model.close()


def test_up_to_30_s_is_one_window():
    assert _api._windows(tone(3.0)) == [(0, 3 * SR)]
    assert _api._windows(tone(30.0)) == [(0, 30 * SR)]
    assert _api._windows(tone(0.1)) == [(0, 1600)]


def test_shorter_than_100_ms_has_no_window():
    assert _api._windows(tone(0.0999)) == [] and _api._windows(np.zeros(0, np.float32)) == []


def test_cut_at_the_quiet_point_in_the_last_10_s():
    x = with_gaps(tone(70.0), 352000, 770400)          # 22.0 s, and 26.0 s after the first cut
    assert _api._windows(x) == [(0, 354400), (354400, 772800), (772800, 70 * SR)]


def test_no_quiet_point_still_cuts_inside_the_last_10_s():
    x = np.full(31 * SR, 0.5, np.float32)              # equally loud everywhere: the first place is taken
    assert _api._windows(x) == [(0, 322400), (322400, 31 * SR)]


def test_windows_cover_the_audio_once_and_in_order():
    x = (np.random.RandomState(0).standard_normal(95 * SR) * 0.05).astype(np.float32)
    spans = _api._windows(x)
    assert len(spans) >= 4 and spans[0][0] == 0 and spans[-1][1] == len(x)
    for (start, end), (next_start, _) in zip(spans, spans[1:]):
        assert end == next_start                        # no gap and no overlap
        assert 20 * SR < end - start <= 30 * SR         # every cut is in the last 10 s of its window
    assert 1600 <= spans[-1][1] - spans[-1][0] <= 30 * SR


def test_a_cut_never_leaves_a_tail_shorter_than_100_ms():
    # 100 samples more than one window, and the quietest place of the last 10 s is right at the end of the window:
    # a rest too short to decode would be lost, so the cut has to stay far enough from the end of the audio
    x = with_gaps(tone(30.0 + 100 / SR), 475200)
    assert len(x) == 480100
    spans = _api._windows(x)
    assert len(spans) == 2 and spans[0][0] == 0 and spans[0][1] == spans[1][0] and spans[1][1] == len(x)
    assert spans[1][1] - spans[1][0] >= 1600 and 20 * SR < spans[0][1] <= 30 * SR


@pytest.mark.parametrize("extra", [1, 800, 1599, 1600, 1601])
def test_every_sample_is_in_a_window_just_past_30_s(extra):
    x = with_gaps(tone(30.0 + extra / SR), 475200)
    spans = _api._windows(x)
    assert [start for start, _ in spans] == [0] + [end for _, end in spans[:-1]] and spans[-1][1] == len(x)
    assert all(1600 <= end - start <= 30 * SR for start, end in spans)


@pytest.mark.parametrize("samples, budget", [(1600, 80), (SR, 80), (SR + 160, 96), (30 * SR, 544)])
def test_token_budget_grows_with_the_window(samples, budget):
    assert _api._token_budget(samples) == budget


# ---------------------------------------------------------------- the constructor
def test_model_and_options_go_to_the_engine(fake_engine, tmp_path):
    asr = _api.KittenASR(f"  {MODEL} ", cache_dir=tmp_path, device="cuda:1", max_batch=8)
    engine = fake_engine.engines[0]
    assert engine.kind == "thread" and engine.model == MODEL
    assert engine.options == {"cache_dir": str(tmp_path), "device": "cuda:1", "max_batch": 8, "precision": "bf16"}
    assert fake_engine.required == [("asr", "cuda:1")]
    assert asr.capabilities["device"] == "cuda:1" and asr.sample_rate == _api.KittenASR.sample_rate == SR
    asr.close()


def test_defaults_and_a_folder_as_model(fake_engine, tmp_path):
    asr = _api.KittenASR(tmp_path)
    engine = fake_engine.engines[0]
    assert engine.model == str(tmp_path) and engine.options == {"cache_dir": None, "precision": "bf16"}
    assert fake_engine.required == [("asr", None)]
    asr.close()


def test_precision_goes_to_the_engine(fake_engine):
    _api.KittenASR(MODEL, precision="int8").close()
    assert fake_engine.engines[0].options == {"cache_dir": None, "precision": "int8"}
    _api.KittenASR(MODEL, dtype="fp16").close()             # the older engine option is passed as it is
    assert fake_engine.engines[1].options == {"cache_dir": None, "dtype": "fp16"}


def test_process_version_does_not_check_the_gpu_here(fake_engine):
    asr = _api.KittenASR(MODEL, process=True, start_timeout_s=60)
    engine = fake_engine.engines[0]
    assert engine.kind == "process" and engine.options == {"cache_dir": None, "start_timeout_s": 60, "precision": "bf16"}
    assert fake_engine.required == [("asr-process", None)]         # PyTorch and the GPU are the engine process's
    assert asr.capabilities["process"] is True and asr.transcribe(tone(1.0)) == "n16000"
    asr.close()
    assert engine.closed == 1


def test_explicit_model():
    with pytest.raises(TypeError):
        _api.KittenASR()


@pytest.mark.parametrize("model", ["", "   ", None, 7, b"KittenML/kitten-asr-tiny"])
def test_model_must_be_a_name_or_a_folder(fake_engine, model):
    with pytest.raises(ValueError, match="model must be a Hugging Face model name or a local model folder"):
        _api.KittenASR(model)
    assert fake_engine.engines == [] and fake_engine.required == []


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4", None])
def test_workers_must_be_a_whole_number(fake_engine, workers):
    with pytest.raises(ValueError, match="workers must be a whole number of 1 or more"):
        _api.KittenASR(MODEL, workers=workers)
    assert fake_engine.engines == []


def test_a_missing_requirement_is_raised_by_the_constructor(fake_engine, monkeypatch):
    from kitten_inference.asr import _deps

    def refuse(feature="asr", device=None):
        raise ImportError("Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch")
    monkeypatch.setattr(_deps, "require", refuse)
    with pytest.raises(ImportError, match="pip install torch"):
        _api.KittenASR(MODEL)
    assert fake_engine.engines == []


def test_capabilities(asr, engine):
    expected = {"streaming": True, "batching": True, "backend": "cuda", "device": "cuda", "attention": "torch",
                "precision": "bf16", "process": False, "sample_rate": 16000, "max_window_seconds": 30.0,
                "system_prompt": "auto:no-file"}
    assert asr.capabilities == expected and asr.capabilities == expected
    assert engine.snapshots == 1                         # the attention kind is asked for once


# ---------------------------------------------------------------- transcribe
def test_one_window_is_one_background_decode(asr, engine):
    x = tone(3.0)
    assert asr.transcribe(x) == "n48000"
    (call,) = engine.calls
    assert call.audio.dtype == np.float32 and call.audio.ndim == 1 and np.array_equal(call.audio, x)
    assert call.prompt == "<prompt:None>" and engine.prompts == [None]
    assert (call.priority, call.timeout_s, call.max_new_tokens) == (10, 600.0, 64 + 16 * 3)
    assert call.draft_ids is None and call.stream_key is None and call.on_tokens is None
    assert engine.released == []


def test_language_is_a_name_and_its_prompt_is_fetched_once(asr, engine):
    assert asr.transcribe(tone(1.0), language=" english ") == "n16000"
    assert engine.prompts == ["English"] and engine.calls[0].prompt == "<prompt:English>"
    asr.transcribe(tone(1.0), language="ENGLISH")
    asr.transcribe(tone(1.0), language="English", details=True)
    assert engine.prompts == ["English"]
    asr.transcribe(tone(1.0))
    asr.transcribe(tone(1.0), language=None)
    assert engine.prompts == ["English", None] and engine.calls[-1].prompt == "<prompt:None>"


@pytest.mark.parametrize("language", [
    "", "  ", 5, b"English", ["English"],
    "en", "en-US", "zh",                                # a code is not a name
    "English<|im_end|>", "English\nx", "English,German", "English 2",    # it goes into the prompt: letters only
])
def test_language_must_be_a_name(asr, engine, language):
    with pytest.raises(ValueError, match="language must be a language name such as 'English'"):
        asr.transcribe(tone(1.0), language=language)
    with pytest.raises(ValueError, match="language must be a language name such as 'English'"):
        asr.stream(language=language)
    assert engine.calls == [] and engine.prompts == []


@pytest.mark.parametrize("language", ["English", "german", "Mandarin Chinese", "Serbo-Croatian", " French "])
def test_language_names_are_accepted(asr, engine, language):
    assert asr.transcribe(tone(1.0), language=language) == "n16000"
    asr.stream(language=language).close()
    (name,) = engine.prompts                            # asked for once, without the spaces around it
    assert name.lower() == language.strip().lower() and name[0].isupper()
    assert engine.calls[0].prompt == "<prompt:%s>" % name


def test_details(asr, engine):
    engine.reply = lambda call: "language French<asr_text> bonjour <non_speech> "
    result = asr.transcribe(tone(2.5), details=True)
    assert result == _api.Transcript(text="bonjour <non_speech>", language="French",
                                     segments=[_api.Segment(0.0, 2.5, "bonjour <non_speech>")],
                                     raw=["language French<asr_text> bonjour <non_speech> "])
    # a language that was asked for is the language of the result, and the model then writes the text only
    engine.reply = lambda call: "guten Tag"
    forced = asr.transcribe(tone(1.0), language="german", details=True)
    assert (forced.text, forced.language, forced.raw) == ("guten Tag", "German", ["guten Tag"])


def test_long_audio_is_cut_decoded_and_joined_in_order(asr, engine):
    x = with_gaps(tone(70.0), 352000, 770400)
    spans = [(0, 354400), (354400, 772800), (772800, 70 * SR)]
    result = asr.transcribe(x, details=True)
    assert result.text == "n354400 n418400 n347200" and result.language == "English"
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (start / SR, end / SR, "n%d" % (end - start)) for start, end in spans]
    assert [s.end for s in result.segments] == pytest.approx([22.15, 48.3, 70.0])
    assert result.raw == ["language English<asr_text>n%d" % (end - start) for start, end in spans]
    by_length = {len(call.audio): call for call in engine.calls}       # the engine got them in any order
    assert sorted(by_length) == sorted(end - start for start, end in spans) and len(engine.calls) == 3
    for start, end in spans:
        call = by_length[end - start]
        assert np.array_equal(call.audio, x[start:end])                # every sample once, none twice
        assert (call.priority, call.timeout_s, call.stream_key, call.draft_ids) == (10, 600.0, None, None)
    assert [by_length[end - start].max_new_tokens for start, end in spans] == [432, 496, 416]


def test_a_silent_window_is_not_sent_but_keeps_its_segment(asr, engine):
    x = np.concatenate([np.zeros(26 * SR, np.float32), tone(14.0)])
    result = asr.transcribe(x, details=True)
    cut = 322400                                        # the first quiet place from 20 s on
    assert result == _api.Transcript(
        "n317600", "English", [_api.Segment(0.0, cut / SR, ""), _api.Segment(cut / SR, 40.0, "n317600")],
        ["", "language English<asr_text>n317600"])
    assert [len(call.audio) for call in engine.calls] == [len(x) - cut]


def test_too_short_and_silent_audio_give_empty_text(asr, engine):
    assert asr.transcribe(tone(0.09)) == "" and asr.transcribe(np.zeros(0, np.float32)) == ""
    assert asr.transcribe(tone(0.05), details=True) == _api.Transcript("", None, [], [])
    silent = asr.transcribe(np.zeros(2 * SR, np.float32), language="German", details=True)
    assert silent == _api.Transcript("", "German", [_api.Segment(0.0, 2.0, "")], [""])
    assert asr.transcribe(np.full(SR, 5e-5, np.float32)) == ""          # below the silence gate
    assert engine.calls == []
    assert asr.transcribe(np.full(SR, 2e-4, np.float32)) == "n16000"    # just above it


def test_arrays_are_mixed_scaled_and_resampled(asr, engine):
    mono = tone(1.0)
    asr.transcribe(np.stack([mono, -0.5 * mono], axis=1))
    assert engine.calls[-1].audio.shape == (SR,)
    np.testing.assert_allclose(engine.calls[-1].audio, 0.25 * mono, rtol=0, atol=1e-6)
    pcm = np.round(mono * 32767).astype(np.int16)
    asr.transcribe(pcm)
    assert np.array_equal(engine.calls[-1].audio, pcm.astype(np.float32) / 32768)
    asr.transcribe(mono.astype(np.float64))
    assert engine.calls[-1].audio.dtype == np.float32 and np.array_equal(engine.calls[-1].audio, mono)
    low = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(8000) / 8000)).astype(np.float32)
    assert asr.transcribe(low, sample_rate=8000) == "n16000"
    np.testing.assert_allclose(engine.calls[-1].audio, _audio.resample(low, 8000), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="sample_rate must be a positive number of Hz"):
        asr.transcribe(mono, sample_rate=0)


def test_files_are_read_at_16_khz_whatever_sample_rate_says(asr, engine, tmp_path):
    import soundfile as sf
    low = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(8000) / 8000)).astype(np.float32)
    path = tmp_path / "speech.wav"
    sf.write(str(path), np.stack([low, low], axis=1), 8000, subtype="FLOAT")
    assert asr.transcribe(path, sample_rate=48000) == "n16000"          # the file's own rate counts
    np.testing.assert_allclose(engine.calls[0].audio, _audio.resample(low, 8000), rtol=0, atol=1e-5)
    assert asr.transcribe([str(path), tone(2.0)]) == ["n16000", "n32000"]


def test_audio_that_is_refused(asr, engine, tmp_path):
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        asr.transcribe(tmp_path / "nothing.wav")
    with pytest.raises(ValueError, match="audio path is empty"):
        asr.transcribe("  ")
    for bad in (3.5, None, {"a": 1}, b"speech.wav"):
        with pytest.raises(ValueError, match="audio must be a file path, a numpy array of samples, or a list"):
            asr.transcribe(bad)
    with pytest.raises(ValueError, match="audio must be a file path"):
        asr.transcribe([[0.0, 0.1, 0.2]])                # a list is a batch, never samples
    with pytest.raises(ValueError, match="must be float or int16"):
        asr.transcribe(np.zeros(SR, np.int32))
    assert engine.calls == []


def test_a_list_is_on_the_engine_together_and_keeps_its_order(asr, engine):
    gate = threading.Barrier(3, timeout=10)

    def reply(call):
        gate.wait()                                     # passes only when three decodes are waiting at once
        seconds = len(call.audio) // SR
        time.sleep(0.05 * (3 - seconds))                # the last item answers first
        return "item %d" % seconds
    engine.reply = reply
    assert asr.transcribe([tone(1.0), tone(2.0), tone(3.0)]) == ["item 1", "item 2", "item 3"]
    assert engine.max_active == 3 and len(engine.calls) == 3 and {c.priority for c in engine.calls} == {10}


def test_list_shapes(asr, engine):
    assert asr.transcribe([]) == [] and asr.transcribe(()) == []
    assert asr.transcribe((tone(1.0),)) == ["n16000"]
    results = asr.transcribe([tone(1.0), np.zeros(SR, np.float32), tone(0.05)], details=True)
    assert [type(r) for r in results] == [_api.Transcript] * 3
    assert [r.text for r in results] == ["n16000", "", ""] and [len(r.segments) for r in results] == [1, 1, 0]


def test_workers_is_how_many_windows_wait_on_the_engine(fake_engine):
    asr = _api.KittenASR(MODEL, workers=1)
    engine = fake_engine.engines[0]

    def reply(call):
        time.sleep(0.02)
        return "n%d" % len(call.audio)
    engine.reply = reply
    try:
        assert asr.transcribe([tone(1.0), tone(2.0), tone(3.0)]) == ["n16000", "n32000", "n48000"]
        assert engine.max_active == 1 and [len(c.audio) for c in engine.calls] == [SR, 2 * SR, 3 * SR]
    finally:
        asr.close()


def test_an_engine_error_reaches_the_caller(asr, engine):
    def reply(call):
        raise PoolExhausted("kitten-asr: KV pool exhausted")
    engine.reply = reply
    with pytest.raises(PoolExhausted, match="KV pool exhausted"):
        asr.transcribe(tone(1.0))
    with pytest.raises(PoolExhausted):
        asr.transcribe([tone(1.0), tone(2.0)])
    engine.reply = lambda call: "fine"
    assert asr.transcribe(tone(1.0)) == "fine"           # the model is still usable


# ---------------------------------------------------------------- lifetime
def test_a_closed_model_refuses_every_call(asr, engine):
    session = asr.stream()
    assert asr.transcribe(tone(1.0)) == "n16000"         # the worker pool exists now
    asr.close()
    asr.close()
    assert engine.closed == 1
    calls = (lambda: asr.transcribe(tone(1.0)), asr.stream, lambda: asr.capabilities, asr.__enter__,
             lambda: session.push(tone(0.5)), session.finish)
    for call in calls:
        with pytest.raises(RuntimeError, match="this KittenASR is closed"):
            call()
    session.close()                                      # nothing is left to free: no error
    assert len(engine.calls) == 1 and engine.released == []


def test_context_manager_closes(fake_engine):
    with _api.KittenASR(MODEL) as asr:
        assert asr.transcribe(tone(1.0)) == "n16000"
    assert fake_engine.engines[0].closed == 1
    with pytest.raises(RuntimeError, match="this KittenASR is closed"):
        asr.transcribe(tone(1.0))


# ---------------------------------------------------------------- streaming
def test_stream_updates_reuse_the_key_and_carry_the_last_ids(asr, engine):
    session = asr.stream()                               # step 0.5 s, window 20 s
    assert session.text == "" and session.language is None
    assert session.push(np.zeros(0, np.float32)) is None
    assert session.push(tone(0.25)) is None and engine.calls == []      # less than one step: only kept
    update = session.push(tone(0.25))
    assert update == _api.StreamUpdate(text="n8000", committed="", partial="n8000")
    first = engine.calls[0]
    assert len(first.audio) == 8000 and first.draft_ids is None and STREAM_KEY.match(first.stream_key)
    assert first.stream_key.endswith(":0") and first.prompt == "<prompt:None>"
    assert (first.priority, first.timeout_s, first.max_new_tokens) == (0, 30.0, 80)     # the engine's defaults

    update = session.push(tone(0.5))
    assert update == _api.StreamUpdate("n16000", "", "n16000") and session.text == "n16000"
    second = engine.calls[1]
    assert len(second.audio) == SR and second.stream_key == first.stream_key
    assert second.draft_ids == first.token_ids            # the previous result is the draft of the next decode
    assert engine.released == [] and session.language == "English"

    # nothing new since the last update: finish() commits that text without another decode, then frees the key
    assert session.finish() == "n16000" and len(engine.calls) == 2
    assert engine.released == [first.stream_key]
    with pytest.raises(RuntimeError, match="this stream is finished"):
        session.push(tone(0.5))
    assert session.finish() == "n16000" and session.text == "n16000"
    session.close()
    assert len(engine.calls) == 2 and engine.released == [first.stream_key]


def noise(seconds, dbfs=-60.0, seed=0):
    n = int(round(seconds * SR))
    return (np.random.default_rng(seed).standard_normal(n) * 10 ** (dbfs / 20)).astype(np.float32)


def test_the_speech_gate_skips_quiet_noise_before_speech(asr, engine):
    session = asr.stream()                               # speech_gate=True by default
    for _ in range(4):                                   # 2 s of quiet noise: no decode, no text
        assert session.push(noise(0.5)) is None
    assert engine.calls == []
    update = session.push(tone(0.25))                    # the first sound: everything heard so far is decoded at once
    assert update is not None and len(engine.calls) == 1 and len(engine.calls[0].audio) == 2 * SR + 4000
    assert engine.calls[0].draft_ids is None
    session.push(tone(0.5))
    assert len(engine.calls) == 2                        # then every step, as without the gate


def test_finish_decodes_even_if_the_gate_never_opened(asr, engine):
    session = asr.stream()
    assert session.push(noise(1.5)) is None and engine.calls == []
    assert session.finish() == "n24000" and len(engine.calls) == 1      # the final text is never skipped


def test_a_later_window_of_quiet_noise_is_not_sent(asr, engine):
    def run(**options):
        session = asr.stream(step=0.5, max_window=2.0, **options)
        session.push(noise(0.5))                         # background, gate closed
        session.push(tone(0.5))                          # opens the gate
        for i in range(6):                               # the same quiet noise: the 2 s window is cut inside it
            session.push(noise(0.5, seed=i + 1))
        return session.finish()
    text = run()
    assert engine.calls and all(call.stream_key.endswith(":0") for call in engine.calls)  # only the first window
    assert " " not in text                               # one window of text, nothing invented on the noise
    engine.calls.clear()
    run(speech_gate=False)
    assert any(not call.stream_key.endswith(":0") for call in engine.calls)  # without the gate they are sent


def test_the_speech_gate_can_be_turned_off(asr, engine):
    session = asr.stream(speech_gate=False)
    assert session.push(noise(0.5)) is not None and len(engine.calls) == 1


def test_the_speech_gate_opens_on_quiet_speech_over_a_quieter_background(asr, engine):
    session = asr.stream()
    assert session.push(noise(1.0, -75.0)) is None
    assert session.push(tone(0.5, level=0.003)) is not None      # -53 dBFS: 22 dB over the background


def test_a_full_window_is_cut_committed_and_released(asr, engine):
    session = asr.stream(step=0.5, max_window=1.0)
    x = with_gaps(tone(1.0), 9600)
    session.push(x[:8000])
    update = session.push(x[8000:])                      # the window is full: cut in its last half second
    # the text before the cut is final; the rest is decoded at once, so its words do not leave the text for a while
    assert update == _api.StreamUpdate(text="n12000 n4000", committed="n12000", partial="n4000")
    assert session.text == "n12000 n4000"
    partial, closing, rest = engine.calls
    key = partial.stream_key
    assert np.array_equal(closing.audio, x[:12000])      # exactly the audio before the cut, decoded once more
    assert closing.stream_key == key and closing.draft_ids == partial.token_ids
    # the rest starts the next window: a new key, no draft
    assert np.array_equal(rest.audio, x[12000:]) and rest.draft_ids is None
    assert rest.stream_key == key[:-1] + "1" and rest.max_new_tokens == 80
    assert engine.log == [("decode", key), ("decode", key), ("release", key), ("decode", rest.stream_key)]

    # nothing new since that decode: finish() commits its text without another one
    assert session.finish() == "n12000 n4000" and len(engine.calls) == 3
    assert engine.log[4:] == [("release", rest.stream_key)]


def test_a_rest_that_holds_no_word_waits_for_the_next_step(asr, engine):
    session = asr.stream(step=0.5, max_window=1.0)
    x = with_gaps(tone(1.0), 11200)                      # the quiet place is at the end: 2400 silent samples are left
    session.push(x[:8000])
    update = session.push(x[8000:])
    assert update == _api.StreamUpdate("n13600", "n13600", "")           # silence holds no word: not sent
    assert [len(call.audio) for call in engine.calls] == [8000, 13600]
    update = session.push(tone(0.5))                     # a step later the new window is decoded from its start
    assert update == _api.StreamUpdate("n13600 n10400", "n13600", "n10400")
    assert np.array_equal(engine.calls[2].audio[:2400], x[13600:]) and engine.calls[2].draft_ids is None
    assert session.finish() == "n13600 n10400" and len(engine.calls) == 3


def test_one_long_push_closes_several_windows_and_loses_no_sample(asr, engine):
    session = asr.stream(step=0.5, max_window=1.0)
    x = with_gaps(tone(2.5), 9600, 20000, 31360)
    update = session.push(x)
    assert update == _api.StreamUpdate("n12000 n10400 n11360 n6240", "n12000 n10400 n11360", "n6240")
    assert len(engine.released) == 3                     # the last window is still open
    assert session.finish() == "n12000 n10400 n11360 n6240"
    assert [len(call.audio) for call in engine.calls] == [12000, 10400, 11360, 6240]
    assert np.array_equal(np.concatenate([call.audio for call in engine.calls]), x)
    prefix = engine.calls[0].stream_key[:-1]
    keys = [prefix + str(i) for i in range(4)]
    assert engine.log == [(what, key) for key in keys for what in ("decode", "release")]
    assert all(call.draft_ids is None and call.max_new_tokens == 80 for call in engine.calls)


def test_stream_language(asr, engine):
    engine.reply = lambda call: "hallo"
    session = asr.stream(language="german")
    assert session.language == "German" and engine.prompts == ["German"]
    assert session.push(tone(0.5)).text == "hallo" and engine.calls[0].prompt == "<prompt:German>"
    assert session.language == "German" and session.finish() == "hallo"


def test_stream_takes_int16(asr, engine):
    pcm = np.round(tone(0.5) * 32767).astype(np.int16)
    session = asr.stream()
    session.push(pcm)
    assert engine.calls[0].audio.dtype == np.float32
    assert np.array_equal(engine.calls[0].audio, pcm.astype(np.float32) / 32768)


def test_silence_in_a_stream_is_not_sent(asr, engine):
    session = asr.stream(speech_gate=False)
    assert session.push(np.zeros(8000, np.float32)) == _api.StreamUpdate("", "", "")
    assert session.finish() == "" and engine.calls == []
    assert len(engine.released) == 1 and STREAM_KEY.match(engine.released[0])
    session = asr.stream()                               # with the speech gate: no update at all
    assert session.push(np.zeros(8000, np.float32)) is None
    assert session.finish() == "" and engine.calls == []


def test_a_failed_decode_changes_nothing_and_the_next_push_tries_again(asr, engine):
    def fail(call):
        raise TimeoutError("kitten-asr: decode timed out after 30.0s")
    engine.reply = fail
    session = asr.stream()
    with pytest.raises(TimeoutError, match="decode timed out"):
        session.push(tone(0.5))
    assert session.text == "" and engine.released == []
    engine.reply = lambda call: "n%d" % len(call.audio)
    update = session.push(tone(0.1))                     # the audio of the failed push is still there
    assert update == _api.StreamUpdate("n9600", "", "n9600")
    assert engine.calls[1].draft_ids is None and engine.calls[1].stream_key == engine.calls[0].stream_key
    assert session.push(tone(0.2)) is None               # less than a step: no decode
    engine.reply = fail
    with pytest.raises(TimeoutError):                    # the same at the end: finish() can be called again
        session.finish()
    assert engine.released == []
    engine.reply = lambda call: "n%d" % len(call.audio)
    assert session.finish() == "n12800" and engine.released == [engine.calls[0].stream_key]


def test_close_frees_the_key_without_a_last_decode(asr, engine):
    session = asr.stream()
    session.push(tone(0.5))
    session.push(tone(0.2))                              # audio no decode has seen
    session.close()
    session.close()
    assert len(engine.calls) == 1 and engine.released == [engine.calls[0].stream_key]
    assert session.text == "" and session.finish() == ""
    with pytest.raises(RuntimeError, match="this stream is finished"):
        session.push(tone(0.5))
    with asr.stream() as other:
        other.push(tone(0.5))
    assert engine.released == [engine.calls[0].stream_key, engine.calls[1].stream_key]
    assert engine.calls[0].stream_key != engine.calls[1].stream_key     # every session has keys of its own


@pytest.mark.parametrize("options, message", [
    ({"max_window": 0.5}, "max_window must be between 1 and 30 seconds, not 0.5"),
    ({"max_window": 31}, "max_window must be between 1 and 30 seconds, not 31"),
    ({"max_window": "20"}, "max_window must be between 1 and 30 seconds"),
    ({"max_window": float("nan")}, "max_window must be between 1 and 30 seconds"),
    ({"step": 0.05}, "step must be between 0.1 and 20 seconds, not 0.05"),
    ({"step": 25}, "step must be between 0.1 and 20 seconds, not 25"),
    ({"step": 3, "max_window": 2}, "step must be between 0.1 and 2 seconds, not 3"),
    ({"step": True}, "step must be between 0.1 and 20 seconds, not True"),
    ({"step": None}, "step must be between 0.1 and 20 seconds, not None"),
])
def test_stream_limits(asr, options, message):
    with pytest.raises(ValueError, match=message):
        asr.stream(**options)


def test_stream_limits_are_inclusive(asr):
    asr.stream(step=0.1, max_window=1.0).close()
    asr.stream(step=30, max_window=30).close()


def test_a_stock_phrase_on_the_first_second_of_live_audio_is_not_shown(fake_engine):
    """The first partial of a session ("Thank you." on 1 s of audio) is a known invention of the model: it is not
    shown, but the final text keeps whatever the model wrote for the finished audio."""
    fake_engine.reply = lambda call: "language English<asr_text>Thank you."
    model = _api.KittenASR(MODEL)
    try:
        session = model.stream(step=0.5)
        update = session.push(tone(1.0))
        assert update is not None and update.partial == "" and update.text == "" and session.text == ""
        assert session.finish() == "Thank you."           # someone who only said "Thank you" keeps it
        session = model.stream(step=0.5)
        session.push(tone(1.0))
        update = session.push(tone(1.5))                   # 2.5 s: shown
        assert update.partial == "Thank you."
        session.close()
    finally:
        model.close()
