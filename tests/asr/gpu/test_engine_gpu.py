"""The real engine on a GPU (run: python -m pytest tests/asr/gpu, with KITTEN_ASR_GPU_MODEL set; see conftest.py).

The model loads, says which attention kind it runs, turns audio into text (one call, two at once, eight at once,
a list), streams, reports its counters, runs with few decode rows, and refuses calls once closed. Generated tones
and noise hold no speech, so only a recording given in KITTEN_ASR_GPU_WAV is compared word by word: with the words
in KITTEN_ASR_GPU_TEXT, and between transcribe() and a stream."""
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np
import pytest

SR = 16000
KINDS = ("triton", "flash", "torch")
# Room for a number or a name written another way. A model that recognises nothing is near 1.
MAX_WORD_ERROR_RATE = 0.2


def clip(seconds=3.0, seed=0):
    """A tone with a little noise: audio without speech, loud enough to reach the model."""
    t = np.arange(int(seconds * SR)) / SR
    hiss = np.random.RandomState(seed).standard_normal(len(t))
    return (0.05 * np.sin(2 * np.pi * 440.0 * t) + 0.01 * hiss).astype(np.float32)


def words(text):
    """Lower-case words without punctuation."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def word_errors(got, want):
    """The fewest words to replace, add or drop to turn `want` into `got`."""
    row = list(range(len(got) + 1))                     # row[j]: errors between want[:i] and got[:j]
    for i, expected in enumerate(want, 1):
        corner, row[0] = row[0], i
        for j, heard in enumerate(got, 1):
            corner, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, corner + (expected != heard))
    return row[-1]


def recording(spoken):
    """(samples at 16 kHz, the words spoken) of the recording named in the environment."""
    from kitten_inference.asr import _audio
    path, text = spoken
    want = words(text)
    assert len(want) >= 3, "KITTEN_ASR_GPU_TEXT holds fewer than three words"
    return _audio.load(path, SR)[0], want


def check_words(text, want):
    got = words(text)
    errors = word_errors(got, want)
    assert errors <= MAX_WORD_ERROR_RATE * len(want), f"{errors} of {len(want)} words differ; the model heard: {got}"


def test_recording_is_recognised(gpu_asr, spoken):
    audio, want = recording(spoken)
    check_words(gpu_asr.transcribe(audio), want)


def test_eight_transcribes_at_once_share_decode_steps_and_are_all_recognised(gpu_asr, spoken):
    audio, want = recording(spoken)
    engine = gpu_asr._engine
    before = engine.snapshot()
    together = threading.Barrier(8, timeout=60)

    def one():
        together.wait()                                 # all eight calls reach the engine at the same time
        return gpu_asr.transcribe(audio)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one) for _ in range(8)]
        texts = [future.result(timeout=600) for future in futures]
    after = engine.snapshot()
    assert len(texts) == 8
    for text in texts:
        check_words(text, want)                         # the same limit as for one call alone
    # batching: the calls were decoded together, so decode steps carried more than one row each
    steps, rows = after["steps"] - before["steps"], after["step_rows"] - before["step_rows"]
    print(f"\n8 calls at once: {steps} decode steps, {rows} rows in them, {len(set(texts))} different text(s)",
          file=sys.stderr)
    assert 0 < steps < rows, f"{steps} decode steps carried {rows} rows: the eight calls ran one after another"


def test_engine_loads_and_reports_its_attention_kind(gpu_asr):
    from kitten_inference.asr import _deps
    caps = gpu_asr.capabilities
    assert caps["attention"] in KINDS, caps
    assert caps["streaming"] is True and caps["batching"] is True and caps["process"] is False
    assert caps["device"] == "cuda" and caps["sample_rate"] == SR == gpu_asr.sample_rate
    usable, reason = _deps.attention_report()[caps["attention"]]
    assert usable, reason                               # the kind in use is one this machine can run


def test_transcribe_returns_text(gpu_asr):
    from kitten_inference.asr import Segment, Transcript
    assert isinstance(gpu_asr.transcribe(clip()), str)
    result = gpu_asr.transcribe(clip(), details=True)
    assert isinstance(result, Transcript) and isinstance(result.text, str)
    assert result.language is None or isinstance(result.language, str)
    assert len(result.segments) == len(result.raw) == 1 and isinstance(result.segments[0], Segment)
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 3.0)
    assert result.segments[0].text == result.text and isinstance(result.raw[0], str)
    assert gpu_asr.transcribe(np.zeros(SR, np.float32)) == ""          # silence never reaches the model


def test_two_transcribes_at_once_and_a_list(gpu_asr):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gpu_asr.transcribe, clip(3.0, seed=1)),
                   pool.submit(gpu_asr.transcribe, clip(5.0, seed=2))]
        texts = [future.result(timeout=600) for future in futures]
    assert all(isinstance(text, str) for text in texts)
    batch = gpu_asr.transcribe([clip(3.0, seed=1), clip(5.0, seed=2)])
    assert isinstance(batch, list) and len(batch) == 2 and all(isinstance(text, str) for text in batch)


def test_stream_ends_with_the_text_of_transcribe(gpu_asr):
    from kitten_inference.asr import StreamUpdate, _audio
    path = os.environ.get("KITTEN_ASR_GPU_WAV")
    if path:
        audio = _audio.load(path, SR)[0][:6 * SR]
        assert len(audio) >= SR, f"{path} holds less than a second of audio"
    else:
        audio = clip(6.0)
    expected = gpu_asr.transcribe(audio)

    updates = []
    with gpu_asr.stream() as session:
        for start in range(0, len(audio), SR // 4):                    # a quarter of a second at a time
            update = session.push(audio[start:start + SR // 4])
            if update is not None:
                updates.append(update)
        final = session.finish()
        assert session.finish() == final == session.text
    assert isinstance(final, str) and updates, "no update in the whole stream"
    for update in updates:
        assert isinstance(update, StreamUpdate)
        assert update.text == " ".join(part for part in (update.committed, update.partial) if part)
    print(f"\ntranscribe(): {expected!r}\nstream:       {final!r}", file=sys.stderr)
    if path:
        # the same audio and the same greedy decoding: the same words, whatever case and punctuation do
        assert len(words(expected)) >= 3, f"transcribe() found fewer than three words in the first 6 s of {path}"
        assert words(final) == words(expected)


def test_snapshot_has_attention_compiled_and_page(gpu_asr):
    snapshot = gpu_asr._engine.snapshot()
    assert snapshot["attention"] in KINDS and snapshot["attention"] == gpu_asr.capabilities["attention"]
    assert "compiled" in snapshot and isinstance(snapshot["compiled"], bool)
    assert isinstance(snapshot["page"], int) and snapshot["page"] >= 1
    assert snapshot["kv_tokens"] >= snapshot["page"] and snapshot["requests"] >= 0


def test_decode_after_close_raises(live):
    from kitten_inference.asr import EngineStopped
    asr = live.take()                                   # closed below: a later test gets a new model
    engine = asr._engine
    prompt = engine.default_prompt()
    first = engine.decode(clip(1.0), prompt, max_new_tokens=8)
    assert isinstance(first.text, str) and isinstance(first.token_ids, list)
    assert first.info["finish_reason"] in ("stop", "length")
    asr.close()
    asr.close()                                         # safe to call again
    with pytest.raises(EngineStopped, match="kitten-asr: the engine is closed"):
        engine.decode(clip(1.0), prompt)
    with pytest.raises(EngineStopped):
        engine.snapshot()
    with pytest.raises(EngineStopped):
        engine.default_prompt()
    engine.release("no such stream")                    # never raises
    with pytest.raises(RuntimeError, match="this KittenASR is closed"):
        asr.transcribe(clip(1.0))


@pytest.mark.parametrize("option, value, least", [
    ("max_batch", 0, 1), ("max_batch", -1, 1), ("max_batch", 2.5, 1), ("max_batch", True, 1),
    ("reserve_rows", -1, 0), ("reserve_rows", "8", 0), ("max_background_rows", 0, 1), ("max_background_rows", None, 1),
])
def test_row_options_must_be_whole_numbers(live, option, value, least):
    from kitten_inference.asr import Engine
    engine = None
    try:
        with pytest.raises(ValueError, match=f"{option} must be a whole number of {least} or more"):
            engine = Engine(live.model, **{option: value})
    finally:
        if engine is not None:                          # it was accepted: do not leave a second model on the GPU
            engine.close()


def test_an_engine_with_few_rows_still_serves_transcribe(live):
    # transcribe() is background work for the engine, and background work leaves decode rows free for live
    # streams. With no more rows than that reserve it would never start, and the call would only time out.
    asr = live.fresh(max_batch=8)                       # the last test: the shared model is closed by now
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        snapshot = asr._engine.snapshot()
        assert 0 <= snapshot["reserve_rows"] < 8 and 1 <= snapshot["max_background_rows"] <= 8
        future = pool.submit(asr.transcribe, clip(2.0))
        try:
            text = future.result(timeout=120)
        except FutureTimeout:
            asr._engine.close()                         # ends the waiting call now, not after its own timeout
            pytest.fail("transcribe() of a 2 s clip did not return within 120 s on an engine with max_batch=8")
        assert isinstance(text, str)
    finally:
        asr.close()
        pool.shutdown()


def test_system_prompt_from_the_model_folder(live, tmp_path):
    """"auto" puts the text a folder's asr_config.json names in the system message; "none" leaves it
    empty; a folder without the file runs as before."""
    import json
    from kitten_inference.asr import KittenASR, _loader, _system
    src = _loader._resolve(live.model, os.environ.get("KITTEN_ASR_GPU_CACHE") or None)
    folder = tmp_path / "m"
    folder.mkdir()
    for name in os.listdir(src):
        if name != _system.SYSTEM_PROMPT_FILE:
            os.symlink(os.path.join(src, name), folder / name)
    text = "Transcribe the audio. Keep the speaker's words."
    (folder / _system.SYSTEM_PROMPT_FILE).write_text(json.dumps({_system.SYSTEM_PROMPT_KEY: text}), encoding="utf-8")
    live.close()                                        # one model on the GPU at a time
    for mode, want, source in (("auto", text, "auto:asr_config.json"), ("none", "", "none")):
        asr = KittenASR(str(folder), backend="cuda", system_prompt=mode)
        try:
            assert asr.capabilities["system_prompt"] == source
            assert asr._engine.snapshot()["system_prompt_chars"] == len(want)
            assert asr._engine.default_prompt("English").startswith(f"<|im_start|>system\n{want}<|im_end|>\n")
            assert isinstance(asr.transcribe(clip(2.0)), str)
        finally:
            asr.close()
