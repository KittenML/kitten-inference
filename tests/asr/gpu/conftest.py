"""GPU tests of kitten_inference.asr: they load a real model on an NVIDIA GPU. Environment:
  KITTEN_ASR_GPU_MODEL      model name on Hugging Face or a local model folder; without it these tests are skipped
  KITTEN_ASR_GPU_WAV        a recording of speech, 6 s or longer, in a language written with spaces between words
  KITTEN_ASR_GPU_TEXT       the words spoken in that recording. The model's text is compared with them; generated
                            tones hold no speech, so without both variables no test shows that words are recognised
  KITTEN_ASR_GPU_CACHE      where a downloaded model is kept (default: the Hugging Face cache)
  KITTEN_ASR_REQUIRE_GPU=1  fail instead of skipping when no model is named, the recording or its text is missing,
                            or the engine cannot run here
"""
import os
import sys
import time

import pytest

MODEL = os.environ.get("KITTEN_ASR_GPU_MODEL") or None
CACHE = os.environ.get("KITTEN_ASR_GPU_CACHE") or None
WAV = os.environ.get("KITTEN_ASR_GPU_WAV") or None
TEXT = os.environ.get("KITTEN_ASR_GPU_TEXT") or None


def _skip(reason):
    if os.environ.get("KITTEN_ASR_REQUIRE_GPU") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


class Live:
    """The one loaded model of the test run. get() loads it when there is none; take() hands it over for good
    (to a test that closes it), so that a test running afterwards gets a new one. fresh(**options) closes it
    and loads a model with other engine options, which the test closes. `model` is the name it is loaded by."""

    model = MODEL

    def __init__(self):
        self.asr = None

    def get(self):
        if self.asr is None:
            from kitten_inference.asr import KittenASR
            t0 = time.monotonic()
            self.asr = KittenASR(MODEL, backend="cuda", cache_dir=CACHE)
            print(f"\nKittenASR({MODEL!r}) ready in {time.monotonic() - t0:.1f} s", file=sys.stderr)
        return self.asr

    def take(self):
        asr, self.asr = self.get(), None
        return asr

    def fresh(self, **engine_options):
        from kitten_inference.asr import KittenASR
        self.close()                                    # one model on the GPU at a time
        return KittenASR(MODEL, backend="cuda", cache_dir=CACHE, **engine_options)

    def close(self):
        asr, self.asr = self.asr, None
        if asr is not None:
            asr.close()


@pytest.fixture(scope="session")
def live():
    if MODEL is None:
        _skip("KITTEN_ASR_GPU_MODEL is not set: name a model or a model folder to run the GPU tests")
    from kitten_inference.asr import _deps
    try:
        _deps.require("asr")
    except (ImportError, RuntimeError) as exc:
        _skip(f"Kitten ASR cannot run here: {exc}")
    holder = Live()
    yield holder
    holder.close()


@pytest.fixture
def gpu_asr(live):
    """The loaded KittenASR (shared by the tests)."""
    return live.get()


@pytest.fixture
def spoken(live):
    """(path of the recording, the words spoken in it)."""
    if WAV is None or TEXT is None:
        _skip("KITTEN_ASR_GPU_WAV and KITTEN_ASR_GPU_TEXT are not both set: no text is compared with known words")
    return WAV, TEXT
