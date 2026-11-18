"""Within a model family, the same behaviour on every backend.

Kitten TTS 0.8 runs on the CPU and Metal; Kitten TTS 2 mini, nano and pico on the CPU, Metal and CUDA. A user must get
the same options, the same errors (type and text), the same voices and languages and the same chunks of text on each
of them; only the numeric kernels differ. Two levels:

- Always: the Python layer, with downloads and the engine replaced. Bad settings are refused with one message per
  family, the same for every backend, before any file is downloaded.
- With KITTEN_FAMILY_MODELS=1 (downloads nano 0.8 and Kitten TTS 2 pico): the real models on the CPU, then every
  other backend of the family simulated by wrapping the CPU session in that backend's session shape (Metal: a CPU
  part and a GPU part; CUDA: the separate library's object). generate() must feed the engine the same tokens and
  styles, and return the same audio, on each; voices, languages and errors must not change.
"""
import contextlib
import importlib
import os
import re

import numpy as np
import pytest

from kitten_inference import KittenTTS, _family

MODELS = {'tts08': 'KittenML/kitten-tts-nano-0.8', 'tts2_tiny': 'KittenML/kitten-tts-2-pico'}
FAMILIES = sorted(MODELS)


def _metal_available():
    try:
        from kitten_inference.tts08 import cpp_model
        cpp_model.check_metal()
        return True
    except Exception:
        return False


@pytest.fixture
def no_download(monkeypatch):
    """Any download fails the test: errors must come first."""
    def refuse(*args, **kwargs):
        raise AssertionError('downloaded before the settings were checked')
    get_model = importlib.import_module('kitten_inference.tts08.get_model')
    monkeypatch.setattr(get_model, '_download_file', refuse)
    monkeypatch.setattr(get_model, 'hf_hub_download', refuse)
    tiny = importlib.import_module('kitten_inference.tts2_tiny.model')
    monkeypatch.setattr(tiny.KittenTTS2Tiny, '_fetch', refuse)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', refuse)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', refuse)


def _error(fn):
    try:
        fn()
    except (ValueError, TypeError) as exc:
        return type(exc), str(exc)
    return None


def _same_template(family_msgs):
    """The messages of the two families are the same once the family name and backend list are taken out."""
    out = set()
    for family, msg in family_msgs.items():
        msg = msg.replace(_family.NAMES[family], '<model>').replace(_family.WHERE[family], '<where>')
        choices = _family.BACKENDS[family]
        msg = msg.replace(_family._or_list(choices), '<backends>')
        msg = re.sub(r'\b(runs|run)\b', 'run', msg)
        msg = re.sub(r'\b(it|them)\b', 'it', msg)
        out.add(msg)
    return out


# ---- level 1: the Python layer (no download, no GPU) ------------------------------------------------------------

@pytest.mark.parametrize('backend', ['rocm', 'amd_gpu', 'vulkan', 'ROCM', 3])
def test_unsupported_backend_same_error_both_families_before_download(no_download, backend):
    errors = {f: _error(lambda f=f: KittenTTS(MODELS[f], backend=backend)) for f in FAMILIES}
    assert {e[0] for e in errors.values()} == {ValueError}, errors
    assert len(_same_template({f: e[1] for f, e in errors.items()})) == 1, errors


def test_cuda_on_tts08_is_a_clear_error_before_download(no_download):
    exc_type, msg = _error(lambda: KittenTTS(MODELS['tts08'], backend='cuda'))
    assert exc_type is ValueError
    assert "backend='cuda' is not available" in msg and "Kitten TTS 2 mini, nano and pico" in msg


@pytest.mark.parametrize('option, value, exc', [('precision', 'fp16', ValueError), ('foo', 1, TypeError)])
def test_bad_model_option_same_error_on_every_backend(no_download, option, value, exc):
    for family in FAMILIES:
        seen = {_error(lambda b=b: KittenTTS(MODELS[family], backend=b, **{option: value}))
                for b in (None,) + _family.BACKENDS[family]}
        assert len(seen) == 1 and next(iter(seen))[0] is exc, (family, seen)


@pytest.mark.parametrize('rule', ['int8', 'low_memory'])
def test_gpu_rules_same_error_on_every_gpu_backend_and_family(no_download, rule):
    options = {'precision': 'int8'} if rule == 'int8' else {'low_memory': True}
    msgs = set()
    for family in FAMILIES:
        for gpu in (b for b in _family.BACKENDS[family] if b in _family.GPU_BACKENDS):
            exc_type, msg = _error(lambda: KittenTTS(MODELS[family], backend=gpu, **options))
            assert exc_type is ValueError, (family, gpu, msg)
            msgs.add(msg.replace(repr(gpu), '<gpu>'))
    assert len(msgs) == 1, msgs


def test_tts08_int8_model_name_on_metal_follows_the_same_rule(no_download):
    exc_type, msg = _error(lambda: KittenTTS('KittenML/kitten-tts-nano-0.8-int8', backend='metal'))
    assert exc_type is ValueError and 'runs the float32 model' in msg


@pytest.mark.skipif(_metal_available(), reason='this computer has Metal')
def test_metal_unavailable_same_error_before_download(no_download):
    errors = {f: _error(lambda f=f: KittenTTS(MODELS[f], backend='metal')) for f in FAMILIES}
    assert len(set(errors.values())) == 1 and errors['tts08'][0] is ValueError, errors


def test_tts08_precision_maps_to_the_int8_model():
    get_model = importlib.import_module('kitten_inference.tts08.get_model')
    assert get_model._with_precision('KittenML/kitten-tts-mini-0.8', 'int8') == 'KittenML/kitten-tts-mini-0.8-int8'
    assert get_model._with_precision('KittenML/kitten-tts-mini-0.8', 'fp32') == 'KittenML/kitten-tts-mini-0.8'
    assert get_model._with_precision('KittenML/kitten-tts-mini-0.8-int8', None) == 'KittenML/kitten-tts-mini-0.8-int8'
    with pytest.raises(ValueError, match='is the int8 model'):
        get_model._with_precision('KittenML/kitten-tts-mini-0.8-int8', 'fp32')
    with pytest.raises(ValueError, match='no int8 version'):
        get_model._with_precision('KittenML/kitten-tts-nano-0.1', 'int8')


class _FakeModel:
    """A loaded model of either family, without the engine (what KittenTTS's checks see)."""
    sample_rate = 24000
    model = type('M', (), {'sample_rate': 24000})()
    available_voices = ['Bella', 'Leo']
    languages = ['en']
    default_voice = 'Leo'

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(4, np.float32)

    def generate_stream(self, text, **kwargs):
        self.calls.append(kwargs)
        yield np.zeros(4, np.float32)

    def voices(self, language=None):
        return [{'name': n} for n in self.available_voices]


def _wrapped(family):
    tts = KittenTTS.__new__(KittenTTS)
    tts.family, tts._closed, tts._model = family, False, _FakeModel()
    return tts


@pytest.mark.parametrize('call', ['generate', 'stream'])
def test_synthesis_options_same_names_and_errors_in_both_families(call):
    def run(tts, **options):
        out = getattr(tts, call)('Hello.', **options)
        return out if call == 'generate' else list(out)
    results = {}
    for family in FAMILIES:
        tts = _wrapped(family)
        run(tts, speed=1.5, normalize=True, language='en')      # accepted by both
        results[family] = {
            'unknown': _error(lambda: run(tts, seed=3)),
            'speed_text': _error(lambda: run(tts, speed='fast')),
            'speed_negative': _error(lambda: run(tts, speed=-1)),
        }
        assert results[family]['unknown'][0] is TypeError
        assert 'does not support: seed' in results[family]['unknown'][1]
        assert results[family]['speed_text'][0] is ValueError and results[family]['speed_negative'][0] is ValueError
    # the speed messages differ only in the family's range
    for key in ('speed_text', 'speed_negative'):
        a, b = (results[f][key][1] for f in FAMILIES)
        assert re.sub(r'(from [\d.]+ to [\d.]+|above 0)', '<range>', a) == \
            re.sub(r'(from [\d.]+ to [\d.]+|above 0)', '<range>', b)


def test_stream_chunk_chars_same_rule_and_errors_in_both_families():
    # stream(chunk_chars=...): None (the model's default) or a positive whole number, the same message in both families
    # (and in the C library, with 0 for None); generate() does not take it (it keeps generate()'s own chunks).
    msgs = {}
    for family in FAMILIES:
        tts = _wrapped(family)
        list(tts.stream('Hello.', chunk_chars=None))
        assert tts._model.calls[-1]['chunk_chars'] is None   # the model picks its default
        list(tts.stream('Hello.', chunk_chars=80))
        assert tts._model.calls[-1]['chunk_chars'] == 80
        msgs[family] = [_error(lambda v=v: list(tts.stream('Hello.', chunk_chars=v)))
                        for v in (0, -5, 2.5, '80', True)]
        assert all(t is ValueError for t, _ in msgs[family])
        assert "chunk_chars must be None (the model's default) or a positive whole number of characters" in \
            msgs[family][0][1]
        exc_type, msg = _error(lambda: tts.generate('Hello.', chunk_chars=80))
        assert exc_type is TypeError and 'does not support: chunk_chars' in msg
    assert msgs['tts08'] == msgs['tts2_tiny']


def test_tts08_speaks_en_and_refuses_other_languages_like_tiny():
    tts = _wrapped('tts08')
    assert tts.languages() == ['en']
    assert tts.list_voices(language='en') == tts.list_voices()
    exc_type, msg = _error(lambda: tts.generate('Hello.', language='de'))
    assert exc_type is ValueError and "does not speak 'de'; languages: ['en']" in msg
    assert _error(lambda: tts.list_voices(language='de'))[0] is ValueError


def test_tiny_accepts_normalize_true_and_refuses_false():
    tts = _wrapped('tts2_tiny')
    tts.generate('Hello.', normalize=True)
    assert 'normalize' not in tts._model.calls[-1]
    assert _error(lambda: tts.generate('Hello.', normalize=False))[0] is ValueError


def test_speed_is_checked_before_synthesis_even_for_empty_text():
    for family in FAMILIES:
        tts = _wrapped(family)
        assert _error(lambda: tts.generate('', speed=-1))[0] is ValueError
        assert tts._model.calls == []


def test_cli_offers_the_same_options_for_both_small_families(monkeypatch, capsys):
    from kitten_inference import _cli
    seen = {}

    class Fake:
        def __init__(self, model, **kwargs):
            seen[model] = kwargs
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def generate_to_file(self, text, out, **kwargs): seen[('file', str(out))] = kwargs
    monkeypatch.setattr(_cli, 'KittenTTS', Fake)
    for family, model in MODELS.items():
        rc = _cli.main(['--model', model, '--text', 'Hi.', '--output', f'{family}.wav', '--backend', 'cpu',
                        '--precision', 'fp32', '--low-memory', '--language', 'en', '--normalize', 'true'])
        assert rc == 0, capsys.readouterr()
    assert seen[MODELS['tts08']] == seen[MODELS['tts2_tiny']]
    assert seen[('file', 'tts08.wav')] == seen[('file', 'tts2_tiny.wav')]


# ---- level 2: real models on the CPU, every backend's session shape ------------------------------------------------

needs_models = pytest.mark.skipif(os.environ.get('KITTEN_FAMILY_MODELS') != '1',
                                  reason='set KITTEN_FAMILY_MODELS=1 to download nano 0.8 and Kitten TTS 2 pico')

TEXT = ('Dr. Smith arrived at 9:30. She said, "It works!" Then the long sentence went on, and on, past several commas, '
        'until it was finally done. Short one. ') * 3   # more than 350 phonemes: several chunks in both families


class _Recorder:
    """Wraps the CPU session; records what each call is fed."""

    def __init__(self, session, log, keep=None):
        self.session, self.log, self.keep = session, log, keep

    def set_speed(self, speed):
        self.log.append(('speed', float(speed)))
        self.session.set_speed(speed)

    def forward(self, feed):
        self.log.append(tuple((k, np.asarray(v).tobytes()) for k, v in sorted(feed.items())))
        out = self.session.forward(feed)
        return out if self.keep is None else {k: v for k, v in out.items() if k in self.keep}


class _GpuPart:
    """Metal's second graph: returns the audio the CPU session already made (passed through as an input)."""

    def set_speed(self, speed):
        pass

    def forward(self, feed):
        return {'t_audio': feed['t_audio']}


@contextlib.contextmanager
def _as_backend(tts, backend, log):
    """Run tts (a real CPU model) through `backend`'s code path, every kernel still on the CPU."""
    m = tts._model.model if tts.family == 'tts08' else tts._model
    cpp_model = importlib.import_module('kitten_inference.tts08.cpp_model')
    gate = cpp_model._ENGINE_GATE
    cpp_model._ENGINE_GATE = type('Gate', (), {'use': staticmethod(lambda engine, b: contextlib.nullcontext())})()
    saved_backend = m.backend
    try:
        if tts.family == 'tts08':
            saved = m.session
            m.session, m.backend = _Recorder(saved, log), backend
            try:
                yield
            finally:
                m.session = saved
        else:
            key = next(iter(m._sessions))
            saved = m._sessions[key]
            if backend == 'metal':
                m._sessions[key] = (_Recorder(saved, log), _GpuPart(), frozenset({'t_audio'}), 'metal')
            else:
                m._sessions[key] = _Recorder(saved, log)
            m.backend = backend
            try:
                yield
            finally:
                m._sessions[key] = saved
    finally:
        m.backend = saved_backend
        cpp_model._ENGINE_GATE = gate


@pytest.fixture(scope='module', params=FAMILIES)
def real(request):
    tts = KittenTTS(MODELS[request.param], backend='cpu')
    tts.generate('Hello.')   # loads the session
    return tts


@needs_models
def test_generate_feeds_and_audio_same_on_every_backend(real):
    feeds, audio = {}, {}
    for backend in _family.BACKENDS[real.family]:
        log = []
        with _as_backend(real, backend, log):
            audio[backend] = real.generate(TEXT, speed=1.1)
        feeds[backend] = log
    first = _family.BACKENDS[real.family][0]
    for backend in feeds:
        assert feeds[backend] == feeds[first], backend
        assert np.array_equal(audio[backend], audio[first]), backend
    assert len([f for f in feeds[first] if f and f[0] != 'speed']) >= 2   # several chunks were planned


@needs_models
def test_errors_voices_and_languages_same_on_every_backend(real):
    bad = {'speed': lambda: real.generate('Hi.', speed=-1),
           'voice': lambda: real.generate('Hi.', voice='nobody'),
           'language': lambda: real.generate('Hi.', language='xx'),
           'option': lambda: real.generate('Hi.', seed=1)}
    seen = {}
    for backend in _family.BACKENDS[real.family]:
        with _as_backend(real, backend, []):
            seen[backend] = ({k: _error(f) for k, f in bad.items()},
                             real.list_voices(), real.languages(), real.default_voice())
    assert len({repr(v) for v in seen.values()}) == 1, seen
    errors, voices, languages, default = next(iter(seen.values()))
    assert all(e is not None for e in errors.values()), errors
    assert default in voices
    assert real.list_voices(language=languages[0])


@needs_models
def test_voice_names_any_case(real):
    voice = real.list_voices()[0]
    assert np.array_equal(real.generate('Hi.', voice=voice.upper()), real.generate('Hi.', voice=voice.lower()))


@needs_models
@pytest.mark.parametrize('family', FAMILIES)
def test_auto_cpu_and_low_memory_list_the_same(family):
    a = KittenTTS(MODELS[family])
    b = KittenTTS(MODELS[family], backend='cpu', low_memory=True)
    assert (a.list_voices(), a.languages(), a.default_voice()) == (b.list_voices(), b.languages(), b.default_voice())
    audio = b.generate('Hello there, this is the low memory mode.')   # mini/nano/pico pick the backend here
    assert (b._model.backend if family == 'tts2_tiny' else b._model.model.backend) == 'cpu'
    assert np.isfinite(audio).all() and len(audio) > 12000


def test_threads_setting_on_a_build_without_openmp_warns(monkeypatch):
    """macOS builds have no OpenMP: OMP_NUM_THREADS is ignored there, with a warning rather than in silence."""
    cpp_model = importlib.import_module('kitten_inference.tts08.cpp_model')
    one_thread = type('E', (), {'max_threads': staticmethod(lambda: 1)})()
    four_threads = type('E', (), {'max_threads': staticmethod(lambda: 4)})()
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    with pytest.warns(RuntimeWarning, match='OMP_NUM_THREADS=4 has no effect'):
        cpp_model._warn_if_threads_ignored(one_thread)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cpp_model._warn_if_threads_ignored(four_threads)       # OpenMP build: no warning
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        cpp_model._warn_if_threads_ignored(one_thread)          # one thread asked, one thread run
