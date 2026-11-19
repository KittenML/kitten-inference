"""Kitten text-to-speech models."""
from pathlib import Path


def model_family(model):
    if not isinstance(model, (str, Path)) or not str(model).strip():
        raise ValueError('model must be a model name or local model directory')
    name = str(model).rstrip('/').lower()
    if name in ('kittenml/kitten-tts-2', 'kitten-tts-2'):
        return 'tts2'
    # Kitten TTS 2 mini, nano and pico are small StyleTTS2-style models: they run on the TTS 0.8 graph engine
    # on the CPU (Metal on Apple silicon; CUDA with backend='cuda', a build of that engine), not on the TTS 2 GPU engine.
    from .tts2_tiny.model import is_tiny_model
    if is_tiny_model(model):
        return 'tts2_tiny'
    # The 0.8 loader validates local and remote configurations.
    return 'tts08'


class KittenTTS:
    """Load a Kitten TTS model by name or local directory."""

    def __init__(self, model, *, backend=None, cache_dir=None, **model_options):
        self.family = model_family(model)
        self._closed = False
        if self.family == 'tts2':
            from .tts2 import KittenTTS2
            self._model = KittenTTS2(backend=backend, cache_dir=cache_dir, **model_options)
            return
        # Kitten TTS 0.8 and Kitten TTS 2 mini/nano/pico: the same options and errors on every backend, checked
        # before any download (kitten_inference/tts08/_family.py).
        from .tts08 import _family
        _family.check_model_options(self.family, model_options)
        if self.family == 'tts2_tiny':
            from .tts2_tiny import KittenTTS2Tiny
            self._model = KittenTTS2Tiny(str(model), backend=backend, cache_dir=cache_dir, **model_options)
        else:
            from .tts08 import KittenTTS as TTS08
            self._model = TTS08(str(model), backend=backend, cache_dir=cache_dir, **model_options)

    def _open(self):
        if self._closed:
            raise RuntimeError('this KittenTTS is closed')
        return self._model

    @property
    def sample_rate(self):
        m = self._open()
        return m.model.sample_rate if self.family == 'tts08' else m.sample_rate

    @property
    def capabilities(self):
        return {'streaming': True,
                'streaming_mode': 'incremental' if self.family == 'tts2' else 'text_chunks',
                'voice_cloning': self.can_clone(),
                'normalization_spans': self.family == 'tts08'}

    def can_clone(self):
        m = self._open()
        return m.can_clone if self.family == 'tts2' else False

    def default_voice(self):
        m = self._open()
        # TTS 0.8: Leo (the file expr-voice-5-m), as list_voices() names it.
        return m.default_voice if self.family != 'tts08' else getattr(m, 'default_voice_name', 'expr-voice-5-m')

    def list_voices(self, language=None):
        m = self._open()
        if self.family == 'tts2_tiny':
            # Without a language: every voice name once (mini gives several languages the same voices).
            return list(dict.fromkeys(v['name'] for v in m.voices(language)))
        if self.family == 'tts08':
            from .tts08._family import check_language, NAMES
            check_language(NAMES['tts08'], language, ['en'])
            return list(m.available_voices)
        if language is not None:
            raise TypeError('language= is available for Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico')
        return m.list_voices()

    def languages(self):
        m = self._open()
        return list(m.languages) if self.family == 'tts2_tiny' else ['en']

    def _options(self, options, *, file=False):
        opts = dict(options)
        if self.family == 'tts2':
            return opts
        # Kitten TTS 0.8 and Kitten TTS 2 mini/nano/pico accept the same synthesis options on every backend and
        # check them before synthesis starts (also for stream(), on its first chunk), in one wording.
        from .tts08 import _family
        _family.check_synth_options(self.family, opts)
        if 'speed' in opts:
            opts['speed'] = _family.check_speed(self.family, opts['speed'])
        if self.family == 'tts08':
            _family.check_language(_family.NAMES['tts08'], opts.pop('language', None), ['en'])
            opts['clean_text'] = opts.pop('normalize', file)
        else:
            # mini/nano/pico always prepare text with their own front end (English: numbers, symbols, ...).
            if not opts.pop('normalize', True):
                raise ValueError('Kitten TTS 2 mini, nano and pico always normalise text with their own front end: '
                                 'normalize=False is not available for them')
        return opts

    def generate(self, text, voice=None, **options):
        m = self._open()
        kwargs = self._options(options)
        if voice is not None:
            kwargs['voice'] = voice
        return m.generate(text, **kwargs)

    def stream(self, text, voice=None, **options):
        m = self._open()
        # Kitten TTS 0.8 and TTS 2 mini/nano/pico: chunk_chars (N: whole sentences packed up to N characters; None, the
        # default: the model's, as KittenML's packages: 0.8 one sentence per chunk, TTS 2 mini/nano/pico 380) and
        # trim_lead (default True: the first chunk starts at its first audible sample), checked here in one wording.
        stream_opts = {}
        if self.family != 'tts2':
            from .tts08 import _family, _stream
            for key in ('chunk_chars', 'trim_lead'):
                if key in options:
                    stream_opts[key] = options.pop(key)
            if 'chunk_chars' in stream_opts:
                stream_opts['chunk_chars'] = _family.check_chunk_chars(stream_opts['chunk_chars'])
            if 'trim_lead' in stream_opts:
                _stream.check_trim_lead(stream_opts['trim_lead'])
        kwargs = self._options(options)
        kwargs.update(stream_opts)
        if voice is not None:
            kwargs['voice'] = voice
        iterator = (m.stream if self.family == 'tts2' else m.generate_stream)(text, **kwargs)
        try:
            for chunk in iterator:
                self._open()
                yield chunk
        finally:
            iterator.close()

    def save_wav(self, path, audio):
        from ._audio import save_wav
        return save_wav(path, audio, self.sample_rate)

    def generate_to_file(self, text, output_path, voice=None, **options):
        # Preserve 0.8's file-generation normalization default.
        if self.family == 'tts08':
            options.setdefault('normalize', True)
        audio = self.generate(text, voice=voice, **options)
        import soundfile as sf
        sf.write(output_path, audio, self.sample_rate)
        return Path(output_path)

    def _tts2(self):
        m = self._open()
        if self.family != 'tts2':
            raise NotImplementedError('this operation requires Kitten TTS 2')
        return m

    def clone_voice(self, reference_audio_path, transcript, **options):
        return self._tts2().clone_voice(reference_audio_path, transcript, **options)

    def save_voice(self, voice, path):
        return self._tts2().save_voice(voice, path)

    def load_voice(self, path):
        return self._tts2().load_voice(path)

    @property
    def revision(self):
        return self._tts2().revision

    @property
    def model_dir(self):
        return self._tts2().model_dir

    def build_info(self):
        return self._tts2().build_info()

    def stats(self):
        return self._tts2().stats()

    def normalize_text(self, text, locale='en-US', return_spans=False):
        self._open()
        if self.family != 'tts08':
            raise NotImplementedError('standalone normalization is available for TTS 0.8')
        return self._model.normalize_text(text, locale=locale, return_spans=return_spans)

    def phonemize(self, text, language=None):
        """Kitten TTS 2 mini, nano and pico: the phoneme chunks the model speaks for `text` (a list of str). Apps that
        run the C library (phones) send these chunks to kt08_synthesize_tiny."""
        m = self._open()
        if self.family != 'tts2_tiny':
            raise NotImplementedError('phonemize is available for Kitten TTS 2 mini, nano and pico')
        return m.phonemize(text, language)

    def close(self):
        if not self._closed:
            self._closed = True
            model = self._model
            self._model = None
            if self.family == 'tts2':
                model.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
