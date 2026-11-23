"""The kitten-tts command."""
import argparse
import sys
from pathlib import Path
from ._api import KittenTTS, model_family


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='kitten-tts', add_help=False, allow_abbrev=False)
    parser.add_argument('--model', required=True,
                        help='Hugging Face model name, or a local copy of a TTS 0.8 model or of Kitten TTS 2 mini, '
                             'nano or pico')
    if argv == ['--version']:
        print(__import__('kitten_inference').__version__)
        return 0
    if argv in (['--help'], ['-h'], []):
        parser.print_help()
        print('\nSelect --model MODEL --help for synthesis and advanced options.')
        return 0 if argv else 2
    from .tts2._cli import NATIVE_VALUE_OPTIONS, OWN_VALUE_OPTIONS
    value_options = NATIVE_VALUE_OPTIONS | OWN_VALUE_OPTIONS | {'--model', '--speed', '--normalize'}
    model_args, rest = [], []
    remaining = iter(argv)
    for arg in remaining:
        key, eq, _ = arg.partition('=')
        target = model_args if key == '--model' else rest
        target.append(arg)
        # A synthesis value such as --text '--model=example' is not a model selector.
        if key in value_options and not eq:
            value = next(remaining, None)
            if value is not None:
                target.append(value)
    args = parser.parse_args(model_args)
    family = model_family(args.model)
    if family == 'tts2':
        from .tts2._cli import main as native_main
        return native_main(rest)
    cpu = argparse.ArgumentParser(prog='kitten-tts --model ' + args.model)
    texts = cpu.add_mutually_exclusive_group()
    texts.add_argument('--text')
    texts.add_argument('--text-file', type=Path)
    cpu.add_argument('--voice')
    cpu.add_argument('--output', type=Path)
    # The same options for Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico; a setting a model cannot use is
    # refused with the same message as in Python (kitten_inference/tts08/_family.py), not by argparse.
    # backend 'cuda': Kitten TTS 2 mini, nano and pico on NVIDIA GPUs (Linux x86-64, Windows x64)
    cpu.add_argument('--backend', choices=['cpu', 'metal', 'cuda'])
    cpu.add_argument('--cache-dir')
    cpu.add_argument('--speed', type=float, default=1.0,
                     help='1.0 is normal; Kitten TTS 2 mini, nano and pico take 0.25 to 4' if family == 'tts2_tiny' else None)
    # Kitten TTS 2 mini, nano and pico always normalise with their own front end (--normalize false is refused)
    cpu.add_argument('--normalize', choices=['true', 'false'], default='true')
    cpu.add_argument('--list-voices', action='store_true')
    cpu.add_argument('--language', help='language code (default: the model\'s default, en)')
    cpu.add_argument('--precision', choices=['fp32', 'int8'],
                     help='default: fp32 (TTS 0.8: the model name\'s, e.g. kitten-tts-mini-0.8-int8 is int8)')
    cpu.add_argument('--low-memory', action='store_true', help='least memory, slower (CPU only)')
    opts = cpu.parse_args(rest)
    if not opts.list_voices and (opts.text is None and opts.text_file is None or opts.output is None):
        cpu.error('synthesis requires --text or --text-file, and --output')
    try:
        tiny = family == 'tts2_tiny'
        model_options = {'low_memory': opts.low_memory}
        if opts.precision is not None:
            model_options['precision'] = opts.precision
        with KittenTTS(args.model, backend=opts.backend, cache_dir=opts.cache_dir, **model_options) as tts:
            if opts.list_voices:
                if tiny and opts.language is None:   # one line per language: several languages share voice names
                    print('\n'.join(f"{lang}: {', '.join(tts.list_voices(lang))}" for lang in tts.languages()))
                else:
                    print('\n'.join(tts.list_voices(opts.language)))
            else:
                text = opts.text if opts.text is not None else opts.text_file.read_text(encoding='utf-8')
                tts.generate_to_file(text, opts.output, voice=opts.voice, speed=opts.speed, language=opts.language,
                                     normalize=opts.normalize == 'true')
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f'kitten-tts: {exc}', file=sys.stderr)
        return 1
    return 0
