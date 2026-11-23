"""Speech recognition from the command line: ``kitten-asr``, or ``python -m kitten_inference.asr``.

    kitten-asr --model KittenML/kitten-asr-tiny audio.wav [more.wav ...]
               [--language NAME] [--backend B] [--precision P] [--cache-dir DIR] [--system-prompt auto|none]
               [--json]

One transcript per line, with the file name in front when several files are given. ``--json`` prints one JSON
object per file instead: file, text, language and segments (start and end in seconds, text).
"""
import argparse
import json
import os
import sys

from . import _precision

PROG = 'kitten-asr'
_ENGINE_PREFIX = 'kitten-asr: '           # the engine's own errors start with its name


def _fail(message):
    """One line on stderr under the command's name, as argparse prints its own errors."""
    message = str(message)
    if message.startswith(_ENGINE_PREFIX):
        message = message[len(_ENGINE_PREFIX):]
    print(f'{PROG}: {message}', file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog=PROG, allow_abbrev=False,
                                     description='Transcribe audio files with a Kitten speech recognition model.')
    parser.add_argument('--model', required=True, help='Hugging Face model name or local model folder')
    parser.add_argument('audio', nargs='+', help='audio files (WAV, FLAC, OGG or MP3)')
    parser.add_argument('--language', help='language name such as English (default: the model detects it)')
    parser.add_argument('--cache-dir', help='where downloaded models are kept')
    parser.add_argument('--backend', choices=_precision.BACKENDS,
                        help='cuda: NVIDIA GPU (needs PyTorch); cpu: the CPU; metal: the Mac GPU '
                             '(default: an NVIDIA GPU when one can be used, otherwise the CPU)')
    parser.add_argument('--precision', choices=tuple(dict.fromkeys(_precision.GPU + _precision.CPU)),
                        help='bf16, fp16 or int8 on an NVIDIA GPU; q8_0, f16 or bf16 on the CPU and Mac GPU. '
                             'Small-enhanced also accepts its published int4 and int8 weights on every backend.')
    parser.add_argument('--system-prompt', choices=('auto', 'none'), default='auto',
                        help="auto: the system message the model folder names in asr_config.json "
                             "(empty without that file); none: always empty (default: auto)")
    parser.add_argument('--json', action='store_true', help='print one JSON object per file')
    if argv == ['--version']:
        print(__import__('kitten_inference').__version__)
        return 0
    if argv in (['--help'], ['-h'], []):
        parser.print_help()
        return 0 if argv else 2
    args = parser.parse_args(argv)
    try:
        for path in args.audio:           # before the model loads: a wrong file name should not cost a minute
            if not os.path.isfile(path):
                raise FileNotFoundError(f'audio file not found: {path}')
        from ._api import KittenASR
        with KittenASR(args.model, cache_dir=args.cache_dir, backend=args.backend, precision=args.precision,
                       system_prompt=args.system_prompt) as asr:
            results = asr.transcribe(list(args.audio), language=args.language, details=True)
    except KeyboardInterrupt:
        _fail('cancelled')
        return 130
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        _fail(exc)
        return 1
    for path, result in zip(args.audio, results):
        if args.json:
            segments = [{'start': round(s.start, 3), 'end': round(s.end, 3), 'text': s.text}
                        for s in result.segments]
            print(json.dumps({'file': path, 'text': result.text, 'language': result.language,
                              'segments': segments}, ensure_ascii=False))
        else:
            print(f'{path}: {result.text}' if len(args.audio) > 1 else result.text)
    return 0
