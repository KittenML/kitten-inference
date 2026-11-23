"""Build-only inputs for release assembly, derived from the root project."""
import os
import shutil
import tempfile
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent


def hooks(part):
    def build(kind, output, config_settings=None, metadata_directory=None):
        import tomli_w
        output = str(Path(output).resolve())
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        with tempfile.TemporaryDirectory(prefix='kitten-build-') as folder:
            stage = Path(folder)
            project = config['project']
            project['license-files'] = ['LICENSE']
            for name in ('LICENSE', 'README.md'):
                shutil.copy2(ROOT / name, stage / name)
            target = stage / 'kitten_inference'
            target.mkdir()
            if part == 'metadata':
                for p in (ROOT / 'src/kitten_inference').glob('*.py'):
                    shutil.copy2(p, target / p.name)
            elif part == 'asr_cpu':
                # The speech recognition CPU engine: compiled libraries only (kitten_inference/_asr_cpu, installed by
                # engines/asr-cpu/CMakeLists.txt), no Python files and no dependencies. Tagged py3-none-<platform>.
                project['name'] += '-asr-cpu'
                project['version'] = project['version'].split('.post', 1)[0]
                project['dependencies'] = []
                project.pop('scripts', None)
                target.rmdir()
                shutil.copytree(ROOT / 'engines/asr-cpu', stage / 'asr-cpu', ignore=shutil.ignore_patterns('python'))
                shutil.copy2(ROOT / 'engines/asr/NOTICE', stage / 'NOTICE')
                project['license-files'] = ['LICENSE', 'NOTICE', 'asr-cpu/llama.cpp/LICENSE']
                config['tool'] = {'scikit-build': {
                    'cmake': {'build-type': 'Release', 'source-dir': 'asr-cpu'}, 'build': {'targets': ['kasr']},
                    'install': {'components': ['python']}, 'wheel': {'packages': [], 'py-api': 'py3'},
                    'sdist': {'include': ['asr-cpu', 'LICENSE', 'NOTICE', 'README.md', 'pyproject.toml']}}}
                from scikit_build_core import build as backend
            else:
                project['name'] += '-' + part
                # Post releases reuse the locked native builds; internal inputs keep their base version.
                project['version'] = project['version'].split('.post', 1)[0]
                project.pop('scripts', None)
                shutil.copytree(ROOT / 'src/kitten_inference' / part, target / part)
            if part == 'asr_cpu':
                pass
            elif part == 'tts08':
                # Kitten TTS 2 mini/nano/pico run on this engine, so their Python package ships with it.
                shutil.copytree(ROOT / 'src/kitten_inference/tts2_tiny', target / 'tts2_tiny')
                shutil.copy2(ROOT / 'engines/lite/NOTICE', stage / 'NOTICE')
                shutil.copytree(ROOT / 'engines/lite/licenses', stage / 'licenses')
                project['license-files'] = ['LICENSE', 'NOTICE', 'licenses/*.txt']
                shutil.copytree(ROOT / 'engines/lite/engine', stage / 'engine')
                config['tool'] = {'scikit-build': dict(config['tool']['scikit-build'])}
                opts = config['tool']['scikit-build']
                opts['cmake']['source-dir'] = 'engine'
                # This wheel holds the TTS engine only. The speech recognition CPU engine (the root project's "kasr"
                # target) is built once per platform by engines/asr-cpu and added when the wheels are assembled.
                opts['build'] = dict(opts.get('build', {}), targets=['_engine'])
                opts['wheel']['packages'] = {'kitten_inference/tts08': 'kitten_inference/tts08',
                                             'kitten_inference/tts2_tiny': 'kitten_inference/tts2_tiny'}
                from scikit_build_core import build as backend
            else:
                if part == 'tts2':
                    shutil.copytree(ROOT / 'engines/tts2/tools/kt2n_assets', target / 'tts2/_assets')
                    shutil.copytree(ROOT / 'engines/tts2/licenses', target / 'tts2/licenses')
                    shutil.copy2(ROOT / 'engines/tts2/NOTICE', target / 'tts2/NOTICE')
                config['build-system'] = {'requires': ['hatchling==1.27.0'], 'build-backend': 'hatchling.build'}
                config['tool'] = {'hatch': {'build': {'targets': {'wheel': {'packages': ['kitten_inference'], 'core-metadata-version': '2.4'}, 'sdist': {'include': ['kitten_inference', 'LICENSE', 'README.md', 'pyproject.toml']}}}}}
                from hatchling import build as backend
            (stage / 'pyproject.toml').write_text(tomli_w.dumps(config))
            previous = Path.cwd()
            try:
                os.chdir(stage)
                return getattr(backend, kind)(output, config_settings)
            finally:
                os.chdir(previous)
    return (lambda wheel_directory, config_settings=None, metadata_directory=None: build('build_wheel', wheel_directory, config_settings),
            lambda sdist_directory, config_settings=None: build('build_sdist', sdist_directory, config_settings))
