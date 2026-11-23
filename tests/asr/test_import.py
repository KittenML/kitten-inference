"""Importing costs nothing: `import kitten_inference` loads no other module of the package, `import
kitten_inference.asr` loads no third-party module, every name is loaded on first use, and every source file parses
on Python 3.8 (the files are in the wheels of every supported Python)."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import kitten_inference
import kitten_inference.asr as asr

ASR_DIR = Path(asr.__file__).parent
SOURCES = sorted(p.name for p in ASR_DIR.glob("*.py"))
THIRD_PARTY = ("numpy", "torch", "triton", "tokenizers", "soundfile", "huggingface_hub", "filelock", "safetensors",
               "transformers", "scipy")
# Files that are imported before anything is known about the machine: the standard library only.
LIGHT = ("__init__.py", "__main__.py", "_cli.py", "_deps.py", "_types.py", "_proc.py", "_child.py")

BARE_IMPORT = """
import json, sys
import kitten_inference
print(json.dumps(sorted(m for m in sys.modules if m.startswith("kitten_inference."))))
"""

# The third-party modules cannot be imported at all here, as on a machine without them.
ASR_IMPORT = f"""
import importlib.abc, json, sys
HIDDEN = {THIRD_PARTY!r}
class Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDDEN:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
sys.meta_path.insert(0, Hide())
before = set(sys.modules)
import kitten_inference.asr as asr
def third_party():
    # "__mp_main__" is the second name multiprocessing gives the main module
    names = {{m.partition(".")[0] for m in set(sys.modules) - before}} - {{"kitten_inference", "__mp_main__"}}
    known = getattr(sys, "stdlib_module_names", None)      # Python 3.10 and newer
    return sorted(n for n in names if (n not in known if known is not None else n in HIDDEN))
report = {{"import": third_party(), "ours": sorted(m for m in sys.modules if m.startswith("kitten_inference."))}}
for name in ("__version__", "DecodeResult", "EngineError", "PoolExhausted", "EngineStopped", "EngineProcess"):
    getattr(asr, name)
report["light names"] = third_party()
try:
    asr.Engine
    report["engine"] = "imported"
except ImportError as e:
    report["engine"] = str(e)
try:
    asr.no_such_name
    report["unknown"] = "found"
except AttributeError as e:
    report["unknown"] = str(e)
print(json.dumps(report))
"""

# PyTorch cannot be imported here; numpy can.
LAZY_NAME = """
import importlib.abc, json, sys
class Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("torch", "triton"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
sys.meta_path.insert(0, Hide())
import kitten_inference
early = sorted(m for m in sys.modules if m.startswith("kitten_inference."))
from kitten_inference import KittenASR
engine = ("torch", "kitten_inference.asr._engine", "kitten_inference.asr._core", "kitten_inference.asr._proc")
print(json.dumps({"early": early, "module": KittenASR.__module__, "name": KittenASR.__name__,
                  "engine": sorted(m for m in sys.modules if m in engine)}))
"""


def _run(script):
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_bare_import_loads_no_other_module():
    assert _run(BARE_IMPORT) == []


def test_asr_import_needs_no_library():
    report = _run(ASR_IMPORT)
    assert report["import"] == [] and report["ours"] == ["kitten_inference.asr"], report
    assert report["light names"] == [], report        # the types and the process version: standard library only
    if sys.version_info < (3, 10):
        this = "%d.%d" % sys.version_info[:2]
        expected = f"Kitten ASR needs Python 3.10 or newer (this is Python {this})."
    else:
        expected = "Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch"
    assert report["engine"] == expected
    assert report["unknown"] == "module 'kitten_inference.asr' has no attribute 'no_such_name'"


def test_top_level_name_resolves_on_first_use():
    report = _run(LAZY_NAME)
    assert report["early"] == [], report
    assert (report["module"], report["name"]) == ("kitten_inference.asr._api", "KittenASR")
    assert report["engine"] == [], report               # the class is there before PyTorch is looked at


def test_lazy_names_are_the_objects_of_their_modules():
    from kitten_inference.asr import _api, _proc, _types
    for name in ("KittenASR", "Transcript", "Segment", "StreamSession", "StreamUpdate"):
        assert getattr(asr, name) is getattr(_api, name), name
    for name in ("DecodeResult", "EngineError", "PoolExhausted", "EngineStopped"):
        assert getattr(asr, name) is getattr(_types, name), name
    assert asr.EngineProcess is _proc.EngineProcess
    assert set(asr.__all__) == set(asr._LAZY) | {"__version__"} and asr._LAZY["Engine"] == "_engine"
    assert kitten_inference.KittenASR is _api.KittenASR and "KittenASR" in kitten_inference.__all__
    # a post release (0.2.0.post1) changes only kitten_inference.__version__; this package keeps the base version
    assert asr.__version__ == re.sub(r"\.post\d+$", "", kitten_inference.__version__)
    with pytest.raises(AttributeError, match="no_such_name"):
        asr.no_such_name


def test_the_source_files_are_found():
    assert {"__init__.py", "__main__.py", "_api.py", "_audio.py", "_cli.py", "_deps.py", "_types.py", "_engine.py",
            "_core.py", "_attn.py", "_loader.py", "_proc.py", "_child.py"} <= set(SOURCES)


def _tree(name):
    return ast.parse((ASR_DIR / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 8))


@pytest.mark.parametrize("name", SOURCES)
def test_source_parses_on_python_3_8(name):
    tree = _tree(name)       # SyntaxError for match statements, parenthesised context managers, ...
    # What parses on 3.8 but fails there when it runs: X | Y and list[int] in annotations that are evaluated
    # (no "from __future__ import annotations"), and the string methods of Python 3.9.
    future = any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
                 and "annotations" in [a.name for a in node.names] for node in tree.body)
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [node.args.vararg, node.args.kwarg]
            annotations += [a.annotation for a in args if a is not None and a.annotation is not None]
            annotations += [node.returns] if node.returns is not None else []
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in ("removeprefix", "removesuffix"), f"{name}:{node.lineno}: str.{node.attr}"
    new_style = [n for a in annotations for n in ast.walk(a)
                 if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr))
                 or (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                     and n.value.id in ("list", "dict", "tuple", "set", "frozenset", "type"))]
    assert future or not new_style, f"{name}:{new_style[0].lineno}: this annotation needs the __future__ import"


def _imports_at_import_time(tree):
    """Top-level names of the modules a file imports when it is imported (function bodies are left out)."""
    names, pending = set(), list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            names.update(a.name.partition(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                names.add(node.module.partition(".")[0])
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return names


@pytest.mark.parametrize("name", LIGHT)
def test_light_modules_import_the_standard_library_only(name):
    assert not _imports_at_import_time(_tree(name)) & set(THIRD_PARTY)


@pytest.mark.parametrize("name", ["_api.py", "_audio.py"])
def test_the_user_layer_imports_numpy_only(name):
    # KittenASR, load and resample can be imported without PyTorch; soundfile is loaded when a file is read
    assert _imports_at_import_time(_tree(name)) & set(THIRD_PARTY) <= {"numpy"}
