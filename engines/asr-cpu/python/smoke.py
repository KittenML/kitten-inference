"""Checks the installed speech recognition CPU engine library: run by packages.yml on the platform that built it.
Needs no model and no other package (the wheel holds libraries only).

1. libkasr loads and reports its API version.
2. The CPU backend variants in the wheel are exactly the expected ones (Linux): x86-64 ships the AVX2-and-newer
   variants; arm64 ships the NEON variants only (an Arm variant with i8mm on repacked q8_0 weights wrote wrong text,
   so no sve, i8mm, sme or armv9 variant may ship). macOS has its backends built into the libraries.
3. The engine starts its CPU backend on this machine (kasr_backend_init): at least one variant loads.
"""
import ctypes
import importlib.util
import os
import platform
import re
import sys

# kitten_inference is a namespace here: only this wheel is installed.
spec = importlib.util.find_spec("kitten_inference")
roots = list(spec.submodule_search_locations) if spec else []
names = {"darwin": "libkasr.dylib", "win32": "kasr.dll"}
name = names.get(sys.platform, "libkasr.so")
paths = [os.path.join(root, "_asr_cpu", "lib", name) for root in roots]
found = [p for p in paths if os.path.isfile(p)]
assert found, f"no {name} under {roots}"
folder = os.path.dirname(found[0])
dll_dir = os.add_dll_directory(folder) if sys.platform == "win32" else None
lib = ctypes.CDLL(found[0])
lib.kasr_api_version.restype = ctypes.c_int32
version = lib.kasr_api_version()
assert version >= 1, version
files = sorted(os.listdir(folder))
variants = sorted(m.group(1) for f in files for m in [re.fullmatch(r"(?:lib)?ggml-cpu-(.+)\.(?:so|dll)", f)] if m)

EXPECTED = {
    ("linux", "x86_64"): ["alderlake", "cascadelake", "haswell", "icelake", "sapphirerapids", "skylakex", "zen4"],
    ("linux", "aarch64"): ["armv8.0_1", "armv8.2_1", "armv8.2_2"],
    ("win32", "amd64"): ["alderlake", "cascadelake", "haswell", "icelake", "skylakex"],
    ("darwin", "arm64"): [],
}
key = (sys.platform, platform.machine().lower())
if key in EXPECTED:
    assert variants == EXPECTED[key], f"CPU backend variants {variants}, expected {EXPECTED[key]} on {key}"
banned = [v for v in variants if re.search(r"sve|i8mm|sme|armv9", v)]
assert not banned, f"an Arm variant that must not ship: {banned}"

lib.kasr_backend_init.restype = ctypes.c_int32
lib.kasr_backend_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.kasr_last_error.restype = ctypes.c_char_p
ok = lib.kasr_backend_init(None, None)
assert ok == 1, f"the CPU engine could not start on this machine: {(lib.kasr_last_error() or b'').decode()}"
print(f"{found[0]}: kasr API {version}; CPU backend started; variants {variants}; {files}")
