"""check_isa.py OBJDUMP NM LIB: lists the functions of an arm64 library that use instructions newer than Armv8.0
(dot product, int8 matrix multiply, fp16 arithmetic, LSE atomics, RDM). In the run-time dispatch build only the
per-level kernel copies (names ending _dp / _i8mm, and the fp16 conv kernels) may use them. Exit 1 if any other
function with a global or weak symbol (where the linker can pick one copy of an inline function for every caller) does."""
import re, subprocess, sys
objdump, nm, lib = sys.argv[1:4]
pat = re.compile(r"\t(sdot|udot|usdot|sudot|smmla|ummla|usmmla|ldadd\w*|cas\w*|swp\w*|ldclr\w*|ldset\w*|ldeor\w*|sqrdml[as]h)\b"
                 r"|\t(fmla|fmls|fadd|fsub|fmul|fdiv|fmax|fmin|fabs|fneg|fcvtzs|scvtf)\s+v\d+\.(8h|4h)"
                 r"|\t(fmla|fmadd|fadd|fmul|fsub)\s+h\d+")
syms = {}
for line in subprocess.run([nm, "--defined-only", lib], capture_output=True, text=True).stdout.splitlines():
    parts = line.split()
    if len(parts) == 3:
        syms[parts[2]] = parts[1]
out = subprocess.run([objdump, "-d", "--no-show-raw-insn", lib], capture_output=True, text=True).stdout
fn = None; hits = {}
for line in out.splitlines():
    m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
    if m:
        fn = m.group(1); continue
    if fn and pat.search(line):
        hits[fn] = hits.get(fn, 0) + 1
allowed = re.compile(r"_(dp|i8mm)\b|_(dp|i8mm)[A-Z_0-9]|_(dp|i8mm)$|hgemm|kmerged_f16|kt_free_f16|conv_fp16|f16")
bad = []
for f, n in sorted(hits.items()):
    dem = subprocess.run(["c++filt", f], capture_output=True, text=True).stdout.strip() or f
    kind = syms.get(f, "?")
    ok = bool(allowed.search(f) or allowed.search(dem)) or kind in ("t", "?")
    print(("ok   " if ok else "BAD  ") + kind + " " + str(n) + " " + dem[:150])
    if not ok:
        bad.append(dem)
print("functions with newer instructions:", len(hits), "global/weak outside the per-level kernels:", len(bad))
sys.exit(1 if bad else 0)
