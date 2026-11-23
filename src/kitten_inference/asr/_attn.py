"""Attention over the paged key/value pool of the decoder, with three interchangeable kernels."""
# Keys and values of all sequences live in one pool per layer: K, V [P, page, nkv, hd], page 0 is a dummy.
# Row b of the block table bt [B, n_pages] int32 lists the pages of sequence b in order (unused entries 0),
# so position p of sequence b is K[bt[b, p // page], p % page].
#
#     fn = make("auto", page=16, nh=16, nkv=8, hd=128, max_seq=2048, device="cuda", dtype=torch.bfloat16)
#     o = fn(q, K, V, cu_q, seqused, bt, max_q, max_k, qpos)          # [T, nh, hd], q's dtype
#
#   q        [T, nh, hd]  query rows of B sequences, concatenated
#   cu_q     [B+1] int32  rows of sequence b are cu_q[b] .. cu_q[b+1]-1; they are its LAST positions
#   seqused  [B] int32    keys of sequence b (the keys of this pass are already in the pool)
#   bt       [B, >= ceil(max_k / page)] int32
#   max_q    int          upper bound on rows per sequence; 1 = decode step
#   max_k    int          upper bound on seqused
#   qpos     [T] int64    position of each row (= seqused[b] - rows_b + j); only the "torch" kind reads it
#
# Row j of sequence b attends to positions 0 .. seqused[b] - rows_b + j of that sequence (causal), scale
# 1/sqrt(hd), query head h uses key/value head h // (nh // nkv).
#
# Kinds
#   triton  the two Triton kernels below; needs triton (comes with torch on Linux) and a C compiler
#   torch   gather each sequence's keys, then scaled_dot_product_attention; needs nothing, copies the keys
#   auto    triton when it can run here, else torch
#   flash   the FlashAttention build that ships inside the vllm package, when that package is installed;
#           it runs only when asked for by name
#
# The triton kind's decode step cuts the keys of each sequence into pieces when few sequences decode at once
# (DECODE_SPLITS): _decode_split_kernel writes one partial result per piece, _decode_merge_kernel combines them.
#
# All kinds can be captured in a CUDA graph when max_q and max_k are fixed for that graph and every tensor
# argument is a static buffer or made inside the graph. "torch" is the exception for max_q > 1 (it needs a
# host sync to place the rows), see fn.graph_prefill.
import math
import os
import shutil
import sys
import sysconfig

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_ERR = ""
except Exception as e:                      # not installed or a broken install
    triton = tl = None
    _TRITON_ERR = "%s: %s" % (type(e).__name__, e)

KINDS = ("triton", "torch", "flash")
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Launch settings of the Triton kernels, read by make("triton"). Block sizes must be powers of two >= 16.
# These are the starting values of the kernels this code follows.
TRITON_OPTS = dict(dec_block_n=64, dec_warps=4, dec_stages=2,
                   pf_block_m=64, pf_block_n=64, pf_warps=8, pf_stages=1)
# Decode steps of few sequences cut each sequence's keys into pieces (split-K) so more of the GPU works on them:
# (largest batch, pieces). With more sequences than the last entry, one piece (the single-pass kernel). One
# sequence has only 8 programs (one per key/value head) otherwise. Measured on an RTX 4090 (3 rounds): 15-26 s
# clips 2-8% faster one at a time, short clips unchanged.
DECODE_SPLITS = ((2, 8), (8, 4), (16, 2))


# ------------------------------------------------------------------ what can run here
def _probe_flash():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: F401
    except Exception as e:
        return False, "vllm.vllm_flash_attn does not import (%s: %s)" % (type(e).__name__, str(e)[:160])
    return True, "vllm.vllm_flash_attn"


def _find_cc():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        path = shutil.which(name) if name else None
        if path:
            return path
    return None


def _python_include():
    # Same lookup as Triton's build step, which compiles a small CPython extension (its kernel launcher).
    try:
        scheme = sysconfig.get_default_scheme()
        return sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]
    except Exception:
        return None


def _probe_triton():
    if triton is None:
        return False, "triton does not import (%s)" % _TRITON_ERR
    ver = getattr(triton, "__version__", "?")
    try:
        if tuple(int(x) for x in ver.split(".")[:2]) < (3, 5):
            return False, "triton %s is older than 3.5; the kernels were written for 3.5 and newer" % ver
    except ValueError:
        pass
    windows = sys.platform == "win32"
    cc = None if windows else _find_cc()
    if not windows and cc is None:
        return False, ("triton %s imports, but no C compiler was found ($CC, cc, gcc, clang); "
                       "Triton builds its launcher with one on first use" % ver)
    inc = _python_include()
    if inc and not os.path.exists(os.path.join(inc, "Python.h")):
        if windows:
            return False, "triton %s found, but Python.h is missing in %s (use Python from python.org)" % (ver, inc)
        return False, ("triton %s and %s found, but Python.h is missing in %s (install the Python development "
                       "headers)" % (ver, cc, inc))
    # The Windows fork bundles its launcher compiler and can also find the Visual Studio tools itself.
    return True, ("triton %s" % ver if windows else "triton %s, C compiler %s" % (ver, cc))


def _probe_torch():
    return True, "plain PyTorch: gathers the keys, then scaled_dot_product_attention"


def probe(kind):
    """(usable here, reason) for one attention kind by name. Only "flash" imports the vllm package."""
    if kind == "triton":
        return _probe_triton()
    if kind == "torch":
        return _probe_torch()
    if kind == "flash":
        return _probe_flash()
    raise ValueError("unknown attention kind %r; choose one of %s" % (kind, ", ".join(KINDS)))


def available():
    """name -> (usable here, reason). Probing "flash" imports the vllm package if it is installed."""
    return {kind: probe(kind) for kind in KINDS}


def _auto():
    # "flash" is never picked automatically: probing it imports the vllm package, which is slow, and the
    # choice should not change with what else happens to be installed.
    return "triton" if probe("triton")[0] else "torch"


PAGE = 16                 # tokens per page when the caller does not choose


def resolve(kind):
    """The attention kind to build for `kind`, before a page size is known. "auto" is "triton" when that
    can run here, else "torch"; it never picks "flash". A named kind is checked: ValueError for an unknown
    name, RuntimeError when it cannot run here."""
    if kind == "auto":
        return _auto()
    if kind not in KINDS:
        raise ValueError("unknown attention kind %r; choose one of %s or 'auto'" % (kind, ", ".join(KINDS)))
    ok, why = probe(kind)
    if not ok:
        raise RuntimeError("attention kind %r is not usable here: %s" % (kind, why))
    return kind


def check_page(kind, page):
    """Raises ValueError when `page` (tokens per page) cannot be used with attention kind `kind`."""
    # At least 2: a prefill graph is padded with sequences of 1..page tokens, spread in steps of page - 1.
    if isinstance(page, bool) or not isinstance(page, int) or page < 2:
        raise ValueError("page must be a whole number of 2 or more tokens, got %r" % (page,))
    if kind == "flash" and page % 16:
        raise ValueError("attention kind 'flash' needs a page size divisible by 16, got %d" % page)


# ------------------------------------------------------------------ flash: the kernel inside the vllm package
def _make_flash(page, nh, nkv, hd, max_seq, device, dtype):
    check_page("flash", page)
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
    except Exception as e:
        raise RuntimeError("attention kind 'flash' needs the vllm package (vllm.vllm_flash_attn): %s: %s"
                           % (type(e).__name__, e)) from e

    def fn(q, K, V, cu_q, seqused, bt, max_q, max_k, qpos=None):
        return fa(q, K, V, max_seqlen_q=max_q, cu_seqlens_q=cu_q, max_seqlen_k=max_k, seqused_k=seqused,
                  causal=True, block_table=bt)
    return fn


# ------------------------------------------------------------------ torch: gather + scaled_dot_product_attention
def _make_torch(page, nh, nkv, hd, max_seq, device, dtype):
    gqa = nh != nkv
    ar_all = torch.arange(max_seq, device=device)

    def fn(q, K, V, cu_q, seqused, bt, max_q, max_k, qpos=None):
        """Gathers each sequence's first max_k keys/values through its block table and runs PyTorch attention
        with a mask "key position <= query position". One query row per sequence when max_q == 1."""
        B = bt.shape[0]
        n_pages = -(-max_k // page)
        if bt.shape[1] < n_pages:
            raise ValueError("block table has %d pages per sequence, max_k=%d needs %d" % (bt.shape[1], max_k, n_pages))
        S = n_pages * page
        idx = bt[:, :n_pages].long()
        Ks = K[idx].reshape(B, S, nkv, hd).transpose(1, 2)
        Vs = V[idx].reshape(B, S, nkv, hd).transpose(1, 2)
        ar = ar_all[:S] if S <= ar_all.shape[0] else torch.arange(S, device=q.device)
        if max_q == 1:
            if qpos is None:
                qpos = seqused.long() - 1
            mask = (ar[None, :] <= qpos[:, None])[:, None, None, :]
            return F.scaled_dot_product_attention(q[:, :, None, :], Ks, Vs, attn_mask=mask, enable_gqa=gqa)[:, :, 0, :]
        nq = (cu_q[1:] - cu_q[:-1]).long()
        row_seq = torch.repeat_interleave(torch.arange(B, device=q.device), nq)      # host sync: not graph safe
        row_j = torch.arange(q.shape[0], device=q.device) - cu_q[:-1].long()[row_seq]
        if qpos is None:
            qpos = (seqused.long() - nq)[row_seq] + row_j
        qp = q.new_zeros(B, max_q, nh, hd)
        qp[row_seq, row_j] = q
        pp = torch.zeros(B, max_q, dtype=torch.long, device=q.device)               # padding rows look at key 0 only
        pp[row_seq, row_j] = qpos
        mask = (ar[None, None, :] <= pp[:, :, None])[:, None]
        o = F.scaled_dot_product_attention(qp.transpose(1, 2), Ks, Vs, attn_mask=mask, enable_gqa=gqa)
        return o.transpose(1, 2)[row_seq, row_j]
    return fn


# ------------------------------------------------------------------ triton: kernels that read the pool in place
# Structure (page-indexed key loads, tl.dot with 16-bit inputs and float32 accumulation, online softmax,
# one program per sequence x head tile, early return for programs past a sequence's rows) follows
#   SGLang, python/sglang/kernels/ops/attention/decode_attention.py and extend_attention.py
#   https://github.com/sgl-project/sglang  (Apache-2.0, Copyright 2023-2024 SGLang Team; its decode kernel is
#   adapted from https://github.com/ModelTC/lightllm, Apache-2.0)
# and Triton's tutorial python/tutorials/06-fused-attention.py, https://github.com/triton-lang/triton (MIT).
# Changes: keys are addressed through a per-sequence block table of pages instead of a token index list,
# decode is a single pass per (sequence, key/value head) without a split/merge stage, prefill takes every
# key from the pool and masks by position, and lengths are read from device tensors.
if triton is not None:

    @triton.jit(do_not_specialize=["bt_stride"])
    def _decode_kernel(Q, K, V, O, CU, SEQ, BT, scale, bt_stride,
                       G: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                       PAGE: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
        # One program per (sequence, key/value head). It computes the G query heads that share this
        # key/value head, so every key and value is read once. The G heads sit in a tile of ROWS rows
        # (16 = one tensor-core tile); the spare rows hold zeros and are never stored.
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        row = tl.load(CU + b)
        if tl.load(CU + b + 1) <= row:                  # sequence without a query row
            return
        klen = tl.load(SEQ + b)                         # the row is at position klen - 1: it sees every key

        offs_d = tl.arange(0, HD)
        g = tl.arange(0, ROWS)
        live_g = g < G
        q_off = (row.to(tl.int64) * NH + kvh * G + g[:, None]) * HD + offs_d[None, :]
        q = tl.load(Q + q_off, mask=live_g[:, None], other=0.0)                     # [ROWS, HD]

        # Running maximum starts at a large finite value, not -inf: a tile with no visible key then gives
        # exp(-inf) = 0 instead of exp(-inf + inf) = NaN.
        e_max = tl.zeros([ROWS], dtype=tl.float32) - 1.0e30
        e_sum = tl.zeros([ROWS], dtype=tl.float32)
        acc = tl.zeros([ROWS, HD], dtype=tl.float32)
        bt_row = BT + b * bt_stride
        d_off = kvh * HD + offs_d                       # offset of this head's vector inside one token
        for start in range(0, klen, BLOCK_N):
            pos = start + tl.arange(0, BLOCK_N)
            live = pos < klen
            page = tl.load(bt_row + pos // PAGE, mask=live, other=0).to(tl.int64)
            tok = (page * PAGE + pos % PAGE) * (NKV * HD)                           # int64: pools can be large
            k = tl.load(K + (tok[None, :] + d_off[:, None]), mask=live[None, :], other=0.0)     # [HD, BLOCK_N]
            s = tl.dot(q, k) * scale                                                # float32 [ROWS, BLOCK_N]
            s = tl.where(live[None, :], s, float("-inf"))
            m = tl.maximum(e_max, tl.max(s, 1))
            alpha = tl.exp(e_max - m)
            p = tl.exp(s - m[:, None])
            v = tl.load(V + (tok[:, None] + d_off[None, :]), mask=live[:, None], other=0.0)     # [BLOCK_N, HD]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * alpha + tl.sum(p, 1)
            e_max = m

        e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)       # no key at all: write zeros, not NaN
        out = acc / e_sum[:, None]
        tl.store(O + q_off, out.to(O.dtype.element_ty), mask=live_g[:, None])

    @triton.jit(do_not_specialize=["bt_stride"])
    def _decode_split_kernel(Q, K, V, MO, ML, CU, SEQ, BT, scale, bt_stride,
                             G: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                             PAGE: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr, NS: tl.constexpr):
        # The decode kernel above with the keys of a sequence cut into NS pieces, one program per (sequence,
        # key/value head, piece), so that one sequence keeps more of the GPU busy (split-K / "flash decoding").
        # Each program writes its piece's normalised output to MO [B, NH, NS, HD] and its log-sum-exp to
        # ML [B, NH, NS]; _decode_merge_kernel combines the pieces.
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        row = tl.load(CU + b)
        if tl.load(CU + b + 1) <= row:
            return
        klen = tl.load(SEQ + b)
        per = tl.cdiv(tl.cdiv(klen, NS), BLOCK_N) * BLOCK_N
        lo = sp * per
        hi = tl.minimum(lo + per, klen)

        offs_d = tl.arange(0, HD)
        g = tl.arange(0, ROWS)
        live_g = g < G
        q_off = (row.to(tl.int64) * NH + kvh * G + g[:, None]) * HD + offs_d[None, :]
        q = tl.load(Q + q_off, mask=live_g[:, None], other=0.0)

        e_max = tl.zeros([ROWS], dtype=tl.float32) - 1.0e30
        e_sum = tl.zeros([ROWS], dtype=tl.float32)
        acc = tl.zeros([ROWS, HD], dtype=tl.float32)
        bt_row = BT + b * bt_stride
        d_off = kvh * HD + offs_d
        for start in range(lo, hi, BLOCK_N):
            pos = start + tl.arange(0, BLOCK_N)
            live = pos < hi
            page = tl.load(bt_row + pos // PAGE, mask=live, other=0).to(tl.int64)
            tok = (page * PAGE + pos % PAGE) * (NKV * HD)
            k = tl.load(K + (tok[None, :] + d_off[:, None]), mask=live[None, :], other=0.0)
            s = tl.dot(q, k) * scale
            s = tl.where(live[None, :], s, float("-inf"))
            m = tl.maximum(e_max, tl.max(s, 1))
            alpha = tl.exp(e_max - m)
            p = tl.exp(s - m[:, None])
            v = tl.load(V + (tok[:, None] + d_off[None, :]), mask=live[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * alpha + tl.sum(p, 1)
            e_max = m
        nonempty = e_sum > 0.0
        out = acc / tl.where(nonempty, e_sum, 1.0)[:, None]
        lse = tl.where(nonempty, e_max + tl.log(tl.where(nonempty, e_sum, 1.0)), float("-inf"))
        h = kvh * G + g
        base = (b.to(tl.int64) * NH + h) * NS + sp
        tl.store(MO + base[:, None] * HD + offs_d[None, :], out, mask=live_g[:, None])
        tl.store(ML + base, lse, mask=live_g)

    @triton.jit
    def _decode_merge_kernel(MO, ML, O, CU, NH: tl.constexpr, HD: tl.constexpr, NS: tl.constexpr):
        # One program per (sequence, query head): output = sum over pieces of exp(lse_i - lse) * out_i.
        b = tl.program_id(0)
        h = tl.program_id(1)
        row = tl.load(CU + b)
        if tl.load(CU + b + 1) <= row:
            return
        offs_s = tl.arange(0, NS)
        offs_d = tl.arange(0, HD)
        base = (b.to(tl.int64) * NH + h) * NS
        lse = tl.load(ML + base + offs_s)
        m = tl.max(lse, 0)
        m = tl.where(m == float("-inf"), 0.0, m)
        w = tl.exp(lse - m)
        tot = tl.sum(w, 0)
        o = tl.load(MO + (base + offs_s)[:, None] * HD + offs_d[None, :])
        out = tl.sum(o * w[:, None], 0) / tl.where(tot > 0.0, tot, 1.0)
        tl.store(O + (row.to(tl.int64) * NH + h) * HD + offs_d, out.to(O.dtype.element_ty))

    @triton.jit(do_not_specialize=["bt_stride"])
    def _prefill_kernel(Q, K, V, O, CU, SEQ, BT, scale, bt_stride,
                        G: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                        PAGE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # One program per (sequence, block of BLOCK_M query rows, query head). The grid is sized for the
        # longest allowed sequence, so most programs of a short sequence stop at the first test. A padded
        # graph bucket (many one-page sequences, max_q in the thousands) is almost only such programs.
        b = tl.program_id(0)
        mb = tl.program_id(1)
        h = tl.program_id(2)
        row0 = tl.load(CU + b)
        nq = tl.load(CU + b + 1) - row0
        if mb * BLOCK_M >= nq:
            return
        klen = tl.load(SEQ + b)
        kvh = h // G

        offs_d = tl.arange(0, HD)
        r = mb * BLOCK_M + tl.arange(0, BLOCK_M)        # row inside the sequence
        live_m = r < nq
        qpos = klen - nq + r                            # the rows are the sequence's last nq positions
        hi = klen - nq + tl.minimum((mb + 1) * BLOCK_M, nq)     # keys the last row of this block can see
        q_off = ((row0 + r)[:, None].to(tl.int64) * NH + h) * HD + offs_d[None, :]
        q = tl.load(Q + q_off, mask=live_m[:, None], other=0.0)                     # [BLOCK_M, HD]

        e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e30      # finite start, see the decode kernel
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HD], dtype=tl.float32)
        bt_row = BT + b * bt_stride
        d_off = kvh * HD + offs_d
        for start in range(0, hi, BLOCK_N):             # stops after the last key block this row block needs
            pos = start + tl.arange(0, BLOCK_N)
            live = pos < hi
            page = tl.load(bt_row + pos // PAGE, mask=live, other=0).to(tl.int64)
            tok = (page * PAGE + pos % PAGE) * (NKV * HD)
            k = tl.load(K + (tok[None, :] + d_off[:, None]), mask=live[None, :], other=0.0)     # [HD, BLOCK_N]
            s = tl.dot(q, k) * scale                                                # float32 [BLOCK_M, BLOCK_N]
            s = tl.where(live[None, :] & (pos[None, :] <= qpos[:, None]), s, float("-inf"))
            m = tl.maximum(e_max, tl.max(s, 1))
            alpha = tl.exp(e_max - m)
            p = tl.exp(s - m[:, None])
            v = tl.load(V + (tok[:, None] + d_off[None, :]), mask=live[:, None], other=0.0)     # [BLOCK_N, HD]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * alpha + tl.sum(p, 1)
            e_max = m

        e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
        out = acc / e_sum[:, None]
        tl.store(O + q_off, out.to(O.dtype.element_ty), mask=live_m[:, None])


def _pow2(n):
    return n >= 1 and n & (n - 1) == 0


def _make_triton(page, nh, nkv, hd, max_seq, device, dtype):
    ok, why = _probe_triton()
    if not ok:
        raise RuntimeError("attention kind 'triton' is not usable here: " + why)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("attention kind 'triton' supports bf16 and fp16 only, got %s" % dtype)
    if not _pow2(hd) or hd < 16:
        raise ValueError("attention kind 'triton' needs a head size that is a power of two >= 16, got %d" % hd)
    if device.type != "cuda":
        raise ValueError("attention kind 'triton' needs a CUDA device, got %s" % device)
    # Triton launches on torch's CURRENT device and stream, not on the device of the tensors it is given.
    if device.index is not None and device.index != torch.cuda.current_device():
        raise RuntimeError("attention kind 'triton' runs on the current CUDA device (%d) but the engine is on %s; "
                           "select the GPU with CUDA_VISIBLE_DEVICES or torch.cuda.set_device"
                           % (torch.cuda.current_device(), device))
    # Triton looks for $CC, gcc and clang only. If just "cc" exists, hand it over through $CC.
    if sys.platform != "win32" and not os.environ.get("CC") and not (shutil.which("gcc") or shutil.which("clang")):
        os.environ["CC"] = _find_cc()

    opts = dict(TRITON_OPTS)
    for name in ("dec_block_n", "pf_block_m", "pf_block_n"):
        if not _pow2(opts[name]) or opts[name] < 16:
            raise ValueError("TRITON_OPTS[%r] must be a power of two >= 16, got %r" % (name, opts[name]))
    G = nh // nkv
    rows = max(16, 1 << (G - 1).bit_length())
    geom = dict(G=G, NH=nh, NKV=nkv, HD=hd, PAGE=page)
    dec = dict(geom, ROWS=rows, BLOCK_N=opts["dec_block_n"], num_warps=opts["dec_warps"], num_stages=opts["dec_stages"])
    pf_m = opts["pf_block_m"]
    pf = dict(geom, BLOCK_M=pf_m, BLOCK_N=opts["pf_block_n"], num_warps=opts["pf_warps"], num_stages=opts["pf_stages"])
    scale = 1.0 / math.sqrt(hd)
    q_tail, kv_tail, i32 = (nh, hd), (page, nkv, hd), torch.int32

    def fn(q, K, V, cu_q, seqused, bt, max_q, max_k, qpos=None):
        # The kernels do raw pointer arithmetic, so a wrong layout would read garbage instead of failing.
        # Lengths are read on the GPU: seqused[b] <= bt.shape[1] * page is the caller's promise.
        # Kernel arguments are the tensors' addresses, so inside a CUDA graph they must be static buffers
        # or tensors made in the graph; max_q fixes the grid. Keep tensors 16-byte aligned (whole
        # allocations or page-sized slices are): another alignment compiles a second variant.
        if bt.dim() != 2 or tuple(q.shape[1:]) != q_tail or tuple(K.shape[1:]) != kv_tail or V.shape != K.shape:
            raise ValueError("triton attention: expected q [T, %d, %d], K and V [P, %d, %d, %d], bt [B, pages]; got "
                             "q %s, K %s, V %s, bt %s" % (nh, hd, page, nkv, hd, tuple(q.shape), tuple(K.shape),
                                                         tuple(V.shape), tuple(bt.shape)))
        B = bt.shape[0]
        if cu_q.shape[0] != B + 1 or seqused.shape[0] != B:
            raise ValueError("triton attention: cu_q must have B+1 = %d entries and seqused B = %d; got %d and %d"
                             % (B + 1, B, cu_q.shape[0], seqused.shape[0]))
        if (q.dtype != dtype or K.dtype != dtype or V.dtype != dtype
                or cu_q.dtype != i32 or seqused.dtype != i32 or bt.dtype != i32):
            raise TypeError("triton attention: q, K, V must be %s and cu_q, seqused, bt int32; got %s %s %s / %s %s %s"
                            % (dtype, q.dtype, K.dtype, V.dtype, cu_q.dtype, seqused.dtype, bt.dtype))
        if not (K.is_contiguous() and V.is_contiguous() and cu_q.is_contiguous() and seqused.is_contiguous()
                and bt.stride(1) == 1):
            raise ValueError("triton attention: K, V, cu_q, seqused must be contiguous and bt rows dense")
        if not q.is_contiguous():
            q = q.contiguous()
        o = torch.empty_like(q)
        ns = next((n for top, n in DECODE_SPLITS if B <= top), 1) if max_q == 1 else 1
        if max_q == 1 and ns > 1:
            mo = torch.empty(B, nh, ns, hd, device=q.device, dtype=torch.float32)
            ml = torch.empty(B, nh, ns, device=q.device, dtype=torch.float32)
            _decode_split_kernel[(B, nkv, ns)](q, K, V, mo, ml, cu_q, seqused, bt, scale, bt.stride(0), NS=ns, **dec)
            _decode_merge_kernel[(B, nh)](mo, ml, o, cu_q, NH=nh, HD=hd, NS=ns)
        elif max_q == 1:
            _decode_kernel[(B, nkv)](q, K, V, o, cu_q, seqused, bt, scale, bt.stride(0), **dec)
        else:
            _prefill_kernel[(B, -(-max_q // pf_m), nh)](q, K, V, o, cu_q, seqused, bt, scale, bt.stride(0), **pf)
        return o
    return fn


_MAKERS = {"triton": _make_triton, "torch": _make_torch, "flash": _make_flash}


def make(kind, *, page, nh, nkv, hd, max_seq, device, dtype):
    """Returns fn(q, K, V, cu_q, seqused, bt, max_q, max_k, qpos) -> o [T, nh, hd] (see the module text).
    fn.kind is the resolved name; fn.graph_prefill says whether passes with max_q > 1 can be captured in
    CUDA graphs."""
    if kind == "auto":
        kind = _auto()
    if kind not in _MAKERS:
        raise ValueError("unknown attention kind %r; choose one of %s or 'auto'" % (kind, ", ".join(KINDS)))
    if nkv < 1 or nh % nkv:
        raise ValueError("query heads (%d) must be a multiple of key/value heads (%d)" % (nh, nkv))
    if page < 1 or max_seq % page:
        raise ValueError("max_seq (%d) must be a multiple of the page size (%d)" % (max_seq, page))
    fn = _MAKERS[kind](page, nh, nkv, hd, max_seq, torch.device(device), _DTYPES.get(dtype, dtype))
    fn.kind = kind
    fn.graph_prefill = kind != "torch"
    return fn
