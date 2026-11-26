"""The model of the ASR engine on one GPU: batched audio encoder, paged-KV decoder forward, CUDA graphs."""
# Everything here runs on one thread (the scheduler thread of _engine.py). Nothing is thread-safe.
#
# Layout
#   K, V            [L, P, page, nkv, hd]   paged KV pool shared by all sequences; page 0 is a dummy
#   block table     [B, max_pages] int32    row b lists the pages of sequence b in order
#   a sequence      head ids | audio embeddings | tail ids | generated text, at positions 0..n-1
from __future__ import annotations

import operator
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from . import _attn, _loader, _quant
from ._quant import mm

SR = 16000
HOP = 160
NFFT = 400
CHUNK_FRAMES = 100         # conv chunk: 1 s of mel frames
CHUNK_TOKENS = 13          # tokens per full conv chunk
BLOCK_FRAMES = 800         # encoder attention block: 8 s
BLOCK_TOKENS = 104
KV_TOKENS = 24576          # smallest default size of the KV pool, in tokens
WHOLE_TOKENS = 4 * BLOCK_TOKENS   # longest window encoded with whole-window attention: 32 s of audio
ENCODER_WINDOWS = ("auto", "blocks", "whole")


def out_len(frames: int) -> int:
    """Tokens the conv stem produces for a chunk of `frames` mel frames (<= 100)."""
    f = (frames - 1) // 2 + 1
    f = (f - 1) // 2 + 1
    return (f - 1) // 2 + 1


def audio_tokens(frames: int) -> int:
    nch = -(-frames // CHUNK_FRAMES)
    return (nch - 1) * CHUNK_TOKENS + out_len(frames - (nch - 1) * CHUNK_FRAMES)


# ------------------------------------------------------------------ fused math (torch.compile targets)
def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


def dec_pre(h, ln_w, qkv_w, qn_w, kn_w, cos, sin, eps: float, nh: int, nkv: int, hd: int):
    """input RMSNorm -> fused QKV GEMM -> per-head q/k RMSNorm -> RoPE. h [T,H]. Linear layers are given as
    _quant.Linear.args(): a 16-bit weight tensor or a (q, s, z, fmt) tuple."""
    T = h.shape[0]
    qkv = mm(_rms(h, ln_w, eps), qkv_w)
    q, k, v = qkv.split([nh * hd, nkv * hd, nkv * hd], -1)
    cos, sin = cos[:, None], sin[:, None]
    q = _rope(_rms(q.view(T, nh, hd), qn_w, eps), cos, sin)
    k = _rope(_rms(k.view(T, nkv, hd), kn_w, eps), cos, sin)
    return q, k, v.view(T, nkv, hd)


def dec_post(h, o, o_w, ln2_w, gu_w, down_w, eps: float):
    """attention out-proj + residual -> RMSNorm -> fused gate/up GEMM -> SiLU*up -> down + residual."""
    h = h + mm(o, o_w)
    g, u = mm(_rms(h, ln2_w, eps), gu_w).chunk(2, -1)
    return h + mm(F.silu(g) * u, down_w)


def dec_head(h, norm_w, eps: float, head_w):
    return torch.argmax(mm(_rms(h, norm_w, eps), head_w), -1)


def enc_layer(x, mask, ln1_w, ln1_b, qkv_w, qkv_b, o_w, o_b, ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b,
              nh: int):
    """Whisper-style pre-LN encoder layer over attention groups. x [G, W, d], mask [G,1,1,W] bool."""
    G, W, d = x.shape
    hs = F.layer_norm(x, (d,), ln1_w, ln1_b)
    q, k, v = _quant.linear_bias(hs, qkv_w, qkv_b).view(G, W, 3, nh, d // nh).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + _quant.linear_bias(a.transpose(1, 2).reshape(G, W, d), o_w, o_b)
    hs = F.layer_norm(x, (d,), ln2_w, ln2_b)
    return x + _quant.linear_bias(F.gelu(_quant.linear_bias(hs, fc1_w, fc1_b)), fc2_w, fc2_b)


def _own(f):
    """A copy of function f with a code object of its own. torch.compile keeps the graphs it compiled, and counts
    recompilations against its limit, per code object; a copy per Core keeps the graphs of one model and
    precision apart from those of another model loaded in the same process."""
    code = f.__code__.replace(co_name=f.__code__.co_name)
    assert code is not f.__code__
    return types.FunctionType(code, f.__globals__, f.__name__, f.__defaults__, f.__closure__)


def _compile_options():
    if sys.platform != "win32":
        return {}
    from torch._inductor import list_options

    workarounds = {
        # PyTorch's older launcher passes CUDA stream handles through a 32-bit C long.
        "use_static_cuda_launcher": False,
        # Long fused kernel names can exceed Windows' path limit in Triton's cache.
        "triton.unique_kernel_names": False,
    }
    supported = set(list_options())
    return {key: value for key, value in workarounds.items() if key in supported}


def plan(attention="auto", compile="auto"):
    """What the `attention` and `compile` options come to here: (attention kind, compile the fused math,
    warning text or None).

    "auto" takes the fast path when Triton can run here (it needs a C compiler and the Python headers, and
    so does torch.compile): the "triton" kind, compiled. Otherwise it is the "torch" kind, not compiled,
    and the warning text says why and what to install. A kind by name and compile=True or False are taken
    as given, without a warning.
    """
    kind = _attn.resolve(attention)
    ok, why = _attn.probe("triton")
    auto_compile = isinstance(compile, str)
    if auto_compile and compile != "auto":
        raise ValueError("compile must be 'auto', True or False, got %r" % (compile,))
    if sys.platform == "win32" and compile is True and not ok:
        raise RuntimeError(f"Kitten ASR: compile=True needs a working triton-windows installation ({why}). "
                           "Install a version that matches PyTorch, or use compile=False.")
    compiled = ok if auto_compile else bool(compile)
    slow = []
    if attention == "auto" and kind == "torch":
        slow.append("plain PyTorch attention")
    if auto_compile and not compiled:
        slow.append("no torch.compile")
    if not slow:
        return kind, compiled, None
    if sys.platform == "win32":
        return kind, compiled, ("Kitten ASR runs below full speed here (%s): %s. Install triton-windows with a "
                                "version that matches PyTorch for Triton acceleration." % (", ".join(slow), why))
    return kind, compiled, ("Kitten ASR runs below full speed here (%s): %s. For full speed install a C compiler "
                            "such as gcc and the Python development headers (Linux only)."
                            % (", ".join(slow), why))


class Core:
    """Weights, paged key/value pool, attention kernel and CUDA graphs of one model. Not thread-safe.

    attention  "auto" ("triton" when it can run here, else "torch") or a kind by name: "triton", "torch",
               "flash" (see _attn.py); the choice is in `attn_kind`
    compile    "auto" (torch.compile for the fused math when Triton can run here, else off), True or False;
               the choice is in `compiled`. `fallback` is the warning text when "auto" gave up speed, else
               None (see plan)
    page       tokens per KV page: "auto" (16) or a number the kind accepts
    kv_tokens  size of the KV pool, at least max_seq; None = 24576 tokens, or two pages per decode row
               when that is more
    precision  "bf16" (default), "fp16", "int8" or "int4"; for on-load quantization, the decoder's linear layers
               and the output head take it, the audio encoder stays in 16 bits. `dtype` is the older name
               for "bf16" / "fp16"
    encoder_window  "auto", "blocks" or "whole": how far the audio encoder's attention reaches for a window
               decoded without a stream cache. "blocks" = 8 s blocks, as the model was trained; "whole" =
               the whole window (up to 32 s) at once. "auto" = "whole" for models with a text width under
               1024 (kitten-asr-tiny), where it lowers the error rate, else "blocks". Streams with a cache
               always use 8 s blocks, so their cached blocks stay exact
    """

    def __init__(self, model, device: str = "cuda", dtype: str | None = None, kv_tokens: int | None = None,
                 max_seq: int = 2048, max_batch: int = 64, compile="auto", cuda_graph: bool = True,
                 batch_buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64), conv_batch: int = 256,
                 head_rows: int = 1024, prefill_buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
                 enc_buckets=(1, 2, 4, 8, 16, 32, 64, 128), prefill_graph: bool = True,
                 enc_graph: bool = True, pad_short: bool = False, attention: str = "auto",
                 kv_buckets=(256, 512, 1024, 2048), page="auto", cache_dir=None, precision=None,
                 encoder_window="auto"):
        self.device = torch.device(device)
        if dtype is not None and dtype not in ("bf16", "fp16"):
            raise ValueError("dtype must be one of bf16, fp16, got %r" % (dtype,))
        if dtype is not None and precision is not None and dtype != precision:
            raise ValueError("give precision=%r or dtype=%r, not both" % (precision, dtype))
        precision = precision or dtype or "bf16"
        _quant.check(precision, self.device)
        self.precision = precision
        self.dtype = _quant.act_dtype(precision)
        if encoder_window not in ENCODER_WINDOWS:
            raise ValueError("encoder_window must be one of %s, got %r" % (", ".join(ENCODER_WINDOWS), encoder_window))

        # ---- the attention kind and whether to compile first, then the page size and the pool size. All
        # are settled before the weights are read, so a wrong option fails at once.
        kind, self.compiled, self.fallback = plan(attention, compile)
        if page is None or (isinstance(page, str) and page == "auto"):
            page = _attn.PAGE
        else:
            try:
                page = operator.index(page)
            except TypeError:
                raise ValueError("page must be 'auto' or a whole number of tokens, got %r" % (page,)) from None
            _attn.check_page(kind, page)
        if max_seq < page or max_seq % page:
            raise ValueError("max_seq (%d) must be a multiple of the page size (%d)" % (max_seq, page))
        if kv_tokens is None:
            # a sequence takes whole pages, so larger pages get a larger default pool: two per decode row
            kv_tokens = max(KV_TOKENS, 2 * max_batch * page, max_seq)
        else:
            try:
                if isinstance(kv_tokens, bool):
                    raise TypeError
                kv_tokens = operator.index(kv_tokens)
            except TypeError:
                raise ValueError("kv_tokens must be a whole number of tokens, got %r" % (kv_tokens,)) from None
            # the pool has to hold one sequence of full length, or the longest request could never run
            if kv_tokens < max_seq:
                raise ValueError("kv_tokens (%d) must be at least max_seq (%d): the pool has to hold one "
                                 "sequence of full length" % (kv_tokens, max_seq))

        w = _loader.load(model, self.device, self.dtype, cache_dir=cache_dir, precision=precision)
        self.w = w
        self.tok = w.tok
        cfg = w.cfg
        self.cfg = cfg
        self.H = cfg["H"]
        self.nh, self.nkv, self.hd = cfg["nh"], cfg["nkv"], cfg["hd"]
        self.eps = cfg["eps"]
        if (cfg["chunk_frames"], cfg["block_frames"], cfg["n_mels"]) != (CHUNK_FRAMES, BLOCK_FRAMES, 128):
            raise ValueError("unsupported audio front end: %r"
                             % ({k: cfg[k] for k in ("chunk_frames", "block_frames", "n_mels")},))
        self.conv_batch, self.head_rows = conv_batch, head_rows
        self.kv_buckets = sorted(b for b in kv_buckets if b <= max_seq)
        if not self.kv_buckets or self.kv_buckets[-1] != max_seq:
            self.kv_buckets.append(max_seq)
        self.pad_short = pad_short          # True = pad windows under 1 s to a full 1 s chunk

        # ---- decoder weights (fused QKV and gate/up) and encoder weights (fused qkv), built by the loader.
        # Legacy on-load int8 converts the decoder linears. Published checkpoints already carry calibrated values.
        lin = precision if precision in _quant.FMT and not w.published_precision else "bf16"
        self.embed, self.norm_w = w.embed, w.norm_w
        self.linears = []
        W = []
        while w.W:
            ln, qkv, qn, kn, o_w, ln2, gu, down = w.W.pop(0)
            ls = [_quant.Linear(x, lin) for x in (qkv, o_w, gu, down)]
            self.linears += ls
            W.append((ln, ls[0].args(), qn, kn, ls[1].args(), ln2, ls[2].args(), ls[3].args()))
            del qkv, o_w, gu, down
        w.W = None
        self.W = W
        self.head = _quant.Linear(w.head_w, lin)
        self.linears.append(self.head)
        self.head_w = self.head.args()
        w.head_w = None
        self.weight_bytes = sum(x.nbytes() for x in self.linears)
        self.L = len(self.W)
        if encoder_window == "auto":
            encoder_window = "whole" if cfg["H"] < 1024 else "blocks"
        self.encoder_window = encoder_window
        self.enc_nh = cfg["enc_nh"]
        self.enc_pos = w.enc_pos[:CHUNK_TOKENS].to(self.device, self.dtype)
        self.EW = w.EW

        options = _compile_options() if self.compiled else {}
        cc = (lambda f: torch.compile(_own(f), dynamic=True, fullgraph=True, options=options)
              if self.compiled else f)
        self._pre, self._post, self._head = cc(dec_pre), cc(dec_post), cc(dec_head)
        self._enc_layer = cc(enc_layer)

        # ---- RoPE tables (all three MRoPE axes carry the same position for text+audio -> plain RoPE)
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        fr = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
        emb = torch.cat([fr, fr], -1)
        self.cos = emb.cos().to(self.device, self.dtype)
        self.sin = emb.sin().to(self.device, self.dtype)

        # ---- paged KV pool: `page` tokens per page, kv_tokens tokens in all
        self.page = page
        self.max_seq = max_seq
        self.max_pages = max_seq // page
        self.kv_tokens = kv_tokens
        self.P = kv_tokens // page + 1                      # + dummy page 0
        self.K = torch.zeros(self.L, self.P, page, self.nkv, self.hd, device=self.device, dtype=self.dtype)
        self.V = torch.zeros_like(self.K)
        self.Kf = self.K.view(self.L, self.P * page, self.nkv, self.hd)
        self.Vf = self.V.view(self.L, self.P * page, self.nkv, self.hd)
        self.kv_bytes = 2 * self.K.numel() * self.K.element_size()

        # ---- attention over the paged pool (the kinds are described in _attn.py)
        self.attn = _attn.make(kind, page=page, nh=self.nh, nkv=self.nkv, hd=self.hd, max_seq=max_seq,
                               device=self.device, dtype=self.dtype)
        self.attn_kind = self.attn.kind
        self.kv_step = self.attn_kind == "torch"   # the gather path needs a step graph per key-length bucket

        # ---- mel
        self.mel_filters = w.mel_filters
        self.window = torch.hann_window(NFFT, device=self.device)

        self.audio_id = cfg["audio_id"]
        self.eos = set(w.eos)

        # ---- decode-step CUDA graphs: static buffers shared by all batch buckets (views [:B])
        self.max_batch = max_batch
        self.buckets = sorted(b for b in batch_buckets if b <= max_batch)
        if not self.buckets or self.buckets[-1] != max_batch:
            self.buckets.append(max_batch)
        dev = self.device
        self.g_tok = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.g_pos = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.g_act = torch.zeros(max_batch, dtype=torch.long, device=dev)     # 1 = live row
        self.g_bt = torch.zeros(max_batch, self.max_pages, dtype=torch.int32, device=dev)
        self.g_out = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.g_cu = torch.arange(max_batch + 1, dtype=torch.int32, device=dev)
        self.use_graph = cuda_graph
        self.graphs = {}
        self.pf_buckets = sorted(prefill_buckets) if (cuda_graph and prefill_graph and self.attn.graph_prefill) else []
        self.enc_buckets = sorted(enc_buckets) if (cuda_graph and enc_graph) else []
        # whole-window encoder graphs: (groups, width) for widths of 2..4 blocks (one block is an 8 s graph above),
        # at most as many tokens as the largest 8 s graph
        top = self.enc_buckets[-1] * BLOCK_TOKENS if self.enc_buckets else 0
        self.whole_buckets = ([(g, k * BLOCK_TOKENS) for k in (2, 3, 4) for g in self.enc_buckets
                               if g * k * BLOCK_TOKENS <= top] if encoder_window == "whole" else [])
        self.pf, self.eg = {}, {}
        self.enc_d = cfg["enc_d"]
        if cuda_graph:
            self.g_pool = torch.cuda.graph_pool_handle()

    # ------------------------------------------------------------ decoder
    def forward(self, x, pos, slots, cu_q, seqused, bt, max_q: int, max_k: int):
        """x [T,H]: rows of all sequences concatenated (each sequence's rows are its LAST positions).
        pos [T] position inside its sequence; slots [T] flat KV slot (page*self.page+offset);
        cu_q [B+1] row offsets; seqused [B] sequence length after this write; bt [B, pages].
        Writes K/V at slots, then each row attends causally to its own sequence. Returns hidden [T,H]."""
        cos, sin = self.cos[pos], self.sin[pos]
        h = x
        T = x.shape[0]
        for i, (ln, qkv, qn, kn, o_w, ln2, gu, down) in enumerate(self.W):
            q, k, v = self._pre(h, ln, qkv, qn, kn, cos, sin, self.eps, self.nh, self.nkv, self.hd)
            self.Kf[i].index_copy_(0, slots, k)
            self.Vf[i].index_copy_(0, slots, v)
            o = self.attn(q, self.K[i], self.V[i], cu_q, seqused, bt, max_q, max_k, pos)
            h = self._post(h, o.reshape(T, self.nh * self.hd), o_w, ln2, gu, down, self.eps)
        return h

    def kv_bucket(self, n: int) -> int:
        return next((b for b in self.kv_buckets if b >= n), self.kv_buckets[-1])

    def greedy(self, h):
        """greedy token for each row of h, in slices (the logits are rows x vocabulary size)."""
        if h.shape[0] <= self.head_rows:
            return self._head(h, self.norm_w, self.eps, self.head_w)
        return torch.cat([self._head(h[i:i + self.head_rows], self.norm_w, self.eps, self.head_w)
                          for i in range(0, h.shape[0], self.head_rows)])

    def _step_body(self, B: int, S: int = 0):
        tok, pos = self.g_tok[:B], self.g_pos[:B]
        bt = self.g_bt[:B]
        page = bt.gather(1, torch.div(pos, self.page, rounding_mode="floor")[:, None]).squeeze(1).long()
        slots = page * self.page + pos % self.page
        seq = (pos + 1).to(torch.int32)
        h = self.forward(_quant.embedding(self.embed, tok, self.dtype), pos, slots, self.g_cu[:B + 1], seq, bt, 1, S or self.max_seq)
        out = self.greedy(h)
        self.g_out[:B].copy_(out)
        tok.copy_(out)                 # feed the token back on the GPU
        pos.add_(self.g_act[:B])       # live rows advance; padding rows stay on the dummy page

    def bucket(self, n: int) -> int:
        return next(b for b in self.buckets if b >= n)

    def capture(self, B: int, S: int = 0):
        """Capture the one-token step for batch bucket B (and, with PyTorch attention, key-length bucket S).
        Rows must all be idle (dummy page) here."""
        assert int(self.g_act.sum()) == 0
        self.g_tok.zero_(); self.g_pos.zero_(); self.g_bt.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step_body(B, S)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.g_pool):
            self._step_body(B, S)
        self.g_tok.zero_(); self.g_pos.zero_()
        self.graphs[(B, S) if S else B] = g

    def step(self, B: int, S: int = 0):
        """One decode step for rows [0, B). Result in g_out[:B]; g_tok/g_pos already advanced."""
        if self.use_graph:
            self.graphs[(B, S) if S else B].replay()
        else:
            self._step_body(B, S)

    # ------------------------------------------------------------ audio front end
    def mel_raw(self, wav: torch.Tensor):
        """log10 mel [128, n//HOP] as WhisperFeatureExtractor computes it, before the dynamic-range clamp."""
        pad = NFFT // 2
        x = F.pad(wav[None, None], (pad, pad), mode="reflect")[0, 0]
        spec = torch.stft(x, NFFT, HOP, window=self.window, center=False, return_complex=True)
        return torch.clamp(self.mel_filters @ (spec.abs() ** 2), min=1e-10).log10()[:, : wav.shape[0] // HOP]

    def _graph(self, body):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.g_pool):
            body()
        return g

    # ------------------------------------------------------------ prefill (CUDA graph per token bucket)
    # A graph has a fixed shape: Tb token rows and Bg sequences. A pass with T real tokens in B sequences
    # fills the rest with dummy sequences of 1..page tokens that live on the dummy page 0, so every graph
    # row has at least one token and nothing real is touched by the padding.
    def _pf_rows(self, Tb: int) -> int:
        return min(self.max_batch, Tb // 8) + Tb // self.page

    def _pf_fill(self, Tb, pos, slots, cu, seq, bt):
        b = self.pf[Tb]
        T, B, Bg = len(pos), len(seq), b["Bg"]
        nd, pad = Bg - B, Tb - T
        lens = np.ones(nd, dtype=np.int64)
        extra = pad - nd
        full, rem = divmod(extra, self.page - 1)
        lens[:full] += self.page - 1
        if rem:
            lens[full] += rem
        starts = np.cumsum(lens) - lens
        dpos = np.arange(pad) - np.repeat(starts, lens)
        btf = np.zeros((Bg, self.max_pages), dtype=np.int32)
        if B:
            btf[:B, :bt.shape[1]] = bt
        f = torch.from_numpy
        b["pos"].copy_(f(np.concatenate([pos, dpos])))
        b["slots"].copy_(f(np.concatenate([slots, dpos])))
        b["cu"].copy_(f(np.concatenate([cu, T + np.cumsum(lens)]).astype(np.int32)))
        b["seq"].copy_(f(np.concatenate([seq, lens]).astype(np.int32)))
        b["bt"].copy_(f(btf))

    def capture_prefill(self, Tb: int):
        dev, Bg = self.device, self._pf_rows(Tb)
        b = dict(Bg=Bg, x=torch.zeros(Tb, self.H, device=dev, dtype=self.dtype),
                 pos=torch.zeros(Tb, dtype=torch.long, device=dev), slots=torch.zeros(Tb, dtype=torch.long, device=dev),
                 cu=torch.zeros(Bg + 1, dtype=torch.int32, device=dev), seq=torch.ones(Bg, dtype=torch.int32, device=dev),
                 bt=torch.zeros(Bg, self.max_pages, dtype=torch.int32, device=dev),
                 h=torch.zeros(Tb, self.H, device=dev, dtype=self.dtype))
        self.pf[Tb] = b
        e = np.zeros(0, dtype=np.int64)
        self._pf_fill(Tb, e, e, np.zeros(1, dtype=np.int64), e, None)
        mq = min(Tb, self.max_seq)

        def body():
            b["h"].copy_(self.forward(b["x"], b["pos"], b["slots"], b["cu"], b["seq"], b["bt"], mq, self.max_seq))
        b["g"] = self._graph(body)

    def prefill(self, ids, arows, audio, pos, slots, cu, seq, bt):
        """One prefill pass. ids/pos/slots [T] and cu [B+1], seq [B], bt [B, pages] are numpy; `audio` is the
        [sum N, H] tensor that goes to rows `arows`. Returns hidden [>=T, H] (valid until the next pass)."""
        T, B, dev = len(ids), len(seq), self.device
        tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)  # noqa: E731
        def fits(t):                            # dummy rows need 1..page tokens each
            nd, pad = self._pf_rows(t) - B, t - T
            return nd >= 0 and nd <= pad <= self.page * nd
        Tb = next((t for t in self.pf_buckets if fits(t)), None)
        if Tb is None:                          # larger than any graph: plain eager pass
            x = _quant.embedding(self.embed, tt(ids, torch.long), self.dtype)
            x.index_copy_(0, tt(arows, torch.long), audio)
            return self.forward(x, tt(pos, torch.long), tt(slots, torch.long), tt(cu, torch.int32),
                                tt(seq, torch.int32), tt(bt, torch.int32), int(np.diff(cu).max()), bt.shape[1] * self.page)
        b = self.pf[Tb]
        self._pf_fill(Tb, pos, slots, cu, seq, bt)
        b["x"][:T] = _quant.embedding(self.embed, tt(ids, torch.long), self.dtype)
        b["x"].index_copy_(0, tt(arows, torch.long), audio)
        b["g"].replay()
        return b["h"]

    # ------------------------------------------------------------ audio encoder
    def _enc_body(self, x, mask):
        for w in self.EW:
            x = self._enc_layer(x, mask, *w, nh=self.enc_nh)
        w = self.w
        x = F.layer_norm(x, (x.shape[-1],), w.ln_post[0], w.ln_post[1])
        return _quant.linear_bias(F.gelu(_quant.linear_bias(x, *w.proj1)), *w.proj2)

    def _stem(self, y):
        """conv stem + linear for [n, 1, 128, frames] chunks -> [n, tokens, d] (positions not added)."""
        for cw, cb in self.w.conv:
            y = F.gelu(F.conv2d(y, cw, cb, stride=2, padding=1))
        b_, c_, f_, t_ = y.shape
        return _quant.linear_bias(y.permute(0, 3, 1, 2).reshape(b_, t_, c_ * f_), self.w.conv_out_w, None)

    def capture_enc(self, Gb: int, W: int = BLOCK_TOKENS):
        dev = self.device
        b = dict(x=torch.zeros(Gb, W, self.enc_d, device=dev, dtype=self.dtype),
                 mask=torch.ones(Gb, 1, 1, W, dtype=torch.bool, device=dev),
                 out=torch.zeros(Gb, W, self.H, device=dev, dtype=self.dtype))
        self.eg[(Gb, W)] = b

        def body():
            b["out"].copy_(self._enc_body(b["x"], b["mask"]))
        b["g"] = self._graph(body)

    def _attend(self, xflat, items, W: int):
        """Encoder layers over attention groups of W tokens. items: (first conv chunk, tokens) per request;
        each request is cut into groups of W tokens (the last one masked). Returns [sum tokens, H] in the order
        of items. One CUDA graph per (group-count bucket, W); padding groups are zeros that attend to
        themselves."""
        src, msk, take, goff = [], [], [], 0
        for coff, N in items:
            G = -(-N // W)
            idx = np.zeros(G * W, dtype=np.int64)
            idx[:N] = coff * CHUNK_TOKENS + np.arange(N)       # token rows of this request in the conv output
            m = np.zeros(G * W, dtype=bool)
            m[:N] = True
            src.append(idx); msk.append(m)
            take.append(goff * W + np.arange(N))
            goff += G
        d, dev = xflat.shape[-1], self.device
        src_t = torch.from_numpy(np.concatenate(src)).to(dev)
        msk_t = torch.from_numpy(np.concatenate(msk)).to(dev).view(goff, 1, 1, W)
        take_t = torch.from_numpy(np.concatenate(take)).to(dev)
        xg = xflat[src_t].view(goff, W, d)
        if W == BLOCK_TOKENS:
            Gb = next((g for g in self.enc_buckets if g >= goff), None)
        else:
            Gb = next((g for g, w in self.whole_buckets if w == W and g >= goff), None)
        if Gb is None:
            out = self._enc_body(xg, msk_t)
        else:
            b = self.eg[(Gb, W)]
            b["x"][:goff] = xg
            b["mask"][:goff] = msk_t
            if goff < Gb:
                b["mask"][goff:] = True
            b["g"].replay()
            out = b["out"]
        return out.reshape(-1, self.H)[take_t]

    def encode(self, feats: list, whole=None):
        """feats: list of [128, n_i] normalised mel. Returns ([sum N_i, H] audio embeddings in request order,
        [N_i]). All 1 s conv chunks of all requests go through the conv stem together. Attention: requests
        flagged in `whole` (when encoder_window is "whole" and the window is at most 32 s) attend over their
        whole window, as one group each; the others in 104-token (8 s) groups. Each kind of group goes through
        the encoder layers as one batch."""
        chunks, ns, items, coff = [], [], [], 0
        short = []                      # (chunk row, features) of windows shorter than one chunk
        for f in feats:
            n = f.shape[1]
            nch = -(-n // CHUNK_FRAMES)
            if nch == 1 and n < CHUNK_FRAMES and not self.pad_short:
                short.append((coff, f))
            chunks.append(F.pad(f, (0, nch * CHUNK_FRAMES - n)).view(128, nch, CHUNK_FRAMES).permute(1, 0, 2))
            N = audio_tokens(n)
            items.append((coff, N))
            ns.append(N)
            coff += nch
        x = torch.cat(chunks)[:, None].to(self.dtype)
        outs = []
        for i in range(0, x.shape[0], self.conv_batch):
            outs.append(self._stem(x[i:i + self.conv_batch]) + self.enc_pos)
        x = torch.cat(outs) if len(outs) > 1 else outs[0]          # [chunks, 13, d]
        for row, f in short:
            # The reference pads a chunk only up to the longest chunk of the same audio, so a window under
            # 1 s is convolved at its own width. Padding it to 100 frames would change its last token.
            y = self._stem(f[None, None].to(self.dtype))[0]
            t_ = y.shape[0]
            x[row, :t_] = y + self.enc_pos[:t_]
        xflat = x.reshape(-1, x.shape[-1])
        use = [self.encoder_window == "whole" and bool(whole is not None and whole[i]) and N <= WHOLE_TOKENS
               for i, (_, N) in enumerate(items)]
        if not any(use):
            return self._attend(xflat, items, BLOCK_TOKENS), ns
        off = np.concatenate([[0], np.cumsum(ns)]).astype(np.int64)
        out = torch.empty(int(off[-1]), self.H, device=self.device, dtype=self.dtype)
        for flag in (False, True):
            sel = [i for i in range(len(items)) if use[i] == flag]
            if not sel:
                continue
            W = BLOCK_TOKENS if not flag else BLOCK_TOKENS * max(-(-items[i][1] // BLOCK_TOKENS) for i in sel)
            rows = torch.from_numpy(np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])).to(self.device)
            out.index_copy_(0, rows, self._attend(xflat, [items[i] for i in sel], W))
        return out, ns
