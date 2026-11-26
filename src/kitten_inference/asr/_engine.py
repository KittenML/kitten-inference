"""Batching scheduler of the ASR engine: many threads call decode(), one thread runs them together on the GPU."""
# The scheduler thread owns the GPU, the key/value pool and every CUDA graph. Its loop:
#   1. take new requests (lowest priority value first) and check each stream's cached 8 s audio blocks,
#   2. one batched encoder pass and one batched prefill pass over all of them. The prefill also carries each
#      request's draft (its previous result): the model's greedy choice at every draft slot comes out of the
#      same pass, so the longest agreeing prefix is kept without decoding it again (exact greedy decoding),
#   3. one batched decode step for every live sequence (a CUDA graph per batch-size bucket),
# and repeat. Sequences join and leave between steps.
from __future__ import annotations

import gc
import heapq
import operator
import queue
import sys
import threading
import time
import traceback
import warnings
from typing import Callable

from . import _deps, _system
from ._types import DecodeResult, EngineStopped, PoolExhausted

try:
    import numpy as np
    import torch

    from ._core import BLOCK_FRAMES, BLOCK_TOKENS, HOP, SR, Core, audio_tokens
except ImportError:
    _deps.require_torch()       # the clear message, when the reason is the Python version or a missing PyTorch
    raise

FLOOR_TOL = 1e-4      # two clamp floors closer than this give the same bf16 features
BACKGROUND = 10       # requests with a priority from here up are background work
_CLOSED = "kitten-asr: the engine is closed"
_STOPPED = "kitten-asr: the scheduler thread has stopped"


class _Stream:
    __slots__ = ("key", "pages", "n_blocks", "head", "prefix", "bmin", "floor", "busy", "last", "dead")

    def __init__(self, key):
        self.key, self.pages, self.n_blocks, self.head, self.prefix = key, [], 0, None, None
        self.bmin, self.floor, self.busy, self.last, self.dead = [], [], False, time.monotonic(), False


class _Req:
    def __init__(self, audio, head, tail, draft, key, max_new, priority, on_tokens):
        self.audio, self.head, self.tail, self.draft = audio, head, tail, draft
        self.key, self.max_new, self.priority, self.on_tokens = key, max_new, priority, on_tokens
        self.t_submit = time.perf_counter()
        self.event = threading.Event()
        self.cancelled = False
        self.error = None
        self.gen = []
        self.info = {}
        self.pages = []
        self.stream = None
        self.row = -1
        self.rem = []          # rejected rest of the draft (resync candidates)
        self.n_resync = 0
        self.cand = None
        self.abort = False     # set from outside (engine process) to stop this request at the next step
        self.took_pages = False   # this request holds its stream's cached pages
        self.settled = False
        nfr = max(len(audio), SR // 2) // HOP
        self.n_audio_full = audio_tokens(nfr)
        self.est = len(head) + self.n_audio_full + len(tail) + len(draft)


def _whole(name, value, low):
    """`value` as an int, ValueError unless it is a whole number of `low` or more."""
    try:
        number = None if isinstance(value, bool) else operator.index(value)
    except TypeError:
        number = None
    if number is None or number < low:
        raise ValueError(f"{name} must be a whole number of {low} or more, got {value!r}")
    return number


class Engine:
    """Speech-to-text engine on one NVIDIA GPU. decode() can be called from many threads at once; one
    scheduler thread batches the calls. The constructor returns after the model is loaded, the CUDA graphs
    are captured and the warm-up decodes have run. Close it with close() or use it as a context manager.

    model                local model folder or Hugging Face repo id
    device               CUDA device, "cuda" or "cuda:N"
    kv_tokens            size of the key/value pool in tokens, fixed at start, at least max_seq. None =
                         24576, or two pages per decode row when that is more
    max_batch            decode rows: sequences that generate at the same time. More streams than this can be
                         open; a row is held only while a request decodes
    max_seq              longest sequence in tokens (prompt + audio + text)
    prefill_cap          most tokens (prompt, audio and draft) admitted in one pass; one larger request is
                         still admitted on its own
    reserve_rows         decode rows that background requests leave free for interactive ones; at most
                         max_batch - 1 is used, so that background work can always run
    max_background_rows  most decode rows that background requests hold at once; at most max_batch is used
    page_ahead           16-token pages (or as many tokens in larger pages) taken ahead of need for every
                         decoding sequence
    stream_ttl_s         idle seconds after which the cache of an unreleased stream_key is dropped
    progress_every       generated tokens between two on_tokens calls
    stream_cache         keep the complete 8 s audio blocks of a stream_key between calls
    resync               when a draft is rejected, find where the new output meets the rest of the draft and
                         check that rest in one pass
    resync_min           fewest remaining draft tokens worth such a pass
    resync_window        how far into the rejected draft to look for the meeting point
    resync_max           most such passes per request
    fuse                 let decoding rows ride in a prefill pass instead of running a separate step
    pass_gap_ms          shortest time between prefill passes while rows are decoding; 0 = none
    warmup               run a few short decodes at start; the counters are reset afterwards
    profile              per-phase timers in snapshot()["phase_s"]; they wait for the GPU, so they slow it down
    system_prompt        "auto": the system message the checkpoint names in asr_config.json (empty
                         when the folder has no such file); "none": always empty. See _system.py
    core_opts            model options: precision ("bf16", "fp16", "int8", "int4"; dtype
                         is the older name for "bf16" / "fp16"), encoder_window ("auto", "blocks", "whole"),
                         attention ("auto" = "triton" when it can run here, else "torch"; "flash" only by
                         name), compile ("auto" = torch.compile when Triton can run here, else off; True or
                         False), page ("auto" = 16, or tokens per page), cache_dir (where a repo id is
                         downloaded to), cuda_graph, prefill_graph, enc_graph, batch_buckets,
                         prefill_buckets, enc_buckets, kv_buckets, conv_batch, head_rows, pad_short

    Requests with priority >= 10 are background work: they only take decode rows that interactive requests
    (priority below 10) are not using, as set by reserve_rows and max_background_rows. snapshot() reports
    the two values in use.

    Full speed needs a C compiler and the Python development headers. Without them "auto" runs plain
    PyTorch attention without torch.compile, and the constructor gives one RuntimeWarning that says why.
    snapshot() reports the choice as "attention" and "compiled".
    """

    name = "kitten-asr"

    def __init__(self, model, device: str = "cuda", kv_tokens: int | None = None, max_batch: int = 64,
                 max_seq: int = 2048, prefill_cap: int = 6144, reserve_rows: int = 8,
                 max_background_rows: int = 16, page_ahead: int = 4, stream_ttl_s: float = 120.0,
                 progress_every: int = 16, stream_cache: bool = True, warmup: bool = True, profile: bool = False,
                 resync: bool = True, resync_min: int = 3, resync_window: int = 48, resync_max: int = 6,
                 pass_gap_ms: float = 0.0, fuse: bool = True, system_prompt: str = "auto", **core_opts):
        self._system_mode = _system.check_mode(system_prompt)
        self._system, self.system_prompt_source = "", None
        _deps.require("asr", device=device)
        max_batch = _whole("max_batch", max_batch, 1)
        reserve_rows = _whole("reserve_rows", reserve_rows, 0)
        max_background_rows = _whole("max_background_rows", max_background_rows, 1)
        self.opts = dict(model=model, device=device, kv_tokens=kv_tokens, max_batch=max_batch, max_seq=max_seq,
                         **core_opts)
        self.prefill_cap = prefill_cap
        # A background request is admitted only while more than reserve_rows rows are free. With a reserve of
        # max_batch or more that is never, and it would wait until its timeout: keep one row for it.
        self.reserve_rows = min(reserve_rows, max_batch - 1)
        self.max_background_rows = min(max_background_rows, max_batch)
        self.page_ahead, self.stream_ttl_s, self.progress_every = page_ahead, stream_ttl_s, progress_every
        self.stream_cache = stream_cache
        self.profile = profile
        self.resync, self.resync_min = resync, resync_min
        self.resync_window, self.resync_max = resync_window, resync_max
        self.pass_gap = pass_gap_ms / 1e3
        self.resume = []
        self.fuse = fuse
        self.t_pass = 0.0
        self.phase = {}
        self.torch = torch
        self.core = None
        self.pending, self.rows, self.streams, self.free_pages, self.free_rows = [], [], {}, [], []
        self.n_live, self.top, self.n_bg = 0, 0, 0
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._tok_lock = threading.Lock()
        self._free_lock = threading.Lock()
        self._prompts: dict = {}
        self._ready = threading.Event()
        self._stop = False
        self._closed = False
        self._err = None
        self._err_tb = None       # traceback of _err as the scheduler thread left it
        self._freed = False
        self._seq = 0
        self.stats = dict(requests=0, errors=0, cancelled=0, prefill_passes=0, prefill_tokens=0, prefill_reqs=0,
                          steps=0, step_rows=0, draft_tokens=0, accepted_tokens=0, generated_tokens=0,
                          cached_tokens=0, cache_resets=0, evictions=0, max_rows=0, pages_peak=0, resyncs=0,
                          fused_steps=0, kv_exhausted=0, parked=0, resync_tokens=0, resync_accepted=0)
        self._thread = threading.Thread(target=self._run, name="kitten-asr-scheduler", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._err is not None:
            self._closed = True
            self._thread.join(timeout=10)
            if not self._thread.is_alive():
                self._free()
            raise self._err
        core = self.core
        if core is not None and core.fallback:       # "auto" could not take the fast path: say so once
            warnings.warn(core.fallback, RuntimeWarning, stacklevel=2)
        if warmup:
            try:
                self._warmup()
            except BaseException:
                self.close()
                raise

    # ------------------------------------------------------------ public API (any thread)
    def _check_running(self):
        if self._closed:
            raise EngineStopped(_CLOSED)
        if self._err is not None:
            raise EngineStopped(_STOPPED) from self._err

    def default_prompt(self, language: str | None = None) -> str:
        """The prompt text for decode(): the model's chat prompt with one audio placeholder, and with the
        language fixed when `language` (a language name such as "English") is given."""
        self._check_running()
        with self._tok_lock:
            return self._prompt(self._system, language)

    def _split(self, prompt: str):
        hit = self._prompts.get(prompt)
        if hit is None:
            with self._tok_lock:
                ids = self._tok.encode(prompt)
            if ids.count(self._audio_id) != 1:
                raise ValueError("prompt must contain exactly one audio placeholder")
            k = ids.index(self._audio_id)
            hit = (tuple(ids[:k]), tuple(ids[k + 1:]))
            if len(self._prompts) < 4096:
                self._prompts[prompt] = hit
        return hit

    def _text(self, ids) -> str:
        with self._tok_lock:
            return self._tok.decode(ids)

    def decode(self, audio: np.ndarray, prompt: str, *, draft_ids: list | None = None,
               stream_key: str | None = None, max_new_tokens: int = 256, priority: int = 0,
               timeout_s: float = 30.0, on_tokens: Callable[[list], bool] | None = None) -> DecodeResult:
        """Transcribes one audio window with greedy decoding and returns a DecodeResult.

        audio           float32 mono 16 kHz, the whole window. Shorter than 0.5 s is zero-padded to 0.5 s.
        prompt          full prompt text with exactly one audio placeholder (see default_prompt), else ValueError
        draft_ids       token ids of an earlier result for the same window. The longest prefix that greedy
                        decoding agrees with is kept, and decoding goes on from the first disagreement.
        stream_key      one key per audio window that grows between calls. Complete 8 s blocks and their
                        decoder state are kept for the next call with the same key; the audio prefix is
                        checked, so a changed window only resets the cache. One request per key runs at a
                        time, a second one waits. Call release(stream_key) when the window ends.
        max_new_tokens  most tokens to generate, cut to what still fits in max_seq; ValueError if nothing fits
        priority        lower runs first; 10 and above is background work (see the class text)
        timeout_s       TimeoutError after this many seconds; the request is dropped at the next step
        on_tokens       on_tokens(token_ids) is called on the scheduler thread every `progress_every`
                        generated tokens; return True to stop the request (finish_reason "abort"). Keep it
                        fast: it blocks every stream.

        info["finish_reason"] is "stop", "length" or "abort". Raises PoolExhausted when the key/value pool
        has no room for the request, and EngineStopped when the engine is closed or its scheduler stopped.
        """
        self._check_running()
        head, tail = self._split(prompt)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        eos = self._eos
        draft = [t for t in (draft_ids or []) if t not in eos]
        r = _Req(audio, head, tail, draft, stream_key if self.stream_cache else None, max_new_tokens, priority,
                 on_tokens)
        self._check_running()
        self._inbox.put(r)
        deadline = time.monotonic() + timeout_s
        while not r.event.wait(min(0.5, max(0.0, deadline - time.monotonic()))):
            if self._err is not None or self._closed:     # the scheduler stopped after this request was queued
                self._check_running()
            if time.monotonic() >= deadline:
                r.cancelled = True
                raise TimeoutError(f"kitten-asr: decode timed out after {timeout_s}s")
        if r.error is not None:
            raise r.error
        return DecodeResult(self._text(r.gen), r.gen, r.info)

    def release(self, stream_key: str) -> None:
        """Drops what is cached for `stream_key`. Does nothing once the engine has stopped."""
        if self._closed or self._err is not None:
            return
        self._inbox.put(("release", stream_key))

    def snapshot(self) -> dict:
        """Counters and settings of the running engine, for monitoring."""
        c = self.core
        if c is None:
            self._check_running()
            raise EngineStopped(_STOPPED)
        torch = self.torch
        s = dict(self.stats)
        s.update(active_rows=self.n_live, pending=len(self.pending), streams=len(self.streams),
                 kv_tokens=c.kv_tokens, kv_pages_total=c.P - 1, kv_pages_free=len(self.free_pages),
                 kv_pool_bytes=c.kv_bytes, gpu_allocated_bytes=torch.cuda.memory_allocated(),
                 gpu_reserved_bytes=torch.cuda.memory_reserved(), graph_buckets=sorted(str(k) for k in c.graphs),
                 attention=c.attn_kind, compiled=c.compiled, page=c.page, precision=c.precision,
                 encoder_window=c.encoder_window, weight_bytes=c.weight_bytes,
                 options={k: v for k, v in self.opts.items()}, prefill_cap=self.prefill_cap,
                 reserve_rows=self.reserve_rows, max_background_rows=self.max_background_rows,
                 stream_cache=self.stream_cache,
                 resync=self.resync, pass_gap_ms=self.pass_gap * 1e3, fuse=self.fuse,
                 system_prompt=self.system_prompt_source, system_prompt_chars=len(self._system),
                 phase_s=dict(self.phase))
        return s

    def close(self) -> None:
        """Stops the scheduler thread and frees the GPU memory. Requests still waiting fail with
        EngineStopped, and so does every later decode(), snapshot() and default_prompt(). Safe to call again."""
        self._closed = True
        self._stop = True
        self._inbox.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)
        if not self._thread.is_alive():
            self._free()        # done by the thread itself after a normal stop; this covers one that had crashed

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _warmup(self):
        rng = np.random.default_rng(0)
        p = self.default_prompt()
        for sec in (1.0, 9.5, 3.0):
            a = (rng.standard_normal(int(sec * SR)) * 0.01).astype(np.float32)
            r = self.decode(a, p, max_new_tokens=8, timeout_s=600)
            self.decode(a, p, draft_ids=r.token_ids + [1, 2, 3], max_new_tokens=8, timeout_s=600)
        for k in self.stats:
            self.stats[k] = 0
        self.phase.clear()

    def _free(self):
        """Drops the model, the key/value pool and the CUDA graphs, then gives the cached GPU memory back."""
        with self._free_lock:
            if self._freed:
                return
            self._freed = True
            self.core = None
            self.pending, self.resume, self.rows, self.streams = [], [], [], {}
            # After a crash or a failed start the error's traceback still holds the scheduler thread's
            # frames, and their local variables hold the model (also one that was only half built). Clear
            # those frames, or the memory stays in use for as long as the error is kept. Frames that are
            # still running are skipped, and frames of other threads are not in this part of the traceback.
            err, tb = self._err, self._err_tb
            seen = set()
            while err is not None and id(err) not in seen:
                seen.add(id(err))
                traceback.clear_frames(tb)
                err = err.__cause__ or err.__context__      # errors it was raised from, same thread
                tb = getattr(err, "__traceback__", None)
            self._err_tb = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ scheduler thread
    def _run(self):
        try:
            with torch.inference_mode():
                self._start()
                self._ready.set()
                self._loop()
        except BaseException as e:  # noqa: BLE001
            self._err, self._err_tb = e, e.__traceback__
            raise
        finally:
            self._ready.set()
            self._fail_waiting(self._err)
            if self._stop:
                self._free()

    def _start(self):
        """Loads the model and captures every CUDA graph."""
        c = self.core = Core(**self.opts)
        # what decode() needs on the callers' threads; none of it lives on the GPU
        self._tok, self._eos, self._audio_id, self._prompt = c.tok, c.eos, c.audio_id, c.w.prompt
        if getattr(c.w, "folder", None):
            self._system, self.system_prompt_source = _system.resolve(c.w.folder, self._system_mode)
        else:
            self.system_prompt_source = "none" if self._system_mode == "none" else "auto:no-folder"
        # page_ahead is given in 16-token pages; keep the same token count for other page sizes
        self.page_ahead = self.page_ahead * 16 // c.page
        self.free_pages = list(range(c.P - 1, 0, -1))          # page 0 is the dummy
        self.free_rows = list(range(c.max_batch))               # min-heap: low rows first
        self.rows = [None] * c.max_batch
        self.n_live, self.top, self.n_bg = 0, 0, 0
        self.pending = []
        self.streams = {}
        if c.use_graph:
            try:
                self._capture_graphs(c)
            except RuntimeError as exc:
                if sys.platform == "win32":
                    raise RuntimeError(f"Kitten ASR could not capture CUDA graphs on Windows ({exc}). "
                                       "Try cuda_graph=False, or compile=False if the compiler failed.") from exc
                raise

    @staticmethod
    def _capture_graphs(c):
        for b in c.buckets:
            if not c.kv_step:
                c.capture(b)
            else:
                for kb in c.kv_buckets:      # PyTorch attention: one graph per key-length bucket too
                    c.capture(b, kb)
        for t in c.pf_buckets:
            c.capture_prefill(t)
        for g in c.enc_buckets:
            c.capture_enc(g)
        for g, w in c.whole_buckets:
            c.capture_enc(g, w)

    def _fail_waiting(self, err):
        """Fails every request that is queued or decoding: with `err` after a crash, with EngineStopped
        after close()."""
        waiting = [it[2] for it in self.pending] + list(self.resume) + [x for x in self.rows if x]
        while True:
            try:
                m = self._inbox.get_nowait()
            except queue.Empty:
                break
            if m is not None and not isinstance(m, tuple):
                waiting.append(m)
        for r in waiting:
            try:
                r.error = err if err is not None else EngineStopped(_CLOSED)
                r.event.set()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self):
        last_gc = time.monotonic()
        while not self._stop:
            block = self.n_live == 0 and not self.pending and not self.resume
            stepped = False
            while True:
                try:
                    m = self._inbox.get(timeout=0.05) if block else self._inbox.get_nowait()
                except queue.Empty:
                    break
                block = False
                if m is None:
                    return
                if isinstance(m, tuple):
                    self._release(m[1])
                else:
                    self._seq += 1
                    self.pending.append((m.priority, self._seq, m))
            if (self.pending or self.resume) and not (
                    self.n_live and self.pass_gap and time.perf_counter() - self.t_pass < self.pass_gap):
                picked = self._pick() if self.pending else []
                resume, self.resume = self.resume, []
                if picked or resume:
                    stepped = self.fuse and self.n_live > 0
                    try:
                        self._prefill(picked, resume)
                    except Exception as e:  # noqa: BLE001
                        # A request of this pass may already sit on a decode row: take it off the row and
                        # bring the GPU-side buffers back in line before failing it.
                        off = []
                        for r in picked + resume:
                            if 0 <= r.row < len(self.rows) and self.rows[r.row] is r:
                                self._leave_row(r.row, r, off)
                        self._post_step(off, ([], [], []))
                        for r in picked + resume:
                            if not r.event.is_set():
                                self._fail(r, e)
                    self.t_pass = time.perf_counter()
            if stepped:
                pass                              # the pass above already advanced every live row
            elif self.n_live:
                self._step()
            elif self.pending and not self.resume:
                time.sleep(0.001)          # waiting for a busy stream key or for pages
            now = time.monotonic()
            if now - last_gc > 5.0:
                last_gc = now
                for st in [s for s in self.streams.values() if not s.busy and now - s.last > self.stream_ttl_s]:
                    self._drop_stream(st)

    # ---- pages and streams
    def _alloc(self, n: int):
        if n > len(self.free_pages):
            idle = sorted((s for s in self.streams.values() if not s.busy and s.pages), key=lambda s: s.last)
            for st in idle:
                self._drop_stream(st)
                self.stats["evictions"] += 1
                if n <= len(self.free_pages):
                    break
            if n > len(self.free_pages):
                return None
        out = [self.free_pages.pop() for _ in range(n)]
        used = self.core.P - 1 - len(self.free_pages)
        if used > self.stats["pages_peak"]:
            self.stats["pages_peak"] = used
        return out

    def _drop_stream(self, st: _Stream):
        self.free_pages.extend(st.pages)
        st.pages = []
        self.streams.pop(st.key, None)

    def _release(self, key):
        st = self.streams.get(key)
        if st is None:
            return
        if st.busy:
            st.dead = True
        else:
            self._drop_stream(st)

    # ---- admission
    def _pick(self):
        self.pending.sort(key=lambda t: (t[0], t[1]))
        # Requests waiting for a resync pass gave their row back but need one again in the next pass.
        free_rows = max(0, len(self.free_rows) - len(self.resume))
        picked, rest, tokens = [], [], 0
        n_bg = self.n_bg + sum(1 for r in self.resume if r.priority >= BACKGROUND)
        avail = len(self.free_pages) + sum(len(s.pages) for s in self.streams.values() if not s.busy)
        avail -= (self.n_live + len(self.resume)) * self.page_ahead     # growth of rows already decoding
        stop = False
        for item in self.pending:
            r = item[2]
            if r.cancelled:
                self.stats["cancelled"] += 1
                continue
            take = not stop and len(picked) < free_rows
            st = self.streams.get(r.key) if r.key else None
            if take and st is not None and st.busy:
                take = False
            # background work only takes rows that interactive requests are not using
            if take and r.priority >= BACKGROUND and (free_rows - len(picked) <= self.reserve_rows
                                                      or n_bg >= self.max_background_rows):
                take = False
            need = -(-(r.est + 1) // self.core.page) + self.page_ahead
            if take and need > avail:
                if not picked and self.n_live == 0 and not any(s.busy for s in self.streams.values()):
                    self._fail(r, PoolExhausted("kitten-asr: KV pool too small for this request"))
                    continue
                take = False
            if take and picked and tokens + r.est > self.prefill_cap:
                take, stop = False, True
            if not take:
                rest.append(item)
                continue
            picked.append(r)
            tokens += r.est
            avail -= need
            if r.priority >= BACKGROUND:
                n_bg += 1
            if r.key:
                if st is None:
                    st = self.streams[r.key] = _Stream(r.key)
                st.busy = True
                r.stream = st
        self.pending = rest
        return picked

    # ---- encoder + prefill + checking the draft, for a batch of new requests
    def _prefill(self, reqs, resume=()):
        torch, c = self.torch, self.core
        dev = c.device
        t0 = time.perf_counter()
        tp = [t0]

        def tick(name):                      # phase timer (profile=True only: it syncs the GPU)
            if self.profile:
                torch.cuda.synchronize()
                now = time.perf_counter()
                self.phase[name] = self.phase.get(name, 0.0) + now - tp[0]
                tp[0] = now
        # mel for the whole window of every request; one host sync for the maxima and block minima
        scal = []
        for r in reqs:
            a = r.audio
            if len(a) < SR // 2:
                a = np.pad(a, (0, SR // 2 - len(a)))
            r.lm = c.mel_raw(torch.from_numpy(a).to(dev))
            r.nb_done = (r.lm.shape[1] - 1) // BLOCK_FRAMES if r.stream is not None else 0
            scal.append(r.lm.max())
            for k in range(r.nb_done):
                scal.append(r.lm[:, k * BLOCK_FRAMES:(k + 1) * BLOCK_FRAMES].min())
        vals = torch.stack(scal).float().tolist() if scal else []
        tick("mel")
        feats, vi = [], 0
        for r in reqs:
            floor = vals[vi] - 8.0
            r.floor = floor
            r.bmin = vals[vi + 1: vi + 1 + r.nb_done]
            vi += 1 + r.nb_done
            st, nb = r.stream, 0
            if st is not None and st.n_blocks:
                # the cached 8 s blocks are reusable only if they are exactly what a from-scratch pass over
                # this window would compute: same prompt head, same audio, same clamped features
                if (st.head == r.head and len(r.audio) >= len(st.prefix)
                        and np.array_equal(r.audio[:len(st.prefix)], st.prefix)):
                    nb = min(st.n_blocks, r.nb_done)
                    for j in range(nb):
                        same = abs(st.floor[j] - floor) <= FLOOR_TOL
                        if not (same or (st.bmin[j] >= floor and st.bmin[j] >= st.floor[j])):
                            nb = j
                            break
                if nb < st.n_blocks:
                    self.stats["cache_resets"] += 1
                    st.n_blocks, st.floor, st.bmin = nb, st.floor[:nb], st.bmin[:nb]
            r.nb = nb
            r.p0 = (len(r.head) + nb * BLOCK_TOKENS) if nb else 0
            if st is not None:
                keep = -(-r.p0 // self.core.page)
                self.free_pages.extend(st.pages[keep:])
                r.pages = st.pages[:keep]
                st.pages = []
                r.took_pages = True
            feats.append((torch.clamp(r.lm[:, nb * BLOCK_FRAMES:], min=floor) + 4.0) / 4.0)
            r.lm = None
        if feats:
            audio_all, ns = c.encode(feats, whole=[r.stream is None for r in reqs])
        else:
            audio_all, ns = torch.zeros(0, c.H, device=dev, dtype=c.dtype), []
        offs = np.concatenate([[0], np.cumsum(ns)]).astype(np.int64)
        keep_a = []
        tick("encode")

        # layout: [head ids if nothing cached] + new audio + tail ids + draft, at positions p0..
        ids_l, pos_l, slot_l, arow_l, hrow_l, cu, seqused, off = [], [], [], [], [], [0], [], 0
        live = []
        for ri, r in enumerate(reqs):
            e = ri
            pre = list(r.head) if r.nb == 0 else []
            n_a = int(ns[ri])
            end = r.p0 + len(pre) + n_a + len(r.tail)
            r.budget = min(r.max_new, c.max_seq - end - 1)
            if r.budget < 1:
                self._fail(r, ValueError(f"window too long for max_seq={c.max_seq}"))
                r.T = 0
                continue
            r.draft = r.draft[: max(0, r.budget - 1)]
            T = len(pre) + n_a + len(r.tail) + len(r.draft)
            need = -(-(r.p0 + T + 1) // self.core.page) + self.page_ahead - len(r.pages)
            got = self._alloc(max(0, min(need, c.max_pages - len(r.pages))))
            if got is None:
                self._fail(r, PoolExhausted("kitten-asr: KV pool exhausted"))
                r.T = 0
                continue
            r.pages.extend(got)
            r.T, r.end, r.n_a = T, end, n_a
            pos = np.arange(r.p0, r.p0 + T)
            pg = np.asarray(r.pages, dtype=np.int64)
            ids = np.zeros(T, dtype=np.int64)
            ids[:len(pre)] = pre
            ids[len(pre) + n_a:] = list(r.tail) + r.draft
            ids_l.append(ids)
            pos_l.append(pos)
            slot_l.append(pg[pos // self.core.page] * self.core.page + pos % self.core.page)
            arow_l.append(np.arange(off + len(pre), off + len(pre) + n_a))
            hrow_l.append(np.arange(off + T - len(r.draft) - 1, off + T))
            off += T
            cu.append(off)
            seqused.append(r.p0 + T)
            live.append((r, e))
            keep_a.append(np.arange(offs[ri], offs[ri + 1]))
        for r in resume:
            # resync: the token just generated (already in r.gen, not yet fed) + the rest of the old draft
            # that we expect to follow it. Every row is checked, exactly like the first verify.
            if r.cancelled:
                self._finish(r, "cancelled")
                continue
            cand = r.cand[: max(0, r.budget - len(r.gen) - 1)]
            T = 1 + len(cand)
            need = -(-(r.pos + T + 1) // self.core.page) - len(r.pages)
            got = self._alloc(max(0, min(need, c.max_pages - len(r.pages))))
            if got is None:
                self.stats["kv_exhausted"] += 1
                self._fail(r, PoolExhausted("kitten-asr: KV pool exhausted"))
                continue
            r.pages.extend(got)
            r.cand, r.T = cand, T
            pos = np.arange(r.pos, r.pos + T)
            pg = np.asarray(r.pages, dtype=np.int64)
            ids_l.append(np.asarray([r.gen[-1]] + cand, dtype=np.int64))
            pos_l.append(pos)
            slot_l.append(pg[pos // self.core.page] * self.core.page + pos % self.core.page)
            arow_l.append(np.zeros(0, dtype=np.int64))
            hrow_l.append(np.arange(off, off + T))
            off += T
            cu.append(off)
            seqused.append(r.pos + T)
            live.append((r, None))
        if self.fuse and self.n_live and live:
            # fuse: the live decode rows ride in this pass as one-token sequences, so the iteration reads
            # the decoder weights once instead of once for the pass and once for a separate step
            for row in range(self.top):
                r = self.rows[row]
                if r is None:
                    continue
                ids_l.append(np.asarray([r.gen[-1]], dtype=np.int64))
                pos_l.append(np.asarray([r.pos], dtype=np.int64))
                slot_l.append(np.asarray([r.pages[r.pos // self.core.page] * self.core.page + r.pos % self.core.page], dtype=np.int64))
                arow_l.append(np.zeros(0, dtype=np.int64))
                hrow_l.append(np.asarray([off], dtype=np.int64))
                off += 1
                cu.append(off)
                seqused.append(r.pos + 1)
                live.append((r, "step"))
        if not live:
            return
        B = len(live)
        maxpg = max(len(r.pages) for r, _ in live)
        bt = np.zeros((B, maxpg), dtype=np.int32)
        for i, (r, _) in enumerate(live):
            bt[i, :len(r.pages)] = r.pages
        tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)  # noqa: E731
        tick("layout")
        if len(keep_a) != len(reqs):             # a request was dropped: keep only the live audio rows
            audio_all = audio_all[tt(np.concatenate(keep_a), torch.long)] if keep_a else audio_all[:0]
        h = c.prefill(np.concatenate(ids_l), np.concatenate(arow_l), audio_all, np.concatenate(pos_l),
                      np.concatenate(slot_l), np.asarray(cu, dtype=np.int64), np.asarray(seqused, dtype=np.int64), bt)
        tick("forward")
        preds = c.greedy(h[tt(np.concatenate(hrow_l), torch.long)]).tolist()        # host sync
        tick("head")
        t1 = time.perf_counter()
        self.stats["prefill_passes"] += 1
        self.stats["prefill_tokens"] += off
        self.stats["prefill_reqs"] += B

        # check the drafts + stream bookkeeping + hand over to the decode rows
        act_rows, act_tok, act_pos, act_bt, pi = [], [], [], [], 0
        off_rows, pg, upd = [], ([], [], []), []
        for r, tag in live:
            if tag == "step":                     # a live decode row that rode along
                t = preds[pi]
                pi += 1
                row = r.row
                self._advance(row, r, t, off_rows, pg)
                if self.rows[row] is r:
                    upd.append((row, t, r.pos))
                continue
            if tag is None:                       # resync row
                d = r.cand
                pr = preds[pi: pi + len(d) + 1]
                pi += len(d) + 1
                n = 0
                while n < len(d) and pr[n] == d[n]:
                    n += 1
                if d:                               # a parked request comes back with an empty candidate
                    self.stats["resyncs"] += 1
                    self.stats["resync_tokens"] += len(d)
                    self.stats["resync_accepted"] += n
                    r.info["resync_accepted"] = r.info.get("resync_accepted", 0) + n
                    r.info["resyncs"] = r.n_resync
                r.gen.extend(d[:n])
                r.rem, r.cand = d[n:], None
                nxt = pr[n]
                r.pos += n + 1
            else:
                nxt = self._verify_new(r, preds[pi: pi + len(r.draft) + 1], t0, t1, B, off)
                pi += len(r.draft) + 1
            if nxt in c.eos:
                self._finish(r, "stop")
                continue
            r.gen.append(nxt)
            if len(r.gen) >= r.budget:
                self._finish(r, "length")
                continue
            if not self.free_rows:
                # No decode row left (admission normally prevents this). Wait for the next pass: the
                # request keeps its pages and its last token is fed again there.
                r.cand = []
                self.resume.append(r)
                self.stats["parked"] += 1
                continue
            r.row = heapq.heappop(self.free_rows)
            self.rows[r.row] = r
            self.n_live += 1
            if r.priority >= BACKGROUND:
                self.n_bg += 1
            act_rows.append(r.row); act_tok.append(nxt); act_pos.append(r.pos)
            row_bt = np.zeros(c.max_pages, dtype=np.int32)
            row_bt[:len(r.pages)] = r.pages
            act_bt.append(row_bt)
        if upd:
            self.stats["steps"] += 1
            self.stats["fused_steps"] += 1
            self.stats["step_rows"] += len(upd) + len(off_rows)
            rows = torch.tensor([u[0] for u in upd], device=dev)
            c.g_tok[rows] = torch.tensor([u[1] for u in upd], device=dev)
            c.g_pos[rows] = torch.tensor([u[2] for u in upd], device=dev)
        self._post_step(off_rows, pg)
        if act_rows:
            rows = torch.tensor(act_rows, device=dev)
            c.g_tok[rows] = torch.tensor(act_tok, device=dev)
            c.g_pos[rows] = torch.tensor(act_pos, device=dev)
            c.g_act[rows] = 1
            c.g_bt[rows] = tt(np.stack(act_bt), torch.int32)
            self.top = max(self.top, max(act_rows) + 1)
            if self.n_live > self.stats["max_rows"]:
                self.stats["max_rows"] = self.n_live

    def _verify_new(self, r, pr, t0, t1, B, off):
        """Check the draft of a new request: pr[j] is the greedy token at draft slot j. Keep the longest
        agreeing prefix; return the first token greedy decoding wants next."""
        d = r.draft
        n = 0
        while n < len(d) and pr[n] == d[n]:
            n += 1
        st = r.stream
        if st is not None:
            # blocks nb..nb_done are now complete and their KV is final: keep them for the next call
            st.floor = st.floor[:r.nb] + [r.floor] * (r.nb_done - r.nb)
            st.bmin = list(r.bmin)
            st.n_blocks, st.head = r.nb_done, r.head
            st.prefix = r.audio[: r.nb_done * BLOCK_FRAMES * HOP + HOP].copy() if r.nb_done else None
        r.t_admit, r.t_pf = t0, t1
        r.info.update(queue_ms=(t0 - r.t_submit) * 1e3, prefill_ms=(t1 - t0) * 1e3,
                      first_token_ms=(t1 - r.t_submit) * 1e3, prompt_tokens=r.end,
                      cached_tokens=r.p0, cached_blocks=r.nb, draft_tokens=len(d), accepted_tokens=n,
                      prefill_batch=B, prefill_batch_tokens=off)
        r.gen = d[:n]
        r.rem = d[n:]
        r.pos = r.end + n                         # where the next token goes
        return pr[n]

    # ---- one decode step for every live row
    def _step(self):
        c = self.core
        B = c.bucket(self.top)
        S = 0
        if c.kv_step:
            S = c.kv_bucket(max(r.pos for r in self.rows[:self.top] if r is not None) + 1)
        c.step(B, S)
        if self.profile:
            ts = time.perf_counter()
        toks = c.g_out[:B].tolist()               # host sync: one per step for the whole batch
        if self.profile:
            self.phase["step"] = self.phase.get("step", 0.0) + time.perf_counter() - ts
        self.stats["steps"] += 1
        self.stats["step_rows"] += self.n_live
        off_rows, pg = [], ([], [], [])
        for row in range(self.top):
            r = self.rows[row]
            if r is not None:
                self._advance(row, r, toks[row], off_rows, pg)
        self._post_step(off_rows, pg)
        if self.profile:
            self.phase["step_host"] = self.phase.get("step_host", 0.0) + time.perf_counter() - ts

    def _leave_row(self, row, r, off_rows):
        off_rows.append(row)
        self.rows[row] = None
        r.row = -1
        heapq.heappush(self.free_rows, row)
        self.n_live -= 1
        if r.priority >= BACKGROUND:
            self.n_bg -= 1

    def _advance(self, row, r, t, off_rows, pg):
        """Row `row` produced token t. Finish, hand over to a resync pass, or stay for the next step."""
        c = self.core
        reason = None
        if r.cancelled:
            reason = "cancelled"
        elif t in c.eos:
            reason = "stop"
        elif r.abort:
            reason = "abort"
        else:
            r.gen.append(t)
            if len(r.gen) >= r.budget:
                reason = "length"
            elif r.on_tokens is not None and len(r.gen) % self.progress_every == 0 and self._progress(r):
                reason = "abort"
            else:
                r.pos += 1
                if self.resync and len(r.rem) > self.resync_min and r.n_resync < self.resync_max:
                    # Has the new output caught up with the old draft? Either the token the draft was
                    # rejected at just came out, or the last two tokens appear in it a little later.
                    rem, j = r.rem, -1
                    if t == rem[0]:
                        j = 0
                    elif len(r.gen) >= 2:
                        p = r.gen[-2]
                        for q in range(1, min(len(rem), self.resync_window)):
                            if rem[q] == t and rem[q - 1] == p:
                                j = q
                                break
                    if j >= 0 and len(rem) - j - 1 >= self.resync_min:
                        r.cand = rem[j + 1:]
                        r.n_resync += 1
                        self._leave_row(row, r, off_rows)
                        self.resume.append(r)
                        return
                if r.pos // self.core.page >= len(r.pages):
                    room = c.max_pages - len(r.pages)
                    got = self._alloc(min(self.page_ahead, room)) or (self._alloc(1) if room > 0 else None)
                    if not got:
                        # Out of KV pages: fail the request. Returning a cut transcript as "length" would
                        # look like a normal result.
                        self.stats["kv_exhausted"] += 1
                        self._leave_row(row, r, off_rows)
                        self._fail(r, PoolExhausted("kitten-asr: KV pool exhausted"))
                        return
                    else:
                        for p in got:
                            pg[0].append(row); pg[1].append(len(r.pages)); pg[2].append(p)
                            r.pages.append(p)
        if reason:
            self._leave_row(row, r, off_rows)
            self._finish(r, reason)

    def _progress(self, r):
        """Run the caller's progress callback. If it raises, stop that request only: an exception here must
        not take the scheduler thread, and every other stream, down with it."""
        try:
            return bool(r.on_tokens(r.gen))
        except Exception as e:  # noqa: BLE001
            r.info["on_tokens_error"] = repr(e)
            return True

    def _post_step(self, off_rows, pg):
        torch, c = self.torch, self.core
        if off_rows:
            rows = torch.tensor(off_rows, device=c.device)
            c.g_act[rows] = 0
            c.g_pos[rows] = 0
            c.g_bt[rows, 0] = 0                  # idle rows write to the dummy page
            while self.top and self.rows[self.top - 1] is None:
                self.top -= 1
        if pg[0]:
            c.g_bt[torch.tensor(pg[0], device=c.device), torch.tensor(pg[1], device=c.device)] = \
                torch.tensor(pg[2], device=c.device, dtype=torch.int32)

    # ---- completion
    def _settle_pages(self, r):
        if r.settled:                       # a second call must not touch the stream again
            self.free_pages.extend(r.pages)
            r.pages = []
            return
        r.settled = True
        st = r.stream
        if st is None:
            self.free_pages.extend(r.pages)
        elif not r.took_pages:
            # failed before it took over the stream's cached pages: those stay with the stream
            self.free_pages.extend(r.pages)
            st.busy = False
            st.last = time.monotonic()
            if st.dead:
                self._drop_stream(st)
        else:
            keep = -(-((len(r.head) + st.n_blocks * BLOCK_TOKENS) if st.n_blocks else 0) // self.core.page)
            if st.dead or not self.stream_cache or keep > len(r.pages):
                keep = 0
                st.n_blocks, st.floor, st.bmin, st.prefix = 0, [], [], None
            self.free_pages.extend(r.pages[keep:])
            st.pages = r.pages[:keep]
            st.busy = False
            st.last = time.monotonic()
            if st.dead:
                self.streams.pop(st.key, None)
        r.pages = []

    def _finish(self, r, reason):
        self._settle_pages(r)
        now = time.perf_counter()
        if reason == "cancelled":
            self.stats["cancelled"] += 1
            return
        i = r.info
        i.update(finish_reason=reason, generated_tokens=len(r.gen), decode_ms=(now - r.t_pf) * 1e3,
                 inference_ms=(now - r.t_admit) * 1e3, scheduler_request_ms=(now - r.t_submit) * 1e3,
                 backend="kitten-asr")
        s = self.stats
        s["requests"] += 1
        s["draft_tokens"] += i["draft_tokens"]
        s["accepted_tokens"] += i["accepted_tokens"]
        s["generated_tokens"] += len(r.gen)
        s["cached_tokens"] += i["cached_tokens"]
        r.event.set()

    def _fail(self, r, e):
        self.stats["errors"] += 1
        if r.stream is not None or r.pages:
            self._settle_pages(r)
        r.error = e
        r.event.set()
