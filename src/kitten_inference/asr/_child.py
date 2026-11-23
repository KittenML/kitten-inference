"""Entry point of the engine process that EngineProcess starts: runs the engine and answers over one connection."""
# Started as `python -m kitten_inference.asr._child <address>`; the auth key of the connection comes in the
# environment variable named by _proc.AUTHKEY_ENV. This process owns the GPU and the scheduler.
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from multiprocessing.connection import Client

from ._proc import AUTHKEY_ENV
from ._types import error_reply


class _Done:
    """Stands in for the request's threading.Event: completion goes to the sender thread instead."""

    def __init__(self, r, outq):
        self.r, self.outq, self.done = r, outq, False

    def set(self):
        if not self.done:
            self.done = True
            r, self.r = self.r, None        # drop the back-reference so the request is freed right away
            self.outq.put(r)

    def is_set(self):
        return self.done


def main() -> None:
    if len(sys.argv) != 2 or AUTHKEY_ENV not in os.environ:
        sys.stderr.write("kitten-asr: this module is the engine process that EngineProcess starts; "
                         "it is not run by hand\n")
        raise SystemExit(2)
    address = sys.argv[1]
    conn = Client(address, authkey=bytes.fromhex(os.environ.pop(AUTHKEY_ENV)))
    model, opts = conn.recv()
    try:
        from . import _deps
        _deps.require("asr", device=opts.get("device"))      # before the engine module, which imports torch
        import numpy as np

        from ._engine import Engine, _Req
        engine = Engine(model, **opts)
    except BaseException as e:  # noqa: BLE001
        conn.send(("fail",) + error_reply(e))
        raise
    conn.send(("ready", os.getpid()))
    outq: queue.SimpleQueue = queue.SimpleQueue()
    live: dict = {}
    eos = engine._eos

    def sender():
        while True:
            r = outq.get()
            if r is None:
                return
            try:
                if isinstance(r, tuple):
                    conn.send(r)
                    continue
                live.pop(r.rid, None)
                if r.error is not None:
                    conn.send(("err", r.rid) + error_reply(r.error))
                else:
                    conn.send(("res", r.rid, engine._text(r.gen), r.gen, r.info))
            except OSError:
                return                          # the calling process is gone; the main loop ends on its own

    threading.Thread(target=sender, name="kitten-asr-sender", daemon=True).start()

    def watch_scheduler():
        # If the scheduler thread dies, this process is useless: let the error messages out, then exit so
        # the calling process sees the connection close and fails its callers at once, not on their timeouts.
        engine._thread.join()
        time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch_scheduler, name="kitten-asr-watch", daemon=True).start()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        kind = msg[0]
        if kind == "dec":
            _, rid, prompt, draft, key, max_new, priority, progress = msg
            audio = np.frombuffer(conn.recv_bytes(), dtype=np.float32)
            try:
                head, tail = engine._split(prompt)
            except Exception as e:  # noqa: BLE001
                outq.put(("err", rid) + error_reply(e))
                continue
            r = _Req(audio, head, tail, [t for t in draft if t not in eos], key if engine.stream_cache else None,
                     max_new, priority, None)
            r.rid, r.abort = rid, False
            r.event = _Done(r, outq)
            if progress:
                def on_tokens(ids, r=r):
                    outq.put(("prog", r.rid, list(ids)))      # the caller answers with "abort" to stop it
                    return r.abort
                r.on_tokens = on_tokens
            live[rid] = r
            engine._inbox.put(r)
        elif kind == "cancel":
            r = live.pop(msg[1], None)
            if r is not None:
                r.cancelled = True
        elif kind == "abort":
            r = live.get(msg[1])
            if r is not None:
                r.abort = True
        elif kind == "rel":
            engine.release(msg[1])
        elif kind in ("snap", "prompt"):
            try:
                value = engine.snapshot() if kind == "snap" else engine.default_prompt(msg[2])
            except Exception as e:  # noqa: BLE001
                outq.put(("err", msg[1]) + error_reply(e))
            else:
                outq.put(("rpc", msg[1], value))
        elif kind == "stop":
            break
    os._exit(0)             # the process exit frees the GPU; nothing here needs an orderly shutdown


if __name__ == "__main__":
    main()
