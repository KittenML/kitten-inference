"""The ASR engine in a process of its own, behind the same decode / release / snapshot calls."""
# The scheduler is one Python thread. In a program whose own threads keep the interpreter busy it has to wait
# for the interpreter lock between GPU steps; in a process of its own it does not. A call then costs one
# small message plus the audio bytes over a local socket. This module imports neither torch nor the engine.
from __future__ import annotations

import itertools
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Listener

from . import _deps
from ._types import DecodeResult, EngineError, EngineStopped, error_from_reply

AUTHKEY_ENV = "KITTEN_ASR_AUTHKEY"            # carries the connection's auth key to the engine process
_CHILD = "kitten_inference.asr._child"
_EXITED = "kitten-asr: engine process exited"
_CLOSED = "kitten-asr: the engine is closed"


class _Wait:
    __slots__ = ("ev", "msg", "cb", "aborted")

    def __init__(self):
        self.ev, self.msg, self.cb, self.aborted = threading.Event(), None, None, False


def _accept(listener, address, check):
    """listener.accept() under the caller's deadline. accept() cannot time out by itself, so it runs on a
    helper thread while this one calls check(what) once a second; check raises when the wait must end."""
    box, stop = {}, threading.Event()

    def run():
        while not stop.is_set():
            try:
                box["conn"] = listener.accept()
                return
            except OSError:
                time.sleep(0.05)                # try again until the caller gives up
            except BaseException as e:  # noqa: BLE001
                box["err"] = e
                return

    t = threading.Thread(target=run, name="kitten-asr-accept", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(1.0)
            if t.is_alive():
                check("connecting")
    except BaseException:
        # Wake the helper: a connection that closes at once makes its accept() return or fail.
        stop.set()
        try:
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(1.0)
            s.connect(address)
            s.close()
        except OSError:
            pass
        t.join(2.0)
        if box.get("conn") is not None:
            box["conn"].close()
        raise
    if "err" in box:
        raise box["err"]
    return box["conn"]


class EngineProcess:
    """Engine(model, **options) running in a child process, with the same methods as Engine.

    The child is started with this interpreter and loads the model there; the constructor returns when it
    is ready, or raises within `start_timeout_s` (one deadline for connecting and loading). `options` are
    the Engine options; they are sent to the child, so they must be plain values (numbers, strings, lists).
    Errors raised by the engine arrive as the same class for EngineError, PoolExhausted, EngineStopped and
    common built-ins such as ValueError, and as EngineError otherwise.
    """

    name = "kitten-asr"

    def __init__(self, model, start_timeout_s: float = 900.0, **options):
        _deps.require("asr-process")
        self.opts = dict(options)
        self._dead = None
        self._closed = False
        self.conn = None
        self.proc = None
        key = secrets.token_bytes(32)
        self._dir = tempfile.mkdtemp(prefix="kitten-asr-")
        address = os.path.join(self._dir, "sock")
        listener = None
        deadline = time.monotonic() + start_timeout_s        # one deadline for connect + model load

        def check(what):
            rc = self.proc.poll()
            if rc is not None:
                raise EngineError(f"kitten-asr: engine process exited with code {rc} while {what}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"kitten-asr: engine process not ready after {start_timeout_s}s ({what})")

        try:
            listener = Listener(address, family="AF_UNIX", authkey=key)
            env = dict(os.environ)
            env[AUTHKEY_ENV] = key.hex()
            self.proc = subprocess.Popen([sys.executable, "-m", _CHILD, address], env=env)
            self.conn = _accept(listener, address, check)
            self.conn.send((os.fspath(model), self.opts))
            while not self.conn.poll(1.0):
                check("loading the model")
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                # the child closed the connection without an answer: report how it ended
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                check("loading the model")
                raise EngineError("kitten-asr: engine process closed the connection while loading the model")
            if msg[0] != "ready":
                raise error_from_reply(msg[1], msg[2], "kitten-asr: engine process failed to start: ")
        except BaseException:
            if self.proc is not None:
                self.proc.kill()
                self.proc.wait()
            if self.conn is not None:
                self.conn.close()
            raise
        finally:
            if listener is not None:
                listener.close()
            shutil.rmtree(self._dir, ignore_errors=True)
        self.engine_pid = msg[1]
        self._ids = itertools.count(1)
        self._wait: dict = {}
        self._wlock = threading.Lock()
        self._prompts: dict = {}
        self._reader_thread = threading.Thread(target=self._reader, name="kitten-asr-reader", daemon=True)
        self._reader_thread.start()

    def _send(self, msg, payload=None):
        with self._wlock:
            self.conn.send(msg)
            if payload is not None:
                self.conn.send_bytes(payload)

    def _reader(self):
        while True:
            try:
                msg = self.conn.recv()
            except Exception:  # noqa: BLE001
                if not self._dead:
                    self._dead = _EXITED
                for w in list(self._wait.values()):       # engine process gone: fail everything waiting
                    w.msg = ("err", 0, "EngineStopped", self._dead)
                    w.ev.set()
                return
            kind, rid = msg[0], msg[1]
            w = self._wait.get(rid)
            if w is None:
                continue
            if kind == "prog":
                try:
                    if w.cb is not None and not w.aborted and w.cb(msg[2]):
                        w.aborted = True            # the caller asked to stop, whatever the engine reports
                        self._send(("abort", rid))
                except Exception:  # noqa: BLE001
                    pass
                continue
            w.msg = msg
            w.ev.set()

    def _check_running(self):
        if self._dead:
            raise EngineStopped(self._dead)

    def _rpc(self, kind, what, *args, timeout=30.0):
        self._check_running()
        rid = -next(self._ids)
        w = self._wait[rid] = _Wait()
        try:
            self._send((kind, rid) + args)
        except (OSError, ValueError) as e:
            self._wait.pop(rid, None)
            raise EngineStopped(self._dead or f"kitten-asr: engine process is unreachable: {e!r}") from e
        ok = w.ev.wait(timeout)
        self._wait.pop(rid, None)
        if not ok:
            raise EngineError(f"kitten-asr: engine process did not answer {what}")
        if w.msg[0] == "err":
            raise error_from_reply(w.msg[2], w.msg[3])
        return w.msg[2]

    def default_prompt(self, language: str | None = None) -> str:
        """The prompt text for decode(); see Engine.default_prompt."""
        self._check_running()
        if language not in self._prompts:
            self._prompts[language] = self._rpc("prompt", "default_prompt", language)
        return self._prompts[language]

    def decode(self, audio, prompt, *, draft_ids=None, stream_key=None, max_new_tokens=256, priority=0,
               timeout_s=30.0, on_tokens=None) -> DecodeResult:
        """Engine.decode in the engine process. on_tokens runs in this process, on the one thread that reads
        every reply, so keep it fast; a stop it asks for takes effect at the engine's next step. If the
        engine had already finished, the full result comes back with finish_reason "abort"."""
        self._check_running()
        import numpy as np          # only for the audio bytes
        rid = next(self._ids)
        w = self._wait[rid] = _Wait()
        w.cb = on_tokens
        a = np.ascontiguousarray(audio, dtype=np.float32)
        try:
            self._send(("dec", rid, prompt, list(draft_ids or []), stream_key, int(max_new_tokens), int(priority),
                        on_tokens is not None), a)
        except (OSError, ValueError) as e:
            self._wait.pop(rid, None)
            raise EngineStopped(self._dead or f"kitten-asr: engine process is unreachable: {e!r}") from e
        if self._dead and not w.ev.is_set():        # died between the check above and the send
            self._wait.pop(rid, None)
            raise EngineStopped(self._dead)
        ok = w.ev.wait(timeout_s)
        self._wait.pop(rid, None)
        if not ok:
            try:
                self._send(("cancel", rid))
            except (OSError, ValueError):
                pass
            raise TimeoutError(f"kitten-asr: decode timed out after {timeout_s}s")
        if w.msg[0] == "err":
            raise error_from_reply(w.msg[2], w.msg[3])
        _, _, text, ids, info = w.msg
        if w.aborted and info.get("finish_reason") != "abort":
            info["finish_reason"] = "abort"         # the stop request arrived after the engine had finished
        return DecodeResult(text, ids, info)

    def release(self, stream_key: str) -> None:
        """Drops what is cached for `stream_key`. Does nothing once the engine has stopped."""
        if self._dead:
            return
        try:
            self._send(("rel", stream_key))
        except (OSError, ValueError):
            pass

    def snapshot(self) -> dict:
        """Engine.snapshot of the engine process, plus its process id as "engine_pid"."""
        s = self._rpc("snap", "snapshot")
        s["engine_pid"] = self.engine_pid
        return s

    def close(self) -> None:
        """Stops the engine process, which frees its GPU memory. Calls still waiting fail with EngineStopped,
        and so does every later decode(), snapshot() and default_prompt(). Safe to call again."""
        if self._closed:
            return
        self._closed = True
        if not self._dead:
            self._dead = _CLOSED
        try:
            self._send(("stop",))
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        if threading.current_thread() is not self._reader_thread:
            self._reader_thread.join(timeout=5)
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
