"""The result and error types, and the engine in a process of its own, without PyTorch and without a second
process. Three layers:
  - EngineProcess against a scripted engine process on the other end of a pipe (what it sends, what it raises);
  - _child.main() on a thread, with a fake engine, against a test that plays the calling side;
  - both together over the real local socket (Linux and macOS), with the fake engine: start, calls, errors, stop.
The fakes are checked against the source of the real engine, so they cannot drift from it unnoticed."""
import ast
import inspect
import itertools
import multiprocessing.connection
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from kitten_inference.asr import _child, _deps, _proc, _types
from kitten_inference.asr._types import (DecodeResult, EngineError, EngineStopped, PoolExhausted, error_from_reply,
                                         error_reply)

ASR_DIR = Path(_types.__file__).parent
# positional names, keyword-only names and their defaults of decode()
DECODE = (["self", "audio", "prompt"],
          ["draft_ids", "stream_key", "max_new_tokens", "priority", "timeout_s", "on_tokens"],
          [None, None, 256, 0, 30.0, None])
DIE = object()            # a scripted reply: the engine process ends without an answer
posix = pytest.mark.skipif(sys.platform == "win32" or not hasattr(socket, "AF_UNIX"),
                           reason="platform: the engine process needs Linux or macOS")


def _until(condition, what, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def _recv(conn, timeout=5.0):
    assert conn.poll(timeout), "no message arrived"
    return conn.recv()


# ---------------------------------------------------------------- types
def test_error_classes():
    assert issubclass(PoolExhausted, EngineError) and issubclass(EngineStopped, EngineError)
    assert issubclass(EngineError, RuntimeError) and not issubclass(PoolExhausted, EngineStopped)


def test_decode_result():
    a, b = DecodeResult("hi", [1, 2]), DecodeResult("hi", [1, 2])
    assert a == b and a.info == {} and a.info is not b.info
    assert DecodeResult("hi", [1], {"finish_reason": "stop"}).info == {"finish_reason": "stop"}


@pytest.mark.parametrize("error", [
    EngineError("kitten-asr: engine process did not answer snapshot"), PoolExhausted("kitten-asr: KV pool exhausted"),
    EngineStopped("kitten-asr: the engine is closed"), ValueError("prompt must contain exactly one audio placeholder"),
    TypeError("t"), ImportError("Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch"),
    ModuleNotFoundError("No module named 'torch'"), FileNotFoundError("model folder not found: ./tiny"),
    TimeoutError("kitten-asr: decode timed out after 30.0s"), MemoryError("m"), RuntimeError("r")])
def test_known_errors_cross_with_class_and_message(error):
    name, message = error_reply(error)
    assert (name, message) == (type(error).__name__, str(error))
    back = error_from_reply(name, message, "never used: ")
    assert type(back) is type(error) and str(back) == str(error)


def test_other_errors_cross_as_engine_error():
    assert error_reply(KeyError("layers")) == ("KeyError", "'layers'")
    back = error_from_reply("KeyError", "'layers'")
    assert type(back) is EngineError and str(back) == "kitten-asr: KeyError: 'layers'"
    start = error_from_reply("ZeroDivisionError", "division by zero", "kitten-asr: engine process failed to start: ")
    assert str(start) == "kitten-asr: engine process failed to start: ZeroDivisionError: division by zero"

    class Custom(ValueError):
        pass
    assert type(error_from_reply(*error_reply(Custom("c")))) is EngineError      # by name, not by base class
    assert error_reply(KeyError()) == ("KeyError", "KeyError()")                 # no message: the repr instead


# ---------------------------------------------------------------- the fakes follow the real engine
def _class(name, cls):
    tree = ast.parse((ASR_DIR / name).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)


def _methods(node):
    return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


def _set_on_self(node):
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name) and n.value.id == "self"}


def _source_shape(fn):
    a = fn.args
    return [x.arg for x in a.args], [x.arg for x in a.kwonlyargs], [ast.literal_eval(d) for d in a.kw_defaults]


def _live_shape(fn):
    params = list(inspect.signature(fn).parameters.values())
    keyword = [p for p in params if p.kind is p.KEYWORD_ONLY]
    return ([p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD], [p.name for p in keyword],
            [p.default for p in keyword])


def test_engine_process_and_fake_take_the_same_decode(fake_engine):
    engine, process = _methods(_class("_engine.py", "Engine")), _methods(_class("_proc.py", "EngineProcess"))
    assert _source_shape(engine["decode"]) == DECODE and _source_shape(process["decode"]) == DECODE
    assert _live_shape(fake_engine.type.decode) == DECODE
    public = {"default_prompt", "decode", "release", "snapshot", "close", "__enter__", "__exit__"}
    assert public <= set(engine) and public <= set(process)
    assert public - {"__enter__", "__exit__"} <= set(vars(fake_engine.type))


def test_file_work_is_background_work_for_the_engine():
    from kitten_inference.asr import _api
    tree = ast.parse((ASR_DIR / "_engine.py").read_text(encoding="utf-8"))
    constants = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
    assert constants["BACKGROUND"] == _api._BACKGROUND == 10


# ---------------------------------------------------------------- EngineProcess against a scripted engine process
class Script:
    """Plays the engine process on the other end of a pipe: keeps every message it gets (`seen`, and the audio
    of a decode in `audio[rid]`) and answers a "dec", "snap" or "prompt" with the messages reply(msg) returns
    (DIE: it ends there, as a process that crashed). "stop" ends it."""

    def __init__(self, conn, reply):
        self.conn, self.reply = conn, reply
        self.seen, self.audio = [], {}
        self._cond = threading.Condition()
        threading.Thread(target=self._run, name="scripted-engine-process", daemon=True).start()

    def _run(self):
        while True:
            try:
                msg = self.conn.recv()
                audio = self.conn.recv_bytes() if msg[0] == "dec" else None
            except (EOFError, OSError):
                return
            with self._cond:
                if audio is not None:
                    self.audio[msg[1]] = np.frombuffer(audio, dtype=np.float32)
                self.seen.append(msg)
                self._cond.notify_all()
            answers = self.reply(msg) if msg[0] in ("dec", "snap", "prompt") else []
            if answers is DIE or msg[0] == "stop":
                self.conn.close()
                return
            try:
                for answer in answers:
                    self.conn.send(answer)
            except OSError:
                return

    def wait_for(self, msg, timeout=5.0):
        with self._cond:
            assert self._cond.wait_for(lambda: msg in self.seen, timeout), f"{msg!r} never arrived: {self.seen!r}"


class _Ended:
    """The process object of an engine process that ends when asked."""

    def __init__(self):
        self.waits = self.kills = 0

    def wait(self, timeout=None):
        self.waits += 1
        return 0

    def kill(self):
        self.kills += 1


def _answer(msg):
    if msg[0] == "dec":
        return [("res", msg[1], "text %d" % msg[1], [1, 2, 3], {"finish_reason": "stop"})]
    if msg[0] == "snap":
        return [("rpc", msg[1], {"attention": "torch", "page": 16})]
    return [("rpc", msg[1], "<audio>%s" % msg[2])]


@pytest.fixture
def scripted():
    """scripted(reply) -> (an EngineProcess whose engine process is a Script, that Script). No process starts."""
    made = []

    def make(reply=_answer):
        ours, theirs = multiprocessing.connection.Pipe()
        script = Script(theirs, reply)
        p = _proc.EngineProcess.__new__(_proc.EngineProcess)
        p.opts, p._dead, p._closed = {}, None, False
        p.conn, p.proc, p.engine_pid = ours, _Ended(), 4242
        p._ids, p._wait, p._wlock, p._prompts = itertools.count(1), {}, threading.Lock(), {}
        p._reader_thread = threading.Thread(target=p._reader, name="kitten-asr-reader", daemon=True)
        p._reader_thread.start()
        made.append((p, theirs))
        return p, script
    yield make
    for p, theirs in made:
        p.close()
        theirs.close()


def _decode(p, **options):
    return p.decode(np.zeros(4, np.float32), "<audio>", timeout_s=5, **options)


def test_names():
    assert _proc.EngineProcess.name == "kitten-asr" and _proc.AUTHKEY_ENV == "KITTEN_ASR_AUTHKEY"


def test_decode_is_one_message_and_the_audio_bytes(scripted):
    p, script = scripted()
    audio = (np.arange(6, dtype=np.float64) / 8)[::2]                # not float32 and not contiguous
    out = p.decode(audio, "<audio>", draft_ids=(4, 5), stream_key="k:1", max_new_tokens=40, priority=10,
                   timeout_s=5)
    assert script.seen == [("dec", 1, "<audio>", [4, 5], "k:1", 40, 10, False)]
    assert script.audio[1].tolist() == [0.0, 0.25, 0.5]
    assert type(out) is DecodeResult
    assert (out.text, out.token_ids, out.info) == ("text 1", [1, 2, 3], {"finish_reason": "stop"})
    assert _decode(p).text == "text 2"
    assert script.seen[1] == ("dec", 2, "<audio>", [], None, 256, 0, False)          # the defaults
    assert p._wait == {}                                             # nothing is left waiting


@pytest.mark.parametrize("name, message, expected", [
    ("ValueError", "prompt must contain exactly one audio placeholder", ValueError),
    ("PoolExhausted", "kitten-asr: KV pool exhausted", PoolExhausted),
    ("EngineStopped", "kitten-asr: the scheduler thread has stopped", EngineStopped),
    ("TimeoutError", "kitten-asr: decode timed out after 30.0s", TimeoutError),
    ("RuntimeError", "CUDA error: out of memory", RuntimeError),
])
def test_engine_errors_keep_their_class(scripted, name, message, expected):
    p, _ = scripted(lambda msg: [("err", msg[1], name, message)])
    for call in (lambda: _decode(p), p.snapshot, lambda: p.default_prompt("English")):
        with pytest.raises(expected) as e:
            call()
        assert type(e.value) is expected and str(e.value) == message


def test_other_errors_arrive_as_engine_error(scripted):
    p, _ = scripted(lambda msg: [("err", msg[1], "KeyError", "'layers'")])
    with pytest.raises(EngineError) as e:
        _decode(p)
    assert type(e.value) is EngineError and str(e.value) == "kitten-asr: KeyError: 'layers'"


def test_decode_timeout_cancels_and_a_late_answer_is_dropped(scripted):
    def reply(msg):
        if msg[1] == 1:
            return []                                               # never answered in time
        late = ("res", 1, "late", [9], {"finish_reason": "stop"})
        return [late, ("res", msg[1], "on time", [1], {"finish_reason": "stop"})]
    p, script = scripted(reply)
    with pytest.raises(TimeoutError) as e:
        p.decode(np.zeros(4, np.float32), "<audio>", timeout_s=0.2)
    assert str(e.value) == "kitten-asr: decode timed out after 0.2s"
    script.wait_for(("cancel", 1))
    assert p._wait == {} and _decode(p).text == "on time"


def test_progress_reaches_the_callback_and_its_stop_reaches_the_engine(scripted):
    def reply(msg):
        rid = msg[1]
        return [("prog", rid, [7]), ("prog", rid, [7, 8]), ("res", rid, "t", [7, 8], {"finish_reason": "stop"})]
    p, script = scripted(reply)
    seen = []
    out = _decode(p, on_tokens=lambda ids: seen.append(ids))
    assert seen == [[7], [7, 8]] and out.info["finish_reason"] == "stop"
    assert script.seen[0][-1] is True and _decode(p).text == "t" and script.seen[1][-1] is False

    stopped = []
    out = _decode(p, on_tokens=lambda ids: stopped.append(ids) or True)
    assert stopped == [[7]]                                         # not called again after asking to stop
    assert out.info["finish_reason"] == "abort"                     # even if the engine had already finished
    script.wait_for(("abort", 3))
    assert script.seen.count(("abort", 3)) == 1 and ("abort", 1) not in script.seen

    def broken(ids):
        raise ZeroDivisionError
    assert _decode(p, on_tokens=broken).text == "t"                 # a failing callback does not stop the reader
    assert _decode(p).text == "t"


def test_snapshot_and_prompt_calls(scripted):
    p, script = scripted()
    assert p.snapshot() == {"attention": "torch", "page": 16, "engine_pid": 4242}
    assert p.default_prompt("English") == "<audio>English" and p.default_prompt("English") == "<audio>English"
    assert p.default_prompt() == "<audio>None"
    assert script.seen == [("snap", -1), ("prompt", -2, "English"), ("prompt", -3, None)]       # asked once each


def test_release_is_sent_and_not_waited_for(scripted):
    p, script = scripted()
    assert p.release("k:1") is None
    script.wait_for(("rel", "k:1"))


def test_engine_process_gone_fails_the_waiting_call_and_every_later_one(scripted):
    p, script = scripted(lambda msg: DIE)
    with pytest.raises(EngineStopped) as e:
        _decode(p)
    assert str(e.value) == "kitten-asr: engine process exited"
    for call in (lambda: _decode(p), p.snapshot, p.default_prompt):
        with pytest.raises(EngineStopped, match="kitten-asr: engine process exited"):
            call()
    assert p.release("k:1") is None                                 # never raises: it is called in finally blocks
    assert p._wait == {}


def test_close_stops_the_process_and_refuses_later_calls(scripted):
    p, script = scripted()
    assert p.default_prompt("English") == "<audio>English"
    p.close()
    script.wait_for(("stop",))
    assert (p.proc.waits, p.proc.kills) == (1, 0)
    for call in (lambda: _decode(p), p.snapshot, lambda: p.default_prompt("English")):        # also a kept prompt
        with pytest.raises(EngineStopped) as e:
            call()
        assert str(e.value) == "kitten-asr: the engine is closed"
    count = len(script.seen)
    assert p.release("k:1") is None
    p.close()
    with p:
        pass
    assert (p.proc.waits, p.proc.kills) == (1, 0) and len(script.seen) == count


def test_close_fails_a_decode_that_is_waiting(scripted):
    p, script = scripted(lambda msg: [])
    errors = []

    def decode():
        try:
            p.decode(np.zeros(4, np.float32), "<audio>", timeout_s=20)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    thread = threading.Thread(target=decode, daemon=True)
    thread.start()
    script.wait_for(("dec", 1, "<audio>", [], None, 256, 0, False))
    p.close()
    thread.join(5)
    assert not thread.is_alive() and len(errors) == 1
    assert type(errors[0]) is EngineStopped and str(errors[0]) == "kitten-asr: the engine is closed"


# ---------------------------------------------------------------- the engine process, on a thread
class _Req:
    """The request record _child.main builds: the constructor of the engine's own (checked below)."""

    def __init__(self, audio, head, tail, draft, key, max_new, priority, on_tokens):
        self.audio, self.head, self.tail, self.draft = audio, head, tail, draft
        self.key, self.max_new, self.priority, self.on_tokens = key, max_new, priority, on_tokens
        self.event = threading.Event()
        self.cancelled, self.abort, self.error = False, False, None
        self.gen, self.info = [], {}


class _Forever:
    """The scheduler thread of the fake engine: it never ends, so the engine process's watcher never acts."""

    def join(self, timeout=None):
        while True:
            threading.Event().wait(3600)


class ChildEngine:
    """What _child.main uses of an engine. A worker thread answers the requests; the text after the audio
    placeholder of the prompt picks what happens: "fail" (the request fails), "hold" (it waits for `go`),
    "wait" (between two progress reports it waits until the caller asked to stop)."""

    def __init__(self, model, **opts):
        self.model, self.opts = model, opts
        self.stream_cache = opts.get("stream_cache", True)
        self._eos = {2}
        self._inbox = queue.SimpleQueue()
        self._thread = _Forever()
        self.requests, self.released = [], []
        self.go, self.stopped = threading.Event(), False
        threading.Thread(target=self._work, name="fake-engine", daemon=True).start()

    def _split(self, prompt):
        if prompt.count("<audio>") != 1:
            raise ValueError("prompt must contain exactly one audio placeholder")
        head, _, tail = prompt.partition("<audio>")
        return tuple(head.encode()), tuple(tail.encode())

    def _text(self, ids):
        return " ".join(str(i) for i in ids)

    def default_prompt(self, language=None):
        if language == "Klingon":
            raise ValueError("no prompt for Klingon")
        return "<audio>%s" % language

    def snapshot(self):
        if self.stopped:
            raise EngineStopped("kitten-asr: the scheduler thread has stopped")
        return {"attention": "torch", "page": 16}

    def release(self, key):
        self.released.append(key)

    def stop(self):
        self.go.set()
        self._inbox.put(None)

    def _work(self):
        while True:
            r = self._inbox.get()
            if r is None:
                return
            self.requests.append(r)
            mode = bytes(r.tail).decode()
            if mode == "fail":
                r.error = PoolExhausted("kitten-asr: KV pool exhausted")
            elif r.on_tokens is not None:
                r.stops = [r.on_tokens([7])]
                deadline = time.monotonic() + 10
                while mode == "wait" and not r.abort and not self.go.is_set() and time.monotonic() < deadline:
                    time.sleep(0.01)
                r.stops.append(r.on_tokens([7, 8]))
                r.gen, r.info = [7, 8], {"finish_reason": "abort" if r.stops[-1] else "stop"}
            else:
                if mode == "hold":
                    self.go.wait(10)
                r.gen, r.info = [len(r.audio)] + list(r.draft), {"finish_reason": "stop"}
            r.event.set()


def test_the_fake_engine_has_what_the_real_one_gives_the_engine_process():
    child = ast.parse((ASR_DIR / "_child.py").read_text(encoding="utf-8"))
    used = {n.attr for n in ast.walk(child) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "engine"}
    assert {"_split", "_text", "_inbox", "_thread", "_eos", "stream_cache", "release", "snapshot",
            "default_prompt"} <= used
    engine = _class("_engine.py", "Engine")
    defined = set(_methods(engine)) | _set_on_self(engine)
    assert used <= defined, sorted(used - defined)                  # the real engine has all of it
    fake = ChildEngine("model")
    fake.stop()
    assert used <= set(dir(fake)), sorted(used - set(dir(fake)))
    request = _methods(_class("_engine.py", "_Req"))["__init__"]
    assert _source_shape(request)[0] == _live_shape(_Req.__init__)[0] == [
        "self", "audio", "head", "tail", "draft", "key", "max_new", "priority", "on_tokens"]
    assert {"event", "cancelled", "abort", "error", "gen", "info", "on_tokens"} <= _set_on_self(request)


class _Exit(BaseException):
    """os._exit(code) of the engine process, which here is only a thread."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


@pytest.fixture
def child_env(monkeypatch):
    """What _child.main() needs to run on a thread of this process: the engine module is a fake (ChildEngine),
    the dependency check only records, and os._exit ends the thread. start(...) runs main() and returns the
    run: done (set when it ended), code (exit code), error (an exception that ended it), conn, connected."""
    env = types.SimpleNamespace(engines=[], built=[], required=[], runs=[], fail_require=None, fail_engine=None)

    def build(model, **opts):
        env.built.append((model, opts))
        if env.fail_engine is not None:
            raise env.fail_engine
        env.engines.append(ChildEngine(model, **opts))
        return env.engines[-1]

    def require(feature="asr", device=None):
        env.required.append((feature, device))
        if env.fail_require is not None and feature == "asr":
            raise env.fail_require

    def leave(code):
        raise _Exit(code)

    module = types.ModuleType("kitten_inference.asr._engine")       # the real one imports torch
    module.Engine, module._Req = build, _Req
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(_deps, "require", require)
    monkeypatch.setattr(_child, "os", types.SimpleNamespace(environ=os.environ, getpid=os.getpid, _exit=leave))
    connect = _child.Client

    def start(address="addr", key="ab" * 32, client=None):
        run = types.SimpleNamespace(code=None, error=None, done=threading.Event(), conn=None, connected=None)

        def open_connection(address, authkey=None):
            run.connected = (address, authkey)
            run.conn = client() if client is not None else connect(address, authkey=authkey)
            return run.conn

        def main():
            try:
                _child.main()
            except (_Exit, SystemExit) as e:
                run.code = e.code
            except BaseException as e:  # noqa: BLE001
                run.error, run.code = e, 1
            finally:
                if run.conn is not None:
                    run.conn.close()                                # a process that ends closes its connection
                run.done.set()

        monkeypatch.setattr(sys, "argv", ["_child", address])
        monkeypatch.setenv(_proc.AUTHKEY_ENV, key)
        monkeypatch.setattr(_child, "Client", open_connection)
        env.runs.append(run)
        threading.Thread(target=main, name="fake-engine-process", daemon=True).start()
        return run
    env.start = start
    yield env
    for engine in env.engines:
        engine.stop()


@pytest.fixture
def piped(child_env):
    """The calling side is the test: parent is its end of a pipe, start(model, **options) runs the engine
    process on the other end."""
    parent, child = multiprocessing.connection.Pipe()

    def start(model="model-dir", **options):
        parent.send((model, options))
        return child_env.start(client=lambda: child)
    yield types.SimpleNamespace(parent=parent, start=start)
    parent.close()


def _send_decode(conn, rid, prompt, audio, draft=(), key=None, max_new=32, priority=0, progress=False):
    conn.send(("dec", rid, prompt, list(draft), key, max_new, priority, progress))
    conn.send_bytes(audio.tobytes())


def test_engine_process_serves_decodes_and_calls(piped, child_env):
    parent = piped.parent
    run = piped.start("model-dir", device="cuda:0", max_batch=4)
    assert _recv(parent) == ("ready", os.getpid())
    engine = child_env.engines[0]
    assert (engine.model, engine.opts) == ("model-dir", {"device": "cuda:0", "max_batch": 4})
    assert child_env.required == [("asr", "cuda:0")]                # checked before the engine is built
    assert run.connected == ("addr", bytes.fromhex("ab" * 32))
    assert _proc.AUTHKEY_ENV not in os.environ                      # the key does not stay in the environment

    audio = np.linspace(-1.0, 1.0, 800, dtype=np.float32)
    _send_decode(parent, 1, "you <audio>", audio, draft=[5, 2, 6], key="k:0", priority=3)
    assert _recv(parent) == ("res", 1, "800 5 6", [800, 5, 6], {"finish_reason": "stop"})
    request = engine.requests[0]
    assert request.audio.dtype == np.float32 and np.array_equal(request.audio, audio)
    assert (request.head, request.tail) == engine._split("you <audio>")
    assert (request.draft, request.key, request.max_new, request.priority) == ([5, 6], "k:0", 32, 3)   # no stop id
    assert request.on_tokens is None

    _send_decode(parent, 2, "no placeholder", audio)                # the caller gets the engine's own error
    assert _recv(parent) == ("err", 2, "ValueError", "prompt must contain exactly one audio placeholder")
    _send_decode(parent, 3, "<audio>fail", audio)
    assert _recv(parent) == ("err", 3, "PoolExhausted", "kitten-asr: KV pool exhausted")

    parent.send(("snap", -4))
    assert _recv(parent) == ("rpc", -4, {"attention": "torch", "page": 16})
    parent.send(("prompt", -5, "English"))
    assert _recv(parent) == ("rpc", -5, "<audio>English")
    parent.send(("prompt", -6, "Klingon"))                          # a failing call is answered, not fatal
    assert _recv(parent) == ("err", -6, "ValueError", "no prompt for Klingon")
    engine.stopped = True
    parent.send(("snap", -7))
    assert _recv(parent) == ("err", -7, "EngineStopped", "kitten-asr: the scheduler thread has stopped")
    parent.send(("rel", "k:0"))
    _until(lambda: engine.released == ["k:0"], "the release")

    assert not run.done.is_set()
    parent.send(("stop",))
    assert run.done.wait(5) and (run.code, run.error) == (0, None)


def test_engine_process_progress_stop_and_cancel(piped, child_env):
    parent = piped.parent
    run = piped.start()
    assert _recv(parent)[0] == "ready"
    engine = child_env.engines[0]
    audio = np.zeros(800, np.float32)

    _send_decode(parent, 1, "<audio>wait", audio, progress=True)
    assert _recv(parent) == ("prog", 1, [7])
    parent.send(("abort", 1))                                       # the caller's callback asked to stop
    assert _recv(parent) == ("prog", 1, [7, 8])
    assert _recv(parent) == ("res", 1, "7 8", [7, 8], {"finish_reason": "abort"})
    assert engine.requests[0].stops == [False, True]

    _send_decode(parent, 2, "<audio>hold", audio)
    _until(lambda: len(engine.requests) == 2, "the second request")
    parent.send(("cancel", 2))                                      # the caller timed out
    _until(lambda: engine.requests[1].cancelled, "the cancel")
    parent.send(("cancel", 99))                                     # an unknown request: ignored
    parent.send(("abort", 99))
    engine.go.set()
    assert _recv(parent)[:2] == ("res", 2)

    parent.close()                                                  # the calling process is gone: the engine ends
    assert run.done.wait(5) and (run.code, run.error) == (0, None)


@pytest.mark.parametrize("where, error", [
    ("require", ImportError("Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch")),
    ("require", RuntimeError("Kitten ASR needs an NVIDIA GPU of the Ampere generation or newer")),
    ("engine", ValueError("max_seq (100) must be a multiple of the page size (16)")),
    ("engine", KeyError("layers")),
])
def test_engine_process_reports_why_it_could_not_start(piped, child_env, where, error):
    setattr(child_env, "fail_" + where, error)
    run = piped.start()
    assert _recv(piped.parent) == ("fail",) + error_reply(error)
    assert run.done.wait(5) and run.error is error and run.code == 1
    assert child_env.engines == [] and len(child_env.built) == (1 if where == "engine" else 0)


def test_engine_process_is_not_run_by_hand(monkeypatch, capsys):
    message = "kitten-asr: this module is the engine process that EngineProcess starts; it is not run by hand\n"
    for argv, key in ((["_child", "addr"], None), (["_child"], "00"), (["_child", "addr", "more"], "00")):
        monkeypatch.setattr(sys, "argv", argv)
        if key is None:
            monkeypatch.delenv(_proc.AUTHKEY_ENV, raising=False)
        else:
            monkeypatch.setenv(_proc.AUTHKEY_ENV, key)
        with pytest.raises(SystemExit) as e:
            _child.main()
        assert e.value.code == 2 and capsys.readouterr().err == message


# ---------------------------------------------------------------- both sides, over the real local socket
@pytest.fixture
def spawned(child_env, monkeypatch):
    """EngineProcess(...) starts its engine process as a thread of this process (see child_env) instead of a
    new Python. mode: "run" (the engine process), "exit" (it ended at once with code 3), "hang" (it never
    connects), "drop" (it connects, takes the model name and ends with code 5). launches: what was started."""
    state = types.SimpleNamespace(mode="run", launches=[])

    class Popen:
        def __init__(self, args, env=None, **kwargs):
            self.args, self.env, self.kwargs = list(args), env, kwargs
            self.returncode, self.run = None, None
            state.launches.append(self)
            address, key = self.args[-1], env[_proc.AUTHKEY_ENV]
            if state.mode == "run":
                self.run = child_env.start(address=address, key=key)
            elif state.mode == "exit":
                self.returncode = 3
            elif state.mode == "drop":
                threading.Thread(target=self._drop, args=(address, key), daemon=True).start()

        def _drop(self, address, key):
            conn = multiprocessing.connection.Client(address, authkey=bytes.fromhex(key))
            conn.recv()
            self.returncode = 5
            conn.close()

        def poll(self):
            if self.returncode is None and self.run is not None and self.run.done.is_set():
                self.returncode = self.run.code
            return self.returncode

        def wait(self, timeout=None):
            if self.poll() is None and self.run is not None:
                self.run.done.wait(timeout)
            if self.poll() is None:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return self.returncode

        def kill(self):
            if self.poll() is None:
                self.returncode = -9

    monkeypatch.setattr(_proc, "subprocess",
                        types.SimpleNamespace(Popen=Popen, TimeoutExpired=subprocess.TimeoutExpired))
    return state


@posix
def test_process_starts_serves_and_stops(spawned, child_env):
    p = _proc.EngineProcess("model-dir", start_timeout_s=20, device="cuda:0", max_batch=4)
    try:
        launch, = spawned.launches
        address = launch.args[3]
        assert launch.args == [sys.executable, "-m", "kitten_inference.asr._child", address]
        assert launch.kwargs == {}                                  # no working folder, nothing else changed
        assert set(launch.env) - set(os.environ) == {_proc.AUTHKEY_ENV}
        assert launch.env.get("PYTHONPATH") == os.environ.get("PYTHONPATH")
        assert len(bytes.fromhex(launch.env[_proc.AUTHKEY_ENV])) == 32
        assert os.path.basename(os.path.dirname(address)).startswith("kitten-asr-")
        assert not os.path.exists(os.path.dirname(address))        # the socket's folder is gone once connected
        assert child_env.required == [("asr-process", None), ("asr", "cuda:0")]
        engine = child_env.engines[0]
        assert (engine.model, engine.opts) == ("model-dir", {"device": "cuda:0", "max_batch": 4})
        assert p.engine_pid == os.getpid() and p.opts == {"device": "cuda:0", "max_batch": 4}

        audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
        out = p.decode(audio[::2], "<audio>", draft_ids=[9, 2], stream_key="s:0", timeout_s=10)     # a view
        assert (out.text, out.token_ids, out.info) == ("800 9", [800, 9], {"finish_reason": "stop"})
        assert np.array_equal(engine.requests[0].audio, audio[::2]) and engine.requests[0].key == "s:0"
        with pytest.raises(ValueError, match="prompt must contain exactly one audio placeholder") as e:
            p.decode(audio, "no placeholder", timeout_s=10)
        assert type(e.value) is ValueError                          # the engine's class, not a wrapped one
        with pytest.raises(PoolExhausted, match="kitten-asr: KV pool exhausted"):
            p.decode(audio, "<audio>fail", timeout_s=10)

        seen = []
        out = p.decode(audio, "<audio>wait", timeout_s=10, on_tokens=lambda ids: seen.append(ids) or True)
        assert seen == [[7]] and out.info["finish_reason"] == "abort" and engine.requests[-1].stops == [False, True]

        assert p.default_prompt("English") == "<audio>English"
        assert p.snapshot() == {"attention": "torch", "page": 16, "engine_pid": os.getpid()}
        p.release("s:0")
        _until(lambda: engine.released == ["s:0"], "the release")
    finally:
        p.close()
    assert launch.run.done.wait(5) and (launch.run.code, launch.run.error) == (0, None)
    with pytest.raises(EngineStopped, match="kitten-asr: the engine is closed"):
        p.decode(audio, "<audio>", timeout_s=10)
    assert p.release("s:0") is None
    p.close()


@posix
def test_engine_process_crash_fails_the_waiting_call(spawned, child_env):
    p = _proc.EngineProcess("model-dir", start_timeout_s=20)
    try:
        engine, run = child_env.engines[0], spawned.launches[0].run
        errors = []

        def decode():
            try:
                p.decode(np.zeros(800, np.float32), "<audio>hold", timeout_s=20)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
        thread = threading.Thread(target=decode, daemon=True)
        thread.start()
        _until(lambda: len(engine.requests) == 1, "the request")
        # The engine process dies: the system hangs up its end of the connection. (Closing the object would
        # not do it here, where the "process" is a thread that is still reading from it.)
        end = socket.socket(fileno=os.dup(run.conn.fileno()))
        end.shutdown(socket.SHUT_RDWR)
        end.close()
        thread.join(5)
        assert not thread.is_alive() and len(errors) == 1
        assert type(errors[0]) is EngineStopped and str(errors[0]) == "kitten-asr: engine process exited"
        with pytest.raises(EngineStopped, match="kitten-asr: engine process exited"):
            p.snapshot()
    finally:
        p.close()


@posix
@pytest.mark.parametrize("where, error, expected, message", [
    ("require", ImportError("Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch"),
     ImportError, "Kitten ASR needs PyTorch, which is not installed. Install it: pip install torch"),
    ("require", RuntimeError("Kitten ASR needs an NVIDIA GPU. Running on the CPU is not supported."),
     RuntimeError, "Kitten ASR needs an NVIDIA GPU. Running on the CPU is not supported."),
    ("engine", FileNotFoundError("model folder not found: ./tiny"), FileNotFoundError,
     "model folder not found: ./tiny"),
    ("engine", KeyError("layers"), EngineError, "kitten-asr: engine process failed to start: KeyError: 'layers'"),
])
def test_start_failure_arrives_with_the_engines_own_words(spawned, child_env, where, error, expected, message):
    setattr(child_env, "fail_" + where, error)
    with pytest.raises(expected) as e:
        _proc.EngineProcess("model-dir", start_timeout_s=20)
    assert type(e.value) is expected and str(e.value) == message
    assert spawned.launches[0].returncode is not None              # nothing is left running


@posix
def test_engine_process_that_ends_before_connecting(spawned):
    spawned.mode = "exit"
    with pytest.raises(EngineError) as e:
        _proc.EngineProcess("model-dir", start_timeout_s=20)
    assert str(e.value) == "kitten-asr: engine process exited with code 3 while connecting"


@posix
def test_engine_process_that_never_connects(spawned):
    spawned.mode = "hang"
    started = time.monotonic()
    with pytest.raises(TimeoutError) as e:
        _proc.EngineProcess("model-dir", start_timeout_s=0.2)
    assert str(e.value) == "kitten-asr: engine process not ready after 0.2s (connecting)"
    assert time.monotonic() - started < 10 and spawned.launches[0].returncode == -9       # and it was killed


@posix
def test_engine_process_that_ends_while_loading(spawned):
    spawned.mode = "drop"
    with pytest.raises(EngineError) as e:
        _proc.EngineProcess("model-dir", start_timeout_s=20)
    assert str(e.value) == "kitten-asr: engine process exited with code 5 while loading the model"
