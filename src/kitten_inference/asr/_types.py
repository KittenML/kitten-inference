"""Result and error types of the ASR engine; imports nothing outside the standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DecodeResult:
    """What one decode call returns: the text, its token ids, and timings and counters in ``info``."""

    text: str
    token_ids: list
    info: dict = field(default_factory=dict)


class EngineError(RuntimeError):
    """The engine could not serve a call."""


class PoolExhausted(EngineError):
    """The key/value pool has no room for the request. The request fails; it is never returned cut short."""


class EngineStopped(EngineError):
    """The engine was closed, or its scheduler (or its process) stopped. A new engine is needed."""


# Errors that cross the process boundary by name. Anything else arrives as EngineError.
_BY_NAME = {cls.__name__: cls for cls in (
    EngineError, PoolExhausted, EngineStopped, ValueError, TypeError, ImportError, ModuleNotFoundError,
    FileNotFoundError, TimeoutError, MemoryError, RuntimeError)}


def error_reply(exc: BaseException) -> Tuple[str, str]:
    """(class name, message) of an exception, to send it to another process."""
    return type(exc).__name__, str(exc) or repr(exc)


def error_from_reply(name: str, message: str, prefix: str = "kitten-asr: ") -> Exception:
    """The exception for a (class name, message) pair: the same class with the same message when it is one
    the engine raises on purpose or a common built-in, else EngineError(prefix + class name + message)."""
    cls = _BY_NAME.get(name)
    if cls is None:
        return EngineError(f"{prefix}{name}: {message}")
    return cls(message)
