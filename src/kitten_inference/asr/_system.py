"""The system message of every decode prompt, chosen from the model folder.

A model folder may set it in ``asr_config.json`` (key ``system_prompt``). Every other
model gets an empty system message, as before.

``system_prompt="auto"`` (the default) follows the folder; ``"none"`` always sends an empty system message.
Light on purpose: no PyTorch, no engine library.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

SYSTEM_PROMPT_FILE = "asr_config.json"
SYSTEM_PROMPT_KEY = "system_prompt"
MODES = ("auto", "none")
# The files both engines download for a Hugging Face model name (a file the repo does not have is skipped).
HUB_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model*.safetensors",
             "model.safetensors.index.json", "preprocessor_config.json", "chat_template.jinja", SYSTEM_PROMPT_FILE]


def check_mode(mode) -> str:
    """The mode in its canonical form, or ValueError. None means "auto"."""
    if mode is None:
        return "auto"
    if not isinstance(mode, str) or mode.strip().lower() not in MODES:
        raise ValueError(f"system_prompt must be one of {', '.join(MODES)}, not {mode!r}")
    return mode.strip().lower()


def resolve(folder: str, mode="auto") -> Tuple[str, str]:
    """(system text, where it came from) for the model in `folder`.

    Sources: "none" (turned off), "auto:no-file" (the folder has no such file: empty text), "auto:<file name>".
    A file that is not JSON, or whose value is not a string, raises ValueError instead of running the model with
    a prompt it does not expect.
    """
    mode = check_mode(mode)
    if mode == "none":
        return "", "none"
    path = os.path.join(folder, SYSTEM_PROMPT_FILE)
    if not os.path.isfile(path):
        return "", "auto:no-file"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} could not be read as JSON: {exc}") from exc
    value = data.get(SYSTEM_PROMPT_KEY) if isinstance(data, dict) else None
    if not isinstance(value, str):
        raise ValueError(f"{path}: {SYSTEM_PROMPT_KEY} must be a string (pass system_prompt='none' to run "
                         "without it)")
    return value, "auto:" + SYSTEM_PROMPT_FILE
