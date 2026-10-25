"""Small shared helpers used by train_jawi_rumi.py, infer_jawi_rumi.py, and
telegram_bot.py. Not a framework -- just the bits that would otherwise be
copy-pasted across all three standalone scripts.
"""
import os
from pathlib import Path

import torch


def load_dotenv_file(path: Path):
    """Load KEY=VALUE lines from a local .env file into os.environ.

    Deliberately not the python-dotenv package to avoid a new dependency for
    a few lines. Existing environment variables always win (never overridden),
    so this only fills in gaps. The .env file itself must never be committed
    -- see .env.example and .gitignore.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_device(requested: str) -> str:
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    return device
