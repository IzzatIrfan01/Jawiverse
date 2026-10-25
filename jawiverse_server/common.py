"""Small shared helpers used by train_jawi_rumi.py, infer_jawi_rumi.py, and
telegram_bot.py. Not a framework -- just the bits that would otherwise be
copy-pasted across all three standalone scripts.
"""
import json
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


def read_base_model_name(model_path: Path) -> str:
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {model_path}. "
            "Point --model-path at the trained adapter directory (e.g. nirnama/v1/final)."
        )
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"adapter_config.json at {adapter_config_path} has no base_model_name_or_path")
    return base_model_name


def full_precision_base_name(base_model_name: str) -> str:
    """Adapters trained with Unsloth record a pre-quantized bitsandbytes base
    (e.g. "...-unsloth-bnb-4bit"), which needs CUDA to load. The LoRA weights
    apply equally to the original full-precision model, so the CPU paths use
    that instead."""
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if base_model_name.endswith(suffix):
            return base_model_name[:-len(suffix)]
    return base_model_name
