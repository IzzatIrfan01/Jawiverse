"""Small shared helpers used by train_jawi_rumi.py, infer_jawi_rumi.py, and
telegram_bot.py. Not a framework -- just the bits that would otherwise be
copy-pasted across all three standalone scripts.
"""
import json
import os
import re
from pathlib import Path

import torch

VERSION_DIR_RE = re.compile(r"^v(\d+)$")


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


def list_models(base_dir: Path) -> dict:
    """Map each dataset folder to its trained versions, oldest first, e.g.
    {"nirnama": ["v1"], "risalah kaum muslimin": ["v1", "v2"]}. A version only
    counts once training finished, i.e. it has final/adapter_config.json."""
    models = {}
    for folder in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        versions = [d for d in folder.iterdir()
                    if VERSION_DIR_RE.match(d.name) and (d / "final" / "adapter_config.json").exists()]
        if versions:
            models[folder.name] = [d.name for d in sorted(versions, key=lambda d: int(d.name[1:]))]
    return models


def format_models(base_dir: Path) -> str:
    models = list_models(base_dir)
    if not models:
        return "  (no trained models found)"
    return "\n".join(f"  {name}: {', '.join(versions)}" for name, versions in models.items())


def resolve_model(spec: str, base_dir: Path) -> Path:
    """Turn a --model value into an adapter directory.

    "risalah" -> newest version of the dataset folder starting with "risalah"
    (so "risalah kaum muslimin/vN/final"); "risalah:v1" or "risalah:1" -> that
    version. Anything that is already an adapter directory is used as-is, so
    a checkpoint path still works.
    """
    for candidate in (Path(spec), base_dir / spec):
        if (candidate / "adapter_config.json").exists():
            return candidate

    name, _, version = spec.partition(":")
    models = list_models(base_dir)
    matches = [folder for folder in models if folder.lower() == name.lower()] or \
              [folder for folder in models if folder.lower().startswith(name.lower())]
    if len(matches) != 1:
        problem = "matches several models" if matches else "matches no trained model"
        raise SystemExit(f"--model '{spec}' {problem}. Available:\n{format_models(base_dir)}")
    folder = matches[0]

    if not version:
        version = models[folder][-1]
    elif not version.startswith("v"):
        version = f"v{version}"
    if version not in models[folder]:
        raise SystemExit(f"'{folder}' has no trained {version}. Available:\n{format_models(base_dir)}")
    return base_dir / folder / version / "final"


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
