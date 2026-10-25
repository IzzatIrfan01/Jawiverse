#!/usr/bin/env python3
"""Print the Python / torch / CUDA / GPU setup, to check an environment
before training or inference.

Package versions are read from installed metadata rather than by importing
them, so this stays fast and doesn't trigger Unsloth's patching.

Examples:
    python check_gpu.py
    CUDA_DEVICE_ORDER=PCI_BUS_ID python check_gpu.py   # number GPUs like nvidia-smi
"""
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "transformers", "peft", "accelerate", "trl", "datasets", "unsloth", "unsloth_zoo",
    "bitsandbytes", "xformers", "triton", "evaluate", "sacrebleu", "jiwer",
]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def driver_info() -> str:
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def main():
    print("== Python ==")
    print(f"  version     : {platform.python_version()}")
    print(f"  executable  : {sys.executable}")
    print(f"  in a venv   : {sys.prefix != sys.base_prefix}")

    print("\n== Environment ==")
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        print(f"  {var:21}: {os.environ.get(var, '(not set)')}")

    print("\n== NVIDIA driver (nvidia-smi: index, name, driver) ==")
    for line in driver_info().splitlines():
        print(f"  {line}")

    print("\n== torch ==")
    try:
        import torch
    except ImportError:
        print("  torch not installed")
        return

    print(f"  version     : {torch.__version__}")
    print(f"  built CUDA  : {torch.version.cuda or 'none (CPU-only build)'}")
    print(f"  cuDNN       : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'}")
    print(f"  CUDA usable : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU count   : {torch.cuda.device_count()} (torch numbering)")
        print(f"  bf16        : {torch.cuda.is_bf16_supported()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            # Tiny matmul to confirm the card actually computes, not just shows up
            try:
                x = torch.randn(256, 256, device=f"cuda:{i}")
                (x @ x).sum().item()
                status = "OK"
            except Exception as e:
                status = f"FAILED ({e})"
            print(f"  [{i}] {props.name} | compute {props.major}.{props.minor} | "
                  f"VRAM {free / 1024 ** 3:.1f} GB free / {total / 1024 ** 3:.1f} GB | test: {status}")
    else:
        print("  -> scripts will run on CPU (--device auto falls back to CPU)")

    print("\n== Packages ==")
    for name in PACKAGES:
        print(f"  {name:13}: {package_version(name)}")


if __name__ == "__main__":
    main()
