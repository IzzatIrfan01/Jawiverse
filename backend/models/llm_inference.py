import os
import sys
import threading
from pathlib import Path

# Reuse the loader/prompt logic from jawiverse_server so the API formats
# prompts exactly like training did (system prompt + Llama chat template),
# and loads correctly on both GPU (Unsloth 4-bit) and CPU (fp32).
JAWIVERSE_SERVER_DIR = Path(__file__).resolve().parents[2] / "jawiverse_server"
sys.path.insert(0, str(JAWIVERSE_SERVER_DIR))

from common import resolve_device  # noqa: E402
from infer_jawi_rumi import DIRECTIONS, load_model, transliterate  # noqa: E402

# Config via environment variables (all optional)
MODEL_PATH = os.environ.get("JAWIVERSE_MODEL_PATH", str(JAWIVERSE_SERVER_DIR / "nirnama" / "v1" / "final"))
DEVICE = os.environ.get("JAWIVERSE_DEVICE", "auto")  # auto, cuda, or cpu
GPU_ID = os.environ.get("JAWIVERSE_GPU_ID")          # physical GPU index, as in nvidia-smi
MAX_NEW_TOKENS = int(os.environ.get("JAWIVERSE_MAX_NEW_TOKENS", "512"))


class JawirumInference:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        if GPU_ID is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
        self.device = resolve_device(device)
        self.model_path = str(Path(model_path).resolve())

        print(f"[LLM] Loading model from {self.model_path} on {self.device}...")
        self.model, self.tokenizer = load_model(Path(self.model_path), self.device)
        # One model instance is shared by all requests; generate one at a time
        self._lock = threading.Lock()
        print("[LLM] Model loaded!")

    def transliterate(self, text: str, mode: str) -> str:
        """
        Translate Jawi ↔ Rumi using LLM

        Args:
            text: Input text
            mode: 'jawi2rumi' or 'rumi2jawi'

        Returns:
            Translated text
        """
        if mode not in DIRECTIONS:
            raise ValueError(f"Invalid mode: {mode}")

        # Greedy decoding (temperature 0): same input always gives the same output
        with self._lock:
            return transliterate(
                self.model, self.tokenizer, text, mode,
                max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
            )

# Load once at startup
inferencer = JawirumInference()
