#!/usr/bin/env python3
"""Chat with the fine-tuned Jawi<->Rumi model through Telegram.

Send it Jawi or Rumi text and it replies with the transliteration. Direction
defaults to auto-detect (Arabic-script input -> jawi2rumi, otherwise
rumi2jawi); override with /jawi2rumi or /rumi2jawi to lock a direction for
your chat, or /auto to go back to auto-detect.

Reuses the same model loading / generation code as infer_jawi_rumi.py --
this file only adds the Telegram polling loop on top.

This is a simple long-polling bot meant for interactive testing (matching
how you'd use infer_jawi_rumi.py --interactive, just from your phone), not a
hardened production service: no retries/rate-limiting beyond what's needed
to not crash on a bad message.

Setup:
    cd jawiverse
    cp .env.example .env   # fill in TG_BOT_TOKEN and TG_CHAT_ID

Run:
    python telegram_bot.py --gpu-id 1

By default only the chat ID in TG_CHAT_ID (from .env) is allowed to talk to
the bot, so a leaked bot username can't be used by strangers to run
inference on your GPU. Pass --allow-all to disable that check, or set
ALLOWED_CHAT_IDS="id1,id2" in .env to allow a specific list of chat IDs.
"""
import argparse
import os
import platform
import re
import time
from pathlib import Path

import torch

from common import load_dotenv_file, resolve_device
from infer_jawi_rumi import DIRECTIONS, load_model, transliterate

API_BASE = "https://api.telegram.org/bot{token}/{method}"

# Arabic script blocks used by Jawi text (incl. presentation forms), vs. plain Latin for Rumi.
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

HELP_TEXT = (
    "Send me Jawi or Rumi text and I'll transliterate it.\n\n"
    "Commands:\n"
    "/auto - auto-detect direction from script (default)\n"
    "/jawi2rumi - lock this chat to Jawi -> Rumi\n"
    "/rumi2jawi - lock this chat to Rumi -> Jawi\n"
    "/help - show this message"
)


def detect_direction(text: str) -> str:
    return "jawi2rumi" if ARABIC_SCRIPT_RE.search(text) else "rumi2jawi"


def describe_model_path(model_path: Path, base_dir: Path) -> str:
    """Human-readable model label built from the path, e.g. 'nirnama (final)' or
    'nirnama (checkpoint-4072)' for .../nirnama/v1/final or
    .../nirnama/v1/checkpoints/checkpoint-4072 -- instead of just the trailing
    directory name ('final', which is the same for every dataset)."""
    model_path = model_path.resolve()
    leaf = model_path.name

    try:
        rel_parts = model_path.relative_to(base_dir.resolve()).parts
    except ValueError:
        rel_parts = None

    dataset_name = rel_parts[0] if rel_parts and len(rel_parts) > 1 else model_path.parent.name

    if not dataset_name or dataset_name == leaf:
        return leaf
    return f"{dataset_name} ({leaf})"


def get_device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_properties(0).name
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.machine() or "unknown CPU"


def log_exchange(chat_id, direction: str, input_text: str, output_text: str,
                  elapsed: float, input_tokens: int, output_tokens: int):
    speed = output_tokens / elapsed if elapsed > 0 else 0.0
    divider = "-" * 64
    print(
        f"\n{divider}\n"
        f"[{time.strftime('%H:%M:%S')}] chat={chat_id} | {direction}\n"
        f"IN  ({input_tokens} tok, {len(input_text)} chars): {input_text}\n"
        f"OUT ({output_tokens} tok, {len(output_text)} chars): {output_text}\n"
        f"time={elapsed:.2f}s | speed={speed:.1f} tok/s\n"
        f"{divider}"
    )


def get_updates(token: str, offset=None, timeout: int = 30):
    url = API_BASE.format(token=token, method="getUpdates")
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    import requests

    resp = requests.get(url, params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json()["result"]


def send_message(token: str, chat_id, text: str):
    import requests

    url = API_BASE.format(token=token, method="sendMessage")
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as e:
        print(f"sendMessage failed for chat {chat_id}: {e}")


def notify_all(token: str, allowed_chat_ids, text: str):
    if not allowed_chat_ids:
        return
    for cid in allowed_chat_ids:
        send_message(token, cid, text)


def poll_loop(token: str, model, tokenizer, args, allowed_chat_ids):
    chat_direction = {}  # chat_id -> locked direction, or absent = auto-detect
    offset = None

    while True:
        try:
            updates = get_updates(token, offset=offset, timeout=args.poll_timeout)
        except Exception as e:
            print(f"getUpdates failed: {e}")
            time.sleep(3)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message or "text" not in message:
                continue

            chat_id = message["chat"]["id"]
            text = message["text"].strip()
            if not text:
                continue

            if allowed_chat_ids is not None and chat_id not in allowed_chat_ids:
                print(f"Ignoring message from unauthorized chat_id {chat_id}")
                continue

            if text in ("/start", "/help"):
                send_message(token, chat_id, HELP_TEXT)
                continue

            if text == "/auto":
                chat_direction.pop(chat_id, None)
                send_message(token, chat_id, "Direction set to auto-detect.")
                continue

            if text.lstrip("/") in DIRECTIONS:
                chat_direction[chat_id] = text.lstrip("/")
                send_message(token, chat_id, f"Direction locked to {chat_direction[chat_id]}.")
                continue

            direction = chat_direction.get(chat_id) or detect_direction(text)

            try:
                start_time = time.time()
                result = transliterate(
                    model, tokenizer, text, direction,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.min_p,
                )
                elapsed = time.time() - start_time

                input_tokens = len(tokenizer.encode(text))
                output_tokens = len(tokenizer.encode(result))
                log_exchange(chat_id, direction, text, result, elapsed, input_tokens, output_tokens)

                send_message(token, chat_id, f"[{direction}]\n{result}")
            except Exception as e:
                print(f"Generation failed for chat {chat_id}: {e}")
                send_message(token, chat_id, f"Sorry, something went wrong: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Telegram chat interface for the Jawi<->Rumi model")
    parser.add_argument("--model-path", type=Path,
                         default=Path(__file__).resolve().parent / "nirnama" / "v1" / "final",
                         help="Path to the trained adapter directory")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=None,
                         help="Physical GPU index to use (as shown in nvidia-smi). Ignored when --device cpu.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument("--allow-all", action="store_true",
                         help="Disable the chat-ID allowlist and respond to anyone who messages the bot")
    parser.add_argument("--poll-timeout", type=int, default=30, help="Telegram long-poll timeout in seconds")
    return parser.parse_args()


def main():
    base_dir = Path(__file__).resolve().parent
    load_dotenv_file(base_dir / ".env")

    args = parse_args()

    token = os.environ.get("TG_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TG_BOT_TOKEN is not set. Copy .env.example to .env and fill in your bot token, "
            "or export TG_BOT_TOKEN before running."
        )

    allowed_chat_ids = None
    if not args.allow_all:
        raw_allowlist = os.environ.get("ALLOWED_CHAT_IDS") or os.environ.get("TG_CHAT_ID")
        if raw_allowlist:
            allowed_chat_ids = {int(x.strip()) for x in raw_allowlist.split(",") if x.strip()}
            print(f"Restricting to chat ID(s): {sorted(allowed_chat_ids)}")
        else:
            print(
                "WARNING: no TG_CHAT_ID / ALLOWED_CHAT_IDS set and --allow-all not passed -- "
                "the bot will respond to anyone who finds it. Set TG_CHAT_ID in .env to restrict access."
            )

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model(args.model_path, device, args.max_seq_length)

    device_name = get_device_name(device)
    model_label = describe_model_path(args.model_path, base_dir)
    startup_msg = (
        f"🤖 Jawiverse bot online\n"
        f"Device: {device.upper()} ({device_name})\n"
        f"Model: {model_label}"
    )
    print(f"Model loaded. {startup_msg.splitlines()[1]}. Starting Telegram bot (Ctrl+C to stop).")

    if allowed_chat_ids:
        notify_all(token, allowed_chat_ids, startup_msg)
    else:
        print("Skipping startup Telegram notification (--allow-all / no TG_CHAT_ID configured).")

    try:
        poll_loop(token, model, tokenizer, args, allowed_chat_ids)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
        notify_all(token, allowed_chat_ids, "🛑 Jawiverse bot stopping (manual stop).")
        raise
    except Exception as e:
        print(f"Bot crashed: {e}")
        notify_all(token, allowed_chat_ids, f"⚠️ Jawiverse bot crashed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
