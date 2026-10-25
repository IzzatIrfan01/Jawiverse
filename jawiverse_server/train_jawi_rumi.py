#!/usr/bin/env python3
"""Fine-tune Llama-3.2-3B-Instruct for Jawi<->Rumi transliteration.

Standalone replacement for TrainingModel(Unsloth).ipynb, meant to run on a
dedicated server (RTX 3090 / RTX 6000 Pro Blackwell / etc.) instead of a
Colab T4 session. Auto-detects CUDA vs CPU:

  - CUDA: uses Unsloth's FastLanguageModel with 4-bit loading (fast path).
  - CPU:  falls back to plain transformers + PEFT LoRA in fp32. This has no
          Unsloth kernels or 4-bit quantization (both require CUDA) and is
          only realistic for a small smoke test, not the full dataset.

Checkpointing uses the Trainer's own save_steps/save_total_limit plus
resume_from_checkpoint — there is no segmented "stop after N steps" logic
here, since that was purely a workaround for Colab's session time limit.
On your own server, just leave it running; if the process dies, rerun the
script and it will resume from the last checkpoint automatically.

Telegram progress notifications are optional: set TG_BOT_TOKEN and
TG_CHAT_ID as environment variables to enable them (never hardcode
credentials in this file).

Example:
    python train_jawi_rumi.py --device cuda
    python train_jawi_rumi.py --device cpu --num-train-epochs 1 --per-device-train-batch-size 1

    # Risalah, continuing from the finished nirnama adapter
    python train_jawi_rumi.py --dataset risalah --init-adapter nirnama/v1/final --gpu-id 1
"""
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch

from common import full_precision_base_name, load_dotenv_file, read_base_model_name, resolve_device


#   folder : <base-dir>/<folder>/dataset/<file> in, <base-dir>/<folder>/v1/ out
#   files  : (Excel file, whether eval holdout pairs may be drawn from it,
#             whether long pairs also get split into extra clause pairs)
#   train_max_seq_length : default for --train-max-seq-length; examples longer
#             than this are truncated, cutting off the end of the answer
DATASETS = {
    "nirnama": {
        "folder": "nirnama",
        "files": [("jawi_rumi_v1.xlsx", False, False), ("jawi_rumi_parallel_v1.xlsx", False, False)],
        "json": "bitransliteration_nirmala_model_v1.json",
        "eval_holdout_ratio": 0.0,
        "train_max_seq_length": 256,
    },
    "risalah": {
        "folder": "risalah kaum muslimin",
        # Holdout only from the sentence file -- single words make a weak eval set
        "files": [("Risalah Untuk Kaum Muslimin.xlsx", True, True), ("Risalah_word_dataset.xlsx", False, False)],
        "json": "bitransliteration_risalah_model_v1.json",
        "eval_holdout_ratio": 0.1,
        # Sentence pairs run up to ~880 tokens; 256 would truncate ~30% of them
        "train_max_seq_length": 1024,
    },
}

ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_SCRIPT = re.compile(r"[A-Za-z]")
# Arabic Presentation Forms (minus the ornate brackets U+FD3E/U+FD3F). Quran
# verses pasted in a Quran font come through as runs of these glyph codes
# instead of real text, with an empty "( )" or OCR-garbled Rumi side.
QURAN_FONT_GLYPH = re.compile(r"[\ufb50-\ufd3d\ufd40-\ufdff\ufe70-\ufeff]")


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"[\u202a-\u202e\u200e\u200f]", "", str(text))
    return text.strip()


def pair_problem(jawi_text: str, rumi_text: str):
    """Reason to skip a Jawi/Rumi pair, or None if it's usable for training."""
    if not jawi_text or not rumi_text:
        return "incomplete"
    if len(QURAN_FONT_GLYPH.findall(jawi_text)) >= 3:
        return "quran_font_glyphs"
    if not ARABIC_SCRIPT.search(jawi_text):
        return "jawi_not_arabic_script"
    if not LATIN_SCRIPT.search(rumi_text) or ARABIC_SCRIPT.search(rumi_text):
        return "rumi_not_latin_script"
    return None


# Split after clause-ending punctuation (kept on the clause) followed by whitespace
CLAUSE_END = re.compile(r"(?<=[.;:!?؛؟])\s+")
JAWI_TO_RUMI_PUNCT = str.maketrans({"؛": ";", "؟": "?"})  # Arabic ؛ ؟
BRACKET_PAIRS = ["()", "[]", "«»", "﴾﴿"]  # ( ) [ ] « » ﴾ ﴿


def split_clauses(jawi_text: str, rumi_text: str) -> list:
    """Split a pair into clause pairs at . ; : ! ? (each clause keeps its own
    punctuation). Returns [] -- keep the pair whole only -- unless both sides
    split at the same sequence of marks, every clause has balanced brackets
    and quotes, a plausible Jawi/Rumi word ratio, and passes pair_problem()."""
    jawi_parts = CLAUSE_END.split(jawi_text)
    rumi_parts = CLAUSE_END.split(rumi_text)
    if len(jawi_parts) < 2 or len(jawi_parts) != len(rumi_parts):
        return []
    jawi_marks = [part[-1].translate(JAWI_TO_RUMI_PUNCT) for part in jawi_parts[:-1]]
    if jawi_marks != [part[-1] for part in rumi_parts[:-1]]:
        return []

    clauses = []
    for jawi_part, rumi_part in zip(jawi_parts, rumi_parts):
        if any(s.count(o) != s.count(c) for s in (jawi_part, rumi_part) for o, c in BRACKET_PAIRS):
            return []
        jawi_words, rumi_words = len(jawi_part.split()), len(rumi_part.split())
        if abs(jawi_words - rumi_words) > 1 and not 0.6 <= rumi_words / jawi_words <= 1.6:
            return []
        if pair_problem(jawi_part, rumi_part):
            return []
        clauses.append((jawi_part, rumi_part))
    return clauses


def build_conversations(input_files, output_json, exclude_sheets=None,
                        eval_holdout_ratio=0.0, holdout_xlsx=None, skipped_csv=None):
    """input_files: list of (xlsx path, holdout allowed, clause split) tuples."""
    exclude_sheets = exclude_sheets or []
    conversations = []
    pairs, skipped, seen = [], [], set()

    for input_xlsx, holdout_allowed, clause_split in input_files:
        print(f"Reading file: {input_xlsx}")
        all_sheets = pd.read_excel(input_xlsx, sheet_name=None)

        for sheet_name, df in all_sheets.items():
            if sheet_name in exclude_sheets:
                print(f"Skipping sheet: {sheet_name}")
                continue

            print(f"Processing sheet: {sheet_name} ({len(df)} rows)")
            sheet_skipped = 0

            for idx, row in df.iterrows():
                jawi_text = clean_text(row.get("Jawi", ""))
                rumi_text = clean_text(row.get("Rumi", ""))
                excel_row = idx + 2  # +1 header row, +1 for 1-based numbering

                reason = pair_problem(jawi_text, rumi_text)
                if reason is None and (jawi_text, rumi_text) in seen:
                    reason = "duplicate"
                if reason:
                    sheet_skipped += 1
                    skipped.append({"File": Path(input_xlsx).name, "Sheet": sheet_name, "ExcelRow": excel_row,
                                    "Reason": reason, "Jawi": jawi_text, "Rumi": rumi_text})
                    continue

                seen.add((jawi_text, rumi_text))
                pairs.append({"File": Path(input_xlsx).name, "Sheet": sheet_name, "ExcelRow": excel_row,
                              "Jawi": jawi_text, "Rumi": rumi_text,
                              "holdout_allowed": holdout_allowed, "clause_split": clause_split})

            if sheet_skipped:
                print(f"  Skipped {sheet_skipped} rows in '{sheet_name}'")

    # Fixed-seed holdout: these pairs never enter the training JSON, so
    # evaluating on them measures generalization rather than memorization.
    holdout_ids = set()
    if eval_holdout_ratio > 0:
        candidates = [i for i, p in enumerate(pairs) if p["holdout_allowed"]]
        n_holdout = round(len(candidates) * eval_holdout_ratio)
        holdout_ids = set(random.Random(3407).sample(candidates, n_holdout))

    train_pairs = [(p["Jawi"], p["Rumi"]) for i, p in enumerate(pairs) if i not in holdout_ids]

    # Extra shorter pairs cut from training pairs only (never from holdout
    # ones, so no piece of an eval sentence leaks into training)
    n_split_sources, clause_pairs = 0, []
    for i, pair in enumerate(pairs):
        if i in holdout_ids or not pair["clause_split"]:
            continue
        clauses = split_clauses(pair["Jawi"], pair["Rumi"])
        n_split_sources += bool(clauses)
        for clause in clauses:
            if clause not in seen:
                seen.add(clause)
                clause_pairs.append(clause)
    if clause_pairs:
        print(f"Clause splits: +{len(clause_pairs)} pairs from {n_split_sources} sentence pairs")
    train_pairs += clause_pairs

    for jawi_text, rumi_text in train_pairs:
        conversations.append({
            "conversations": [
                {"role": "system", "content": "You are an expert AI that transliterates Classical Jawi into Rumi accurately."},
                {"role": "user", "content": f"<jawi2rumi> {jawi_text}"},
                {"role": "assistant", "content": rumi_text},
            ]
        })
        conversations.append({
            "conversations": [
                {"role": "system", "content": "You are an expert AI that transliterates Rumi into Classical Jawi accurately."},
                {"role": "user", "content": f"<rumi2jawi> {rumi_text}"},
                {"role": "assistant", "content": jawi_text},
            ]
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(conversations)} examples ({len(train_pairs)} pairs x 2 directions) "
          f"saved -> {output_json}")

    if holdout_ids and holdout_xlsx:
        holdout_df = pd.DataFrame([pairs[i] for i in sorted(holdout_ids)]).drop(
            columns=["holdout_allowed", "clause_split"])
        holdout_df.to_excel(holdout_xlsx, sheet_name="holdout", index=False)
        print(f"Eval holdout: {len(holdout_ids)} pairs kept out of training -> {holdout_xlsx}")

    if skipped:
        skipped_df = pd.DataFrame(skipped)
        print(f"Total skipped rows: {len(skipped)}")
        for reason, count in skipped_df["Reason"].value_counts().items():
            print(f"  {reason}: {count}")
        if skipped_csv:
            # utf-8-sig so Excel shows the Jawi text correctly
            skipped_df.to_csv(skipped_csv, index=False, encoding="utf-8-sig")
            print(f"Skipped rows listed in: {skipped_csv}")

    return output_json


def telegram_notify(message: str):
    token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not token or not chat_id:
        return
    import requests

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "disable_web_page_preview": True}
    for attempt in range(2):
        try:
            requests.post(url, json=payload, timeout=5)
            return
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
            else:
                print(f"Telegram notify failed: {e}")


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} hr"


def make_eta_callback_cls():
    from transformers import TrainerCallback

    class TelegramETACallback(TrainerCallback):
        def __init__(self, notify_every_steps=100):
            self.notify_every_steps = notify_every_steps
            self.start_time = None
            self.last_notified_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step - self.last_notified_step < self.notify_every_steps:
                return
            elapsed = time.time() - self.start_time
            sec_per_step = elapsed / max(1, step)
            remaining = state.max_steps - step
            eta_sec = remaining * sec_per_step
            telegram_notify(
                f"Training update\nStep: {step}/{state.max_steps}\n"
                f"Elapsed: {elapsed / 60:.1f} min\nETA: {format_eta(eta_sec)}\n"
                f"Speed: {1 / sec_per_step:.2f} it/s"
            )
            self.last_notified_step = step

    return TelegramETACallback


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_model_and_tokenizer(args, device: str):
    if device == "cuda":
        from unsloth import FastLanguageModel

        # With --init-adapter, Unsloth loads the adapter's recorded base model
        # plus the existing LoRA weights (trainable), so training continues
        # from them instead of starting a fresh LoRA.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.init_adapter) if args.init_adapter else args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=not args.no_4bit,
        )
        if not args.init_adapter:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    print(
        "WARNING: running on CPU. Unsloth's fused kernels and 4-bit "
        "quantization require CUDA, so this path uses plain transformers + "
        "PEFT in fp32. Fine-tuning a 3B model this way is only realistic for "
        "a small smoke test, not the full dataset."
    )
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.init_adapter) if args.init_adapter else args.base_model)
    if tokenizer.pad_token is None:
        # Common Llama gotcha: some checkpoints ship with no pad token defined,
        # which crashes DataCollatorForSeq2Seq(padding=True) later. Unsloth
        # patches this automatically on the CUDA path; here we do it by hand.
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_adapter:
        base_model_name = full_precision_base_name(read_base_model_name(args.init_adapter))
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
        return PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True), tokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the Jawi<->Rumi transliteration model")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                         help="jawiverse root; dataset/output paths are derived from this")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nirnama",
                         help="Which dataset to train on; also decides the output folder (<dataset folder>/v1/)")
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--init-adapter", type=Path, default=None,
                         help="Continue fine-tuning from an existing LoRA adapter (e.g. nirnama/v1/final) "
                              "instead of a fresh one. --base-model/--lora-r/--lora-alpha are then ignored.")
    parser.add_argument("--eval-holdout-ratio", type=float, default=None,
                         help="Fraction of eligible pairs kept out of training for evaluation "
                              "(default: per dataset, nirnama 0, risalah 0.1). Only applied when the "
                              "dataset JSON is (re)built -- pass --rebuild-dataset after changing it.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=None,
                         help="Physical GPU index to use (as shown in nvidia-smi), e.g. 0 for the Blackwell, "
                              "1 for the 3090. Sets CUDA_VISIBLE_DEVICES. Ignored when --device cpu.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--train-max-seq-length", type=int, default=None,
                         help="Truncate training examples to this many tokens (default: per dataset, "
                              "nirnama 256, risalah 1024)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading on CUDA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--num-checkpoint-segments", type=int, default=5,
                         help="Roughly how many save points across the whole run")
    parser.add_argument("--rebuild-dataset", action="store_true",
                         help="Rebuild the conversations JSON from the Excel files even if it already exists")
    return parser.parse_args()


def main():
    load_dotenv_file(Path(__file__).resolve().parent / ".env")

    args = parse_args()

    if args.gpu_id is not None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(
                f"--gpu-id {args.gpu_id} overrides the existing "
                f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = resolve_device(args.device)

    print(f"Using device: {device}")
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} | VRAM: {round(gpu.total_memory / 1024 ** 3, 2)} GB")

    dataset_cfg = DATASETS[args.dataset]
    base_dir = args.base_dir
    dataset_dir = base_dir / dataset_cfg["folder"] / "dataset"
    output_dir = base_dir / dataset_cfg["folder"] / "v1"
    checkpoint_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final"
    output_json = output_dir / dataset_cfg["json"]
    if args.train_max_seq_length is None:
        args.train_max_seq_length = dataset_cfg["train_max_seq_length"]
    print(f"Dataset: {args.dataset} | output: {output_dir} | train max seq length: {args.train_max_seq_length}")

    if (final_dir / "adapter_model.safetensors").exists():
        print(f"Training already completed. Final model at: {final_dir}")
        return

    if args.init_adapter and not (args.init_adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"--init-adapter {args.init_adapter} has no adapter_model.safetensors")

    if args.rebuild_dataset or not output_json.exists():
        eval_holdout_ratio = (dataset_cfg["eval_holdout_ratio"] if args.eval_holdout_ratio is None
                              else args.eval_holdout_ratio)
        build_conversations(
            input_files=[(dataset_dir / name, holdout, clause_split)
                         for name, holdout, clause_split in dataset_cfg["files"]],
            output_json=output_json,
            eval_holdout_ratio=eval_holdout_ratio,
            holdout_xlsx=output_dir / "eval_holdout.xlsx",
            skipped_csv=output_dir / "skipped_rows.csv",
        )
    else:
        print(f"Reusing existing dataset: {output_json}")

    from datasets import load_dataset

    model, tokenizer = load_model_and_tokenizer(args, device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("Model has no trainable parameters -- the LoRA adapter was loaded frozen.")
    print(f"Trainable parameters: {trainable_params:,}"
          + (f" (continuing from adapter {args.init_adapter})" if args.init_adapter else ""))

    dataset = load_dataset("json", data_files=str(output_json), split="train")

    if device == "cuda":
        # Unsloth-only helpers: normalizes conversation dict shapes, and applies
        # its bundled llama-3.1 chat template. Our data is already produced in
        # {"role", "content"} shape by build_conversations(), so standardize_sharegpt
        # is a no-op here -- but both are CUDA-only imports, so skip them on CPU.
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt

        dataset = standardize_sharegpt(dataset)
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    else:
        print(
            "CPU path: using the tokenizer's own built-in chat template "
            "(Unsloth's get_chat_template/standardize_sharegpt are CUDA-only)."
        )

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset ready: {len(dataset)} examples")

    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    num_samples = len(dataset)
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(num_samples / effective_batch)
    total_steps = steps_per_epoch * args.num_train_epochs
    save_steps = max(1, total_steps // args.num_checkpoint_segments)

    print("Training plan")
    print(f"  Samples          : {num_samples}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(f"  Total steps      : {total_steps}")
    print(f"  Checkpoint every : {save_steps} steps")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    from transformers.trainer_utils import get_last_checkpoint

    def is_valid_checkpoint(path):
        if path is None:
            return False
        files = set(os.listdir(path))
        return "trainer_state.json" in files or "adapter_model.safetensors" in files

    last_ckpt = get_last_checkpoint(str(checkpoint_dir))
    resume_ckpt = last_ckpt if is_valid_checkpoint(last_ckpt) else None

    def read_previous_precision(checkpoint_path):
        """fp16/bf16 the checkpoint was originally saved under, read from its
        training_args.bin. Resuming under a different precision than the one
        that produced the checkpoint crashes: the Trainer tries to restore a
        saved fp16 GradScaler state into a bf16 run that never created one
        (accelerator.scaler is None), raising
        AttributeError: 'NoneType' object has no attribute 'load_state_dict'."""
        args_path = Path(checkpoint_path) / "training_args.bin"
        if not args_path.exists():
            return None
        prev_args = torch.load(args_path, weights_only=False)
        return {"fp16": bool(getattr(prev_args, "fp16", False)), "bf16": bool(getattr(prev_args, "bf16", False))}

    sft_config_kwargs = dict(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        logging_steps=10,
        report_to="none",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
    )
    if device == "cuda":
        sft_config_kwargs["optim"] = "adamw_8bit"
        sft_config_kwargs["bf16"] = torch.cuda.is_bf16_supported()
        sft_config_kwargs["fp16"] = not torch.cuda.is_bf16_supported()
    else:
        sft_config_kwargs["optim"] = "adamw_torch"

    if resume_ckpt:
        prev_precision = read_previous_precision(resume_ckpt)
        if prev_precision and (
            prev_precision["fp16"] != sft_config_kwargs.get("fp16", False)
            or prev_precision["bf16"] != sft_config_kwargs.get("bf16", False)
        ):
            print(
                f"Resuming a checkpoint saved with fp16={prev_precision['fp16']} "
                f"bf16={prev_precision['bf16']} (likely trained on different hardware); "
                f"overriding this run's precision to match it instead of "
                f"fp16={sft_config_kwargs.get('fp16', False)} bf16={sft_config_kwargs.get('bf16', False)}, "
                "since mixing precision modes across a resume breaks the saved GradScaler/optimizer state."
            )
            sft_config_kwargs["fp16"] = prev_precision["fp16"]
            sft_config_kwargs["bf16"] = prev_precision["bf16"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.train_max_seq_length,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        args=SFTConfig(**sft_config_kwargs),
    )

    if device == "cuda":
        # CUDA-only helper: masks the loss so only assistant response tokens
        # are trained on (not the instruction/system tokens). No CPU/plain
        # transformers equivalent is wired up here -- the CPU path is a smoke
        # test and trains on the full sequence loss instead.
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    else:
        print(
            "CPU path: training on full-sequence loss (Unsloth's "
            "train_on_responses_only response-masking is CUDA-only). Fine "
            "for a smoke test, but expect worse convergence than the GPU path."
        )

    if os.environ.get("TG_BOT_TOKEN") and os.environ.get("TG_CHAT_ID"):
        trainer.add_callback(make_eta_callback_cls()(notify_every_steps=100))

    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        telegram_notify(f"Training resumed from {os.path.basename(resume_ckpt)}")
    else:
        print("Starting training from scratch")
        telegram_notify(f"Training started ({args.dataset})\nSamples: {num_samples}\nTotal steps: {total_steps}")

    start_time = time.time()
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except Exception as e:
        telegram_notify(f"Training crashed: {str(e)[:300]}")
        raise

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed_min:.1f} min")

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final model saved to: {final_dir}")
    telegram_notify(f"Training completed. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
