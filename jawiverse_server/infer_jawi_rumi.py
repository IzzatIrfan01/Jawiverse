#!/usr/bin/env python3
"""Run / evaluate the fine-tuned Jawi<->Rumi transliteration model.

Standalone replacement for TestingModel.ipynb. Auto-detects CUDA vs CPU:

  - CUDA: loads via Unsloth's FastLanguageModel with 4-bit inference (fast path).
  - CPU:  loads via plain transformers + PEFT in fp32 (no Unsloth import at
          all on this path, since Unsloth itself pulls in CUDA-only deps at
          import time). Slower, but works anywhere.

The base model is read automatically from the adapter's own
adapter_config.json ("base_model_name_or_path"), so you only need to point
--model-path at the trained adapter directory (e.g. nirnama/v1/final).

Examples:
    # Single prompt, GPU
    python infer_jawi_rumi.py --prompt "چينتا سجاتي" --direction jawi2rumi

    # Single prompt, force CPU
    python infer_jawi_rumi.py --device cpu --prompt "Cinta sejati" --direction rumi2jawi

    # Interactive REPL
    python infer_jawi_rumi.py --interactive --direction jawi2rumi

    # Batch from a file (one input per line in a .txt, or a .json/.csv/.xlsx),
    # no ground truth needed -- just runs predictions
    python infer_jawi_rumi.py --input-file prompts.txt --direction jawi2rumi --output-file results.csv

    # Evaluate against an Excel dataset with ground truth (all sheets, 10 random samples each)
    python infer_jawi_rumi.py --eval-xlsx "risalah kaum muslimin/dataset/Risalah Untuk Kaum Muslimin.xlsx" \\
        --direction jawi2rumi --n-samples 10 --save-results
"""
import argparse
import json
import os
from pathlib import Path

import torch

from common import resolve_device

DIRECTIONS = ("jawi2rumi", "rumi2jawi")

SYSTEM_PROMPTS = {
    "jawi2rumi": "You are an expert AI that transliterates Classical Jawi into Rumi accurately.",
    "rumi2jawi": "You are an expert AI that transliterates Rumi into Classical Jawi accurately.",
}


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


def load_model_cuda(model_path: Path, max_seq_length: int):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


def load_model_cpu(model_path: Path, max_seq_length: int):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_name = read_base_model_name(model_path)
    # Adapters trained with Unsloth record a pre-quantized bitsandbytes base
    # (e.g. "...-unsloth-bnb-4bit"), which needs CUDA to load. The LoRA weights
    # apply equally to the original full-precision model, so use that instead.
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if base_model_name.endswith(suffix):
            base_model_name = base_model_name[:-len(suffix)]
            break
    print(f"CPU inference: loading base model '{base_model_name}' + adapter '{model_path}' (fp32, no Unsloth)")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()
    return model, tokenizer


def load_model(model_path: Path, device: str, max_seq_length: int = 2048):
    if device == "cuda":
        return load_model_cuda(model_path, max_seq_length)
    return load_model_cpu(model_path, max_seq_length)


def transliterate(model, tokenizer, prompt: str, direction: str,
                   max_new_tokens: int = 128, temperature: float = 0.1, min_p: float = 0.1,
                   stream: bool = False) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[direction]},
        {"role": "user", "content": f"<{direction}> {prompt}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = inputs.ne(pad_id).long()

    streamer = None
    if stream:
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            min_p=min_p,
        )

    generated = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return generated.strip()


def read_inputs_from_file(path: str, input_column: str = "Input") -> list:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return []
        if isinstance(data[0], dict):
            return [str(item[input_column]) for item in data]
        return [str(item) for item in data]

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        return df[input_column].astype(str).tolist()

    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(file_path)
        return df[input_column].astype(str).tolist()

    raise ValueError(f"Unsupported --input-file type: {suffix} (use .txt, .json, .csv, or .xlsx)")


def run_batch_file(model, tokenizer, input_file: str, direction: str, output_file: str = None,
                    input_column: str = "Input", max_new_tokens: int = 128,
                    temperature: float = 0.1, min_p: float = 0.1):
    from tqdm import tqdm

    inputs = read_inputs_from_file(input_file, input_column=input_column)
    print(f"Loaded {len(inputs)} input(s) from {input_file}")

    results = []
    for text in tqdm(inputs, desc=f"Generating {direction}"):
        predicted = transliterate(
            model, tokenizer, text, direction,
            max_new_tokens=max_new_tokens, temperature=temperature, min_p=min_p,
        )
        results.append({"Input": text, "Predicted": predicted})

    if output_file:
        out_path = Path(output_file)
        if out_path.suffix.lower() == ".csv":
            import pandas as pd
            pd.DataFrame(results).to_csv(out_path, index=False)
        elif out_path.suffix.lower() == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                for row in results:
                    f.write(row["Predicted"] + "\n")
        print(f"Saved {len(results)} result(s) to: {out_path}")
    else:
        for row in results:
            print(f"\nInput:     {row['Input']}")
            print(f"Predicted: {row['Predicted']}")

    return results


def compute_metrics(predictions, references):
    import evaluate
    from jiwer import cer, wer

    predictions = [str(p) for p in predictions]
    references = [str(r) for r in references]

    bleu_metric = evaluate.load("bleu")
    chrf_metric = evaluate.load("chrf")

    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )["bleu"]
    chrf_score = chrf_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )["score"]
    cer_score = cer(references, predictions)
    wer_score = wer(references, predictions)

    return {
        "BLEU": round(bleu_score, 4),
        "ChrF": round(chrf_score, 4),
        "CER": round(cer_score, 4),
        "WER": round(wer_score, 4),
    }


def evaluate_xlsx(model, tokenizer, file_path: str, direction: str,
                   sheet_name: str = None, n_samples: int = 10, preview_rows: int = 5,
                   random_state: int = 42, save_results: bool = False,
                   max_new_tokens: int = 128, temperature: float = 0.1, min_p: float = 0.1):
    import pandas as pd
    from tqdm import tqdm

    input_col = "Jawi" if direction == "jawi2rumi" else "Rumi"
    target_col = "Rumi" if direction == "jawi2rumi" else "Jawi"

    sheet_list = [sheet_name] if sheet_name else pd.ExcelFile(file_path).sheet_names
    print(f"Evaluating {len(sheet_list)} sheet(s): {sheet_list}")

    results_summary = []

    for idx, sheet in enumerate(sheet_list, start=1):
        print(f"\n[{idx}/{len(sheet_list)}] Sheet: '{sheet}'")
        df = pd.read_excel(file_path, sheet_name=sheet)

        if input_col not in df.columns or target_col not in df.columns:
            print(f"  Skipping (missing {input_col}/{target_col} column)")
            continue

        sample_df = df.sample(min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)

        rows = []
        for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"Generating [{sheet}]"):
            input_text = str(row[input_col])
            actual_output = str(row[target_col])
            predicted_output = transliterate(
                model, tokenizer, input_text, direction,
                max_new_tokens=max_new_tokens, temperature=temperature, min_p=min_p,
            )
            rows.append({
                "Sheet": sheet, "No": i + 1,
                "Input": input_text, "Predicted": predicted_output, "Actual": actual_output,
            })

        comparison_df = pd.DataFrame(rows)
        metrics = compute_metrics(comparison_df["Predicted"].tolist(), comparison_df["Actual"].tolist())
        results_summary.append({"Sheet": sheet, "Samples": len(sample_df), **metrics})

        print(f"Sample predictions from '{sheet}':")
        print(comparison_df.head(preview_rows).to_string(index=False))

        if save_results:
            out_path = f"results_{sheet}_{direction}.csv"
            comparison_df.to_csv(out_path, index=False)
            print(f"Saved detailed results to: {out_path}")

    summary_df = pd.DataFrame(results_summary)
    print("\nOverall evaluation summary:")
    print(summary_df.to_string(index=False))

    if save_results:
        summary_path = f"evaluation_summary_{direction}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved metrics summary to: {summary_path}")

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run / evaluate the Jawi<->Rumi transliteration model")
    parser.add_argument("--model-path", type=Path,
                         default=Path(__file__).resolve().parent / "nirnama" / "v1" / "final",
                         help="Path to the trained adapter directory")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=None,
                         help="Physical GPU index to use (as shown in nvidia-smi). Ignored when --device cpu.")
    parser.add_argument("--direction", choices=DIRECTIONS, default="jawi2rumi")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-p", type=float, default=0.1)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", type=str, help="Single text to transliterate")
    mode.add_argument("--interactive", action="store_true", help="Interactive REPL")
    mode.add_argument("--input-file", type=str,
                       help="Batch inference from a .txt (one input per line), .json, .csv, or .xlsx file "
                            "(no ground truth needed)")
    mode.add_argument("--eval-xlsx", type=str, help="Path to an Excel file with ground truth to evaluate against")

    parser.add_argument("--input-column", type=str, default="Input",
                         help="Column/key name to read inputs from for .json/.csv/.xlsx --input-file")
    parser.add_argument("--output-file", type=str, default=None,
                         help="Where to save --input-file results (.csv, .json, or plain text). "
                              "Prints to stdout if omitted.")

    parser.add_argument("--sheet", type=str, default=None, help="Specific sheet name for --eval-xlsx (default: all sheets)")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-results", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} | VRAM: {round(gpu.total_memory / 1024 ** 3, 2)} GB")

    model, tokenizer = load_model(args.model_path, device, args.max_seq_length)

    if args.prompt is not None:
        result = transliterate(
            model, tokenizer, args.prompt, args.direction,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.min_p,
            stream=True,
        )
        print("\n--- Result ---")
        print(result)

    elif args.interactive:
        print(f"Interactive mode ({args.direction}). Type an empty line or Ctrl+C to quit.")
        try:
            while True:
                text = input("\n> ").strip()
                if not text:
                    break
                transliterate(
                    model, tokenizer, text, args.direction,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.min_p,
                    stream=True,
                )
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")

    elif args.input_file is not None:
        run_batch_file(
            model, tokenizer, args.input_file, args.direction,
            output_file=args.output_file, input_column=args.input_column,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.min_p,
        )

    elif args.eval_xlsx is not None:
        evaluate_xlsx(
            model, tokenizer, args.eval_xlsx, args.direction,
            sheet_name=args.sheet, n_samples=args.n_samples, random_state=args.random_state,
            save_results=args.save_results,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, min_p=args.min_p,
        )


if __name__ == "__main__":
    main()
