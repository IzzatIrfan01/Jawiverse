# Jawi <-> Rumi Transliteration: Training & Inference

Three scripts, replacing the original Colab notebooks:

| Script | Replaces | Purpose |
|---|---|---|
| `train_jawi_rumi.py` | `TrainingModel(Unsloth).ipynb` | Fine-tune the LoRA adapter |
| `infer_jawi_rumi.py` | `TestingModel.ipynb` | Run / evaluate the trained adapter |
| `telegram_bot.py` | (new) | Chat with the model from Telegram |

`common.py` holds the small bits shared by all three (`.env` loading, device resolution) — not something you run directly.

All three auto-detect GPU vs CPU and accept `--device auto|cuda|cpu` and `--gpu-id <N>` to pick a specific physical card (see [GPU selection](#gpu-selection)).

---

## 1. Setup (on the server)

```bash
# 1. Copy the jawiverse folder to the server (scp/rsync/git), then:
cd jawiverse

# 2. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install torch FIRST, matching the server's CUDA driver -- check
#    https://pytorch.org/get-started/locally/ for the right command
#    (the RTX 3090 and the RTX PRO 6000 Blackwell may want different
#    cu12x wheels; nvidia-smi's "CUDA Version" is the driver's max
#    supported version, not necessarily what you should install).
pip install torch --index-url https://download.pytorch.org/whl/cu124   # adjust cu-tag as needed

# 4. Install everything else
pip install -r requirements.txt

# 5. Set up Telegram notifications (optional)
cp .env.example .env
# then edit .env and fill in your real TG_BOT_TOKEN / TG_CHAT_ID
```

`requirements.txt` covers both the GPU path (Unsloth + bitsandbytes + 4-bit) and what's needed for `--device cpu` runs on the same machine (plain `transformers` + `peft`), plus the optional evaluation-mode deps (`evaluate`, `jiwer`). Since your server has two working GPUs, you won't hit the "unsloth fails to import" problem from the earlier Windows CPU-only test — `--device cpu` there just skips the 4-bit/Unsloth kernels for that specific run, it doesn't require a GPU-less machine.

If you're running on the RTX PRO 6000 Blackwell specifically: verify `torch`/`bitsandbytes`/`xformers`/`unsloth` actually support it before a long training run — it's very new hardware and support can lag behind.

---

## 2. Training — `train_jawi_rumi.py`

Reads `nirnama/dataset/jawi_rumi_v1.xlsx` + `jawi_rumi_parallel_v1.xlsx`, builds the ShareGPT-style conversations JSON (skipped if it already exists — pass `--rebuild-dataset` to force), fine-tunes a LoRA adapter on top of `unsloth/Llama-3.2-3B-Instruct`, and saves checkpoints to `nirnama/v1/checkpoints/` and the final adapter to `nirnama/v1/final/`.

### Basic usage
```bash
# Auto-detect GPU, train with defaults (matches the original Colab run)
python train_jawi_rumi.py

# Force a specific physical GPU (see nvidia-smi index)
python train_jawi_rumi.py --gpu-id 1        # RTX 3090
python train_jawi_rumi.py --gpu-id 0        # RTX PRO 6000 Blackwell

# CPU smoke test only (small run, not the full 54k-example dataset)
python train_jawi_rumi.py --device cpu --num-train-epochs 1 --per-device-train-batch-size 1
```

### Resuming
There's no Colab-style "segmented" training here — just leave it running. If the process dies or you Ctrl+C, rerun the exact same command; it picks up from the last checkpoint in `nirnama/v1/checkpoints/` automatically. If `nirnama/v1/final/adapter_model.safetensors` already exists, the script reports training as complete and exits without doing anything.

### Key flags
| Flag | Default | Notes |
|---|---|---|
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--gpu-id` | none | Physical GPU index (nvidia-smi numbering) |
| `--base-model` | `unsloth/Llama-3.2-3B-Instruct` | HF/Unsloth repo id |
| `--no-4bit` | off | Disable 4-bit loading on CUDA (uses more VRAM, may improve quality slightly) |
| `--lora-r` / `--lora-alpha` | `16` / `16` | LoRA rank/alpha |
| `--per-device-train-batch-size` | `4` | Raise this on the Blackwell's 96GB if you want faster throughput |
| `--gradient-accumulation-steps` | `8` | Effective batch = batch size × this |
| `--num-train-epochs` | `3` | |
| `--learning-rate` | `5e-5` | |
| `--num-checkpoint-segments` | `5` | Roughly how many save points across the whole run |
| `--save-total-limit` | `5` | How many checkpoints to keep on disk |
| `--rebuild-dataset` | off | Force-regenerate the conversations JSON from the Excel files |
| `--dataset` | `nirnama` | `nirnama` or `risalah`; also picks the output folder (`<dataset folder>/v1/`) |
| `--init-adapter` | none | Continue fine-tuning from an existing adapter (e.g. `nirnama/v1/final`) instead of a fresh LoRA |
| `--eval-holdout-ratio` | per dataset | Fraction of pairs kept out of training for evaluation (nirnama `0`, risalah `0.1`) |
| `--train-max-seq-length` | per dataset | Tokens per training example; longer ones are truncated, cutting off the answer (nirnama `256`, risalah `1024`) |

### Data cleaning
When the conversations JSON is built, rows are skipped if the pair is incomplete (Jawi or Rumi empty), the Jawi is a Quran verse pasted as Quran-font glyphs, the Jawi has no Arabic-script letters, the Rumi has no Latin letters or contains Arabic script (e.g. number-only rows), or the exact pair already appeared. Counts per reason are printed, and every skipped row (file, sheet, Excel row, reason) is written to `<dataset folder>/v1/skipped_rows.csv` so you can fix the source Excel. An existing JSON is reused as-is — pass `--rebuild-dataset` to apply the cleaning to it.

For Risalah, long sentence pairs are additionally split into extra clause pairs at `. ; : ! ?` (Jawi `؛ ؟` count as `; ?`), each clause keeping its own punctuation. A pair is only split when both sides break at the same sequence of marks, every clause has balanced brackets/quotes and a plausible Jawi/Rumi word ratio, and passes the checks above — otherwise it's kept whole only. Splits are made from training pairs only, never from eval holdout pairs. Currently this adds ~600 pairs.

### Training on Risalah Untuk Kaum Muslimin
```bash
# Separate Risalah-only model (fresh LoRA, no nirnama data involved)
python train_jawi_rumi.py --dataset risalah --gpu-id 1

# Or continue from the finished nirnama model instead
python train_jawi_rumi.py --dataset risalah --init-adapter nirnama/v1/final --gpu-id 1
```
Reads both files in `risalah kaum muslimin/dataset/`, writes to `risalah kaum muslimin/v1/{checkpoints,final}`. 10% of the sentence pairs (fixed seed) are kept out of training in `risalah kaum muslimin/v1/eval_holdout.xlsx` — score the result on them with:
```bash
python infer_jawi_rumi.py --model-path "risalah kaum muslimin/v1/final" \
    --eval-xlsx "risalah kaum muslimin/v1/eval_holdout.xlsx" --direction jawi2rumi --n-samples 1000 \
    --max-new-tokens 512 --save-results
```
(`--max-new-tokens 512` matters here: the default 128 cuts off long Risalah sentences and scores them unfairly.)
Run the same command with `--model-path nirnama/v1/final` to compare against the nirnama-only model, and spot-check nirnama-style text too to make sure the continued model hasn't forgotten it.

### Optional Telegram progress notifications
Set `TG_BOT_TOKEN` and `TG_CHAT_ID` to enable (never hardcode these in the script). Easiest way is a local `.env` file, auto-loaded at startup:
```bash
cp .env.example .env
# edit .env and fill in your real TG_BOT_TOKEN / TG_CHAT_ID
python train_jawi_rumi.py
```
(Or `export TG_BOT_TOKEN=... TG_CHAT_ID=...` instead of using `.env` — either works, exported env vars take precedence over `.env`.) If either is unset, notifications are silently skipped.

> **Security note:** the original notebook had a live Telegram bot token hardcoded in plaintext. If that token is still active, revoke it in BotFather and use `.env`/env vars instead.

---

## 3. Inference — `infer_jawi_rumi.py`

Loads the adapter from `--model-path` (default `nirnama/v1/final`). The base model name is read automatically from the adapter's own `adapter_config.json`, so you don't need to specify it separately.

Exactly one of four modes is required per run: `--prompt`, `--interactive`, `--input-file`, or `--eval-xlsx`.

### Mode 1 — Single prompt
```bash
python infer_jawi_rumi.py --prompt "چينتا سجاتي اداله سوات ڤركارا" --direction jawi2rumi
python infer_jawi_rumi.py --device cpu --prompt "Cinta sejati" --direction rumi2jawi
```

### Mode 2 — Interactive REPL
```bash
python infer_jawi_rumi.py --interactive --direction jawi2rumi
```
Type a line, get a transliteration, repeat. Empty line or Ctrl+C to quit.

### Mode 3 — Batch from a file (no ground truth needed)
Accepts `.txt` (one input per line), `.json` (list of strings, or list of `{"Input": "..."}` objects), `.csv`, or `.xlsx`:
```bash
python infer_jawi_rumi.py --input-file prompts.txt --direction jawi2rumi
python infer_jawi_rumi.py --input-file prompts.csv --input-column Text --output-file results.csv
```
- `--input-column` (default `Input`): which column/key to read for `.json`/`.csv`/`.xlsx`.
- `--output-file`: `.csv` → `Input,Predicted` table; `.json` → structured records; anything else → plain predicted text, one per line. Omit to print to stdout instead.

### Mode 4 — Evaluate against labeled Excel data (BLEU / ChrF / CER / WER)
```bash
python infer_jawi_rumi.py --eval-xlsx "risalah kaum muslimin/dataset/Risalah Untuk Kaum Muslimin.xlsx" \
    --direction jawi2rumi --n-samples 10 --save-results
```
- Expects `Jawi` and `Rumi` columns (direction determines which is input vs. target).
- `--sheet NAME` to test one sheet only; omit for all sheets.
- `--n-samples` / `--random-state` control sampling per sheet.
- `--save-results` writes `results_<sheet>_<direction>.csv` per sheet plus `evaluation_summary_<direction>.csv`.

### Shared flags (all modes)
| Flag | Default | Notes |
|---|---|---|
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--gpu-id` | none | Physical GPU index |
| `--model-path` | `nirnama/v1/final` | Adapter directory (or a specific checkpoint dir) |
| `--direction` | `jawi2rumi` | or `rumi2jawi` |
| `--max-new-tokens` | `128` | |
| `--temperature` | `0.1` | `0` = greedy |
| `--min-p` | `0.1` | Nucleus sampling floor |

---

## 4. Chat testing — `telegram_bot.py`

Loads the model once (same loading code as `infer_jawi_rumi.py`) and lets you transliterate by messaging your bot from your phone, instead of typing `--prompt` on the server.

Uses GPU or CPU depending on `--device`/`--gpu-id`, exactly like the other two scripts — default is `--device auto` (GPU if one's visible), or force it with `--device cpu` / `--gpu-id <N>`.

### Step by step

**1. Confirm `.env` has your bot's credentials** (copy from the template if you haven't already):
```bash
cp .env.example .env
# edit .env and fill in:
#   TG_BOT_TOKEN=<your bot token from BotFather>
#   TG_CHAT_ID=<your own chat id>
```

**2. Start the bot**, picking a free GPU (check `nvidia-smi` first):
```bash
cd jawiverse
source jawi_env/bin/activate     # or whatever your venv is called
python telegram_bot.py --gpu-id 1
```
Once loaded, it prints the device to the terminal **and** sends a startup message to every allowed chat ID, e.g.:
```
🤖 Jawiverse bot online
Device: CUDA (NVIDIA GeForce RTX 3090)
Model: final
```
That message is your confirmation the bot is up and which device/GPU it's actually running on — check for it on your phone before you start testing.

**3. Open Telegram on your phone**, go to your chat with the bot, and just send it text:
- Send Jawi script → auto-detected, replies with Rumi.
- Send Rumi text → auto-detected, replies with Jawi.
- `/jawi2rumi` or `/rumi2jawi` — lock the direction for that chat (skip auto-detection).
- `/auto` — go back to auto-detect.
- `/help` — show the command list.

Each exchange is also logged to the terminal in a readable block, with timing/speed:
```
----------------------------------------------------------------
[14:32:07] chat=123456789 | jawi2rumi
IN  (6 tok, 12 chars): چينتا سجاتي
OUT (4 tok, 13 chars): Cinta sejati
time=0.87s | speed=4.6 tok/s
----------------------------------------------------------------
```

**4. Stop it** with Ctrl+C in the terminal when done.

**Keeping it running after you disconnect SSH**: the process dies when your terminal session ends unless it's kept alive some other way. Easiest is `tmux`:
```bash
tmux new -s jawibot
python telegram_bot.py --gpu-id 1
# Ctrl+B then D to detach -- it keeps running
# tmux attach -t jawibot   to reconnect later and see logs / Ctrl+C it
```

### Notes
- **Access control**: by default only the chat ID in `TG_CHAT_ID` (or the comma-separated list in `ALLOWED_CHAT_IDS`) can talk to the bot — otherwise anyone who finds your bot's username could run inference on your GPU. Pass `--allow-all` to disable that check.
- This is a simple long-polling bot for interactive testing (like `infer_jawi_rumi.py --interactive`, just from Telegram) — not a hardened production service.

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | `nirnama/v1/final` | Adapter directory |
| `--device` / `--gpu-id` | `auto` / none | Same as the other scripts |
| `--max-new-tokens` / `--temperature` / `--min-p` | `128` / `0.1` / `0.1` | Generation params |
| `--allow-all` | off | Respond to any chat, ignoring the allowlist |
| `--poll-timeout` | `30` | Telegram long-poll timeout (seconds) |

---

## GPU selection

All three scripts accept `--gpu-id <physical index>` (as shown in `nvidia-smi`), which sets `CUDA_VISIBLE_DEVICES` before any CUDA call:
```bash
nvidia-smi   # check which index is idle / has free VRAM
python train_jawi_rumi.py --gpu-id 1   # 3090
python infer_jawi_rumi.py --gpu-id 0 --prompt "..." --direction jawi2rumi   # Blackwell
```
Note: this does **not** enable multi-GPU training across both cards at once — it just picks which single GPU to use. Running one job per GPU independently (e.g. training on one, evaluating on the other) works fine with two separate invocations.

---

## CPU inference caveats

- No 4-bit quantization (bitsandbytes requires CUDA) — runs the base model in fp32, so expect higher RAM use and noticeably slower generation than GPU.
- Unsloth itself is not imported on the CPU path at all (it fails to import without CUDA deps present) — CPU inference uses plain `transformers` + `PEFT` instead.
- For genuinely fast CPU-only deployment, the better long-term option is merging the adapter into the base model and converting to **GGUF** to run via **llama.cpp**, rather than running the raw fp32 HF model — worth revisiting if CPU inference latency becomes a real requirement.
