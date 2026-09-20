# voxcpm-tn

Fine-tune [VoxCPM2](https://github.com/OpenBMB/VoxCPM) for Tunisian Derja.

Clone, install, run.

```bash
git clone <this repo> /workspace/voxcpm-tn
cd /workspace/voxcpm-tn

bash install.sh          # 1. torch + VoxCPM + deps
source /workspace/env.sh

python data.py pull      # 2. data from your HF dataset repo
python check.py          # 3. verify + download model weights

tmux new -s train
python train.py          # 4. train
```

Edit `config.yaml` and nothing else. Every script reads it.

---

## Files

| | |
|---|---|
| `config.yaml` | the only file you edit |
| `install.sh` | torch (CUDA auto-detected), VoxCPM, deps |
| `check.py` | environment + config + data verification, downloads weights |
| `data.py` | `prepare` / `push` / `pull` / `stats` |
| `train.py` | training driver — checkpointing, validation, auto-eval |
| `evaluate.py` | quality harness + blind A/B export |
| `common.py` | config loading, Arabic normalization, CER/WER |
| `eval_sentences.txt` | frozen eval set — **never** put these in training data |

---

## Data

Build it once, then push so any pod can pull it:

```bash
python data.py prepare --limit 50    # smoke test
python data.py prepare               # full, ~10-20 min
export HF_TOKEN=hf_...
python data.py push                  # private by default
```

Output is WAVs plus a JSONL manifest — the only format the trainer reads:

```jsonl
{"audio": "/workspace/data/tunswitch/wav/tn_train_000001.wav", "text": "(Tunisian Dialect) عسلامة، شنوة أحوالك؟"}
{"audio": ".../tn_train_000002.wav", "text": "(Tunisian Dialect, calm) الطقس اليوم سخون برشة.", "ref_audio": ".../tn_train_000001.wav"}
```

Audio is **16 kHz mono, 1–20 s**. The 48 kHz in the config is output-only.

Every transcript carries `(Tunisian Dialect)`. Because 100% of rows are tagged, untagged text is out of distribution for the fine-tuned model — always prefix it at inference. `evaluate.py` does this for you.

Pushed manifests store relative paths; `pull` rewrites them to absolute.

---

## Training

`train.py` generates the trainer's config from `config.yaml`, runs the official
VoxCPM trainer, and adds what it lacks.

**Checkpointing** — `save_interval: 50`. Checkpoints land as
`step_XXXXXXX/` under `save_path`, plus a `latest/` pointer. Tight interval on
purpose: the best checkpoint with full SFT arrives early and is easy to blow
past. Each full-SFT checkpoint carries weights plus `optimizer.pth` and
`scheduler.pth` (~24 GB), so budget a 150–200 GB volume.

**Resume is automatic** — the trainer picks up from `latest/` if it exists, and
saves a checkpoint on SIGINT/SIGTERM. Delete `latest/` to start fresh.

**Validation** — `valid_interval: 50` computes validation loss on `val.jsonl`,
split by speaker where speaker labels exist. If validation loss rises while
training loss falls, stop and roll back: that specific pattern means the model
has stopped conditioning on your text.

**Automatic evaluation** — a watcher evaluates each checkpoint as it lands,
appends to `eval_trend.csv`, and tracks the best in `best.json`.

```bash
python train.py --dry-run       # write config + run checks, don't train
python train.py --resume        # continue from the latest checkpoint
python train.py --no-auto-eval  # train only
```

Switch to LoRA by setting `train.mode: lora` and `learning_rate: 1.0e-4`.
The `lora` block's presence in the generated config is what selects the mode.

`config.yaml` uses VoxCPM2's own key names (`save_interval`, `valid_interval`,
`grad_accum_steps`, …) so nothing is silently dropped. The trainer is
**step-based**; `train.epochs` is a convenience that `train.py` converts to
`max_steps` from your manifest length.

---

## Is it actually better?

Training loss won't tell you. `evaluate.py` measures four things against a
frozen eval set:

| metric | direction | catches |
|---|---|---|
| `asr_cer` / `asr_wer` | lower | intelligibility — synthesize, transcribe with a Tunisian ASR, compare |
| `style_separation` | higher | **forgetting** — same English sentence under 4 style prefixes; if they converge, instruction-following is being overwritten |
| `duration_ratio` | stable | runaway or truncated generation |
| `gen_failures` | 0 | hard breakage |

Arabic is normalized before scoring, so orthographic variants (`برشة` vs
`برشه`) cost nothing — it measures pronunciation, not spelling.

```bash
python evaluate.py --compare base tn_sft_v1_s100
python evaluate.py --ab base tn_sft_v1_s100
```

`--ab` writes randomized A/B pairs with a key you don't show the listeners.
Score two questions **separately**: is this natural speech, and does it sound
like a native Tunisian rather than someone reading Tunisian. They come apart,
and only the second says the dialect adaptation worked.

No automatic metric hears an accent. `asr_cer` will happily reward confident
MSA-accented speech that a Tunisian ASR still decodes correctly.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `Python.h: No such file` | `apt install python3.12-dev build-essential` — triton JIT-compiles a CUDA helper in C |
| CUDA OOM | lower `max_batch_tokens` (16384 → 8192) before touching `batch_size` |
| val loss up, train loss down | stop, roll back — the model is ignoring input text |
| style_separation dropping | raise en/zh replay, lower LR or LoRA rank |
| metrics identical to baseline | checkpoint loaded wrong; `evaluate.py --ckpt` auto-detects LoRA vs full |
| garbage audio | see [OpenBMB/VoxCPM#202](https://github.com/OpenBMB/VoxCPM/issues/202) |

`check.py` also diffs your config keys against the repo's reference configs.
Unrecognized YAML keys are ignored **silently**, so a `learning_rate` that
doesn't apply would never raise — worth reading that section of the output.
