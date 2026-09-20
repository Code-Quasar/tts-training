#!/usr/bin/env python3
"""
Push a trained checkpoint to the HF Hub for inference.

Uploads ONLY what inference needs. `optimizer.pth` and `scheduler.pth` are
training state - typically ~16 GB - and are excluded, which usually cuts the
upload by two thirds.

    export HF_TOKEN=hf_...
    python push_model.py --ckpt /workspace/runs/tn_sft_v1/step_0000147 \
                         --repo-id Code-Quasar/voxcpm-tn

Then anywhere:

    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained("Code-Quasar/voxcpm-tn", load_denoiser=False)
    wav = m.generate(text="(Tunisian Dialect) عسلامة، شنوة أحوالك؟")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from common import load_config

# Everything the model needs to generate audio. Nothing else.
INFERENCE_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",          # only if safetensors is absent
    "audiovae.pth",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]

EXCLUDE = {"optimizer.pth", "scheduler.pth", "rng_state.pth", "trainer_state.json"}

CARD = """---
language: [ar]
license: apache-2.0
base_model: openbmb/VoxCPM2
pipeline_tag: text-to-speech
tags: [tunisian, derja, arabic-dialect, voxcpm, tts]
---

# {repo_id}

VoxCPM2 fine-tuned for **Tunisian Derja**.

- base: `openbmb/VoxCPM2` (2B, tokenizer-free, 48 kHz)
- method: {mode} fine-tuning, {steps} steps
- data: ~{hours} h Tunisian read speech, 16 kHz mono

## Important: the dialect tag

Every training transcript was prefixed with `({tag})`, so **untagged text is
out of distribution**. Always prefix it:

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("{repo_id}", load_denoiser=False)
wav = model.generate(text="({tag}) عسلامة، شنوة أحوالك اليوم؟")

import soundfile as sf
sf.write("out.wav", wav, 48000)
```

On a GPU with under ~8 GB, disable compilation:

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
```

## Limitations

Trained on a small corpus of read speech, so expect limited prosodic range and
weaker long-form phrasing. Derived from source corpora with their own licence
terms; the voices belong to real speakers.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint directory")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--include-optimizer", action="store_true",
                    help="also upload training state (~16 GB, rarely useful)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("export HF_TOKEN first")

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        sys.exit(f"not a directory: {ckpt}")
    if not (ckpt / "config.json").exists():
        sys.exit(f"{ckpt} has no config.json — loading would silently fall "
                 f"back to the base model")

    present = {f.name for f in ckpt.iterdir() if f.is_file()}
    upload = [f for f in INFERENCE_FILES if f in present]
    if "model.safetensors" in upload and "pytorch_model.bin" in upload:
        upload.remove("pytorch_model.bin")      # safetensors is enough
    if args.include_optimizer:
        upload += [f for f in present if f in EXCLUDE]

    skipped = present - set(upload)
    size = sum((ckpt / f).stat().st_size for f in upload) / 1024 ** 3
    skip_size = sum((ckpt / f).stat().st_size for f in skipped) / 1024 ** 3

    print(f"checkpoint: {ckpt}")
    print(f"\nuploading ({size:.1f} GiB):")
    for f in upload:
        print(f"  {f:<28} {(ckpt/f).stat().st_size/1024**3:6.2f} GiB")
    if skipped:
        print(f"\nskipping ({skip_size:.1f} GiB of training state):")
        for f in sorted(skipped):
            print(f"  {f}")

    if not any(f.startswith("model.") for f in upload):
        sys.exit("\nno model weights found — refusing to push")

    cfg = load_config(args.config)
    stats_f = Path(cfg["paths"]["data"]) / "stats.json"
    hours = json.loads(stats_f.read_text()).get("hours", "?") if stats_f.exists() else "?"
    steps = "".join(c for c in ckpt.name if c.isdigit()).lstrip("0") or "?"

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model",
                    private=not args.public, exist_ok=True)

    card = CARD.format(repo_id=args.repo_id, mode=cfg["train"]["mode"],
                       steps=steps, hours=hours,
                       tag=cfg["data"]["dialect_tag"])
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")

    print(f"\npushing to {args.repo_id} "
          f"({'public' if args.public else 'private'})…")
    for f in upload:
        print(f"  {f}")
        api.upload_file(path_or_fileobj=str(ckpt / f), path_in_repo=f,
                        repo_id=args.repo_id, repo_type="model")

    print(f"\ndone -> https://huggingface.co/{args.repo_id}")
    print(f"""
test locally:

  pip install voxcpm soundfile
  python - <<'EOF'
import os, soundfile as sf
os.environ["TORCHDYNAMO_DISABLE"] = "1"        # for GPUs under ~8 GB
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("{args.repo_id}", load_denoiser=False)
wav = m.generate(text="({cfg['data']['dialect_tag']}) عسلامة، شنوة أحوالك اليوم؟")
sf.write("tn_test.wav", wav, 48000)
print("wrote tn_test.wav")
EOF""")


if __name__ == "__main__":
    main()
