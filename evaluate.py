#!/usr/bin/env python3
"""
Quality harness. Training loss will not tell you whether the model improved.

Four automatic signals against a FIXED eval set:

  asr_cer / asr_wer   synthesize -> transcribe with a Tunisian ASR -> compare
                      to the intended text. Intelligibility. LOWER better.
  style_separation    same English sentence under 4 style prefixes; how far
                      apart they land acoustically. Collapsing toward 0 means
                      instruction-following is being overwritten by the
                      fine-tune. HIGHER better. This is the forgetting tripwire.
  duration_ratio      seconds per character - runaway/truncated generation.
  gen_failures        hard errors.

Then listen, because none of these hears an accent.

    python evaluate.py --tag base
    python evaluate.py --ckpt /workspace/runs/tn_sft_v1/step_100 --tag s100
    python evaluate.py --compare base s100
    python evaluate.py --ab base s100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from common import HERE, cer, free_cuda, load_config, norm_ar, wer

OUT = HERE / "eval_out"
SENTENCES = HERE / "eval_sentences.txt"
SR = 48000

PROBE_TEXT = "I need to tell you something before you go."
PROBE_STYLES = [
    ("whisper", "(whispering, secretive, quiet)"),
    ("angry", "(angry, shouting, forceful)"),
    ("sad", "(sad, slow, subdued)"),
    ("excited", "(excited, fast, energetic)"),
]


def load_sentences():
    rows = []
    for line in SENTENCES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cat, _, text = line.partition("\t")
        if text.strip():
            rows.append((cat.strip(), text.strip()))
    return rows


# ---------------------------------------------------------------- acoustics

def features(y, sr=SR) -> dict:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {"rms": 0.0, "f0": 0.0, "dur": 0.0, "centroid": 0.0}
    try:
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
        f0 = float(np.nanmedian(f0[np.isfinite(f0)])) if f0.size else 0.0
    except Exception:
        f0 = 0.0
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    except Exception:
        centroid = 0.0
    return {"rms": float(np.sqrt(np.mean(y ** 2))), "f0": f0,
            "dur": len(y) / sr, "centroid": centroid}


def style_separation(feats) -> float:
    if len(feats) < 2:
        return 0.0
    keys = ["rms", "f0", "dur", "centroid"]
    M = np.array([[f[k] for k in keys] for f in feats], dtype=np.float64)
    sd = M.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (M - M.mean(axis=0)) / sd
    d = [np.linalg.norm(Z[i] - Z[j])
         for i in range(len(Z)) for j in range(i + 1, len(Z))]
    return float(np.mean(d)) if d else 0.0


# ---------------------------------------------------------------- model

def is_lora(path) -> bool:
    p_ = Path(path)
    return any((p_ / f).exists() for f in
               ("lora_config.json", "lora_weights.safetensors", "lora_weights.ckpt"))


def build_model(cfg, ckpt, no_compile):
    """Full-SFT checkpoints are complete models, not adapters.

    LoRA  -> base model + lora_weights_path=<adapter dir>
    Full  -> from_pretrained(<checkpoint dir>) directly; the dir carries
             model.safetensors, config.json and audiovae.pth.

    Passing a full checkpoint as lora_weights_path would silently load the
    BASE model, and every metric would come back identical to baseline.

    load_denoiser=False skips the zipenhancer model: ~1 GB of VRAM and a
    separate download we do not need for evaluation.
    """
    if no_compile:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from voxcpm import VoxCPM
    base = cfg["train"]["base_model"]

    def _load(src, **kw):
        try:
            return VoxCPM.from_pretrained(src, load_denoiser=False, **kw)
        except TypeError:          # older signature without load_denoiser
            return VoxCPM.from_pretrained(src, **kw)

    if not ckpt:
        print(f"loading {base} (base)")
        return _load(base)
    if is_lora(ckpt):
        print(f"loading {base} + LoRA adapter {ckpt}")
        return _load(base, lora_weights_path=str(ckpt))
    print(f"loading FULL checkpoint {ckpt}")
    missing = [f for f in ("config.json",) if not (Path(ckpt) / f).exists()]
    if missing:
        print(f"  WARNING {ckpt} lacks {missing} — the trainer writes "
              f"config.json alongside weights; loading may fall back to base")
    return _load(str(ckpt))


def generate_all(cfg, model, tag):
    import soundfile as sf
    tagstr = cfg["data"]["dialect_tag"]
    wav_dir = OUT / tag / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    recs, probes, fails = [], [], 0

    print("\n[1/2] Tunisian eval set")
    for i, (cat, text) in enumerate(load_sentences()):
        name = f"{cat}_{i:03d}"
        try:
            y = np.asarray(model.generate(text=f"({tagstr}) {text}"), dtype=np.float32)
            sf.write(wav_dir / f"{name}.wav", y, SR)
            recs.append({"name": name, "category": cat, "text": text,
                         "wav": str(wav_dir / f"{name}.wav"), **features(y)})
            print(f"  ok  {name}  {len(y)/SR:5.2f}s")
        except Exception as e:
            fails += 1
            print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:80]}")
        finally:
            free_cuda()

    print("\n[2/2] English style probes (forgetting tripwire)")
    for label, prefix in PROBE_STYLES:
        try:
            y = np.asarray(model.generate(text=f"{prefix} {PROBE_TEXT}"),
                           dtype=np.float32)
            sf.write(wav_dir / f"probe_{label}.wav", y, SR)
            f = features(y)
            probes.append({"name": f"probe_{label}", "style": label, **f})
            print(f"  ok  {label:<9} rms={f['rms']:.4f} f0={f['f0']:6.1f} dur={f['dur']:.2f}")
        except Exception as e:
            fails += 1
            print(f"  FAIL probe_{label}: {type(e).__name__}")
        finally:
            free_cuda()
    return recs, probes, fails


def asr_roundtrip(cfg, recs):
    try:
        import torch
        from transformers import pipeline
        model_id = cfg["eval"]["asr_model"]
        print(f"\nASR round-trip: {model_id}")
        asr = pipeline("automatic-speech-recognition", model=model_id,
                       device=0 if torch.cuda.is_available() else -1,
                       chunk_length_s=30)
    except Exception as e:
        print(f"ASR unavailable ({type(e).__name__}) — skipping")
        return recs
    for r in recs:
        try:
            hyp = (asr(r["wav"], generate_kwargs={"language": "ar",
                                                  "task": "transcribe"}).get("text") or "").strip()
        except Exception:
            try:
                hyp = (asr(r["wav"]).get("text") or "").strip()
            except Exception:
                continue
        r["asr"], r["cer"], r["wer"] = hyp, round(cer(r["text"], hyp), 4), round(wer(r["text"], hyp), 4)
    free_cuda()
    return recs


def summarize(cfg, recs, probes, fails, tag):
    cers = [r["cer"] for r in recs if "cer" in r]
    wers = [r["wer"] for r in recs if "wer" in r]
    durs = [r["dur"] for r in recs]
    chars = [max(len(r["text"]), 1) for r in recs]
    by_cat = {}
    for r in recs:
        if "cer" in r:
            by_cat.setdefault(r["category"], []).append(r["cer"])

    res = {"tag": tag, "n": len(recs), "gen_failures": fails,
           "asr_model": cfg["eval"]["asr_model"],
           "asr_cer": round(float(np.mean(cers)), 4) if cers else None,
           "asr_wer": round(float(np.mean(wers)), 4) if wers else None,
           "cer_by_category": {k: round(float(np.mean(v)), 4) for k, v in by_cat.items()},
           "style_separation": round(style_separation(probes), 4),
           "duration_ratio": round(float(np.sum(durs) / np.sum(chars)), 5) if durs else None}

    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    with open(d / "records.jsonl", "w", encoding="utf-8") as f:
        for r in recs + probes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 56}\n{tag}\n{'=' * 56}")
    for k in ("asr_cer", "asr_wer", "style_separation", "duration_ratio", "gen_failures"):
        print(f"  {k:<18} {res[k]}")
    for k, v in sorted(res["cer_by_category"].items()):
        print(f"    {k:<12} {v}")
    return res


DIRECTION = {"asr_cer": True, "asr_wer": True,
             "style_separation": False, "gen_failures": True}


def compare(a_tag, b_tag):
    a = json.loads((OUT / a_tag / "results.json").read_text())
    b = json.loads((OUT / b_tag / "results.json").read_text())
    print(f"\n{'metric':<20}{a_tag:>12}{b_tag:>12}{'delta':>12}   verdict")
    print("-" * 70)
    worse = []
    for k, lower in DIRECTION.items():
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        delta = vb - va
        good = (delta < 0) if lower else (delta > 0)
        rel = abs(delta) / max(abs(va), 1e-9)
        v = "~ same" if rel < 0.02 else ("BETTER" if good else "WORSE")
        if v == "WORSE":
            worse.append(k)
        print(f"{k:<20}{va:>12.4f}{vb:>12.4f}{delta:>+12.4f}   {v}")
    print()
    if "style_separation" in worse:
        print("WARNING style_separation fell — instruction-following eroding.")
        print("        raise en/zh replay, or lower LR / LoRA rank.")
    if "asr_cer" in worse:
        print("WARNING intelligibility regressed — usually too many steps or")
        print("        noisy transcripts, not a hyperparameter.")
    if not worse:
        print(f"No regression. Confirm by ear:  python evaluate.py --ab {a_tag} {b_tag}")


def make_ab(a_tag, b_tag):
    import shutil
    ab = OUT / f"ab_{a_tag}_vs_{b_tag}"
    ab.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    wa, wb = OUT / a_tag / "wav", OUT / b_tag / "wav"
    key = []
    for f in sorted(wa.glob("*.wav")):
        if not (wb / f.name).exists():
            continue
        flip = rng.random() < 0.5
        left, right = (b_tag, a_tag) if flip else (a_tag, b_tag)
        shutil.copy(wa / f.name if left == a_tag else wb / f.name, ab / f"{f.stem}__A.wav")
        shutil.copy(wa / f.name if right == a_tag else wb / f.name, ab / f"{f.stem}__B.wav")
        key.append({"item": f.stem, "A": left, "B": right})
    (ab / "KEY.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (ab / "SCORESHEET.csv").write_text(
        "item,more_natural(A/B),more_tunisian(A/B),notes\n"
        + "".join(f"{k['item']},,,\n" for k in key), encoding="utf-8")
    print(f"{len(key)} blind pairs -> {ab}")
    print("Give SCORESHEET.csv to native listeners; do NOT show KEY.json.")
    print("Score two questions separately: natural speech, and native Tunisian")
    print("versus someone reading Tunisian. They come apart.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag")
    ap.add_argument("--ckpt")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--ab", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.ab:
        return make_ab(*args.ab)
    if not args.tag:
        sys.exit("--tag required (or --compare / --ab)")

    cfg = load_config(args.config)
    try:
        model = build_model(cfg, args.ckpt, args.no_compile)
    except Exception:
        traceback.print_exc()
        sys.exit("failed to load model")
    recs, probes, fails = generate_all(cfg, model, args.tag)
    if not args.skip_asr:
        del model
        free_cuda()
        recs = asr_roundtrip(cfg, recs)
    summarize(cfg, recs, probes, fails, args.tag)


if __name__ == "__main__":
    main()
