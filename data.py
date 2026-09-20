#!/usr/bin/env python3
"""
Data: prepare locally, push to HF, pull on the pod.

    python data.py prepare                 # HF corpus -> wav + train/val.jsonl
    python data.py prepare --limit 50      # smoke test first
    python data.py push                    # -> config.data.hf_repo (private)
    python data.py pull                    # pod: download + fix paths
    python data.py stats                   # what do I have

Manifest format (the only thing the trainer reads):
    {"audio": "<abs 16kHz mono wav>", "text": "(Tunisian Dialect) ...",
     "ref_audio": "<abs wav>"}          # ref_audio optional

Pushed manifests use RELATIVE paths so they survive the trip; `pull` rewrites
them to absolute for wherever they landed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

from common import load_config, p, read_jsonl, word_count, write_jsonl

EMOTIONS = ["calm", "conversational", "measured", "cheerful", "serious"]


def guess_speaker(path) -> str:
    """No speaker column in TunSwitch; filenames sometimes encode one.
    Only affects how the val split is drawn."""
    if not path:
        return "unk"
    m = re.match(r"^([A-Za-z]*\d{1,5})[_\-]", Path(str(path)).stem)
    return m.group(1) if m else "unk"


# ------------------------------------------------------------------ prepare

def prepare(cfg, args):
    import librosa
    import soundfile as sf
    from datasets import Audio, Value, load_dataset
    from tqdm.auto import tqdm

    d = cfg["data"]
    rng = random.Random(0)
    out = p(cfg, "data")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    sr_target = d["sample_rate"]
    tag = d["dialect_tag"]

    ds = load_dataset(d["hf_dataset"])
    first = next(iter(ds.values()))
    audio_col = next(c for c, f in first.features.items() if isinstance(f, Audio))
    strs = [c for c, f in first.features.items()
            if isinstance(f, Value) and f.dtype == "string"]
    text_col = next((c for c in strs if "original" in c.lower()),
                    next((c for c in strs if "diacritiz" not in c.lower()), strs[0]))
    print(f"audio={audio_col}  text={text_col}")

    kept, skip = [], {"text": 0, "duration": 0, "decode": 0}
    for split_name, split in ds.items():
        s = split.cast_column(audio_col, Audio(decode=False))   # no torchcodec
        n = s.num_rows if not args.limit else min(args.limit, s.num_rows)
        for i in tqdm(range(n), desc=split_name):
            ex = s[i]
            text = str(ex[text_col] or "").strip()
            if word_count(text) < d["min_words"]:
                skip["text"] += 1
                continue
            a = ex[audio_col]
            try:
                import io
                src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
                y, sr = sf.read(src, dtype="float32", always_2d=False)
            except Exception:
                skip["decode"] += 1
                continue
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            dur = len(y) / sr_target
            if not (d["min_duration"] <= dur <= d["max_duration"]):
                skip["duration"] += 1
                continue
            peak = float(np.abs(y).max()) if y.size else 0.0
            if peak < 1e-6:
                skip["decode"] += 1
                continue

            uid = f"tn_{split_name}_{i:06d}"
            wav = out / "wav" / f"{uid}.wav"
            sf.write(wav, y / peak * 0.95, sr_target)
            kept.append({"uid": uid, "audio": str(wav.resolve()), "text": text,
                         "duration": round(dur, 3),
                         "speaker": guess_speaker(a.get("path"))})

    if not kept:
        sys.exit(f"nothing kept. skips={skip}")
    hours = sum(r["duration"] for r in kept) / 3600
    print(f"\nkept {len(kept)} clips | {hours:.2f} h | skips {skip}")

    # ---- split ----------------------------------------------------------
    spk = sorted({r["speaker"] for r in kept if r["speaker"] != "unk"})
    if len(spk) >= 10:
        rng.shuffle(spk)
        val_spk = set(spk[:max(1, int(len(spk) * d["val_frac"]))])
        for r in kept:
            r["_val"] = r["speaker"] in val_spk
        how = f"speaker-held-out ({len(val_spk)} speakers)"
    else:
        idx = list(range(len(kept)))
        rng.shuffle(idx)
        val_ids = set(idx[:max(1, int(len(kept) * d["val_frac"]))])
        for i, r in enumerate(kept):
            r["_val"] = i in val_ids
        how = "random (no speaker labels — val is slightly optimistic)"
    print(f"split: {how}")

    # ---- manifests ------------------------------------------------------
    by_spk: dict[str, list[str]] = {}
    for r in kept:
        by_spk.setdefault(r["speaker"], []).append(r["audio"])

    def line(r):
        # dialect tag on EVERY row; emotion word on a fraction
        prefix = (f"({tag}, {rng.choice(EMOTIONS)}) "
                  if rng.random() < d["description_rate"] else f"({tag}) ")
        row = {"audio": r["audio"], "text": prefix + r["text"]}
        pool = by_spk.get(r["speaker"], [])
        if rng.random() < d["ref_rate"] and len(pool) > 1:
            row["ref_audio"] = rng.choice([x for x in pool if x != r["audio"]])
        return row

    train = [line(r) for r in kept if not r["_val"]]
    val = [line(r) for r in kept if r["_val"]]
    assert all(f"({tag}" in r["text"] for r in train + val), "dialect tag missing"

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    stats = {"clips": len(kept), "hours": round(hours, 3), "train": len(train),
             "val": len(val), "split": how, "skips": skip, "dialect_tag": tag,
             "median_duration": round(float(np.median([r["duration"] for r in kept])), 3)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\n-> {out}/train.jsonl  ({len(train)})\n-> {out}/val.jsonl  ({len(val)})")


# ------------------------------------------------------------------ push/pull

def _rewrite(rows, root: Path, absolute: bool):
    out = []
    for r in rows:
        r = dict(r)
        for k in ("audio", "ref_audio"):
            if k not in r:
                continue
            if absolute:
                r[k] = str((root / r[k]).resolve())
            else:
                try:
                    r[k] = str(Path(r[k]).resolve().relative_to(root.resolve()))
                except ValueError:
                    r[k] = f"wav/{Path(r[k]).name}"
        out.append(r)
    return out


def push(cfg, args):
    """Push as a NATIVE HF audio dataset (Parquet + embedded Audio feature).

    Uploading loose wavs + a JSONL of path strings breaks the dataset viewer:
    it parses the JSONL as the dataset, so `audio` renders as an unplayable
    string, and the unreferenced wav files make the reported size incoherent.
    Parquet with a real Audio() column fixes both.

    `ref_audio` is stored as ref_index (a row index), not a path - paths do
    not survive the trip. `pull` maps it back.
    """
    from datasets import Audio, Dataset, DatasetDict

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("export HF_TOKEN first")
    src, repo_id = p(cfg, "data"), args.repo_id or cfg["data"]["hf_repo"]
    sr = cfg["data"]["sample_rate"]

    splits = {}
    for name, key in (("train.jsonl", "train"), ("val.jsonl", "validation")):
        rows = read_jsonl(src / name)
        if not rows:
            continue
        missing = [r for r in rows if not Path(r["audio"]).exists()]
        if missing:
            sys.exit(f"{len(missing)} audio files missing in {name}")
        idx = {r["audio"]: i for i, r in enumerate(rows)}
        d = Dataset.from_dict({
            "audio": [r["audio"] for r in rows],
            "text": [r["text"] for r in rows],
            "ref_index": [idx.get(r.get("ref_audio", ""), -1) for r in rows],
        }).cast_column("audio", Audio(sampling_rate=sr))
        splits[key] = d
        print(f"  {key}: {len(rows)} rows")

    if not splits:
        sys.exit("nothing to push — run: python data.py prepare")

    print(f"\npushing to {repo_id} ({'public' if args.public else 'private'})")
    print("audio is embedded in parquet shards — this is the slow part")
    DatasetDict(splits).push_to_hub(repo_id, private=not args.public, token=token)

    # human-readable card on top of the auto-generated dataset_info
    try:
        from huggingface_hub import DatasetCard
        tag = cfg["data"]["dialect_tag"]
        card = DatasetCard.load(repo_id, token=token)
        card.text = (
            f"# {repo_id}\n\n"
            f"Tunisian Derja speech for VoxCPM2 fine-tuning.\n\n"
            f"- `audio` — {sr} Hz mono, decoded by the `datasets` Audio feature\n"
            f"- `text` — transcript, every row prefixed with `({tag})`\n"
            f"- `ref_index` — row index of a reference clip for in-context "
            f"cloning, or -1\n\n"
            "## Use\n\n```bash\npython data.py pull  # materializes wavs + manifests\n```\n\n"
            "Derived from corpora with their own licence terms; the voices "
            "belong to real speakers. Check before redistributing.\n")
        card.push_to_hub(repo_id, token=token)
    except Exception as e:
        print(f"(card update skipped: {type(e).__name__})")

    print(f"done -> https://huggingface.co/datasets/{repo_id}")


def pull(cfg, args):
    """Download and materialize wavs + manifests. Handles the parquet layout
    and the older loose-file layout."""
    import soundfile as sf
    from datasets import load_dataset
    from tqdm.auto import tqdm

    repo_id = args.repo_id or cfg["data"]["hf_repo"]
    out = p(cfg, "data")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    sr = cfg["data"]["sample_rate"]

    # legacy layout: loose wavs + jsonl of relative paths
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception as e:
        sys.exit(f"cannot read {repo_id}: {e}")

    if "train.jsonl" in files and not any(f.endswith(".parquet") for f in files):
        from huggingface_hub import snapshot_download
        print(f"{repo_id} (file layout) -> {out}")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out),
                          token=token, max_workers=args.workers)
        for name in ("train.jsonl", "val.jsonl"):
            rows = read_jsonl(out / name)
            if rows:
                rows = _rewrite(rows, out, absolute=True)
                write_jsonl(rows, out / name)
                miss = sum(1 for r in rows if not Path(r["audio"]).exists())
                print(f"  {name}: {len(rows)} rows"
                      + (f"  WARNING {miss} missing" if miss else "  ok"))
        return

    print(f"{repo_id} (parquet layout) -> {out}")
    ds = load_dataset(repo_id, token=token)
    for split, manifest in (("train", "train.jsonl"), ("validation", "val.jsonl")):
        if split not in ds:
            continue
        # decode=False: reading ex["audio"]["array"] makes `datasets` decode
        # via torchcodec, which is an extra CUDA-coupled dependency that
        # frequently mismatches the torch build. Pull raw bytes and decode
        # with soundfile instead - same result, one less thing to break.
        s = ds[split].cast_column("audio", Audio(decode=False))
        paths = []
        for i, ex in enumerate(tqdm(s, desc=split)):
            a = ex["audio"]
            src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
            y, sr_in = sf.read(src, dtype="float32", always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            if sr_in != sr:
                import librosa
                y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)
            w = out / "wav" / f"{split}_{i:06d}.wav"
            sf.write(w, np.asarray(y, dtype=np.float32), sr)
            paths.append(str(w.resolve()))

        rows = []
        for i, ex in enumerate(s):
            r = {"audio": paths[i], "text": ex["text"]}
            ri = ex.get("ref_index", -1)
            if isinstance(ri, int) and 0 <= ri < len(paths):
                r["ref_audio"] = paths[ri]
            rows.append(r)
        write_jsonl(rows, out / manifest)
        print(f"  {manifest}: {len(rows)} rows")

    print(f"\n-> {out}/train.jsonl\n-> {out}/val.jsonl")


def stats(cfg, args):
    d = p(cfg, "data")
    f = d / "stats.json"
    if f.exists():
        print(f.read_text(encoding="utf-8"))
    for name in ("train.jsonl", "val.jsonl"):
        rows = read_jsonl(d / name)
        print(f"{name}: {len(rows)} rows")
        if rows:
            print("  e.g.", json.dumps(rows[0], ensure_ascii=False)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare"); a.add_argument("--limit", type=int); a.set_defaults(f=prepare)
    b = sub.add_parser("push"); b.add_argument("--repo-id"); b.add_argument("--public", action="store_true"); b.set_defaults(f=push)
    c = sub.add_parser("pull"); c.add_argument("--repo-id"); c.add_argument("--workers", type=int, default=8); c.set_defaults(f=pull)
    e = sub.add_parser("stats"); e.set_defaults(f=stats)

    args = ap.parse_args()
    args.f(load_config(args.config), args)


if __name__ == "__main__":
    main()
