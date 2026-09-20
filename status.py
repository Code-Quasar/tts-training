#!/usr/bin/env python3
"""
Live training dashboard. Run in a second shell while train.py works.

    python status.py            # snapshot
    python status.py --watch    # refresh every 20s

Shows checkpoints on disk, the quality trend per checkpoint, the current best,
GPU utilisation, disk headroom, and the tail of the training log.

The column to watch is style_separation. asr_cer falling means the model is
getting more intelligible; style_separation falling means it is losing the
instruction-following you fine-tuned it for. Both matter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from common import load_config, p, run_dir

CKPT_GLOBS = ("step_*", "step-*", "checkpoint-*", "checkpoint_*",
              "ckpt_*", "ckpt-*", "epoch_*", "epoch-*")


def dir_size_gb(d: Path) -> float:
    try:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 ** 3
    except Exception:
        return 0.0


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        rows = []
        for line in out.splitlines():
            u, mu, mt, t = [x.strip() for x in line.split(",")]
            rows.append(f"{u}% util | {int(mu)/1024:.1f}/{int(mt)/1024:.1f} GiB | {t}C")
        return rows
    except Exception:
        return []


def tail(path: Path, n=12):
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


def show(cfg):
    rd = run_dir(cfg)
    print("=" * 74)
    print(f"{cfg['project']} — {cfg['train']['run_name']} "
          f"({cfg['train']['mode'].upper()})   {time.strftime('%H:%M:%S')}")
    print("=" * 74)

    if not rd.exists():
        print(f"run dir not created yet: {rd}")
        return

    # ---- checkpoints
    cks = {}
    for pat in CKPT_GLOBS:
        for c in rd.glob(pat):
            if c.is_dir():
                m = re.search(r"(\d+)\s*$", c.name)
                cks[c] = int(m.group(1)) if m else 0
    cks = sorted(cks, key=lambda c: cks[c])
    total_gb = sum(dir_size_gb(c) for c in cks)
    print(f"\ncheckpoints: {len(cks)}"
          + (f"   latest {cks[-1].name}   {total_gb:.0f} GiB on disk" if cks else ""))

    # ---- quality trend
    trend = rd / "eval_trend.csv"
    if trend.exists():
        rows = list(csv.DictReader(trend.open()))
        if rows:
            print(f"\n{'step':>6} {'asr_cer':>9} {'asr_wer':>9} {'style_sep':>10} "
                  f"{'fails':>6}  verdict")
            print("-" * 74)
            for r in rows[-12:]:
                def f(k, w, d=4):
                    v = r.get(k) or ""
                    try:
                        return f"{float(v):{w}.{d}f}"
                    except ValueError:
                        return f"{'-':>{w}}"
                print(f"{r.get('step',''):>6} {f('asr_cer',9)} {f('asr_wer',9)} "
                      f"{f('style_separation',10)} {r.get('gen_failures','-'):>6}  "
                      f"{r.get('verdict','')}")
            # direction of travel
            cers = [(r["step"], float(r["asr_cer"])) for r in rows
                    if r.get("asr_cer") not in (None, "", "None")]
            if len(cers) >= 2:
                d = cers[-1][1] - cers[0][1]
                print(f"\ncer {cers[0][1]:.4f} -> {cers[-1][1]:.4f} "
                      f"({d:+.4f})  {'improving' if d < 0 else 'REGRESSING'}")
            styles = [float(r["style_separation"]) for r in rows
                      if r.get("style_separation") not in (None, "", "None")]
            if len(styles) >= 2 and styles[-1] < styles[0] * 0.9:
                print("WARNING style_separation is falling — instruction-following")
                print("        is eroding. Lower LR, or add en/zh replay data.")
    else:
        print("\nno evaluations yet (first checkpoint not reached)")

    best = rd / "best.json"
    if best.exists():
        b = json.loads(best.read_text())
        print(f"\nbest: {b['tag']}  cer={b['asr_cer']}\n      {b['checkpoint']}")

    # ---- machine
    for g in gpu_line():
        print(f"\ngpu: {g}")
    try:
        free = shutil.disk_usage(p(cfg, "root")).free / 1024 ** 3
        print(f"disk: {free:.0f} GiB free"
              + ("   LOW — checkpoints are ~24 GiB each" if free < 60 else ""))
    except Exception:
        pass

    # ---- log tail
    log = rd / "train.log"
    if log.exists():
        print(f"\n--- {log.name} ---")
        for line in tail(log, 10):
            print("  " + line[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not args.watch:
        return show(cfg)
    try:
        while True:
            print("\033[2J\033[H", end="")   # clear
            show(cfg)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
