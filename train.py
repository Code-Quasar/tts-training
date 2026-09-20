#!/usr/bin/env python3
"""
3/3 - training driver.

Generates the trainer config from config.yaml, runs the official VoxCPM
trainer, and adds the parts it does not provide: a baseline measurement,
automatic evaluation of every checkpoint as it lands, a best-checkpoint
record, and resume.

    python train.py                  # full run
    python train.py --dry-run        # write config + checks, do not train
    python train.py --resume         # continue from the latest checkpoint
    python train.py --no-auto-eval   # train only

Checkpointing and validation are driven by config.yaml:
    train.save_interval     checkpoint interval (steps)
    train.valid_interval    validation-loss interval on val.jsonl
    train.epochs            converted to max_steps from the manifest size

Resume is automatic: the trainer picks up from `latest/` under save_path if
it exists, and saves a checkpoint on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from common import HERE, load_config, p, read_jsonl, run_dir

TREND_FIELDS = ["step", "tag", "asr_cer", "asr_wer", "style_separation",
                "gen_failures", "verdict"]

# Checkpoint directory naming is not guaranteed across VoxCPM releases.
# Globbing only "step_*" would silently find nothing and you'd get no
# evaluation at all, with no error. Match the common conventions instead.
CKPT_GLOBS = ("step_*", "step-*", "checkpoint-*", "checkpoint_*",
              "ckpt_*", "ckpt-*", "epoch_*", "epoch-*")


def step_of(d: Path) -> int:
    m = re.search(r"(\d+)\s*$", d.name)
    return int(m.group(1)) if m else 0


def find_checkpoints(rd: Path) -> list[Path]:
    found = {}
    for pat in CKPT_GLOBS:
        for c in rd.glob(pat):
            if c.is_dir():
                found[c] = step_of(c)
    return sorted(found, key=lambda c: found[c])


def resolve_model_path(model_id: str) -> str:
    """The trainer's `pretrained_path` must be a LOCAL DIRECTORY.

    Passing a repo id makes it look for 'openbmb/VoxCPM2/config.json' as a
    filesystem path. So keep the portable repo id in config.yaml and resolve
    it to the cached snapshot dir here. With HF_HOME on the network volume
    that cache survives pod restarts; a hard-coded /root/.cache path does not.
    """
    p = Path(model_id)
    if p.exists() and (p / "config.json").exists():
        return str(p.resolve())
    if p.is_absolute():
        raise SystemExit(
            f"base_model points at {p}, which has no config.json.\n"
            f"Set it to 'openbmb/VoxCPM2' in config.yaml.")

    from huggingface_hub import snapshot_download
    print(f"resolving {model_id} -> local snapshot ...")
    path = snapshot_download(model_id, token=os.environ.get("HF_TOKEN"))
    if not (Path(path) / "config.json").exists():
        raise SystemExit(f"snapshot at {path} has no config.json")
    print(f"  {path}")
    return path


def prune_checkpoints(rd: Path, keep: int, protect=()) -> list:
    """Delete old checkpoints, keeping the newest `keep` plus anything
    protected. VoxCPM2 keeps every checkpoint and has no save_total_limit,
    so a full-SFT run (~24 GB each) fills the volume quickly.

    Never removes `latest/` or whatever it points at - the trainer resumes
    from there, and deleting it breaks restart-after-crash.
    """
    import shutil

    cks = find_checkpoints(rd)
    if len(cks) <= max(keep, 1):
        return []

    protected = {c.resolve() for c in cks[-keep:]}      # newest N
    latest = rd / "latest"
    if latest.exists():
        try:
            protected.add(latest.resolve())            # symlink target too
        except OSError:
            pass
    for extra in protect:
        if extra:
            try:
                protected.add(Path(extra).resolve())
            except OSError:
                pass

    removed = []
    for c in cks[:-keep]:
        try:
            if c.resolve() in protected:
                continue
            gb = sum(f.stat().st_size for f in c.rglob("*") if f.is_file()) / 1024 ** 3
            shutil.rmtree(c)
            removed.append((c.name, gb))
        except Exception as e:
            print(f"[prune] could not remove {c.name}: {type(e).__name__}: {e}")
    if removed:
        freed = sum(g for _, g in removed)
        print(f"[prune] removed {', '.join(n for n, _ in removed)} "
              f"— freed {freed:.1f} GiB", flush=True)
    return removed


def trainer_help(script: Path) -> str:
    """Ask the trainer what it supports rather than assuming."""
    try:
        out = subprocess.run([sys.executable, str(script), "--help"],
                             capture_output=True, text=True, timeout=180)
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def detect_flag(help_text: str, candidates, default=None):
    for flag in candidates:
        if re.search(rf"(^|\s){re.escape(flag)}(\s|=|,|\b)", help_text):
            return flag
    return default


CONFIG_FLAGS = ("--config_path", "--config-path", "--config", "--cfg", "--conf")
RESUME_FLAGS = ("--resume_from_checkpoint", "--resume-from-checkpoint",
                "--resume_from", "--resume")


def detect_resume_flag(script: Path) -> str | None:
    return detect_flag(trainer_help(script), RESUME_FLAGS)


# ------------------------------------------------------------------ config

def build_trainer_config(cfg, out_path: Path) -> Path:
    """Emit VoxCPM2's own fine-tuning schema.

    Key names come from the official guide, not invented - the trainer
    ignores unknown keys silently, so `save_every` instead of `save_interval`
    would leave checkpointing on its default with no warning.

    Note the trainer is STEP-based. `epochs` in our config is converted to
    max_steps using the actual manifest length.
    """
    t, d = cfg["train"], cfg["data"]
    rd = run_dir(cfg)
    data_dir = p(cfg, "data")

    n_rows = len(read_jsonl(data_dir / "train.jsonl"))
    eff_batch = max(1, t["batch_size"] * t["grad_accum_steps"])
    steps_per_epoch = max(1, -(-n_rows // eff_batch))          # ceil
    max_steps = max(1, int(steps_per_epoch * float(t["epochs"])))
    warmup = max(1, int(max_steps * float(t["warmup_ratio"])))

    val = data_dir / "val.jsonl"
    tc = {
        # must be a local directory, not a repo id
        "pretrained_path": resolve_model_path(str(t["base_model"])),
        "train_manifest": str(data_dir / "train.jsonl"),
        # documented as optional; empty string disables validation
        "val_manifest": str(val) if val.exists() else "",
        "save_path": str(rd),
        "tensorboard": str(rd / "tb"),
        "sample_rate": d["sample_rate"],
        "out_sample_rate": 48000,
        "batch_size": t["batch_size"],
        "grad_accum_steps": t["grad_accum_steps"],
        "num_workers": t.get("num_workers", 4),
        "max_batch_tokens": t["max_batch_tokens"],
        "num_iters": max_steps,
        "max_steps": max_steps,
        "warmup_steps": warmup,
        "learning_rate": float(t["learning_rate"]),
        "weight_decay": float(t["weight_decay"]),
        "log_interval": t.get("log_interval", 10),
        "valid_interval": t["valid_interval"],
        "save_interval": t["save_interval"],
    }
    tc.update(t.get("extra") or {})       # verified-only passthrough
    if t["mode"] == "lora":
        tc["lora"] = cfg["lora"]          # presence of this key == LoRA mode

    print(f"\n{n_rows} train rows / effective batch {eff_batch} "
          f"= {steps_per_epoch} steps per epoch")
    print(f"epochs {t['epochs']} -> max_steps {max_steps}, warmup {warmup}")
    print(f"checkpoint every {tc['save_interval']} steps "
          f"(~{max_steps // tc['save_interval']} checkpoints), "
          f"validation every {tc['valid_interval']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(tc, sort_keys=False), encoding="utf-8")
    return out_path


def sanity(cfg) -> bool:
    ok = True
    d = p(cfg, "data")
    train = read_jsonl(d / "train.jsonl")
    val = read_jsonl(d / "val.jsonl")
    if not train:
        print("FAIL  no train.jsonl — run: python data.py prepare")
        ok = False
    if not val:
        print("WARN  no val.jsonl — validation loss will be unavailable")
    if train:
        missing = [r for r in train[:200] if not Path(r["audio"]).exists()]
        if missing:
            print(f"FAIL  {len(missing)}/200 audio files missing "
                  f"— on a new machine run: python data.py pull")
            ok = False
        tag = cfg["data"]["dialect_tag"]
        tagged = sum(f"({tag}" in r["text"] for r in train)
        print(f"      train {len(train)} | val {len(val)} | tagged {tagged/len(train):.0%}")

    repo = p(cfg, "voxcpm_repo")
    script = next(repo.rglob("train_voxcpm_finetune.py"), None) if repo.exists() else None
    if not script:
        print(f"FAIL  trainer not found under {repo} — run install.sh")
        ok = False
    t = cfg["train"]
    if t["mode"] == "full" and float(t["learning_rate"]) > 5e-5:
        print(f"WARN  lr={t['learning_rate']} is high for full SFT (use ~1e-5)")
    if t["mode"] == "lora" and float(t["learning_rate"]) < 5e-5:
        print(f"WARN  lr={t['learning_rate']} is low for LoRA (use ~1e-4)")
    return ok, script


# ------------------------------------------------------------------ eval

def run_eval(args_list) -> bool:
    cmd = [sys.executable, str(HERE / "evaluate.py"), *args_list]
    return subprocess.run(cmd, cwd=HERE).returncode == 0


def load_results(tag):
    f = HERE / "eval_out" / tag / "results.json"
    return json.loads(f.read_text()) if f.exists() else None


def verdict(base, cur) -> str:
    if not base:
        return "no-baseline"
    bits = []
    bc, cc = base.get("asr_cer"), cur.get("asr_cer")
    if bc is not None and cc is not None:
        bits.append("cer+" if cc < bc else ("cer-" if cc > bc else "cer="))
    bs, cs = base.get("style_separation"), cur.get("style_separation")
    if bs and cs is not None:
        bits.append("STYLE-DROP" if cs < bs * 0.9 else
                    ("style+" if cs > bs * 1.1 else "style="))
    if cur.get("gen_failures"):
        bits.append(f"fails={cur['gen_failures']}")
    return " ".join(bits)


def watcher(cfg, stop: threading.Event):
    """Evaluate each checkpoint as it appears; record the trend."""
    rd = run_dir(cfg)
    trend = rd / "eval_trend.csv"
    if not trend.exists():
        rd.mkdir(parents=True, exist_ok=True)
        with open(trend, "w", newline="") as f:
            csv.DictWriter(f, TREND_FIELDS).writeheader()

    base = load_results(cfg["eval"]["baseline_tag"])
    seen, best = set(), None

    waited = 0
    while not stop.is_set():
        found = find_checkpoints(rd)
        if not found and waited >= 600:
            print(f"[watcher] no checkpoint dirs under {rd} after 10 min. "
                  f"Expected one of {CKPT_GLOBS}. Auto-eval is idle — check "
                  f"what the trainer actually writes.", flush=True)
            waited = 0
        for ck in found:
            if ck.name in seen:
                continue
            seen.add(ck.name)
            step = step_of(ck)
            if step <= 0:
                # VoxCPM2 writes step_0000000 before training starts. It is
                # byte-identical to the base model, so evaluating it wastes
                # minutes of GPU and adds a meaningless row to the trend.
                print(f"[watcher] skipping {ck.name} (pre-training checkpoint)",
                      flush=True)
                continue
            time.sleep(15)                      # let the writer finish
            if not any(ck.iterdir()):
                continue
            tag = f"{rd.name}_s{step}"
            print(f"\n[watcher] evaluating {ck.name}", flush=True)
            if not run_eval(["--ckpt", str(ck), "--tag", tag]):
                continue
            res = load_results(tag)
            if not res:
                continue
            v = verdict(base, res)
            with open(trend, "a", newline="") as f:
                csv.DictWriter(f, TREND_FIELDS).writerow({
                    "step": step, "tag": tag,
                    "asr_cer": res.get("asr_cer"), "asr_wer": res.get("asr_wer"),
                    "style_separation": res.get("style_separation"),
                    "gen_failures": res.get("gen_failures"), "verdict": v})
            print(f"[watcher] {tag}: cer={res.get('asr_cer')} "
                  f"style={res.get('style_separation')} -> {v}", flush=True)
            if "STYLE-DROP" in v:
                print("[watcher] !! style control eroding — consider stopping, "
                      "then lower LR or add en/zh replay", flush=True)
            cer = res.get("asr_cer")
            if cer is not None and (best is None or cer < best[1]):
                best = (tag, cer, str(ck))
                (rd / "best.json").write_text(json.dumps(
                    {"tag": tag, "asr_cer": cer, "checkpoint": str(ck)},
                    indent=2), encoding="utf-8")
                print(f"[watcher] new best: {tag} (cer {cer})", flush=True)

            # prune only AFTER evaluating - otherwise we'd delete a
            # checkpoint before knowing whether it was the good one
            t = cfg["train"]
            keep = int(t.get("keep_checkpoints", 2))
            protect = [best[2]] if (best and t.get("protect_best", True)) else []
            prune_checkpoints(rd, keep, protect)
        stop.wait(60)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-auto-eval", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--prune", action="store_true",
                    help="delete old checkpoints now and exit (safe mid-run)")
    ap.add_argument("--keep", type=int,
                    help="override train.keep_checkpoints for --prune")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)

    if args.prune:
        keep = args.keep or int(cfg["train"].get("keep_checkpoints", 2))
        protect = []
        bj = rd / "best.json"
        if bj.exists() and cfg["train"].get("protect_best", True):
            protect.append(json.loads(bj.read_text()).get("checkpoint"))
        cks = find_checkpoints(rd)
        print(f"{len(cks)} checkpoint(s) in {rd}; keeping newest {keep}"
              + (f" + best ({Path(protect[0]).name})" if protect else ""))
        removed = prune_checkpoints(rd, keep, protect)
        if not removed:
            print("nothing to remove")
        import shutil as _sh
        print(f"free: {_sh.disk_usage(rd).free / 1024 ** 3:.0f} GiB")
        return

    print("=" * 66)
    print(f"{cfg['project']} — {cfg['train']['run_name']} "
          f"({cfg['train']['mode'].upper()} SFT)")
    print("=" * 66)

    ok, script = sanity(cfg)
    if not ok:
        sys.exit(1)

    tconf = build_trainer_config(cfg, rd / "trainer_config.yaml")
    print(f"\ntrainer config -> {tconf}")
    print(f"checkpoints    -> {rd}/step_XXXXXXX  + latest/")
    print(f"validation     -> every {cfg['train']['valid_interval']} steps")

    if args.dry_run:
        print("\n--dry-run: stopping before training")
        print(tconf.read_text())
        return

    # baseline, so "better" has a reference point
    btag = cfg["eval"]["baseline_tag"]
    if not args.no_baseline and not load_results(btag):
        print(f"\nbaseline eval ({btag}) — before any training")
        if not run_eval(["--tag", btag]):
            print("baseline failed; comparisons will be unavailable")

    stop = threading.Event()
    th = None
    if cfg["eval"]["auto"] and not args.no_auto_eval:
        th = threading.Thread(target=watcher, args=(cfg, stop), daemon=True)
        th.start()
        print("checkpoint watcher started")

    help_text = trainer_help(script)
    cfg_flag = detect_flag(help_text, CONFIG_FLAGS)
    if cfg_flag is None:
        if help_text:
            print(f"\nERROR: the trainer exposes none of {CONFIG_FLAGS}.")
            print("Its --help says:\n" + help_text[:1200])
            print("\nEdit the `cmd = [...]` line in train.py to match.")
            sys.exit(1)
        cfg_flag = "--config_path"      # --help unavailable; use the documented name
        print("note: could not read trainer --help; assuming --config_path")
    elif cfg_flag != "--config_path":
        print(f"note: trainer uses {cfg_flag} (not --config_path)")

    cmd = [sys.executable, str(script), cfg_flag, str(tconf)]

    # VoxCPM2 resumes AUTOMATICALLY from `latest/` under save_path - there is
    # normally no flag to pass. Only add one if this build exposes it.
    latest = rd / "latest"
    if latest.exists():
        print(f"note: {latest} exists — the trainer will resume from it "
              f"automatically. Delete it to start fresh.")
    if args.resume:
        flag = detect_flag(help_text, RESUME_FLAGS)
        cks = find_checkpoints(rd)
        if flag and cks:
            cmd += [flag, str(cks[-1])]
            print(f"resuming from {cks[-1].name} via {flag}")
        elif latest.exists():
            print("--resume: handled automatically via latest/")
        else:
            print("--resume: nothing to resume from, starting fresh")

    log = rd / "train.log"
    print(f"\n{' '.join(cmd)}\nlog -> {log}\n")
    rc = 0
    try:
        with open(log, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=p(cfg, "voxcpm_repo"),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
                lf.flush()
            rc = proc.wait()
    except KeyboardInterrupt:
        print("\ninterrupted — checkpoints on disk are still usable")
        rc = 130
    finally:
        if th:
            print("waiting for the watcher to finish the last checkpoint…")
            time.sleep(20)
            stop.set()
            th.join(timeout=600)

    trend = rd / "eval_trend.csv"
    print("\n" + "=" * 66)
    if trend.exists():
        print(trend.read_text())
    if (rd / "best.json").exists():
        best = json.loads((rd / "best.json").read_text())
        print(f"best: {best['tag']}  cer={best['asr_cer']}\n  {best['checkpoint']}")
        print(f"\ncompare + listen:\n"
              f"  python evaluate.py --compare {btag} {best['tag']}\n"
              f"  python evaluate.py --ab {btag} {best['tag']}")
    print("=" * 66)
    sys.exit(rc)


if __name__ == "__main__":
    main()
