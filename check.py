#!/usr/bin/env python3
"""
2/3 - environment check. Download weights and verify the box can train.

Fails loudly on the things that otherwise surface hours into a run:
python version, CUDA, Python.h (triton needs it), disk, config keys the
trainer would silently ignore, and manifest/audio format.

    python check.py                 # full
    python check.py --quick         # skip weight download + generation
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common import load_config, p, read_jsonl  # noqa: E402

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
R = []


def rep(name, status, detail=""):
    R.append((name, status, detail))
    print(f"[{ {OK: '  ok ', WARN: ' warn', FAIL: ' FAIL'}[status] }] {name}"
          + (f"  —  {detail}" if detail else ""))
    return status == OK


# ------------------------------------------------------------------ system

def check_python():
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) < (3, 13):
        return rep("python", OK, f"{v.major}.{v.minor}.{v.micro}")
    return rep("python", FAIL, f"{v.major}.{v.minor} — VoxCPM2 needs >=3.10,<3.13")


def check_headers():
    import sysconfig
    inc = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if inc.exists():
        return rep("python headers", OK, str(inc))
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    return rep("python headers", FAIL,
               f"Python.h missing — triton JIT fails. apt install python{v}-dev")


def check_deps():
    missing = []
    for m in ("yaml", "numpy", "soundfile", "librosa", "tqdm", "huggingface_hub"):
        try:
            __import__(m)
        except ImportError:
            missing.append("pyyaml" if m == "yaml" else m)
    return rep("deps", FAIL if missing else OK,
               f"missing {missing}" if missing else "all present")


def check_torch(need_vram):
    try:
        import torch
    except ImportError:
        return rep("torch", FAIL, "not installed — run install.sh")
    rep("torch", OK, torch.__version__)

    if not torch.cuda.is_available():
        return rep("cuda", FAIL, "no CUDA — CPU build or driver mismatch")
    d = torch.cuda.get_device_properties(0)
    vram = d.total_memory / 1024 ** 3
    rep("cuda", OK, f"{torch.version.cuda}")
    if vram >= need_vram:
        rep("gpu", OK, f"{d.name} — {vram:.1f} GiB")
    elif vram >= 8:
        rep("gpu", WARN, f"{d.name} — {vram:.1f} GiB (< {need_vram}); "
                         f"inference ok, training will be tight")
    else:
        rep("gpu", FAIL, f"{d.name} — {vram:.1f} GiB; cannot train a 2B model")
    rep("bf16", OK if torch.cuda.is_bf16_supported() else WARN,
        "supported" if torch.cuda.is_bf16_supported() else "unsupported — set bf16: false")

    try:  # a real kernel; import success proves nothing
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        (a @ a).sum().item()
        torch.cuda.synchronize()
        rep("cuda kernel", OK, "bf16 matmul")
    except Exception as e:
        rep("cuda kernel", FAIL, f"{type(e).__name__}: {str(e)[:90]}")
    return True


def check_disk(path, full_sft):
    need = 90 if full_sft else 40
    path = Path(path) if Path(path).exists() else Path(".")
    free = shutil.disk_usage(path).free / 1024 ** 3
    return rep("disk", OK if free >= need else FAIL,
               f"{free:.0f} GiB free at {path} (need ~{need})")


# ------------------------------------------------------------------ project

def check_voxcpm(cfg):
    import subprocess

    # Report the interpreter first: a venv/system-python mix-up is the most
    # common cause of "No module named voxcpm" on a pod whose image already
    # ships torch.
    venv = Path(cfg["paths"].get("venv", "")) / "bin" / "python"
    in_venv = str(venv) in sys.executable or sys.prefix != sys.base_prefix
    rep("interpreter", OK if in_venv else WARN,
        f"{sys.executable}" + ("" if in_venv else
                               f" — NOT a venv. expected {venv}; "
                               f"run: source {venv.parent}/activate"))

    imported = True
    try:
        import voxcpm
        rep("voxcpm", OK, getattr(voxcpm, "__version__", "importable"))
    except ImportError as e:
        imported = False
        rep("voxcpm", FAIL, f"{e} — run install.sh (do NOT skip its output)")

    # keep going even if the import failed - the repo state is the diagnosis
    repo = p(cfg, "voxcpm_repo")
    if not repo.exists():
        return rep("trainer repo", FAIL,
                   f"{repo} not found — install.sh never cloned it")
    n = len(list(repo.glob("*")))
    rep("trainer repo", OK if n else FAIL,
        f"{repo} ({n} entries)" + ("" if n else " — empty, clone failed"))
    if not imported:
        rep("voxcpm install", FAIL,
            f"repo exists but not importable — run: pip install -e {repo}")

    # what is actually checked out vs what config.yaml pins
    want = cfg.get("voxcpm", {}).get("version", "")
    try:
        got = subprocess.run(["git", "describe", "--tags", "--always"],
                             cwd=repo, capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except Exception:
        got = ""
    if want and got:
        match = want.lstrip("v") in got.lstrip("v")
        rep("voxcpm version", OK if match else WARN,
            f"checked out {got}, config pins {want}"
            + ("" if match else " — mismatch; delete the repo and rerun install.sh"))
    elif got:
        rep("voxcpm version", WARN, f"checked out {got} (config pins nothing)")

    script = next(repo.rglob("train_voxcpm_finetune.py"), None)
    if not script:
        return rep("trainer script", FAIL,
                   "train_voxcpm_finetune.py not found — version too old? "
                   "fine-tuning needs >= 2.0.3")
    rep("trainer script", OK, str(script.relative_to(repo)))

    if shutil.which("voxcpm"):
        rep("voxcpm CLI", OK, "`voxcpm validate` available")
    else:
        rep("voxcpm CLI", WARN, "not on PATH — manifest validation unavailable")
    return True


def check_config_keys(cfg):
    """Unknown YAML keys are ignored SILENTLY - a wrong learning_rate would
    never raise. Diff our keys against the repo's own example configs."""
    import yaml
    repo = p(cfg, "voxcpm_repo")
    refs = []
    for pat in ("conf/**/*.yaml", "configs/**/*.yaml", "examples/**/*.yaml"):
        refs += list(repo.glob(pat))
    if not refs:
        return rep("config keys", WARN, "no reference configs found in repo")

    known = set()
    for f in refs:
        try:
            d = yaml.safe_load(f.read_text())
            if isinstance(d, dict):
                known |= set(d)
                for v in d.values():
                    if isinstance(v, dict):
                        known |= set(v)
        except Exception:
            pass

    mine = set(cfg["train"]) - {"run_name", "mode", "base_model"}
    unknown = sorted(mine - known)
    if unknown:
        return rep("config keys", WARN,
                   f"not in repo configs (may be ignored): {', '.join(unknown)}")
    return rep("config keys", OK, f"checked against {len(refs)} reference config(s)")


def check_data(cfg):
    d = p(cfg, "data")
    ok = True
    for name in ("train.jsonl", "val.jsonl"):
        f = d / name
        rows = read_jsonl(f)
        if not rows:
            ok &= rep(f"data/{name}", FAIL if name == "train.jsonl" else WARN,
                      f"missing or empty — run: python data.py prepare")
            continue
        bad = [r for r in rows if "audio" not in r or "text" not in r]
        missing = [r for r in rows[:300] if not Path(r["audio"]).exists()]
        tag = cfg["data"]["dialect_tag"]
        tagged = sum(f"({tag}" in r.get("text", "") for r in rows)
        detail = f"{len(rows)} rows, {tagged/len(rows):.0%} tagged"
        if bad or missing:
            ok &= rep(f"data/{name}", FAIL,
                      detail + f" — {len(bad)} malformed, {len(missing)}/300 audio missing")
        else:
            rep(f"data/{name}", OK, detail)
            try:
                import soundfile as sf
                srs = {sf.info(r["audio"]).samplerate for r in rows[:100]}
                chs = {sf.info(r["audio"]).channels for r in rows[:100]}
                want = cfg["data"]["sample_rate"]
                ok &= rep("audio format",
                          OK if srs == {want} and chs == {1} else FAIL,
                          f"sr={srs} channels={chs} (need {want}, mono)")
            except ImportError:
                rep("audio format", WARN, "soundfile missing")
    return ok


def download_and_generate(cfg, quick):
    if quick:
        rep("model weights", WARN, "skipped (--quick)")
        return rep("generation", WARN, "skipped (--quick)")
    model_id = cfg["train"]["base_model"]
    try:
        from huggingface_hub import snapshot_download
        print(f"\ndownloading {model_id} (~10 GB, cached)…")
        path = snapshot_download(model_id, token=os.environ.get("HF_TOKEN"))
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        rep("model weights", OK, f"{size/1024**3:.1f} GiB")
    except Exception as e:
        return rep("model weights", FAIL, f"{type(e).__name__}: {str(e)[:90]}")

    try:
        import numpy as np
        import soundfile as sf
        from voxcpm import VoxCPM
        print("generating one sentence…")
        m = VoxCPM.from_pretrained(model_id)
        tag = cfg["data"]["dialect_tag"]
        wav = np.asarray(m.generate(text=f"({tag}) عسلامة، هذا اختبار."),
                         dtype=np.float32)
        out = Path(tempfile.gettempdir()) / "voxcpm_check.wav"
        sf.write(out, wav, 48000)
        return rep("generation", OK if wav.size else FAIL,
                   f"{len(wav)/48000:.2f}s -> {out}")
    except Exception as e:
        msg = str(e)
        hint = " (apt install python3.x-dev)" if "Python.h" in msg else \
               " (low VRAM)" if "out of memory" in msg.lower() else ""
        return rep("generation", FAIL, f"{type(e).__name__}: {msg[:90]}{hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vram", type=float, default=24.0)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("=" * 66)
    print(f"{cfg['project']} — environment check")
    print("=" * 66)

    check_python()
    check_headers()
    check_deps()
    check_torch(args.vram)
    check_disk(cfg["paths"]["root"], cfg["train"]["mode"] == "full")
    check_voxcpm(cfg)
    check_config_keys(cfg)
    check_data(cfg)
    download_and_generate(cfg, args.quick)

    fails = [r for r in R if r[1] == FAIL]
    warns = [r for r in R if r[1] == WARN]
    print("\n" + "=" * 66)
    print(f"{len(R)-len(fails)-len(warns)} pass, {len(warns)} warn, {len(fails)} fail")
    for n, _, d in fails:
        print(f"  FAIL  {n}: {d}")
    for n, _, d in warns:
        print(f"  warn  {n}: {d}")
    print("=" * 66)
    if fails:
        sys.exit(1)
    print("Ready.  next:  python train.py")


if __name__ == "__main__":
    main()
