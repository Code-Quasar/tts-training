"""Shared config loading and small helpers."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CONFIG = HERE / "config.yaml"


def _rebase(cfg: dict, new_root: str) -> None:
    old = cfg["paths"]["root"]
    for k, v in cfg["paths"].items():
        if isinstance(v, str) and v.startswith(old):
            cfg["paths"][k] = v.replace(old, new_root, 1)


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and resolve paths for wherever we're running.

    Resolution order:
      1. $VOL if set
      2. paths.root from the file, if it exists (the pod case)
      3. the repo's parent directory (the laptop case)

    Then, if paths.data still doesn't exist but a sibling of the repo does,
    use that. This is why the scripts work from any working directory - the
    config is found relative to the SCRIPT, not the cwd.
    """
    cfg = yaml.safe_load(Path(path or CONFIG).read_text(encoding="utf-8"))

    vol = os.environ.get("VOL")
    if vol:
        _rebase(cfg, vol)
    elif not Path(cfg["paths"]["root"]).exists():
        local = str(HERE.parent)            # e.g. .../TTS when repo is .../TTS/voxcpm-tn
        _rebase(cfg, local)
        print(f"[config] {cfg['project']}: root not found, using {local}")

    data = Path(cfg["paths"]["data"])
    if not data.exists():
        guess = HERE.parent / "data" / data.name
        if guess.exists():
            cfg["paths"]["data"] = str(guess)
            print(f"[config] data -> {guess}")

    return cfg


def p(cfg, key) -> Path:
    return Path(cfg["paths"][key])


def run_dir(cfg) -> Path:
    return p(cfg, "runs") / cfg["train"]["run_name"]


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- arabic

_TASHKEEL = re.compile(r"[ً-ْٰـ]")
_PUNCT = re.compile(r"[،؛؟٪-٭۔!-/:-@\[-`{-~]")
_WS = re.compile(r"\s+")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def norm_ar(t: str) -> str:
    """Comparison form only. Derja spelling is unstable, so without this any
    CER measures orthographic convention rather than pronunciation."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = _TASHKEEL.sub("", t).translate(_ALEF)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def word_count(t: str) -> int:
    return len(norm_ar(t).split())


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = norm_ar(ref).replace(" ", ""), norm_ar(hyp).replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = norm_ar(ref).split(), norm_ar(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)


def free_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
