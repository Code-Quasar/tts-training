#!/usr/bin/env bash
# One command: install -> data -> verify -> train.
#
#   git clone <repo> /workspace/voxcpm-tn
#   cd /workspace/voxcpm-tn
#   export HF_TOKEN=hf_...
#   tmux new -s train
#   bash run.sh
#
# Every stage is skipped if already done, so re-running after a crash or a
# pod restart resumes where it left off.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

cfg() {
  grep -E "^[[:space:]]*$1:" config.yaml | head -1 \
    | sed -E "s/^[[:space:]]*$1:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//" \
    | tr -d '"'"'"
}
VOL=${VOL:-$(cfg root)}
DATA=$(cfg data); DATA=${DATA/\/workspace/$VOL}

step() { echo -e "\n\033[1m━━━ $* ━━━\033[0m\n"; }
ask()  { read -r -p "$1 [y/N] " a; [[ "${a:-n}" =~ ^[Yy]$ ]]; }

# ---------------------------------------------------------------- 0. tmux
if [[ -z "${TMUX:-}" ]] && [[ -t 0 ]]; then
  echo "WARNING: not inside tmux — an SSH drop will kill training."
  command -v tmux >/dev/null || echo "         (tmux not installed: apt install -y tmux)"
  ask "continue anyway?" || { echo "run: tmux new -s train"; exit 0; }
fi

# ---------------------------------------------------------------- 1. install
if [[ -x "$VOL/venv/bin/python" ]]; then
  step "1/4  install — already done"
else
  step "1/4  install  (~5-10 min)"
  bash install.sh
fi
source "$VOL/venv/bin/activate"
export HF_HOME="${HF_HOME:-$VOL/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------- 2. data
if [[ -f "$DATA/train.jsonl" ]]; then
  step "2/4  data — already present"
  wc -l "$DATA"/train.jsonl "$DATA"/val.jsonl 2>/dev/null || true
else
  step "2/4  data"
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN not set. Needed for a private dataset repo."
    echo "  export HF_TOKEN=hf_..."
    ask "try anyway (public dataset)?" || exit 1
  fi
  python data.py pull
fi

# ---------------------------------------------------------------- 3. verify
step "3/4  verify  (downloads ~10GB of model weights on first run)"
if ! python check.py; then
  echo
  echo "check.py reported blockers. Unknown config keys are ignored SILENTLY"
  echo "by the trainer, so a setting that does not apply never raises."
  ask "train anyway?" || exit 1
fi

# ---------------------------------------------------------------- 4. train
step "4/4  train"
echo "Watch progress from a second shell:"
echo "    python status.py --watch"
echo
python train.py "$@"
