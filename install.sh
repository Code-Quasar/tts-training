#!/usr/bin/env bash
# 1/3 - installation. Torch, VoxCPM, deps. Idempotent.
#
#   bash install.sh
#
# Reads paths from config.yaml. Override the volume with VOL=/some/path.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Minimal YAML read - pyyaml isn't installed yet at this point.
# Strips at the FIRST colon (URLs contain colons) and drops inline comments.
cfg() {
  grep -E "^[[:space:]]*$1:" "$HERE/config.yaml" | head -1 \
    | sed -E "s/^[[:space:]]*$1:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//" \
    | tr -d '"'"'"
}
VOL=${VOL:-$(cfg root)}
VENV="${VOL}/venv"
REPO="${VOL}/VoxCPM"

say() { echo -e "\n\033[1m>>> $*\033[0m"; }

say "python (need >=3.10,<3.13)"
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  V=$("$c" -c 'import sys;print(sys.version_info.major*100+sys.version_info.minor)')
  if [[ "$V" -ge 310 && "$V" -lt 313 ]]; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || {
  echo "ERROR: no python in >=3.10,<3.13. found $(python3 --version 2>&1)"
  echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
  exit 1; }
echo "$($PY --version)"

PYV=$("$PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
"$PY" - <<'EOF' || echo "WARNING: Python.h missing -> torch.compile/triton will fail. sudo apt install python3.X-dev build-essential"
import os, sys, sysconfig
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)
EOF

say "venv -> $VENV"
[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip wheel setuptools -q

say "torch"
if python -c "import torch" 2>/dev/null; then
  python -c "import torch;print('present:',torch.__version__)"
else
  if [[ -n "${CUDA:-}" ]]; then IDX="$CUDA"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if   [[ "$DRV" -ge 550 ]]; then IDX=cu124
    elif [[ "$DRV" -ge 530 ]]; then IDX=cu121
    else IDX=cu118; fi
    echo "driver $DRV -> $IDX"
  else IDX=cpu; echo "no GPU detected -> cpu wheels"; fi
  pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$IDX"
fi

say "voxcpm"
VOXCPM_VERSION=${VOXCPM_VERSION:-$(cfg version)}
VOXCPM_URL=$(cfg repo_url)
VOXCPM_URL=${VOXCPM_URL:-https://github.com/OpenBMB/VoxCPM.git}
echo "pinned version: $VOXCPM_VERSION"

if [[ ! -d "$REPO/.git" ]]; then
  command -v git >/dev/null || { echo "ERROR: git not installed"; exit 1; }
  # release tags have appeared as both "v2.0.3" and "2.0.3"
  CLONED=0
  for TAG in "$VOXCPM_VERSION" "${VOXCPM_VERSION#v}" "v${VOXCPM_VERSION#v}"; do
    if git clone --depth 1 --branch "$TAG" "$VOXCPM_URL" "$REPO" 2>/dev/null; then
      echo "cloned at tag $TAG"; CLONED=1; break
    fi
  done
  if [[ "$CLONED" -eq 0 ]]; then
    echo "WARNING: tag $VOXCPM_VERSION unreachable — falling back to main."
    echo "         Pin a real tag in config.yaml for reproducibility."
    git clone --depth 1 "$VOXCPM_URL" "$REPO"
  fi
else
  echo "repo already present at $REPO"
  git -C "$REPO" describe --tags --always 2>/dev/null | sed 's/^/checked out: /'
fi

git -C "$REPO" rev-parse HEAD > "$REPO/.installed_commit" 2>/dev/null || true
pip install -e "$REPO" -q          # editable: the trainer lives in the repo
python - <<'EOF'
import voxcpm
print("voxcpm ok", getattr(voxcpm, "__version__", "(no __version__)"))
EOF
command -v voxcpm >/dev/null && echo "voxcpm CLI on PATH" \
  || echo "WARNING: no 'voxcpm' CLI — `voxcpm validate` will be unavailable"

say "deps"
pip install -q -r "$HERE/requirements.txt"

say "env"
mkdir -p "$VOL/hf_cache" "$VOL/data" "$VOL/runs"
cat > "$VOL/env.sh" <<EOF
source $VENV/bin/activate
export VOL=$VOL
export HF_HOME=$VOL/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
EOF
grep -qF "source $VOL/env.sh" ~/.bashrc 2>/dev/null || echo "source $VOL/env.sh" >> ~/.bashrc
echo "wrote $VOL/env.sh"

say "done"
cat <<EOF
  source $VOL/env.sh
  python check.py            # 2/3 verify + download weights
  python train.py            # 3/3 train
EOF
