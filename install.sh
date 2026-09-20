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
# A venv on a NETWORK VOLUME outlives the container that made it. If the new
# image has a different python, the venv's symlinks dangle and `activate`
# silently leaves you on the system interpreter. Detect and rebuild.
if [[ -d "$VENV" ]]; then
  if ! "$VENV/bin/python" -c "import sys" 2>/dev/null; then
    echo "existing venv is broken (built by another image) — rebuilding"
    rm -rf "$VENV"
  fi
fi
[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
source "$VENV/bin/activate"

# Confirm activation actually took. `source activate` only edits shell vars;
# if the venv is unusable, pip silently targets the system python instead.
ACTIVE=$(python -c 'import sys;print(sys.prefix)')
if [[ "$ACTIVE" != "$VENV" ]]; then
  echo "WARNING: venv did not activate (sys.prefix=$ACTIVE, expected $VENV)"
  echo "         installing into the system python instead."
  echo "         that is fine as long as you use the SAME python everywhere:"
  echo "           $(command -v python)"
fi
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

# NOT quiet: this is the step that fails, and -q hides the reason.
echo "installing voxcpm (editable) with $(command -v python)"
if ! pip install -e "$REPO"; then
  echo
  echo "ERROR: pip install -e $REPO failed (output above)."
  echo "Common causes:"
  echo "  - dependency conflict with this image's torch ($(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '?'))"
  echo "  - stale /workspace/venv built by a previous pod: rm -rf $VENV && bash install.sh"
  exit 1
fi

if ! python -c "import voxcpm" 2>/dev/null; then
  echo
  echo "ERROR: pip reported success but 'import voxcpm' still fails."
  echo "  interpreter: $(python -c 'import sys;print(sys.executable)')"
  echo "  pip target : $(pip -V)"
  echo "These two must match. If they do not, the venv is not active:"
  echo "  rm -rf $VENV && bash install.sh"
  exit 1
fi
python -c "import voxcpm;print('voxcpm ok', getattr(voxcpm,'__version__','(no __version__)'))"
command -v voxcpm >/dev/null && echo "voxcpm CLI on PATH" \
  || echo "note: no 'voxcpm' CLI on PATH — 'voxcpm validate' unavailable"

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
