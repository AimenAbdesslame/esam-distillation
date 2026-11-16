#!/usr/bin/env bash
# =============================================================================
#  ESAM / ESAM-E environment builder for Kaggle GPU (T4), Python 3.8 + CUDA 11.6
#  Idempotent: every step writes a marker in /kaggle/working/.state/*.ok
#  Logs:       /kaggle/working/logs/<step>.log  and logs/setup.log
#  Re-run:     bash /kaggle/working/setup.sh
#  Notes:      single GPU only (CUDA_VISIBLE_DEVICES=0)
# =============================================================================
set -uo pipefail

W=/kaggle/working
E=$W/mc/envs/ESAM
S=$W/.state
L=$W/logs
mkdir -p "$S" "$L" "$W/outputs" "$W/data" "$W/tools" "$W/tmp"
: > "$W/logs/00_disk_start.txt"
df -h "$W" >> "$W/logs/00_disk_start.txt" 2>&1 || true

export CUDA_HOME="$E"
export LD_LIBRARY_PATH="$E/lib:$E/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=4
export PIP_NO_CACHE_DIR=1
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_TELEMETRY=1
export PATH="$W/mc/bin:$E/bin:$PATH"

LOG=$L/setup.log
: > "$LOG"
say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mark() { touch "$S/$1.ok"; }
done_(){ [ -f "$S/$1.ok" ]; }
FAILED=()

step(){
  local n=$1; shift
  if done_ "$n"; then say "SKIP $n (marker present)"; return 0; fi
  say ">>>  $n"
  local t0; t0=$(date +%s)
  if ( "$@" ) >>"$L/$n.log" 2>&1; then
    mark "$n"
    say "OK   $n ($(( $(date +%s) - t0 ))s)"
    return 0
  else
    FAILED+=("$n")
    say "FAIL $n ($(( $(date +%s) - t0 ))s) -> logs/$n.log"
    tail -n 15 "$L/$n.log" | sed 's/^/     | /' | tee -a "$LOG"
    return 1
  fi
}

# ---------------------------------------------------------------- step 00 ----
apt_libs(){
  local SUDO=""
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1
}

# ---------------------------------------------------------------- step 01 ----
miniconda(){
  [ -x "$W/mc/bin/conda" ] && return 0
  curl -fL --retry 3 -o "$W/m.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$W/m.sh" -b -p "$W/mc"
  rm -f "$W/m.sh"
  "$W/mc/bin/conda" --version
}

# ---------------------------------------------------------------- step 02 ----
py38(){
  if [ -x "$E/bin/python" ]; then
    "$E/bin/python" -c 'import sys; assert sys.version_info[:2]==(3,8), sys.version; print(sys.version)'
    return 0
  fi
  # --override-channels is mandatory to avoid the Anaconda ToS error
  "$W/mc/bin/conda" create -y -n ESAM --override-channels -c conda-forge python=3.8
  "$E/bin/python" --version
}

# ---------------------------------------------------------------- step 03 ----
torch_install(){
  "$E/bin/pip" install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
  "$E/bin/python" - <<'PY'
import torch
assert torch.__version__.startswith('1.13.1'), torch.__version__
assert torch.version.cuda == '11.6', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA not available'
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))
PY
}

# ---------------------------------------------------------------- step 04 ----
conda_tools(){
  # CUDA 11.6 compiler (nvcc 11.6.124) + gcc 11 (system g++13 is too new for CUDA 11.6)
  "$W/mc/bin/conda" install -y -n ESAM --override-channels \
    -c "nvidia/label/cuda-11.6.2" cuda-toolkit
  # 'openblas-devel' does NOT exist on conda-forge: 'openblas' ships cblas.h
  "$W/mc/bin/conda" install -y -n ESAM --override-channels \
    -c conda-forge gxx_linux-64=11 openblas ninja
  "$E/bin/pip" install "setuptools<66" wheel
  "$E/bin/nvcc" --version | tail -n 2
  find "$E" -name cblas.h | head -n 3
}

# ---------------------------------------------------------------- step 05 ----
minkowski_install(){
  [ -d "$W/MinkowskiEngine" ] || \
    git clone --recursive --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git "$W/MinkowskiEngine"
  cd "$W/MinkowskiEngine" || return 1
  git submodule update --init --recursive || true
  rm -rf build
  local H
  H=$(dirname "$(find "$E" -name cblas.h | head -n 1)")
  [ -f "$H/cblas.h" ] || { echo "cblas.h not found under $E"; return 1; }
  local CXX_BIN GCC_BIN
  CXX_BIN=$(find "$E/bin" -maxdepth 1 -name '*-g++' | head -n 1)
  [ -x "$CXX_BIN" ] || { echo "conda g++ not found"; return 1; }
  GCC_BIN="${CXX_BIN%g++}gcc"
  echo "CXX=$CXX_BIN"; "$CXX_BIN" --version | head -n 1
  echo "BLAS include=$H"
  CUDA_HOME="$E" CC="$GCC_BIN" CXX="$CXX_BIN" \
    "$E/bin/python" setup.py install --blas_include_dirs="$H" --blas=openblas
}

# ---------------------------------------------------------------- step 06 ----
minkowski_verify(){
  "$E/bin/python" -c "import MinkowskiEngine as ME; print('MinkowskiEngine', ME.__version__)"
}

# ---------------------------------------------------------------- step 07 ----
mm_stack(){
  "$E/bin/pip" install mmengine==0.10.3
  "$E/bin/pip" install mmcv==2.0.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
  "$E/bin/pip" install mmdet==3.2.0 mmdet3d==1.4.0
  "$E/bin/python" - <<'PY'
import mmcv, mmdet, mmdet3d, mmengine
print('mmcv', mmcv.__version__, '| mmdet', mmdet.__version__,
      '| mmdet3d', mmdet3d.__version__, '| mmengine', mmengine.__version__)
PY
}

# ---------------------------------------------------------------- step 08 ----
esam_repo(){
  [ -d "$W/ESAM/.git" ] || git clone --depth 1 https://github.com/xuxw98/ESAM.git "$W/ESAM"
  [ -d "$W/FastSAM/.git" ] || git clone --depth 1 https://github.com/CASIA-IVA-Lab/FastSAM.git "$W/FastSAM"
  # ESAM does `from fastsam import FastSAM`; make the clone importable everywhere
  echo "$W/FastSAM" > "$E/lib/python3.8/site-packages/fastsam_path.pth"
  mkdir -p "$W/ESAM/data" "$W/ESAM/work_dirs/ckpt" "$W/ESAM/work_dirs/tmp"
  # Do not import fastsam here: ultralytics/psutil are installed in later
  # steps, and the import check in v2 failed for that reason (false negative).
  [ -d "$W/FastSAM/fastsam" ] || { echo "FastSAM package missing"; return 1; }
  [ -f "$W/ESAM/vis_demo/stream_demo.py" ] || { echo "ESAM clone incomplete"; return 1; }
  echo "ESAM and FastSAM clones ready"
}

# ---------------------------------------------------------------- step 09 ----
pointops(){
  cd "$W/ESAM/thirdparty/pointops" || return 1
  local CXX_BIN GCC_BIN
  CXX_BIN=$(find "$E/bin" -maxdepth 1 -name '*-g++' | head -n 1)
  GCC_BIN="${CXX_BIN%g++}gcc"
  CUDA_HOME="$E" CC="$GCC_BIN" CXX="$CXX_BIN" "$E/bin/python" setup.py install
  cd "$W/ESAM"
  "$E/bin/python" -c "import pointops; print('pointops OK', pointops.__file__)"
}

# ---------------------------------------------------------------- step 10 ----
reqs(){
  cd "$W/ESAM" || return 1
  # trimesh==3.23.6 does not exist for py3.8 -> install latest trimesh instead
  grep -viE '^[[:space:]]*trimesh' requirements.txt > "$W/req.txt"
  "$E/bin/pip" install -r "$W/req.txt"
  "$E/bin/pip" install trimesh
  "$E/bin/python" - <<'PY'
import open3d, cv2, trimesh
print('open3d', open3d.__version__, '| cv2', cv2.__version__, '| trimesh', trimesh.__version__)
PY
}

# ---------------------------------------------------------------- step 11 ----
pyg(){
  # oneformer3d/__init__.py imports spconv + torch_scatter + networkx at module
  # load, even when the ESAM-E (MinkowskiEngine) model is the one being used.
  "$E/bin/pip" install "networkx==3.1" "terminaltables==3.1.10"
  # Pin the prebuilt cp38/cu116 wheel.  With only -f <index>, pip prefers the
  # newer 2.1.2 sdist and tries to compile it; that failed in v2.
  "$E/bin/pip" install --no-deps \
    "https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp38-cp38-linux_x86_64.whl"
  "$E/bin/pip" install spconv-cu116==2.3.6
  "$E/bin/python" - <<'PY'
import networkx, terminaltables
import torch_scatter
import spconv.pytorch as spconv
print('networkx', networkx.__version__, '| terminaltables', terminaltables.__version__)
print('torch_scatter', torch_scatter.__version__)
print('spconv', getattr(spconv, '__version__', 'ok'))
PY
}

# ---------------------------------------------------------------- step 12 ----
extras(){
  "$E/bin/pip" install git+https://github.com/facebookresearch/segment-anything.git
  "$E/bin/pip" install ultralytics==8.0.120
  "$E/bin/pip" install psutil gdown huggingface_hub
  "$E/bin/python" - <<'PY'
import segment_anything, ultralytics
print('segment_anything OK | ultralytics', ultralytics.__version__)
PY
}

# ---------------------------------------------------------------- step 12 ----
fastsam_weights(){
  local f="$W/ESAM/data/FastSAM-x.pt"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 100000000 ]; then
    echo "FastSAM weights already present: $(stat -c%s "$f") bytes"; return 0
  fi
  rm -f "$f"
  if ! "$E/bin/gdown" 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv -O "$f"; then
    echo "gdown failed, falling back to Hugging Face mirror"
    curl -fL --retry 3 -o "$f" \
      https://huggingface.co/Uminosachi/FastSAM/resolve/main/FastSAM-x.pt
  fi
  local n; n=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "FastSAM-x.pt = $n bytes"
  [ "$n" -gt 100000000 ]
}

# ---------------------------------------------------------------- step 13 ----
esam_ckpt(){
  local f="$W/ESAM/work_dirs/ckpt/ESAM-E_CA_online_epoch_128.pth"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 500000000 ]; then
    echo "checkpoint already present: $(stat -c%s "$f") bytes"; return 0
  fi
  rm -f "$f"
  curl -fL --retry 3 -o "$f" \
    "https://huggingface.co/XXXCARREY/EmbodiedSAM/resolve/main/ESAM-E_CA_online_epoch_128.pth?download=true"
  local n; n=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "ESAM-E ckpt = $n bytes"
  [ "$n" -gt 500000000 ]
}

# ---------------------------------------------------------------- step 14 ----
scenenn(){
  local T=$W/tmp/scenenn
  local SRC=$T/oni/322
  local BASE="https://huggingface.co/XXXCARREY/EmbodiedSAM/resolve/main/SceneNN"
  local MEMBERS="oni/322/image oni/322/depth oni/322/pose oni/322/timestamp.txt"
  # The 2 GB tar plus the conda package cache filled the 20 GB volume in v2.
  # The env is built by now; drop caches/source before streaming the archive.
  "$W/mc/bin/conda" clean -afy >/dev/null 2>&1 || true
  rm -rf "$W/mc/pkgs" "$W/MinkowskiEngine" "$T"
  mkdir -p "$T"
  echo "disk before SceneNN:"; df -h "$W" | tail -1
  rm -rf "$SRC"
  # Stream instead of saving SceneNN.tar.0: no 2 GB file on the volume.
  curl -fL --retry 3 "$BASE/SceneNN.tar.0?download=true" \
    | tar -xf - -C "$T" $MEMBERS || true
  local n_img n_dep n_pose
  n_img=$(find "$SRC/image" -name '*.png' 2>/dev/null | wc -l)
  n_dep=$(find "$SRC/depth" -name '*.png' 2>/dev/null | wc -l)
  n_pose=$(find "$SRC/pose" -name '*.npy' 2>/dev/null | wc -l)
  echo "part 0: image=$n_img depth=$n_dep pose=$n_pose"
  if [ "$n_img" -lt 20 ] || [ "$n_dep" -lt 20 ] || [ "$n_pose" -lt 20 ]; then
    echo "scene 322 incomplete in part 0 -> streaming all parts"
    { curl -fL --retry 3 "$BASE/SceneNN.tar.0?download=true"
      curl -fL --retry 3 "$BASE/SceneNN.tar.1?download=true"
      curl -fL --retry 3 "$BASE/SceneNN.tar.2?download=true"
    } | tar -xf - -C "$T" $MEMBERS || true
    n_img=$(find "$SRC/image" -name '*.png' 2>/dev/null | wc -l)
    n_dep=$(find "$SRC/depth" -name '*.png' 2>/dev/null | wc -l)
    n_pose=$(find "$SRC/pose" -name '*.npy' 2>/dev/null | wc -l)
  fi
  rm -f "$T"/SceneNN.tar.*
  echo "scene 322: image=$n_img depth=$n_dep pose=$n_pose"
  df -h "$W" | tail -1
  [ "$n_img" -ge 20 ] && [ "$n_dep" -ge 20 ] && [ "$n_pose" -ge 20 ]
}

# ---------------------------------------------------------------- step 15 ----
scene_convert(){
  "$E/bin/python" "$W/tools/convert_scenenn.py" \
    --src "$W/tmp/scenenn/oni/322" \
    --dst "$W/data/scenenn_stream/322" \
    --fx 544.47329 --fy 544.47329 --cx 320 --cy 240 \
    --max_frames 80
  "$E/bin/python" - <<'PY'
import os
root = '/kaggle/working/data/scenenn_stream/322'
for sub in ('color', 'depth', 'pose'):
    n = len(os.listdir(os.path.join(root, sub)))
    print(sub, n)
assert os.path.exists(os.path.join(root, 'intrinsic.txt'))
PY
}

# ---------------------------------------------------------------- step 16 ----
cleanup(){
  # ME source is no longer needed once the extension is installed; removing it
  # keeps the version output small enough to archive and commit.
  rm -rf "$W/tmp/scenenn" "$W/MinkowskiEngine"
  rm -f "$W/m.sh" "$W/req.txt"
  "$W/mc/bin/conda" clean -afy || true
  df -h "$W" | tee -a "$L/setup.log"
}

say "==================== ESAM setup start ===================="
say "host: $(hostname) | user: $(id -un) | disk: $(df -h "$W" | tail -n1)"

step apt            apt_libs
step miniconda      miniconda
step py38           py38
step torch          torch_install
step conda_tools    conda_tools
step minkowski      minkowski_install
step minkowski_ok   minkowski_verify
step mm_stack       mm_stack
step esam_repo      esam_repo
step pointops       pointops
step requirements   reqs
step pyg            pyg
step extras         extras
step fastsam        fastsam_weights
step esam_ckpt      esam_ckpt
step scenenn        scenenn
step scene_convert  scene_convert
step cleanup        cleanup

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL_STEPS_OK $(date -u +%FT%TZ)" > "$S/setup_status.txt"
else
  printf '%s\n' "${FAILED[@]}" > "$S/setup_failed.txt"
  echo "FAILED ${FAILED[*]}" > "$S/setup_status.txt"
fi
say "FAILED steps: ${FAILED[*]:-none}"
say "==================== ESAM setup end ======================"
df -h "$W" | tee -a "$L/setup.log"
exit 0
