"""Build the Kaggle notebook of one experiment.

usage:
    python tools/make_notebook.py exp2_frozen_shared

Reads `experiments/<name>/settings.json` and writes
`notebooks/<name>.ipynb` plus `notebooks/kernel-metadata.json`.
"""
import json
import os
import sys

ROOT = r"D:\ESAM_Work"
CODE = os.path.join(ROOT, "esam_distill")
EXPERIMENTS = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "notebooks")

MODULES = [("config.py", "PYEOF_CONFIG"),
           ("common.py", "PYEOF_COMMON"),
           ("data.py", "PYEOF_DATA"),
           ("teacher.py", "PYEOF_TEACHER"),
           ("tiny_model.py", "PYEOF_TINY"),
           ("student.py", "PYEOF_STUDENT")]


# ============================================================
# 1. NOTEBOOK HELPERS
# ============================================================
def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def script_cell(modules):
    """Cell that writes the experiment code into the Kaggle working directory."""
    lines = ["%%bash", "set -e",
             "mkdir -p /kaggle/working/tools/distill /kaggle/working/logs /kaggle/working/distill",
             ""]
    for filename, delimiter in modules:
        lines.append(f"cat > /kaggle/working/tools/distill/{filename} <<'{delimiter}'")
        lines.append(read(os.path.join(CODE, filename)))
        lines.append(delimiter)
        lines.append("")
    return code("\n".join(lines) + "\n")


# ============================================================
# 2. NOTEBOOK CELLS
# ============================================================
def build_cells(settings):
    env = settings["env"]
    exports = "\n".join(f"export {key}={value}" for key, value in env.items())
    return [
        markdown(f"""# {settings.get("name", settings["title"])}

{settings['why']}

Pipeline on Kaggle (1x T4): restore the ESAM environment, download SceneNN
scene 322, build 80 frames, run the ESAM-E teacher, distil the student,
evaluate and export the point cloud.

Code: `/kaggle/working/tools/distill/` &nbsp;|&nbsp; results: `{settings['student_dir']}`
"""),
        script_cell(MODULES),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
SRC=/kaggle/input/notebooks/abdesselamaimen01/notebook8491230ab9
mkdir -p "$W/logs" "$W/.state" "$W/tools"
echo "restoring saved environment"
tar -xzf "$SRC/esam_env.tar.gz" -C "$W" --ignore-zeros \
  2>&1 | tee "$W/logs/00_restore.log" || true
for d in ESAM FastSAM data; do
  rm -rf "$W/$d"
  cp -a "$SRC/$d" "$W/"
done
cp "$SRC/setup.sh" "$W/setup.sh" 2>/dev/null || true
export PATH="$W/mc/envs/ESAM/bin:$PATH"
export LD_LIBRARY_PATH="$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/ESAM:$W/FastSAM"
env_ok=1
$W/mc/envs/ESAM/bin/python - <<'PY' || env_ok=0
import torch, MinkowskiEngine, mmdet3d, pointops
print('torch', torch.__version__, '| ME', MinkowskiEngine.__version__,
      '| cuda', torch.cuda.is_available())
PY
if [ "$env_ok" -eq 0 ]; then
  echo "saved env incomplete; rebuilding with setup.sh"
  rm -rf "$W/mc"
  for s in esam_repo fastsam esam_ckpt scenenn scene_convert; do
    touch "$W/.state/$s.ok"
  done
  bash "$W/setup.sh" 2>&1 | tee "$W/logs/setup_fallback.log"
  grep -q 'ALL_STEPS_OK' "$W/.state/setup_status.txt"
  export PATH="$W/mc/envs/ESAM/bin:$PATH"
  export LD_LIBRARY_PATH="$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$W/ESAM:$W/FastSAM"
fi
echo "$W/FastSAM" > "$W/mc/envs/ESAM/lib/python3.8/site-packages/fastsam_path.pth"
$W/mc/envs/ESAM/bin/python - <<'PY'
import torch, MinkowskiEngine, mmdet3d, pointops
print('ready:', torch.__version__, MinkowskiEngine.__version__,
      'cuda', torch.cuda.is_available())
PY
df -h "$W" | tail -1
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
if [ ! -d "$W/esamplusplus/.git" ]; then
  git clone --depth 1 https://github.com/google/esamplusplus.git "$W/esamplusplus"
fi
rm -rf "$W/tools/esampp_src"
cp -a "$W/esamplusplus/src" "$W/tools/esampp_src"
touch "$W/tools/esampp_src/__init__.py"
cp "$W/esamplusplus/configs/esamplusplus_online_scannet200_CA.py" "$W/tools/esampp_tiny_scenenn.py"
ls -lh "$W/tools/esampp_src" "$W/tools/esampp_tiny_scenenn.py"
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
mkdir -p "$W/tmp/scenenn" "$W/distill/frames" "$W/logs"
URL="https://huggingface.co/XXXCARREY/EmbodiedSAM/resolve/main/SceneNN/SceneNN.tar.0?download=true"
echo "streaming SceneNN part 0: point, ins, pose"
curl -fL --retry 3 "$URL" | tar -xf - -C "$W/tmp/scenenn" --ignore-zeros \
  --wildcards 'oni/322/point/*' 'oni/322/ins/*' 'oni/322/pose/*' \
  2>&1 | tee "$W/logs/10_extract_npy.log" || true
P=$(find "$W/tmp/scenenn" -path '*/322/point/*.npy' | wc -l)
I=$(find "$W/tmp/scenenn" -path '*/322/ins/*.npy' | wc -l)
echo "point/ins npy count: $P / $I"
[ "$P" -ge 80 ] && [ "$I" -ge 80 ]
df -h "$W" | tail -1
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
export PATH=$W/mc/envs/ESAM/bin:$PATH
export LD_LIBRARY_PATH=$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
cd $W
$W/mc/envs/ESAM/bin/python $W/tools/distill/data.py \
  2>&1 | tee $W/logs/20_prepare.log
ls -lh $W/distill/frames | tail -3
df -h $W | tail -1
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
export PATH=$W/mc/envs/ESAM/bin:$PATH
export LD_LIBRARY_PATH=$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
cd $W/ESAM
$W/mc/envs/ESAM/bin/python $W/tools/distill/teacher.py \
  2>&1 | tee $W/logs/30_teacher.log
cat $W/distill/teacher/teacher_metrics.json | tail -40
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
$W/mc/envs/ESAM/bin/python - <<'PY'
import json
m = json.load(open('/kaggle/working/distill/teacher/teacher_metrics.json'))
print(json.dumps(m['gate'], indent=2))
if not m['gate']['ok']:
    raise SystemExit('TEACHER_GATE_FAILED: inspect previews before training')
print('teacher gate OK')
PY
"""),
        code(f"""%%bash
set +e
W=/kaggle/working
export PATH=$W/mc/envs/ESAM/bin:$PATH
export LD_LIBRARY_PATH=$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${{LD_LIBRARY_PATH:-}}
export CUDA_VISIBLE_DEVICES=0
{exports}
export ESAM_STUDENT=$W/{settings['student_dir']}
cd $W/ESAM
$W/mc/envs/ESAM/bin/python $W/tools/distill/student.py \\
  2>&1 | tee $W/logs/40_student.log
ls -lh $W/{settings['student_dir']} 2>/dev/null | tail -30
"""),
        code(r"""%%bash
set +e
W=/kaggle/working
echo "================ FINAL STATUS ================"
$W/mc/envs/ESAM/bin/python - <<'PY'
import json, os, glob
W = '/kaggle/working'
paths = [f'{W}/distill/teacher/teacher_metrics.json']
paths += sorted(glob.glob(f'{W}/distill/student*/student_ap.json'))
paths += sorted(glob.glob(f'{W}/distill/student*/distill_metrics.json'))
for p in paths:
    print('\n---', p, '---')
    if os.path.isfile(p):
        d = json.load(open(p))
        d.pop('history', None)
        print(json.dumps(d, indent=2)[:3000])
    else:
        print('MISSING')
for root, dirs, files in os.walk(f'{W}/distill'):
    if root.endswith('/frames') or root.endswith('/teacher'):
        continue
    for f in sorted(files):
        q = os.path.join(root, f)
        print(f'{os.path.getsize(q)/1e6:8.2f} MB  {os.path.relpath(q, W)}')
PY
df -h $W | tail -1
"""),
    ]


# ============================================================
# 3. MAIN
# ============================================================
def main(experiment):
    settings = json.load(open(os.path.join(EXPERIMENTS, experiment, "settings.json"),
                              encoding="utf-8"))
    os.makedirs(os.path.join(OUT, experiment), exist_ok=True)
    notebook = {
        "cells": build_cells(settings),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.8"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(OUT, experiment, f"{experiment}.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    slug = f"abdesselamaimen01/esam-{experiment.replace('_', '-')}"
    metadata = {
        "id": slug, "id_no": 0, "title": settings["title"],
        "code_file": f"{experiment}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [],
        "kernel_sources": ["abdesselamaimen01/notebook8491230ab9"],
        "competition_sources": [], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(OUT, experiment, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1)
    print("wrote", path, "and kernel-metadata.json")
    print("kernel id:", slug)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exp1_all_layers")
