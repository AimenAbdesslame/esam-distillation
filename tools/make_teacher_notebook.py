"""Build a lightweight Kaggle notebook that ONLY runs the ESAM-E teacher.

Runs data.py (3 min) + teacher.py (4 min) -> produces teacher_val.ply and exits.
No student training (saves 110 minutes of GPU time).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "esam_distill")
NOTEBOOKS = os.path.join(ROOT, "notebooks")

MODULES = [("config.py", "PYEOF_CONFIG"),
           ("common.py", "PYEOF_COMMON"),
           ("data.py", "PYEOF_DATA"),
           ("teacher.py", "PYEOF_TEACHER")]


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook():
    cells = [
        md("# ESAM-E Teacher - Export 3D Point Cloud (.ply)\n\n"
           "Runs the online teacher on SceneNN 322 to generate `teacher_val.ply`.\n"
           "Runtime: ~5-7 minutes on T4 GPU."),
        code("%%bash\nset -e\nmkdir -p /kaggle/working/tools/distill /kaggle/working/logs /kaggle/working/distill\n\n" +
             "\n\n".join(f"cat > /kaggle/working/tools/distill/{fn} <<'{delim}'\n" +
                         open(os.path.join(CODE, fn), encoding="utf-8").read().rstrip() + f"\n{delim}"
                         for fn, delim in MODULES)),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
# Auto-detect where Kaggle mounted the input files
SRC=/kaggle/input/notebooks/abdesselamaimen01/notebook8491230ab9
if [ ! -f "$SRC/esam_env.tar.gz" ]; then
  SRC=$(find /kaggle/input -name "esam_env.tar.gz" -exec dirname {} \; | head -n 1)
fi
echo "Using input directory: $SRC"
if [ -z "$SRC" ] || [ ! -f "$SRC/esam_env.tar.gz" ]; then
  echo "ERROR: Could not find esam_env.tar.gz in /kaggle/input!"
  ls -la /kaggle/input
  exit 1
fi

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
df -h "$W" | tail -1
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
export PATH=$W/mc/envs/ESAM/bin:$PATH
export LD_LIBRARY_PATH=$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
cd $W
$W/mc/envs/ESAM/bin/python $W/tools/distill/data.py 2>&1 | tee $W/logs/20_prepare.log
ls -lh $W/distill/frames | tail -3
"""),
        code(r"""%%bash
set -euo pipefail
W=/kaggle/working
export PATH=$W/mc/envs/ESAM/bin:$PATH
export LD_LIBRARY_PATH=$W/mc/envs/ESAM/lib:$W/mc/envs/ESAM/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
cd $W/ESAM
$W/mc/envs/ESAM/bin/python $W/tools/distill/teacher.py 2>&1 | tee $W/logs/30_teacher.log
cat $W/distill/teacher/teacher_metrics.json | tail -40
"""),
        code(r"""%%bash
set +e
W=/kaggle/working
echo "================ OUTPUT POINT CLOUDS ================"
ls -lh $W/distill/teacher/*.ply
""")
    ]

    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}

    out_dir = os.path.join(NOTEBOOKS, "export_teacher_ply")
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, "export_teacher_ply.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    meta = {
        "id": "abdesselamaimen01/esam-export-teacher-ply",
        "id_no": 0,
        "title": "ESAM Export Teacher PLY",
        "code_file": "export_teacher_ply.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "kernel_sources": ["abdesselamaimen01/notebook8491230ab9"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
        "machine_shape": "NvidiaTeslaT4"
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"Generated: {nb_path}")


if __name__ == "__main__":
    build_notebook()
