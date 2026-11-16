"""Build the data exploration notebook of the SceneNN 322 study.

usage: python tools/make_exploration_notebook.py

Writes notebooks/data_exploration/data_exploration.ipynb and its kernel metadata.
"""
import json
import os

ROOT = r"D:\ESAM_Work"
OUT = os.path.join(ROOT, "notebooks", "data_exploration")

CELLS = [
    ("markdown", """# SceneNN 322 - data exploration

What the raw SceneNN archive gives us for scene 322, before any model runs.

* `oni/322/point/{frame}.npy` - 20 000 points per frame (xyz in the camera frame + rgb)
* `oni/322/ins/{frame}.npy`   - 20 000 instance ids, 0 = background
* `oni/322/pose/{frame}.npy`  - 4x4 camera to world matrix

The frames used by the study are the ids `1, 41, 81, ...` (80 of them).
"""),
    ("bash", r"""%%bash
set -euo pipefail
W=/kaggle/working
mkdir -p "$W/tmp/scenenn"
URL="https://huggingface.co/XXXCARREY/EmbodiedSAM/resolve/main/SceneNN/SceneNN.tar.0?download=true"
curl -fL --retry 3 "$URL" | tar -xf - -C "$W/tmp/scenenn" --ignore-zeros \
  --wildcards 'oni/322/point/*' 'oni/322/ins/*' 'oni/322/pose/*'
P=$(find "$W/tmp/scenenn" -path '*/322/point/*.npy' | wc -l)
I=$(find "$W/tmp/scenenn" -path '*/322/ins/*.npy' | wc -l)
echo "point/ins frames: $P / $I"
"""),
    ("code", """# ============================================================
# 1. RAW ARRAYS
# ============================================================
import numpy as np
from pathlib import Path

SCENE = Path('/kaggle/working/tmp/scenenn/oni/322')
FRAME_IDS = list(range(1, 3162, 40))

point = np.load(SCENE / 'point' / f'{FRAME_IDS[0]:05d}.npy')
ins = np.load(SCENE / 'ins' / f'{FRAME_IDS[0]:05d}.npy')
pose = np.load(SCENE / 'pose' / f'{FRAME_IDS[0]:05d}.npy')

print('point :', point.shape, point.dtype, 'columns = x, y, z, r, g, b, index')
print('ins   :', ins.shape, ins.dtype, 'unique ids:', np.unique(ins))
print('pose  :', pose.shape, pose.dtype)
print('rgb range:', point[:, 3:6].min(), point[:, 3:6].max())
"""),
    ("code", """# ============================================================
# 2. STATISTICS OVER THE 80 STUDY FRAMES
# ============================================================
frames, counts, sizes = [], [], []
for frame_id in FRAME_IDS:
    ids = np.load(SCENE / 'ins' / f'{frame_id:05d}.npy')
    ids = ids[ids > 0]
    frames.append(frame_id)
    counts.append(len(np.unique(ids)))
    sizes += list(np.bincount(ids)[1:])

print(f'frames                : {len(frames)}')
print(f'instances per frame   : min {min(counts)}, median {int(np.median(counts))}, max {max(counts)}')
print(f'instance size (points): min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}')
print(f'total instances       : {sum(counts)}')
"""),
    ("code", """# ============================================================
# 3. VISUAL CHECK OF ONE FRAME
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

frame_id = FRAME_IDS[0]
point = np.load(SCENE / 'point' / f'{frame_id:05d}.npy')
ids = np.load(SCENE / 'ins' / f'{frame_id:05d}.npy')
pose = np.load(SCENE / 'pose' / f'{frame_id:05d}.npy')

xyz = point[:, :3]
xyz = (pose @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
xyz = xyz @ R_ZUP.T

rng = np.random.default_rng(0)
palette = rng.integers(0, 256, size=(int(ids.max()) + 1, 3))
labels = np.where(ids[:, None] > 0, palette[ids], point[:, 3:6])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)
axes[0].scatter(xyz[:, 0], xyz[:, 2], c=point[:, 3:6] / 255.0, s=0.3, linewidths=0)
axes[0].set_title('RGB colours, front view (X / Z)')
axes[1].scatter(xyz[:, 0], xyz[:, 2], c=labels / 255.0, s=0.3, linewidths=0)
axes[1].set_title('ground truth instances, front view (X / Z)')
for ax in axes:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
plt.tight_layout()
plt.savefig('scenenn322_frame000.png', bbox_inches='tight')
print('saved scenenn322_frame000.png')
"""),
    ("markdown", """**Takeaways**

* The ground truth is exact: 20 000 instance ids per frame, so every model can
  be scored against the same labels (this is what the teacher is measured with).
* The scene is sparse but well populated: about 9 instances per frame with a
  wide spread of instance sizes.
* Points live in the camera frame, the pose moves them to world, and
  `R_x(+90)` turns SceneNN Y-up into the Z-up the ESAM models expect.
"""),
]


def build():
    cells = []
    for kind, source in CELLS:
        if kind == "bash":
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": source.splitlines(True)})
        elif kind == "code":
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": source.splitlines(True)})
        else:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": source.splitlines(True)})
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.8"}},
                "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "data_exploration.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    metadata = {"id": "abdesselamaimen01/esam-data-exploration", "id_no": 0,
                "title": "ESAM data exploration", "code_file": "data_exploration.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "keywords": [], "dataset_sources": [],
                "kernel_sources": ["abdesselamaimen01/notebook8491230ab9"],
                "competition_sources": [], "model_sources": [],
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
                "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    build()
