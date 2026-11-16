# ESAM-E &rarr; ESAM++ Tiny distillation on SceneNN 322

Reproducible study of distilling the ESAM-E online teacher into the ESAM++ Tiny
student on one SceneNN scene. Every experiment runs end to end on Kaggle GPUs
from a single notebook.

## Repository layout

```
esam_distill/       the shared library: config + 5 modules
experiments/        one folder per experiment (settings, README, results)
notebooks/          the Kaggle notebook of each experiment, generated from its settings
tools/              notebook builder, Kaggle helpers, HTML to PDF
env/                setup.sh, the environment recipe of the Kaggle GPU image
results/            shared outputs (run logs)
docs/               reports and reference material
upstream/           local checkouts of ESAM and ESAM++ (git ignored)
```

## The library (`esam_distill/`)

| file | role |
|---|---|
| `config.py` | paths, data and training parameters, experiment switches |
| `common.py` | rotation, instance colours, PLY writer, preview, class agnostic AP |
| `data.py` | builds the 80 SceneNN 322 frames (points, super points, instance ids) |
| `teacher.py` | ESAM-E online teacher: pseudo instances, per frame AP, sanity gate |
| `tiny_model.py` | `ScanNet200MixFormer3D_Online_Tiny`, the ESAM++ student model |
| `student.py` | distillation training, evaluation, PLY export |

No upstream ESAM file is modified.

## Experiments

| folder | shared layers | targets | AP | AP50 | AP25 | predictions |
|---|---|---|---|---|---|---|
| `exp1_all_layers` | trained | ESAM-E pseudo instances | 0.0000 | 0.0000 | 0.0000 | 0 |
| `exp2_frozen_shared` | frozen | ESAM-E pseudo instances | 0.0021 | 0.0065 | 0.0177 | 100 |
| `exp3_frozen_gt_targets` | frozen | exact SceneNN ids | **0.0129** | **0.0290** | **0.0623** | 100 |

Teacher reference on the same validation frames: AP 0.0319 / AP50 0.0545 / AP25 0.0729.

Each experiment folder contains `settings.json` (the exact configuration),
`README.md` (why the experiment, what came out), `results/` (metrics, previews,
clouds, run log) and a `notebooks/<experiment>/results.ipynb` that summarises
them. The comparison and its reading are in `docs/experiments_comparison.md`.

## Notebooks

| notebook | where it runs | content |
|---|---|---|
| `notebooks/data_exploration/` | Kaggle (CPU) | raw SceneNN arrays, instance statistics, one frame visualised |
| `notebooks/<experiment>/<experiment>.ipynb` | Kaggle (GPU) | the full pipeline of that experiment |
| `notebooks/<experiment>/results.ipynb` | local | metrics, loss history, previews, cloud statistics |

## Checkpoints

The best checkpoint of every run is available for direct download on GitHub Releases:
* 📦 **[GitHub Release v1.0.0 Checkpoints](https://github.com/AimenAbdesslame/esam-distillation/releases/tag/v1.0.0)**
  * `exp1_all_layers_student_best.pth` (94 MB)
  * `exp2_frozen_shared_student_best.pth` (94 MB)
  * `exp3_frozen_gt_targets_student_best.pth` (94 MB)

## Pipeline

```
data.py  ->  teacher.py  ->  student.py
```

* data: `oni/322/{point,ins,pose}` from the SceneNN archive of `XXXCARREY/EmbodiedSAM`,
  80 frames (ids 1, 41, ... 3161), 20 000 points each
* teacher: ESAM-E CA online, checkpoint `ESAM-E_CA_online_epoch_128.pth`
* student: ESAM++ `TinyExpert` backbone plus the shared ESAM layers, initialised
  from the published ESAM-E weights wherever the tensors match (103 of 400)
* split: frames 0-39 train (40 frames), frames 40-79 validate (40 frames)

## Running an experiment

```powershell
python tools\make_notebook.py exp2_frozen_shared       # build the notebook
kaggle kernels push -p notebooks\exp2_frozen_shared    # run it on Kaggle
python tools\kaggle_fetch.py abdesselamaimen01/esam-exp2-frozen-shared
python tools\fetch_distill_outputs.py abdesselamaimen01/esam-exp2-frozen-shared experiments\exp2_frozen_shared\results
```

## Switches (`esam_distill/config.py`)

| variable | default | meaning |
|---|---|---|
| `ESAM_FREEZE_SHARED` | `0` | `1` freezes every shared layer, trains only the TinyExpert |
| `ESAM_TARGET` | `teacher` | `teacher` distils the ESAM-E pseudo instances, `gt` uses the exact SceneNN instance ids |
| `ESAM_MAX_MINUTES` | `110` | training budget |
| `ESAM_LR` | `1e-3` | learning rate |
| `ESAM_STUDENT` | `distill/student` | output folder of the run |
