# ESAM++ distillation - state (read this first)

## Runs

| experiment | what | slug | status |
|---|---|---|---|
| exp1 | all layers trained, targets = ESAM-E pseudo instances | `abdesselamaimen01/esam-distill-v9` | done |
| exp2 | shared layers frozen, same targets | `abdesselamaimen01/esam-distill-v10` | done |
| exp3 | shared layers frozen, targets = exact SceneNN ids | `abdesselamaimen01/esam-exp3-frozen-gt-targets` | done |

## Results

| | teacher ESAM-E | exp1 (all layers) | exp2 (frozen) | exp3 (frozen + GT) |
|---|---|---|---|---|
| AP / AP50 / AP25 | 0.032 / 0.055 / 0.073 | 0.000 / 0.000 / 0.000 | 0.0021 / 0.0065 / 0.0177 | 0.0129 / 0.0290 / 0.0623 |
| predictions | - | 0 | 100 | 100 |
| best validation loss | - | 1.606 | 1.074 | 0.956 |
| instance colours in the cloud | - | 1 | 94 | 86 |

Freezing the shared layers is necessary, and the teacher pseudo labels were the
second bottleneck (exp3 is six times better in AP with the exact ids).

## Repository layout

```
esam_distill/     config.py common.py data.py teacher.py tiny_model.py student.py
experiments/      exp1_all_layers/ exp2_frozen_shared/ exp3_frozen_gt_targets/
                   each with settings.json, README.md, results/
notebooks/        data_exploration/ + one folder per experiment
                   (<experiment>.ipynb = Kaggle run, results.ipynb = local summary)
tools/            make_notebook.py, make_exploration_notebook.py, make_results_notebooks.py,
                  kaggle_fetch.py, fetch_distill_outputs.py, html_to_pdf.py
env/              setup.sh
outputs/          checkpoints/<experiment>/student_best.pth (98.6 MB, git ignored)
docs/             model_comparison.md, experiments_comparison.md, reports
upstream/         ESAM and ESAM++ checkouts (git ignored)
```

## Commands

```powershell
python tools\make_notebook.py exp3_frozen_gt_targets        # rebuild a pipeline notebook
python tools\make_exploration_notebook.py                   # rebuild the exploration notebook
python tools\make_results_notebooks.py                      # rebuild the result notebooks
kaggle kernels push -p notebooks\exp3_frozen_gt_targets     # run it
python tools\kaggle_fetch.py <slug>                         # status + log
python tools\fetch_distill_outputs.py <slug> <folder>       # metrics, previews, PLYs
python tools\fetch_distill_outputs.py <slug> --checkpoints outputs\checkpoints\<name>
```

## Still open

* Next experiment candidate: feature-level distillation (the paper's recipe)
  with exact ground-truth targets and more scenes.
* Threshold sweep on exp2/exp3 (both output exactly 100 predictions; optimal threshold unknown).
* ScanNet++ dataset access: requested, not yet granted.
* ESAM++ pretrained weights: not publicly released; author contacted, no reply.
* MeshLab screen recordings: TODO (add links in Presentation.pdf cover or README).
