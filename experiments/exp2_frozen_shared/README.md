# Experiment 2 - shared layers frozen

**Question.** Does the official ESAM++ recipe work: freeze the shared layers
with their published ESAM-E weights and train only the TinyExpert backbone?

**Why this experiment.** ESAM++ keeps every ESAM layer except the 3D encoder.
The published decoder is calibrated to the heavy sparse U-Net features, so a
random TinyExpert produces features it cannot read. Freezing the shared layers
keeps them intact; the tiny backbone then has to learn a representation the
frozen decoder understands. This is the only change with respect to exp1 -
same teacher, same frames, same split, same budget.

**Setup** (`settings.json`)

| item | value |
|---|---|
| shared layers | frozen (`ESAM_FREEZE_SHARED=1`) |
| backbone | TinyExpert, random init, the only trainable part (153 of 256 tensors) |
| initialisation | published ESAM-E weights, 103 of 400 tensors copied |
| targets | ESAM-E online pseudo instances |
| frames | 0-39 train, 40-59 validate |
| budget | 110 minutes, AdamW, lr 1e-3 |

**Result** (`results/`)

| metric | value |
|---|---|
| steps | 57,814 (110.2 min) |
| best validation loss | **1.074** (exp1: 1.606) |
| AP / AP50 / AP25 on frames 40-59 | **0.0021 / 0.0065 / 0.0177** (exp1: 0.000 / 0.000 / 0.000) |
| predictions above threshold | 100 (exp1: 0) |
| exported cloud | 800,000 points, 94 instance colours (exp1: 1 background colour) |

**What it shows.** Freezing the shared layers is clearly better than training
them: the validation loss drops by a third, the student starts emitting
instances and the exported cloud carries real instance structure. The student is
still far from the teacher (AP 0.0021 against 0.032), so the remaining gap is
not caused by the shared layers any more - it is either the weak teacher targets
on this scene or the fact that a randomly initialised tiny encoder has to learn
from scratch on 40 frames. `exp3_frozen_gt_targets` separates those two.

**Run.** Kaggle kernel `abdesselamaimen01/esam-distill-v10`,
log in `results/run.log`, checkpoint in `outputs/checkpoints/exp2_frozen_shared/`.
