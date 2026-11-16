# Experiment 3 - frozen shared layers, ground truth targets

**Question.** Is the ESAM-E teacher the limiting factor of the distillation?

**Why this experiment.** `exp2_frozen_shared` shows that freezing the shared
layers makes the student learn (loss 1.61 -> 1.07, 100 predictions instead of
0), but the instances stay weak. Two explanations remain:

1. the targets are weak - the teacher itself only reaches AP 0.032 on this scene,
2. the recipe is the limit - a randomly initialised TinyExpert behind a frozen
   decoder has to learn a lot from 40 frames.

This run keeps the frozen recipe of exp2 and changes **only the targets**: the
exact per-point instance ids of SceneNN (`ins`) instead of the teacher pseudo
instances. The teacher stage still runs, so both runs share the same data and
the same teacher for comparison.

**Setup** (`settings.json`)

| item | value |
|---|---|
| shared layers | frozen |
| backbone | TinyExpert, random init, the only trainable part (153 of 256 tensors) |
| targets | exact SceneNN instance ids (frames 0-39) |
| frames | 0-39 train, 40-59 validate |
| budget | 110 minutes, AdamW, lr 1e-3 |

**Result** (`results/`)

| metric | value | exp2 (teacher targets) | teacher ESAM-E |
|---|---|---|---|
| steps | 56,229 | 57,814 | - |
| best validation loss | **0.956** | 1.074 | - |
| AP | **0.0129** | 0.0021 | 0.0319 |
| AP50 | **0.0290** | 0.0065 | 0.0545 |
| AP25 | **0.0623** | 0.0177 | 0.0729 |
| predictions above threshold | 100 | 100 | - |
| instance colours in the exported cloud | 86 (top 3 cover 56 %) | 94 (74 %) | - |

**What it shows.** The teacher was indeed the second bottleneck: with the exact
labels the same frozen recipe is six times better in AP (0.0021 -> 0.0129),
the loss drops again, and the student reaches 41 % of the teacher AP and 85 % of
its AP25. The remaining distance is the recipe itself: a randomly initialised
tiny encoder, 40 training frames and two hours, with no feature level
distillation.

**Run.** Kaggle kernel `abdesselamaimen01/esam-exp3-frozen-gt-targets`,
log in `results/run.log`, checkpoint in `outputs/checkpoints/exp3_frozen_gt_targets/`.
