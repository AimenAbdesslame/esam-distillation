# Experiment 1 - all layers trained

**Question.** What happens when the published ESAM-E weights are copied into the
ESAM++ student and *every* layer is then trained on the ESAM-E pseudo instances?

**Why this experiment.** It is the straightforward way to run a distillation:
clone what can be cloned, train the whole network. It gives the baseline against
which the frozen variant is compared, and it exposes the failure mode of a
pretrained decoder being pulled around by a randomly initialised backbone.

**Setup** (`settings.json`)

| item | value |
|---|---|
| shared layers | trained (not frozen) |
| backbone | TinyExpert, random init |
| initialisation | published ESAM-E weights, 103 of 400 tensors copied |
| targets | ESAM-E online pseudo instances |
| frames | 0-39 train, 40-59 validate (80 frames, 20 000 points each) |
| budget | 110 minutes, AdamW, lr 1e-3 |

**Result** (`results/`)

| metric | value |
|---|---|
| steps | 50,139 |
| best validation loss | 1.606 (flat between 1.60 and 1.65) |
| AP / AP50 / AP25 on frames 40-59 | 0.000 / 0.000 / 0.000 |
| predictions above threshold | 0 |
| exported cloud | 800,000 points, 1 colour (background only) |

**What it shows.** The loss never moved and the objectness head never fired, so
the student emits no instance at all. Training the shared layers together with
a random backbone destroys the pretrained decoder faster than the new backbone
can rebuild a compatible representation, at least on 40 frames and two hours.

**Reference.** The same teacher on the same frames scores AP 0.032 / AP50 0.055 /
AP25 0.073, so the targets themselves are weak but not empty.

**Run.** Kaggle kernel `abdesselamaimen01/esam-distill-v9`, log in `results/run.log`,
checkpoint in `outputs/checkpoints/exp1_all_layers/`.
