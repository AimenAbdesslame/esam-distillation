# SceneNN distillation (ESAM-E teacher -> ESAM++ Tiny student)

Short experiment, not full ESAM++ training.

| file | role |
|---|---|
| `config.py` | every path and constant, overridable with `ESAM_*` env vars |
| `common.py` | rotation, colours, PLY/preview, class agnostic AP |
| `data.py` | builds the 80 SceneNN 322 frames (points, super points, instance labels) |
| `teacher.py` | runs ESAM-E online and stores the pseudo instances |
| `tiny_model.py` | `ScanNet200MixFormer3D_Online_Tiny`, the ESAM++ Tiny student |
| `student.py` | distils the teacher instances into the Tiny student and evaluates it |

## Data

SceneNN part 0 of `XXXCARREY/EmbodiedSAM` contains `oni/322/{point,ins,pose}`.
Each frame is a 20 000 point cloud (`point`), the matching instance ids
(`ins`) and the camera pose (`pose`).  The pose moves the points to world
coordinates, `R_x(+90)` turns SceneNN Y-up into the Z-up the models expect.

## Run order

```
python data.py        # 80 frames, ~1 min
python teacher.py     # ESAM-E pseudo instances, ~25 s
python student.py     # distillation, 110 min budget + evaluation
```

`teacher.py` writes `teacher_metrics.json` with the validation AP and a small
sanity gate; the notebook only starts the student when that gate passes.

## Experiment switches (environment variables)

| variable | default | meaning |
|---|---|---|
| `ESAM_FREEZE_SHARED` | `0` | `1` freezes every shared layer and trains only the TinyExpert backbone |
| `ESAM_TARGET` | `teacher` | `teacher` distils the ESAM-E pseudo instances, `gt` uses the exact SceneNN instance ids |
| `ESAM_STUDENT` | `$WORK/distill/student` | output folder |
| `ESAM_MAX_MINUTES` | `110` | training budget |
| `ESAM_LR` | `1e-3` | learning rate |

Both experiments share the same teacher, frames, split and budget, so the only
difference between them is the switch above.

## Notes

* Frames 0-39 train, frames 40-59 validate.
* The student starts from the ESAM-E weights that still fit its shape, the
  TinyExpert backbone itself is random.
* Loss: Hungarian match between student queries and teacher instances,
  mask BCE + Dice, objectness BCE.
* No upstream ESAM file is modified.
