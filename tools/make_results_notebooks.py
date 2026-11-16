"""Build one results notebook per experiment.

usage: python tools/make_results_notebooks.py

Writes notebooks/<experiment>/results.ipynb for every folder of experiments/.
The notebooks are local (no GPU): they read the result files of the experiment
and summarise them.
"""
import json
import os

ROOT = r"D:\ESAM_Work"
EXPERIMENTS = os.path.join(ROOT, "experiments")
NOTEBOOKS = os.path.join(ROOT, "notebooks")

CELLS = [
    ("markdown", """# {name}

{why}

Everything below is read from `experiments/{experiment}/results/`.
"""),
    ("code", """# ============================================================
# 1. METRICS
# ============================================================
import json
from pathlib import Path

RESULTS = Path(r'{root}') / 'experiments' / '{experiment}' / 'results'

student = json.loads((RESULTS / 'student_ap.json').read_text())
distill = json.loads((RESULTS / 'distill_metrics.json').read_text())
teacher = json.loads((RESULTS / 'teacher_metrics.json').read_text())

print('teacher (ESAM-E) on the validation frames:', teacher['teacher_ap_val'])
print('student                              :', student)
print('best validation loss                 :', round(distill['best_val_loss'], 4))
print('steps / minutes                      :', distill['steps'], round(distill['train_minutes'], 1))
print('trainable tensors                    :', distill.get('trainable_tensors'))
print('frozen shared layers                 :', distill.get('frozen_shared'))
print('targets                              :', distill.get('target'))
"""),
    ("code", """# ============================================================
# 2. LOSS HISTORY
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

history = distill.get('history', [])
losses = [h['loss'] for h in history if 'loss' in h]
validations = [(h['step'], h['val_loss']) for h in history if 'val_loss' in h]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses, label='training loss')
ax.set_xlabel('logged step')
ax.set_ylabel('loss')
if validations:
    ax2 = ax.twinx()
    ax2.plot([s for s, _ in validations], [v for _, v in validations],
             color='tab:orange', label='validation loss')
    ax2.set_ylabel('validation loss')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'loss_history.png', dpi=120)
plt.show()
"""),
    ("code", """# ============================================================
# 3. PREDICTED INSTANCES
# ============================================================
from IPython.display import Image, display

for name in ('teacher_preview_frame000.png', 'student_preview.png'):
    path = RESULTS / name
    if path.exists():
        print(name)
        display(Image(filename=str(path)))
"""),
    ("code", """# ============================================================
# 4. EXPORTED POINT CLOUD
# ============================================================
import numpy as np

path = RESULTS / 'student_val.ply'
raw = path.read_bytes()
end = raw.index(b'end_header\\n') + len(b'end_header\\n')
vertices = int([l for l in raw[:end].decode().splitlines()
                 if l.startswith('element vertex')][0].split()[-1])
record = np.frombuffer(raw[end:], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                         ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
colors = np.stack([record['r'], record['g'], record['b']], axis=1)
unique, counts = np.unique(colors, axis=0, return_counts=True)

print('vertices        :', vertices)
print('instance colours:', len(unique))
print('largest 3 colours cover %.1f %% of the points' %
      (100 * np.sort(counts)[::-1][:3].sum() / vertices))
print('bbox            : x', record['x'].min().round(2), record['x'].max().round(2),
      '| y', record['y'].min().round(2), record['y'].max().round(2),
      '| z', record['z'].min().round(2), record['z'].max().round(2))
"""),
]


def build(experiment):
    folder = os.path.join(EXPERIMENTS, experiment)
    settings = json.loads(open(os.path.join(folder, "settings.json"), encoding="utf-8").read())
    cells = []
    for kind, source in CELLS:
        text = (source.replace("{name}", settings.get("name", settings["title"]))
                    .replace("{why}", settings["why"])
                    .replace("{experiment}", experiment)
                    .replace("{root}", ROOT))
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": text.splitlines(True)})
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.8"}},
                "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(NOTEBOOKS, experiment)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    for name in sorted(os.listdir(EXPERIMENTS)):
        if os.path.isfile(os.path.join(EXPERIMENTS, name, "settings.json")):
            build(name)
