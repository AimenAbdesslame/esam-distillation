"""
Run data_exploration.ipynb locally (outside Kaggle).
- Streams SceneNN 322 data directly from HuggingFace via tarfile (no full-file scan)
- Executes all notebook cells via nbconvert
- Writes outputs back into data_exploration.ipynb (ready to push to GitHub)
"""
import os, sys, subprocess, pathlib, urllib.request, tarfile, json, copy

# ── paths ────────────────────────────────────────────────────────────────────
NB_DIR  = pathlib.Path(__file__).parent
NB_PATH = NB_DIR / "data_exploration.ipynb"
TMP     = NB_DIR / "tmp" / "scenenn"
SCENE   = TMP / "oni" / "322"

# ── 1. download + stream-extract SceneNN 322 ─────────────────────────────────
PREFIXES = ("oni/322/point/", "oni/322/ins/", "oni/322/pose/")

if not (SCENE / "point").exists():
    print("Streaming SceneNN 322 from HuggingFace …")
    TMP.mkdir(parents=True, exist_ok=True)
    URL = "https://huggingface.co/XXXCARREY/EmbodiedSAM/resolve/main/SceneNN/SceneNN.tar.0?download=true"

    with urllib.request.urlopen(URL) as resp:
        # Stream into tarfile — 'r|' means read as a stream (no seeking)
        with tarfile.open(fileobj=resp, mode="r|") as tf:
            extracted = 0
            for member in tf:
                if any(member.name.startswith(p) for p in PREFIXES):
                    tf.extract(member, path=TMP)
                    extracted += 1
                    if extracted % 50 == 0:
                        print(f"  extracted {extracted} files …")
                # No early-stop: let stream finish naturally (tar ends itself)
    print(f"Done. Extracted {extracted} files to {TMP}")
else:
    print("Data already present — skipping download.")

# verify
npy_count = len(list(TMP.rglob("*.npy")))
print(f"Total .npy files on disk: {npy_count}")
if npy_count < 200:
    print("WARNING: fewer files than expected — extraction may be incomplete.")
    sys.exit(1)

# ── 2. patch the notebook: replace /kaggle/working with local path ────────────
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
local_scene = str(TMP).replace("\\", "/")

patched_nb = copy.deepcopy(nb)
for cell in patched_nb["cells"]:
    if cell["cell_type"] == "code":
        joined = "".join(cell["source"])
        # skip the bash download cell
        if "%%bash" in joined or "curl" in joined:
            cell["source"] = ["print('Data already downloaded by run_local.py')"]
            continue
        new_src = []
        for line in cell["source"]:
            line = line.replace("/kaggle/working/tmp/scenenn", local_scene)
            line = line.replace("/kaggle/working", str(NB_DIR).replace("\\", "/"))
            new_src.append(line)
        cell["source"] = new_src

PATCHED = NB_DIR / "data_exploration_local.ipynb"
PATCHED.write_text(json.dumps(patched_nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched notebook → {PATCHED}")

# ── 3. execute with nbconvert ─────────────────────────────────────────────────
print("Executing notebook …")
out_path = NB_DIR / "data_exploration_executed.ipynb"
result = subprocess.run(
    [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=300",
        "--output", str(out_path),
        str(PATCHED),
    ],
    cwd=str(NB_DIR),
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
    sys.exit(result.returncode)
print(f"Executed notebook written → {out_path}")

# ── 4. copy cell outputs back into the tracked notebook ──────────────────────
executed = json.loads(out_path.read_text(encoding="utf-8"))
original = json.loads(NB_PATH.read_text(encoding="utf-8"))

exec_code = [c for c in executed["cells"] if c["cell_type"] == "code"]
orig_code = [c for c in original["cells"] if c["cell_type"] == "code"]

for oc, ec in zip(orig_code, exec_code):
    oc["outputs"]         = ec.get("outputs", [])
    oc["execution_count"] = ec.get("execution_count")

NB_PATH.write_text(json.dumps(original, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\n✅ Outputs written back to {NB_PATH}")
print("   Ready to: git add notebooks/data_exploration/data_exploration.ipynb && git commit && git push")
