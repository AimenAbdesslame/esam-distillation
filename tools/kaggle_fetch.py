"""Check a Kaggle distillation run and pull its results.

usage:
    python kaggle_fetch.py esam-distill-v10
    python kaggle_fetch.py esam-distill-v10 --outputs results/exp2_frozen
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

WORK = Path(r"D:\ESAM_Work")


def status(slug):
    out = subprocess.run(["kaggle", "kernels", "status", slug],
                         capture_output=True, text=True,
                         env={**os.environ, "KAGGLE_ENABLE_OAUTH": "1"})
    return (out.stdout or out.stderr).strip()


def log(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    api = KaggleApi(enable_oauth=True)
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = slug.split("/")[0]
        request.kernel_slug = slug.split("/")[1]
        raw = client.kernels.kernels_api_client.list_kernel_session_output(request).log or ""
    try:
        raw = "".join(str(event.get("data", "")) for event in json.loads(raw))
    except Exception:
        pass
    return raw


def download(slug, target):
    target = WORK / target
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(target)],
                   env={**os.environ, "KAGGLE_ENABLE_OAUTH": "1"})
    for folder in ("mc", ".state", "tmp"):
        shutil.rmtree(target / folder, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--outputs", default="")
    args = parser.parse_args()
    print("status:", status(args.slug))
    try:
        text = log(args.slug)
    except Exception as exc:
        text = f"log not available yet: {exc}"
    (WORK / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (WORK / "results" / "logs" / f"{args.slug.split('/')[-1]}.log").write_text(text, encoding="utf-8")
    print("\n".join(text.splitlines()[-40:]))
    if args.outputs:
        download(args.slug, args.outputs)


if __name__ == "__main__":
    main()
