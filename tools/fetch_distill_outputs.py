"""Download the result files of a distillation run from Kaggle.

usage:
    python fetch_distill_outputs.py <slug> <results folder>
    python fetch_distill_outputs.py <slug> --checkpoints outputs/checkpoints/<name>
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

RESULTS = {"student_preview.png", "student_ap.json", "distill_metrics.json",
           "teacher_metrics.json", "student_val.ply", "student_val_voxel02.ply",
           "teacher_preview_0000.png", "teacher_val.ply", "teacher_val_voxel02.ply"}
CHECKPOINTS = {"student_best.pth"}


def list_files(slug):
    """Map every output file of the kernel to its download url."""
    api = KaggleApi(enable_oauth=True)
    api.authenticate()
    files, token = {}, None
    with api.build_kaggle_client() as client:
        for _ in range(40):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = slug.split("/")[0]
            request.kernel_slug = slug.split("/")[1]
            if token:
                request.page_token = token
            try:
                page = client.kernels.kernels_api_client.list_kernel_session_output(request)
            except Exception as exc:                 # Kaggle rate limits the listing
                print("retry after error:", exc)
                time.sleep(60)
                continue
            for entry in page.files:
                files[entry.file_name] = entry.to_dict()
            token = page.next_page_token
            if not token:
                break
            time.sleep(2)
    return files


def download(files, wanted, folder):
    """Download the wanted files into the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, meta in files.items():
        if name.split("/")[-1] not in wanted:
            continue
        target = folder / name.split("/")[-1]
        if not target.exists():
            with urllib.request.urlopen(meta["url"]) as response:
                target.write_bytes(response.read())
        print("saved", target, target.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--checkpoints", default="")
    args = parser.parse_args()
    files = list_files(args.slug)
    if args.checkpoints:
        download(files, CHECKPOINTS, Path(args.checkpoints))
    if args.folder:
        download(files, RESULTS, Path(args.folder))


if __name__ == "__main__":
    main()
