"""Run the ESAM-E online teacher over the prepared frames.

The teacher produces the pseudo instances that the student imitates, and the
reference AP that every experiment is measured against.
"""
import glob
import sys
import time

import numpy as np
import torch

import config as C
from common import (instance_ap, mask_to_colors, palette, save_preview,
                    voxel_downsample, write_json, write_ply)

sys.path.insert(0, str(C.ESAM_ROOT))


# ============================================================
# 1. TEACHER MODEL
# ============================================================
def build_teacher():
    """Build ESAM-E with the published CA online checkpoint."""
    import oneformer3d  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(C.TEACHER_CFG)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(C.ESAM_CKPT), map_location="cpu")
    model.cfg = cfg
    model.map_to_rec_pcd = False
    return model.to(C.DEVICE).eval()


def teacher_frame(model, frame):
    """Predict the instances of one frame with the online ESAM-E model."""
    from mmengine.dataset import pseudo_collate
    from mmdet3d.structures import Det3DDataSample, PointData

    points = torch.from_numpy(frame["points"]).float().to(C.DEVICE)
    sample = Det3DDataSample()
    gt_pts = PointData()
    gt_pts["sp_pts_mask"] = torch.from_numpy(frame["sp_pts_mask"]).long().to(C.DEVICE)
    sample.gt_pts_seg = gt_pts
    data = pseudo_collate([dict(inputs={"points": points}, data_samples=sample)])
    with torch.no_grad():
        result = model.test_step(data)
    pred = result[0].pred_pts_seg
    return (np.asarray(pred.pts_instance_mask[0], dtype=bool),
            np.asarray(pred.instance_scores, dtype=np.float32))


# ============================================================
# 2. MAIN
# ============================================================
def main():
    """Predict all frames, store the pseudo instances and write the metrics."""
    files = sorted(glob.glob(str(C.FRAMES / "frame_*.npz")))
    C.TEACHER.mkdir(parents=True, exist_ok=True)
    colors = palette()
    model = build_teacher()

    points_all, gt_all, counts = [], [], []
    frame_ap, stats = [], []
    total = 0
    start = time.time()

    for index, path in enumerate(files):
        data = np.load(path)
        points = data["points"].astype(np.float32)
        gt = data["gt_ins"].astype(np.int64)
        num_points = len(points)

        merged, scores = teacher_frame(model, {k: data[k] for k in data.files})
        current = merged[:, total:total + num_points]
        frame_ap.append(instance_ap(current, scores, gt))

        keep = [i for i in range(len(current))
                if current[i].sum() >= C.MIN_POINTS and scores[i] >= C.SCORE_THR]
        masks = current[keep] if keep else np.zeros((0, num_points), dtype=bool)
        mask_scores = scores[keep] if keep else np.zeros((0,), dtype=np.float32)
        np.savez_compressed(C.TEACHER / f"pseudo_{index:04d}.npz",
                            masks=masks, scores=mask_scores,
                            frame_id=data["frame_id"])

        points_all.append(points[:, :3])
        gt_all.append(gt)
        counts.append(len(masks))
        stats.append({"index": index, "frame_id": int(data["frame_id"][0]),
                      "instances": int(len(masks)),
                      "max_mask_points": int(masks.sum(axis=1).max()) if len(masks) else 0})
        total += num_points

        if index < 4:
            save_preview(C.TEACHER / f"teacher_preview_{index:04d}.png",
                         points[:, :3], mask_to_colors(masks, mask_scores, colors),
                         title=f"ESAM-E teacher frame {index} ({len(masks)} instances)")
        if (index + 1) % 20 == 0:
            print(f"teacher frame {index + 1}/{len(files)}")

    np.savez_compressed(C.TEACHER / "teacher_final.npz",
                        points=np.concatenate(points_all),
                        gt_ins=np.concatenate(gt_all),
                        instances=np.array(counts))

    val_pts = np.concatenate(points_all[C.VAL_START:])
    val_offset = sum(len(p) for p in points_all[:C.VAL_START])
    val_masks = merged[:, val_offset:]
    val_colors = mask_to_colors(val_masks, scores, colors)
    write_ply(C.TEACHER / "teacher_val.ply", val_pts, val_colors)
    down_xyz, down_colors = voxel_downsample(val_pts, val_colors, 0.02)
    write_ply(C.TEACHER / "teacher_val_voxel02.ply", down_xyz, down_colors)

    # ============================================================
    # 3. METRICS AND SANITY GATE
    # ============================================================
    def mean_ap(selection):
        chosen = [a for i, a in enumerate(frame_ap)
                  if selection(i) and a["num_gt"] > 0]
        if not chosen:
            return {"AP": 0.0, "AP50": 0.0, "AP25": 0.0, "frames": 0}
        return {"AP": float(np.mean([a["AP"] for a in chosen])),
                "AP50": float(np.mean([a["AP50"] for a in chosen])),
                "AP25": float(np.mean([a["AP25"] for a in chosen])),
                "frames": len(chosen)}

    ap_val = mean_ap(lambda i: i >= C.VAL_START)
    ap_all = mean_ap(lambda i: True)
    fractions = [s["max_mask_points"] / 20000.0 for s in stats if s["max_mask_points"]]
    gate = {"ok": bool(len(files) >= 20 and np.median(counts) >= 2
                       and 0.01 <= np.median(fractions) <= 0.95),
            "frames": len(files),
            "median_instances": float(np.median(counts)),
            "median_max_mask_fraction": float(np.median(fractions))}
    write_json(C.TEACHER / "teacher_metrics.json",
               {"frames": len(files), "elapsed_sec": time.time() - start,
                "teacher_ap_val": ap_val, "teacher_ap_all": ap_all,
                "gate": gate, "frame_stats": stats})
    print("teacher AP (val):", ap_val)
    print("gate:", gate)


if __name__ == "__main__":
    main()
