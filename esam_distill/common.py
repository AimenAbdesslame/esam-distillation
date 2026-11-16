"""Geometry, colour, PLY and evaluation helpers shared by the experiment scripts."""
import json
import os

import numpy as np

# ============================================================
# 1. GEOMETRY
# ============================================================
# SceneNN is Y-up, the models expect Z-up: R_x(+90) maps (x, y, z) -> (x, -z, y).
R_ZUP = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])


def zup(points):
    """Rotate SceneNN world points into the Z-up frame of the models."""
    return (np.asarray(points, dtype=np.float64) @ R_ZUP.T).astype(np.float32)


# ============================================================
# 2. COLOURS
# ============================================================
def palette(seed=0, n=256):
    """Deterministic instance colours."""
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    colors[-1] = (190, 190, 190)
    return colors


def mask_to_colors(masks, scores=None, colors=None):
    """One stable colour per mask row.  Row order is preserved."""
    masks = np.asarray(masks, dtype=bool)
    colors = palette() if colors is None else colors
    index = np.full(masks.shape[1], -1, dtype=np.int64)
    order = np.argsort(np.asarray(scores)) if scores is not None else range(len(masks))
    for row in order:
        index[masks[row]] = row
    return colors[index]


# ============================================================
# 3. PLY AND POINT CLOUD IO
# ============================================================
def write_ply(path, xyz, rgb):
    """Write a binary little-endian PLY."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    record = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    record["x"], record["y"], record["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    record["red"], record["green"], record["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        record.tofile(f)


def voxel_downsample(xyz, rgb, voxel=0.02):
    """Keep one point per occupied voxel."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    keys = np.clip(keys, 0, (1 << 21) - 1)
    packed = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    keep = np.sort(np.unique(packed, return_index=True)[1])
    return xyz[keep], rgb[keep]


def save_preview(path, xyz, rgb, title=""):
    """Save a two-view matplotlib preview of a point cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if len(xyz) > 90000:
        sel = np.random.default_rng(0).choice(len(xyz), 90000, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=140)
    for ax, (name, a, b) in zip(axes, [("X / Z", 0, 2), ("X / Y", 0, 1)]):
        ax.scatter(xyz[:, a], xyz[:, b], c=rgb / 255.0, s=0.25, linewidths=0)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. EVALUATION
# ============================================================
def iou_matrix(pred, gt):
    """IoU between predicted masks [P, N] and ground truth masks [G, N]."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    inter = pred @ gt.T
    union = pred.sum(1)[:, None] + gt.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)


def average_precision(pred_masks, pred_scores, gt_masks, threshold=0.5):
    """Class agnostic AP at one IoU threshold.  gt_masks: [G, N] bool."""
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0
    ious = iou_matrix(pred_masks, gt_masks)
    order = np.argsort(-np.asarray(pred_scores))
    used = np.zeros(len(gt_masks), dtype=bool)
    tp = np.zeros(len(order), dtype=np.float32)
    for rank, row in enumerate(order):
        best = int(np.argmax(ious[row]))
        if ious[row, best] >= threshold and not used[best]:
            used[best] = True
            tp[rank] = 1.0
    if tp.sum() == 0:
        return 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1.0 - tp)
    recall = tp_c / len(gt_masks)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-6)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def instance_ap(pred_masks, pred_scores, gt_ins, min_gt_points=50):
    """AP averaged over IoU 0.50:0.95, plus AP50 and AP25, for one frame."""
    pred_masks = np.asarray(pred_masks, dtype=bool)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    gt_ins = np.asarray(gt_ins, dtype=np.int64)
    if len(pred_masks) > 300:
        top = np.argsort(-pred_scores)[:300]
        pred_masks, pred_scores = pred_masks[top], pred_scores[top]
    gt_ids = [x for x in np.unique(gt_ins) if x > 0]
    gt_masks = [gt_ins == gid for gid in gt_ids
                if (gt_ins == gid).sum() >= min_gt_points]
    if not gt_masks:
        return {"AP": 0.0, "AP50": 0.0, "AP25": 0.0,
                "num_pred": int(len(pred_masks)), "num_gt": 0}
    gt_masks = np.stack(gt_masks)
    thresholds = list(np.arange(0.50, 0.96, 0.05))
    values = [average_precision(pred_masks, pred_scores, gt_masks, t) for t in thresholds]
    return {"AP": float(np.mean(values)),
            "AP50": float(average_precision(pred_masks, pred_scores, gt_masks, 0.50)),
            "AP25": float(average_precision(pred_masks, pred_scores, gt_masks, 0.25)),
            "num_pred": int(len(pred_masks)), "num_gt": int(len(gt_masks))}


# ============================================================
# 5. JSON IO
# ============================================================
def write_json(path, data):
    """Write a JSON report file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)
