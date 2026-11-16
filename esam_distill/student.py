"""Train the ESAM++ Tiny student to imitate the ESAM-E teacher instances.

The experiment switches in `config.py` decide what is trained:

* `ESAM_FREEZE_SHARED=1` freezes every shared ESAM++ layer and trains only the
  TinyExpert backbone (the official ESAM++ recipe).
* `ESAM_FREEZE_SHARED=0` trains the whole model.
* `ESAM_TARGET=teacher` imitates the ESAM-E pseudo instances, `gt` uses the
  exact SceneNN instance ids.
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import config as C
from common import (instance_ap, mask_to_colors, palette, save_preview,
                    voxel_downsample, write_json, write_ply)

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (str(C.ESAM_ROOT), HERE, os.path.dirname(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# 1. DATA
# ============================================================
def load_frame(path):
    """Points, super points, instance ids and raw colours of one frame."""
    data = np.load(path)
    return (torch.from_numpy(data["points"]).float(),
            torch.from_numpy(data["sp_pts_mask"]).long(),
            data["gt_ins"].astype(np.int64),
            data["rgb_raw"].astype(np.uint8))


def load_pseudo(path):
    """Teacher instances of one frame: masks and scores."""
    data = np.load(path)
    return torch.from_numpy(data["masks"]).bool(), torch.from_numpy(data["scores"]).float()


def load_gt_targets(path, min_points=50):
    """Instance masks built from the ground truth ids of one frame."""
    ids = np.load(path)["gt_ins"].astype(np.int64)
    masks = [ids == gid for gid in np.unique(ids)
             if gid > 0 and (ids == gid).sum() >= min_points]
    if not masks:
        return torch.zeros((0, len(ids)), dtype=torch.bool), torch.zeros(0)
    return torch.from_numpy(np.stack(masks)), torch.ones(len(masks))


def build_cache(frame_paths, pseudo_paths):
    """Cache the frames and their targets in RAM."""
    cache = []
    for index, frame_path in enumerate(frame_paths):
        points, sp, _, _ = load_frame(frame_path)
        masks, scores = (load_gt_targets(frame_path) if C.TARGET == "gt"
                         else load_pseudo(pseudo_paths[index]))
        cache.append((points, sp, masks, scores))
    return cache


# ============================================================
# 2. MODEL
# ============================================================
def build_student():
    """Build the ESAM++ Tiny student from the ESAM++ config."""
    import oneformer3d  # noqa: F401
    import tiny_model  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(C.STUDENT_CFG)
    cfg.custom_imports = dict(imports=["oneformer3d", "tiny_model"])
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    return model.to(C.DEVICE)


def partial_init(model):
    """Copy the published ESAM-E weights that still fit the student."""
    checkpoint = torch.load(str(C.ESAM_CKPT), map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    source = {k.replace("module.", "", 1): v for k, v in source.items()}
    own = model.state_dict()
    matched = {k: v for k, v in source.items()
               if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    return {"matched": len(matched), "total": len(own)}


def set_trainable(model, freeze_shared):
    """Freeze the shared layers, train only the TinyExpert backbone."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not freeze_shared) or name.startswith("backbone.")
    return [p for p in model.parameters() if p.requires_grad]


def set_train_mode(model):
    """Backbone in train mode, the frozen shared layers stay in eval mode."""
    model.train()
    if C.FREEZE_SHARED:
        for name, module in model.named_modules():
            if not name.startswith("backbone"):
                module.eval()


def student_forward(model, points, sp):
    """Student queries: mask logits and objectness logits."""
    from mmdet3d.structures import Det3DDataSample, PointData

    sample = Det3DDataSample()
    gt_pts = PointData()
    gt_pts["sp_pts_mask"] = sp
    sample.gt_pts_seg = gt_pts
    x, point_features, all_xyz, sp_xyz, _, _ = model.extract_feat(
        {"points": [points]}, [sample])
    out = model.decoder(x, point_features, x, ([sp], all_xyz))
    masks, cls = out["masks"][0], out["cls_preds"][0]
    if isinstance(masks, (list, tuple)):
        masks = masks[-1]
    if isinstance(cls, (list, tuple)):
        cls = cls[-1]
    return masks, cls


# ============================================================
# 3. LOSS
# ============================================================
def distill_loss(logits, cls, target_masks, target_scores):
    """Hungarian matching between student queries and target instances."""
    if len(target_masks) == 0:
        return F.binary_cross_entropy_with_logits(
            cls[:, 1], torch.zeros_like(cls[:, 1])) * 0.1
    prob = logits.sigmoid().detach()
    target = target_masks.float()
    inter = prob @ target.T
    union = prob.sum(1)[:, None] + target.sum(1)[None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    rows, cols = linear_sum_assignment((1.0 - iou).detach().cpu().numpy())
    rows = torch.as_tensor(rows, device=logits.device)
    cols = torch.as_tensor(cols, device=logits.device)

    chosen, wanted = logits[rows], target[cols]
    weight = target_scores[cols].clamp(0.05, 1.0)
    bce = F.binary_cross_entropy_with_logits(
        chosen, wanted, reduction="none").mean(dim=1)
    bce = (bce * weight).sum() / weight.sum().clamp_min(1e-6)
    prob_chosen = chosen.sigmoid()
    dice = (2 * (prob_chosen * wanted).sum(dim=1) + 1) / (
        prob_chosen.sum(dim=1) + wanted.sum(dim=1) + 1)
    dice = (1 - dice).mean()
    objectness = torch.zeros_like(cls[:, 1])
    objectness[rows] = 1.0
    cls_loss = F.binary_cross_entropy_with_logits(
        cls[:, 1], objectness, pos_weight=torch.tensor(2.0, device=cls.device))
    return bce + dice + 0.5 * cls_loss


# ============================================================
# 4. EVALUATION
# ============================================================
def evaluate(model, frames):
    """Class agnostic AP of the student, averaged over the given frames."""
    from mmengine.dataset import pseudo_collate
    from mmdet3d.structures import Det3DDataSample, PointData

    if hasattr(model, "online_merger"):
        delattr(model, "online_merger")
    model.eval()
    prev = 0
    frame_ap = []
    xyz_parts, rgb_parts, gt_parts = [], [], []
    merged = np.zeros((0, 0), dtype=bool)
    scores = np.zeros((0,), dtype=np.float32)
    with torch.no_grad():
        for path in frames:
            points, sp, gt, rgb = load_frame(path)
            num_points = len(points)
            sample = Det3DDataSample()
            gt_pts = PointData()
            gt_pts["sp_pts_mask"] = sp.to(C.DEVICE)
            sample.gt_pts_seg = gt_pts
            data = pseudo_collate([dict(
                inputs={"points": points.to(C.DEVICE)}, data_samples=sample)])
            result = model.test_step(data)
            pred = result[0].pred_pts_seg
            merged = np.asarray(pred.pts_instance_mask[0], dtype=bool)
            scores = np.asarray(pred.instance_scores, dtype=np.float32)
            frame_ap.append(instance_ap(merged[:, prev:prev + num_points], scores, gt))
            xyz_parts.append(points[:, :3].numpy())
            rgb_parts.append(rgb)
            gt_parts.append(gt)
            prev += num_points

    chosen = [a for a in frame_ap if a["num_gt"] > 0]
    ap = {"AP": float(np.mean([a["AP"] for a in chosen])),
          "AP50": float(np.mean([a["AP50"] for a in chosen])),
          "AP25": float(np.mean([a["AP25"] for a in chosen])),
          "num_pred": int(len(scores)),
          "num_gt": int(sum(a["num_gt"] for a in chosen)),
          "frames": len(chosen)}
    keep = [i for i in range(len(merged)) if merged[i].any()]
    masks = merged[keep] if keep else np.zeros((0, prev), dtype=bool)
    keep_scores = scores[keep] if keep else scores
    gt = np.concatenate(gt_parts)
    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    return ap, masks, keep_scores, xyz, rgb, gt


# ============================================================
# 5. TRAINING
# ============================================================
def train(model, train_cache, val_cache, minutes):
    """Train the student for the given budget and return the best checkpoint."""
    trainable = set_trainable(model, C.FREEZE_SHARED)
    optimizer = torch.optim.AdamW(trainable, lr=C.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(200, int(minutes * 12)), eta_min=C.LR * 0.05)
    best_path = C.STUDENT / "student_best.pth"
    last_path = C.STUDENT / "student_last.pth"
    history, best_val, step = [], float("inf"), 0
    start = time.time()

    def save(path):
        torch.save({"state_dict": model.state_dict(), "step": step}, path)

    def val_loss():
        model.eval()
        values = []
        with torch.no_grad():
            for points, sp, masks, scores in val_cache[:5]:
                logits, cls = student_forward(model, points.to(C.DEVICE), sp.to(C.DEVICE))
                values.append(float(distill_loss(logits, cls,
                                                 masks.to(C.DEVICE),
                                                 scores.to(C.DEVICE)).item()))
        set_train_mode(model)
        return float(np.mean(values)) if values else 0.0

    set_train_mode(model)
    while time.time() - start < minutes * 60:
        points, sp, masks, scores = train_cache[step % len(train_cache)]
        logits, cls = student_forward(model, points.to(C.DEVICE), sp.to(C.DEVICE))
        loss = distill_loss(logits, cls, masks.to(C.DEVICE), scores.to(C.DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 20 == 0:
            history.append({"step": step, "loss": float(loss.item())})
            print(f"step {step} loss {float(loss.item()):.4f} "
                  f"{(time.time() - start) / 60:.1f}m")
        if step % 200 == 0:
            value = val_loss()
            history.append({"step": step, "val_loss": value})
            print(f"val step {step} loss {value:.4f}")
            if value < best_val:
                best_val = value
                save(best_path)
        if step % 100 == 0:
            save(last_path)

    save(last_path)
    if not best_path.exists():
        save(best_path)
    return best_path, {"steps": step, "train_minutes": (time.time() - start) / 60,
                        "best_val_loss": best_val, "history": history,
                        "trainable_tensors": len(trainable)}


# ============================================================
# 6. MAIN
# ============================================================
def main():
    """Run one experiment: prepare, train, evaluate and export."""
    frame_files = sorted(glob.glob(str(C.FRAMES / "frame_*.npz")))
    pseudo_files = sorted(glob.glob(str(C.TEACHER / "pseudo_*.npz")))
    train_frames = frame_files[:C.VAL_START]
    val_frames = frame_files[C.VAL_START:]

    C.STUDENT.mkdir(parents=True, exist_ok=True)
    model = build_student()
    init_info = partial_init(model)
    train_cache = build_cache(train_frames, pseudo_files[:C.VAL_START])
    val_cache = build_cache(val_frames, pseudo_files[C.VAL_START:])

    points, sp, masks, scores = train_cache[0]
    logits, cls = student_forward(model, points.to(C.DEVICE), sp.to(C.DEVICE))
    print("student forward:", tuple(logits.shape), tuple(cls.shape))

    best_path, summary = train(model, train_cache, val_cache, C.MAX_MINUTES)
    model.load_state_dict(torch.load(str(best_path), map_location="cpu")["state_dict"])

    ap, masks, scores, xyz, rgb, gt = evaluate(model, val_frames)
    colors = mask_to_colors(masks, scores, palette())
    write_ply(C.STUDENT / "student_val.ply", xyz, colors)
    down_xyz, down_colors = voxel_downsample(xyz, colors, 0.02)
    write_ply(C.STUDENT / "student_val_voxel02.ply", down_xyz, down_colors)
    save_preview(C.STUDENT / "student_preview.png", xyz, colors, title=f"Tiny student {ap}")
    write_json(C.STUDENT / "student_ap.json", ap)
    write_json(C.STUDENT / "distill_metrics.json",
               {**summary, "student_ap": ap, "init_info": init_info,
                "frozen_shared": C.FREEZE_SHARED, "target": C.TARGET,
                "train_frames": len(train_frames), "val_frames": len(val_frames)})
    print("student AP (val):", ap)


if __name__ == "__main__":
    main()
