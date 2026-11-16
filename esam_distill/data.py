"""Build the SceneNN 322 frames used by every experiment.

Each frame holds the 20 000 points of the scene in world Z-up coordinates, the
super point grouping and the exact per-point instance ids.
"""
import numpy as np
from sklearn.cluster import KMeans

import config as C
from common import write_json, zup

# ImageNet statistics used by the ESAM models for the colour channels.
RGB_MEAN = np.array([0.47793125906962, 0.4303257521323044, 0.3749598901421883])
RGB_STD = np.array([0.2294476613552524, 0.22136130327957347, 0.22412925035485958])


# ============================================================
# 1. FRAME LOADING
# ============================================================
def load_frame(frame_id):
    """Load one SceneNN frame: points, super points, instance ids, colours."""
    name = f"{frame_id:05d}.npy"
    scene = C.RAW_ROOT / "oni" / "322"
    point = np.load(scene / "point" / name)
    ins = np.load(scene / "ins" / name)
    pose = np.load(scene / "pose" / name)

    xyz = point[:, :3]
    xyz = (pose @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    xyz = zup(xyz)

    rgb = point[:, 3:6].astype(np.float32)
    norm_rgb = (rgb / 255.0 - RGB_MEAN) / RGB_STD
    super_points = KMeans(n_clusters=C.CLUSTERS, n_init=10,
                           random_state=0).fit_predict(xyz)
    return (np.c_[xyz, norm_rgb].astype(np.float32),
            super_points.astype(np.int64),
            ins.astype(np.int64),
            rgb.astype(np.uint8))


def frame_ids():
    """Frame ids of the study: the first NUM_FRAMES ids of the converted stream."""
    ids = [int(x) for x in (C.DATA_ROOT / "frame_ids.txt").read_text().split()]
    return ids[:C.NUM_FRAMES]


# ============================================================
# 2. MAIN
# ============================================================
def main():
    """Write one npz per frame plus a small meta file."""
    C.FRAMES.mkdir(parents=True, exist_ok=True)
    ids = frame_ids()
    for index, frame_id in enumerate(ids):
        points, super_points, ins, rgb = load_frame(frame_id)
        np.savez_compressed(C.FRAMES / f"frame_{index:04d}.npz",
                            points=points, sp_pts_mask=super_points,
                            gt_ins=ins, rgb_raw=rgb,
                            frame_id=np.array([frame_id]))
        if (index + 1) % 20 == 0:
            print(f"prepared {index + 1}/{len(ids)} frames")
    write_json(C.FRAMES / "meta.json",
               {"frames": len(ids), "frame_ids": ids,
                "points_per_frame": 20000, "clusters": C.CLUSTERS,
                "rotation": "R_x(+90) Y-up -> Z-up"})
    print("frames ready:", C.FRAMES)


if __name__ == "__main__":
    main()
