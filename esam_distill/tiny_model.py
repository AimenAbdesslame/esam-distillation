"""ESAM++ Tiny student model.

ESAM++ keeps every layer of ESAM except the 3D encoder: the heavy sparse
U-Net is replaced by the TinyExpert, a small hierarchical sparse feature
pyramid.  This module registers the student model without touching any
upstream ESAM file.
"""
import os
import sys

import MinkowskiEngine as ME
import torch
from mmdet3d.registry import MODELS
from oneformer3d.mixformer3d import ScanNet200MixFormer3D_Stream

# The notebook copies google/esamplusplus/src to tools/esampp_src.
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
for path in (TOOLS, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from esampp_src.backbone import TinyExpert  # noqa: F401,E402  (registers the backbone)


# ============================================================
# 1. STUDENT MODEL
# ============================================================
@MODELS.register_module()
class ScanNet200MixFormer3D_Online_Tiny(ScanNet200MixFormer3D_Stream):
    """Single frame online variant whose backbone takes no `memory` kwarg."""

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Replace the heavy sparse U-Net by the TinyExpert, keep the rest."""
        coordinates, features = [], []
        for points in batch_inputs_dict["points"]:
            coordinates.append(points[:, :3])
            features.append(points[:, 3:])
        all_xyz = coordinates

        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        x = self.backbone(field.sparse())          # TinyExpert has no memory
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1)
                          for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x, all_xyz_w = self.pool(x, torch.cat(sp_pts_masks), all_xyz, with_xyz=True)

        features, sp_xyz_list = [], []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin:end, :-3])
            sp_xyz_list.append(x[begin:end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list, None, None
