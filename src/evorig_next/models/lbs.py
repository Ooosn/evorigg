from __future__ import annotations

import torch

from evorig_next.utils.geometry import assert_shape


def lbs_deform(
    rest_vertices: torch.Tensor,
    skinning_weights: torch.Tensor,
    bind_transforms: torch.Tensor,
    posed_joint_transforms: torch.Tensor,
) -> torch.Tensor:
    assert_shape(rest_vertices, (None, 3), "rest_vertices")
    assert_shape(skinning_weights, (rest_vertices.shape[0], posed_joint_transforms.shape[1]), "skinning_weights")
    assert_shape(bind_transforms, (posed_joint_transforms.shape[1], 4, 4), "bind_transforms")
    assert_shape(posed_joint_transforms, (None, None, 4, 4), "posed_joint_transforms")
    rest_h = torch.cat([rest_vertices, torch.ones(rest_vertices.shape[0], 1, dtype=rest_vertices.dtype, device=rest_vertices.device)], dim=-1)
    joint_mats = posed_joint_transforms @ bind_transforms.unsqueeze(0)
    transformed = torch.matmul(joint_mats[:, :, None], rest_h[None, None, :, :, None]).squeeze(-1)
    weighted = (skinning_weights.transpose(0, 1)[None, :, :, None] * transformed).sum(dim=1)
    return weighted[..., :3]

