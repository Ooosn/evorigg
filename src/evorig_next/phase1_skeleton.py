from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from evorig_next.utils.geometry import assert_shape
from evorig_next.phase1_rotations import axis_angle_to_matrix_stable, compute_bone_frames, invert_transform, make_transform


class Phase1Skeleton(nn.Module):
    def __init__(
        self,
        parent_idx: torch.Tensor,
        rest_joints: torch.Tensor,
        frame_count: int,
        init_pose: torch.Tensor | None = None,
        birth_steps: Iterable[int] | None = None,
        inserted: Iterable[bool] | None = None,
        birth_modes: Iterable[str] | None = None,
        connected_to_parent: Iterable[bool] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert_shape(parent_idx, (rest_joints.shape[0],), "parent_idx")
        assert_shape(rest_joints, (None, 3), "rest_joints")
        self.register_buffer("parent_idx", parent_idx.long().clone())
        if connected_to_parent is None:
            connected = parent_idx >= 0
        else:
            connected = torch.as_tensor(
                list(connected_to_parent) if not isinstance(connected_to_parent, torch.Tensor) else connected_to_parent,
                dtype=torch.bool,
                device=parent_idx.device,
            ).reshape(-1)
            assert_shape(connected, (rest_joints.shape[0],), "connected_to_parent")
            connected = connected & (parent_idx >= 0)
        self.register_buffer("connected_to_parent", connected.bool().clone())
        self.register_buffer("init_rest_joints", rest_joints.clone())
        self.rest_joints = nn.Parameter(rest_joints.clone())
        joint_count = int(rest_joints.shape[0])
        if init_pose is None:
            pose_rot = torch.zeros(frame_count, joint_count, 3, dtype=rest_joints.dtype, device=rest_joints.device)
        else:
            assert_shape(init_pose, (frame_count, joint_count, 3), "init_pose")
            pose_rot = init_pose.clone()
        self.pose_rot = nn.Parameter(pose_rot)
        self.root_trans = nn.Parameter(torch.zeros(frame_count, 3, dtype=rest_joints.dtype, device=rest_joints.device))
        self.birth_steps = list(birth_steps) if birth_steps is not None else [0] * joint_count
        self.is_inserted = list(inserted) if inserted is not None else [False] * joint_count
        self.birth_modes = list(birth_modes) if birth_modes is not None else ["seed"] * joint_count
        self._refresh_bones()

    @property
    def joint_count(self) -> int:
        return int(self.rest_joints.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.pose_rot.shape[0])

    @property
    def bone_count(self) -> int:
        return int(self.bone_child_idx.shape[0])

    def _replace_parameter(self, name: str, value: torch.Tensor) -> None:
        setattr(self, name, nn.Parameter(value))

    def _refresh_bones(self) -> None:
        if not hasattr(self, "connected_to_parent"):
            self.register_buffer("connected_to_parent", (self.parent_idx >= 0).bool().clone())
        child_idx = torch.nonzero((self.parent_idx >= 0) & self.connected_to_parent.bool(), as_tuple=False).flatten()
        if hasattr(self, "bone_child_idx"):
            self.bone_child_idx = child_idx.long()
            self.bone_parent_idx = self.parent_idx[child_idx].long()
        else:
            self.register_buffer("bone_child_idx", child_idx.long())
            self.register_buffer("bone_parent_idx", self.parent_idx[child_idx].long())

    def support_parent_idx(self) -> torch.Tensor:
        support_parent = torch.full_like(self.parent_idx, -1)
        mask = (self.parent_idx >= 0) & self.connected_to_parent.bool()
        support_parent[mask] = self.parent_idx[mask]
        return support_parent

    def compute_local_transforms(
        self,
        pose_rot: torch.Tensor | None = None,
        root_trans: torch.Tensor | None = None,
        frame_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pose_rot_full = self.pose_rot if pose_rot is None else pose_rot
        root_trans_full = self.root_trans if root_trans is None else root_trans
        assert_shape(pose_rot_full, (None, self.joint_count, 3), "pose_rot")
        assert_shape(root_trans_full, (pose_rot_full.shape[0], 3), "root_trans")
        if frame_idx is not None:
            frame_idx = frame_idx.to(device=pose_rot_full.device, dtype=torch.long).reshape(-1)
            pose_rot_sel = pose_rot_full.index_select(0, frame_idx)
            root_trans_sel = root_trans_full.index_select(0, frame_idx)
            frame_count = int(frame_idx.shape[0])
        else:
            pose_rot_sel = pose_rot_full
            root_trans_sel = root_trans_full
            frame_count = int(pose_rot_full.shape[0])
        rotations = axis_angle_to_matrix_stable(pose_rot_sel.reshape(-1, 3)).reshape(frame_count, self.joint_count, 3, 3)
        offsets = torch.zeros(self.joint_count, 3, dtype=self.rest_joints.dtype, device=self.rest_joints.device)
        root_idx = torch.nonzero(self.parent_idx == -1, as_tuple=False).flatten()[0]
        offsets[root_idx] = self.rest_joints[root_idx]
        non_root = torch.nonzero(self.parent_idx >= 0, as_tuple=False).flatten()
        offsets[non_root] = self.rest_joints[non_root] - self.rest_joints[self.parent_idx[non_root]]
        translations = offsets.unsqueeze(0).expand(frame_count, -1, -1).clone()
        translations[:, root_idx] = self.rest_joints[root_idx] + root_trans_sel
        return make_transform(rotations, translations)

    def forward_kinematics(
        self,
        pose_rot: torch.Tensor | None = None,
        root_trans: torch.Tensor | None = None,
        frame_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local = self.compute_local_transforms(pose_rot=pose_rot, root_trans=root_trans, frame_idx=frame_idx)
        global_parts: list[torch.Tensor | None] = [None] * self.joint_count

        def compute_joint(joint_idx: int) -> torch.Tensor:
            cached = global_parts[joint_idx]
            if cached is not None:
                return cached
            parent = int(self.parent_idx[joint_idx].item())
            if parent < 0:
                value = local[:, joint_idx]
            else:
                value = compute_joint(parent) @ local[:, joint_idx]
            global_parts[joint_idx] = value
            return value

        for joint_idx in range(self.joint_count):
            compute_joint(joint_idx)
        return torch.stack([part for part in global_parts if part is not None], dim=1)

    def compute_bind_transforms(self) -> torch.Tensor:
        rest_pose = torch.zeros(1, self.joint_count, 3, dtype=self.pose_rot.dtype, device=self.pose_rot.device)
        rest_root = torch.zeros(1, 3, dtype=self.root_trans.dtype, device=self.root_trans.device)
        rest_global = self.forward_kinematics(pose_rot=rest_pose, root_trans=rest_root)[0]
        return invert_transform(rest_global)

    def compute_bone_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_pos, frame = compute_bone_frames(self.rest_joints, self.bone_parent_idx, self.bone_child_idx)
        return parent_pos, frame, self.bone_parent_idx, self.bone_child_idx

    def insert_joint(
        self,
        parent_joint: int,
        rest_position: torch.Tensor,
        pose_init: torch.Tensor | None = None,
        birth_step: int = 0,
        birth_mode: str = "branch",
        connected_to_parent: bool = True,
    ) -> int:
        parent_joint = int(parent_joint)
        if parent_joint < 0 or parent_joint >= self.joint_count:
            raise IndexError(f"parent_joint out of range: {parent_joint}")
        rest_position = rest_position.detach().to(self.rest_joints.device, self.rest_joints.dtype).reshape(1, 3)
        self.parent_idx = torch.cat(
            [
                self.parent_idx,
                torch.tensor([parent_joint], dtype=torch.long, device=self.parent_idx.device),
            ],
            dim=0,
        )
        self.connected_to_parent = torch.cat(
            [
                self.connected_to_parent,
                torch.tensor([bool(connected_to_parent)], dtype=torch.bool, device=self.connected_to_parent.device),
            ],
            dim=0,
        )
        self.init_rest_joints = torch.cat([self.init_rest_joints, rest_position.clone()], dim=0)
        self._replace_parameter("rest_joints", torch.cat([self.rest_joints.detach(), rest_position], dim=0))
        if pose_init is None:
            pose_init = torch.zeros(self.frame_count, 1, 3, dtype=self.pose_rot.dtype, device=self.pose_rot.device)
        else:
            assert_shape(pose_init, (self.frame_count, 3), "pose_init")
            pose_init = pose_init.detach().to(self.pose_rot.device, self.pose_rot.dtype).unsqueeze(1)
        self._replace_parameter("pose_rot", torch.cat([self.pose_rot.detach(), pose_init], dim=1))
        self.birth_steps.append(int(birth_step))
        self.is_inserted.append(True)
        self.birth_modes.append(str(birth_mode))
        self._refresh_bones()
        return self.joint_count - 1

    def split_bone(
        self,
        bone_index: int,
        rest_position: torch.Tensor,
        pose_init: torch.Tensor | None = None,
        child_pose_init: torch.Tensor | None = None,
        birth_step: int = 0,
        birth_mode: str = "split",
    ) -> tuple[int, int]:
        bone_index = int(bone_index)
        if bone_index < 0 or bone_index >= self.bone_count:
            raise IndexError(f"bone_index out of range: {bone_index}")
        parent_joint = int(self.bone_parent_idx[bone_index].item())
        child_joint = int(self.bone_child_idx[bone_index].item())
        new_joint_id = self.joint_count
        rest_position = rest_position.detach().to(self.rest_joints.device, self.rest_joints.dtype).reshape(1, 3)

        updated_parent_idx = self.parent_idx.clone()
        updated_parent_idx[child_joint] = new_joint_id
        updated_parent_idx = torch.cat(
            [
                updated_parent_idx,
                torch.tensor([parent_joint], dtype=torch.long, device=self.parent_idx.device),
            ],
            dim=0,
        )
        self.parent_idx = updated_parent_idx
        self.connected_to_parent = torch.cat(
            [
                self.connected_to_parent,
                torch.tensor([True], dtype=torch.bool, device=self.connected_to_parent.device),
            ],
            dim=0,
        )
        self.init_rest_joints = torch.cat([self.init_rest_joints, rest_position.clone()], dim=0)
        self._replace_parameter("rest_joints", torch.cat([self.rest_joints.detach(), rest_position], dim=0))

        if pose_init is None:
            pose_init = torch.zeros(self.frame_count, 1, 3, dtype=self.pose_rot.dtype, device=self.pose_rot.device)
        else:
            assert_shape(pose_init, (self.frame_count, 3), "pose_init")
            pose_init = pose_init.detach().to(self.pose_rot.device, self.pose_rot.dtype).unsqueeze(1)

        updated_pose = self.pose_rot.detach().clone()
        if child_pose_init is not None:
            assert_shape(child_pose_init, (self.frame_count, 3), "child_pose_init")
            updated_pose[:, child_joint] = child_pose_init.detach().to(updated_pose.device, updated_pose.dtype)
        self._replace_parameter("pose_rot", torch.cat([updated_pose, pose_init], dim=1))

        self.birth_steps.append(int(birth_step))
        self.is_inserted.append(True)
        self.birth_modes.append(str(birth_mode))
        self._refresh_bones()
        return new_joint_id, child_joint
