from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from evorig_next.utils.geometry import EPS, assert_shape, safe_normalize
from evorig_next.utils.rotations import axis_angle_to_matrix, invert_transform, make_transform, stable_up_vector


class Skeleton(nn.Module):
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
        self._refresh_initial_bone_lengths()

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

    def _refresh_initial_bone_lengths(self) -> None:
        if self.bone_child_idx.numel() == 0:
            lengths = torch.zeros(0, dtype=self.rest_joints.dtype, device=self.rest_joints.device)
        else:
            lengths = (self.init_rest_joints[self.bone_child_idx] - self.init_rest_joints[self.bone_parent_idx]).norm(dim=-1)
        if hasattr(self, "init_bone_lengths"):
            self.init_bone_lengths = lengths
        else:
            self.register_buffer("init_bone_lengths", lengths)

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
        rotations = axis_angle_to_matrix(pose_rot_sel.reshape(-1, 3)).reshape(frame_count, self.joint_count, 3, 3)
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

    def compute_rest_global_transforms(self) -> torch.Tensor:
        rest_pose = torch.zeros(1, self.joint_count, 3, dtype=self.pose_rot.dtype, device=self.pose_rot.device)
        rest_root = torch.zeros(1, 3, dtype=self.root_trans.dtype, device=self.root_trans.device)
        return self.forward_kinematics(pose_rot=rest_pose, root_trans=rest_root)[0]

    def compute_bind_transforms(self) -> torch.Tensor:
        return invert_transform(self.compute_rest_global_transforms())

    def compute_bone_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_pos = self.rest_joints[self.bone_parent_idx]
        child_pos = self.rest_joints[self.bone_child_idx]
        direction = child_pos - parent_pos
        length = direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
        x_axis = direction / length
        up = stable_up_vector(x_axis)
        z_axis = safe_normalize(torch.cross(x_axis, up, dim=-1))
        y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
        frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        fallback = torch.eye(3, dtype=frame.dtype, device=frame.device).expand_as(frame)
        degenerate = (length.squeeze(-1) <= EPS).unsqueeze(-1).unsqueeze(-1)
        frame = torch.where(degenerate, fallback, frame)
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
        rest_position = rest_position.detach().to(self.rest_joints.device, self.rest_joints.dtype).reshape(1, 3)
        self.parent_idx = torch.cat([self.parent_idx, torch.tensor([parent_joint], dtype=torch.long, device=self.parent_idx.device)])
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
            pose_init = pose_init.detach().unsqueeze(1)
        self._replace_parameter("pose_rot", torch.cat([self.pose_rot.detach(), pose_init], dim=1))
        self.birth_steps.append(int(birth_step))
        self.is_inserted.append(True)
        self.birth_modes.append(str(birth_mode))
        self._refresh_bones()
        self._refresh_initial_bone_lengths()
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
            ]
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
            pose_init = pose_init.detach().unsqueeze(1)

        updated_pose = self.pose_rot.detach().clone()
        if child_pose_init is not None:
            assert_shape(child_pose_init, (self.frame_count, 3), "child_pose_init")
            updated_pose[:, child_joint] = child_pose_init.detach().to(updated_pose.device, updated_pose.dtype)
        self._replace_parameter("pose_rot", torch.cat([updated_pose, pose_init], dim=1))

        self.birth_steps.append(int(birth_step))
        self.is_inserted.append(True)
        self.birth_modes.append(str(birth_mode))
        self._refresh_bones()
        self._refresh_initial_bone_lengths()
        return new_joint_id, child_joint

    def reparent_joint(self, joint_id: int, new_parent_joint: int) -> None:
        joint_id = int(joint_id)
        new_parent_joint = int(new_parent_joint)
        if joint_id < 0 or joint_id >= self.joint_count:
            raise IndexError(f"joint_id out of range: {joint_id}")
        if new_parent_joint < 0 or new_parent_joint >= self.joint_count:
            raise IndexError(f"new_parent_joint out of range: {new_parent_joint}")
        current_parent = int(self.parent_idx[joint_id].item())
        if current_parent < 0:
            raise ValueError("cannot reparent the root joint")
        if new_parent_joint == joint_id:
            raise ValueError("a joint cannot parent itself")
        descendants = {joint_id}
        stack = [joint_id]
        while stack:
            current = stack.pop()
            children = torch.nonzero(self.parent_idx == current, as_tuple=False).flatten()
            for child in children.tolist():
                if int(child) in descendants:
                    continue
                descendants.add(int(child))
                stack.append(int(child))
        if new_parent_joint in descendants:
            raise ValueError("cannot reparent a joint under its own descendant")
        if new_parent_joint == current_parent:
            return
        updated_parent_idx = self.parent_idx.clone()
        updated_parent_idx[joint_id] = new_parent_joint
        self.parent_idx = updated_parent_idx
        self._refresh_bones()
        self._refresh_initial_bone_lengths()

    def collapse_joint_into_target(self, joint_id: int, target_joint: int) -> dict[str, int | list[int]]:
        joint_id = int(joint_id)
        target_joint = int(target_joint)
        if joint_id < 0 or joint_id >= self.joint_count:
            raise IndexError(f"joint_id out of range: {joint_id}")
        if target_joint < 0 or target_joint >= self.joint_count:
            raise IndexError(f"target_joint out of range: {target_joint}")
        parent_joint = int(self.parent_idx[joint_id].item())
        if parent_joint < 0:
            raise ValueError("cannot collapse the root joint")
        if target_joint == joint_id:
            raise ValueError("cannot collapse a joint into itself")

        descendants = {joint_id}
        stack = [joint_id]
        while stack:
            current = stack.pop()
            children = torch.nonzero(self.parent_idx == current, as_tuple=False).flatten()
            for child in children.tolist():
                if int(child) in descendants:
                    continue
                descendants.add(int(child))
                stack.append(int(child))
        if target_joint in descendants:
            raise ValueError("cannot collapse a joint into its own descendant")

        child_joint_ids = torch.nonzero(self.parent_idx == joint_id, as_tuple=False).flatten().tolist()
        updated_parent_idx = self.parent_idx.detach().clone()
        for child_joint in child_joint_ids:
            updated_parent_idx[int(child_joint)] = target_joint

        keep_mask = torch.ones(self.joint_count, dtype=torch.bool, device=self.parent_idx.device)
        keep_mask[joint_id] = False
        compact_parent_idx = updated_parent_idx[keep_mask]
        compact_parent_idx[compact_parent_idx > joint_id] -= 1
        self.parent_idx = compact_parent_idx
        compact_connected = self.connected_to_parent[keep_mask].clone()
        for child_joint in child_joint_ids:
            adjusted_child = int(child_joint) - 1 if int(child_joint) > int(joint_id) else int(child_joint)
            if 0 <= adjusted_child < int(compact_connected.numel()):
                compact_connected[adjusted_child] = bool(compact_connected[adjusted_child].item())
        self.connected_to_parent = compact_connected

        self.init_rest_joints = self.init_rest_joints[keep_mask].clone()
        self._replace_parameter("rest_joints", self.rest_joints.detach()[keep_mask].clone())
        self._replace_parameter("pose_rot", self.pose_rot.detach()[:, keep_mask].clone())

        self.birth_steps.pop(joint_id)
        self.is_inserted.pop(joint_id)
        self.birth_modes.pop(joint_id)
        self._refresh_bones()
        self._refresh_initial_bone_lengths()
        return {
            "removed_joint_id": joint_id,
            "kept_joint_id": target_joint if target_joint < joint_id else target_joint - 1,
            "old_parent_id": parent_joint,
            "target_joint_id": target_joint,
            "reparented_children": [int(child_id) for child_id in child_joint_ids],
        }

    def collapse_joint_into_parent(self, joint_id: int) -> dict[str, int | list[int]]:
        joint_id = int(joint_id)
        parent_joint = int(self.parent_idx[joint_id].item())
        return self.collapse_joint_into_target(joint_id, parent_joint)
