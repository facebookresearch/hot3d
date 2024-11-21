# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data_loaders.HandDataProviderBase import HandDataProviderBase

from data_loaders.loader_hand_poses import Handedness, HandPose

from data_loaders.umetrack_layer import get_skinning_weights, skin_points


@dataclass
class UmeTrackHandModelData:
    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor
    joint_frame_index: torch.Tensor
    joint_parent: torch.Tensor
    joint_first_child: torch.Tensor
    joint_next_sibling: torch.Tensor
    landmark_rest_positions: torch.Tensor
    landmark_rest_bone_weights: torch.Tensor
    landmark_rest_bone_indices: torch.Tensor
    hand_scale: Optional[torch.Tensor] = None
    mesh_vertices: Optional[torch.Tensor] = None
    mesh_triangles: Optional[torch.Tensor] = None
    dense_bone_weights: Optional[torch.Tensor] = None
    joint_limits: Optional[torch.Tensor] = None


def from_dict(j: Dict[str, Any]) -> UmeTrackHandModelData:
    model = UmeTrackHandModelData(**{k: torch.tensor(v) for k, v in j.items()})
    MM_TO_M = 1e-3
    model.joint_rest_positions *= MM_TO_M
    model.landmark_rest_positions *= MM_TO_M
    if model.mesh_vertices is not None:
        model.mesh_vertices *= MM_TO_M
    return model


def load_hand_model_from_file(filename: str) -> Optional[UmeTrackHandModelData]:
    with open(filename, "rb") as f:
        hand_model_dict = json.load(f)
        if "hand_model" in hand_model_dict.keys():
            return from_dict(hand_model_dict["hand_model"])
    return None


class UmeTrackHandDataProvider(HandDataProviderBase):
    def __init__(
        self, hand_pose_trajectory_filepath: str, hand_profile_filepath: str
    ) -> None:
        super().__init__()
        super()._init_hand_poses(hand_pose_trajectory_filepath)

        # Hand profile
        self._hand_model = (
            None
            if len(self._hand_poses) == 0
            else load_hand_model_from_file(hand_profile_filepath)
        )

    def get_hand_mesh_vertices(
        self, hand_wrist_data: HandPose
    ) -> Optional[torch.Tensor]:
        """
        Return the hand mesh corresponding to given HandPose
        """
        if hand_wrist_data.wrist_pose is not None and self._hand_model is not None:
            hand_wrist_pose_matrix = hand_wrist_data.wrist_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == Handedness.Right:
                hand_wrist_pose_tensor[:, 0] *= -1

            mesh_vertices = skin_vertices(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            return mesh_vertices
        return None

    def get_hand_mesh_faces_and_normals(
        self, hand_wrist_data: HandPose
    ) -> Optional[List[np.ndarray]]:
        """
        Return the hand mesh faces and normals
        """
        if self._hand_model is not None and self._hand_model.mesh_triangles is not None:
            hand_triangles = self._hand_model.mesh_triangles.int().numpy()
            vertices = self.get_hand_mesh_vertices(hand_wrist_data)
            assert vertices is not None
            normals = HandDataProviderBase.get_triangular_mesh_normals(
                vertices.float().numpy(), hand_triangles
            )
            return [hand_triangles, normals]
        else:
            return None

    def get_hand_landmarks(self, hand_wrist_data: HandPose) -> Optional[torch.Tensor]:
        """
        Return the hand joint landmarks corresponding to given HandPose
        See how to map the vertices together to represent a Hand as linked lines using LANDMARK_CONNECTIVITY
        """
        if self._hand_model is not None and self._hand_model.mesh_triangles is not None:
            hand_wrist_pose_matrix = hand_wrist_data.wrist_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == Handedness.Right:
                hand_wrist_pose_tensor[:, 0] *= -1

            hand_landmarks = skin_landmarks(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            return hand_landmarks

        return None


NUM_JOINT_FRAMES: int = 1 + 1 + 3 * 5  # root + wrist + finger frames * 5


def skin_landmarks(
    hand_model: UmeTrackHandModelData,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    leading_dims = joint_angles.shape[:-1]
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1
    max_weights = hand_model.landmark_rest_bone_indices.shape[-1]
    skin_mat = get_skinning_weights(
        hand_model.landmark_rest_bone_indices.reshape(numel, -1, max_weights),
        hand_model.landmark_rest_bone_weights.reshape(numel, -1, max_weights),
        NUM_JOINT_FRAMES,
    )
    return skin_points(
        hand_model.joint_rest_positions.double(),
        hand_model.joint_rotation_axes.double(),
        skin_mat.double(),
        joint_angles.double(),
        hand_model.landmark_rest_positions.double(),
        wrist_transforms.double(),
    )


def skin_vertices(
    hand_model: UmeTrackHandModelData,
    joint_angles: torch.Tensor,
    wrist_transforms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert hand_model.mesh_vertices is not None, "mesh vertices should not be none"
    assert (
        hand_model.dense_bone_weights is not None
    ), "dense bone weights should not be none"
    vertices = skin_points(
        hand_model.joint_rest_positions.double(),
        hand_model.joint_rotation_axes.double(),
        # pyre-fixme[16]: `Optional` has no attribute `double`.
        hand_model.dense_bone_weights.double(),
        joint_angles.double(),
        hand_model.mesh_vertices.double(),
        wrist_transforms.double(),
    )

    leading_dims = joint_angles.shape[:-1]
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    return vertices
