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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytorch3d
import torch
from data_loaders.HandDataProviderBase import HandDataProviderBase

from data_loaders.loader_hand_poses import (
    Handedness,
    HandModelType,
    HandPose,
    load_hand_poses,
    load_mano_shape_params,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual

from .mano_layer import MANOHandModel


@dataclass
class HandPosesWithDt:
    hand_poses: List[HandPose]
    time_delta_ns: int


class MANOHandDataProvider(HandDataProviderBase):

    def __init__(
        self,
        hand_pose_trajectory_filepath: str,
        mano_layer: MANOHandModel,
    ) -> None:

        super().__init__()
        self._hand_poses = load_hand_poses(
            hand_pose_trajectory_filepath, HandModelType.MANO
        )
        self._sorted_timestamp_ns_list: List[int] = sorted(self._hand_poses.keys())

        # Hand profile
        self._mano_shape_params = load_mano_shape_params(hand_pose_trajectory_filepath)
        if self._mano_shape_params is not None:
            self._mano_shape_params = torch.from_numpy(
                np.array(self._mano_shape_params)
            )

        self.mano_layer = mano_layer

    def get_hand_mesh_vertices(
        self, hand_wrist_data: HandPose
    ) -> Optional[torch.Tensor]:
        """
        Return the hand mesh corresponding to given HandPose
        """
        if (
            hand_wrist_data.wrist_pose is not None
            and self._mano_shape_params is not None
        ):
            hand_wrist_pose_matrix = hand_wrist_data.wrist_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            hand_wrist_rotation_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(
                hand_wrist_pose_tensor[:3, :3]
            )
            hand_wrist_pose_tensor = torch.cat(
                [hand_wrist_rotation_axis_angle, hand_wrist_pose_tensor[:3, 3]]
            )

            mesh_vertices, landmarks = self.mano_layer.forward_kinematics(
                self._mano_shape_params,
                torch.from_numpy(np.array(hand_wrist_data.joint_angles)),
                hand_wrist_pose_tensor,
                torch.tensor([hand_wrist_data.handedness == Handedness.Right]),
            )

            # Rescale and translate the vertices
            scale = 1e-3
            mesh_vertices = mesh_vertices.mul(scale)
            translation = torch.Tensor(hand_wrist_data.wrist_pose.translation()[0])
            return mesh_vertices + translation.expand_as(mesh_vertices)
        return None

    def get_hand_mesh_faces_and_normals(
        self, hand_wrist_data: HandPose
    ) -> Optional[List[np.ndarray]]:
        """
        Return the hand mesh faces and normals
        """
        if self.mano_layer is not None:
            if hand_wrist_data.handedness == Handedness.Right:
                hand_triangles = self.mano_layer.mano_layer_right.faces
            else:
                hand_triangles = self.mano_layer.mano_layer_left.faces

            vertices = self.get_hand_mesh_vertices(hand_wrist_data)
            assert vertices is not None
            normals = MANOHandDataProvider.get_triangular_mesh_normals(
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
        if (
            hand_wrist_data.wrist_pose is not None
            and self._mano_shape_params is not None
        ):
            hand_wrist_pose_matrix = hand_wrist_data.wrist_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            hand_wrist_rotation_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(
                hand_wrist_pose_tensor[:3, :3]
            )
            hand_wrist_pose_tensor = torch.cat(
                [hand_wrist_rotation_axis_angle, hand_wrist_pose_tensor[:3, 3]]
            )

            mesh_vertices, hand_landmarks = self.mano_layer.forward_kinematics(
                self._mano_shape_params,
                torch.from_numpy(np.array(hand_wrist_data.joint_angles)),
                hand_wrist_pose_tensor,
                torch.tensor([hand_wrist_data.handedness == Handedness.Right]),
            )
            # Rescale and translate the vertices
            scale = 1e-3
            hand_landmarks = hand_landmarks.mul(scale)
            translation = torch.Tensor(hand_wrist_data.wrist_pose.translation()[0])
            return hand_landmarks + translation.expand_as(hand_landmarks)

        return None
