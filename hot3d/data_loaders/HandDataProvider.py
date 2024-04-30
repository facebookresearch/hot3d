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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data_loaders.loader_hand_poses import (
    Handedness,
    HandPose,
    HandPose3dCollection,
    load_hand_poses,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual

from UmeTrack.common.hand_skinning import skin_landmarks, skin_vertices
from UmeTrack.common.loader_handmodel import load_hand_model_from_file

from .pose_utils import lookup_timestamp


@dataclass
class HandPose3dCollectionWithDt:
    pose3d_collection: HandPose3dCollection
    time_delta_ns: int


class HandDataProvider:

    def __init__(
        self, hand_pose_trajectory_filepath: str, hand_profile_filepath: str
    ) -> None:

        self._hand_poses = load_hand_poses(hand_pose_trajectory_filepath)
        self._sorted_timestamp_ns_list: List[int] = sorted(self._hand_poses.keys())

        # Hand profile
        self._hand_model = load_hand_model_from_file(hand_profile_filepath)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Returns the stats of the Hand data
        """
        stats = {}
        stats["num_frames"] = len(self._sorted_timestamp_ns_list)
        stats["num_right_hands"] = sum(
            [
                1
                for it in self._hand_poses.values()
                if Handedness.Right in it.poses.keys()
            ]
        )
        stats["num_left_hands"] = sum(
            [
                1
                for it in self._hand_poses.values()
                if Handedness.Left in it.poses.keys()
            ]
        )
        return stats

    def get_pose_at_timestamp(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain,
        acceptable_time_delta: Optional[int] = None,
    ) -> Optional[HandPose3dCollectionWithDt]:
        """
        Return the list of hands available at a given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        hand_pose_collection, time_delta_ns = lookup_timestamp(
            time_indexed_dict=self._hand_poses,
            sorted_timestamp_list=self._sorted_timestamp_ns_list,
            query_timestamp=timestamp_ns,
            time_query_options=time_query_options,
        )

        if (
            hand_pose_collection is None
            or time_delta_ns is None
            or (
                acceptable_time_delta is not None
                and abs(time_delta_ns) > acceptable_time_delta
            )
        ):
            return None
        else:
            return HandPose3dCollectionWithDt(
                pose3d_collection=hand_pose_collection, time_delta_ns=time_delta_ns
            )

    def get_hand_mesh_vertices(self, hand_wrist_data: HandPose) -> torch.Tensor:
        """
        Return the hand mesh corresponding to given HandPose
        """
        if hand_wrist_data.hand_pose is not None:
            hand_wrist_pose_matrix = hand_wrist_data.hand_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == Handedness.Right:
                hand_wrist_pose_tensor[:, 0] *= -1

            mesh_vertices = skin_vertices(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            # Rescale and translate the vertices
            scale = 1e-3
            mesh_vertices = mesh_vertices.mul(scale)
            translation = torch.Tensor(hand_wrist_data.hand_pose.translation()[0])
            return mesh_vertices + translation.expand_as(mesh_vertices)
        return None

    def get_hand_mesh_faces_and_normals(
        self, hand_wrist_data: HandPose
    ) -> Optional[tuple[np.array, np.array]]:
        """
        Return the hand mesh faces and normals
        """
        if self._hand_model is not None:
            hand_triangles = self._hand_model.mesh_triangles.int().numpy()
            vertices = self.get_hand_mesh_vertices(hand_wrist_data)
            normals = HandDataProvider.get_triangular_mesh_normals(
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
        if hand_wrist_data.hand_pose is not None:
            hand_wrist_pose_matrix = hand_wrist_data.hand_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == Handedness.Right:
                hand_wrist_pose_tensor[:, 0] *= -1
            hand_landmarks = skin_landmarks(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            # Rescale and translate the vertices
            scale = 1e-3
            hand_landmarks = hand_landmarks.mul(scale)
            translation = torch.Tensor(hand_wrist_data.hand_pose.translation()[0])
            return hand_landmarks + translation.expand_as(hand_landmarks)

        return None

    @staticmethod
    def normalized(
        vecs: np.ndarray, axis: int = -1, add_const_to_denom: bool = True
    ) -> np.ndarray:
        """
        Normalize a set of vectors.
        Args:
            vecs: np.ndarray of shape (..., V).
            axis: axis along which to normalize.
            add_const_to_denom: if True, add a small constant to the denominator to prevent numerical issues.
        Returns:
            np.ndarray of the same shape as vecs.
        """
        denom = np.linalg.norm(vecs, axis=axis, keepdims=True)
        if add_const_to_denom:
            denom += 1e-5
        return vecs / denom

    @staticmethod
    def get_triangular_mesh_normals(
        vertices: np.ndarray, triangles: np.ndarray
    ) -> np.ndarray:
        """
        Compute the normals of a triangular mesh.
        Args:
            vertices: np.ndarray of shape (..., V, 3).
            triangles: np.ndarray of shape (..., F, 3).
        Returns:
            normals: np.ndarray of shape (..., F, 3).
        """
        norm = np.zeros_like(vertices)
        tris = vertices[triangles]
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        n = HandDataProvider.normalized(n)
        norm[triangles[:, 0]] += n
        norm[triangles[:, 1]] += n
        norm[triangles[:, 2]] += n
        return HandDataProvider.normalized(norm)
