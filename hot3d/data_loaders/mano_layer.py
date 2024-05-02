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

import os
from typing import List, Optional

import smplx

import torch

mano_to_nimble_joint_mapping = [
    16,
    17,
    18,
    19,
    20,
    0,
    14,
    15,
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    11,
    12,
    7,
    8,
    9,
]


class MANOHandModel:
    N_VERT = 778
    N_LANDMARKS = 21
    MANO_FINGERTIP_VERT_INDICES = {
        "thumb": 744,
        "index": 320,
        "middle": 443,
        "ring": 554,
        "pinky": 671,
    }

    def __init__(self, mano_model_files_dir: str, joint_mapper: Optional[List] = None):

        mano_left_filename = os.path.join(mano_model_files_dir, "MANO_LEFT.pkl")
        mano_right_filename = os.path.join(mano_model_files_dir, "MANO_RIGHT.pkl")

        self.use_pose_pca = True
        self.num_pose_coeffs = 15
        self.num_shape_params = 10
        self.device = "cpu"
        self.dtype = torch.float32
        self.joint_mapper = joint_mapper

        self.mano_layer_left = smplx.create(
            mano_left_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=False,
            num_pca_comps=self.num_pose_coeffs,
        )
        self.mano_layer_left.to(self.device)

        self.mano_layer_right = smplx.create(
            mano_right_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=True,
            num_pca_comps=self.num_pose_coeffs,
        )
        self.mano_layer_right.to(self.device)

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if (
            torch.sum(
                torch.abs(
                    self.mano_layer_left.shapedirs[:, 0, :]
                    - self.mano_layer_right.shapedirs[:, 0, :]
                )
            )
            < 1
        ):
            self.mano_layer_left.shapedirs[:, 0, :] *= -1

    def forward_kinematics(
        self,
        shape_params: torch.Tensor,
        joint_angles: torch.Tensor,
        global_xfrom: torch.Tensor,
        is_right_hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert shape_params.shape[0] == self.num_shape_params
        is_batched = len(joint_angles.shape) == 2
        if len(global_xfrom.shape) == 1:
            global_xfrom = torch.unsqueeze(global_xfrom, 0)
        assert global_xfrom.shape[1] == 6
        if len(joint_angles.shape) == 1:
            joint_angles = torch.unsqueeze(joint_angles, 0)
        if self.use_pose_pca:
            assert joint_angles.shape[1] == self.num_pose_coeffs
        assert is_right_hand.shape[0] == joint_angles.shape[0]

        num_frames = joint_angles.shape[0]

        # Left hand FK
        if torch.any(torch.logical_not(is_right_hand)):
            left_global_xform = global_xfrom[torch.logical_not(is_right_hand)]
            left_joint_angles = joint_angles[torch.logical_not(is_right_hand)]
            left_mano_output = self.mano_layer_left(
                betas=shape_params[None]
                .repeat(left_global_xform.shape[0], 1)
                .to(self.dtype),
                global_orient=left_global_xform[:, :3].to(self.dtype),
                hand_pose=left_joint_angles.to(self.dtype),
                transl=left_global_xform[:, 3:].to(self.dtype),
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        # Right hand FK
        if torch.any(is_right_hand):
            right_global_xform = global_xfrom[is_right_hand]
            right_joint_angles = joint_angles[is_right_hand]
            right_mano_output = self.mano_layer_right(
                betas=shape_params[None]
                .repeat(right_global_xform.shape[0], 1)
                .to(self.dtype),
                global_orient=right_global_xform[:, :3].to(self.dtype),
                hand_pose=right_joint_angles.to(self.dtype),
                transl=right_global_xform[:, 3:].to(self.dtype),
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        # Merge the left and right hand outputs
        out_vertices = torch.zeros(
            (
                num_frames,
                self.N_VERT,
                3,
            )
        ).to(self.device)
        if torch.any(torch.logical_not(is_right_hand)):
            out_vertices[torch.logical_not(is_right_hand)] = left_mano_output.vertices
        if torch.sum(is_right_hand) > 0:
            out_vertices[is_right_hand] = right_mano_output.vertices

        out_landmarks = torch.zeros(
            (
                num_frames,
                self.N_LANDMARKS,
                3,
            )
        ).to(self.device)
        if torch.any(torch.logical_not(is_right_hand)):
            if left_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = torch.index_select(
                    left_mano_output.vertices,
                    1,
                    torch.tensor(
                        list(self.MANO_FINGERTIP_VERT_INDICES.values()),
                        dtype=torch.long,
                    ),
                )
                joints = torch.cat([left_mano_output.joints, extra_joints], dim=1)
            else:
                joints = left_mano_output.joints
            out_landmarks[torch.logical_not(is_right_hand)] = joints
        if torch.sum(is_right_hand) > 0:
            if right_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = torch.index_select(
                    right_mano_output.vertices,
                    1,
                    torch.tensor(
                        list(self.MANO_FINGERTIP_VERT_INDICES.values()),
                        dtype=torch.long,
                    ),
                )
                joints = torch.cat([right_mano_output.joints, extra_joints], dim=1)
            else:
                joints = right_mano_output.joints
            out_landmarks[is_right_hand] = joints

        assert out_landmarks.shape[1] == self.N_LANDMARKS

        if self.joint_mapper is not None:
            out_landmarks = out_landmarks[:, self.joint_mapper]

        # convert to mm
        out_vertices *= 1e3
        out_landmarks *= 1e3

        if not is_batched:
            out_vertices = torch.squeeze(out_vertices, 0)
            out_landmarks = torch.squeeze(out_landmarks, 0)

        return out_vertices, out_landmarks
