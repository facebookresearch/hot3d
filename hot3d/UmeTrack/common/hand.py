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

from enum import Enum
from typing import NamedTuple, Optional

import torch

NUM_HANDS = 2
NUM_LANDMARKS_PER_HAND = 21
NUM_FINGERTIPS_PER_HAND = 5
NUM_JOINTS_PER_HAND = 22
LEFT_HAND_INDEX = 0
RIGHT_HAND_INDEX = 1

NUM_DIGITS: int = 5
NUM_JOINT_FRAMES: int = 1 + 1 + 3 * 5  # root + wrist + finger frames * 5
DOF_PER_FINGER: int = 4


class LANDMARK(Enum):
    THUMB_FINGERTIP = "Thumb fingertip"
    INDEX_FINGER_FINGERTIP = "Index finger fingertip"
    MIDDLE_FINGER_FINGERTIP = "Middle finger fingertip"
    RING_FINGER_FINGERTIP = "Ring finger fingertip"
    PINKY_FINGER_FINGERTIP = "Pinky finger fingertip"
    WRIST_JOINT = "Wrist joint"
    THUMB_INTERMEDIATE_FRAME = "Thumb intermediate frame"
    THUMB_DISTAL_FRAME = "Thumb distal frame"
    INDEX_PROXIMAL_FRAME = "Index proximal frame"
    INDEX_INTERMEDIATE_FRAME = "Index intermediate frame"
    INDEX_DISTAL_FRAME = "Index distal frame"
    MIDDLE_PROXIMAL_FRAME = "Middle proximal frame"
    MIDDLE_INTERMEDIATE_FRAME = "Middle intermediate frame"
    MIDDLE_DISTAL_FRAME = "Middle distal frame"
    RING_PROXIMAL_FRAME = "Ring proximal frame"
    RING_INTERMEDIATE_FRAME = "Ring intermediate frame"
    RING_DISTAL_FRAME = "Ring distal frame"
    PINKY_PROXIMAL_FRAME = "Pinky proximal frame"
    PINKY_INTERMEDIATE_FRAME = "Pinky intermediate frame"
    PINKY_DISTAL_FRAME = "Pinky distal frame"
    PALM_CENTER = "Palm center"


LANDMARK_INDEX_TO_NAMING = [
    LANDMARK.THUMB_FINGERTIP,
    LANDMARK.INDEX_FINGER_FINGERTIP,
    LANDMARK.MIDDLE_FINGER_FINGERTIP,
    LANDMARK.RING_FINGER_FINGERTIP,
    LANDMARK.PINKY_FINGER_FINGERTIP,
    LANDMARK.WRIST_JOINT,
    LANDMARK.THUMB_INTERMEDIATE_FRAME,
    LANDMARK.THUMB_DISTAL_FRAME,
    LANDMARK.INDEX_PROXIMAL_FRAME,
    LANDMARK.INDEX_INTERMEDIATE_FRAME,
    LANDMARK.INDEX_DISTAL_FRAME,
    LANDMARK.MIDDLE_PROXIMAL_FRAME,
    LANDMARK.MIDDLE_INTERMEDIATE_FRAME,
    LANDMARK.MIDDLE_DISTAL_FRAME,
    LANDMARK.RING_PROXIMAL_FRAME,
    LANDMARK.RING_INTERMEDIATE_FRAME,
    LANDMARK.RING_DISTAL_FRAME,
    LANDMARK.PINKY_PROXIMAL_FRAME,
    LANDMARK.PINKY_INTERMEDIATE_FRAME,
    LANDMARK.PINKY_DISTAL_FRAME,
    LANDMARK.PALM_CENTER,
]

LANDMARK_NAMING_TO_INDEX = {k: i for i, k in enumerate(LANDMARK_INDEX_TO_NAMING)}

LANDMARK_CONNECTIVITY = [
    # pinky
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.WRIST_JOINT],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_INTERMEDIATE_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_DISTAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_DISTAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_FINGER_FINGERTIP],
    ],
    # ring
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.WRIST_JOINT],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_INTERMEDIATE_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_DISTAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_DISTAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_FINGER_FINGERTIP],
    ],
    # middle
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.WRIST_JOINT],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_INTERMEDIATE_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_DISTAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_DISTAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_FINGER_FINGERTIP],
    ],
    # index
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.WRIST_JOINT],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_INTERMEDIATE_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_DISTAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_DISTAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_FINGER_FINGERTIP],
    ],
    # thumb
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.WRIST_JOINT],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_INTERMEDIATE_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_DISTAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_DISTAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_FINGERTIP],
    ],
    # palm
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.THUMB_INTERMEDIATE_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.INDEX_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.MIDDLE_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_PROXIMAL_FRAME],
    ],
    [
        LANDMARK_NAMING_TO_INDEX[LANDMARK.RING_PROXIMAL_FRAME],
        LANDMARK_NAMING_TO_INDEX[LANDMARK.PINKY_PROXIMAL_FRAME],
    ],
]


class HandModel(NamedTuple):
    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor
    joint_frame_index: torch.Tensor
    joint_parent: torch.Tensor
    joint_first_child: torch.Tensor
    joint_next_sibling: torch.Tensor
    landmark_rest_positions: torch.Tensor
    landmark_rest_bone_weights: torch.Tensor
    landmark_rest_bone_indices: torch.Tensor
    hand_scale: Optional[torch.Tensor]
    mesh_vertices: Optional[torch.Tensor] = None
    mesh_triangles: Optional[torch.Tensor] = None
    dense_bone_weights: Optional[torch.Tensor] = None
    joint_limits: Optional[torch.Tensor] = None


def scaled_hand_model(hand: HandModel, multiplier: float) -> HandModel:
    leading_dims = hand.joint_rest_positions.shape[:-2]
    multiplier = (
        torch.ones(
            leading_dims,
            dtype=hand.joint_rest_positions.dtype,
            device=hand.joint_rest_positions.device,
        )
        * multiplier
    )

    joint_rest_positions = hand.joint_rest_positions * multiplier[..., None, None]
    landmark_rest_positions = hand.landmark_rest_positions * multiplier[..., None, None]
    mesh_vertices = hand.mesh_vertices
    if mesh_vertices is not None:
        mesh_vertices = mesh_vertices * multiplier[..., None, None]

    return HandModel(
        joint_rotation_axes=hand.joint_rotation_axes,
        joint_rest_positions=joint_rest_positions,
        joint_frame_index=hand.joint_frame_index,
        joint_parent=hand.joint_parent,
        joint_first_child=hand.joint_first_child,
        joint_next_sibling=hand.joint_next_sibling,
        landmark_rest_positions=landmark_rest_positions,
        landmark_rest_bone_weights=hand.landmark_rest_bone_weights,
        landmark_rest_bone_indices=hand.landmark_rest_bone_indices,
        # Below are optional fields
        hand_scale=hand.hand_scale,
        mesh_vertices=mesh_vertices,
        mesh_triangles=hand.mesh_triangles,
        dense_bone_weights=hand.dense_bone_weights,
        joint_limits=hand.joint_limits,
    )


def mirrored_hand_model(hand: HandModel, to_mirror: torch.Tensor) -> HandModel:
    joint_rotation_axes = hand.joint_rotation_axes.clone()
    joint_rest_positions = hand.joint_rest_positions.clone()
    landmark_rest_positions = hand.landmark_rest_positions.clone()
    # Only 1d masks work correctly when using it to index another tensor.
    # So we flat the masks here to make it work with higher dimensionalities.
    to_mirror_flat = to_mirror.reshape(-1)
    flat_end = len(to_mirror.shape) - 1
    torch.flatten(joint_rotation_axes, 0, flat_end)[to_mirror_flat, ..., 1:] *= -1
    torch.flatten(joint_rest_positions, 0, flat_end)[to_mirror_flat, ..., 0] *= -1
    torch.flatten(landmark_rest_positions, 0, flat_end)[to_mirror_flat, ..., 0] *= -1

    mesh_vertices = hand.mesh_vertices
    if mesh_vertices is not None:
        mesh_vertices = mesh_vertices.clone()
        torch.flatten(mesh_vertices, 0, flat_end)[to_mirror_flat, ..., 0] *= 1

    return HandModel(
        joint_rotation_axes=joint_rotation_axes,
        joint_rest_positions=joint_rest_positions,
        joint_frame_index=hand.joint_frame_index,
        joint_parent=hand.joint_parent,
        joint_first_child=hand.joint_first_child,
        joint_next_sibling=hand.joint_next_sibling,
        landmark_rest_positions=landmark_rest_positions,
        landmark_rest_bone_weights=hand.landmark_rest_bone_weights,
        landmark_rest_bone_indices=hand.landmark_rest_bone_indices,
        # Below are optional fields
        hand_scale=hand.hand_scale,
        mesh_vertices=mesh_vertices,
        mesh_triangles=hand.mesh_triangles,
        dense_bone_weights=hand.dense_bone_weights,
        joint_limits=hand.joint_limits,
    )
