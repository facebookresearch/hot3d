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
from typing import Dict, List, Optional

import numpy as np
from projectaria_tools.core.sophus import SE3  # @manual


def _get_hand_pose(handedness: str, hand_poses_json: Dict) -> Optional[SE3]:
    if handedness in hand_poses_json.keys():
        wrist = hand_poses_json[handedness]["wrist_xform"]
        quaternion_w = wrist["q_wxyz"][0]
        quaternion_xyz = wrist["q_wxyz"][1:4]
        translation = wrist["t_xyz"]
        hand_pose = SE3.from_quat_and_translation(
            float(quaternion_w),
            np.array([float(o) for o in quaternion_xyz]),
            np.array([float(o) for o in translation]),
        )[0]
        return hand_pose
    return None


def _get_joint_angles(handedness: str, hand_poses_json: Dict) -> Optional[List[float]]:
    if handedness in hand_poses_json.keys():
        joint_angles = hand_poses_json[handedness]["joint_angles"]
        return joint_angles
    return None


class HandPose:
    """Define a Hand as hand_pose, handedness (0: left, 1: right) and joint_angles.
    - Note that joint_angles and Hand_pose can be set to None if the hand was not seen.
    """

    def __init__(
        self,
        handedness: str,
        hand_pose: Optional[SE3],
        joint_angles: Optional[List[float]],
    ):
        self._handedness = handedness
        self._hand_pose = hand_pose
        self._joint_angles = joint_angles

    @property
    def handedness(self) -> str:
        return self._handedness

    @handedness.setter
    def handedness(self, value: str):
        self._handedness = value

    @property
    def hand_pose(self) -> Optional[SE3]:
        return self._hand_pose

    @property
    def joint_angles(self) -> Optional[List[float]]:
        return self._joint_angles

    def is_left_hand(self) -> bool:
        return self._handedness == "0"

    def is_right_hand(self) -> bool:
        return self._handedness == "1"

    def handedness_label(self) -> str:
        return "left" if self.is_left_hand() else "right"


def load_hand_poses(filename: str) -> Dict[int, List[HandPose]]:
    """Load Hand Poses meta data from a JSONL file.

    Keyword arguments:
    filename -- the jsonl file i.e. sequence_folder + "/hand_pose_trajectory.jsonl"
    """
    hand_poses_per_timestamp = {}
    hand_poses_count = {}
    hand_poses_count["0"] = 0
    hand_poses_count["1"] = 0
    # Open the CSV file for reading
    f = open(filename, "r")

    for line in f:
        # Parse the JSON file line
        hand_pose_instance = json.loads(line)
        timestamp_ns = hand_pose_instance["timestamp_ns"]
        hand_poses_json = hand_pose_instance["hand_poses"]

        # Read hand pose data
        left_hand_pose = _get_hand_pose("0", hand_poses_json)
        right_hand_pose = _get_hand_pose("1", hand_poses_json)

        left_joint_angles = _get_joint_angles("0", hand_poses_json)
        right_joint_angles = _get_joint_angles("1", hand_poses_json)

        if (
            timestamp_ns not in hand_poses_per_timestamp
            and left_hand_pose is not None
            or right_hand_pose is not None
        ):
            hand_poses_per_timestamp[timestamp_ns] = []

        if left_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].append(
                HandPose("0", left_hand_pose, left_joint_angles)
            )
            hand_poses_count["0"] += 1

        if right_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].append(
                HandPose("1", right_hand_pose, right_joint_angles)
            )
            hand_poses_count["1"] += 1

    # Print statistics
    print(
        f"Hand pose data loading stats: \n\
        \tNumber of timestamps: {len(hand_poses_per_timestamp.keys())}\n\
        \tNumber of Left Hand pose: {hand_poses_count['0']}\n\
        \tNumber of Right Hand pose: {hand_poses_count['1']}"
    )
    return hand_poses_per_timestamp
