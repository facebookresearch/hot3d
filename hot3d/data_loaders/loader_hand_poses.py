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
from typing import Dict, Optional

import numpy as np
from projectaria_tools.core.sophus import SE3


def getHandPose(handedness: str, hand_poses_json: Dict) -> Optional[SE3]:
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


def load_hand_poses(filename: str) -> Dict[int, Dict[str, SE3]]:
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
        print(line)
        # Parse the JSON file
        hand_pose_instance = json.loads(line)
        timestamp_ns = hand_pose_instance["timestamp_ns"]
        hand_poses_json = hand_pose_instance["hand_poses"]

        # Read hand pose data
        left_hand_pose = getHandPose("0", hand_poses_json)
        right_hand_pose = getHandPose("1", hand_poses_json)

        if timestamp_ns not in hand_poses_per_timestamp:
            hand_poses_per_timestamp[timestamp_ns] = {}
        hand_poses_per_timestamp[timestamp_ns]["0"] = left_hand_pose
        hand_poses_per_timestamp[timestamp_ns]["1"] = right_hand_pose
        if left_hand_pose is not None:
            hand_poses_count["0"] += 1
        if right_hand_pose is not None:
            hand_poses_count["1"] += 1

        # if '0' not in hand_poses_json.keys() or '1' not in hand_poses_json.keys():
        #     print('ERROR')
        #     print(hand_poses_json.keys())

        # print(hand_poses_json["0"])

        # print(hand_pose_instance)
    # Print statistics
    print(
        f"Hand pose data loading stats: \n\
        \tNumber of timestamps: {len(hand_poses_per_timestamp.keys())}\n\
        \tNumber of Left Hand pose: {hand_poses_count['0']}\n\
        \tNumber of Right Hand pose: {hand_poses_count['1']}"
    )
    return hand_poses_per_timestamp

    #             device_pose_per_timestamp[timestamp] = {}
    #         device_pose_per_timestamp[timestamp] = object_pose
    #         device_pose_count.add(object_uid)

    # # Print statistics
    # print(
    #     f"Device trajectory data loading stats: \n\
    #     \tNumber of timestamps: {len(device_pose_per_timestamp.keys())}\n\
    #     \tNumber of Device: {len(device_pose_count)}"
    # )
    # assert len(device_pose_count) == 1  # Only one device should be tracked
    # return device_pose_per_timestamp
