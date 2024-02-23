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

DYNAMIC_OBJECT_POSES_FILE = "dynamic_objects_v2.csv"
DEVICE_POSES_FILE = "headset_trajectory.csv"
HAND_POSES_FILE = "hand_pose_trajectory.jsonl"
VRS_FILE = "recording.vrs"
OBJECT_LIBRARY_FOLDER = "assets"
OBJECT_LIBRARY_INSTANCES = "instance.json"  # map UID to object name


class Hot3DDataPathProvider:
    """
    High Level interface to retrieve the metadata PATH used by the dataset
    """

    def __init__(self, sequence_folder: str) -> None:
        """
        INIT_DOC_STRING
        """
        self.dynamic_objects_file = None
        self.device_poses_file = None

        # Check if expected file are present
        possible_path = os.path.join(sequence_folder, DYNAMIC_OBJECT_POSES_FILE)
        self.dynamic_objects_file = (
            possible_path if os.path.exists(possible_path) else None
        )

        possible_path = os.path.join(sequence_folder, VRS_FILE)
        self.vrs_file = possible_path if os.path.exists(possible_path) else None

        possible_path = os.path.join(sequence_folder, HAND_POSES_FILE)
        self.hand_poses_file = possible_path if os.path.exists(possible_path) else None

        possible_path = os.path.join(sequence_folder, DEVICE_POSES_FILE)
        self.device_poses_file = (
            possible_path if os.path.exists(possible_path) else None
        )

        # Looking one level up for the object library
        possible_path = os.path.join(
            os.path.dirname(sequence_folder), OBJECT_LIBRARY_INSTANCES
        )
        self.object_library_instances_file = (
            possible_path if os.path.exists(possible_path) else None
        )

    def is_valid(self) -> str:
        """
        FUNC_DOC_STRING
        """
        return (
            self.dynamic_objects_file
            and self.device_poses_file
            and self.object_library_instances_file
        )
