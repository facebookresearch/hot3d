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

from .headsets import Headset
from .io_utils import load_json


class Hot3DDataPathProvider(object):
    @staticmethod
    def fromRecordingFolder(recording_instance_folderpath):

        metadata_filepath = os.path.join(recording_instance_folderpath, "metadata.json")
        metadata_json = load_json(metadata_filepath)
        headset = Headset[metadata_json["headset"]]

        if headset is Headset.Aria:
            return AriaDatasetPaths(
                recording_instance_folderpath=recording_instance_folderpath
            )
        elif headset is Headset.Quest3:
            return Quest3DatasetPaths(
                recording_instance_folderpath=recording_instance_folderpath
            )
        else:
            raise NotImplementedError(f"{headset} not supported at the moment.")


class Quest3DatasetPaths(object):
    def __init__(self, recording_instance_folderpath):
        self._recording_instance_folderpath = recording_instance_folderpath

    @property
    def recording_instance_folderpath(self):
        return self._recording_instance_folderpath

    @property
    def dynamic_objects_filepath(self):
        return f"{self._recording_instance_folderpath}/dynamic_objects.csv"

    @property
    def headset_trajectory_filepath(self):
        return f"{self._recording_instance_folderpath}/headset_trajectory.csv"

    @property
    def headset_metadata_filepath(self):
        return f"{self._recording_instance_folderpath}/headset_metadata.json"

    @property
    def hand_pose_trajectory_filepath(self):
        return f"{self._recording_instance_folderpath}/hand_pose_trajectory.jsonl"

    @property
    def hand_user_profile_filepath(self):
        return f"{self._recording_instance_folderpath}/hand_user_profile.json"

    @property
    def vrs_filepath(self):
        return f"{self._recording_instance_folderpath}/recording.vrs"

    @property
    def required_filepaths(self):
        return [
            self.vrs_filepath,
            self.dynamic_objects_filepath,
            self.headset_trajectory_filepath,
            self.headset_metadata_filepath,
            self.hand_pose_trajectory_filepath,
            self.hand_user_profile_filepath,
            self.markers_frames_filepath,
        ]

    def is_valid(self) -> bool:
        return all(os.path.exists(filepath) for filepath in self.required_filepaths)


class AriaDatasetPaths(object):
    def __init__(self, recording_instance_folderpath):
        self._recording_instance_folderpath = recording_instance_folderpath

    @property
    def recording_instance_folderpath(self):
        return self._recording_instance_folderpath

    @property
    def dynamic_objects_filepath(self):
        return f"{self._recording_instance_folderpath}/dynamic_objects.csv"

    @property
    def headset_trajectory_filepath(self):
        return f"{self._recording_instance_folderpath}/headset_trajectory.csv"

    @property
    def headset_metadata_filepath(self):
        return f"{self._recording_instance_folderpath}/headset_metadata.json"

    @property
    def hand_pose_trajectory_filepath(self):
        return f"{self._recording_instance_folderpath}/hand_pose_trajectory.jsonl"

    @property
    def hand_user_profile_filepath(self):
        return f"{self._recording_instance_folderpath}/hand_user_profile.json"

    @property
    def vrs_filepath(self):
        return f"{self._recording_instance_folderpath}/recording.vrs"

    @property
    def mps_folderpath(self):
        return f"{self._recording_instance_folderpath}/mps"

    @property
    def required_filepaths(self):
        return [
            self.vrs_filepath,
            self.dynamic_objects_filepath,
            self.headset_trajectory_filepath,
            self.headset_metadata_filepath,
            self.hand_pose_trajectory_filepath,
        ]

    def is_valid(self) -> bool:
        return all(os.path.exists(filepath) for filepath in self.required_filepaths)
