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

import importlib.resources
import os
import unittest
from pathlib import Path

from data_loaders.ObjectPose3dProvider import load_pose_provider_from_csv
from data_loaders.PathProvider import Hot3dDataPathProvider

# pyre-fixme[21]: Could not find name `TimeDomain` in
#  `projectaria_tools.core.sensor_data`.
# pyre-fixme[21]: Could not find name `TimeQueryOptions` in
#  `projectaria_tools.core.sensor_data`.
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

try:
    data_path = Path(
        str(
            importlib.resources.files(__package__).joinpath(
                "test_data/",
            )
        )
    )
except ImportError:
    data_path = Path(__file__).parent

sequence_path = data_path / "data_sample/Aria/P0003_c701bd11"


class TestObjectPoseDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.path_provider = Hot3dDataPathProvider.fromRecordingFolder(
            recording_instance_folderpath=sequence_path
        )

    def test_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.path_provider.dynamic_objects_filepath))

    def test_object_pose_provider(self) -> None:
        object_pose_provider = load_pose_provider_from_csv(
            self.path_provider.dynamic_objects_filepath
        )
        self.assertIsNotNone(object_pose_provider)

        # Statistics must report a non empty dictionary detailing the loaded data
        self.assertIsNotNone(object_pose_provider.get_data_statistics())

        #
        # Ability to retrieve the timestamps of the annotations
        #
        timestamps = object_pose_provider.timestamp_ns_list
        self.assertGreater(len(timestamps), 0)

        #
        # Ability to retrieve the Unique Identified of the tracked objects
        #
        object_uids = object_pose_provider.object_uids_with_poses
        self.assertGreater(len(object_uids), 0)

        #
        # Ability to retrieve the poses of the tracked objects
        #
        object_poses_with_dt = object_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamps[0],
            # pyre-fixme[16]: Module `sensor_data` has no attribute `TimeQueryOptions`.
            time_query_options=TimeQueryOptions.CLOSEST,
            # pyre-fixme[16]: Module `sensor_data` has no attribute `TimeDomain`.
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=0,  # Retrieve perfect match
        )
        self.assertIsNotNone(object_poses_with_dt)

        # Test the collection at this timestamp
        objects_pose3d_collection = object_poses_with_dt.pose3d_collection
        # Test that all UIDs are present in the pose data at this timestamp
        self.assertEqual(len(objects_pose3d_collection.poses), len(object_uids))

        # Test that annotations have valid pose data
        for (
            object_uid,
            object_pose3d,
        ) in objects_pose3d_collection.poses.items():
            self.assertIsNotNone(object_pose3d.T_world_object)
            self.assertTrue(object_uid in object_uids)
