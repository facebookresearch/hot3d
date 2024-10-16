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

from data_loaders.HandDataProviderBase import HandDataProviderBase

from data_loaders.loader_hand_poses import Handedness

from data_loaders.mano_layer import MANOHandModel
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from data_loaders.UmeTrackHandDataProvider import UmeTrackHandDataProvider
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
mano_hand_pose_trajectory_filepath = str(
    data_path / sequence_path / "mano_hand_pose_trajectory.jsonl"
)

umetrack_trajectory_resource = str(
    sequence_path / "umetrack_hand_pose_trajectory.jsonl"
)
umetrack_profile_resource = str(sequence_path / "umetrack_hand_user_profile.json")


class TestHandDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_file_exists(self) -> None:
        self.assertTrue(os.path.exists(mano_hand_pose_trajectory_filepath))
        self.assertTrue(os.path.exists(umetrack_trajectory_resource))
        self.assertTrue(os.path.exists(umetrack_profile_resource))

    def test_provider_mano_hands(self) -> None:
        hand_data_provider = MANOHandDataProvider(
            mano_hand_pose_trajectory_filepath,
            None,
            # Using this with no mano_layer=None provide the ability to recover wrist pose, but no FK (hand vertices, landmarks)
        )
        self.hand_provider_test(hand_data_provider)

    def test_provider_hand_umetrack(self) -> None:
        hand_data_provider = UmeTrackHandDataProvider(
            umetrack_trajectory_resource,
            umetrack_profile_resource,
        )
        self.hand_provider_test(hand_data_provider)

    def hand_provider_test(self, hand_data_provider: HandDataProviderBase) -> None:
        # Function to test a generic hand provider, i.e being either:
        #  - MANOHandDataProvider
        #  - UmeTrackHandDataProvider
        self.assertIsNotNone(hand_data_provider)

        self.assertIsNotNone(hand_data_provider)

        timestamps = hand_data_provider.timestamp_ns_list
        self.assertTrue(len(timestamps) > 0)

        hand_statistics = hand_data_provider.get_data_statistics()
        self.assertGreater(hand_statistics["num_frames"], 0)
        self.assertGreater(hand_statistics["num_right_hands"], 0)
        self.assertGreater(hand_statistics["num_left_hands"], 0)

        for timestamp_it in timestamps:
            hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_it,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(hand_poses_with_dt)

            hand_pose_collection = hand_poses_with_dt.pose3d_collection
            self.assertIsNotNone(hand_pose_collection)
            self.assertTrue(len(hand_pose_collection.poses) > 0)
            for hand_pose_data in hand_pose_collection.poses.values():
                # Check that Handedness label and type are matching
                handedness_label = hand_pose_data.handedness_label()
                if hand_pose_data.handedness is Handedness.Left:
                    self.assertTrue(handedness_label == "left")
                if hand_pose_data.handedness is Handedness.Right:
                    self.assertTrue(handedness_label == "right")

                # Check that the wrist pose can be retrieved
                self.assertIsNotNone(hand_pose_data.wrist_pose)
