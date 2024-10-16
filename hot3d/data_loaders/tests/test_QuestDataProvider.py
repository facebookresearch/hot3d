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

from data_loaders.QuestDataProvider import QuestDataProvider
from projectaria_tools.core.calibration import FISHEYE624, LINEAR
from projectaria_tools.core.stream_id import StreamId

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

sequence_path = data_path / "data_sample/Quest3/P0002_273c2819"
vrs_file_filepath = str(sequence_path / "recording.vrs")
device_calibration_filepath = str(sequence_path / "camera_models.json")


class TestQuestDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_quest_recording(self) -> None:
        self.assertTrue(os.path.exists(vrs_file_filepath))
        provider = QuestDataProvider(vrs_file_filepath, device_calibration_filepath)

        self.assertIsNotNone(provider)
        self.assertIsNotNone(provider.get_device_calibration())

        # Quest VRS files contains 2 monochrome Image streams
        stream_ids = provider.get_image_stream_ids()
        self.assertEqual(len(stream_ids), 2)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        timestamps = provider.get_sequence_timestamps()
        self.assertTrue(len(timestamps) > 0)

        for stream_id in stream_ids:
            timestamp = timestamps[1]
            img_array = provider.get_image(timestamp, stream_id)
            self.assertIsNotNone(img_array)
            self.assertEqual(img_array.shape, (1024, 1280))

            undistorted_img_array = provider.get_image(timestamp, stream_id)
            self.assertIsNotNone(undistorted_img_array)

            # Retrieve camera calibration
            self.assertIsNotNone(provider.get_camera_calibration(stream_id))
            self.assertIsNotNone(provider.get_camera_calibration(stream_id, FISHEYE624))
            self.assertIsNotNone(provider.get_camera_calibration(stream_id, LINEAR))

            # Assert we have the right camera type
            self.assertEqual(
                provider.get_camera_calibration(stream_id)[1].model_name(),
                FISHEYE624,
            )
            self.assertEqual(
                provider.get_camera_calibration(stream_id, LINEAR)[1].model_name(),
                LINEAR,
            )

        # check the frameset grouping logic
        frameset_acceptable_time_diff_ns = 1e6
        reference_timestamps = provider.get_sequence_timestamps()
        expected_streamid_strs = sorted(str(x) for x in provider.get_image_stream_ids())
        for ref_tsns in reference_timestamps:
            for tsns in [ref_tsns, ref_tsns + 100, ref_tsns - 200]:
                out_frameset_a = provider.get_frameset_from_timestamp(
                    timestamp_ns=tsns,
                    frameset_acceptable_time_diff_ns=frameset_acceptable_time_diff_ns,
                )
                self.assertIsNotNone(out_frameset_a)
                self.assertEqual(sorted(out_frameset_a.keys()), expected_streamid_strs)

                # For Quest, timestamps for all streams should be the same
                self.assertEqual(len(set(out_frameset_a.values())), 1)

                ## check the timestamps are within the acceptable time difference
                for _, frameset_tsns in out_frameset_a.items():
                    self.assertTrue(
                        abs(frameset_tsns - tsns) < frameset_acceptable_time_diff_ns,
                    )
        # sanity check that an out of bounds timestamp returns None values inside the frameset
        outofbounds_frameset = provider.get_frameset_from_timestamp(
            timestamp_ns=-2 * 1e9,
            frameset_acceptable_time_diff_ns=frameset_acceptable_time_diff_ns,
        )
        self.assertIsNotNone(outofbounds_frameset)
        self.assertTrue(all(x is None for x in outofbounds_frameset.values()))
