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
import unittest
from pathlib import Path

from data_loaders.QuestDataProvider import QuestDataProvider
from projectaria_tools.core.calibration import FISHEYE624, LINEAR
from projectaria_tools.core.stream_id import StreamId

try:
    from libfb.py import parutil

    data_path = Path(parutil.get_file_path("test_data/", pkg=__package__))
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
