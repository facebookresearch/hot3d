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

import unittest

from data_loaders.AriaDataProvider import AriaDataProvider
from libfb.py import parutil
from projectaria_tools.core.stream_id import StreamId  # @manual

aria_vrs_resource = "test_data/aria_sample_recording/sequence.vrs"


class TestAriaDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_aria_recording(self) -> None:
        vrs_file_filepath = parutil.get_file_path(aria_vrs_resource, pkg=__package__)
        provider = AriaDataProvider(vrs_file_filepath, "")

        self.assertIsNotNone(provider)
        self.assertIsNotNone(provider.get_device_calibration())

        # Aria VRS files contains 3 Image streams (1 RGB, 2 SLAM)
        stream_ids = provider.get_image_stream_ids()
        self.assertEqual(len(stream_ids), 3)
        self.assertTrue(StreamId("214-1") in stream_ids)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        timestamps = provider.get_sequence_timestamps()
        self.assertTrue(len(timestamps) > 0)

        for stream_id in stream_ids:
            img_array = provider.get_image(timestamps[0], stream_id)
            self.assertIsNotNone(img_array)
            if "rgb" in provider.get_image_stream_label(stream_id):
                self.assertEqual(img_array.shape, (1408, 1408, 3))
            elif "slam" in provider.get_image_stream_label(stream_id):
                self.assertEqual(img_array.shape, (480, 640))

            undistorted_img_array = provider.get_image(timestamps[0], stream_id)
            self.assertIsNotNone(undistorted_img_array)

            self.assertIsNotNone(provider.get_camera_calibration(stream_id))

        # MPS resources are empty since not initialized
        self.assertIsNone(provider.get_point_cloud())