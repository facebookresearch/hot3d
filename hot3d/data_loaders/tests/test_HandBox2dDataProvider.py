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

from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual


try:
    from libfb.py import parutil

    data_path = Path(parutil.get_file_path("test_data/", pkg=__package__))
except ImportError:
    data_path = Path(__file__).parent

sequence_path = data_path / "data_sample/Aria/P0003_c701bd11"
box2d_hands_filepath = str(sequence_path / "box2d_hands.csv")


class TestHandBox2dDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_aria_recording(self) -> None:
        self.assertTrue(os.path.exists(box2d_hands_filepath))
        provider = load_box2d_trajectory_from_csv(filename=box2d_hands_filepath)

        stream_ids = provider.stream_ids
        self.assertEqual(len(stream_ids), 3)

        self.assertTrue(StreamId("214-1") in stream_ids)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        for stream_id in stream_ids:
            timestamp_ns_list = provider.get_timestamp_ns_list(stream_id=stream_id)
            self.assertIsNotNone(timestamp_ns_list)
            self.assertGreater(len(timestamp_ns_list), 0)

            query_timestamp_ns = timestamp_ns_list[len(timestamp_ns_list) // 2]

            box2d_collection_with_dt = provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=query_timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )

            self.assertIsNotNone(box2d_collection_with_dt)
            self.assertIsNotNone(box2d_collection_with_dt.box2d_collection)