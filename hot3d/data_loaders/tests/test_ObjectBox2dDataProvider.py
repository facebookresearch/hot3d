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

from data_loaders.ObjectBox2dDataProvider import load_box2d_trajectory_from_csv

# pyre-fixme[21]: Could not find name `TimeDomain` in
#  `projectaria_tools.core.sensor_data`.
# pyre-fixme[21]: Could not find name `TimeQueryOptions` in
#  `projectaria_tools.core.sensor_data`.
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual

# pyre-fixme[21]: Could not find name `StreamId` in `projectaria_tools.core.stream_id`.
from projectaria_tools.core.stream_id import StreamId  # @manual


try:
    data_path = Path(
        # pyre-fixme[6]: For 1st argument expected `Union[PathLike[str], str]` but
        #  got `Traversable`.
        importlib.resources.files(__package__).joinpath(
            "test_data/",
        )
    )

except ImportError:
    data_path = Path(__file__).parent

sequence_path = data_path / "data_sample/Aria/P0003_c701bd11"
box2d_objects_filepath = str(sequence_path / "box2d_objects.csv")


class TestObjectBox2dDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_aria_recording(self) -> None:
        self.assertTrue(os.path.exists(box2d_objects_filepath))
        provider = load_box2d_trajectory_from_csv(filename=box2d_objects_filepath)
        # pyre-fixme[16]: Optional type has no attribute `object_uids`.
        object_uids = provider.object_uids
        self.assertEqual(len(object_uids), 6)

        # pyre-fixme[16]: Optional type has no attribute `stream_ids`.
        stream_ids = provider.stream_ids
        self.assertEqual(len(stream_ids), 3)

        self.assertTrue(StreamId("214-1") in stream_ids)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        for stream_id in stream_ids:
            # pyre-fixme[16]: Optional type has no attribute `get_timestamp_ns_list`.
            timestamp_ns_list = provider.get_timestamp_ns_list(stream_id=stream_id)
            self.assertIsNotNone(timestamp_ns_list)
            self.assertGreater(len(timestamp_ns_list), 0)

            query_timestamp_ns = timestamp_ns_list[len(timestamp_ns_list) // 2]

            # pyre-fixme[16]: Optional type has no attribute `get_bbox_at_timestamp`.
            box2d_collection_with_dt = provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=query_timestamp_ns,
                # pyre-fixme[16]: Module `sensor_data` has no attribute
                #  `TimeQueryOptions`.
                time_query_options=TimeQueryOptions.CLOSEST,
                # pyre-fixme[16]: Module `sensor_data` has no attribute `TimeDomain`.
                time_domain=TimeDomain.TIME_CODE,
            )

            self.assertIsNotNone(box2d_collection_with_dt)
            self.assertIsNotNone(box2d_collection_with_dt.box2d_collection)
            object_uids_at_query_timestamp = (
                box2d_collection_with_dt.box2d_collection.object_uid_list
            )
            self.assertGreater(len(object_uids_at_query_timestamp), 0)

            # pyre-fixme[16]: Optional type has no attribute `get_data_statistics`.
            data_statistics = provider.get_data_statistics()
            print(f"data_statistics: {data_statistics}")
            self.assertEqual(len(data_statistics["num_frames"]), 3)
            self.assertEqual(data_statistics["num_frames"]["214-1"], 34)
            self.assertEqual(data_statistics["num_objects"], 6)
            self.assertEqual(len(data_statistics["stream_ids"]), 3)
            self.assertEqual(len(data_statistics["object_uids"]), 6)
