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

from data_loaders.frameset import compute_frameset_for_timestamp


class TestFrameSet(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_compute_framesets(self) -> None:
        stream_timestamps = {
            "s0": sorted([100, 200, 300, 400, 500, 600, 702]),
            "s1": sorted([103, 202, 298, 400, 450, 604, 710]),
            "s2": sorted([101, 198, 299, 400, 452, 502, 718]),
        }

        expected_framesets = {
            -1: {"s0": None, "s1": None, "s2": None},
            0: {"s0": None, "s1": None, "s2": None},
            50: {"s0": None, "s1": None, "s2": None},
            10000: {"s0": None, "s1": None, "s2": None},
            100: {"s0": 100, "s1": 103, "s2": 101},
            202: {"s0": 200, "s1": 202, "s2": 198},
            299: {"s0": 300, "s1": 298, "s2": 299},
            400: {"s0": 400, "s1": 400, "s2": 400},
            450: {"s0": None, "s1": 450, "s2": 452},
            500: {"s0": 500, "s1": None, "s2": 502},
            600: {"s0": 600, "s1": 604, "s2": None},
            702: {"s0": 702, "s1": 710, "s2": None},
            710: {"s0": 702, "s1": 710, "s2": 718},
            718: {"s0": None, "s1": 710, "s2": 718},
        }

        for target_timestamp in expected_framesets.keys():
            out_frameset = compute_frameset_for_timestamp(
                stream_timestamps_sorted=stream_timestamps,
                target_timestamp=target_timestamp,
                frameset_acceptable_time_diff=10,
            )
            expected_frameset = expected_framesets[target_timestamp]

            self.assertEqual(
                sorted(out_frameset.keys()),
                sorted(expected_frameset.keys()),
            )

            for stream_id_str in expected_frameset.keys():
                self.assertEqual(
                    out_frameset[stream_id_str],
                    expected_frameset[stream_id_str],
                )
