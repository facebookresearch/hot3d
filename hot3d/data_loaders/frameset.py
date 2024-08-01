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

import bisect
from typing import Dict, List, Optional


def find_closest(sorted_input_list: List[int], target: int) -> int:
    """
    Find the closest value in a sorted list to a target value
    """
    index = bisect.bisect_left(sorted_input_list, target)
    if index == 0:
        closest = sorted_input_list[0]
    elif index == len(sorted_input_list):
        closest = sorted_input_list[-1]
    else:
        before = sorted_input_list[index - 1]
        after = sorted_input_list[index]
        if abs(target - after) > abs(target - before):
            closest = before
        else:
            closest = after
    return closest


def compute_frameset_for_timestamp(
    stream_timestamps_sorted: Dict[str, List[int]],
    target_timestamp: int,
    frameset_acceptable_time_diff: int,
) -> Dict[str, Optional[int]]:
    """
    Compute the frameset for a given timestamp.
    The frameset consists of the closest timestamps for each stream that are within the acceptable time difference.
    Args:
        stream_timestamps_sorted: A dictionary containing lists of sorted timestamps indexed by str(StreamId).
        target_timestamp: The target timestamp to compute the frameset for.
        frameset_acceptable_time_diff: The maximum difference between the target timestamp and the closest timestamp in each stream.
    Returns:
        A dictionary of str(StreamId) to timestamps which defines a frameset at the target timestamp.
    """

    stream_id_strs = list(stream_timestamps_sorted.keys())
    frameset = {}
    for stream_id_str in stream_id_strs:
        closest_timestamp = find_closest(
            sorted_input_list=stream_timestamps_sorted[stream_id_str],
            target=target_timestamp,
        )
        if abs(closest_timestamp - target_timestamp) < frameset_acceptable_time_diff:
            frameset[stream_id_str] = closest_timestamp
        else:
            frameset[stream_id_str] = None

    return frameset
