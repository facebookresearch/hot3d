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
from typing import Any, Dict, List, Optional, Tuple

from projectaria_tools.core.sensor_data import TimeQueryOptions  # @manual


def query_left_right(ordered_timestamps: List, query_timestamp: int):
    """
    Return the left and right timestamp of the query timestamp by using bisection.
    Assumption: timestamps are monotonically sorted in the ordered_timestamps List
    """
    index_less_than_query = bisect.bisect_left(ordered_timestamps, query_timestamp) - 1
    index_greater_than_query = index_less_than_query + 1
    lower_timestamp = ordered_timestamps[index_less_than_query]
    upper_timestamp = ordered_timestamps[index_greater_than_query]

    alpha = (query_timestamp - lower_timestamp) / (upper_timestamp - lower_timestamp)

    return lower_timestamp, upper_timestamp, alpha


def lookup_timestamp(
    time_indexed_dict: Dict[int, Any],
    sorted_timestamp_list: Optional[List[int]],
    query_timestamp: int,
    time_query_options: TimeQueryOptions,
) -> Tuple[Any, int]:
    """
    Lookup the object corresponding to query_timestamp based on the time_query_options.
    time_indexed_dict: a dictionary of timestamp to object
    sorted_timestamp_list: a precomputed list of sorted timestamps in time_indexed_dict. If None, it will be computed from time_indexed_dict.
    query_timestamp: the timestamp to query
    time_query_options: the time query options to use (CLOSEST, BEFORE, AFTER)
    """

    if sorted_timestamp_list is None:
        sorted_timestamp_list = sorted(time_indexed_dict.keys())

    obj = None
    time_delta_ns = None

    if query_timestamp in time_indexed_dict:
        obj = time_indexed_dict[query_timestamp]
        time_delta_ns = 0

    left_frame_tsns, right_frame_tsns, alpha = query_left_right(
        ordered_timestamps=sorted_timestamp_list,
        query_timestamp=query_timestamp,
    )
    if time_query_options == TimeQueryOptions.BEFORE:
        obj = time_indexed_dict[left_frame_tsns]
        time_delta_ns = query_timestamp - left_frame_tsns
    elif time_query_options == TimeQueryOptions.AFTER:
        obj = time_indexed_dict[right_frame_tsns]
        time_delta_ns = query_timestamp - right_frame_tsns
    elif time_query_options == TimeQueryOptions.CLOSEST:
        if abs(query_timestamp - left_frame_tsns) > abs(
            query_timestamp - right_frame_tsns
        ):
            obj = time_indexed_dict[right_frame_tsns]
            time_delta_ns = query_timestamp - right_frame_tsns
        else:
            obj = time_indexed_dict[left_frame_tsns]
            time_delta_ns = query_timestamp - left_frame_tsns
    return obj, time_delta_ns
