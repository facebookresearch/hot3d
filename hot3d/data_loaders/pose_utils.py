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
from typing import List


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
