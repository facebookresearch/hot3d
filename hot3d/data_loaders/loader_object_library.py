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

import json

from typing import Dict


def load_object_instance(filename: str) -> Dict:
    """Load Objects library meta data.

    Keyword arguments:
    filename -- the json file i.e. sequence_folder + "/instance.csv"
    """

    object_instance = {}

    # Open the JSON file for reading
    with open(filename, "r") as f:

        # Parse the JSON file
        object_instance = json.load(f)

    print(object_instance)
    print(
        f"Object instance data loading stats: \n\
        \tNumber of instances: {len(object_instance.keys())}"
    )
    return object_instance
