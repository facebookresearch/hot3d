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


def load_json(json_filepath):
    with open(json_filepath, "r") as fp:
        return json.loads(fp.read())


def write_json(payload, json_filepath):
    with open(json_filepath, "w") as fp:
        json.dump(payload, fp, indent=4, sort_keys=True)
