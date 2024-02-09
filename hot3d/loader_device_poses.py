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

import csv

from typing import Dict

import numpy as np
from projectaria_tools.core.sophus import SE3


def loadDevicePoses(filename: str) -> Dict[str, SE3]:

    device_pose_per_timestamp = {}
    device_count = set()
    # Open the CSV file for reading
    with open(filename, "r") as f:
        reader = csv.reader(f)

        # Read the header row
        header = next(reader)

        # Ensure we have the desired columns
        DESIRED_COLUMNS = [
            "object_uid",
            "timestamp[ns]",
            "t_wo_x[m]",
            "t_wo_y[m]",
            "t_wo_z[m]",
            "q_wo_w",
            "q_wo_x",
            "q_wo_y",
            "q_wo_z",
        ]
        for column in header:
            if column not in DESIRED_COLUMNS:
                raise ValueError(
                    "Invalid Object CSV format. Expected columns are: {}".format(
                        ", ".join(DESIRED_COLUMNS)
                    )
                )

        # Read the rest of the rows in the CSV file
        for row in reader:
            translation = [
                row[header.index("t_wo_x[m]")],
                row[header.index("t_wo_y[m]")],
                row[header.index("t_wo_z[m]")],
            ]
            quaternion_xyz = [
                row[header.index("q_wo_x")],
                row[header.index("q_wo_y")],
                row[header.index("q_wo_z")],
            ]
            quaternion_w = row[header.index("q_wo_w")]
            timestamp = row[header.index("timestamp[ns]")]
            object_uid = row[header.index("object_uid")]

            object_pose = SE3.from_quat_and_translation(
                float(quaternion_w),
                np.array([float(o) for o in quaternion_xyz]),
                np.array([float(o) for o in translation]),
            )[0]

            if timestamp not in device_pose_per_timestamp:
                device_pose_per_timestamp[timestamp] = {}
            device_pose_per_timestamp[timestamp] = object_pose
            device_count.add(object_uid)

    assert len(device_count) == 1  # Only one device should be tracked
    print(
        f"Objects data loading stats: \n\
        \tNumber of timestamps: {len(device_pose_per_timestamp.keys())}\n\
        \tNumber of objects: {len(device_count)}"
    )
    return device_pose_per_timestamp
