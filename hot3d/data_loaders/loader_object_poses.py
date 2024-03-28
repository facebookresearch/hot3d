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
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Set

import numpy as np
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual

from .constants import POSE_DATA_CSV_COLUMNS
from .loader_poses_utils import check_csv_columns
from .pose_utils import lookup_timestamp


@dataclass
class Pose3D:
    """
    Class to store pose of a single entity (object/headset)
    """

    T_world_object: Optional[SE3] = None


@dataclass
class Pose3DCollection:
    """
    Class to store the poses for a given timestamp
    """

    timestamp_ns: int
    poses: Dict[str, Pose3D]

    @property
    def object_uid_list(self) -> Set[str]:
        return set(self.poses.keys())


Pose3DTrajectory = Dict[int, Pose3DCollection]


@dataclass
class Pose3DCollectionWithDt:
    pose3d_collection: Pose3DCollection
    time_delta_ns: int


class Pose3DProvider(object):
    def __init__(self, pose3d_trajectory: Pose3DTrajectory):
        self._pose3d_trajectory: Pose3DTrajectory = pose3d_trajectory
        self._sorted_timestamp_ns_list: List[int] = sorted(
            self._pose3d_trajectory.keys()
        )
        self._object_uids_with_poses: Set = {
            x for v in self._pose3d_trajectory.values() for x in v.object_uid_list
        }

    @property
    def timestamp_ns_list(self) -> List[int]:
        return self._sorted_timestamp_ns_list

    @property
    def object_uids_with_poses(self) -> Set[str]:
        return set(self._object_uids_with_poses)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Returns the stats of the trajectory
        """
        stats = {}
        stats["num_frames"] = len(self._sorted_timestamp_ns_list)
        stats["num_objects"] = len(self._object_uids_with_poses)
        stats["object_uids"] = [str(x) for x in self._object_uids_with_poses]
        return stats

    def get_pose_at_timestamp(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain,
    ) -> Optional[Pose3DCollectionWithDt]:
        """
        Return the list of poses at the given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        pose3d_collection, time_delta_ns = lookup_timestamp(
            time_indexed_dict=self._pose3d_trajectory,
            sorted_timestamp_list=self._sorted_timestamp_ns_list,
            query_timestamp=timestamp_ns,
            time_query_options=time_query_options,
        )

        if pose3d_collection is None or time_delta_ns is None:
            return None
        else:
            return Pose3DCollectionWithDt(
                pose3d_collection=pose3d_collection, time_delta_ns=time_delta_ns
            )


def load_pose_trajectory_from_csv(filename: str) -> Pose3DTrajectory:
    """Load Dynamic Objects meta data from a CSV file.

    Keyword arguments:
    filename -- the csv file i.e. sequence_folder + "/dynamic_objects.csv"
    """

    pose3d_trajectory: Pose3DTrajectory = {}
    # Open the CSV file for reading
    with open(filename, "r") as f:
        reader = csv.reader(f)

        # Read the header row
        header = next(reader)

        # Ensure we have the desired columns
        check_csv_columns(header, POSE_DATA_CSV_COLUMNS)

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
            timestamp_ns = int(row[header.index("timestamp[ns]")])
            object_uid = str(row[header.index("object_uid")])

            T_world_object = SE3.from_quat_and_translation(
                float(quaternion_w),
                np.array([float(o) for o in quaternion_xyz]),
                np.array([float(o) for o in translation]),
            )[0]

            pose3d = Pose3D(T_world_object=T_world_object)

            if timestamp_ns not in pose3d_trajectory:
                pose3d_trajectory[timestamp_ns] = Pose3DCollection(
                    timestamp_ns=timestamp_ns, poses={}
                )

            pose3d_trajectory[timestamp_ns].poses[object_uid] = pose3d

    return pose3d_trajectory


def load_pose_provider_from_csv(filename: str) -> Pose3DProvider:
    """
    Load pose_provider from csv
    """
    return Pose3DProvider(pose3d_trajectory=load_pose_trajectory_from_csv(filename))
