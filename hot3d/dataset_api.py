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

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data_loaders.AriaDataProvider import AriaDataProvider
from data_loaders.HandDataProvider import HandDataProvider
from data_loaders.headsets import Headset
from data_loaders.io_utils import load_json

from data_loaders.loader_device_poses import (
    HeadsetPose3DWithDt,
    load_headset_pose_provider_from_csv,
)
from data_loaders.loader_hand_poses import HandPose, load_hand_poses
from data_loaders.loader_object_library import ObjectLibrary

from data_loaders.loader_object_poses import (
    load_pose_provider_from_csv,
    Pose3DCollectionWithDt,
)
from data_loaders.PathProvider import Hot3DDataPathProvider
from data_loaders.pose_utils import query_left_right

from projectaria_tools.core.mps import (  # @manual
    get_eyegaze_point_at_depth,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual


# 3D assets
# - object_uid

# 3D transform
# Aria Device to Optitrack

# Generic idea around the DataProvider is that is allow to initialize the data reading
# and offer a generic interface to retrieve timestamp data by TYPE (Image, Object, Hand, etc.)


class Hot3DDataProvider:
    """
    High Level interface to retrieve and use data from the hot3d dataset
    """

    def __init__(self, sequence_folder: str, object_library: ObjectLibrary) -> None:
        """
        INIT_DOC_STRING
        """
        # Will read all required metadata
        # Hands
        # Objects
        # Device type, ...
        self.path_provider = Hot3DDataPathProvider.fromRecordingFolder(
            recording_instance_folderpath=sequence_folder
        )

        if not self.path_provider.is_valid():
            raise RuntimeError(
                "Invalid hot3d path.. Not all expected data are present."
            )

        self._dynamic_objects_provider = load_pose_provider_from_csv(
            self.path_provider.dynamic_objects_filepath
        )

        self._device_pose_provider = load_headset_pose_provider_from_csv(
            self.path_provider.headset_trajectory_filepath
        )

        self._object_library: ObjectLibrary = object_library

        self._hand_data_provider = HandDataProvider(
            self.path_provider.hand_pose_trajectory_filepath,
            self.path_provider.hand_user_profile_filepath,
        )

        if self.get_device_type() == Headset.Aria:
            # Aria specifics

            # VRS data provider
            self._device_data_provider = AriaDataProvider(
                self.path_provider.vrs_filepath,
                self.path_provider.mps_folderpath,
            )

        else:
            raise RuntimeError(f"Unsupported device type {self.get_device_type()}")

    def get_data_statistics(self) -> Dict[str, Any]:
        statistics_dict = {}
        statistics_dict["dynamic_objects"] = (
            self._dynamic_objects_provider.get_data_statistics()
        )
        return statistics_dict

    @property
    def object_library(self) -> ObjectLibrary:
        """
        Return the object library used for initializing the Hot3DDataProvider
        """
        return self._object_library

    @property
    def device_data_provider(self):
        """
        Return the device data provider
        """
        return self._device_data_provider

    @property
    def hand_data_provider(self):
        """
        Return the hand data provider
        """
        return self._hand_data_provider

    def get_object_poses(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ) -> Optional[Pose3DCollectionWithDt]:
        """
        Return the list of object poses at the given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        return self._dynamic_objects_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=time_query_options,
            time_domain=time_domain,
        )

    def get_device_pose(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ) -> Optional[HeadsetPose3DWithDt]:
        """
        Return the list of headset poses at the given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        return self._device_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=time_query_options,
            time_domain=time_domain,
        )

    def get_device_type(self) -> Headset:
        """
        Return the type of device used for recording (e.g. Quest3, Aria, etc.)
        """
        return Headset[self.get_sequence_metadata()["headset"]]

    def get_sequence_metadata(self) -> Dict:
        """
        Return the metadata associated with the sequence
        """
        metadata_json = load_json(self.path_provider.scene_metadata_filepath)

        return metadata_json
