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

from enum import Enum

from typing import Dict, List

import numpy as np

from data_loaders.loader_device_poses import load_device_poses
from data_loaders.loader_object_poses import load_dynamic_objects
from data_loaders.PathProvider import Hot3DDataPathProvider

from projectaria_tools.core.sensor_data import TimeDomain


class DeviceType(Enum):
    QUEST3 = 1
    ARIA = 2


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

    def __init__(self, sequence_folder: str) -> None:
        """
        INIT_DOC_STRING
        """
        # Will read all required metadata
        # Hands
        # Objects
        # Device type, ...
        self.path_provider = Hot3DDataPathProvider(sequence_folder)
        if not self.path_provider.is_valid():
            raise RuntimeError(
                "Invalid hot3d path.. Not all expected data are present."
            )

        self._dynamic_objects = load_dynamic_objects(
            self.path_provider.dynamic_objects_file
        )
        self._device_poses = load_device_poses(self.path_provider.device_poses_file)
        self._timestamp_list = self._dynamic_objects.keys()

    def get_timestamps(
        self, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> List[int]:
        """
        Returns the list of device timestamp for the specified sequence
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        return self._timestamp_list  # default is timecode time domain

    def get_image(self, timestamp_ns: int) -> np.ndarray:
        """
        Return the RGB image at the given timestamp
        """
        # For Aria we would have a streamId
        # Note Aria files would be cut from EyeGaze images and Audio (GPS, ...)

    def get_camera_calibration(self):
        """
        Return the camera calibration of the device of the sequence
        """
        # Ideally we have the same calibration for both devices

    def get_object_poses(
        self, timestamp_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ):
        """
        Return the list of object poses at the given timestamp
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        # Interpolated
        # Closed GT
        # Before
        # After
        # Export the TimeDelta, or the corresponding timestamp of the return data?

        # Visibility
        # Exclusion list (is_valid)

        # If something not supported -> return Exception
        if timestamp_ns in self._dynamic_objects:
            return self._dynamic_objects[timestamp_ns]
        else:
            return None

    def get_hand_poses(self, timestamp_ns: int):
        """
        Return the list of hand poses at the given timestamp
        """

    def get_device_pose(
        self, timestamp_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ):
        """
        Return the device pose at the given timestamp
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        # BBox 2D, 3D
        # OptiTrack
        # MPS (would come from the MPSDataProvider)
        if timestamp_ns in self._device_poses:
            return self._device_poses[timestamp_ns]
        else:
            return None

    def get_device_type(self) -> DeviceType:
        """
        Return the type of device used for recording (e.g. Quest3, Aria, etc.)
        """
        # => ENUM
        return DeviceType.ARIA

    def get_sequence_metadata(self) -> Dict:
        """
        Return the metadata associated with the sequence
        """
        pass

        # High level functions that are considered in the sequence
        # Details on the scenario, hardware used and sequences ...
        # Device Ids
        # Number of objects
        # Participant Ids
        # Length of the sequence

    # Todo
    # Need a mechanism to add filtering (visible in camera frustum , etc.)
    # Interface to retrieve MPS data if available (see the getDeviceType())
