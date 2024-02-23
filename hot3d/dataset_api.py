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
from data_loaders.loader_hand_poses import load_hand_poses
from data_loaders.loader_object_library import load_object_instance
from data_loaders.loader_object_poses import load_dynamic_objects
from data_loaders.PathProvider import Hot3DDataPathProvider
from data_loaders.pose_utils import query_left_right

from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.calibration import DeviceCalibration, distort_by_calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId


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
        self._object_instance_mapping = load_object_instance(
            self.path_provider.object_library_instances_file
        )
        self._timestamp_list = self._dynamic_objects.keys()

        self._vrs_data_provider = None
        self._vrs_data_provider = data_provider.create_vrs_data_provider(
            self.path_provider.vrs_file
        )

        self._hand_poses = load_hand_poses(self.path_provider.hand_poses_file)

        # rgb_stream_id = StreamId("214-1")
        # timecode_vec = self._vrs_data_provider.get_timestamps_ns(
        #     rgb_stream_id, TimeDomain.TIME_CODE
        # )
        # print(timecode_vec)

    def get_valid_recording_range(
        self, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> tuple[int, int]:
        """
        Return the valid recording range corresponding to the Device sequence
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        device_start_timestamp = self._vrs_data_provider.get_first_time_ns_all_streams(
            TimeDomain.DEVICE_TIME
        )
        device_end_timestamp = self._vrs_data_provider.get_last_time_ns_all_streams(
            TimeDomain.DEVICE_TIME
        )

        device_start_timestamp = (
            self._vrs_data_provider.convert_from_device_time_to_timecode_ns(
                device_start_timestamp
            )
        )
        device_end_timestamp = (
            self._vrs_data_provider.convert_from_device_time_to_timecode_ns(
                device_end_timestamp
            )
        )

        return [device_start_timestamp, device_end_timestamp]

    def get_timestamps(
        self, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> List[int]:
        """
        Returns the list of device timestamp for the specified sequence
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        return self._timestamp_list  # default is TimeDomain.TIME_CODE

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> np.ndarray:
        """
        Return the image corresponding to the requested timestamp and streamId
        """
        if self._vrs_data_provider:
            # Map to corresponding timestamp
            device_timestamp_ns = (
                self._vrs_data_provider.convert_from_timecode_to_device_time_ns(
                    timestamp_ns
                )
            )
            # Get corresponding image
            image = self._vrs_data_provider.get_image_data_by_time_ns(
                stream_id,
                device_timestamp_ns,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )
            return image[0].to_numpy_array()

        return None

    def get_undistorted_image(
        self, timestamp_ns: int, stream_id: StreamId
    ) -> np.ndarray:
        """
        Return the undistorted image corresponding to the requested timestamp and streamId
        """
        if self._vrs_data_provider:
            # Map to corresponding timestamp
            device_timestamp_ns = (
                self._vrs_data_provider.convert_from_timecode_to_device_time_ns(
                    timestamp_ns
                )
            )
            image = self._vrs_data_provider.get_image_data_by_time_ns(
                stream_id,
                device_timestamp_ns,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )

            [T_device_camera, camera_calibration] = self.get_camera_calibration(
                stream_id
            )
            focal_lengths = camera_calibration.get_focal_lengths()
            image_size = camera_calibration.get_image_size()
            pinhole_calib = calibration.get_linear_camera_calibration(
                image_size[0], image_size[1], focal_lengths[0]
            )

            # Perform the actual undistortion
            undistorted_image = distort_by_calibration(
                image[0].to_numpy_array(), pinhole_calib, camera_calibration
            )

            return undistorted_image
        return None

    def get_camera_calibration(
        self, stream_id: StreamId
    ) -> tuple[SE3, DeviceCalibration]:
        """
        Return the camera calibration of the device of the sequence
        """
        # Ideally we have the same calibration for both devices

        if self.get_device_type() == DeviceType.ARIA:
            # Should we return [EXTRINSICS, INTRINSICS]
            rgb_stream_label = self._vrs_data_provider.get_label_from_stream_id(
                stream_id
            )
            device_calibration = self._vrs_data_provider.get_device_calibration()
            camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
            T_device_camera = camera_calibration.get_transform_device_camera()
            return [T_device_camera, camera_calibration]
        else:
            raise ValueError("TODO Implement for Quest device.")

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
            # We use bisection to find the closest timestamp
            lower, upper, alpha = query_left_right(
                list(self._dynamic_objects.keys()), timestamp_ns
            )
            return self._dynamic_objects[lower]

        return None

    def get_object_instance_name(self, instance_id: str) -> str:
        """
        Return the "name" of the object instance from its unique instance id
        """
        if instance_id not in self._object_instance_mapping.keys():
            raise ValueError("Instance id {} not found".format(instance_id))
        return self._object_instance_mapping[instance_id]["instance_name"]

    def get_hand_poses(self, timestamp_ns: int):
        """
        Return the list of hand poses at the given timestamp
        """
        if timestamp_ns in self._hand_poses:
            return self._hand_poses[timestamp_ns]
        else:
            # We use bisection to find the closest timestamp
            lower, upper, alpha = query_left_right(
                list(self._hand_poses.keys()), timestamp_ns
            )
            return self._hand_poses[lower]

        return None

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
            # We use bisection to find the closest timestamp
            lower, upper, alpha = query_left_right(
                list(self._device_poses.keys()), timestamp_ns
            )
            return self._device_poses[lower]

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
