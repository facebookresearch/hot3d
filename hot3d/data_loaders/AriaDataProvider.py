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

import os
from typing import List

import numpy as np

from projectaria_tools.core import calibration, data_provider  # @manual
from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
    DeviceCalibration,
    distort_by_calibration,
)
from projectaria_tools.core.mps import (  # @manual
    get_eyegaze_point_at_depth,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual


class AriaDataProvider:

    def __init__(self, vrs_filepath: str, mps_folder_path: str) -> None:
        self._vrs_data_provider = data_provider.create_vrs_data_provider(vrs_filepath)

        # MPS data provider
        if os.path.exists(mps_folder_path):
            mps_data_paths_provider = MpsDataPathsProvider(mps_folder_path)
            mps_data_paths = mps_data_paths_provider.get_data_paths()
            self._mps_data_provider = MpsDataProvider(mps_data_paths)
            print(mps_data_paths)

    def get_image_stream_ids(self) -> List[StreamId]:
        # retrieve all streams ids and filter the one that are image based
        stream_ids = self._vrs_data_provider.get_all_streams()
        image_stream_ids = [
            p
            for p in stream_ids
            if self._vrs_data_provider.get_label_from_stream_id(p).startswith("camera-")
        ]
        return image_stream_ids

    def get_sequence_timestamps(self) -> List[int]:
        """
        Returns the list of "time code" timestamp for the sequence
        """
        stream_id = StreamId(
            "214-1"
        )  # use rgb as default stream used for timestamp reference
        return self._vrs_data_provider.get_timestamps_ns(
            stream_id, TimeDomain.TIME_CODE
        )

    def get_image_stream_label(self, stream_id: StreamId) -> str:
        return self._vrs_data_provider.get_label_from_stream_id(stream_id)

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> np.ndarray:
        image = self._vrs_data_provider.get_image_data_by_time_ns(
            stream_id,
            timestamp_ns,
            TimeDomain.TIME_CODE,
            TimeQueryOptions.CLOSEST,
        )
        return image[0].to_numpy_array()

    def get_undistorted_image(
        self, timestamp_ns: int, stream_id: StreamId
    ) -> np.ndarray:
        image = self.get_image(timestamp_ns, stream_id)

        [T_device_camera, camera_calibration] = self.get_camera_calibration(stream_id)
        focal_lengths = camera_calibration.get_focal_lengths()
        image_size = camera_calibration.get_image_size()
        pinhole_calibration = calibration.get_linear_camera_calibration(
            image_size[0], image_size[1], focal_lengths[0]
        )

        # Compute the actual undistorted image
        undistorted_image = distort_by_calibration(
            image, pinhole_calibration, camera_calibration
        )

        return undistorted_image

    def get_device_calibration(self) -> DeviceCalibration:
        """
        Return the device calibration (factory calibration of all sensors)
        """
        return self._vrs_data_provider.get_device_calibration()

    def get_camera_calibration(
        self, stream_id: StreamId
    ) -> tuple[SE3, CameraCalibration]:
        """
        Return the camera calibration of the device of the sequence as [Extrinsics, Intrinsics]
        """
        device_calibration = self.get_device_calibration()
        rgb_stream_label = self._vrs_data_provider.get_label_from_stream_id(stream_id)
        camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
        T_device_camera = camera_calibration.get_transform_device_camera()
        return [T_device_camera, camera_calibration]

    def _timestamp_convert(
        self, timestamp: int, time_domain_in: TimeDomain, time_domain_out: TimeDomain
    ) -> int:
        """
        Returns the converted timestamp between two domains (TimeCode <-> Aria DeviceTime)
        """
        if (
            self._vrs_data_provider
            and time_domain_in == TimeDomain.TIME_CODE
            and time_domain_out == TimeDomain.DEVICE_TIME
        ):
            # Map to corresponding timestamp
            device_timestamp_ns = (
                self._vrs_data_provider.convert_from_timecode_to_device_time_ns(
                    timestamp
                )
            )
            return device_timestamp_ns
        return None

    ###
    # Add MPS data specifics
    ###

    def get_point_cloud(self) -> np.ndarray:
        """
        Return the point cloud of the scene
        """
        if self._mps_data_provider.has_semidense_point_cloud():
            point_cloud_data = self._mps_data_provider.get_semidense_point_cloud()
            # Point cloud filtering is left to the user
            return point_cloud_data

        return None

    def get_eye_gaze_in_camera(
        self,
        stream_id: StreamId,
        timestamp_ns: int,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
        raw_image=False,  # If False we project for corresponding pinhole camera model
        depth_m: float = 1.0,
    ):
        """
        Return the eye_gaze at the given timestamp projected in the given stream for the given time_domain
        """
        # Map to corresponding timestamp
        if time_domain == TimeDomain.TIME_CODE:
            device_timestamp_ns = self._timestamp_convert(
                timestamp_ns, TimeDomain.TIME_CODE, TimeDomain.DEVICE_TIME
            )
        elif time_domain == TimeDomain.DEVICE_TIME:
            device_timestamp_ns = timestamp_ns
        else:
            raise ValueError("Unsupported time domain")

        if device_timestamp_ns:
            eye_gaze = self._mps_data_provider.get_general_eyegaze(device_timestamp_ns)
            if eye_gaze:
                # Compute eye_gaze vector at depth_m and project it in the image
                depth_m = 1.0
                gaze_vector_in_cpf = get_eyegaze_point_at_depth(
                    eye_gaze.yaw, eye_gaze.pitch, depth_m
                )
                [T_device_camera, camera_calibration] = self.get_camera_calibration(
                    stream_id
                )
                focal_lengths = camera_calibration.get_focal_lengths()
                image_size = camera_calibration.get_image_size()
                image_calibration = (
                    camera_calibration
                    if raw_image
                    else calibration.get_linear_camera_calibration(
                        image_size[0], image_size[1], focal_lengths[0]
                    )
                )
                device_calibration = self.get_device_calibration()
                T_device_CPF = device_calibration.get_transform_device_cpf()
                gaze_center_in_camera = (
                    T_device_camera.inverse() @ T_device_CPF @ gaze_vector_in_cpf
                )
                gaze_projection = image_calibration.project(gaze_center_in_camera)
                return gaze_projection
        return None
