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

from typing import List

import numpy as np

from projectaria_tools.core import calibration, data_provider  # @manual

from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
    distort_by_calibration,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual


class AriaDataProvider:

    def __init__(self, vrs_filepath: str) -> None:
        self._vrs_data_provider = data_provider.create_vrs_data_provider(vrs_filepath)

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

    def get_device_calibration(self):
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
