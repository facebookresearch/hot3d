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
from PIL import Image
from projectaria_tools.core.calibration import CameraCalibration  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual
from pyvrs import ImageConversion, SyncVRSReader


class QuestDataProvider:

    def __init__(self, vrs_filepath: str) -> None:
        self._vrs_reader = SyncVRSReader(vrs_filepath)
        # Configure Image conversion
        self._vrs_reader.set_image_conversion(ImageConversion.NORMALIZE)
        self._vrs_reader.set_stream_type_image_conversion(
            8010, ImageConversion.NORMALIZE_GREY8
        )

    def get_image_stream_ids(self) -> List[StreamId]:
        # retrieve all streams ids and filter the one that are image based
        image_stream_ids = []
        for stream_id in self._vrs_reader.stream_ids:
            if self._vrs_reader.might_contain_images(stream_id):
                image_stream_ids.append(StreamId(stream_id))
        print(image_stream_ids)
        return image_stream_ids

    def get_sequence_timestamps(self) -> List[int]:
        """
        Returns the list of "time code" timestamp for the sequence
        """
        timestamps = self._vrs_reader.get_timestamp_list()
        # convert timestamp from float to int in ns
        return sorted({int(x * 1e9) for x in timestamps})

    def get_image_stream_label(self, stream_id: StreamId) -> str:
        return str(stream_id)

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> np.ndarray:
        record = self._vrs_reader.read_record_by_time(
            stream_id=self.get_image_stream_label(stream_id),
            timestamp=timestamp_ns / 1e9,
        )

        if record.record_type == "data":
            grey8 = Image.fromarray(record.image_blocks[0]).convert("RGB")
            return np.array(grey8)
        return None

    def get_undistorted_image(
        self, timestamp_ns: int, stream_id: StreamId
    ) -> np.ndarray:
        # TODO: will be added when calibration data will be accessible
        return self.get_image(timestamp_ns, stream_id)

    def get_camera_calibration(
        self, stream_id: StreamId
    ) -> tuple[SE3, CameraCalibration]:
        """
        Return the camera calibration of the device of the sequence as [Extrinsics, Intrinsics]
        """
        # TODO: will be added when calibration data will be accessible
        return None
