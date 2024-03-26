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
from enum import Enum
from pathlib import Path

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data_loaders.loader_device_poses import load_device_poses
from data_loaders.loader_hand_poses import HandPose, load_hand_poses
from data_loaders.loader_object_library import ObjectLibrary

from data_loaders.loader_object_poses import (
    load_pose_provider_from_csv,
    Pose3DCollectionWithDt,
)
from data_loaders.PathProvider import Hot3DDataPathProvider
from data_loaders.pose_utils import query_left_right

from projectaria_tools.core import calibration, data_provider  # @manual
from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
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

from UmeTrack.common.hand_skinning import skin_landmarks, skin_vertices
from UmeTrack.common.loader_handmodel import load_hand_model_from_file


def normalized(
    vecs: np.ndarray, axis: int = -1, add_const_to_denom: bool = True
) -> np.ndarray:
    denom = np.linalg.norm(vecs, axis=axis, keepdims=True)
    if add_const_to_denom:
        denom += 1e-5
    return vecs / denom


def get_triangular_mesh_normals(
    vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    norm = np.zeros_like(vertices)
    tris = vertices[triangles]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalized(n)
    norm[triangles[:, 0]] += n
    norm[triangles[:, 1]] += n
    norm[triangles[:, 2]] += n
    return normalized(norm)


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

        self._device_poses = load_device_poses(
            self.path_provider.headset_trajectory_filepath
        )

        self._object_library: ObjectLibrary = object_library

        self._vrs_data_provider = None
        self._vrs_data_provider = data_provider.create_vrs_data_provider(
            self.path_provider.vrs_filepath
        )

        self._hand_poses = load_hand_poses(
            self.path_provider.hand_pose_trajectory_filepath
        )
        # Hand profile
        self._hand_model = load_hand_model_from_file(
            self.path_provider.hand_user_profile_filepath
        )

        # Aria specifics
        mps_possible_path = self.path_provider.mps_folderpath
        if os.path.exists(mps_possible_path):
            mps_data_paths_provider = MpsDataPathsProvider(mps_possible_path)
            mps_data_paths = mps_data_paths_provider.get_data_paths()
            self.mps_data_provider = MpsDataProvider(mps_data_paths)
            print(mps_data_paths)

    def get_data_statistics(self) -> Dict[str, Any]:
        statistics_dict = {}
        statistics_dict["dynamic_objects"] = (
            self._dynamic_objects_provider.get_data_statistics()
        )
        return statistics_dict

    @property
    def object_library(self):
        """
        Return the object library used for initializing the Hot3DDataProvider
        """
        return self._object_library

    def get_valid_recording_range(
        self, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> tuple[int, int]:
        """
        Return the valid recording range corresponding to the Device sequence
        """
        if time_domain != TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        device_start_timestamp = self._vrs_data_provider.get_first_time_ns_all_streams(
            time_domain
        )
        device_end_timestamp = self._vrs_data_provider.get_last_time_ns_all_streams(
            time_domain
        )

        return [device_start_timestamp, device_end_timestamp]

    def get_sequence_timestamps(
        self, stream_id: StreamId, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> List[int]:
        """
        Returns the list of device timestamp for the specified StreamId
        """
        if self.get_device_type() == DeviceType.ARIA:
            return self._vrs_data_provider.get_timestamps_ns(stream_id, time_domain)
        else:
            return None

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

    def get_image_stream_ids(self) -> List[StreamId]:
        """
        Return the list of image stream ids
        """
        if self._vrs_data_provider:
            stream_ids = self._vrs_data_provider.get_all_streams()
            image_stream_ids = [
                p
                for p in stream_ids
                if self._vrs_data_provider.get_label_from_stream_id(p).startswith(
                    "camera-"
                )
            ]
            return image_stream_ids
        return None

    def get_image_stream_label(self, stream_id: StreamId) -> str:
        """
        Return the label of the image stream
        """
        if self._vrs_data_provider:
            return self._vrs_data_provider.get_label_from_stream_id(stream_id)
        return None

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> np.ndarray:
        """
        Return the image corresponding to the requested timestamp and streamId
        """
        if self._vrs_data_provider:
            # Map to corresponding timestamp
            device_timestamp_ns = self._timestamp_convert(
                timestamp_ns, TimeDomain.TIME_CODE, TimeDomain.DEVICE_TIME
            )

            if device_timestamp_ns:
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
            device_timestamp_ns = self._timestamp_convert(
                timestamp_ns, TimeDomain.TIME_CODE, TimeDomain.DEVICE_TIME
            )

            if device_timestamp_ns:
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

                # Compute the actual undistorted image
                undistorted_image = distort_by_calibration(
                    image[0].to_numpy_array(), pinhole_calib, camera_calibration
                )

                return undistorted_image
        return None

    def device_calibration(self):
        """
        Return the device calibration (factory calibration of all sensors)
        """
        if self.get_device_type() == DeviceType.ARIA:
            return self._vrs_data_provider.get_device_calibration()
        else:
            raise ValueError("TODO Implement for Quest device.")
            return None

    def get_camera_calibration(
        self, stream_id: StreamId
    ) -> tuple[SE3, CameraCalibration]:
        """
        Return the camera calibration of the device of the sequence as [Extrinsics, Intrinsics]
        """
        if self.get_device_type() == DeviceType.ARIA:

            device_calibration = self.device_calibration()
            rgb_stream_label = self._vrs_data_provider.get_label_from_stream_id(
                stream_id
            )
            camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
            T_device_camera = camera_calibration.get_transform_device_camera()
            return [T_device_camera, camera_calibration]
        else:
            raise ValueError("TODO Implement for Quest device.")

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

    def get_hand_poses(self, timestamp_ns: int) -> Optional[List[HandPose]]:
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

    def get_hand_mesh_vertices(self, hand_wrist_data: HandPose) -> torch.Tensor:
        """
        Return the hand mesh corresponding to given HandPose
        """
        if hand_wrist_data.hand_pose is not None:
            hand_wrist_pose_matrix = hand_wrist_data.hand_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == "1":
                hand_wrist_pose_tensor[:, 0] *= -1

            mesh_vertices = skin_vertices(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            # Rescale and translate the vertices
            scale = 1e-3
            mesh_vertices = mesh_vertices.mul(scale)
            translation = torch.Tensor(hand_wrist_data.hand_pose.translation()[0])
            return mesh_vertices + translation.expand_as(mesh_vertices)
        return None

    def get_hand_mesh_faces_and_normals(
        self, hand_wrist_data: HandPose
    ) -> Optional[tuple[np.array, np.array]]:
        """
        Return the hand mesh faces and normals
        """
        if self._hand_model is not None:
            hand_triangles = self._hand_model.mesh_triangles.int().numpy()
            vertices = self.get_hand_mesh_vertices(hand_wrist_data)
            normals = get_triangular_mesh_normals(
                vertices.float().numpy(), hand_triangles
            )
            return [hand_triangles, normals]
        else:
            return None

    def get_hand_landmarks(self, hand_wrist_data: HandPose) -> Optional[torch.Tensor]:
        """
        Return the hand joint landmarks corresponding to given HandPose
        See how to map the vertices together to represent a Hand as linked lines using LANDMARK_CONNECTIVITY
        """
        if hand_wrist_data.hand_pose is not None:
            hand_wrist_pose_matrix = hand_wrist_data.hand_pose.to_matrix()
            hand_wrist_pose_tensor = torch.from_numpy(hand_wrist_pose_matrix)

            # Set translation to 0. Fix scaling and translation as a post processing step
            hand_wrist_pose_tensor[:, 3] = torch.zeros(1, 4)

            # self._hand_model is defined for the Left hand,
            #  flipping here the pose X axis is moving the Left Hand to a Right Hand
            if hand_wrist_data.handedness == "1":
                hand_wrist_pose_tensor[:, 0] *= -1
            hand_landmarks = skin_landmarks(
                self._hand_model,
                torch.Tensor(hand_wrist_data.joint_angles),
                hand_wrist_pose_tensor,
            )
            # Rescale and translate the vertices
            scale = 1e-3
            hand_landmarks = hand_landmarks.mul(scale)
            translation = torch.Tensor(hand_wrist_data.hand_pose.translation()[0])
            return hand_landmarks + translation.expand_as(hand_landmarks)

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

    def get_eye_gaze_in_camera(
        self,
        stream_id: StreamId,
        timestamp_ns: int,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
        depth_m: float = 1.0,
    ):
        """
        Return the eye_gaze at the given timestamp projected in the given stream for the given time_domain
        """
        if self.get_device_type() != DeviceType.ARIA:
            raise ValueError("Eye Gaze not available for this device.")

        # We have an Aria Device
        #
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
            eye_gaze = self.mps_data_provider.get_general_eyegaze(device_timestamp_ns)
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
                pinhole_calib = calibration.get_linear_camera_calibration(
                    image_size[0], image_size[1], focal_lengths[0]
                )
                device_calibration = self.device_calibration()
                T_device_CPF = device_calibration.get_transform_device_cpf()
                gaze_center_in_camera = (
                    T_device_camera.inverse() @ T_device_CPF @ gaze_vector_in_cpf
                )
                gaze_projection = pinhole_calib.project(gaze_center_in_camera)
                return gaze_projection
        return None

    def get_point_cloud(self) -> np.ndarray:
        """
        Return the point cloud of the scene
        """
        if self.get_device_type() != DeviceType.ARIA:
            raise ValueError("Point cloud data is not available for this device.")

        if self.mps_data_provider.has_semidense_point_cloud():
            point_cloud_data = self.mps_data_provider.get_semidense_point_cloud()
            # Todo: Should we clean it?
            return point_cloud_data

        return None

        pass

    # Todo
    # Need a mechanism to add filtering (visible in camera frustum , etc.)
    # Interface to retrieve MPS data if available (see the getDeviceType())
