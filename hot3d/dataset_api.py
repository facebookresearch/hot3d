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

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data_loaders.AriaDataProvider import AriaDataProvider
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

from projectaria_tools.core import calibration  # @manual
from projectaria_tools.core.mps import (  # @manual
    get_eyegaze_point_at_depth,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
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

        self._hand_poses = load_hand_poses(
            self.path_provider.hand_pose_trajectory_filepath
        )
        # Hand profile
        self._hand_model = load_hand_model_from_file(
            self.path_provider.hand_user_profile_filepath
        )

        if self.get_device_type() == Headset.Aria:
            # Aria specifics

            # VRS data provider
            self._device_data_provider = AriaDataProvider(
                self.path_provider.vrs_filepath
            )

            # MPS data provider
            mps_possible_path = self.path_provider.mps_folderpath
            if os.path.exists(mps_possible_path):
                mps_data_paths_provider = MpsDataPathsProvider(mps_possible_path)
                mps_data_paths = mps_data_paths_provider.get_data_paths()
                self.mps_data_provider = MpsDataProvider(mps_data_paths)
                print(mps_data_paths)

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
        if self.get_device_type() != Headset.Aria:
            raise ValueError("Eye Gaze not available for this device.")

        # We have an Aria Device
        #
        # Map to corresponding timestamp
        if time_domain == TimeDomain.TIME_CODE:
            device_timestamp_ns = self._device_data_provider._timestamp_convert(
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
                [T_device_camera, camera_calibration] = (
                    self._device_data_provider.get_camera_calibration(stream_id)
                )
                focal_lengths = camera_calibration.get_focal_lengths()
                image_size = camera_calibration.get_image_size()
                pinhole_calib = calibration.get_linear_camera_calibration(
                    image_size[0], image_size[1], focal_lengths[0]
                )
                device_calibration = self._device_data_provider.get_device_calibration()
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
        if self.get_device_type() != Headset.Aria:
            raise ValueError("Point cloud data is not available for this device.")

        if self.mps_data_provider.has_semidense_point_cloud():
            point_cloud_data = self.mps_data_provider.get_semidense_point_cloud()
            # Todo: Should we clean it?
            return point_cloud_data

        return None

        pass
