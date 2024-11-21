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

from typing import Dict, List, Optional

import matplotlib.pyplot as plt

import numpy as np
import rerun as rr  # @manual

from data_loaders.hand_common import LANDMARK_CONNECTIVITY
from data_loaders.headsets import Headset
from data_loaders.loader_hand_poses import HandType
from data_loaders.loader_object_library import ObjectLibrary

# pyre-fixme[21]: Could not find name `StreamId` in `projectaria_tools.core.stream_id`.
from projectaria_tools.core.stream_id import StreamId  # @manual

try:
    # pyre-fixme[21]: Could not find module `dataset_api`.
    from dataset_api import Hot3dDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3dDataProvider

from data_loaders.HandDataProviderBase import (  # @manual
    HandDataProviderBase,
    HandPose3dCollectionWithDt,
)
from data_loaders.ObjectBox2dDataProvider import (  # @manual
    ObjectBox2dCollectionWithDt,
    ObjectBox2dProvider,
)

from data_loaders.ObjectPose3dProvider import (  # @manual
    ObjectPose3dCollectionWithDt,
    ObjectPose3dProvider,
)

# pyre-fixme[21]: Could not find name `CameraCalibration` in
#  `projectaria_tools.core.calibration`.
# pyre-fixme[21]: Could not find name `DeviceCalibration` in
#  `projectaria_tools.core.calibration`.
# pyre-fixme[21]: Could not find name `FISHEYE624` in
#  `projectaria_tools.core.calibration`.
# pyre-fixme[21]: Could not find name `LINEAR` in `projectaria_tools.core.calibration`.
from projectaria_tools.core.calibration import (
    CameraCalibration,
    DeviceCalibration,
    FISHEYE624,
    LINEAR,
)

# pyre-fixme[21]: Could not find name `get_eyegaze_point_at_depth` in
#  `projectaria_tools.core.mps`.
from projectaria_tools.core.mps import get_eyegaze_point_at_depth  # @manual

from projectaria_tools.core.mps.utils import (  # @manual
    filter_points_from_confidence,
    filter_points_from_count,
)

# pyre-fixme[21]: Could not find name `TimeDomain` in
#  `projectaria_tools.core.sensor_data`.
# pyre-fixme[21]: Could not find name `TimeQueryOptions` in
#  `projectaria_tools.core.sensor_data`.
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual

# pyre-fixme[21]: Could not find name `SE3` in `projectaria_tools.core.sophus`.
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.utils.rerun_helpers import (  # @manual
    AriaGlassesOutline,
    ToTransform3D,
)


class Hot3DVisualizer:
    def __init__(
        self,
        hot3d_data_provider: Hot3dDataProvider,
        hand_type: HandType = HandType.Umetrack,
    ) -> None:
        self._hot3d_data_provider = hot3d_data_provider
        # Device calibration and Image stream data
        self._device_data_provider = hot3d_data_provider.device_data_provider
        # Data provider at time T (for device & objects & hand poses)
        self._device_pose_provider = hot3d_data_provider.device_pose_data_provider
        self._hand_data_provider = (
            hot3d_data_provider.umetrack_hand_data_provider
            if hand_type == HandType.Umetrack
            else hot3d_data_provider.mano_hand_data_provider
        )
        if hand_type is HandType.Umetrack:
            print("Hot3DVisualizer is using UMETRACK hand model")
        elif hand_type is HandType.Mano:
            print("Hot3DVisualizer is using MANO hand model")
        self._object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        self._object_box2d_data_provider = (
            hot3d_data_provider.object_box2d_data_provider
        )
        # Object library
        self._object_library = hot3d_data_provider.object_library

        # If required
        # Retrieve a distinct color mapping for object bounding box to show consistent color across stream_ids
        # - Use a Colormap for visualizing object bounding box
        self._object_box2d_colors = None
        if self._object_box2d_data_provider is not None:
            color_map = plt.get_cmap("viridis")
            self._object_box2d_colors = color_map(
                np.linspace(0, 1, len(self._object_box2d_data_provider.object_uids))
            )

        # Keep track of what 3D assets has been loaded/unloaded so we will load them only when needed
        self._object_cache_status = {}

        # To be parametrized later
        self._jpeg_quality = 75

    def log_static_assets(
        self,
        # pyre-fixme[11]: Annotation `StreamId` is not defined as a type.
        image_stream_ids: List[StreamId],
    ) -> None:
        """
        Log all static assets (aka Timeless assets)
        - assets that are immutable (but can still move if attached to a 3D Pose)
        """

        # Configure the world coordinate system to ease navigation
        if self._hot3d_data_provider.get_device_type() is Headset.Aria:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        else:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        if self._hot3d_data_provider.get_device_type() is Headset.Aria:
            ## for Aria devices, we use online calibration which is a dynamic asset
            pass
        elif self._hot3d_data_provider.get_device_type() is Headset.Quest3:
            # For each of the stream ids we want to use, export the camera calibration (intrinsics and extrinsics)
            for stream_id in image_stream_ids:
                #
                # Plot the camera configuration
                [extrinsics, intrinsics] = (
                    self._device_data_provider.get_camera_calibration(stream_id)
                )
                Hot3DVisualizer.log_pose(
                    f"world/device/{stream_id}", extrinsics, static=True
                )
                Hot3DVisualizer.log_calibration(f"world/device/{stream_id}", intrinsics)

        # Deal with Aria specifics
        # - Glasses outline
        # - Point cloud
        if self._hot3d_data_provider.get_device_type() is Headset.Aria:
            Hot3DVisualizer.log_aria_glasses(
                "world/device/glasses_outline",
                self._device_data_provider.get_device_calibration(),
            )

            # Point cloud (downsampled for visualization)
            point_cloud = self._device_data_provider.get_point_cloud()
            if point_cloud:
                # Filter out low confidence points
                threshold_invdep = 5e-4
                threshold_dep = 5e-4
                point_cloud = filter_points_from_confidence(
                    point_cloud, threshold_invdep, threshold_dep
                )
                # Down sample points
                points_data_down_sampled = filter_points_from_count(
                    point_cloud, 500_000
                )
                # Retrieve point position
                point_positions = [it.position_world for it in points_data_down_sampled]
                POINT_COLOR = [200, 200, 200]
                rr.log(
                    "world/points",
                    rr.Points3D(point_positions, colors=POINT_COLOR, radii=0.002),
                    static=True,
                )

    def log_dynamic_assets(
        self,
        stream_ids: List[StreamId],
        timestamp_ns: int,
    ) -> None:
        """
        Log dynamic assets:
        I.e assets that are moving, such as:
        - 3D assets
        - Device pose
        - Hands
        - Object poses
        - Image related specifics assets
        - images (stream_ids)
        - Object Bounding boxes
        - Aria Eye Gaze
        """

        #
        ## Retrieve and log data that is not stream_id dependent (pure 3D data)
        #
        acceptable_time_delta = 0

        if self._hot3d_data_provider.get_device_type() is Headset.Aria:
            # For each of the stream ids we want to use, export the camera calibration (intrinsics and extrinsics)
            for stream_id in stream_ids:
                #
                # Plot the camera configuration
                [extrinsics, intrinsics] = (
                    self._device_data_provider.get_online_camera_calibration(
                        stream_id=stream_id, timestamp_ns=timestamp_ns
                    )
                )
                Hot3DVisualizer.log_pose(f"world/device/{stream_id}", extrinsics)
                Hot3DVisualizer.log_calibration(f"world/device/{stream_id}", intrinsics)

        elif self._hot3d_data_provider.get_device_type() is Headset.Quest3:
            ## for Quest devices we will use factory calibration which is a static asset
            pass

        headset_pose3d_with_dt = None
        if self._device_data_provider is not None:
            headset_pose3d_with_dt = self._device_pose_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                # pyre-fixme[16]: Module `sensor_data` has no attribute
                #  `TimeQueryOptions`.
                time_query_options=TimeQueryOptions.CLOSEST,
                # pyre-fixme[16]: Module `sensor_data` has no attribute `TimeDomain`.
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=acceptable_time_delta,
            )

        hand_poses_with_dt = None
        if self._hand_data_provider is not None:
            hand_poses_with_dt = self._hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                # pyre-fixme[16]: Module `sensor_data` has no attribute
                #  `TimeQueryOptions`.
                time_query_options=TimeQueryOptions.CLOSEST,
                # pyre-fixme[16]: Module `sensor_data` has no attribute `TimeDomain`.
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=acceptable_time_delta,
            )

        object_poses_with_dt = None
        if self._object_pose_data_provider is not None:
            object_poses_with_dt = (
                self._object_pose_data_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp_ns,
                    # pyre-fixme[16]: Module `sensor_data` has no attribute
                    #  `TimeQueryOptions`.
                    time_query_options=TimeQueryOptions.CLOSEST,
                    # pyre-fixme[16]: Module `sensor_data` has no attribute
                    #  `TimeDomain`.
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=acceptable_time_delta,
                )
            )

        aria_eye_gaze_data = (
            self._device_data_provider.get_eye_gaze(timestamp_ns)
            if self._hot3d_data_provider.get_device_type() is Headset.Aria
            else None
        )

        #
        ## Log Device pose
        #
        if headset_pose3d_with_dt is not None:
            headset_pose3d = headset_pose3d_with_dt.pose3d
            Hot3DVisualizer.log_pose(
                "world/device", headset_pose3d.T_world_device, static=False
            )

        #
        ## Log Hand poses
        #
        Hot3DVisualizer.log_hands(
            "world/hands",  # /{handedness_label}/... will be added as necessary
            self._hand_data_provider,
            hand_poses_with_dt,
            show_hand_mesh=True,
            show_hand_vertices=False,
            show_hand_landmarks=False,
        )

        #
        ## Log Object poses
        #
        Hot3DVisualizer.log_object_poses(
            "world/objects",
            object_poses_with_dt,
            self._object_pose_data_provider,
            self._object_library,
            self._object_cache_status,
        )

        #
        ## Log stream dependent data
        #
        for stream_id in stream_ids:
            #
            ## Log Image data
            #

            # Undistorted image (required if you want see reprojected 3D mesh on the images)
            image_data = self._device_data_provider.get_undistorted_image(
                timestamp_ns, stream_id
            )
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}",
                    rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
                )

            # Raw device images (required for object bounding box visualization)
            image_data = self._device_data_provider.get_image(timestamp_ns, stream_id)
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}_raw",
                    rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
                )

            if (
                self._object_box2d_data_provider is not None
                and stream_id in self._object_box2d_data_provider.stream_ids
            ):
                box2d_collection_with_dt = (
                    self._object_box2d_data_provider.get_bbox_at_timestamp(
                        stream_id=stream_id,
                        timestamp_ns=timestamp_ns,
                        # pyre-fixme[16]: Module `sensor_data` has no attribute
                        #  `TimeQueryOptions`.
                        time_query_options=TimeQueryOptions.CLOSEST,
                        # pyre-fixme[16]: Module `sensor_data` has no attribute
                        #  `TimeDomain`.
                        time_domain=TimeDomain.TIME_CODE,
                    )
                )
                Hot3DVisualizer.log_object_bounding_boxes(
                    stream_id,
                    box2d_collection_with_dt,
                    self._object_box2d_data_provider,
                    self._object_library,
                    self._object_box2d_colors,
                )

            #
            ## Eye Gaze image reprojection
            #
            if self._hot3d_data_provider.get_device_type() is Headset.Aria:
                # We are showing EyeGaze reprojection only on the RGB image stream
                # pyre-fixme[16]: Module `stream_id` has no attribute `StreamId`.
                if stream_id != StreamId("214-1"):
                    continue

                # Reproject EyeGaze for raw and pinhole images
                # pyre-fixme[16]: Module `calibration` has no attribute `FISHEYE624`.
                # pyre-fixme[16]: Module `calibration` has no attribute `LINEAR`.
                camera_configurations = [FISHEYE624, LINEAR]
                for camera_model in camera_configurations:
                    eye_gaze_reprojection_data = (
                        self._device_data_provider.get_eye_gaze_in_camera(
                            stream_id, timestamp_ns, camera_model=camera_model
                        )
                    )
                    if (
                        eye_gaze_reprojection_data is None
                        or not eye_gaze_reprojection_data.any()
                    ):
                        continue

                    label = (
                        f"world/device/{stream_id}/eye-gaze_projection"
                        # pyre-fixme[16]: Module `calibration` has no attribute
                        #  `LINEAR`.
                        if camera_model == LINEAR
                        else f"world/device/{stream_id}_raw/eye-gaze_projection_raw"
                    )
                    rr.log(
                        label,
                        rr.Points2D(eye_gaze_reprojection_data, radii=20),
                        # TODO consistent color and size depending of camera resolution
                    )
        #
        ## Log device dependent remaining 3D data
        #

        # Log 3D eye gaze
        if aria_eye_gaze_data is not None:
            T_device_CPF = self._device_data_provider.get_device_calibration().get_transform_device_cpf()
            # Compute eye_gaze vector at depth_m (30cm for a proxy 3D vector to display)
            # pyre-fixme[16]: Module `mps` has no attribute
            #  `get_eyegaze_point_at_depth`.
            gaze_vector_in_cpf = get_eyegaze_point_at_depth(
                aria_eye_gaze_data.yaw, aria_eye_gaze_data.pitch, depth_m=0.3
            )
            # Draw EyeGaze vector
            rr.log(
                "world/device/eye-gaze",
                rr.Arrows3D(
                    origins=[T_device_CPF @ [0, 0, 0]],
                    vectors=[T_device_CPF @ gaze_vector_in_cpf],
                ),
            )

    @staticmethod
    def log_aria_glasses(
        label: str,
        # pyre-fixme[11]: Annotation `DeviceCalibration` is not defined as a type.
        device_calibration: DeviceCalibration,
        use_cad_calibration: bool = True,
    ) -> None:
        ## Plot Project Aria Glasses outline (as lines)
        aria_glasses_point_outline = AriaGlassesOutline(
            device_calibration, use_cad_calibration
        )
        rr.log(label, rr.LineStrips3D([aria_glasses_point_outline]), static=True)

    @staticmethod
    def log_calibration(
        label: str,
        # pyre-fixme[11]: Annotation `CameraCalibration` is not defined as a type.
        camera_calibration: CameraCalibration,
    ) -> None:
        rr.log(
            label,
            rr.Pinhole(
                resolution=[
                    camera_calibration.get_image_size()[0],
                    camera_calibration.get_image_size()[1],
                ],
                focal_length=float(camera_calibration.get_focal_lengths()[0]),
            ),
            static=True,
        )

    @staticmethod
    # pyre-fixme[11]: Annotation `SE3` is not defined as a type.
    def log_pose(label: str, pose: SE3, static=False) -> None:
        rr.log(label, ToTransform3D(pose, False), static=static)

    @staticmethod
    def log_hands(
        label: str,
        hand_data_provider: HandDataProviderBase,
        hand_poses_with_dt: HandPose3dCollectionWithDt,
        show_hand_mesh=True,
        show_hand_vertices=True,
        show_hand_landmarks=True,
    ):
        logged_right_hand_data = False
        logged_left_hand_data = False
        if hand_poses_with_dt is None:
            return

        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        for hand_pose_data in hand_pose_collection.poses.values():
            if hand_pose_data.is_left_hand():
                logged_left_hand_data = True
            elif hand_pose_data.is_right_hand():
                logged_right_hand_data = True

            handedness_label = hand_pose_data.handedness_label()

            # Skeleton/Joints landmark representation
            if show_hand_landmarks:
                hand_landmarks = hand_data_provider.get_hand_landmarks(hand_pose_data)
                # convert landmarks to connected lines for display
                # (i.e retrieve points along the HAND LANDMARK_CONNECTIVITY as a list)
                points = [
                    connections
                    for connectivity in LANDMARK_CONNECTIVITY
                    for connections in [
                        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                        [hand_landmarks[it].numpy().tolist() for it in connectivity]
                    ]
                ]
                rr.log(
                    f"{label}/{handedness_label}/joints",
                    rr.LineStrips3D(points, radii=0.002),
                )

            # Update mesh vertices if required
            hand_mesh_vertices = (
                hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                if show_hand_vertices or show_hand_mesh
                else None
            )

            # Vertices representation
            if show_hand_vertices:
                rr.log(
                    f"{label}/{handedness_label}/mesh",
                    rr.Points3D(hand_mesh_vertices),
                )

            # Triangular Mesh representation
            if show_hand_mesh:
                # pyre-fixme[23]: Unable to unpack `list[ndarray[Any, Any]] | None`
                #  into 2 values.
                [hand_triangles, hand_vertex_normals] = (
                    hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)
                )
                rr.log(
                    f"{label}/{handedness_label}/mesh_faces",
                    rr.Mesh3D(
                        vertex_positions=hand_mesh_vertices,
                        vertex_normals=hand_vertex_normals,
                        triangle_indices=hand_triangles,  # TODO: we could avoid sending this list if we want to save memory
                    ),
                )
        # If some hand data has not been logged, do not show it in the visualizer
        if logged_left_hand_data is False:
            rr.log(f"{label}/left", rr.Clear.recursive())
        if logged_right_hand_data is False:
            rr.log(f"{label}/right", rr.Clear.recursive())

    @staticmethod
    def log_object_poses(
        label: str,  # "world/objects",
        object_poses_with_dt: ObjectPose3dCollectionWithDt,
        object_pose_data_provider: ObjectPose3dProvider,
        object_library: ObjectLibrary,
        object_cache_status: Dict[int, bool],
    ):
        if object_poses_with_dt is None:
            return

        objects_pose3d_collection = object_poses_with_dt.pose3d_collection

        # Keep a mapping to know what object has been seen, and which one has not
        object_uids = object_pose_data_provider.object_uids_with_poses
        logging_status = {x: False for x in object_uids}

        for (
            object_uid,
            object_pose3d,
        ) in objects_pose3d_collection.poses.items():
            object_name = object_library.object_id_to_name_dict[object_uid]
            object_name = object_name + "_" + str(object_uid)
            object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
                object_library_folderpath=object_library.asset_folder_name,
                object_id=object_uid,
            )

            Hot3DVisualizer.log_pose(
                f"world/objects/{object_name}",
                object_pose3d.T_world_object,
                False,
            )
            # Mark object has been seen
            logging_status[object_uid] = True

            # Link the corresponding 3D object
            if object_uid not in object_cache_status.keys():
                # pyre-fixme[6]: For 1st argument expected `int` but got `str`.
                object_cache_status[object_uid] = True
                rr.log(
                    f"world/objects/{object_name}",
                    rr.Asset3D(
                        path=object_cad_asset_filepath,
                    ),
                )

        # If some object are not visible, we clear the entity (last known mesh and pose will not be displayed)
        for object_uid, displayed in logging_status.items():
            if not displayed:
                object_name = object_library.object_id_to_name_dict[object_uid]
                object_name = object_name + "_" + str(object_uid)
                rr.log(
                    f"world/objects/{object_name}",
                    rr.Clear.recursive(),
                )
                if object_uid in object_cache_status.keys():
                    # pyre-fixme[6]: For 1st argument expected `int` but got `str`.
                    del object_cache_status[object_uid]  # We will log the mesh again

    @staticmethod
    def log_object_bounding_boxes(
        stream_id: StreamId,
        box2d_collection_with_dt: Optional[ObjectBox2dCollectionWithDt],
        object_box2d_data_provider: ObjectBox2dProvider,
        object_library: ObjectLibrary,
        bbox_colors: np.ndarray,
    ):
        """
        Object bounding boxes (valid for native raw images).
        - We assume that the image corresponding to the stream_id has been logged beforehand as 'world/device/{stream_id}_raw/'
        """

        # Keep a mapping to know what object has been seen, and which one has not
        object_uids = list(object_box2d_data_provider.object_uids)
        logging_status = {x: False for x in object_uids}

        if (
            box2d_collection_with_dt is None
            or box2d_collection_with_dt.box2d_collection is None
        ):
            # No bounding box are retrieved, we clear all the bounding box visualization existing so far
            rr.log(f"world/device/{stream_id}_raw/bbox", rr.Clear.recursive())
            return

        object_uids_at_query_timestamp = (
            box2d_collection_with_dt.box2d_collection.object_uid_list
        )

        for object_uid in object_uids_at_query_timestamp:
            object_name = object_library.object_id_to_name_dict[object_uid]
            axis_aligned_box2d = box2d_collection_with_dt.box2d_collection.box2ds[
                object_uid
            ]
            box = axis_aligned_box2d.box2d
            if box is None:
                continue

            logging_status[object_uid] = True
            rr.log(
                f"world/device/{stream_id}_raw/bbox/{object_name}",
                rr.Boxes2D(
                    mins=[box.left, box.top],
                    sizes=[box.width, box.height],
                    colors=bbox_colors[object_uids.index(object_uid)],
                ),
            )
        # If some object are not visible, we clear the bounding box visualization
        for key, value in logging_status.items():
            if not value:
                object_name = object_library.object_id_to_name_dict[key]
                rr.log(
                    f"world/device/{stream_id}_raw/bbox/{object_name}",
                    rr.Clear.flat(),
                )
