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

import argparse
import os
from typing import Optional, Type

import rerun as rr  # @manual
from data_loaders.headsets import Headset
from data_loaders.loader_object_library import load_object_library, ObjectLibrary

try:
    from dataset_api import Hot3DDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3DDataProvider

from projectaria_tools.core.mps.utils import (  # @manual
    filter_points_from_confidence,
    filter_points_from_count,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.utils.rerun_helpers import (  # @manual
    AriaGlassesOutline,
    ToTransform3D,
)

from tqdm import tqdm

from UmeTrack.common.hand import LANDMARK_CONNECTIVITY  # @manual


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to hot3d data sequence",
        required=True,
    )
    parser.add_argument(
        "--object_library_folder",
        type=str,
        help="path to object library folder containing instance.json and assets/*.glb cad files",
        required=True,
    )

    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    # If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument(
        "--rrd_output_path", type=str, default=None, help=argparse.SUPPRESS
    )

    return parser.parse_args()


def execute_rerun(
    sequence_folder: str,
    object_library_folder: str,
    rrd_output_path: Optional[str],
    jpeg_quality: int,
    timestamps_slice: Type[slice],
):

    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(
            f"Object Library folder {object_library_folder} does not exist"
        )

    object_library: ObjectLibrary = load_object_library(
        object_library_folderpath=object_library_folder
    )

    # Initialize hot3d data provider
    data_provider = Hot3DDataProvider(
        sequence_folder=sequence_folder, object_library=object_library
    )
    print(f"data_provider statistics: {data_provider.get_data_statistics()}")

    # Device calibration and Image stream
    device_data_provider = data_provider.device_data_provider
    # X at time T
    device_pose_provider = data_provider.device_pose_data_provider
    hand_data_provider = data_provider.hand_data_provider
    object_pose_data_provider = data_provider.object_pose_data_provider
    # Bounding box at time T
    object_box2d_data_provider = data_provider.object_box2d_data_provider

    # Initializing rerun log configuration
    rr.init("hot3d Data Viewer", spawn=(rrd_output_path is None))
    if rrd_output_path is not None:
        print(f"Saving .rrd file to {rrd_output_path}")
        rr.save(rrd_output_path)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)

    # TODO:
    # For convenience LOG the camera trajectory as a 3D line to help user understand the type of user motion in the sequence

    image_stream_ids = device_data_provider.get_image_stream_ids()

    # Log STATIC assets (aka Timeless assets)
    for stream_id in image_stream_ids:
        #
        # Plot the camera configuration
        [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(
            stream_id
        )
        rr.log(
            f"world/device/{stream_id}",
            ToTransform3D(extrinsics, False),
            timeless=True,
        )
        rr.log(
            f"world/device/{stream_id}",
            rr.Pinhole(
                resolution=[
                    intrinsics.get_image_size()[0],
                    intrinsics.get_image_size()[1],
                ],
                focal_length=float(intrinsics.get_focal_lengths()[0]),
            ),
            timeless=True,
        )

    # Deal with Aria specifics
    if data_provider.get_device_type() is Headset.Aria:
        device_calibration = device_data_provider.get_device_calibration()
        aria_glasses_point_outline = AriaGlassesOutline(device_calibration)
        rr.log(
            "world/device/glasses_outline",
            rr.LineStrips3D([aria_glasses_point_outline]),
            timeless=True,
        )

        # Point cloud
        point_cloud = device_data_provider.get_point_cloud()
        if point_cloud:
            # Filter out low confidence points
            threshold_invdep = 5e-4
            threshold_dep = 5e-4
            point_cloud = filter_points_from_confidence(
                point_cloud, threshold_invdep, threshold_dep
            )
            # Down sample points
            points_data_down_sampled = filter_points_from_count(point_cloud, 500_000)
            # Retrieve point position
            point_positions = [it.position_world for it in points_data_down_sampled]
            POINT_COLOR = [200, 200, 200]
            rr.log(
                "world/points",
                rr.Points3D(point_positions, colors=POINT_COLOR, radii=0.002),
                timeless=True,
            )

    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    timestamps = device_data_provider.get_sequence_timestamps()

    object_table = (
        {}
    )  # We want to log 3D assets once, we keep track of their addition here
    for timestamp in tqdm(timestamps[timestamps_slice]):

        rr.set_time_nanos("synchronization_time", int(timestamp))
        rr.set_time_sequence("timestamp", timestamp)

        # Retrieve METADATA object and visualize them
        # 1. Plot 3D assets
        #    - 1.a Device pose
        #    - 1.b hands
        #    - 1.c Object poses
        # 2. Plot image specifics assets

        # 1. Plot 3D assets
        # 1.a Device pose
        headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        headset_pose3d = headset_pose3d_with_dt.pose3d

        if headset_pose3d:
            rr.log("world/device", ToTransform3D(headset_pose3d.T_world_device, False))

        #  1.b hands
        hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )

        logged_right_hand_data = False
        logged_left_hand_data = False
        if hand_poses_with_dt is not None:
            hands_data = hand_poses_with_dt.hand_poses
            for hand_data in hands_data:
                if hand_data.hand_pose is not None:

                    if hand_data.is_left_hand():
                        logged_left_hand_data = True
                    elif hand_data.is_right_hand():
                        logged_right_hand_data = True

                    handedness_label = hand_data.handedness_label()

                    # Wrist pose representation
                    rr.log(
                        f"/world/hands/{handedness_label}/pose",
                        ToTransform3D(hand_data.hand_pose, False),
                    )

                    # Skeleton/Joints landmark representation
                    hand_landmarks = hand_data_provider.get_hand_landmarks(hand_data)
                    # convert landmarks to connected lines for display
                    points = []
                    for connectivity in LANDMARK_CONNECTIVITY:
                        connections = []
                        for it in connectivity:
                            connections.append(hand_landmarks[it].numpy().tolist())
                        points.append(connections)
                    rr.log(
                        f"/world/hands/{handedness_label}/joints",
                        rr.LineStrips3D(points),
                    )

                    # Vertices representation
                    hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(
                        hand_data
                    )
                    rr.log(
                        f"/world/hands/{handedness_label}/mesh",
                        rr.Points3D(hand_mesh_vertices),
                    )

                    # Triangular Mesh representation
                    [hand_triangles, hand_vertex_normals] = (
                        hand_data_provider.get_hand_mesh_faces_and_normals(hand_data)
                    )
                    rr.log(
                        f"/world/hands/{handedness_label}/mesh_faces",
                        rr.Mesh3D(
                            vertex_positions=hand_mesh_vertices,
                            vertex_normals=hand_vertex_normals,
                            indices=hand_triangles,  # TODO: we could avoid sending this list if we want to save memory
                        ),
                    )
        # If some hand data has not been logged, do not show it in the visualizer
        if logged_left_hand_data is False:
            rr.log("/world/hands/left", rr.Clear.recursive())
        if logged_right_hand_data is False:
            rr.log("/world/hands/right", rr.Clear.recursive())

        # 1.c Object poses
        object_poses_with_dt = object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if object_poses_with_dt is not None:
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
                    object_library_folderpath=object_library_folder,
                    object_id=object_uid,
                )

                rr.log(
                    f"/world/objects/{object_name}",
                    ToTransform3D(object_pose3d.T_world_object, False),
                )
                # Mark object has been seen
                logging_status[object_uid] = True

                # Link the corresponding 3D object
                scale = rr.datatypes.Scale3D(1e-3)
                if object_uid not in object_table.keys():
                    object_table[object_uid] = True
                    rr.log(
                        f"/world/objects/{object_name}",
                        rr.Asset3D(
                            path=object_cad_asset_filepath,
                            transform=rr.TranslationRotationScale3D(scale=scale),
                        ),
                    )
            # If some object are not visible, we clear the bounding box visualization
            for object_uid, displayed in logging_status.items():
                if not displayed:
                    object_name = object_library.object_id_to_name_dict[object_uid]
                    object_name = object_name + "_" + str(object_uid)
                    rr.log(
                        f"world/objects/{object_name}",
                        rr.Clear.recursive(),
                    )
                    if object_uid in object_table.keys():
                        del object_table[object_uid]  # We will log the mesh again

        # 2. Plot image specifics assets
        #    - 2.a Image
        #    - 2.b Eye Gaze image reprojection
        #    - 2.c Object bounding boxes
        for stream_id in image_stream_ids:

            # 2.a Image
            image_data = device_data_provider.get_undistorted_image(
                timestamp, stream_id
            )
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}",
                    rr.Image(image_data).compress(jpeg_quality=jpeg_quality),
                )

            if data_provider.get_device_type() is Headset.Aria:
                # 2.b Eye Gaze image reprojection
                # Note: Eye Gaze is only available for the Aria device
                eye_gaze_reprojection_data = (
                    device_data_provider.get_eye_gaze_in_camera(stream_id, timestamp)
                )
                if (
                    eye_gaze_reprojection_data is not None
                    and eye_gaze_reprojection_data.any()
                ):
                    rr.log(
                        f"world/device/{stream_id}/eye-gaze_projection",
                        rr.Points2D(eye_gaze_reprojection_data, radii=20),
                    )

            # 2.c Object bounding boxes (valid for native raw images)
            image_data = device_data_provider.get_image(timestamp, stream_id)
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}_raw",
                    rr.Image(image_data).compress(jpeg_quality=jpeg_quality),
                )

                # Keep a mapping to know what object has been seen, and which one has not
                object_uids = list(object_box2d_data_provider.object_uids)
                logging_status = {x: False for x in object_uids}

                # Plot 2d bounding boxes
                box2d_collection_with_dt = (
                    object_box2d_data_provider.get_box2d_at_timestamp(
                        stream_id=stream_id,
                        timestamp_ns=timestamp,
                        time_query_options=TimeQueryOptions.CLOSEST,
                        time_domain=TimeDomain.TIME_CODE,
                    )
                )

                if (
                    box2d_collection_with_dt is not None
                    and box2d_collection_with_dt.box2d_collection is not None
                ):
                    object_uids_at_query_timestamp = (
                        box2d_collection_with_dt.box2d_collection.object_uid_list
                    )

                    for object_uid in object_uids_at_query_timestamp:
                        object_name = object_library.object_id_to_name_dict[object_uid]
                        axis_aligned_box2d = (
                            box2d_collection_with_dt.box2d_collection.box2ds[object_uid]
                        )
                        # TODO: do we want use visibility_ratio = axis_aligned_box2d.visibility_ratio
                        box = axis_aligned_box2d.box2d
                        logging_status[object_uid] = True
                        rr.log(
                            f"world/device/{stream_id}_raw/bbox/{object_name}",
                            rr.Boxes2D(
                                mins=[box.left, box.top], sizes=[box.width, box.height]
                            ),
                            # Todo: use the same colors for many objects
                        )
                    # If some object are not visible, we clear the bounding box visualization
                    for key, value in logging_status.items():
                        if not value:
                            object_name = object_library.object_id_to_name_dict[key]
                            rr.log(
                                f"world/device/{stream_id}_raw/bbox/{object_name}",
                                rr.Clear.flat(),
                            )
                else:
                    # No bounding box are retrieved, we clear all the bounding box visualization
                    rr.log(f"world/device/{stream_id}_raw/bbox", rr.Clear.recursive())


def main():
    args = parse_args()
    print(f"args provided: {args}")

    execute_rerun(
        sequence_folder=args.sequence_folder,
        object_library_folder=args.object_library_folder,
        rrd_output_path=args.rrd_output_path,
        jpeg_quality=args.jpeg_quality,
        timestamps_slice=slice(None, None, None),
    )


if __name__ == "__main__":
    main()
