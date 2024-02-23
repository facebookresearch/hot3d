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

import rerun as rr

from dataset_api import DeviceType, Hot3DDataProvider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import ToTransform3D

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to hot3d data sequence",
    )

    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()

    #
    # Gather data input
    #
    if args.folder is None:
        print("Nothing to display.")
        exit(1)

    # Initialize hot3d data provider
    data_provider = Hot3DDataProvider(args.folder)

    # Initializing Rerun viewer
    rr.init("MPS Data Viewer", spawn=True)

    # TODO:
    # For convenience LOG the camera trajectory as a 3D line to help user understand the type of motion in the sequence

    rgb_stream_id = StreamId("214-1")
    # Plot the camera configuration
    [extrinsics, intrinsics] = data_provider.get_camera_calibration(rgb_stream_id)
    rr.log(
        f"world/device/{rgb_stream_id}", ToTransform3D(extrinsics, False), timeless=True
    )
    rr.log(
        f"world/device/{rgb_stream_id}",
        rr.Pinhole(
            resolution=[
                intrinsics.get_image_size()[0],
                intrinsics.get_image_size()[1],
            ],
            focal_length=float(intrinsics.get_focal_lengths()[0]),
        ),
        timeless=True,
    )

    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    timestamps = data_provider.get_timestamps()
    # Crop timestamp to the valid timing of the Image recording
    [min_timestamp, max_timestamp] = data_provider.get_valid_recording_range()
    timestamps = [
        x for x in timestamps if int(x) >= min_timestamp and int(x) <= max_timestamp
    ]

    object_table = (
        {}
    )  # We want to log 3D assets once, we keep track of their addition here
    for timestamp in tqdm(timestamps):

        rr.set_time_nanos("synchronization_time", int(timestamp))
        rr.set_time_sequence("timestamp", timestamp)

        # Retrieve METADATA object and visualize them

        # Plot Hand poses
        hands_data = data_provider.get_hand_poses(timestamp)
        if hands_data:
            for hand_data in hands_data:
                rr.log(
                    f"/world/hands/{hand_data}",
                    ToTransform3D(hands_data[hand_data][0], False),
                )

        # Plot Object poses
        objects_data = data_provider.get_object_poses(timestamp)
        for object_data_key, object_data_value in objects_data.items():

            object_name = data_provider.get_object_instance_name(object_data_key)
            object_name = object_name + "_" + str(object_data_key)
            rr.log(
                f"/world/objects/{object_name}",
                ToTransform3D(object_data_value[0], False),
            )

            # If desired (display the corresponding 3D object)
            scale = rr.datatypes.Scale3D(10e-4)
            if object_data_key not in object_table.keys():
                object_table[object_data_key] = True
                rr.log(
                    f"/world/objects/{object_name}",
                    rr.Asset3D(
                        path=os.path.join(
                            os.path.dirname(args.folder),
                            "assets",
                            object_data_key + ".glb",
                        ),
                        transform=rr.TranslationRotationScale3D(scale=scale),
                    ),
                    timeless=True,
                )

        # Plot device pose
        device_pose_data = data_provider.get_device_pose(timestamp)
        if device_pose_data:
            rr.log("world/device", ToTransform3D(device_pose_data, False))

            # Plot image (corresponding to this pose)
            # image_data = data_provider.get_image(timestamp, rgb_stream_id)
            image_data = data_provider.get_undistorted_image(timestamp, rgb_stream_id)
            if image_data is not None:
                rr.log(
                    f"world/device/{rgb_stream_id}",
                    # f"{rgb_stream_id}",
                    rr.Image(image_data).compress(jpeg_quality=args.jpeg_quality),
                )

        # Deal with device specifics
        # if data_provider.get_device_type() == DeviceType.ARIA:
        # print("MPS specifics")
        # Display MPS artefact if desired


if __name__ == "__main__":
    main()
