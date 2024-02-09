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

import rerun as rr

from dataset_api import DeviceType, Hot3DDataProvider
from projectaria_tools.utils.rerun_helpers import ToTransform3D

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to hot3d data sequence",
    )

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

    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    timestamps = data_provider.get_timestamps()
    for timestamp_us in tqdm(timestamps):

        rr.set_time_nanos("synchronization_time", int(timestamp_us))
        # rr.set_time_sequence("timestamp", timestamp_us)

        # Retrieve METADATA object and visualize them

        # Plot image
        image_data = data_provider.get_image(timestamp_us)
        if image_data:
            rr.log(
                f"world/device/{image_data.label}",
                rr.Image(image_data.img).compress(jpeg_quality=args.jpeg_quality),
            )

        # Plot Hand poses
        hands_data = data_provider.get_hand_poses(timestamp_us)
        if hands_data:
            for hand_data in hands_data:
                print(f"Hand pose: {hand_data}")

        # Plot Object poses
        objects_data = data_provider.get_object_poses(timestamp_us)
        for object_data_key, object_data_value in objects_data.items():

            rr.log(
                f"/world/{object_data_key}", ToTransform3D(object_data_value[0], False)
            )

            # If desired (display the corresponding 3D object)

        # Plot device pose
        # device_pose_data = data_provider.get_device_pose(timestamp_us)
        # T_world_device = device_pose_data.T_world_device
        # rr.log("world/device", ToTransform3D(T_world_device, False))

        # Deal with device specifics
        # if data_provider.get_device_type() == DeviceType.ARIA:
        # print("MPS specifics")
        # Display MPS artefact if desired


if __name__ == "__main__":
    main()
