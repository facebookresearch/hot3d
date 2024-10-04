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
from data_loaders.loader_hand_poses import HandType
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import loadManoHandModel

try:
    from dataset_api import Hot3dDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3dDataProvider

try:
    from Hot3DVisualizer import Hot3DVisualizer
except ImportError:
    from hot3d.Hot3DVisualizer import Hot3DVisualizer

from tqdm import tqdm


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
        help="path to object library folder containing instance.json and *.glb cad files",
        required=True,
    )
    parser.add_argument(
        "--mano_model_folder",
        type=str,
        default=None,
        help="path to MANO models containing the MANO_RIGHT/LEFT.pkl files",
        required=False,
    )
    parser.add_argument(
        "--hand_type",
        type=str,
        default="UMETRACK",
        help="type of HAND (MANO or UMETRACK)",
        required=False,
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
    mano_model_folder: Optional[str],
    rrd_output_path: Optional[str],
    jpeg_quality: int,
    timestamps_slice: Type[slice],
    fail_on_missing_data: bool,
    hand_type: str,
):
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(
            f"Object Library folder {object_library_folder} does not exist"
        )

    hand_enum_type = HandType.Umetrack if hand_type == "UMETRACK" else HandType.Mano
    if hand_type not in ["UMETRACK", "MANO"]:
        raise RuntimeError(
            f"Invalid hand type: {hand_type}. hand_type must be either UMETRACK or MANO"
        )

    object_library = load_object_library(
        object_library_folderpath=object_library_folder
    )

    mano_hand_model = loadManoHandModel(mano_model_folder)

    #
    # Initialize hot3d data provider
    #
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=fail_on_missing_data,
    )
    print(f"data_provider statistics: {data_provider.get_data_statistics()}")

    #
    # Prepare the rerun rerun log configuration
    #
    rr.init("hot3d Data Viewer", spawn=(rrd_output_path is None))
    if rrd_output_path is not None:
        print(f"Saving .rrd file to {rrd_output_path}")
        rr.save(rrd_output_path)

    #
    # Initialize the rerun hot3d visualizer interface
    #
    rr_visualizer = Hot3DVisualizer(data_provider, hand_enum_type)

    # Define which image stream will be shown
    image_stream_ids = data_provider.device_data_provider.get_image_stream_ids()

    #
    # Log static assets (aka Timeless assets)
    rr_visualizer.log_static_assets(image_stream_ids)

    timestamps = data_provider.device_data_provider.get_sequence_timestamps()
    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    for timestamp in tqdm(timestamps[timestamps_slice]):
        rr.set_time_nanos("synchronization_time", int(timestamp))
        rr.set_time_sequence("timestamp", timestamp)

        rr_visualizer.log_dynamic_assets(image_stream_ids, timestamp)


def main():
    args = parse_args()
    print(f"args provided: {args}")

    try:
        execute_rerun(
            sequence_folder=args.sequence_folder,
            object_library_folder=args.object_library_folder,
            mano_model_folder=args.mano_model_folder,
            rrd_output_path=args.rrd_output_path,
            jpeg_quality=args.jpeg_quality,
            timestamps_slice=slice(None, None, None),
            fail_on_missing_data=False,
            hand_type=args.hand_type,
        )
    except Exception as error:
        print(f"An exception occurred: {error}")


if __name__ == "__main__":
    main()
