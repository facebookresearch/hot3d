"""
This script converts the Hot3D-Clips dataset used for the BOP challenge 2024 to the standard BOP format
"""

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image
import tarfile
from typing import Any, Dict, List, Optional
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing

import clip_util
from hand_tracking_toolkit import rasterizer


def main():
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset_path", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/")
    #parser.add_argument("--dataset_path", type=str, required=True)
    # BOP dataset split name
    parser.add_argument("--split", type=str, default="train_aria_subsample")
    # Quest3 or Aria
    parser.add_argument("--dataset", type=str, default="aria")
    # output directory
    parser.add_argument("--output_dataset_path", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/train_aria_subsample_bop")
    # object models directory
    parser.add_argument("--object_models_dir", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval")
    # number of threads
    parser.add_argument("--num_threads", type=int, default=4)

    args = parser.parse_args()

    if args.dataset == "quest3":
        args.camera_streams_id = ["1201-1", "1201-2"]
        args.camera_streams_names = ["gray1", "gray2"]
    elif args.dataset == "aria":
        args.camera_streams_id = ["214-1", "1201-1", "1201-2"]
        args.camera_streams_names = ["rgb", "gray1", "gray2"]

    # paths
    clips_input_dir = os.path.join(args.input_dataset_path, args.split)
    scenes_output_dir = os.path.join(args.output_dataset_path, args.split)

    # list all clips names in the dataset
    split_clips = sorted([p for p in os.listdir(clips_input_dir) if p.endswith(".tar")])

    # create output directory
    os.makedirs(scenes_output_dir, exist_ok=True)  # TODO change exist_ok to False

    # Progress bar setup
    with tqdm(total=len(split_clips), desc="Processing clips") as pbar:
        # Use a Pool of 8 processes
        with multiprocessing.Pool(processes=args.num_threads) as pool:
            # Use imap_unordered to get results as soon as they're ready
            for _ in pool.imap_unordered(worker, ((clip, clips_input_dir, scenes_output_dir, args) for clip in split_clips)):
                pbar.update(1)


def worker(args):
    clip, clips_input_dir, scenes_output_dir, args = args
    process_clip(clip, clips_input_dir, scenes_output_dir, args)


def process_clip(clip, clips_input_dir, scenes_output_dir, args):
    # get clip id
    clip_name = clip.split(".")[0].split("-")[1]

    # extract clip
    tar = tarfile.open(os.path.join(clips_input_dir, clip), "r")

    # make scene folder and files for the scene
    scene_output_dir = os.path.join(scenes_output_dir, clip_name)
    os.makedirs(scene_output_dir, exist_ok=True)

    # make path of folders and folders
    # eg: STREAM_NAME, mask_STREAM_NAME, mask_visib_STREAM_NAME
    # also create path for each json file
    # eg: scene_camera_STREAM_NAME.json, scene_gt_STREAM_NAME.json, scene_gt_info_STREAM_NAME.json
    # create a dictionary for all camera streams
    clip_stream_paths = {}
    for stream_name in args.camera_streams_names:
        # directories
        stream_image_dir = os.path.join(scene_output_dir, stream_name)
        os.makedirs(stream_image_dir, exist_ok=True)
        clip_stream_paths[stream_name] = stream_image_dir
        stream_mask_dir = os.path.join(scene_output_dir, f"mask_{stream_name}")
        os.makedirs(stream_mask_dir, exist_ok=True)
        clip_stream_paths[f"mask_{stream_name}"] = stream_mask_dir
        stream_mask_visib_dir = os.path.join(scene_output_dir, f"mask_visib_{stream_name}")
        os.makedirs(stream_mask_visib_dir, exist_ok=True)
        clip_stream_paths[f"mask_visib_{stream_name}"] = stream_mask_visib_dir
        # json files
        stream_scene_camera_json_path = os.path.join(scene_output_dir, f"scene_camera_{stream_name}.json")
        clip_stream_paths[f"scene_camera_{stream_name}"] = stream_scene_camera_json_path
        stream_scene_gt_json_path = os.path.join(scene_output_dir, f"scene_gt_{stream_name}.json")
        clip_stream_paths[f"scene_gt_{stream_name}"] = stream_scene_gt_json_path
        stream_scene_gt_info_json_path = os.path.join(scene_output_dir, f"scene_gt_info_{stream_name}.json")
        clip_stream_paths[f"scene_gt_info_{stream_name}"] = stream_scene_gt_info_json_path

    # make a dict of dicts with stream name as keys
    scene_camera_data = {}
    scene_gt_data = {}
    scene_gt_info_data = {}
    for stream_name in args.camera_streams_names:
        # add an empty dict indicating the stream name
        scene_camera_data[stream_name] = {}
        scene_gt_data[stream_name] = {}
        scene_gt_info_data[stream_name] = {}

    # loop over all frames
    for frame_id in range(clip_util.get_number_of_frames(tar)):
        frame_key = f"{frame_id:06d}"

        # Load camera parameters.
        # from FRAME_ID.cameras.json
        frame_camera = clip_util.load_cameras(tar, frame_key)
        ## read FRAME_ID.objects.json
        frame_objects = clip_util.load_object_annotations(tar, frame_key)

        # TODO can i read this from the camera model?
        # read calibration json as it is
        camera_json_file_name = f"{frame_id:06d}.cameras.json"
        camera_json_file = tar.extractfile(camera_json_file_name)
        frame_camera_data = json.load(camera_json_file)

        # read FRAME_ID.info.json
        frame_info_file_name = f"{frame_id:06d}.info.json"
        frame_info_file = tar.extractfile(frame_info_file_name)
        frame_info_data = json.load(frame_info_file)

        # loop over all camera streams
        for stream_index, stream_name in enumerate(args.camera_streams_names):
            stream_id = args.camera_streams_id[stream_index]

            # load the image corresponding to the stream and frame
            image = clip_util.load_image(tar, frame_key, stream_id)
            # if image is rgb (3 channels), convert to BGR
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # save the image
            image_path = os.path.join(clip_stream_paths[stream_name], frame_key+".jpg")
            cv2.imwrite(image_path, image)

            # filling scene_camera.json

            # get T_world_from_camera
            T_world_from_camera = frame_camera[stream_id].T_world_from_eye

            # TODO check that inverting this homogenous transformation is correct
            T_world_to_camera = np.linalg.inv(T_world_from_camera)

            # get camera parameters
            calibration = frame_camera_data[stream_id]["calibration"]

            # add frame scene_camera data
            scene_camera_data[stream_name][int(frame_id)] = {
                # TODO change this after we agree on the final format
                "cam_model": calibration,
                "device": frame_info_data["device"],
                "image_timestamps_ns": frame_info_data["image_timestamps_ns"][stream_id],
                #"cam_K":  # not used as cam_model exists
                #"depth_scale":  # also not used
                # convert translation from meter to mm
                "cam_R_w2c": T_world_to_camera[:3, :3].flatten().tolist(),
                "cam_t_w2c": (T_world_to_camera[:3, 3] * 1000).tolist(),
            }

            # Camera parameters of the current image.
            camera_model = frame_camera[stream_id]

            frame_scene_gt_data = []
            frame_scene_gt_info_data = []
            # loop with enumerate over all objects in the frame
            for anno_id, obj_key in enumerate(frame_objects):
                obj_data = frame_objects[obj_key][0]

                # check if the object is visible in the current stream - stream id in keys of visibilities_modeled
                if stream_id not in obj_data["visibilities_modeled"]:
                    continue

                #bop_id = int(obj_data["object_bop_id"])  # same as obj_key

                # Transformation from the model to the world space.
                T_world_from_model = clip_util.se3_from_dict(obj_data["T_world_from_object"])

                # get object pose in camera frame
                T_camera_from_model = np.linalg.inv(T_world_from_camera) @ T_world_from_model

                object_frame_scene_gt_anno = {
                    "obj_id": int(obj_key),
                    "cam_R_m2c": T_camera_from_model[:3, :3].flatten().tolist(),
                    "cam_t_m2c": (T_camera_from_model[:3, 3] * 1000).tolist(),
                }

                #print('frame_id', frame_id, 'anno_id', anno_id, 'stream_id', stream_id, 'obj_key', obj_key)
                #if frame_id == 8 and anno_id == 1 and stream_id == '1201-2':
                #    print('break')

                # read amodal masks
                rle_dict = obj_data['masks_amodal'][stream_id]
                if not rle_dict['rle']:
                    # if 'rle' is an empty list, continue to the next object
                    continue
                else:
                    mask = custom_rle_to_mask(rle_dict['height'], rle_dict['width'], rle_dict['rle'])
                    mask = Image.fromarray(mask * 255)
                    mask = mask.convert("L")

                # read modal mask
                rle_dict = obj_data['masks_modal'][stream_id]
                # if 'rle' is an empty list, make an empty mask
                if not rle_dict['rle']:
                    mask_visib = Image.new("L", (rle_dict['width'], rle_dict['height']), 0)
                else:
                    mask_visib = custom_rle_to_mask(rle_dict['height'], rle_dict['width'], rle_dict['rle'])
                    mask_visib = Image.fromarray(mask_visib * 255)
                    mask_visib = mask_visib.convert("L")

                anno_id = f"{anno_id:06d}"

                # save mask FRAME-ID_ANNO-ID.png
                mask_path = os.path.join(clip_stream_paths[f"mask_{stream_name}"], frame_key+"_"+anno_id+".png")
                # save mask
                mask.save(mask_path)
                # save mask_visib FRAME-ID_ANNO-ID.png
                mask_visib_path = os.path.join(clip_stream_paths[f"mask_visib_{stream_name}"], frame_key+"_"+anno_id+".png")
                # save mask_visib
                mask_visib.save(mask_visib_path)

                # add scene_gt_info data
                # calculate bbox from mask with cv2 (x, y, width, height)
                bbox_obj = obj_data['boxes_amodal'][stream_id]
                bbox_obj = [int(val) for val in bbox_obj]
                # bbox_visib
                x_min, y_min, x_max, y_max = mask.getbbox()
                bbox_visib = [x_min, y_min, x_max - x_min, y_max - y_min]
                px_count_all = cv2.countNonZero(np.array(mask))
                px_count_visib = cv2.countNonZero(np.array(mask_visib))
                # visibile fraction
                visibilities_modeled = obj_data['visibilities_modeled'][stream_id]
                visibilities_predicted = obj_data['visibilities_predicted'][stream_id]
                visib_fract = min(visibilities_modeled, visibilities_predicted)
                object_frame_scene_gt_info_anno = {
                    "bbox_obj": bbox_obj,
                    "bbox_visib": bbox_visib,
                    "px_count_all": px_count_all,
                    "px_count_valid": px_count_all,  # excluded as Hot3D is RGB only
                    "px_count_visib": px_count_visib,
                    "visib_fract": visib_fract,
                }

                frame_scene_gt_data.append(object_frame_scene_gt_anno)
                frame_scene_gt_info_data.append(object_frame_scene_gt_info_anno)

            scene_gt_data[stream_name][int(frame_id)] = frame_scene_gt_data
            scene_gt_info_data[stream_name][int(frame_id)] = frame_scene_gt_info_data

    # save scene_gt.json, scene_gt_info.json, scene_camera.json for each camera stream
    for stream_name in args.camera_streams_names:
        with open(clip_stream_paths[f"scene_camera_{stream_name}"], "w") as f:
            json.dump(scene_camera_data[stream_name], f, indent=4)
        with open(clip_stream_paths[f"scene_gt_{stream_name}"], "w") as f:
            json.dump(scene_gt_data[stream_name], f, indent=4)
        with open(clip_stream_paths[f"scene_gt_info_{stream_name}"], "w") as f:
            json.dump(scene_gt_info_data[stream_name], f, indent=4)


def custom_rle_to_mask(height, width, rle):
    """
    Convert custom RLE (Run-Length Encoding) to a binary mask using vectorized operations.

    Parameters:
    - height (int): The height of the mask.
    - width (int): The width of the mask.
    - rle (list): The custom RLE list [start, length, start, length, ...].

    Returns:
    - np.ndarray: The binary mask.
    """
    # Create an empty mask
    mask = np.zeros(height * width, dtype=np.uint8)

    # Convert RLE pairs into start and end indices
    starts = np.array(rle[0::2])
    lengths = np.array(rle[1::2])
    ends = starts + lengths

    # Create an array of indices corresponding to the runs
    run_lengths = np.concatenate([np.arange(start, end) for start, end in zip(starts, ends)])

    # Set those indices in the mask to 1
    mask[run_lengths] = 1

    # Reshape the flat array into a 2D mask
    return mask.reshape((height, width))


if __name__ == "__main__":
    main()
