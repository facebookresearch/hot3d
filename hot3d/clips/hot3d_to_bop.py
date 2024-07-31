"""
This script converts the Hot3D-Clips dataset used for the BOP challenge 2024 to the standard BOP format
"""

import os
import argparse
import json
import cv2
import numpy as np
import tarfile
from typing import Any, Dict, List, Optional
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import clip_util
from hand_tracking_toolkit import rasterizer


RGB_STREAM = "214-1"

def main():
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset_path", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/")
    #parser.add_argument("--dataset_path", type=str, required=True)
    # BOP dataset split name
    parser.add_argument("--split", type=str, default="train_aria_subsample")
    # output directory
    parser.add_argument("--output_dataset_path", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/train_aria_subsample_bop")
    # object models directory
    parser.add_argument("--object_models_dir", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval")
    # list of camera streams
    parser.add_argument("--camera_streams_id", type=str, nargs="+", default=["214-1", "1201-1", "1201-2"])
    # list of camera streams names
    parser.add_argument("--camera_streams_names", type=str, nargs="+", default=["rgb", "gray1", "gray2"])

    args = parser.parse_args()

    # paths
    clips_input_dir = os.path.join(args.input_dataset_path, args.split)
    scenes_output_dir = os.path.join(args.output_dataset_path, args.split)

    # Load object models.
    object_models: Dict[int, trimesh.Trimesh] = {}
    object_model_filenames = sorted(
        [p for p in os.listdir(args.object_models_dir) if p.endswith(".glb")]
    )
    #for model_filename in object_model_filenames[0:2]:  # TODO delete debug - load all models
    for model_filename in object_model_filenames:
        model_path = os.path.join(args.object_models_dir, model_filename)
        print(f"Loading model: {model_path}")
        object_id = int(model_filename.split(".glb")[0].split("obj_")[1])
        object_models[object_id] = clip_util.load_mesh(model_path)

    # list all clips names in the dataset
    split_clips = sorted([p for p in os.listdir(clips_input_dir) if p.endswith(".tar")])

    # create output directory
    os.makedirs(scenes_output_dir, exist_ok=True)  # TODO change exist_ok to False

    # loop over all clips
    for clip in split_clips:  # TODO add tqdm
        print(f"Processing clip: {clip}")
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
        for frame_id in tqdm(range(clip_util.get_number_of_frames(tar))):
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
                T_camera_to_world = frame_camera[stream_id].T_world_from_eye

                # TODO check that inverting this homogenous transformation is correct
                # get T_world_2_camera as in the BOP format
                T_world_to_camera = np.eye(4)
                T_world_to_camera[:3, :3] = T_camera_to_world[:3, :3]
                T_world_to_camera[:3, 3] = -T_camera_to_world[:3, :3].T @ T_camera_to_world[:3, 3]

                # get camera parameters
                calibration = frame_camera_data[stream_id]["calibration"]

                # add frame scene_camera data
                scene_camera_data[stream_name][int(frame_id)] = {
                    "scene_camera": {
                        # TODO change this after we agree on the final format
                        "cam_model": calibration,
                        #"cam_K":  # not used as cam_model exists
                        #"depth_scale":  # also not used
                        "cam_t_w2c": T_world_to_camera[:3, 3].flatten().tolist(),
                        "cam_R_w2c": T_world_to_camera[:3, :3].flatten().tolist(),
                    }
                }

                # Camera parameters of the current image.
                camera_model = frame_camera[stream_id]

                frame_scene_gt_data = []
                frame_scene_gt_info_data = []
                # loop with enumerate over all objects in the frame
                for anno_id, obj_key in enumerate(frame_objects):
                    obj_data = frame_objects[obj_key][0]
                    bop_id = int(obj_data["object_bop_id"])

                    # Transformation from the model to the world space.
                    T_object_to_world = clip_util.se3_from_dict(obj_data["T_world_from_object"])

                    # get object pose in camera frame
                    T_camera_to_object = T_world_to_camera @ T_object_to_world

                    object_frame_scene_gt_anno = {
                        "obj_id": obj_key,
                        "cam_R_m2c": T_camera_to_object[:3, :3].flatten().tolist(),
                        "cam_t_m2c": T_camera_to_object[:3, 3].flatten().tolist(),
                    }

                    # Transformation from the model to the world space.
                    T = clip_util.se3_from_dict(obj_data["T_world_from_object"])

                    # Vertices in the model space.
                    verts_in_m = object_models[bop_id].vertices

                    # Vertices in the world space (can be brought to the camera
                    # space by the inverse of camera_model.T_world_from_eye).
                    verts_in_w = (T[:3, :3] @ verts_in_m.T + T[:3, 3:]).T

                    # Render the object model (outputs: rgb, mask, depth).
                    _, mask, _ = rasterizer.rasterize_mesh(
                        verts=verts_in_w,
                        faces=object_models[bop_id].faces,
                        vert_normals=object_models[bop_id].vertex_normals,
                        camera=camera_model,
                    )

                    # if no pixel is one in the mask, skip this object
                    if np.count_nonzero(mask) == 0:
                        continue

                    mask *= 255

                    anno_id = f"{anno_id:06d}"

                    # save mask FRAME-ID_ANNO-ID.png
                    mask_path = os.path.join(clip_stream_paths[f"mask_{stream_name}"], frame_key+"_"+anno_id+".png")
                    # save mask
                    cv2.imwrite(mask_path, mask)
                    # TODO for now save mask as mask_visib - change it later after generating the correct mask_visib
                    # save mask_visib FRAME-ID_ANNO-ID.png
                    mask_visib_path = os.path.join(clip_stream_paths[f"mask_visib_{stream_name}"], frame_key+"_"+anno_id+".png")
                    # save mask_visib
                    cv2.imwrite(mask_visib_path, mask)

                    # add scene_gt_info data
                    ## calculate bbox from mask with cv2 (x, y, width, height
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    bbox_obj = [x, y, w, h]
                    bbox_visib = bbox_obj  # TODO change to visib mask after getting it for Hot3D
                    px_count_all = cv2.countNonZero(mask)
                    px_count_visib = px_count_all  # TODO change to visib mask after getting it for Hot3D
                    visib_fract = px_count_visib / px_count_all
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

        # debug - check world to camera transformation
        # plot the camera pose in the world frame
        # make a loop with all frames
        # user click enter to add the next frame
        # use open3d visualization
        # ------
        #import open3d as o3d
        #import time
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        ## create origin
        #origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #vis.add_geometry(origin)
        ## add a sphere at the origin
        #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #sphere.translate([0, 0, 0])
        #vis.add_geometry(sphere)
        #vis.poll_events()
        #vis.update_renderer()
        ## make a loop with all frames
        #for frame_id in frames_ids[0:15]:
        #    # create camera pose
        #    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #    # color first frame yellow and the last in loop black
        #    if frame_id == frames_ids[0]:
        #        camera_frame.paint_uniform_color([1, 1, 0])
        #    elif frame_id == frames_ids[14]:
        #        camera_frame.paint_uniform_color([0, 0, 0])
        #    # get T_world_2_camera from scene_camera_data
        #    T_world_2_camera = np.eye(4)
        #    T_world_2_camera[:3, :3] = np.array(scene_camera_data[int(frame_id)]["cam_R_w2c"])
        #    T_world_2_camera[:3, 3] = np.array(scene_camera_data[int(frame_id)]["cam_t_w2c"])
        #    # transform camera_frame to the camera pose
        #    camera_frame.transform(T_world_2_camera)
        #    vis.add_geometry(camera_frame, reset_bounding_box=True)
        #    vis.poll_events()
        #    vis.update_renderer()
        #    # sleep 1 second
        #    #time.sleep(1)
        #vis.run()
        #vis.destroy_window()
        # ----

        ## debug check transformation of object pose in each camera frame
        #import open3d as o3d
        #vis = o3d.visualization.Visualizer()
        ## make a loop with all frames
        #for frame_id in frames_ids:
        #    vis.create_window()
        #    # create origin
        #    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #    vis.add_geometry(origin)
        #    # add a sphere at the origin in red color
        #    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #    sphere.translate([0, 0, 0])
        #    vis.add_geometry(sphere)

        #    # create camera pose
        #    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #    # get T_world_2_camera from scene_camera_data
        #    T_world_2_camera = np.eye(4)
        #    T_world_2_camera[:3, :3] = np.array(scene_camera_data[int(frame_id)]["cam_R_w2c"])
        #    T_world_2_camera[:3, 3] = np.array(scene_camera_data[int(frame_id)]["cam_t_w2c"])
        #    # transform camera_frame to the camera pose
        #    camera_frame.transform(T_world_2_camera)
        #    # add sphere at the camera origin in green color
        #    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #    sphere.translate(T_world_2_camera[:3, 3])
        #    sphere.paint_uniform_color([0, 1, 0])
        #    vis.add_geometry(sphere)
        #    # get object pose in the camera frame
        #    vis.add_geometry(camera_frame, reset_bounding_box=True)
        #    for obj_key in scene_gt_data[int(frame_id)]:
        #        # create object pose
        #        object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #        T_object_to_camera= np.eye(4)
        #        T_object_to_camera[:3, :3] = np.array(obj_key["cam_R_m2c"])
        #        T_object_to_camera[:3, 3] = np.array(obj_key["cam_t_m2c"])
        #        object_frame.transform(T_object_to_camera)
        #        vis.add_geometry(object_frame, reset_bounding_box=True)
        #        # create sphere at the object origin in blue color
        #        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #        sphere.translate(T_object_to_camera[:3, 3])
        #        sphere.paint_uniform_color([0, 0, 1])
        #        vis.add_geometry(sphere)
        #    vis.poll_events()
        #    vis.update_renderer()
        #    vis.run()
        #    vis.clear_geometries()

    # remove tmp directory  - TODO uncomment
    #os.system(f"rm -r {tmp_tar_dir}")


if __name__ == "__main__":
    main()
