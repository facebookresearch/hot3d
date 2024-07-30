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
    parser.add_argument("--split", type=str, default="train_aria_one")
    # output directory
    parser.add_argument("--output_dataset_path", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/train_aria_one_bop")
    # object models directory
    parser.add_argument("--object_models_dir", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval")

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
        # get clip id
        clip_name = clip.split(".")[0].split("-")[1]

        # extract clip
        tar = tarfile.open(os.path.join(clips_input_dir, clip), "r")

        # make scene folder and files for the scene
        scene_output_dir = os.path.join(scenes_output_dir, clip_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        rgb_dir = os.path.join(scene_output_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        mask_dir = os.path.join(scene_output_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        mask_visib_dir = os.path.join(scene_output_dir, "mask_visib")
        os.makedirs(mask_visib_dir, exist_ok=True)

        scene_gt_json_path = os.path.join(scene_output_dir, "scene_gt.json")
        scene_camera_json_path = os.path.join(scene_output_dir, "scene_camera.json")
        #scene_gt_info_json_path = os.path.join(args.dataset_path, args.split, "scene_gt_info.json")  # TODO if cannot be generated with BOP toolkit create it in this script.

        # create scene_gt_info.json
        scene_gt_data = {}
        scene_camera_data = {}
        #"scene_gt_info" = {}  # TODO if cannot be generated with BOP toolkit create it in this script.
        
        # loop over all frames
        for frame_id in range(clip_util.get_number_of_frames(tar)):
            frame_key = f"{frame_id:06d}"

            # get rgb image from the clip tar file
            rgb_image = clip_util.load_image(tar, frame_key, RGB_STREAM)
            # convert to BGR
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            # rotate the image 90 degrees
            rgb_image = np.rot90(rgb_image, k=3)
            # save rgb image
            cv2.imwrite(os.path.join(rgb_dir, frame_key+".jpg"), rgb_image)

            # filling scene_gt data

            # Load camera parameters.
            # from FRAME_ID.cameras.json
            frame_camera = clip_util.load_cameras(tar, frame_key)
            ## read FRAME_ID.objects.json
            frame_objects = clip_util.load_object_annotations(tar, frame_key)

            # get T_world_from_camera
            T_camera_to_world = frame_camera[RGB_STREAM].T_world_from_eye

            # TODO check that inverting this homogenous transformation is correct
            # get T_world_2_camera as in the BOP format
            T_world_to_camera = np.eye(4)
            T_world_to_camera[:3, :3] = T_camera_to_world[:3, :3]
            T_world_to_camera[:3, 3] = -T_camera_to_world[:3, :3].T @ T_camera_to_world[:3, 3]

            # TODO can i read this from the camera model?
            # read max_solid_angle and projection_params from FRAME_ID.cameras.json
            camera_json_file_name = f"{frame_id:06d}.cameras.json"
            camera_json_file = tar.extractfile(camera_json_file_name)
            frame_camera_data = json.load(camera_json_file)
            # get camera parameters
            image_height = frame_camera_data[RGB_STREAM]["calibration"]["image_height"]
            image_width = frame_camera_data[RGB_STREAM]["calibration"]["image_width"]
            max_solid_angle = frame_camera_data[RGB_STREAM]["calibration"]["max_solid_angle"]
            projection_params = frame_camera_data[RGB_STREAM]["calibration"]["projection_params"]

            # add frame scene_camera data
            scene_camera_data[int(frame_id)] = {
                # TODO change this after we agree on the final format
                "cam_model": {
                    "image_height": image_height,
                    "image_width": image_width,
                    "max_solid_angle": max_solid_angle,
                    "projection_params": projection_params,
                },
                #"cam_K":  # not used as cam_model exists
                #"depth_scale": 1.0,  # also not used
                "cam_t_w2c": T_world_to_camera[:3, 3].tolist(),
                "cam_R_w2c": T_world_to_camera[:3, :3].tolist()
            }

            # Camera parameters of the current image.
            camera_model = frame_camera[RGB_STREAM]

            frame_objects_data = []
            # loop with enumerate over all objects in the frame
            for anno_id, obj_key in enumerate(frame_objects):
                obj_data = frame_objects[obj_key][0]
                bop_id = int(obj_data["object_bop_id"])

                # Transformation from the model to the world space.
                T_object_to_world = clip_util.se3_from_dict(obj_data["T_world_from_object"])

                # get object pose in camera frame
                T_camera_to_object = T_camera_to_world @ np.linalg.inv(T_object_to_world)

                object_bop_anno = {
                    "obj_id": obj_key,
                    "cam_R_m2c": T_camera_to_object[:3, :3].tolist(),
                    "cam_t_m2c": T_camera_to_object[:3, 3].tolist(),
                }
                frame_objects_data.append(object_bop_anno)

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

                mask *= 255
                # rotate the mask 90 degrees
                mask = np.rot90(mask, k=3)

                anno_id = f"{anno_id:06d}"

                # save mask FRAME-ID_ANNO-ID.png
                mask_path = os.path.join(mask_dir, frame_key+"_"+anno_id+".png")
                # save mask
                cv2.imwrite(mask_path, mask)
                # TODO for now save mask as mask_visib - change it later after generating the correct mask_visib
                # save mask_visib FRAME-ID_ANNO-ID.png
                mask_visib_path = os.path.join(mask_visib_dir, frame_key+"_"+anno_id+".png")
                # save mask_visib
                cv2.imwrite(mask_visib_path, mask)

            scene_gt_data[int(frame_id)] = frame_objects_data

        # save scene_gt.json
        with open(scene_gt_json_path, "w") as f:
            json.dump(scene_gt_data, f)
        # save scene_camera.json
        with open(scene_camera_json_path, "w") as f:
            json.dump(scene_camera_data, f)

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
