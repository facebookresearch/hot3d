"""
This script converts the Hot3D-Clips dataset used for the BOP challenge 2024 to the standard BOP format
"""

import os
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset_path", type=str, default="/home/gouda/Downloads/hot3d/hot3d_native_format")
    #parser.add_argument("--dataset_path", type=str, required=True)
    # BOP dataset split name
    parser.add_argument("--split", type=str, default="train_aria")
    # output directory
    parser.add_argument("--output_dataset_path", type=str, default="/home/gouda/Downloads/hot3d/hot3d_BOP_format")

    args = parser.parse_args()

    # paths
    clips_input_dir = os.path.join(args.input_dataset_path, args.split)
    scenes_output_dir = os.path.join(args.output_dataset_path, args.split)
    tmp_tar_dir = os.path.join(args.output_dataset_path, "tmp_tar")

    # list all clips names in the dataset
    split_clips = sorted([p for p in os.listdir(clips_input_dir) if p.endswith(".tar")])

    # create output directory
    os.makedirs(scenes_output_dir, exist_ok=True)  # TODO change exist_ok to False

    # create tmp directory for extracting clips from the webdataset format
    os.makedirs(tmp_tar_dir, exist_ok=True)  # TODO change exist_ok to False

    # loop over all clips
    for clip in split_clips:  # TODO add tqdm
        # get clip id
        clip_id = clip.split(".")[0].split("-")[1]

        # extract clip
        clip_extract_dir = os.path.join(tmp_tar_dir, clip_id)
        os.makedirs(clip_extract_dir, exist_ok=True)
        os.system(f"tar -xf {os.path.join(clips_input_dir, clip)} -C {clip_extract_dir}")

        # make scene folder and files for the scene
        scene_output_dir = os.path.join(scenes_output_dir, clip_id)
        os.makedirs(scene_output_dir, exist_ok=True)

        rgb_dir = os.path.join(scene_output_dir, "rgb")
        # make 2 folders that are not in the BOP format
        monocular_left = os.path.join(scene_output_dir, "monocular_left")
        monocular_right = os.path.join(scene_output_dir, "monocular_right")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(monocular_left, exist_ok=True)
        os.makedirs(monocular_right, exist_ok=True)

        scene_gt_json_path = os.path.join(scene_output_dir, "scene_gt.json")
        scene_camera_json_path = os.path.join(scene_output_dir, "scene_camera.json")
        #scene_gt_info_json_path = os.path.join(args.dataset_path, args.split, "scene_gt_info.json")  # TODO if cannot be generated with BOP toolkit create it in this script.


        # create scene_gt_info.json
        scene_gt_data = {}
        scene_camera_data = {}
        #"scene_gt_info" = {}  # TODO if cannot be generated with BOP toolkit create it in this script.
        
        # get frames ids of all frames in the clip - read all files with *.cameras.json
        frames_ids = sorted([f.split(".")[0] for f in os.listdir(clip_extract_dir) if f.endswith(".cameras.json")])

        # loop over all frames
        for frame_id in frames_ids:
            # copy rgb image
            os.system(f"cp {os.path.join(clip_extract_dir, frame_id)}.image_214-1.jpg {os.path.join(rgb_dir, frame_id.zfill(6))}.jpg")
            # copy monocular images  # TODO not in BOP format - should i ignore them?
            os.system(f"cp {os.path.join(clip_extract_dir, frame_id)}.image_1201-1.jpg {os.path.join(monocular_left, frame_id.zfill(6))}.jpg")
            os.system(f"cp {os.path.join(clip_extract_dir, frame_id)}.image_1201-2.jpg {os.path.join(monocular_right, frame_id.zfill(6))}.jpg")

            # filling scene_gt data
            ## read FRAME_ID.cameras.json
            with open(os.path.join(clip_extract_dir, f"{frame_id}.cameras.json"), "r") as f:
                frame_camera = json.load(f)
            ## read FRAME_ID.objects.json
            with open(os.path.join(clip_extract_dir, f"{frame_id}.objects.json"), "r") as f:
                frame_objects = json.load(f)

            # get T_world_from_camera
            translation_world_from_camera = frame_camera["214-1"]["T_world_from_camera"]["translation_xyz"]
            rotation_world_from_camera = frame_camera["214-1"]["T_world_from_camera"]["quaternion_wxyz"]
            # order of rotation is wxyz, convert to xyzw as SciPy uses
            rotation_world_from_camera = [rotation_world_from_camera[1], rotation_world_from_camera[2], rotation_world_from_camera[3], rotation_world_from_camera[0]]
            # convert rotation to rotation matrix (use SciPy)
            rotation_world_from_camera = R.from_quat(rotation_world_from_camera).as_matrix()
            # get T_camera_from_world
            T_world_from_camera = np.eye(4)
            T_world_from_camera[:3, :3] = rotation_world_from_camera
            T_world_from_camera[:3, 3] = translation_world_from_camera
            # get T_world_2_camera as in the BOP format
            T_world_2_camera = np.eye(4)
            T_world_2_camera[:3, :3] = rotation_world_from_camera.T
            T_world_2_camera[:3, 3] = -rotation_world_from_camera.T @ translation_world_from_camera

            # add frame scene_camera data
            scene_camera_data[int(frame_id)] = {
                # TODO I need to check how the Fisheye model is going to be added to the BOP toolkit
                "cam_model": "FISHEYE624",
                "cam_K": frame_camera["214-1"]["calibration"]["projection_params"],
                "depth_scale": 1.0,  # TODO check if Hot3D is in mm
                "cam_t_w2c": T_world_2_camera[:3, 3].tolist(),
                "cam_R_w2c": T_world_2_camera[:3, :3].tolist()
            }

            frame_objects_data = []
            for obj_key in frame_objects:
                obj_data = frame_objects[obj_key][0]
                # read object pose
                translation_world_from_object = obj_data["T_world_from_object"]["translation_xyz"]
                rotation_world_from_object = obj_data["T_world_from_object"]["quaternion_wxyz"]
                # order of rotation is wxyz, convert to xyzw as SciPy uses
                rotation_world_from_object = [rotation_world_from_object[1], rotation_world_from_object[2], rotation_world_from_object[3], rotation_world_from_object[0]]
                T_world_from_object = np.eye(4)
                T_world_from_object[:3, 3] = translation_world_from_object
                T_world_from_object[:3, :3] = R.from_quat(rotation_world_from_object).as_matrix()

                # get object pose in camera frame
                T_object_to_camera = T_world_2_camera @ T_world_from_object

                object_bop_anno = {
                    "obj_id": obj_key,
                    "cam_R_m2c": T_object_to_camera[:3, :3].tolist(),
                    "cam_t_m2c": T_object_to_camera[:3, 3].tolist(),
                }
                frame_objects_data.append(object_bop_anno)

                # create masks using the vis script


            scene_gt_data[int(frame_id)] = frame_objects_data

        # save scene_gt.json
        with open(scene_gt_json_path, "w") as f:
            json.dump(scene_gt_data, f)
        # save scene_camera.json
        with open(scene_camera_json_path, "w") as f:
            json.dump(scene_camera_data, f)

        # remove extracted clip
        os.system(f"rm -r {clip_extract_dir}")

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

        # debug check transformation of object pose in each camera frame
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        # make a loop with all frames
        for frame_id in frames_ids:
            vis.create_window()
            # create origin
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(origin)
            # add a sphere at the origin in red color
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate([0, 0, 0])
            vis.add_geometry(sphere)

            # create camera pose
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # get T_world_2_camera from scene_camera_data
            T_world_2_camera = np.eye(4)
            T_world_2_camera[:3, :3] = np.array(scene_camera_data[int(frame_id)]["cam_R_w2c"])
            T_world_2_camera[:3, 3] = np.array(scene_camera_data[int(frame_id)]["cam_t_w2c"])
            # transform camera_frame to the camera pose
            camera_frame.transform(T_world_2_camera)
            # add sphere at the camera origin in green color
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(T_world_2_camera[:3, 3])
            sphere.paint_uniform_color([0, 1, 0])
            vis.add_geometry(sphere)
            # get object pose in the camera frame
            vis.add_geometry(camera_frame, reset_bounding_box=True)
            for obj_key in scene_gt_data[int(frame_id)]:
                # create object pose
                object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                T_object_to_camera= np.eye(4)
                T_object_to_camera[:3, :3] = np.array(obj_key["cam_R_m2c"])
                T_object_to_camera[:3, 3] = np.array(obj_key["cam_t_m2c"])
                object_frame.transform(T_object_to_camera)
                vis.add_geometry(object_frame, reset_bounding_box=True)
                # create sphere at the object origin in blue color
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(T_object_to_camera[:3, 3])
                sphere.paint_uniform_color([0, 0, 1])
                vis.add_geometry(sphere)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.clear_geometries()

    # remove tmp directory  - TODO uncomment
    #os.system(f"rm -r {tmp_tar_dir}")



if __name__ == "__main__":
    main()
