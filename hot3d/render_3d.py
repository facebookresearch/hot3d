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

#
# Code Sample
#
# Installation:
# - pip install pyrender trimesh
#
# Details:
# Demonstrate how to use PyRender to render HOT3D meshes (objects, hands) for a given timestamp & stream_id
# - As OpenGL rendering is rectilinear, color, segmentation and depth buffer are also rectilinear rendering
# - We then show how to map the rectilinear image back to the original fisheye image (but do not we are loosing some field of view)

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
    from pyrender import (
        IntrinsicsCamera,
        Mesh,
        Node,
        OffscreenRenderer,
        RenderFlags,
        Scene,
    )
except ImportError:
    print("trimesh or pyrender modules are missing. Please install them.")

from data_loaders.HandDataProviderBase import HandDataProviderBase
from data_loaders.headsets import Headset
from data_loaders.loader_object_library import load_object_library, ObjectLibrary
from dataset_api import Hot3dDataProvider
from PIL import Image
from projectaria_tools.core.calibration import (
    CameraCalibration,
    distort_by_calibration,
    FISHEYE624,
    LINEAR,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId

from tqdm import tqdm

# Matrix transform to change Aria camera pose to PyRender coordinate system
# PyRender: +Z = back, +Y = up, +X = right
# Aria: +Z = forward, +Y = down, +X = right
T_ARIA_OPENGL = SE3.from_matrix(
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
)

ACCEPTABLE_TIME_DELTA = 0  # To retrieve exact GT


def load_meshes_scene(
    hot3d_data_provider: Hot3dDataProvider,
) -> Dict[str, Mesh]:
    """
    Load all meshes in the scene and hash them by object_uid
    """

    object_library = hot3d_data_provider.object_library
    object_library_folderpath = object_library.asset_folder_name

    object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
    object_uids = object_pose_data_provider.object_uids_with_poses

    #
    # Load all meshes in the scene and store them in a dict
    #
    meshes: Dict[str, Mesh] = {}
    for object_uid in tqdm(object_uids):
        object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
            object_library_folderpath=object_library_folderpath,
            object_id=object_uid,
        )
        # Load the mesh, merge its component
        scene = trimesh.load_mesh(
            object_cad_asset_filepath,
            process=True,
            merge_primitives=True,
            file_type="glb",
        )
        # Represent the scene by a single mesh
        glb_mesh = scene.to_mesh()
        # Store the resulting mesh in the dict
        meshes[object_uid] = Mesh.from_trimesh(glb_mesh)

    return meshes


def setup_objects_at_timestamp(
    scene: Scene,
    meshes: Dict[str, Mesh],
    hot3d_data_provider: Hot3dDataProvider,
    timestamp_ns: int,
) -> Dict[str, Node]:
    """
    Setup object meshes in the scene for the specified timestamp
    """

    object_pose_data_provider = hot3d_data_provider.object_pose_data_provider

    pyrender_node_meshes = {}
    object_poses_with_dt = None
    if object_pose_data_provider is not None:
        object_poses_with_dt = object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=ACCEPTABLE_TIME_DELTA,
        )
        if object_poses_with_dt is not None:
            objects_pose3d_collection = object_poses_with_dt.pose3d_collection
            for (
                object_uid,
                object_pose3d,
            ) in objects_pose3d_collection.poses.items():
                transform = object_pose3d.T_world_object.to_matrix()
                pyrender_node_meshes[object_uid] = scene.add(
                    meshes[object_uid], pose=transform
                )

    return pyrender_node_meshes


def get_camera_calibration(
    hot3d_data_provider: Hot3dDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
    camera_model=LINEAR,
) -> Optional[Tuple[SE3, CameraCalibration]]:
    """
    Return the camera calibration
    """
    device_data_provider = hot3d_data_provider.device_data_provider
    if hot3d_data_provider.get_device_type() is Headset.Aria:
        return device_data_provider.get_online_camera_calibration(
            stream_id=stream_id,
            timestamp_ns=timestamp_ns,
            camera_model=camera_model,
        )
    elif hot3d_data_provider.get_device_type() is Headset.Quest3:
        return device_data_provider.get_camera_calibration(
            stream_id=stream_id,
            camera_model=camera_model,
        )
    else:
        return None


def setup_camera_at_timestamp(
    scene: Scene,
    hot3d_data_provider: Hot3dDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
) -> Tuple[Node, List[int]]:
    """
    Setup a rectilinear camera for the specified stream_id and timestamp
    """

    device_data_provider = hot3d_data_provider.device_data_provider
    device_pose_provider = hot3d_data_provider.device_pose_data_provider

    [T_device_camera, intrinsics] = get_camera_calibration(
        hot3d_data_provider=hot3d_data_provider,
        stream_id=stream_id,
        timestamp_ns=timestamp_ns,
        camera_model=LINEAR,
    )

    headset_pose3d_with_dt = None
    if device_data_provider is not None:
        headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=ACCEPTABLE_TIME_DELTA,
        )
        if headset_pose3d_with_dt is not None:
            headset_pose3d = headset_pose3d_with_dt.pose3d
            focal_lengths = intrinsics.get_focal_lengths()
            principal_point = intrinsics.get_principal_point()
            camera = IntrinsicsCamera(
                focal_lengths[0],
                focal_lengths[0],
                principal_point[0],
                principal_point[1],
                znear=0.05,
                zfar=100.0,
                name=None,
            )

            camera_pose = (
                (headset_pose3d.T_world_device @ T_device_camera) @ T_ARIA_OPENGL
            ).to_matrix()

            camera_node = scene.add(camera, pose=camera_pose)
    return [camera_node, intrinsics.get_image_size().tolist()]


def setup_hand_at_timestamp(
    scene: Scene,
    hot3d_data_provider: Hot3dDataProvider,
    timestamp_ns: int,
    hand_data_provider: HandDataProviderBase,
) -> Dict[str, Mesh]:
    """
    Add hand meshes to the scene for the specified timestamp
    """

    pyrender_node_meshes = {}

    if hand_data_provider is None:
        return []

    hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
        acceptable_time_delta=ACCEPTABLE_TIME_DELTA,
    )
    if hand_poses_with_dt is not None:
        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        for hand_pose_data in hand_pose_collection.poses.values():
            handedness_label = hand_pose_data.handedness_label()

            hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(
                hand_pose_data
            )

            [hand_triangles, hand_vertex_normals] = (
                hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)
            )

            pyrender_node_meshes[handedness_label] = scene.add(
                Mesh.from_trimesh(
                    trimesh.Trimesh(
                        vertices=hand_mesh_vertices,
                        normals=hand_vertex_normals,
                        faces=hand_triangles,
                    )
                )
            )
    return pyrender_node_meshes


def offscreen_render(
    scene: Scene,
    resolution: List[int],  # [width, height]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return COLOR and DEPTH images
    """
    renderer = OffscreenRenderer(resolution[0], resolution[1])
    color, depth = renderer.render(scene)  # , flags=RenderFlags.RGBA)

    nm = {
        node: 20 * (i + 1) for i, node in enumerate(scene.mesh_nodes)
    }  # Node->Seg Id map
    seg = renderer.render(scene, RenderFlags.SEG, nm)[0]

    renderer.delete()
    return [color, depth, seg]


def distort_rendering(
    image: np.ndarray,
    hot3d_data_provider: Hot3dDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
) -> np.ndarray:
    """
    Map a rectilinear image to the native Fisheye camera model.
    - Do notice that we are loosing some field of view.
    """

    # Retrieve the camera model we want to distort to
    [T_device_camera, intrinsics_raw] = get_camera_calibration(
        hot3d_data_provider=hot3d_data_provider,
        stream_id=stream_id,
        timestamp_ns=timestamp_ns,
        camera_model=FISHEYE624,
    )

    # Retrieve the camera model we used for rendering
    [T_device_camera, intrinsics_linear] = get_camera_calibration(
        hot3d_data_provider=hot3d_data_provider,
        stream_id=stream_id,
        timestamp_ns=timestamp_ns,
        camera_model=LINEAR,
    )

    re_distorted_image = distort_by_calibration(
        image,
        intrinsics_raw,
        intrinsics_linear,
    )

    return re_distorted_image


#
# Initialize hot3d data provider
#
# For simplicity we are using the data from the data_sample folder
# But you can replace this with any HOT3D sequence

object_library = load_object_library(
    object_library_folderpath="./data_loaders/tests/data_sample/object_library"
)

hot3d_data_provider = Hot3dDataProvider(
    sequence_folder="./data_loaders/tests/data_sample/Aria/P0003_c701bd11",
    # sequence_folder="./data_loaders/tests/data_sample/Quest3/P0002_273c2819",
    object_library=object_library,
)
print(f"data_provider statistics: {hot3d_data_provider.get_data_statistics()}")

# Load scene meshes
print("Loading meshes...")
scene_meshes = load_meshes_scene(hot3d_data_provider)

# Define timestamps and stream ids that need rendering
# Default attempts all stream ids and a timestamp in the middle of the sequence
timestamps = hot3d_data_provider.device_data_provider.get_sequence_timestamps()
timestamp_list = [timestamps[len(timestamps) // 2]]
stream_id_list = (
    [StreamId("1201-1"), StreamId("1201-2"), StreamId("214-1")]
    if hot3d_data_provider.get_device_type() is Headset.Aria
    else [StreamId("1201-1"), StreamId("1201-2")]
)

# Main rendering loop
print(f"Rendering for: {stream_id_list}")
for stream_id in stream_id_list:
    for timestamp_ns in timestamp_list:
        # Initialize the scene
        scene = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))

        # Add objects
        setup_objects_at_timestamp(
            scene=scene,
            meshes=scene_meshes,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
        )

        # Add hands
        hands_meshes = setup_hand_at_timestamp(
            scene=scene,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
            hand_data_provider=hot3d_data_provider.umetrack_hand_data_provider,
        )

        # Setup camera rendering (for the specific stream_id and timestamp)
        camera_node_and_resolution = setup_camera_at_timestamp(
            scene=scene,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
            stream_id=stream_id,
        )

        print(camera_node_and_resolution)
        # Setup off screen rendering (to save rendering buffer to disk as image)
        [color, depth, seg] = offscreen_render(scene, camera_node_and_resolution[1])

        im = Image.fromarray(color)
        im.save(f"render_native_{stream_id}_{timestamp_ns}.png")

        camera_node_and_resolution = setup_camera_at_timestamp(
            scene=scene,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
            stream_id=stream_id,
        )

        # Save the rectilinear/pinhole rendering color buffer
        im = Image.fromarray(color)
        im.save(f"render_native_{stream_id}_{timestamp_ns}.png")

        #
        # Show how to distort the rendering to match the device FishEye camera calibration
        #
        # - Do note that we are missing some field of view in the distorted image
        #   - To avoid such effect, we would need to render a pinhole image with a larger field of view

        # Save "color" buffer and map it to the native Fisheye camera model
        distorted_color = distort_rendering(
            image=color,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
            stream_id=stream_id,
        )

        im = Image.fromarray(distorted_color)
        im.save(f"render_ref_{stream_id}_{timestamp_ns}.png")

        # Save "segmentation id" buffer and map it to the native Fisheye camera model
        distorted_seg = distort_rendering(
            image=seg,
            hot3d_data_provider=hot3d_data_provider,
            timestamp_ns=timestamp_ns,
            stream_id=stream_id,
        )

        im = Image.fromarray(distorted_seg)
        im.save(f"seg_ref_{stream_id}_{timestamp_ns}.png")

        # Save "depth" buffer
        im = Image.fromarray(depth)
        im.save(f"depth_ref_{stream_id}_{timestamp_ns}.tiff")

        # Save raw input image for comparison and overlay
        image_data_raw = hot3d_data_provider.device_data_provider.get_image(
            timestamp_ns, stream_id
        )
        if image_data_raw is not None:
            im = Image.fromarray(image_data_raw)
            im.save(f"image_ref_{stream_id}_{timestamp_ns}.png")
