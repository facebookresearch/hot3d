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

"""
This script converts the object's models from the original GLTF format used in Hot3D to the PLY format as in the standard BOP format.
Note: the models_info.json file should be copied from the HOT3D dataset model directory to the output directory.
      The data in this native models_info.json file actually contains more data than the standard BOP models_info.json.
"""

import argparse
import os

import numpy as np
import trimesh
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    # add arg gltf dir and output dir
    parser.add_argument("--input-gltf-dir", required=True, type=str)
    parser.add_argument("--output-bop-dir", required=True, type=str)

    args = parser.parse_args()

    # make the output directory
    os.makedirs(args.output_bop_dir, exist_ok=False)

    for gltf_filename in os.listdir(args.input_gltf_dir):
        if gltf_filename.endswith(".glb"):
            gltf_filepath = os.path.join(args.input_gltf_dir, gltf_filename)
            ply_filepath = os.path.join(
                args.output_bop_dir, gltf_filename.replace(".glb", ".ply")
            )
            texture_filepath = os.path.join(
                args.output_bop_dir, gltf_filename.replace(".glb", ".png")
            )

            # Save mesh as PLY and texture as PNG
            save_mesh_as_ply_with_uv_and_texture(
                gltf_filepath, ply_filepath, texture_filepath
            )


def save_mesh_as_ply_with_uv_and_texture(gltf_filepath, ply_filepath, texture_filepath):
    # Load the GLTF/GLB file using trimesh
    scene = trimesh.load(gltf_filepath, process=False, maintain_order=True)

    # Dump the scene to a single mesh
    mesh = scene.dump(concatenate=True)

    # Extract vertex positions, normals, and UVs
    vertices = mesh.vertices * 1000.0
    normals = mesh.vertex_normals

    # Handle cases where UV coordinates might be missing
    uv = mesh.visual.uv if mesh.visual.uv is not None else np.zeros((len(vertices), 2))

    # Prepare vertex data including UV coordinates
    vertex_data = np.hstack([vertices, normals, uv])

    # Prepare faces
    faces = mesh.faces

    # Create a PLY header with texture file comment
    header = f"""ply
format ascii 1.0
comment TextureFile {os.path.basename(texture_filepath)}
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float texture_u
property float texture_v
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""

    # Write the PLY file
    with open(ply_filepath, "w") as ply_file:
        # Write the header
        ply_file.write(header)

        # Write the vertex data
        for v in vertex_data:
            ply_file.write(f"{v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]} {v[6]} {v[7]}\n")

        # Write the face data
        for f in faces:
            ply_file.write(f"3 {f[0]} {f[1]} {f[2]}\n")

    print(f"Mesh saved as {ply_filepath}")

    # Save the texture as a PNG image
    if mesh.visual.material.to_simple().image is not None:
        mesh.visual.material = mesh.visual.material.to_simple()
        texture_image = mesh.visual.material.image

    else:
        print("No texture found in the GLTF file, using the main_color instead.")
        # make an image 2048x2048 with the main color
        main_color = mesh.visual.material.main_color[0:3]
        texture_image = np.ones((2048, 2048, 3), dtype=np.uint8) * main_color.astype(
            np.uint8
        )

    # Convert the texture to a PIL Image and save as PNG
    image = Image.fromarray(np.array(texture_image))
    image.save(texture_filepath)
    print(f"Texture saved as {texture_filepath}")


if __name__ == "__main__":
    main()
