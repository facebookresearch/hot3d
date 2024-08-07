"""
This script converts the model from the original GLT format used in Hot3D to the PLY format as in the standard BOP format.
"""

import trimesh
import numpy as np
from PIL import Image
import os


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
    with open(ply_filepath, 'w') as ply_file:
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
        texture_image = np.ones((2048, 2048, 3), dtype=np.uint8) * main_color.astype(np.uint8)

    # Convert the texture to a PIL Image and save as PNG
    image = Image.fromarray(np.array(texture_image))
    image.save(texture_filepath)
    print(f"Texture saved as {texture_filepath}")


if __name__ == "__main__":
    gltf_dir = "/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models"
    output_dir = "/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_ply"

    # make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for gltf_filename in os.listdir(gltf_dir):
        if gltf_filename.endswith(".glb"):
            gltf_filepath = os.path.join(gltf_dir, gltf_filename)
            ply_filepath = os.path.join(output_dir , gltf_filename.replace(".glb", ".ply"))
            texture_filepath = os.path.join(output_dir , gltf_filename.replace(".glb", ".png"))

            # Save mesh as PLY and texture as PNG
            save_mesh_as_ply_with_uv_and_texture(gltf_filepath, ply_filepath, texture_filepath)

