
import glob
import os
import trimesh


src_path = "/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval"
dst_path = "/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval_bop"


def load_mesh(
        path: str,
) -> trimesh.Trimesh:
    """Loads a 3D mesh model from a specified path.

    Args:
        path: Path to the model to load.
    Returns.
        Loaded mesh.
    """

    # Load the scene.
    scene = trimesh.load_mesh(
        path,
        process=True,
        merge_primitives=True,
        skip_materials=True,
    )

    # Represent the scene by a single mesh.
    mesh = scene.dump(concatenate=True)

    # Clean the mesh.
    mesh.process(validate=True)

    return mesh


# make dst folder
os.makedirs(dst_path, exist_ok=True)

mesh_in_paths = sorted(glob.glob(f"{src_path}/*.glb"))

for mesh_in_path in mesh_in_paths:
    print(f"src: {mesh_in_path}")
    mesh = load_mesh(mesh_in_path)

    # Convert from meters to millimeters.
    mesh.vertices *= 1000.0

    # calculate normals
    vertex_normals = mesh.vertex_normals

    # save the mesh as a PLY file ascii format
    mesh_out_path = os.path.join(dst_path, os.path.basename(mesh_in_path).replace(".glb", ".ply"))
    print(f"dst: {mesh_out_path}")
    ply_file = trimesh.exchange.ply.export_ply(mesh, encoding="ascii")
    with open(mesh_out_path, "wb") as f:
        f.write(ply_file)
