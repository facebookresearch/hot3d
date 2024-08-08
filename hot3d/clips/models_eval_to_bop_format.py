
import glob
import os
import trimesh
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gltf_dir", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval")
    parser.add_argument("--output_dir", type=str, default="/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_eval_bop")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mesh_in_paths = sorted(glob.glob(f"{args.gltf_dir}/*.glb"))

    for mesh_in_path in mesh_in_paths:
        print(f"src: {mesh_in_path}")
        mesh = load_mesh(mesh_in_path)

        # Convert from meters to millimeters.
        mesh.vertices *= 1000.0

        # save the mesh as a PLY file ascii format
        mesh_out_path = os.path.join(args.output_dir, os.path.basename(mesh_in_path).replace(".glb", ".ply"))
        print(f"dst: {mesh_out_path}")
        ply_file = trimesh.exchange.ply.export_ply(mesh, encoding="ascii")
        with open(mesh_out_path, "wb") as f:
            f.write(ply_file)


def load_mesh(path: str) -> trimesh.Trimesh:
    # Load the scene.
    scene = trimesh.load_mesh(path, process=False, merge_primitives=True, skip_materials=True, maintain_order=True)

    # Represent the scene by a single mesh.
    mesh = scene.dump(concatenate=True)

    # Clean the mesh.  # don't use it as it will change indices and normals
    #mesh.process(validate=True)

    return mesh


if __name__ == "__main__":
    main()
