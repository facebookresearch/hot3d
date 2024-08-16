# HOT3D-Clips

HOT3D-Clips is a set of curated sub-sequences of the [HOT3D dataset](https://facebookresearch.github.io/hot3d/).
Each clip has 150 frames (5 seconds) which are all annotated with ground-truth poses of all modeled objects and hands and which passed our visual inspection.
There are 3832 clips in total, 2804 clips extracted from the training split and 1028 from the test split of HOT3D.

HOT3D-Clips are hosted on [Hugging Face](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d) and used in [BOP Challenge 2024](https://bop.felk.cvut.cz/challenges/bop-challenge-2024) and [Multiview Egocentric Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation). The full HOT3D dataset is provided in a [VRS](https://github.com/facebookresearch/vrs)-based format on [projectaria.com](https://www.projectaria.com/datasets/hot3D/) (see [tutorial](https://github.com/facebookresearch/hot3d/blob/main/hot3d/HOT3D_Tutorial.ipynb)).

More details can be found in the [HOT3D whitepaper](https://arxiv.org/pdf/2406.09598).


## Data format

HOT3D-Clips are distributed via a [folder on Hugging Face](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d) which includes:

- `object_models` - 3D object models in GLB format with PBR materials.
- `object_models_eval` - Simplified 3D object models in GLB format without PBR materials (used for [BOP evaluation](https://bop.felk.cvut.cz/challenges/bop-challenge-2024/)).
- `object_ref_quest3_dynamic` - Dynamic object onboarding sequences from Quest 3. GT object pose is available only in the first frame.
- `object_ref_quest3_dynamic_vis` - Visualizations of dynamic onboarding sequences from Quest 3.
- `object_ref_quest3_static` - Static object onboarding sequences from Quest 3. GT object poses are available in all frames.
- `object_ref_quest3_static_vis` - Visualizations of static onboarding sequences from Quest 3.
- `test_aria` - Test Aria clips with some GT annotations removed (see the below description of splits for details).
- `test_quest3` - Test Quest3 clips with some GT annotations removed (see the below description of splits for details).
- `train_aria` - Training Aria clips with all GT annotations.
- `train_quest3` - Training Quest3 clips with all GT annotations.
- `vis_mano` - Visualizations of GT object annotations and GT [MANO](https://github.com/facebookresearch/hot3d?tab=readme-ov-file#mano) hand annotations.
- `vis_umetrack` - Visualizations of GT object annotations and GT [UmeTrack](https://dl.acm.org/doi/pdf/10.1145/3550469.3555378) hand annotations.
- `clip_definitions.json` - Includes the source HOT3D sequence, device and timestamps for each clip.
- `clip_splits.json` - Defines the following clip splits:
    - `train`
        - Training clips which include all GT annotations. Extracted from sequences of 13 participants that are not seen in any other split.
    - `test_ht_pose`
        - Test split for the Pose Estimation track of the [Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation).
        - Files `hands.json` with GT hand poses and `objects.json` with GT object poses are not publicly available.
        - Extracted from sequences of 3 participants that are not seen in any other split.
    - `test_ht_shape`
        - Test split for the Shape Estimation track of the [Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation).
        - Files `hands.json` with GT hand poses, `__hand_shapes.json__` with GT hand shapes, and `objects.json` with GT object poses are not publicly available.
        - Extracted from sequences of 3 participants that are not seen in any other split.
    - `test_bop`
        - Test split for [BOP Challenge 2024](https://bop.felk.cvut.cz/challenges/bop-challenge-2024/). Union of `test_ht_pose` and `test_ht_shape`.


Folders `train_aria`, `train_quest3`, `test_aria`, and `test_quest3` include one `tar` archive per clip. A clip from Aria contains one RGB image stream (`214-1`) and two monochrome image streams (`1201-1` and `1201-2`), while a clip from Quest 3 contains two monochrome image streams (`1201-1` and `1201-2`).

A clip has the following structure:

```
├─ clip-<CLIP-ID>.tar
│  ├─ <FRAME-ID>.image_214-1.jpg
│  ├─ <FRAME-ID>.image_1201-1.jpg
│  ├─ <FRAME-ID>.image_1201-2.jpg
│  ├─ <FRAME-ID>.cameras.json
│  ├─ <FRAME-ID>.hands.json
│  ├─ <FRAME-ID>.hand_crops.json
│  ├─ <FRAME-ID>.objects.json
│  ├─ <FRAME-ID>.info.json
│  ├─ ...
│  ├─ __hand_shapes.json__
```

Files `<FRAME-ID>.cameras.json` provide camera parameters for each image stream:

- `calibration`:
    - `label`: Label of the camera stream (e.g. `camera-slam-left`).
    - `stream_id`: Stream id (e.g. `214-1`).
    - `serial_number`: Serial number of the camera.
    - `image_width`: Image width.
    - `image_height`: Image height.
    - `projection_model_type`: Projection model type (e.g. `CameraModelType.FISHEYE624`).
    - `projection_params`: Projection parameters.
    - `T_device_from_camera`:
        - `translation_xyz`: Translation from device to the camera.
        - `quaternion_wxyz`: Rotation from device to the camera.
    - `max_solid_angle`: Max solid angle of the camera.
- `T_world_from_camera`:
    - `translation_xyz`: Translation from world to the camera.
    - `quaternion_wxyz`: Rotation from world to the camera.

Files `<FRAME-ID>.objects.json` provide for each annotated object the following:

- `T_world_from_object`:
    - `translation_xyz`: Translation from world to the object.
    - `quaternion_wxyz`: Rotation from world to the object.
- `boxes_amodal`: A map from a stream ID to an amodal 2D bounding box.
- `masks_modal` [currently not available]: A map from a stream ID to an modal binary mask.
- `visibilities_modeled`: A map from a stream ID to the fraction of the projected surface area that is visibile.
        (reflecting only occlusions by modeled scene elements).
- `visibilities_full` [currently not available]: A map from a stream ID to the fraction of the projected surface area that is visibile
    (reflecting occlusions by modeled and unmodeled, such as arms, scene elements).

Files `<FRAME-ID>.hands.json` provide hand parameters:

- `left`: Parameters of the left hand (may be missing).
    - `mano_pose`:
        - `thetas`: MANO pose parameters.
        - `wrist_xform`: 3D rigid transformation from world to wrist, in the axis-angle + translation format expected by the smplx library
            (`wrist_xform[0:3]` is the axis-angle orientation and `wrist_xform[3:6]` is the 3D translation).
    - `umetrack_pose`:
        - `joint_angles`: 20 floats.
        - `wrist_xform`: 4x4 3D rigid transformation matrix.
    - `boxes_amodal`: A map from a stream ID to an amodal 2D bounding box.
    - `masks_modal` [currently not available]: A map from a stream ID to an modal binary mask.
    - `visibilities_modeled`: A map from a stream ID to the fraction of the projected surface area that is visibile.
        (reflecting only occlusions by modeled scene elements).
    - `visibilities_full` [currently not available]: A map from a stream ID to the fraction of the projected surface area that is visibile
        (reflecting occlusions by modeled and unmodeled, such as arms, scene elements).
- `right`: As for `left`.

Files `<FRAME-ID>.hand_crops.json` provide hand crop parameters (used in [Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation); a crop camera is saved only if the hand visibility > 0.1):

- `left`: A map from a stream ID to a dictionary with these items:
    - `T_world_from_crop_camera`:
        - `translation_xyz`: Translation from world to the crop camera.
        - `quaternion_wxyz`: Rotation from world to the crop camera.
    - `crop_camera_fov`: Field of view of the crop camera in degrees.
- `right`: As for `left`.

File `__hand_shapes.json__` provides hand shape parameters (shared by all frames in a clip):

- `mano`: MANO shape (beta) parameters shared by the left and right hands.
- `umetrack`: Serialized UmeTrack UserProfile.


## Loading and visualizing HOT3D-Clips

First, make sure all dependencies are available in your environment (tested with Python 3.10):
```
pip install opencv-python
pip install imageio
pip install trimesh
pip install git+https://github.com/facebookresearch/hand_tracking_toolkit
```

If you want to use hand annotations in the [MANO](https://github.com/facebookresearch/hot3d?tab=readme-ov-file#mano) format, you will additionally need:
```
pip install git+https://github.com/vchoutas/smplx.git
pip install git+https://github.com/mattloper/chumpy
```

Then, run the following command (from folder `hot3d/clips`) to load selected clips and visualize ground-truth hand and object poses:
```
python vis_clips.py --clips_dir <clips_dir> --object_models_dir <object_models_dir> --output_dir <output_dir>
```
Required arguments:
- `--clips_dir` is a folder with clips saved as tar files (can be, e.g., your local copy of [`test_aria`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/test_aria), [`test_quest3`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/test_quest3), [`train_aria`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/train_aria) or [`train_quest3`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/train_quest3)).
- `--object_models_dir` is a folder with 3D object models (can be a local copy of [`object_models_eval`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/object_models_eval) or [`object_models`](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d/object_models); the first is recommended for speed).
- `--output_dir` is an output folder for visualizations.

Optional arguments:
- `--hand_type` indicates which hand format to use ([UmeTrack](https://dl.acm.org/doi/pdf/10.1145/3550469.3555378) and [MANO](https://github.com/facebookresearch/hot3d?tab=readme-ov-file#mano) formats are available; UmeTrack is default).
- `--mano_model_dir` is a folder with the MANO hand model (needs to be specified if `--hand_type mano`).
- `--clip_start` and `--clip_end` can be used to specify a range of clips to consider.
- `--undistort` is a binary flag indicating whether the images should be undistorted (warped from the original fisheye cameras to pinhole cameras; disabled by default).

An example command to visualize Quest3 training clips (`$HOT3DC` is assumed to be a path to [HOT3D-Clips](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d)):
```
python vis_clips.py --clips_dir $HOT3DC/train_quest3 --object_models_dir $HOT3DC/object_models_eval --output_dir $HOT3DC/output
```


## License

HOT3D-Clips are distributed under [the same license](https://github.com/facebookresearch/hot3d?tab=readme-ov-file#license) as the full HOT3D dataset.
