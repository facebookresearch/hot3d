# HOT3D-Clips

HOT3D-Clips is a set of curated sub-sequences of the [HOT3D dataset](https://facebookresearch.github.io/hot3d/).
Each clip has 150 frames (5 seconds) which are all annotated with ground-truth poses of all modeled objects and hands and which passed our visual inspection.
There are 4117 clips in total, 2969 clips extracted from the training split and 1148 from the test split of HOT3D.

The clips are provided to facilitate benchmarking of 3D hand/object tracking methods and are used in [BOP Challenge 2024](https://bop.felk.cvut.cz/challenges/bop-challenge-2024) and [Multiview Egocentric Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation).

- [Download HOT3D-Clips from Hugging Face](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d) (provided in the [Webdataset format](https://github.com/webdataset/webdataset))
- [Download full HOT3D dataset from projectaria.com](https://www.projectaria.com/datasets/hot3D/) (provided in a [VRS](https://github.com/facebookresearch/vrs)-based format; see [tutorial](https://github.com/facebookresearch/hot3d/blob/main/hot3d/HOT3D_Tutorial.ipynb))
- [Read HOT3D whitepaper](https://arxiv.org/pdf/2406.09598)


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
