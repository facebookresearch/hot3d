# HOT3D: A dataset for egocentric 3D hand and object tracking

This repository hosts the API for downloading and utilizing HOT3D, a benchmark dataset designed for the vision-based understanding of 3D hand-object interactions.

# Installation and usage

This python repository can be used with pixi and conda environments.

## Using Pixi

[Pixi](https://prefix.dev/) is a package management tool for developers. It allows the developer to install libraries and applications in a reproducible way and ease the installation and usage of a python environment for the HOT3D API.

```
# 1. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Checkout this repository
git clone https://github.com/facebookresearch/hot3d.git
cd hot3d

# 3. Call `pixi install` to setup the environment
pixi install

# 4. (Optional) Install the third-party dependencies required for hands by reviewing and accepting the licenses provided on the corresponding third-party repositories
pixi run setup_hands

```

### A quick introduction to [PIXI environment](https://prefix.dev/)
- Execute Pixi environment commands from within the `hot3d` folder.
- The Pixi environment is located in the `.pixi` folder.
- Activate the Pixi HOT3D environment by using the command `pixi shell`.
- Exit the Pixi HOT3D environment by typing `exit`.
- Remove the environment by executing the command `rm -rf .pixi`.

## Using Conda

[Conda](https://conda.io/projects/conda/en/latest/index.html) is a package manager used for managing software environments and dependencies in data science and scientific computing.

```
# 1. Install conda -> https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
# 2. Create your environment
conda create --name hot3d
conda activate hot3d

# 2. Install dependencies
python3 -m ensurepip
python3 -m pip install projectaria_tools==1.5.1 torch requests rerun-sdk==0.16.0
python3 -m pip install vrs
python3 -m pip install matplotlib

# 3. (Optional) Install the third-party dependencies required for hands by reviewing and accepting the licenses provided on the corresponding third-party repositories
python3 -m pip install 'git+https://github.com/vchoutas/smplx.git'
python3 -m pip install 'git+https://github.com/mattloper/chumpy'
```

### A quick introduction to [CONDA environment](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html#managing-environments)

- Activate the Conda HOT3D environment by executing `conda activate hot3d`.
- Exit the Conda HOT3D environment using `source deactivate`.
- Remove the Conda HOT3D environment by executing `conda remove --name hot3d --all`.

# Dataset downloading

Please review the [HOT3D license agreement](https://www.projectaria.com/datasets/hot3d/license/) first, and then examine the specific licenses applicable to the data types you wish to use, such as Sequence, Hand annotations, and 3D object models.

Required:
- After agreeing to the license, download the cdn url files from the "Access HOT3D Dataset and Accompanying Tools" section on [HOT3D projectaria.com website](https://www.projectaria.com/datasets/hot3d). You should see buttons corresponding to each download data type as follows
  - "Download the HOT3D Aria Dataset"
  - "Download the HOT3D Quest Dataset"
  - "Download the HOT3D Assets Dataset"

```
# 1. Activate your environment (assuming your are in the hot3d folder):
# conda: conda activate hot3d
# pixi: pixi shell

# 2. Go to the hot3d/data_downloader directory
cd hot3d/data_downloader
mkdir -p ../dataset

# 3. Run the dataset downloader
# Download HOT3D Object Library data
python3 dataset_downloader_base_main.py -c Hot3DAssets_download_urls.json -o ../dataset --sequence_name all

# Download HOT3D Aria data (here one sequence)
python3 dataset_downloader_base_main.py -c Hot3DAria_download_urls.json -o ../dataset --sequence_name P0003_c701bd11 --data_types all
# Type answer `y`

# Download HOT3D Quest data (here one sequence)
python3 dataset_downloader_base_main.py -c Hot3DQuest_download_urls.json -o ../dataset --sequence_name P0002_1464cbdc --data_types all
# Type answer `y`
```


# Running the dataset viewer

## Viewing objects and headset pose trajectory
```
python3 viewer --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets/
```

When using `pixi`, you can directly launch the viewer without explicitly activating the environment by using the following command:
```
pixi run viewer --sequence_folder --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets/
```


## Using hand annotations

Hand pose annotations in HOT3D are provided in the [UmeTrack](https://github.com/facebookresearch/UmeTrack) and [MANO](https://mano.is.tue.mpg.de/) formats.

### UmeTrack

Instructions coming soon.

### MANO

Hand annotations in the MANO format can be downloaded after accepting their [license agreement](https://mano.is.tue.mpg.de/).
- HOT3D only requires the `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` files for loading and rendering of hand poses. These files can be obtained from the `mano_v1_2.zip` file located in the "Models & Code" section of the `MANO` website. After downloading, extract the zip file to your local disk, and the `*.pkl` files can be found at the following path: `mano_v1_2/models/`.

```
python3 viewer --sequence_folder --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets --mano_model_folder <PATH>/mano_v1_2/models/
```


# License

- [HOT3D Dataset API](https://github.com/facebookresearch/hot3d) (aka. this repository) is released by Meta under the [Apache 2.0 license](LICENSE)
- [HOT3D dataset](https://www.projectaria.com/datasets/hot3d/) is released under the [HOT3D license agreement](https://www.projectaria.com/datasets/hot3d/license/)
  - Using hands annotation requires installation of  [SMPLX/MANO](https://github.com/vchoutas/smplx) third-party dependencies, please review and agree to their license listed on their website.


# Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
