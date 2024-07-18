# HOT3D Toolkit

This repository hosts an official toolkit for [HOT3D](https://arxiv.org/pdf/2406.09598), an egocentric dataset for 3D hand and object tracking.

The toolkit offers:

- An API for downloading and using the [full HOT3D dataset](https://www.projectaria.com/datasets/hot3D/) provided in a [VRS](https://github.com/facebookresearch/vrs)-based format (see [tutorial](https://github.com/facebookresearch/hot3d/blob/main/hot3d/HOT3D_Tutorial.ipynb)).
- An API for [HOT3D-Clips](https://github.com/facebookresearch/hot3d/tree/main/hot3d/clips) which is a curated HOT3D subset provided in the [Webdataset](https://github.com/webdataset/webdataset) format and used in [BOP Challenge 2024](https://bop.felk.cvut.cz/challenges/bop-challenge-2024) and [Multiview Egocentric Hand Tracking Challenge](https://github.com/facebookresearch/hand_tracking_toolkit?tab=readme-ov-file#evaluation).

Resources:
- [Download full HOT3D dataset from projectaria.com](https://www.projectaria.com/datasets/hot3D/)
- [Download HOT3D-Clips from Hugging Face](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d)
- [Read HOT3D whitepaper](https://arxiv.org/pdf/2406.09598)

## Step 1: Install the downloader

This Python repository can be used with Pixi and Conda environments and can run on:

* x64 Linux distributions of:
    * Fedora 36, 37, 38
    * Ubuntu jammy (22.04 LTS) and focal (20.04 LTS)
* Mac Intel or Mac ARM-based (M1) with MacOS 11 (Big Sur) or newer

Python 3.9+ (3.10+ if you are on [Apple Silicon](https://support.apple.com/en-us/116943)).


### Using Pixi

[Pixi](https://prefix.dev/) is a package management tool for developers. Developers can install libraries and applications in a reproducible way, which makes it easier to install and use a Python environment for the HOT3D API.

```
# 1. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Checkout this repository
git clone https://github.com/facebookresearch/hot3d.git
cd hot3d

# 3. Call `pixi install` to setup the environment
pixi install

# 4. (Optional) Install the third-party dependencies required for hands, by reviewing and accepting the licenses provided on the corresponding third-party repositories
pixi run setup_hands

```

#### A quick introduction to the [PIXI environment](https://prefix.dev/)
- Execute Pixi environment commands from within the `hot3d` folder.
- The Pixi environment is located in the `.pixi` folder.
- Activate the Pixi HOT3D environment by using the command `pixi shell`.
- Exit the Pixi HOT3D environment by typing `exit`.
- Remove the environment by executing the command `rm -rf .pixi`.


### Using Conda

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

#### A quick introduction to the [CONDA environment](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html#managing-environments)

- Activate the Conda HOT3D environment by executing `conda activate hot3d`.
- Exit the Conda HOT3D environment using `source deactivate`.
- Remove the Conda HOT3D environment by executing `conda remove --name hot3d --all`.


## Step 2: Sign up and get the download links file

1. Review the [HOT3D license agreement](https://www.projectaria.com/datasets/hot3d/license/).
    * Examine the specific licenses applicable to the data types you wish to use, such as Sequence, Hand annotations, and 3D object models.
2. Go to the [HOT3D website](https://www.projectaria.com/datasets/hot3D/) and sign up.
    * Scroll down to the bottom of the page.
    * Enter your email and select **Access the Datasets**.
3. The HOT3D page will be refreshed to contain instructions and download links
    * The download view is ephemeral, keep the tab open to access instructions and links
    * Download links that last for 14 days
    * Enter your email again on the HOT3D main page to get fresh links
4. Select the Download button for any of the data types:
    * “Download the HOT3D Aria Dataset"
    * "Download the HOT3D Quest Dataset"
    * "Download the HOT3D Assets Dataset"
5. These will swiftly download JSON files with urls that the downloader will use


## Step 3: Download the data
Use the HOT3D downloader to download some, or all of the data.

```
# 1. Activate your environment (assuming from the hot3d folder):
# conda: conda activate hot3d
# pixi: pixi shell

# 2. Go to the hot3d/data_downloader directory
cd hot3d/data_downloader
mkdir -p ../dataset

# 3. Run the dataset downloader
# Download HOT3D Object Library data
python3 dataset_downloader_base_main.py -c Hot3DAssets_download_urls.json -o ../dataset --sequence_name all

# Download one HOT3D Aria data sequence
python3 dataset_downloader_base_main.py -c Hot3DAria_download_urls.json -o ../dataset --sequence_name P0003_c701bd11 --data_types all
# Type answer `y`

# Download one HOT3D Quest data sequence
python3 dataset_downloader_base_main.py -c Hot3DQuest_download_urls.json -o ../dataset --sequence_name P0002_1464cbdc --data_types all
# Type answer `y`
```

**Tip:** To download all sequences in a download links JSON file (such as the HOT3D Object Library data in step 3), do not include the sequence_name argument.

## Step 4: Run the dataset viewer

### Viewing objects and headset pose trajectory
```
python3 viewer --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets/
```

When using `pixi`, you can directly launch the viewer without explicitly activating the environment by using the following command:
```
pixi run viewer --sequence_folder --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets/
```


## Step 5: Run the python notebook tutorial

```
# Assuming you have downloaded the Aria `P0003_c701bd11` sequence and the object library above.
#
# Install Jupyter Notebook for your environment:
python3 -m pip install Jupyter
# Run Jupyter and open the notebook (conda)
jupyter notebook ./HOT3D_Tutorial.ipynb
# Run Jupyter and open the notebook (pixi, use a direct path to ensure jupyter will take the right python path)
.pixi/envs/default/bin/jupyter notebook ./HOT3D_Tutorial.ipynb
```

### Using hand annotations

Hand pose annotations in HOT3D are provided in the [UmeTrack](https://github.com/facebookresearch/UmeTrack) and [MANO](https://mano.is.tue.mpg.de/) formats. Both hand poses annotation are accessible in the API by using either the `mano_hand_data_provider`, `umetrack_hand_data_provider` property once the `Hot3dDataProvider` is initialized. In order to choose the representation for the viewer, use the following:

#### UmeTrack

```
python3 viewer --sequence_folder --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets --hand_type UMETRACK
```

#### MANO

Hand annotations in the MANO format can be downloaded after accepting their [license agreement](https://mano.is.tue.mpg.de/).
- HOT3D only requires the `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` files for loading and rendering of hand poses. These files can be obtained from the `mano_v1_2.zip` file located in the "Models & Code" section of the `MANO` website. After downloading, extract the zip file to your local disk, and the `*.pkl` files can be found at the following path: `mano_v1_2/models/`.

```
python3 viewer --sequence_folder --sequence_folder <PATH>/hot3d_dataset/P0001_0444739e> --object_library_folder <PATH>/hot3d_dataset/assets --mano_model_folder <PATH>/mano_v1_2/models/  --hand_type MANO
```


## License

- [HOT3D API](https://github.com/facebookresearch/hot3d) (aka. this repository) is released by Meta under the [Apache 2.0 license](LICENSE)
- [HOT3D dataset](https://www.projectaria.com/datasets/hot3d/) is released under the [HOT3D license agreement](https://www.projectaria.com/datasets/hot3d/license/)
  - Using hands annotation requires installation of [SMPLX/MANO](https://github.com/vchoutas/smplx) third-party dependencies, please review and agree to their license listed on their website.


## Contributing

Go to [Contributing](.github/CONTRIBUTING.md) and the [Code of Conduct](.github/CODE_OF_CONDUCT.md).
