# HOT-3D: A dataset for egocentric 3D hand and object tracking

# Install and use

## Using Conda

 ```
conda create --name hot3d
conda activate hot3d
# install dependencies
python3 -m ensurepip
python3 -m pip install projectaria_tools torch
python3 -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
python3 -m pip install vrs
python3 -m pip install matplotlib
python3 -m pip install 'git+https://github.com/vchoutas/smplx.git'
python3 -m pip install 'git+https://github.com/mattloper/chumpy'
python3 viewer_mockup --sequence_folder <Sequence i.e <PATH>/aria_P0009_PickupDropOff_cf323827> --object_library_folder <PATH>
```

## Using pixi

```
git clone https://github.com/facebookresearch/hot3d.git
cd hot3d
# install dependencies
pixi run setup
# run the viewer
pixi run viewer --sequence_folder <Sequence i.e <PATH>/aria_P0009_PickupDropOff_cf323827> --object_library_folder <PATH>
# run the pixi environment and the hot3d viewer
pixi shell
python3 viewer_mockup --sequence_folder <Sequence i.e <PATH>/aria_P0009_PickupDropOff_cf323827> --object_library_folder <PATH>
```

## License

HOT-3D is released by Meta under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
