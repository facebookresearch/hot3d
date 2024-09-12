
# Scripts to convert HOT3D dataset from its native format to the BOP format.

### Set Environment Variables

Before running the scripts, set the following environment variables:

- `HOT3D_DIR`: Path to the HOT3D dataset directory
- `BOP_DIR`: Path to the output BOP directory

```bash
export HOT3D_DIR=<PATH_TO_HOT3D_DATASET>
export BOP_DIR=<PATH_TO_OUTPUT_BOP>
```

### Convert the object models from HOT3D format to BOP format

To convert (full) models:

```bash
python hot3d_models_to_bop.py --input-gltf-dir $HOT3D_DIR/object_models --output-bop-dir $BOP_DIR/models
```

To convert eval models:

```bash
python hot3d_models_eval_to_bop.py --input-gltf-dir $HOT3D_DIR/object_models_eval --output-bop-dir $BOP_DIR/models_eval
```

Copy the models info from both models and models_eval to the same directory:

```bash
cp $HOT3D_DIR/object_models/models_info.json $BOP_DIR/models/models_info.json
cp $HOT3D_DIR/object_models_eval/models_info.json $BOP_DIR/models_eval/models_info.json
```

### Convert HOT3D clips to BOP format

To convert HOT3D clips to BOP format, run the following command:

Parameters:
- --split: Options are "train_aria", "train_quest3", "test_aria", or "test_quest3"
- --num-threads: Optional, with a default of 4. You can use 4 or 8 threads for better performance.

```bash
python hot3d_clips_to_bop_scenewise.py \
  --hot3d-dataset-path $HOT3D_DIR \
  --split <SPLIT_NAME> \
  --output-bop-path $BOP_DIR \
  --num-threads <NUM_THREADS>
```
