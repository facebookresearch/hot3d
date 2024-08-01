import json

input_json_path = '/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models/models_info.json'
output_json_path = '/media/gouda/ssd_data/datasets/hot3d/hot3d/object_models_ply/models_info.json'

with open(input_json_path, 'r') as f:
    input_json_data = json.load(f)

output_json_data = {}

# only keep keys: diameter, min_x, min_y, min_z, size_x, size_y, size_z and multiply by 1000
for key, value in input_json_data.items():
    output_json_data[key] = {
        'diameter': value['diameter'] * 1000,
        'min_x': value['min_x'] * 1000,
        'min_y': value['min_y'] * 1000,
        'min_z': value['min_z'] * 1000,
        'size_x': value['size_x'] * 1000,
        'size_y': value['size_y'] * 1000,
        'size_z': value['size_z'] * 1000
    }

with open(output_json_path, 'w') as f:
    json.dump(output_json_data, f, indent=4)
print(f"Saved to {output_json_path}")
