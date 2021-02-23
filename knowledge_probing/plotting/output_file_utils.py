import os
import json


def handle_mean_values_string(mean_vals_string):
    values_string = mean_vals_string.split(':')[1]

    values = values_string.split(',')

    numeric_vals = []

    for val in values:
        val = val.strip()
        numeric_vals.append(float(val))

    return numeric_vals


def make_plots_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_sample_data_file(dir):
    json_file = get_json_data_file_for_layer(dir, layer=12)
    sample_data = load_json_data(json_file)

    return sample_data


def get_subfolders(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def load_json_data(file):
    with open(file) as json_data:
        data = json.load(json_data)
    return data


def get_layer_folder(data_base_dir, layer):
    list_subfolders_with_paths = get_subfolders(data_base_dir)
    matching = [
        s for s in list_subfolders_with_paths if "layer_{}".format(layer) in s]

    for match in matching:
        file_name = os.path.basename(os.path.normpath(match))
        if file_name == 'layer_{}'.format(layer):
            print(f'Found the right folder: {match}')
            return match

    raise Exception('Could not find folder for layer {layer}')


def get_files_in_folder(dir):
    return [f.path for f in os.scandir(dir) if f.is_file()]


def get_json_data_file_for_layer(dir, layer):
    layer_dir = get_layer_folder(dir, layer)
    all_files = get_files_in_folder(layer_dir)
    json_files = [f for f in all_files if ".json" in f]
    if len(json_files) > 1:
        print('Found more than one json data file in dir. Using this one: {}'.format(
            json_files[0]))
    if len(json_files) == 0:
        print('No json files found in {}'.format(layer_dir))
        return None
    return json_files[0]
