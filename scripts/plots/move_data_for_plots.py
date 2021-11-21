import os
from argparse import ArgumentParser
import shutil


def main(data_dir):
    assert os.path.isdir(data_dir)

    model_name = os.path.split(data_dir)[-1]

    # Create folder layerdata
    layer_data_dir = '{}/layer_data/'.format(data_dir)
    os.mkdir(layer_data_dir)

    # For ever folder like *layer_X*
    # Find json file in it

    if args.mode_24:
        for layer in range(1, 25):
            try:
                json_file = get_json_data_file_for_layer(data_dir, layer)
                # Move to layerdata/layer_X/data.json
                layer_folder = '{}layer_{}'.format(layer_data_dir, layer)
                os.mkdir(layer_folder)
                shutil.move(json_file, layer_folder)
            except Exception as e:
                print('Could not find layerdata for layer ', layer)
                print(e)
    else:
        for layer in range(1, 13):
            json_file = get_json_data_file_for_layer(data_dir, layer)
            # Move to layerdata/layer_X/data.json
            layer_folder = '{}layer_{}'.format(layer_data_dir, layer)
            os.mkdir(layer_folder)
            shutil.move(json_file, layer_folder)

    # Tar/zip the folder with layerdata
    archive = shutil.make_archive('{}_layer_data'.format(model_name), 'zip',
                                  base_dir=layer_data_dir, root_dir=data_dir)
    shutil.move(archive, data_dir)
    print('Finished archiving. File can be found at: {}'.format(data_dir))
    # Be able to download the data


def get_subfolders(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def get_layer_folder(data_base_dir, layer):
    list_subfolders_with_paths = get_subfolders(data_base_dir)
    matching = [
        s for s in list_subfolders_with_paths if "layer_{}".format(layer) in s]

    shortest = matching[0]
    if len(matching) > 1:
        # We use the shortest file name as layer_1 is shorter than layer_10
        for file in matching:
            if len(file) < len(shortest):
                shortest = file

        print('Found more than one dir with for layer {}. Using this one: {}'.format(
            layer, shortest))
    return shortest


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--mode_24', default=False, action='store_true')

    args = parser.parse_args()

    main(args.data_dir)
