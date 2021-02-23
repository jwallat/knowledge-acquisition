from tqdm import tqdm
import os
import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import collections
# from knowledge_probing.plotting.output_file_utils import get_json_data_file_for_layer, load_json_data

layer_range = range(1, 25)
# layer_range = range(1, 13)
plot_dir = '/home/jonas/git/knowledge-probing-private/data/plots/wordclouds_top_1/'


def obj_label_stats(data_sample):
    obj_labels_data = {}

    datasets = data_sample.keys()

    for dataset in datasets:
        print(50 * '*' + '  Dataset  {}  '.format(dataset) + 50 * '*')
        for relation in tqdm(data_sample[dataset].keys()):
            print(relation)
            if relation == 'means':
                # Skip, no data here
                pass
            else:
                layer_data = data_sample

                obj_labels_data[relation] = {}

                obj_labels = []

                for individual_prediction in layer_data[dataset][relation][0]['individual_predictions']:
                    obj_labels.append(
                        individual_prediction['sample']['obj_label'])

                obj_labels_data[relation]['labels'] = obj_labels
                counter = collections.Counter(obj_labels)
                obj_labels_data[relation]['frequencies'] = counter

                print(50 * '-' + '  Relation  {}  '.format(relation) + 50 * '-')
                print(counter.most_common(100))

    return obj_labels_data

    # for individual_prediction in layer_data[dataset][relation][0]['individual_predictions']:
    #     pass
    # pass


def main():

    make_plots_dir(plot_dir)
    selected_models = select_model_for_comparison()

    for model in selected_models:
        print(60 * "*" + '    Loading model {}    '. format(model['name']))
        model['data'] = smart_load_data(model['data_dir'])

    data_sample = selected_models[0]['data']['1']
    obj_labels_data = obj_label_stats(data_sample)

    with open('{}/{}-obj_labels_and_frequencies_data.json'.format(plot_dir, selected_models[0]['name']), 'w') as outfile:
        json.dump(obj_labels_data, outfile)

    # print(data_sample)

    print('done loading')

    datasets = data_sample.keys()

    for dataset in datasets:
        for relation in tqdm(data_sample[dataset].keys()):
            print(relation)
            if relation == 'means':
                # Skip, no data here
                pass
            else:
                for model in selected_models:
                    model_data = model['data']
                    # model_values = []

                    print('Model ', model['name'])

                    for layer in layer_range:
                        layer_data = model_data[str(layer)]

                        print('Layer', layer)

                        if layer_data is not None:
                            print('Before make wordcloud')
                            make_wordcloud(layer_data, dataset,
                                           relation, layer, model['name'])


def make_wordcloud(layer_data, dataset, relation, layer, model_name):
    print('In make_wordcloud')
    fig_path = '{}/{}/{}/{}/'.format(plot_dir, model_name, dataset, relation)
    print('Saving to {}'.format(fig_path))
    make_plots_dir(fig_path)

    all_top_tokens = []

    for individual_prediction in layer_data[dataset][relation][0]['individual_predictions']:
        top_tokens = individual_prediction['top_k_tokens'][:1]
        for i, token in enumerate(top_tokens):
            if token == '':
                top_tokens[i] = '<blank>'
        # top_tokens = [x.replace(' ', '<blank>') for x in top_tokens]
        # top_tokens.replace(' ', '<blank>')
        all_top_tokens.extend(top_tokens)

    # print(all_top_tokens)
    counter = collections.Counter(all_top_tokens)
    print(counter.most_common(200))

    # all_top_tokens_string = (" ").join(all_top_tokens)

    wordcloud = WordCloud(width=1600, height=1000,
                          background_color='white',
                          stopwords=[],
                          min_font_size=10).generate_from_frequencies(counter)

    # plot the WordCloud image
    plt.figure(figsize=(16, 10), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    # plt.show()
    plt.savefig('{}/{}-{}-{}'.format(fig_path, dataset, relation, layer))
    plt.close()


def smart_load_data(dir):
    all_data = {}

    for layer in tqdm(layer_range):
        try:
            print('Loading layer file: ', layer)
            data_file = get_json_data_file_for_layer(
                dir, layer)
            layer_data = load_json_data(data_file)
            all_data[str(layer)] = layer_data
        except Exception as e:
            print('No data found for layer ', layer)
            print(e)
            print('-> will be skipped')
            all_data[str(layer)] = None

    return all_data


def select_model_for_comparison():
    selected_models = []

    default = {
        'data_dir': '/media/jonas/TOSHIBA EXT/latest_knowledge_probing/outputs/bert/',
        'name': 'BERT',
        'marker': 'circle',
        'color': 'black'
    }
    selected_models.append(default)

    # squad_uncased = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_1/',
    #     'name': 'QA-SQuAD-1',
    #     'marker': 'triangle-down',
    #     'color': 'coral'
    # }
    # selected_models.append(squad_uncased)

    # squad_2_uncased = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_2/',
    #     'name': 'QA-SQuAD-2',
    #     'marker': 'triangle-up',
    #     'color': 'red'
    # }
    # selected_models.append(squad_2_uncased)

    # squad_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/10/',
    #     'name': 'MLM-SQuAD',
    #     'marker': 'diamond',
    #     'color': 'darkred'
    # }
    # selected_models.append(squad_mlm)

    # msmarco_ranking = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_rank/',
    #     'name': 'RANK-MSMarco',
    #     'marker': 'star',
    #     'color': 'blue'
    # }
    # selected_models.append(msmarco_ranking)

    # msmarco_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_mlm/',
    #     'name': 'MLM-MSMarco',
    #     'marker': 'pentagon',
    #     'color': 'dodgerblue'
    # }
    # selected_models.append(msmarco_mlm)

    # ner = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/ner/',
    #     'name': 'NER-CoNLL',
    #     'marker': 'cross',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(ner)

    ######################################################################################################################################################

    # squad_mlm_pre_trained = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/rnd_vs_pre-trained/pre-trained/layer_data/',
    #     'name': 'Pre-trained',
    #     'marker': 'diamond',
    #     'color': 'orange'
    # }
    # selected_models.append(squad_mlm_pre_trained)

    # squad_mlm_rnd = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/rnd_vs_pre-trained/rnd/layer_data/',
    #     'name': 'Random',
    #     'marker': 'diamond-open',
    #     'color': 'purple'
    # }
    # selected_models.append(squad_mlm_rnd)

    # fc = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/fc_no_fc/fc/',
    #     'name': 'FC layer',
    #     'marker': 'star-triangle-up',
    #     'color': 'blue'
    # }
    # selected_models.append(fc)

    # no_fc = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/fc_no_fc/no_fc/',
    #     'name': 'No FC layer',
    #     'marker': 'star-triangle-up-open',
    #     'color': 'brown'
    # }
    # selected_models.append(no_fc)

    # warmup = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/warmup_no_warmup/warmup/',
    #     'name': 'Warmup',
    #     'marker': 'star-triangle-down',
    #     'color': 'saddlebrown'
    # }
    # selected_models.append(warmup)

    # no_warmup = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/warmup_no_warmup/no_warmup/',
    #     'name': 'No warmup',
    #     'marker': 'star-triangle-down-open',
    #     'color': 'olive'
    # }
    # selected_models.append(no_warmup)

    # sq_mlm_1 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/1/',
    #     'name': 'MLM-SQUAD-1',
    #     'marker': 'diamond',
    #     'color': 'darkred'
    # }
    # selected_models.append(sq_mlm_1)

    # sq_mlm_6 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/6/',
    #     'name': 'MLM-SQUAD-6',
    #     'marker': 'diamond-tall',
    #     'color': 'orange'
    # }
    # selected_models.append(sq_mlm_6)

    # sq_mlm_10 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/10/',
    #     'name': 'MLM-SQUAD-10',
    #     'marker': 'diamond-wide',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(sq_mlm_10)

    # old_squad_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/old_squad/',
    #     'name': 'MLM-SQUAD-OLD',
    #     'marker': 'diamond-open',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(old_squad_mlm)

    # old_marco_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/old_marco_mlm/',
    #     'name': 'MLM-MSMARCO-OLD',
    #     'marker': 'pentagon-open',
    #     'color': 'purple'
    # }
    # selected_models.append(old_marco_mlm)

    # t5_small_last_layer_trained = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_small/',
    #     'name': 'T5-Small-trained_last_layer',
    #     'marker': 'pentagon-open',
    #     'color': 'purple'
    # }
    # selected_models.append(t5_small_last_layer_trained)

    # t5_small = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_small_og_ll/',
    #     'name': 'T5-Small',
    #     'marker': 'diamond',
    #     'color': 'darkred'
    # }
    # selected_models.append(t5_small)

    # t5_base = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_base_og_ll/',
    #     'name': 'T5-BASE',
    #     'marker': 'circle',
    #     'color': 'green'
    # }
    # selected_models.append(t5_base)

    # t5_base_last_layer_trained = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_base/',
    #     'name': 'T5-Base-trained_last_layer',
    #     'marker': 'diamond',
    #     'color': 'blue'
    # }
    # selected_models.append(t5_base_og_ll)

    # t5_rank = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_rank/og_last_layer/',
    #     'name': 'T5-RANK',
    #     'marker': 'triangle-down',
    #     'color': 'purple'
    # }
    # selected_models.append(t5_rank)

    # t5_rank_trained_ll = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_rank/',
    #     'name': 'T5-RANK-Trained-Last-Layer',
    #     'marker': 'pentagon-open',
    #     'color': 'brown'
    # }
    # selected_models.append(t5_rank_trained_ll)

    # t5_qa = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_qa/og_last_layer',
    #     'name': 'T5-QA',
    #     'marker': 'pentagon-open',
    #     'color': 'coral'
    # }
    # selected_models.append(t5_qa)

    # t5_qa_trained_ll = {
    #     'data_dir': '/home/jonas/git/knowledge-probing-private/data/outputs/t5_qa/',
    #     'name': 'T5-QA-Trained-Last-Layer',
    #     'marker': 'pentagon-open',
    #     'color': 'blue'
    # }
    # selected_models.append(t5_qa_trained_ll)

    return selected_models


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
    else:
        print('Path exists: ', output_dir)


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


if __name__ == "__main__":
    main()
