import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import copy
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def parallel_plots():

    data_dir = 'data/layer_data/classifier_per_layer/long_e100_p15_d0_01/hug_decoder/'
    # data_dir = 'data/layer_data/classifier_per_layer/msmarco_reranking_vs_qa_vs_default_long/hugging_decoder/'

    output_dir = '{}plots/paralel/'.format(data_dir, )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use this to load file other than the qa one
    # qa_file_name = 'marco_data'  # 'qa_data'
    qa_file_name = 'qa_data'
    default_file_name = 'default_data'

    metrics = ['P_AT_1', 'P_AT_10', 'P_AT_K']

    # If not all 12 layer data elements are in the folder
    layer_range = range(0, 12)
    # layer_range = range(4, 12)

    # Pick one arbitrary file to get all the datasets and relations
    data_example = get_file_data(default_file_name, layer_range[-1], data_dir)
    datasets = data_example.keys()

    for dataset in datasets:
        for metric in metrics:

            default_relations_data = []
            qa_relations_data = []
            relations = []

            for relation in tqdm(data_example[dataset].keys()):
                if relation != 'means':
                    relations.append(relation)
                    # list of a specific relations precision values
                    qa_relations_item = {
                        'relation': relation
                    }
                    default_relations_item = {
                        'relation': relation,
                        'precision': []
                    }

                    for layer in layer_range:
                        qa_data = get_file_data(
                            qa_file_name, layer + 1, data_dir)
                        qa_relations_item['p_{}'.format(
                            layer + 1)] = qa_data[dataset][relation][0][metric] * 100

                        default_data = get_file_data(
                            default_file_name, layer + 1, data_dir)
                        default_relations_item['p_{}'.format(
                            layer + 1)] = default_data[dataset][relation][0][metric] * 100
                        default_relations_item['precision'].append(
                            default_data[dataset][relation][0][metric] * 100)

                    default_relations_data.append(default_relations_item)
                    qa_relations_data.append(qa_relations_item)

            # # Gather data
            # l1 = []
            # l2 = []
            # l3 = []
            # l4 = []
            # l5 = []
            # l6 = []
            # l7 = []
            # l8 = []
            # l9 = []
            # l10 = []
            # l11 = []
            # l12 = []
            # for item in default_relations_data:
            #     l1.append(item['p_1'])
            #     l2.append(item['p_2'])
            #     l3.append(item['p_3'])
            #     l4.append(item['p_4'])
            #     l5.append(item['p_5'])
            #     l6.append(item['p_6'])
            #     l7.append(item['p_7'])
            #     l8.append(item['p_8'])
            #     l9.append(item['p_9'])
            #     l10.append(item['p_10'])
            #     l11.append(item['p_11'])
            #     l12.append(item['p_12'])

            # wide_df = pd.DataFrame(
            #     dict(Relations=relations, L1=l1, L2=l2, L3=l3, L4=l4, L5=l5, L6=l6, L7=l7, L8=l8, L9=l9, L10=l10, L11=l11, L12=l12, layer=layer_range))
            # print('Wide: {}'.format(wide_df))
            # tidy_df = wide_df.melt(id_vars='Relations')
            # print('Tidy: {}'.format(tidy_df))

            # Plot
            # fig = px.scatter(tidy_df, x='layer', y='value')

            # fig.update_layout(
            #     xaxis_title="Layer",
            #     yaxis_title=metric,
            # )

            # fig.show()

            fig = go.Figure()

            for item in default_relations_data:
                fig.add_trace(go.Scatter(
                    x=list(layer_range), y=item['precision'], name=item['relation'], hoverinfo=['all']))

            fig.update_layout(title='{}  -  {}'.format(dataset, metric),
                              xaxis_title='Layer',
                              yaxis_title='{}'.format(metric))

            fig.show()

            fig.write_image(
                "{}{}-{}.png".format(output_dir, dataset, metric))

    # plot relations data paralel plot - once per dataset and metric and one for qa and one for default


def get_file_data(file_name, layer, data_dir):
    file_path = '{}{}/{}.json'.format(
        data_dir, layer, file_name)

    assert os.path.isfile(file_path)
    with open(file_path) as json_data:
        file_data = json.load(json_data)

    return file_data
