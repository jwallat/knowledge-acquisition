import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import copy
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans


def produce_layer_wise_plots():

    data_dir = 'data/layer_data/classifier_per_layer/ner_vs_qa_squad2_vs_default_cased_e100_p15_d0_01/hugging_decoder/'
    MODEL_NAME = 'MSMARCO'
    third_model_file_name = 'ner_data.json'

    with open('{}5/default_data.json'.format(data_dir)) as json_data:
        default_data_1 = json.load(json_data)
        # print(default_data)

    clustering_data = []

    for dataset in ['Google_RE', 'TREx', 'ConceptNet', 'Squad']:
        for relation in tqdm(default_data_1[dataset].keys()):
            if relation == 'means':
                continue
            for metric in ['P_AT_1', 'P_AT_10', 'P_AT_K']:
                layers = []
                qa_vals = []
                third_model_vals = []
                default_vals = []

                for layer in range(4, 12):
                    layers.append(layer + 1)

                    # Load data from files

                    qa_file = '{}{}/qa_data.json'.format(
                        data_dir, layer + 1)
                    default_file = '{}{}/default_data.json'.format(
                        data_dir, layer + 1)
                    third_model_file = '{}{}/{}'.format(
                        data_dir, layer + 1, third_model_file_name)

                    if os.path.isfile(qa_file):
                        with open(qa_file) as json_data:
                            qa_data = json.load(json_data)

                            qa_vals.append(qa_data[dataset]
                                           [relation][0][metric] * 100)
                    else:
                        qa_vals.append(-1)

                    if os.path.isfile(default_file):
                        with open(default_file) as json_data:
                            default_data = json.load(json_data)

                            default_vals.append(
                                default_data[dataset][relation][0][metric] * 100)
                    else:
                        default_vals.append(-1)

                    if os.path.isfile(third_model_file):
                        with open(third_model_file) as json_data:
                            third_model_data = json.load(json_data)

                            third_model_vals.append(
                                third_model_data[dataset][relation][0][metric] * 100)
                    else:
                        third_model_vals.append(-1)

                # Plotting data
                wide_df = pd.DataFrame(
                    dict(Layer=layers, Default=default_vals, QA_SQuAD2=qa_vals, NER=third_model_vals))
                tidy_df = wide_df.melt(id_vars='Layer')

                # Plot
                fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                              title='{} {} [{}]'.format(dataset, relation, metric), )

                fig.update_layout(
                    xaxis_title="Layer",
                    yaxis_title=metric,
                )

                if not os.path.exists("{}plots_compare_3/png/".format(data_dir)):
                    os.mkdir("{}plots_compare_3/png/".format(data_dir))

                fig.write_image(
                    "{}plots_compare_3/png/{}_{}_{}.png".format(data_dir, dataset, relation, metric))
