import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


def produce_layer_wise_plots():
    with open('data/layer_data/classifier_per_layer/1/default_data.json') as json_data:
        default_data_1 = json.load(json_data)
        # print(default_data)

    for dataset in ['Google_RE', 'TREx', 'ConceptNet', 'Squad']:
        for relation in tqdm(default_data_1[dataset].keys()):
            if relation == 'means':
                continue
            for metric in ['P_AT_1', 'P_AT_10', 'P_AT_K']:
                layers = []
                qa_vals = []
                default_vals = []

                for layer in range(0, 12):
                    layers.append(layer + 1)

                    # Load data from files

                    qa_file = 'data/layer_data/classifier_per_layer/{}/qa_data.json'.format(
                        layer + 1)
                    default_file = 'data/layer_data/classifier_per_layer/{}/default_data.json'.format(
                        layer + 1)

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

                wide_df = pd.DataFrame(
                    dict(Layer=layers, Default=default_vals, QA=qa_vals))
                tidy_df = wide_df.melt(id_vars='Layer')

                # Plot
                fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                              title='{} {} [{}]'.format(dataset, relation, metric), )

                fig.update_layout(
                    xaxis_title="Layer",
                    yaxis_title=metric,
                )

                if not os.path.exists("data/layer_data/classifier_per_layer/plots/png/"):
                    os.mkdir("data/layer_data/classifier_per_layer/plots/png/")

                fig.write_image(
                    "data/layer_data/classifier_per_layer/plots/png/{}_{}_{}.png".format(dataset, relation, metric))
