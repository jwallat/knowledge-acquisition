import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


def produce_plots():
    with open('data/layer_data/default/default_data.json') as json_data:
        default_data = json.load(json_data)
        # print(default_data)

    with open('data/layer_data/qa/qa_data.json') as json_data:
        qa_data = json.load(json_data)
        # print(qa_data)

    with open('data/layer_data/qa_trained/qa_trained_data.json') as json_data:
        qa_trained_data = json.load(json_data)
        # print(qa_data)

    # dataset = 'Google_RE'
    # relation = 'date_of_birth'
    # metric = 'P_AT_1'

    for dataset in ['Google_RE', 'TREx', 'ConceptNet', 'Squad']:
        for relation in tqdm(default_data[dataset].keys()):
            if relation == 'means':
                continue
            for metric in ['P_AT_1', 'P_AT_10', 'P_AT_K']:
                layers = []
                qa_vals = []
                qa_trained_vals = []
                default_vals = []

                for layer in range(0, 12):
                    layers.append(layer + 1)
                    qa_vals.append(qa_data[dataset]
                                   [relation][layer][metric] * 100)
                    qa_trained_vals.append(qa_trained_data[dataset]
                                           [relation][layer][metric] * 100)
                    default_vals.append(
                        default_data[dataset][relation][layer][metric] * 100)

                wide_df = pd.DataFrame(
                    dict(Layer=layers, Default=default_vals, QA=qa_vals, QA_trained=qa_trained_vals))
                tidy_df = wide_df.melt(id_vars='Layer')

                # print(tidy_df)

                # Plot
                fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                              title='{} {} [{}]'.format(dataset, relation, metric), )

                fig.update_layout(
                    xaxis_title="Layer",
                    yaxis_title=metric,
                )

                if not os.path.exists("data/layer_data/plots/png/"):
                    os.mkdir("data/layer_data/plots/png/")

                fig.write_image(
                    "data/layer_data/plots/png/{}_{}_{}.png".format(dataset, relation, metric))

                # fig.show()
