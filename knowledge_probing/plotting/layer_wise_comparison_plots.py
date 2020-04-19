import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


def produce_layer_wise_comparison_plots():

    data_dir1 = 'data/layer_data/classifier_per_layer/long_e100_p15_d0_01/hug_decoder/'
    data_dir2 = 'data/layer_data/classifier_per_layer/short_e35_p12_d0_02/'

    with open('{}1/default_data.json'.format(data_dir1)) as json_data:
        default_data_1 = json.load(json_data)
        # print(default_data)

    for dataset in ['Google_RE', 'TREx', 'ConceptNet', 'Squad']:
        for relation in tqdm(default_data_1[dataset].keys()):
            if relation == 'means':
                continue
            for metric in ['P_AT_1', 'P_AT_10', 'P_AT_K']:
                layers = []
                qa_vals1 = []
                default_vals1 = []
                qa_vals2 = []
                default_vals2 = []

                for layer in range(0, 12):
                    layers.append(layer + 1)

                    # Load data from files

                    qa_file1 = '{}{}/qa_data.json'.format(
                        data_dir1, layer + 1)
                    default_file1 = '{}{}/default_data.json'.format(
                        data_dir1, layer + 1)

                    qa_file2 = '{}{}/qa_data.json'.format(
                        data_dir2, layer + 1)
                    default_file2 = '{}{}/default_data.json'.format(
                        data_dir2, layer + 1)

                    # Data for folder 1
                    with open(qa_file1) as json_data:
                        qa_data1 = json.load(json_data)

                        qa_vals1.append(qa_data1[dataset]
                                        [relation][0][metric] * 100)

                    with open(default_file1) as json_data:
                        default_data1 = json.load(json_data)

                        default_vals1.append(
                            default_data1[dataset][relation][0][metric] * 100)

                    # Data for folder 2
                    with open(qa_file2) as json_data:
                        qa_data2 = json.load(json_data)

                        qa_vals2.append(qa_data2[dataset]
                                        [relation][0][metric] * 100)

                    with open(default_file2) as json_data:
                        default_data2 = json.load(json_data)

                        default_vals2.append(
                            default_data2[dataset][relation][0][metric] * 100)

                wide_df = pd.DataFrame(
                    dict(Layer=layers, Default_Hug_Long=default_vals1, QA_Hug_Long=qa_vals1, Default_Hug_Short=default_vals2, QA_Hug_Short=qa_vals2))
                tidy_df = wide_df.melt(id_vars='Layer')

                # Plot
                fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                              title='{} {} [{}]'.format(dataset, relation, metric), )

                fig.update_layout(
                    xaxis_title="Layer",
                    yaxis_title=metric,
                )

                plot_dir = 'data/layer_data/classifier_per_layer/plots_long_short_comparison/'

                if not os.path.exists("{}png/".format(plot_dir)):
                    os.mkdir("{}png/".format(plot_dir))

                fig.write_image(
                    "{}png/{}_{}_{}.png".format(plot_dir, dataset, relation, metric))
