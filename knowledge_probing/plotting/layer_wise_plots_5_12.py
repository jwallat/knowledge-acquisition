import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import copy
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans


def produce_layer_wise_plots(orderType):

    data_dir = 'data/layer_data/classifier_per_layer/msmarco_reranking_vs_qa_vs_default_long/hugging_decoder/'
    MODEL_NAME = 'MSMARCO'
    third_model_file_name = 'marco_data.json'

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
                default_vals = []

                for layer in range(4, 12):
                    layers.append(layer + 1)

                    # Load data from files

                    qa_file = '{}{}/{}'.format(
                        data_dir, layer + 1, third_model_file_name)
                    default_file = '{}{}/default_data.json'.format(
                        data_dir, layer + 1)

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

                # Plotting data
                wide_df = pd.DataFrame(
                    dict(Layer=layers, Default=default_vals, MSMARCO=qa_vals))
                tidy_df = wide_df.melt(id_vars='Layer')

                # Plot
                fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                              title='{} {} [{}]'.format(dataset, relation, metric), )

                fig.update_layout(
                    xaxis_title="Layer",
                    yaxis_title=metric,
                )

                if orderType == 'None':
                    if not os.path.exists("{}plots/png/".format(data_dir)):
                        os.mkdir("{}plots/png/".format(data_dir))

                    fig.write_image(
                        "{}plots/png/{}_{}_{}.png".format(data_dir, dataset, relation, metric))

                elif orderType == 'naive':

                    # Getting an order on this data
                    diff_layer = find_layer_differences(qa_vals, default_vals)

                    if diff_layer == -1:
                        ordering = -1
                    elif diff_layer >= 1 and diff_layer <= 3:
                        ordering = 'early'
                    elif diff_layer >= 4 and diff_layer <= 9:
                        ordering = 'middle'
                    elif diff_layer >= 10 and diff_layer <= 12:
                        ordering = 'late'
                    else:
                        ordering = 'Unknown-ordering'

                    if not os.path.exists("{}plots/png_naive_ordering/".format(data_dir)):
                        os.mkdirs(
                            "{}plots/png_naive_ordering/".format(data_dir))

                    fig.write_image(
                        "{}plots/png_naive_ordering/{}-layer_{}_{}_{}.png".format(data_dir, ordering, dataset, relation, metric))

                elif orderType == 'clustering':

                    data_point = {
                        'name': '{}_{}_{}'.format(dataset, relation, metric),
                        'default_vals': default_vals,
                        'qa_vals': qa_vals
                    }

                    for layer in range(12):
                        data_point['diff_{}'.format(
                            layer + 1)] = abs(default_vals[layer] - qa_vals[layer])
                        # Curve going down or up?
                        if layer > 0:
                            if default_vals[layer] - default_vals[layer - 1] == 0:
                                data_point['curve_default_{}'.format(
                                    layer + 1)] = 0
                            elif default_vals[layer] - default_vals[layer - 1] > 0:
                                data_point['curve_default_{}'.format(
                                    layer + 1)] = 1
                            else:
                                data_point['curve_default_{}'.format(
                                    layer + 1)] = -1

                            if qa_vals[layer] - qa_vals[layer - 1] == 0:
                                data_point['curve_qa_{}'.format(layer + 1)] = 0
                            elif qa_vals[layer] - qa_vals[layer - 1] > 0:
                                data_point['curve_qa_{}'.format(layer + 1)] = 1
                            else:
                                data_point['curve_qa_{}'.format(
                                    layer + 1)] = -1
                        else:
                            data_point['curve_default_{}'.format(
                                layer + 1)] = 1 if default_vals[layer] > 0 else 0
                            data_point['curve_qa_{}'.format(
                                layer + 1)] = 1 if qa_vals[layer] > 0 else 0
                    clustering_data.append(data_point)

                else:
                    print('Unknown order type')

    if orderType == 'clustering':
        num_clusters = 8

        print(clustering_data[0])

        cluster_assignments = do_clustered_plots(
            data_dir, clustering_data, num_clusters)

        print(clustering_data[0])

        if not os.path.exists("{}plots/png_clustered/".format(data_dir)):
            os.mkdirs("{}plots/png_clustered/".format(data_dir))

        for i, ele in enumerate(clustering_data):

            qa_vals = ele['qa_vals']
            default_vals = ele['default_vals']

            # Plotting data
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

            cluster = cluster_assignments[i]

            fig.write_image(
                "{}plots/png_clustered/{}-cluster_{}.png".format(data_dir, cluster, ele['name']))


def find_layer_differences(qa_vals, default_vals):
    ''' 
    This function is supposed to retun the layer in which the difference between qa val and default 
    val gets bigger than the mean difference on all layers.
    '''
    mean_difference = get_mean_difference(default_vals, qa_vals)

    for layer in range(12):
        difference = abs(default_vals[layer] - qa_vals[layer])

        if difference > mean_difference:
            return layer + 1

    return -1


def get_mean_difference(default, qa):
    sum_differences = 0
    for layer in range(12):
        sum_differences = sum_differences + abs(default[layer] - qa[layer])

    mean_difference = sum_differences / len(default)
    return mean_difference


def do_clustered_plots(data_dir, cluster_data, num_clusters):
    print('Clustering....')

    data = copy.deepcopy(cluster_data)

    for ele in data:
        ele.pop('name', None)
        ele.pop('qa_vals', None)
        ele.pop('default_vals', None)

    vec = DictVectorizer()

    transformed = vec.fit_transform(data).toarray()

    # KMeans
    kmeans = KMeans(n_clusters=num_clusters)

    # fit
    kmeans.fit(transformed)

    # Predict
    out = kmeans.fit_predict(transformed)

    return out
