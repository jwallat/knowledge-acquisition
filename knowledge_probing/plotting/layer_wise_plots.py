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


def produce_layer_wise_plots(orderType):

    data_dir = 'data/layer_data/classifier_per_layer/long_e100_p15_d0_01/hug_decoder/'
    # data_dir = 'data/layer_data/classifier_per_layer/msmarco_reranking_vs_qa_vs_default_long/hugging_decoder/'

    output_dir = '{}plots/order_{}/'.format(data_dir, orderType)
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

    compare_all_precisions = False

    # Pick one arbitrary file to get all the datasets and relations
    data_example = get_file_data(default_file_name, layer_range[-1], data_dir)
    datasets = data_example.keys()

    clustering_data = []

    for dataset in datasets:
        for relation in tqdm(data_example[dataset].keys()):
            if relation == 'means':
                if len(data_example[dataset][relation]) > 0:
                    for i, metric in enumerate(metrics):
                        default_mean_values = []
                        qa_mean_values = []
                        layers = []

                        for layer in layer_range:
                            layers.append(layer + 1)

                            # Load data from files\
                            qa_data = get_file_data(
                                qa_file_name, layer + 1, data_dir)
                            means_string = qa_data[dataset]['means'][0]
                            values = handle_mean_values_string(means_string)
                            qa_mean_values.append(values[i] * 100)

                            default_data = get_file_data(
                                default_file_name, layer + 1, data_dir)
                            means_string = default_data[dataset]['means'][0]
                            values = handle_mean_values_string(means_string)
                            default_mean_values.append(values[i] * 100)

                        fig = plot_and_get_figure(
                            default_mean_values, qa_mean_values, layers, dataset, relation, metric)

                        fig.write_image(
                            "{}/{}_{}_{}.png".format(output_dir, dataset, relation, metric))

                continue
            for metric in metrics:
                layers = []
                qa_vals = []
                default_vals = []

                for layer in layer_range:
                    layers.append(layer + 1)

                    # Load data from files\
                    qa_data = get_file_data(qa_file_name, layer + 1, data_dir)
                    # print('Out of range: {}'.format(dataset))
                    # print('Out of range: {}'.format(relation))
                    # print('Out of range: {}'.format(metric))
                    qa_vals.append(
                        qa_data[dataset][relation][0][metric] * 100)

                    default_data = get_file_data(
                        default_file_name, layer + 1, data_dir)
                    default_vals.append(
                        default_data[dataset][relation][0][metric] * 100)

                # Plotting data
                fig = plot_and_get_figure(
                    default_vals, qa_vals, layers, dataset, relation, metric)

                if orderType == 'None':
                    fig.write_image(
                        "{}/{}_{}_{}.png".format(output_dir, dataset, relation, metric))

                elif orderType == 'naive':
                    handle_naive_ordering(
                        fig, default_vals, qa_vals, output_dir, dataset, relation, metric)

                elif orderType == 'clustering':
                    # Prepare the cummulated clustering data for the clustering

                    # Data that will be carried over into clustering. Be aware that not all of this data will
                    # be used in the actual clustering but it may be needed in the generation of plots or to
                    # show additional information in the plots.
                    data_point = {
                        'name': '{}_{}_{}'.format(dataset, relation, metric),
                        'metric': metric,
                        'dataset': dataset,
                        'relation': relation,
                        'default_vals': default_vals,
                        'qa_vals': qa_vals
                    }

                    for layer, _ in enumerate(default_vals):
                        data_point['diff_{}'.format(
                            layer + 1)] = abs(default_vals[layer] - qa_vals[layer])
                        # Curve going down or up?
                        # if layer > 0:
                        #     if default_vals[layer] - default_vals[layer - 1] == 0:
                        #         data_point['curve_default_{}'.format(
                        #             layer + 1)] = 0
                        #     elif default_vals[layer] - default_vals[layer - 1] > 0:
                        #         data_point['curve_default_{}'.format(
                        #             layer + 1)] = 1
                        #     else:
                        #         data_point['curve_default_{}'.format(
                        #             layer + 1)] = -1

                        #     if qa_vals[layer] - qa_vals[layer - 1] == 0:
                        #         data_point['curve_qa_{}'.format(layer + 1)] = 0
                        #     elif qa_vals[layer] - qa_vals[layer - 1] > 0:
                        #         data_point['curve_qa_{}'.format(layer + 1)] = 1
                        #     else:
                        #         data_point['curve_qa_{}'.format(
                        #             layer + 1)] = -1
                        # else:
                        #     data_point['curve_default_{}'.format(
                        #         layer + 1)] = 1 if default_vals[layer] > 0 else 0
                        #     data_point['curve_qa_{}'.format(
                        #         layer + 1)] = 1 if qa_vals[layer] > 0 else 0
                    clustering_data.append(data_point)

    if orderType == 'clustering':
        print('\n\n\n----------------------------------- CLUSTERING -----------------------------------------')

        # print('Many instances: {}'.format(len(clustering_data)))

        # filter by metric
        p1s = []
        p10s = []
        pks = []

        if compare_all_precisions:

            for ele in clustering_data:
                if ele['metric'] == 'P_AT_1':
                    p1s.append(ele)
                elif ele['metric'] == 'P_AT_10':
                    p10s.append(ele)
                else:
                    pks.append(ele)

            for i, clustering_data in enumerate([p1s, p10s, pks]):
                print('-------------------    {}    -----------------------'.format(i))
                make_clustering_plots(output_dir, clustering_data, layer)
        else:
            make_clustering_plots(output_dir, clustering_data, layers)


def handle_mean_values_string(mean_vals_string):
    values_string = mean_vals_string.split(':')[1]

    values = values_string.split(',')

    numeric_vals = []

    for val in values:
        val = val.strip()
        numeric_vals.append(float(val))

    return numeric_vals


def find_layer_differences(qa_vals, default_vals):
    ''' 
    This function is supposed to retun the layer in which the difference between qa val and default 
    val gets bigger than the mean difference on all layers.
    '''
    mean_difference = get_mean_difference(default_vals, qa_vals)

    for layer, _ in enumerate(default_vals):
        difference = abs(default_vals[layer] - qa_vals[layer])

        if difference > mean_difference:
            return layer + 1

    return -1


def get_mean_difference(default, qa):
    sum_differences = 0
    for layer, _ in enumerate(default):
        sum_differences = sum_differences + abs(default[layer] - qa[layer])

    mean_difference = sum_differences / len(default)
    return mean_difference


def do_clustered_plots(cluster_data):
    print('Clustering....')

    data = copy.deepcopy(cluster_data)

    for ele in data:
        ele.pop('name', None)
        ele.pop('metric', None)
        ele.pop('dataset', None)
        ele.pop('relation', None)

        # Also use qa and def values
        for i, val in enumerate(ele['default_vals']):
            ele['qa_{}'.format(i)] = ele['qa_vals'][i]
            ele['default_{}'.format(i)] = val

        ele.pop('qa_vals', None)
        ele.pop('default_vals', None)

    print('Clustering element: {}'.format(data[0]))

    vec = DictVectorizer()

    data = vec.fit_transform(data).toarray()

    best_num_clusters = -1
    best_silhouette_score = -1

    for num_clusters in range(2, len(data) - 1):
        kmeans = KMeans(n_clusters=num_clusters)
        preds = kmeans.fit_predict(data)
        # centers = kmeans.cluster_centers_

        score = silhouette_score(data, preds)
        if score > best_silhouette_score:
            best_num_clusters = num_clusters
            best_silhouette_score = score

        print("For n_clusters = {}, silhouette score is {})".format(
            num_clusters, score))

    print('Chose {} - clusters with the silhouette score of {}'.format(
        best_num_clusters, best_silhouette_score))

    # KMeans
    kmeans = KMeans(n_clusters=best_num_clusters)

    # Predict
    out = kmeans.fit_predict(data)

    tsne_projection(cluster_data, data, out)

    return out


def tsne_projection(data, clustering_data, cluster_assignments):
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE projection")
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    X_tsne = tsne.fit_transform(clustering_data)
    print(X_tsne)

    plot_projection(X_tsne, cluster_assignments, data)


def plot_projection(tsne_data, cluster_assignments, data):
    print('data i {}'.format(data[0]))

    x_data = []
    y_data = []
    metrics = []
    metrics_sizes = []
    relations = []
    clusters = []
    datasets = []

    for i, ele in enumerate(tsne_data):
        x, y = ele
        x_data.append(x)
        y_data.append(y)
        clusters.append('{}-cluster'.format(cluster_assignments[i]))
        relations.append(data[i]['dataset'] + '_' + data[i]['relation'])
        datasets.append(data[i]['dataset'])

        metric = data[i]['metric']
        metrics.append(metric)
        if metric == 'P_AT_1':
            metrics_sizes.append(1)
        elif metric == 'P_AT_10':
            metrics_sizes.append(3)
        else:
            metrics_sizes.append(6)

    # Plotting data
    wide_df = pd.DataFrame(
        dict(x=x_data, y=y_data, cluster=clusters, metric=metrics, metric_size=metrics_sizes, relation=relations, dataset=datasets))
    # tidy_df = wide_df.melt(id_vars='Layer')

    # Plot
    fig = px.scatter(wide_df, x='x', y='y',
                     color='cluster', size='metric_size', hover_data=['metric', 'relation'], symbol='dataset')

    fig.update_layout(
        xaxis_title="D1",
        yaxis_title='D2',
    )

    fig.show()
    # fig.write_image(
    #     "{}plots/png_clustered/{}/{}-cluster_{}.png".format(data_dir, metric, cluster, ele['name']))


def make_clustering_plots(output_dir, clustering_data, layers):
    cluster_assignments = do_clustered_plots(clustering_data)

    os.makedirs(
        "{}P_AT_1/".format(output_dir), exist_ok=True)
    os.makedirs(
        "{}P_AT_10/".format(output_dir), exist_ok=True)
    os.makedirs(
        "{}P_AT_K/".format(output_dir), exist_ok=True)

    for i, ele in tqdm(enumerate(clustering_data)):

        qa_vals = ele['qa_vals']
        default_vals = ele['default_vals']
        dataset = ele['dataset']
        relation = ele['relation']
        metric = ele['metric']

        fig = plot_and_get_figure(
            default_vals, qa_vals, layers, dataset, relation, metric)

        cluster = cluster_assignments[i]

        fig.write_image(
            "{}{}/{}-cluster_{}.png".format(output_dir, metric, cluster, ele['name']))


def get_file_data(file_name, layer, data_dir):
    file_path = '{}{}/{}.json'.format(
        data_dir, layer, file_name)

    assert os.path.isfile(file_path)
    with open(file_path) as json_data:
        file_data = json.load(json_data)

    return file_data


def plot_and_get_figure(default_vals, qa_vals, layers, dataset, relation, metric):
    wide_df = pd.DataFrame(
        dict(Layer=layers, Default=default_vals, QA=qa_vals))
    tidy_df = wide_df.melt(id_vars='Layer')
    print(wide_df)
    print(tidy_df)

    # Plot
    fig = px.line(tidy_df, x='Layer', y='value', color='variable',
                  title='{} {} [{}]'.format(dataset, relation, metric), )

    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title=metric,
    )

    return fig


def handle_naive_ordering(fig, default_vals, qa_vals, output_dir, dataset, relation, metric):
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

    fig.write_image(
        "{}/{}-layer_{}_{}_{}.png".format(output_dir, ordering, dataset, relation, metric))
