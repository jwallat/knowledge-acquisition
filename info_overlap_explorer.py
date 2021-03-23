import os
from datasets import dataset_dict
import torch
from torch.utils import data
import streamlit as st
from dotmap import DotMap
from datasets import load_dataset
from knowledge_probing.probing.probing_args import build_args
from knowledge_probing.file_utils import load_file
from knowledge_probing.datasets.cloze_data_utils import parse_template
from info_overlap import prepare_dataset_for_indexing
import json


def main():
    st.title('Info Overlap Explorer')
    # MODEL
    st.subheader('Dataset selection')

    probing_dir = '/home/jonas/git/knowledge-probing-private/data/probing_data/'

    overlap_datasets = ['SQuAD', 'Wikitext-2', 'Wikitext-103', 'MSMarco']
    overlap_dataset = st.selectbox(
        'For which training dataset do you want to inspect the overlapping text passages?', overlap_datasets)

    overlap_data = get_overlap_data(overlap_dataset)

    probing_dataset_name = st.selectbox(
        'Please select a probing dataset', [dataset for dataset in overlap_data.keys()])

    relation = st.selectbox('What relation do you want to inspect?', [
                            x for x in overlap_data[probing_dataset_name]])

    sample_sentence = st.selectbox('Which one?', [
        x['fact_sentence'][0] for x in overlap_data[probing_dataset_name][relation] if len(x['occ']) > 0])

    print(sample_sentence)
    index = -1
    for i, x in enumerate(overlap_data[probing_dataset_name][relation]):
        if x['fact_sentence'][0] == sample_sentence:
            index = i

    st.write('Matched index at: ', index)

    st.write('The query was observed in the following documents:')

    passages = get_overlap_dataset_passages(overlap_dataset)

    occurences = overlap_data[probing_dataset_name][relation][index]['occ']

    st.write('Occurred in the documents: ', occurences)
    # st.text_area(occurences)
    st.write(passages[int(occurences[0])])
    # st.write(passages[10] + '\n\n\n' + passages[15])


@ st.cache
def get_overlap_data(overlap_dataset):
    overlap_paths = {
        'SQuAD': '/home/jonas/git/knowledge-probing-private/overlap_squad_data.json',
        'Wikitext-2': '/home/jonas/git/knowledge-probing-private/overlap_wikitext-2_data.json',
        'Wikitext-103': '/home/jonas/git/knowledge-probing-private/overlap_wikitext-103_data.json',
        'MSMarco': '/home/jonas/git/knowledge-probing-private/overlap_msmarco_data.json'
    }

    # overlap_data = load_file(overlap_paths[overlap_dataset])

    with open(overlap_paths[overlap_dataset], 'r') as json_file:
        overlap_data = json.load(json_file)
        return overlap_data


@st.cache
def get_overlap_dataset_passages(overlap_dataset):
    if 'Wikitext-2' in overlap_dataset:
        dataset = load_dataset(
            'wikitext', 'wikitext-2-raw-v1', split='train')
    elif 'Wikitext-103' in overlap_dataset:
        dataset = load_dataset(
            'wikitext', 'wikitext-103-raw-v1', split='train')
    elif 'SQuAD' in overlap_dataset:
        dataset = load_dataset('squad', split='train')
    elif 'MSMarco' in overlap_dataset:
        dataset = load_dataset('ms_marco', 'v2.1')['test']

    # Build the dataset e.g., by appending the context to the question for squad
    dataset = prepare_dataset_for_indexing(dataset)

    return dataset


@ st.cache
def load_probing_data(dir):
    dataset_args = []
    dataset_args.append(('Google_RE', build_args(
        'Google_RE', False, dir, 0)))
    dataset_args.append(('TREx', build_args(
        'TREx', False, dir, 0)))
    # dataset_args.append(('ConceptNet', build_args(
    #     'ConceptNet', False, dir, 0)))
    dataset_args.append(('Squad', build_args(
        'Squad', False, dir, 0)))

    dataset_dict = {}
    names = []

    for dataset_touple in dataset_args:
        dataset_dict[dataset_touple[0]] = dataset_touple[1]
        names.append(dataset_touple[0])

    return dataset_dict, names


if __name__ == "__main__":
    main()
