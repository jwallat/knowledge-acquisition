from os import remove
from re import search
from knowledge_probing.probing.probing_args import build_args
from dotmap import DotMap
from knowledge_probing.file_utils import write_metrics, write_to_execution_log, stringify_dotmap, get_vocab
from knowledge_probing.models.lightning.og_t5_model import OGT5Model
from knowledge_probing.models.lightning.t5_decoder import T5Decoder
from knowledge_probing.models.lightning.bert_decoder import BertDecoder
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from knowledge_probing.training.training import training
from knowledge_probing.probing.probing import probing
from knowledge_probing.config.config_helper import handle_config
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_dataset import ClozeDataset
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, parse_template
from krovetzstemmer import Stemmer
import sys
import os
import torch
import functools
import multiprocessing as mp
import json
from tqdm import tqdm
from knowledge_probing.file_utils import load_file
from datasets import load_dataset
import wandb
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('stopwords')


def prepare_dataset_for_indexing(dataset):
    patches = []
    print('prepare dataset')
    for ele in tqdm(dataset):
        text = ''
        if 'text' in ele.keys():
            # Wiki
            text = ele['text']
        elif 'context' in ele.keys() and 'question' in ele.keys():
            # Squad
            text = ele['question'] + ' ' + ele['context']
        elif 'query' in ele.keys() and 'passages' in ele.keys():
            concatenated_passages = ''
            for passage in ele['passages']['passage_text']:
                concatenated_passages = concatenated_passages + " " + passage
            text = ele['query'] + concatenated_passages
        else:
            print('Could not find something in the dataset element: ', ele)
        patches.append(text)

    return patches


def remove_stopwords_and_stem(data):
    # ps = PorterStemmer()
    lemm = WordNetLemmatizer()
    stem = Stemmer()

    filtered_data = []
    for i, text in enumerate(tqdm(data)):
        tokens = word_tokenize(text)

        filtered_tokens = [
            token for token in tokens if not token in stopwords.words()]
        stemmed_tokens = [lemm.lemmatize(token) for token in filtered_tokens]
        filtered_data.append(stemmed_tokens)

    return filtered_data


def get_inverted_index(dataset, num_proccesses, dataset_name):
    inv_index_file = 'inv_index_{}.json'.format(dataset_name)

    if os.path.isfile(inv_index_file):
        with open(inv_index_file, 'r') as json_file:
            inv_index = json.load(json_file)
            return inv_index

    text_dataset = prepare_dataset_for_indexing(dataset)
    print(text_dataset[8])
    # text_dataset = text_dataset[:1000]
    print('Remove punctuation')
    text_dataset = remove_punctuation(text_dataset)

    sample_chunks = chunkIt(text_dataset, num_proccesses)

    pool = mp.Pool(num_proccesses)
    results = pool.map(remove_stopwords_and_stem, sample_chunks)

    results_from_mp = []
    for res in results:
        results_from_mp.extend(res)

    print('Build index')
    inv_index = compute_inverted_index(results_from_mp)

    # Safe inverted index
    with open(inv_index_file, 'w') as outfile:
        json.dump(inv_index, outfile)
    # print(inv_index)


def compute_inverted_index(tokens_dataset):
    inv_index = {}

    for document_idx, document in enumerate(tqdm(tokens_dataset)):
        for token in document:
            if token not in inv_index:
                inv_index[token] = [document_idx]
            else:
                if document_idx not in inv_index[token]:
                    inv_index[token].append(document_idx)

    return inv_index


def remove_punctuation(data):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~='''

    for i, text in enumerate(tqdm(data)):
        # print(text)
        for ele in text:
            if ele in punc:
                # print('Ele is in punc: ', ele)
                text = text.replace(ele, " ")

        data[i] = text.lower()

    return data


def main(args):

    wandb.init(project='Info-Overlap', name='Wiki')

    probing_data_dir = '/home/jonas/git/knowledge-probing-private/data/probing_data/'

    dataset_name = 'squad'

    num_proccesses = 9

    # Dataset to ceck against
    # dataset = load_dataset(
    #     'wikitext', 'wikitext-103-raw-v1', split='train')
    # dataset = load_dataset(
    #     'wikitext', 'wikitext-2-raw-v1', split='train')

    dataset = load_dataset(dataset_name, split='train')
    # dataset = load_dataset('ms_marco', 'v2.1')['train']

    # Prepare dataset for matching
    inv_index = get_inverted_index(dataset, num_proccesses, dataset_name)
    # return

    # dataset = prepare_dataset_for_matching(dataset)

    # Get the stuff the we need for the script
    # print(args)

    # Get probing data stuff
    dataset_args = []
    dataset_args.append(('Google_RE', build_args(
        'Google_RE', False, probing_data_dir, 0)))
    dataset_args.append(('TREx', build_args(
        'TREx', False, probing_data_dir, 0)))
    # dataset_args.append(('ConceptNet', build_args(
    #     'ConceptNet', False, probing_data_dir, 0)))
    dataset_args.append(('Squad', build_args(
        'Squad', False, probing_data_dir, 0)))

    occurrences = {}

    print('$$$$$$$$$$$$$$$$$$$$$$$    Probing model of type       $$$$$$$$$$$$$$$$$$$$$$$$')
    for ele in dataset_args:
        ds_name, relation_args_list = ele

        occurrences[ds_name] = {}

        print('*****************   {}   **********************'.format(ds_name))

        for i, relation_args in enumerate(relation_args_list):
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            print(
                '---------------- {} {} of {} ----------------------'.format(relation_args.template, i+1, len(relation_args_list)))
            print(stringify_dotmap(relation_args))

            # Build ClozeDataset the get the same examples as in the real case
            samples = load_file(args.relation_args.dataset_filename)

            occurrences[ds_name][relation_args['relation']] = []

            # handle multiprocessing
            sample_chunks = chunkIt(samples, num_proccesses)

            partial_handle = functools.partial(
                handle_samples, args=args, dataset=dataset, inv_index=inv_index)

            pool = mp.Pool(num_proccesses)
            results = pool.map(partial_handle, sample_chunks)

            results_from_mp = []
            for res in results:
                results_from_mp.extend(res)

            occurrences[ds_name][relation_args['relation']] = results_from_mp

            # print statistics
            print_statistics(occurrences[ds_name][relation_args['relation']])

    # Print overall statistics:
    print_global_stats(occurrences)

    # Safe the accumulated data
    with open('overlap_{}_data.json'.format(dataset_name), 'w') as outfile:
        json.dump(occurrences, outfile)


def print_global_stats(data):

    for dataset in data.keys():
        print('$$$$$$$$$$$$$$$$$$$    {}    $$$$$$$$$$$$$$$$$$$'.format(dataset))
        sum_aggregates_overlap_percentages = 0
        num_relations = len(data[dataset].keys())
        for relation in data[dataset].keys():
            print('------------------  {}  ------------------'.format(relation))
            num_instances = len(data[dataset][relation])
            instances = data[dataset][relation]
            num_with_fact_sentences = 0
            for item in instances:
                num_spans = len(item['occ'])
                if num_spans > 0:
                    num_with_fact_sentences = num_with_fact_sentences + 1
            print('{} of facts covered. This is {}/{}'.format((num_with_fact_sentences /
                                                               num_instances), num_with_fact_sentences, num_instances))
            sum_aggregates_overlap_percentages = sum_aggregates_overlap_percentages + \
                (num_with_fact_sentences/num_instances)
            wandb.log(
                {'Relation': relation, 'Relation_Overlap': (num_with_fact_sentences/num_instances)})

        print('Total stats for {}: Of {} relation, the average coverage is {}'.format(
            dataset, num_relations, (sum_aggregates_overlap_percentages/num_relations)))
        wandb.log(
            {'Dataset': dataset, 'Dataset_Overlap': (sum_aggregates_overlap_percentages/num_relations)})


def handle_samples(samples, args, dataset, inv_index=None):
    items = []

    for sample in tqdm(samples):
        template_sentence = parse_template(
            args.relation_args.template.strip(
            ), sample["sub_label"].strip(), sample['obj_label'].strip()
        )
        item = {
            'fact_sentence': template_sentence,
            'sub': sample['sub_label'],
            'obj': sample['obj_label'],
            'occ': []
        }

        # search in dataset
        # fact_occurrences = find_occurrence(
        #     dataset, sample['sub_label'], sample['obj_label'])

        # search in inverted index
        fact_occurrences = search_in_index(
            dataset, sample['sub_label'], sample['obj_label'], inv_index)

        # Add occurence to item
        item['occ'] = fact_occurrences
        items.append(item)

    return items


def search_in_index(dataset, sub, obj, inv_index):
    # get stemmed tokens for the inverted index searching
    sub_tokens = make_words_index_searchable(sub)
    obj_token = make_words_index_searchable(obj)

    if len(sub_tokens) == 0 or len(obj_token) == 0:
        return []

    sub_tokens.extend(obj_token)
    search_tokens = sub_tokens[1:]
    # print(sub_tokens)
    # print('modded sub: ', sub_tokens[1:])
    # print(obj_token)
    # print('Search tokens', search_tokens)
    if sub_tokens[0] in inv_index:
        potential_documents = inv_index[sub_tokens[0]]
        for token in search_tokens:
            if token in inv_index:
                token_documents = inv_index[token]
                # Filter out all documents for that do not have both tokens in them
                potential_documents = [
                    value for value in potential_documents if value in token_documents]
            else:
                # print('Token < {} > not in inverted index, aborting'.format(token))
                return []
        # Found documents that contain all tokens
        # Now find the fulltext in the dataset
        # print('{} documents left after filtering'.format(
        #     len(potential_documents)))
        occs = []
        for doc in potential_documents:
            occs.append(get_text_from_dataset(dataset, doc))
        return occs
    else:
        # print('First Token < {} > not in inverted index, aborting'.format(
        # sub_tokens[0]))
        return []


def get_text_from_dataset(dataset, doc_id):
    if 'text' in dataset[doc_id].keys():
        # Wiki
        text = dataset[doc_id]['text']
    elif 'context' in dataset[doc_id].keys() and 'question' in dataset[doc_id].keys():
        # Squad
        text = dataset[doc_id]['question'] + ' ' + dataset[doc_id]['context']
    elif 'query' in dataset[doc_id].keys() and 'passages' in dataset[doc_id].keys():
        # TODO: Decide if I want to add a the query in front of all passages or not
        concatenated_passages = ''


def print_statistics(items):
    num_items = len(items)
    num_one_span = 0
    num_multiple_spans = 0

    for item in items:
        num_spans = len(item['occ'])
        if num_spans > 1:
            num_multiple_spans = num_multiple_spans + 1
        elif num_spans > 0:
            num_one_span = num_one_span + 1
            # print(item)

    print('Of {}, {} have one and {} have multiple spans found containing the answer'.format(
        num_items, num_one_span, num_multiple_spans))


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def make_words_index_searchable(text):
    lemm = WordNetLemmatizer()
    stem = Stemmer()

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = [
        token for token in tokens if not token in stopwords.words()]
    stemmed_tokens = [lemm.lemmatize(token) for token in filtered_tokens]

    return stemmed_tokens


def find_occurrence(dataset, sub, obj):
    occurrences = []
    avg_word_len = 7

    for text in dataset:
        found = text.find(sub)

        if found != -1:
            # check if the answer is in a window of 15 words around the given answer
            lower_index = found - 15 * avg_word_len
            if lower_index < 0:
                lower_index = 0
            upper_index = found + len(sub) + 15 * avg_word_len
            window_text = text[lower_index:upper_index]

            found_answer = window_text.find(obj)
            if found_answer != -1:
                occurrences.append(window_text)

    return occurrences


if __name__ == '__main__':
    parser = ArgumentParser(add_help=True)

    parser = BaseDecoder.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)

    # Decoder
    parser.add_argument('--decoder_initialization', required='--do_training' in sys.argv,
                        choices=['pre-trained', 'random'],
                        help='Use either the huggingface_pretrained_decoder, which was used during pre-training of BERT, or a randomly initialized decoder')

    # Training
    parser.add_argument('--do_training', default=False, action='store_true')
    parser.add_argument('--training_early_stop_delta', default=0.01, type=float,
                        help='The minimum validation-loss-delta between #patience iterations that has to happen for the computation not to stop')
    parser.add_argument('--training_early_stop_patience', default=15, type=int,
                        help='The patience for the models validation loss to improve by [training_early_stop_delta] for the computation not to stop')

    # Probing
    parser.add_argument('--do_probing', default=False, action='store_true')
    parser.add_argument('--use_original_model',
                        default=False, action='store_true')
    # parser.add_argument('--probing_layer', default=12,
    #                     choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int)
    parser.add_argument('--probing_layer', default=12, type=int)

    # Other
    parser.add_argument('--run_name', default='',
                        help='Name of the run that will be used when building the run_identifier')
    parser.add_argument('--output_base_dir', default='data/outputs/',
                        help='Path to the output dir that will contain the logs and trained models')
    parser.add_argument('--seed', default=42, type=int)

    # Wandb
    parser.add_argument('--use_wandb_logging', default=False, action='store_true',
                        help='Use this flag to use wandb logging. Otherwise we will use the pytorch-lightning tensorboard logger')
    parser.add_argument('--wandb_project_name', required='--use_wandb_logging' in sys.argv, type=str,
                        help='Name of wandb project')
    parser.add_argument('--wandb_run_name', default='',
                        type=str, help='Name of wandb run')
    parser.add_argument('--python_executable', required='--use_wandb_logging' in sys.argv, type=str, default='/usr/bin/python3',
                        help='Some cluster environments might require to set the sys.executable for wandb to work')

    args = parser.parse_args()

    main(args)
