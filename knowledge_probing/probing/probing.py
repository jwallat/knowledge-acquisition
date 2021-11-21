from argparse import Namespace
from transformers import AutoTokenizer
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.datasets.cloze_dataset import ClozeDataset
from knowledge_probing.datasets.cloze_dataset4paq import ClozeDataset4paq
from knowledge_probing.datasets.cbqa_dataset import cbqa_Dataset
from knowledge_probing.models.t5_model_util import qa_tokens
from knowledge_probing.probing.metrics import calculate_metrics, mean_precisions, aggregate_metrics_elements, \
    cbqa_metrics, compute_exact_match, compute_f1
from knowledge_probing.probing.probing_args import build_args
from knowledge_probing.plotting.plots import handle_mean_values_string
from knowledge_probing.file_utils import write_metrics, write_to_execution_log, stringify_dotmap, get_vocab
from dotmap import DotMap
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import gc, sys, torch
import os
import json
import functools
import wandb


def probing(args, decoder):
    # Use the correct probing model
    probing_model, tokenizer = get_probing_model(args, decoder)
    # vocab = get_vocab(args.model_type)
    # dataset_args = get_probing_dataset_args(args)
    # Do the probing
    # data = probe(args, probing_model, tokenizer, dataset_args, layer=args.probing_layer, vocab=vocab)
    # save_probing_data(args, data)
    # probe4paq(args, probing_model, tokenizer, layer=args.probing_layer)
    probe4cbqa(args, probing_model, tokenizer, layer=args.probing_layer)
    return


def get_probing_dataset_args(args):
    """
    Choose datasets to probe
    The dataset loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)
    """
    dataset_args = []
    dataset_args.append(('Google_RE', build_args(
        'Google_RE', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    # dataset_args.append(('Google_RE_UHN', build_args(
    #     'Google_RE_UHN', args.lowercase, args.probing_data_dir, args.precision_at_k, model_type=args.model_type)))
    dataset_args.append(('TREx', build_args(
        'TREx', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    # dataset_args.append(('TREx_UHN', build_args(
    #     'TREx_UHN', args.lowercase, args.probing_data_dir, args.precision_at_k, model_type=args.model_type)))
    dataset_args.append(('ConceptNet', build_args(
        'ConceptNet', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    dataset_args.append(('Squad', build_args(
        'Squad', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    # print(dataset_args)
    # print(len(dataset_args))
    return dataset_args


def get_probing_model(args, decoder):
    tokenizer = decoder.tokenizer
    probing_model = decoder

    probing_model.to(args.device)
    probing_model.zero_grad()
    probing_model.set_to_eval()

    return probing_model, tokenizer


def probe4paq(args: Namespace, probing_model: BaseDecoder, tokenizer: AutoTokenizer, layer: int):
    write_to_execution_log(
        100 * '+' + '\t Probing layer with PAQ: {} \t'.format(layer) + 100 * '+', append_newlines=True,
        path=args.execution_log)
    print('##################################      Layer: {}      #########################################'.format(
        layer))
    print('$$$$$$$$$$$$$$$$$$$$$$$    Probing model of type with PAQ: {}      $$$$$$$$$$$$$$$$$$$$$$$$'.format(
        args.model_type))

    probing_model.total_num_probing_steps = 0
    dataset = ClozeDataset4paq(probing_model, tokenizer, args, tokenizer.model_max_length, output_debug_info=False)
    my_collate = functools.partial(cloze_collate4paq, tokenizer=tokenizer)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.probing_batch_size, collate_fn=my_collate)

    metrics_elements_num = 0
    correct_num = 0
    grad_sum = 0
    num_tokens = 0
    token_1 = 0
    token_2 = 0
    token_3 = 0
    token_4 = 0
    token_more = 0
    P1_num = 0
    P10_num = 0
    Pk_num = 0
    wrong_answer = []
    for batch in dataloader:
        if probing_model.total_num_probing_steps % 5 == 0:
            show = True
        else:
            show = False
        probing_model.total_num_probing_steps += 1
        metrics_elements_from_batch = probing_model.probe4paq(batch, layer=layer, show=show,
                                                              log_path=args.execution_log)
        for i in metrics_elements_from_batch:
            metrics_elements_num += 1
            wrong_answer_dict = {}
            if i['correct']:
                correct_num += 1
            else:
                if len(wrong_answer) <= 100:
                    wrong_answer_dict['answer_token'] = i['answer_token']
                    wrong_answer_dict['top10_span'] = i['top10_span']
                    wrong_answer_dict['answer'] = i['answer']
                    wrong_answer_dict['question'] = i['question']
                    wrong_answer.append(wrong_answer_dict)
            grad_sum += i['grad']
            num_tokens += i['num_tokens']
            P1_num += i['P_AT_1']
            P10_num += i['P_AT_10']
            Pk_num += i['P_AT_K']
            if i['num_tokens'] == 1:
                token_1 += 1
            elif i['num_tokens'] == 2:
                token_2 += 1
            elif i['num_tokens'] == 3:
                token_3 += 1
            elif i['num_tokens'] == 4:
                token_4 += 1
            elif i['num_tokens'] >= 5:
                token_more += 1
    # gc.collect()
    mrr_mean = grad_sum / num_tokens
    P1_mean = P1_num / num_tokens
    P10_mean = P10_num / num_tokens
    Pk_mean = Pk_num / num_tokens
    print('num_QA-pairs: {}'.format(metrics_elements_num))
    print('sum_tokens: {}'.format(num_tokens))
    print('MRR_mean: {}'.format(mrr_mean))
    print('Total completely correct: {}'.format(correct_num))
    print('token_1: ', token_1)
    print('token_2: ', token_2)
    print('token_3: ', token_3)
    print('token_4: ', token_4)
    print('token_>=5: ', token_more)
    print('Mean P@1,10,k: {}, {}, {}'.format(P1_mean, P10_mean, Pk_mean))
    # print('!!', args.execution_log)
    write_to_execution_log('=' * 45, append_newlines=True, path=args.execution_log)
    write_to_execution_log('100 examples that are not completely correct', append_newlines=True,
                           path=args.execution_log)
    write_to_execution_log('↓' * 45, append_newlines=True, path=args.execution_log)
    for i in wrong_answer:
        write_to_execution_log(
            '-------------------------------------------------\nQuestion: {}\nAnswer: {}\n{:<10}, Top10 predictions'.format(
                i['question'], i['answer'], 'GT token'),
            append_newlines=True, path=args.execution_log)
        for id, token in enumerate(i['answer_token']):
            write_to_execution_log('{:<10}, {}'.format(token, i['top10_span'][id]), append_newlines=True,
                                   path=args.execution_log)

    write_to_execution_log('----num_QA-pairs: {}'.format(metrics_elements_num),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('-----sum_tokens:  {}'.format(num_tokens),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('-----MRR_mean:    {}'.format(mrr_mean),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('-----total completely correct: {}'.format(correct_num),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('Mean P@1,10,k: {}, {}, {}'.format(P1_mean, P10_mean, Pk_mean),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('token_1: {} \ntoken_2: {} \ntoken_3: {} \ntoken_4: {} \ntoken_>=5: {} \n'.format(
        token_1, token_2, token_3, token_4, token_more),
        append_newlines=True, path=args.execution_log)

    return


def probe(args: Namespace, probing_model: BaseDecoder, tokenizer: AutoTokenizer, dataset_args, layer: int, vocab):
    write_to_execution_log(
        100 * '+' + '\t Probing layer: {} \t'.format(layer) + 100 * '+', append_newlines=True, path=args.execution_log)
    print('##################################      Layer: {}      #########################################'.format(
        layer))

    layer_data = {}

    google_re_metrices = []
    trex_metrices = []

    my_collate = functools.partial(
        probing_model.cloze_collate, tokenizer=tokenizer)

    print('$$$$$$$$$$$$$$$$$$$$$$$    Probing model of type: {}      $$$$$$$$$$$$$$$$$$$$$$$$'.format(
        args.model_type))
    for ele in dataset_args:
        ds_name, relation_args_list = ele

        layer_data[ds_name] = {}
        layer_data[ds_name]['means'] = []

        print('*****************   {}   **********************'.format(ds_name))

        for relation_args in relation_args_list:

            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            print('---------------- {} ----------------------'.format(args.relation_args.template))
            print(stringify_dotmap(args.relation_args))
            layer_data[ds_name][args.relation_args.relation] = []
            dataset = ClozeDataset(probing_model, tokenizer, args, vocab, tokenizer.model_max_length,
                                   output_debug_info=False)
            # Create dataloader
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.probing_batch_size, collate_fn=my_collate)

            metrics_elements = []
            for batch in dataloader:
                # for _, batch in enumerate(tqdm(dataloader)):
                if probing_model.total_num_probing_steps % 5 == 0:
                    show = True
                else:
                    show = False
                probing_model.total_num_probing_steps += 1
                metrics_elements_from_batch = probing_model.probe(batch, layer=layer, relation_args=relation_args,
                                                                  show=show)
                metrics_elements.extend(metrics_elements_from_batch)

                gc.collect()
            print('Number metrics elements: {}'.format(len(metrics_elements)))
            aggregated_metrics = aggregate_metrics_elements(metrics_elements)
            print('Aggregated P@1: {}'.format(aggregated_metrics['P_AT_1']))
            print('Aggregated P@10: {}'.format(aggregated_metrics['P_AT_10']))
            if ds_name == 'Google_RE':
                google_re_metrices.append(aggregated_metrics)
            elif ds_name == 'TREx':
                trex_metrices.append(aggregated_metrics)

            layer_data[ds_name][args.relation_args.relation].append(
                aggregated_metrics)
            write_to_execution_log(ds_name + ': ' + args.relation_args.relation +
                                   '\t' + str(aggregated_metrics['P_AT_1']), append_newlines=True,
                                   path=args.execution_log)
    # Write results to logfile
    if len(google_re_metrices) > 0:
        write_to_execution_log(
            '\n\nGoogle_RE: ' + mean_precisions(google_re_metrices), append_newlines=True, path=args.execution_log)
        layer_data['Google_RE']['means'].append(
            mean_precisions(google_re_metrices))
    if len(trex_metrices) > 0:
        write_to_execution_log(
            'Trex: ' + mean_precisions(trex_metrices), append_newlines=True, path=args.execution_log)
        layer_data['TREx']['means'].append(mean_precisions(trex_metrices))
    write_to_execution_log(
        220 * '-', append_newlines=True, path=args.execution_log)

    if args.use_wandb_logging:
        wandb.init(name=args.wandb_run_name, project=args.wandb_project_name,
                   settings=wandb.Settings(start_method='thread'))
        wandb_log_metrics(layer_data, layer)
    return layer_data


def probe4cbqa(args: Namespace, probing_model: BaseDecoder, tokenizer: AutoTokenizer, layer: int):
    write_to_execution_log(
        100 * '+' + '\t CBQA probing layer: {} \t'.format(layer) + 100 * '+', append_newlines=True,
        path=args.execution_log)
    print('##################################      Layer: {}      #########################################'.format(
        layer))
    print('$$$$$$$$$$$$$$$$$$$$$$$   CBQA Probing model of type with PAQ: {}      $$$$$$$$$$$$$$$$$$$$$$$$'.format(
        args.model_type))

    probing_model.total_num_probing_steps = 0
    probe_dataset = cbqa_Dataset(probing_model.tokenizer, probing_model.hparams,
                                 file_path=args.test_file,
                                 block_size=probing_model.tokenizer.model_max_length)
    sampler = RandomSampler(probe_dataset)
    dataloader = DataLoader(probe_dataset, sampler=sampler, batch_size=args.probing_batch_size,
                            collate_fn=probing_model.mlm_collate)
    if args.log_prefix is not None:
        contain_file = '/{}_contain_answer.txt'.format(args.log_prefix)
        correct_file = '/{}_correct_answer.txt'.format(args.log_prefix)
        wrong_file = '/{}_wrong_answer.txt'.format(args.log_prefix)
    else:
        contain_file = '/contain_answer.txt'
        correct_file = '/correct_answer.txt'
        wrong_file = '/wrong_answer.txt'
    metrics_elements_num = 0
    contain_list = []
    correct_list = []
    wrong_list = []
    contain_num = 0
    correct_num = 0
    wrong_num = 0
    f1_sum = 0
    for batch in tqdm(dataloader):
        # if probing_model.total_num_probing_steps % 5 == 0:
        #     show = True
        # else:
        #     show = False
        # probing_model.total_num_probing_steps += 1
        inputs, t5_labels = qa_tokens(batch, probing_model.tokenizer, probing_model.hparams)
        inputs = inputs.to(args.device)
        # t5_labels = t5_labels.to(args.device)
        question_batch = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        answer_batch = tokenizer.batch_decode(t5_labels, skip_special_tokens=True)
        outputs = probing_model.model.generate(input_ids=inputs, num_beams=5, max_length=50, num_return_sequences=5,
                                               early_stopping=True)
        generated_txt_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for k in range(1, int(len(generated_txt_batch) / 5 + 1)):
            metrics_elements_num += 1
            gold_answers = answer_batch[k - 1].split(',')
            f1_max = []
            em_max = []
            gen_lst = []
            for i, gen in enumerate(generated_txt_batch[(k - 1) * 5:k * 5]):
                em_score = max((compute_exact_match(gen, answer)) for answer in gold_answers)
                f1_score = max((compute_f1(gen, answer)) for answer in gold_answers)
                # print(em_score)
                # print(f1_score)
                gen_lst.append(gen)
                f1_max.append(f1_score)
                em_max.append(em_score)
            try:
                ind = em_max.index(1)
                f1_score = f1_max[ind]
                em_score = 1
            except:
                em_score = 0
                f1_score = max(f1_max)
                ind = f1_max.index(f1_score)
            f1_sum += f1_score
            if f1_score > 0 and em_score != 1:
                # if (answer_batch[i].strip() in gen.strip()) or (gen.strip() in answer_batch[i].strip()):
                contain_num += 1
                contain_dict = {'Question': question_batch[k - 1], 'Ground-truth answer': answer_batch[k - 1],
                                'generated answer': gen_lst, 'f1 score': f1_score}
                # print('contain_dict', contain_dict)
                contain_list.append(contain_dict)
                with open(args.output_dir + contain_file, 'a', encoding='utf-8') as f:
                    f.write(str(contain_dict))
                    f.write('\n')

            if em_score == 1:
                # if answer_batch[i].strip() == gen.strip():
                correct_num += 1
                correct_dict = {'Question': question_batch[k - 1], 'Ground-truth answer': answer_batch[k - 1],
                                'generated answer': gen_lst, 'exact match rank': ind + 1, 'f1_score': f1_score}
                print('correct_dict', correct_dict)
                correct_list.append(correct_dict)
                with open(args.output_dir + correct_file, 'a', encoding='utf-8') as f:
                    f.write(str(correct_dict))
                    f.write('\n')
            elif em_score == 0:
                wrong_num += 1
                wrong_dict = {'Question': question_batch[k - 1], 'Ground-truth answer': answer_batch[k - 1],
                              'generated answer': gen_lst, 'exact match': 'False', 'f1_score': f1_score}
                # print('wrong_dict', wrong_dict)
                with open(args.output_dir + wrong_file, 'a', encoding='utf-8') as f:
                    f.write(str(wrong_dict))
                    f.write('\n')
                if len(wrong_list) <= 100:
                    wrong_list.append(wrong_dict)

    print('num_QA-pairs: {}'.format(metrics_elements_num))
    print('Total completely correct: {}'.format(len(correct_list)))
    print('Total num of contain the answer: {}'.format(len(contain_list)))
    print('probing log path', args.execution_log)
    write_to_execution_log('=' * 45, append_newlines=True, path=args.execution_log)
    write_to_execution_log('100 examples that are not completely correct', append_newlines=True,
                           path=args.execution_log)
    write_to_execution_log('↓' * 45, append_newlines=True, path=args.execution_log)
    for i in wrong_list:
        write_to_execution_log(
            '-' * 100 + '\nQuestion: {}\nAnswer: {}\n{:<10}: {}'.format(i['Question'], i['Ground-truth answer'],
                                                                        'generated answer', i['generated answer']),
            append_newlines=True, path=args.execution_log)
        # for id, token in enumerate(i['answer_token']):
        # write_to_execution_log('{:<10}, {}'.format(token, i['top10_span'][id]), append_newlines=True, path=args.execution_log)

    write_to_execution_log('Total QA-pairs: {}'.format(metrics_elements_num),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('Total completely correct: {}'.format(len(correct_list)),
                           append_newlines=True, path=args.execution_log)
    # write_to_execution_log('Total num of contain the answer: {}'.format(len(contain_list)),
    #                        append_newlines=True, path=args.execution_log)
    write_to_execution_log('Total wrong answer: {}'.format(wrong_num),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('Exact match rate: {:.2%}'.format(len(correct_list) / metrics_elements_num),
                           append_newlines=True, path=args.execution_log)
    write_to_execution_log('Average f1 score: {:.2%}'.format(f1_sum / metrics_elements_num),
                           append_newlines=True, path=args.execution_log)
    if args.log_prefix is not None:
        log_prefix_path = os.path.join(args.output_dir, 'results_collect')
        write_to_execution_log('Checkpoint: {}'.format(args.log_prefix),
                               append_newlines=True, path=log_prefix_path)
        write_to_execution_log('Total QA-pairs: {}'.format(metrics_elements_num),
                               append_newlines=True, path=log_prefix_path)
        write_to_execution_log('Total completely correct: {}'.format(len(correct_list)),
                               append_newlines=True, path=log_prefix_path)
        # write_to_execution_log('Total num of contain the answer: {}'.format(len(contain_list)),
        #                        append_newlines=True, path=args.execution_log)
        write_to_execution_log('Total wrong answer: {}'.format(wrong_num),
                               append_newlines=True, path=log_prefix_path)
        write_to_execution_log('Exact match rate: {:.2%}'.format(len(correct_list) / metrics_elements_num),
                               append_newlines=True, path=log_prefix_path)
        write_to_execution_log('Average f1 score: {:.2%}'.format(f1_sum / metrics_elements_num),
                               append_newlines=True, path=log_prefix_path)


def save_probing_data(args, data):
    with open('{}/{}_data.json'.format(args.output_dir, 'model'), 'w') as outfile:
        json.dump(data, outfile)


def cloze_collate4paq(examples, tokenizer: AutoTokenizer):
    """
        This is a function that makes sure all entries in the batch are padded
        to the correct length.
    """
    inputs_id = [x['inputs_id'] for x in examples]
    labels = [x['labels'] for x in examples]
    t5_labels = [x['t5_labels'] for x in examples]
    uuids = [x['uuid'] for x in examples]
    length = [x['len'] for x in examples]
    answer = [x['answer'] for x in examples]
    inputs_tokens = [x['inputs_tokens'] for x in examples]
    question = [x['question'] for x in examples]
    t5_labels = torch.tensor(t5_labels)
    padded_sentences_masked = pad_sequence(
        inputs_id, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_sentences_input = pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask = padded_sentences_masked.clone()
    attention_mask[attention_mask != tokenizer.pad_token_id] = 1
    attention_mask[attention_mask == tokenizer.pad_token_id] = 0

    examples_batch = {
        "inputs_id": padded_sentences_masked,
        "labels": padded_sentences_input,
        "t5_labels": t5_labels,
        "attention_mask": attention_mask,
        "len": length,
        "uuid": uuids,
        "answer": answer,
        "inputs_tokens": inputs_tokens,
        "question": question
    }

    return examples_batch


def wandb_log_metrics(layer_data, layer):
    # log datasets with just one relation
    wandb.log(
        {'layer': layer, 'ConceptNet P@1': layer_data['ConceptNet']['test'][0]['P_AT_1']})
    wandb.log(
        {'layer': layer, 'Squad P@1': layer_data['Squad']['test'][0]['P_AT_1']})

    # log dataset with messy means
    # Google_RE
    p1 = handle_mean_values_string(layer_data['Google_RE']['means'][0])[0]
    wandb.log({'layer': layer, 'Google_RE P@1': p1})

    # TREx
    p1 = handle_mean_values_string(layer_data['TREx']['means'][0])[0]
    wandb.log({'layer': layer, 'TREx P@1': p1})
