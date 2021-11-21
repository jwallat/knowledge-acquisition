from knowledge_probing.datasets.cloze_data_utils import topk
from knowledge_probing.file_utils import write_to_execution_log
import sys


def calculate_metrics(batch, index, prediction_scores, precision_at_k, tokenizer=None, total_top_k_words=100):
    metrics_element = {}

    # print(batch)

    # sample information (masked sentences, obj_label, uuid)
    metrics_element['sample'] = {
        'masked_sentences': tokenizer.convert_ids_to_tokens(batch['masked_sentences'][index]),
        'obj_label': batch['obj_label'][index],
        'uuid': batch['uuid'][index]
    }
    # print(metrics_element)
    # Initialize values
    metrics_element['P_AT_K'] = 0.0
    metrics_element['P_AT_10'] = 0.0
    metrics_element['P_AT_1'] = 0.0
    metrics_element['MRR'] = 0.0
    metrics_element['PERPLEXITY'] = None
    # get topk predictions
    topk_tokens, topk_values = topk(prediction_scores, batch['mask_index'][index], k=total_top_k_words,
                                    tokenizer=tokenizer, return_likelihoods=True)
    # print('topk_tokens', topk_tokens)
    # print('topk_values', topk_values)
    # Might need to be done for T5 as it adds a \u2581 (_) to all tokens
    for i, token in enumerate(topk_tokens):
        topk_tokens[i] = str(token).replace('\u2581', '')

    # print(topk_tokens)
    metrics_element['top_k_tokens'] = topk_tokens[:precision_at_k]
    metrics_element['top_k_values'] = topk_values[:precision_at_k]

    try:
        # get rank of our expected word
        rank = topk_tokens.index(batch['obj_label'][index])
        # print(rank)
        rank += 1
        metrics_element['rank'] = rank

        # MRR

        metrics_element['MRR'] = (1. / rank)

        # precision @ 1, 10, k
        if rank <= precision_at_k:
            metrics_element['P_AT_K'] = 1.
        if rank <= 10:
            metrics_element['P_AT_10'] = 1.
        if rank == 1:
            metrics_element['P_AT_1'] = 1.

        # perplexity
        # perplexity = 0

        # metrics_element['PERPLEXITY'] = perplexity

    except:
        metrics_element['rank'] = 'not found in top {} words'.format(
            total_top_k_words)

    # judgement
    if 'judgments' in batch:
        num_yes = 0
        num_no = 0
        for judgment_ele in batch['judgments']:
            for judgment in judgment_ele:
                if judgment['judgment'] == 'yes':
                    num_yes += 1
                else:
                    num_no += 1

        if num_yes > num_no:
            metrics_element['sample']['judgment'] = 'positive'
        elif num_no <= num_yes:
            metrics_element['sample']['judgment'] = 'negative'

    # print(metrics_element)

    # print('Masked sentence: ', metrics_element['sample']['masked_sentences'])
    # print('Answer: ', metrics_element['sample']['obj_label'])
    # print('Top predictions: ', metrics_element['top_k_tokens'][:10])
    # print('Rank of GT: ', metrics_element['rank'])

    return metrics_element


def calculate_metrics4paq(batch, index, prediction_scores, precision_at_k, tokenizer=None, log_path=None):
    metrics_element = {}
    answer = batch['answer'][index]
    answer_label = tokenizer(answer)['input_ids']
    answer_label.pop()
    answer_token = tokenizer.convert_ids_to_tokens(answer_label)
    correct = True
    grad_sum = 0
    P_at_1 = 0
    P_at_10 = 0
    P_at_k = 0
    top10_span = []
    for i in range(batch['len'][index]):
        mask_index = 1 + i
        topk_tokens, topk_values = topk(prediction_scores, mask_index, k=precision_at_k,
                                        tokenizer=tokenizer, return_likelihoods=True)
        top10_span.append(topk_tokens[:10])
        if topk_tokens[0] != answer_token[i]:
            correct = False
        try:
            rank = topk_tokens.index(answer_token[i])

            mrr = (1. / (rank + 1))
            if rank <= (precision_at_k - 1):
                P_at_k += 1
            if rank <= 9:
                P_at_10 += 1
            if rank == 0:
                P_at_1 += 1
        except:
            mrr = 0

        grad_sum += mrr
    # print some wrong prediction in excution-log
    if correct:
        write_to_execution_log(
            'Completely correct : ' + 'Q: ' + batch['question'][index] + ' A:' + batch['answer'][index] +
            ', num of tokens: ' + str(batch['len'][index]), append_newlines=True, path=log_path)
    else:
        metrics_element['answer_token'] = answer_token
        metrics_element['top10_span'] = top10_span
        metrics_element['answer'] = batch['answer'][index]
        metrics_element['question'] = batch['question'][index]
    metrics_element['correct'] = correct
    metrics_element['grad'] = grad_sum
    metrics_element['num_tokens'] = batch['len'][index]
    metrics_element['P_AT_1'] = P_at_1
    metrics_element['P_AT_10'] = P_at_10
    metrics_element['P_AT_K'] = P_at_k
    return metrics_element


def calculate_metrics4cbqa(inputs, t5_labels, index, prediction_scores, precision_at_k, tokenizer, log_path):
    metrics_element = {}
    answer = t5_labels
    # answer_label = tokenizer(answer, skip_special_tokens=True)['input_ids']
    # answer_label.pop()
    answer_token = tokenizer.convert_ids_to_tokens(answer, skip_special_tokens=True)
    correct = True
    grad_sum = 0
    P_at_1 = 0
    P_at_10 = 0
    P_at_k = 0
    top10_span = []
    for i in range(len(answer_token)):
        mask_index = i
        topk_tokens, topk_values = topk(prediction_scores, mask_index, k=precision_at_k,
                                        tokenizer=tokenizer, return_likelihoods=True)
        top10_span.append(topk_tokens[:10])
        if topk_tokens[0] != answer_token[i]:
            correct = False
        try:
            rank = topk_tokens.index(answer_token[i])

            mrr = (1. / (rank + 1))
            if rank <= (precision_at_k - 1):
                P_at_k += 1
            if rank <= 9:
                P_at_10 += 1
            if rank == 0:
                P_at_1 += 1
        except:
            mrr = 0

        grad_sum += mrr
    # print some wrong prediction in excution-log
    qq = tokenizer.decode(inputs, skip_special_tokens=True)
    aa = tokenizer.decode(answer, skip_special_tokens=True)
    length = len(answer_token)
    print('metric log_path', log_path)
    if correct:
        write_to_execution_log(
            'Completely correct : ' + 'Q: ' + qq + ' A:' + aa +
            ', num of tokens: ' + str(length), append_newlines=True, path=log_path)
    else:
        metrics_element['answer_token'] = answer_token
        metrics_element['top10_span'] = top10_span
        metrics_element['answer'] = aa
        metrics_element['question'] = qq
    metrics_element['correct'] = correct
    metrics_element['grad'] = grad_sum
    metrics_element['num_tokens'] = length
    metrics_element['P_AT_1'] = P_at_1
    metrics_element['P_AT_10'] = P_at_10
    metrics_element['P_AT_K'] = P_at_k
    return metrics_element


def cbqa_metrics(batch_prediction_scores, inputs, t5_labels, precision_at_k, tokenizer, log_path):
    metrics_elements_from_batch = []
    for i, prediction_scores in enumerate(batch_prediction_scores):
        prediction_scores = prediction_scores[None, :, :]
        t5_label = t5_labels[i].clone()
        input = inputs[i].clone()
        metrics_element = calculate_metrics4cbqa(
            input, t5_label, i, prediction_scores, precision_at_k=precision_at_k
            , tokenizer=tokenizer, log_path=log_path)
        metrics_elements_from_batch.append(metrics_element)

    return metrics_elements_from_batch


def aggregate_metrics_elements(metrics_elements):
    # Calc mean p1,p10, pk, MRR
    # print(len([x['MRR'] for x in metrics_elements]))
    # Mean reciprocal rank
    MRR = sum([x['MRR'] for x in metrics_elements]) / \
          len([x['MRR'] for x in metrics_elements])
    MRR_negative = 0.0
    MRR_positive = 0.0

    # Precision at (default 10)
    Precision10 = sum([x['P_AT_10'] for x in metrics_elements]) / \
                  len([x['P_AT_10'] for x in metrics_elements])
    Precision1 = sum([x['P_AT_1'] for x in metrics_elements]) / \
                 len([x['P_AT_1'] for x in metrics_elements])
    PrecisionK = sum([x['P_AT_K'] for x in metrics_elements]) / \
                 len([x['P_AT_K'] for x in metrics_elements])
    Precision_negative = 0.0
    Precision_positive = 0.0

    total_positive = 0
    total_negative = 0

    # the judgment of the annotators recording whether they are
    # evidence in the sentence that indicates a relation between two entities.
    for element in metrics_elements:
        if 'judgment' in element['sample']:
            if element['sample']['judgment'] == 'negative':
                total_negative += 1
                MRR_negative += element['MRR']
                Precision_negative += element['P_AT_K']
            else:
                total_positive += 1
                MRR_positive += element['MRR']
                Precision_positive += element['P_AT_K']

    if total_negative > 0:
        Precision_negative = Precision_negative / total_negative
        MRR_negative = MRR_negative / total_negative

    if total_positive > 0:
        Precision_positive = Precision_positive / total_positive
        MRR_positive = MRR_positive / total_positive

    aggregated = {
        'MRR': MRR,
        'MRR_negative': MRR_negative,
        'MRR_positive': MRR_positive,
        'P_AT_1': Precision1,
        'P_AT_10': Precision10,
        'P_AT_K': PrecisionK,
        'P_AT_K_positive': Precision_positive,
        'P_AT_K_negative': Precision_negative,
        'individual_predictions': metrics_elements
    }

    return aggregated


# Calc means for google-re and trex
def mean_precisions(data):
    p1s = []
    p10s = []
    pks = []

    for relation_metric in data:
        p1s.append(relation_metric['P_AT_1'])
        p10s.append(relation_metric['P_AT_10'])
        pks.append(relation_metric['P_AT_K'])

    mean_p1 = sum(p1s) / len(p1s)
    mean_p10 = sum(p10s) / len(p10s)
    mean_pk = sum(pks) / len(pks)

    return 'Mean P@1,10,k: {}, {}, {}'.format(mean_p1, mean_p10, mean_pk)


"""
https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
"""


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers
