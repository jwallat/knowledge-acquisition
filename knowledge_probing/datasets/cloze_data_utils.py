from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import torch
import functools


def predicted_sentence(scores, tokenizer):
    predicted_ids = torch.reshape(torch.argmax(scores, dim=2), (-1,))
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    return predicted_tokens


def topk(prediction_scores, token_index, k=10, tokenizer=None, return_likelihoods=False):
    # Get the ids for the masked index:
    prediction_for_masked = prediction_scores[0][token_index]
    top_values, tops_indices = torch.topk(input=prediction_for_masked, k=k)
    top_words = tokenizer.convert_ids_to_tokens(tops_indices)
    # print(top_words)

    if return_likelihoods:
        return top_words, top_values.tolist()

    return top_words


# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)
def lowercase_samples(samples, mask, use_negated_probes=False):
    new_samples = []
    for sample in samples:
        if "obj_label" in sample:
            sample["obj_label"] = sample["obj_label"].lower()
        if "sub_label" in sample:
            sample["sub_label"] = sample["sub_label"].lower()
        if 'masked_sentences' in sample:
            lower_masked_sentences = []
            for sentence in sample["masked_sentences"]:
                sentence = sentence.lower()
                sentence = sentence.replace(mask.lower(), mask)
                lower_masked_sentences.append(sentence)
            sample["masked_sentences"] = lower_masked_sentences

            # if "negated" in sample and use_negated_probes:
            #     for sentence in sample["negated"]:
            #         sentence = sentence.lower()
            #         sentence = sentence.replace("[MASK]".lower(), mask)
            #         lower_masked_sentences.append(sentence)
            #     sample["negated"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)
def filter_samples(samples, tokenizer: AutoTokenizer, vocab, template):
    msg = ""
    new_samples = []
    samples_exluded = 0

    for sample in samples:
        excluded = False
        if "obj_label" in sample and "sub_label" in sample:
            obj_label_ids = tokenizer.encode(
                sample["obj_label"], add_special_tokens=False)
            if obj_label_ids:
                # reconstructed_word = " ".join(
                #     [vocab[x] for x in obj_label_ids]
                # ).strip()
                reconstructed_word = tokenizer.decode(obj_label_ids)

                if len(obj_label_ids) > 1:
                    reconstructed_word = None
                # TODO: Find good solution for comparing two models
            else:
                reconstructed_word = None

            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
                text = " ".join(masked_sentences)
                if len(text.split()) > tokenizer.model_max_length:
                    msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True

            # # MAKE SURE THAT obj_label IS IN VOCABULARIES
            # (Removed as we do not have multiple different models that require different vocabularies)

            if excluded:
                pass
            elif obj_label_ids is None:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            elif not reconstructed_word or reconstructed_word != sample["obj_label"]:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            elif "judgments" in sample:
                # only for Google-RE
                num_no = 0
                num_yes = 0
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no > num_yes:
                    # SKIP NEGATIVE EVIDENCE
                    pass
                else:
                    new_samples.append(sample)
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]
