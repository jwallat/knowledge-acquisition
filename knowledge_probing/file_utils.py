from transformers import AutoTokenizer
from dotmap import DotMap
import yaml
import json
import os, sys
import torch


def stringify_dotmap(args):
    stringy = 'Args:\n'
    for k, v in args.items():
        if type(v) != DotMap:
            # print(k)
            if (str(k) == 'getdoc') or (str(k) == 'shape'):
                continue
            else:
                stringy = stringy + '\t' + k + ':' + '\t' + str(v) + '\n'
        elif type(v) == DotMap:
            stringy = stringy + '\t' + k + ':\n'
            for key, value in v.items():
                # print(key)
                if (str(k) == 'getdoc') or (str(k) == 'shape'):
                    continue
                else:
                    stringy = stringy + '\t\t' + key + \
                              ':' + '\t' + str(value) + '\n'
    return stringy


# Write args to args.execution_log


def write_to_execution_log(text, path, append_newlines=False):
    # print("Saving args into file %s", args.execution_log)
    with open(path, "a", encoding='utf-8') as handle:
        handle.write(text)
        handle.write('\n\n') if append_newlines else None


def load_vocab(path):
    assert os.path.exists(path)

    with open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab


def get_vocab(model_type):
    vocab = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        vocab_path = tokenizer.save_vocabulary(save_directory='.')

        # Load vocabulary
        vocab = load_vocab(vocab_path)
    except:
        print('Could not read vocab, will use tokenizer.decode() in datasets.cloze_dataset')
    return vocab


def load_config(path):
    assert os.path.exists(path)

    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def load_file(filename):
    assert os.path.exists(filename)
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def load_model_config(filename):
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        data = f.read()

    return data


def write_metrics(run_identifier, dataset, relation, dir, metrics):
    metrics_file = os.path.join(
        dir, dataset + "_" + relation + "_" + run_identifier
    )
    # print("Saving metrics into file %s", metrics_file)
    with open(metrics_file, "w") as handle:
        handle.write('Results: {}'.format(metrics))


def find_checkpoint_in_dir(dir):
    for file in os.listdir(dir):
        if file.endswith(".ckpt"):
            checkpoint = os.path.join(dir, file)
            return checkpoint


def load_cpt_torch(decoder, decoder_save_dir):
    if 'ckpt' not in decoder_save_dir:
        checkpoint_file = find_checkpoint_in_dir(decoder_save_dir)
    else:
        checkpoint_file = decoder_save_dir
    print('Loading best checkpoint: {}'.format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(checkpoint['state_dict'])
    return decoder


def update_args4ft(args):
    args.wandb_run_name = args.wandb_run_name + '_FT'
    args.adafactor_relative_step = False
    args.adafactor_warmup = False
    args.adafactor_scale_params = False
    args.finetuning = True
    args.old_dataset4ewc = args.train_file
    args.train_file = args.FT_train_file
    args.valid_file = args.FT_valid_file
    args.test_file = args.FT_test_file
    args.did_multitask = args.multitask
    args.multitask = False
    args.warmup_steps = 0
    args.total_steps = 0
    args.batch_size = args.ft_batch_size
    args.accumulate_grad_batches = args.ft_accumulate_grad_batches
    args.max_epochs = args.ft_max_epochs
    args.ft_ewc = args.ewc
    args.ft_ewc_lambda = args.ewc_lambda
    if args.save_all:
        args.save_top_k = -1
    return args
