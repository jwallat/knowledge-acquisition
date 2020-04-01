from knowledge_probing.file_utils import load_config
from datetime import datetime
from dotmap import DotMap
import torch
import os


def handle_config(args):
    # Load the config:
    config = load_config(args.config_path)
    args = DotMap(vars(args))

    args = add_config_to_args(config, args)

    # The idea is to load the config and set all necessary implications (e.g. lowercase True/False from bert_model_type)
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.run_identifier = build_run_identifier(args)

    # Data dir
    assert os.path.exists(args.data_dir)

    # Output dir
    args.output_dir = '{}{}'.format(args.output_base_dir, args.run_identifier)

    # Model dirs
    args.decoder_save_dir = '{}/decoder/{}'.format(
        args.output_dir, args.bert_type)

    # Execution log
    args.execution_log = args.output_dir + '/execution_log.txt'

    os.makedirs(args.output_dir, exist_ok=True)
    # os.makedirs(args.pre_trained_model_dir, exist_ok=True)
    os.makedirs(args.qa_model_dir, exist_ok=True)
    os.makedirs(args.decoder_save_dir, exist_ok=True)

    # Lowercase
    args.lowercase = False
    if 'uncased' in args.bert_model_type:
        args.lowercase = True

    # Choose the correct training dataset
    if args.do_training:
        if args.training_dataset == 'wikitext-2':
            args.train_data_file = args.wiki2_train_data_file
            args.eval_data_file = args.wiki2_eval_data_file
            args.test_data_file = args.wiki2_test_data_file
        elif args.training_dataset == 'wikitext-103':
            args.train_data_file = args.wiki103_train_data_file
            args.eval_data_file = args.wiki103_eval_data_file
            args.test_data_file = args.wiki103_test_data_file
        else:
            print('If you can read this, you have probabily added a new training dataset. Please make sure to add the path to the dataset files in the config.yaml and the config_helper.py')

    # Fix the type of learning_rate
    args.learning_rate = float(args.learning_rate)

    return args


def add_config_to_args(config, args):
    for key in config:
        args[key] = config[key]

    return args


def build_run_identifier(args):

    time = datetime.now()
    timestamp = '{}_{}_{}__{}-{}'.format(time.day,
                                         time.month, time.year, time.hour + 1, time.minute)

    model_type_postfix = args.bert_model_type.split('-')[-1]

    run_identifier = '{}_{}_{}_trained-{}_{}_{}'.format(
        args.run_name, args.bert_type, model_type_postfix, args.do_training, args.probing_model, timestamp)

    print('Run identifier: ', run_identifier)
    return run_identifier
