from pytorch_lightning import Trainer
from argparse import ArgumentParser
from dotmap import DotMap
from transformers import BertConfig, AutoTokenizer, BertModel
from knowledge_probing.models.lightning.decoder import Decoder
from knowledge_probing.models.lightning.hugging_decoder import HuggingDecoder
from knowledge_probing.training.training import training
from knowledge_probing.models.models_helper import get_model
from knowledge_probing.probing.probing import probing
from knowledge_probing.config.config_helper import handle_config
import sys
import os


def main(args):
    args.run_name = 'probe_layer_{}'.format(args.probing_layer)
    args.training_decoder = args.decoder
    args.probing_model = args.decoder

    model_selection = ['qa'] if args.use_only_qa_model else ['default', 'qa']
    # Start with default
    args.bert_type = model_selection[0]
    args.do_training = True
    args.do_probing = True

    # Compute the missing args ##########################################################################
    args = handle_config(args)
    print(args.output_dir)

    for bert_type in model_selection:
        print('************************     Using model type: {}'.format(bert_type))
        args.bert_type = bert_type

        # Update the logdir for the default/qa models
        args.decoder_save_dir = '{}/decoder/{}'.format(
            args.output_dir, args.bert_type)
        os.makedirs(args.decoder_save_dir, exist_ok=True)

        # Set up bert-config and model ###########################################################################
        decoder = get_model(args)

        print(args.skip_training)
        # Training: #########################################################################################
        if not args.skip_training:
            training(args, decoder)

        # Probing ###########################################################################################
        probing(args, decoder)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--distribution_strategy',
                        required=False, choices=['dp', 'ddp', 'ddp2', None], help="Distribution strategy according to https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#data-parallel-dp")

    parser.add_argument('--num_nodes', required='--distribution_strategy' in sys.argv, type=int,
                        help='Please set the number of gpus to per node and specify the number of nodes when doing multi gpu training')

    parser.add_argument('--training_early_stop_delta', default=0.01, type=int,
                        help='The minimum validation-loss-delta between #patience iterations that has to happen for the computation not to stop')

    parser.add_argument('--training_early_stop_patience', default=15, type=int,
                        help='The patience for the models validation loss to improve by [training_early_stop_delta] for the computation not to stop')

    # General BERT
    parser.add_argument('--bert_model_type', default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-base-cased'],)

    parser.add_argument('--decoder', required=True,
                        choices=['Huggingface_pretrained_decoder', 'Decoder'],)

    # Training
    parser.add_argument('--skip_training',
                        action='store_true', default=False)
    parser.add_argument('--training_dataset', default='wikitext-2',
                        choices=['wikitext-2', 'wikitext-103'], required='--do_training' in sys.argv)

    parser.add_argument('--training_epochs', default=100,
                        required='--do_training' in sys.argv, type=int)

    # Probing
    parser.add_argument('--probing_layer', default=12,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int)

    # Other
    parser.add_argument('--config_path', default='knowledge_probing/config/config.yaml',
                        help='Path to your config file in case you do not want to use the config in /config/config.yaml')

    parser.add_argument('--output_base_dir', default='/data/outputs/',
                        help='Path to the output dir that will contain the logs and trained models')

    parser.add_argument('--use_only_qa_model',
                        default=False, action='store_true')

    args = parser.parse_args()

    main(args)
