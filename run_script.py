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


def main(args):

    # Compute the missing args ##########################################################################
    args = handle_config(args)

    # Set up bert-config and model ###########################################################################
    decoder = get_model(args)

    # Training: #########################################################################################
    if args.do_training:
        # Load data
        training(args, decoder)

    # Probing ###########################################################################################
    if args.do_probing:
        # Load data
        probing(args, decoder)


if __name__ == '__main__':
    parser = ArgumentParser()

    # TODO: Other PL informations?
    parser.add_argument('--gpus', default=1, type=int)

    # General BERT
    parser.add_argument('--bert_type', default='default',
                        choices=['default', 'qa'],)

    parser.add_argument('--bert_model_type', default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-base-cased'],)

    # Training
    parser.add_argument('--do_training', default=False, action='store_true')

    parser.add_argument('--training_dataset', default='wikitext-2',
                        choices=['wikitext-2', 'wikitext-103'],)

    parser.add_argument('--training_epochs', default=None,
                        required='--do_training' in sys.argv, type=int)

    parser.add_argument('--training_decoder', required='--do_training' in sys.argv,
                        choices=['Huggingface_pretrained_decoder', 'Decoder'],)

    parser.add_argument('--training_early_stop_delta', default=0.01, type=int,
                        help='The minimum validation-loss-delta between #patience iterations that has to happen for the computation not to stop')

    parser.add_argument('--training_early_stop_patience', default=15, type=int,
                        help='The patience for the models validation loss to improve by [training_early_stop_delta] for the computation not to stop')

    # Probing
    parser.add_argument('--do_probing', default=False, action='store_true')

    parser.add_argument('--probing_model', required='--do_probing' in sys.argv,
                        choices=['Huggingface_pretrained_decoder', 'Decoder', 'BertForMaskedLM'],)

    parser.add_argument('--probing_layer', default=12,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int)

    parser.add_argument('--probe_all_layers',
                        default=False, action='store_true')

    # Other
    parser.add_argument('--save_logs', default=False, action='store_true')

    parser.add_argument('--save_model', default=False, action='store_true')

    parser.add_argument('--config_path', default='knowledge_probing/config/config.yaml',
                        help='Path to your config file in case you do not want to use the config in /config/config.yaml')

    parser.add_argument('--run_name', default='',
                        help='Name of the run that will be used when building the run_identifier')

    parser.add_argument('--output_base_dir', default='/data/outputs/',
                        help='Path to the output dir that will contain the logs and trained models')

    args = parser.parse_args()

    main(args)
