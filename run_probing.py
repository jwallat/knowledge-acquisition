from knowledge_probing.models.lightning.og_t5_model import OGT5Model
from knowledge_probing.models.lightning.t5_decoder import T5Decoder
from knowledge_probing.models.lightning.bert_decoder import BertDecoder
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from knowledge_probing.training.training import training
from knowledge_probing.probing.probing import probing
from knowledge_probing.config.config_helper import handle_config
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
import sys
import os


def main(args):


    seed_everything(args.seed)
    print('PID: ', os.getpid())

    args = handle_config(args)
    print('Config handled')
    if 't5' in args.model_type:
        print('Using a T5 model')
        if args.use_original_model:
            print('Using the original model with the last layer')
            decoder = OGT5Model(hparams=args)
        else:
            decoder = T5Decoder(hparams=args)
    else:
        print('Using a BERT model')
        decoder = BertDecoder(hparams=args)

    print('Got a model')

    if args.do_training:
        training(args, decoder)

    if args.do_probing:
        probing(args, decoder)

    return args.run_name


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
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
    parser.add_argument('--select_specific_gpu_id', type=str)
    

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
