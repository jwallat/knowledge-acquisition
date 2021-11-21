from knowledge_probing.models.lightning.t5_decoder import T5Decoder
from knowledge_probing.models.lightning.bert_decoder import BertDecoder
from knowledge_probing.models.lightning.t5_multitask4ewc_decoder import T5MultitaskDecoder
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from knowledge_probing.training.fine_tuning import fine_tuning
from knowledge_probing.probing.probing import probing
from knowledge_probing.config.config_helper import handle_config
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from knowledge_probing.file_utils import load_cpt_torch, find_checkpoint_in_dir, update_args4ft
import functools
import sys
import os


def load_cpt(decoder, checkpoint_path):
    decoder_save_dir = checkpoint_path
    checkpoint_file = find_checkpoint_in_dir(decoder_save_dir)
    print('Loading best checkpoint: {}'.format(checkpoint_file))
    return decoder.load_from_checkpoint(checkpoint_file)


def fine_tune(args, decoder):
    decoder.hparams.mask_way = 'normal'
    decoder.collate = functools.partial(decoder.mlm_collate, tokenizer=decoder.tokenizer)
    decoder.hparams.extend4probing = False
    decoder = fine_tuning(args, decoder)
    decoder.hparams.extend4probing = True
    return decoder


def main(args):
    seed_everything(args.seed)
    print('PID: ', os.getpid())
    args = handle_config(args)
    print('Config handled')
    if 't5' in args.model_type:
        print('Using a T5 model')
        #if args.ewc and args.multitask:
        #    print('---------- Multitask training with EWC ----------')
        #    decoder = T5MultitaskDecoder(hparams=args)
        #else:
        decoder = T5Decoder(hparams=args)
    else:
        print('Using a BERT model')
        decoder = BertDecoder(hparams=args)

    args = update_args4ft(args=args)
    decoder = fine_tune(args, decoder)
    decoder.hparams.extend4probing = True
    decoder.hparams.probing_batch_size = 32
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
    parser.add_argument('--load_model_ckpt_path', default=None, type=str)
    parser.add_argument('--old_dataset4ewc',
                        default="/home/tzhang/tmp/nlp/knowledge-probing-private/data/probing_data/QAsets/paq_train_10k.tsv")
    # Training
    parser.add_argument('--do_training', default=False, action='store_true')
    parser.add_argument('--training_early_stop_delta', default=0.01, type=float,
                        help='The minimum validation-loss-delta between #patience iterations that has to happen for the computation not to stop')
    parser.add_argument('--training_early_stop_patience', default=15, type=int,
                        help='The patience for the models validation loss to improve by [training_early_stop_delta] for the computation not to stop')
    parser.add_argument('--freeze_encoder', default=False, type=bool)
    parser.add_argument('--freeze_decoder', default=False, type=bool)
    parser.add_argument('--use_raw_model', default=False, action='store_true')
    parser.add_argument('--multitask', default=False, type=bool)
    # fine-tuning
    parser.add_argument('--save_all', default=False, action='store_true')
    parser.add_argument('--log_prefix', default=None, type=str)
    parser.add_argument('--finetuning', default=False, action='store_true')
    parser.add_argument('--ft_batch_size', default=32, type=int)
    parser.add_argument('--ft_accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--ft_max_epochs', default=100, type=int)
    parser.add_argument('--ewc', default=False, action='store_true')
    parser.add_argument('--ft_ewc', default=False, action='store_true')
    parser.add_argument('--ewc_lambda', default=1.0, type=float)
    parser.add_argument('--ft_ewc_lambda', default=1.0, type=float)
    # Probing
    parser.add_argument('--extend4probing', default=False, action='store_true')
    parser.add_argument('--do_probing', default=False, action='store_true')
    parser.add_argument('--use_original_model',
                        default=False, action='store_true')
    # parser.add_argument('--probing_layer', default=12,
    #                     choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int)
    parser.add_argument('--probing_layer', default=12, type=int)
    # mask method
    parser.add_argument('--mask_way', default='normal', help='mask strategy', type=str)
    parser.add_argument('--ssm_all', default=False, action='store_true')
    parser.add_argument('--pmi_path',
                        default="/home/tzhang/tmp/nlp/knowledge-probing-private/data/training_data/PAQ/pmi_tokens.jsonl")
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
    parser.add_argument('--python_executable', required='--use_wandb_logging' in sys.argv, type=str,
                        default='/usr/bin/python3',
                        help='Some cluster environments might require to set the sys.executable for wandb to work')

    args = parser.parse_args()

    main(args)
