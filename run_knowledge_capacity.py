from knowledge_probing.models.lightning.bert_probe_train import BertProbeTrain
from knowledge_probing.datasets.cloze_dataset_train import TrainClozeDataset
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
from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from knowledge_probing.file_utils import write_to_execution_log
import sys
import torch


def main(args):

    seed_everything(args.seed)
    print('PID: ', os.getpid())

    args = handle_config(args)
    print('Config handled')
    # if 't5' in args.model_type:
    #     print('Using a T5 model')
    #     if args.use_original_model:
    #         print('Using the original model with the last layer')
    #         decoder = OGT5Model(hparams=args)
    #     else:
    #         decoder = T5Decoder(hparams=args)
    # else:
    #     print('Using a BERT model')
    #     decoder = BertDecoder(hparams=args)

    model = BertProbeTrain(hparams=args)

    # cloze_training_data = TrainClozeDataset(
    #     probing_model=decoder, tokenizer=decoder.tokenizer, args=args)

    #########################################################################################################################

    if args.do_training:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.decoder_save_dir,
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=args.training_early_stop_delta,
            patience=args.training_early_stop_patience,
            verbose=True,
            mode='min'
        )

        if args.use_wandb_logging:
            print('If you are having issues with wandb, make sure to give the correct python executable to --python_executable')
            sys.executable = args.python_executable
            logger = WandbLogger(project=args.wandb_project_name,
                                 name=args.wandb_run_name)
        else:
            logger = TensorBoardLogger("{}/tb_logs".format(args.output_dir))

        # if 'select_specific_gpu_id' in args:
        #     gpu_selection = args.select_specific_gpu_id
        # else:
        #     gpu_selection = args.gpus

        trainer = Trainer.from_argparse_args(
            args, checkpoint_callback=checkpoint_callback, callbacks=[early_stop_callback], logger=logger, gpus=args.gpus)

        write_to_execution_log('Run: {} \nArgs: {}\n'.format(
            args.run_identifier, args), path=args.execution_log)

        model.set_to_train()

        try:
            trainer.fit(model)
        except:
            print('Skip training')
        # model = model.load_best_model_checkpoint(hparams=args)

        trainer.test(model)

    ###########################################################################################################################################

    if args.do_probing:
        probing(args, model)

    return args.run_name


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = BaseDecoder.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Capacity specific
    parser.add_argument('--capacity_text_mode', required=True,
                        choices=['templates', 'evidences'])
    parser.add_argument('--capacity_masking_mode', required=True,
                        choices=['object', 'random'])

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
