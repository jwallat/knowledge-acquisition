from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from knowledge_probing.file_utils import write_to_execution_log, load_cpt_torch
import sys
import os


def training(args, decoder: BaseDecoder):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.decoder_save_dir,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor='training_loss',
        mode='min'
    )  # , prefix=''

    early_stop_callback = EarlyStopping(
        monitor='training_loss',
        min_delta=args.training_early_stop_delta,
        patience=args.training_early_stop_patience,
        verbose=True,
        mode='min'
    )

    if args.use_wandb_logging:
        print(
            'If you are having issues with wandb, make sure to give the correct python executable to --python_executable')
        sys.executable = args.python_executable
        filepath = "{}/train/wandb_logs".format(args.output_dir)
        if not os.path.exists(filepath):
            os.makedirs("{}/train/wandb_logs".format(args.output_dir))
        logger = WandbLogger(project=args.wandb_project_name,
                             name=args.wandb_run_name, save_dir=filepath)
    else:
        logger = TensorBoardLogger("{}/tb_logs".format(args.output_dir))

    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=True, callbacks=[checkpoint_callback, early_stop_callback], logger=logger, gpus=args.gpus, accelerator='dp')  # gpu_selection

    write_to_execution_log('Run: {} \nArgs: {}\n'.format(
        args.run_identifier, args), path=args.execution_log)  # knowledge_probing.file_utils.write_to_execution_log

    if not args.use_original_model:
        decoder.set_to_train()

        trainer.fit(decoder)
        # todo: If here a logical error?   line 52 to line 56
        #decoder = decoder.load_best_model_checkpoint(hparams=args)
        decoder = load_cpt_torch(decoder=decoder, decoder_save_dir=args.decoder_save_dir)
        decoder.hparams.update(vars(args))
    else:
        try:
            trainer.fit(decoder)
            # decoder = decoder.load_best_model_checkpoint(hparams=args)
        except:
            print('Training failed for some reason')
    if not decoder.hparams.multitask:
        trainer.test(decoder)
    return decoder
