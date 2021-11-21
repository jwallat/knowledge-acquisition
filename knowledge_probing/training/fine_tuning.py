from knowledge_probing.models.lightning.base_decoder import BaseDecoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from knowledge_probing.file_utils import write_to_execution_log, load_cpt_torch
import sys
import os


def fine_tuning(args, decoder: BaseDecoder):
    args.decoder_save_dir = '{}/ft/'.format(
        args.decoder_save_dir)
    os.makedirs(args.decoder_save_dir, exist_ok=True)
    print('args.save_top_k', args.save_top_k)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.decoder_save_dir,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True
    )  # ,prefix=''

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=args.training_early_stop_delta,
        patience=args.training_early_stop_patience,
        verbose=True,
        mode='min'
    )
    decoder.hparams.update(vars(args))
    print('ewc2', decoder.hparams.ewc)
    if args.use_wandb_logging:
        print(
            'If you are having issues with wandb, make sure to give the correct python executable to --python_executable')
        sys.executable = args.python_executable
        os.makedirs("{}/wandb_logs".format(args.output_dir))
        logger = WandbLogger(project=decoder.hparams.wandb_project_name,
                             name=decoder.hparams.wandb_run_name, save_dir="{}/wandb_logs".format(args.output_dir))
    else:
        logger = TensorBoardLogger("{}/tb_logs".format(args.output_dir))

    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=True, callbacks=[checkpoint_callback, early_stop_callback], logger=logger,
        gpus=args.gpus)  # gpu_selection

    write_to_execution_log('Run: {} \nArgs: {}\n'.format(
        args.run_identifier, args), path=args.execution_log)  # knowledge_probing.file_utils.write_to_execution_log

    if not args.use_original_model:
        decoder.set_to_train()

        trainer.fit(decoder)
        #decoder = decoder.load_best_model_checkpoint(hparams=args)
        if args.save_top_k == 1:
            decoder = load_cpt_torch(decoder=decoder, decoder_save_dir=args.decoder_save_dir)
            decoder.hparams.update(vars(args))
    else:
        try:
            trainer.fit(decoder)
        except:
            print('Training failed for some reason')

    return decoder
    # trainer.test(decoder)
