from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler
import torch
from knowledge_probing.file_utils import write_to_execution_log, stringify_dotmap


def training(args, decoder):
    checkpoint_callback = ModelCheckpoint(
        filepath=args.decoder_model_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.02,
        patience=6,
        verbose=True,
        mode='min'
    )

    # logger = TensorBoardLogger("tb_logs", args.run_identifier)

    # profiler = AdvancedProfiler(output_filename='performance_logs')

    if args.use_tpu:
        print('Training is using TPU')
        # log_save_interval=100
        trainer = Trainer(row_log_interval=10, num_tpu_cores=8, max_epochs=args.train.training_epochs,
                          precision=32, checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback)
    elif args.device == torch.device('cuda'):
        print('Training is using GPU')
        trainer = Trainer(row_log_interval=50, log_save_interval=500, gpus=1, max_epochs=args.train.training_epochs,
                          checkpoint_callback=checkpoint_callback, val_check_interval=args.train.val_check_interval, early_stop_callback=early_stop_callback)

        # early stop
        # trainer = Trainer(gpus=1, fast_dev_run=False, checkpoint_callback=checkpoint_callback, val_check_interval=args.train.val_check_interval, early_stop_callback=early_stop_callback)
    else:
        print('Training is using CPU')
        trainer = Trainer(max_epochs=args.train.training_epochs, fast_dev_run=False, checkpoint_callback=checkpoint_callback,
                          val_check_interval=args.train.val_check_interval, early_stop_callback=early_stop_callback)

    write_to_execution_log('Run: {} \nArgs: {}\n'.format(
        args.run_identifier, stringify_dotmap(args)))
    trainer.fit(decoder)

    trainer.test()


def save_training_logs(logs_dir, path):
    # TODO: Implement save training logs
    raise NotImplementedError


def save_model(model, path):
    # TODO: Implement save model
    raise NotImplementedError
