import argparse
import datetime
import json
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler

from pl.module import DetrModule
from pl.datamodule import VocDataset


def parse_args():
    parser = argparse.ArgumentParser(prog='CIFAR10 classifier trainer')
    parser.add_argument(
        '--device', type=str,
        default='cpu',
        help='Execution device',
    )
    parser.add_argument(
        '--epochs', type=int,
        default=100,
        help='Epochs to train',
    )
    parser.add_argument(
        '--logdir', type=str,
        default='logs',
        help='Path to train logs',
    )
    parser.add_argument(
        '--val_interval', type=int,
        default=50,
        help='Validation check interval',
    )
    parser.add_argument(
        '--train_batch', type=int,
        default=4,
        help='Train batch size',
    )
    parser.add_argument(
        '--val_batch', type=int,
        default=16,
        help='Validation batch size',
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Flag to download the dataset',
    )
    return parser.parse_args()



def get_session_tstamp():
    session_timestamp = str(datetime.datetime.now())
    session_timestamp = session_timestamp.replace(' ', '').replace(':', '-').replace('.', '-')
    return session_timestamp


def run_training(args):
    seed_everything(42, workers=True)
    logger = TensorBoardLogger(save_dir=args.logdir, name=get_session_tstamp())
    profiler = SimpleProfiler(filename='profiler_report')
    trainer = pl.Trainer(
        accelerator=args.device,
        strategy='auto',
        devices='auto',
        num_nodes=1,
        precision='32-true',
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=None,
                filename='epoch-{epoch:04d}-loss-{loss/val:.6f}-acc-{accuracy/val:.6f}',
                monitor='loss/total',
                verbose=True,
                save_last=True,
                save_top_k=2,
                mode='min',
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor()
        ],
        fast_dev_run=False,
        max_epochs=args.epochs,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        limit_train_batches=None,
        limit_val_batches=None,
        limit_test_batches=None,
        limit_predict_batches=None,
        overfit_batches=0.0,
        val_check_interval=args.val_interval,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=None,
        log_every_n_steps=50,
        enable_checkpointing=None,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm='norm',
        deterministic=None,
        benchmark=None,
        inference_mode=True,
        use_distributed_sampler=True,
        profiler=profiler,
        detect_anomaly=False,
        barebones=False,
        plugins=None,
        sync_batchnorm=False,
        reload_dataloaders_every_n_epochs=0,
        default_root_dir=None,
    )
    model = DetrModule()
    datamodule = VocDataset(
        train_batch=args.train_batch,
        val_batch=args.val_batch,
        download=args.download,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    run_training(parse_args())
