#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import logging
import hydra
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
)

from project.dataloader.data_loader import DriverDataModule

#####################################
# select different experiment trainer
#####################################

# baseline
from project.trainer.baseline.train_3dcnn import Res3DCNNTrainer

# attention based
from project.trainer.mid.train_pose_attn import PoseAttnTrainer
from project.trainer.mid.train_se_attn import SEAttnTrainer
from project.trainer.early.train_early_fusion import EarlyFusion3DCNNTrainer
from project.trainer.late.train_late_fusion import LateFusion3DCNNTrainer

from project.cross_validation import DefineCrossValidation

logger = logging.getLogger(__name__)


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    # TODO: add more experiment trainer here.
    if hparams.train.view == "multi":
        if hparams.model.backbone == "3dcnn":

            if hparams.model.fuse_method in ["add", "mul", "concat", "avg"]:
                classification_module = EarlyFusion3DCNNTrainer(hparams)
            elif hparams.model.fuse_method == "late":
                classification_module = LateFusion3DCNNTrainer(hparams)
            else:
                raise ValueError("the experiment fuse method is not supported.")
        else:
            raise ValueError("the experiment backbone is not supported.")
    elif hparams.train.view == "single":
        classification_module = Res3DCNNTrainer(hparams)
    else:
        raise ValueError("the experiment view is not supported.")

    # * prepare data module
    data_module = DriverDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    cvs_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv_logs"),
        name="fold_" + str(fold) + "_csv",  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=10)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(
            hparams.log_path, "checkpoints", "fold_" + str(fold)
        ),
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=True,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=5,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            # progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
            DeviceStatsMonitor(),  # monitor the device stats.
        ],
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #######################
    # prepare dataset index
    #######################

    fold_dataset_idx = DefineCrossValidation(config)()

    logger.info("#" * 50)
    logger.info("Start train all fold")
    logger.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold, dataset_value in fold_dataset_idx.items():
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train all fold")
    logger.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
