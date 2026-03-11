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

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from project.dataloader.data_loader import UnityDataModule

#####################################
# select different experiment trainer
#####################################
# baseline
from project.trainer.baseline.train_3dcnn import Res3DCNNTrainer
from project.trainer.early.train_early_fusion import EarlyFusion3DCNNTrainer
from project.trainer.late.train_late_fusion import LateFusion3DCNNTrainer
from project.trainer.train_fusion_SSM import FusionSSMTrainer

from project.map_config import UnityDataConfig

logger = logging.getLogger(__name__)


def load_fold_dataset_idx_from_index_mapping(config: DictConfig):
    """Load precomputed fold mapping from index json file.

    This removes CV split preparation from training entry.
    """
    index_mapping_cfg = Path(str(config.data.index_mapping))
    index_file_name = str(config.data.index_mapping_file)

    # Backward/forward compatible:
    # 1) data.index_mapping points to directory + data.index_mapping_file
    # 2) data.index_mapping points directly to a json file
    if index_mapping_cfg.suffix == ".json":
        index_file = index_mapping_cfg
    else:
        index_file = index_mapping_cfg / index_file_name

    if not index_file.exists():
        raise FileNotFoundError(
            f"Index mapping file not found: {index_file}. "
            f"Please generate it first (e.g. cross_validation/generate_cv_index.py)."
        )

    with open(index_file, "r", encoding="utf-8") as f:
        serial = json.load(f)

    # Skip metadata entry if exists.
    serial.pop("_metadata", None)

    fold_dataset_idx: Dict[int, Dict[str, List[UnityDataConfig]]] = {}
    for kfold, d in serial.items():
        if not isinstance(d, dict):
            raise ValueError(f"Fold {kfold} must be a dict, got {type(d)}")

        fold = int(kfold)
        fold_dataset_idx[fold] = {"train": [], "val": []}

        # Accept both val/valid naming from different generators.
        split_aliases = {"train": ["train"], "val": ["val", "valid"]}

        for split, aliases in split_aliases.items():
            src_list = None
            for alias in aliases:
                if alias in d:
                    src_list = d[alias]
                    break
            if src_list is None:
                raise KeyError(
                    f"Fold {kfold} missing split '{split}' (aliases: {aliases})"
                )
            if not isinstance(src_list, list):
                raise TypeError(
                    f"Fold {kfold} split '{split}' must be a list, got {type(src_list)}"
                )

            for item in src_list:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Index item in fold {kfold}/{split} must be dict, got {type(item)}"
                    )

                # camera-pair index format: build UnityDataConfig directly.
                if "cam1_frames_dir" in item and "cam2_frames_dir" in item:
                    required_fields = [
                        "person_id",
                        "action_id",
                        "cam1_id",
                        "cam2_id",
                        "cam1_path",
                        "cam2_path",
                        "label_path",
                        "cam1_frames_dir",
                        "cam2_frames_dir",
                        "cam1_kpt2d_dir",
                        "cam2_kpt2d_dir",
                        "kpt3d_dir",
                        "sam3d_cam1_dir",
                        "sam3d_cam2_dir",
                        "sequence_meta_path",
                        "joint_names_path",
                    ]
                    missing = [k for k in required_fields if k not in item]
                    if missing:
                        raise KeyError(
                            f"Fold {kfold}/{split} missing required UnityDataConfig keys: {missing}"
                        )

                    fold_dataset_idx[fold][split].append(
                        UnityDataConfig(
                            person_id=str(item["person_id"]),
                            action_id=str(item["action_id"]),
                            cam1_id=str(item["cam1_id"]),
                            cam2_id=str(item["cam2_id"]),
                            cam1_path=str(item["cam1_path"]),
                            cam2_path=str(item["cam2_path"]),
                            label_path=str(item["label_path"]),
                            cam1_frames_dir=str(item["cam1_frames_dir"]),
                            cam2_frames_dir=str(item["cam2_frames_dir"]),
                            cam1_kpt2d_dir=str(item["cam1_kpt2d_dir"]),
                            cam2_kpt2d_dir=str(item["cam2_kpt2d_dir"]),
                            kpt3d_dir=str(item["kpt3d_dir"]),
                            sam3d_cam1_dir=str(item["sam3d_cam1_dir"]),
                            sam3d_cam2_dir=str(item["sam3d_cam2_dir"]),
                            sequence_meta_path=str(item["sequence_meta_path"]),
                            joint_names_path=str(item["joint_names_path"]),
                        )
                    )
                    continue

                raise ValueError(
                    "Unsupported index item format. "
                    f"Expected camera-pair fields, got keys: {list(item.keys())}"
                )

    return fold_dataset_idx


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
    monitor_metric = "val/video_acc"
    monitor_mode = "max"
    ckpt_filename = "{epoch}-{val/loss:.2f}-{val/video_acc:.4f}"

    if hparams.train.view == "multi":
        if hparams.model.backbone == "3dcnn":
            if hparams.model.fuse_method in ["add", "mul", "concat", "avg"]:
                classification_module = EarlyFusion3DCNNTrainer(hparams)
            elif hparams.model.fuse_method == "late":
                classification_module = LateFusion3DCNNTrainer(hparams)
            elif hparams.model.fuse_method in ["ssm", "mamba", "mamba_ssm"]:
                classification_module = FusionSSMTrainer(hparams)
                monitor_metric = "val/loss"
                monitor_mode = "min"
                ckpt_filename = "{epoch}-{val/loss:.4f}"
            else:
                raise ValueError("the experiment fuse method is not supported.")
        else:
            raise ValueError("the experiment backbone is not supported.")
    elif hparams.train.view == "single":
        classification_module = Res3DCNNTrainer(hparams)
    else:
        raise ValueError("the experiment view is not supported.")

    # * prepare data module
    data_module = UnityDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=10)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", "fold_" + str(fold)),
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        mode=monitor_mode,
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
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
            # DeviceStatsMonitor(),  # monitor the device stats.
        ],
        limit_train_batches=5,
        limit_val_batches=5,
        limit_test_batches=5,
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
    config_name="train.yaml",
)
def init_params(config):
    # Load precomputed fold mapping only; do not prepare CV splits here.
    fold_dataset_idx: Dict[int, Dict[str, List[UnityDataConfig]]] = (
        load_fold_dataset_idx_from_index_mapping(config)
    )

    requested_fold = int(config.train.fold)
    available_folds = sorted(fold_dataset_idx.keys())

    # train.fold >= 0: run only the specified fold (recommended for multi-node jobs)
    # train.fold < 0: run all folds sequentially (backward compatible mode)
    if requested_fold >= 0:
        if requested_fold not in fold_dataset_idx:
            raise KeyError(
                f"Requested fold {requested_fold} is not in index mapping. "
                f"Available folds: {available_folds}"
            )
        target_folds = [requested_fold]
    else:
        target_folds = available_folds

    logger.info("#" * 50)
    logger.info(
        "Start training folds: %s (requested train.fold=%s)",
        target_folds,
        requested_fold,
    )
    logger.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold in target_folds:
        dataset_value = fold_dataset_idx[fold]
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train folds: %s", target_folds)
    logger.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
