#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/data_loader.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Saturday January 24th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 24th 2026 10:51:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
)

from project.dataloader.whole_video_dataset import whole_video_dataset
from project.dataloader.utils import Div255


class UnityDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._experiment = opt.experiment

        self.mapping_transform = Compose(
            [
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

    @staticmethod
    def _merge_bt_pose(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,T,J,C) or (T,J,C) -> (T,1,J,C)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected {name} shape (T,J,C), got {tuple(x.shape)}")
        return x.unsqueeze(1)

    @staticmethod
    def _merge_bt_video(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,C,T,H,W) -> (T,C,H,W)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"Expected {name} shape (1,C,T,H,W), got {tuple(x.shape)}")
        return x[0].permute(1, 0, 2, 3)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep variable T per sample by flattening B and T into pseudo-batch."""
        if not batch:
            return {}

        frames_cam1: List[torch.Tensor] = []
        frames_cam2: List[torch.Tensor] = []
        gt2d_cam1: List[torch.Tensor] = []
        gt2d_cam2: List[torch.Tensor] = []
        sam2d_cam1: List[torch.Tensor] = []
        sam2d_cam2: List[torch.Tensor] = []
        sam3d_cam1: List[torch.Tensor] = []
        sam3d_cam2: List[torch.Tensor] = []
        gt3d: List[torch.Tensor] = []
        frame_indices: List[torch.Tensor] = []
        meta_rows: List[Dict[str, Any]] = []

        for sample in batch:
            frames_cam1.append(
                self._merge_bt_video(sample["frames"]["cam1"], "frames/cam1")
            )
            frames_cam2.append(
                self._merge_bt_video(sample["frames"]["cam2"], "frames/cam2")
            )

            gt2d_cam1.append(
                self._merge_bt_pose(sample["kpt2d_gt"]["cam1"], "kpt2d_gt/cam1")
            )
            gt2d_cam2.append(
                self._merge_bt_pose(sample["kpt2d_gt"]["cam2"], "kpt2d_gt/cam2")
            )
            sam2d_cam1.append(
                self._merge_bt_pose(sample["kpt2d_sam"]["cam1"], "kpt2d_sam/cam1")
            )
            sam2d_cam2.append(
                self._merge_bt_pose(sample["kpt2d_sam"]["cam2"], "kpt2d_sam/cam2")
            )
            sam3d_cam1.append(
                self._merge_bt_pose(sample["kpt3d_sam"]["cam1"], "kpt3d_sam/cam1")
            )
            sam3d_cam2.append(
                self._merge_bt_pose(sample["kpt3d_sam"]["cam2"], "kpt3d_sam/cam2")
            )
            gt3d.append(self._merge_bt_pose(sample["kpt3d_gt"], "kpt3d_gt"))

            idx = sample.get("frame_indices")
            if isinstance(idx, torch.Tensor):
                frame_indices.append(idx.view(-1))

            sample_meta = sample.get("meta", {})
            num_frames = int(sample["kpt3d_gt"].shape[0])
            for t in range(num_frames):
                row = (
                    dict(sample_meta)
                    if isinstance(sample_meta, dict)
                    else {"meta": sample_meta}
                )
                row["time_index_in_sample"] = t
                meta_rows.append(row)

        return {
            "frames": {
                "cam1": torch.cat(frames_cam1, dim=0),
                "cam2": torch.cat(frames_cam2, dim=0),
            },
            "kpt2d_gt": {
                "cam1": torch.cat(gt2d_cam1, dim=0),
                "cam2": torch.cat(gt2d_cam2, dim=0),
            },
            "kpt3d_gt": torch.cat(gt3d, dim=0),
            "kpt2d_sam": {
                "cam1": torch.cat(sam2d_cam1, dim=0),
                "cam2": torch.cat(sam2d_cam2, dim=0),
            },
            "kpt3d_sam": {
                "cam1": torch.cat(sam3d_cam1, dim=0),
                "cam2": torch.cat(sam3d_cam2, dim=0),
            },
            "frame_indices": torch.cat(frame_indices, dim=0)
            if frame_indices
            else torch.empty(0, dtype=torch.long),
            "meta": meta_rows,
        }

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # train dataset
        self.train_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            transform=self.mapping_transform,
        )

        # val dataset
        self.val_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
        )

        # test dataset
        self.test_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
        )

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        return test_data_loader
