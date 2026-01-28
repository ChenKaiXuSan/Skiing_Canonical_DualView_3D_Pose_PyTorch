#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/cross_validation.py
Project: /workspace/skeleton/project
Created Date: Friday March 22nd 2024
Author: Kaixu Chen
-----
Comment:
This module defines a cross-validation strategy based on GroupKFold,
按照 person_id 进行分组划分，确保同一人的数据不会同时出现在训练集和验证集中。
根据label文件夹中的标注文件，配对对应的视频文件，构建样本列表。
划分结果会被保存到指定的index_mapping目录下的index.json文件中，以便后续加载使用。
不实用person22，23的数据。

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import GroupKFold
from project.map_config import (
    label_mapping_Dict,
    environment_mapping_Dict,
    ENV_KEY_TO_FOLDER,
    CAM_NAMES,
    VideoSample,
)


class DefineCrossValidation(object):
    """
    New behavior:
      - build samples from:
          videos/{person}/{env_folder}/{cam}.mp4
          label/person_{person}_{day|night}_{high|low}_h265.json
      - GroupKFold split by person_id
      - no sampler
    """

    def __init__(self, config) -> None:
        self.video_path: Path = Path(
            config.data.video_path
        )  # e.g. /workspace/data/videos
        self.annotation_path: Path = Path(
            config.data.annotation_path
        )  # e.g. /workspace/data/label

        self.fold_count: int = int(config.data.fold)
        self.index_mapping: Path = Path(
            config.data.index_mapping
        )  # folder to save/load index.json

    # --------- helpers ---------
    @staticmethod
    def _parse_label_filename(p: Path) -> Tuple[str, str, str]:
        """
        person_01_night_high_h265.json -> ("01", "night", "high")
        """
        stem = p.stem  # person_01_night_high_h265
        parts = stem.split("_")
        # 最少应满足：person, 01, night, high, h265
        if len(parts) < 5 or parts[0] != "person":
            raise ValueError(f"Unexpected label filename: {p.name}")
        person_id = parts[1]
        daynight = parts[2]
        highlow = parts[3]
        return person_id, daynight, highlow

    def _collect_one_sample(self, label_path: Path) -> VideoSample | None:
        person_id, daynight, highlow = self._parse_label_filename(label_path)

        # label中的环境 -> 视频文件夹中文名
        if (daynight, highlow) not in ENV_KEY_TO_FOLDER:
            # 不认识的命名就跳过
            return None

        env_folder = ENV_KEY_TO_FOLDER[(daynight, highlow)]
        env_key = f"{daynight}_{highlow}"

        # video root: videos/01/夜多/
        vid_dir = self.video_path / person_id / env_folder
        if not vid_dir.exists():
            # 你的数据可能是 videos/01/... 但 label 是 person_01...
            # 如果视频路径是 01 而 person_id 是 "01" 这没问题；
            # 如果是 "1" vs "01" 才会找不到，需要你统一命名
            return None

        videos: Dict[str, Path] = {}
        for cam in CAM_NAMES:
            mp4 = vid_dir / f"{cam}.mp4"
            if mp4.exists():
                videos[cam] = mp4

        # 至少要有一个视频才算 sample
        if len(videos) == 0:
            return None

        return VideoSample(
            person_id=person_id,
            env_folder=env_folder,
            env_key=env_key,
            label_path=label_path,
            videos=videos,
        )

    def build_samples(self) -> List[VideoSample]:
        """
        Scan label directory, pair videos, return samples list.
        """
        label_files = sorted(self.annotation_path.glob("person_*_*.json"))
        samples: List[VideoSample] = []
        for lp in label_files:
            try:
                s = self._collect_one_sample(lp)
            except Exception:
                s = None
            if s is not None:
                samples.append(s)
        return samples

    def split_by_person(
        self, samples: List[VideoSample]
    ) -> Dict[int, Dict[str, List[VideoSample]]]:
        """
        GroupKFold by person_id
        """
        if self.fold_count <= 1:
            # fold=1 时，给一个“全量train + 空val”或你也可以改成 train_test_split
            return {0: {"train": samples, "val": []}}

        groups = [s.person_id for s in samples]
        indices = list(range(len(samples)))

        gkf = GroupKFold(n_splits=self.fold_count)
        fold_dict: Dict[int, Dict[str, List[VideoSample]]] = {}

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(indices, groups=groups)):
            train_samples = [samples[i] for i in tr_idx]
            val_samples = [samples[i] for i in va_idx]
            fold_dict[fold] = {"train": train_samples, "val": val_samples}

        return fold_dict

    # --------- main entry ---------
    def prepare(self):
        samples = self.build_samples()
        if len(samples) == 0:
            raise RuntimeError(
                f"No valid samples found. Please check:\n"
                f"  video_path={self.video_path}\n"
                f"  annotation_path={self.annotation_path}\n"
                f"  label filename format: person_XX_(day|night)_(high|low)_h265.json\n"
                f"  video structure: videos/XX/(夜多|夜少|昼多|昼少)/(front|right|left).mp4"
            )

        fold_samples = self.split_by_person(samples)
        return fold_samples

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Save/load fold index json
        """
        target_dir = self.index_mapping
        target_dir.mkdir(parents=True, exist_ok=True)
        index_file = target_dir / "index.json"

        if not index_file.exists():
            fold_samples = self.prepare()

            # serialize
            serial: Dict[str, Any] = {}
            for fold, d in fold_samples.items():
                serial[str(fold)] = {
                    "train": [
                        {
                            "person_id": s.person_id,
                            "env_folder": s.env_folder,
                            "env_key": s.env_key,
                            "label_path": str(s.label_path),
                            "videos": {k: str(v) for k, v in s.videos.items()},
                        }
                        for s in d["train"]
                    ],
                    "val": [
                        {
                            "person_id": s.person_id,
                            "env_folder": s.env_folder,
                            "env_key": s.env_key,
                            "label_path": str(s.label_path),
                            "videos": {k: str(v) for k, v in s.videos.items()},
                        }
                        for s in d["val"]
                    ],
                }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(serial, f, ensure_ascii=False, indent=2)

            return fold_samples

        # load
        with open(index_file, "r", encoding="utf-8") as f:
            serial = json.load(f)

        fold_samples: Dict[int, Dict[str, List[VideoSample]]] = {}
        for kfold, d in serial.items():
            fold = int(kfold)
            fold_samples[fold] = {"train": [], "val": []}
            for split in ["train", "val"]:
                for item in d[split]:
                    fold_samples[fold][split].append(
                        VideoSample(
                            person_id=item["person_id"],
                            env_folder=item["env_folder"],
                            env_key=item["env_key"],
                            label_path=Path(item["label_path"]),
                            videos={kk: Path(vv) for kk, vv in item["videos"].items()},
                        )
                    )

        return fold_samples
