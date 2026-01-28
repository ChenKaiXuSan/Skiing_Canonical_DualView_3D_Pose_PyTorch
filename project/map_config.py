#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/map_config.py
Project: /workspace/MultiView_DriverAction_PyTorch/project
Created Date: Sunday January 25th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday January 25th 2026 9:48:24 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

label_mapping_Dict: Dict = {
    0: "left",
    1: "right",
    2: "down",
    3: "up",
    4: "right_up",
    5: "right_down",
    6: "left_down",
    7: "left_up",
    8: "front",
}

environment_mapping_Dict: Dict = {
    0: "夜多い",  # night_high
    1: "夜少ない",  # night_low
    2: "昼多い",  # day_high
    3: "昼少ない",  # day_low
}


# 反向映射：label文件名里的 (day/night, high/low) -> 文件夹名
ENV_KEY_TO_FOLDER = {
    ("night", "high"): "夜多い",
    ("night", "low"): "夜少ない",
    ("day", "high"): "昼多い",
    ("day", "low"): "昼少ない",
}

# 你期望的相机视频文件（按需增减）
CAM_NAMES = ["front", "right", "left"]


@dataclass
class VideoSample:
    person_id: str  # "01"
    env_folder: str  # "夜多"
    env_key: str  # "night_high"
    label_path: Path  # .../label/person_01_night_high_h265.json
    videos: Dict[str, Path]  # {"front": ..., "right": ..., "left": ...}
