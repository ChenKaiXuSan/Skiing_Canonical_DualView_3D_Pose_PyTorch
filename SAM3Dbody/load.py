#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/load.py
Project: /workspace/code/SAM3Dbody
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:50:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_data(input_video_path: Dict[str, Path]) -> Dict[str, List[np.ndarray]]:
    """
    動画ファイルからすべてのフレームを読み込み、RGB形式のリストとして返します。

    引数:
        input_video_path: 動画ファイルのパスの辞書 (キーは視点名、値は Path オブジェクト)

    戻り値:
        List[np.ndarray]: 全フレームのリスト。各フレームは RGB 形式の numpy 配列。
    """
    view_frames_list: Dict[str, List[np.ndarray]] = {}

    for view, video_path in input_video_path.items():

        # 2. 動画のキャプチャ開始
        logger.info(f"動画ファイルからフレームを抽出中: {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(
                f"エラー: 動画を開くことができませんでした -> {video_path}"
            )
            return []

        # 3. 全フレームをループで読み込み
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # OpenCVはBGR形式なのでRGBに変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                view_frames_list.setdefault(view, []).append(frame_rgb)
        finally:
            cap.release()

        logger.info(
            f"動画の読み込み完了。合計 {len(view_frames_list[view])} フレームを抽出しました。"
        )
    return view_frames_list


def load_capture_frames(capture_dir: Path) -> List[np.ndarray]:
    """Load one capture folder into an RGB frame list."""
    frame_files = sorted(
        [
            p
            for p in capture_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
    )

    frames: List[np.ndarray] = []
    for frame_file in frame_files:
        frame_bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning("[Skip] Failed to read image: %s", frame_file)
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    return frames


def collect_action_dirs(source_root: Path) -> List[Path]:
    """Collect all action directories that contain a ``frames`` subdirectory.

    This keeps worker partitioning at action granularity regardless of the
    dataset hierarchy (e.g., gender/action or person/action).
    """
    action_dirs = sorted(
        [
            p.parent
            for p in source_root.rglob("frames")
            if p.is_dir() and p.parent.is_dir()
        ]
    )
    return action_dirs