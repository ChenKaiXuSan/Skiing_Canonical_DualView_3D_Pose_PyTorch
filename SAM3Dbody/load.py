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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


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
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
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


def _has_image_files(folder: Path) -> bool:
    """Check whether a folder directly contains image files."""
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def collect_capture_dirs(action_dir: Path, camera_layers=None) -> List[Path]:
    """Collect camera folders under one action directory.

    Supported layout:
    - person/action/frames/camera
    
    Args:
        action_dir: Action directory containing frames/
        camera_layers: Optional list of layer indices (0-4) to filter.
                      E.g., [0, 1] means only L0 and L1.
                      None means all layers.
    """
    frames_dir = action_dir / "frames"
    if not frames_dir.is_dir():
        return []

    capture_dirs = sorted(
        [x for x in frames_dir.iterdir() if x.is_dir() and _has_image_files(x)]
    )
    
    # Apply camera layer filter if specified
    if camera_layers is not None and len(camera_layers) > 0:
        filtered_dirs = []
        for capture_dir in capture_dirs:
            # Expected format: capture_L{layer}_A{angle}
            name = capture_dir.name
            if name.startswith("capture_L"):
                try:
                    layer_str = name.split("_")[1]  # "L0", "L1", etc.
                    layer_num = int(layer_str[1:])  # Extract 0, 1, 2, 3, 4
                    if layer_num in camera_layers:
                        filtered_dirs.append(capture_dir)
                except (IndexError, ValueError):
                    logger.warning("[Skip] Unexpected capture dir name: %s", name)
            else:
                # If not matching expected pattern, keep it (backward compatibility)
                filtered_dirs.append(capture_dir)
        return filtered_dirs
    
    return capture_dirs


def collect_action_dirs(
    source_root: Path,
    camera_layers=None,
    person_filter=None,
    action_filter=None,
) -> List[Path]:
    """Collect action directories for camera-based inference.

    Supported layout:
    - person/action/frames/camera

    An action directory is identified if it has a ``frames`` folder containing
    camera subdirectories with image files.
    
    Args:
        source_root: Root directory containing person folders
        camera_layers: Optional list of layer indices to filter captures
        person_filter: Optional person name to filter (e.g., "male", "female")
        action_filter: Optional action name to filter (exact match)
    """
    action_dirs_set = set()

    # Strict scan: only one level below each person directory.
    for person_dir in sorted([x for x in source_root.iterdir() if x.is_dir()]):
        # Apply person filter
        if person_filter is not None and person_dir.name != person_filter:
            continue
            
        for candidate in sorted([x for x in person_dir.iterdir() if x.is_dir()]):
            # Apply action filter
            if action_filter is not None and candidate.name != action_filter:
                continue
                
            if collect_capture_dirs(candidate, camera_layers):
                action_dirs_set.add(candidate)

    # Keep sorted deterministic ordering for sharding.
    action_dirs = sorted(action_dirs_set)
    return action_dirs