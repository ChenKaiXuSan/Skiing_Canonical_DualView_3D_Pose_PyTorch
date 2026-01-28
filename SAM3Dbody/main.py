#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/main_multi_gpu_process.py
Project: /workspace/code/SAM3Dbody
Created Date: Monday January 26th 2026
Author: Kaixu Chen
-----
Comment:
根据多GPU并行处理SAM-3D-Body推理任务。

Have a good code time :)
-----
Last Modified: Monday January 26th 2026 5:12:10 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

# 假设这些是从你的其他模块导入的
from .infer import process_frame_list
from .load import load_data

# --- 常量定义 ---
REQUIRED_VIEWS = {"front", "left", "right"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------
def process_single_person(
    person_dir: Path,
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
):
    """处理单个人员的所有环境和视角"""
    person_id = person_dir.name
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    # --- 1. Person専用のログ設定 ---
    log_dir = out_root / "person_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    person_log_file = log_dir / f"{person_id}.log"

    # 新しいハンドラを作成
    handler = logging.FileHandler(person_log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(person_id)  # このPerson専用のロガーを取得
    logger.addHandler(handler)
    logger.propagate = False  # 親（Root）ロガーにログを流さない（混ざるのを防ぐ）

    logger.info(f"==== Starting Process for Person: {person_id} ====")

    env_dirs = sorted([x for x in person_dir.iterdir() if x.is_dir()])
    if not env_dirs:
        logger.warning(f"跳过：{person_dir} 下没有环境目录")
        return

    for env_dir in env_dirs:
        env_name = env_dir.name
        rel_env = env_dir.relative_to(source_root)

        # --- 视频处理逻辑 ---
        view_map: Dict[str, Path] = {}
        for pat in vid_patterns:
            for f in env_dir.glob(pat):
                stem = f.stem.lower()
                if stem in REQUIRED_VIEWS:
                    view_map[stem] = f.resolve()

        if not all(v in view_map for v in REQUIRED_VIEWS):
            logger.warning(f"[Skip] {rel_env}: 视角不全 {list(view_map.keys())}")
            continue

        view_frames: Dict[str, List[np.ndarray]] = load_data(view_map)

        for view, frames in view_frames.items():
            logger.info(f"  视角 {view} 处理了 {len(frames)} 帧数据。")
            _out_root = out_root / rel_env / view
            _out_root.mkdir(parents=True, exist_ok=True)
            _infer_root = infer_root / rel_env / view
            _infer_root.mkdir(parents=True, exist_ok=True)

            process_frame_list(
                frame_list=frames,
                out_dir=_out_root,
                inference_output_path=_infer_root,
                cfg=cfg,
            )


# ---------------------------------------------------------------------
# GPU Worker：进程执行函数
# ---------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    person_dirs: List[Path],
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """
    # 1. 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg_dict["infer"]["gpu"] = 0  # 因为上面已经隔离了 GPU，所以这里设为 0

    # 2. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🟢 GPU {gpu_id} 进程启动，待处理人数: {len(person_dirs)}")

    for p_dir in person_dirs:
        try:
            process_single_person(p_dir, source_root, out_root, infer_root, cfg)
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 处理 {p_dir.name} 时出错: {e}")

    logger.info(f"🏁 GPU {gpu_id} 所有任务处理完毕")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
# @hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
# def main(cfg: DictConfig) -> None:
#     # 1. 路径准备
#     out_root = Path(cfg.paths.log_path).resolve()
#     infer_root = Path(cfg.paths.result_output_path).resolve()
#     source_root = Path(cfg.paths.video_path).resolve()

#     gpu_ids = cfg.infer.get("gpu", [0, 1])  # 从配置文件读取 GPU 列表，默认 [0, 1]

#     all_person_dirs = sorted([x for x in source_root.iterdir() if x.is_dir()])
#     if not all_person_dirs:
#         logger.error(f"未找到数据目录: {source_root}")
#         return

#     # 2. 自动分组逻辑 (Task Chunking)
#     # 将所有目录分成 N 份，N 等于 GPU 的数量
#     num_gpus = len(gpu_ids)
#     # 使用 np.array_split 可以确保即使除不尽，分配也尽可能均匀
#     chunks = np.array_split(all_person_dirs, num_gpus)

#     logger.info(f"检测到 {num_gpus} 个 GPU: {gpu_ids}")
#     for i, gpu_id in enumerate(gpu_ids):
#         logger.info(f"  - GPU {gpu_id} 分配任务数: {len(chunks[i])}")

#     # 3. 启动并行进程
#     cfg_dict = OmegaConf.to_container(cfg, resolve=True)
#     mp.set_start_method("spawn", force=True)

#     processes = []
#     for i, gpu_id in enumerate(gpu_ids):
#         person_list = chunks[i].tolist()  # 转回普通列表
#         if not person_list:
#             continue

#         p = mp.Process(
#             target=gpu_worker,
#             args=(
#                 gpu_id,
#                 person_list,
#                 source_root,
#                 out_root,
#                 infer_root,
#                 cfg_dict,
#             ),
#         )
#         p.start()
#         processes.append(p)

#     # 4. 等待所有进程完成
#     for p in processes:
#         p.join()

#     logger.info("🎉 [SUCCESS] 所有 GPU 任务已圆满完成！")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 経路準備
    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    source_root = Path(cfg.paths.video_path).resolve()

    # --- 設定の追加 ---
    gpu_ids = cfg.infer.get("gpu", [0, 1])  # 使用するGPUのリスト
    workers_per_gpu = cfg.infer.get("workers_per_gpu", 2)  # 1枚あたりのプロセス数
    
    # 実際に起動するプロセスの数だけGPU IDを並べる (例: [0, 0, 1, 1])
    expanded_gpu_ids = []
    for gid in gpu_ids:
        expanded_gpu_ids.extend([gid] * workers_per_gpu)
    
    total_workers = len(expanded_gpu_ids)
    # ------------------

    all_person_dirs = sorted([x for x in source_root.iterdir() if x.is_dir()])
    if not all_person_dirs:
        logger.error(f"未找到数据目录: {source_root}")
        return

    # 2. 自動分组逻辑 (プロセスの総数で分割)
    chunks = np.array_split(all_person_dirs, total_workers)

    logger.info(f"使用 GPU: {gpu_ids} (各 {workers_per_gpu} ワーカー)")
    logger.info(f"総プロセス数: {total_workers}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, gpu_id in enumerate(expanded_gpu_ids):
        person_list = chunks[i].tolist()
        if not person_list:
            continue

        logger.info(f"  - Worker {i} (GPU {gpu_id}) 分配任务数: {len(person_list)}")

        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                person_list,
                source_root,
                out_root,
                infer_root,
                cfg_dict,
            ),
        )
        p.start()
        processes.append(p)

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    logger.info("🎉 [SUCCESS] 所有 GPU 任务已圆满完成！")

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
