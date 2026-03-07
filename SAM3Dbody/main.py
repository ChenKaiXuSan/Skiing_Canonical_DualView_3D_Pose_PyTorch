#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run SAM-3D-Body inference on unity image frames by action in parallel."""

import logging
import multiprocessing as mp
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_frame_list
from .load import collect_action_dirs, load_capture_frames

logger = logging.getLogger(__name__)

def split_evenly(items: List[Path], num_chunks: int) -> List[List[Path]]:
    """Split a list into near-even contiguous chunks."""
    if num_chunks <= 0:
        return []

    n = len(items)
    base = n // num_chunks
    extra = n % num_chunks

    chunks: List[List[Path]] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(items[start:end])
        start = end
    return chunks


def process_single_action(
    action_dir: Path,
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    cfg: DictConfig,
) -> None:
    """Process all captures in one action directory."""
    rel_action = action_dir.relative_to(source_root)
    action_id = str(rel_action).replace("/", "__")

    log_dir = infer_root.parent / "logs" / "action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    action_log_file = log_dir / f"{action_id}.log"

    handler = logging.FileHandler(action_log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    action_logger = logging.getLogger(f"action_{action_id}")
    action_logger.handlers.clear()
    action_logger.addHandler(handler)
    action_logger.propagate = False

    action_logger.info("==== Start Action: %s ====", rel_action)

    frames_dir = action_dir / "frames"
    capture_dirs = sorted([x for x in frames_dir.iterdir() if x.is_dir()])
    if not capture_dirs:
        action_logger.warning("[Skip] No capture dirs in: %s", frames_dir)
        return

    for capture_dir in capture_dirs:
        rel_capture = capture_dir.relative_to(source_root)
        frame_list = load_capture_frames(capture_dir)
        if not frame_list:
            action_logger.warning("[Skip] Empty capture: %s", rel_capture)
            continue

        action_logger.info(
            "Processing %s, frame_count=%d",
            rel_capture,
            len(frame_list),
        )

        out_dir = vis_root / rel_capture
        out_dir.mkdir(parents=True, exist_ok=True)

        infer_dir = infer_root / rel_capture
        infer_dir.mkdir(parents=True, exist_ok=True)

        process_frame_list(
            frame_list=frame_list,
            out_dir=out_dir,
            inference_output_path=infer_dir,
            cfg=cfg,
        )

    action_logger.info("==== Finished Action: %s ====", rel_action)


def gpu_worker(
    gpu_id: Union[int, str],
    action_dirs: List[Path],
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    cfg_dict: dict,
    worker_id: int,
) -> None:
    """Worker entrypoint: pin device and process assigned actions."""
    is_cpu = isinstance(gpu_id, str) and gpu_id.lower() == "cpu"
    if is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    local_cfg_dict = deepcopy(cfg_dict)
    local_cfg_dict.setdefault("infer", {})
    local_cfg_dict["infer"]["gpu"] = "cpu" if is_cpu else 0
    cfg = OmegaConf.create(local_cfg_dict)

    logger.info(
        "[Worker %d] GPU %s started, actions=%d",
        worker_id,
        gpu_id,
        len(action_dirs),
    )

    for action_dir in action_dirs:
        try:
            process_single_action(action_dir, source_root, vis_root, infer_root, cfg)
        except Exception as exc:
            logger.error(
                "[Worker %d] Failed on action %s: %s",
                worker_id,
                action_dir.name,
                exc,
            )

    logger.info("[Worker %d] GPU %s finished", worker_id, gpu_id)


def normalize_gpu_ids(raw_gpu_ids) -> List[Union[int, str]]:
    """Normalize gpu config to a list of integer ids."""
    if isinstance(raw_gpu_ids, str) and raw_gpu_ids.lower() == "cpu":
        return ["cpu"]

    if isinstance(raw_gpu_ids, int):
        return [raw_gpu_ids]

    if isinstance(raw_gpu_ids, str):
        if "," in raw_gpu_ids:
            parsed_ids: List[Union[int, str]] = []
            for x in raw_gpu_ids.split(","):
                x = x.strip()
                if not x:
                    continue
                parsed_ids.append("cpu" if x.lower() == "cpu" else int(x))
            return parsed_ids
        return [int(raw_gpu_ids)]

    if isinstance(raw_gpu_ids, (list, tuple)):
        parsed_ids: List[Union[int, str]] = []
        for x in raw_gpu_ids:
            if isinstance(x, str) and x.lower() == "cpu":
                parsed_ids.append("cpu")
            else:
                parsed_ids.append(int(x))
        return parsed_ids

    return [0]


def select_action_shard(
    action_dirs: List[Path],
    shard_count: int,
    shard_index: int,
) -> List[Path]:
    """Select one shard of actions for multi-node execution."""
    if shard_count <= 1:
        return action_dirs

    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"Invalid shard index {shard_index} for shard_count {shard_count}"
        )

    shard_chunks = split_evenly(action_dirs, shard_count)
    return shard_chunks[shard_index]


@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    source_root = Path(cfg.paths.unity.unity_dataset_data_path).resolve()
    result_root = Path(cfg.paths.unity.unity_sam3d_result_root).resolve()

    vis_root = result_root / "visualization"
    infer_root = result_root / "inference"
    vis_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = normalize_gpu_ids(cfg.infer.gpu)
    workers_per_gpu = int(cfg.infer.workers_per_gpu)
    workers_per_gpu = max(workers_per_gpu, 1)

    expanded_gpu_ids: List[Union[int, str]] = []
    for gid in gpu_ids:
        expanded_gpu_ids.extend([gid] * workers_per_gpu)

    total_workers = len(expanded_gpu_ids)
    if total_workers < 1:
        logger.error("No worker created. Check infer.gpu / infer.workers_per_gpu")
        return

    action_dirs_all = collect_action_dirs(source_root)
    if not action_dirs_all:
        logger.error("No action dirs found in: %s", source_root)
        return

    shard_count = max(int(getattr(cfg.infer, "shard_count", 1)), 1)
    shard_index = int(getattr(cfg.infer, "shard_index", 0))

    try:
        action_dirs = select_action_shard(action_dirs_all, shard_count, shard_index)
    except ValueError as exc:
        logger.error("%s", exc)
        return

    if not action_dirs:
        logger.warning(
            "No actions assigned to this shard (index=%d/%d). Exit.",
            shard_index,
            shard_count,
        )
        return

    chunks = split_evenly(action_dirs, total_workers)

    logger.info("Source data root: %s", source_root)
    logger.info("Result root: %s", result_root)
    logger.info("GPU ids: %s, workers_per_gpu=%d", gpu_ids, workers_per_gpu)
    logger.info(
        "Shard: index=%d/%d, actions_in_shard=%d, total_actions=%d",
        shard_index,
        shard_count,
        len(action_dirs),
        len(action_dirs_all),
    )
    logger.info("Total workers: %d, total actions: %d", total_workers, len(action_dirs))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.error("Failed to convert config to dict")
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes: List[mp.Process] = []
    for i, gpu_id in enumerate(expanded_gpu_ids):
        action_list = chunks[i]
        if not action_list:
            continue

        logger.info(
            "Assign worker=%d, gpu=%s, action_count=%d",
            i,
            gpu_id,
            len(action_list),
        )

        process = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                action_list,
                source_root,
                vis_root,
                infer_root,
                cfg_dict,
                i,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    logger.info("[SUCCESS] All action workers completed")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
