#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
预处理数据到高速缓存格式 - 加速训练数据加载

用法:
    python analysis/preprocess_data_cache.py \
        --dataset-idx data/dual_view_pose/cv_indices.pkl \
        --output-dir data/cached_frames_224 \
        --img-size 224 \
        --num-workers 16
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import cv2
from tqdm import tqdm
import multiprocessing as mp


def process_frame_pair(args: tuple) -> tuple:
    """处理单个frame pair，保存为压缩的.pt文件（含RGB+元数据）"""
    (cam1_path, cam2_path, cam1_idx, cam2_idx, output_dir, img_size) = args
    
    try:
        # 读取、转换、resize、保存
        img1 = cv2.imread(str(cam1_path), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(cam2_path), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return None
        
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (img_size, img_size))
        img2 = cv2.resize(img2, (img_size, img_size))
        
        # 保存为float32 (0-255) 而不是uint8，这样torch的Div255更快
        t1 = torch.from_numpy(np.ascontiguousarray(img1, dtype=np.float32)).permute(2, 0, 1)
        t2 = torch.from_numpy(np.ascontiguousarray(img2, dtype=np.float32)).permute(2, 0, 1)
        
        # 创建缓存标识符
        cache_key = f"cam1_{cam1_idx}_cam2_{cam2_idx}"
        cache_path = Path(output_dir) / f"{cache_key}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({"cam1": t1, "cam2": t2, "key": cache_key}, cache_path)
        return (cache_key, str(cache_path))
    except Exception as e:
        print(f"Error processing {cam1_path}: {e}")
        return None


def preprocess_dataset(dataset_idx_path: str, output_dir: str, img_size: int = 224, num_workers: int = 16):
    """
    预处理所有数据集的图像
    """
    
    with open(dataset_idx_path, 'rb') as f:
        dataset_idx = pickle.load(f)
    
    tasks = []
    for split_name in ["train", "val"]:
        if split_name not in dataset_idx:
            continue
        
        split_data = dataset_idx[split_name]
        for item in split_data:
            cam1_frames_dir = Path(item.get("cam1_frames_dir", ""))
            cam2_frames_dir = Path(item.get("cam2_frames_dir", ""))
            
            if not cam1_frames_dir.exists() or not cam2_frames_dir.exists():
                continue
            
            # 列举所有frame索引
            cam1_frames = sorted(cam1_frames_dir.glob("frame_*.png"))
            cam2_frames = sorted(cam2_frames_dir.glob("frame_*.png"))
            
            for idx, (f1, f2) in enumerate(zip(cam1_frames, cam2_frames)):
                tasks.append((
                    f1, f2, 
                    f1.stem.replace("frame_", ""),
                    f2.stem.replace("frame_", ""),
                    output_dir, img_size
                ))
    
    print(f"准备缓存 {len(tasks)} 个frame pairs...")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_frame_pair, tasks),
            total=len(tasks),
            desc="Caching frames"
        ))
    
    # 保存映射表
    cache_map = {k: v for res in results if res for k, v in [res]}
    output_path = Path(output_dir) / "cache_map.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(cache_map, f)
    
    print(f"✓ 已缓存 {len(cache_map)} frames 到 {output_dir}")
    print(f"✓ 映射表已保存到 {output_path}")
    print(f"✓ 节省空间: {len(tasks)} 次I/O调用 -> 1次加载")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理数据到缓存")
    parser.add_argument("--dataset-idx", required=True, help="数据集索引文件路径 (.pkl)")
    parser.add_argument("--output-dir", required=True, help="缓存输出目录")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=16)
    
    args = parser.parse_args()
    
    preprocess_dataset(
        args.dataset_idx,
        args.output_dir,
        args.img_size,
        args.num_workers
    )
