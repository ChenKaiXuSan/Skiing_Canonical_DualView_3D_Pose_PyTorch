#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/cross_validation_camera_pairs.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project
Created Date: Sunday March 9th 2026
Author: Kaixu Chen
-----
Comment:
交叉验证脚本 - 用于摄像头两两组合的场景
针对2个人物、12个动作、每个动作108个摄像头的数据集。
支持三种划分策略：
1. by_person: 按人物划分（Leave-One-Person-Out）
2. by_action: 按动作划分（K-Fold on actions）
3. by_camera_pair: 按摄像头对划分（K-Fold on camera pairs）

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
from sklearn.model_selection import KFold


@dataclass
class CameraPairSample:
    """表示一个训练样本：人物 + 动作 + 摄像头对"""
    person_id: str  # "01", "02"
    action_id: str  # "action_01", "action_02", ..., "action_12"
    cam1_id: str    # "cam_001", "cam_002", ..., "cam_108"
    cam2_id: str    # "cam_001", "cam_002", ..., "cam_108"
    cam1_path: Optional[str] = None  # 兼容旧字段：camera1 frames目录
    cam2_path: Optional[str] = None  # 兼容旧字段：camera2 frames目录
    label_path: Optional[str] = None  # 兼容旧字段：sequence.json路径
    cam1_frames_dir: Optional[str] = None
    cam2_frames_dir: Optional[str] = None
    cam1_kpt2d_dir: Optional[str] = None
    cam2_kpt2d_dir: Optional[str] = None
    kpt3d_dir: Optional[str] = None
    sam3d_cam1_dir: Optional[str] = None
    sam3d_cam2_dir: Optional[str] = None
    sequence_meta_path: Optional[str] = None
    joint_names_path: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


class CameraPairCrossValidation:
    """
    针对摄像头对的交叉验证策略
    
    参数:
        data_root: 数据根目录
        num_persons: 人物数量（默认2）
        num_actions: 动作数量（默认12）
        num_cameras: 每个动作的摄像头数量（默认108）
        split_strategy: 划分策略，可选 'by_person', 'by_action', 'by_camera_pair'
        n_splits: K折交叉验证的折数（仅用于 by_action 和 by_camera_pair 策略）
        index_save_path: 索引文件保存路径
    """
    
    def __init__(
        self,
        data_root: str,
        num_persons: int = 2,
        num_actions: int = 12,
        num_cameras: int = 108,
        split_strategy: str = "by_person",  # by_person, by_action, by_camera_pair
        n_splits: int = 5,
        index_save_path: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.data_dir = self.data_root / "data"
        self.sam3d_root = self.data_root / "sam3d_body_results" / "inference"
        self.num_persons = num_persons
        self.num_actions = num_actions
        self.num_cameras = num_cameras
        self.split_strategy = split_strategy
        self.n_splits = n_splits
        
        if index_save_path is None:
            self.index_save_path: Path = self.data_root / "index_mapping" / f"camera_pairs_{split_strategy}.json"
        else:
            self.index_save_path = Path(index_save_path)
        
        self.index_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _capture_to_kpt2d_id(capture_name: str) -> str:
        # frames目录是 capture_Lx_Ayyy；kpt2d目录是 Lx_Ayyy
        return capture_name.replace("capture_", "", 1)

    def _discover_people_actions(self) -> Dict[str, List[str]]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data目录不存在: {self.data_dir}")

        people_actions: Dict[str, List[str]] = {}
        for person_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            # 跳过辅助目录
            if person_dir.name.lower() in {"logs", "cameras"}:
                continue
            action_names: List[str] = []
            for action_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                if (action_dir / "frames").exists():
                    action_names.append(action_dir.name)
            if action_names:
                people_actions[person_dir.name] = action_names
        return people_actions
    
    def build_all_samples(self) -> List[CameraPairSample]:
        """
        扫描真实目录，构建样本：person × action × camera_capture_pairs
        
        Returns:
            所有样本的列表
        """
        samples: List[CameraPairSample] = []
        people_actions = self._discover_people_actions()

        action_count_total = 0
        per_action_pair_count: List[int] = []

        for person_id, actions in people_actions.items():
            for action_id in actions:
                action_count_total += 1
                action_dir = self.data_dir / person_id / action_id
                frames_root = action_dir / "frames"
                kpt2d_root = action_dir / "kpt2d"
                kpt3d_dir = action_dir / "kpt3d"
                meta_dir = action_dir / "meta"
                sam3d_frames_root = self.sam3d_root / person_id / action_id / "frames"

                capture_dirs = sorted(p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("capture_"))
                if len(capture_dirs) < 2:
                    continue

                per_action_pair_count.append(len(capture_dirs) * (len(capture_dirs) - 1) // 2)

                for cam1_dir, cam2_dir in combinations(capture_dirs, 2):
                    cam1_id = cam1_dir.name
                    cam2_id = cam2_dir.name

                    kpt2d_cam1 = kpt2d_root / self._capture_to_kpt2d_id(cam1_id)
                    kpt2d_cam2 = kpt2d_root / self._capture_to_kpt2d_id(cam2_id)

                    sam3d_cam1 = sam3d_frames_root / cam1_id
                    sam3d_cam2 = sam3d_frames_root / cam2_id

                    sequence_meta = meta_dir / "sequence.json"
                    joint_meta = meta_dir / "joint_names.json"

                    sample = CameraPairSample(
                        person_id=person_id,
                        action_id=action_id,
                        cam1_id=cam1_id,
                        cam2_id=cam2_id,
                        cam1_path=str(cam1_dir),
                        cam2_path=str(cam2_dir),
                        label_path=str(sequence_meta),
                        cam1_frames_dir=str(cam1_dir),
                        cam2_frames_dir=str(cam2_dir),
                        cam1_kpt2d_dir=str(kpt2d_cam1),
                        cam2_kpt2d_dir=str(kpt2d_cam2),
                        kpt3d_dir=str(kpt3d_dir),
                        sam3d_cam1_dir=str(sam3d_cam1),
                        sam3d_cam2_dir=str(sam3d_cam2),
                        sequence_meta_path=str(sequence_meta),
                        joint_names_path=str(joint_meta),
                    )
                    samples.append(sample)

        people_count = len(people_actions)
        action_count = sum(len(v) for v in people_actions.values())
        avg_pairs = int(np.mean(per_action_pair_count)) if per_action_pair_count else 0

        print(f"✓ 总共生成 {len(samples)} 个样本")
        print(f"  - {people_count} 个人物")
        print(f"  - {action_count} 个动作")
        print(f"  - 每个动作平均 {avg_pairs} 个摄像头对")
        
        return samples
    
    def split_by_person(self, samples: List[CameraPairSample]) -> Dict[int, Dict[str, Any]]:
        """
        策略1: 按人物划分 (Leave-One-Person-Out)
        每一折使用一个人的数据作为验证集，其他人的数据作为训练集
        """
        fold_dict: Dict[int, Dict[str, Any]] = {}
        
        person_ids = sorted(set(s.person_id for s in samples))
        
        if len(person_ids) <= 1:
            return {0: {"train": samples, "val": [], "val_person": person_ids[0] if person_ids else "unknown"}}

        for fold_idx, val_person in enumerate(person_ids):
            train_samples = [s for s in samples if s.person_id != val_person]
            val_samples = [s for s in samples if s.person_id == val_person]
            
            fold_dict[fold_idx] = {
                "train": train_samples,
                "val": val_samples,
                "val_person": val_person,
            }
            
            print(f"Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)} (person={val_person})")
        
        return fold_dict
    
    def split_by_action(self, samples: List[CameraPairSample]) -> Dict[int, Dict[str, Any]]:
        """
        策略2: 按动作划分 (K-Fold on actions)
        将动作分成K折，每折某些动作用于验证，其他用于训练
        """
        fold_dict: Dict[int, Dict[str, Any]] = {}
        
        action_ids = sorted(set(s.action_id for s in samples))
        n_splits = min(self.n_splits, len(action_ids))
        if n_splits <= 1:
            return {0: {"train": samples, "val": [], "val_actions": []}}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_action_indices, val_action_indices) in enumerate(kf.split(action_ids)):
            train_actions = [action_ids[i] for i in train_action_indices]
            val_actions = [action_ids[i] for i in val_action_indices]
            
            train_samples = [s for s in samples if s.action_id in train_actions]
            val_samples = [s for s in samples if s.action_id in val_actions]
            
            fold_dict[fold_idx] = {
                "train": train_samples,
                "val": val_samples,
                "val_actions": val_actions,
            }
            
            print(f"Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)} (actions={val_actions})")
        
        return fold_dict
    
    def split_by_camera_pair(self, samples: List[CameraPairSample]) -> Dict[int, Dict[str, Any]]:
        """
        策略3: 按摄像头对划分 (K-Fold on camera pairs)
        将摄像头对分成K折进行交叉验证
        
        注意：这种方式会产生大量的样本，可能需要采样或分层
        """
        fold_dict: Dict[int, Dict[str, List[CameraPairSample]]] = {}
        
        # 按照 (person, action, cam_pair) 分组
        # 为了简化，我们可以只按摄像头对来划分
        camera_pairs = list(set((s.cam1_id, s.cam2_id) for s in samples))
        
        # 如果摄像头对太多，可以考虑采样
        if len(camera_pairs) > self.n_splits * 100:
            print(f"⚠ 摄像头对数量过多 ({len(camera_pairs)})，考虑使用随机采样")
            np.random.seed(42)
            sampled_indices = np.random.choice(len(samples), size=min(len(samples), 10000), replace=False)
            samples = [samples[i] for i in sampled_indices]
            camera_pairs = list(set((s.cam1_id, s.cam2_id) for s in samples))
        
        n_splits = min(self.n_splits, len(camera_pairs))
        if n_splits <= 1:
            return {0: {"train": samples, "val": []}}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_pair_indices, val_pair_indices) in enumerate(kf.split(camera_pairs)):
            train_pairs = set(camera_pairs[i] for i in train_pair_indices)
            val_pairs = set(camera_pairs[i] for i in val_pair_indices)
            
            train_samples = [s for s in samples if (s.cam1_id, s.cam2_id) in train_pairs]
            val_samples = [s for s in samples if (s.cam1_id, s.cam2_id) in val_pairs]
            
            fold_dict[fold_idx] = {
                "train": train_samples,
                "val": val_samples,
            }
            
            print(f"Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)}")
        
        return fold_dict
    
    def prepare_folds(self) -> Dict[int, Dict[str, Any]]:
        """
        根据选择的策略准备交叉验证的折
        """
        print(f"\n{'='*60}")
        print("准备交叉验证数据集")
        print(f"{'='*60}")
        print(f"策略: {self.split_strategy}")
        print(f"数据根目录: {self.data_root}")
        print(f"{'='*60}\n")
        
        samples = self.build_all_samples()
        
        if self.split_strategy == "by_person":
            fold_dict = self.split_by_person(samples)
        elif self.split_strategy == "by_action":
            fold_dict = self.split_by_action(samples)
        elif self.split_strategy == "by_camera_pair":
            fold_dict = self.split_by_camera_pair(samples)
        else:
            raise ValueError(f"未知的划分策略: {self.split_strategy}")
        
        return fold_dict
    
    def save_folds(self, fold_dict: Dict[int, Dict[str, Any]]):
        """
        保存交叉验证的划分结果到JSON文件
        """
        # 序列化
        serialized: Dict[str, Any] = {}
        for fold_idx, fold_data in fold_dict.items():
            serialized[str(fold_idx)] = {
                "train": [s.to_dict() for s in fold_data["train"]],
                "val": [s.to_dict() for s in fold_data["val"]],
            }
            # 保存额外信息（如验证集的人物或动作）
            for key in fold_data:
                if key not in ["train", "val"]:
                    serialized[str(fold_idx)][key] = fold_data[key]
        
        # 添加元数据
        serialized["_metadata"] = {
            "num_persons": self.num_persons,
            "num_actions": self.num_actions,
            "num_cameras": self.num_cameras,
            "split_strategy": self.split_strategy,
            "n_splits": len(fold_dict),
            "total_samples": sum(len(fold_data["train"]) + len(fold_data["val"]) 
                                for fold_data in fold_dict.values()) // len(fold_dict),
        }
        
        with open(self.index_save_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 交叉验证索引已保存到: {self.index_save_path}")
    
    def load_folds(self) -> Dict[int, Dict[str, Any]]:
        """
        从JSON文件加载交叉验证的划分结果
        """
        if not self.index_save_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_save_path}")
        
        with open(self.index_save_path, "r", encoding="utf-8") as f:
            serialized = json.load(f)
        
        # 提取元数据
        metadata = serialized.pop("_metadata", {})
        print("\n加载交叉验证数据:")
        print(f"  策略: {metadata.get('split_strategy', 'unknown')}")
        print(f"  折数: {metadata.get('n_splits', 'unknown')}")
        print(f"  总样本数: {metadata.get('total_samples', 'unknown')}")
        
        # 反序列化
        fold_dict: Dict[int, Dict[str, Any]] = {}
        for fold_idx_str, fold_data in serialized.items():
            fold_idx = int(fold_idx_str)
            fold_dict[fold_idx] = {
                "train": [CameraPairSample.from_dict(d) for d in fold_data["train"]],
                "val": [CameraPairSample.from_dict(d) for d in fold_data["val"]],
            }
            # 恢复额外信息
            for key in fold_data:
                if key not in ["train", "val"]:
                    fold_dict[fold_idx][key] = fold_data[key]
        
        return fold_dict
    
    def __call__(self, force_recreate: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        主入口：创建或加载交叉验证划分
        
        Args:
            force_recreate: 是否强制重新创建索引文件
        """
        if self.index_save_path.exists() and not force_recreate:
            print("✓ 发现已存在的索引文件，直接加载")
            return self.load_folds()
        else:
            print("✓ 创建新的交叉验证划分")
            fold_dict = self.prepare_folds()
            self.save_folds(fold_dict)
            return fold_dict


def main():
    """
    示例使用
    """
    for strategy in ["by_person", "by_action", "by_camera_pair"]:
        cv = CameraPairCrossValidation(
            data_root="/workspace/data/skiing_unity_dataset",
            split_strategy=strategy,
            n_splits=5,
            index_save_path=f"/workspace/data/skiing_unity_dataset/index_mapping/camera_pairs_{strategy}.json",
        )
        folds = cv(force_recreate=True)
        print(f"\n示例 - {strategy}:")
        print(f"折数: {len(folds)}")


if __name__ == "__main__":
    main()
