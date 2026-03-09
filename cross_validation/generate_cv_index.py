#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
生成交叉验证索引文件
用于保存摄像头两两组合的交叉验证划分结果
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validation.cross_validation_camera_pairs import CameraPairCrossValidation


def generate_index_files(
    data_root: str,
    num_persons: int = 2,
    num_actions: int = 12,
    num_cameras: int = 108,
    strategies: Optional[List[str]] = None,
    n_splits: int = 5,
    force_recreate: bool = False,
):
    """
    生成所有策略的索引文件
    
    Args:
        data_root: 数据根目录
        num_persons: 人物数量
        num_actions: 动作数量
        num_cameras: 摄像头数量
        strategies: 要生成的策略列表，默认生成所有策略
        n_splits: K折数量
        force_recreate: 是否强制重新生成
    """
    if strategies is None:
        strategies = ["by_person", "by_action", "by_camera_pair"]
    
    index_mapping_dir = Path(data_root) / "index_mapping"
    index_mapping_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("生成交叉验证索引文件")
    print("="*80)
    print(f"数据根目录: {data_root}")
    print(f"人物数: {num_persons}")
    print(f"动作数: {num_actions}")
    print(f"摄像头数: {num_cameras}")
    print(f"策略: {', '.join(strategies)}")
    print(f"保存目录: {index_mapping_dir}")
    print("="*80 + "\n")
    
    results: Dict[str, Dict[str, object]] = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"生成策略: {strategy}")
        print(f"{'='*60}")
        
        # 创建交叉验证对象
        cv = CameraPairCrossValidation(
            data_root=data_root,
            num_persons=num_persons,
            num_actions=num_actions,
            num_cameras=num_cameras,
            split_strategy=strategy,
            n_splits=n_splits,
            index_save_path=str(index_mapping_dir / f"camera_pairs_{strategy}.json")
        )
        
        # 生成并保存索引
        folds = cv(force_recreate=force_recreate)
        
        # 记录结果
        results[strategy] = {
            "file": str(index_mapping_dir / f"camera_pairs_{strategy}.json"),
            "n_folds": len(folds),
            "total_samples": sum(len(fold["train"]) + len(fold["val"]) 
                                for fold in folds.values()) // len(folds)
        }
        
        print(f"\n✓ 策略 '{strategy}' 索引文件已生成")
        print(f"  文件路径: {results[strategy]['file']}")
        print(f"  折数: {results[strategy]['n_folds']}")
        print(f"  每折样本数: {results[strategy]['total_samples']}")
    
    # 打印汇总信息
    print("\n" + "="*80)
    print("✅ 所有索引文件生成完成")
    print("="*80)
    
    for strategy, info in results.items():
        print(f"\n[{strategy}]")
        print(f"  文件: {info['file']}")
        print(f"  折数: {info['n_folds']}")
        print(f"  样本数: {info['total_samples']:,}")
        
        # 获取文件大小
        file_path = Path(str(info['file']))
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  文件大小: {size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("📁 所有索引文件位置:")
    print(f"   {index_mapping_dir}")
    print("="*80 + "\n")
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def hydra_main(cfg: DictConfig):
    """
    通过 Hydra 配置生成 index mapping。

    默认读取 configs/train.yaml 中 data.cross_validation 字段，
    也支持命令行 override，例如：
    python cross_validation/generate_cv_index.py data.cross_validation.force_recreate=true
    """
    cv_cfg = cfg.data.cross_validation

    # 打印当前配置，便于排查。
    print("\nHydra cross validation config:")
    print(OmegaConf.to_yaml(cv_cfg))

    generate_index_files(
        data_root=str(cfg.data.root_path),
        num_persons=int(cv_cfg.num_persons),
        num_actions=int(cv_cfg.num_actions),
        num_cameras=int(cv_cfg.num_cameras),
        strategies=list(cv_cfg.strategies),
        n_splits=int(cv_cfg.n_splits),
        force_recreate=bool(cv_cfg.force_recreate),
    )


def main() -> None:
    cast(Any, hydra_main)()


if __name__ == "__main__":
    main()
