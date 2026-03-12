#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Export SAM3D inference npz into separated kpt2d/kpt3d files.

Input structure:
  inference/<person>/<action>/frames/<camera>/*_sam3d_body.npz

Output structure:
  <output_root>/<person>/<action>/kpt2d/<camera>/kpt2d_XXXXXX.npy
  <output_root>/<person>/<action>/kpt3d/<camera>/kpt3d_XXXXXX.npy
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

NPZ_PATTERN = re.compile(r"^(\d+)_sam3d_body\.npz$")


@dataclass
class ExportStats:
    people: int = 0
    actions: int = 0
    cameras: int = 0
    npz_files: int = 0
    kpt2d_saved: int = 0
    kpt3d_saved: int = 0
    npz_errors: int = 0


def iter_npz_files(camera_dir: Path) -> Iterable[Tuple[int, Path]]:
    for npz_path in sorted(camera_dir.glob("*_sam3d_body.npz")):
        m = NPZ_PATTERN.match(npz_path.name)
        if m is None:
            continue
        frame_idx = int(m.group(1))
        yield frame_idx, npz_path


def squeeze_pose(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out


def load_sam_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if "output" not in data.files:
        raise KeyError("Missing key 'output'")

    output = data["output"]
    if isinstance(output, np.ndarray) and output.shape == ():
        output = output.item()
    if not isinstance(output, dict):
        raise TypeError(f"Unexpected output type: {type(output)}")

    if "pred_keypoints_3d" in output:
        arr_3d = output["pred_keypoints_3d"]
    elif "pred_joint_coords" in output:
        arr_3d = output["pred_joint_coords"]
    else:
        raise KeyError(f"No 3D key in output: {list(output.keys())}")

    if "pred_keypoints_2d" in output:
        arr_2d = output["pred_keypoints_2d"]
    else:
        arr_2d = np.asarray(arr_3d, dtype=np.float32)[..., :2]

    return squeeze_pose(arr_2d), squeeze_pose(arr_3d)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def export_one_camera(
    npz_camera_dir: Path,
    output_root: Path,
    person: str,
    action: str,
    camera: str,
    overwrite: bool,
    dry_run: bool,
    stats: ExportStats,
) -> None:
    out_kpt2d_dir = output_root / person / action / "kpt2d" / camera
    out_kpt3d_dir = output_root / person / action / "kpt3d" / camera

    none_txt = npz_camera_dir / "none_detected_frames.txt"
    if none_txt.exists() and none_txt.is_file() and not dry_run:
        out_kpt2d_dir.mkdir(parents=True, exist_ok=True)
        out_kpt3d_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(none_txt, out_kpt2d_dir / none_txt.name)
        shutil.copy2(none_txt, out_kpt3d_dir / none_txt.name)

    for frame_idx, npz_path in iter_npz_files(npz_camera_dir):
        stats.npz_files += 1

        try:
            kpt2d, kpt3d = load_sam_npz(npz_path)
        except Exception as ex:
            stats.npz_errors += 1
            print(f"[WARN] load npz failed: {npz_path} ({ex})")
            continue

        out_2d = out_kpt2d_dir / f"kpt2d_{frame_idx:06d}.npy"
        out_3d = out_kpt3d_dir / f"kpt3d_{frame_idx:06d}.npy"

        if dry_run:
            stats.kpt2d_saved += 1
            stats.kpt3d_saved += 1
        else:
            if overwrite or not out_2d.exists():
                ensure_parent(out_2d)
                np.save(out_2d, kpt2d)
                stats.kpt2d_saved += 1
            if overwrite or not out_3d.exists():
                ensure_parent(out_3d)
                np.save(out_3d, kpt3d)
                stats.kpt3d_saved += 1


def run(args: argparse.Namespace) -> None:
    inference_root = args.inference_root
    output_root = args.output_root

    stats = ExportStats()

    people = [p for p in sorted(inference_root.iterdir()) if p.is_dir()]
    stats.people = len(people)

    for person_dir in people:
        person = person_dir.name
        action_dirs = [d for d in sorted(person_dir.iterdir()) if d.is_dir()]
        if args.max_actions is not None:
            action_dirs = action_dirs[: args.max_actions]

        for action_dir in action_dirs:
            action = action_dir.name
            npz_frames_root = action_dir / "frames"
            if not npz_frames_root.exists() or not npz_frames_root.is_dir():
                continue

            stats.actions += 1
            camera_dirs = [d for d in sorted(npz_frames_root.iterdir()) if d.is_dir()]
            if args.max_cameras is not None:
                camera_dirs = camera_dirs[: args.max_cameras]
            stats.cameras += len(camera_dirs)

            for camera_dir in camera_dirs:
                export_one_camera(
                    npz_camera_dir=camera_dir,
                    output_root=output_root,
                    person=person,
                    action=action,
                    camera=camera_dir.name,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    stats=stats,
                )

    print("\n=== Export Summary ===")
    print(f"people:         {stats.people}")
    print(f"actions:        {stats.actions}")
    print(f"cameras:        {stats.cameras}")
    print(f"npz files:      {stats.npz_files}")
    print(f"kpt2d saved:    {stats.kpt2d_saved}")
    print(f"kpt3d saved:    {stats.kpt3d_saved}")
    print(f"npz errors:     {stats.npz_errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAM3D npz into separated frames/kpt2d/kpt3d directories.",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=Path("/workspace/data/skiing_unity_dataset/sam3d_body_results/inference"),
        help="Root of SAM3D inference outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/data/skiing_unity_dataset/sam3d_body_results/modalities_from_sam3d"),
        help="Output root for exported modalities.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and count only, do not write files.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Only process first N actions per person (debug/quick run).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Only process first N cameras per action (debug/quick run).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
