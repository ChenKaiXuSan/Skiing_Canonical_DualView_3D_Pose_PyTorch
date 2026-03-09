#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check SAM3D-Body inference data completeness against source frames.

Expected layout:
- source_root: person/action/frames/camera/*.png|jpg|jpeg|bmp
- result_root: inference/person/action/frames/camera/*_sam3d_body.npz

The script reports per-capture coverage and missing frame indices.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
NPZ_PATTERN = re.compile(r"^(\d+)_sam3d_body\.npz$")


@dataclass
class CaptureCheck:
    rel_capture: str
    source_frame_count: int
    output_npz_count: int
    coverage: float
    missing_indices: List[int]
    extra_indices: List[int]
    malformed_outputs: List[str]
    corrupt_outputs: List[str]
    output_dir_exists: bool


@dataclass
class IntegritySummary:
    source_root: str
    result_root: str
    total_actions: int
    total_captures: int
    total_source_frames: int
    total_output_npz: int
    captures_missing_output_dir: int
    captures_with_missing_frames: int
    captures_with_extra_frames: int
    captures_with_malformed_outputs: int
    captures_with_corrupt_outputs: int
    strict_mode: bool
    passed: bool


def split_action_dirs(source_root: Path) -> List[Path]:
    """Collect action directories by finding all folders named 'frames'."""
    action_dirs = sorted(
        [p.parent for p in source_root.rglob("frames") if p.is_dir() and p.parent.is_dir()]
    )
    return action_dirs


def list_capture_dirs(action_dir: Path) -> List[Path]:
    frames_dir = action_dir / "frames"
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []
    return sorted([p for p in frames_dir.iterdir() if p.is_dir()])


def list_source_images(capture_dir: Path) -> List[Path]:
    return sorted(
        [p for p in capture_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def parse_output_indices(output_dir: Path) -> Tuple[List[int], List[str], List[str]]:
    """Return (valid_indices, malformed_files, corrupt_files)."""
    valid_indices: List[int] = []
    malformed_files: List[str] = []
    corrupt_files: List[str] = []

    if not output_dir.exists() or not output_dir.is_dir():
        return valid_indices, malformed_files, corrupt_files

    for p in sorted(output_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".npz":
            continue

        m = NPZ_PATTERN.match(p.name)
        if not m:
            malformed_files.append(p.name)
            continue

        idx = int(m.group(1))
        valid_indices.append(idx)

        # Lightweight integrity check: NPZ is readable and has key "output".
        try:
            import numpy as np

            with np.load(p, allow_pickle=True) as data:
                if "output" not in data:
                    corrupt_files.append(p.name)
        except Exception:
            corrupt_files.append(p.name)

    return valid_indices, malformed_files, corrupt_files


def check_integrity(source_root: Path, result_root: Path) -> Tuple[IntegritySummary, List[CaptureCheck]]:
    action_dirs = split_action_dirs(source_root)

    checks: List[CaptureCheck] = []
    total_source_frames = 0
    total_output_npz = 0

    for action_dir in action_dirs:
        for capture_dir in list_capture_dirs(action_dir):
            rel_capture = capture_dir.relative_to(source_root)
            source_images = list_source_images(capture_dir)
            n_src = len(source_images)
            total_source_frames += n_src

            output_dir = result_root / "inference" / rel_capture
            indices, malformed, corrupt = parse_output_indices(output_dir)

            actual_set = set(indices)
            expected_set = set(range(n_src))

            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)

            n_out = len(indices)
            total_output_npz += n_out
            coverage = (n_out / n_src) if n_src > 0 else 1.0

            checks.append(
                CaptureCheck(
                    rel_capture=rel_capture.as_posix(),
                    source_frame_count=n_src,
                    output_npz_count=n_out,
                    coverage=coverage,
                    missing_indices=missing,
                    extra_indices=extra,
                    malformed_outputs=malformed,
                    corrupt_outputs=corrupt,
                    output_dir_exists=output_dir.exists() and output_dir.is_dir(),
                )
            )

    summary = IntegritySummary(
        source_root=str(source_root),
        result_root=str(result_root),
        total_actions=len(action_dirs),
        total_captures=len(checks),
        total_source_frames=total_source_frames,
        total_output_npz=total_output_npz,
        captures_missing_output_dir=sum(1 for c in checks if not c.output_dir_exists),
        captures_with_missing_frames=sum(1 for c in checks if len(c.missing_indices) > 0),
        captures_with_extra_frames=sum(1 for c in checks if len(c.extra_indices) > 0),
        captures_with_malformed_outputs=sum(1 for c in checks if len(c.malformed_outputs) > 0),
        captures_with_corrupt_outputs=sum(1 for c in checks if len(c.corrupt_outputs) > 0),
        strict_mode=False,
        passed=False,
    )

    return summary, checks


def print_report(summary: IntegritySummary, checks: Sequence[CaptureCheck], show_top: int) -> None:
    print("=== SAM3D-Body Integrity Report ===")
    print(f"source_root: {summary.source_root}")
    print(f"result_root: {summary.result_root}")
    print(f"actions: {summary.total_actions}")
    print(f"captures: {summary.total_captures}")
    print(f"source frames: {summary.total_source_frames}")
    print(f"output npz: {summary.total_output_npz}")
    print(f"missing output dirs: {summary.captures_missing_output_dir}")
    print(f"captures with missing frames: {summary.captures_with_missing_frames}")
    print(f"captures with extra frames: {summary.captures_with_extra_frames}")
    print(f"captures with malformed outputs: {summary.captures_with_malformed_outputs}")
    print(f"captures with corrupt outputs: {summary.captures_with_corrupt_outputs}")
    print(f"strict mode: {summary.strict_mode}")
    print(f"passed: {summary.passed}")

    problems = [
        c
        for c in checks
        if (not c.output_dir_exists)
        or c.missing_indices
        or c.extra_indices
        or c.malformed_outputs
        or c.corrupt_outputs
    ]

    if not problems:
        print("No capture-level problems found.")
        return

    print(f"\nTop {min(show_top, len(problems))} problematic captures:")
    for c in problems[:show_top]:
        print(f"- {c.rel_capture}")
        print(
            f"  src={c.source_frame_count}, out={c.output_npz_count}, coverage={c.coverage:.3f}, output_dir_exists={c.output_dir_exists}"
        )
        if c.missing_indices:
            print(f"  missing={len(c.missing_indices)} (first: {c.missing_indices[:10]})")
        if c.extra_indices:
            print(f"  extra={len(c.extra_indices)} (first: {c.extra_indices[:10]})")
        if c.malformed_outputs:
            print(f"  malformed={len(c.malformed_outputs)} (first: {c.malformed_outputs[:5]})")
        if c.corrupt_outputs:
            print(f"  corrupt={len(c.corrupt_outputs)} (first: {c.corrupt_outputs[:5]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAM3D-Body inference integrity")
    parser.add_argument("--source-root", type=Path, required=True, help="Dataset data root")
    parser.add_argument("--result-root", type=Path, required=True, help="SAM3D result root")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("analysis/sam3d_body_integrity_report.json"),
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=20,
        help="Show top N problematic captures in stdout",
    )
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help="Do not fail on missing frame indices (model may skip undetected frames)",
    )

    args = parser.parse_args()

    source_root = args.source_root.resolve()
    result_root = args.result_root.resolve()

    if not source_root.exists():
        print(f"[ERROR] source_root not found: {source_root}", file=sys.stderr)
        return 2

    if not result_root.exists():
        print(f"[ERROR] result_root not found: {result_root}", file=sys.stderr)
        return 2

    summary, checks = check_integrity(source_root, result_root)

    has_hard_errors = (
        summary.captures_missing_output_dir > 0
        or summary.captures_with_extra_frames > 0
        or summary.captures_with_malformed_outputs > 0
        or summary.captures_with_corrupt_outputs > 0
    )
    has_missing = summary.captures_with_missing_frames > 0

    strict_mode = not args.allow_missing_frames
    passed = (not has_hard_errors) and ((not has_missing) or args.allow_missing_frames)

    summary.strict_mode = strict_mode
    summary.passed = passed

    report = {
        "summary": asdict(summary),
        "captures": [asdict(c) for c in checks],
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(summary, checks, args.show_top)
    print(f"\nJSON report saved to: {args.report_json}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
