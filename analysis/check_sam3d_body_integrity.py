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
CAMERA_PATTERN = re.compile(r"^capture_L(\d+)_A(\d{3})$")
EXPECTED_LEVELS = set(range(5))
EXPECTED_ANGLES = set(range(0, 360, 10))
# DEFAULT_SOURCE_ROOT = Path("/work/SSR/share/data/skiing/skiing_unity_dataset/data")
DEFAULT_SOURCE_ROOT = Path("/workspace/data/skiing_unity_dataset/data")

# DEFAULT_RESULT_ROOT = Path("/work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results")
DEFAULT_RESULT_ROOT = Path("/workspace/data/skiing_unity_dataset/sam3d_body_results")


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
    none_file_exists: bool
    none_indices: List[int]
    none_file_invalid_lines: List[str]
    none_mismatch_indices: List[int]
    output_dir_exists: bool


@dataclass
class IntegritySummary:
    source_root: str
    result_root: str
    total_persons: int
    unique_action_names: int
    total_actions: int
    camera_level_depth: int
    unique_cameras: int
    total_cameras: int
    detected_camera_levels: List[int]
    detected_angles: List[int]
    captures_with_bad_camera_name: int
    actions_with_missing_levels: int
    actions_with_missing_angles: int
    actions_with_wrong_camera_count: int
    actions_with_camera_config_issues: int
    camera_config_passed: bool
    total_captures: int
    total_source_frames: int
    total_output_npz: int
    captures_missing_output_dir: int
    captures_with_missing_frames: int
    captures_with_extra_frames: int
    captures_with_malformed_outputs: int
    captures_with_corrupt_outputs: int
    captures_with_none_file: int
    captures_with_none_parse_errors: int
    captures_with_none_mismatch: int
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


def parse_none_detected_indices(output_dir: Path) -> Tuple[bool, List[int], List[str]]:
    """Return (exists, parsed_indices, invalid_lines)."""
    none_file = output_dir / "none_detected_frames.txt"
    if not none_file.exists() or not none_file.is_file():
        return False, [], []

    indices: List[int] = []
    invalid_lines: List[str] = []
    for raw in none_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            idx = int(line)
        except ValueError:
            invalid_lines.append(line)
            continue
        if idx < 0:
            invalid_lines.append(line)
            continue
        indices.append(idx)

    return True, sorted(set(indices)), invalid_lines


def check_integrity(source_root: Path, result_root: Path) -> Tuple[IntegritySummary, List[CaptureCheck]]:
    action_dirs = split_action_dirs(source_root)
    person_names = {
        action_dir.relative_to(source_root).parts[0]
        for action_dir in action_dirs
        if len(action_dir.relative_to(source_root).parts) >= 1
    }
    action_names = {
        action_dir.name
        for action_dir in action_dirs
    }

    checks: List[CaptureCheck] = []
    total_source_frames = 0
    total_output_npz = 0
    capture_name_set = set()
    capture_depths: List[int] = []
    detected_levels: set[int] = set()
    detected_angles: set[int] = set()
    bad_camera_name_count = 0
    action_to_camera_pairs: Dict[str, set[Tuple[int, int]]] = {}
    action_to_bad_names: Dict[str, int] = {}

    for action_dir in action_dirs:
        action_rel = action_dir.relative_to(source_root).as_posix()
        action_to_camera_pairs[action_rel] = set()
        action_to_bad_names[action_rel] = 0

        for capture_dir in list_capture_dirs(action_dir):
            rel_capture = capture_dir.relative_to(source_root)
            rel_from_frames = capture_dir.relative_to(action_dir / "frames")
            capture_depths.append(len(rel_from_frames.parts))
            if rel_from_frames.parts:
                camera_name = rel_from_frames.parts[-1]
                capture_name_set.add(camera_name)
                m_cam = CAMERA_PATTERN.match(camera_name)
                if m_cam:
                    level = int(m_cam.group(1))
                    angle = int(m_cam.group(2))
                    detected_levels.add(level)
                    detected_angles.add(angle)
                    action_to_camera_pairs[action_rel].add((level, angle))
                else:
                    bad_camera_name_count += 1
                    action_to_bad_names[action_rel] += 1

            source_images = list_source_images(capture_dir)
            n_src = len(source_images)
            total_source_frames += n_src

            output_dir = result_root / "inference" / rel_capture
            indices, malformed, corrupt = parse_output_indices(output_dir)
            none_file_exists, none_indices, none_invalid_lines = parse_none_detected_indices(output_dir)

            actual_set = set(indices)
            expected_set = set(range(n_src))

            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            none_mismatch = sorted(set(missing).symmetric_difference(set(none_indices)))

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
                    none_file_exists=none_file_exists,
                    none_indices=none_indices,
                    none_file_invalid_lines=none_invalid_lines,
                    none_mismatch_indices=none_mismatch,
                    output_dir_exists=output_dir.exists() and output_dir.is_dir(),
                )
            )

    expected_pairs = {(l, a) for l in EXPECTED_LEVELS for a in EXPECTED_ANGLES}
    actions_with_missing_levels = 0
    actions_with_missing_angles = 0
    actions_with_wrong_camera_count = 0
    actions_with_camera_config_issues = 0

    for action_rel, pairs in action_to_camera_pairs.items():
        levels = {l for l, _ in pairs}
        angles = {a for _, a in pairs}
        bad_names = action_to_bad_names[action_rel]

        missing_levels = EXPECTED_LEVELS - levels
        missing_angles = EXPECTED_ANGLES - angles
        wrong_count = len(pairs) != len(expected_pairs)

        if missing_levels:
            actions_with_missing_levels += 1
        if missing_angles:
            actions_with_missing_angles += 1
        if wrong_count:
            actions_with_wrong_camera_count += 1

        if bad_names > 0 or missing_levels or missing_angles or wrong_count:
            actions_with_camera_config_issues += 1

    camera_config_passed = (
        bad_camera_name_count == 0
        and actions_with_missing_levels == 0
        and actions_with_missing_angles == 0
        and actions_with_wrong_camera_count == 0
    )

    summary = IntegritySummary(
        source_root=str(source_root),
        result_root=str(result_root),
        total_persons=len(person_names),
        unique_action_names=len(action_names),
        total_actions=len(action_dirs),
        camera_level_depth=max(capture_depths) if capture_depths else 0,
        unique_cameras=len(capture_name_set),
        total_cameras=len(checks),
        detected_camera_levels=sorted(detected_levels),
        detected_angles=sorted(detected_angles),
        captures_with_bad_camera_name=bad_camera_name_count,
        actions_with_missing_levels=actions_with_missing_levels,
        actions_with_missing_angles=actions_with_missing_angles,
        actions_with_wrong_camera_count=actions_with_wrong_camera_count,
        actions_with_camera_config_issues=actions_with_camera_config_issues,
        camera_config_passed=camera_config_passed,
        total_captures=len(checks),
        total_source_frames=total_source_frames,
        total_output_npz=total_output_npz,
        captures_missing_output_dir=sum(1 for c in checks if not c.output_dir_exists),
        captures_with_missing_frames=sum(1 for c in checks if len(c.missing_indices) > 0),
        captures_with_extra_frames=sum(1 for c in checks if len(c.extra_indices) > 0),
        captures_with_malformed_outputs=sum(1 for c in checks if len(c.malformed_outputs) > 0),
        captures_with_corrupt_outputs=sum(1 for c in checks if len(c.corrupt_outputs) > 0),
        captures_with_none_file=sum(1 for c in checks if c.none_file_exists),
        captures_with_none_parse_errors=sum(1 for c in checks if len(c.none_file_invalid_lines) > 0),
        captures_with_none_mismatch=sum(1 for c in checks if len(c.none_mismatch_indices) > 0),
        strict_mode=False,
        passed=False,
    )

    return summary, checks


def print_report(summary: IntegritySummary, checks: Sequence[CaptureCheck], show_top: int) -> None:
    print("=== SAM3D-Body Integrity Report ===")
    print(f"source_root: {summary.source_root}")
    print(f"result_root: {summary.result_root}")
    print(f"persons: {summary.total_persons}")
    print(f"unique action names: {summary.unique_action_names}")
    print(f"actions: {summary.total_actions}")
    print(f"camera level depth: {summary.camera_level_depth}")
    print(f"unique cameras: {summary.unique_cameras}")
    print(f"total cameras: {summary.total_cameras}")
    print(f"detected camera levels: {summary.detected_camera_levels}")
    print(
        f"detected angle range: {summary.detected_angles[:1]} ... {summary.detected_angles[-1:] if summary.detected_angles else []} (count={len(summary.detected_angles)})"
    )
    print(f"captures with bad camera name: {summary.captures_with_bad_camera_name}")
    print(f"actions with missing levels: {summary.actions_with_missing_levels}")
    print(f"actions with missing angles: {summary.actions_with_missing_angles}")
    print(f"actions with wrong camera count: {summary.actions_with_wrong_camera_count}")
    print(f"actions with camera config issues: {summary.actions_with_camera_config_issues}")
    print(f"camera config passed (5 levels x 10deg): {summary.camera_config_passed}")
    print(f"captures: {summary.total_captures}")
    print(f"source frames: {summary.total_source_frames}")
    print(f"output npz: {summary.total_output_npz}")
    print(f"missing output dirs: {summary.captures_missing_output_dir}")
    print(f"captures with missing frames: {summary.captures_with_missing_frames}")
    print(f"captures with extra frames: {summary.captures_with_extra_frames}")
    print(f"captures with malformed outputs: {summary.captures_with_malformed_outputs}")
    print(f"captures with corrupt outputs: {summary.captures_with_corrupt_outputs}")
    print(f"captures with none file: {summary.captures_with_none_file}")
    print(f"captures with none parse errors: {summary.captures_with_none_parse_errors}")
    print(f"captures with none mismatch: {summary.captures_with_none_mismatch}")
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
        or c.none_file_invalid_lines
        or c.none_mismatch_indices
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
        if c.none_file_invalid_lines:
            print(f"  none_parse_errors={len(c.none_file_invalid_lines)} (first: {c.none_file_invalid_lines[:5]})")
        if c.none_mismatch_indices:
            print(f"  none_mismatch={len(c.none_mismatch_indices)} (first: {c.none_mismatch_indices[:10]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAM3D-Body inference integrity")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Dataset data root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"SAM3D result root (default: {DEFAULT_RESULT_ROOT})",
    )
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
        or summary.captures_with_none_parse_errors > 0
        or summary.captures_with_none_mismatch > 0
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
