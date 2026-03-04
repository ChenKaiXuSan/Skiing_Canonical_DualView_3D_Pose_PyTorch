#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


RE_FRAME = re.compile(r"^frame_(\d+)\.png$")
RE_KPT2D_PER_FRAME = re.compile(r"^kpt2d_(\d+)\.npy$")


@dataclass
class Issue:
	severity: str  # ERROR / WARN / INFO
	scope: str
	message: str


@dataclass
class CameraReport:
	camera_id: str
	has_intrinsics: bool = False
	has_extrinsics: bool = False
	frame_count: int = 0
	frame_indices_min: int | None = None
	frame_indices_max: int | None = None
	frame_index_gaps: int = 0
	has_kpt2d_npy: bool = False
	kpt2d_shape: list[int] | None = None
	kpt2d_per_frame_count: int = 0
	kpt2d_per_frame_gaps: int = 0
	kpt2d_per_frame_shape_ok: bool | None = None
	issues: list[Issue] = field(default_factory=list)


@dataclass
class ActionReport:
	action_name: str
	action_path: str
	has_meta_sequence: bool = False
	sequence: dict[str, Any] | None = None
	has_kpt3d_npy: bool = False
	kpt3d_shape: list[int] | None = None
	camera_reports: list[CameraReport] = field(default_factory=list)
	issues: list[Issue] = field(default_factory=list)


@dataclass
class Summary:
	checked_at_utc: str
	dataset_root: str
	action_count: int = 0
	camera_count: int = 0
	error_count: int = 0
	warn_count: int = 0
	info_count: int = 0
	passed: bool = True


@dataclass
class FullReport:
	summary: Summary
	actions: list[ActionReport]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Validate exported ski dataset and generate detailed report.")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=(Path(__file__).resolve().parents[1] / "SkiDataset"),
		help="Dataset root path (default: ../SkiDataset)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=(Path(__file__).resolve().parent / "reports"),
		help="Directory to write report files",
	)
	parser.add_argument(
		"--sample-per-frame-shape-check",
		type=int,
		default=0,
		help="Check shape for first N per-frame kpt2d npy files per camera (0 = check all)",
	)
	return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None


def parse_npy_shape(path: Path) -> list[int] | None:
	try:
		with path.open("rb") as f:
			magic = f.read(6)
			if magic != b"\x93NUMPY":
				return None
			major = f.read(1)
			minor = f.read(1)
			if not major or not minor:
				return None
			major_v = major[0]
			if major_v == 1:
				header_len = struct.unpack("<H", f.read(2))[0]
			else:
				header_len = struct.unpack("<I", f.read(4))[0]
			header = f.read(header_len).decode("latin1")
			m = re.search(r"'shape':\s*\((.*?)\)", header)
			if not m:
				return None
			raw = m.group(1).strip()
			if not raw:
				return []
			parts = [p.strip() for p in raw.split(",") if p.strip()]
			shape = [int(p) for p in parts]
			return shape
	except Exception:
		return None


def extract_indices(file_paths: list[Path], regex: re.Pattern[str]) -> list[int]:
	indices: list[int] = []
	for p in file_paths:
		m = regex.match(p.name)
		if m:
			indices.append(int(m.group(1)))
	return sorted(indices)


def count_gaps(indices: list[int]) -> int:
	if not indices:
		return 0
	gap = 0
	prev = indices[0]
	for cur in indices[1:]:
		if cur != prev + 1:
			gap += 1
		prev = cur
	return gap


def discover_action_dirs(dataset_root: Path) -> list[Path]:
	if not dataset_root.exists():
		return []

	# New layout: SkiDataset/<ActionName>/...
	candidates = [
		d
		for d in dataset_root.iterdir()
		if d.is_dir() and (d / "frames").exists() and (d / "kpt2d").exists()
	]
	if candidates:
		return sorted(candidates)

	# Legacy layout fallback: SkiDataset/actions/<ActionName>/...
	actions_root = dataset_root / "actions"
	if actions_root.exists() and actions_root.is_dir():
		return sorted([d for d in actions_root.iterdir() if d.is_dir()])

	return []


def validate_action(action_dir: Path, sample_per_frame_shape_check: int) -> ActionReport:
	report = ActionReport(action_name=action_dir.name, action_path=str(action_dir))

	meta_seq = action_dir / "meta" / "sequence.json"
	if meta_seq.exists():
		seq = read_json(meta_seq)
		if seq is None:
			report.issues.append(Issue("ERROR", "action", "meta/sequence.json 无法解析为合法 JSON"))
		else:
			report.has_meta_sequence = True
			report.sequence = seq
	else:
		report.issues.append(Issue("ERROR", "action", "缺少 meta/sequence.json"))

	kpt3d = action_dir / "kpt3d" / "kpt3d.npy"
	report.has_kpt3d_npy = kpt3d.exists()
	if report.has_kpt3d_npy:
		report.kpt3d_shape = parse_npy_shape(kpt3d)
		if report.kpt3d_shape is None:
			report.issues.append(Issue("ERROR", "action", "kpt3d/kpt3d.npy 不是合法 npy 或 shape 读取失败"))
	else:
		report.issues.append(Issue("WARN", "action", "缺少 kpt3d/kpt3d.npy"))

	cameras_root = action_dir / "cameras"
	frames_root = action_dir / "frames"
	kpt2d_root = action_dir / "kpt2d"

	camera_ids = set()
	if cameras_root.exists():
		camera_ids.update([d.name for d in cameras_root.iterdir() if d.is_dir()])
	if frames_root.exists():
		for d in frames_root.iterdir():
			if not d.is_dir():
				continue
			if d.name.startswith("capture_"):
				camera_ids.add(d.name.replace("capture_", "", 1))
	if kpt2d_root.exists():
		camera_ids.update([d.name for d in kpt2d_root.iterdir() if d.is_dir()])

	if not camera_ids:
		report.issues.append(Issue("ERROR", "action", "未发现任何相机目录（cameras/frames/kpt2d）"))
		return report

	expected_sampled = None
	expected_joints = None
	if report.sequence:
		expected_sampled = report.sequence.get("sampled_frames")
		expected_joints = report.sequence.get("joints_count")

	for cam_id in sorted(camera_ids):
		cam = CameraReport(camera_id=cam_id)

		intr = action_dir / "cameras" / cam_id / "intrinsics.json"
		extr = action_dir / "cameras" / cam_id / "extrinsics.json"
		cam.has_intrinsics = intr.exists()
		cam.has_extrinsics = extr.exists()
		if not cam.has_intrinsics:
			cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 cameras/<cam>/intrinsics.json"))
		if not cam.has_extrinsics:
			cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 cameras/<cam>/extrinsics.json"))

		frame_dir = action_dir / "frames" / f"capture_{cam_id}"
		frame_files = sorted([p for p in frame_dir.glob("frame_*.png")]) if frame_dir.exists() else []
		frame_indices = extract_indices(frame_files, RE_FRAME)
		cam.frame_count = len(frame_files)
		if frame_indices:
			cam.frame_indices_min = frame_indices[0]
			cam.frame_indices_max = frame_indices[-1]
			cam.frame_index_gaps = count_gaps(frame_indices)
			if cam.frame_index_gaps > 0:
				cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"frame 序号存在 {cam.frame_index_gaps} 处间断"))
		else:
			cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "未找到 frame_*.png"))

		kpt2d_dir = action_dir / "kpt2d" / cam_id
		kpt2d_main = kpt2d_dir / "kpt2d.npy"
		cam.has_kpt2d_npy = kpt2d_main.exists()
		if cam.has_kpt2d_npy:
			cam.kpt2d_shape = parse_npy_shape(kpt2d_main)
			if cam.kpt2d_shape is None:
				cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "kpt2d.npy shape 读取失败"))
		else:
			cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 kpt2d/<cam>/kpt2d.npy"))

		per_frame_files = sorted([p for p in kpt2d_dir.glob("kpt2d_*.npy")]) if kpt2d_dir.exists() else []
		per_idx = extract_indices(per_frame_files, RE_KPT2D_PER_FRAME)
		cam.kpt2d_per_frame_count = len(per_frame_files)
		cam.kpt2d_per_frame_gaps = count_gaps(per_idx)
		if per_frame_files and cam.kpt2d_per_frame_gaps > 0:
			cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"kpt2d_*.npy 序号存在 {cam.kpt2d_per_frame_gaps} 处间断"))

		if per_frame_files:
			check_files = per_frame_files
			if sample_per_frame_shape_check > 0:
				check_files = per_frame_files[:sample_per_frame_shape_check]

			shape_ok = True
			for p in check_files:
				shp = parse_npy_shape(p)
				if shp is None or len(shp) != 2 or shp[1] != 3:
					shape_ok = False
					cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"{p.name} 形状异常，期望 (J,3)"))
					break
			cam.kpt2d_per_frame_shape_ok = shape_ok

		if expected_sampled is not None:
			if cam.frame_count != expected_sampled:
				cam.issues.append(
					Issue("WARN", f"camera:{cam_id}", f"frame 数({cam.frame_count}) != sequence.sampled_frames({expected_sampled})")
				)
			if cam.kpt2d_per_frame_count not in (0, expected_sampled):
				cam.issues.append(
					Issue(
						"WARN",
						f"camera:{cam_id}",
						f"kpt2d per-frame 数({cam.kpt2d_per_frame_count}) != sequence.sampled_frames({expected_sampled})",
					)
				)

		if cam.kpt2d_shape and len(cam.kpt2d_shape) == 3 and expected_joints is not None:
			if cam.kpt2d_shape[1] != expected_joints or cam.kpt2d_shape[2] != 3:
				cam.issues.append(
					Issue(
						"ERROR",
						f"camera:{cam_id}",
						f"kpt2d.npy shape={tuple(cam.kpt2d_shape)} 与 sequence.joints_count={expected_joints} 不一致",
					)
				)

		report.camera_reports.append(cam)

	if report.kpt3d_shape and report.sequence:
		sampled = report.sequence.get("sampled_frames")
		joints = report.sequence.get("joints_count")
		shp = report.kpt3d_shape
		if len(shp) != 3 or shp[2] != 3:
			report.issues.append(Issue("ERROR", "action", f"kpt3d.npy shape={tuple(shp)} 非法，期望 (T,J,3)"))
		else:
			if sampled is not None and shp[0] != sampled:
				report.issues.append(Issue("WARN", "action", f"kpt3d T={shp[0]} != sampled_frames={sampled}"))
			if joints is not None and shp[1] != joints:
				report.issues.append(Issue("WARN", "action", f"kpt3d J={shp[1]} != joints_count={joints}"))

	return report


def to_markdown(report: FullReport) -> str:
	s = report.summary
	lines: list[str] = []
	lines.append("# Dataset Validation Report")
	lines.append("")
	lines.append(f"- Checked at (UTC): {s.checked_at_utc}")
	lines.append(f"- Dataset root: {s.dataset_root}")
	lines.append(f"- Actions: {s.action_count}")
	lines.append(f"- Cameras: {s.camera_count}")
	lines.append(f"- Errors: {s.error_count}")
	lines.append(f"- Warnings: {s.warn_count}")
	lines.append(f"- Infos: {s.info_count}")
	lines.append(f"- Passed: {s.passed}")
	lines.append("")

	for action in report.actions:
		lines.append(f"## Action: {action.action_name}")
		lines.append("")
		lines.append(f"- Path: {action.action_path}")
		if action.sequence:
			lines.append(
				"- Sequence: "
				f"total_frames={action.sequence.get('total_frames')}, "
				f"sampled_frames={action.sequence.get('sampled_frames')}, "
				f"joints_count={action.sequence.get('joints_count')}, "
				f"pose_every_n_frames={action.sequence.get('pose_every_n_frames')}"
			)
		else:
			lines.append("- Sequence: (missing)")

		if action.kpt3d_shape is not None:
			lines.append(f"- kpt3d shape: {tuple(action.kpt3d_shape)}")
		else:
			lines.append("- kpt3d shape: (missing or unreadable)")

		lines.append("")
		lines.append("### Cameras")
		lines.append("")
		lines.append("| Camera | Intr | Extr | Frames | FrameRange | kpt2d.npy | kpt2d shape | kpt2d per-frame |")
		lines.append("|---|---:|---:|---:|---|---:|---|---:|")
		for cam in action.camera_reports:
			fr = "-"
			if cam.frame_indices_min is not None and cam.frame_indices_max is not None:
				fr = f"{cam.frame_indices_min}..{cam.frame_indices_max}"
			kshape = "-" if cam.kpt2d_shape is None else str(tuple(cam.kpt2d_shape))
			lines.append(
				f"| {cam.camera_id} | {'Y' if cam.has_intrinsics else 'N'} | {'Y' if cam.has_extrinsics else 'N'} | "
				f"{cam.frame_count} | {fr} | {'Y' if cam.has_kpt2d_npy else 'N'} | {kshape} | {cam.kpt2d_per_frame_count} |"
			)

		all_issues = action.issues + [i for c in action.camera_reports for i in c.issues]
		lines.append("")
		lines.append("### Issues")
		lines.append("")
		if not all_issues:
			lines.append("- None")
		else:
			for issue in all_issues:
				lines.append(f"- [{issue.severity}] ({issue.scope}) {issue.message}")
		lines.append("")

	return "\n".join(lines)


def main() -> int:
	args = parse_args()
	dataset_root = args.dataset_root.resolve()
	output_dir = args.output_dir.resolve()

	now = datetime.utcnow().isoformat() + "Z"
	summary = Summary(checked_at_utc=now, dataset_root=str(dataset_root))

	actions = discover_action_dirs(dataset_root)
	action_reports: list[ActionReport] = []

	if not actions:
		summary.passed = False
		summary.error_count = 1
		report = FullReport(
			summary=summary,
			actions=[
				ActionReport(
					action_name="(none)",
					action_path=str(dataset_root),
					issues=[Issue("ERROR", "dataset", "未发现动作目录。请检查 dataset_root 是否正确。")],
				)
			],
		)
	else:
		for action_dir in actions:
			action_reports.append(validate_action(action_dir, args.sample_per_frame_shape_check))

		summary.action_count = len(action_reports)
		summary.camera_count = sum(len(a.camera_reports) for a in action_reports)

		all_issues = [i for a in action_reports for i in (a.issues + [ii for c in a.camera_reports for ii in c.issues])]
		summary.error_count = sum(1 for i in all_issues if i.severity == "ERROR")
		summary.warn_count = sum(1 for i in all_issues if i.severity == "WARN")
		summary.info_count = sum(1 for i in all_issues if i.severity == "INFO")
		summary.passed = summary.error_count == 0

		report = FullReport(summary=summary, actions=action_reports)

	output_dir.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	json_path = output_dir / f"dataset_check_report_{stamp}.json"
	md_path = output_dir / f"dataset_check_report_{stamp}.md"

	json_path.write_text(
		json.dumps(
			{
				"summary": asdict(report.summary),
				"actions": [asdict(a) for a in report.actions],
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)
	md_path.write_text(to_markdown(report), encoding="utf-8")

	print(f"[check.py] dataset_root = {dataset_root}")
	print(f"[check.py] actions = {summary.action_count}, cameras = {summary.camera_count}")
	print(f"[check.py] errors = {summary.error_count}, warnings = {summary.warn_count}, passed = {summary.passed}")
	print(f"[check.py] report(json) = {json_path}")
	print(f"[check.py] report(md)   = {md_path}")

	return 0 if summary.passed else 2


if __name__ == "__main__":
	raise SystemExit(main())

