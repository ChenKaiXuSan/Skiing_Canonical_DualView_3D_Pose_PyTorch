#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import statistics
import struct
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RE_FRAME = re.compile(r"^frame_(\d+)\.png$")
RE_KPT2D_PER_FRAME = re.compile(r"^kpt2d_(\d+)\.npy$")
RE_HELPER_JOINT = re.compile(r"(twist|roll|end|helper|ik|pole|weapon|prop|socket)", re.IGNORECASE)


@dataclass
class Issue:
    severity: str  # ERROR / WARN / INFO
    scope: str
    message: str


@dataclass
class ActionNode:
    action_name: str
    action_path: Path
    character_name: str = ""


@dataclass
class CameraReport:
    camera_id: str
    has_intrinsics: bool = False
    has_extrinsics: bool = False
    camera_meta_source: str = ""

    frame_count: int = 0
    frame_indices_min: int | None = None
    frame_indices_max: int | None = None
    frame_index_gaps: int = 0

    has_kpt2d_npy: bool = False
    kpt2d_shape: list[int] | None = None

    has_kpt2d_npz: bool = False
    kpt2d_npz_arrays: dict[str, list[int] | None] = field(default_factory=dict)

    kpt2d_per_frame_count: int = 0
    kpt2d_per_frame_gaps: int = 0
    kpt2d_per_frame_shape_ok: bool | None = None

    frame_vs_expected_delta: int | None = None
    overlay_svg: str = ""
    overlay_all_count: int = 0
    overlay_point_source: str = ""

    issues: list[Issue] = field(default_factory=list)


@dataclass
class ActionReport:
    action_name: str
    action_path: str
    character_name: str = ""

    has_meta_sequence: bool = False
    sequence: dict[str, Any] | None = None

    has_kpt3d_npy: bool = False
    kpt3d_shape: list[int] | None = None

    has_kpt3d_npz: bool = False
    kpt3d_npz_arrays: dict[str, list[int] | None] = field(default_factory=dict)

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
    visuals: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ski dataset, NPZ content, and generate visual reports.")
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
        default=32,
        help="Check shape for first N per-frame kpt2d npy files per camera (0=all)",
    )
    parser.add_argument(
        "--overlay-per-action",
        type=int,
        default=2,
        help="How many camera overlay examples to generate per action",
    )
    parser.add_argument(
        "--overlay-all-frames",
        action="store_true",
        default=True,
        help="Generate overlays for all actions, all cameras, all frames (default: enabled)",
    )
    parser.add_argument(
        "--no-overlay-all-frames",
        dest="overlay_all_frames",
        action="store_false",
        help="Disable full-frame overlay generation and only keep sample overlays",
    )
    parser.add_argument(
        "--viz-conf-threshold",
        type=float,
        default=0.0,
        help="Only draw keypoints with confidence >= threshold (default: 0.0)",
    )
    parser.add_argument(
        "--viz-main-joints",
        type=str,
        default="",
        help="Comma-separated joint indices to visualize, e.g. '0,1,2,5,8'. Empty means visualize all",
    )
    parser.add_argument(
        "--viz-auto-filter-helper-joints",
        action="store_true",
        default=True,
        help="Auto-filter helper/twist/end joints in visualization using meta/joint_names.json (default: enabled)",
    )
    parser.add_argument(
        "--no-viz-auto-filter-helper-joints",
        dest="viz_auto_filter_helper_joints",
        action="store_false",
        help="Disable automatic helper-joint filtering",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def auto_select_main_joint_indices(joint_names: list[str]) -> set[int]:
    selected: set[int] = set()
    for idx, name in enumerate(joint_names):
        if not name:
            continue
        if RE_HELPER_JOINT.search(name):
            continue
        selected.add(idx)
    return selected


def parse_npy_header_from_bytes(raw: bytes) -> tuple[str | None, list[int] | None, int | None]:
    try:
        if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
            return None, None, None
        major = raw[6]
        if major == 1:
            hlen = struct.unpack("<H", raw[8:10])[0]
            header_start = 10
        else:
            hlen = struct.unpack("<I", raw[8:12])[0]
            header_start = 12

        header = raw[header_start:header_start + hlen].decode("latin1")
        m_dtype = re.search(r"'descr':\s*'([^']+)'", header)
        m_shape = re.search(r"'shape':\s*\((.*?)\)", header)
        if not m_dtype or not m_shape:
            return None, None, None

        dtype = m_dtype.group(1)
        shape_raw = m_shape.group(1).strip()
        shape = [int(p.strip()) for p in shape_raw.split(",") if p.strip()] if shape_raw else []
        data_offset = header_start + hlen
        return dtype, shape, data_offset
    except Exception:
        return None, None, None


def parse_npy_shape(path: Path) -> list[int] | None:
    try:
        raw = path.read_bytes()
        _, shape, _ = parse_npy_header_from_bytes(raw)
        return shape
    except Exception:
        return None


def parse_npy_float32_data(path: Path) -> tuple[list[int] | None, list[float] | None]:
    try:
        raw = path.read_bytes()
        return parse_npy_float32_data_from_bytes(raw)
    except Exception:
        return None, None


def parse_npy_float32_data_from_bytes(raw: bytes) -> tuple[list[int] | None, list[float] | None]:
    try:
        dtype, shape, data_offset = parse_npy_header_from_bytes(raw)
        if dtype not in ("<f4", "|f4") or shape is None or data_offset is None:
            return shape, None
        count = 1
        for d in shape:
            count *= d
        data = struct.unpack("<" + "f" * count, raw[data_offset:data_offset + 4 * count])
        return shape, list(data)
    except Exception:
        return None, None


def parse_npz_shapes(path: Path) -> dict[str, list[int] | None]:
    out: dict[str, list[int] | None] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                raw = zf.read(name)
                _, shape, _ = parse_npy_header_from_bytes(raw)
                out[name] = shape
    except Exception:
        return {}
    return out


def parse_npz_float32_arrays(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    out: dict[str, tuple[list[int], list[float]]] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                shape, data = parse_npy_float32_data_from_bytes(zf.read(name))
                if shape is None or data is None:
                    continue
                out[name] = (shape, data)
    except Exception:
        return {}
    return out


def extract_indices(file_paths: list[Path], regex: re.Pattern[str]) -> list[int]:
    out: list[int] = []
    for p in file_paths:
        m = regex.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def count_gaps(indices: list[int]) -> int:
    if len(indices) <= 1:
        return 0
    return sum(1 for prev, cur in zip(indices, indices[1:]) if cur != prev + 1)


def is_action_dir(path: Path) -> bool:
    return path.is_dir() and (path / "frames").exists() and (path / "kpt2d").exists()


def discover_action_nodes(dataset_root: Path) -> list[ActionNode]:
    if not dataset_root.exists():
        return []

    direct = [d for d in dataset_root.iterdir() if is_action_dir(d)]
    if direct:
        return [ActionNode(action_name=d.name, action_path=d) for d in sorted(direct)]

    nested: list[ActionNode] = []
    for maybe_char in sorted(dataset_root.iterdir()):
        if not maybe_char.is_dir() or maybe_char.name in {"cameras", "reports"}:
            continue
        for maybe_action in sorted(maybe_char.iterdir()):
            if is_action_dir(maybe_action):
                nested.append(ActionNode(action_name=maybe_action.name, action_path=maybe_action, character_name=maybe_char.name))
    if nested:
        return nested

    legacy = dataset_root / "actions"
    if legacy.exists():
        cand = [d for d in legacy.iterdir() if is_action_dir(d)]
        if cand:
            return [ActionNode(action_name=d.name, action_path=d) for d in sorted(cand)]

    return []


def resolve_shared_camera_root(dataset_root: Path, node: ActionNode) -> Path:
    if node.character_name:
        return dataset_root / node.character_name / "cameras"
    return dataset_root / "cameras"


def reshape_kpt2d_points(shape: list[int], data: list[float], frame_idx: int | None = None) -> list[tuple[float, float, float]]:
    if len(shape) == 2 and shape[1] == 3:
        pts = []
        for j in range(shape[0]):
            base = j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    if len(shape) == 3 and shape[2] == 3:
        t = shape[0]
        jn = shape[1]
        idx = 0 if frame_idx is None else max(0, min(frame_idx, t - 1))
        start = idx * jn * 3
        pts = []
        for j in range(jn):
            base = start + j * 3
            pts.append((data[base], data[base + 1], data[base + 2]))
        return pts

    return []


def svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_points_to_pixels(points: list[tuple[float, float, float]], width: int, height: int) -> list[tuple[float, float, float]]:
    if not points:
        return points

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_abs = max(max(abs(v) for v in xs), max(abs(v) for v in ys))

    # normalized [0,1] or [-1,1] range
    if max_abs <= 2.0:
        normalized: list[tuple[float, float, float]] = []
        for x, y, c in points:
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and (x < 0.0 or y < 0.0):
                x = (x + 1.0) * 0.5
                y = (y + 1.0) * 0.5
            normalized.append((x * (width - 1), y * (height - 1), c))
        return normalized

    return points


def save_kpt_overlay_svg(
    out_path: Path,
    image_path: Path,
    width: int,
    height: int,
    points: list[tuple[float, float, float]],
    title: str,
    y_flip: bool = False,
    embed_image: bool = True,
    conf_threshold: float = 0.0,
    main_joint_indices: set[int] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if embed_image:
        png_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        href = f"data:image/png;base64,{png_b64}"
    else:
        href = os.path.relpath(image_path, out_path.parent).replace("\\", "/")
    points = normalize_points_to_pixels(points, width, height)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<image href="{href}" x="0" y="0" width="{width}" height="{height}"/>',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="white" stroke-width="1"/>',
        f'<text x="12" y="22" font-size="16" fill="yellow" font-family="Arial">{svg_escape(title)}</text>',
    ]

    for i, (x, y, c) in enumerate(points):
        if c < conf_threshold:
            continue
        if main_joint_indices is not None and i not in main_joint_indices:
            continue
        if y_flip:
            y = (height - 1) - y
        color = "#00ff66" if c > 0 else "#ff3b30"
        r = 2.8 if c > 0 else 2.0
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" fill-opacity="0.9"/>')
        if i % 8 == 0:
            lines.append(f'<text x="{x + 3:.2f}" y="{y - 3:.2f}" font-size="8" fill="white" font-family="Arial">{i}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_local_image_for_overlay(svg_path: Path, src_image_path: Path) -> Path:
    local_image = svg_path.with_suffix(src_image_path.suffix)
    if local_image.exists():
        return local_image

    local_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_image_path, local_image)
    except Exception:
        try:
            os.symlink(src_image_path.resolve(), local_image)
        except Exception:
            shutil.copy2(src_image_path, local_image)
    return local_image


def validate_action(
    node: ActionNode,
    dataset_root: Path,
    sample_per_frame_shape_check: int,
    overlay_dir: Path,
    overlay_per_action: int,
    overlay_all_frames: bool,
    overlay_all_root: Path,
    viz_conf_threshold: float,
    viz_main_joint_indices: set[int] | None,
    viz_auto_filter_helper_joints: bool,
) -> ActionReport:
    action_dir = node.action_path
    report = ActionReport(action_name=node.action_name, action_path=str(action_dir), character_name=node.character_name)

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

    kpt3d_npy = action_dir / "kpt3d" / "kpt3d.npy"
    report.has_kpt3d_npy = kpt3d_npy.exists()
    if report.has_kpt3d_npy:
        report.kpt3d_shape = parse_npy_shape(kpt3d_npy)
        if report.kpt3d_shape is None:
            report.issues.append(Issue("ERROR", "action", "kpt3d.npy 形状读取失败"))
    else:
        report.issues.append(Issue("WARN", "action", "缺少 kpt3d/kpt3d.npy"))

    kpt3d_npz = action_dir / "kpt3d" / "kpt3d.npz"
    report.has_kpt3d_npz = kpt3d_npz.exists()
    if report.has_kpt3d_npz:
        report.kpt3d_npz_arrays = parse_npz_shapes(kpt3d_npz)
        if not report.kpt3d_npz_arrays:
            report.issues.append(Issue("ERROR", "action", "kpt3d.npz 存在但无法解析"))

    camera_ids: set[str] = set()
    frames_root = action_dir / "frames"
    kpt2d_root = action_dir / "kpt2d"
    action_cam_root = action_dir / "cameras"

    if action_cam_root.exists():
        camera_ids.update(d.name for d in action_cam_root.iterdir() if d.is_dir())
    if frames_root.exists():
        for d in frames_root.iterdir():
            if d.is_dir() and d.name.startswith("capture_"):
                camera_ids.add(d.name.replace("capture_", "", 1))
    if kpt2d_root.exists():
        camera_ids.update(d.name for d in kpt2d_root.iterdir() if d.is_dir())

    if not camera_ids:
        report.issues.append(Issue("ERROR", "action", "未发现任何相机目录（cameras/frames/kpt2d）"))
        return report

    expected_sampled = int(report.sequence.get("sampled_frames")) if report.sequence and report.sequence.get("sampled_frames") is not None else None
    expected_joints = int(report.sequence.get("joints_count")) if report.sequence and report.sequence.get("joints_count") is not None else None
    image_w = int(report.sequence.get("width", 1920)) if report.sequence else 1920
    image_h = int(report.sequence.get("height", 1080)) if report.sequence else 1080
    joint_names_path = action_dir / "meta" / "joint_names.json"
    joint_names_json = read_json(joint_names_path) if joint_names_path.exists() else None
    auto_main_joint_indices: set[int] | None = None
    if joint_names_json and isinstance(joint_names_json.get("joint_names"), list):
        names = [str(x) for x in joint_names_json.get("joint_names")]
        auto_main_joint_indices = auto_select_main_joint_indices(names)

    shared_cam_root = resolve_shared_camera_root(dataset_root, node)
    overlays_made = 0

    for cam_id in sorted(camera_ids):
        cam = CameraReport(camera_id=cam_id)

        action_intr = action_cam_root / cam_id / "intrinsics.json"
        action_extr = action_cam_root / cam_id / "extrinsics.json"
        shared_intr = shared_cam_root / cam_id / "intrinsics.json"
        shared_extr = shared_cam_root / cam_id / "extrinsics.json"

        if action_intr.exists() or action_extr.exists():
            intr, extr = action_intr, action_extr
            cam.camera_meta_source = "action"
        else:
            intr, extr = shared_intr, shared_extr
            cam.camera_meta_source = "shared"

        cam.has_intrinsics = intr.exists()
        cam.has_extrinsics = extr.exists()
        if not cam.has_intrinsics:
            cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 intrinsics.json（action/shared cameras）"))
        if not cam.has_extrinsics:
            cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 extrinsics.json（action/shared cameras）"))

        frame_dir = frames_root / f"capture_{cam_id}"
        frame_files = sorted(frame_dir.glob("frame_*.png")) if frame_dir.exists() else []
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

        kpt2d_dir = kpt2d_root / cam_id
        kpt2d_main = kpt2d_dir / "kpt2d.npy"
        kpt2d_npz = kpt2d_dir / "kpt2d.npz"
        per_frame_files = sorted(kpt2d_dir.glob("kpt2d_*.npy")) if kpt2d_dir.exists() else []

        cam.has_kpt2d_npy = kpt2d_main.exists()
        if cam.has_kpt2d_npy:
            cam.kpt2d_shape = parse_npy_shape(kpt2d_main)
            if cam.kpt2d_shape is None:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "kpt2d.npy shape 读取失败"))
        else:
            if per_frame_files:
                cam.issues.append(Issue("WARN", f"camera:{cam_id}", "缺少 kpt2d.npy（但存在 kpt2d_*.npy）"))
            else:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "缺少 kpt2d.npy 与 kpt2d_*.npy"))

        cam.has_kpt2d_npz = kpt2d_npz.exists()
        npz_kpt_shape: list[int] | None = None
        npz_kpt_data: list[float] | None = None
        if cam.has_kpt2d_npz:
            cam.kpt2d_npz_arrays = parse_npz_shapes(kpt2d_npz)
            if not cam.kpt2d_npz_arrays:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", "kpt2d.npz 存在但无法解析"))
            else:
                npz_float_arrays = parse_npz_float32_arrays(kpt2d_npz)
                preferred_names = ["kpt2d.npy", "kpt2d", "arr_0.npy", "arr_0"]
                ordered_names = [n for n in preferred_names if n in npz_float_arrays] + [n for n in npz_float_arrays.keys() if n not in preferred_names]
                for name in ordered_names:
                    shape, data = npz_float_arrays[name]
                    if (len(shape) == 3 and shape[2] == 3) or (len(shape) == 2 and shape[1] == 3):
                        npz_kpt_shape, npz_kpt_data = shape, data
                        break

        per_idx = extract_indices(per_frame_files, RE_KPT2D_PER_FRAME)
        cam.kpt2d_per_frame_count = len(per_frame_files)
        cam.kpt2d_per_frame_gaps = count_gaps(per_idx)
        if per_frame_files and cam.kpt2d_per_frame_gaps > 0:
            cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"kpt2d_*.npy 序号存在 {cam.kpt2d_per_frame_gaps} 处间断"))

        if per_frame_files:
            check_files = per_frame_files if sample_per_frame_shape_check <= 0 else per_frame_files[:sample_per_frame_shape_check]
            shape_ok = True
            for p in check_files:
                shp = parse_npy_shape(p)
                if shp is None or len(shp) != 2 or shp[1] != 3:
                    shape_ok = False
                    cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"{p.name} 形状异常，期望 (J,3)"))
                    break
            cam.kpt2d_per_frame_shape_ok = shape_ok

        if expected_sampled is not None:
            cam.frame_vs_expected_delta = cam.frame_count - expected_sampled
            if cam.frame_count != expected_sampled:
                cam.issues.append(Issue("WARN", f"camera:{cam_id}", f"frame 数({cam.frame_count}) != sampled_frames({expected_sampled})"))

        if cam.kpt2d_shape and len(cam.kpt2d_shape) == 3 and expected_joints is not None:
            if cam.kpt2d_shape[1] != expected_joints or cam.kpt2d_shape[2] != 3:
                cam.issues.append(Issue("ERROR", f"camera:{cam_id}", f"kpt2d.npy shape={tuple(cam.kpt2d_shape)} 与 joints_count={expected_joints} 不一致"))

        main_kpt_shape: list[int] | None = None
        main_kpt_data: list[float] | None = None
        if kpt2d_main.exists():
            main_kpt_shape, main_kpt_data = parse_npy_float32_data(kpt2d_main)

        def get_points_for_frame(frame_index: int) -> list[tuple[float, float, float]]:
            if npz_kpt_shape and npz_kpt_data is not None:
                cam.overlay_point_source = "npz"
                return reshape_kpt2d_points(npz_kpt_shape, npz_kpt_data, frame_index)

            per_frame_kpt = kpt2d_dir / f"kpt2d_{frame_index:06d}.npy"
            if per_frame_kpt.exists():
                shp, data = parse_npy_float32_data(per_frame_kpt)
                if shp and data is not None:
                    cam.overlay_point_source = "npy-per-frame"
                    return reshape_kpt2d_points(shp, data)

            if main_kpt_shape and main_kpt_data is not None:
                cam.overlay_point_source = "npy-main"
                return reshape_kpt2d_points(main_kpt_shape, main_kpt_data, frame_index)

            return []

        if overlay_all_frames and frame_files:
            for frame_path in frame_files:
                m = RE_FRAME.match(frame_path.name)
                if not m:
                    continue
                target_idx = int(m.group(1))
                points = get_points_for_frame(target_idx)
                if not points:
                    continue

                rel = frame_path.relative_to(dataset_root)
                overlay_path = (overlay_all_root / rel).with_suffix(".svg")
                title = f"{report.action_name} | {cam_id} | {frame_path.name}"
                draw_indices = viz_main_joint_indices
                if draw_indices is None and auto_main_joint_indices is not None and viz_auto_filter_helper_joints:
                    draw_indices = auto_main_joint_indices
                save_kpt_overlay_svg(
                    overlay_path,
                    frame_path,
                    image_w,
                    image_h,
                    points,
                    title,
                    y_flip=False,
                    embed_image=True,
                    conf_threshold=viz_conf_threshold,
                    main_joint_indices=draw_indices,
                )
                cam.overlay_all_count += 1
                if not cam.overlay_svg:
                    cam.overlay_svg = str(overlay_path)

        # overlay examples: generate for first N cameras per action
        if not overlay_all_frames and overlays_made < overlay_per_action and frame_indices:
            target_idx = frame_indices[len(frame_indices) // 2]
            frame_path = frame_dir / f"frame_{target_idx:06d}.png"
            points = get_points_for_frame(target_idx)

            if points and frame_path.exists():
                safe_action = re.sub(r"[^A-Za-z0-9_.-]+", "_", report.action_name)
                overlay_path = overlay_dir / f"overlay_{safe_action}_{cam_id}_{target_idx:06d}.svg"
                title = f"{report.action_name} | {cam_id} | frame_{target_idx:06d}"
                draw_indices = viz_main_joint_indices
                if draw_indices is None and auto_main_joint_indices is not None and viz_auto_filter_helper_joints:
                    draw_indices = auto_main_joint_indices
                save_kpt_overlay_svg(
                    overlay_path,
                    frame_path,
                    image_w,
                    image_h,
                    points,
                    title,
                    y_flip=False,
                    conf_threshold=viz_conf_threshold,
                    main_joint_indices=draw_indices,
                )
                cam.overlay_svg = str(overlay_path)
                overlays_made += 1

        report.camera_reports.append(cam)

    if report.kpt3d_shape and report.sequence:
        sampled = report.sequence.get("sampled_frames")
        joints = report.sequence.get("joints_count")
        shp = report.kpt3d_shape
        if len(shp) != 3 or shp[2] != 3:
            report.issues.append(Issue("ERROR", "action", f"kpt3d.npy shape={tuple(shp)} 非法，期望 (T,J,3)"))
        else:
            if sampled is not None and shp[0] != int(sampled):
                report.issues.append(Issue("WARN", "action", f"kpt3d T={shp[0]} != sampled_frames={sampled}"))
            if joints is not None and shp[1] != int(joints):
                report.issues.append(Issue("WARN", "action", f"kpt3d J={shp[1]} != joints_count={joints}"))

    return report


def save_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width = max(1000, 80 + 26 * max(1, len(labels)))
    height = 520
    left, right, top, bottom = 70, 30, 60, 150
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    bars = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + i * (plot_w / max(1, len(labels))) + 2
        bw = max(4, (plot_w / max(1, len(labels))) - 4)
        bh = (val / max_v) * plot_h
        y = top + (plot_h - bh)
        bars.append((x, y, bw, bh, lab, val))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{svg_escape(title)}</text>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20,{top+plot_h/2})" text-anchor="middle" font-size="12" font-family="Arial">{svg_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]

    for t in range(6):
        v = max_v * t / 5
        y = top + plot_h - (plot_h * t / 5)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#eee"/>')
        lines.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial">{int(v)}</text>')

    for x, y, bw, bh, lab, val in bars:
        lines.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="#4e79a7"/>')
        lines.append(f'<text x="{x+bw/2}" y="{y-4}" text-anchor="middle" font-size="10" font-family="Arial">{int(val)}</text>')
        lines.append(
            f'<text x="{x+bw/2}" y="{top+plot_h+14}" text-anchor="end" transform="rotate(-65 {x+bw/2},{top+plot_h+14})" '
            f'font-size="10" font-family="Arial">{svg_escape(lab)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_svg_issue_summary(path: Path, errors: int, warns: int, infos: int) -> None:
    labels = ["ERROR", "WARN", "INFO"]
    values = [errors, warns, infos]
    colors = ["#e15759", "#f28e2b", "#59a14f"]

    width, height = 640, 420
    left, top = 80, 70
    plot_w, plot_h = 500, 260
    max_v = max(values) if max(values) > 0 else 1
    bw = 100
    gap = (plot_w - bw * 3) / 4

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="320" y="30" text-anchor="middle" font-size="20" font-family="Arial">Issue Severity Summary</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]

    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x = left + gap + i * (bw + gap)
        bh = 0 if max_v == 0 else (val / max_v) * plot_h
        y = top + (plot_h - bh)
        lines.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{col}"/>')
        lines.append(f'<text x="{x+bw/2}" y="{y-6}" text-anchor="middle" font-size="12" font-family="Arial">{val}</text>')
        lines.append(f'<text x="{x+bw/2}" y="{top+plot_h+24}" text-anchor="middle" font-size="12" font-family="Arial">{lab}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_visuals(report: FullReport, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"
    overlay_dir = fig_dir / "overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    sev_path = fig_dir / "issues_summary.svg"
    save_svg_issue_summary(sev_path, report.summary.error_count, report.summary.warn_count, report.summary.info_count)
    generated.append(str(sev_path))

    for action in report.actions:
        cams = sorted(action.camera_reports, key=lambda c: c.camera_id)
        if cams:
            labels = [c.camera_id for c in cams]
            frame_vals = [float(c.frame_count) for c in cams]
            per_vals = [float(c.kpt2d_per_frame_count) for c in cams]
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.action_name)

            p1 = fig_dir / f"{safe_name}_frames_per_camera.svg"
            save_svg_bar_chart(p1, f"{action.action_name} - Frames per Camera", labels, frame_vals, "#frames")
            generated.append(str(p1))

            p2 = fig_dir / f"{safe_name}_kpt2d_per_frame_per_camera.svg"
            save_svg_bar_chart(p2, f"{action.action_name} - kpt2d per-frame count", labels, per_vals, "#kpt2d files")
            generated.append(str(p2))

        for cam in cams:
            if cam.overlay_svg:
                generated.append(cam.overlay_svg)

    return generated


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

    if report.visuals:
        lines.append("## Visualizations")
        lines.append("")
        for v in report.visuals:
            lines.append(f"- {v}")
        lines.append("")

    for action in report.actions:
        lines.append(f"## Action: {action.action_name}")
        lines.append("")
        lines.append(f"- Character: {action.character_name or '(none)'}")
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

        lines.append(f"- kpt3d.npy: {'Y' if action.has_kpt3d_npy else 'N'} shape={tuple(action.kpt3d_shape) if action.kpt3d_shape else '-'}")
        lines.append(f"- kpt3d.npz: {'Y' if action.has_kpt3d_npz else 'N'} arrays={list(action.kpt3d_npz_arrays.keys()) if action.kpt3d_npz_arrays else []}")

        frame_counts = [c.frame_count for c in action.camera_reports]
        if frame_counts:
            lines.append(f"- Frame stats: min={min(frame_counts)}, max={max(frame_counts)}, mean={statistics.mean(frame_counts):.2f}")

        lines.append("")
        lines.append("### Cameras")
        lines.append("")
        lines.append("| Camera | MetaSrc | Intr | Extr | Frames | FrameRange | Gap# | kpt2d.npy | kpt2d.npz | kpt2d per-frame | Overlay(all) | Overlay src | Overlay(sample) |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|")
        for cam in sorted(action.camera_reports, key=lambda c: c.camera_id):
            fr = "-"
            if cam.frame_indices_min is not None and cam.frame_indices_max is not None:
                fr = f"{cam.frame_indices_min}..{cam.frame_indices_max}"
            ov = cam.overlay_svg if cam.overlay_svg else "-"
            lines.append(
                f"| {cam.camera_id} | {cam.camera_meta_source or '-'} | {'Y' if cam.has_intrinsics else 'N'} | {'Y' if cam.has_extrinsics else 'N'} | "
                f"{cam.frame_count} | {fr} | {cam.frame_index_gaps} | {'Y' if cam.has_kpt2d_npy else 'N'} | {'Y' if cam.has_kpt2d_npz else 'N'} | {cam.kpt2d_per_frame_count} | {cam.overlay_all_count} | {cam.overlay_point_source or '-'} | {ov} |"
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
    viz_main_joint_indices: set[int] | None = None
    if args.viz_main_joints.strip():
        try:
            viz_main_joint_indices = {int(part.strip()) for part in args.viz_main_joints.split(",") if part.strip()}
        except ValueError:
            print("[check.py] ERROR: --viz-main-joints 格式非法，应为逗号分隔整数，如 0,1,2,5")
            return 2

    summary = Summary(checked_at_utc=datetime.utcnow().isoformat() + "Z", dataset_root=str(dataset_root))
    nodes = discover_action_nodes(dataset_root)

    if not nodes:
        summary.passed = False
        summary.error_count = 1
        report = FullReport(
            summary=summary,
            actions=[ActionReport(action_name="(none)", action_path=str(dataset_root), issues=[Issue("ERROR", "dataset", "未发现动作目录")])],
        )
    else:
        overlay_dir = output_dir / "figures" / "overlays"
        overlay_all_root = output_dir / "figures" / "overlays_all"
        actions = [
            validate_action(
                n,
                dataset_root,
                args.sample_per_frame_shape_check,
                overlay_dir,
                args.overlay_per_action,
                args.overlay_all_frames,
                overlay_all_root,
                args.viz_conf_threshold,
                viz_main_joint_indices,
                args.viz_auto_filter_helper_joints,
            )
            for n in nodes
        ]
        summary.action_count = len(actions)
        summary.camera_count = sum(len(a.camera_reports) for a in actions)

        all_issues = [i for a in actions for i in (a.issues + [ii for c in a.camera_reports for ii in c.issues])]
        summary.error_count = sum(1 for i in all_issues if i.severity == "ERROR")
        summary.warn_count = sum(1 for i in all_issues if i.severity == "WARN")
        summary.info_count = sum(1 for i in all_issues if i.severity == "INFO")
        summary.passed = summary.error_count == 0

        report = FullReport(summary=summary, actions=actions)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report.visuals = generate_visuals(report, output_dir)

    json_path = output_dir / f"dataset_check_report_{stamp}.json"
    md_path = output_dir / f"dataset_check_report_{stamp}.md"

    json_path.write_text(
        json.dumps({"summary": asdict(report.summary), "visuals": report.visuals, "actions": [asdict(a) for a in report.actions]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"[check.py] dataset_root = {dataset_root}")
    print(f"[check.py] actions = {summary.action_count}, cameras = {summary.camera_count}")
    print(f"[check.py] errors = {summary.error_count}, warnings = {summary.warn_count}, passed = {summary.passed}")
    print(f"[check.py] report(json) = {json_path}")
    print(f"[check.py] report(md)   = {md_path}")
    print(f"[check.py] visuals = {len(report.visuals)} files in {output_dir / 'figures'}")

    return 0 if summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
