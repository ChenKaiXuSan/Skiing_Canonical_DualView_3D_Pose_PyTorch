#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_numpy_pose(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected tensor for {name}, got {type(x)}")
    arr = x.detach().cpu().numpy()
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected {name} shape (N,T,J,3), got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _flatten_frames(x: np.ndarray) -> np.ndarray:
    n, t, j, c = x.shape
    return x.reshape(n * t, j, c)


def mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def n_mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    denom = np.sum(pred * pred, axis=(1, 2))
    numer = np.sum(pred * gt, axis=(1, 2))
    scale = np.where(denom > 1e-12, numer / denom, 1.0)
    pred_scaled = pred * scale[:, None, None]
    return mpjpe_per_frame(pred_scaled, gt)


def _procrustes_align_one(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # pred, gt: (J,3), align pred to gt with similarity transform.
    mu_pred = pred.mean(axis=0, keepdims=True)
    mu_gt = gt.mean(axis=0, keepdims=True)

    x0 = pred - mu_pred
    y0 = gt - mu_gt

    var_x = np.sum(x0 * x0)
    if var_x <= 1e-12:
        return pred

    k = x0.T @ y0
    u, s, vt = np.linalg.svd(k)
    v = vt.T
    r = v @ u.T

    if np.linalg.det(r) < 0:
        v[:, -1] *= -1.0
        s[-1] *= -1.0
        r = v @ u.T

    scale = float(np.sum(s) / var_x)
    t = mu_gt - scale * (mu_pred @ r)
    return scale * (pred @ r) + t


def p_mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(pred)
    for i in range(pred.shape[0]):
        aligned[i] = _procrustes_align_one(pred[i], gt[i])
    return mpjpe_per_frame(aligned, gt)


def _as_scalar(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return v.item()
        return v.tolist()
    return v


def _expand_meta_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(entry, dict) or len(entry) == 0:
        return []

    candidate_len = 1
    for v in entry.values():
        if isinstance(v, (list, tuple)):
            candidate_len = max(candidate_len, len(v))
        elif isinstance(v, (torch.Tensor, np.ndarray)) and np.asarray(v).ndim > 0:
            candidate_len = max(candidate_len, int(np.asarray(v).shape[0]))

    out: List[Dict[str, Any]] = []
    for i in range(candidate_len):
        row: Dict[str, Any] = {}
        for k, v in entry.items():
            if isinstance(v, (list, tuple)):
                row[k] = _as_scalar(v[i]) if i < len(v) else None
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                arr = np.asarray(v)
                if arr.ndim == 0:
                    row[k] = _as_scalar(arr)
                elif i < arr.shape[0]:
                    row[k] = _as_scalar(arr[i])
                else:
                    row[k] = None
            else:
                row[k] = _as_scalar(v)
        out.append(row)
    return out


def _expand_payload_meta(payload: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
    raw_meta = payload.get("meta", None)
    if not isinstance(raw_meta, list) or len(raw_meta) == 0:
        return [{} for _ in range(n_samples)]

    rows: List[Dict[str, Any]] = []
    for entry in raw_meta:
        if isinstance(entry, dict):
            rows.extend(_expand_meta_entry(entry))
        else:
            rows.append({"meta": _as_scalar(entry)})

    if len(rows) < n_samples:
        rows.extend([{} for _ in range(n_samples - len(rows))])
    return rows[:n_samples]


def _fold_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_pose_outputs"):
        return stem.replace("_pose_outputs", "")
    return stem


def evaluate_one_file(path: Path, unit_scale: float = 1.0) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "p_hat" not in payload:
        raise KeyError(f"Missing 'p_hat' in {path}")
    if "label" not in payload:
        raise KeyError(f"Missing 'label' in {path}; cannot compute supervised metrics")

    p_hat = _to_numpy_pose(payload["p_hat"], "p_hat")
    label = _to_numpy_pose(payload["label"], "label")
    if p_hat.shape != label.shape:
        raise ValueError(f"Shape mismatch p_hat {p_hat.shape} vs label {label.shape} in {path}")

    n_samples, n_frames, _, _ = p_hat.shape
    pred_f = _flatten_frames(p_hat)
    gt_f = _flatten_frames(label)

    e_mpjpe = mpjpe_per_frame(pred_f, gt_f) * unit_scale
    e_nmpjpe = n_mpjpe_per_frame(pred_f, gt_f) * unit_scale
    e_pmpjpe = p_mpjpe_per_frame(pred_f, gt_f) * unit_scale

    sample_meta = _expand_payload_meta(payload, n_samples)

    frame_rows: List[Dict[str, Any]] = []
    flat_idx = 0
    for s_idx in range(n_samples):
        base_meta = sample_meta[s_idx] if s_idx < len(sample_meta) else {}
        for t_idx in range(n_frames):
            row = {
                "fold": _fold_name_from_path(path),
                "sample_index": s_idx,
                "frame_index_in_sample": t_idx,
                "mpjpe": float(e_mpjpe[flat_idx]),
                "n_mpjpe": float(e_nmpjpe[flat_idx]),
                "p_mpjpe": float(e_pmpjpe[flat_idx]),
            }
            row.update(base_meta)
            frame_rows.append(row)
            flat_idx += 1

    summary = {
        "file": str(path),
        "fold": _fold_name_from_path(path),
        "num_samples": int(n_samples),
        "num_frames_per_sample": int(n_frames),
        "num_total_frames": int(n_samples * n_frames),
        "mpjpe": float(e_mpjpe.mean()),
        "n_mpjpe": float(e_nmpjpe.mean()),
        "p_mpjpe": float(e_pmpjpe.mean()),
    }

    return {"summary": summary, "frame_rows": frame_rows}


def _mean_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if len(rows) == 0:
        return {"mpjpe": float("nan"), "n_mpjpe": float("nan"), "p_mpjpe": float("nan")}
    return {
        "mpjpe": float(np.mean([r["mpjpe"] for r in rows])),
        "n_mpjpe": float(np.mean([r["n_mpjpe"] for r in rows])),
        "p_mpjpe": float(np.mean([r["p_mpjpe"] for r in rows])),
    }


def _group_rows(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(r.get(key, "unknown"))
        buckets.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for k in sorted(buckets.keys()):
        m = _mean_metrics(buckets[k])
        out.append(
            {
                "group_by": key,
                "group": k,
                "frames": len(buckets[k]),
                "mpjpe": m["mpjpe"],
                "n_mpjpe": m["n_mpjpe"],
                "p_mpjpe": m["p_mpjpe"],
            }
        )
    return out


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if len(rows) == 0:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MPJPE / N-MPJPE / P-MPJPE from fold_*_pose_outputs.pt files"
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        required=True,
        help="Directory containing fold_*_pose_outputs.pt",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="fold_*_pose_outputs.pt",
        help="Glob pattern for prediction files",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Multiply final errors by this scale (e.g. 1000 for m->mm)",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["fold", "person_id", "action_id", "cam_pair"],
        help="Grouping keys for breakdown CSV",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="metrics",
        help="Output filename prefix under pose-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_dir: Path = args.pose_dir
    if not pose_dir.exists() or not pose_dir.is_dir():
        raise FileNotFoundError(f"pose_dir not found: {pose_dir}")

    files = sorted(pose_dir.rglob(args.pattern))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' in {pose_dir}"
        )

    all_frame_rows: List[Dict[str, Any]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for path in files:
        ret = evaluate_one_file(path, unit_scale=float(args.unit_scale))
        summary = ret["summary"]
        frame_rows = ret["frame_rows"]
        for r in frame_rows:
            if "cam1_id" in r or "cam2_id" in r:
                r["cam_pair"] = f"{r.get('cam1_id', 'NA')}__{r.get('cam2_id', 'NA')}"
        fold_summaries.append(summary)
        all_frame_rows.extend(frame_rows)

    overall = _mean_metrics(all_frame_rows)
    summary_payload: Dict[str, Any] = {
        "unit_scale": float(args.unit_scale),
        "num_files": len(files),
        "num_total_frames": len(all_frame_rows),
        "overall": overall,
        "per_file": fold_summaries,
    }

    group_rows: List[Dict[str, Any]] = []
    for g in args.group_by:
        group_rows.extend(_group_rows(all_frame_rows, g))

    summary_json = pose_dir / f"{args.save_prefix}_summary.json"
    frame_csv = pose_dir / f"{args.save_prefix}_per_frame.csv"
    group_csv = pose_dir / f"{args.save_prefix}_grouped.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    _write_csv(frame_csv, all_frame_rows)
    _write_csv(group_csv, group_rows)

    print("=== Pose Metrics ===")
    print(f"Files: {len(files)}")
    print(f"Frames: {len(all_frame_rows)}")
    print(f"MPJPE:   {overall['mpjpe']:.6f}")
    print(f"N-MPJPE: {overall['n_mpjpe']:.6f}")
    print(f"P-MPJPE: {overall['p_mpjpe']:.6f}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved per-frame: {frame_csv}")
    print(f"Saved grouped: {group_csv}")


if __name__ == "__main__":
    main()
