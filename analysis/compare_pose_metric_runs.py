#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def _n_mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    denom = np.sum(pred * pred, axis=(1, 2))
    numer = np.sum(pred * gt, axis=(1, 2))
    scale = np.where(denom > 1e-12, numer / denom, 1.0)
    pred_scaled = pred * scale[:, None, None]
    return _mpjpe_per_frame(pred_scaled, gt)


def _procrustes_align_one(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
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


def _p_mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(pred)
    for i in range(pred.shape[0]):
        aligned[i] = _procrustes_align_one(pred[i], gt[i])
    return _mpjpe_per_frame(aligned, gt)


def _evaluate_one_file(path: Path, unit_scale: float = 1.0) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "p_hat" not in payload:
        raise KeyError(f"Missing 'p_hat' in {path}")
    if "label" not in payload:
        raise KeyError(f"Missing 'label' in {path}; cannot compute supervised metrics")

    p_hat = _to_numpy_pose(payload["p_hat"], "p_hat")
    label = _to_numpy_pose(payload["label"], "label")
    if p_hat.shape != label.shape:
        raise ValueError(f"Shape mismatch p_hat {p_hat.shape} vs label {label.shape} in {path}")

    pred_f = _flatten_frames(p_hat)
    gt_f = _flatten_frames(label)

    e_mpjpe = _mpjpe_per_frame(pred_f, gt_f) * unit_scale
    e_nmpjpe = _n_mpjpe_per_frame(pred_f, gt_f) * unit_scale
    e_pmpjpe = _p_mpjpe_per_frame(pred_f, gt_f) * unit_scale

    n_samples, n_frames = p_hat.shape[0], p_hat.shape[1]
    return {
        "summary": {
            "file": str(path),
            "num_samples": int(n_samples),
            "num_frames_per_sample": int(n_frames),
            "num_total_frames": int(n_samples * n_frames),
            "mpjpe": float(e_mpjpe.mean()),
            "n_mpjpe": float(e_nmpjpe.mean()),
            "p_mpjpe": float(e_pmpjpe.mean()),
        }
    }


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _read_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid summary json (not object): {path}")
    if "overall" not in data or not isinstance(data["overall"], dict):
        raise KeyError(f"Missing overall metrics in {path}")
    return data


def _infer_run_id(summary_path: Path) -> str:
    # expected: .../<run_ts>/pose_analysis/metrics_summary.json
    # fallback to parent paths when directory depth differs
    parts = summary_path.parts
    if len(parts) >= 3 and parts[-2] == "pose_analysis":
        return parts[-3]
    return summary_path.parent.name


def _iter_summaries(root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(root.glob(pattern))


def _iter_pose_dirs(root: Path) -> Iterable[Path]:
    for d in sorted(root.glob("**/pose_analysis")):
        if d.is_dir():
            yield d


def _row_from_summary(summary_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    overall = payload.get("overall", {})
    run_id = _infer_run_id(summary_path)
    pose_dir = summary_path.parent

    return {
        "run_id": run_id,
        "summary_file": str(summary_path),
        "pose_dir": str(pose_dir),
        "unit_scale": payload.get("unit_scale", 1.0),
        "num_files": payload.get("num_files", 0),
        "num_total_frames": payload.get("num_total_frames", 0),
        "mpjpe": _to_float(overall.get("mpjpe")),
        "n_mpjpe": _to_float(overall.get("n_mpjpe")),
        "p_mpjpe": _to_float(overall.get("p_mpjpe")),
    }


def _row_from_pose_dir(pose_dir: Path, unit_scale: float = 1.0) -> Dict[str, Any]:
    files = sorted(pose_dir.glob("fold_*_pose_outputs.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No fold pose files found in {pose_dir}")

    all_mpjpe: List[float] = []
    all_nmpjpe: List[float] = []
    all_pmpjpe: List[float] = []
    total_frames = 0
    for f in files:
        ret = _evaluate_one_file(f, unit_scale=unit_scale)
        s = ret["summary"]
        all_mpjpe.append(float(s["mpjpe"]))
        all_nmpjpe.append(float(s["n_mpjpe"]))
        all_pmpjpe.append(float(s["p_mpjpe"]))
        total_frames += int(s["num_total_frames"])

    run_id = pose_dir.parent.name
    return {
        "run_id": run_id,
        "summary_file": "",
        "pose_dir": str(pose_dir),
        "unit_scale": unit_scale,
        "num_files": len(files),
        "num_total_frames": total_frames,
        "mpjpe": sum(all_mpjpe) / len(all_mpjpe),
        "n_mpjpe": sum(all_nmpjpe) / len(all_nmpjpe),
        "p_mpjpe": sum(all_pmpjpe) / len(all_pmpjpe),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
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


def _safe_min(rows: List[Dict[str, Any]], key: str) -> Tuple[str, float]:
    valid = [r for r in rows if isinstance(r.get(key), (float, int))]
    if not valid:
        return "N/A", float("nan")
    best = min(valid, key=lambda r: float(r[key]))
    return str(best.get("run_id", "N/A")), float(best[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare metrics_summary.json across multiple training runs"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/train"),
        help="Search root for metrics_summary.json",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/pose_analysis/metrics_summary.json",
        help="Glob pattern relative to root",
    )
    parser.add_argument(
        "--fallback-fold-files",
        action="store_true",
        help="If no metrics_summary.json found, scan pose_analysis/fold_*_pose_outputs.pt and compute run metrics on the fly",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="p_mpjpe",
        choices=["mpjpe", "n_mpjpe", "p_mpjpe", "run_id"],
        help="Primary sort key for comparison table",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="metrics_run_compare",
        help="Output filename prefix in root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"root not found: {root}")

    files = list(_iter_summaries(root, args.pattern))

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    if len(files) > 0:
        for f in files:
            try:
                payload = _read_summary(f)
                rows.append(_row_from_summary(f, payload))
            except Exception as e:
                errors.append({"file": str(f), "error": str(e)})
    elif args.fallback_fold_files:
        pose_dirs = list(_iter_pose_dirs(root))
        if len(pose_dirs) == 0:
            raise FileNotFoundError(
                f"No pose_analysis dirs found under {root}"
            )
        for d in pose_dirs:
            try:
                rows.append(_row_from_pose_dir(d))
            except Exception as e:
                errors.append({"file": str(d), "error": str(e)})
        files = [d / "fold_*_pose_outputs.pt" for d in pose_dirs]
    else:
        raise FileNotFoundError(
            f"No metrics summary found under {root} with pattern {args.pattern}. "
            "Use --fallback-fold-files to compute directly from fold pose outputs."
        )

    rows.sort(key=lambda r: r.get(args.sort_by, float("inf")))

    best_mpjpe_run, best_mpjpe_val = _safe_min(rows, "mpjpe")
    best_nmpjpe_run, best_nmpjpe_val = _safe_min(rows, "n_mpjpe")
    best_pmpjpe_run, best_pmpjpe_val = _safe_min(rows, "p_mpjpe")

    summary = {
        "search_root": str(root),
        "pattern": args.pattern,
        "num_found": len(files),
        "num_valid": len(rows),
        "num_invalid": len(errors),
        "sort_by": args.sort_by,
        "best": {
            "mpjpe": {"run_id": best_mpjpe_run, "value": best_mpjpe_val},
            "n_mpjpe": {"run_id": best_nmpjpe_run, "value": best_nmpjpe_val},
            "p_mpjpe": {"run_id": best_pmpjpe_run, "value": best_pmpjpe_val},
        },
        "errors": errors,
    }

    json_out = root / f"{args.save_prefix}_summary.json"
    csv_out = root / f"{args.save_prefix}.csv"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(csv_out, rows)

    print("=== Run Metrics Comparison ===")
    print(f"Found summaries: {len(files)}")
    print(f"Valid summaries: {len(rows)}")
    print(f"Sorted by: {args.sort_by}")
    print(f"Best MPJPE:   {best_mpjpe_val:.6f} ({best_mpjpe_run})")
    print(f"Best N-MPJPE: {best_nmpjpe_val:.6f} ({best_nmpjpe_run})")
    print(f"Best P-MPJPE: {best_pmpjpe_val:.6f} ({best_pmpjpe_run})")
    print(f"Saved table: {csv_out}")
    print(f"Saved summary: {json_out}")

    if rows:
        print("Top runs:")
        topn = min(5, len(rows))
        for i in range(topn):
            r = rows[i]
            print(
                f"  {i+1}. {r['run_id']} | "
                f"mpjpe={float(r['mpjpe']):.6f}, "
                f"n_mpjpe={float(r['n_mpjpe']):.6f}, "
                f"p_mpjpe={float(r['p_mpjpe']):.6f}"
            )


if __name__ == "__main__":
    main()
