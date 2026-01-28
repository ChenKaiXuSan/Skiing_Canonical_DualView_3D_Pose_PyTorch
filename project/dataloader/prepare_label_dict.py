#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Label utilities

- Load label JSON into dict: {label: [{start, end}, ...]}
- Automatically fill unlabeled gaps as `front`
- Convert to a time-ordered timeline list:
    [{"label": "front", "start": 0, "end": 5}, {"label":"left","start":5,"end":10}, ...]
- Print label stats: names + counts
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------
# Loading labels
# ---------------------------------------------------------------------
def load_label_dict(
    json_path: str | Path,
    *,
    annotator: Optional[int] = None,
    annotation_id: Optional[int] = None,
    merge_all: bool = False,
) -> Dict[str, List[dict]]:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if not isinstance(anns, list):
        raise ValueError("Invalid json: 'annotations' must be a list")

    if merge_all:
        chosen = anns
    else:
        chosen = []
        for a in anns:
            if annotation_id is not None and a.get("annotation_id") != annotation_id:
                continue
            if annotator is not None and a.get("annotator") != annotator:
                continue
            chosen.append(a)

        if annotator is None and annotation_id is None:
            if not anns:
                return {}

            def _ts(x: dict) -> str:
                return x.get("updated_at") or x.get("created_at") or ""

            chosen = [max(anns, key=_ts)]

    out: Dict[str, List[dict]] = defaultdict(list)

    for ann in chosen:
        video_labels = ann.get("videoLabels", [])
        if not isinstance(video_labels, list):
            continue

        for item in video_labels:
            labels = item.get("timelinelabels", [])
            ranges = item.get("ranges", [])
            if not labels or not ranges:
                continue

            for lb in labels:
                for r in ranges:
                    if r is None:
                        continue
                    s = r.get("start")
                    e = r.get("end")
                    if s is None or e is None:
                        continue
                    out[str(lb)].append({"start": float(s), "end": float(e)})

    for lb in out:
        out[lb].sort(key=lambda x: (x["start"], x["end"]))

    return dict(out)


# ---------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------
def _merge_intervals(
    intervals: List[Tuple[float, float]], eps: float = 1e-9
) -> List[Tuple[float, float]]:
    intervals = [
        (float(s), float(e))
        for s, e in intervals
        if e is not None and s is not None and e > s
    ]
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + eps:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _clip_intervals(
    intervals: List[Tuple[float, float]], start: float, end: float
) -> List[Tuple[float, float]]:
    out = []
    for s, e in intervals:
        s2, e2 = max(start, s), min(end, e)
        if e2 > s2:
            out.append((s2, e2))
    return out


# ---------------------------------------------------------------------
# Fill unlabeled as front
# ---------------------------------------------------------------------
def fill_unlabeled_as_front(
    label_dict: Dict[str, List[dict]],
    *,
    total_end: Optional[float] = None,
    total_start: float = 0.0,
    front_label: str = "front",
    eps: float = 1e-9,
) -> Dict[str, List[dict]]:
    all_intervals: List[Tuple[float, float]] = []
    for segs in label_dict.values():
        for r in segs:
            s, e = r.get("start"), r.get("end")
            if s is None or e is None:
                continue
            all_intervals.append((float(s), float(e)))

    if total_end is None:
        total_end = max([e for _, e in all_intervals], default=total_start)

    covered = _merge_intervals(
        _clip_intervals(all_intervals, total_start, float(total_end)), eps=eps
    )

    gaps: List[Tuple[float, float]] = []
    cur = float(total_start)
    for s, e in covered:
        if s > cur + eps:
            gaps.append((cur, s))
        cur = max(cur, e)
    if float(total_end) > cur + eps:
        gaps.append((cur, float(total_end)))

    out = {k: [dict(x) for x in v] for k, v in label_dict.items()}
    front_list = out.get(front_label, [])
    front_list.extend([{"start": s, "end": e} for s, e in gaps])

    merged_front = _merge_intervals(
        [(r["start"], r["end"]) for r in front_list], eps=eps
    )
    out[front_label] = [{"start": s, "end": e} for s, e in merged_front]

    for k in out:
        out[k].sort(key=lambda x: (x["start"], x["end"]))
    return out


# ---------------------------------------------------------------------
# Timeline conversion (your requested output)
# ---------------------------------------------------------------------
def label_dict_to_timeline(
    label_dict: Dict[str, List[dict]],
    *,
    eps: float = 1e-9,
    sort: bool = True,
) -> List[dict]:
    """
    Convert {label:[{start,end},...]} -> [{"label":lb,"start":s,"end":e}, ...] sorted by time.
    """
    timeline: List[dict] = []
    for lb, segs in label_dict.items():
        for r in segs:
            s, e = r.get("start"), r.get("end")
            if s is None or e is None:
                continue
            s = float(s)
            e = float(e)
            if e <= s + eps:
                continue
            timeline.append({"label": str(lb), "start": s, "end": e})

    if sort:
        timeline.sort(key=lambda x: (x["start"], x["end"], x["label"]))

    return timeline


def print_label_stats(label_dict: Dict[str, List[dict]]) -> None:
    names = sorted(label_dict.keys())
    counts = {k: len(v) for k, v in label_dict.items()}
    total = sum(counts.values())

    print(f"[labels] names({len(names)}): {names}")
    print(f"[labels] total_segments: {total}")
    for k in names:
        print(f"  - {k}: {counts[k]} segments")


# ---------------------------------------------------------------------
# Main prepare function
# ---------------------------------------------------------------------
def prepare_label_dict(
    path: str | Path,
    *,
    total_end: Optional[float] = 3000.0,
    merge_all: bool = True,
    fill_front: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict containing both:
      - raw label_dict: {label:[{start,end},...]}
      - timeline_list: [{"label":..., "start":..., "end":...}, ...] (sorted)
      - timeline_dict: {"front":{"start":0,"end":5}, ...}  (may overwrite if repeated)

    Also prints label stats.
    """
    labels = load_label_dict(path, merge_all=merge_all)
    if fill_front:
        labels = fill_unlabeled_as_front(labels, total_end=total_end)

    print_label_stats(labels)

    timeline_list = label_dict_to_timeline(labels)

    return {
        "label_dict": labels,
        "timeline_list": timeline_list,
    }
