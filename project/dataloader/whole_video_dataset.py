#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from project.dataloader.prepare_label_dict import prepare_label_dict
from project.map_config import VideoSample, label_mapping_Dict

logger = logging.getLogger(__name__)

ViewName = Literal["front", "left", "right"]


class LabeledVideoDataset(Dataset):
    """
    Multi-view labeled video dataset.

    Output:
        sample["video"][view] : Tensor (B, T, C, H, W)  # segments split by label timeline
        sample["label"]       : LongTensor (B,)
        sample["label_info"]  : List[str]
        sample["meta"]        : dict
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[Sample],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decode_audio: bool = False,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self._decode_audio = bool(decode_audio)

        # label mapping: {class_id: "label_name"} -> {"label_name": class_id}
        self._label_to_id: Dict[str, int] = {
            v: int(k) for k, v in label_mapping_Dict.items()
        }

    def __len__(self) -> int:
        return len(self._index_mapping)

    # ---------------- IO ----------------
    def _load_one_view(self, path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load one view video and return (video_tchw, fps).

        read_video(output_format="TCHW") returns:
            vframes: (T, C, H, W)
        """
        vframes, aframes, info = read_video(
            str(path),
            pts_unit="sec",
            output_format="TCHW",
        )
        fps = int(info.get("video_fps", 0))
        if fps <= 0:
            raise ValueError(f"Invalid fps={fps} for video: {path}")
        return vframes, fps

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    # ---------------- Timeline utils ----------------
    @staticmethod
    def _fill_tail_as_front(
        timeline: List[Dict[str, Any]],
        total_frames: int,
        front_label: str = "front",
    ) -> List[Dict[str, Any]]:
        """
        If the timeline doesn't cover [0, total_frames), fill uncovered gaps as front.
        Assumes timeline items have int-like start/end and end is exclusive.
        """
        if total_frames <= 0:
            return []

        # sort by start
        tl = sorted(
            (
                {
                    "start": int(x["start"]),
                    "end": int(x["end"]),
                    "label": str(x["label"]),
                }
                for x in timeline
                if x is not None and "start" in x and "end" in x and "label" in x
            ),
            key=lambda d: (d["start"], d["end"]),
        )

        filled: List[Dict[str, Any]] = []
        cur = 0

        for seg in tl:
            s, e, lb = seg["start"], seg["end"], seg["label"]
            s = max(0, min(s, total_frames))
            e = max(0, min(e, total_frames))
            if e <= s:
                continue

            # gap before this seg
            if s > cur:
                filled.append({"start": cur, "end": s, "label": front_label})

            filled.append({"start": s, "end": e, "label": lb})
            cur = max(cur, e)

        # tail gap
        if cur < total_frames:
            filled.append({"start": cur, "end": total_frames, "label": front_label})

        return filled

    def split_frame_with_label(
        self,
        front_view: torch.Tensor,  # (T,C,H,W)
        left_view: torch.Tensor,  # (T,C,H,W)
        right_view: torch.Tensor,  # (T,C,H,W)
        timeline_list: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.LongTensor]:
        """
        Split video frames according to label timeline.

        Returns:
            batch_front: (B, Tseg, C, H, W)  (NOTE: variable Tseg across segments -> stacking requires equal length)
            ...
        Important:
            If segments have variable length, torch.stack will fail.
            In that case you should pad to same length or keep as list.
        """
        assert (
            front_view.shape[0] == left_view.shape[0] == right_view.shape[0]
        ), "All views must have the same number of frames"

        T = int(front_view.shape[0])

        # 1) fill uncovered as front
        timeline = self._fill_tail_as_front(
            timeline_list, total_frames=T, front_label="front"
        )

        batch_front: List[torch.Tensor] = []
        batch_left: List[torch.Tensor] = []
        batch_right: List[torch.Tensor] = []
        labels: List[str] = []
        mapped: List[int] = []

        for seg in timeline:
            s, e, lb = int(seg["start"]), int(seg["end"]), str(seg["label"])
            if e <= s:
                continue

            seg_front = self._apply_transform(front_view[s:e])
            seg_left = self._apply_transform(left_view[s:e])
            seg_right = self._apply_transform(right_view[s:e])

            batch_front.append(seg_front)
            batch_left.append(seg_left)
            batch_right.append(seg_right)

            labels.append(lb)
            mapped.append(self._label_to_id.get(lb, -1))  # unknown -> -1

        # ⚠️ 如果每段长度不同，这里 stack 会报错。
        # 你有两种选择：
        #  A) 先 padding 到同样长度再 stack
        #  B) 保持 list 形式返回（推荐用于变长）
        #
        # 这里为了兼容你原逻辑，我保留 stack，但你要确保每段长度一致或你已对 timeline 做了固定长度切分。
        batch_front_t = torch.stack(batch_front, dim=0).permute(
            0, 2, 1, 3, 4
        )  # (B,T,C,H,W) > (B,C,T,H,W)
        batch_left_t = torch.stack(batch_left, dim=0).permute(
            0, 2, 1, 3, 4
        )  # (B,T,C,H,W) > (B,C,T,H,W)
        batch_right_t = torch.stack(batch_right, dim=0).permute(
            0, 2, 1, 3, 4
        )  # (B,T,C,H,W) > (B,C,T,H,W)

        mapped_t = torch.tensor(mapped, dtype=torch.long)

        return batch_front_t, batch_left_t, batch_right_t, labels, mapped_t

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._index_mapping[index]

        # load 3 views (T,C,H,W)
        front_frames, _ = self._load_one_view(item.videos["front"])
        left_frames, _ = self._load_one_view(item.videos["left"])
        right_frames, _ = self._load_one_view(item.videos["right"])

        assert (
            front_frames.shape[0] == left_frames.shape[0] == right_frames.shape[0]
        ), "All views must have the same number of frames"

        # labels (ensure total_end = T)
        label_dict = prepare_label_dict(
            item.label_path, total_end=int(front_frames.shape[0])
        )
        timeline_list = label_dict.get("timeline_list", [])

        (
            batch_front,
            batch_left,
            batch_right,
            labels,
            mapped_labels,
        ) = self.split_frame_with_label(
            front_frames,
            left_frames,
            right_frames,
            timeline_list,
        )

        assert (
            batch_front.shape[0]
            == batch_left.shape[0]
            == batch_right.shape[0]
            == mapped_labels.shape[0]
            == len(labels)
        ), "Batch size mismatch after splitting"

        return {
            "video": {
                "front": batch_front,
                "left": batch_left,
                "right": batch_right,
            },
            "label": mapped_labels,  # LongTensor (B,)
            "label_info": labels,  # List[str]
            "meta": {
                "experiment": self._experiment,
                "index": index,
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
            },
        }


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[Sample],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LabeledVideoDataset:
    return LabeledVideoDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
    )
