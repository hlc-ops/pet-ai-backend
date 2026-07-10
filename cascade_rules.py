"""三层级联规则引擎

架构:
    L1 (bbox IoU) → L2 (mask overlap) → L3 (LLM verification)

L1 便宜大量过滤,L2 精准判定,L3 关键时刻兜底 + 提供解释性。

每层触发条件:
    L1: bbox IoU >= 0.05                        (99% 帧被过滤)
    L2: 头部 mask ∩ 盆 mask >= MIN_OVERLAP      (剩下的 5-10%)
    L3: LLM 视觉复核 confirmed=True            (触发/中期/结束 各 1 次)

事件流转:
    - 单帧满足 L1+L2 -> 进入 ongoing
    - ongoing 累积到 MIN_DURATION_SEC -> 触发 L3 首次复核
    - 事件持续满 LLM_MID_CHECK_SEC -> 触发 L3 中期复核
    - 事件结束 -> 触发 L3 结束复核
    - 3 次 LLM 至少 2 次 confirmed -> 最终事件产出
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from behavior_rules import iou
from mask_utils import (polygon_to_mask, mask_overlap_area,
                         head_region_mask)
from llm_verifier import get_verifier, VerifyResult


logger = logging.getLogger(__name__)


# ==================== 参数 ====================
L1_IOU_THRESHOLD = 0.05           # bbox 粗过滤
L2_MIN_OVERLAP_PX = 50            # mask 像素相交阈值
L2_MIN_OVERLAP_RATIO = 0.05       # 或头部占比 5%

# 事件时长
MIN_DURATION_SEC = 2.0
MAX_GAP_SEC = 1.5
LLM_MID_CHECK_SEC = 8.0           # 事件持续 8 秒时触发中期复核

# LLM 判定
LLM_MIN_CONFIRMS = 1              # 3 次至少 1 次 confirmed = 通过(严格改 2)


@dataclass
class OngoingEvent:
    event_id: str
    animal_cls: str
    event_type: str
    start_time: float
    last_seen: float
    hit_count: int = 0
    conf_sum: float = 0.0
    max_overlap: int = 0
    # LLM 复核记录
    llm_start_result: Optional[VerifyResult] = None
    llm_mid_result: Optional[VerifyResult] = None
    llm_end_result: Optional[VerifyResult] = None
    llm_start_frame: Optional[np.ndarray] = None
    llm_mid_triggered: bool = False


@dataclass
class FinalizedEvent:
    event_id: str
    animal_cls: str
    event_type: str
    start_time: float
    end_time: float
    duration_sec: float
    confidence: float
    hit_count: int
    max_overlap: int
    llm_confirmed_count: int
    llm_reasons: List[str] = field(default_factory=list)
    llm_pass: bool = False


class CascadeRuleEngine:
    """三层级联规则引擎(线程不安全,单视频/单摄像头用)"""

    def __init__(self, use_llm: bool = True):
        self.ongoing: Dict[Tuple, OngoingEvent] = {}
        self.verifier = get_verifier() if use_llm else None

    def update(self, animals: List[dict], bowls: List[dict],
               frame_time: float, frame_bgr: Optional[np.ndarray] = None
               ) -> Tuple[List[dict], List[FinalizedEvent]]:
        """
        输入:
            animals: [{cls, box, conf, mask_pts}, ...]
            bowls:   [{box, conf, mask_pts}, ...]
            frame_time: 视频/时间戳(秒)
            frame_bgr: 用于 LLM 复核的原始帧(可选)
        输出:
            debug_pairs: 每个 (animal, bowl) 对的判定详情,给 UI 用
            finalized: 已确认完成的事件
        """
        if frame_bgr is None:
            frame_h, frame_w = 480, 640  # 兜底
        else:
            frame_h, frame_w = frame_bgr.shape[:2]

        # ---------- 每一对做级联判定 ----------
        debug_pairs = []
        active_keys = set()

        for a_idx, a in enumerate(animals):
            # L1: bbox IoU
            head_mask_cached = None
            for b_idx, b in enumerate(bowls):
                l1_iou = iou(a["box"], b["box"])
                debug = {
                    "animal": a["cls"],
                    "bowl_idx": b_idx,
                    "l1_iou": l1_iou,
                    "l1_pass": False,
                    "l2_overlap_px": 0,
                    "l2_ratio": 0.0,
                    "l2_pass": False,
                }
                if l1_iou < L1_IOU_THRESHOLD:
                    debug_pairs.append(debug)
                    continue
                debug["l1_pass"] = True

                # L2: mask 相交
                if head_mask_cached is None:
                    a_mask = polygon_to_mask(
                        a.get("mask_pts"), frame_h, frame_w)
                    head_mask_cached = head_region_mask(
                        a_mask, top_ratio=0.3)
                    head_area = int(np.sum(head_mask_cached > 0))
                b_mask = polygon_to_mask(
                    b.get("mask_pts"), frame_h, frame_w)
                overlap_px = mask_overlap_area(head_mask_cached, b_mask)
                overlap_ratio = (overlap_px / head_area
                                  if head_area > 0 else 0)
                debug["l2_overlap_px"] = overlap_px
                debug["l2_ratio"] = overlap_ratio
                debug["l2_pass"] = (overlap_px >= L2_MIN_OVERLAP_PX
                                     or overlap_ratio >= L2_MIN_OVERLAP_RATIO)
                debug_pairs.append(debug)

                if not debug["l2_pass"]:
                    continue

                # L1 + L2 通过 -> 记入 ongoing
                key = (a["cls"], b_idx)
                active_keys.add(key)

                if key in self.ongoing:
                    ev = self.ongoing[key]
                    ev.last_seen = frame_time
                    ev.hit_count += 1
                    ev.conf_sum += a["conf"]
                    ev.max_overlap = max(ev.max_overlap, overlap_px)
                else:
                    ev = OngoingEvent(
                        event_id=f"evt-{uuid.uuid4().hex[:12]}",
                        animal_cls=a["cls"],
                        event_type="drinking",
                        start_time=frame_time,
                        last_seen=frame_time,
                        hit_count=1,
                        conf_sum=a["conf"],
                        max_overlap=overlap_px,
                    )
                    self.ongoing[key] = ev
                    # L3: 首次触发时保存关键帧,持续到 MIN_DURATION 后调 LLM
                    if frame_bgr is not None:
                        ev.llm_start_frame = frame_bgr.copy()

                # 检查是否满足首次 LLM 复核(持续时间够 + 未复核)
                dur = ev.last_seen - ev.start_time
                if (ev.llm_start_result is None
                        and dur >= MIN_DURATION_SEC
                        and self.verifier
                        and ev.llm_start_frame is not None):
                    ev.llm_start_result = self.verifier.verify_behavior(
                        ev.llm_start_frame, ev.event_type,
                        ev.animal_cls, ev.event_id,
                        extra_context=f"YOLO 置信度 {ev.conf_sum/ev.hit_count:.2f}")

                # 中期复核
                if (not ev.llm_mid_triggered
                        and dur >= LLM_MID_CHECK_SEC
                        and frame_bgr is not None
                        and self.verifier):
                    ev.llm_mid_triggered = True
                    ev.llm_mid_result = self.verifier.verify_behavior(
                        frame_bgr, ev.event_type,
                        ev.animal_cls, ev.event_id + "-mid",
                        extra_context=f"事件已持续 {dur:.1f} 秒")

        # ---------- 结束过期事件 ----------
        finalized = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = frame_time - ev.last_seen
            if gap < MAX_GAP_SEC:
                continue
            dur = ev.last_seen - ev.start_time
            if dur < MIN_DURATION_SEC:
                to_remove.append(key)
                continue

            # 结束复核
            if (self.verifier and frame_bgr is not None
                    and ev.llm_end_result is None):
                ev.llm_end_result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type,
                    ev.animal_cls, ev.event_id + "-end",
                    extra_context=f"事件即将结束,总时长 {dur:.1f} 秒")

            fe = self._finalize(ev)
            finalized.append(fe)
            to_remove.append(key)

        for k in to_remove:
            del self.ongoing[k]

        return debug_pairs, finalized

    def force_flush(self, frame_time: float,
                     frame_bgr: Optional[np.ndarray] = None
                     ) -> List[FinalizedEvent]:
        finalized = []
        for ev in self.ongoing.values():
            dur = ev.last_seen - ev.start_time
            if dur < MIN_DURATION_SEC:
                continue
            if (self.verifier and frame_bgr is not None
                    and ev.llm_end_result is None):
                ev.llm_end_result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type,
                    ev.animal_cls, ev.event_id + "-end")
            finalized.append(self._finalize(ev))
        self.ongoing.clear()
        return finalized

    def _finalize(self, ev: OngoingEvent) -> FinalizedEvent:
        results = [r for r in (ev.llm_start_result,
                                ev.llm_mid_result,
                                ev.llm_end_result) if r is not None]
        confirmed_count = sum(1 for r in results if r.confirmed)
        reasons = [r.reason for r in results]
        llm_pass = (not self.verifier or not self.verifier.available
                     or confirmed_count >= LLM_MIN_CONFIRMS)

        return FinalizedEvent(
            event_id=ev.event_id,
            animal_cls=ev.animal_cls,
            event_type=ev.event_type,
            start_time=ev.start_time,
            end_time=ev.last_seen,
            duration_sec=ev.last_seen - ev.start_time,
            confidence=ev.conf_sum / max(1, ev.hit_count),
            hit_count=ev.hit_count,
            max_overlap=ev.max_overlap,
            llm_confirmed_count=confirmed_count,
            llm_reasons=reasons,
            llm_pass=llm_pass,
        )
