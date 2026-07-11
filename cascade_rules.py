"""级联规则引擎 · V9 现代版(4 层触发 + LLM 复核)

替代早期只有"bbox IoU + 头 30% mask"的老 CascadeRuleEngine。
本文件把 V9 (video_player_v9.py 里) 的 V9RuleEngine 提取成通用模块,
V10/V11 都用这个。

4 层触发(OR 关系):
    L1 bbox 面积比 >= 20%
    L2 动物 mask ∩ 盆 mask >= 100 像素
    L3 动物 mask 侵入盆 bbox 区域 >= 30%
    L4 盆消失 >= 3s 且动物在原位

任一层通过即计入 ongoing。LLM 复核在 ongoing 事件达到 2 秒时触发。

数据结构兼容:
    debug_pairs[i] 包含 l1_pass / l2_pass / trigger / bbox_ratio / mask_px / ...
    OngoingEvent 有 llm_start_result / llm_mid_result / llm_end_result 3 个字段
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from behavior_rules import iou
from mask_utils import polygon_to_mask, mask_overlap_area
from llm_verifier import get_verifier, VerifyResult


logger = logging.getLogger(__name__)


# ==================== 参数(与 V9 一致) ====================
BBOX_MIN_RATIO = 0.20              # L1 bbox 面积比
MASK_MIN_PIXELS = 100              # L2 mask 相交像素
INVASION_MIN_RATIO = 0.30          # L3 侵入盆区
OCCLUSION_MEMORY_SEC = 6.0
OCCLUSION_MIN_GONE_SEC = 3.0       # L4 遮挡触发

MIN_DURATION_SEC = 2.0
MAX_GAP_SEC = 1.5
LLM_MID_CHECK_SEC = 8.0
LLM_MAX_PER_EVENT = 3


# ==================== 几何工具 ====================
def bbox_overlap_ratio(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / min(a1, a2) if min(a1, a2) > 0 else 0.0


def bbox_center_distance(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def bbox_to_mask(box, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    x2 = min(w, x2); y2 = min(h, y2)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 1
    return mask


# ==================== 盆记忆(处理遮挡) ====================
class BowlMemory:
    def __init__(self):
        self.slots = {}
        self._next_id = 0

    def _match_or_create(self, box, now):
        best_id, best_dist = None, float("inf")
        for bid, s in self.slots.items():
            d = bbox_center_distance(box, s["box"])
            avg = ((box[2] - box[0]) + (box[3] - box[1])) / 2
            if d < avg * 1.0 and d < best_dist:
                best_dist = d; best_id = bid
        if best_id is not None:
            self.slots[best_id]["box"] = box
            self.slots[best_id]["last_seen"] = now
            return best_id
        bid = self._next_id
        self._next_id += 1
        self.slots[bid] = {"box": box, "last_seen": now, "first_seen": now}
        return bid

    def update(self, bowls, now):
        ids = []
        for b in bowls:
            ids.append(self._match_or_create(b["box"], now))
        expired = [bid for bid, s in self.slots.items()
                    if now - s["last_seen"] > OCCLUSION_MEMORY_SEC]
        for bid in expired:
            del self.slots[bid]
        return ids


# ==================== 事件对象 ====================
@dataclass
class OngoingEvent:
    """兼容 V9/V10/V11 的字段结构"""
    event_id: str
    animal_cls: str
    event_type: str
    bowl_id: int
    start_time: float
    last_seen: float
    hit_count: int = 0
    conf_sum: float = 0.0
    max_bbox: float = 0.0
    max_mask: int = 0
    max_invasion: float = 0.0
    trigger_types: set = field(default_factory=set)
    # LLM 复核 3 段
    llm_calls: int = 0
    llm_mid_triggered: bool = False
    llm_start_result: Optional[VerifyResult] = None
    llm_mid_result: Optional[VerifyResult] = None
    llm_end_result: Optional[VerifyResult] = None

    # V9 兼容属性别名
    @property
    def hit(self): return self.hit_count


@dataclass
class FinalizedEvent:
    event_id: str
    animal_cls: str
    event_type: str
    bowl_id: int
    start_time: float
    end_time: float
    duration_sec: float
    confidence: float
    hit_count: int
    max_bbox: float
    max_mask: int
    max_invasion: float
    trigger_types: List[str]
    llm_calls: int
    llm_confirmed_count: int
    llm_pass: bool
    llm_reasons: List[str] = field(default_factory=list)


# ==================== 规则引擎 ====================
class CascadeRuleEngine:
    """V9 4 层规则 + LLM 复核

    debug_pairs 每项含:
        animal, bowl_id, bbox_ratio, mask_px, invasion_ratio,
        l1_pass, l2_pass, trigger, types, occluded
    """

    def __init__(self, use_llm: bool = True):
        self.bowl_memory = BowlMemory()
        self.ongoing: Dict[Tuple, OngoingEvent] = {}
        self.verifier = get_verifier() if use_llm else None

    def update(self, animals, bowls, now, frame_bgr=None):
        if frame_bgr is None:
            fh, fw = 720, 1280
        else:
            fh, fw = frame_bgr.shape[:2]

        bowl_ids = self.bowl_memory.update(bowls, now)
        debug_pairs = []

        # 预计算动物 mask
        for a in animals:
            a["_mask"] = (polygon_to_mask(a.get("mask_pts"), fh, fw)
                          if a.get("mask_pts") else None)

        # ---------- 判定 1:L1+L2+L3 ----------
        for a in animals:
            for bid, b in zip(bowl_ids, bowls):
                bbox_r = bbox_overlap_ratio(a["box"], b["box"])
                mask_px = 0
                if a["_mask"] is not None and b.get("mask_pts"):
                    b_mask = polygon_to_mask(b["mask_pts"], fh, fw)
                    mask_px = mask_overlap_area(a["_mask"], b_mask)
                invasion_r = 0
                if a["_mask"] is not None:
                    bbb_mask = bbox_to_mask(b["box"], fh, fw)
                    invade_px = mask_overlap_area(a["_mask"], bbb_mask)
                    ba = int(np.sum(bbb_mask > 0))
                    invasion_r = invade_px / ba if ba > 0 else 0

                trig_b = bbox_r >= BBOX_MIN_RATIO
                trig_m = mask_px >= MASK_MIN_PIXELS
                trig_i = invasion_r >= INVASION_MIN_RATIO
                trigger = trig_b or trig_m or trig_i

                types = []
                if trig_b: types.append("bbox")
                if trig_m: types.append("mask")
                if trig_i: types.append("侵入")

                debug_pairs.append({
                    "animal": a["cls"], "bowl_id": bid,
                    "bbox_ratio": bbox_r,
                    "mask_px": mask_px,
                    "invasion_ratio": invasion_r,
                    "l1_pass": trig_b,
                    "l2_pass": trigger,       # V11 draw_trigger 用 l2_pass
                    "trigger": trigger,        # 兼容旧代码
                    "types": types,
                    "occluded": False,
                })

                if trigger:
                    self._add_hit((a["cls"], bid), a, bbox_r, mask_px,
                                   invasion_r, now, types, frame_bgr)

        # ---------- 判定 2:L4 遮挡 ----------
        for bid, mem in self.bowl_memory.slots.items():
            gone = now - mem["last_seen"]
            if gone < OCCLUSION_MIN_GONE_SEC: continue
            if gone > OCCLUSION_MEMORY_SEC: continue
            for a in animals:
                bbox_r = bbox_overlap_ratio(a["box"], mem["box"])
                if bbox_r < 0.15: continue
                invasion_r = 0
                if a["_mask"] is not None:
                    bm = bbox_to_mask(mem["box"], fh, fw)
                    px = mask_overlap_area(a["_mask"], bm)
                    ba = int(np.sum(bm > 0))
                    invasion_r = px / ba if ba > 0 else 0
                if bbox_r >= 0.15 or invasion_r >= 0.20:
                    debug_pairs.append({
                        "animal": a["cls"], "bowl_id": bid,
                        "bbox_ratio": bbox_r, "mask_px": 0,
                        "invasion_ratio": invasion_r,
                        "l1_pass": True, "l2_pass": True,
                        "trigger": True,
                        "types": [f"遮挡{gone:.1f}s"],
                        "occluded": True,
                    })
                    self._add_hit((a["cls"], bid), a, bbox_r, 0, invasion_r,
                                   now, [f"遮挡{gone:.1f}s"], frame_bgr)

        # ---------- LLM 复核触发 / 中期 ----------
        self._maybe_llm_verify(now, frame_bgr)

        # ---------- 结束事件 ----------
        finalized = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = now - ev.last_seen
            if gap < MAX_GAP_SEC: continue
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_DURATION_SEC:
                # 结束 LLM
                if (self.verifier and self.verifier.available
                        and frame_bgr is not None
                        and ev.llm_end_result is None
                        and ev.llm_calls < LLM_MAX_PER_EVENT):
                    ev.llm_end_result = self.verifier.verify_behavior(
                        frame_bgr, ev.event_type, ev.animal_cls,
                        f"{ev.event_id}-end",
                        extra_context=f"事件结束前 dur={dur:.1f}s")
                    if ev.llm_end_result is not None:
                        ev.llm_calls += 1
                finalized.append(self._finalize(ev))
            to_remove.append(key)
        for k in to_remove:
            del self.ongoing[k]

        return debug_pairs, finalized

    def _add_hit(self, key, a, bbox_r, mask_px, inv_r, now, types, frame_bgr):
        if key not in self.ongoing:
            ev = OngoingEvent(
                event_id=f"evt-{uuid.uuid4().hex[:12]}",
                animal_cls=a["cls"],
                event_type="drinking",
                bowl_id=key[1],
                start_time=now, last_seen=now,
                hit_count=1, conf_sum=a["conf"],
                max_bbox=bbox_r, max_mask=mask_px,
                max_invasion=inv_r,
            )
            self.ongoing[key] = ev
        ev = self.ongoing[key]
        ev.last_seen = now
        ev.hit_count += 1
        ev.conf_sum += a["conf"]
        ev.max_bbox = max(ev.max_bbox, bbox_r)
        ev.max_mask = max(ev.max_mask, mask_px)
        ev.max_invasion = max(ev.max_invasion, inv_r)
        for t in types:
            ev.trigger_types.add(t)

    def _maybe_llm_verify(self, now, frame_bgr):
        if not self.verifier or not self.verifier.available or frame_bgr is None:
            return
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            # 首次
            if ev.llm_start_result is None and dur >= MIN_DURATION_SEC and \
                    ev.llm_calls < LLM_MAX_PER_EVENT:
                ev.llm_start_result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type, ev.animal_cls,
                    f"{ev.event_id}-start",
                    extra_context=(
                        f"YOLO 置信度 {ev.conf_sum/max(1,ev.hit_count):.2f}"))
                if ev.llm_start_result is not None:
                    ev.llm_calls += 1
            # 中期
            if (not ev.llm_mid_triggered and dur >= LLM_MID_CHECK_SEC
                    and ev.llm_calls < LLM_MAX_PER_EVENT):
                ev.llm_mid_triggered = True
                ev.llm_mid_result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type, ev.animal_cls,
                    f"{ev.event_id}-mid",
                    extra_context=f"事件持续 {dur:.1f} 秒")
                if ev.llm_mid_result is not None:
                    ev.llm_calls += 1

    def _finalize(self, ev: OngoingEvent) -> FinalizedEvent:
        results = [r for r in (ev.llm_start_result,
                                ev.llm_mid_result,
                                ev.llm_end_result) if r is not None]
        confirmed_count = sum(1 for r in results if r.confirmed)
        llm_pass = (not self.verifier or not self.verifier.available
                     or confirmed_count >= 1)
        return FinalizedEvent(
            event_id=ev.event_id,
            animal_cls=ev.animal_cls,
            event_type=ev.event_type,
            bowl_id=ev.bowl_id,
            start_time=ev.start_time,
            end_time=ev.last_seen,
            duration_sec=ev.last_seen - ev.start_time,
            confidence=ev.conf_sum / max(1, ev.hit_count),
            hit_count=ev.hit_count,
            max_bbox=ev.max_bbox,
            max_mask=ev.max_mask,
            max_invasion=ev.max_invasion,
            trigger_types=list(ev.trigger_types),
            llm_calls=ev.llm_calls,
            llm_confirmed_count=confirmed_count,
            llm_pass=llm_pass,
            llm_reasons=[r.reason for r in results],
        )

    def force_flush(self, now, frame_bgr=None):
        finalized = []
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_DURATION_SEC:
                if (self.verifier and self.verifier.available
                        and frame_bgr is not None
                        and ev.llm_end_result is None):
                    ev.llm_end_result = self.verifier.verify_behavior(
                        frame_bgr, ev.event_type, ev.animal_cls,
                        f"{ev.event_id}-flush")
                    if ev.llm_end_result is not None:
                        ev.llm_calls += 1
                finalized.append(self._finalize(ev))
        self.ongoing.clear()
        return finalized
