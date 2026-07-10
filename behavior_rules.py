"""行为规则引擎

MVP 版本支持:
- drinking(饮水):animal + bowl 接触 + 头低于盆 + 持续 >= MIN_DURATION_SEC

TODO(待姿态识别接入后完善):
- excretion(排泄):需 SuperAnimal-Quadruped 关键点 + 姿态角度判定
- eating(吃饭)vs drinking:目前统一按 drinking 上报,姿态接入后区分

规则引擎是**状态机**:
    每帧调 update() 更新进行中事件
    动物离开盆 >= MAX_GAP_SEC 后:如果时长够,产出"完成事件"
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import Config


logger = logging.getLogger(__name__)


# ==================== 事件类型 ====================
EVENT_DRINKING = "drinking"
EVENT_EXCRETION = "excretion"


# ==================== 几何工具 ====================
def iou(box1, box2) -> float:
    """两个 xyxy 的 IoU(交并比)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def head_below_bowl_top(animal_box, bowl_box) -> bool:
    """动物"头部"(上 30% 处)是否低于 bowl 顶部 —— 判定喝水的必要条件"""
    animal_top = animal_box[1]
    animal_head_y = animal_top + (animal_box[3] - animal_top) * 0.3
    bowl_top = bowl_box[1]
    return animal_head_y > bowl_top


def bbox_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ==================== 事件数据结构 ====================
@dataclass
class CompletedEvent:
    """已完成的事件,准备推送"""
    event_id: str
    event_type: str
    kennel_id: str
    camera_id: str
    pet_id: str
    detected_class: str      # cat / dog / monkey / other_primate
    start_time: float        # unix 秒
    end_time: float
    duration_sec: float
    confidence: float        # 平均置信度
    hit_count: int           # 触发帧数
    snapshot_path: Optional[str] = None


@dataclass
class OngoingEvent:
    """内存中跟踪的进行中事件"""
    event_type: str
    detected_class: str
    animal_box: Tuple[float, float, float, float]
    bowl_box: Tuple[float, float, float, float]
    start_time: float
    last_seen: float
    confidence_sum: float = 0.0
    hit_count: int = 0
    last_frame_bgr: Optional[object] = None  # 保留最后一帧供截图


# ==================== 规则引擎 ====================
class BehaviorRuleEngine:
    """
    对每一帧的检测结果调 update(),内部维护"进行中事件"状态机。
    动物离开盆 >= MAX_GAP_SEC 时把事件"完结",若时长够返回 CompletedEvent。
    """

    def __init__(self,
                 kennel_id: str,
                 camera_id: str = "",
                 pet_id: str = ""):
        self.kennel_id = kennel_id
        self.camera_id = camera_id
        self.pet_id = pet_id
        # key = (detected_class, bowl_box_int_tuple)
        self.ongoing: Dict[Tuple, OngoingEvent] = {}

    # ---------- 主循环 ----------
    def update(self, animals: List[dict], bowls: List[dict],
               frame_time: float,
               frame_bgr=None) -> List[CompletedEvent]:
        """
        每帧调一次。
        - animals: [{"cls":"cat","box":[x,y,x,y],"conf":0.9}, ...]
        - bowls:   [{"box":[x,y,x,y],"conf":0.85}, ...]
        - frame_time: 帧时间戳(unix 秒)
        - frame_bgr: 可选,原始帧用于截图取证

        返回本帧新完结的事件列表(通常是 0-1 个)
        """
        # 1. 找出所有当前触发对(animal, bowl)
        active_keys = set()
        for animal in animals:
            for bowl in bowls:
                if not self._is_drinking(animal, bowl):
                    continue

                bowl_key = tuple(int(v) for v in bowl["box"])
                key = (animal["cls"], bowl_key)
                active_keys.add(key)

                if key in self.ongoing:
                    ev = self.ongoing[key]
                    ev.last_seen = frame_time
                    ev.confidence_sum += animal["conf"]
                    ev.hit_count += 1
                    if frame_bgr is not None:
                        ev.last_frame_bgr = frame_bgr
                else:
                    self.ongoing[key] = OngoingEvent(
                        event_type=EVENT_DRINKING,
                        detected_class=animal["cls"],
                        animal_box=tuple(animal["box"]),
                        bowl_box=tuple(bowl["box"]),
                        start_time=frame_time,
                        last_seen=frame_time,
                        confidence_sum=animal["conf"],
                        hit_count=1,
                        last_frame_bgr=frame_bgr,
                    )

        # 2. 结束超过 MAX_GAP_SEC 没触发的事件
        completed = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = frame_time - ev.last_seen
            if gap >= Config.MAX_EVENT_GAP_SEC:
                duration = ev.last_seen - ev.start_time
                if duration >= Config.MIN_EVENT_DURATION_SEC:
                    completed.append(self._finalize(ev))
                to_remove.append(key)
        for key in to_remove:
            del self.ongoing[key]

        return completed

    def force_flush(self, frame_time: float) -> List[CompletedEvent]:
        """强制结束所有进行中事件(视频处理结束时用)"""
        completed = []
        for ev in self.ongoing.values():
            duration = ev.last_seen - ev.start_time
            if duration >= Config.MIN_EVENT_DURATION_SEC:
                completed.append(self._finalize(ev))
        self.ongoing.clear()
        return completed

    # ---------- 判定 ----------
    def _is_drinking(self, animal: dict, bowl: dict) -> bool:
        """drinking 触发条件:
        1. IoU >= IOU_THRESHOLD
        2. 动物头(上 30% 位置)低于盆顶
        """
        if iou(animal["box"], bowl["box"]) < Config.IOU_THRESHOLD:
            return False
        if not head_below_bowl_top(animal["box"], bowl["box"]):
            return False
        return True

    # ---------- 完结事件 ----------
    def _finalize(self, ev: OngoingEvent) -> CompletedEvent:
        avg_conf = ev.confidence_sum / max(1, ev.hit_count)
        duration = ev.last_seen - ev.start_time
        return CompletedEvent(
            event_id=f"evt-{uuid.uuid4().hex}",
            event_type=ev.event_type,
            kennel_id=self.kennel_id,
            camera_id=self.camera_id,
            pet_id=self.pet_id,
            detected_class=ev.detected_class,
            start_time=ev.start_time,
            end_time=ev.last_seen,
            duration_sec=duration,
            confidence=round(avg_conf, 3),
            hit_count=ev.hit_count,
            snapshot_path=None,  # 后面单独存
        )
