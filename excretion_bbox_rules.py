"""排泄识别 · bbox 特征版(方案 A · 不需要姿态模型)

基于 6 张排泄图分析的关键特征(D:\wenti63):
1. 姿态改变:排泄时动物 bbox 高宽比变小(蹲下,变矮)
2. 位置稳定:排泄时 bbox 中心 5+ 秒不移动
3. 尾部区域:排泄时 bbox 底部 20% 会有排泄物出现(可选,难识别)

参数(基于反例调过):
- 站立参考 h/w 比:1.0 ~ 1.4(正例反例区分)
- 蹲姿 h/w 比:0.7 ~ 0.95(蹲下)
- 稳定性:5 秒内 bbox 中心移动 < bbox 宽度的 15%

精度期望:70-75%(bbox 版)
升级到 SuperAnimal 姿态版可到 90%+
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ==================== 参数 ====================
CROUCH_ASPECT_MAX = 0.95         # 排泄蹲姿 h/w 阈值(小于表示压扁)
STAND_ASPECT_MIN = 1.05          # 站立参考 h/w
STABLE_WINDOW_SEC = 5.0          # 观察窗口
STABLE_MAX_MOVE_RATIO = 0.15     # bbox 中心移动 < bbox 宽度 15%
MIN_DURATION_SEC = 5.0           # 排泄至少持续 5 秒
MAX_GAP_SEC = 2.0
BBOX_HISTORY_LEN = 60            # 保留最近 60 帧(约 12 秒 @ 5FPS)


@dataclass
class BBoxSample:
    time: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2
    @property
    def w(self): return max(1e-6, self.box[2] - self.box[0])
    @property
    def h(self): return max(1e-6, self.box[3] - self.box[1])
    @property
    def aspect(self): return self.h / self.w


@dataclass
class ExcretionEvent:
    animal_key: str      # 用于追踪同一动物
    start_time: float
    last_seen: float
    hit: int = 0
    max_stability_frames: int = 0
    min_aspect: float = 999
    reason_snapshot: str = ""


class BBoxExcretionDetector:
    """基于 bbox 特征的排泄检测器

    使用方法:
        det = BBoxExcretionDetector()
        for frame_time, animals in stream:
            for a in animals:
                key = f"{a['cls']}-{track_id}"  # 或简单用 index
                result = det.update(key, a['box'], frame_time)
                if result and result['triggered']:
                    print(f"排泄事件: {result}")
    """

    def __init__(self):
        # 每只动物的 bbox 历史 + 事件状态
        self.history: dict = {}    # key -> deque[BBoxSample]
        self.ongoing: dict = {}    # key -> ExcretionEvent

    def _get_history(self, key) -> Deque[BBoxSample]:
        if key not in self.history:
            self.history[key] = deque(maxlen=BBOX_HISTORY_LEN)
        return self.history[key]

    def update(self, animal_key: str, box, now: float) -> dict:
        """
        每帧调 update,返回本帧对该动物的判定
        返回字段:
            in_ongoing_event: bool  (处于事件中)
            triggered_now: bool     (本帧刚触发/持续)
            just_finished: dict or None (完成的事件)
            metrics: dict           (调试信息)
        """
        h = self._get_history(animal_key)
        sample = BBoxSample(now, tuple(box))
        h.append(sample)

        # 需要至少 STABLE_WINDOW_SEC 秒历史才能判定
        window_samples = [s for s in h if now - s.time <= STABLE_WINDOW_SEC]

        metrics = {
            "aspect_now": sample.aspect,
            "window_len": len(window_samples),
        }

        # 特征 1:蹲姿(bbox 变压扁)
        aspect_crouched = sample.aspect < CROUCH_ASPECT_MAX
        metrics["aspect_crouched"] = aspect_crouched

        # 特征 2:稳定性(中心不动)
        stable = False
        max_move_ratio = 0
        if len(window_samples) >= 5:
            cx_arr = np.array([s.cx for s in window_samples])
            cy_arr = np.array([s.cy for s in window_samples])
            move_x = cx_arr.max() - cx_arr.min()
            move_y = cy_arr.max() - cy_arr.min()
            move = max(move_x, move_y)
            avg_w = np.mean([s.w for s in window_samples])
            max_move_ratio = move / max(1, avg_w)
            stable = max_move_ratio <= STABLE_MAX_MOVE_RATIO
        metrics["max_move_ratio"] = max_move_ratio
        metrics["stable"] = stable

        # 综合判定
        is_excreting_frame = aspect_crouched and stable

        result = {
            "in_ongoing_event": False,
            "triggered_now": False,
            "just_finished": None,
            "metrics": metrics,
        }

        # 事件状态机
        if is_excreting_frame:
            if animal_key not in self.ongoing:
                self.ongoing[animal_key] = ExcretionEvent(
                    animal_key=animal_key,
                    start_time=now, last_seen=now,
                    hit=1,
                    max_stability_frames=len(window_samples),
                    min_aspect=sample.aspect,
                    reason_snapshot=f"h/w={sample.aspect:.2f} 稳定")
            else:
                ev = self.ongoing[animal_key]
                ev.last_seen = now
                ev.hit += 1
                ev.max_stability_frames = max(
                    ev.max_stability_frames, len(window_samples))
                ev.min_aspect = min(ev.min_aspect, sample.aspect)
            result["in_ongoing_event"] = True
            result["triggered_now"] = True

        # 检查是否结束
        if animal_key in self.ongoing:
            ev = self.ongoing[animal_key]
            gap = now - ev.last_seen
            if gap >= MAX_GAP_SEC:
                dur = ev.last_seen - ev.start_time
                if dur >= MIN_DURATION_SEC:
                    result["just_finished"] = {
                        "animal_key": animal_key,
                        "start_time": ev.start_time,
                        "end_time": ev.last_seen,
                        "duration": dur,
                        "hit": ev.hit,
                        "min_aspect": ev.min_aspect,
                        "reason": ev.reason_snapshot,
                    }
                del self.ongoing[animal_key]
            else:
                result["in_ongoing_event"] = True

        return result

    def force_flush(self, now: float) -> List[dict]:
        finished = []
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_DURATION_SEC:
                finished.append({
                    "animal_key": key,
                    "start_time": ev.start_time,
                    "end_time": ev.last_seen,
                    "duration": dur,
                    "hit": ev.hit,
                    "min_aspect": ev.min_aspect,
                    "reason": ev.reason_snapshot,
                })
        self.ongoing.clear()
        return finished
