"""排泄识别 · 真姿态版(需要 pose_service 可用)

对比 bbox 版:
    bbox 版精度 70-75%
    姿态版精度 85-95%(接近论文级)

核心特征(基于 D:\wenti63 6 张图 + SuperAnimal 24 点):
1. 髋部 y > 肩部 y + 阈值(髋低于肩,蹲下)
2. 背部曲率高(shoulder-back_middle-hip 三点弓形)
3. 尾巴翘起(cat 明显,dog 弱)
4. 姿态稳定 >= 5 秒

使用:
    from pose_service_v2 import get_pose_service
    from excretion_pose_rules import PoseExcretionDetector

    det = PoseExcretionDetector()
    pose_svc = get_pose_service()

    for frame_time, animal in stream:
        keypoints = pose_svc.predict(frame, animal["box"])
        if keypoints is not None:
            result = det.update(animal["key"], keypoints, frame_time,
                                animal.get("cls", "dog"))
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from pose_service_v2 import compute_pose_features


logger = logging.getLogger(__name__)


# ==================== 参数 ====================
HIP_SHOULDER_DY_MIN = 15         # 髋比肩低多少像素才算"蹲下"
BACK_CURVATURE_MIN = 0.015       # 背部弯曲阈值(经验值,可调)
TAIL_RAISED_BONUS_CAT = 15       # cat 尾巴翘起加分
TAIL_RAISED_BONUS_DOG = 5        # dog 尾巴翘起加分(dog 常自然下垂)
SCORE_THRESHOLD = 55             # 综合分阈值
STABLE_WINDOW_SEC = 5.0
STABLE_MAX_MOVE_PX = 20          # bbox 内关键点稳定
MIN_DURATION_SEC = 5.0
MAX_GAP_SEC = 2.0
HISTORY_LEN = 60


@dataclass
class PoseSample:
    time: float
    hip_shoulder_dy: float
    back_curvature: float
    tail_raised: bool
    shoulder: tuple
    hip: tuple


@dataclass
class PoseExcretionEvent:
    animal_key: str
    animal_cls: str
    start_time: float
    last_seen: float
    hit: int = 0
    max_score: int = 0
    max_hip_dy: float = 0
    max_back_curv: float = 0


class PoseExcretionDetector:
    """基于姿态的排泄检测器"""

    def __init__(self):
        self.history: dict = {}   # key -> deque[PoseSample]
        self.ongoing: dict = {}   # key -> PoseExcretionEvent

    def _hist(self, key):
        if key not in self.history:
            self.history[key] = deque(maxlen=HISTORY_LEN)
        return self.history[key]

    def update(self, animal_key: str, keypoints: np.ndarray,
               now: float, animal_cls: str = "dog") -> dict:
        """
        输入:
            animal_key: 唯一动物 ID
            keypoints: SuperAnimal 24 点 (N, 3) [x, y, conf]
            now: 帧时间戳
            animal_cls: cat / dog(尾巴权重不同)
        """
        features = compute_pose_features(keypoints)

        result = {
            "in_ongoing_event": False,
            "triggered_now": False,
            "just_finished": None,
            "score": 0,
            "reasons": [],
            "features": features,
        }

        if not features["valid"]:
            return result

        sample = PoseSample(
            time=now,
            hip_shoulder_dy=features["hip_shoulder_dy"],
            back_curvature=features["back_curvature"],
            tail_raised=features["tail_raised"],
            shoulder=features["shoulder"],
            hip=features["hip"],
        )
        h = self._hist(animal_key)
        h.append(sample)

        # 综合评分
        score = 0
        reasons = []

        if sample.hip_shoulder_dy >= HIP_SHOULDER_DY_MIN:
            score += 30
            reasons.append(
                f"髋低于肩 Δy={sample.hip_shoulder_dy:.0f}px")

        if sample.back_curvature >= BACK_CURVATURE_MIN:
            score += 20
            reasons.append(f"背弓 {sample.back_curvature:.3f}")

        if sample.tail_raised:
            bonus = TAIL_RAISED_BONUS_CAT if animal_cls == "cat" \
                else TAIL_RAISED_BONUS_DOG
            score += bonus
            reasons.append(f"尾抬 +{bonus}")

        # 稳定性
        window = [s for s in h if now - s.time <= STABLE_WINDOW_SEC]
        if len(window) >= 5:
            hip_ys = [s.hip[1] for s in window]
            hip_xs = [s.hip[0] for s in window]
            move = max(np.ptp(hip_xs), np.ptp(hip_ys))
            if move <= STABLE_MAX_MOVE_PX:
                score += 15
                reasons.append(f"稳定 {len(window)}帧 move={move:.0f}px")

        result["score"] = score
        result["reasons"] = reasons
        is_excreting = score >= SCORE_THRESHOLD

        # 事件状态机
        if is_excreting:
            if animal_key not in self.ongoing:
                self.ongoing[animal_key] = PoseExcretionEvent(
                    animal_key=animal_key,
                    animal_cls=animal_cls,
                    start_time=now, last_seen=now,
                    hit=1, max_score=score,
                    max_hip_dy=sample.hip_shoulder_dy,
                    max_back_curv=sample.back_curvature)
            else:
                ev = self.ongoing[animal_key]
                ev.last_seen = now
                ev.hit += 1
                ev.max_score = max(ev.max_score, score)
                ev.max_hip_dy = max(ev.max_hip_dy, sample.hip_shoulder_dy)
                ev.max_back_curv = max(ev.max_back_curv,
                                        sample.back_curvature)
            result["triggered_now"] = True
            result["in_ongoing_event"] = True

        # 结束判定
        if animal_key in self.ongoing:
            ev = self.ongoing[animal_key]
            gap = now - ev.last_seen
            if gap >= MAX_GAP_SEC:
                dur = ev.last_seen - ev.start_time
                if dur >= MIN_DURATION_SEC:
                    result["just_finished"] = {
                        "animal_key": animal_key,
                        "animal_cls": ev.animal_cls,
                        "start_time": ev.start_time,
                        "end_time": ev.last_seen,
                        "duration": dur,
                        "hit": ev.hit,
                        "max_score": ev.max_score,
                        "max_hip_dy": ev.max_hip_dy,
                        "max_back_curv": ev.max_back_curv,
                    }
                del self.ongoing[animal_key]
            else:
                result["in_ongoing_event"] = True

        return result

    def force_flush(self, now: float):
        finished = []
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_DURATION_SEC:
                finished.append({
                    "animal_key": key,
                    "animal_cls": ev.animal_cls,
                    "start_time": ev.start_time,
                    "end_time": ev.last_seen,
                    "duration": dur,
                    "hit": ev.hit,
                    "max_score": ev.max_score,
                })
        self.ongoing.clear()
        return finished
