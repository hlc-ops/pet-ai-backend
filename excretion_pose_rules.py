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
# 主要特征
REAR_LEG_ANGLE_MAX = 100         # 后腿角度小于此值 = 深蹲 (直腿=180, 深蹲=60-90)
HIP_SHOULDER_DY_MIN = 15         # 髋比肩低多少像素才算"蹲下"
BACK_CURVATURE_MIN = 0.015       # 背部弯曲阈值

# 各特征评分权重
SCORE_REAR_LEG = 35              # ⭐ 后腿深蹲 = 排泄最强信号
SCORE_HIP_DROP = 25              # 髋部下沉
SCORE_BACK_CURVE = 15            # 背部弓形
SCORE_STABLE = 15                # 位置稳定
TAIL_RAISED_BONUS_CAT = 15       # cat 尾巴翘起加分
TAIL_RAISED_BONUS_DOG = 5        # dog 尾巴翘起(dog 常自然下垂)
NO_BOWL_BONUS = 10               # 现场无盆 = 更可能是排泄
# ⭐ 前后腿不对称度:排泄 = 前直后弯 (>30°), 喝水/坐 = 前后都弯 (~0)
SCORE_ASYMMETRY = 25             # 前后腿不对称加分
FRONT_LEG_STRAIGHT_MIN = 150     # 前腿直立最低角度
FRONT_REAR_ASYMMETRY_MIN = 30    # 前后腿角度差最小值

SCORE_THRESHOLD = 60             # 无盆时综合分阈值
SCORE_THRESHOLD_WITH_BOWL = 90   # 有盆时更严 (避免"猫坐着喝水"误判为排泄)
STABLE_WINDOW_SEC = 3.0
STABLE_MAX_MOVE_PX = 25
MIN_DURATION_SEC = 4.0           # 3.0 → 4.0: 短暂坐姿/低头喝水不算排泄
MAX_GAP_SEC = 6.0
HISTORY_LEN = 90


@dataclass
class PoseSample:
    time: float
    hip_shoulder_dy: float
    back_curvature: float
    tail_raised: bool
    rear_leg_angle: float
    legs_bent: bool
    front_leg_angle: float
    front_rear_asymmetry: float
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
    min_rear_leg_angle: float = 180.0


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
               now: float, animal_cls: str = "dog",
               has_bowl_nearby: bool = False) -> dict:
        """
        输入:
            animal_key: 唯一动物 ID
            keypoints: SuperAnimal 39 点 (N, 3) [x, y, conf]
            now: 帧时间戳
            animal_cls: cat / dog(尾巴权重不同)
            has_bowl_nearby: 现场有盆? 有的话就是喝水/进食不是排泄
        """
        features = compute_pose_features(keypoints)

        result = {
            "in_ongoing_event": False,
            "triggered_now": False,
            "just_finished": None,
            "score": 0,
            "reasons": [],
            "features": features,
            "strong_excretion_pose": False,
        }

        if not features["valid"]:
            return result

        sample = PoseSample(
            time=now,
            hip_shoulder_dy=features["hip_shoulder_dy"],
            back_curvature=features["back_curvature"],
            tail_raised=features["tail_raised"],
            rear_leg_angle=features.get("rear_leg_angle", 180.0),
            legs_bent=features.get("legs_bent", False),
            front_leg_angle=features.get("front_leg_angle", 180.0),
            front_rear_asymmetry=features.get("front_rear_asymmetry", 0.0),
            shoulder=features["shoulder"],
            hip=features["hip"],
        )
        h = self._hist(animal_key)
        h.append(sample)

        # ==================== 综合评分 ====================
        score = 0
        reasons = []

        # ⭐ 特征 1: 后腿深蹲(最强信号)
        if sample.rear_leg_angle <= REAR_LEG_ANGLE_MAX:
            score += SCORE_REAR_LEG
            reasons.append(f"后腿深蹲 {sample.rear_leg_angle:.0f}°")

        # 特征 2: 髋部下沉
        if sample.hip_shoulder_dy >= HIP_SHOULDER_DY_MIN:
            score += SCORE_HIP_DROP
            reasons.append(
                f"髋低于肩 Δy={sample.hip_shoulder_dy:.0f}px")

        # 特征 3: 背部弓形
        if sample.back_curvature >= BACK_CURVATURE_MIN:
            score += SCORE_BACK_CURVE
            reasons.append(f"背弓 {sample.back_curvature:.3f}")

        # 特征 4: 尾巴翘起(cat 权重高)
        if sample.tail_raised:
            bonus = TAIL_RAISED_BONUS_CAT if animal_cls == "cat" \
                else TAIL_RAISED_BONUS_DOG
            score += bonus
            reasons.append(f"尾抬 +{bonus}")

        # 特征 5: 位置稳定
        window = [s for s in h if now - s.time <= STABLE_WINDOW_SEC]
        if len(window) >= 3:
            hip_ys = [s.hip[1] for s in window]
            hip_xs = [s.hip[0] for s in window]
            move = max(np.ptp(hip_xs), np.ptp(hip_ys))
            if move <= STABLE_MAX_MOVE_PX:
                score += SCORE_STABLE
                reasons.append(f"稳定 {len(window)}帧 move={move:.0f}px")

        # ⭐ 特征 6: 前后腿不对称 (排泄专属)
        # 前腿直 (>=150°) + 前后差 >= 30° = 排泄姿态
        # 前后腿都弯 = 喝水/坐,不对称度小
        if (sample.front_leg_angle >= FRONT_LEG_STRAIGHT_MIN
                and sample.front_rear_asymmetry >= FRONT_REAR_ASYMMETRY_MIN):
            score += SCORE_ASYMMETRY
            reasons.append(
                f"前直后弯 asym={sample.front_rear_asymmetry:.0f}°")
        elif sample.front_leg_angle < 130:
            # 前腿明显弯曲 → 高度怀疑喝水/趴, 扣分
            score -= 15
            reasons.append(f"前腿也弯 -15 ({sample.front_leg_angle:.0f}°)")

        # ⭐ 特征 7: 现场无盆(排除喝水/进食)
        # 若姿态铁证如山(强排泄证据), 忽略 bowl 扣分
        # 排泄姿态: 后腿深弯(猫弓背深蹲/狗后腿岔开) + 前腿撑直 + 髋下沉
        strong_excretion = (
            sample.rear_leg_angle <= 90       # 后腿深弯 (蹲 or 岔开)
            and sample.front_leg_angle >= 150  # 前腿撑直
            and sample.hip_shoulder_dy >= 15   # 髋低于肩
        )
        result["strong_excretion_pose"] = strong_excretion

        if has_bowl_nearby:
            if strong_excretion:
                reasons.append("[强排泄证据] 忽略 bowl 扣分")
            else:
                score -= 20
                reasons.append("现场有盆 -20")
        else:
            score += NO_BOWL_BONUS
            reasons.append(f"无盆 +{NO_BOWL_BONUS}")

        result["score"] = score
        result["reasons"] = reasons
        # 有盆时门槛更严 (避免坐姿喝水误判排泄), 除非姿态铁证
        threshold = SCORE_THRESHOLD_WITH_BOWL if has_bowl_nearby else SCORE_THRESHOLD
        is_excreting = score >= threshold

        # 事件状态机
        if is_excreting:
            if animal_key not in self.ongoing:
                self.ongoing[animal_key] = PoseExcretionEvent(
                    animal_key=animal_key,
                    animal_cls=animal_cls,
                    start_time=now, last_seen=now,
                    hit=1, max_score=score,
                    max_hip_dy=sample.hip_shoulder_dy,
                    max_back_curv=sample.back_curvature,
                    min_rear_leg_angle=sample.rear_leg_angle)
            else:
                ev = self.ongoing[animal_key]
                ev.last_seen = now
                ev.hit += 1
                ev.max_score = max(ev.max_score, score)
                ev.max_hip_dy = max(ev.max_hip_dy, sample.hip_shoulder_dy)
                ev.max_back_curv = max(ev.max_back_curv,
                                        sample.back_curvature)
                ev.min_rear_leg_angle = min(ev.min_rear_leg_angle,
                                             sample.rear_leg_angle)
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
                        "min_rear_leg_angle": ev.min_rear_leg_angle,
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
                    "max_hip_dy": ev.max_hip_dy,
                    "max_back_curv": ev.max_back_curv,
                    "min_rear_leg_angle": ev.min_rear_leg_angle,
                })
        self.ongoing.clear()
        return finished
