"""排泄(excretion)行为识别规则

基于姿态关键点 + 几何特征判定。**必须先接入 SuperAnimal-Quadruped 得到关键点**。

关键特征(来自 8 张典型排泄图分析):
1. 髋部 y 坐标 > 肩部 y 坐标(后半身低于前半身,鲁棒特征)
2. 后腿蹲姿:髋-膝-踝角度 < 100°
3. 背部弓形:shoulder-hip-tail 三点非直线,曲率明显
4. 尾巴位置:
   - 猫:通常明显上翘(tail_base y < hip y)
   - 狗:自然下垂或微翘,不作强约束
5. 头部姿势:相对水平,不是深俯(区别于吃/喝)
6. 姿态稳定:排泄通常 5-30 秒,姿态帧间波动小

集成方式:
    from excretion_rules import ExcretionDetector
    detector = ExcretionDetector(animal_class='dog')
    # 每帧调用:
    is_excreting, score, reasons = detector.detect(keypoints, frame_time)
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ==================== 关键点索引(以 SuperAnimal-Quadruped 为例) ====================
# 实际接入时按你选的 pose 模型的 keypoint 顺序调整
KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "withers": 3,        # 肩胛骨(shoulder)
    "l_shoulder": 4,
    "r_shoulder": 5,
    "throat": 6,
    "l_hip": 7,
    "r_hip": 8,
    "tail_base": 9,
    "l_knee": 10,
    "l_ankle": 11,
    "r_knee": 12,
    "r_ankle": 13,
    "l_paw_front": 14,
    "r_paw_front": 15,
}


# ==================== 阈值(可调) ====================
# 来自 8 张图的经验值,实际部署需按测试视频微调
class ExcretionThresholds:
    # 髋部相对肩部下沉(髋 y 比肩 y 大多少)
    HIP_BELOW_SHOULDER_PX = 5     # 像素,或用比例更好
    HIP_BELOW_SHOULDER_RATIO = 0.05  # bbox 高度的 5%

    # 后腿角度(髋-膝-踝)—— 蹲姿时应 < 100°
    LEG_BEND_ANGLE_DEG = 100

    # 背部弓形曲率(shoulder-hip-tail 偏离直线的比例)
    BACK_CURVATURE = 0.10

    # 尾巴翘起(仅猫用)
    TAIL_RAISED_PX_RATIO = 0.03  # bbox 高度的 3%

    # 头部水平(0=水平,±90=垂直)
    HEAD_TILT_DEG = 45  # 头俯 > 45° = 在吃/喝,不算排泄

    # 姿态稳定性:相邻帧关键点变化 <= 该值算稳定
    STABILITY_PX_TOL = 15
    STABILITY_MIN_FRAMES = 15  # 至少 15 帧稳定(5 秒 @ 3 FPS)

    # 综合分阈值(0-100)
    SCORE_THRESHOLD = 60


# ==================== 几何工具 ====================
def angle_deg(p1, p2, p3) -> float:
    """三点夹角(p2 是顶点),返回度"""
    if p1 is None or p2 is None or p3 is None:
        return 180.0
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def curvature_3points(p1, p2, p3) -> float:
    """三点连线的曲率(0 = 直线, 越大越弯)。返回 p2 到 p1p3 直线的距离 / p1p3 长度"""
    if p1 is None or p2 is None or p3 is None:
        return 0.0
    # 点到直线距离公式
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    num = abs((y3 - y1) * x2 - (x3 - x1) * y2 + x3 * y1 - y3 * x1)
    den = math.hypot(y3 - y1, x3 - x1) or 1e-6
    d = num / den
    line_len = math.hypot(y3 - y1, x3 - x1) or 1e-6
    return d / line_len


def kp_visible(kp_arr, idx) -> bool:
    """关键点是否可见(SuperAnimal 输出的 confidence > 阈值)"""
    if kp_arr is None or len(kp_arr) <= idx:
        return False
    if len(kp_arr[idx]) >= 3:
        return kp_arr[idx][2] > 0.3
    return True


def get_kp(kp_arr, idx) -> Optional[Tuple[float, float]]:
    """安全取点,不可见返回 None"""
    if not kp_visible(kp_arr, idx):
        return None
    return (kp_arr[idx][0], kp_arr[idx][1])


# ==================== 主检测器 ====================
@dataclass
class FeatureSnapshot:
    """一帧的关键特征"""
    hip_below_shoulder: bool
    hip_shoulder_delta_ratio: float
    avg_leg_angle: float
    legs_bent: bool
    back_curvature: float
    arched_back: bool
    tail_raised: bool
    head_tilt_deg: float
    head_not_diving: bool
    valid: bool  # 关键点齐全


class ExcretionDetector:
    """
    对单个动物实例(track_id / bbox)维护姿态历史,判定是否在排泄。

    使用方法:
        detector = ExcretionDetector(animal_class='dog', bbox_height=200)
        for frame_time, keypoints in stream:
            result = detector.update(keypoints, frame_time)
            if result.is_excreting:
                emit_event(...)
    """
    def __init__(self, animal_class: str,
                 bbox_height: float = 200,
                 thresholds: ExcretionThresholds = None):
        self.animal_class = animal_class.lower()
        self.bbox_height = bbox_height
        self.thresh = thresholds or ExcretionThresholds()
        self._history: List[FeatureSnapshot] = []
        self._recent_kps: List[np.ndarray] = []
        self._triggered = False
        self._start_time: Optional[float] = None
        self._last_trigger_time: Optional[float] = None

    def update(self, keypoints, frame_time: float) -> "ExcretionResult":
        """
        keypoints: shape (N, 2) 或 (N, 3),SuperAnimal 输出的关键点
        frame_time: 视频时间戳(秒)
        """
        snap = self._extract_features(keypoints)
        self._history.append(snap)
        if len(self._history) > 60:
            self._history = self._history[-60:]

        self._recent_kps.append(np.asarray(keypoints))
        if len(self._recent_kps) > 30:
            self._recent_kps = self._recent_kps[-30:]

        if not snap.valid:
            return ExcretionResult(False, 0, ["关键点不齐全"])

        # 计算综合分
        score = 0
        reasons: List[str] = []

        if snap.hip_below_shoulder:
            score += 25
            reasons.append(
                f"髋低于肩 (Δ={snap.hip_shoulder_delta_ratio:.2%})")

        if snap.legs_bent:
            score += 20
            reasons.append(f"后腿蹲 ({snap.avg_leg_angle:.0f}°)")

        if snap.arched_back:
            score += 20
            reasons.append(f"背弓 ({snap.back_curvature:.2f})")

        # 尾巴:只对猫强判(狗常自然下垂,不作硬要求)
        if self.animal_class == "cat" and snap.tail_raised:
            score += 15
            reasons.append("猫尾巴翘起")
        elif self.animal_class == "dog" and snap.tail_raised:
            score += 5
            reasons.append("狗尾巴微翘")

        if snap.head_not_diving:
            score += 10
            reasons.append("头不俯冲(区别于吃/喝)")

        # 稳定性
        stable_frames = self._compute_stability_frames()
        if stable_frames >= self.thresh.STABILITY_MIN_FRAMES:
            score += 10
            reasons.append(f"姿态稳定 {stable_frames} 帧")

        is_excreting = score >= self.thresh.SCORE_THRESHOLD
        return ExcretionResult(is_excreting, score, reasons, snap)

    # ---------- 内部特征提取 ----------
    def _extract_features(self, keypoints) -> FeatureSnapshot:
        withers = get_kp(keypoints, KP["withers"])
        l_hip = get_kp(keypoints, KP["l_hip"])
        r_hip = get_kp(keypoints, KP["r_hip"])
        tail = get_kp(keypoints, KP["tail_base"])
        nose = get_kp(keypoints, KP["nose"])
        l_knee = get_kp(keypoints, KP["l_knee"])
        l_ankle = get_kp(keypoints, KP["l_ankle"])
        r_knee = get_kp(keypoints, KP["r_knee"])
        r_ankle = get_kp(keypoints, KP["r_ankle"])

        hip = midpoint(l_hip, r_hip)

        # 若基础关键点缺失,标 invalid
        if withers is None or hip is None or tail is None:
            return FeatureSnapshot(
                False, 0, 180, False, 0, False, False, 0, False, valid=False)

        # 特征 1:髋部低于肩部
        # 图像坐标 y 向下,所以髋 y 更大 = 更低
        delta_y = hip[1] - withers[1]
        delta_ratio = delta_y / max(1, self.bbox_height)
        hip_below = delta_ratio >= self.thresh.HIP_BELOW_SHOULDER_RATIO

        # 特征 2:后腿角度
        l_angle = angle_deg(l_hip, l_knee, l_ankle) \
            if (l_hip and l_knee and l_ankle) else 180
        r_angle = angle_deg(r_hip, r_knee, r_ankle) \
            if (r_hip and r_knee and r_ankle) else 180
        avg_angle = (l_angle + r_angle) / 2
        legs_bent = avg_angle < self.thresh.LEG_BEND_ANGLE_DEG

        # 特征 3:背部弓形
        curvature = curvature_3points(withers, hip, tail)
        arched = curvature > self.thresh.BACK_CURVATURE

        # 特征 4:尾巴翘起(tail y 比 hip y 小)
        tail_delta = hip[1] - tail[1]
        tail_raised = (tail_delta / max(1, self.bbox_height)) > \
            self.thresh.TAIL_RAISED_PX_RATIO

        # 特征 5:头部角度(nose 相对 withers 的连线)
        head_tilt = 0
        if nose is not None:
            dx = nose[0] - withers[0]
            dy = nose[1] - withers[1]
            head_tilt = abs(math.degrees(math.atan2(dy, dx)))
        head_not_diving = head_tilt < self.thresh.HEAD_TILT_DEG

        return FeatureSnapshot(
            hip_below_shoulder=hip_below,
            hip_shoulder_delta_ratio=delta_ratio,
            avg_leg_angle=avg_angle,
            legs_bent=legs_bent,
            back_curvature=curvature,
            arched_back=arched,
            tail_raised=tail_raised,
            head_tilt_deg=head_tilt,
            head_not_diving=head_not_diving,
            valid=True,
        )

    def _compute_stability_frames(self) -> int:
        """从最近历史往前数,姿态相似的连续帧数"""
        if len(self._recent_kps) < 2:
            return 0
        latest = self._recent_kps[-1]
        stable = 1
        for prev in reversed(self._recent_kps[:-1]):
            if prev.shape != latest.shape:
                break
            # 比较 x, y 变化
            xy_prev = prev[:, :2] if prev.shape[1] >= 2 else prev
            xy_latest = latest[:, :2] if latest.shape[1] >= 2 else latest
            max_delta = np.max(np.abs(xy_prev - xy_latest))
            if max_delta > self.thresh.STABILITY_PX_TOL:
                break
            stable += 1
        return stable


@dataclass
class ExcretionResult:
    is_excreting: bool
    score: int
    reasons: List[str]
    features: Optional[FeatureSnapshot] = None
