"""姿态服务 V2 · 真 SuperAnimal-Quadruped 集成

架构:
    主 Python 3.12 环境 (app.py, 播放器)
        ↓  HTTP 调用
    姿态微服务 (Python 3.11 + DLC 3.0)
        ↓
    SuperAnimal-Quadruped 模型
        ↓
    24 个关键点输出

为什么用微服务架构:
    - DLC 3.0 需要 numpy<2, PyTorch<2.5 (旧栈)
    - 主环境用最新 Ultralytics YOLOv8 (新栈)
    - 强行合并会冲突,拆开最干净

启动微服务:
    D:\venvs\dlc\Scripts\python pose_micro_service.py

调用:
    from pose_service_v2 import get_pose_service
    svc = get_pose_service()
    keypoints = svc.predict(image, bbox)   # 24 x (x, y, conf)
"""
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import requests


logger = logging.getLogger(__name__)


# ==================== SuperAnimal 39 关键点定义(DLC 3.0 官方) ====================
# 从 D:\venvs\dlc\Lib\site-packages\deeplabcut\modelzoo\project_configs\superanimal_quadruped.yaml 读取
# 注:"thai" 是官方笔误,实际意思是 "thigh" (大腿)
SUPERANIMAL_KEYPOINTS = [
    "nose",              # 0
    "upper_jaw",         # 1
    "lower_jaw",         # 2
    "mouth_end_right",   # 3
    "mouth_end_left",    # 4
    "right_eye",         # 5
    "right_earbase",     # 6
    "right_earend",      # 7
    "right_antler_base", # 8  (鹿角,猫狗无)
    "right_antler_end",  # 9
    "left_eye",          # 10
    "left_earbase",      # 11
    "left_earend",       # 12
    "left_antler_base",  # 13
    "left_antler_end",   # 14
    "neck_base",         # 15
    "neck_end",          # 16
    "throat_base",       # 17
    "throat_end",        # 18
    "back_base",         # 19  肩胛(前肩位)
    "back_end",          # 20  髋部(骨盆位)
    "back_middle",       # 21  腰
    "tail_base",         # 22  尾根
    "tail_end",          # 23  尾尖
    "front_left_thai",   # 24  前左大腿
    "front_left_knee",   # 25  前左膝
    "front_left_paw",    # 26  前左爪
    "front_right_thai",  # 27
    "front_right_knee",  # 28
    "front_right_paw",   # 29
    "back_left_paw",     # 30  ← 后左爪
    "back_left_thai",    # 31  ← 后左大腿(排泄核心特征)
    "back_right_thai",   # 32  ← 后右大腿
    "back_left_knee",    # 33  ← 后左膝
    "back_right_knee",   # 34
    "back_right_paw",    # 35
    "belly_bottom",      # 36  腹底
    "body_middle_right", # 37
    "body_middle_left",  # 38
]

# 骨架连线(基于真解剖学)
SKELETON_LINKS = [
    # 头部
    ("nose", "upper_jaw"),
    ("upper_jaw", "lower_jaw"),
    ("nose", "right_eye"),
    ("nose", "left_eye"),
    ("right_eye", "right_earbase"),
    ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"),
    ("left_earbase", "left_earend"),
    # 头 → 颈 → 背(脊柱)
    ("upper_jaw", "throat_base"),
    ("throat_base", "neck_base"),
    ("neck_base", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),
    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),
    # 前腿(左右)
    ("back_base", "front_left_thai"),
    ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("back_base", "front_right_thai"),
    ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    # 后腿(左右) ← 排泄识别主战场
    ("back_end", "back_left_thai"),
    ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"),
    ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
    # 身体侧线
    ("body_middle_right", "belly_bottom"),
    ("body_middle_left", "belly_bottom"),
]


class PoseServiceClient:
    """姿态微服务 HTTP 客户端(主 Python 用)

    ⚠ 关键: 用户机器上 Clash TUN 模式会劫持 127.0.0.1:8090 的响应回程
    所以必须用 Session + trust_env=False 强制绕过环境代理
    否则 pose 服务处理了 15 次请求, V11 只收到 1 次响应
    """

    def __init__(self, url: str = "http://127.0.0.1:8090"):
        self.url = url
        self._available = None
        # 独立 session, 忽略 HTTP_PROXY/HTTPS_PROXY 环境变量
        self._session = requests.Session()
        self._session.trust_env = False
        # 显式禁代理, 防止 Clash TUN 劫持 127.0.0.1
        self._session.proxies = {"http": "", "https": ""}

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                r = self._session.get(f"{self.url}/health", timeout=2)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        return self._available

    def predict(self, image_bgr: np.ndarray,
                bbox=None) -> Optional[np.ndarray]:
        """
        V2: 送整帧不裁 crop(避免 DLC detector 在小 crop 里找不到动物)

        输入:
            image_bgr: 原始 BGR 图 (H, W, 3)
            bbox: 忽略,DLC 内部会做 detection
        输出:
            keypoints: (39, 3) [x, y, confidence],绝对坐标
        """
        if not self.available:
            return None
        import cv2, base64
        # 送整帧
        _, buf = cv2.imencode(".jpg", image_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()

        try:
            r = self._session.post(
                f"{self.url}/predict",
                json={"image_b64": b64},
                timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            kps_list = data.get("keypoints", [])
            if not kps_list:
                return None
            return np.array(kps_list, dtype=np.float32)
        except Exception as e:
            logger.debug(f"pose 请求失败: {e}")
            return None


_client: Optional[PoseServiceClient] = None


def get_pose_service() -> PoseServiceClient:
    global _client
    if _client is None:
        _client = PoseServiceClient()
    return _client


# ==================== 姿态特征提取(排泄识别用) ====================
def compute_pose_features(keypoints) -> dict:
    """
    从姿态提取排泄识别的关键特征

    keypoints 可以是 numpy array 或 list,形状 (N, 3) [x, y, conf]
    """
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.ndim == 1:
        # 单个点 flat 数据 -> reshape
        if kps.size % 3 == 0:
            kps = kps.reshape(-1, 3)
    if kps.ndim != 2 or kps.shape[1] < 2:
        return {
            "valid": False,
            "hip_shoulder_dy": 0,
            "back_curvature": 0,
            "tail_raised": False,
        }

    N = kps.shape[0]

    def get_pt(kp_name, min_conf=0.3):
        """按关键点名称索引;不存在或置信度低返回 None"""
        try:
            i = SUPERANIMAL_KEYPOINTS.index(kp_name)
        except ValueError:
            return None
        if i >= N:
            return None
        pt = kps[i]
        if pt.shape[0] >= 3 and pt[2] < min_conf:
            return None
        return pt[:2]

    def first_available(*names):
        """按顺序找第一个可用的关键点"""
        for n in names:
            p = get_pt(n)
            if p is not None:
                return p
        return None

    # 尝试各种可能的"肩部"和"髋部"关键点
    shoulder = first_available("neck_base", "back_base", "withers")
    hip = first_available("back_end", "tail_base")
    back_middle = first_available("back_middle")
    tail_base = get_pt("tail_base")

    # ⭐ 后腿关键点(排泄识别核心)
    bl_thai = get_pt("back_left_thai")
    bl_knee = get_pt("back_left_knee")
    bl_paw = get_pt("back_left_paw")
    br_thai = get_pt("back_right_thai")
    br_knee = get_pt("back_right_knee")
    br_paw = get_pt("back_right_paw")

    # 前腿关键点(区分喝水 vs 排泄)
    fl_thai = get_pt("front_left_thai")
    fl_knee = get_pt("front_left_knee")
    fl_paw = get_pt("front_left_paw")
    fr_thai = get_pt("front_right_thai")
    fr_knee = get_pt("front_right_knee")
    fr_paw = get_pt("front_right_paw")

    if shoulder is None or hip is None:
        return {
            "valid": False,
            "hip_shoulder_dy": 0,
            "back_curvature": 0,
            "tail_raised": False,
            "rear_leg_angle": 180,
            "legs_bent": False,
        }

    hip_shoulder_dy = float(hip[1] - shoulder[1])

    # 背部曲率:三点 shoulder-back_middle-hip 偏离直线距离
    back_curvature = 0.0
    if back_middle is not None:
        v1 = np.asarray(shoulder, dtype=np.float32)
        v2 = np.asarray(back_middle, dtype=np.float32)
        v3 = np.asarray(hip, dtype=np.float32)
        line_len = float(np.linalg.norm(v3 - v1)) + 1e-6
        area = abs((v3[0] - v1[0]) * (v1[1] - v2[1]) -
                    (v1[0] - v2[0]) * (v3[1] - v1[1]))
        back_curvature = float(area / (line_len * line_len))

    # 尾巴翘起:tail_base y < hip y (即抬高)
    tail_raised = False
    if tail_base is not None:
        tail_raised = bool(float(tail_base[1]) < float(hip[1]) - 5)

    # ⭐ 后腿弯曲角度(排泄核心特征)
    def _angle_deg(p1, p2, p3):
        """三点夹角(p2 是顶点),返回度"""
        v1 = np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
        v2 = np.asarray(p3, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos = float(np.dot(v1, v2)) / (n1 * n2)
        cos = max(-1.0, min(1.0, cos))
        return float(np.degrees(np.arccos(cos)))

    # 左右后腿 thai-knee-paw 三点角度,180 = 直腿站,60-90 = 深蹲
    rear_angles = []
    if bl_thai is not None and bl_knee is not None and bl_paw is not None:
        rear_angles.append(_angle_deg(bl_thai, bl_knee, bl_paw))
    if br_thai is not None and br_knee is not None and br_paw is not None:
        rear_angles.append(_angle_deg(br_thai, br_knee, br_paw))
    rear_leg_angle = float(np.mean(rear_angles)) if rear_angles else 180.0
    legs_bent = rear_leg_angle < 110  # 排泄蹲姿

    # 前腿角度: 区分排泄 vs 喝水
    # 排泄: 前腿撑地基本直 (>150°)
    # 喝水: 前腿也弯 (<130°) 因为脖子够到低处食盆
    front_angles = []
    if fl_thai is not None and fl_knee is not None and fl_paw is not None:
        front_angles.append(_angle_deg(fl_thai, fl_knee, fl_paw))
    if fr_thai is not None and fr_knee is not None and fr_paw is not None:
        front_angles.append(_angle_deg(fr_thai, fr_knee, fr_paw))
    front_leg_angle = float(np.mean(front_angles)) if front_angles else 180.0
    # 前后腿不对称度: 排泄 = 前直后弯 (差值大), 喝水/坐 = 前后都弯 (差值小)
    front_rear_asymmetry = front_leg_angle - rear_leg_angle

    return {
        "valid": True,
        "hip_shoulder_dy": hip_shoulder_dy,
        "back_curvature": back_curvature,
        "tail_raised": tail_raised,
        "rear_leg_angle": rear_leg_angle,
        "legs_bent": legs_bent,
        "front_leg_angle": front_leg_angle,
        "front_rear_asymmetry": front_rear_asymmetry,
        "shoulder": (float(shoulder[0]), float(shoulder[1])),
        "hip": (float(hip[0]), float(hip[1])),
        "tail_base": (float(tail_base[0]), float(tail_base[1]))
            if tail_base is not None else None,
    }
