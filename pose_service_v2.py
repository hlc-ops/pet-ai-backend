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


# ==================== SuperAnimal 24 关键点定义 ====================
SUPERANIMAL_KEYPOINTS = [
    "nose",             # 0
    "upper_jaw",        # 1
    "lower_jaw",        # 2
    "mouth_end_right",  # 3
    "mouth_end_left",   # 4
    "right_eye",        # 5
    "right_earbase",    # 6
    "right_earend",     # 7
    "right_antler_base",# 8
    "right_antler_end", # 9
    "left_eye",         # 10
    "left_earbase",     # 11
    "left_earend",      # 12
    "left_antler_base", # 13
    "left_antler_end",  # 14
    "neck_base",        # 15
    "neck_end",         # 16
    "throat_base",      # 17
    "throat_end",       # 18
    "back_base",        # 19
    "back_end",         # 20
    "back_middle",      # 21
    "tail_base",        # 22
    "tail_end",         # 23
]

# 骨架连线(可视化用)
SKELETON_LINKS = [
    # 头部
    ("nose", "upper_jaw"),
    ("upper_jaw", "lower_jaw"),
    ("upper_jaw", "right_eye"),
    ("upper_jaw", "left_eye"),
    ("right_eye", "right_earbase"),
    ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"),
    ("left_earbase", "left_earend"),
    # 脖子背部
    ("nose", "neck_base"),
    ("neck_base", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),
    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),
    # 喉咙(前颈)
    ("neck_base", "throat_base"),
    ("throat_base", "throat_end"),
]


class PoseServiceClient:
    """姿态微服务 HTTP 客户端(主 Python 用)"""

    def __init__(self, url: str = "http://127.0.0.1:8090"):
        self.url = url
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                r = requests.get(f"{self.url}/health", timeout=1)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        return self._available

    def predict(self, image_bgr: np.ndarray,
                bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        输入:
            image_bgr: 原始 BGR 图 (H, W, 3)
            bbox: (x1, y1, x2, y2) 动物边界框
        输出:
            keypoints: (24, 3) [x, y, confidence],绝对坐标
            或 None(服务不可用/失败)
        """
        if not self.available:
            return None
        # 裁 bbox 内区域
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        crop = image_bgr[y1:y2, x1:x2]

        # 编码 + POST
        import cv2, base64
        _, buf = cv2.imencode(".jpg", crop,
                              [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf).decode()

        try:
            r = requests.post(
                f"{self.url}/predict",
                json={"image_b64": b64},
                timeout=5)
            if r.status_code != 200:
                return None
            data = r.json()
            kps = np.array(data["keypoints"])  # (24, 3) 相对 crop
            # 转成绝对坐标
            kps[:, 0] += x1
            kps[:, 1] += y1
            return kps
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
def compute_pose_features(keypoints: np.ndarray) -> dict:
    """
    从 24 点姿态提取排泄识别的关键特征

    返回:
        hip_shoulder_dy: 髋-肩 y 差(髋比肩低 = 蹲)
        back_curvature: 背部弯曲度
        rear_leg_bent: 后腿是否弯曲(通过 back_end vs tail_base 角度近似)
        tail_raised: 尾巴翘起(cat 主要)
        valid: 特征是否可信(关键点存在)
    """
    def get(idx, min_conf=0.3):
        if idx >= len(keypoints) or keypoints[idx, 2] < min_conf:
            return None
        return keypoints[idx][:2]

    idx = SUPERANIMAL_KEYPOINTS.index
    nose = get(idx("nose"))
    neck_base = get(idx("neck_base"))
    back_base = get(idx("back_base"))
    back_middle = get(idx("back_middle"))
    back_end = get(idx("back_end"))
    tail_base = get(idx("tail_base"))
    tail_end = get(idx("tail_end"))

    # 用 back_base 代替肩(neck_base 顶部)
    # 用 back_end 代替髋(靠近尾根)
    shoulder = neck_base or back_base
    hip = back_end
    if shoulder is None or hip is None:
        return {
            "valid": False,
            "hip_shoulder_dy": 0,
            "back_curvature": 0,
            "tail_raised": False,
        }

    hip_shoulder_dy = hip[1] - shoulder[1]

    # 背部曲率:三点 shoulder-back_middle-hip 偏离直线距离
    back_curvature = 0
    if back_middle is not None:
        v1 = np.array(shoulder)
        v2 = np.array(back_middle)
        v3 = np.array(hip)
        # 点 v2 到线 v1v3 的距离
        line_len = np.linalg.norm(v3 - v1) + 1e-6
        area = abs((v3[0] - v1[0]) * (v1[1] - v2[1]) -
                    (v1[0] - v2[0]) * (v3[1] - v1[1]))
        back_curvature = area / line_len / line_len

    # 尾巴翘起:tail_base y < hip y (即抬高)
    tail_raised = False
    if tail_base is not None and hip is not None:
        tail_raised = tail_base[1] < hip[1] - 5

    return {
        "valid": True,
        "hip_shoulder_dy": float(hip_shoulder_dy),
        "back_curvature": float(back_curvature),
        "tail_raised": bool(tail_raised),
        "shoulder": tuple(shoulder),
        "hip": tuple(hip),
        "tail_base": tuple(tail_base) if tail_base is not None else None,
    }
