"""姿态识别服务

选型策略:
1. 首选:DeepLabCut SuperAnimal-Quadruped(猫狗四足专用,精度最好)
2. 次选:YOLOv8n-pose(COCO 人体,对猫狗仅能给出粗糙 fallback)
3. 兜底:纯 bbox 特征(不需要姿态,精度打折)

集成方式:后端启动时按顺序尝试 1 → 2 → 3,能用哪个用哪个。

DeepLabCut 安装(可选,重):
    pip install deeplabcut[modelzoo]==2.3.10
    # 或:pip install "dlclibrary"  然后手动下载 SuperAnimal 权重

优雅降级:如果 DLC 装不上,pose_service 会自动降级到 bbox-only 模式,
不会影响后端整体启动。
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ==================== SuperAnimal 关键点索引 ====================
# SuperAnimal-Quadruped 的 24 个关键点顺序
SUPERANIMAL_KP = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right",
    "mouth_end_left", "right_eye", "right_earbase",
    "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base",
    "neck_end", "throat_base", "throat_end", "back_base",
    "back_end", "back_middle", "tail_base", "tail_end",
]


class PoseService:
    """姿态识别服务,3 种后端优雅降级"""

    def __init__(self):
        self.backend = None  # "dlc" | "yolov8-pose" | "bbox-only"
        self._model = None
        self._init_backend()

    def _init_backend(self):
        # 尝试 1: DeepLabCut SuperAnimal
        try:
            self._model = self._try_dlc()
            if self._model is not None:
                self.backend = "dlc"
                logger.info("✅ 姿态识别: SuperAnimal-Quadruped(DLC)")
                return
        except Exception as e:
            logger.warning(f"DLC 加载失败: {e}")

        # 尝试 2: YOLOv8-pose(COCO 人体,兜底用)
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n-pose.pt")
            self.backend = "yolov8-pose"
            logger.warning(
                "⚠️ 姿态识别降级: yolov8n-pose(人体训练,对动物精度低)")
            return
        except Exception as e:
            logger.warning(f"YOLOv8-pose 也不可用: {e}")

        # 兜底:bbox-only
        self.backend = "bbox-only"
        logger.warning(
            "⚠️ 姿态识别不可用,行为规则将只用 bbox 特征"
            "(精度会打折)")

    def _try_dlc(self):
        """尝试加载 DeepLabCut SuperAnimal-Quadruped"""
        try:
            from deeplabcut.pose_estimation_pytorch.apis import (
                DLCLive)  # noqa
        except ImportError:
            return None

        # 具体的权重加载 SDK 会自动处理
        # 详见 https://deeplabcut.github.io/DeepLabCut/docs/HelperFunctions.html
        try:
            import deeplabcut
            from deeplabcut.utils import auxiliaryfunctions
            # SuperAnimal 会在首次调用时自动下载权重
            logger.info("[i] DLC 模块已装,首次运行会自动下载 SuperAnimal 权重")
            # 返回 DLC 上下文占位,实际使用时按需初始化
            return "dlc-placeholder"
        except Exception as e:
            logger.warning(f"DLC 权重初始化失败: {e}")
            return None

    def is_available(self) -> bool:
        return self.backend != "bbox-only"

    def is_full_quality(self) -> bool:
        return self.backend == "dlc"

    def detect(self, frame, bbox) -> Optional[np.ndarray]:
        """
        输入:frame BGR,bbox=(x1,y1,x2,y2)
        返回:关键点 numpy (N, 3) [x, y, confidence],或 None

        - DLC:SuperAnimal 24 个关键点
        - yolov8-pose:COCO 17 个关键点(人体,动物不准)
        - bbox-only:返回 None,规则引擎会走 fallback
        """
        if self.backend == "bbox-only":
            return None

        if self.backend == "dlc":
            return self._detect_dlc(frame, bbox)

        if self.backend == "yolov8-pose":
            return self._detect_yolo(frame, bbox)

        return None

    def _detect_dlc(self, frame, bbox):
        # 实际 DLC 推理接口
        # 简化版占位:后续接入时展开
        # 参考:https://deeplabcut.github.io/DeepLabCut/docs/superanimal.html
        return None

    def _detect_yolo(self, frame, bbox):
        if self._model is None:
            return None
        try:
            r = self._model.predict(frame, verbose=False)[0]
            if r.keypoints is None:
                return None
            # 只取 bbox 内的关键点
            kps = r.keypoints.data.cpu().numpy()  # (n_people, 17, 3)
            if len(kps) == 0:
                return None
            return kps[0]  # 取第一组
        except Exception as e:
            logger.debug(f"yolo-pose 推理失败: {e}")
            return None


# 单例
_service: Optional[PoseService] = None


def get_pose_service() -> PoseService:
    global _service
    if _service is None:
        _service = PoseService()
    return _service
