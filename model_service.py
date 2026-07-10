"""模型服务:封装 YOLO 加载 + 推理 + 结果解析。

对外只暴露 detect(frame) 方法,返回统一格式的结果,
不管底层是训练好的 best.pt 还是 COCO 占位模型。
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from ultralytics import YOLO

from config import Config


logger = logging.getLogger(__name__)


class DetectionResult:
    """一帧的检测结果"""
    def __init__(self):
        self.animals: List[Dict] = []
        self.bowls: List[Dict] = []
        # animals[i] = {"cls": "cat", "box": [x1,y1,x2,y2], "conf": 0.9,
        #               "mask": <optional polygon points>}
        # bowls[i]   = {"box": [x1,y1,x2,y2], "conf": 0.85, "mask": ...}


class ModelService:
    """YOLO 推理封装。

    构造后就可以调 detect(frame) 得到 DetectionResult。
    如果 MODEL_PATH 不存在,自动回退到 COCO 占位模型。
    """
    def __init__(self, config: Config = Config):
        self.config = config
        self.model = None
        self.mode = None  # "trained" | "coco_placeholder"
        self._name_to_id = {}   # 项目类别名 -> 模型 class id
        self._load()

    def _load(self):
        model_path = Path(self.config.MODEL_PATH)
        if model_path.exists() and not self.config.USE_COCO_MAPPING:
            # 用户训练的模型
            logger.info(f"加载训练模型: {model_path}")
            self.model = YOLO(str(model_path))
            self.mode = "trained"
            # 项目 5 类的 ID
            self._name_to_id = dict(self.config.CLASSES)
        else:
            # 占位模式:用 COCO 预训练模型
            reason = "USE_COCO_MAPPING=true" if self.config.USE_COCO_MAPPING \
                else f"训练模型 {model_path} 不存在"
            logger.warning(
                f"⚠️ 占位模式 ({reason}): 用 yolov8n.pt (COCO 预训练)")
            logger.warning(
                "占位模式只支持 cat / dog / bowl,不支持 monkey / other_primate")
            self.model = YOLO("yolov8n.pt")
            self.mode = "coco_placeholder"
            # COCO 里 cat=15 dog=16 bowl=45
            self._name_to_id = dict(self.config.COCO_MAPPING)

        logger.info(f"模型加载完成,模式={self.mode},类别映射={self._name_to_id}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def info(self) -> Dict:
        return {
            "mode": self.mode,
            "model_path": self.config.MODEL_PATH,
            "classes": list(self._name_to_id.keys()),
        }

    def detect(self, frame) -> DetectionResult:
        """
        输入: cv2 读的 BGR ndarray (H, W, 3)
        输出: DetectionResult,已按 animals / bowls 分好
        """
        result = DetectionResult()
        if frame is None or self.model is None:
            return result

        r = self.model.predict(
            frame,
            imgsz=self.config.MODEL_IMGSZ,
            conf=self.config.MODEL_CONF,
            verbose=False,
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return result

        # 收集数据
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        # 分割 mask 可能有可能没有
        masks_xy = None
        if r.masks is not None:
            masks_xy = r.masks.xy  # list of numpy arrays

        animal_ids = {
            self._name_to_id[name]: name
            for name in ("cat", "dog", "monkey", "other_primate")
            if name in self._name_to_id
        }
        bowl_id = self._name_to_id.get("bowl")

        for i, (box, cls, conf) in enumerate(
                zip(xyxy, cls_arr, conf_arr)):
            box_list = box.tolist()
            mask = masks_xy[i].tolist() if masks_xy is not None else None

            if cls in animal_ids:
                result.animals.append({
                    "cls": animal_ids[cls],
                    "box": box_list,
                    "conf": float(conf),
                    "mask": mask,
                })
            elif cls == bowl_id:
                result.bowls.append({
                    "cls": "bowl",
                    "box": box_list,
                    "conf": float(conf),
                    "mask": mask,
                })

        return result


# 单例
_service: Optional[ModelService] = None


def get_service() -> ModelService:
    """全局单例,首次调用时加载模型"""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
