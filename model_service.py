"""模型服务:双模型并行架构

主模型: 检测动物 (cat/dog/monkey/other_primate)
副模型: 检测容器 (bowl/plate/cup/bottle) — 可选, 未配则用主模型的 bowl 类

两个模型独立线程并行推理, 结果合并.
支持切换任意开源 YOLO-seg 模型 (只要有 mask 输出).
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from ultralytics import YOLO

from config import Config


logger = logging.getLogger(__name__)


# COCO 类别映射(常用宠物/容器 · yolo11n-seg 用)
COCO_ANIMAL_IDS = {15: "cat", 16: "dog"}  # COCO 没 monkey/other_primate
COCO_CONTAINER_IDS = {
    39: "bottle", 41: "cup", 45: "bowl",  # 46/47 = wine_glass/fork 不算
}


class DetectionResult:
    """一帧的检测结果"""
    def __init__(self):
        self.animals: List[Dict] = []
        self.bowls: List[Dict] = []
        # animals[i] = {"cls":"cat", "box":[..], "conf":0.9, "mask":[[x,y]..]}
        # bowls[i]   = {"cls":"bowl"|"cup"|"bottle", ...}


class _SubModel:
    """单个 YOLO 模型的封装, 内部管 class-id 到项目类别名的映射"""

    def __init__(self, path: str, use_coco: bool, role: str):
        """
        role: 'primary'(动物) 或 'secondary'(容器)
        """
        self.path = path
        self.role = role
        self.use_coco = use_coco
        self.model = None
        self._name_map: Dict[int, str] = {}
        self._load()

    def _load(self):
        p = Path(self.path)
        if not p.exists() and not (self.path.endswith(".pt")
                                    and "/" not in self.path.replace("\\","/")):
            logger.warning(f"[{self.role}] 模型不存在: {self.path}")
            self.model = None
            return

        self.model = YOLO(str(self.path))
        # 验证是分割模型 (有 mask 头)
        # ultralytics 会自动填 task 属性
        task = getattr(self.model, "task", None) or "detect"
        if task != "segment":
            logger.warning(
                f"[{self.role}] {self.path} 不是分割模型 (task={task}), "
                f"没 mask 输出, 行为规则可能失效")

        if self.use_coco:
            # COCO 映射
            if self.role == "primary":
                # 主模型: 动物 (+ 若无副模型, 也要出容器)
                self._name_map = dict(COCO_ANIMAL_IDS)
                # 单模型模式下主模型也拿容器 (由 ModelService 决定后设 flag)
                self._include_containers = False
            else:
                self._name_map = dict(COCO_CONTAINER_IDS)
                self._include_containers = True
        else:
            # 项目类别映射 (用户自训 best.pt)
            if self.role == "primary":
                self._name_map = {v: k for k, v in Config.CLASSES.items()
                                    if k in ("cat","dog","monkey","other_primate")}
                self._include_containers = False
            else:
                self._name_map = {v: k for k, v in Config.CLASSES.items()
                                    if k in ("bowl",)}
                self._include_containers = True

        logger.info(
            f"[{self.role}] 加载 {self.path} · task={task} · "
            f"关注类别={list(self._name_map.values())}")

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, frame) -> tuple:
        """返回 (list[animal_dict], list[container_dict])"""
        animals = []
        containers = []
        if not self.ready:
            return animals, containers

        r = self.model.predict(
            frame, imgsz=Config.MODEL_IMGSZ,
            conf=Config.MODEL_CONF, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return animals, containers

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        masks_xy = r.masks.xy if r.masks is not None else None

        for i, (box, cls, conf) in enumerate(zip(xyxy, cls_arr, conf_arr)):
            name = self._name_map.get(int(cls))
            if name is None: continue
            mask = masks_xy[i].tolist() if masks_xy is not None else None
            info = {
                "cls": name,
                "box": box.tolist(),
                "conf": float(conf),
                "mask": mask,
            }
            if name in ("cat","dog","monkey","other_primate"):
                animals.append(info)
            else:
                containers.append(info)

        return animals, containers


class ModelService:
    """双模型服务

    主模型必须有. 副模型可选, 未配时主模型的 bowl 类会用.
    """

    def __init__(self, config: Config = Config):
        self.config = config
        # 主模型: 训练模型 或 COCO 占位
        primary_path = config.MODEL_PATH
        primary_use_coco = config.USE_COCO_MAPPING
        if not Path(primary_path).exists():
            logger.warning(
                f"⚠️ 主模型 {primary_path} 不存在, 用 yolov8n-seg.pt 占位")
            primary_path = "yolov8n-seg.pt"
            primary_use_coco = True

        self.primary = _SubModel(primary_path, primary_use_coco, "primary")

        # 副模型: 可选
        self.secondary: Optional[_SubModel] = None
        if config.SECONDARY_MODEL_PATH:
            try:
                self.secondary = _SubModel(
                    config.SECONDARY_MODEL_PATH,
                    config.SECONDARY_USE_COCO,
                    "secondary")
                logger.info(
                    f"✅ 双模型启用: 主={primary_path} 副={config.SECONDARY_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"副模型加载失败, 降级单模型: {e}")
                self.secondary = None
        else:
            logger.info("单模型模式 (主模型同时出动物和容器)")
            # 单模型模式: 主模型的 name_map 加入容器类
            if self.primary.use_coco:
                self.primary._name_map.update(COCO_CONTAINER_IDS)
            else:
                # 项目类别里的 bowl
                for k, v in Config.CLASSES.items():
                    if k == "bowl":
                        self.primary._name_map[v] = k
            self.primary._include_containers = True

        # 线程池 for 并行
        self._pool = ThreadPoolExecutor(max_workers=2)

    @property
    def is_ready(self) -> bool:
        return self.primary.ready

    @property
    def info(self) -> Dict:
        info = {
            "mode": "dual" if self.secondary else "single",
            "primary": {
                "path": self.primary.path,
                "classes": list(self.primary._name_map.values()),
            },
        }
        if self.secondary:
            info["secondary"] = {
                "path": self.secondary.path,
                "classes": list(self.secondary._name_map.values()),
            }
        # 兼容老字段 (health 检查用)
        info["classes"] = list(self.primary._name_map.values())
        if self.secondary:
            info["classes"] += list(self.secondary._name_map.values())
        info["model_path"] = self.primary.path
        return info

    def detect(self, frame) -> DetectionResult:
        """输入 BGR ndarray, 输出 DetectionResult (animals+bowls 合并)"""
        result = DetectionResult()
        if frame is None or not self.primary.ready:
            return result

        # 主模型 + 副模型 并行
        if self.secondary and self.secondary.ready:
            fut_p = self._pool.submit(self.primary.predict, frame)
            fut_s = self._pool.submit(self.secondary.predict, frame)
            animals_p, containers_p = fut_p.result()
            animals_s, containers_s = fut_s.result()
            # 主模型的动物 + 副模型的容器
            # (主模型出的 bowl 会被覆盖为副模型的更精准结果)
            result.animals = animals_p
            result.bowls = containers_s if containers_s else containers_p
        else:
            # 单模型: 主模型出所有
            animals, containers = self.primary.predict(frame)
            result.animals = animals
            result.bowls = containers

        return result


# 单例
_service: Optional[ModelService] = None


def get_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
