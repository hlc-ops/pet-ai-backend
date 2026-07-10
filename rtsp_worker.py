"""RTSP 拉流 worker(占位,批次 3+ 完善)

架构预留:
- POST /api/kennels/*/rtsp 注册后,启动一个后台线程
- 线程持续从 RTSP 拉帧,喂给 YOLO + 规则引擎
- 断线自动重连
- 支持多摄像头并发

当前状态:占位,只做接口,不启动实际拉流。
如果甲方明确要 RTSP,后续完善这里。
"""
import logging
import threading
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class RtspWorker(threading.Thread):
    """单个 RTSP 拉流线程(占位实现)"""

    def __init__(self, kennel_id: str, camera_id: str,
                 rtsp_url: str, pet_id: str = ""):
        super().__init__(daemon=True)
        self.kennel_id = kennel_id
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.pet_id = pet_id
        self.stop_flag = False

    def run(self):
        logger.info(
            f"[占位] RTSP worker 启动: {self.kennel_id}/{self.camera_id} "
            f"url={self.rtsp_url}")
        # TODO: 实际实现
        # cap = cv2.VideoCapture(self.rtsp_url)
        # rules = BehaviorRuleEngine(...)
        # while not self.stop_flag:
        #     ret, frame = cap.read()
        #     if not ret: 重连
        #     det = model.detect(frame)
        #     completed = rules.update(...)
        #     for event in completed: reporter.submit(event)

    def stop(self):
        self.stop_flag = True


class RtspRegistry:
    """管理所有 RTSP worker"""

    _instance: Optional["RtspRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.workers: Dict[str, RtspWorker] = {}
        return cls._instance

    def register(self, kennel_id: str, camera_id: str,
                 rtsp_url: str, pet_id: str = ""):
        key = f"{kennel_id}/{camera_id}"
        if key in self.workers:
            self.workers[key].stop()
        w = RtspWorker(kennel_id, camera_id, rtsp_url, pet_id)
        w.start()
        self.workers[key] = w
        return key

    def stop_all(self):
        for w in self.workers.values():
            w.stop()
        self.workers.clear()


def get_registry() -> RtspRegistry:
    return RtspRegistry()
