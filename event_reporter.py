"""事件推送客户端

- 异步队列:主流程 submit() 后立刻返回,不阻塞视频处理
- 重试:指数退避 1s / 2s / 4s
- 失败落盘:重试都失败的事件写 failed_events.jsonl,可后期人工重传
"""
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from behavior_rules import CompletedEvent
from config import Config


logger = logging.getLogger(__name__)


def event_to_payload(event: CompletedEvent) -> dict:
    """把内部事件对象转成 HTTP 推送格式"""
    return {
        "eventId": event.event_id,
        "eventType": event.event_type,
        "kennelId": event.kennel_id,
        "cameraId": event.camera_id,
        "petId": event.pet_id,
        "detectedClass": event.detected_class,
        "eventTime": datetime.fromtimestamp(event.start_time).strftime(
            "%Y-%m-%d %H:%M:%S"),
        "durationSeconds": int(round(event.duration_sec)),
        "confidence": event.confidence,
        "hitCount": event.hit_count,
        "imageUrl": event.snapshot_path or "",
        "videoUrl": "",
    }


class EventReporter:
    """单例推送客户端。

    submit(event) 后异步推送,失败落盘。
    在 app 启动时初始化,关闭时 close()。
    """

    _instance: Optional["EventReporter"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inited = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_inited", False):
            return
        self._inited = True

        self.q: queue.Queue[CompletedEvent] = queue.Queue()
        self.stop_flag = False
        self.fail_log = Config.LOG_DIR / "failed_events.jsonl"
        self.headers = {"Content-Type": "application/json"}
        if Config.CALLBACK_AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {Config.CALLBACK_AUTH_TOKEN}"

        # 后台工作线程
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

        if Config.CALLBACK_URL:
            logger.info(f"事件推送已启用: {Config.CALLBACK_URL}")
        else:
            logger.warning(
                "CALLBACK_URL 未配置,事件将只写本地日志不推送")

    def submit(self, event: CompletedEvent):
        """主流程调这个,立刻返回"""
        self.q.put(event)

    def _run(self):
        while not self.stop_flag:
            try:
                event = self.q.get(timeout=1)
            except queue.Empty:
                continue
            self._process(event)

    def _process(self, event: CompletedEvent):
        payload = event_to_payload(event)

        # 无论如何先写本地日志
        logger.info(
            f"📡 事件产出: {payload['eventType']} class={payload['detectedClass']} "
            f"kennel={payload['kennelId']} camera={payload['cameraId']} "
            f"duration={payload['durationSeconds']}s conf={payload['confidence']} "
            f"eventId={payload['eventId']}")

        if not Config.CALLBACK_URL:
            return  # 没配 URL 就只写日志

        # 3 次重试 + 指数退避
        for attempt in range(Config.CALLBACK_MAX_RETRIES):
            try:
                r = requests.post(
                    Config.CALLBACK_URL,
                    json=payload,
                    headers=self.headers,
                    timeout=Config.CALLBACK_TIMEOUT_SEC,
                )
                if 200 <= r.status_code < 300:
                    logger.info(
                        f"✅ 事件推送成功 HTTP {r.status_code} "
                        f"eventId={payload['eventId']}")
                    return
                else:
                    logger.warning(
                        f"⚠️ HTTP {r.status_code} 响应: "
                        f"{r.text[:200]}")
            except requests.RequestException as e:
                logger.warning(
                    f"⚠️ 推送失败 (第 {attempt + 1} 次): {e}")

            if attempt < Config.CALLBACK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

        # 全失败 -> 落盘
        try:
            with self.fail_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "failedAt": datetime.now().isoformat(),
                    "payload": payload,
                }, ensure_ascii=False) + "\n")
            logger.error(
                f"❌ 事件推送最终失败,已入队 {self.fail_log.name} "
                f"eventId={payload['eventId']}")
        except Exception as e:
            logger.exception(f"落盘也失败了: {e}")

    def close(self):
        self.stop_flag = True


# 全局单例
def get_reporter() -> EventReporter:
    return EventReporter()
