"""视频处理流水线:视频文件 → 抽帧 → YOLO → 规则 → 事件

给 app.py 的 process_video_async 用。
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

from behavior_rules import BehaviorRuleEngine, CompletedEvent
from config import Config
from event_reporter import get_reporter
from model_service import get_service


logger = logging.getLogger(__name__)


def process_video(video_path: str, kennel_id: str,
                  camera_id: str = "", pet_id: str = "",
                  task_id: str = "") -> dict:
    """处理一个视频文件,返回本次识别到的事件统计"""
    task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"

    logger.info(
        f"[{task_id}] 视频处理开始: {video_path} kennel={kennel_id} "
        f"camera={camera_id}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{task_id}] 无法打开视频")
        return {"error": "无法打开视频"}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps if src_fps > 0 else 0
    step = max(1, int(src_fps / Config.INFERENCE_FPS)) if src_fps > 0 else 1
    logger.info(
        f"[{task_id}] 视频信息: {n_frames} 帧 / {src_fps:.1f} FPS "
        f"/ {duration:.1f} 秒 / 每 {step} 帧抽 1")

    model_svc = get_service()
    rules = BehaviorRuleEngine(
        kennel_id=kennel_id,
        camera_id=camera_id,
        pet_id=pet_id,
    )
    reporter = get_reporter()

    # 使用视频"内部时间"而不是 wall clock,规则判定用视频秒
    frame_idx = 0
    inferred = 0
    total_events = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # 视频内部时间戳
            video_time = frame_idx / src_fps if src_fps > 0 else 0
            inferred += 1

            # 推理
            det = model_svc.detect(frame)

            # 规则引擎
            completed = rules.update(
                animals=det.animals,
                bowls=det.bowls,
                frame_time=video_time,
                frame_bgr=frame,
            )

            # 事件产出
            for event in completed:
                _save_snapshot(event, task_id)
                reporter.submit(event)
                total_events += 1

            frame_idx += 1

        # 视频结束,强制 flush 未完的
        end_time = frame_idx / src_fps if src_fps > 0 else 0
        for event in rules.force_flush(end_time):
            _save_snapshot(event, task_id)
            reporter.submit(event)
            total_events += 1

    finally:
        cap.release()

    logger.info(
        f"[{task_id}] 视频处理完成: 抽帧 {inferred} 张, "
        f"产出事件 {total_events} 个")
    return {
        "taskId": task_id,
        "framesInferred": inferred,
        "eventsProduced": total_events,
    }


def _save_snapshot(event: CompletedEvent, task_id: str):
    """从 event 的最后帧存截图作证据"""
    # last_frame_bgr 在 rules 里保存在 OngoingEvent,但 CompletedEvent 不带
    # 简化:如果配置需要截图,可在 rules._finalize 里带上 frame
    # 目前不实现,batch 3 再补
    pass
