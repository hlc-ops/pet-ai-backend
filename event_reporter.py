"""事件推送客户端 · 按 AI_ALGORITHM_API v1.0 实现

两个回调:
- 行为事件: POST CALLBACK_URL          /api/ai/behavior-events
- 任务状态: POST TASK_CALLBACK_URL     /api/ai/analysis-jobs/status

策略:
- 异步队列: submit() 后立刻返回, 不阻塞视频处理
- 重试: 指数退避 1s / 2s / 4s (共 3 次), 只对 5xx/408/429/网络错误重试
- 幂等: 事件 eventId 唯一, Java 侧去重
- 落盘: 最终失败写 failed_events.jsonl
"""
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from behavior_rules import CompletedEvent
from config import Config


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """带时区 ISO-8601 (北京时间), 甲方要求格式"""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).isoformat(timespec="seconds")


def _iso_from_ts(ts: float) -> str:
    """timestamp → ISO-8601 带时区"""
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(ts, tz).isoformat(timespec="seconds")


def event_to_payload(event: CompletedEvent) -> dict:
    """行为事件 → HTTP payload (完全按甲方 API 4.2)"""
    return {
        "eventId": event.event_id,
        "source": "ai",
        "taskId": getattr(event, "task_id", "") or "",
        "requestId": getattr(event, "request_id", "") or "",
        "eventType": event.event_type,
        "kennelId": event.kennel_id,
        "kennelCode": getattr(event, "kennel_code", "") or "",
        "cameraId": event.camera_id,
        "petId": event.pet_id,
        "detectedClass": event.detected_class,
        "eventTime": _iso_from_ts(event.start_time),
        "videoOffsetSeconds": float(getattr(event, "video_offset_sec", 0.0)),
        "durationSeconds": int(round(event.duration_sec)),
        "confidence": float(event.confidence),
        "hitCount": int(event.hit_count),
        "imageUrl": event.snapshot_path or "",
        "videoUrl": "",
    }


def task_status_payload(status: str, task_id: str, ctx: dict,
                         result: Optional[dict] = None,
                         error: Optional[str] = None) -> dict:
    """任务状态回调 payload (甲方 API 4.1)"""
    payload = {
        "taskId": task_id,
        "requestId": ctx.get("requestId", "") or "",
        "status": status,
        "kennelId": ctx.get("kennelId", "") or "",
        "kennelCode": ctx.get("kennelCode", "") or "",
        "cameraId": ctx.get("cameraId", "") or "",
        "petId": ctx.get("petId", "") or "",
        "occurredAt": _now_iso(),
    }
    if status == "completed" and result is not None:
        payload["result"] = result
    if status == "failed" and error:
        payload["error"] = str(error)[:2000]
    return payload


def _should_retry(status_code: int) -> bool:
    """哪些 HTTP 状态需要重试 (甲方 API 4.3)"""
    return status_code in (408, 429) or 500 <= status_code < 600


def _post_with_retry(url: str, payload: dict, headers: dict,
                      max_retries: int = None, tag: str = "") -> bool:
    """POST 带指数退避重试. 返回是否成功"""
    if max_retries is None:
        max_retries = Config.CALLBACK_MAX_RETRIES

    # 独立 session 绕过 Clash 系统代理拦截
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": "", "https": ""}

    for attempt in range(max_retries):
        try:
            r = session.post(
                url, json=payload, headers=headers,
                timeout=Config.CALLBACK_TIMEOUT_SEC,
            )
            if 200 <= r.status_code < 300:
                # 也检查 Java 返回的 code < 400
                try:
                    body = r.json()
                    if isinstance(body, dict) and body.get("code", 200) >= 400:
                        logger.warning(
                            f"⚠️ {tag} 甲方业务码 {body.get('code')} "
                            f"msg={body.get('message', '')[:100]}")
                        return False
                except Exception:
                    pass
                logger.info(f"✅ {tag} 推送成功 HTTP {r.status_code}")
                return True
            elif _should_retry(r.status_code):
                logger.warning(
                    f"⚠️ {tag} HTTP {r.status_code} "
                    f"body={r.text[:200]}  (将重试)")
            else:
                # 4xx 参数/鉴权错误 不重试
                logger.error(
                    f"❌ {tag} HTTP {r.status_code} 不重试. "
                    f"body={r.text[:300]}")
                return False
        except requests.RequestException as e:
            logger.warning(
                f"⚠️ {tag} 网络错误 (第 {attempt + 1} 次): {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s

    logger.error(f"❌ {tag} 重试 {max_retries} 次全失败")
    return False


class EventReporter:
    """行为事件推送单例"""

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

        self.q: queue.Queue = queue.Queue(maxsize=Config.MAX_QUEUED_EVENTS)
        self.stop_flag = False
        self.fail_log = Config.LOG_DIR / "failed_events.jsonl"
        self.headers = {"Content-Type": "application/json"}
        if Config.CALLBACK_AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {Config.CALLBACK_AUTH_TOKEN}"

        # 事件冷却池: (kennelId, eventType) -> last_sent_timestamp
        self._cooldown_pool: dict = {}
        self._cooldown_lock = threading.Lock()

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

        if Config.CALLBACK_URL:
            logger.info(
                f"事件推送已启用: {Config.CALLBACK_URL} · "
                f"冷却窗口 {Config.EVENT_COOLDOWN_SEC}s")
        else:
            logger.warning(
                "CALLBACK_URL 未配置,事件将只写本地日志不推送")

    def submit(self, event: CompletedEvent):
        try:
            self.q.put_nowait(event)
        except queue.Full:
            logger.error(
                f"事件队列已满 ({Config.MAX_QUEUED_EVENTS}), 丢弃 "
                f"eventId={event.event_id}")

    def _run(self):
        while not self.stop_flag:
            try:
                event = self.q.get(timeout=1)
            except queue.Empty:
                continue
            self._process(event)

    def _process(self, event: CompletedEvent):
        payload = event_to_payload(event)
        logger.info(
            f"📡 事件产出: {payload['eventType']} "
            f"class={payload['detectedClass']} "
            f"kennel={payload['kennelId']}({payload['kennelCode']}) "
            f"duration={payload['durationSeconds']}s "
            f"conf={payload['confidence']} eventId={payload['eventId']}")

        # 冷却检查: 同一犬位 + 同行为在冷却窗口内只推 1 次
        if Config.EVENT_COOLDOWN_SEC > 0:
            key = (payload["kennelId"], payload["eventType"])
            now = time.time()
            with self._cooldown_lock:
                last_ts = self._cooldown_pool.get(key, 0)
                gap = now - last_ts
                if gap < Config.EVENT_COOLDOWN_SEC:
                    logger.info(
                        f"🛑 冷却池跳过: kennel={key[0]} type={key[1]} "
                        f"距上次 {gap:.1f}s < {Config.EVENT_COOLDOWN_SEC}s")
                    return
                # 通过冷却, 记录时间
                self._cooldown_pool[key] = now

        # ⭐ 无论 CALLBACK_URL 通不通, 都把事件存到 TASKS[taskId].events
        # 供朋友前端 GET /api/tasks/{taskId} 轮询兜底
        task_id = payload.get("taskId", "")
        if task_id:
            try:
                from app import append_event_to_task
                append_event_to_task(task_id, payload)
            except Exception as e:
                logger.debug(f"存 task events 失败: {e}")

        if not Config.CALLBACK_URL:
            return

        ok = _post_with_retry(
            Config.CALLBACK_URL, payload, self.headers,
            tag=f"behavior[{payload['eventId']}]")
        if not ok:
            # 落盘
            try:
                with self.fail_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "failedAt": _now_iso(),
                        "target": Config.CALLBACK_URL,
                        "payload": payload,
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.exception(f"落盘也失败: {e}")

    def close(self):
        self.stop_flag = True


def get_reporter() -> EventReporter:
    return EventReporter()


# ============ 任务状态回调 (同步版, 因量少不需要队列) ============
def report_task_status(status: str, task_id: str, ctx: dict,
                        result: Optional[dict] = None,
                        error: Optional[str] = None):
    """发任务状态到 TASK_CALLBACK_URL

    status: processing / completed / failed
    """
    if not Config.TASK_CALLBACK_URL:
        logger.warning(
            f"TASK_CALLBACK_URL 未配置, {task_id} status={status} 只记日志")
        return

    payload = task_status_payload(status, task_id, ctx, result, error)
    headers = {"Content-Type": "application/json"}
    if Config.CALLBACK_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {Config.CALLBACK_AUTH_TOKEN}"

    logger.info(f"📮 任务状态: {task_id} → {status}")
    _post_with_retry(
        Config.TASK_CALLBACK_URL, payload, headers,
        tag=f"task-status[{task_id}:{status}]")
