"""视频处理流水线 · V11 全栈版

流程: 视频 → 抽帧 → YOLO → 姿态(异步) → cascade_rules V9 → 排泄判定 → 事件推送

事件类型:
- drinking: bbox 20% / mask 100px / 侵入 30% / 遮挡 3s 任一触发, LLM 复核
- excretion: 后腿深弯 + 髋下沉 + 前腿撑直, 有盆时门槛更严

姿态服务不可用时优雅降级到纯 bbox 规则(与旧版同).
"""
import base64
import logging
import queue
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
import requests

from config import Config
from event_reporter import get_reporter
from model_service import get_service


logger = logging.getLogger(__name__)


# ============ 姿态客户端 (进程内单例, 独立 session 绕 Clash) ============
_POSE_URL = "http://127.0.0.1:8090"
_pose_session = requests.Session()
_pose_session.trust_env = False
_pose_session.proxies = {"http": "", "https": ""}
_pose_available = None


def _check_pose_available() -> bool:
    """缓存的姿态服务可用性检查, 首次多试几次应对冷启动"""
    global _pose_available
    if _pose_available is None:
        for _ in range(3):
            try:
                r = _pose_session.get(f"{_POSE_URL}/health", timeout=5)
                if r.status_code == 200:
                    _pose_available = True
                    logger.info(f"✅ 姿态服务可用: {_POSE_URL}")
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        _pose_available = False
        logger.warning(
            f"⚠️ 姿态服务不可用: {_POSE_URL} — 降级到纯 bbox 规则")
    return _pose_available


def _pose_predict_crop(crop_bgr):
    """送 crop 到姿态服务, 返回 (39, 3) kps 或 None"""
    try:
        _, buf = cv2.imencode(".jpg", crop_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()
        r = _pose_session.post(
            f"{_POSE_URL}/predict",
            json={"image_b64": b64}, timeout=30)
        if r.status_code != 200:
            return None
        kps_list = r.json().get("keypoints", [])
        if not kps_list:
            return None
        return np.array(kps_list, dtype=np.float32)
    except Exception as e:
        logger.debug(f"pose 请求失败: {e}")
        return None


class _PoseWorker:
    """后台线程跑姿态推理, 主流程读缓存"""

    def __init__(self):
        self.available = _check_pose_available()
        self._latest_kps: Optional[np.ndarray] = None
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._stop = False
        if self.available:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def request(self, frame, animals):
        """主线程调, 有空槽才提交"""
        if not self.available: return
        if self._q.full(): return
        if not animals: return
        # 选最大 bbox 裁 20% padding
        best = max(animals, key=lambda a:
                   (a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1]))
        x1, y1, x2, y2 = best["box"]
        bw = x2 - x1; bh = y2 - y1
        if bw < 120 or bh < 120: return  # 太小跳过
        fh, fw = frame.shape[:2]
        px = int(bw * 0.2); py = int(bh * 0.2)
        cx1 = max(0, int(x1 - px)); cy1 = max(0, int(y1 - py))
        cx2 = min(fw, int(x2 + px)); cy2 = min(fh, int(y2 + py))
        if cx2 - cx1 < 40 or cy2 - cy1 < 40: return
        crop = frame[cy1:cy2, cx1:cx2].copy()
        try:
            self._q.put_nowait((crop, (cx1, cy1)))
        except queue.Full:
            pass

    def _loop(self):
        while not self._stop:
            try:
                crop, (ox, oy) = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            kps = _pose_predict_crop(crop)
            if kps is not None:
                # 翻回全帧坐标
                kps[:, 0] += ox
                kps[:, 1] += oy
                with self._lock:
                    self._latest_kps = kps

    @property
    def latest(self):
        with self._lock:
            return self._latest_kps

    def stop(self):
        self._stop = True


def process_video(video_path: str, kennel_id: str,
                  camera_id: str = "", pet_id: str = "",
                  task_id: str = "",
                  request_id: str = "",
                  kennel_code: str = "") -> dict:
    """全栈视频处理 pipeline"""
    task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"

    logger.info(
        f"[{task_id}] 开始: {video_path} kennel={kennel_id}"
        f"({kennel_code}) camera={camera_id} req={request_id}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{task_id}] 无法打开视频")
        raise RuntimeError("无法打开视频")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps if src_fps > 0 else 0
    # 从 INFERENCE_FPS 反算步长
    step = max(1, int(src_fps / Config.INFERENCE_FPS))
    logger.info(
        f"[{task_id}] {n_frames} 帧 / {src_fps:.1f}fps / {duration:.1f}s / "
        f"步长 {step}")

    model_svc = get_service()
    reporter = get_reporter()
    pose_worker = _PoseWorker()

    # 懒导入 V11 规则引擎
    from cascade_rules import CascadeRuleEngine
    from excretion_pose_rules import PoseExcretionDetector
    from activity_detector import ActivityDetector
    from behavior_rules import CompletedEvent

    drink_engine = CascadeRuleEngine(use_llm=True)
    exc_detector = PoseExcretionDetector()
    act_detector = ActivityDetector()
    exc_memory_until = 0.0

    frame_idx = 0
    inferred = 0
    total_events = 0
    total_exc_events = 0

    # LLM 发现层用:
    # 1) 处理中每 LLM_PERIODIC_SEC 秒定时抽 1 帧问 LLM (实时)
    # 2) 视频结束若还是 0 事件, 抽 3 帧兜底
    llm_scan_candidates = []   # list of (video_time, frame_bgr, detected_class)
    LLM_PERIODIC_SEC = 20.0     # 定时采样间隔
    llm_last_check = 0.0
    llm_periodic_events = 0     # 记录定时 LLM 触发的事件数

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % step != 0:
                frame_idx += 1; continue

            video_time = frame_idx / src_fps
            inferred += 1

            det = model_svc.detect(frame)
            # 转换 mask 字段名: model_svc 返回 "mask", cascade_rules 期望 "mask_pts"
            animals = [dict(a, mask_pts=a.get("mask")) for a in det.animals]
            bowls   = [dict(b, mask_pts=b.get("mask")) for b in det.bowls]

            # 存 LLM 发现层的候选帧 (有动物就存, 视频末若 0 事件再抽)
            if animals:
                llm_scan_candidates.append(
                    (video_time, frame.copy(), animals[0]["cls"]))

            # 每秒发一次姿态请求
            if pose_worker.available and animals and frame_idx % int(src_fps) == 0:
                pose_worker.request(frame, animals)

            # 姿态判定 (排泄优先)
            strong_excretion = middle_excretion = False
            kps = pose_worker.latest
            if kps is not None and animals:
                a = animals[0]
                has_bowl = len(bowls) > 0
                exc_r = exc_detector.update(
                    "animal-0", kps, video_time, a["cls"],
                    has_bowl_nearby=has_bowl)
                strong_excretion = exc_r.get("strong_excretion_pose", False)
                middle_excretion = exc_r.get("score", 0) >= 50
                if exc_r.get("just_finished"):
                    e = exc_r["just_finished"]
                    _push_excretion_event(
                        e, task_id, request_id, kennel_id, kennel_code,
                        camera_id, pet_id, video_time, reporter)
                    total_exc_events += 1
                    total_events += 1

            if strong_excretion:
                exc_memory_until = video_time + 8.0
            excretion_active = (
                strong_excretion or middle_excretion
                or video_time < exc_memory_until
                or len(exc_detector.ongoing) > 0
            )

            # Drink 判定 (强排泄证据时清空 bowls 抑制 drink)
            drink_bowls = [] if excretion_active else bowls
            _, drink_completed = drink_engine.update(
                animals, drink_bowls, video_time, frame_bgr=frame)
            if excretion_active and drink_engine.ongoing:
                drink_engine.ongoing.clear()

            for e in drink_completed:
                _push_drink_event(
                    e, task_id, request_id, kennel_id, kennel_code,
                    camera_id, pet_id, reporter)
                total_events += 1

            # === 活动检测 (优先级最低)===
            # 只有当无排泄 + 无饮水 ongoing 时才推 activity 事件
            act_events = act_detector.update(animals, video_time)
            higher_priority_active = (
                len(exc_detector.ongoing) > 0
                or len(drink_engine.ongoing) > 0
            )
            if not higher_priority_active:
                for ae in act_events:
                    _push_activity_event(
                        ae, task_id, request_id, kennel_id, kennel_code,
                        camera_id, pet_id, reporter)
                    total_events += 1

            # === LLM 定时采样 (每 LLM_PERIODIC_SEC 秒抽 1 帧问 Qwen VL) ===
            # 条件: 有动物 + 无 drink/exc ongoing + 距上次检查 >= 20s
            if (animals
                    and not higher_priority_active
                    and video_time - llm_last_check >= LLM_PERIODIC_SEC):
                llm_last_check = video_time
                pushed = _llm_periodic_check(
                    frame, video_time, animals[0]["cls"],
                    task_id, request_id, kennel_id, kennel_code,
                    camera_id, pet_id, reporter)
                if pushed:
                    llm_periodic_events += pushed
                    total_events += pushed

            frame_idx += 1

        # 视频结束, force_flush
        end_time = frame_idx / src_fps
        for e in drink_engine.force_flush(end_time, frame):
            _push_drink_event(
                e, task_id, request_id, kennel_id, kennel_code,
                camera_id, pet_id, reporter)
            total_events += 1
        for e in exc_detector.force_flush(end_time):
            _push_excretion_event(
                e, task_id, request_id, kennel_id, kennel_code,
                camera_id, pet_id, end_time, reporter)
            total_exc_events += 1
            total_events += 1
        # 活动收尾 (最后, 若无更高优先级正在跟踪)
        for ae in act_detector.force_flush(end_time):
            if not (len(exc_detector.ongoing) or len(drink_engine.ongoing)):
                _push_activity_event(
                    ae, task_id, request_id, kennel_id, kennel_code,
                    camera_id, pet_id, reporter)
                total_events += 1

        # ⭐ LLM 发现层: 规则完全 0 事件时, 抽 3 帧问 LLM
        llm_discovered_events = 0
        if total_events == 0 and llm_scan_candidates:
            llm_discovered_events = _llm_discover_scan(
                llm_scan_candidates, task_id, request_id,
                kennel_id, kennel_code, camera_id, pet_id, reporter)
            total_events += llm_discovered_events
    finally:
        cap.release()
        pose_worker.stop()

    logger.info(
        f"[{task_id}] 完成: 抽帧 {inferred} 张, 事件 {total_events} 个 "
        f"(排泄 {total_exc_events}, "
        f"LLM 定时 {llm_periodic_events}, LLM 末尾发现 {llm_discovered_events})")
    return {
        "framesInferred": inferred,
        "eventsProduced": total_events,
        "eventsReported": total_events,
        "excretionEvents": total_exc_events,
        "llmPeriodicEvents": llm_periodic_events,
        "llmDiscoveredEvents": llm_discovered_events,
    }


def _llm_periodic_check(frame, video_time: float, animal_cls: str,
                         task_id: str, request_id: str,
                         kennel_id: str, kennel_code: str,
                         camera_id: str, pet_id: str, reporter) -> int:
    """定时 LLM 采样: 单帧问 Qwen VL 有无 drinking/excretion 行为
    有则立刻推事件. 返回推送事件数.
    """
    from llm_verifier import get_verifier
    verifier = get_verifier()
    if not verifier or not verifier.available:
        return 0
    pushed = 0
    for evt_type in ("drinking", "excretion"):
        r = verifier.verify_behavior(
            frame, evt_type, animal_cls,
            f"{task_id}-llm-periodic-{int(video_time)}",
            extra_context=f"定时采样 t={video_time:.0f}s")
        if r is not None and r.confirmed and r.confidence >= 0.7:
            from behavior_rules import CompletedEvent
            ev = CompletedEvent(
                event_id=f"evt-llm-{uuid.uuid4().hex[:12]}",
                event_type=evt_type,
                kennel_id=kennel_id,
                camera_id=camera_id,
                pet_id=pet_id,
                detected_class=animal_cls,
                start_time=time.time(),
                end_time=time.time(),
                duration_sec=0,
                hit_count=1,
                confidence=float(r.confidence),
                snapshot_path=None,
            )
            setattr(ev, "task_id", task_id)
            setattr(ev, "request_id", request_id)
            setattr(ev, "kennel_code", kennel_code)
            setattr(ev, "video_offset_sec", video_time)
            reporter.submit(ev)
            pushed += 1
            logger.info(
                f"[{task_id}] 🔍 LLM 定时采样命中: {evt_type} "
                f"t={video_time:.0f}s conf={r.confidence:.2f} "
                f"reason={r.reason[:60]}")
            break  # 单帧只推一个类型的事件
    return pushed


def _llm_discover_scan(candidates, task_id: str, request_id: str,
                        kennel_id: str, kennel_code: str,
                        camera_id: str, pet_id: str, reporter) -> int:
    """LLM 发现层: 规则漏检时, 均匀抽 3 帧问 Qwen VL

    candidates: [(video_time, frame_bgr, cls), ...]
    返回 LLM 判定为 True 后推送的事件数
    """
    from llm_verifier import get_verifier
    verifier = get_verifier()
    if not verifier or not verifier.available:
        logger.info(f"[{task_id}] LLM 未启用, 跳过 LLM 发现")
        return 0
    if len(candidates) < 3:
        return 0
    # 均匀抽 3 帧: 1/4, 1/2, 3/4 位置
    n = len(candidates)
    picks = [candidates[n // 4], candidates[n // 2], candidates[3 * n // 4]]
    logger.info(
        f"[{task_id}] 🔎 LLM 发现层触发: 规则 0 事件, 抽 3 帧问 Qwen VL")

    pushed = 0
    votes = {"drinking": 0, "excretion": 0, "feeding": 0}
    reasons = []
    for vt, frame, cls in picks:
        # 对每帧都试 drinking + excretion
        for evt_type in ("drinking", "excretion"):
            r = verifier.verify_behavior(
                frame, evt_type, cls,
                f"{task_id}-llm-discover-{int(vt)}",
                extra_context=f"规则未触发, LLM 发现层")
            if r is not None and r.confirmed:
                votes[evt_type] += 1
                reasons.append(f"t={vt:.0f}s {evt_type}: {r.reason}")
                break

    # 3 次里 >=2 次同意 → 推送事件
    for evt_type, cnt in votes.items():
        if cnt >= 2:
            from behavior_rules import CompletedEvent
            ev = CompletedEvent(
                event_id=f"evt-llm-{uuid.uuid4().hex[:12]}",
                event_type=evt_type,
                kennel_id=kennel_id,
                camera_id=camera_id,
                pet_id=pet_id,
                detected_class=picks[0][2],
                start_time=time.time(),
                end_time=time.time(),
                duration_sec=0,
                hit_count=cnt,
                confidence=cnt / 3.0,
                snapshot_path=None,
            )
            setattr(ev, "task_id", task_id)
            setattr(ev, "request_id", request_id)
            setattr(ev, "kennel_code", kennel_code)
            setattr(ev, "video_offset_sec", picks[0][0])
            reporter.submit(ev)
            pushed += 1
            logger.info(
                f"[{task_id}] ✨ LLM 发现事件: {evt_type} · "
                f"3帧中{cnt}帧同意 · " + " | ".join(reasons))
    if not pushed:
        logger.info(
            f"[{task_id}] LLM 发现层: 3 次抽帧均无一致行为, 视频真无事件")
    return pushed


def _push_drink_event(finalized, task_id: str, request_id: str,
                      kennel_id: str, kennel_code: str,
                      camera_id: str, pet_id: str, reporter):
    """把 cascade FinalizedEvent 转成 CompletedEvent 推给 Java"""
    from behavior_rules import CompletedEvent
    ev = CompletedEvent(
        event_id=finalized.event_id,
        event_type="drinking",
        kennel_id=kennel_id,
        camera_id=camera_id,
        pet_id=pet_id,
        detected_class=finalized.animal_cls,
        start_time=time.time(),
        end_time=time.time() + finalized.duration_sec,
        duration_sec=float(finalized.duration_sec),
        hit_count=int(finalized.hit_count),
        confidence=round(float(finalized.confidence), 3),
        snapshot_path=None,
    )
    setattr(ev, "task_id", task_id)
    setattr(ev, "request_id", request_id)
    setattr(ev, "kennel_code", kennel_code)
    setattr(ev, "video_offset_sec", float(finalized.start_time))
    reporter.submit(ev)


def _push_activity_event(a_dict, task_id: str, request_id: str,
                          kennel_id: str, kennel_code: str,
                          camera_id: str, pet_id: str, reporter):
    """把活动检测器返回的 dict 转成 CompletedEvent 推 Java"""
    from behavior_rules import CompletedEvent
    ev = CompletedEvent(
        event_id=a_dict.get("event_id", f"evt-act-{uuid.uuid4().hex[:12]}"),
        event_type="activity",
        kennel_id=kennel_id,
        camera_id=camera_id,
        pet_id=pet_id,
        detected_class=a_dict.get("animal_cls", "unknown"),
        start_time=time.time(),
        end_time=time.time() + a_dict.get("duration", 0),
        duration_sec=float(a_dict.get("duration", 0)),
        hit_count=1,
        confidence=0.9,
        snapshot_path=None,
    )
    setattr(ev, "task_id", task_id)
    setattr(ev, "request_id", request_id)
    setattr(ev, "kennel_code", kennel_code)
    setattr(ev, "video_offset_sec", float(a_dict.get("start_time", 0)))
    reporter.submit(ev)


def _push_excretion_event(e_dict, task_id: str, request_id: str,
                           kennel_id: str, kennel_code: str,
                           camera_id: str, pet_id: str,
                           video_end_time: float, reporter):
    """把排泄检测器返回的 dict 转成 CompletedEvent 推给 Java"""
    from behavior_rules import CompletedEvent
    ev = CompletedEvent(
        event_id=f"evt-exc-{uuid.uuid4().hex[:12]}",
        event_type="excretion",
        kennel_id=kennel_id,
        camera_id=camera_id,
        pet_id=pet_id,
        detected_class=e_dict.get("animal_cls", "unknown"),
        start_time=time.time(),
        end_time=time.time() + e_dict.get("duration", 0),
        duration_sec=float(e_dict.get("duration", 0)),
        hit_count=int(e_dict.get("hit", 0)),
        confidence=min(1.0, e_dict.get("max_score", 0) / 125.0),
        snapshot_path=None,
    )
    setattr(ev, "task_id", task_id)
    setattr(ev, "request_id", request_id)
    setattr(ev, "kennel_code", kennel_code)
    setattr(ev, "video_offset_sec", float(e_dict.get("start_time", 0)))
    reporter.submit(ev)
