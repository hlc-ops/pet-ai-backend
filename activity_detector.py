"""活动检测器 · 动物 bbox 位移超阈值判为活动

规则:
- 追踪每只动物的 bbox 中心, 过去 3 秒轨迹
- 中心移动 > 40 像素 (或 bbox 宽度 30%) 视为在动
- 连续活动 >= 3 秒触发 activity 事件
- 事件冷却 60s (甲方要求, 同时上层 event_reporter 也有冷却)
"""
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, List, Optional


logger = logging.getLogger(__name__)


# ==================== 参数 ====================
MOVEMENT_THRESHOLD_PX = 40       # 3 秒内中心位移超过多少像素算在动
MOVEMENT_BBOX_RATIO = 0.30       # 或位移 > bbox 宽 30%
MIN_ACTIVE_DURATION_SEC = 3.0    # 连续活动多久才算事件
STATIC_TIMEOUT_SEC = 1.5         # 多久没动了算停下
HISTORY_WINDOW_SEC = 3.0
CENTROID_QUANT = 60              # 简单追踪: bbox 中心量化到 60px 网格


@dataclass
class _Track:
    animal_key: str
    animal_cls: str
    positions: Deque   # (time, cx, cy)
    active_since: Optional[float] = None
    last_activity_end: float = 0.0


class ActivityDetector:
    """活动检测: 追踪 bbox 位移, 位移超阈值判为活动"""

    def __init__(self):
        self.tracks: dict = {}
        self._next_id = 0

    def _find_or_create_track(self, cx, cy, cls, now):
        """按位置就近匹配, 找不到就建新"""
        best_id = None
        best_dist = float("inf")
        for tid, t in self.tracks.items():
            if not t.positions: continue
            _, lx, ly = t.positions[-1]
            d = ((cx-lx)**2 + (cy-ly)**2)**0.5
            if d < 200 and d < best_dist:  # 200px 内认为同一动物
                best_dist = d; best_id = tid
        if best_id is not None:
            return self.tracks[best_id]
        # 新轨迹
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = _Track(
            animal_key=f"animal-{tid}",
            animal_cls=cls,
            positions=deque(),
        )
        return self.tracks[tid]

    def update(self, animals: List[dict], now: float) -> List[dict]:
        """输入当前帧所有动物, 输出触发的活动事件"""
        finished_events = []

        # 1. 更新每只动物的轨迹
        active_tracks = set()
        for a in animals:
            x1, y1, x2, y2 = a["box"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1

            track = self._find_or_create_track(cx, cy, a["cls"], now)
            active_tracks.add(id(track))
            track.animal_cls = a["cls"]  # 更新类别
            track.positions.append((now, cx, cy))
            # 保留 3 秒历史
            while track.positions and now - track.positions[0][0] > HISTORY_WINDOW_SEC:
                track.positions.popleft()

            # 计算过去 3s 位移
            if len(track.positions) < 3: continue
            xs = [p[1] for p in track.positions]
            ys = [p[2] for p in track.positions]
            move = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)**0.5

            is_moving = (
                move > MOVEMENT_THRESHOLD_PX
                or move > bw * MOVEMENT_BBOX_RATIO
            )

            if is_moving:
                if track.active_since is None:
                    track.active_since = now
            else:
                # 停下, 判定是否触发事件
                if track.active_since is not None:
                    duration = now - track.active_since
                    if duration >= MIN_ACTIVE_DURATION_SEC:
                        finished_events.append({
                            "event_id": f"evt-act-{uuid.uuid4().hex[:12]}",
                            "animal_key": track.animal_key,
                            "animal_cls": track.animal_cls,
                            "start_time": track.active_since,
                            "end_time": now,
                            "duration": duration,
                            "avg_move_px": move,
                        })
                    track.active_since = None
                    track.last_activity_end = now

        # 2. 清理长时间未见的轨迹
        stale = [tid for tid, t in self.tracks.items()
                  if id(t) not in active_tracks
                  and t.positions
                  and now - t.positions[-1][0] > STATIC_TIMEOUT_SEC * 2]
        for tid in stale:
            t = self.tracks[tid]
            if t.active_since is not None:
                duration = t.positions[-1][0] - t.active_since
                if duration >= MIN_ACTIVE_DURATION_SEC:
                    finished_events.append({
                        "event_id": f"evt-act-{uuid.uuid4().hex[:12]}",
                        "animal_key": t.animal_key,
                        "animal_cls": t.animal_cls,
                        "start_time": t.active_since,
                        "end_time": t.positions[-1][0],
                        "duration": duration,
                        "avg_move_px": 0,
                    })
            del self.tracks[tid]

        return finished_events

    def force_flush(self, now: float) -> List[dict]:
        """视频结束时收尾"""
        events = []
        for tid, t in list(self.tracks.items()):
            if t.active_since is not None and t.positions:
                duration = t.positions[-1][0] - t.active_since
                if duration >= MIN_ACTIVE_DURATION_SEC:
                    events.append({
                        "event_id": f"evt-act-{uuid.uuid4().hex[:12]}",
                        "animal_key": t.animal_key,
                        "animal_cls": t.animal_cls,
                        "start_time": t.active_since,
                        "end_time": t.positions[-1][0],
                        "duration": duration,
                        "avg_move_px": 0,
                    })
        self.tracks.clear()
        return events
