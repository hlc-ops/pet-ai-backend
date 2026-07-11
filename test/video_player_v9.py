"""视频播放器 V9 · 带 LLM 复核

V8.1 基础上增加:
- L5:LLM 视觉复核(Qwen-VL / GLM-4V)
- 触发瞬间/中期/结束 各调用 1 次
- 全局 3 秒节流,单事件最多 3 次
- LLM 结果直接在横幅显示

需要 .env 里配 LLM_API_KEY(申请:https://dashscope.aliyun.com/apiKey)

用法:
    python video_player_v9.py
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 提前读 .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from mask_utils import polygon_to_mask, mask_overlap_area
from llm_verifier import get_verifier


CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
    "pet": (200, 200, 200),
}
TRIGGER_COLOR = (0, 100, 255)
GHOST_BOWL_COLOR = (128, 220, 128)
LLM_OK_COLOR = (0, 255, 100)     # 绿:LLM 确认
LLM_NO_COLOR = (0, 100, 255)     # 红:LLM 否认
LLM_WAIT_COLOR = (100, 200, 255) # 淡青:等待复核

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


# ==================== 参数 ====================
BBOX_MIN_RATIO = 0.20
MASK_MIN_PIXELS = 100
INVASION_MIN_RATIO = 0.30
OCCLUSION_MEMORY_SEC = 6.0
OCCLUSION_MIN_GONE_SEC = 3.0
MIN_EVENT_DURATION_SEC = 2.0
MAX_GAP_SEC = 1.5
MAX_GHOSTS_SHOWN = 2

AGNOSTIC_NMS = True
PET_UNCERTAINTY_THRESHOLD = 0.15
PET_MAX_CONF = 0.60

# LLM 复核
LLM_MID_CHECK_SEC = 8.0
LLM_MAX_PER_EVENT = 3


def pick_file(title, filetypes):
    r = Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    r.destroy()
    return p


def get_screen_size():
    r = Tk()
    w, h = r.winfo_screenwidth(), r.winfo_screenheight()
    r.destroy()
    return w, h


def resolve_model():
    if len(sys.argv) > 1: return sys.argv[1]
    here = Path(__file__).parent.parent
    for p in [here / "model" / "best.pt",
              here / "model" / "best_openvino_model"]:
        if p.exists(): return str(p)
    return pick_file("选模型", [("模型", "*.pt *.onnx")])


def cv2_zh(img, text, org, size=18, color=(255, 255, 255)):
    if FONT_PATH is None:
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(
        org, text, font=ImageFont.truetype(FONT_PATH, size),
        fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def bbox_overlap_ratio(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / min(a1, a2) if min(a1, a2) > 0 else 0.0


def bbox_center_distance(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


def bbox_to_mask(box, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    x2 = min(w, x2); y2 = min(h, y2)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 1
    return mask


def maybe_downgrade_to_pet(cls_name, conf, other_animal_confs):
    if cls_name not in ("cat", "dog", "monkey", "other_primate"):
        return cls_name
    if conf < PET_MAX_CONF:
        return "pet"
    max_other = max(
        [c for k, c in other_animal_confs.items() if k != cls_name],
        default=0)
    if conf - max_other < PET_UNCERTAINTY_THRESHOLD and max_other > 0.3:
        return "pet"
    return cls_name


# ==================== V9 事件带 LLM 状态 ====================
class LLMBackedOngoingEvent:
    def __init__(self, animal_cls, bowl_id, event_type, start_time):
        self.animal_cls = animal_cls
        self.bowl_id = bowl_id
        self.event_type = event_type
        self.start_time = start_time
        self.last_seen = start_time
        self.hit = 0
        self.conf_sum = 0
        self.max_bbox = 0
        self.max_mask = 0
        self.max_invasion = 0
        self.trigger_types = set()
        # LLM 状态
        self.llm_calls = 0
        self.llm_results = []  # list of VerifyResult
        self.llm_start_frame = None
        self.llm_mid_triggered = False


class BowlMemory:
    def __init__(self):
        self.slots = {}
        self._next_id = 0

    def _match_or_create(self, box, now):
        best_id, best_dist = None, float("inf")
        for bid, s in self.slots.items():
            d = bbox_center_distance(box, s["box"])
            avg = ((box[2] - box[0]) + (box[3] - box[1])) / 2
            if d < avg * 1.0 and d < best_dist:
                best_dist = d; best_id = bid
        if best_id is not None:
            self.slots[best_id]["box"] = box
            self.slots[best_id]["last_seen"] = now
            return best_id
        bid = self._next_id
        self._next_id += 1
        self.slots[bid] = {"box": box, "last_seen": now, "first_seen": now}
        return bid

    def update(self, bowls, now):
        ids = []
        for b in bowls:
            ids.append(self._match_or_create(b["box"], now))
        expired = [bid for bid, s in self.slots.items()
                    if now - s["last_seen"] > OCCLUSION_MEMORY_SEC]
        for bid in expired:
            del self.slots[bid]
        return ids

    def top_ghosts(self, now, n=MAX_GHOSTS_SHOWN):
        g = [(bid, s) for bid, s in self.slots.items()
             if 0.3 <= now - s["last_seen"] <= OCCLUSION_MEMORY_SEC]
        g.sort(key=lambda x: now - x[1]["last_seen"])
        return g[:n]


class V9RuleEngine:
    def __init__(self, use_llm=True):
        self.bowl_memory = BowlMemory()
        self.ongoing = {}
        self.event_id_ctr = 0
        self.verifier = get_verifier() if use_llm else None

    def update(self, animals, bowls, now, fh, fw, frame_bgr=None):
        bowl_ids = self.bowl_memory.update(bowls, now)
        debug_pairs = []

        for a in animals:
            a["_mask"] = (polygon_to_mask(a.get("mask_pts"), fh, fw)
                          if a.get("mask_pts") else None)

        # L1/L2/L3 判定
        for a in animals:
            for bid, b in zip(bowl_ids, bowls):
                bbox_r = bbox_overlap_ratio(a["box"], b["box"])
                mask_px = 0
                if a["_mask"] is not None and b.get("mask_pts"):
                    b_mask = polygon_to_mask(b["mask_pts"], fh, fw)
                    mask_px = mask_overlap_area(a["_mask"], b_mask)
                inv_r = 0
                if a["_mask"] is not None:
                    bm = bbox_to_mask(b["box"], fh, fw)
                    px = mask_overlap_area(a["_mask"], bm)
                    ba = int(np.sum(bm > 0))
                    inv_r = px / ba if ba > 0 else 0

                trig_b = bbox_r >= BBOX_MIN_RATIO
                trig_m = mask_px >= MASK_MIN_PIXELS
                trig_i = inv_r >= INVASION_MIN_RATIO
                trigger = trig_b or trig_m or trig_i
                types = []
                if trig_b: types.append("bbox")
                if trig_m: types.append("mask")
                if trig_i: types.append("侵入")

                debug_pairs.append({
                    "animal": a["cls"], "bowl_id": bid,
                    "bbox_ratio": bbox_r, "mask_px": mask_px,
                    "invasion_ratio": inv_r,
                    "trigger": trigger, "types": types,
                    "occluded": False,
                })
                if trigger:
                    self._add_hit((a["cls"], bid), a, bbox_r, mask_px,
                                   inv_r, now, types, frame_bgr)

        # L4 遮挡
        for bid, mem in self.bowl_memory.slots.items():
            gone = now - mem["last_seen"]
            if gone < OCCLUSION_MIN_GONE_SEC: continue
            if gone > OCCLUSION_MEMORY_SEC: continue
            for a in animals:
                bbox_r = bbox_overlap_ratio(a["box"], mem["box"])
                if bbox_r < 0.15: continue
                inv_r = 0
                if a["_mask"] is not None:
                    bm = bbox_to_mask(mem["box"], fh, fw)
                    px = mask_overlap_area(a["_mask"], bm)
                    ba = int(np.sum(bm > 0))
                    inv_r = px / ba if ba > 0 else 0
                if bbox_r >= 0.15 or inv_r >= 0.20:
                    debug_pairs.append({
                        "animal": a["cls"], "bowl_id": bid,
                        "bbox_ratio": bbox_r, "mask_px": 0,
                        "invasion_ratio": inv_r,
                        "trigger": True,
                        "types": [f"遮挡{gone:.1f}s"],
                        "occluded": True,
                    })
                    self._add_hit((a["cls"], bid), a, bbox_r, 0, inv_r,
                                   now, [f"遮挡{gone:.1f}s"], frame_bgr)

        # LLM 复核
        self._maybe_llm_verify(now, frame_bgr)

        # 结束事件
        finalized = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = now - ev.last_seen
            if gap < MAX_GAP_SEC: continue
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_EVENT_DURATION_SEC:
                # 结束前最后一次 LLM
                if (self.verifier and self.verifier.available
                        and frame_bgr is not None
                        and ev.llm_calls < LLM_MAX_PER_EVENT):
                    result = self.verifier.verify_behavior(
                        frame_bgr, ev.event_type, ev.animal_cls,
                        f"final-{ev.animal_cls}-{ev.bowl_id}",
                        extra_context=f"事件结束前,持续 {dur:.1f} 秒")
                    if result is not None:
                        ev.llm_results.append(result)
                        ev.llm_calls += 1

                self.event_id_ctr += 1
                confirmed = sum(1 for r in ev.llm_results if r.confirmed)
                llm_pass = (not self.verifier or not self.verifier.available
                             or confirmed >= 1)  # 至少 1 次确认
                finalized.append({
                    "id": f"evt-{self.event_id_ctr}",
                    "animal": ev.animal_cls, "bowl_id": ev.bowl_id,
                    "duration": dur, "hit": ev.hit,
                    "confidence": ev.conf_sum / max(1, ev.hit),
                    "max_bbox": ev.max_bbox,
                    "max_mask": ev.max_mask,
                    "max_invasion": ev.max_invasion,
                    "trigger_types": list(ev.trigger_types),
                    "llm_calls": ev.llm_calls,
                    "llm_confirmed": confirmed,
                    "llm_pass": llm_pass,
                    "llm_reasons": [r.reason for r in ev.llm_results],
                })
            to_remove.append(key)
        for k in to_remove:
            del self.ongoing[k]
        return debug_pairs, finalized

    def _maybe_llm_verify(self, now, frame_bgr):
        """检查每个 ongoing 事件是否需要 LLM 复核(触发/中期)"""
        if not self.verifier or not self.verifier.available:
            return
        if frame_bgr is None:
            return
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            # 首次复核:满足 MIN_DURATION 且未复核过
            if ev.llm_calls == 0 and dur >= MIN_EVENT_DURATION_SEC:
                result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type, ev.animal_cls,
                    f"start-{ev.animal_cls}-{ev.bowl_id}-{int(ev.start_time*10)}",
                    extra_context=(
                        f"YOLO 置信度 {ev.conf_sum/max(1,ev.hit):.2f}, "
                        f"触发方式: {','.join(sorted(ev.trigger_types))[:30]}"))
                if result is not None:
                    ev.llm_results.append(result)
                    ev.llm_calls += 1
            # 中期复核
            if (not ev.llm_mid_triggered and dur >= LLM_MID_CHECK_SEC
                    and ev.llm_calls < LLM_MAX_PER_EVENT):
                ev.llm_mid_triggered = True
                result = self.verifier.verify_behavior(
                    frame_bgr, ev.event_type, ev.animal_cls,
                    f"mid-{ev.animal_cls}-{ev.bowl_id}-{int(ev.start_time*10)}",
                    extra_context=f"事件已持续 {dur:.1f} 秒")
                if result is not None:
                    ev.llm_results.append(result)
                    ev.llm_calls += 1

    def _add_hit(self, key, a, bbox_r, mask_px, inv_r, now, types,
                  frame_bgr):
        if key not in self.ongoing:
            ev = LLMBackedOngoingEvent(a["cls"], key[1], "drinking", now)
            if frame_bgr is not None:
                ev.llm_start_frame = frame_bgr.copy()
            self.ongoing[key] = ev
        ev = self.ongoing[key]
        ev.last_seen = now
        ev.hit += 1
        ev.conf_sum += a["conf"]
        ev.max_bbox = max(ev.max_bbox, bbox_r)
        ev.max_mask = max(ev.max_mask, mask_px)
        ev.max_invasion = max(ev.max_invasion, inv_r)
        for t in types:
            ev.trigger_types.add(t)

    def force_flush(self, now, frame_bgr=None):
        finalized = []
        for key, ev in self.ongoing.items():
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_EVENT_DURATION_SEC:
                self.event_id_ctr += 1
                confirmed = sum(1 for r in ev.llm_results if r.confirmed)
                llm_pass = (not self.verifier or not self.verifier.available
                             or confirmed >= 1)
                finalized.append({
                    "id": f"evt-{self.event_id_ctr}",
                    "animal": ev.animal_cls, "bowl_id": ev.bowl_id,
                    "duration": dur, "hit": ev.hit,
                    "confidence": ev.conf_sum / max(1, ev.hit),
                    "max_bbox": ev.max_bbox,
                    "max_mask": ev.max_mask,
                    "max_invasion": ev.max_invasion,
                    "trigger_types": list(ev.trigger_types),
                    "llm_calls": ev.llm_calls,
                    "llm_confirmed": confirmed,
                    "llm_pass": llm_pass,
                    "llm_reasons": [r.reason for r in ev.llm_results],
                })
        self.ongoing.clear()
        return finalized


# ==================== 绘制(用 V8 的,略) ====================
def parse_and_draw(frame, r, names, downgrade_stats):
    class_counts = {}
    animals, bowls = [], []
    if r.boxes is None or len(r.boxes) == 0:
        return frame, class_counts, animals, bowls
    boxes = r.boxes.xyxy.cpu().numpy()
    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
    conf_arr = r.boxes.conf.cpu().numpy()
    masks_xy = r.masks.xy if r.masks is not None else None
    animal_conf_map = {}
    for i, cls in enumerate(cls_arr):
        n = names.get(int(cls), "")
        if n in ("cat", "dog", "monkey", "other_primate"):
            animal_conf_map[n] = max(animal_conf_map.get(n, 0),
                                       float(conf_arr[i]))
    overlay = frame.copy()
    for i, (box, cls, conf) in enumerate(zip(boxes, cls_arr, conf_arr)):
        original = names.get(int(cls), str(cls))
        name = maybe_downgrade_to_pet(original, float(conf), animal_conf_map)
        if name != original:
            downgrade_stats["count"] += 1
        color = CLASS_COLORS.get(name, (200, 200, 200))
        class_counts[name] = class_counts.get(name, 0) + 1
        mask_pts = None
        if masks_xy is not None and i < len(masks_xy):
            mask_pts = np.asarray(masks_xy[i]).tolist()
            if len(mask_pts) >= 3:
                cv2.fillPoly(
                    overlay, [np.asarray(mask_pts).astype(np.int32)],
                    color)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        info = {"box": box, "cls": name, "conf": float(conf),
                "mask_pts": mask_pts, "original": original}
        if name in ("cat", "dog", "monkey", "other_primate", "pet"):
            animals.append(info)
        elif name == "bowl":
            bowls.append(info)
    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return frame, class_counts, animals, bowls


def draw_ghost_bowls(frame, memory, now):
    for bid, mem in memory.top_ghosts(now):
        gone = now - mem["last_seen"]
        x1, y1, x2, y2 = [int(v) for v in mem["box"]]
        for x in range(x1, x2, 14):
            cv2.line(frame, (x, y1), (min(x + 7, x2), y1),
                     GHOST_BOWL_COLOR, 2)
            cv2.line(frame, (x, y2), (min(x + 7, x2), y2),
                     GHOST_BOWL_COLOR, 2)
        for y in range(y1, y2, 14):
            cv2.line(frame, (x1, y), (x1, min(y + 7, y2)),
                     GHOST_BOWL_COLOR, 2)
            cv2.line(frame, (x2, y), (x2, min(y + 7, y2)),
                     GHOST_BOWL_COLOR, 2)
        stat = ("等 3s" if gone < OCCLUSION_MIN_GONE_SEC
                else "遮挡触发!")
        frame = cv2_zh(
            frame, f"记忆盆#{bid} {gone:.1f}s {stat}",
            (x1, y2 + 4), 12, GHOST_BOWL_COLOR)
    return frame


def draw_trigger_indicator(frame, animals, debug_pairs):
    trig = set()
    for i, a in enumerate(animals):
        for p in debug_pairs:
            if p["animal"] == a["cls"] and p["trigger"]:
                trig.add(i)
    for i in trig:
        box = animals[i]["box"]
        x1, y1, x2, y2 = box.astype(int)
        cv2.circle(frame, (x1 + 20, y1 + 20), 12, TRIGGER_COLOR, -1)
        cv2.putText(frame, "!", (x1 + 15, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)


def draw_panel(h, debug_pairs, ongoing, latest, model_info, memory_slots,
                now, downgrade_stats, verifier):
    w = 360
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V9 · 5 层带 LLM",
                   (10, y), 15, (255, 255, 255))
    y += 26
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 15
    llm_state = f"LLM: {verifier.provider}" if verifier and verifier.available else "LLM: ✗ 未配 KEY"
    llm_color = LLM_OK_COLOR if verifier and verifier.available else LLM_NO_COLOR
    panel = cv2_zh(panel, llm_state, (10, y), 11, llm_color)
    y += 18
    panel = cv2_zh(panel, f"Pet 降级: {downgrade_stats['count']}",
                   (10, y), 11, (150, 200, 255))
    y += 22

    panel = cv2_zh(panel, "═ 5 层触发 ═",
                   (10, y), 13, (200, 200, 255))
    y += 18
    for text, c in [
        (f" L1 bbox ≥ {BBOX_MIN_RATIO:.0%}", (200, 200, 200)),
        (f" L2 mask ≥ {MASK_MIN_PIXELS}px", (200, 200, 200)),
        (f" L3 侵入 ≥ {INVASION_MIN_RATIO:.0%}", (255, 200, 0)),
        (f" L4 遮挡 ≥ {OCCLUSION_MIN_GONE_SEC:.0f}s", GHOST_BOWL_COLOR),
        (f" L5 LLM 至少 1 次确认", LLM_OK_COLOR),
    ]:
        panel = cv2_zh(panel, text, (10, y), 10, c)
        y += 14

    y += 8
    panel = cv2_zh(panel, "═ 当前判定 ═", (10, y), 13, (200, 200, 255))
    y += 18
    if not debug_pairs:
        panel = cv2_zh(panel, " (无)", (10, y), 11, (150, 150, 150))
        y += 15
    for p in debug_pairs[:4]:
        c = ((0, 200, 255) if p.get("occluded") else
             (100, 255, 100) if p["trigger"] else (170, 170, 170))
        panel = cv2_zh(
            panel, f" {p['animal']}<->bowl#{p['bowl_id']}",
            (10, y), 11, c)
        y += 13
        panel = cv2_zh(
            panel,
            f"  b={p['bbox_ratio']:.0%} m={p['mask_px']} 侵入={p['invasion_ratio']:.0%}",
            (10, y), 10, c)
        y += 15

    y += 5
    panel = cv2_zh(panel, "═ 进行中 + LLM 状态 ═",
                   (10, y), 13, (255, 255, 100))
    y += 18
    if not ongoing:
        panel = cv2_zh(panel, " (无)", (10, y), 11, (150, 150, 150))
        y += 15
    for key, ev in list(ongoing.items())[:3]:
        dur = ev.last_seen - ev.start_time
        confirmed = sum(1 for r in ev.llm_results if r.confirmed)
        panel = cv2_zh(
            panel, f" {ev.animal_cls} #{ev.bowl_id} {dur:.1f}s",
            (10, y), 11, (255, 255, 100))
        y += 13
        llm_line = f"  LLM: {ev.llm_calls}/{LLM_MAX_PER_EVENT} 次 ✓{confirmed}"
        if ev.llm_results:
            reason = ev.llm_results[-1].reason[:22]
            llm_line += f" '{reason}'"
        panel = cv2_zh(panel, llm_line, (10, y), 10,
                       LLM_OK_COLOR if confirmed > 0 else LLM_WAIT_COLOR)
        y += 15

    y += 8
    panel = cv2_zh(panel, "═ 盆记忆 ═", (10, y), 13, GHOST_BOWL_COLOR)
    y += 18
    for bid, mem in list(memory_slots.items())[:3]:
        gone = now - mem["last_seen"]
        if gone < 0.3: s = "🟢 见"
        elif gone < OCCLUSION_MIN_GONE_SEC: s = f"🟡 等{gone:.1f}s"
        else: s = f"🔴 遮挡{gone:.1f}s"
        panel = cv2_zh(panel, f" #{bid}: {s}", (10, y), 10, GHOST_BOWL_COLOR)
        y += 13

    y += 8
    panel = cv2_zh(panel, "═ 已完成 ═", (10, y), 13, (100, 255, 100))
    y += 18
    for e in latest[-4:]:
        panel = cv2_zh(panel, f" • {e}", (10, y), 10, (100, 255, 100))
        y += 13
    return panel


def draw_banner(w, active, flash, verifier):
    h = 70
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if active:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 12
        for t in active:
            banner = cv2_zh(banner, t, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_zh(banner, "监控中 · 无进行中行为",
                       (20, 12), 17, (180, 180, 180))
        llm_note = "LLM 层已启用,触发后 2s 内会复核" if verifier and verifier.available \
                    else "⚠️ LLM 未配 API_KEY,只用规则判断"
        banner = cv2_zh(banner, f"V9 · 5 层 · {llm_note}",
                       (20, 42), 12, (150, 150, 150))
    return banner


def format_active(ongoing):
    out = []
    for k, ev in ongoing.items():
        dur = ev.last_seen - ev.start_time
        types = ",".join(sorted(ev.trigger_types))[:20]
        confirmed = sum(1 for r in ev.llm_results if r.confirmed)
        llm_tag = f" LLM✓{confirmed}/{ev.llm_calls}" if ev.llm_calls else ""
        out.append(
            f"🐾 {ev.animal_cls.upper()} DRINKING "
            f"{dur:.1f}s [{types}]{llm_tag}")
    return out


def fit(display, max_w, max_h, zoom):
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h) * zoom
    if scale >= 1.0 and zoom == 1.0:
        return display
    return cv2.resize(display, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA)


def main():
    from ultralytics import YOLO
    mp = resolve_model()
    if not mp: return
    print(f"[+] 模型: {mp}")
    model = YOLO(mp)
    names = model.names
    print(f"[+] 类别: {names}")

    model_info = ("INT8 OpenVINO" if "openvino" in mp.lower()
                   else "ONNX" if ".onnx" in mp.lower()
                   else "PyTorch")

    verifier = get_verifier()
    if verifier.available:
        print(f"[+] ✅ LLM 复核层启用: {verifier.provider}/{verifier.model}")
    else:
        print(f"[!] ⚠️ LLM 未启用: 请在 .env 里配 LLM_API_KEY")

    vp = pick_file("选视频",
                    [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not vp: return
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened(): return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.80)
    max_h = int(scr_h * 0.80)

    rules = V9RuleEngine(use_llm=True)
    win = "宠物 AI V9 · LLM 复核"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max_w, max_h)
    cv2.moveWindow(win, 50, 50)

    paused = False
    frame_idx = 0
    prev_t = time.time()
    fps_smooth = src_fps
    latest = []
    flash = 0
    show_debug = True
    zoom = 1.0
    downgrade_stats = {"count": 0}
    save_dir = Path(__file__).parent / "screenshots"
    save_dir.mkdir(exist_ok=True)

    print(f"\n===== V9 =====")
    print(f"5 层触发: bbox / mask / 侵入 / 遮挡 / LLM 复核")
    print(f"LLM 调用: 每事件最多 3 次(触发/中期/结束)\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for e in rules.force_flush(frame_idx / src_fps, frame):
                    line = (f"{e['animal']} 吃/喝 {int(e['duration'])}s "
                            f"LLM {e['llm_confirmed']}/{e['llm_calls']} "
                            f"{'✅' if e['llm_pass'] else '❌'}")
                    latest.append(line)
                    print(f"[事件] {line}")
                    if e["llm_reasons"]:
                        for r in e["llm_reasons"]:
                            print(f"       LLM: {r}")
                paused = True
                if frame_idx == 0: break
                continue
            frame_idx += 1

            fh, fw = frame.shape[:2]
            r = model.predict(frame, conf=0.35,
                              agnostic_nms=AGNOSTIC_NMS,
                              verbose=False)[0]
            frame, cnt, animals, bowls = parse_and_draw(
                frame, r, names, downgrade_stats)

            now = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            debug_pairs, completed = rules.update(
                animals, bowls, now, fh, fw, frame_bgr=frame)
            now_set = set(rules.ongoing.keys())
            if now_set - was:
                flash = 6
                for k in now_set - was:
                    ev = rules.ongoing[k]
                    print(f"[!] 触发: {ev.animal_cls} bowl#{ev.bowl_id} "
                          f"[{','.join(sorted(ev.trigger_types))}]")
            if flash > 0: flash -= 1

            frame = draw_ghost_bowls(frame, rules.bowl_memory, now)
            draw_trigger_indicator(frame, animals, debug_pairs)

            for e in completed:
                line = (f"{e['animal']} 吃/喝 {int(e['duration'])}s "
                        f"LLM {e['llm_confirmed']}/{e['llm_calls']} "
                        f"{'✅' if e['llm_pass'] else '❌'}")
                latest.append(line)
                print(f"[事件] {line}")
                if e["llm_reasons"]:
                    for r in e["llm_reasons"]:
                        print(f"       LLM: {r}")

            t = time.time()
            dt = t - prev_t; prev_t = t
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            active = format_active(rules.ongoing)
            banner = draw_banner(fw, active, flash, verifier)
            info = np.zeros((30, fw, 3), dtype=np.uint8)
            cs = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(
                info, f"帧 {frame_idx}  {fps_smooth:.1f}FPS  {cs}",
                (10, 6), 13, (200, 200, 200))
            main_view = np.vstack([banner, frame, info])
            if show_debug:
                p = draw_panel(
                    main_view.shape[0], debug_pairs, rules.ongoing,
                    latest, model_info, rules.bowl_memory.slots, now,
                    downgrade_stats, verifier)
                display = np.hstack([main_view, p])
            else:
                display = main_view
            display = fit(display, max_w, max_h, zoom)
            cv2.imshow(win, display)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27): break
        elif key == ord(" "): paused = not paused
        elif key == ord("s"):
            fp = save_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(fp), display)
            print(f"[+] {fp}")
        elif key == ord("d"): show_debug = not show_debug
        elif key in (ord("+"), ord("=")): zoom = min(2.0, zoom + 0.1)
        elif key == ord("-"): zoom = max(0.3, zoom - 0.1)
        elif key == ord("r"): zoom = 1.0

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n共 {frame_idx} 帧, {len(latest)} 事件")
    for l in latest: print(f"  • {l}")


if __name__ == "__main__":
    main()
