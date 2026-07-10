"""视频播放器 V8

修 V7 三处:
1. 猫身上两个 bbox(cat 和 dog 重叠)→ 加 agnostic_nms=True 跨类去重
2. 猫狗互认时智能降级到 "pet" 标签(不改模型,后处理)
3. 底层触发规则同 V7(4 层 OR)

新增:
- 显示"pet"降级标签(当 top1 和 top2 差 < 10% 时)
- 底部信息条显示"NMS 去重了 N 个重复框"
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from mask_utils import polygon_to_mask, mask_overlap_area


CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
    "pet": (200, 200, 200),        # 灰色兜底
}
TRIGGER_COLOR = (0, 100, 255)
GHOST_BOWL_COLOR = (128, 220, 128)

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

# ==================== V8 新增 ====================
AGNOSTIC_NMS = True                # 跨类 NMS(修双框)
PET_UNCERTAINTY_THRESHOLD = 0.15   # top1-top2 conf 差 < 0.15 → 用 pet
PET_MAX_CONF = 0.60                # top1 < 0.60 → 用 pet


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
              here / "model" / "best.onnx",
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


# ==================== V8 新增:后处理 "pet" 降级 ====================
def maybe_downgrade_to_pet(cls_name: str, conf: float,
                           other_animal_confs: dict) -> str:
    """
    如果:
      - top1 置信度 < PET_MAX_CONF (0.60)
      OR
      - top2 存在且 conf 与 top1 差 < PET_UNCERTAINTY_THRESHOLD (0.15)
    → 降级到 "pet"
    """
    if cls_name not in ("cat", "dog", "monkey", "other_primate"):
        return cls_name

    if conf < PET_MAX_CONF:
        return "pet"

    # 有没有另一个动物类的高置信度?
    max_other = max(
        [c for k, c in other_animal_confs.items() if k != cls_name],
        default=0)
    if conf - max_other < PET_UNCERTAINTY_THRESHOLD and max_other > 0.3:
        return "pet"

    return cls_name


# ==================== BowlMemory ====================
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


# ==================== 规则(和 V7 一样)====================
class V8RuleEngine:
    def __init__(self):
        self.bowl_memory = BowlMemory()
        self.ongoing = {}
        self.event_id_ctr = 0

    def update(self, animals, bowls, now, fh, fw):
        bowl_ids = self.bowl_memory.update(bowls, now)
        debug_pairs = []

        for a in animals:
            a["_mask"] = (polygon_to_mask(a.get("mask_pts"), fh, fw)
                          if a.get("mask_pts") else None)

        for a in animals:
            for bid, b in zip(bowl_ids, bowls):
                bbox_r = bbox_overlap_ratio(a["box"], b["box"])
                mask_px = 0
                if a["_mask"] is not None and b.get("mask_pts"):
                    b_mask = polygon_to_mask(b["mask_pts"], fh, fw)
                    mask_px = mask_overlap_area(a["_mask"], b_mask)
                invasion_ratio = 0
                if a["_mask"] is not None:
                    bbb_mask = bbox_to_mask(b["box"], fh, fw)
                    invade_px = mask_overlap_area(a["_mask"], bbb_mask)
                    ba = int(np.sum(bbb_mask > 0))
                    invasion_ratio = invade_px / ba if ba > 0 else 0

                trig_bbox = bbox_r >= BBOX_MIN_RATIO
                trig_mask = mask_px >= MASK_MIN_PIXELS
                trig_inv = invasion_ratio >= INVASION_MIN_RATIO
                trigger = trig_bbox or trig_mask or trig_inv
                types = []
                if trig_bbox: types.append("bbox")
                if trig_mask: types.append("mask")
                if trig_inv: types.append("侵入")

                debug_pairs.append({
                    "animal": a["cls"], "bowl_id": bid,
                    "bbox_ratio": bbox_r, "mask_px": mask_px,
                    "invasion_ratio": invasion_ratio,
                    "trigger": trigger, "types": types,
                    "occluded": False,
                })
                if trigger:
                    self._add_hit(
                        (a["cls"], bid), a, bbox_r, mask_px,
                        invasion_ratio, now, types)

        for bid, mem in self.bowl_memory.slots.items():
            gone = now - mem["last_seen"]
            if gone < OCCLUSION_MIN_GONE_SEC: continue
            if gone > OCCLUSION_MEMORY_SEC: continue
            for a in animals:
                bbox_r = bbox_overlap_ratio(a["box"], mem["box"])
                if bbox_r < 0.15: continue
                invasion_r = 0
                if a["_mask"] is not None:
                    bm = bbox_to_mask(mem["box"], fh, fw)
                    px = mask_overlap_area(a["_mask"], bm)
                    ba = int(np.sum(bm > 0))
                    invasion_r = px / ba if ba > 0 else 0
                if bbox_r >= 0.15 or invasion_r >= 0.20:
                    debug_pairs.append({
                        "animal": a["cls"], "bowl_id": bid,
                        "bbox_ratio": bbox_r, "mask_px": 0,
                        "invasion_ratio": invasion_r,
                        "trigger": True,
                        "types": [f"遮挡{gone:.1f}s"],
                        "occluded": True,
                    })
                    self._add_hit(
                        (a["cls"], bid), a, bbox_r, 0, invasion_r, now,
                        [f"遮挡{gone:.1f}s"])

        finalized = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = now - ev["last_seen"]
            if gap < MAX_GAP_SEC: continue
            dur = ev["last_seen"] - ev["start"]
            if dur >= MIN_EVENT_DURATION_SEC:
                self.event_id_ctr += 1
                finalized.append({
                    "id": f"evt-{self.event_id_ctr}",
                    "animal": ev["animal"], "bowl_id": ev["bowl_id"],
                    "duration": dur, "hit": ev["hit"],
                    "confidence": ev["conf_sum"] / max(1, ev["hit"]),
                    "max_bbox": ev["max_bbox"],
                    "max_mask": ev["max_mask"],
                    "max_invasion": ev["max_invasion"],
                    "trigger_types": list(ev["trigger_types"]),
                })
            to_remove.append(key)
        for k in to_remove:
            del self.ongoing[k]
        return debug_pairs, finalized

    def _add_hit(self, key, a, bbox_r, mask_px, inv_r, now, types):
        if key not in self.ongoing:
            self.ongoing[key] = {
                "start": now, "last_seen": now, "hit": 0,
                "conf_sum": 0, "max_bbox": 0, "max_mask": 0,
                "max_invasion": 0,
                "animal": a["cls"], "bowl_id": key[1],
                "trigger_types": set(),
            }
        ev = self.ongoing[key]
        ev["last_seen"] = now
        ev["hit"] += 1
        ev["conf_sum"] += a["conf"]
        ev["max_bbox"] = max(ev["max_bbox"], bbox_r)
        ev["max_mask"] = max(ev["max_mask"], mask_px)
        ev["max_invasion"] = max(ev["max_invasion"], inv_r)
        for t in types:
            ev["trigger_types"].add(t)

    def force_flush(self, now):
        finalized = []
        for key, ev in self.ongoing.items():
            dur = ev["last_seen"] - ev["start"]
            if dur >= MIN_EVENT_DURATION_SEC:
                self.event_id_ctr += 1
                finalized.append({
                    "id": f"evt-{self.event_id_ctr}",
                    "animal": ev["animal"], "bowl_id": ev["bowl_id"],
                    "duration": dur, "hit": ev["hit"],
                    "confidence": ev["conf_sum"] / max(1, ev["hit"]),
                    "max_bbox": ev["max_bbox"],
                    "max_mask": ev["max_mask"],
                    "max_invasion": ev["max_invasion"],
                    "trigger_types": list(ev["trigger_types"]),
                })
        self.ongoing.clear()
        return finalized


# ==================== 绘制 ====================
def parse_and_draw(frame, r, names, downgrade_stats):
    """V8: 加 pet 降级 + 记录 NMS 效果"""
    class_counts = {}
    animals, bowls = [], []
    if r.boxes is None or len(r.boxes) == 0:
        return frame, class_counts, animals, bowls

    boxes = r.boxes.xyxy.cpu().numpy()
    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
    conf_arr = r.boxes.conf.cpu().numpy()
    masks_xy = r.masks.xy if r.masks is not None else None

    # 收集本帧动物类的置信度分布(为 pet 降级用)
    animal_conf_map = {}
    for i, cls in enumerate(cls_arr):
        n = names.get(int(cls), "")
        if n in ("cat", "dog", "monkey", "other_primate"):
            animal_conf_map[n] = max(animal_conf_map.get(n, 0),
                                       float(conf_arr[i]))

    overlay = frame.copy()
    for i, (box, cls, conf) in enumerate(zip(boxes, cls_arr, conf_arr)):
        original = names.get(int(cls), str(cls))
        # V8: 降级
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
        if name == "pet":
            label += f" (原{original})"
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
            frame,
            f"记忆盆#{bid} {gone:.1f}s {stat}",
            (x1, y2 + 4), 12, GHOST_BOWL_COLOR)
    return frame


def draw_trigger_highlight(frame, animals, debug_pairs):
    """V8.1 修:改成在触发的动物 bbox 内画一层"感叹号"提示,
    不再画外框(避免看起来像双 bbox)"""
    trig = set()
    for i, a in enumerate(animals):
        for p in debug_pairs:
            if p["animal"] == a["cls"] and p["trigger"]:
                trig.add(i)
    for i in trig:
        box = animals[i]["box"]
        x1, y1, x2, y2 = box.astype(int)
        # 在 bbox 左上角内侧画一个红色感叹号提示
        cv2.circle(frame, (x1 + 20, y1 + 20), 12, TRIGGER_COLOR, -1)
        cv2.putText(frame, "!", (x1 + 15, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)


def draw_panel(h, debug_pairs, ongoing, latest, model_info,
                memory_slots, now, nms_agnostic, pet_downgrades):
    w = 340
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V8 · 跨类NMS+Pet降级",
                   (10, y), 15, (255, 255, 255))
    y += 26
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 18
    panel = cv2_zh(panel,
                   f"跨类 NMS: {'✓' if nms_agnostic else '✗'} "
                   f"| Pet 降级次数: {pet_downgrades}",
                   (10, y), 11, (150, 200, 255))
    y += 20

    panel = cv2_zh(panel, "═ 4 层触发 ═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    for text, c in [
        (f" L1 bbox 面积比 ≥ {BBOX_MIN_RATIO:.0%}", (200, 200, 200)),
        (f" L2 mask 相交 ≥ {MASK_MIN_PIXELS}px", (200, 200, 200)),
        (f" L3 侵入盆区 ≥ {INVASION_MIN_RATIO:.0%}", (255, 200, 0)),
        (f" L4 遮挡 ≥ {OCCLUSION_MIN_GONE_SEC:.0f}s", GHOST_BOWL_COLOR),
    ]:
        panel = cv2_zh(panel, text, (10, y), 11, c)
        y += 15

    y += 5
    panel = cv2_zh(panel, "═ 当前对判 ═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    if not debug_pairs:
        panel = cv2_zh(panel, " (无对)", (10, y), 11, (150, 150, 150))
        y += 18
    for p in debug_pairs[:5]:
        c = ((0, 200, 255) if p.get("occluded") else
             (100, 255, 100) if p["trigger"] else (170, 170, 170))
        panel = cv2_zh(
            panel, f" {p['animal']}<->bowl#{p['bowl_id']}",
            (10, y), 12, c)
        y += 14
        panel = cv2_zh(
            panel,
            f"  bbox={p['bbox_ratio']:.0%} mask={p['mask_px']}",
            (10, y), 10, c)
        y += 12
        panel = cv2_zh(
            panel,
            f"  侵入={p['invasion_ratio']:.0%}  "
            f"{','.join(p['types']) if p['types'] else '无'}",
            (10, y), 10, c)
        y += 18

    y += 5
    panel = cv2_zh(panel, "═ 盆记忆 ═",
                   (10, y), 13, GHOST_BOWL_COLOR)
    y += 20
    for bid, mem in list(memory_slots.items())[:4]:
        gone = now - mem["last_seen"]
        if gone < 0.3:
            s = "🟢 见"
        elif gone < OCCLUSION_MIN_GONE_SEC:
            s = f"🟡 等{gone:.1f}s"
        else:
            s = f"🔴 遮挡{gone:.1f}s"
        panel = cv2_zh(
            panel, f" bowl#{bid}: {s}",
            (10, y), 11, GHOST_BOWL_COLOR)
        y += 15

    y += 5
    panel = cv2_zh(panel, "═ 已完成 ═",
                   (10, y), 13, (100, 255, 100))
    y += 20
    for e in latest[-5:]:
        panel = cv2_zh(panel, f" • {e}", (10, y), 10, (100, 255, 100))
        y += 14
    return panel


def draw_banner(w, active, flash):
    h = 65
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if active:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 12
        for t in active:
            banner = cv2_zh(banner, t, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_zh(
            banner, "监控中 · 无进行中行为",
            (20, 12), 17, (180, 180, 180))
        banner = cv2_zh(
            banner,
            "V8 · 跨类 NMS 修双框 + Pet 降级修错认 + 4 层触发",
            (20, 38), 12, (150, 150, 150))
    return banner


def format_active(ongoing):
    out = []
    for k, ev in ongoing.items():
        dur = ev["last_seen"] - ev["start"]
        types = ",".join(sorted(ev["trigger_types"]))[:30]
        out.append(f"🐾 {ev['animal'].upper()} DRINKING "
                    f"{dur:.1f}s [{types}]")
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

    if "openvino" in mp.lower():
        model_info = "INT8 OpenVINO"
    elif ".onnx" in mp.lower():
        model_info = "ONNX FP16"
    else:
        model_info = "PyTorch"

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

    rules = V8RuleEngine()
    win = "宠物 AI V8"
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

    print(f"\n===== V8 =====")
    print(f"跨类 NMS: {'开' if AGNOSTIC_NMS else '关'}")
    print(f"Pet 降级阈值: conf < {PET_MAX_CONF} 或 top1-top2 < {PET_UNCERTAINTY_THRESHOLD}\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for e in rules.force_flush(frame_idx / src_fps):
                    line = (f"{e['animal']} 吃/喝 {int(e['duration'])}s "
                            f"[{','.join(e['trigger_types'])[:30]}]")
                    latest.append(line)
                    print(f"[事件] {line}")
                paused = True
                if frame_idx == 0: break
                continue
            frame_idx += 1

            fh, fw = frame.shape[:2]
            # V8: agnostic_nms=True 跨类去重
            r = model.predict(frame, conf=0.35,
                              agnostic_nms=AGNOSTIC_NMS,
                              verbose=False)[0]
            frame, cnt, animals, bowls = parse_and_draw(
                frame, r, names, downgrade_stats)

            now = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            debug_pairs, completed = rules.update(
                animals, bowls, now, fh, fw)
            now_set = set(rules.ongoing.keys())
            if now_set - was:
                flash = 6
                for k in now_set - was:
                    ev = rules.ongoing[k]
                    print(f"[!] 触发: {ev['animal']} bowl#{ev['bowl_id']} "
                          f"[{','.join(sorted(ev['trigger_types']))}]")
            if flash > 0: flash -= 1

            frame = draw_ghost_bowls(frame, rules.bowl_memory, now)
            draw_trigger_highlight(frame, animals, debug_pairs)

            for e in completed:
                line = (f"{e['animal']} 吃/喝 {int(e['duration'])}s "
                        f"[{','.join(e['trigger_types'])[:30]}]")
                latest.append(line)
                print(f"[事件] {line}")

            t = time.time()
            dt = t - prev_t; prev_t = t
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            active = format_active(rules.ongoing)
            banner = draw_banner(fw, active, flash)
            info = np.zeros((30, fw, 3), dtype=np.uint8)
            cs = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(
                info,
                f"帧 {frame_idx}  {fps_smooth:.1f}FPS  {cs}  "
                f"Pet降级={downgrade_stats['count']}",
                (10, 6), 13, (200, 200, 200))
            main_view = np.vstack([banner, frame, info])
            if show_debug:
                p = draw_panel(
                    main_view.shape[0], debug_pairs, rules.ongoing,
                    latest, model_info, rules.bowl_memory.slots, now,
                    AGNOSTIC_NMS, downgrade_stats["count"])
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
    print(f"Pet 降级触发 {downgrade_stats['count']} 次")
    for l in latest: print(f"  • {l}")


if __name__ == "__main__":
    main()
