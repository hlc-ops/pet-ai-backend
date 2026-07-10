"""视频播放器 V6

用户反馈:
- 去掉头部 30% 的鸡肋标注
- 直接:动物整体 bbox ∩ 盆 bbox,以及动物 mask ∩ 盆 mask
- 双判(bbox 或 mask 任一满足即触发)
- 加时序记忆:遮挡时用 5 秒前的盆位置推理

核心:更简单,更符合行业主流,能处理遮挡

触发条件(OR):
- bbox_ratio: 动物 bbox ∩ 盆 bbox 面积 / min(bbox 面积) >= 20%
- mask_pixels: 动物 mask ∩ 盆 mask >= 100 像素
- occluded: 盆消失 + 动物在原盆位置附近 3+ 秒
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque

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
}
TRIGGER_COLOR = (0, 100, 255)      # 红
GHOST_BOWL_COLOR = (128, 220, 128)  # 淡绿(过去的盆)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


# ==================== 参数 ====================
BBOX_MIN_RATIO = 0.20            # bbox 交集 / min bbox 面积
MASK_MIN_PIXELS = 100            # mask 相交像素数
OCCLUSION_MEMORY_SEC = 5.0       # 记住 5 秒内的盆位置
OCCLUSION_MIN_DURATION_SEC = 3.0 # 遮挡 + 动物在原位 3 秒 = 触发
MIN_EVENT_DURATION_SEC = 2.0     # 事件最短
MAX_GAP_SEC = 1.5                # 事件"消失"判定


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
    return pick_file("选模型", [("模型", "*.pt")])


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


# ==================== 几何工具 ====================
def bbox_overlap_ratio(box1, box2):
    """两个 xyxy 的交集面积 / 较小 bbox 面积"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    smaller = min(area1, area2)
    return inter / smaller if smaller > 0 else 0.0


def bbox_center_distance(box1, box2):
    """两 bbox 中心点距离"""
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ==================== 时序记忆 ====================
class BowlMemory:
    """记住最近见到的每个盆的 bbox + 消失时间"""
    def __init__(self, memory_sec=OCCLUSION_MEMORY_SEC):
        self.memory_sec = memory_sec
        # {bowl_id: {box, last_seen, first_seen}}
        self.slots = {}
        self._next_id = 0

    def _match_or_create(self, box, now):
        # 简单最近距离匹配
        best_id, best_dist = None, float("inf")
        for bid, b in self.slots.items():
            d = bbox_center_distance(box, b["box"])
            # 中心距离 < 平均尺寸的 50% 认为是同一个盆
            avg = ((box[2] - box[0]) + (box[3] - box[1])) / 2
            if d < avg * 0.5 and d < best_dist:
                best_dist = d
                best_id = bid
        if best_id is not None:
            self.slots[best_id]["box"] = box
            self.slots[best_id]["last_seen"] = now
            return best_id
        # 新建
        bid = self._next_id
        self._next_id += 1
        self.slots[bid] = {"box": box, "last_seen": now, "first_seen": now}
        return bid

    def update(self, bowls, now):
        """输入本帧检测到的盆列表 [{box, ...}, ...],返回 bowl_id 列表"""
        ids = []
        for b in bowls:
            bid = self._match_or_create(b["box"], now)
            ids.append(bid)
        # 清理过期
        expired = [bid for bid, b in self.slots.items()
                    if now - b["last_seen"] > self.memory_sec]
        for bid in expired:
            del self.slots[bid]
        return ids

    def ghost_bowls(self, now):
        """返回:最近失踪(消失 < memory_sec 且没在本帧被 update)的盆"""
        # 需要在 update 之前调用
        return [(bid, b["box"]) for bid, b in self.slots.items()
                if now - b["last_seen"] > 0.5]  # 至少 0.5 秒没见到才算


# ==================== 规则引擎 ====================
class V6RuleEngine:
    def __init__(self):
        self.bowl_memory = BowlMemory()
        # ongoing[(animal_cls, bowl_id)] = {start, last_seen, hit, conf_sum,
        #                                    max_bbox_ratio, max_mask_px,
        #                                    trigger_type}
        self.ongoing = {}
        self.event_id_ctr = 0

    def update(self, animals, bowls, now, frame_h, frame_w):
        # 先记录本帧盆位置
        bowl_ids = self.bowl_memory.update(bowls, now)
        # 消失(遮挡)的盆
        ghosts_before_update = []  # 已经处理,不用

        debug_pairs = []
        active_keys = set()

        # ---------- 判定 1:动物 vs 当前帧盆 ----------
        for a in animals:
            a_mask = polygon_to_mask(a.get("mask_pts"), frame_h, frame_w) \
                if a.get("mask_pts") else None
            for bid, b, cur_frame in [(bid, b, True) for bid, b in
                                       zip(bowl_ids, bowls)]:
                bbox_r = bbox_overlap_ratio(a["box"], b["box"])
                mask_px = 0
                if a_mask is not None and b.get("mask_pts"):
                    b_mask = polygon_to_mask(
                        b["mask_pts"], frame_h, frame_w)
                    mask_px = mask_overlap_area(a_mask, b_mask)

                trigger_by_bbox = bbox_r >= BBOX_MIN_RATIO
                trigger_by_mask = mask_px >= MASK_MIN_PIXELS
                trigger = trigger_by_bbox or trigger_by_mask
                trigger_type = ("bbox+mask" if trigger_by_bbox and trigger_by_mask
                                 else "bbox" if trigger_by_bbox
                                 else "mask" if trigger_by_mask
                                 else "无")

                debug = {
                    "animal": a["cls"], "bowl_id": bid,
                    "bbox_ratio": bbox_r, "mask_px": mask_px,
                    "trigger": trigger, "trigger_type": trigger_type,
                    "occluded": False,
                }
                debug_pairs.append(debug)

                if not trigger:
                    continue

                self._add_hit(
                    (a["cls"], bid), a, bbox_r, mask_px, now, trigger_type)
                active_keys.add((a["cls"], bid))

        # ---------- 判定 2:遮挡推理(盆消失,动物在原位置) ----------
        for bid, mem in self.bowl_memory.slots.items():
            gone_time = now - mem["last_seen"]
            if gone_time < 0.3 or gone_time > OCCLUSION_MEMORY_SEC:
                continue  # 太短或太长

            # 有没有动物在原盆位置附近?
            for a in animals:
                bbox_r = bbox_overlap_ratio(a["box"], mem["box"])
                if bbox_r < 0.15:
                    continue

                # 有动物在原位,认为可能遮挡进食
                debug_pairs.append({
                    "animal": a["cls"], "bowl_id": bid,
                    "bbox_ratio": bbox_r, "mask_px": 0,
                    "trigger": True, "trigger_type": "遮挡",
                    "occluded": True,
                })
                self._add_hit(
                    (a["cls"], bid), a, bbox_r, 0, now, "遮挡")
                active_keys.add((a["cls"], bid))

        # ---------- 结束过期事件 ----------
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
                    "animal": ev["animal"],
                    "bowl_id": ev["bowl_id"],
                    "duration": dur, "hit": ev["hit"],
                    "confidence": ev["conf_sum"] / max(1, ev["hit"]),
                    "max_bbox_ratio": ev["max_bbox"],
                    "max_mask_px": ev["max_mask"],
                    "trigger_types": ev["trigger_types"],
                })
            to_remove.append(key)
        for k in to_remove:
            del self.ongoing[k]

        return debug_pairs, finalized

    def _add_hit(self, key, animal, bbox_r, mask_px, now, ttype):
        if key not in self.ongoing:
            self.ongoing[key] = {
                "start": now, "last_seen": now, "hit": 0,
                "conf_sum": 0, "max_bbox": 0, "max_mask": 0,
                "animal": animal["cls"], "bowl_id": key[1],
                "trigger_types": set(),
            }
        ev = self.ongoing[key]
        ev["last_seen"] = now
        ev["hit"] += 1
        ev["conf_sum"] += animal["conf"]
        ev["max_bbox"] = max(ev["max_bbox"], bbox_r)
        ev["max_mask"] = max(ev["max_mask"], mask_px)
        ev["trigger_types"].add(ttype)

    def force_flush(self, now):
        finalized = []
        for key, ev in self.ongoing.items():
            dur = ev["last_seen"] - ev["start"]
            if dur >= MIN_EVENT_DURATION_SEC:
                self.event_id_ctr += 1
                finalized.append({
                    "id": f"evt-{self.event_id_ctr}",
                    "animal": ev["animal"],
                    "bowl_id": ev["bowl_id"],
                    "duration": dur, "hit": ev["hit"],
                    "confidence": ev["conf_sum"] / max(1, ev["hit"]),
                    "max_bbox_ratio": ev["max_bbox"],
                    "max_mask_px": ev["max_mask"],
                    "trigger_types": ev["trigger_types"],
                })
        self.ongoing.clear()
        return finalized


# ==================== 检测绘制 ====================
def parse_and_draw(frame, r, names):
    class_counts = {}
    animals, bowls = [], []
    if r.boxes is None or len(r.boxes) == 0:
        return frame, class_counts, animals, bowls

    boxes = r.boxes.xyxy.cpu().numpy()
    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
    conf_arr = r.boxes.conf.cpu().numpy()
    masks_xy = r.masks.xy if r.masks is not None else None

    overlay = frame.copy()
    for i, (box, cls, conf) in enumerate(zip(boxes, cls_arr, conf_arr)):
        name = names.get(int(cls), str(cls))
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
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        info = {"box": box, "cls": name, "conf": float(conf),
                "mask_pts": mask_pts}
        if name in ("cat", "dog", "monkey", "other_primate"):
            animals.append(info)
        elif name == "bowl":
            bowls.append(info)

    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return frame, class_counts, animals, bowls


def draw_ghost_bowls(frame, memory, now):
    """画消失中的盆(遮挡),用淡绿虚线"""
    for bid, mem in memory.slots.items():
        gone = now - mem["last_seen"]
        if gone < 0.3:
            continue  # 还在见,不画
        box = mem["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        # 虚线矩形
        for x in range(x1, x2, 12):
            cv2.line(frame, (x, y1), (min(x + 6, x2), y1),
                     GHOST_BOWL_COLOR, 2)
            cv2.line(frame, (x, y2), (min(x + 6, x2), y2),
                     GHOST_BOWL_COLOR, 2)
        for y in range(y1, y2, 12):
            cv2.line(frame, (x1, y), (x1, min(y + 6, y2)),
                     GHOST_BOWL_COLOR, 2)
            cv2.line(frame, (x2, y), (x2, min(y + 6, y2)),
                     GHOST_BOWL_COLOR, 2)
        frame = cv2_zh(
            frame,
            f"记忆盆 #{bid} 消失{gone:.1f}s",
            (x1, y2 + 4), 12, GHOST_BOWL_COLOR)
    return frame


def draw_trigger_highlight(frame, animals, debug_pairs):
    """给触发的动物 bbox 画红边"""
    triggered_indices = set()
    for i, a in enumerate(animals):
        for p in debug_pairs:
            if p["animal"] == a["cls"] and p["trigger"]:
                triggered_indices.add(i)
    for i in triggered_indices:
        box = animals[i]["box"]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                     TRIGGER_COLOR, 4)


def draw_panel(h, debug_pairs, ongoing, latest, model_info,
                memory_slots, now):
    w = 340
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V6 · 双判 + 遮挡感知",
                   (10, y), 16, (255, 255, 255))
    y += 28
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 22

    # 触发规则
    panel = cv2_zh(panel, "═ 触发规则(OR)═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    panel = cv2_zh(
        panel, f" bbox 面积比 ≥ {BBOX_MIN_RATIO:.0%}",
        (10, y), 11, (200, 200, 200))
    y += 15
    panel = cv2_zh(
        panel, f" mask 像素 ≥ {MASK_MIN_PIXELS}",
        (10, y), 11, (200, 200, 200))
    y += 15
    panel = cv2_zh(
        panel, f" 遮挡: 盆消失+动物在原位",
        (10, y), 11, (200, 200, 200))
    y += 22

    panel = cv2_zh(panel, "═ 当前对判 ═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    if not debug_pairs:
        panel = cv2_zh(panel, " (无对)",
                       (10, y), 11, (150, 150, 150))
        y += 18
    for p in debug_pairs[:6]:
        c = ((0, 200, 255) if p.get("occluded") else
             (100, 255, 100) if p["trigger"] else (170, 170, 170))
        panel = cv2_zh(
            panel, f" {p['animal']}<->bowl#{p['bowl_id']}",
            (10, y), 12, c)
        y += 15
        panel = cv2_zh(
            panel,
            f"  bbox={p['bbox_ratio']:.2%}  mask={p['mask_px']}px",
            (10, y), 10, c)
        y += 13
        panel = cv2_zh(
            panel, f"  {p['trigger_type']} "
                    f"{'🟢' if p['trigger'] else '🔴'}",
            (10, y), 10, c)
        y += 18

    # 时序记忆
    y += 5
    panel = cv2_zh(panel, "═ 盆记忆库 ═",
                   (10, y), 13, (128, 220, 128))
    y += 20
    for bid, mem in list(memory_slots.items())[:5]:
        gone = now - mem["last_seen"]
        state = "在" if gone < 0.3 else f"消{gone:.1f}s"
        panel = cv2_zh(
            panel, f" bowl#{bid}: {state}",
            (10, y), 11, (128, 220, 128))
        y += 16

    y += 5
    panel = cv2_zh(panel, "═ 进行中 ═",
                   (10, y), 13, (255, 255, 100))
    y += 20
    if not ongoing:
        panel = cv2_zh(panel, " (无)",
                       (10, y), 11, (150, 150, 150))
        y += 18
    for key, ev in list(ongoing.items())[:3]:
        dur = ev["last_seen"] - ev["start"]
        panel = cv2_zh(
            panel,
            f" {ev['animal']} #{ev['bowl_id']} {dur:.1f}s",
            (10, y), 11, (255, 255, 100))
        y += 14
        panel = cv2_zh(
            panel,
            f"  {','.join(ev['trigger_types'])} hits={ev['hit']}",
            (10, y), 10, (255, 255, 100))
        y += 18

    y += 5
    panel = cv2_zh(panel, "═ 已完成事件 ═",
                   (10, y), 13, (100, 255, 100))
    y += 20
    for e in latest[-5:]:
        panel = cv2_zh(panel, f" • {e}",
                       (10, y), 10, (100, 255, 100))
        y += 15
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
            "V6:动物 vs 盆 直接判(bbox+mask+遮挡三重)",
            (20, 38), 12, (150, 150, 150))
    return banner


def format_active(ongoing):
    out = []
    for key, ev in ongoing.items():
        dur = ev["last_seen"] - ev["start"]
        types = ",".join(sorted(ev["trigger_types"]))
        out.append(
            f"🐾 {ev['animal'].upper()} 正在 DRINKING "
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

    model_info = (
        "INT8(可能掉 mask)" if "openvino" in mp.lower()
        else "ONNX" if ".onnx" in mp.lower()
        else "PyTorch(best.pt)")

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

    rules = V6RuleEngine()
    win = "宠物 AI V6 · 双判 + 遮挡"
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
    save_dir = Path(__file__).parent / "screenshots"
    save_dir.mkdir(exist_ok=True)

    print("\n===== V6 播放开始 =====")
    print("触发规则(OR):")
    print(f"  bbox 面积比 >= {BBOX_MIN_RATIO:.0%}")
    print(f"  mask 像素 >= {MASK_MIN_PIXELS}")
    print(f"  遮挡: 盆消失 + 动物在原位")
    print(f"事件最短 {MIN_EVENT_DURATION_SEC}秒\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for e in rules.force_flush(frame_idx / src_fps):
                    line = (f"{e['animal']} 吃/喝 "
                            f"{int(e['duration'])}s "
                            f"bbox={e['max_bbox_ratio']:.0%} "
                            f"mask={e['max_mask_px']}px "
                            f"[{','.join(sorted(e['trigger_types']))}]")
                    latest.append(line)
                    print(f"[事件] {line}")
                paused = True
                if frame_idx == 0: break
                continue
            frame_idx += 1

            fh, fw = frame.shape[:2]
            r = model.predict(frame, conf=0.35, verbose=False)[0]
            frame, cnt, animals, bowls = parse_and_draw(frame, r, names)

            now = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            debug_pairs, completed = rules.update(
                animals, bowls, now, fh, fw)
            now_set = set(rules.ongoing.keys())
            if now_set - was:
                flash = 6
                for k in now_set - was:
                    ev = rules.ongoing[k]
                    types = ",".join(sorted(ev["trigger_types"]))
                    print(f"[!] 触发: {ev['animal']} bowl#{ev['bowl_id']} "
                          f"[{types}]")
            if flash > 0: flash -= 1

            # 画消失的盆(遮挡记忆)
            frame = draw_ghost_bowls(frame, rules.bowl_memory, now)
            # 画触发的动物红边
            draw_trigger_highlight(frame, animals, debug_pairs)

            for e in completed:
                line = (f"{e['animal']} 吃/喝 "
                        f"{int(e['duration'])}s "
                        f"bbox={e['max_bbox_ratio']:.0%} "
                        f"mask={e['max_mask_px']}px "
                        f"[{','.join(sorted(e['trigger_types']))}]")
                latest.append(line)
                print(f"[事件] {line}")

            t = time.time()
            dt = t - prev_t
            prev_t = t
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            active = format_active(rules.ongoing)
            banner = draw_banner(fw, active, flash)

            info = np.zeros((30, fw, 3), dtype=np.uint8)
            cs = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(
                info,
                f"帧 {frame_idx}  {fps_smooth:.1f}FPS  {cs}",
                (10, 6), 13, (200, 200, 200))

            main_view = np.vstack([banner, frame, info])
            if show_debug:
                p = draw_panel(
                    main_view.shape[0], debug_pairs, rules.ongoing,
                    latest, model_info, rules.bowl_memory.slots, now)
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
