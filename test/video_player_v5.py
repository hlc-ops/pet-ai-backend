"""视频播放器 V5 - 像素级 mask 判定

用户的关键洞察:
    "像素分割的意义就是,只有真实像素重合才判定进食"

V4 之前用 bbox IoU 判定,这是浪费了分割能力。
V5 用 mask 交集判定:动物头部 mask ∩ 盆 mask > 0 才触发。

可视化:
- 半透明动物 mask
- 半透明盆 mask
- 头部 mask 用亮青色描边
- 头部 ∩ 盆 的**像素级重叠区**用**闪烁红色高亮**
- 一眼看到"到底哪些像素在接触"

默认加载:优先 best.pt,兜底 INT8。
指定 INT8:python video_player_v5.py model/best_openvino_model
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
from mask_utils import (polygon_to_mask, polygon_area,
                         head_region_mask, mask_overlap_area,
                         mask_overlap_ratio, visualize_overlap)


CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
}
HEAD_ZONE_COLOR = (255, 255, 0)      # 青
TRIGGER_COLOR = (0, 100, 255)        # 红

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


# 触发参数(mask 版)
MIN_OVERLAP_PIXELS = 50        # 头部 ∩ 盆 至少 50 像素才算触发
MIN_OVERLAP_RATIO = 0.05       # 或者 交集 / 头部mask >= 5%
MIN_DURATION_SEC = 2.0         # 事件最短持续
MAX_GAP_SEC = 1.5              # 事件"结束"判定间隔


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


# ==================== V5 核心:mask 版本判定 ====================
class MaskOngoingEvent:
    def __init__(self, animal_cls, event_type, start_time):
        self.animal_cls = animal_cls
        self.event_type = event_type
        self.start_time = start_time
        self.last_seen = start_time
        self.hit_count = 0
        self.conf_sum = 0
        self.max_overlap = 0


class MaskRuleEngine:
    def __init__(self):
        # key = (animal_cls_id, bowl_id)
        self.ongoing = {}
        self.completed_history = []

    def update(self, animals, bowls, frame_time, frame_h, frame_w):
        """
        animals: [{"cls": "cat", "conf": 0.9, "mask_pts": [...], "box": [x1,y1,x2,y2]}, ...]
        bowls:   [{"conf": 0.85, "mask_pts": [...], "box": [x1,y1,x2,y2]}, ...]
        """
        triggered_keys = set()
        debug_pairs = []

        for a_idx, animal in enumerate(animals):
            a_mask = polygon_to_mask(animal.get("mask_pts"), frame_h, frame_w)
            head_mask = head_region_mask(a_mask, top_ratio=0.3)
            head_area = int(np.sum(head_mask > 0))

            for b_idx, bowl in enumerate(bowls):
                b_mask = polygon_to_mask(bowl.get("mask_pts"), frame_h, frame_w)
                overlap_px = mask_overlap_area(head_mask, b_mask)
                overlap_ratio = overlap_px / head_area if head_area > 0 else 0

                trigger = (overlap_px >= MIN_OVERLAP_PIXELS or
                           overlap_ratio >= MIN_OVERLAP_RATIO)

                debug_pairs.append({
                    "animal": animal["cls"],
                    "overlap_px": overlap_px,
                    "overlap_ratio": overlap_ratio,
                    "trigger": trigger,
                    "head_mask": head_mask,
                    "bowl_mask": b_mask,
                })

                if not trigger:
                    continue

                key = (animal["cls"], b_idx)
                triggered_keys.add(key)

                if key in self.ongoing:
                    ev = self.ongoing[key]
                    ev.last_seen = frame_time
                    ev.hit_count += 1
                    ev.conf_sum += animal["conf"]
                    ev.max_overlap = max(ev.max_overlap, overlap_px)
                else:
                    ev = MaskOngoingEvent(
                        animal_cls=animal["cls"],
                        event_type="drinking",
                        start_time=frame_time,
                    )
                    ev.hit_count = 1
                    ev.conf_sum = animal["conf"]
                    ev.max_overlap = overlap_px
                    self.ongoing[key] = ev

        # 结束过期事件
        completed = []
        to_remove = []
        for key, ev in self.ongoing.items():
            gap = frame_time - ev.last_seen
            if gap >= MAX_GAP_SEC:
                dur = ev.last_seen - ev.start_time
                if dur >= MIN_DURATION_SEC:
                    completed.append({
                        "animal": ev.animal_cls,
                        "event_type": ev.event_type,
                        "duration": dur,
                        "confidence": ev.conf_sum / max(1, ev.hit_count),
                        "hit_count": ev.hit_count,
                        "max_overlap": ev.max_overlap,
                    })
                to_remove.append(key)
        for k in to_remove:
            del self.ongoing[k]

        return debug_pairs, completed

    def force_flush(self, frame_time):
        completed = []
        for ev in self.ongoing.values():
            dur = ev.last_seen - ev.start_time
            if dur >= MIN_DURATION_SEC:
                completed.append({
                    "animal": ev.animal_cls,
                    "event_type": ev.event_type,
                    "duration": dur,
                    "confidence": ev.conf_sum / max(1, ev.hit_count),
                    "hit_count": ev.hit_count,
                    "max_overlap": ev.max_overlap,
                })
        self.ongoing.clear()
        return completed


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

        info = {"box": box, "cls": name,
                "conf": float(conf), "mask_pts": mask_pts}
        if name in ("cat", "dog", "monkey", "other_primate"):
            animals.append(info)
        elif name == "bowl":
            bowls.append(info)

    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return frame, class_counts, animals, bowls


def draw_head_and_overlap(frame, debug_pairs, flash):
    """给每个 pair 画头部轮廓 + 交集高亮"""
    for pair in debug_pairs:
        head = pair["head_mask"]
        bowl = pair["bowl_mask"]
        if head is None:
            continue

        # 头部边缘描边
        contours, _ = cv2.findContours(
            head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, HEAD_ZONE_COLOR, 2)

        # 头部 ∩ 盆 的重叠区域
        if pair["trigger"]:
            inter = cv2.bitwise_and(head, bowl)
            if np.sum(inter) > 0:
                overlay = frame.copy()
                color = TRIGGER_COLOR
                if flash % 2 == 0:  # 闪烁
                    overlay[inter > 0] = color
                    frame[:] = cv2.addWeighted(
                        frame, 0.5, overlay, 0.5, 0)


def draw_panel(h, debug_pairs, ongoing, latest, model_info):
    w = 340
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V5 · Mask 像素判定",
                   (10, y), 16, (255, 255, 255))
    y += 30
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 25

    panel = cv2_zh(panel, "═ 图例 ═", (10, y), 13, (200, 200, 255))
    y += 22
    cv2.rectangle(panel, (10, y), (30, y + 12), HEAD_ZONE_COLOR, 2)
    panel = cv2_zh(panel, " 头部 mask 轮廓",
                   (35, y - 2), 11, HEAD_ZONE_COLOR)
    y += 18
    cv2.rectangle(panel, (10, y), (30, y + 12), TRIGGER_COLOR, -1)
    panel = cv2_zh(panel, " 头 ∩ 盆 像素重叠 = 触发",
                   (35, y - 2), 11, TRIGGER_COLOR)
    y += 25

    panel = cv2_zh(panel, "═ 当前对判 ═",
                   (10, y), 13, (200, 200, 255))
    y += 22
    if not debug_pairs:
        panel = cv2_zh(panel, " (无动物+盆对)",
                       (10, y), 11, (150, 150, 150))
        y += 20
    for p in debug_pairs[:5]:
        c = (100, 255, 100) if p["trigger"] else (170, 170, 170)
        panel = cv2_zh(
            panel, f" {p['animal']} <-> bowl",
            (10, y), 12, c)
        y += 17
        panel = cv2_zh(
            panel,
            f"  重叠像素={p['overlap_px']}  "
            f"占比={p['overlap_ratio']:.2%}",
            (10, y), 10, c)
        y += 15
        panel = cv2_zh(
            panel,
            f"  阈值≥{MIN_OVERLAP_PIXELS}像素 或 ≥{MIN_OVERLAP_RATIO:.0%}  "
            f"{'🟢' if p['trigger'] else '🔴'}",
            (10, y), 10, c)
        y += 20

    y += 6
    panel = cv2_zh(panel, "═ 进行中 ═",
                   (10, y), 13, (255, 255, 100))
    y += 22
    if not ongoing:
        panel = cv2_zh(panel, " (无)",
                       (10, y), 11, (150, 150, 150))
        y += 18
    for ev in ongoing[:3]:
        dur = ev.last_seen - ev.start_time
        panel = cv2_zh(
            panel,
            f" {ev.animal_cls} {ev.event_type} {dur:.1f}s",
            (10, y), 12, (255, 255, 100))
        y += 18
        panel = cv2_zh(
            panel,
            f"  hits={ev.hit_count} 峰值={ev.max_overlap}px",
            (10, y), 10, (255, 255, 100))
        y += 20

    y += 6
    panel = cv2_zh(panel, "═ 已完成 ═",
                   (10, y), 13, (100, 255, 100))
    y += 22
    for line in latest[-5:]:
        panel = cv2_zh(panel, f" • {line}",
                       (10, y), 10, (100, 255, 100))
        y += 15

    return panel


def draw_banner(w, active_texts, flash):
    h = 65
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if active_texts:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 12
        for t in active_texts:
            banner = cv2_zh(banner, t, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_zh(
            banner, "监控中 · 无进行中行为",
            (20, 12), 17, (180, 180, 180))
        banner = cv2_zh(
            banner,
            "V5 用 mask 像素相交判定,比 bbox 精确得多",
            (20, 38), 12, (150, 150, 150))
    return banner


def format_active(ongoing_dict):
    return [f"🐾 {ev.animal_cls.upper()} 正在 "
            f"{ev.event_type.upper()} - "
            f"{ev.last_seen - ev.start_time:.1f}秒 "
            f"({ev.max_overlap}像素接触)"
            for ev in ongoing_dict.values()]


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

    # 判断模型类型
    if "openvino" in mp.lower():
        model_info = "INT8 (OpenVINO)"
    elif ".onnx" in mp.lower():
        model_info = "ONNX"
    else:
        model_info = "PyTorch (best.pt)"

    vp = pick_file("选视频", [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not vp: return
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened(): return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.80)
    max_h = int(scr_h * 0.80)

    rules = MaskRuleEngine()
    win = "宠物 AI V5 - Mask 像素判定"
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

    print("\n===== 播放 =====")
    print(f"模型格式: {model_info}")
    print("V5: 用 mask 像素相交判定 drinking\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for ev in rules.force_flush(frame_idx / src_fps):
                    line = (f"{ev['animal']} {ev['event_type']} "
                            f"{int(ev['duration'])}s "
                            f"峰值{ev['max_overlap']}px "
                            f"conf={ev['confidence']:.2f}")
                    latest.append(line)
                    print(f"[事件] {line}")
                paused = True
                if frame_idx == 0: break
                continue
            frame_idx += 1

            fh, fw = frame.shape[:2]
            r = model.predict(frame, conf=0.35, verbose=False)[0]
            frame, cnt, animals, bowls = parse_and_draw(frame, r, names)

            vt = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            debug_pairs, completed = rules.update(
                animals, bowls, vt, fh, fw)
            now_set = set(rules.ongoing.keys())
            if now_set - was:
                flash = 6
                for k in now_set - was:
                    ev = rules.ongoing[k]
                    print(f"[!] 触发: {ev.animal_cls} {ev.event_type}")
            if flash > 0: flash -= 1

            # 画头部 + 交集
            draw_head_and_overlap(frame, debug_pairs, flash)

            for c in completed:
                line = (f"{c['animal']} {c['event_type']} "
                        f"{int(c['duration'])}s "
                        f"峰值{c['max_overlap']}px "
                        f"conf={c['confidence']:.2f}")
                latest.append(line)
                print(f"[事件] {line}")

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            ongo_list = list(rules.ongoing.values())
            active = format_active(rules.ongoing)

            banner = draw_banner(fw, active, flash)
            info = np.zeros((30, fw, 3), dtype=np.uint8)
            counts = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(
                info, f"帧 {frame_idx}  {fps_smooth:.1f}FPS  {counts}",
                (10, 6), 13, (200, 200, 200))
            info = cv2_zh(
                info,
                f"缩放{int(zoom*100)}%  Q 空格 S D +/- R",
                (fw - 350, 6), 11, (150, 150, 150))

            main_view = np.vstack([banner, frame, info])
            if show_debug:
                p = draw_panel(main_view.shape[0], debug_pairs,
                              ongo_list, latest, model_info)
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
