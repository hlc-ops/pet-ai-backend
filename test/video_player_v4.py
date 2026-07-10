"""视频播放器 V4 - 诚实版

设计原则:
- 不显示"假关节点"(V3 那些点是 bbox 角点,不是解剖学关键点)
- 显示"规则判定逻辑":头部区域、盆顶线、接触区域,让用户看得懂
- 如果 SuperAnimal 装了才画真关节点(等 DLC 集成)

关键可视化:
- 青色虚线框 = 动物 bbox 上 30%(算法认为的"头部区域")
- 橙色横线 = 盆顶
- 红色高亮 = 头部区域和盆有 IoU 且头低于盆顶 → 规则触发
- 用户能一眼看到"为什么触发 / 为什么不触发"

按键(同 V3):Q 空格 S D +/- R
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
from behavior_rules import BehaviorRuleEngine, iou, head_below_bowl_top


CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
}
HEAD_ZONE_COLOR = (255, 255, 0)      # 青色
BOWL_TOP_COLOR = (0, 165, 255)       # 橙色
TRIGGER_COLOR = (0, 100, 255)        # 红色

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


def pick_file(title, filetypes):
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return p


def get_screen_size():
    root = Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def resolve_model():
    if len(sys.argv) > 1: return sys.argv[1]
    here = Path(__file__).parent.parent
    for p in [here / "model" / "best.pt"]:
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


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash=8):
    """画虚线矩形"""
    x1, y1 = pt1; x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_dashed_line(img, p1, p2, color, thickness=2, dash=10):
    x1, y1 = p1; x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0: return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    for i in range(0, length, dash * 2):
        s = (int(x1 + dx * i), int(y1 + dy * i))
        e = (int(x1 + dx * min(i + dash, length)),
             int(y1 + dy * min(i + dash, length)))
        cv2.line(img, s, e, color, thickness)


def draw_detections_and_rules(frame, r, names):
    """核心:画检测 + 头部区域 + 盆顶线 + 触发高亮"""
    class_counts = {}
    animals, bowls = [], []

    if r.boxes is None or len(r.boxes) == 0:
        return frame, class_counts, animals, bowls

    boxes = r.boxes.xyxy.cpu().numpy()
    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
    conf_arr = r.boxes.conf.cpu().numpy()
    masks_xy = r.masks.xy if r.masks is not None else None

    # ---------- 第 1 层:mask 半透明 + bbox + 类别 ----------
    overlay = frame.copy()
    for i, (box, cls, conf) in enumerate(zip(boxes, cls_arr, conf_arr)):
        name = names.get(int(cls), str(cls))
        color = CLASS_COLORS.get(name, (200, 200, 200))
        class_counts[name] = class_counts.get(name, 0) + 1

        if masks_xy is not None and i < len(masks_xy):
            pts = np.asarray(masks_xy[i]).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)

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

        if name in ("cat", "dog", "monkey", "other_primate"):
            animals.append({"box": box, "cls": name, "conf": float(conf)})
        elif name == "bowl":
            bowls.append({"box": box, "conf": float(conf)})

    frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # ---------- 第 2 层:规则可视化(头部区域 + 盆顶) ----------

    # 盆的顶部线(橙色虚线,横穿盆宽度延长)
    for b in bowls:
        x1, y1, x2, y2 = b["box"].astype(int)
        draw_dashed_line(
            frame, (x1 - 30, y1), (x2 + 30, y1),
            BOWL_TOP_COLOR, 2)
        frame = cv2_zh(
            frame, "盆顶", (x2 + 32, y1 - 12), 14, BOWL_TOP_COLOR)

    # 每个动物画"头部区域"(bbox 上 30%)
    for a in animals:
        x1, y1, x2, y2 = a["box"].astype(int)
        head_y = int(y1 + (y2 - y1) * 0.30)  # 头区结束的 y

        # 检查是否与任何盆触发规则
        triggered = False
        for b in bowls:
            if iou(a["box"], b["box"]) >= 0.10 and head_below_bowl_top(
                    a["box"], b["box"]):
                triggered = True
                break

        if triggered:
            # 红色实线框 + 半透明填充
            fill = frame.copy()
            cv2.rectangle(fill, (x1, y1), (x2, head_y),
                         TRIGGER_COLOR, -1)
            frame[:] = cv2.addWeighted(frame, 0.6, fill, 0.4, 0)
            cv2.rectangle(frame, (x1, y1), (x2, head_y),
                         TRIGGER_COLOR, 3)
            frame = cv2_zh(
                frame, "🍽 触发!",
                (x1, head_y + 3), 18, TRIGGER_COLOR)
        else:
            # 青色虚线框(未触发,仅提示)
            draw_dashed_rect(frame, (x1, y1), (x2, head_y),
                            HEAD_ZONE_COLOR, 2)
            frame = cv2_zh(
                frame, "头区",
                (x1 + 4, head_y + 3), 12, HEAD_ZONE_COLOR)

    return frame, class_counts, animals, bowls


def compute_debug_pairs(animals, bowls):
    pairs = []
    for a in animals:
        for b in bowls:
            i = iou(a["box"], b["box"])
            hl = head_below_bowl_top(a["box"], b["box"])
            pairs.append({
                "animal": a["cls"], "iou": i,
                "head_low": hl, "trigger": i >= 0.10 and hl})
    return pairs


def draw_panel(h, debug_pairs, ongoing, latest):
    w = 320
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🔍 判定详情",
                   (10, y), 18, (255, 255, 255))
    y += 32

    # 图例(顶部醒目)
    panel = cv2_zh(panel, "═ 图例 ═", (10, y), 14, (200, 200, 255))
    y += 22
    cv2.rectangle(panel, (10, y), (30, y + 12),
                 HEAD_ZONE_COLOR, 2)
    panel = cv2_zh(panel, " 头部区域(未触发)",
                   (35, y - 2), 12, HEAD_ZONE_COLOR)
    y += 20
    cv2.rectangle(panel, (10, y), (30, y + 12),
                 TRIGGER_COLOR, -1)
    panel = cv2_zh(panel, " 头区触发(饮/食)",
                   (35, y - 2), 12, TRIGGER_COLOR)
    y += 20
    cv2.line(panel, (10, y + 6), (30, y + 6),
             BOWL_TOP_COLOR, 2)
    panel = cv2_zh(panel, " 盆顶横线",
                   (35, y - 2), 12, BOWL_TOP_COLOR)
    y += 28

    # 当前判定
    panel = cv2_zh(panel, "═ 当前对判 ═",
                   (10, y), 14, (200, 200, 255))
    y += 22
    if not debug_pairs:
        panel = cv2_zh(panel, " (无动物+盆对)",
                       (10, y), 12, (150, 150, 150))
        y += 20
    for p in debug_pairs[:5]:
        c = (100, 255, 100) if p["trigger"] else (170, 170, 170)
        panel = cv2_zh(panel, f" {p['animal']}<->bowl",
                       (10, y), 13, c)
        y += 18
        panel = cv2_zh(
            panel,
            f"  IoU={p['iou']:.2f}  头低={'✓' if p['head_low'] else '✗'}  "
            f"{'🟢触发' if p['trigger'] else '🔴未触发'}",
            (10, y), 11, c)
        y += 22

    y += 8
    panel = cv2_zh(panel, "═ 进行中 ═",
                   (10, y), 14, (255, 255, 100))
    y += 22
    if not ongoing:
        panel = cv2_zh(panel, " (无)",
                       (10, y), 12, (150, 150, 150))
        y += 20
    for ev in ongoing[:3]:
        dur = ev.last_seen - ev.start_time
        panel = cv2_zh(
            panel,
            f" {ev.detected_class} {ev.event_type} {dur:.1f}s",
            (10, y), 13, (255, 255, 100))
        y += 22

    y += 8
    panel = cv2_zh(panel, "═ 已完成事件 ═",
                   (10, y), 14, (100, 255, 100))
    y += 22
    for line in latest[-5:]:
        panel = cv2_zh(panel, f" • {line}",
                       (10, y), 11, (100, 255, 100))
        y += 17

    # 姿态说明(底部)
    y = h - 80
    panel = cv2_zh(panel, "═ 说明 ═",
                   (10, y), 12, (200, 200, 200))
    y += 20
    panel = cv2_zh(
        panel, "本版不画伪关节点",
        (10, y), 11, (200, 200, 200))
    y += 18
    panel = cv2_zh(
        panel, "真姿态需 SuperAnimal",
        (10, y), 11, (200, 200, 200))
    y += 18
    panel = cv2_zh(
        panel, "(见 docs/SUPERANIMAL_SETUP.md)",
        (10, y), 10, (150, 150, 150))

    return panel


def draw_banner(w, texts, flash):
    h = 70
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if texts:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 15
        for t in texts:
            banner = cv2_zh(banner, t, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_zh(
            banner, "监控中 · 无进行中行为",
            (20, 15), 18, (180, 180, 180))
        banner = cv2_zh(
            banner,
            "画面里青虚线框=头区,橙线=盆顶,红色高亮=触发",
            (20, 42), 12, (150, 150, 150))
    return banner


def format_active(rules):
    return [f"🐾 {ev.detected_class.upper()} 正在 "
            f"{ev.event_type.upper()} - "
            f"{ev.last_seen - ev.start_time:.1f}秒"
            for ev in rules.ongoing.values()]


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

    vp = pick_file(
        "选视频",
        [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not vp: return
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened(): return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.80)
    max_h = int(scr_h * 0.80)
    print(f"[+] 窗口最大: {max_w}x{max_h}")

    rules = BehaviorRuleEngine("test", "local", "")

    win = ("宠物 AI V4 (Q 退出 空格 暂停 S 截图 D 面板 +/- 缩放 R 重置)")
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

    print("\n===== 播放开始 =====\n"
          "看画面里的青色虚线框(头区)和橙色横线(盆顶)")
    print("一旦头区侵入盆顶,会变红色高亮 + 顶部横幅变亮黄\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for ev in rules.force_flush(frame_idx / src_fps):
                    line = (f"{ev.detected_class} {ev.event_type} "
                            f"{int(ev.duration_sec)}s "
                            f"conf={ev.confidence:.2f}")
                    latest.append(line)
                    print(f"[事件产出] {line}")
                paused = True
                if frame_idx == 0: break
                continue
            frame_idx += 1

            r = model.predict(frame, conf=0.35, verbose=False)[0]
            frame, cnt, animals, bowls = draw_detections_and_rules(
                frame, r, names)

            vt = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            completed = rules.update(animals, bowls, vt, frame_bgr=frame)
            new_now = set(rules.ongoing.keys())
            if new_now - was:
                flash = 3
                for k in new_now - was:
                    ev = rules.ongoing[k]
                    print(f"[!] 触发: {ev.detected_class} {ev.event_type}")
            if flash > 0: flash -= 1
            for ev in completed:
                line = (f"{ev.detected_class} {ev.event_type} "
                        f"{int(ev.duration_sec)}s "
                        f"conf={ev.confidence:.2f}")
                latest.append(line)
                print(f"[事件产出] {line}")

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            dbg = compute_debug_pairs(animals, bowls)
            ongo = list(rules.ongoing.values())
            act = format_active(rules)

            fh, fw = frame.shape[:2]
            banner = draw_banner(fw, act, flash)

            info = np.zeros((32, fw, 3), dtype=np.uint8)
            counts = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(
                info, f"帧 {frame_idx}  {fps_smooth:.1f}FPS  本帧: {counts}",
                (10, 8), 14, (200, 200, 200))
            info = cv2_zh(
                info,
                f"缩放{int(zoom*100)}%  Q 退出 空格 S D +/- R",
                (fw - 400, 8), 12, (150, 150, 150))

            main_view = np.vstack([banner, frame, info])
            if show_debug:
                pan = draw_panel(main_view.shape[0], dbg, ongo, latest)
                display = np.hstack([main_view, pan])
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
            print(f"[+] 截图: {fp}")
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
