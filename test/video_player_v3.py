"""视频播放器 V3

修复:
- V2 窗口太大 → V3 自动检测屏幕尺寸,80% 显示,可 [+/-] 缩放,可拖边
- V2 没关节点 → V3 双模型:seg 出 bbox+mask,同时跑 pose 模型出关键点
- V3 兜底:即使 pose 模型不识别动物,也从 mask 几何估算 5 个关键点画伪骨架

按键:
    Q / ESC   退出
    空格      暂停
    S         截图
    D         切换调试面板
    + / -     缩放画面 10%
    R         重置缩放到自动大小

姿态渲染:
    绿色实圆点 = pose 模型输出的真关键点
    黄色空心圆 = 从 mask 几何估算的伪关键点(fallback)
    白色细线连接 = 骨架
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


# ==================== 配色 ====================
CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

FONT_PATH = next(
    (p for p in FONT_CANDIDATES if Path(p).exists()), None)


def pick_file(title, filetypes):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def get_screen_size():
    """获取当前屏幕分辨率"""
    root = Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def resolve_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    here = Path(__file__).parent.parent
    for p in [here / "model" / "best.pt",
              here / "model" / "best_openvino_model"]:
        if p.exists():
            return str(p)
    return pick_file("选模型", [("模型", "*.pt *.onnx")])


def cv2_puttext_zh(img, text, org, size=18, color=(255, 255, 255)):
    if FONT_PATH is None:
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, size)
    draw.text(org, text, font=font,
              fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==================== 姿态估算(mask 几何回退) ====================
def estimate_pose_from_mask(mask_pts):
    """从 mask 多边形估算 5 个关键点:头/肩/髋/尾根/中心

    返回:{ "head": (x,y), "shoulder": (x,y), ... }
    """
    if mask_pts is None or len(mask_pts) < 4:
        return None
    pts = np.asarray(mask_pts).astype(np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    # 判断朝向:宽 > 高 = 水平躺/走(狗蹲侧面就是这种)
    #          高 > 宽 = 竖直站
    w = x_max - x_min
    h = y_max - y_min

    if w >= h:  # 侧面(横向)
        head = (x_max, y_min + h * 0.35)     # 右上
        tail = (x_min, y_min + h * 0.35)     # 左上
        shoulder = (x_min + w * 0.75, y_min + h * 0.30)
        hip = (x_min + w * 0.25, y_min + h * 0.30)
        # 腿部近似:底部中间偏 30% / 70%
        l_foot = (x_min + w * 0.30, y_max)
        r_foot = (x_min + w * 0.70, y_max)
    else:  # 竖直(蹲/正面/背面)
        head = (cx, y_min)
        shoulder = (cx, y_min + h * 0.30)
        hip = (cx, y_min + h * 0.70)
        tail = (cx, y_max)
        l_foot = (x_min + w * 0.30, y_max)
        r_foot = (x_min + w * 0.70, y_max)

    return {
        "head": head,
        "shoulder": shoulder,
        "center": (cx, cy),
        "hip": hip,
        "tail": tail,
        "l_foot": l_foot,
        "r_foot": r_foot,
    }


def draw_estimated_skeleton(frame, kps):
    """画伪骨架:黄色空心圆 + 白色细线"""
    if kps is None:
        return
    # 关键点
    for name, (x, y) in kps.items():
        cv2.circle(frame, (int(x), int(y)), 5, (0, 200, 255), 2)

    # 骨架线
    connections = [
        ("head", "shoulder"),
        ("shoulder", "center"),
        ("center", "hip"),
        ("hip", "tail"),
        ("hip", "l_foot"),
        ("hip", "r_foot"),
    ]
    for a, b in connections:
        p1 = tuple(int(v) for v in kps[a])
        p2 = tuple(int(v) for v in kps[b])
        cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)


def draw_pose_keypoints(frame, kps_arr):
    """画真关键点(绿色实圆)"""
    if kps_arr is None:
        return
    for i in range(len(kps_arr)):
        pt = kps_arr[i]
        if len(pt) >= 3 and pt[2] < 0.3:
            continue
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)


# ==================== 检测绘制 ====================
def draw_detections(frame, r, names, pose_model=None):
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
            mask_pts = np.asarray(masks_xy[i]).astype(np.int32)
            if len(mask_pts) >= 3:
                cv2.fillPoly(overlay, [mask_pts], color)

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if name in ("cat", "dog", "monkey", "other_primate"):
            animals.append({"box": box, "cls": name, "conf": float(conf),
                            "mask_pts": mask_pts})
        elif name == "bowl":
            bowls.append({"box": box, "conf": float(conf)})

    frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 姿态:动物 bbox 上跑 pose(如果有 pose 模型)+ mask 估算兜底
    for animal in animals:
        pose_kps = None
        # 尝试真 pose 模型
        if pose_model is not None:
            x1, y1, x2, y2 = animal["box"].astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                try:
                    pr = pose_model.predict(crop, conf=0.25, verbose=False)[0]
                    if pr.keypoints is not None and len(pr.keypoints.data) > 0:
                        kp = pr.keypoints.data[0].cpu().numpy()  # (17, 3)
                        # 反归一化到原图坐标
                        kp_abs = kp.copy()
                        kp_abs[:, 0] += x1
                        kp_abs[:, 1] += y1
                        pose_kps = kp_abs
                except Exception:
                    pass

        # 画真 pose(有则画,绿色)
        if pose_kps is not None:
            draw_pose_keypoints(frame, pose_kps)

        # 无论如何画 mask 估算的伪骨架(黄色空心圆 + 灰线)
        est = estimate_pose_from_mask(animal.get("mask_pts"))
        if est is not None:
            draw_estimated_skeleton(frame, est)

    return frame, class_counts, animals, bowls


def compute_debug_pairs(animals, bowls):
    pairs = []
    for a in animals:
        for b in bowls:
            i = iou(a["box"], b["box"])
            head_low = head_below_bowl_top(a["box"], b["box"])
            pairs.append({
                "animal": a["cls"], "iou": i,
                "head_low": head_low,
                "trigger": i >= 0.10 and head_low,
            })
    return pairs


def draw_debug_panel(w_h, debug_pairs, ongoing_events, latest_events,
                     pose_backend):
    w, h = 300, w_h
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_puttext_zh(
        panel, "🔍 调试面板", (10, y), 18, (255, 255, 255))
    y += 32
    panel = cv2_puttext_zh(
        panel, f"姿态后端: {pose_backend}", (10, y), 12, (150, 200, 255))
    y += 25
    panel = cv2_puttext_zh(
        panel, "═══ 当前对判 ═══", (10, y), 14, (200, 200, 255))
    y += 22
    if not debug_pairs:
        panel = cv2_puttext_zh(
            panel, " (无 animal+bowl 对)", (10, y), 12, (150, 150, 150))
        y += 20
    for pair in debug_pairs[:5]:
        color = (100, 255, 100) if pair["trigger"] else (150, 150, 150)
        panel = cv2_puttext_zh(
            panel, f" {pair['animal']}<->bowl", (10, y), 13, color)
        y += 18
        panel = cv2_puttext_zh(
            panel,
            f"  IoU={pair['iou']:.2f} 头低={'Y' if pair['head_low'] else 'N'} "
            f"{'✓触发' if pair['trigger'] else '✗'}",
            (10, y), 11, color)
        y += 22

    y += 8
    panel = cv2_puttext_zh(
        panel, "═══ 进行中 ═══", (10, y), 14, (255, 255, 100))
    y += 22
    if not ongoing_events:
        panel = cv2_puttext_zh(panel, " (无)", (10, y), 12, (150, 150, 150))
        y += 20
    for ev in ongoing_events[:3]:
        dur = ev.last_seen - ev.start_time
        panel = cv2_puttext_zh(
            panel, f" {ev.detected_class} {ev.event_type}",
            (10, y), 13, (255, 255, 100))
        y += 18
        panel = cv2_puttext_zh(
            panel, f"  {dur:.1f}s hits={ev.hit_count}",
            (10, y), 11, (255, 255, 100))
        y += 22

    y += 8
    panel = cv2_puttext_zh(
        panel, "═══ 已完成 ═══", (10, y), 14, (100, 255, 100))
    y += 22
    for line in latest_events[-6:]:
        panel = cv2_puttext_zh(panel, f" • {line}", (10, y),
                               11, (100, 255, 100))
        y += 17

    # 图例
    y = h - 100
    panel = cv2_puttext_zh(panel, "═══ 图例 ═══", (10, y),
                          12, (200, 200, 200))
    y += 20
    cv2.circle(panel, (20, y + 4), 4, (0, 255, 0), -1)
    panel = cv2_puttext_zh(panel, "  真姿态关键点", (30, y), 11,
                          (200, 200, 200))
    y += 20
    cv2.circle(panel, (20, y + 4), 5, (0, 200, 255), 2)
    panel = cv2_puttext_zh(panel, "  Mask 估算关键点", (30, y), 11,
                          (200, 200, 200))
    y += 20
    cv2.line(panel, (10, y + 4), (28, y + 4), (200, 200, 200), 1)
    panel = cv2_puttext_zh(panel, "  骨架连线", (30, y), 11,
                          (200, 200, 200))
    return panel


def draw_banner(w, active_texts, flash):
    h = 70
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if active_texts:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 15
        for text in active_texts:
            banner = cv2_puttext_zh(banner, text, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_puttext_zh(
            banner, "监控中 · 无进行中行为",
            (20, 15), 18, (180, 180, 180))
        banner = cv2_puttext_zh(
            banner, "看右侧调试面板 → 一旦 IoU>0.1 且头低,横幅变亮黄",
            (20, 42), 12, (150, 150, 150))
    return banner


def format_active_texts(rules):
    return [f"🐾 {ev.detected_class.upper()} 正在 "
            f"{ev.event_type.upper()} - "
            f"{ev.last_seen - ev.start_time:.1f}秒"
            for ev in rules.ongoing.values()]


def fit_frame_to_screen(display, max_w, max_h, zoom=1.0):
    """把整个显示区(横幅+视频+调试)按比例缩小到屏幕内"""
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h) * zoom
    if scale >= 1.0 and zoom == 1.0:
        return display
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(display, (new_w, new_h),
                      interpolation=cv2.INTER_AREA)


def main():
    from ultralytics import YOLO
    model_path = resolve_model()
    if not model_path:
        return
    print(f"[+] 检测/分割模型: {model_path}")
    model = YOLO(model_path)
    names = model.names
    print(f"[+] 类别: {names}")

    # 试着加载姿态模型
    pose_model = None
    pose_backend = "无(mask估算)"
    try:
        print("[i] 尝试加载 YOLOv8n-pose(首次运行会下载 6MB)...")
        pose_model = YOLO("yolov8n-pose.pt")
        pose_backend = "yolov8n-pose(人体训练,动物可能不识别)"
        print("[+] Pose 模型已加载")
    except Exception as e:
        print(f"[!] YOLOv8-pose 加载失败,用 mask 估算兜底: {e}")

    video_path = pick_file(
        "选视频",
        [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    # 屏幕尺寸,窗口占 80%
    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.80)
    max_h = int(scr_h * 0.80)
    print(f"[+] 屏幕 {scr_w}x{scr_h},窗口最大 {max_w}x{max_h}")

    rules = BehaviorRuleEngine("test", "local", "")

    win = "宠物 AI - V3 (Q退出 空格暂停 S截图 D调试 +/-缩放 R重置)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max_w, max_h)
    cv2.moveWindow(win, 50, 50)

    paused = False
    frame_idx = 0
    prev_t = time.time()
    fps_smooth = src_fps
    latest_events = []
    flash = 0
    show_debug = True
    zoom = 1.0
    save_dir = Path(__file__).parent / "screenshots"
    save_dir.mkdir(exist_ok=True)

    print("\n===== 播放开始 =====")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for ev in rules.force_flush(frame_idx / src_fps):
                    line = (f"{ev.detected_class} {ev.event_type} "
                            f"{int(ev.duration_sec)}s "
                            f"conf={ev.confidence:.2f}")
                    latest_events.append(line)
                    print(f"[事件产出] {line}")
                paused = True
                if frame_idx == 0:
                    break
                continue
            frame_idx += 1

            r = model.predict(frame, conf=0.35, verbose=False)[0]
            frame, class_counts, animals, bowls = draw_detections(
                frame, r, names, pose_model)

            vtime = frame_idx / src_fps
            was = set(rules.ongoing.keys())
            completed = rules.update(animals, bowls, vtime, frame_bgr=frame)
            now_set = set(rules.ongoing.keys())
            if now_set - was:
                flash = 3
                for k in now_set - was:
                    ev = rules.ongoing[k]
                    print(f"[!] 触发: {ev.detected_class} {ev.event_type}")
            if flash > 0:
                flash -= 1
            for ev in completed:
                line = (f"{ev.detected_class} {ev.event_type} "
                        f"{int(ev.duration_sec)}s "
                        f"conf={ev.confidence:.2f}")
                latest_events.append(line)
                print(f"[事件产出] {line}")

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            debug_pairs = compute_debug_pairs(animals, bowls)
            ongoing = list(rules.ongoing.values())
            active = format_active_texts(rules)

            fh, fw = frame.shape[:2]
            banner = draw_banner(fw, active, flash)

            info = np.zeros((32, fw, 3), dtype=np.uint8)
            counts_str = "  ".join(
                f"{k}={v}" for k, v in sorted(class_counts.items())) \
                or "无检测"
            info = cv2_puttext_zh(
                info, f"帧 {frame_idx}  {fps_smooth:.1f} FPS  {counts_str}",
                (10, 8), 14, (200, 200, 200))
            info = cv2_puttext_zh(
                info,
                f"缩放 {int(zoom*100)}%  Q退出 空格 S截图 D面板 +/- R重置",
                (fw - 500, 8), 12, (150, 150, 150))

            main_view = np.vstack([banner, frame, info])

            if show_debug:
                panel = draw_debug_panel(
                    main_view.shape[0], debug_pairs, ongoing,
                    latest_events, pose_backend)
                display = np.hstack([main_view, panel])
            else:
                display = main_view

            display = fit_frame_to_screen(display, max_w, max_h, zoom)
            cv2.imshow(win, display)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            fp = save_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(fp), display)
            print(f"[+] 截图: {fp}")
        elif key == ord("d"):
            show_debug = not show_debug
        elif key in (ord("+"), ord("=")):
            zoom = min(2.0, zoom + 0.1)
        elif key == ord("-"):
            zoom = max(0.3, zoom - 0.1)
        elif key == ord("r"):
            zoom = 1.0

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n共播放 {frame_idx} 帧")
    print(f"识别 {len(latest_events)} 个事件:")
    for line in latest_events:
        print(f"  • {line}")


if __name__ == "__main__":
    main()
