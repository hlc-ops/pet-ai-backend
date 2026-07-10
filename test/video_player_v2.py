"""视频播放器 V2

修复 V1 问题:
- 中文乱码 -> 改用 PIL 画,支持中文字体
- 横幅太安静 -> 触发瞬间"跳灯"闪 3 帧
- 调试面板 -> 右侧显示实时判定值,能看到"为什么没触发"

用法:
    python video_player_v2.py                    # 弹窗选模型和视频
    python video_player_v2.py model/best.pt      # 直接指定模型

按键:
    Q / ESC   退出
    空格      暂停/继续
    S         保存当前帧到 screenshots/
    D         切换调试面板(默认打开)
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


# ==================== 类别配色(BGR)====================
CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
}

# 中文字体路径(Windows)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\simsun.ttc",     # 宋体
]


def find_font():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


FONT_PATH = find_font()


def pick_file(title, filetypes):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def resolve_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    here = Path(__file__).parent.parent
    for p in [here / "model" / "best.pt",
              here / "model" / "best_openvino_model"]:
        if p.exists():
            return str(p)
    return pick_file("选模型", [("模型", "*.pt *.onnx")])


# ==================== 用 PIL 画中文 ====================
def cv2_puttext_zh(img, text, org, font_size=20, color=(255, 255, 255)):
    """在 cv2 BGR 图上画中文(BGR 颜色元组)"""
    if FONT_PATH is None:
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    # PIL 用 RGB
    rgb_color = (color[2], color[1], color[0])
    draw.text(org, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==================== 画检测 ====================
def draw_detections(frame, r, names):
    """返回:frame(已画 bbox + 半透明 mask),class_counts,animals,bowls"""
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

        if masks_xy is not None and i < len(masks_xy):
            pts = np.asarray(masks_xy[i]).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 用英文画标签(避免中文字体拖慢)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 分类
        if name in ("cat", "dog", "monkey", "other_primate"):
            animals.append({"box": box, "cls": name, "conf": float(conf)})
        elif name == "bowl":
            bowls.append({"box": box, "conf": float(conf)})

    frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    return frame, class_counts, animals, bowls


def compute_debug_pairs(animals, bowls):
    """算 animal-bowl 对的 IoU + 头低于盆顶,给调试面板"""
    pairs = []
    for a in animals:
        for b in bowls:
            i = iou(a["box"], b["box"])
            head_low = head_below_bowl_top(a["box"], b["box"])
            pairs.append({
                "animal": a["cls"],
                "iou": i,
                "head_low": head_low,
                "trigger": i >= 0.10 and head_low,
            })
    return pairs


def draw_debug_panel(w_h, debug_pairs, ongoing_events, latest_events):
    """右侧一个 300 宽的调试面板"""
    w, h = 300, w_h
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = 20
    panel = cv2_puttext_zh(
        panel, "🔍 调试面板", (10, y), 20, (255, 255, 255))
    y += 40

    # 当前帧的 animal-bowl 判定
    panel = cv2_puttext_zh(
        panel, "当前对判:", (10, y), 16, (200, 200, 255))
    y += 25
    if not debug_pairs:
        panel = cv2_puttext_zh(
            panel, " (无 animal+bowl 对)", (10, y), 14, (150, 150, 150))
        y += 25
    for pair in debug_pairs[:5]:
        color = (100, 255, 100) if pair["trigger"] else (150, 150, 150)
        line1 = f" {pair['animal']}<->bowl"
        line2 = (f"  IoU={pair['iou']:.2f} "
                 f"头低={'Y' if pair['head_low'] else 'N'} "
                 f"{'✓' if pair['trigger'] else '✗'}")
        panel = cv2_puttext_zh(panel, line1, (10, y), 14, color)
        y += 20
        panel = cv2_puttext_zh(panel, line2, (10, y), 12, color)
        y += 25

    y += 10
    panel = cv2_puttext_zh(
        panel, "进行中事件:", (10, y), 16, (255, 255, 100))
    y += 25
    if not ongoing_events:
        panel = cv2_puttext_zh(
            panel, " (无)", (10, y), 14, (150, 150, 150))
        y += 20
    for ev in ongoing_events[:3]:
        dur = ev.last_seen - ev.start_time
        panel = cv2_puttext_zh(
            panel,
            f" {ev.detected_class} {ev.event_type}",
            (10, y), 14, (255, 255, 100))
        y += 20
        panel = cv2_puttext_zh(
            panel, f"  时长 {dur:.1f}s hits={ev.hit_count}",
            (10, y), 12, (255, 255, 100))
        y += 25

    y += 10
    panel = cv2_puttext_zh(
        panel, "🟢 已完成事件:", (10, y), 16, (100, 255, 100))
    y += 25
    for line in latest_events[-8:]:
        panel = cv2_puttext_zh(
            panel, f" {line}", (10, y), 12, (100, 255, 100))
        y += 18

    return panel


def draw_banner(w, active_texts, flash_countdown):
    """顶部横幅:进行中行为醒目显示"""
    h = 80
    banner = np.zeros((h, w, 3), dtype=np.uint8)

    if active_texts:
        # 触发中 -> 亮黄底,闪烁 3 帧
        if flash_countdown > 0:
            banner[:] = (0, 220, 220)  # 亮黄
        else:
            banner[:] = (0, 150, 150)  # 暗黄
        y = 30
        for text in active_texts:
            banner = cv2_puttext_zh(
                banner, text, (20, y), 26, (0, 0, 0))
            y += 40
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_puttext_zh(
            banner, "监控中 · 无进行中行为", (20, 30), 22, (180, 180, 180))
        banner = cv2_puttext_zh(
            banner, "看右侧调试面板 → 一旦 IoU>0.1 且头低,横幅会变亮黄",
            (20, 55), 14, (150, 150, 150))

    return banner


def get_ongoing_events(rules: BehaviorRuleEngine):
    return list(rules.ongoing.values())


def format_active_texts(ongoing_events, video_time):
    texts = []
    for ev in ongoing_events:
        dur = ev.last_seen - ev.start_time
        texts.append(
            f"🐾 {ev.detected_class.upper()} 正在 "
            f"{ev.event_type.upper()} - {dur:.1f}秒")
    return texts


def main():
    from ultralytics import YOLO
    model_path = resolve_model()
    if not model_path:
        return
    print(f"[+] 模型: {model_path}")
    model = YOLO(model_path)
    names = model.names
    print(f"[+] 类别: {names}")
    print(f"[+] 中文字体: {FONT_PATH or '未找到,回退英文'}")

    video_path = pick_file(
        "选视频",
        [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not video_path:
        return
    print(f"[+] 视频: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    rules = BehaviorRuleEngine(
        kennel_id="test", camera_id="local", pet_id="")

    win = "宠物 AI - 检测 + 行为识别(V2 修复版)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 850)

    paused = False
    frame_idx = 0
    prev_t = time.time()
    fps_smooth = src_fps
    latest_events = []
    flash_countdown = 0
    show_debug = True
    save_dir = Path(__file__).parent / "screenshots"
    save_dir.mkdir(exist_ok=True)

    print("\n===== 播放开始 =====")
    print("看顶部横幅 + 右侧调试面板")
    print("控制台会同步打印事件\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for ev in rules.force_flush(frame_idx / src_fps):
                    line = (f"{ev.detected_class} {ev.event_type} "
                            f"{int(ev.duration_sec)}s "
                            f"(conf={ev.confidence:.2f})")
                    latest_events.append(line)
                    print(f"[事件产出] {line}")
                print("[i] 视频结束")
                paused = True
                if frame_idx == 0:
                    break
                continue
            frame_idx += 1

            r = model.predict(frame, conf=0.35, verbose=False)[0]
            frame, class_counts, animals, bowls = draw_detections(
                frame, r, names)

            video_time = frame_idx / src_fps
            was_ongoing = set(rules.ongoing.keys())
            completed = rules.update(
                animals, bowls, video_time, frame_bgr=frame)
            now_ongoing = set(rules.ongoing.keys())

            # 新触发的
            newly = now_ongoing - was_ongoing
            if newly:
                flash_countdown = 3
                for key in newly:
                    ev = rules.ongoing[key]
                    print(f"[!] 触发: {ev.detected_class} "
                          f"{ev.event_type} 开始")
            if flash_countdown > 0:
                flash_countdown -= 1

            for ev in completed:
                line = (f"{ev.detected_class} {ev.event_type} "
                        f"{int(ev.duration_sec)}s "
                        f"(conf={ev.confidence:.2f})")
                latest_events.append(line)
                print(f"[事件产出] {line}")

            # FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            # 调试信息
            debug_pairs = compute_debug_pairs(animals, bowls)
            ongoing = get_ongoing_events(rules)
            active_texts = format_active_texts(ongoing, video_time)

            # 拼图
            fh, fw = frame.shape[:2]
            banner = draw_banner(fw, active_texts, flash_countdown)

            # 底部信息
            info_h = 40
            info = np.zeros((info_h, fw, 3), dtype=np.uint8)
            counts_str = "  ".join(
                f"{k}={v}" for k, v in sorted(class_counts.items())) \
                or "无检测"
            info = cv2_puttext_zh(
                info,
                f"帧 {frame_idx}  {fps_smooth:.1f} FPS  "
                f"本帧: {counts_str}",
                (10, 20), 16, (200, 200, 200))
            info = cv2_puttext_zh(
                info,
                "Q退出 · 空格暂停 · S截图 · D调试面板",
                (fw - 400, 20), 14, (150, 150, 150))

            main_view = np.vstack([banner, frame, info])

            if show_debug:
                panel = draw_debug_panel(
                    main_view.shape[0], debug_pairs, ongoing,
                    latest_events)
                display = np.hstack([main_view, panel])
            else:
                display = main_view

            cv2.imshow(win, display)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            fp = save_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(fp), display)
            print(f"[+] 截图存到 {fp}")
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n===== 结束 =====")
    print(f"共播放 {frame_idx} 帧")
    print(f"识别到 {len(latest_events)} 个事件:")
    for line in latest_events:
        print(f"  • {line}")


if __name__ == "__main__":
    main()
