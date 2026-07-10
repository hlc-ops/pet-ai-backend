"""增强版视频播放器:检测 + 行为识别(实时叠加)

功能:
- 底层:YOLO seg 模型(现有 best.pt)
- 中层:BehaviorRuleEngine(drinking / eating 规则)
- 顶层:实时叠加行为标签,直接看到 "EATING" / "DRINKING" 等

依赖同 video_player_gui.py,复用 behavior_rules.py 的规则引擎

用法:
    python video_player_with_behavior.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tkinter import Tk, filedialog

# 把项目根加入 sys.path,能 import behavior_rules
sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior_rules import BehaviorRuleEngine


CLASS_COLORS = {
    "cat": (255, 128, 0),
    "dog": (0, 0, 255),
    "monkey": (0, 255, 255),
    "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0),
}


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
            print(f"[i] 模型: {p}")
            return str(p)
    return pick_file("选模型", [("模型", "*.pt *.onnx")])


def draw_frame(frame, r, names, behavior_labels, frame_idx, fps,
               ongoing_texts):
    """画 detection + behavior 标签"""
    class_counts = {}

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        masks_xy = r.masks.xy if r.masks is not None else None

        # 半透明 mask
        overlay = frame.copy()
        for i, (box, cls, conf) in enumerate(
                zip(boxes, cls_arr, conf_arr)):
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
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    h, w = frame.shape[:2]

    # 顶部:行为标签横幅(醒目)
    banner_h = 60
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 40)  # 深蓝底

    if ongoing_texts:
        # 有进行中的行为 -> 亮显
        y = 40
        for text in ongoing_texts:
            cv2.putText(
                banner, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
            y += 30
    else:
        cv2.putText(
            banner, "监控中 - 无进行中行为", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (150, 150, 150), 2)

    # 已完成事件的历史(小字滚动)
    if behavior_labels:
        y = 45
        for line in behavior_labels[-2:]:
            cv2.putText(
                banner, line, (w // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (100, 255, 100), 1)
            y += 15

    # 底部:统计
    stats_h = 40
    stats = np.zeros((stats_h, w, 3), dtype=np.uint8)
    counts_str = "本帧: " + "  ".join(
        f"{k}={v}" for k, v in sorted(class_counts.items())) \
        if class_counts else "本帧: 无检测"
    cv2.putText(
        stats, f"帧 {frame_idx} | {fps:.1f} FPS | {counts_str}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (200, 200, 200), 1)
    cv2.putText(
        stats, "Q/ESC 退出  空格暂停", (w - 250, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (180, 180, 180), 1)

    return np.vstack([banner, frame, stats])


def get_ongoing_texts(rules: BehaviorRuleEngine, now: float):
    """从 rule engine 提取当前进行中的行为,用于顶部横幅显示"""
    texts = []
    for key, ev in rules.ongoing.items():
        dur = ev.last_seen - ev.start_time
        emoji = {"cat": "🐱", "dog": "🐶", "monkey": "🐒",
                 "other_primate": "🦍"}.get(ev.detected_class, "🐾")
        # emoji 用不上(cv2 不支持),换成英文首字母
        cls_short = ev.detected_class[:3].upper()
        texts.append(
            f"[{cls_short}] {ev.event_type.upper()} - {dur:.1f}s "
            f"(hits={ev.hit_count})")
    return texts


def main():
    from ultralytics import YOLO
    model_path = resolve_model()
    if not model_path:
        return
    print(f"[+] 加载模型: {model_path}")
    model = YOLO(model_path)
    names = model.names
    print(f"[+] 类别: {names}")

    video_path = pick_file(
        "选视频",
        [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not video_path:
        return
    print(f"[+] 视频: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[X] 打不开视频")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    # 规则引擎
    rules = BehaviorRuleEngine(
        kennel_id="test", camera_id="local", pet_id="")

    win = "宠物 AI 检测 + 行为识别(Q 退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1024, 780)

    paused = False
    frame_idx = 0
    prev_t = time.time()
    fps_smooth = src_fps
    behavior_history = []  # 已完成的事件字符串

    print("\n===== 播放开始 =====")
    print("看顶部横幅:进行中行为(黄字)+ 历史事件(绿字)")
    print("看底部:检测统计\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # 视频结束,flush 未完成事件
                for ev in rules.force_flush(frame_idx / src_fps):
                    behavior_history.append(
                        f"{ev.detected_class} {ev.event_type} "
                        f"{int(ev.duration_sec)}s (conf={ev.confidence:.2f})")
                    print(f"[事件] {behavior_history[-1]}")
                print("[i] 视频结束")
                # 暂停最后一帧
                paused = True
                if frame_idx == 0:
                    break
                continue

            frame_idx += 1

            # 推理
            r = model.predict(frame, conf=0.35, verbose=False)[0]

            # 收集 animals 和 bowls
            animals, bowls = [], []
            if r.boxes is not None and len(r.boxes) > 0:
                for box, cls, conf in zip(
                        r.boxes.xyxy.cpu().numpy(),
                        r.boxes.cls.cpu().numpy().astype(int),
                        r.boxes.conf.cpu().numpy()):
                    name = names.get(int(cls), "")
                    if name in ("cat", "dog", "monkey", "other_primate"):
                        animals.append({
                            "box": box, "cls": name,
                            "conf": float(conf)})
                    elif name == "bowl":
                        bowls.append({
                            "box": box, "conf": float(conf)})

            # 用视频内部时间戳跑规则
            video_time = frame_idx / src_fps
            completed = rules.update(
                animals, bowls, video_time, frame_bgr=frame)

            # 打印新完成的事件
            for ev in completed:
                line = (f"{ev.detected_class} {ev.event_type} "
                        f"{int(ev.duration_sec)}s "
                        f"(conf={ev.confidence:.2f})")
                behavior_history.append(line)
                print(f"[事件] {line}")

            # FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = 0.8 * fps_smooth + 0.2 * inst_fps

            # 画
            ongoing_texts = get_ongoing_texts(rules, video_time)
            annotated = draw_frame(
                frame, r, names, behavior_history,
                frame_idx, fps_smooth, ongoing_texts)
            cv2.imshow(win, annotated)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n===== 结束 =====")
    print(f"共播放 {frame_idx} 帧")
    print(f"识别到 {len(behavior_history)} 个行为事件:")
    for line in behavior_history:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
