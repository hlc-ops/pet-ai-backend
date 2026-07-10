"""可视化视频测试工具

功能:
- 弹 Windows 文件选择框选视频
- OpenCV 窗口实时播放,叠加检测框 + mask + 类别标签
- 底部显示 FPS / 帧号 / 每类检测数

依赖:tkinter(Windows Python 自带) + ultralytics + opencv

用法:
    python video_player_gui.py                    # 弹窗选 best.pt
    python video_player_gui.py model/best.pt      # 直接指定模型
    python video_player_gui.py model/best_openvino_model  # 用 OpenVINO
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tkinter import Tk, filedialog


# ==================== 类别配色(cat 蓝 / dog 红 / monkey 黄 / other_primate 紫 / bowl 绿)====================
CLASS_COLORS = {
    "cat": (255, 128, 0),         # BGR 蓝
    "dog": (0, 0, 255),           # BGR 红
    "monkey": (0, 255, 255),      # BGR 黄
    "other_primate": (255, 0, 255),  # BGR 紫
    "bowl": (0, 255, 0),          # BGR 绿
}


def pick_file(title: str, filetypes):
    """弹 Windows 文件对话框"""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def resolve_model_path(argv):
    if len(argv) > 1:
        return argv[1]

    # 默认查找顺序
    here = Path(__file__).parent.parent  # 项目根
    candidates = [
        here / "model" / "best.pt",
        here / "model" / "best_openvino_model",
        here / "model" / "best.onnx",
    ]
    for p in candidates:
        if p.exists():
            print(f"[i] 自动使用模型: {p}")
            return str(p)

    print("[!] 没找到默认模型,请选择")
    return pick_file(
        "选择 YOLO 模型",
        [("PyTorch/ONNX 模型", "*.pt *.onnx"), ("所有文件", "*.*")],
    )


def draw_overlay(frame, r, names, class_counts, frame_idx, fps):
    """在 frame 上画 bbox + mask + label + 底部信息"""
    if r.boxes is None or len(r.boxes) == 0:
        pass
    else:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        masks_xy = r.masks.xy if r.masks is not None else None

        # 半透明 mask 覆盖
        overlay = frame.copy()
        for i, (box, cls, conf) in enumerate(
                zip(boxes, cls_arr, conf_arr)):
            name = names.get(int(cls), str(cls))
            color = CLASS_COLORS.get(name, (200, 200, 200))
            class_counts[name] = class_counts.get(name, 0) + 1

            # 画 mask
            if masks_xy is not None and i < len(masks_xy):
                pts = np.asarray(masks_xy[i]).astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], color)

            # 画 bbox
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 画 label
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 混合 overlay(mask 半透明 30%)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 底部信息栏
    h, w = frame.shape[:2]
    info_h = 60
    info_bar = np.zeros((info_h, w, 3), dtype=np.uint8)
    cv2.putText(
        info_bar,
        f"帧 {frame_idx}  |  {fps:.1f} FPS  |  按 Q/ESC 退出, 空格暂停",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1)

    counts_str = "本帧: " + "  ".join(
        f"{k}={v}" for k, v in sorted(class_counts.items()))
    cv2.putText(
        info_bar, counts_str, (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    return np.vstack([frame, info_bar])


def main():
    # 1. 拿模型
    model_path = resolve_model_path(sys.argv)
    if not model_path:
        print("[X] 未选择模型,退出")
        return
    print(f"[+] 加载模型: {model_path}")

    from ultralytics import YOLO
    model = YOLO(model_path)
    names = model.names
    print(f"[+] 类别: {names}")

    # 2. 选视频
    video_path = pick_file(
        "选择要检测的视频",
        [("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"),
         ("所有文件", "*.*")],
    )
    if not video_path:
        print("[X] 未选视频,退出")
        return
    print(f"[+] 打开视频: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[X] 无法打开视频")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps) if src_fps > 0 else 40

    win = "宠物 AI 检测测试(Q/ESC 退出,空格暂停)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 640)

    paused = False
    frame_idx = 0
    prev_t = time.time()
    fps_smooth = src_fps

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[i] 视频结束")
                break
            frame_idx += 1

            # 推理
            r = model.predict(frame, conf=0.35, verbose=False)[0]
            class_counts = {}
            annotated = draw_overlay(
                frame, r, names, class_counts, frame_idx, fps_smooth)

            # 估算 FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = 0.8 * fps_smooth + 0.2 * inst_fps

            cv2.imshow(win, annotated)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            print("[i] " + ("暂停" if paused else "继续"))

    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] 结束,共播放 {frame_idx} 帧")


if __name__ == "__main__":
    main()
