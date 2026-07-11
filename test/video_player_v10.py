"""V10 · 真姿态骨骼 + 6 层触发(旗舰版)

在 V9 基础上加:
- L6 姿态识别(SuperAnimal-Quadruped, PyTorch 版)
- 排泄识别用真姿态点(hip_shoulder_dy, back_curvature, tail)
- 骨骼实时绘制(24 关节点 + 连线)

姿态服务方式:
- 优先 pose_service_v2 微服务(端口 8090)
- 兜底 mask 估算

启动前:
- D:\venvs\dlc\Scripts\python D:\pet_ai_delivery\pose_micro_service.py
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 读 .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from mask_utils import polygon_to_mask, mask_overlap_area
from llm_verifier import get_verifier
from pose_service_v2 import (get_pose_service, SKELETON_LINKS,
                              SUPERANIMAL_KEYPOINTS, compute_pose_features)
from excretion_pose_rules import PoseExcretionDetector


CLASS_COLORS = {
    "cat": (255, 128, 0), "dog": (0, 0, 255),
    "monkey": (0, 255, 255), "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0), "pet": (200, 200, 200),
}
TRIGGER_COLOR = (0, 100, 255)
GHOST_BOWL_COLOR = (128, 220, 128)
KP_COLOR = (0, 255, 100)          # 绿色关节点
SKELETON_LINE_COLOR = (255, 255, 100)  # 青色骨架线
EXCRETION_COLOR = (150, 0, 255)   # 紫红排泄标志

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
AGNOSTIC_NMS = True
PET_UNCERTAINTY_THRESHOLD = 0.15
PET_MAX_CONF = 0.60
POSE_INFERENCE_INTERVAL = 3       # 每 3 帧调一次姿态


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
    for p in [here / "model" / "best.pt"]:
        if p.exists(): return str(p)
    return pick_file("选 YOLO 模型", [("模型", "*.pt")])


def cv2_zh(img, text, org, size=18, color=(255, 255, 255)):
    if FONT_PATH is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(
        org, text, font=ImageFont.truetype(FONT_PATH, size),
        fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def maybe_downgrade_to_pet(cls_name, conf, other_animal_confs):
    if cls_name not in ("cat", "dog", "monkey", "other_primate"):
        return cls_name
    if conf < PET_MAX_CONF: return "pet"
    max_other = max(
        [c for k, c in other_animal_confs.items() if k != cls_name],
        default=0)
    if conf - max_other < PET_UNCERTAINTY_THRESHOLD and max_other > 0.3:
        return "pet"
    return cls_name


def draw_pose_skeleton(frame, keypoints, kp_color=KP_COLOR,
                        line_color=SKELETON_LINE_COLOR):
    """画 24 关节点 + 骨骼连线"""
    if keypoints is None or len(keypoints) == 0:
        return
    # 关节点
    for i, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] < 0.3:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, kp_color, -1)

    # 骨架线
    name_to_idx = {n: i for i, n in enumerate(SUPERANIMAL_KEYPOINTS)}
    for a, b in SKELETON_LINKS:
        ia, ib = name_to_idx.get(a), name_to_idx.get(b)
        if ia is None or ib is None:
            continue
        if ia >= len(keypoints) or ib >= len(keypoints):
            continue
        p1, p2 = keypoints[ia], keypoints[ib]
        if len(p1) >= 3 and p1[2] < 0.3: continue
        if len(p2) >= 3 and p2[2] < 0.3: continue
        cv2.line(frame,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 line_color, 2, cv2.LINE_AA)


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
                cv2.fillPoly(overlay,
                             [np.asarray(mask_pts).astype(np.int32)], color)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info = {"box": box, "cls": name, "conf": float(conf),
                "mask_pts": mask_pts}
        if name in ("cat", "dog", "monkey", "other_primate", "pet"):
            animals.append(info)
        elif name == "bowl":
            bowls.append(info)
    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return frame, class_counts, animals, bowls


def draw_banner(w, texts, flash):
    h = 65
    banner = np.zeros((h, w, 3), dtype=np.uint8)
    if texts:
        banner[:] = (0, 220, 220) if flash > 0 else (0, 150, 150)
        y = 12
        for t in texts:
            banner = cv2_zh(banner, t, (20, y), 22, (0, 0, 0))
            y += 30
    else:
        banner[:] = (40, 40, 60)
        banner = cv2_zh(banner, "监控中 · 无进行中行为",
                       (20, 12), 17, (180, 180, 180))
        banner = cv2_zh(banner,
                        "V10 · 6 层触发 · 姿态骨骼",
                        (20, 38), 12, (150, 150, 150))
    return banner


def draw_panel(h, pose_available, llm_available, ongoing_drink,
                ongoing_exc, latest, model_info, downgrade_count,
                pose_debug):
    w = 360
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V10 · 旗舰版",
                   (10, y), 16, (255, 255, 255))
    y += 26
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 16
    ptxt = "姿态: SuperAnimal ✓" if pose_available else "姿态: ✗ 微服务未启"
    panel = cv2_zh(panel, ptxt, (10, y), 11,
                   KP_COLOR if pose_available else (200, 100, 100))
    y += 16
    ltxt = "LLM: qwen ✓" if llm_available else "LLM: ✗"
    panel = cv2_zh(panel, ltxt, (10, y), 11,
                   (0, 255, 100) if llm_available else (200, 100, 100))
    y += 22

    panel = cv2_zh(panel, "═ 6 层触发 ═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    for text, c in [
        (" L1 bbox 面积 ≥ 20%", (200, 200, 200)),
        (" L2 mask 相交 ≥ 100px", (200, 200, 200)),
        (" L3 侵入 ≥ 30%", (255, 200, 0)),
        (" L4 遮挡 ≥ 3s", GHOST_BOWL_COLOR),
        (" L5 LLM 复核", (100, 255, 100)),
        (" L6 姿态排泄", EXCRETION_COLOR),
    ]:
        panel = cv2_zh(panel, text, (10, y), 10, c)
        y += 14

    y += 5
    panel = cv2_zh(panel, "═ 姿态状态 ═", (10, y), 13, KP_COLOR)
    y += 20
    if pose_debug:
        for line in pose_debug[:5]:
            panel = cv2_zh(panel, f" {line}", (10, y), 10, KP_COLOR)
            y += 14

    y += 5
    panel = cv2_zh(panel, "═ 进行事件 ═", (10, y), 13, (255, 255, 100))
    y += 20
    if ongoing_drink:
        for k, ev in list(ongoing_drink.items())[:2]:
            dur = ev.last_seen - ev.start_time
            confirmed = sum(1 for r in ev.llm_results if r.confirmed)
            panel = cv2_zh(panel,
                          f" {ev.animal_cls} DRINK {dur:.1f}s",
                          (10, y), 11, (255, 255, 100))
            y += 13
            panel = cv2_zh(panel,
                          f"  LLM {confirmed}/{ev.llm_calls}",
                          (10, y), 10, (255, 255, 100))
            y += 15
    if ongoing_exc:
        for k, ev in list(ongoing_exc.items())[:2]:
            dur = ev.last_seen - ev.start_time
            panel = cv2_zh(panel,
                          f" {ev.animal_cls} 排泄 {dur:.1f}s",
                          (10, y), 11, EXCRETION_COLOR)
            y += 13
            panel = cv2_zh(panel, f"  score={ev.max_score}",
                          (10, y), 10, EXCRETION_COLOR)
            y += 15

    y += 5
    panel = cv2_zh(panel, "═ 已完成 ═", (10, y), 13, (100, 255, 100))
    y += 20
    for e in latest[-4:]:
        panel = cv2_zh(panel, f" • {e}", (10, y), 10, (100, 255, 100))
        y += 13

    return panel


def fit(display, max_w, max_h, zoom):
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h) * zoom
    if scale >= 1.0 and zoom == 1.0:
        return display
    return cv2.resize(display, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA)


def main():
    from ultralytics import YOLO
    from cascade_rules import CascadeRuleEngine

    mp = resolve_model()
    if not mp: return
    print(f"[+] YOLO 模型: {mp}")
    model = YOLO(mp)
    names = model.names

    verifier = get_verifier()
    print(f"[+] LLM: {verifier.available}")

    pose_svc = get_pose_service()
    print(f"[+] 姿态服务(端口 8090): {pose_svc.available}")

    # 规则引擎(饮水)
    drink_rules = CascadeRuleEngine(use_llm=True)
    # 排泄检测器(姿态)
    exc_detector = PoseExcretionDetector()

    vp = pick_file("选视频", [("视频", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not vp: return
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened(): return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / src_fps)

    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.80)
    max_h = int(scr_h * 0.80)

    win = "宠物 AI V10 · 旗舰版"
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

    model_info = "PyTorch (best.pt)"

    print(f"\n===== V10 =====")
    print(f"6 层触发: bbox / mask / 侵入 / 遮挡 / LLM / 姿态排泄")
    print(f"姿态服务: {'✓' if pose_svc.available else '❌ 请先启动 pose_micro_service.py'}\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                for e in drink_rules.force_flush(
                        frame_idx / src_fps, frame):
                    line = f"drink {e.animal_cls} {int(e.duration_sec)}s LLM✓{e.llm_confirmed_count}"
                    latest.append(line)
                    print(f"[事件] {line}")
                for e in exc_detector.force_flush(frame_idx / src_fps):
                    line = f"排泄 {e['animal_cls']} {int(e['duration'])}s score={e['max_score']}"
                    latest.append(line)
                    print(f"[事件] {line}")
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
            # 饮水检测
            was = set(drink_rules.ongoing.keys())
            debug_pairs, drink_completed = drink_rules.update(
                animals, bowls, now, frame_bgr=frame)
            if set(drink_rules.ongoing.keys()) - was:
                flash = 6

            # 姿态 + 排泄检测(节流)
            pose_debug = []
            if pose_svc.available and frame_idx % POSE_INFERENCE_INTERVAL == 0:
                for i, a in enumerate(animals):
                    animal_key = f"{a['cls']}-{i}"
                    kps = pose_svc.predict(frame, tuple(a["box"]))
                    if kps is not None and len(kps) > 0:
                        # 画骨骼
                        draw_pose_skeleton(frame, kps)
                        # 排泄判定
                        exc_result = exc_detector.update(
                            animal_key, kps, now, a["cls"])
                        pose_debug.append(
                            f"{a['cls']} score={exc_result['score']}")
                        if exc_result["just_finished"]:
                            e = exc_result["just_finished"]
                            line = f"排泄 {e['animal_cls']} {int(e['duration'])}s"
                            latest.append(line)
                            print(f"[!排泄] {line}")

            for e in drink_completed:
                line = f"drink {e.animal_cls} {int(e.duration_sec)}s LLM✓{e.llm_confirmed_count}"
                latest.append(line)
                print(f"[事件] {line}")

            if flash > 0: flash -= 1

            t = time.time()
            dt = t - prev_t; prev_t = t
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            active_texts = []
            for k, ev in drink_rules.ongoing.items():
                dur = ev.last_seen - ev.start_time
                active_texts.append(
                    f"🐾 {ev.animal_cls.upper()} DRINK {dur:.1f}s")
            for k, ev in exc_detector.ongoing.items():
                dur = ev.last_seen - ev.start_time
                active_texts.append(
                    f"💩 {ev.animal_cls.upper()} 排泄 {dur:.1f}s")

            banner = draw_banner(fw, active_texts, flash)
            info = np.zeros((30, fw, 3), dtype=np.uint8)
            cs = "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) \
                if cnt else "无"
            info = cv2_zh(info,
                        f"帧 {frame_idx}  {fps_smooth:.1f}FPS  {cs}",
                        (10, 6), 13, (200, 200, 200))
            main_view = np.vstack([banner, frame, info])
            if show_debug:
                p = draw_panel(
                    main_view.shape[0], pose_svc.available,
                    verifier.available, drink_rules.ongoing,
                    exc_detector.ongoing, latest, model_info,
                    downgrade_stats["count"], pose_debug)
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
