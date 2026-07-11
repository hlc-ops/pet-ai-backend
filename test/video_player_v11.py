"""V11 · 异步姿态 + 39 点骨架(最终版)

V10 的问题:
- 同步调用 pose(2 秒/帧)→ 主线程卡死 → "关节点闪一下就没了"
- 24 关键点映射错(DLC 3.0 实际 39 点)
- 送 crop 导致猫检测不到

V11 修复:
- 独立线程跑 pose,主线程不阻塞
- 39 点正确骨架连线
- 送整帧,猫狗都能识别到
- 关键点 conf >= 0.25 就画(之前 0.3 阈值太严)
"""
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

# 读 .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# 禁本地代理(pose 服务在本地 8090)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

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
from cascade_rules import CascadeRuleEngine


CLASS_COLORS = {
    "cat": (255, 128, 0), "dog": (0, 0, 255),
    "monkey": (0, 255, 255), "other_primate": (255, 0, 255),
    "bowl": (0, 255, 0), "pet": (200, 200, 200),
}
TRIGGER_COLOR = (0, 100, 255)
GHOST_BOWL_COLOR = (128, 220, 128)
KP_COLOR = (0, 255, 100)
SKELETON_LINE_COLOR = (255, 255, 100)
EXCRETION_COLOR = (150, 0, 255)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


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
KP_MIN_CONF = 0.25


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


# ==================== 异步姿态工作线程 ====================
class AsyncPoseWorker:
    """
    后台异步跑姿态推理,主线程随时读最新缓存
    - request(frame) 提交推理请求(不阻塞)
    - latest 属性:最新一次成功的关键点
    """
    def __init__(self):
        self.pose_svc = get_pose_service()
        self.available = self.pose_svc.available
        self._req_queue = Queue(maxsize=2)  # 最多堆积 2 帧
        self._latest = None
        self._latest_time = 0
        self._latest_features = None
        self._lock = threading.Lock()
        self._stop = False
        self._call_count = 0
        self._last_latency_ms = 0
        if self.available:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def request(self, frame):
        """主线程调:提交姿态请求(有空槽才提交,免堆积)"""
        if not self.available: return
        if self._req_queue.full(): return
        try:
            self._req_queue.put_nowait(frame.copy())
        except Exception:
            pass

    def _loop(self):
        while not self._stop:
            try:
                frame = self._req_queue.get(timeout=0.5)
            except Empty:
                continue
            t0 = time.time()
            kps = self.pose_svc.predict(frame)
            latency = int((time.time() - t0) * 1000)
            with self._lock:
                if kps is not None:
                    self._latest = kps
                    self._latest_time = time.time()
                    self._latest_features = compute_pose_features(kps)
                self._call_count += 1
                self._last_latency_ms = latency

    @property
    def latest(self):
        with self._lock:
            return self._latest, self._latest_time, self._latest_features

    @property
    def stats(self):
        with self._lock:
            return {
                "calls": self._call_count,
                "latency_ms": self._last_latency_ms,
                "age_sec": time.time() - self._latest_time
                    if self._latest_time else -1,
            }

    def stop(self):
        self._stop = True


# ==================== 骨架绘制 ====================
def draw_pose_skeleton(frame, keypoints, min_conf=KP_MIN_CONF):
    """画 39 关节点 + 骨骼线"""
    if keypoints is None or len(keypoints) == 0:
        return
    n = len(keypoints)
    # 关键点
    for i, kp in enumerate(keypoints):
        c = kp[2] if len(kp) >= 3 else 1.0
        if c < min_conf: continue
        radius = 4 if c > 0.5 else 3
        cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, KP_COLOR, -1)

    # 骨架线
    name_to_idx = {n: i for i, n in enumerate(SUPERANIMAL_KEYPOINTS)}
    for a, b in SKELETON_LINKS:
        ia, ib = name_to_idx.get(a), name_to_idx.get(b)
        if ia is None or ib is None: continue
        if ia >= n or ib >= n: continue
        p1, p2 = keypoints[ia], keypoints[ib]
        c1 = p1[2] if len(p1) >= 3 else 1.0
        c2 = p2[2] if len(p2) >= 3 else 1.0
        if c1 < min_conf or c2 < min_conf: continue
        cv2.line(frame,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 SKELETON_LINE_COLOR, 2, cv2.LINE_AA)


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
            animal_conf_map[n] = max(animal_conf_map.get(n, 0), float(conf_arr[i]))
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


def draw_trigger_indicator(frame, animals, debug_pairs):
    """cascade_rules debug_pairs 用 l2_pass 作为最终触发标志"""
    trig = set()
    for i, a in enumerate(animals):
        for p in debug_pairs:
            if p.get("animal") == a["cls"] and (
                    p.get("l2_pass") or p.get("trigger")):
                trig.add(i)
    for i in trig:
        box = animals[i]["box"]
        x1, y1 = box.astype(int)[:2]
        cv2.circle(frame, (x1 + 20, y1 + 20), 12, TRIGGER_COLOR, -1)
        cv2.putText(frame, "!", (x1 + 15, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)


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
                        "V11 · 异步姿态(39 点)· 骨架实时",
                        (20, 38), 12, (150, 150, 150))
    return banner


def draw_panel(h, pose_worker, llm_available, ongoing_drink,
                ongoing_exc, latest, model_info, downgrade_count,
                pose_features):
    w = 380
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 15
    panel = cv2_zh(panel, "🎯 V11 · 异步姿态",
                   (10, y), 16, (255, 255, 255))
    y += 26
    panel = cv2_zh(panel, f"模型: {model_info}",
                   (10, y), 11, (150, 200, 255))
    y += 16
    stats = pose_worker.stats
    ptxt = f"姿态: 已调 {stats['calls']} 次, 最近 {stats['latency_ms']}ms"
    panel = cv2_zh(panel, ptxt, (10, y), 11,
                   KP_COLOR if pose_worker.available else (200, 100, 100))
    y += 16
    if stats['age_sec'] >= 0:
        panel = cv2_zh(panel,
                       f"缓存关键点年龄: {stats['age_sec']:.1f}s",
                       (10, y), 11, KP_COLOR)
        y += 16
    ltxt = "LLM: qwen ✓" if llm_available else "LLM: ✗"
    panel = cv2_zh(panel, ltxt, (10, y), 11,
                   (0, 255, 100) if llm_available else (200, 100, 100))
    y += 22

    # 姿态特征实时数据
    if pose_features is not None and pose_features.get("valid"):
        panel = cv2_zh(panel, "═ 姿态特征(实时) ═",
                       (10, y), 13, KP_COLOR)
        y += 20
        panel = cv2_zh(panel,
                       f" 髋-肩 Δy={pose_features['hip_shoulder_dy']:.0f}px",
                       (10, y), 11, KP_COLOR)
        y += 14
        panel = cv2_zh(panel,
                       f" 后腿角度={pose_features['rear_leg_angle']:.0f}°"
                       f" {'蹲' if pose_features['legs_bent'] else '直'}",
                       (10, y), 11, KP_COLOR)
        y += 14
        panel = cv2_zh(panel,
                       f" 背弓={pose_features['back_curvature']:.3f}"
                       f" 尾抬={pose_features['tail_raised']}",
                       (10, y), 11, KP_COLOR)
        y += 20

    panel = cv2_zh(panel, "═ 6 层触发 ═",
                   (10, y), 13, (200, 200, 255))
    y += 20
    for text, c in [
        (" L1 bbox 面积 ≥ 20%", (200, 200, 200)),
        (" L2 mask 相交 ≥ 100px", (200, 200, 200)),
        (" L3 侵入 ≥ 30%", (255, 200, 0)),
        (" L4 遮挡 ≥ 3s", GHOST_BOWL_COLOR),
        (" L5 LLM 复核", (100, 255, 100)),
        (" L6 姿态排泄(rear_leg_angle<110)", EXCRETION_COLOR),
    ]:
        panel = cv2_zh(panel, text, (10, y), 10, c)
        y += 14

    y += 5
    panel = cv2_zh(panel, "═ 进行事件 ═", (10, y), 13, (255, 255, 100))
    y += 20
    if ongoing_drink:
        for k, ev in list(ongoing_drink.items())[:2]:
            dur = ev.last_seen - ev.start_time
            llm_res = [r for r in (getattr(ev, 'llm_start_result', None),
                                    getattr(ev, 'llm_mid_result', None),
                                    getattr(ev, 'llm_end_result', None))
                        if r is not None]
            confirmed = sum(1 for r in llm_res if r.confirmed)
            panel = cv2_zh(panel,
                          f" {ev.animal_cls} DRINK {dur:.1f}s LLM {confirmed}/{len(llm_res)}",
                          (10, y), 11, (255, 255, 100))
            y += 15
    if ongoing_exc:
        for k, ev in list(ongoing_exc.items())[:2]:
            dur = ev.last_seen - ev.start_time
            panel = cv2_zh(panel,
                          f" {ev.animal_cls} 排泄 {dur:.1f}s score={ev.max_score}",
                          (10, y), 11, EXCRETION_COLOR)
            y += 15

    y += 5
    panel = cv2_zh(panel, "═ 已完成 ═", (10, y), 13, (100, 255, 100))
    y += 20
    for e in latest[-5:]:
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

    mp = resolve_model()
    if not mp: return
    print(f"[+] YOLO 模型: {mp}")
    model = YOLO(mp)
    names = model.names

    verifier = get_verifier()
    print(f"[+] LLM: {verifier.available}")

    pose_worker = AsyncPoseWorker()
    print(f"[+] 异步姿态: {pose_worker.available}")

    drink_rules = CascadeRuleEngine(use_llm=True)
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

    win = "宠物 AI V11 · 异步姿态"
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

    print(f"\n===== V11 =====")
    print(f"异步姿态: {'✓' if pose_worker.available else '❌'}")
    print(f"关键点阈值: {KP_MIN_CONF}")
    print()

    # 每 N 帧提交一次姿态请求(避免队列爆)
    POSE_REQ_EVERY_N = int(src_fps)  # 每秒 1 次

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
            was = set(drink_rules.ongoing.keys())
            debug_pairs, drink_completed = drink_rules.update(
                animals, bowls, now, frame_bgr=frame)
            if set(drink_rules.ongoing.keys()) - was:
                flash = 6

            # === 异步姿态:每秒提交 1 次 ===
            if animals and frame_idx % POSE_REQ_EVERY_N == 0:
                pose_worker.request(frame)

            # 读缓存关键点画骨架 + 排泄判定
            kps, kp_time, pose_features = pose_worker.latest
            if kps is not None:
                draw_pose_skeleton(frame, kps)
                # 用第一只动物 key 做排泄判定(简化)
                if animals:
                    a = animals[0]
                    exc_result = exc_detector.update(
                        f"{a['cls']}-0", kps, now, a["cls"])
                    if exc_result.get("just_finished"):
                        e = exc_result["just_finished"]
                        line = f"排泄 {e['animal_cls']} {int(e['duration'])}s"
                        latest.append(line)
                        print(f"[!排泄] {line}")

            for e in drink_completed:
                line = f"drink {e.animal_cls} {int(e.duration_sec)}s LLM✓{e.llm_confirmed_count}"
                latest.append(line)
                print(f"[事件] {line}")

            draw_trigger_indicator(frame, animals, debug_pairs)

            if flash > 0: flash -= 1

            t = time.time()
            dt = t - prev_t; prev_t = t
            if dt > 0:
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

            active_texts = []
            for k, ev in drink_rules.ongoing.items():
                dur = ev.last_seen - ev.start_time
                active_texts.append(
                    f"CAT DRINK {dur:.1f}s" if ev.animal_cls == "cat"
                    else f"{ev.animal_cls.upper()} DRINK {dur:.1f}s")
            for k, ev in exc_detector.ongoing.items():
                dur = ev.last_seen - ev.start_time
                active_texts.append(f"[排泄] {ev.animal_cls.upper()} {dur:.1f}s")

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
                    main_view.shape[0], pose_worker,
                    verifier.available, drink_rules.ongoing,
                    exc_detector.ongoing, latest, model_info,
                    downgrade_stats["count"], pose_features)
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

    pose_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n共 {frame_idx} 帧, {len(latest)} 事件")
    for l in latest: print(f"  • {l}")


if __name__ == "__main__":
    main()
