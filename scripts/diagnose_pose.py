"""诊断姿态识别的实际输出

对多张图 POST 到姿态微服务,记录:
1. 关键点数量(是不是 24?)
2. 关键点位置有效性(在图内?)
3. 置信度分布(有多少高置信度点)
4. 处理耗时(帧率上限)

结论用于修 V10 的骨骼绘制。
"""
import base64
import json
import os
import statistics
import time
from pathlib import Path

# 禁本地代理
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

import cv2
import requests


TEST_IMAGES = [
    # 猫吃盆(V9 触发场景)
    r"D:\pet_project_data\frames\july10_clip1_00050.jpg",
    r"D:\pet_project_data\frames\july10_clip1_00100.jpg",
    r"D:\pet_project_data\frames\july10_clip1_00200.jpg",
    # 狗排泄
    r"D:\wenti63\QQ截图20260710163403.png",   # 黄色 Lab 侧面
    r"D:\wenti63\QQ截图20260710163453.png",   # 棕色狗蓝背带
    r"D:\wenti63\QQ截图20260710163512.png",   # 黄色狗侧面
    # 狗站立(反例)
    r"D:\wenti63\QQ截图20260710163255.png",   # 米格鲁站立
    r"D:\wenti63\QQ截图20260710163334.png",   # 金毛小狗
]

POSE_URL = "http://127.0.0.1:8090"


def send_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, f"读不到 {img_path}"
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()
    t0 = time.time()
    try:
        r = requests.post(f"{POSE_URL}/predict",
                          json={"image_b64": b64}, timeout=30)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        kps = data.get("keypoints", [])
        return {
            "shape": (h, w),
            "elapsed": elapsed,
            "kp_count": len(kps),
            "kp_shape": data.get("shape", []),
            "raw_kps": kps,
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def analyze_kps(name, result):
    if result is None:
        return
    kps = result["raw_kps"]
    if not kps:
        print(f"  ❌ 无关键点返回")
        return
    n = len(kps)
    # 统计置信度
    confs = [k[2] for k in kps if len(k) >= 3]
    high_conf = sum(1 for c in confs if c > 0.5)
    med_conf = sum(1 for c in confs if 0.3 <= c <= 0.5)
    print(f"  关键点数: {n}")
    print(f"  形状元信息: {result['kp_shape']}")
    print(f"  置信度分布: 高(>0.5)={high_conf} 中(0.3-0.5)={med_conf}"
          f" 低(<0.3)={n-high_conf-med_conf}")
    if confs:
        print(f"  置信度: 最高{max(confs):.2f} 最低{min(confs):.2f} "
              f"中位数{statistics.median(confs):.2f}")
    # 前 5 个关键点具体位置
    h, w = result["shape"]
    print(f"  图片尺寸: {w}x{h}")
    print(f"  前 5 个关键点:")
    for i, k in enumerate(kps[:5]):
        x, y, c = k[0], k[1], k[2] if len(k) >= 3 else None
        in_frame = 0 <= x <= w and 0 <= y <= h
        print(f"    [{i}] x={x:.0f} y={y:.0f} conf={c:.2f} "
              f"{'✓' if in_frame else '✗超出范围'}")
    print(f"  推理耗时: {result['elapsed']*1000:.0f}ms "
          f"(单帧上限 {1/result['elapsed']:.1f} FPS)")


def main():
    # 先测服务健康
    try:
        r = requests.get(f"{POSE_URL}/health", timeout=3)
        print(f"服务健康: {r.json()}")
    except Exception as e:
        print(f"❌ 服务不通: {e}")
        return

    print()
    for img_path in TEST_IMAGES:
        name = Path(img_path).name
        print(f"===== {name} =====")
        result, err = send_image(img_path)
        if err:
            print(f"  ERROR: {err}")
            continue
        analyze_kps(name, result)
        print()


if __name__ == "__main__":
    main()
