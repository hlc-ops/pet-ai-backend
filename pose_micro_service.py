"""姿态微服务 · DLC 3.0 + SuperAnimal-Quadruped

启动:
    D:\venvs\dlc\Scripts\python D:\pet_ai_delivery\pose_micro_service.py

API:
    GET  /health         → {"status": "ok", "model_ready": true}
    POST /predict
        Body: {"image_b64": "..."}
        Response: {"keypoints": [[x, y, conf], ...]}

策略:
- 用 DLC 的 superanimal_analyze_images(),把请求图存临时目录跑推理
- 首次调用会加载模型(慢,10-30 秒)
- 后续每帧 300-800ms (CPU rtmpose_s)
"""
import base64
import io
import logging
import os
import ssl
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

# ===== 环境准备 =====
os.environ.setdefault('DLC_HOME_DIR', 'D:/ai_models/dlc_home')
os.environ.setdefault('TORCH_HOME', 'D:/ai_models/torch_home')
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')

# 忽略 SSL 校验
ssl._create_default_https_context = ssl._create_unverified_context

# 强制 torch 用 CPU
import torch
_orig_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _orig_load(*args, **kwargs)
torch.load = cpu_load


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("pose_service")

app = Flask(__name__)

TMP_DIR = Path("D:/pytemp/pose_service")
TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("D:/pytemp/pose_service_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 加载 DLC API ====================
_analyze_fn = None


def _load_dlc():
    global _analyze_fn
    from deeplabcut.pose_estimation_pytorch.apis.analyze_images import (
        superanimal_analyze_images)
    _analyze_fn = superanimal_analyze_images
    logger.info("✅ DLC superanimal_analyze_images 已加载")


def _predict_keypoints(image_bgr):
    """
    输入 BGR 图,返回 keypoints (N_animals, N_kp, 3) 或 None
    N_kp = 24 for SuperAnimal-Quadruped
    """
    if _analyze_fn is None:
        return None

    # 存临时文件
    tmp_name = f"{uuid.uuid4().hex}.jpg"
    tmp_path = TMP_DIR / tmp_name
    cv2.imwrite(str(tmp_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    try:
        result = _analyze_fn(
            superanimal_name='superanimal_quadruped',
            model_name='rtmpose_s',
            detector_name='fasterrcnn_mobilenet_v3_large_fpn',
            images=[str(tmp_path)],
            out_folder=str(OUT_DIR),
            max_individuals=1,   # 只要 1 只!调 3 会强制输出 3 组幻觉 kps
            device='cpu',
        )

        # 解析返回值
        if not isinstance(result, dict):
            return None

        # result 应该有 predictions per image
        image_key = None
        for k in result.keys():
            if tmp_name in str(k) or str(tmp_path) in str(k):
                image_key = k
                break
        if image_key is None:
            # 取第一个 key
            keys = list(result.keys())
            if not keys:
                return None
            image_key = keys[0]

        pred = result[image_key]
        # pred 结构参考 DLC 3.0 输出
        # 通常是 dict 有 'bodyparts' 或 'poses' 字段
        if isinstance(pred, dict):
            for key in ['poses', 'bodyparts', 'keypoints']:
                if key in pred:
                    return np.asarray(pred[key])
            # 兜底:直接取 dict values
            for v in pred.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    return v
        elif isinstance(pred, np.ndarray):
            return pred

        return None
    except Exception as e:
        logger.error(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


# ==================== 路由 ====================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if _analyze_fn is not None else "not-ready",
        "model_ready": _analyze_fn is not None,
        "backend": "DLC 3.0 + SuperAnimal-Quadruped rtmpose_s",
        "device": "cpu",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image_b64" not in data:
        return jsonify({"error": "missing image_b64"}), 400

    try:
        img_bytes = base64.b64decode(data["image_b64"])
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "decode failed"}), 400
    except Exception as e:
        return jsonify({"error": f"decode: {e}"}), 400

    kps = _predict_keypoints(img)
    if kps is None:
        return jsonify({"keypoints": []})

    # 归一化到 (N, 3) [x, y, conf]
    if kps.ndim == 3:
        # 多候选 → 选包围盒面积最大的那一个,避免 DLC 抓到椅子腿等噪声
        best_idx = 0
        best_area = 0.0
        for i in range(kps.shape[0]):
            one = kps[i]
            good = one[one[:, 2] > 0.2] if one.shape[1] >= 3 else one
            if len(good) < 3:
                continue
            xs, ys = good[:, 0], good[:, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            if area > best_area:
                best_area = area
                best_idx = i
        kps_out = kps[best_idx]
        logger.info(f"[predict] 检出 {kps.shape[0]} 候选, "
                    f"选 idx={best_idx} area={best_area:.0f}")
    elif kps.ndim == 2:
        kps_out = kps
    else:
        return jsonify({"keypoints": []})

    # 报告 kps 分布范围, 帮助定位坐标问题
    if kps_out.shape[1] >= 3:
        good = kps_out[kps_out[:, 2] > 0.15]
        if len(good):
            logger.info(
                f"[predict] kps 范围 x=[{good[:,0].min():.0f},{good[:,0].max():.0f}] "
                f"y=[{good[:,1].min():.0f},{good[:,1].max():.0f}] "
                f"img={img.shape[1]}x{img.shape[0]}")

    return jsonify({
        "keypoints": kps_out.tolist(),
        "shape": list(kps_out.shape),
    })


if __name__ == "__main__":
    print("===== 姿态微服务 · SuperAnimal-Quadruped =====")
    print("首次加载 5-15 秒...")
    _load_dlc()
    print(f"监听: http://127.0.0.1:8090")
    print(f"健康检查: curl http://127.0.0.1:8090/health")
    app.run(host="127.0.0.1", port=8090, debug=False, threaded=False)
