"""姿态微服务 · 独立跑在 D:\venvs\dlc 环境

启动:
    D:\venvs\dlc\Scripts\python D:\pet_ai_delivery\pose_micro_service.py

API:
    GET  /health         → {"status": "ok", "model": "..."}
    POST /predict
        Body: {"image_b64": "base64 编码 JPG"}
        Response: {"keypoints": [[x, y, conf], ...24]}

主进程通过 pose_service_v2.py 里的 PoseServiceClient 调用。
"""
import base64
import io
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pose_service")

SUPERANIMAL_MODEL_DIR = r"D:\ai_models\superanimal"
DEFAULT_IMGSZ = 256


app = Flask(__name__)


# ==================== 模型加载 ====================
_predictor = None


def _load_dlc():
    """加载 DLC SuperAnimal-Quadruped

    尝试 3 种加载方式,任一成功就返回:
    1. DLC 3.0 官方 API (deeplabcut.pose_estimation_pytorch)
    2. dlclive 快速推理
    3. 直接 TF checkpoint(备用)
    """
    global _predictor

    # 尝试 1: DLC 3.0 官方(PyTorch 版)
    try:
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch.apis import PosePredictor
        pose_cfg = Path(SUPERANIMAL_MODEL_DIR) / "pose_cfg.yaml"
        if pose_cfg.exists():
            logger.info(f"[+] 加载 DLC PyTorch 预测器 from {pose_cfg}")
            _predictor = ("dlc3", pose_cfg)
            return
    except ImportError as e:
        logger.warning(f"DLC 3 未装或版本不对: {e}")
    except Exception as e:
        logger.warning(f"DLC 3 加载失败: {e}")

    # 尝试 2: dlclive
    try:
        from dlclive import DLCLive
        logger.info("[+] 尝试 dlclive")
        dlc_live = DLCLive(SUPERANIMAL_MODEL_DIR, resize=1.0)
        dlc_live.init_inference(
            np.zeros((DEFAULT_IMGSZ, DEFAULT_IMGSZ, 3), dtype=np.uint8))
        _predictor = ("dlclive", dlc_live)
        logger.info("[+] dlclive 就绪")
        return
    except ImportError:
        logger.warning("dlclive 未装")
    except Exception as e:
        logger.warning(f"dlclive 加载失败: {e}")

    logger.error("❌ 所有姿态后端加载失败,服务将返回空关键点")
    _predictor = None


def _predict_keypoints(image_bgr):
    """输入 BGR 图,返回 (24, 3) [x, y, conf] 或 None"""
    if _predictor is None:
        return None

    backend, obj = _predictor
    try:
        if backend == "dlc3":
            # DLC 3.0 PyTorch predictor
            # 这里需要根据实际 DLC 3.0 API 完善
            from deeplabcut.pose_estimation_pytorch.apis import (
                get_inference_runner)
            # TODO: 完善 DLC 3.0 推理调用
            return None
        elif backend == "dlclive":
            kps = obj.get_pose(image_bgr)
            # dlclive 返回 (N_keypoints, 3) = [x, y, conf]
            return kps
    except Exception as e:
        logger.error(f"推理失败: {e}")
        return None


# ==================== HTTP 路由 ====================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if _predictor is not None else "degraded",
        "backend": _predictor[0] if _predictor else None,
        "model_dir": SUPERANIMAL_MODEL_DIR,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
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

    return jsonify({
        "keypoints": kps.tolist(),
        "shape": list(kps.shape),
    })


if __name__ == "__main__":
    print(f"===== SuperAnimal 姿态微服务 =====")
    print(f"模型目录: {SUPERANIMAL_MODEL_DIR}")
    _load_dlc()
    print(f"后端: {_predictor[0] if _predictor else 'unavailable'}")
    print(f"监听: http://127.0.0.1:8090")
    app.run(host="127.0.0.1", port=8090, debug=False, threaded=True)
