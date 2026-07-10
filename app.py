"""AI 后端主入口:Flask 应用工厂 + 路由。

启动:
    python app.py                # 用默认配置
    HOST=0.0.0.0 PORT=8080 python app.py  # 覆盖端口

接口:
    POST /api/kennels/<kennel_id>/stream-video    上传视频分析
    POST /api/kennels/<kennel_id>/rtsp             注册 RTSP 摄像头(占位)
    GET  /api/health                                健康检查
    GET  /api/config                                查看当前配置(不含敏感)
"""
import logging
import os
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config


def _setup_logging(app: Flask):
    """日志切割:单文件 5MB × 5 份"""
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = RotatingFileHandler(
        Config.LOG_DIR / "ai-backend.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(fmt)
    handler.setLevel(logging.INFO)

    # 同时也打到控制台
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console)


def _allowed_video(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in Config.ALLOWED_VIDEO_EXTS


def create_app() -> Flask:
    Config.ensure_dirs()
    _setup_logging(None)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    CORS(app)

    logger = logging.getLogger(__name__)

    # ==================== 预热模型(可选,速度快首帧不卡) ====================
    if Config.PRELOAD_MODEL:
        from model_service import get_service
        try:
            get_service()
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"⚠️ 模型预热失败(启动继续): {e}")

    # ==================== 视频处理调度 ====================
    def process_video_async(video_path: str, kennel_id: str, task_id: str,
                             meta: dict):
        """后台线程处理视频:抽帧 → YOLO → 规则 → 推事件"""
        logger = logging.getLogger("video_task")
        try:
            from video_pipeline import process_video
            result = process_video(
                video_path=video_path,
                kennel_id=kennel_id,
                camera_id=meta.get("cameraId", ""),
                pet_id=meta.get("petId", ""),
                task_id=task_id,
            )
            logger.info(f"[{task_id}] 处理完成: {result}")
        except Exception as e:
            logger.exception(f"[{task_id}] 处理失败: {e}")

    # ==================== 路由 ====================

    @app.route("/api/health", methods=["GET"])
    def health():
        """健康检查:监控探活用"""
        from model_service import get_service
        try:
            svc = get_service()
            return jsonify({
                "status": "ok",
                "modelLoaded": svc.is_ready,
                "modelInfo": svc.info,
            })
        except Exception as e:
            return jsonify({
                "status": "degraded",
                "modelLoaded": False,
                "error": str(e),
            }), 503

    @app.route("/api/config", methods=["GET"])
    def get_config():
        """查看当前非敏感配置(方便调试)"""
        return jsonify({
            "inferenceFps": Config.INFERENCE_FPS,
            "maxVideoSizeMb": Config.MAX_VIDEO_SIZE_MB,
            "iouThreshold": Config.IOU_THRESHOLD,
            "minEventDurationSec": Config.MIN_EVENT_DURATION_SEC,
            "maxEventGapSec": Config.MAX_EVENT_GAP_SEC,
            "callbackConfigured": bool(Config.CALLBACK_URL),
        })

    @app.route("/api/kennels/<kennel_id>/stream-video", methods=["POST"])
    def upload_video(kennel_id: str):
        """接收视频文件并后台分析

        请求:
            multipart/form-data
            - video: 视频文件(必填)
            - cameraId: 摄像头编号(可选)
            - petId: 宠物编号(可选)

        响应:
            { "status": "processing", "taskId": "xxx",
              "kennelId": "A02", "cameraId": "c002" }
        """
        if "video" not in request.files:
            return jsonify({"error": "缺少 video 文件字段"}), 400

        video_file = request.files["video"]
        if not video_file.filename:
            return jsonify({"error": "video 字段无文件名"}), 400
        if not _allowed_video(video_file.filename):
            return jsonify({
                "error": f"不支持的视频格式,允许: "
                         f"{Config.ALLOWED_VIDEO_EXTS}"
            }), 400

        # 生成任务 ID + 落盘
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        safe_name = secure_filename(video_file.filename)
        save_path = Config.UPLOAD_DIR / f"{task_id}_{safe_name}"
        video_file.save(str(save_path))

        # 元数据
        meta = {
            "cameraId": request.form.get("cameraId", ""),
            "petId": request.form.get("petId", ""),
        }

        # 后台跑
        thread = threading.Thread(
            target=process_video_async,
            args=(str(save_path), kennel_id, task_id, meta),
            daemon=True,
        )
        thread.start()

        logger.info(
            f"接收视频: kennel={kennel_id} task={task_id} "
            f"size={save_path.stat().st_size} bytes")

        return jsonify({
            "status": "processing",
            "taskId": task_id,
            "kennelId": kennel_id,
            "cameraId": meta["cameraId"],
            "message": "视频接收完成,后台分析中,事件会推送到 CALLBACK_URL",
        }), 202

    @app.route("/api/kennels/<kennel_id>/rtsp", methods=["POST"])
    def register_rtsp(kennel_id: str):
        """注册 RTSP 摄像头(占位,批次 2 完善)

        请求:
            { "rtspUrl": "rtsp://xxx",
              "cameraId": "c002",
              "petId": "p002" }
        """
        data = request.get_json(silent=True) or {}
        rtsp_url = data.get("rtspUrl", "").strip()
        if not rtsp_url:
            return jsonify({"error": "缺少 rtspUrl"}), 400

        # 批次 2 会完成:启动后台 RTSP worker
        logger.info(
            f"[占位] 注册 RTSP: kennel={kennel_id} "
            f"camera={data.get('cameraId', '')} url={rtsp_url}")
        return jsonify({
            "status": "accepted",
            "message": "RTSP 注册已接受(批次 2 后完整生效)",
        }), 202

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({
            "error": f"文件过大,上限 {Config.MAX_VIDEO_SIZE_MB} MB"
        }), 413

    @app.errorhandler(500)
    def server_error(e):
        logging.getLogger("flask").exception("500 内部错误")
        return jsonify({"error": "服务器内部错误",
                        "detail": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
