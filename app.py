"""AI 后端主入口:Flask 应用工厂 + 路由。

启动:
    python app.py                # 用默认配置
    HOST=0.0.0.0 PORT=8080 python app.py  # 覆盖端口

接口(按 AI_ALGORITHM_API v1.0):
    GET  /api/health                                健康检查
    GET  /api/config                                查看当前配置(不含敏感)
    POST /api/kennels/<kennel_id>/stream-video     上传视频分析
    GET  /api/tasks/<task_id>                       查询任务状态
    POST /api/kennels/<kennel_id>/rtsp             注册 RTSP 摄像头(占位)
"""
import logging
import os
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config


# ============ 内存任务表 (进程内, 演示级) ============
# task_id -> dict{status, requestId, kennelId, kennelCode, cameraId, petId, result, error}
TASKS = {}
TASKS_LOCK = threading.Lock()

# 简单信号量: 限制并发推理 + 队列上限
_INFERENCE_SEM = threading.Semaphore(Config.MAX_CONCURRENT_TASKS)
_QUEUE_COUNT = 0
_QUEUE_LOCK = threading.Lock()


def _setup_logging(app: Flask):
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


def _check_bearer_token() -> bool:
    """如果配置了 API_AUTH_TOKEN, 校验 Bearer Token
    未配置时(Demo) 直接通过
    """
    if not Config.API_AUTH_TOKEN:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {Config.API_AUTH_TOKEN}"


def _now_iso() -> str:
    """带时区的 ISO-8601 (甲方要求格式)"""
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=8))  # 北京时间
    return datetime.now(tz).isoformat(timespec="seconds")


def create_app() -> Flask:
    Config.ensure_dirs()
    _setup_logging(None)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    CORS(app)

    logger = logging.getLogger(__name__)

    # 预热模型
    if Config.PRELOAD_MODEL:
        from model_service import get_service
        try:
            get_service()
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"⚠️ 模型预热失败(启动继续): {e}")

    def process_video_async(video_path: str, task_id: str, ctx: dict):
        """后台线程: 抽帧 → YOLO → 规则 → 事件回调"""
        _log = logging.getLogger("video_task")
        global _QUEUE_COUNT

        # 状态: processing
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["status"] = "processing"

        # 发 processing 回调
        try:
            from event_reporter import report_task_status
            report_task_status("processing", task_id, ctx)
        except Exception as e:
            _log.warning(f"[{task_id}] processing 回调失败: {e}")

        # 占并发槽
        with _INFERENCE_SEM:
            try:
                from video_pipeline import process_video
                result = process_video(
                    video_path=video_path,
                    kennel_id=ctx.get("kennelId", ""),
                    camera_id=ctx.get("cameraId", ""),
                    pet_id=ctx.get("petId", ""),
                    task_id=task_id,
                    request_id=ctx.get("requestId", ""),
                    kennel_code=ctx.get("kennelCode", ""),
                )
                _log.info(f"[{task_id}] 完成: {result}")

                with TASKS_LOCK:
                    if task_id in TASKS:
                        TASKS[task_id]["status"] = "completed"
                        TASKS[task_id]["result"] = result

                # 发 completed 回调
                try:
                    from event_reporter import report_task_status
                    report_task_status("completed", task_id, ctx, result=result)
                except Exception as e:
                    _log.warning(f"[{task_id}] completed 回调失败: {e}")
            except Exception as e:
                _log.exception(f"[{task_id}] 处理失败: {e}")
                err_msg = str(e)[:1900]
                with TASKS_LOCK:
                    if task_id in TASKS:
                        TASKS[task_id]["status"] = "failed"
                        TASKS[task_id]["error"] = err_msg
                try:
                    from event_reporter import report_task_status
                    report_task_status("failed", task_id, ctx, error=err_msg)
                except Exception as ce:
                    _log.warning(f"[{task_id}] failed 回调失败: {ce}")
            finally:
                # 队列出列
                with _QUEUE_LOCK:
                    _QUEUE_COUNT -= 1
                # 上传文件清理
                if Config.DELETE_UPLOAD_AFTER_PROCESSING:
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

    # ==================== 路由 ====================

    @app.route("/api/health", methods=["GET"])
    def health():
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
        return jsonify({
            "inferenceFps": Config.INFERENCE_FPS,
            "maxVideoSizeMb": Config.MAX_VIDEO_SIZE_MB,
            "iouThreshold": Config.IOU_THRESHOLD,
            "minEventDurationSec": Config.MIN_EVENT_DURATION_SEC,
            "maxEventGapSec": Config.MAX_EVENT_GAP_SEC,
            "maxConcurrentTasks": Config.MAX_CONCURRENT_TASKS,
            "maxQueuedTasks": Config.MAX_QUEUED_TASKS,
            "allowedVideoExts": sorted(Config.ALLOWED_VIDEO_EXTS),
            "callbackConfigured": bool(Config.CALLBACK_URL),
            "taskCallbackConfigured": bool(Config.TASK_CALLBACK_URL),
        })

    @app.route("/api/kennels/<kennel_id>/stream-video", methods=["POST"])
    def upload_video(kennel_id: str):
        """接收视频文件并后台异步分析

        multipart/form-data:
            video       (必填) 视频文件
            requestId   (必填) Java 分析任务 ID, 需原样回调
            kennelCode  (必填) 页面展示犬位编号
            cameraId    (可选) 摄像头编号
            petId       (可选) 宠物编号

        响应:
            202 {status, taskId, requestId, kennelId, kennelCode, cameraId, message}
            400 {error} · 401 {error} · 413 {error} · 429 {error, Retry-After}
        """
        # Token 校验
        if not _check_bearer_token():
            return jsonify({"error": "Bearer Token 不正确"}), 401

        if "video" not in request.files:
            return jsonify({"error": "缺少 video 文件字段"}), 400

        video_file = request.files["video"]
        if not video_file.filename:
            return jsonify({"error": "video 字段无文件名"}), 400
        if not _allowed_video(video_file.filename):
            return jsonify({
                "error": f"不支持的视频格式,允许: "
                         f"{sorted(Config.ALLOWED_VIDEO_EXTS)}"
            }), 400

        # 队列上限检查
        global _QUEUE_COUNT
        with _QUEUE_LOCK:
            if _QUEUE_COUNT >= Config.MAX_QUEUED_TASKS:
                resp = jsonify({
                    "error": "AI 任务队列已满",
                    "queuedTasks": _QUEUE_COUNT,
                    "maxQueuedTasks": Config.MAX_QUEUED_TASKS,
                })
                resp.status_code = 429
                resp.headers["Retry-After"] = "10"
                return resp
            _QUEUE_COUNT += 1

        # 生成 task_id + 落盘
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        safe_name = secure_filename(video_file.filename)
        save_path = Config.UPLOAD_DIR / f"{task_id}_{safe_name}"
        video_file.save(str(save_path))

        # 完整上下文 (回调需要)
        ctx = {
            "requestId": request.form.get("requestId", ""),
            "kennelId": kennel_id,
            "kennelCode": request.form.get("kennelCode", ""),
            "cameraId": request.form.get("cameraId", ""),
            "petId": request.form.get("petId", ""),
        }

        # 注册任务
        with TASKS_LOCK:
            TASKS[task_id] = {
                "taskId": task_id,
                "requestId": ctx["requestId"],
                "status": "queued",
                "kennelId": kennel_id,
                "kennelCode": ctx["kennelCode"],
                "cameraId": ctx["cameraId"],
                "petId": ctx["petId"],
                "result": None,
                "error": None,
                "createdAt": _now_iso(),
            }

        # 启动后台任务
        thread = threading.Thread(
            target=process_video_async,
            args=(str(save_path), task_id, ctx),
            daemon=True,
        )
        thread.start()

        logger.info(
            f"接收视频: kennel={kennel_id} code={ctx['kennelCode']} "
            f"task={task_id} req={ctx['requestId']} "
            f"size={save_path.stat().st_size} bytes")

        return jsonify({
            "status": "processing",
            "taskId": task_id,
            "requestId": ctx["requestId"],
            "kennelId": kennel_id,
            "kennelCode": ctx["kennelCode"],
            "cameraId": ctx["cameraId"],
            "message": "视频接收完成,可通过 taskId 查询处理状态",
        }), 202

    @app.route("/api/tasks/<task_id>", methods=["GET"])
    def query_task(task_id: str):
        """查询任务状态 (甲方联调辅助接口)

        返回:
            200 {taskId, requestId, status, kennelId, kennelCode,
                 cameraId, petId, result, error}
            404 {error: "task not found"}
        """
        if not _check_bearer_token():
            return jsonify({"error": "Bearer Token 不正确"}), 401

        with TASKS_LOCK:
            task = TASKS.get(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        return jsonify(task)

    @app.route("/api/kennels/<kennel_id>/rtsp", methods=["POST"])
    def register_rtsp(kennel_id: str):
        """占位, 批次 2 完成"""
        if not _check_bearer_token():
            return jsonify({"error": "Bearer Token 不正确"}), 401
        data = request.get_json(silent=True) or {}
        rtsp_url = data.get("rtspUrl", "").strip()
        if not rtsp_url:
            return jsonify({"error": "缺少 rtspUrl"}), 400
        logger.info(
            f"[占位] RTSP: kennel={kennel_id} url={rtsp_url}")
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
