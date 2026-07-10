"""AI 后端配置。

所有关键项都可通过环境变量覆盖,方便部署时按需调整:
- 部署到 A 环境:直接 python app.py,用默认配置
- 部署到 B 环境:改 .env 或 export 环境变量,不改代码
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()

# 自动加载 .env(如果存在)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


class Config:
    # ==================== 服务基础 ====================
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "8080"))
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

    # ==================== 目录 ====================
    DATA_DIR = BASE_DIR / "data"
    LOG_DIR = BASE_DIR / "logs"
    SNAPSHOT_DIR = DATA_DIR / "snapshots"
    UPLOAD_DIR = DATA_DIR / "uploads"

    # ==================== 模型 ====================
    # 模型路径:训练完的 best.pt。占位期先用 yolov8n.pt(COCO 预训练)
    MODEL_PATH = os.environ.get(
        "MODEL_PATH", str(BASE_DIR / "model" / "best.pt"))
    MODEL_IMGSZ = int(os.environ.get("MODEL_IMGSZ", "640"))
    MODEL_CONF = float(os.environ.get("MODEL_CONF", "0.35"))
    PRELOAD_MODEL = os.environ.get(
        "PRELOAD_MODEL", "true").lower() == "true"

    # 类别映射(**训练完的 best.pt**)
    # 如果 MODEL_PATH 指向占位 yolov8n.pt(COCO),类别 ID 不同,
    # 用 USE_COCO_MAPPING=true 切换
    USE_COCO_MAPPING = os.environ.get(
        "USE_COCO_MAPPING", "false").lower() == "true"

    # 你的项目类别定义
    CLASSES = {
        "cat": 0, "dog": 1, "monkey": 2,
        "other_primate": 3, "bowl": 4,
    }

    # COCO 类别映射(占位模型用)
    COCO_MAPPING = {
        "cat": 15, "dog": 16, "bowl": 45,
    }

    # ==================== 推理 ====================
    INFERENCE_FPS = int(os.environ.get("INFERENCE_FPS", "5"))
    MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", "500"))

    # ==================== 规则引擎 ====================
    # IoU 阈值:animal 和 bowl bbox 重叠度
    IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.10"))
    # 事件最短持续秒数
    MIN_EVENT_DURATION_SEC = float(
        os.environ.get("MIN_EVENT_DURATION_SEC", "3.0"))
    # 事件结束判定:多久没触发算结束
    MAX_EVENT_GAP_SEC = float(
        os.environ.get("MAX_EVENT_GAP_SEC", "2.0"))

    # ==================== 事件推送 ====================
    # 甲方后端接收事件的 URL。空 = 不推送,只本地记录
    CALLBACK_URL = os.environ.get("CALLBACK_URL", "")
    # 推送鉴权(可选)
    CALLBACK_AUTH_TOKEN = os.environ.get("CALLBACK_AUTH_TOKEN", "")
    # 推送重试次数
    CALLBACK_MAX_RETRIES = int(
        os.environ.get("CALLBACK_MAX_RETRIES", "3"))
    # 推送超时(秒)
    CALLBACK_TIMEOUT_SEC = int(
        os.environ.get("CALLBACK_TIMEOUT_SEC", "5"))

    # ==================== 文件上传限制 ====================
    MAX_CONTENT_LENGTH = MAX_VIDEO_SIZE_MB * 1024 * 1024
    ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "flv"}

    # ==================== 初始化目录 ====================
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.LOG_DIR,
                  cls.SNAPSHOT_DIR, cls.UPLOAD_DIR]:
            d.mkdir(parents=True, exist_ok=True)
