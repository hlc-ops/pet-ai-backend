# 宠物 / 灵长类行为识别 AI 后端

> 独立可运行的 AI 服务:接视频输入 → YOLO 检测 → 行为识别 → 推送事件

## 一句话概览

Flask 后端 + YOLOv8-seg 模型 + 行为规则引擎。你部署在 Windows(GPU 可用)上,
业务方(前后端团队)通过 HTTP 接口调用本服务,识别到"喝水""排泄"等行为后
POST 到你配置的回调 URL。

## 30 秒起跑(占位模型)

```powershell
# 1. 装依赖
pip install -r requirements.txt

# 2. 启动(model/best.pt 不存在会自动用 COCO 预训练占位)
python app.py

# 3. 另开一个 PowerShell 测试上传视频
curl.exe -F "video=@some.mp4" http://localhost:8080/api/kennels/A02/stream-video

# 4. 看日志
Get-Content logs\ai-backend.log -Wait
```

## 特性

- **视频文件上传**:multipart/form-data,后台异步分析
- **多种模型格式**:`.pt` / `.onnx` / `_openvino_model/`  自动识别
- **GPU 自动加速**:检测到 CUDA 就走 GPU
- **事件推送**:异步 + 3 次重试 + 失败落盘
- **可配置**:所有参数走环境变量,不改代码
- **健康检查**:`GET /api/health` 供监控用
- **日志切割**:5MB × 5 份,方便运维

## 目录结构

```
pet_ai_delivery/
├── app.py                    Flask 入口
├── config.py                 配置(env 优先)
├── model_service.py          YOLO 推理封装
├── behavior_rules.py         规则引擎(drinking / excretion)
├── event_reporter.py         事件推送客户端
├── video_pipeline.py         视频抽帧 → 推理 → 事件 流水线
├── rtsp_worker.py            RTSP 拉流(可选)
├── requirements.txt          Python 依赖
├── .env.example              环境变量示例
├── Dockerfile                容器化(可选)
├── model/
│   └── best.pt               训好的模型(多格式并存)
├── data/
│   ├── uploads/              上传视频存这里
│   └── snapshots/            事件截图
├── logs/
│   ├── ai-backend.log        应用日志
│   └── failed_events.jsonl   推送失败的事件
├── docs/
│   ├── API.md                HTTP 接口文档
│   ├── EVENTS.md             事件格式说明
│   └── DEPLOY.md             部署详细说明
└── scripts/
    └── export_model.py       best.pt -> 多格式导出
```

## 支持的行为(MVP)

| 事件类型 | 状态 | 触发条件 |
|---|---|---|
| `drinking` | ✅ 已实现 | 动物 bbox 与 bowl bbox IoU ≥ 阈值,动物头低于盆顶,持续 ≥ 3 秒 |
| `excretion` | ⏳ 待接入姿态识别 | 需 SuperAnimal-Quadruped 关键点 + 姿态角度判定 |

**扩展**:后续加姿态识别可支持吃饭 vs 饮水区分、精细的排泄识别。

## 关键配置

见 `.env.example`,重点:

| 环境变量 | 作用 |
|---|---|
| `MODEL_PATH` | 模型路径,支持 .pt / .onnx / OpenVINO 文件夹 |
| `CALLBACK_URL` | 事件推送目标 URL(空 = 不推送) |
| `CALLBACK_AUTH_TOKEN` | 推送鉴权 token(可选) |
| `INFERENCE_FPS` | 视频抽帧率,默认 5 |
| `MIN_EVENT_DURATION_SEC` | 事件最短持续时长,默认 3 秒 |

## 详细文档

- `docs/API.md` - 完整 HTTP 接口清单 + 示例
- `docs/EVENTS.md` - 事件 JSON 格式定义
- `docs/DEPLOY.md` - 部署到生产环境的完整流程

## 联系

任何接入疑问看 `docs/`,或直接联系交付方。
