# HTTP 接口文档

**Base URL**:`http://<部署主机>:8080`(默认端口 8080,可通过 `PORT` 环境变量改)

**认证**:目前无认证。如需增加,联系交付方。

---

## 1. `POST /api/kennels/<kennel_id>/stream-video`

上传视频文件,后台分析并**异步**推送事件。

### 请求

- **URL 参数**
  - `kennel_id` string:狗舍/摄像头单元编号,如 `A02`

- **Body**(multipart/form-data)
  - `video` file(**必填**):视频文件,支持 mp4/avi/mov/mkv/flv,上限 500 MB
  - `cameraId` string(可选):摄像头编号
  - `petId` string(可选):宠物编号

### 响应

**202 Accepted**(视频接收成功,进入后台处理):

```json
{
    "status": "processing",
    "taskId": "task-abc123",
    "kennelId": "A02",
    "cameraId": "c002",
    "message": "视频接收完成,后台分析中,事件会推送到 CALLBACK_URL"
}
```

**400 Bad Request**:
```json
{ "error": "缺少 video 文件字段" }
```

**413 Payload Too Large**:
```json
{ "error": "文件过大,上限 500 MB" }
```

### 示例(curl)

```bash
curl -F "video=@/path/to/video.mp4" \
     -F "cameraId=c002" \
     -F "petId=p002" \
     http://localhost:8080/api/kennels/A02/stream-video
```

Windows PowerShell:

```powershell
curl.exe -F "video=@C:\videos\cat.mp4" `
    http://localhost:8080/api/kennels/A02/stream-video
```

### 处理流程

1. 视频存到 `data/uploads/{taskId}_{filename}`
2. 后台线程按 `INFERENCE_FPS`(默认 5)抽帧
3. 每帧过 YOLO 检测 + 规则引擎
4. 满足触发条件的事件推送到 `CALLBACK_URL`
5. 事件详见 `docs/EVENTS.md`

---

## 2. `POST /api/kennels/<kennel_id>/rtsp`

注册 RTSP 摄像头,后台持续拉流分析。

⚠️ **本接口在批次 3 后完善**,目前只接受不启动。

### 请求

- **URL 参数**:同上 `kennel_id`
- **Body**(JSON):
```json
{
    "rtspUrl": "rtsp://user:pass@192.168.1.102:554/stream1",
    "cameraId": "c002",
    "petId": "p002"
}
```

### 响应

**202 Accepted**:
```json
{
    "status": "accepted",
    "message": "RTSP 注册已接受"
}
```

---

## 3. `GET /api/health`

健康检查,监控探活。

### 响应

**200 OK**:
```json
{
    "status": "ok",
    "modelLoaded": true,
    "modelInfo": {
        "mode": "trained",
        "model_path": "./model/best.pt",
        "classes": ["cat", "dog", "monkey", "other_primate", "bowl"]
    }
}
```

**503 Service Unavailable**(模型未加载):
```json
{
    "status": "degraded",
    "modelLoaded": false,
    "error": "..."
}
```

**`modelInfo.mode`** 取值:
- `trained`:使用训练好的 `best.pt`,支持完整 5 类
- `coco_placeholder`:占位模式,只支持 cat/dog/bowl

---

## 4. `GET /api/config`

查看当前运行时配置(不含敏感字段)。

### 响应

**200 OK**:
```json
{
    "inferenceFps": 5,
    "maxVideoSizeMb": 500,
    "iouThreshold": 0.10,
    "minEventDurationSec": 3.0,
    "maxEventGapSec": 2.0,
    "callbackConfigured": true
}
```

---

## 错误码约定

| HTTP | 含义 |
|---|---|
| 200 | 成功(GET) |
| 202 | 接受(POST,后台异步处理) |
| 400 | 请求参数错误 |
| 413 | 上传文件过大 |
| 500 | 服务器内部错误 |
| 503 | 模型未加载或不可用 |

所有错误响应格式:
```json
{ "error": "错误说明", "detail": "..." }
```
