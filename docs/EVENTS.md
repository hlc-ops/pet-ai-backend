# 事件格式说明

本服务识别到"喝水""排泄"等行为后,通过 HTTP POST 推送 JSON 事件到
`CALLBACK_URL` 配置的地址。

## 推送目标

| 项 | 值 |
|---|---|
| **URL** | 由 `CALLBACK_URL` 环境变量配置 |
| **方法** | POST |
| **Content-Type** | application/json |
| **鉴权** | 可选,由 `CALLBACK_AUTH_TOKEN` 配置为 `Authorization: Bearer <token>` |
| **重试** | 失败 3 次(1s / 2s / 4s 指数退避) |
| **超时** | 5 秒 |
| **失败落盘** | 3 次都失败的事件写入 `logs/failed_events.jsonl` |

## 事件 JSON 格式

```json
{
    "eventId": "evt-abc123def456",
    "eventType": "drinking",
    "kennelId": "A02",
    "cameraId": "c002",
    "petId": "p002",
    "detectedClass": "dog",
    "eventTime": "2026-07-10 11:42:30",
    "durationSeconds": 8,
    "confidence": 0.91,
    "hitCount": 42,
    "imageUrl": "",
    "videoUrl": ""
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| **eventId** | string | UUID 唯一 ID,**用于去重** |
| **eventType** | string | 行为类型,见下表 |
| **kennelId** | string | 狗舍编号,与请求 URL 里的一致 |
| **cameraId** | string | 摄像头编号,上传视频时传的 |
| **petId** | string | 宠物编号(可能为空,可由业务方按 kennelId 补) |
| **detectedClass** | string | 识别到的动物物种:`cat` / `dog` / `monkey` / `other_primate` |
| **eventTime** | string | 事件**开始**时间,`YYYY-MM-DD HH:MM:SS`(服务器本地时间) |
| **durationSeconds** | int | 事件持续秒数(四舍五入) |
| **confidence** | float | 置信度,0-1,是命中帧的平均检测置信度 |
| **hitCount** | int | 触发帧数(用于评估事件强度) |
| **imageUrl** | string | 事件截图 URL(未实现时为空字符串) |
| **videoUrl** | string | 事件视频片段 URL(暂未实现,总是空) |

## 事件类型

| eventType | 中文 | 状态 |
|---|---|---|
| `drinking` | 饮水 | ✅ MVP 已实现 |
| `excretion` | 排泄 | ⏳ 待姿态识别接入 |

⚠️ **注意**:目前 MVP 不区分"吃饭"和"饮水",都上报为 `drinking`。
姿态识别接入后会补 `eating` 类型。业务方接入时可预留字段。

## 去重建议

强烈建议业务方用 `eventId` 做去重键。虽然本服务不会主动发重复事件,
但网络重试等情况下可能造成同一事件到达 2 次。

## 收到事件后建议响应

- **HTTP 2xx**:视为成功,本服务不会重试
- **HTTP 4xx**:视为失败但**不重试**(视为业务方拒绝)
- **HTTP 5xx / 网络错误**:重试 3 次

回复内容不要求特定格式,状态码是唯一判据。

## 失败事件的处理

3 次重试都失败后,事件写入 `logs/failed_events.jsonl`,格式:

```json
{"failedAt": "2026-07-10T11:42:35.123456", "payload": { ... }}
```

业务方可后期人工同步这些事件,或联系交付方补发。

## 示例 curl 模拟接收

```bash
# 你的后端接收接口应该长得像:
POST http://your-backend:8000/api/ai/behavior-events
Content-Type: application/json
Authorization: Bearer optional-token

{ ...上面的 JSON... }

# 返回 200 OK 即可
```
