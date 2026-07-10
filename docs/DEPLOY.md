# 部署说明

支持 Windows / Linux / Mac,推荐 Windows + NVIDIA GPU 环境。

---

## 前置要求

- Python **3.10+**(推荐 3.11 或 3.12)
- (可选)NVIDIA GPU + CUDA 12.x + 相应 PyTorch cu12 版本
- 磁盘空间 ≥ 5 GB(视频缓存 + 日志 + 模型)

---

## 方式一:直接 Python 启动(**推荐,最简单**)

### 1. 装依赖

```powershell
cd D:\pet_ai_delivery
pip install -r requirements.txt
```

**GPU 用户**(可选,可显著加速):
```powershell
# 若 pip 装的 torch 是 CPU 版,手动装 CUDA 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 放模型

把 `best.pt`(或其他格式)放到 `model/` 目录:

```
model/
├── best.pt                       (PyTorch,通用)
├── best.onnx                     (可选,ONNX)
└── best_openvino_model/          (可选,OpenVINO)
```

**默认加载 `model/best.pt`**。想切换格式,设 `MODEL_PATH`:

```powershell
$env:MODEL_PATH = "./model/best.onnx"; python app.py
```

### 3. 配置环境变量

复制并修改 `.env.example` → `.env`:

```
CALLBACK_URL=http://your-backend:8000/api/ai/behavior-events
CALLBACK_AUTH_TOKEN=  (如有)
INFERENCE_FPS=5
```

### 4. 启动

```powershell
python app.py
```

期望看到:

```
✅ 模型预热完成
 * Running on http://0.0.0.0:8080
```

### 5. 冒烟测试

新开一个 PowerShell:

```powershell
curl.exe http://localhost:8080/api/health
```

应返回 `"status": "ok"`。

---

## 方式二:Docker(**推荐生产环境**)

### 1. 构建镜像

```bash
cd D:\pet_ai_delivery
docker build -t pet-ai-backend:latest .
```

### 2. 运行

```bash
docker run -d \
    --name pet-ai \
    -p 8080:8080 \
    -e CALLBACK_URL=http://your-backend:8000/api/ai/behavior-events \
    -v $(pwd)/model:/app/model \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    --gpus all \
    pet-ai-backend:latest
```

**Windows PowerShell**:

```powershell
docker run -d `
    --name pet-ai `
    -p 8080:8080 `
    -e CALLBACK_URL=http://your-backend:8000/api/ai/behavior-events `
    -v ${PWD}/model:/app/model `
    -v ${PWD}/data:/app/data `
    -v ${PWD}/logs:/app/logs `
    --gpus all `
    pet-ai-backend:latest
```

**没 GPU 就去掉 `--gpus all`**。

### 3. 查看日志

```bash
docker logs -f pet-ai
```

---

## 生产运维

### 后台运行(Windows Service)

推荐用 [NSSM](https://nssm.cc/) 把 python 脚本装成服务:

```powershell
nssm install PetAiBackend "D:\Python\Python\python.exe" "D:\pet_ai_delivery\app.py"
nssm set PetAiBackend AppDirectory "D:\pet_ai_delivery"
nssm start PetAiBackend
```

### 日志监控

- 应用日志:`logs/ai-backend.log`(5 MB × 5 份自动切割)
- 事件推送失败:`logs/failed_events.jsonl`(定期人工同步)

### 磁盘管理

- `data/uploads/` 会累积上传的视频,建议定期清理:
  ```powershell
  # 删除 7 天前的上传
  Get-ChildItem data\uploads -File | 
      Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
      Remove-Item
  ```

### 端口 & 防火墙

默认监听 8080。开放防火墙:

```powershell
netsh advfirewall firewall add rule name="Pet AI 8080" `
    dir=in action=allow protocol=TCP localport=8080
```

---

## 排错

### Q1:启动时报 "模型未加载"

**A**:检查 `model/best.pt` 是否存在。如果不存在,服务会自动降级到占位模式
(用 COCO YOLOv8n),只支持 cat/dog/bowl。可通过 `GET /api/health` 查看:
`modelInfo.mode == "coco_placeholder"` 就是占位。

### Q2:GPU 装了但推理走 CPU

**A**:验证 CUDA 可用:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
如果 False,重新装 CUDA 版 torch:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Q3:上传大视频卡住

**A**:检查 `MAX_VIDEO_SIZE_MB`(默认 500 MB)。加大或分段上传。

### Q4:事件没推送到我后端

**A**:先看 `logs/ai-backend.log`:
- 有 `📡 事件产出` 说明识别到了
- 有 `✅ 事件推送成功` 说明推送成功
- 有 `❌ 事件推送最终失败` 说明推送失败,查 `failed_events.jsonl`
- 只有产出没推送:检查 `CALLBACK_URL` 环境变量是否配了

### Q5:识别效果不好

**A**:
1. 确认 `modelInfo.mode == "trained"`(不是占位)
2. 调 `MODEL_CONF`(默认 0.35,越低越敏感)
3. 调 `IOU_THRESHOLD`(默认 0.10)
4. 联系交付方,可能需要补数据重训

---

## 生产 checklist

上线前逐项打勾:

- [ ] Python 环境装好(≥ 3.10)
- [ ] 依赖装完(`pip install -r requirements.txt`)
- [ ] GPU 用户装 CUDA 版 torch
- [ ] `model/best.pt` 就位(或多格式并存)
- [ ] `.env` 配好 `CALLBACK_URL`
- [ ] 冒烟测试 `/api/health` 返回 `"modelLoaded": true, "mode": "trained"`
- [ ] 用一段测试视频调用 `/api/kennels/*/stream-video`,能看到事件日志
- [ ] 甲方后端能收到事件并 200 响应
- [ ] Windows Service / systemd / docker restart:always 保证挂了自动拉起
- [ ] 磁盘清理策略生效
- [ ] 防火墙 / 网络策略允许连接
