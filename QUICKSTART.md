# 宠物行为识别系统 · 快速上手

**技术栈**:YOLO v8n-seg + SuperAnimal-Quadruped (DLC 3.0) + Qwen VL Plus LLM 复核

**功能**:
- 5 类目标检测 + 分割:cat / dog / monkey / other_primate / bowl
- 39 关键点姿态识别(通过 DLC 微服务)
- 4 层级联规则触发喝水/进食事件(bbox 20% + mask 100px + 侵入 30% + 遮挡 3s)
- LLM 双验证(事件首/中/尾各 1 次)
- 排泄行为姿态判定(后腿角度 + 髋部下沉 + 背部弯曲)

---

## 一、环境准备(两个 venv)

**为什么两个**:主环境跑最新 Ultralytics + Flask 需要 numpy 2.x;DLC 3.0 需要 numpy<2 + torch<2.5,栈冲突,拆开最干净。

### 主环境(Python 3.12)
```powershell
# 已有 3.12 就 skip
python -m venv D:\venvs\main
D:\venvs\main\Scripts\pip install -r requirements.txt
```

### 姿态微服务环境(Python 3.11 · DLC)
```powershell
# 装 Python 3.11
winget install --id Python.Python.3.11
# 建 venv
py -3.11 -m venv D:\venvs\dlc
D:\venvs\dlc\Scripts\pip install deeplabcut==3.0.0 flask requests opencv-python
```

### 骨骼模型权重就位(离线,免下载)

包里已经带了 SuperAnimal-Quadruped 的 2 个权重(共 96 MB),
在 `superanimal_weights/` 目录。**一键就位到 DLC 的 checkpoints 目录**:

```powershell
D:\venvs\dlc\Scripts\python install_weights.py
```

看到 "✓ 复制完成" 就行。如果不跑这步,DLC 首次会去 HuggingFace 下载,
国内没代理容易卡住。

---

## 二、配置

复制 `.env.example` 到 `.env`,填 Qwen API Key:
```
LLM_PROVIDER=qwen
LLM_API_KEY=sk-你自己的密钥
LLM_MODEL=qwen-vl-plus
LLM_TIMEOUT=30
```

阿里云百炼申请 Key:https://bailian.console.aliyun.com/

---

## 三、运行(要开两个终端)

**终端 A** — 启动姿态微服务(常驻,首次加载 5-15 秒):
```powershell
D:\venvs\dlc\Scripts\python pose_micro_service.py
```
看到 `Running on http://127.0.0.1:8090` 就成。

**终端 B** — 启动主播放器:
```powershell
D:\venvs\main\Scripts\python test\video_player_v11.py
```
弹窗选视频文件即开播。

---

## 四、注意事项(踩过的坑)

### Clash / V2Ray 用户
如果本机开了 Clash 系统代理 + TUN 模式,会:
- 劫持 DNS,把 `dashscope.aliyuncs.com` 解析成假 IP,LLM 复核 SSL 断连
- 甚至劫持 127.0.0.1:8090 本地端口

**建议**:跑本项目时**关掉 Clash 系统代理**,或在 Clash 配置里加规则:
```yaml
- DOMAIN-SUFFIX,dashscope.aliyuncs.com,DIRECT
- IP-CIDR,127.0.0.1/32,DIRECT,no-resolve
```

启动 V11 会自动检测并告警。

### 骨骼延迟出现
CPU 上 rtmpose_s 单帧 2-10 秒。V11 每秒发 1 次姿态请求,
所以骨骼比视频画面滞后 2-5 秒是正常的。想实时得 GPU。

### 前几秒无骨骼
DLC 的 fasterrcnn_mobilenet 检测器不稳,有 25% 面积门槛过滤
低质量误检 —— 宁可不画,不要画错。等动物走到清晰侧面姿态骨骼才出。

---

## 五、快捷键

| 键 | 作用 |
|---|---|
| Q / ESC | 退出(自动 finalize 进行中事件) |
| Space | 暂停 / 恢复 |
| S | 截图保存 |
| D | 切换右侧调试面板 |
| +/- | 缩放 |
| R | 复位缩放 |

---

## 六、文件结构

```
pet_ai_delivery/
├── test/video_player_v11.py    # 主入口
├── pose_micro_service.py       # 姿态微服务(DLC venv 跑)
├── pose_service_v2.py          # 姿态客户端
├── llm_verifier.py             # LLM 复核
├── cascade_rules.py            # V9 级联规则引擎
├── excretion_pose_rules.py     # 排泄姿态识别
├── mask_utils.py               # mask 工具
├── behavior_rules.py           # bbox 工具
├── .env.example                # LLM 配置模板
├── requirements.txt            # 主环境依赖
├── model/best.pt               # YOLOv8n-seg 5 类,mAP≈0.917
└── docs/                       # 详细文档
```

---

## 七、常见问题

**Q**: LLM 一直失败但代理已关?
**A**: 检查 `HTTP_PROXY` 环境变量。PowerShell 里跑 `Get-ChildItem env: | Where-Object Name -Match Proxy` 应该为空。清代理:`$env:HTTP_PROXY=$null; $env:HTTPS_PROXY=$null`

**Q**: 姿态微服务提示 "端口 8090 已被占用"?
**A**: 上一次没退出干净。`netstat -ano | findstr :8090` 找到 PID,`taskkill /F /PID xxx`。

**Q**: 想接摄像头 RTSP 流?
**A**: 参考 `docs/DEPLOY.md`。

---

**遇到问题联系原作者**。
