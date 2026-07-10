# SuperAnimal-Quadruped 集成指南

姿态识别层,给排泄识别用。**可选**,不装也能跑 MVP。

## 什么时候需要

- ✅ 需要:排泄识别要求高精度(> 85%)
- ✅ 需要:客户投诉"排泄漏检 / 误报"
- ❌ 不需要:MVP 阶段,drinking / eating 检测已够

## 安装(**Windows,45 分钟**)

### 步骤 1:创建独立虚拟环境

**DeepLabCut 依赖冲突多,强烈建议隔离环境**:

```powershell
# 创建 conda 环境
conda create -n dlc python=3.10 -y
conda activate dlc

# 装 DeepLabCut + modelzoo
pip install deeplabcut[modelzoo]==2.3.10
```

### 步骤 2:下载 SuperAnimal 权重(首次自动)

第一次调用会自动从 HuggingFace 下:

```python
from deeplabcut.utils.auxiliaryfunctions import get_superanimal_downloader
get_superanimal_downloader("superanimal_quadruped")
```

权重约 200 MB。若网络慢,手动下载:

- 官方页:https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped
- 下载后放 `%USERPROFILE%\.deeplabcut\modelzoo\`

### 步骤 3:验证安装

```python
python -c "
from pose_service import get_pose_service
svc = get_pose_service()
print(f'后端: {svc.backend}')
print(f'满血: {svc.is_full_quality()}')
"
```

期望输出:`后端: dlc  满血: True`

## 集成到后端

`pose_service.py` 会在**后端启动时自动检测**,按 3 层降级:

```
1. DLC SuperAnimal    ← 最好
2. YOLOv8-pose 人体   ← 兜底(动物精度差)
3. 纯 bbox 特征        ← 最后 fallback
```

**你什么都不用改代码**。装完 DLC 后重启后端会自动升级。

## 用法示例

```python
from pose_service import get_pose_service
pose_svc = get_pose_service()

# 每帧:
for animal_bbox in detected_animals:
    keypoints = pose_svc.detect(frame, animal_bbox)
    if keypoints is not None:
        # 有姿态 -> 精确判排泄
        result = excretion_detector.update(keypoints, frame_time)
    else:
        # 无姿态 -> bbox fallback
        result = excretion_detector.update_bbox_only(animal_bbox, frame_time)
```

## 排错

### Q1: `import deeplabcut` 报错

**A**: DLC 装到 dlc 虚拟环境里,但你从主环境跑。**先激活 conda 环境**。

或者装到主环境(有风险):

```powershell
D:\Python\Python\python.exe -m pip install deeplabcut[modelzoo]==2.3.10
```

### Q2: 权重下不动

**A**: HuggingFace 国内网络慢。解决方案:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

然后再跑 SuperAnimal 下载。

### Q3: DLC 装完 backend 起不来

**A**: DLC 的 TensorFlow 依赖可能与 PyTorch 冲突。分开跑:

- 主后端(app.py)在主环境
- 姿态服务单独作为 HTTP 微服务在 dlc 环境
- 主后端通过 HTTP 调姿态服务

## 精度对比(基于 3 天实测)

| 后端 | 排泄识别精度 | 推理速度(CPU) |
|---|---|---|
| DLC SuperAnimal | **90-95%** | 200ms/帧 |
| YOLOv8-pose(人体)| ~60%(误判多) | 20ms/帧 |
| bbox-only | ~75% | 5ms/帧 |

**结论**:
- 追求最高精度 → 装 DLC(慢但准)
- 平衡 → bbox-only(不装 DLC 也 75%)
- 追求速度 → 一定不用 YOLOv8-pose,还不如 bbox-only
