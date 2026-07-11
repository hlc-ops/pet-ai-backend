"""接力完成 DLC 姿态推理准备

你(用户)在 PowerShell 里跑:

    $env:HTTP_PROXY = "http://127.0.0.1:7897"
    $env:HTTPS_PROXY = "http://127.0.0.1:7897"
    D:\venvs\dlc\Scripts\python D:\pet_ai_delivery\scripts\complete_dlc_setup.py

会:
1. 触发 DLC 3.0 下所有需要的 PyTorch 权重
2. 强制 CPU 模式
3. 单张图测试推理
4. 输出关节点结果 + 可视化图到 D:/pose_test_out

预计:
- 首次:下载 ~500MB,约 5-15 分钟
- 之后:推理 1-3 秒/张(CPU)
"""
import os
import ssl
import sys
import time

# ==================== 环境准备 ====================
os.environ['DLC_HOME_DIR'] = 'D:/ai_models/dlc_home'
os.environ['TORCH_HOME'] = 'D:/ai_models/torch_home'

# 忽略 SSL 校验(公网权重下载)
ssl._create_default_https_context = ssl._create_unverified_context

# 强制 torch 用 CPU
import torch
_orig_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _orig_load(*args, **kwargs)
torch.load = cpu_load

# ==================== 推理测试 ====================
def main():
    from deeplabcut.pose_estimation_pytorch.apis.analyze_images import \
        superanimal_analyze_images

    test_img = r'D:\pet_project_data\frames\july10_clip1_00050.jpg'
    if not os.path.exists(test_img):
        print(f"❌ 测试图不存在: {test_img}")
        print("请改成任意宠物图片路径")
        return

    print(f"[+] 测试图: {test_img}")
    print(f"[+] 输出目录: D:/pose_test_out")
    print()
    print("===== 开始推理(首次会自动下 ~500MB 权重)=====")
    t0 = time.time()

    try:
        result = superanimal_analyze_images(
            superanimal_name='superanimal_quadruped',
            model_name='rtmpose_s',
            detector_name='fasterrcnn_mobilenet_v3_large_fpn',
            images=[test_img],
            out_folder='D:/pose_test_out',
            max_individuals=3,
            device='cpu',
        )
        elapsed = time.time() - t0
        print(f"\n✅ 成功,耗时 {elapsed:.1f}s")

        # 分析返回值
        print(f"\n返回类型: {type(result)}")
        if isinstance(result, tuple):
            print(f"元素数: {len(result)}")
            for i, x in enumerate(result):
                print(f"  [{i}] {type(x).__name__}")
                if hasattr(x, 'keys'):
                    print(f"      keys: {list(x.keys())[:5]}")

        print("\n===== 输出目录 =====")
        for f in os.listdir('D:/pose_test_out'):
            fp = os.path.join('D:/pose_test_out', f)
            size = os.path.getsize(fp) / 1024
            print(f"  {f} ({size:.1f}KB)")

        print()
        print("===== 下一步 =====")
        print("1. 打开 D:/pose_test_out/ 看叠加了骨骼的图")
        print("2. 骨骼看起来正确 -> 姿态服务可用")
        print("3. 通知我(claude)开始集成到 V10")

    except Exception as e:
        print(f"\n❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("常见解决:")
        print("- 网络问题 -> 换 VPN 或明天再试")
        print("- SSL 问题 -> 确认代理稳定")
        print("- 内存不足 -> 用小图片测试")


if __name__ == '__main__':
    main()
