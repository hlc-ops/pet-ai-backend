"""SuperAnimal-Quadruped 权重就位脚本

朋友首次运行:
    D:\\venvs\\dlc\\Scripts\\python install_weights.py

作用:把 superanimal_weights/ 里的 2 个 .pt 权重
复制到 DLC 3.0 期待的 checkpoints 目录,免去联网下载。
成功后就可以启动 pose_micro_service.py 了。
"""
import shutil
import sys
from pathlib import Path


def main():
    here = Path(__file__).parent
    src_dir = here / "superanimal_weights"
    if not src_dir.exists():
        print(f"❌ 找不到权重目录 {src_dir}")
        print(f"   请确认解压完整")
        return 1

    # 定位 DLC 的 checkpoints 目录
    try:
        import deeplabcut
    except ImportError:
        print("❌ 当前 Python 环境未装 deeplabcut")
        print("   请用 DLC venv 跑: D:\\venvs\\dlc\\Scripts\\python install_weights.py")
        return 1

    dlc_dir = Path(deeplabcut.__file__).parent
    dst_dir = dlc_dir / "modelzoo" / "checkpoints"
    dst_dir.mkdir(parents=True, exist_ok=True)

    weights = [
        "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt",
        "superanimal_quadruped_rtmpose_s.pt",
    ]
    copied = 0
    for w in weights:
        s = src_dir / w
        d = dst_dir / w
        if not s.exists():
            print(f"❌ 缺文件: {s}")
            continue
        if d.exists() and d.stat().st_size == s.stat().st_size:
            print(f"✓ 已存在(跳过): {w}")
            continue
        shutil.copy2(s, d)
        size_mb = d.stat().st_size / 1024 / 1024
        print(f"✓ 复制完成: {w}  ({size_mb:.1f} MB)")
        copied += 1

    print(f"\n目标目录: {dst_dir}")
    print(f"完成 {copied} 个新权重复制,总共 {len(weights)} 个就位。")
    print(f"现在可以运行: D:\\venvs\\dlc\\Scripts\\python pose_micro_service.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
