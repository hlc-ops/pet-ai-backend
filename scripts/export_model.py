"""训完后一键导出多格式模型

用法:
    python scripts/export_model.py --pt model/best.pt

会生成:
    model/best.pt                   (原样)
    model/best.onnx                 (ONNX,跨平台)
    model/best_openvino_model/     (OpenVINO INT8,CPU 优化)

TensorRT 引擎因为要绑定目标 GPU,让部署方在他自己机器上导出:
    yolo export model=best.pt format=engine device=0
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True,
                        help="best.pt 路径")
    parser.add_argument("--data",
                        default=None,
                        help="data.yaml 路径(INT8 校准需要)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-openvino", action="store_true",
                        help="跳过 OpenVINO 导出(比较慢)")
    args = parser.parse_args()

    pt_path = Path(args.pt).resolve()
    if not pt_path.exists():
        print(f"❌ 找不到: {pt_path}")
        return

    model = YOLO(str(pt_path))

    # 1. ONNX
    print("\n===== 导出 ONNX =====")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"✅ 生成: {onnx_path}")

    # 2. OpenVINO(可选 INT8)
    if not args.skip_openvino:
        print("\n===== 导出 OpenVINO =====")
        if args.data:
            ov_path = model.export(
                format="openvino",
                int8=True,
                data=args.data,
                imgsz=args.imgsz,
            )
        else:
            print("⚠️ 未给 --data,导出 FP32 版(建议给 data.yaml 用 INT8)")
            ov_path = model.export(format="openvino", imgsz=args.imgsz)
        print(f"✅ 生成: {ov_path}")

    print("\n" + "=" * 50)
    print("导出完成。检查 model/ 目录下有:")
    print(f"  {pt_path.name}                 (PyTorch, 通用)")
    print(f"  {pt_path.stem}.onnx            (ONNX, 跨平台)")
    if not args.skip_openvino:
        print(f"  {pt_path.stem}_openvino_model/ (OpenVINO)")
    print("\n目标机器有 NVIDIA GPU 想极致优化,让部署方跑:")
    print(f"  yolo export model={pt_path} format=engine device=0")


if __name__ == "__main__":
    main()
