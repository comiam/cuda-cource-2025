import torch
import torchvision
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def find_trtexec():
    trtexec = shutil.which("trtexec")
    if trtexec:
        return trtexec
    for path in ["/usr/local/TensorRT/bin/trtexec", "/opt/TensorRT/bin/trtexec"]:
        if Path(path).exists():
            return path
    return None

def quantize_to_int8(onnx_model_path, output_engine_path, trtexec_path=None):
    if not Path(onnx_model_path).exists():
        print(f"Error: ONNX model not found at {onnx_model_path}", file=sys.stderr)
        return False
    
    if trtexec_path is None:
        trtexec_path = find_trtexec()
    
    if trtexec_path is None:
        print("Error: trtexec not found. Use --trtexec to specify path.", file=sys.stderr)
        return False
    
    Path(output_engine_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [trtexec_path, f"--onnx={onnx_model_path}", f"--saveEngine={output_engine_path}", "--int8", "--fp16"]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: trtexec failed: {e.stderr.decode()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: trtexec not found at {trtexec_path}", file=sys.stderr)
        return False

def export_retinanet(quantize_int8=False, trtexec_path=None, output_dir="models"):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()

    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    labels = []
    try:
        meta = getattr(weights, "meta", {}) or {}
        labels = list(meta.get("categories", []) or [])
    except Exception:
        pass

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onnx_file = output_path / ("retinanet_raw.onnx" if quantize_int8 else "retinanet.onnx")
    
    export_kwargs = {
        "opset_version": 18,
        "input_names": ["input"],
        "output_names": ["boxes", "scores", "labels"],
        "do_constant_folding": True,
    }
    
    try:
        torch.onnx.export(model, torch.randn(1, 3, 640, 640), str(onnx_file), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, torch.randn(1, 3, 640, 640), str(onnx_file), **export_kwargs)

    if labels:
        (output_path / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    if quantize_int8:
        engine_path = output_path / "retinanet_int8.engine"
        if not quantize_to_int8(str(onnx_file), str(engine_path), trtexec_path):
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8")
    parser.add_argument("--trtexec", type=str, help="Path to trtexec")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()
    export_retinanet(quantize_int8=args.int8, trtexec_path=args.trtexec, output_dir=args.output_dir)