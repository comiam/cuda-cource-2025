import torch
import torchvision
import time
import sys
from pathlib import Path

def export_retinanet():
  
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights, progress=True)
   

    model.eval()
    print("Model loaded successfully.")


    labels = []
    try:
        meta = getattr(weights, "meta", {}) or {}
        labels = list(meta.get("categories", []) or [])
    except Exception:
        labels = []

    if labels:
        print(f"[{time.strftime('%H:%M:%S')}] Labels found in weights meta: {len(labels)} classes")
        print("First 10 labels:", labels[:10])
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [WARN] No labels found in weights meta. labels.txt will not be created.")

    print(f"[{time.strftime('%H:%M:%S')}] Creating dummy input (1x3x640x640)...")
    dummy_input = torch.randn(1, 3, 640, 640)

    output_file = "retinanet.onnx"
    print(f"[{time.strftime('%H:%M:%S')}] Exporting to ONNX...")
    
    start_time = time.time()
    export_kwargs = dict(
        opset_version=18,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy_input, output_file, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, output_file, **export_kwargs)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"[{time.strftime('%H:%M:%S')}] Export completed in {duration:.2f} seconds.")
    print(f"Model saved to {output_file}")

    if labels:
        out_dir = Path(output_file).resolve().parent
        labels_path = out_dir / "labels.txt"
        try:
            labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
            print(f"[{time.strftime('%H:%M:%S')}] Labels saved to {labels_path}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [WARN] Failed to write labels.txt: {e}")

if __name__ == "__main__":
    export_retinanet()
