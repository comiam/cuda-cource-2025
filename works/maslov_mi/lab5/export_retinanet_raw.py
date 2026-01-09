import torch
import torchvision
from collections import OrderedDict
from pathlib import Path

class RetinaNetRaw(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.transform = original_model.transform
        self.backbone = original_model.backbone
        self.head = original_model.head

    def forward(self, images):
        # transform
        images, targets = self.transform(images, None)
        
        # backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        # Convert OrderedDict to list of tensors as expected by head
        features = list(features.values())
            
        # head
        head_outputs = self.head(features)
        
        # head_outputs['cls_logits'] is [N, Anchors, 91]
        # head_outputs['bbox_regression'] is [N, Anchors, 4]
        return head_outputs['cls_logits'], head_outputs['bbox_regression']

def save_class_list(labels, output_dir: Path):
    """Сохраняет список классов в файл labels.txt."""
    if not labels:
        print("[WARN] Метки классов не найдены.")
        return

    dest_file = output_dir / "labels.txt"
    try:
        dest_file.write_text("\n".join(labels), encoding="utf-8")
        print(f"[INFO] Список классов сохранен в: {dest_file}")
    except Exception as e:
        print(f"[WARN] Ошибка записи файла меток: {e}")

def main():
    print("Loading RetinaNet V2...")
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=weights)
    model.eval()
    
    # Save labels
    try:
        labels = weights.meta.get("categories", [])
        save_class_list(labels, Path("."))
    except Exception as e:
        print(f"[WARN] Failed to extract labels: {e}")

    # Wrap model
    raw_model = RetinaNetRaw(model)
    raw_model.eval()

    # Fixed input size
    dummy_input = torch.randn(1, 3, 640, 640)
    
    onnx_path = "retinanet_640_raw.onnx"

    output_names = ["cls_logits", "bbox_regression"]
    input_names = ["input"]

    print(f"Exporting raw model to {onnx_path}...")
    torch.onnx.export(
        raw_model, 
        dummy_input, 
        onnx_path,
        verbose=False, 
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "cls_logits": {0: "batch_size"},
            "bbox_regression": {0: "batch_size"}
        },
        dynamo=False
    )
    
    print("Export complete.")

if __name__ == "__main__":
    main()
