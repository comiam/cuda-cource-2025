import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision

def load_pretrained_model():
    """Загружает предобученную модель RetinaNet с весами по умолчанию."""
    print("[INFO] Загрузка модели RetinaNet (ResNet50 FPN)...")
    try:
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights, progress=True)
        model.eval()
        return model, weights
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель: {e}")
        sys.exit(1)

def extract_classes(weights):
    """Извлекает названия классов из метаданных весов."""
    try:
        meta = getattr(weights, "meta", {})
        return meta.get("categories", [])
    except Exception:
        return []

def run_export(model, output_path: Path):
    """Экспортирует модель в формат ONNX."""
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print(f"[INFO] Экспорт ONNX модели в: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_settings = {
        "opset_version": 11,
        "input_names": ["input"],
        "output_names": ["boxes", "scores", "labels"],
        "do_constant_folding": True,
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"},
            "labels": {0: "batch_size"},
        }
    }

    start_ts = time.time()
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_settings)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(output_path), **export_settings)
    
    elapsed = time.time() - start_ts
    print(f"[SUCCESS] Экспорт завершен за {elapsed:.2f} сек.")

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
    parser = argparse.ArgumentParser(description="Скачивание и экспорт RetinaNet в ONNX")
    parser.add_argument("--out", type=str, default="onnx/retinanet.onnx", 
                        help="Путь для сохранения ONNX файла")
    args = parser.parse_args()
    
    output_path = Path(args.out).resolve()
    
    model, weights = load_pretrained_model()
    
    labels = extract_classes(weights)
    if labels:
        print(f"[INFO] Обнаружено {len(labels)} классов (примеры: {labels[:3]}...)")
    
    run_export(model, output_path)
    
    save_class_list(labels, output_path.parent)

if __name__ == "__main__":
    main()
