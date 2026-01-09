## Lab 5: TensorRT RetinaNet - Детекция объектов

### Описание задачи
Запустить и использовать предварительно обученную модель RetinaNet-R50 (quantized INT8) на платформе TensorRT. На вход подаётся видеоряд, на выходе — тот же видеоряд с разметкой bounding boxes и confidence scores.

### Запуск

**Требования**
- CUDA 12+
- TensorRT 10+
- OpenCV 4.x

Для сборки 

```bash
make build
```

Для инференса версии fp16

```bash
make run_fp16
```

Для инференса версии int8

```bash
make run_i8
```

### Примеры

Примеры расположены в директории videos/