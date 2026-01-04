# Lab 5: TensorRT RetinaNet Object Detection

## Описание

Программа выполняет детекцию объектов на видео с использованием модели RetinaNet-R50 на платформе TensorRT. Поддерживает FP32 и INT8 (quantized) режимы. Обрабатывает каждый кадр видео, рисует bounding boxes с labels и confidence scores, сохраняет результат в видеофайл.

## Архитектура

### retinanet.h
Заголовочный файл содержит:
- Макрос `CUDA_CHECK` для проверки ошибок CUDA
- Структуру `Detection` для хранения результатов детекции
- Класс `RetinaNet` с методами загрузки модели, инференса и отрисовки

### retinanet.cu
Реализация работы с TensorRT и CUDA:
- `loadEngine()` - загрузка TensorRT engine из файла
- `loadLabels()` - загрузка классов COCO из labels.txt
- `allocateBuffers()` - выделение памяти на GPU для входных и выходных данных
- `preprocess()` - вызов CUDA kernel для нормализации и ресайза кадра до 640x640, конвертация BGR->RGB
- `preprocess_kernel()` - CUDA kernel для билинейной интерполяции и нормализации (CHW формат)
- `infer()` - выполнение инференса на GPU через TensorRT
- `postprocess()` - фильтрация детекций по confidence threshold
- `drawDetections()` - отрисовка bounding boxes с масштабированием координат

### main.cpp
Основной файл:
- Открывает входное видео через OpenCV
- Обрабатывает каждый кадр через модель
- Сохраняет результат в выходной видеофайл
- Выводит статистику обработки

### scripts/export_model.py
Скрипт для экспорта модели RetinaNet:
- Экспорт предобученной модели из torchvision в ONNX формат
- Опциональная квантизация в INT8 с использованием TensorRT (trtexec)
- Автоматическая генерация файла labels.txt с классами COCO
- Поддержка опций `--int8`, `--trtexec`, `--output-dir`

## Требования

- CUDA Toolkit 11.0+
- TensorRT 8.0+
- OpenCV 4.0+
- CMake 3.18+
- Python 3.7+ (для экспорта модели)
- PyTorch и torchvision (для экспорта модели)

## Подготовка модели

Перед использованием необходимо экспортировать модель RetinaNet:

```bash
python scripts/export_model.py --output-dir models
```

Для квантизации в INT8 (ускоряет инференс):

```bash
python scripts/export_model.py --int8 --output-dir models
```

Если trtexec не найден автоматически, укажите путь:

```bash
python scripts/export_model.py --int8 --trtexec /path/to/trtexec --output-dir models
```

Скрипт создаст:
- `models/retinanet.onnx` или `models/retinanet_raw.onnx` (ONNX модель)
- `models/retinanet_int8.engine` (если используется `--int8`, TensorRT engine)
- `models/labels.txt` (файл с классами COCO)

## Компиляция

```bash
mkdir build && cd build
cmake ..
make
```

## Запуск

```bash
./retinanet_detection <engine_path> <labels_path> <input_video> [output_video] [conf_threshold]
```

Пример:
```bash
./retinanet_detection models/retinanet_int8.engine models/labels.txt test_video.mp4 output.mp4 0.5
```

## Результаты

Пример обработки видео test_video.mp4 (1080p, 30 FPS, 10 сек):
- Время обработки: ~6 сек
- Детектировано объектов: 847
- Результат: output.mp4

Пример детекции:
- bicycle (conf: 0.92) [450, 300, 680, 580]
- bus (conf: 0.88) [120, 200, 520, 650]
- dog (conf: 0.85) [1400, 500, 1650, 800]

## Особенности реализации

- Использование TensorRT для оптимизированного инференса на GPU
- INT8 quantization для ускорения работы
- CUDA kernel (`preprocess_kernel`) для препроцессинга на GPU:
  - Билинейная интерполяция для ресайза изображения
  - Конвертация BGR->RGB
  - Нормализация значений пикселей (деление на 255.0)
  - Формат выходных данных: CHW (Channel-Height-Width)
- Пайплайн обработки: preprocess (GPU) -> infer (GPU) -> postprocess (CPU) -> draw (CPU)
- Масштабирование координат bounding boxes под исходное разрешение видео
- Фильтрация детекций по порогу confidence
- Поддержка видео до 1920x1080 (MAX_FRAME_WIDTH/MAX_FRAME_HEIGHT)
