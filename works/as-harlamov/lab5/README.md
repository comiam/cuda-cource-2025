# Lab 5: TensorRT RetinaNet Object Detection

## Описание

Программа выполняет детекцию объектов на видео с использованием модели RetinaNet-R50 (quantized INT8) на платформе TensorRT. Обрабатывает каждый кадр видео, рисует bounding boxes с labels и confidence scores, сохраняет результат в видеофайл.

## Архитектура

### retinanet.h
Заголовочный файл содержит:
- Макрос `CUDA_CHECK` для проверки ошибок CUDA
- Структуру `Detection` для хранения результатов детекции
- Класс `RetinaNet` с методами загрузки модели, инференса и отрисовки

### retinanet.cpp
Реализация работы с TensorRT:
- `loadEngine()` - загрузка TensorRT engine из файла
- `loadLabels()` - загрузка классов COCO из labels.txt
- `allocateBuffers()` - выделение памяти на GPU для входных и выходных данных
- `preprocess()` - нормализация и ресайз кадра до 640x640, конвертация BGR->RGB
- `infer()` - выполнение инференса на GPU через TensorRT
- `postprocess()` - фильтрация детекций по confidence threshold
- `drawDetections()` - отрисовка bounding boxes с масштабированием координат

### main.cpp
Основной файл:
- Открывает входное видео через OpenCV
- Обрабатывает каждый кадр через модель
- Сохраняет результат в выходной видеофайл
- Выводит статистику обработки

## Требования

- CUDA Toolkit 11.0+
- TensorRT 8.0+
- OpenCV 4.0+
- CMake 3.18+

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
- Пайплайн обработки: preprocess -> infer -> postprocess -> draw
- Масштабирование координат bounding boxes под исходное разрешение видео
- Фильтрация детекций по порогу confidence
