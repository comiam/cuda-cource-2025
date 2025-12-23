# Lab 3: Sobel Edge Detection - CUDA Implementation
## Компиляция

```bash
make
```

## Использование

### Использование через Makefile

```bash
make run

```

### Очистка артефактов сборки
```bash
make clean
```

## Реузльтаты

В качестве тестов использовались 2 картинки png разных размеров и с несколькими каналами (RGB и RGBA):
```bash
Loaded image: data/image.png (640x640, 3 channels)
Время на GPU: 0.48 мс
Saved PGM: image_sobel.pgm
Saved PNG: image_sobel.png
Loaded image: data/MSI.png (3840x2160, 4 channels)
Время на GPU: 0.18 мс
Saved PGM: MSI_sobel.pgm
Saved PNG: MSI_sobel.png
```