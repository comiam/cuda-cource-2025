# Lab 3: Sobel Operator

## Описание

Реализация оператора Собеля для обнаружения границ в изображениях на CUDA. Поддерживает форматы PGM и PNG.

## Компиляция

```bash
make
```

Очистка скомпилированных файлов:
```bash
make clean
```

## Запуск

```bash
./sobel images/bird.pgm images/bird_sobel.pgm
```

### Пример вывода:

```
Loading image: images/bird.pgm
Image size: 321 x 481
Launching kernel: grid(21, 31), block(16, 16)
GPU execution time: 0.273 ms
Saving result: images/bird_sobel.pgm
Processing completed successfully!
```

## Результаты

```
Loading image: images/bird.pgm
Image size: 321 x 481
Launching kernel: grid(21, 31), block(16, 16)
GPU execution time: 0.273 ms
Saving result: images/bird_sobel.pgm
Processing completed successfully!
```

```
Loading image: images/2k.png
Loaded image: images/2k.png (2560x1440, 4 channels)
Image size: 2560 x 1440
Launching kernel: grid(160, 90), block(16, 16)
GPU execution time: 0.186 ms
Saving result: images/2k_sobel.png
Processing completed successfully!
```

```
Loading image: images/test.png
Loaded image: images/test.png (1080x640, 4 channels)
Image size: 1080 x 640
Launching kernel: grid(68, 40), block(16, 16)
GPU execution time: 0.144 ms
Saving result: images/test_sobel.png
Processing completed successfully!
```

## Замечания

- Программа автоматически определяет формат входного изображения по расширению или сигнатуре файла
- Выходной формат определяется по расширению выходного файла
- Для PNG изображений выполняется автоматическое преобразование в grayscale