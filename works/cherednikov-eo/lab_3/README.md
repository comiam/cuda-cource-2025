# Lab 3: Sobel Operator

## Описание


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

## Результаты

### Пример вывода:

```
Loading image: images/bird.pgm
Image size: 321 x 481
Launching kernel: grid(21, 31), block(16, 16)
GPU execution time: 0.273 ms
Saving result: images/bird_sobel.pgm
Processing completed successfully!
```



## Замечания
