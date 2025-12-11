# Makefile для проекта
# Структура:
# include/   <- сюда скачиваются stb-заголовки
# src/       <- исходники (lab3.cu)
# bin/       <- скомпилированный binary

NVCC      := nvcc
CUDACFLAGS:= -O3 -arch=sm_60
INCLUDE   := include
SRC_DIR   := src
BIN_DIR   := bin

STB_IMAGE_URL       := https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
STB_IMAGE_WRITE_URL := https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

STB_IMAGE_FILE       := $(INCLUDE)/stb_image.h
STB_IMAGE_WRITE_FILE := $(INCLUDE)/stb_image_write.h

TARGET := $(BIN_DIR)/lab3
SRC    := $(SRC_DIR)/lab3.cu

.PHONY: all clean download update dirs

all: dirs $(STB_IMAGE_FILE) $(STB_IMAGE_WRITE_FILE) $(TARGET)

dirs:
	@mkdir -p $(INCLUDE)
	@mkdir -p $(BIN_DIR)

# скачать заголовки через curl (если файла нет)
$(STB_IMAGE_FILE):
	@echo "Downloading stb_image.h..."
	@curl -L $(STB_IMAGE_URL) -o $(STB_IMAGE_FILE)

$(STB_IMAGE_WRITE_FILE):
	@echo "Downloading stb_image_write.h..."
	@curl -L $(STB_IMAGE_WRITE_URL) -o $(STB_IMAGE_WRITE_FILE)

# цель всегда пересобирает бинарник при изменениях исходника
$(TARGET): $(SRC)
	$(NVCC) $(CUDACFLAGS) -I$(INCLUDE) -o $@ $<

# Принудительное обновление заголовков
update: dirs
	@echo "Updating stb headers..."
	@curl -L $(STB_IMAGE_URL) -o $(STB_IMAGE_FILE)
	@curl -L $(STB_IMAGE_WRITE_URL) -o $(STB_IMAGE_WRITE_FILE)
	@echo "Updated."

clean:
	-rm -f $(TARGET)
	-rm -f $(STB_IMAGE_FILE) $(STB_IMAGE_WRITE_FILE)
	-rm -rf $(BIN_DIR)

