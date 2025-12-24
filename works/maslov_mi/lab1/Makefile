NVCC := nvcc
TARGET := lab1
SRC := src/lab1.cu

NVCCFLAGS := -O2 -std=c11
RM ?= rm -f

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	-@$(RM) $(TARGET) $(TARGET).exe *.o *.obj

.PHONY: all clean run
