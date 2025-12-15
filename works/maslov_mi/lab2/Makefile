NVCC := nvcc
TARGET := lab2
SRC := lab2.cu

NVCCFLAGS := -O2 -std=c++14
RM ?= rm -f

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	-@$(RM) $(TARGET) $(TARGET).exe *.o *.obj

.PHONY: all clean run