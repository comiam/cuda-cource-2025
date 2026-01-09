# Variables
ONNX_SCRIPT = export_retinanet_raw.py
ONNX_MODEL = retinanet_640_raw.onnx
ENGINE_NAME = model_640x640.engine
APP_NAME = RetinaTRT
BUILD_DIR = build
VIDEO_INPUT = videos/test.mp4
LABELS_SOURCE = labels.txt
TRTEXEC = /usr/src/tensorrt/bin/trtexec

# Default target
all: export build_app

# 1. Export ONNX
export: $(ONNX_MODEL)

$(ONNX_MODEL): $(ONNX_SCRIPT)
	@echo "[INFO] Exporting RetinaNet Raw..."
	python3 $(ONNX_SCRIPT)

# 2. Build TensorRT Engine explicitly via trtexec (FP16 mode)
# Note: This creates an FP16 engine. If you want INT8 with video calibration, 
# use 'make run_i8' (letting the C++ app build it) or 'make clean_engine' first.
engine: $(ONNX_MODEL)
	@echo "[INFO] Building TensorRT Engine (FP16) via trtexec..."
	$(TRTEXEC) --onnx=$(ONNX_MODEL) --saveEngine=$(ENGINE_NAME) --fp16

# 3. Build C++ Application
build_app:
	@echo "[INFO] Building C++ Application..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j$(shell nproc)
	@if [ -f $(LABELS_SOURCE) ]; then cp $(LABELS_SOURCE) $(BUILD_DIR)/labels.txt; fi
	@if [ -f labels.txt ]; then cp labels.txt $(BUILD_DIR)/labels.txt; fi

# Clean engine to force rebuild
clean_engine:
	@echo "[INFO] Removing engine to force rebuild..."
	rm -f $(ENGINE_NAME)

# Run with INT8 (triggers internal C++ engine build/calibration if engine missing)
run_i8: all
	@echo "[INFO] Running $(APP_NAME) with INT8 (Auto-build engine if missing)..."
	./$(BUILD_DIR)/$(APP_NAME) $(ONNX_MODEL) video $(VIDEO_INPUT) true

# Run using the pre-built engine (via 'make engine') or FP16 mode
run_fp16: all
	@echo "[INFO] Running $(APP_NAME) with FP16..."
	./$(BUILD_DIR)/$(APP_NAME) $(ONNX_MODEL) video $(VIDEO_INPUT) false

# Build engine explicitly then run_i8
run_engine: engine build_app
	@echo "[INFO] Running with explicitly built FP16 engine..."
	./$(BUILD_DIR)/$(APP_NAME) $(ONNX_MODEL) video $(VIDEO_INPUT) false

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(ONNX_MODEL)
	rm -f $(ENGINE_NAME)
	rm -f output_annotated.mp4
	rm -f *.engine
	rm -f labels.txt

.PHONY: all export engine build_app clean_engine run_i8 run_fp16 run_engine clean
