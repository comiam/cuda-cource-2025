LD_LIBRARY_PATH=/home/anastasia/TensorRT-10.14.1.48/targets/x86_64-linux-gnu/lib/:/usr/lib/x86_64-linux-gnu/

build:
	rm -rf build/*
	cmake . -B build/
	cmake --build build/

drop_engine:
	rm -rf models/*.engine

run_video:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	./build/DetectionModelTRT models/yolov10s_dyn.onnx video videos/v_J41d0cHAfSM_c008.mp4 false

run_images:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	./build/DetectionModelTRT models/yolov10s_dyn.onnx images assets/ false

compare:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	echo "No INT8 optimization:"
	./build/DetectionModelTRT models/yolov10s_dyn.onnx images assets/ false
	echo "\nWith INT8 optimization:"
	./build/DetectionModelTRT models/yolov10s_dyn.onnx images assets/ true

.PHONY: build drop_engine run_images run_video compare