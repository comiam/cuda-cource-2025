LD_LIBRARY_PATH=/home/anastasia/TensorRT-10.14.1.48/targets/x86_64-linux-gnu/lib/:/usr/lib/x86_64-linux-gnu/

build:
	rm -rf build/*
	cmake . -B build/
	cmake --build build/

drop_engine:
	rm -rf models/*.engine

run:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	./build/DetectionModelTRT models/yolov10s_dyn.onnx 
