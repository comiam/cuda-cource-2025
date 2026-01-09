LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$(shell echo $$LD_LIBRARY_PATH)

build:
	rm -rf build/*
	cmake . -B build/
	cmake --build build/

drop_engine:
	rm -f model.engine

run_video_int8:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	./build/RetinaTRT onnx/retinanet.onnx video data/input.mp4 true

run_video:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	./build/RetinaTRT onnx/retinanet.onnx video data/input.mp4 false

clean:
	rm -rf build
	rm -f model.engine
	rm -f output_annotated.mp4

.PHONY: build drop_engine run_video run_video_no_int8 clean
