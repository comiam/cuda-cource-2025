TRTEXEC="/usr/src/tensorrt/bin/trtexec"


ONNX_MODEL="models/retinanet_raw.onnx"
OUTPUT_ENGINE="models/retinanet_int8.engine"


if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found at $ONNX_MODEL"
    echo "Please make sure the model is exported and placed in the models/ directory."
    exit 1
fi

echo "[INFO] Building INT8 engine..."
echo "[INFO] ONNX Input : $ONNX_MODEL"
echo "[INFO] Engine Out : $OUTPUT_ENGINE"

$TRTEXEC --onnx="$ONNX_MODEL" --saveEngine="$OUTPUT_ENGINE" --int8 --fp16

if [ $? -eq 0 ]; then
    echo "[OK] Engine saved to $OUTPUT_ENGINE"
else
    echo "[ERROR] trtexec failed."
    exit 1
fi

