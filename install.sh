#!/bin/bash

echo "-------------------------------------------------------------"
echo "*************** F5-TTS-ONNX with GUI INSTALL ****************"
echo "-------------------------------------------------------------"

VIRTUAL_ENV=".venv"
start_time=$(date +%s)

echo "::$(date +%H:%M:%S):: - Setting up the virtual environment"
if [ ! -f "${VIRTUAL_ENV}/bin/activate" ]; then
  uv venv --python 3.10
fi

if [ ! -f "${VIRTUAL_ENV}/bin/activate" ]; then
  exit 1
fi

echo "::$(date +%H:%M:%S):: - Virtual environment activation"
source "${VIRTUAL_ENV}/bin/activate"

echo "::$(date +%H:%M:%S):: - Installing required packages"
uv pip install -r requirements.txt

echo "::$(date +%H:%M:%S):: - Downloading vocab file for F5-TTS"
curl -s -L https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/vocab.txt >models/vocab.txt

echo "::$(date +%H:%M:%S):: - Downloading F5-TTS-ONNX model (F5_Decode.onnx)"
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Decode.onnx >models/onnx/F5_Decode.onnx

echo "::$(date +%H:%M:%S):: - Downloading F5-TTS-ONNX model (F5_Preprocess.onnx)"
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Preprocess.onnx >models/onnx/F5_Preprocess.onnx

echo "::$(date +%H:%M:%S):: - Downloading F5-TTS-ONNX model (F5_Transformer.onnx)"
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Transformer.onnx >models/onnx/F5_Transformer.onnx

end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "\nInstallation completed in $duration seconds."
echo "Run with: source .venv/bin/activate && python inference.py"
