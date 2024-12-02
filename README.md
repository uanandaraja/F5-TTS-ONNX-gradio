---- Original README ---
# F5-TTS-ONNX
Running the F5-TTS  by ONNX Runtime
- 我们已经更新了代码，以适配截至2024年11月28日的SWivid/F5-TTS，成功地导出为ONNX格式。如果您之前遇到了错误，可以下载最新的代码并再试一次。
- 适用于配备 AMD GPU + Windows 操作系统的简单解决方案，使用 ONNX-DirectML 执行F5-TTS。（`pip install onnxruntime-directml --upgrade`）
- 简单的GUI版本 -> https://github.com/patientx/F5-TTS-ONNX-gui
- 看更多項目 -> https://dakeqq.github.io/overview/
- We have updated the code to adapt to SWivid/F5-TTS as of 2024/11/28, successfully exporting to the ONNX format. If you encountered an error before, you can download the latest code and try again.
- It is an easy solution for Windows OS with an AMD GPU using the ONNX-DirectML execution provider. (`pip install onnxruntime-directml --upgrade`)
- Get try with easy GUI version -> https://github.com/patientx/F5-TTS-ONNX-gui.
- See more -> https://dakeqq.github.io/overview/
-------------------------------------------------------------------
# F5-TTS-ONNX-gui standalone & GPU accelerated 
- have python & git installed.
- clone this repo with "git clone https://github.com/patientx/F5-TTS-ONNX-gui"
- run install.bat
- download ONNX models for default english version of F5-TTS model from this link and put them into /models/onnx/ folder. (three files , F5_Decode.onnx , F5_Preprocess.onnx , F5_Transformer.onnx)
- https://drive.google.com/file/d/1_QNdX-6l8iwDF9c6HxOxW5bpCB6HJGnR/view
- after you put those models , you can run the app by simply running "inference.bat" , there would be a simple GUI with default values.
- put your sample audio inside audio folder , change the first box to whatever you want the audio to say , and the last box to write what your reference audio is saying.
- change "generated audio file" 's name if you don't want it overwritten 
- this fork is using onnxruntime to use your GPU. AMD (from 6000 onwards better) and Nvidia GPU's should work at an acceptable speed. 

- Here is an alternative ONNX model which accepts audio files twice longer than default.
- https://www.mediafire.com/file/dkwdbfswqks414u/f5-tts-onnx-4096.zip/file
  
