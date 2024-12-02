# F5-TTS-ONNX-gui standalone & GPU accelerated 

- Have python & git installed.
- Clone this repo with "git clone https://github.com/patientx/F5-TTS-ONNX-gui"

## INSTALLATION 
*** Run install.bat
- Default english version of F5-TTS models are automatically downloaded from huggingface into /models/onnx/ folder.
- (If you can't reach huggingface you can use this link as a backup : (https://drive.google.com/file/d/1_QNdX-6l8iwDF9c6HxOxW5bpCB6HJGnR/view)

## RUNNING
* After you put those models , you can run the app by simply running "inference.bat" , there would be a simple GUI with default values.
- If you want to manually run it, get into the folder via commandline , venv\scripts\activate , "python F5-TTS-ONNX-Inference.py" .
- Make sure your sample audio is inside audio folder , change the first box to whatever you want the model to say , and the last box to write what your reference audio is saying.
* Change "generated audio file" 's name if you don't want it overwritten 

* This fork is using onnxruntime to use your GPU. AMD (from 6000 onwards better) and Nvidia GPU's should work at an acceptable speed. 

*** Here is an alternative ONNX model which accepts audio files twice longer than default.
--- https://www.mediafire.com/file/dkwdbfswqks414u/f5-tts-onnx-4096.zip/file ---


  
