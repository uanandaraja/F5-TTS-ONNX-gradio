@echo off
title F5-TTS-ONNX with GUI Installer

setlocal EnableDelayedExpansion
set "startTime=%time: =0%"

cls
echo -------------------------------------------------------------
Echo *************** F5-TTS-ONNX with GUI INSTALL ****************
echo -------------------------------------------------------------
echo.
echo  ::  %time:~0,8%  ::  - Setting up the virtual enviroment
Set "VIRTUAL_ENV=venv"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    python.exe -m venv %VIRTUAL_ENV%
)

If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo  ::  %time:~0,8%  ::  - Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"
echo  ::  %time:~0,8%  ::  - Updating the pip package 
python.exe -m pip install --upgrade pip --quiet
echo.
echo  ::  %time:~0,8%  ::  Beginning installation ...
echo.
echo  ::  %time:~0,8%  ::  - Installing required packages
pip install -r requirements.txt --quiet
echo. 
echo  ::  %time:~0,8%  ::  - Downloading vocab file for F5-TTS
curl -s -L https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/vocab.txt > models\vocab.txt
echo.
echo  ::  %time:~0,8%  ::  - Downloading F5-TTS-ONNX model (F5_Decode.onnx)
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Decode.onnx > models\onnx\F5_Decode.onnx
echo. 
echo  ::  %time:~0,8%  ::  - Downloading F5-TTS-ONNX model (F5_Preprocess.onnx)
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Preprocess.onnx > models\onnx\F5_Preprocess.onnx
echo.
echo  ::  %time:~0,8%  ::  - Downloading F5-TTS-ONNX model (F5_Transformer.onnx)
curl --progress-bar -L https://huggingface.co/huggingfacess/F5-TTS-ONNX/resolve/main/F5_Transformer.onnx > models\onnx\F5_Transformer.onnx
echo.
echo.
set "endTime=%time: =0%"
set "end=!endTime:%time:~8,1%=%%100)*100+1!"  &  set "start=!startTime:%time:~8,1%=%%100)*100+1!"
set /A "elap=((((10!end:%time:~2,1%=%%100)*60+1!%%100)-((((10!start:%time:~2,1%=%%100)*60+1!%%100), elap-=(elap>>31)*24*60*60*100"
set /A "cc=elap%%100+100,elap/=100,ss=elap%%60+100,elap/=60,mm=elap%%60+100,hh=elap/60+100"
echo.
echo ............................................................... 
echo *** Installation is completed in %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1% . 
echo *** You can run app with "inference.bat" on windows 
echo *** Or manually, get into the directory, activate venv with "venv\scripts\activate" , then "python F5-TTS-ONNX-Inference.py" . 
echo *** You can close this window.
echo ...............................................................
echo.
