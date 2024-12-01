@echo off
title F5-TTS-ONNX with GUI Installer

setlocal EnableDelayedExpansion
set "startTime=%time: =0%"

cls
echo -------------------------------------------------------------
Echo ************ F5-TTS-ONNX with GUI INSTALL ************
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
set "endTime=%time: =0%"
set "end=!endTime:%time:~8,1%=%%100)*100+1!"  &  set "start=!startTime:%time:~8,1%=%%100)*100+1!"
set /A "elap=((((10!end:%time:~2,1%=%%100)*60+1!%%100)-((((10!start:%time:~2,1%=%%100)*60+1!%%100), elap-=(elap>>31)*24*60*60*100"
set /A "cc=elap%%100+100,elap/=100,ss=elap%%60+100,elap/=60,mm=elap%%60+100,hh=elap/60+100"
echo ............................................................... 
echo *** Installation is completed in %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1% . 
echo *** Please download models from this url and extract them into models\onnx folder. 
echo *** https://drive.google.com/file/d/1_QNdX-6l8iwDF9c6HxOxW5bpCB6HJGnR/view?usp=sharing
echo ...............................................................
echo.
pause
