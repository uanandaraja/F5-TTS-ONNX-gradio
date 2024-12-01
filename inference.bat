@echo off

set PYTHON="%~dp0/venv/Scripts/python.exe"
set VENV_DIR=./venv

echo *** Activating virtual enviroment
call venv\scripts\activate

echo *** Starting GUI
python F5-TTS-ONNX-Inference.py

echo.
pause