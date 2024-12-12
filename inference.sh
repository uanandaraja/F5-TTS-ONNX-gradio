#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
VENV_DIR=.venv

echo "*** Activating virtual environment"
source $VENV_DIR/bin/activate

echo "*** Starting gradio app"
python inference.py
