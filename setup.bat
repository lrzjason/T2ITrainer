@echo off
setlocal EnableDelayedExpansion

echo 0. Setup venv
set /p setup_venv=Do you want to setup venv? [Y/n]:
if "!setup_venv!" == "y" (
   python -m venv venv
   rem Activate the virtual environment
   call venv\Scripts\activate
)


echo 1. Upgrade pip
python -m pip install --upgrade pip


echo 2. Install torch
set /p install_torch=Do you want to install torch? [Y/n]:
if "!install_torch!" == "y" (
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo 3. Install other dependencies from requirements.txt
pip install -r requirements.txt

echo 4. Download the Model File
set /p install_model=Do you want to download the model file? (y/n):
if "!install_model!" == "y" (
    REM Try running huggingface-cli with --version
    huggingface-cli --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo huggingface-cli is not installed. Installing now...
        pip install --upgrade huggingface_hub
    )
   huggingface-cli download Kwai-Kolors/Kolors --local-dir Kolors_models/
)


endlocal