@echo off
setlocal EnableDelayedExpansion

echo Step 1. Setup venv

set /a setup_venv="n"
if exist venv\ (
   rem Activate the virtual environment
   call venv\Scripts\activate
   set /a setup_venv="y"
) else (
   set /p setup_venv=Do you want to setup venv? [Y/n]:
   if "!setup_venv!" == "y" (
      python -m venv venv
      rem Activate the virtual environment
      call venv\Scripts\activate
      set /a setup_venv="y"
   )
)


echo Step 2. Upgrade pip
python -m pip install --upgrade pip


echo Step 3. Install torch
if "!setup_venv!" == "y" (
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
   set /p install_torch=Do you want to install torch? [Y/n]:
   if "!install_torch!" == "y" (
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   )
)

echo Step 4. Install other dependencies from requirements.txt
pip install -r requirements.txt

echo Step 5. Download the Kolors Model Files (Optional)
set /p install_kolors_model=Do you want to download the kolors model files? (y/n):
if "!install_kolors_model!" == "y" (
   REM Try running huggingface-cli with --version 
   huggingface-cli --version >nul 2>&1
   IF ERRORLEVEL 1 (
      echo huggingface-cli is not installed. Installing now...
      pip install --upgrade huggingface_hub
   )
   huggingface-cli download Kwai-Kolors/Kolors --local-dir kolors_models/
)


echo Step 5. Download the SD3.5 Large Model Files (Optional)
set /p install_sd35_model=Do you want to download the SD3.5 Large model files? (y/n):
if "!install_sd35_model!" == "y" (
   REM Try running huggingface-cli with --version 
   huggingface-cli --version >nul 2>&1
   IF ERRORLEVEL 1 (
      echo huggingface-cli is not installed. Installing now...
      pip install --upgrade huggingface_hub
   )
   huggingface-cli download "stabilityai/stable-diffusion-3.5-large" --local-dir "sd3.5L/"
)


echo Step 5. Download the Flux Fill Model Files (Optional)
set /p install_fill_model=Do you want to download the Flux Fill model files? (y/n):
if "!install_fill_model!" == "y" (
   REM Try running huggingface-cli with --version 
   huggingface-cli --version >nul 2>&1
   IF ERRORLEVEL 1 (
      echo huggingface-cli is not installed. Installing now...
      pip install --upgrade huggingface_hub
   )
   huggingface-cli download "black-forest-labs/FLUX.1-fill-dev" --local-dir "flux_models/fill/"
)

endlocal