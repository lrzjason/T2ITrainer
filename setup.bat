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


echo Step 2. Check Node.js installation
node --version >nul 2>&1
if ERRORLEVEL 1 (
    echo Node.js is not installed or not in PATH.
    echo Please install Node.js version 14 or higher from https://nodejs.org/
    echo Visit https://nodejs.org/ to download and install Node.js.
    pause
    exit /b 1
) else (
    echo Node.js is installed.
    node --version
)

npm --version >nul 2>&1
if ERRORLEVEL 1 (
    echo npm is not installed properly with Node.js.
    echo Please reinstall Node.js from https://nodejs.org/
    pause
    exit /b 1
) else (
    echo npm is installed.
    npm --version
)

echo Step 3. Upgrade pip
python -m pip install --upgrade pip


echo Step 4. Install torch
if "!setup_venv!" == "y" (
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
   set /p install_torch=Do you want to install torch? [Y/n]:
   if "!install_torch!" == "y" (
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   )
)

echo Step 5. Install mandatory diffusers from GitHub
pip install git+https://github.com/huggingface/diffusers.git -U

echo Step 6. Install other dependencies from requirements.txt
pip install -r requirements.txt

echo Step 7. Setup frontend with npm
if exist frontend\ (
    cd frontend
    npm install
    npm run build
    cd ..
) else (
    echo frontend directory not found. Skipping frontend setup.
)

echo Step 8. Download the Flux Fill Model Files (Optional)
set /p install_fill_model=Do you want to download the Flux Fill model files? (y/n):
if "!install_fill_model!" == "y" (
   REM Try running huggingface-cli with --version 
   huggingface-cli --version >nul 2>&1
   IF ERRORLEVEL 1 (
      echo huggingface-cli is not installed. Installing now...
      pip install --upgrade huggingface_hub
   )
   hf download"black-forest-labs/FLUX.1-fill-dev" --local-dir "flux_models/fill/"
)


echo Step 9. Download additional model files (Optional)
set /p install_additional_models=Do you want to download additional models (Qwen Image, Flux Kontext, Original Qwen for lokr)? (y/n):
if "!install_additional_models!" == "y" (
   REM Try running huggingface-cli with --version 
   huggingface-cli --version >nul 2>&1
   IF ERRORLEVEL 1 (
      echo huggingface-cli is not installed. Installing now...
      pip install --upgrade huggingface_hub
   )
   
   set /p install_qwen_image=Download NF4 Qwen Image? (y/n):
   if "!install_qwen_image!" == "y" (
      hf download "lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/
   )
   
   set /p install_flux_kontext=Download NF4 Flux kontext? (y/n):
   if "!install_flux_kontext!" == "y" (
      hf download "lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/
   )
   
   set /p install_flux_fill_nf4=Download NF4 Flux Fill for low gpu? (y/n):
   if "!install_flux_fill_nf4!" == "y" (
      hf download "lrzjason/flux-fill-nf4" --local-dir flux_models/fill/
   )
   
   set /p install_original_qwen=Download original Qwen models for lokr training? (y/n):
   if "!install_original_qwen!" == "y" (
      hf download "Qwen/Qwen-Image" --local-dir qwen_models/qwen_image/
      hf download "Qwen/Qwen-Image-Edit" --local-dir qwen_models/qwen_image_edit/
   )
)

endlocal