#!/bin/bash

echo "Step 1. Setup venv"
read -p "Do you want to setup venv? [Y/n]: " setup_venv
if [ "$setup_venv" == "y" ]; then
   python -m venv venv
   # Activate the virtual environment
   source venv/bin/activate
elif [ -d "venv" ]; then
   # Activate the virtual environment
   source venv/bin/activate
   setup_venv="y"
fi

echo "Step 2. Check Node.js installation"
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed or not in PATH."
    echo "Please install Node.js version 14 or higher from https://nodejs.org/"
    echo "Visit https://nodejs.org/ to download and install Node.js."
    exit 1
else
    echo "Node.js is installed."
    node --version
fi

if ! command -v npm &> /dev/null; then
    echo "npm is not installed properly with Node.js."
    echo "Please reinstall Node.js from https://nodejs.org/"
    exit 1
else
    echo "npm is installed."
    npm --version
fi

echo "Step 3. Upgrade pip"
python -m pip install --upgrade pip

echo "Step 4. Install torch"
if [ "$setup_venv" == "y" ]; then
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
   read -p "Do you want to install torch? [Y/n]: " install_torch
   if [ "$install_torch" == "y" ]; then
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   fi
fi

echo "Step 5. Install mandatory diffusers from GitHub"
pip install git+https://github.com/huggingface/diffusers.git -U

echo "Step 6. Install other dependencies from requirements.txt"
pip install -r requirements.txt

echo "Step 7. Setup frontend with npm"
if [ -d "frontend" ]; then
    cd frontend
    npm install
    npm run build
    cd ..
else
    echo "frontend directory not found. Skipping frontend setup."
fi

echo "Step 8. Download the Flux Fill Model Files (Optional)
read -p "Do you want to download the Flux Fill model files? (y/n): " install_fill_model
if [ "$install_fill_model" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   hf download"black-forest-labs/FLUX.1-fill-dev" --local-dir "flux_models/fill/"
fi


echo "Step 9. Download additional model files (Optional)
read -p "Do you want to download additional models (Qwen Image, Flux Kontext, Original Qwen for lokr)? (y/n): " install_additional_models
if [ "$install_additional_models" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   
   read -p "Download NF4 Qwen Image? (y/n): " install_qwen_image
   if [ "$install_qwen_image" == "y" ]; then
      hf download "lrzjason/qwen_image_nf4" --local-dir qwen_models/qwen_image_nf4/
   fi
   
   read -p "Download NF4 Flux kontext? (y/n): " install_flux_kontext
   if [ "$install_flux_kontext" == "y" ]; then
      hf download "lrzjason/flux-kontext-nf4" --local-dir flux_models/kontext/
   fi
   
   read -p "Download NF4 Flux Fill for low gpu? (y/n): " install_flux_fill_nf4
   if [ "$install_flux_fill_nf4" == "y" ]; then
      hf download "lrzjason/flux-fill-nf4" --local-dir flux_models/fill/
   fi
   
   read -p "Download original Qwen models for lokr training? (y/n): " install_original_qwen
   if [ "$install_original_qwen" == "y" ]; then
      hf download "Qwen/Qwen-Image" --local-dir qwen_models/qwen_image/
      hf download "Qwen/Qwen-Image-Edit" --local-dir qwen_models/qwen_image_edit/
   fi
fi