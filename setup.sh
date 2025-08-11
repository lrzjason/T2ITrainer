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

echo "Step 2. Upgrade pip"
python -m pip install --upgrade pip

echo "Step 3. Install torch"
if [ "$setup_venv" == "y" ]; then
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
   read -p "Do you want to install torch? [Y/n]: " install_torch
   if [ "$install_torch" == "y" ]; then
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   fi
fi

echo "Step 4. Install other dependencies from requirements.txt"
pip install -r requirements.txt

echo "Step 5. Download the Kolors Model Files (Optional)"
read -p "Do you want to download the kolors model files? (y/n): " install_kolors_model
if [ "$install_kolors_model" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   hf downloadKwai-Kolors/Kolors --local-dir kolors_models/
fi


echo "Step 5. Download the SD3.5 Large Model Files (Optional)"
read -p "Do you want to download the SD3.5 Large model files? (y/n): " install_sd35_model
if [ "$install_sd35_model" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   hf download"stabilityai/stable-diffusion-3.5-large" --local-dir "sd3.5L/"
fi


echo "Step 5. Download the Flux Fill Model Files (Optional)"
read -p "Do you want to download the Flux Fill model files? (y/n): " install_fill_model
if [ "$install_fill_model" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   hf download"black-forest-labs/FLUX.1-fill-dev" --local-dir "flux_models/fill/"
fi