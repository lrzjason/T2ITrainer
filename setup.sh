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

echo "Step 5. Download the Model Files"
read -p "Do you want to download the model files? (y/n): " install_model
if [ "$install_model" == "y" ]; then
   # Try running huggingface-cli with --version
   if ! huggingface-cli --version &> /dev/null; then
      echo "huggingface-cli is not installed. Installing now..."
      pip install --upgrade huggingface_hub
   fi
   huggingface-cli download Kwai-Kolors/Kolors --local-dir kolors_models/
fi