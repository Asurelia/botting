#!/bin/bash
#
# Script d'installation PyTorch + ROCm pour AMD RX 7800 XT
# À exécuter DANS le venv: source venv_rocm/bin/activate && bash setup_pytorch_rocm.sh
#

set -e

echo "🎮 Installation PyTorch avec support ROCm pour AMD RX 7800 XT..."
echo

# Check if in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERREUR: Vous devez activer le venv d'abord!"
    echo "   Exécutez: source venv_rocm/bin/activate"
    exit 1
fi

echo "✅ Environnement virtuel détecté: $VIRTUAL_ENV"
echo

# Install PyTorch with ROCm 5.7
echo "📦 Installation PyTorch + torchvision avec ROCm 5.7..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

echo
echo "🔧 Installation dépendances deep learning..."
pip install ultralytics>=8.0.0  # YOLOv8
pip install easyocr>=1.7.0      # OCR avancé

echo
echo "🧪 Installation dépendances tests..."
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0
pip install pytest-asyncio>=0.21.0
pip install pytube>=15.0.0

echo
echo "📊 Installation autres dépendances..."
pip install python-xlib  # X11 bindings pour Python

echo
echo "✅ Installation terminée!"
echo

# Test PyTorch + ROCm
echo "🔍 Test PyTorch + ROCm..."
python3 << 'PYTHON_TEST'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    print("✅ ROCm fonctionnel!")
else:
    print("⚠️  CUDA/ROCm non disponible - vérifier installation ROCm système")
PYTHON_TEST

echo
echo "🎉 Setup PyTorch + ROCm terminé!"
echo
echo "Prochaine étape: Créer structure de tests"
echo "  python tests/setup_test_data.py"
