#!/bin/bash
#
# Script d'installation des dépendances Linux pour Bot Dofus
# À exécuter avec: bash setup_linux_dependencies.sh
#

set -e  # Exit on error

echo "🚀 Installation des dépendances système Linux..."
echo

# Update apt
echo "📦 Mise à jour apt..."
sudo apt update

# Install system tools
echo "🔧 Installation outils système..."
sudo apt install -y \
    xdotool \
    wmctrl \
    tesseract-ocr \
    tesseract-ocr-fra \
    python3-xlib \
    python3-dev \
    build-essential

echo
echo "✅ Dépendances système installées!"
echo

# Check installations
echo "🔍 Vérification installations..."
echo -n "xdotool: "
which xdotool && echo "✅" || echo "❌"

echo -n "wmctrl: "
which wmctrl && echo "✅" || echo "❌"

echo -n "tesseract: "
which tesseract && echo "✅" || echo "❌"

echo
echo "🎉 Setup système terminé!"
echo
echo "Prochaine étape: Installer PyTorch avec ROCm"
echo "  cd /home/spoukie/Documents/Botting"
echo "  source venv_rocm/bin/activate"
echo "  bash setup_pytorch_rocm.sh"
