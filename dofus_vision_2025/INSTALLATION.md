# 🛠️ INSTALLATION GUIDE - DOFUS Unity World Model AI

**Version 2025.1.0** | **Guide d'Installation Complet** | **Septembre 2025**

---

## 📋 Table des Matières

1. [Prérequis Système](#-prérequis-système)
2. [Installation Rapide](#-installation-rapide)
3. [Installation Avancée](#-installation-avancée)
4. [Configuration](#-configuration)
5. [Vérification](#-vérification)
6. [Troubleshooting](#-troubleshooting)
7. [Optimisations](#-optimisations)

---

## 💻 Prérequis Système

### Configuration Minimale

| Composant | Minimum | Recommandé | Optimal |
|-----------|---------|------------|---------|
| **OS** | Windows 10+ | Windows 11 | Windows 11 Pro |
| **Python** | 3.8+ | 3.11+ | 3.12+ |
| **RAM** | 4GB | 8GB | 16GB+ |
| **CPU** | 4 cores | 6 cores | 8+ cores |
| **GPU** | Intégré | AMD/NVIDIA | AMD 6000+ / RTX 3000+ |
| **Stockage** | 2GB libre | 5GB libre | 10GB+ libre |
| **Réseau** | 10 Mbps | 50 Mbps | 100+ Mbps |

### Logiciels Requis

#### **Python 3.8+**
```bash
# Vérification version Python
python --version
# Doit retourner Python 3.8.x ou supérieur

# Si Python non installé, télécharger depuis:
# https://www.python.org/downloads/
```

#### **Git (optionnel mais recommandé)**
```bash
# Vérification Git
git --version

# Installation Git:
# https://git-scm.com/downloads
```

#### **Visual C++ Redistributable (Windows)**
```bash
# Télécharger et installer:
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
```

### Pilotes GPU (pour accélération)

#### **AMD GPU (ROCm)**
```bash
# Installation ROCm pour Windows (optionnel)
# https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html
```

#### **NVIDIA GPU (CUDA)**
```bash
# Installation CUDA Toolkit (optionnel)
# https://developer.nvidia.com/cuda-downloads
```

---

## ⚡ Installation Rapide

### Méthode 1 : Clone et Setup Automatique

```bash
# 1. Cloner le repository
git clone <repository-url> dofus_vision_2025
cd dofus_vision_2025

# 2. Créer environnement virtuel
python -m venv venv_dofus_ai

# 3. Activer l'environnement
# Windows
venv_dofus_ai\Scripts\activate
# Linux/Mac
source venv_dofus_ai/bin/activate

# 4. Installation des dépendances
pip install -r requirements.txt

# 5. Test de fonctionnement
python tests/test_complete_system.py
```

### Méthode 2 : Script d'Installation Automatique

```bash
# Télécharger et exécuter le script d'installation
curl -O https://raw.githubusercontent.com/.../install.bat
install.bat

# Ou pour Linux/Mac
curl -O https://raw.githubusercontent.com/.../install.sh
chmod +x install.sh
./install.sh
```

### Méthode 3 : Package Wheel (Future)

```bash
# Installation via pip (quand disponible)
pip install dofus-vision-ai
```

---

## 🔧 Installation Avancée

### Étape 1 : Préparation Environnement

#### **Créer Environnement Virtuel**
```bash
# Environnement virtuel avec Python spécifique
python3.11 -m venv venv_dofus_ai --prompt="DOFUS-AI"

# Activation avec verification
venv_dofus_ai\Scripts\activate
python --version  # Vérifier version dans venv
```

#### **Mise à jour pip et setuptools**
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Étape 2 : Installation Dépendances Core

#### **Installation Base**
```bash
# Dépendances principales
pip install opencv-python==4.8.1.78
pip install easyocr==1.7.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install pillow==10.0.0
pip install scikit-learn==1.3.0
```

#### **Installation Vision/OCR**
```bash
# Outils vision par ordinateur
pip install pytesseract==0.3.10
pip install imutils==0.5.4

# Installation Tesseract (Windows)
# Télécharger: https://github.com/UB-Mannheim/tesseract/wiki
# Ajouter au PATH: C:\Program Files\Tesseract-OCR
```

#### **Installation Automation**
```bash
# Automation interface
pip install pyautogui==0.9.54
pip install pygetwindow==0.0.9
pip install pynput==1.7.6
```

#### **Installation Base de Données**
```bash
# SQLite et outils DB
pip install sqlalchemy==2.0.19
pip install sqlite-utils==3.34
```

### Étape 3 : Installation Dépendances Avancées

#### **Machine Learning**
```bash
# Dépendances ML avancées
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==2.13.0
pip install xgboost==1.7.6
```

#### **Interface Graphique**
```bash
# Tkinter est inclus avec Python
# Pour interfaces avancées (optionnel)
pip install customtkinter==5.2.0
pip install tkinter-tooltip==2.2.0
```

#### **Optimisation Performance**
```bash
# Outils performance
pip install numba==0.57.1
pip install cython==3.0.0
pip install psutil==5.9.5
```

### Étape 4 : Installation Environnement Développement

#### **Outils Développement**
```bash
# Linting et formatage
pip install flake8==6.0.0
pip install black==23.7.0
pip install mypy==1.5.1
pip install isort==5.12.0

# Tests
pip install pytest==7.4.0
pip install pytest-cov==4.1.0
pip install pytest-mock==3.11.1

# Documentation
pip install sphinx==7.1.2
pip install sphinx-rtd-theme==1.3.0
```

### Étape 5 : Installation Spécifique OS

#### **Windows**
```bash
# Dépendances Windows spécifiques
pip install pywin32==306
pip install wmi==1.5.1

# Installation Visual Studio Build Tools si erreurs compilation
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### **Linux (Ubuntu/Debian)**
```bash
# Dépendances système
sudo apt update
sudo apt install python3-dev python3-venv
sudo apt install libopencv-dev python3-opencv
sudo apt install tesseract-ocr libtesseract-dev
sudo apt install libgtk-3-dev libboost-all-dev

# Dépendances Python
pip install -r requirements_linux.txt
```

#### **macOS**
```bash
# Installation via Homebrew
brew install python@3.11 opencv tesseract

# Dépendances Python
pip install -r requirements_macos.txt
```

---

## ⚙️ Configuration

### Configuration Système

#### **Variables d'Environnement**
```bash
# Copier et modifier le fichier d'environnement
cp .env.example .env

# Éditer avec vos paramètres
notepad .env  # Windows
nano .env     # Linux
```

#### **Contenu .env**
```bash
# DOFUS Vision 2025 Configuration

# Chemins système
DOFUS_INSTALL_PATH=C:/Program Files (x86)/Dofus/
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe
DATA_PATH=./data/

# Base de données
DATABASE_URL=sqlite:///./data/databases/dofus_knowledge.db
CACHE_DATABASE=sqlite:///./data/cache/cache.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/dofus_vision.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# Performance
MAX_MEMORY_USAGE=512MB
CACHE_TTL=3600
WORKER_THREADS=4

# Vision
OCR_LANGUAGES=fr,en
SCREENSHOT_QUALITY=95
ANALYSIS_TIMEOUT=5000

# Learning
LEARNING_RATE=0.01
BATCH_SIZE=32
MODEL_SAVE_INTERVAL=300

# Human Simulation
DEFAULT_BEHAVIOR_PROFILE=natural
MOUSE_SPEED_FACTOR=1.0
KEYBOARD_DELAY_FACTOR=1.0

# Sécurité
ENABLE_TELEMETRY=false
LOG_SENSITIVE_DATA=false
ENABLE_CRASH_REPORTING=true

# Développement
DEBUG_MODE=false
ENABLE_PROFILING=false
MOCK_DOFUS_WINDOW=false
```

### Configuration Base de Données

#### **Initialisation DB**
```bash
# Script d'initialisation des bases de données
python scripts/init_databases.py

# Vérification
python -c "
from core.knowledge_base import get_knowledge_base
kb = get_knowledge_base()
print('Base de données initialisée avec succès')
"
```

#### **Update Databases**
```bash
# Mise à jour manuelle des données
python scripts/database_updater.py

# Mise à jour simple
python scripts/simple_database_updater.py
```

### Configuration Modules

#### **Configuration Vision Engine**
```python
# config/vision_config.py
VISION_CONFIG = {
    "window_title": "Dofus",
    "capture_method": "win32",  # win32, mss, pyautogui
    "ocr_engine": "easyocr",    # easyocr, tesseract
    "preprocessing": {
        "gaussian_blur": (3, 3),
        "threshold_type": "adaptive",
        "noise_reduction": True
    },
    "combat_grid": {
        "cell_size": 43,
        "grid_offset": (100, 150),
        "detection_threshold": 0.8
    }
}
```

#### **Configuration Learning Engine**
```python
# config/learning_config.py
LEARNING_CONFIG = {
    "algorithm": "q_learning",
    "learning_rate": 0.01,
    "discount_factor": 0.95,
    "exploration_rate": 0.1,
    "memory_size": 10000,
    "batch_size": 32,
    "update_frequency": 100
}
```

#### **Configuration Human Simulation**
```python
# config/human_config.py
HUMAN_CONFIG = {
    "profiles": {
        "natural": {
            "movement_style": "bezier",
            "reaction_time": (0.2, 0.5),
            "click_duration": (0.05, 0.15),
            "error_rate": 0.02
        },
        "nervous": {
            "movement_style": "jittery",
            "reaction_time": (0.1, 0.3),
            "click_duration": (0.03, 0.10),
            "error_rate": 0.05
        }
    }
}
```

---

## ✅ Vérification

### Tests d'Installation

#### **Test Système Complet**
```bash
# Test de tous les modules
python tests/test_complete_system.py

# Sortie attendue:
# [OK] Vision Engine operationnel
# [OK] Knowledge Base operationnel
# [OK] Learning Engine operationnel
# [OK] Human Simulation operationnel
# [OK] Assistant Interface pret
# [OK] Data Extraction operationnel
# [ERROR/OK] HRM Integration: dépendances externes
```

#### **Tests Modules Individuels**
```bash
# Test Knowledge Base
python tests/test_knowledge_base.py

# Test Vision (nécessite DOFUS ouvert)
python -c "
from core.vision_engine import DofusWindowCapture
capture = DofusWindowCapture()
screenshot = capture.capture_screenshot()
print(f'Capture: {screenshot is not None}')
"

# Test OCR
python -c "
from core.vision_engine import DofusUnityInterfaceReader
reader = DofusUnityInterfaceReader()
print('OCR initialisé avec succès')
"
```

#### **Test Performance**
```bash
# Benchmark performance
python -c "
import time
from core import get_knowledge_base

start = time.time()
kb = get_knowledge_base()
init_time = time.time() - start

print(f'Temps initialisation: {init_time:.2f}s')
print('Performance: OK' if init_time < 3.0 else 'Performance: LENT')
"
```

### Validation Configuration

#### **Vérification Chemins**
```bash
# Script de vérification
python -c "
import os
from pathlib import Path

# Vérifications
paths = {
    'Data': './data/',
    'Logs': './data/logs/',
    'Cache': './data/cache/',
    'Databases': './data/databases/'
}

for name, path in paths.items():
    exists = Path(path).exists()
    print(f'{name}: {\"✅\" if exists else \"❌\"} {path}')
"
```

#### **Test Dépendances**
```bash
# Vérification toutes les dépendances
python -c "
import pkg_resources
import sys

required = [
    'opencv-python>=4.8.0',
    'easyocr>=1.7.0',
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'pillow>=10.0.0',
    'scikit-learn>=1.3.0'
]

for requirement in required:
    try:
        pkg_resources.require(requirement)
        print(f'✅ {requirement}')
    except Exception as e:
        print(f'❌ {requirement}: {e}')
"
```

---

## 🔧 Troubleshooting

### Problèmes Fréquents

#### **Erreur 1 : Module 'cv2' not found**
```bash
# Solution 1: Réinstaller OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.1.78

# Solution 2: Vérifier conflits
pip list | grep opencv

# Solution 3: Installation alternative
conda install -c conda-forge opencv
```

#### **Erreur 2 : EasyOCR downloading models**
```bash
# Problème: Téléchargement initial lent
# Solution: Pré-télécharger les modèles
python -c "
import easyocr
reader = easyocr.Reader(['en', 'fr'])
print('Modèles téléchargés')
"

# Ou définir cache personnalisé
export EASYOCR_MODULE_PATH=/path/to/cache
```

#### **Erreur 3 : Permission denied sur Windows**
```bash
# Exécuter en tant qu'administrateur
# Ou modifier permissions dossier:
icacls "C:\path\to\dofus_vision_2025" /grant Users:F /T
```

#### **Erreur 4 : DLL load failed**
```bash
# Installer Visual C++ Redistributable
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

# Ou réinstaller packages problématiques
pip install --force-reinstall --no-deps numpy
```

#### **Erreur 5 : Tesseract not found**
```bash
# Windows: Installer Tesseract OCR
# https://github.com/UB-Mannheim/tesseract/wiki

# Ajouter au PATH ou configurer:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Diagnostics Avancés

#### **Script de Diagnostic**
```bash
# Créer et exécuter diagnostic.py
python -c "
import sys
import platform
import cv2
import numpy as np
import sqlite3
from pathlib import Path

print('=== DIAGNOSTIC SYSTÈME ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')

print('\n=== CHEMINS ===')
paths = ['./data/', './core/', './tests/']
for path in paths:
    exists = Path(path).exists()
    print(f'{path}: {\"✅\" if exists else \"❌\"}')

print('\n=== BASE DE DONNÉES ===')
try:
    conn = sqlite3.connect('./data/databases/dofus_knowledge.db')
    print('✅ Connexion DB OK')
    conn.close()
except Exception as e:
    print(f'❌ DB Error: {e}')

print('\n=== MODULES ===')
modules = ['core', 'core.vision_engine', 'core.knowledge_base']
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except Exception as e:
        print(f'❌ {module}: {e}')
"
```

#### **Log Analysis**
```bash
# Analyser les logs pour problèmes
python -c "
import re
from pathlib import Path

log_file = Path('./data/logs/dofus_vision.log')
if log_file.exists():
    with open(log_file) as f:
        content = f.read()

    errors = re.findall(r'ERROR.*', content)
    warnings = re.findall(r'WARNING.*', content)

    print(f'Erreurs: {len(errors)}')
    print(f'Warnings: {len(warnings)}')

    if errors:
        print('\nDernières erreurs:')
        for error in errors[-5:]:
            print(f'  {error}')
else:
    print('Aucun fichier de log trouvé')
"
```

### Performance Issues

#### **Optimisation Mémoire**
```bash
# Monitoring mémoire
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024

print(f'Mémoire utilisée: {memory_mb:.1f} MB')
print('Status:', 'OK' if memory_mb < 200 else 'ÉLEVÉ')
"

# Réduction utilisation mémoire
export PYTHONHASHSEED=0
export PYTHONMALLOC=malloc
```

#### **Optimisation CPU**
```bash
# Vérifier utilisation multi-core
python -c "
import multiprocessing as mp
import psutil

cores = mp.cpu_count()
usage = psutil.cpu_percent(interval=1)

print(f'Cores disponibles: {cores}')
print(f'Utilisation CPU: {usage}%')
"
```

---

## 🚀 Optimisations

### Performance GPU

#### **Configuration AMD ROCm**
```bash
# Installation ROCm (AMD GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2

# Test GPU AMD
python -c "
import torch
print(f'ROCm disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
"
```

#### **Configuration NVIDIA CUDA**
```bash
# Installation CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Test GPU NVIDIA
python -c "
import torch
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

### Optimisation Base de Données

#### **Index et Vacuum**
```bash
# Optimisation SQLite
python -c "
import sqlite3

conn = sqlite3.connect('./data/databases/dofus_knowledge.db')
cursor = conn.cursor()

# Analyse et optimisation
cursor.execute('PRAGMA optimize')
cursor.execute('VACUUM')
cursor.execute('ANALYZE')

conn.close()
print('Base de données optimisée')
"
```

### Cache Configuration

#### **Configuration Cache Avancé**
```python
# config/cache_config.py
CACHE_CONFIG = {
    "memory_cache": {
        "maxsize": 1000,
        "ttl": 3600
    },
    "disk_cache": {
        "path": "./data/cache/",
        "max_size": "500MB",
        "compression": True
    },
    "redis_cache": {  # Future
        "host": "localhost",
        "port": 6379,
        "db": 0
    }
}
```

### Démarrage Optimisé

#### **Script de Démarrage Rapide**
```bash
# fast_start.py
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def fast_initialization():
    """Initialisation parallèle des modules"""

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Chargement parallèle des modules
        futures = [
            executor.submit(init_knowledge_base),
            executor.submit(init_vision_engine),
            executor.submit(init_learning_engine),
            executor.submit(init_human_simulator)
        ]

        # Attendre completion
        results = await asyncio.gather(*[
            asyncio.wrap_future(future) for future in futures
        ])

    print("Initialisation rapide terminée")
    return results

if __name__ == "__main__":
    asyncio.run(fast_initialization())
```

---

## 📚 Ressources Additionnelles

### Documentation Système

- **[Requirements.txt](requirements.txt)** - Dépendances exactes
- **[Configuration Examples](config/examples/)** - Exemples configuration
- **[Scripts](scripts/)** - Scripts utilitaires
- **[Tests](tests/)** - Suite de tests complète

### Support Installation

- **GitHub Issues** - Problèmes spécifiques
- **Wiki** - Base de connaissances communautaire
- **Discord** - Support temps réel
- **Documentation** - Guides détaillés

### Alternatives Installation

#### **Docker (Future)**
```bash
# Installation via Docker
docker pull dofusvision/ai:latest
docker run -it dofusvision/ai
```

#### **Conda Environment**
```bash
# Alternative avec Conda
conda create -n dofus_ai python=3.11
conda activate dofus_ai
conda install -c conda-forge opencv pytorch numpy pandas
pip install -r requirements_conda.txt
```

---

## ✅ Checklist Installation

### ☐ Prérequis
- [ ] Python 3.8+ installé
- [ ] Git installé (recommandé)
- [ ] 4GB+ RAM disponible
- [ ] 2GB+ espace disque libre

### ☐ Installation
- [ ] Repository cloné
- [ ] Environnement virtuel créé
- [ ] Dépendances installées
- [ ] Configuration .env créée

### ☐ Configuration
- [ ] Chemins système configurés
- [ ] Base de données initialisée
- [ ] Modules testés individuellement
- [ ] Tests système passés

### ☐ Vérification
- [ ] Test complet exécuté avec succès
- [ ] Interface assistant lancée
- [ ] Logs générés correctement
- [ ] Performance acceptable

### ☐ Optimisation (Optionnel)
- [ ] GPU configuré (si disponible)
- [ ] Cache optimisé
- [ ] Base de données indexée
- [ ] Profils comportementaux configurés

---

*Guide d'Installation maintenu par Claude Code - AI Development Specialist*
*Version 2025.1.0 - Septembre 2025*
*Testé sur Windows 10/11, Ubuntu 22.04, macOS 13+*