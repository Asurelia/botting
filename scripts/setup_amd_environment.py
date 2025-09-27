#!/usr/bin/env python3
"""
Configuration automatique de l'environnement IA pour AMD 7800XT sous Windows 11
Optimise PyTorch, OpenCV et les dépendances pour performances maximales
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMDEnvironmentSetup:
    """Configurateur d'environnement optimisé AMD"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.requirements_dir = self.project_root / "requirements"
        self.requirements_dir.mkdir(exist_ok=True)

        # Vérification système
        self.system_info = self._get_system_info()

    def _get_system_info(self):
        """Collecte les informations système"""
        return {
            "os": platform.system(),
            "version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": sys.version
        }

    def check_amd_gpu(self):
        """Vérifie la présence et configuration du GPU AMD"""
        logger.info("🔍 Vérification GPU AMD...")

        try:
            # Tentative de détection via WMI (Windows)
            if platform.system() == "Windows":
                result = subprocess.run([
                    "wmic", "path", "win32_VideoController",
                    "get", "name,adapterram"
                ], capture_output=True, text=True, shell=True)

                if result.returncode == 0:
                    gpu_info = result.stdout
                    if "AMD" in gpu_info or "Radeon" in gpu_info:
                        logger.info("✅ GPU AMD détecté")
                        if "7800" in gpu_info:
                            logger.info("✅ AMD 7800XT confirmé - Configuration optimisée disponible")
                            return True
                        else:
                            logger.info("✅ GPU AMD détecté (modèle à confirmer)")
                            return True
                    else:
                        logger.warning("⚠️ GPU AMD non détecté clairement")
                        return False

        except Exception as e:
            logger.error(f"❌ Erreur détection GPU: {e}")

        return False

    def create_optimized_requirements(self):
        """Crée les fichiers requirements optimisés"""
        logger.info("📝 Création requirements optimisés...")

        # Requirements de base IA
        base_requirements = """# IA DOFUS - Base Requirements
# Optimisé pour AMD 7800XT + Windows 11

# Computer Vision & Deep Learning
ultralytics>=8.0.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0

# Machine Learning & Data Science
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Graph Databases & Knowledge Management
networkx>=3.1
neo4j>=5.12.0
rdflib>=7.0.0

# Performance & Optimization
numba>=0.58.0
joblib>=1.3.0
cython>=3.0.0

# Async & Concurrency
aiohttp>=3.8.0

# Utilities
tqdm>=4.65.0
click>=8.1.0
pyyaml>=6.0
python-dotenv>=1.0.0
psutil>=5.9.0
"""

        # Requirements pour GPU AMD
        amd_requirements = """# IA DOFUS - AMD GPU Requirements
# Support DirectML pour AMD 7800XT

# PyTorch avec DirectML
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
torch-directml>=0.2.0

# ONNX Runtime pour optimisation
onnxruntime-directml>=1.16.0
onnx>=1.14.0

# Alternative TensorFlow avec DirectML (optionnel)
tensorflow-directml>=1.15.8
"""

        # Requirements développement
        dev_requirements = """# IA DOFUS - Development Requirements

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Code Quality
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0

# Jupyter & Analysis
jupyter>=1.0.0
ipykernel>=6.25.0
plotly>=5.15.0

# Profiling & Debugging
line_profiler>=4.1.0
memory_profiler>=0.61.0
py-spy>=0.3.14
"""

        # Sauvegarde des fichiers
        requirements_files = {
            "base.txt": base_requirements,
            "amd_gpu.txt": amd_requirements,
            "dev.txt": dev_requirements
        }

        for filename, content in requirements_files.items():
            file_path = self.requirements_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"✅ Créé: {file_path}")

        return True

    def install_base_requirements(self):
        """Installe les requirements de base"""
        logger.info("📦 Installation requirements de base...")

        requirements_file = self.requirements_dir / "base.txt"
        if not requirements_file.exists():
            logger.error("❌ Fichier requirements de base non trouvé")
            return False

        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Requirements de base installés")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur installation: {e.stderr}")
            return False

    def install_amd_pytorch(self):
        """Installe PyTorch optimisé pour AMD"""
        logger.info("📦 Installation PyTorch pour AMD...")

        pytorch_commands = [
            # Désinstallation versions existantes
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],

            # Installation PyTorch standard + DirectML
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"],
            [sys.executable, "-m", "pip", "install", "torch-directml"]
        ]

        for cmd in pytorch_commands:
            try:
                logger.info(f"Exécution: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("✅ Commande réussie")
            except subprocess.CalledProcessError as e:
                if "torch-directml" in cmd:
                    logger.error("❌ Impossible d'installer torch-directml")
                    return False
                else:
                    logger.info("ℹ️ Tentative méthode alternative...")
                    continue

        return True

    def install_amd_requirements(self):
        """Installe les requirements AMD spécifiques"""
        logger.info("📦 Installation requirements AMD...")

        requirements_file = self.requirements_dir / "amd_gpu.txt"
        if not requirements_file.exists():
            logger.error("❌ Fichier requirements AMD non trouvé")
            return False

        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Requirements AMD installés")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Erreur installation AMD: {e.stderr}")
            logger.info("ℹ️ Tentative installation alternative...")

            # Installation alternative une par une
            alternative_packages = [
                "onnxruntime-directml"
            ]

            for package in alternative_packages:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    subprocess.run(cmd, check=True)
                    logger.info(f"✅ {package} installé")
                except:
                    logger.warning(f"⚠️ Échec installation {package}")

            return True

    def test_gpu_acceleration(self):
        """Teste l'accélération GPU"""
        logger.info("🧪 Test accélération GPU...")

        test_script = """
import torch
import sys

print("=== Test GPU AMD ===")
print(f"PyTorch version: {torch.__version__}")

# Test CUDA (peu probable sur AMD mais on vérifie)
if torch.cuda.is_available():
    print(f"✅ CUDA disponible - Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ℹ️ CUDA non disponible (normal pour AMD)")

# Test DirectML
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"✅ DirectML disponible - Device: {device}")

        # Test simple
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("✅ Test calcul GPU réussi")
        exit(0)
    else:
        print("❌ DirectML non disponible")
except ImportError:
    print("❌ torch-directml non installé")
except Exception as e:
    print(f"❌ Erreur DirectML: {e}")

# Test CPU en fallback
print("ℹ️ Utilisation CPU en fallback")
x = torch.randn(100, 100)
y = torch.randn(100, 100)
z = torch.mm(x, y)
print("✅ Test calcul CPU réussi")
exit(1)
"""

        try:
            result = subprocess.run([sys.executable, "-c", test_script],
                                 capture_output=True, text=True, timeout=30)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.error("❌ Test GPU timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur test GPU: {e}")
            return False

    def create_gpu_config(self):
        """Crée la configuration GPU optimisée"""
        logger.info("⚙️ Création configuration GPU...")

        gpu_available = self.test_gpu_acceleration()

        config = {
            "gpu": {
                "vendor": "AMD",
                "model": "7800XT",
                "backend": "directml",
                "available": gpu_available,
                "optimization": {
                    "mixed_precision": True,
                    "memory_efficient": True,
                    "batch_size_auto": True
                }
            },
            "pytorch": {
                "device": "directml" if gpu_available else "cpu",
                "amp_enabled": True,
                "compile_enabled": True
            },
            "yolo": {
                "device": "0" if gpu_available else "cpu",
                "half": gpu_available,   # FP16 pour performance
                "optimize": True
            },
            "performance": {
                "max_workers": 8,  # Adapté au processeur
                "memory_limit_gb": 16,
                "cache_enabled": True
            }
        }

        config_file = self.project_root / "config" / "gpu_config.json"
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ Configuration GPU sauvegardée: {config_file}")
        return config

    def run_full_setup(self):
        """Lance l'installation complète"""
        logger.info("🚀 Démarrage installation complète IA DOFUS...")

        steps = [
            ("Vérification GPU AMD", self.check_amd_gpu),
            ("Création requirements", self.create_optimized_requirements),
            ("Installation base", self.install_base_requirements),
            ("Installation PyTorch AMD", self.install_amd_pytorch),
            ("Installation AMD", self.install_amd_requirements),
            ("Configuration GPU", self.create_gpu_config)
        ]

        results = {}

        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 {step_name}...")
            logger.info(f"{'='*50}")

            try:
                result = step_func()
                results[step_name] = result

                if result:
                    logger.info(f"✅ {step_name} - SUCCÈS")
                else:
                    logger.warning(f"⚠️ {step_name} - ÉCHEC (continuant...)")

            except Exception as e:
                logger.error(f"❌ {step_name} - ERREUR: {e}")
                results[step_name] = False

        # Rapport final
        logger.info(f"\n{'='*60}")
        logger.info("📊 RAPPORT FINAL D'INSTALLATION")
        logger.info(f"{'='*60}")

        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)

        for step, result in results.items():
            status = "✅ SUCCÈS" if result else "❌ ÉCHEC"
            logger.info(f"{step}: {status}")

        logger.info(f"\nScore: {success_count}/{total_count}")

        if success_count >= total_count * 0.7:  # 70% de succès minimum
            logger.info("🎉 INSTALLATION RÉUSSIE ! Prêt pour l'IA DOFUS")
            return True
        else:
            logger.error("💥 INSTALLATION INCOMPLÈTE - Vérifiez les erreurs")
            return False

def main():
    """Point d'entrée principal"""
    print("🚀 IA DOFUS - Configuration Environnement AMD 7800XT")
    print("=" * 60)

    setup = AMDEnvironmentSetup()

    # Affichage info système
    print(f"🖥️ Système: {setup.system_info['os']} {setup.system_info['version']}")
    print(f"🐍 Python: {setup.system_info['python_version']}")
    print("=" * 60)

    success = setup.run_full_setup()

    if success:
        print("\n🎯 PROCHAINES ÉTAPES:")
        print("1. Lancez: python core/ai_framework.py --init")
        print("2. Testez: python scripts/gemini_consensus.py autonomy_architecture")
        print("3. Consultez: docs/PLAN_AUTONOMIE_ENRICHI.md")
        print("\n🚀 L'IA DOFUS vous attend !")
    else:
        print("\n🔧 ACTIONS CORRECTIVES:")
        print("1. Vérifiez les logs d'erreur ci-dessus")
        print("2. Installez manuellement les packages échoués")
        print("3. Relancez le script après corrections")

if __name__ == "__main__":
    main()