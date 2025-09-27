#!/usr/bin/env python3
"""
Script Automatisé - Création Environnement IA DOFUS
Crée un environnement conda optimal avec Python 3.11 et torch-directml
"""

import subprocess
import sys
import os
import platform
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIEnvironmentCreator:
    """Créateur d'environnement IA automatisé"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_name = "ia-dofus"
        self.python_version = "3.11"

        # Détection du système conda
        self.conda_available = self._check_conda()

    def _check_conda(self) -> bool:
        """Vérifie la disponibilité de conda"""
        try:
            result = subprocess.run(["conda", "--version"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Conda détecté: {result.stdout.strip()}")
                return True
            else:
                logger.error("❌ Conda non accessible")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("❌ Conda non installé")
            return False

    def check_existing_environment(self) -> bool:
        """Vérifie si l'environnement existe déjà"""
        try:
            result = subprocess.run(["conda", "env", "list"],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                env_exists = self.env_name in result.stdout
                if env_exists:
                    logger.info(f"ℹ️ Environnement {self.env_name} existe déjà")
                return env_exists

            return False
        except Exception as e:
            logger.error(f"Erreur vérification environnement: {e}")
            return False

    def create_conda_environment(self) -> bool:
        """Crée l'environnement conda"""
        try:
            logger.info(f"🚀 Création environnement conda: {self.env_name}")

            # Suppression si existe (pour recréation propre)
            if self.check_existing_environment():
                logger.info("🗑️ Suppression environnement existant...")
                subprocess.run(["conda", "env", "remove", "-n", self.env_name, "-y"],
                             capture_output=True)

            # Création nouvel environnement
            cmd = [
                "conda", "create", "-n", self.env_name,
                f"python={self.python_version}", "-y"
            ]

            logger.info(f"Exécution: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Environnement conda créé avec succès")
                return True
            else:
                logger.error(f"❌ Erreur création environnement: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout création environnement (>5min)")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur création environnement: {e}")
            return False

    def install_pytorch_directml(self) -> bool:
        """Installe PyTorch + DirectML dans l'environnement"""
        try:
            logger.info("📦 Installation PyTorch + DirectML...")

            # Commandes d'installation dans l'environnement
            install_commands = [
                # PyTorch CPU d'abord
                [
                    "conda", "run", "-n", self.env_name,
                    "pip", "install", "torch", "torchvision", "torchaudio"
                ],
                # DirectML pour AMD
                [
                    "conda", "run", "-n", self.env_name,
                    "pip", "install", "torch-directml"
                ],
                # ONNX Runtime DirectML (backup)
                [
                    "conda", "run", "-n", self.env_name,
                    "pip", "install", "onnxruntime-directml"
                ]
            ]

            success_count = 0

            for i, cmd in enumerate(install_commands, 1):
                try:
                    logger.info(f"  {i}/3 Installation: {cmd[-1]}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                    if result.returncode == 0:
                        logger.info(f"  ✅ {cmd[-1]} installé")
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️ Échec {cmd[-1]}: {result.stderr}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"  ⚠️ Timeout installation {cmd[-1]}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Erreur {cmd[-1]}: {e}")

            # Au moins 2/3 requis pour succès
            if success_count >= 2:
                logger.info(f"✅ PyTorch/DirectML installé ({success_count}/3)")
                return True
            else:
                logger.error(f"❌ Installation insuffisante ({success_count}/3)")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur installation PyTorch: {e}")
            return False

    def install_ai_requirements(self) -> bool:
        """Installe les requirements IA spécifiques"""
        try:
            logger.info("📦 Installation requirements IA...")

            # Requirements de base pour IA
            base_packages = [
                "ultralytics>=8.0.0",
                "opencv-python>=4.8.0",
                "numpy>=1.24.0",
                "pillow>=10.0.0",
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "matplotlib>=3.7.0",
                "networkx>=3.1",
                "aiohttp>=3.8.0",
                "tqdm>=4.65.0",
                "pyyaml>=6.0",
                "psutil>=5.9.0"
            ]

            # Installation par batch pour éviter conflits
            batch_size = 4
            success_count = 0
            total_packages = len(base_packages)

            for i in range(0, len(base_packages), batch_size):
                batch = base_packages[i:i + batch_size]

                cmd = ["conda", "run", "-n", self.env_name, "pip", "install"] + batch

                try:
                    logger.info(f"  Installation batch {i//batch_size + 1}: {len(batch)} packages")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

                    if result.returncode == 0:
                        success_count += len(batch)
                        logger.info(f"  ✅ Batch {i//batch_size + 1} installé")
                    else:
                        logger.warning(f"  ⚠️ Erreurs batch {i//batch_size + 1}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"  ⚠️ Timeout batch {i//batch_size + 1}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Erreur batch {i//batch_size + 1}: {e}")

            success_rate = success_count / total_packages
            logger.info(f"📊 Requirements installés: {success_count}/{total_packages} ({success_rate:.1%})")

            return success_rate >= 0.8  # 80% minimum requis

        except Exception as e:
            logger.error(f"❌ Erreur installation requirements: {e}")
            return False

    def test_gpu_acceleration(self) -> bool:
        """Teste l'accélération GPU dans l'environnement"""
        try:
            logger.info("🧪 Test accélération GPU...")

            test_script = '''
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# Test DirectML
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"DirectML: {device}")

        # Test calcul simple
        x = torch.randn(500, 500, device=device)
        y = torch.randn(500, 500, device=device)
        z = torch.mm(x, y)
        print("GPU_TEST_SUCCESS")
    else:
        print("DirectML non disponible")
        print("GPU_TEST_PARTIAL")
except ImportError:
    print("torch-directml non installé")
    print("GPU_TEST_FAIL")
except Exception as e:
    print(f"Erreur DirectML: {e}")
    print("GPU_TEST_FAIL")
'''

            cmd = ["conda", "run", "-n", self.env_name, "python", "-c", test_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            output = result.stdout
            print("📋 Résultat test GPU:")
            print(output)

            if "GPU_TEST_SUCCESS" in output:
                logger.info("✅ Accélération GPU fonctionnelle")
                return True
            elif "GPU_TEST_PARTIAL" in output:
                logger.warning("⚠️ GPU partiellement fonctionnel")
                return True
            else:
                logger.warning("❌ GPU non fonctionnel - CPU fallback")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout test GPU")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur test GPU: {e}")
            return False

    def create_activation_scripts(self) -> bool:
        """Crée des scripts d'activation pratiques"""
        try:
            logger.info("📝 Création scripts d'activation...")

            # Script PowerShell
            ps_script = f"""# Script d'activation IA DOFUS
Write-Host "🚀 Activation environnement IA DOFUS..." -ForegroundColor Green
conda activate {self.env_name}

Write-Host "✅ Environnement activé !" -ForegroundColor Green
Write-Host "🎯 Commandes disponibles:" -ForegroundColor Yellow
Write-Host "  python launch_ai_dofus.py --mode demo" -ForegroundColor Cyan
Write-Host "  python scripts/test_amd_integration.py" -ForegroundColor Cyan
Write-Host "  python scripts/gemini_consensus.py autonomy_architecture" -ForegroundColor Cyan
"""

            ps_file = self.project_root / "activate_ia_dofus.ps1"
            with open(ps_file, 'w', encoding='utf-8') as f:
                f.write(ps_script)

            # Script Batch
            bat_script = f"""@echo off
echo 🚀 Activation environnement IA DOFUS...
call conda activate {self.env_name}
echo ✅ Environnement activé !
echo 🎯 Commandes disponibles:
echo   python launch_ai_dofus.py --mode demo
echo   python scripts/test_amd_integration.py
echo   python scripts/gemini_consensus.py autonomy_architecture
cmd /k
"""

            bat_file = self.project_root / "activate_ia_dofus.bat"
            with open(bat_file, 'w', encoding='utf-8') as f:
                f.write(bat_script)

            logger.info(f"✅ Scripts créés: {ps_file.name}, {bat_file.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur création scripts: {e}")
            return False

    def generate_environment_info(self) -> dict:
        """Génère les informations d'environnement"""
        try:
            # Test des modules installés
            test_imports = [
                "torch", "torch_directml", "cv2", "numpy",
                "ultralytics", "sklearn", "pandas"
            ]

            installed_modules = {}

            for module in test_imports:
                cmd = ["conda", "run", "-n", self.env_name, "python", "-c", f"import {module}; print({module}.__version__ if hasattr({module}, '__version__') else 'OK')"]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        installed_modules[module] = version
                    else:
                        installed_modules[module] = "ERROR"
                except:
                    installed_modules[module] = "NOT_FOUND"

            # Informations environnement
            env_info = {
                "environment_name": self.env_name,
                "python_version": self.python_version,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "platform": platform.platform(),
                "installed_modules": installed_modules
            }

            # Sauvegarde
            info_file = self.project_root / "config" / "environment_info.json"
            info_file.parent.mkdir(exist_ok=True)

            with open(info_file, 'w') as f:
                json.dump(env_info, f, indent=2)

            logger.info(f"✅ Informations environnement sauvegardées: {info_file}")
            return env_info

        except Exception as e:
            logger.error(f"❌ Erreur génération infos environnement: {e}")
            return {}

    def run_complete_setup(self):
        """Lance l'installation complète"""
        print("🚀 IA DOFUS - Création Environnement Automatisé")
        print("=" * 60)
        print(f"🐍 Conda: Détecté")
        print(f"🎯 Environnement: {self.env_name}")
        print(f"🐍 Python: {self.python_version}")
        print("=" * 60)

        if not self.conda_available:
            print("❌ Conda non disponible ! Installez Miniconda/Anaconda d'abord.")
            return False

        steps = [
            ("Création environnement conda", self.create_conda_environment),
            ("Installation PyTorch DirectML", self.install_pytorch_directml),
            ("Installation requirements IA", self.install_ai_requirements),
            ("Test accélération GPU", self.test_gpu_acceleration),
            ("Création scripts activation", self.create_activation_scripts),
            ("Génération infos environnement", lambda: bool(self.generate_environment_info()))
        ]

        results = {}

        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            print("-" * 40)

            try:
                start_time = time.perf_counter()
                result = step_func()
                duration = time.perf_counter() - start_time

                results[step_name] = result
                status = "✅ SUCCÈS" if result else "❌ ÉCHEC"
                print(f"{status} {step_name} ({duration:.1f}s)")

            except Exception as e:
                results[step_name] = False
                print(f"💥 {step_name} - ERREUR: {e}")

        # Rapport final
        print("\n" + "=" * 60)
        print("📊 RAPPORT FINAL")
        print("=" * 60)

        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)

        for step, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {step}")

        success_rate = success_count / total_count
        print(f"\n📈 Score: {success_count}/{total_count} ({success_rate:.1%})")

        if success_rate >= 0.8:
            print("\n🎉 ENVIRONNEMENT PRÊT !")
            print("\n🚀 ÉTAPES SUIVANTES:")
            print("1. Activez l'environnement:")
            print(f"   conda activate {self.env_name}")
            print("2. Lancez l'IA DOFUS:")
            print("   python launch_ai_dofus.py --mode demo")
            print("3. Ou utilisez les scripts d'activation:")
            print("   .\\activate_ia_dofus.ps1  (PowerShell)")
            print("   .\\activate_ia_dofus.bat  (Command Prompt)")

            return True
        else:
            print("\n⚠️ INSTALLATION INCOMPLÈTE")
            print("🔧 Vérifiez les erreurs ci-dessus et relancez si nécessaire")
            return False

def main():
    """Point d'entrée principal"""
    creator = AIEnvironmentCreator()
    success = creator.run_complete_setup()

    if not success:
        print("\n❌ Échec création environnement")
        sys.exit(1)

if __name__ == "__main__":
    main()