#!/usr/bin/env python3
"""
Setup Environment - Assistant IA DOFUS Ultime
Automatise la création de l'environnement de développement complet
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import urllib.request
import zipfile
import json

class EnvironmentSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.venv_dir = self.root_dir / "venv_dofus_ai"
        self.python_exe = sys.executable
        self.os_type = platform.system()

        print(f"🚀 Configuration Assistant IA DOFUS Ultime")
        print(f"📁 Répertoire: {self.root_dir}")
        print(f"🐍 Python: {self.python_exe} ({sys.version})")
        print(f"💻 OS: {self.os_type}")
        print("-" * 60)

    def check_python_version(self):
        """Vérifier la version Python"""
        print("🔍 Vérification version Python...")

        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ requis")
            return False
        elif sys.version_info >= (3, 13):
            print("✅ Python 3.13+ détecté - Excellent!")
        else:
            print("✅ Version Python compatible")

        return True

    def check_system_requirements(self):
        """Vérifier les prérequis système"""
        print("🔍 Vérification prérequis système...")

        checks = {
            "Git": shutil.which("git"),
            "Visual Studio Build Tools": self._check_vs_tools(),
        }

        all_good = True
        for tool, available in checks.items():
            if available:
                print(f"✅ {tool}: Disponible")
            else:
                print(f"❌ {tool}: Manquant")
                all_good = False

        return all_good

    def _check_vs_tools(self):
        """Vérifier Visual Studio Build Tools"""
        if self.os_type != "Windows":
            return True

        vs_paths = [
            "C:/Program Files/Microsoft Visual Studio",
            "C:/Program Files (x86)/Microsoft Visual Studio",
            "C:/BuildTools"
        ]

        return any(Path(path).exists() for path in vs_paths)

    def create_virtual_environment(self):
        """Créer l'environnement virtuel"""
        print("📦 Création environnement virtuel...")

        if self.venv_dir.exists():
            print("⚠️  Environnement existant détecté")
            response = input("Voulez-vous le recréer? (o/N): ")
            if response.lower() == 'o':
                print("🗑️  Suppression ancien environnement...")
                shutil.rmtree(self.venv_dir)
            else:
                print("📦 Utilisation environnement existant")
                return True

        try:
            subprocess.run([
                self.python_exe, "-m", "venv", str(self.venv_dir)
            ], check=True)
            print("✅ Environnement virtuel créé")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur création environnement: {e}")
            return False

    def get_venv_python(self):
        """Obtenir le chemin Python de l'environnement virtuel"""
        if self.os_type == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"

    def upgrade_pip(self):
        """Mettre à jour pip"""
        print("📈 Mise à jour pip...")

        venv_python = self.get_venv_python()

        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "--upgrade", "pip", "setuptools", "wheel"
            ], check=True)
            print("✅ pip mis à jour")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur mise à jour pip: {e}")
            return False

    def install_basic_requirements(self):
        """Installer les dépendances de base"""
        print("📚 Installation dépendances de base...")

        venv_python = self.get_venv_python()
        basic_requirements = self.root_dir / "requirements.txt"

        if basic_requirements.exists():
            try:
                subprocess.run([
                    str(venv_python), "-m", "pip", "install",
                    "-r", str(basic_requirements)
                ], check=True)
                print("✅ Dépendances de base installées")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur installation de base: {e}")
                return False
        else:
            print("⚠️  requirements.txt de base non trouvé")
            return True

    def install_advanced_requirements(self):
        """Installer les dépendances avancées"""
        print("🧠 Installation dépendances IA avancées...")

        venv_python = self.get_venv_python()
        advanced_requirements = self.root_dir / "requirements_advanced.txt"

        if not advanced_requirements.exists():
            print("❌ requirements_advanced.txt non trouvé")
            return False

        try:
            # Installation avec timeout étendu pour les gros packages
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "-r", str(advanced_requirements),
                "--timeout", "300",
                "--retries", "3"
            ], check=True, timeout=1800)  # 30 minutes max
            print("✅ Dépendances IA avancées installées")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation avancée: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Timeout installation (packages trop volumineux)")
            return False

    def install_amd_gpu_support(self):
        """Installer le support GPU AMD"""
        print("🎮 Configuration support GPU AMD...")

        venv_python = self.get_venv_python()

        # Vérification si GPU AMD détecté
        if not self._detect_amd_gpu():
            print("⚠️  GPU AMD non détecté - installation optionnelle")
            response = input("Installer quand même le support AMD? (o/N): ")
            if response.lower() != 'o':
                return True

        # Installation torch-directml pour AMD
        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "torch-directml", "--force-reinstall"
            ], check=True)
            print("✅ Support GPU AMD installé")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation GPU AMD: {e}")
            return False

    def _detect_amd_gpu(self):
        """Détecter GPU AMD"""
        try:
            result = subprocess.run([
                "wmic", "path", "win32_VideoController", "get", "name"
            ], capture_output=True, text=True, timeout=10)

            output = result.stdout.lower()
            return "amd" in output or "radeon" in output
        except:
            return False

    def download_external_tools(self):
        """Télécharger outils externes"""
        print("📥 Téléchargement outils externes...")

        tools_dir = self.root_dir / "tools"
        tools_dir.mkdir(exist_ok=True)

        # Tesseract OCR (Windows)
        if self.os_type == "Windows":
            if not self._check_tesseract_installed():
                print("⚠️  Tesseract OCR non installé")
                print("📝 Téléchargez depuis: https://github.com/UB-Mannheim/tesseract/wiki")
                print("   Ou utilisez: winget install UB-Mannheim.TesseractOCR")

        return True

    def _check_tesseract_installed(self):
        """Vérifier installation Tesseract"""
        return shutil.which("tesseract") is not None

    def create_project_structure(self):
        """Créer la structure de projet"""
        print("📁 Création structure projet...")

        dirs_to_create = [
            "modules/vision",
            "modules/ai_core",
            "modules/overlay",
            "modules/external",
            "modules/learning",
            "modules/advisor",
            "data/models",
            "data/cache",
            "data/logs",
            "data/knowledge",
            "config",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "scripts",
            "docs"
        ]

        for dir_path in dirs_to_create:
            (self.root_dir / dir_path).mkdir(parents=True, exist_ok=True)
            # Créer __init__.py pour les modules Python
            if dir_path.startswith("modules/"):
                init_file = self.root_dir / dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Module Assistant IA DOFUS\\n")

        print("✅ Structure projet créée")
        return True

    def create_configuration_files(self):
        """Créer fichiers de configuration"""
        print("⚙️  Création fichiers configuration...")

        # Configuration principale
        config = {
            "version": "2.0.0",
            "environment": "development",
            "gpu": {
                "enable_amd": True,
                "memory_fraction": 0.8,
                "mixed_precision": True
            },
            "vision": {
                "capture_fps": 60,
                "ocr_language": "fra+eng",
                "detection_threshold": 0.7
            },
            "ai": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_memory_mb": 4096
            },
            "overlay": {
                "enable": True,
                "transparency": 0.8,
                "position": "top_left"
            },
            "external_tools": {
                "dofus_guide_path": "",
                "ganymede_path": "",
                "dofus_unity_path": ""
            }
        }

        config_file = self.root_dir / "config" / "main_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Script d'activation
        if self.os_type == "Windows":
            activate_script = self.root_dir / "activate_env.bat"
            activate_script.write_text(f"""@echo off
echo 🚀 Activation environnement Assistant IA DOFUS
call "{self.venv_dir}\\Scripts\\activate.bat"
echo ✅ Environnement activé
echo 💡 Utilisez: python main.py --mode gui
cmd /k
""")

        print("✅ Fichiers configuration créés")
        return True

    def verify_installation(self):
        """Vérifier l'installation"""
        print("🔍 Vérification installation...")

        venv_python = self.get_venv_python()

        # Test imports critiques
        critical_modules = [
            "cv2",
            "torch",
            "numpy",
            "PIL",
            "tkinter"
        ]

        all_good = True
        for module in critical_modules:
            try:
                result = subprocess.run([
                    str(venv_python), "-c", f"import {module}; print('✅ {module}')"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print(f"❌ {module}: {result.stderr.strip()}")
                    all_good = False
            except subprocess.TimeoutExpired:
                print(f"❌ {module}: Timeout")
                all_good = False

        return all_good

    def run_setup(self):
        """Exécuter configuration complète"""
        print("🎯 Début configuration Assistant IA DOFUS Ultime")
        print("=" * 60)

        steps = [
            ("Vérification Python", self.check_python_version),
            ("Prérequis système", self.check_system_requirements),
            ("Environnement virtuel", self.create_virtual_environment),
            ("Mise à jour pip", self.upgrade_pip),
            ("Dépendances de base", self.install_basic_requirements),
            ("Dépendances IA", self.install_advanced_requirements),
            ("Support GPU AMD", self.install_amd_gpu_support),
            ("Outils externes", self.download_external_tools),
            ("Structure projet", self.create_project_structure),
            ("Configuration", self.create_configuration_files),
            ("Vérification", self.verify_installation),
        ]

        success_count = 0
        for step_name, step_func in steps:
            print(f"\n📋 Étape: {step_name}")
            try:
                if step_func():
                    success_count += 1
                    print(f"✅ {step_name} - Succès")
                else:
                    print(f"❌ {step_name} - Échec")
            except Exception as e:
                print(f"💥 {step_name} - Erreur: {e}")

        print(f"\n🏁 Configuration terminée: {success_count}/{len(steps)} étapes réussies")

        if success_count == len(steps):
            print("🎉 Installation complète réussie !")
            print(f"💡 Activez l'environnement: {self.venv_dir}")
            print("🚀 Lancez: python main.py --mode gui")
        else:
            print("⚠️  Installation partiellement réussie")
            print("🔧 Vérifiez les erreurs ci-dessus")

        return success_count == len(steps)

if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run_setup()