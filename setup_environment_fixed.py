#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Configuration encodage pour Windows
if platform.system() == "Windows":
    import locale
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class EnvironmentSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.venv_dir = self.root_dir / "venv_dofus_ai"
        self.python_exe = sys.executable
        self.os_type = platform.system()

        print("Configuration Assistant IA DOFUS Ultime")
        print(f"Repertoire: {self.root_dir}")
        print(f"Python: {self.python_exe} ({sys.version})")
        print(f"OS: {self.os_type}")
        print("-" * 60)

    def check_python_version(self):
        """Vérifier la version Python"""
        print("Verification version Python...")

        if sys.version_info < (3, 8):
            print("ERREUR: Python 3.8+ requis")
            return False
        elif sys.version_info >= (3, 13):
            print("SUCCES: Python 3.13+ détecté - Excellent!")
        else:
            print("SUCCES: Version Python compatible")

        return True

    def check_system_requirements(self):
        """Vérifier les prérequis système"""
        print("Verification prérequis système...")

        checks = {
            "Git": shutil.which("git"),
            "Visual Studio Build Tools": self._check_vs_tools(),
        }

        all_good = True
        for tool, available in checks.items():
            if available:
                print(f"SUCCES: {tool}: Disponible")
            else:
                print(f"ATTENTION: {tool}: Manquant")
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
        print("Creation environnement virtuel...")

        if self.venv_dir.exists():
            print("ATTENTION: Environnement existant détecté - suppression...")
            shutil.rmtree(self.venv_dir)

        try:
            subprocess.run([
                self.python_exe, "-m", "venv", str(self.venv_dir)
            ], check=True)
            print("SUCCES: Environnement virtuel créé")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERREUR: Création environnement: {e}")
            return False

    def get_venv_python(self):
        """Obtenir le chemin Python de l'environnement virtuel"""
        if self.os_type == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"

    def upgrade_pip(self):
        """Mettre à jour pip"""
        print("Mise à jour pip...")

        venv_python = self.get_venv_python()

        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "--upgrade", "pip", "setuptools", "wheel"
            ], check=True)
            print("SUCCES: pip mis à jour")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERREUR: Mise à jour pip: {e}")
            return False

    def install_basic_requirements(self):
        """Installer les dépendances de base"""
        print("Installation dépendances de base...")

        venv_python = self.get_venv_python()
        basic_requirements = self.root_dir / "requirements.txt"

        if basic_requirements.exists():
            try:
                subprocess.run([
                    str(venv_python), "-m", "pip", "install",
                    "-r", str(basic_requirements)
                ], check=True)
                print("SUCCES: Dépendances de base installées")
                return True
            except subprocess.CalledProcessError as e:
                print(f"ERREUR: Installation de base: {e}")
                return False
        else:
            print("ATTENTION: requirements.txt de base non trouvé - ignoré")
            return True

    def install_core_packages(self):
        """Installer les packages essentiels"""
        print("Installation packages essentiels...")

        venv_python = self.get_venv_python()

        # Packages critiques en premier
        core_packages = [
            "numpy==1.24.4",
            "opencv-python==4.8.1.78",
            "pillow==10.0.0",
            "pytesseract==0.3.10",
            "torch==2.0.1",
            "torchvision==0.15.2",
            "requests==2.31.0",
            "psutil==5.9.5",
            "pyyaml==6.0.1",
            "pywin32==306",
        ]

        for package in core_packages:
            try:
                print(f"Installation: {package}")
                subprocess.run([
                    str(venv_python), "-m", "pip", "install", package
                ], check=True, timeout=300)
                print(f"SUCCES: {package}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"ERREUR: {package} - {e}")
                return False

        return True

    def install_amd_gpu_support(self):
        """Installer le support GPU AMD"""
        print("Configuration support GPU AMD...")

        venv_python = self.get_venv_python()

        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "torch-directml"
            ], check=True, timeout=300)
            print("SUCCES: Support GPU AMD installé")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"ATTENTION: Support GPU AMD - {e}")
            return True  # Non-critique

    def create_project_structure(self):
        """Créer la structure de projet"""
        print("Création structure projet...")

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
            "tests",
            "scripts"
        ]

        for dir_path in dirs_to_create:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

            # Créer __init__.py pour les modules Python
            if dir_path.startswith("modules/"):
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Module Assistant IA DOFUS\n")

        print("SUCCES: Structure projet créée")
        return True

    def create_configuration_files(self):
        """Créer fichiers de configuration"""
        print("Création fichiers configuration...")

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
            }
        }

        config_dir = self.root_dir / "config"
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "main_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Script d'activation Windows
        if self.os_type == "Windows":
            activate_script = self.root_dir / "activate_env.bat"
            activate_script.write_text(f"""@echo off
echo Activation environnement Assistant IA DOFUS
call "{self.venv_dir}\\Scripts\\activate.bat"
echo Environnement active
echo Utilisez: python main.py --mode gui
cmd /k
""")

        print("SUCCES: Fichiers configuration créés")
        return True

    def verify_installation(self):
        """Vérifier l'installation"""
        print("Verification installation...")

        venv_python = self.get_venv_python()

        # Test imports critiques
        critical_modules = [
            "cv2",
            "numpy",
            "PIL",
            "yaml"
        ]

        all_good = True
        for module in critical_modules:
            try:
                result = subprocess.run([
                    str(venv_python), "-c", f"import {module}; print('OK: {module}')"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print(f"ERREUR: {module}")
                    all_good = False
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: {module}")
                all_good = False

        return all_good

    def run_setup(self):
        """Exécuter configuration complète"""
        print("Debut configuration Assistant IA DOFUS Ultime")
        print("=" * 60)

        steps = [
            ("Verification Python", self.check_python_version),
            ("Prerequis système", self.check_system_requirements),
            ("Environnement virtuel", self.create_virtual_environment),
            ("Mise à jour pip", self.upgrade_pip),
            ("Packages essentiels", self.install_core_packages),
            ("Support GPU AMD", self.install_amd_gpu_support),
            ("Structure projet", self.create_project_structure),
            ("Configuration", self.create_configuration_files),
            ("Verification", self.verify_installation),
        ]

        success_count = 0
        for step_name, step_func in steps:
            print(f"\nETAPE: {step_name}")
            try:
                if step_func():
                    success_count += 1
                    print(f"SUCCES: {step_name}")
                else:
                    print(f"ECHEC: {step_name}")
            except Exception as e:
                print(f"ERREUR: {step_name} - {e}")

        print(f"\nConfiguration terminée: {success_count}/{len(steps)} étapes réussies")

        if success_count >= len(steps) - 1:  # Allow 1 failure
            print("Installation réussie !")
            print(f"Activez l'environnement avec: {self.venv_dir}")
        else:
            print("Installation partiellement réussie")

        return success_count >= len(steps) - 1

if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run_setup()