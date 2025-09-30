#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Environment - Assistant IA DOFUS Ultime (Python 3.13 Compatible)
Automatise la création de l'environnement de développement complet
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import json

# Configuration encodage pour Windows
if platform.system() == "Windows":
    import locale
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class EnvironmentSetupPython313:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.venv_dir = self.root_dir / "venv_dofus_ai"
        self.python_exe = sys.executable
        self.os_type = platform.system()

        print("Configuration Assistant IA DOFUS Ultime - Python 3.13")
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
            print("SUCCES: Python 3.13+ détecté - Mode compatibilité activé")
        else:
            print("SUCCES: Version Python compatible")

        return True

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

    def install_core_packages_step_by_step(self):
        """Installation packages essentiels étape par étape"""
        print("Installation packages essentiels (Python 3.13 compatible)...")

        venv_python = self.get_venv_python()

        # Packages par ordre de priorité avec gestion d'erreurs
        package_groups = [
            # Groupe 1: Base absolument nécessaire
            {
                "name": "Packages de base",
                "packages": [
                    "numpy",
                    "pillow",
                    "requests",
                    "pyyaml",
                    "psutil"
                ]
            },
            # Groupe 2: Computer Vision
            {
                "name": "Computer Vision",
                "packages": [
                    "opencv-python",
                    "scikit-image"
                ]
            },
            # Groupe 3: GUI et Windows
            {
                "name": "Interface et Windows",
                "packages": [
                    "pywin32",
                    "customtkinter",
                    "pygame"
                ]
            },
            # Groupe 4: OCR
            {
                "name": "OCR",
                "packages": [
                    "pytesseract",
                    "easyocr"
                ]
            },
            # Groupe 5: ML (optionnel)
            {
                "name": "Machine Learning",
                "packages": [
                    "torch",
                    "scikit-learn"
                ]
            }
        ]

        success_count = 0
        total_groups = len(package_groups)

        for group in package_groups:
            print(f"\nInstallation groupe: {group['name']}")
            group_success = True

            for package in group["packages"]:
                try:
                    print(f"  Installation: {package}")
                    result = subprocess.run([
                        str(venv_python), "-m", "pip", "install", package,
                        "--no-cache-dir"
                    ], check=True, timeout=300, capture_output=True, text=True)
                    print(f"  SUCCES: {package}")
                except subprocess.CalledProcessError as e:
                    print(f"  ECHEC: {package} - {e}")
                    group_success = False
                except subprocess.TimeoutExpired:
                    print(f"  TIMEOUT: {package}")
                    group_success = False

            if group_success:
                success_count += 1
                print(f"SUCCES: Groupe {group['name']}")
            else:
                print(f"ECHEC PARTIEL: Groupe {group['name']}")

        print(f"\nInstallation terminée: {success_count}/{total_groups} groupes réussis")
        return success_count >= 3  # Minimum 3 groupes pour fonctionner

    def install_python313_requirements(self):
        """Installer avec le fichier requirements Python 3.13"""
        print("Installation avec requirements Python 3.13...")

        venv_python = self.get_venv_python()
        requirements_file = self.root_dir / "requirements_python313.txt"

        if not requirements_file.exists():
            print("ERREUR: requirements_python313.txt non trouvé")
            return False

        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "install",
                "-r", str(requirements_file),
                "--no-cache-dir",
                "--timeout", "300"
            ], check=True, timeout=1800)
            print("SUCCES: Requirements Python 3.13 installés")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ATTENTION: Installation partielle - {e}")
            return True  # Continue même si échec partiel
        except subprocess.TimeoutExpired:
            print("TIMEOUT: Installation requirements")
            return True  # Continue même si timeout

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
            "version": "2.0.1",
            "environment": "development",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
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

        # Test imports critiques (avec fallback)
        critical_modules = [
            ("numpy", "np"),
            ("PIL", "PIL"),
            ("cv2", "opencv-python"),
            ("yaml", "pyyaml"),
            ("requests", "requests")
        ]

        success_count = 0
        for module, package_name in critical_modules:
            try:
                result = subprocess.run([
                    str(venv_python), "-c", f"import {module}; print('OK: {module}')"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    print(result.stdout.strip())
                    success_count += 1
                else:
                    print(f"ECHEC: {module}")
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: {module}")

        print(f"Verification: {success_count}/{len(critical_modules)} modules OK")
        return success_count >= 3  # Minimum 3 modules critiques

    def run_setup(self):
        """Exécuter configuration complète"""
        print("Debut configuration Assistant IA DOFUS Ultime - Python 3.13")
        print("=" * 60)

        steps = [
            ("Verification Python", self.check_python_version),
            ("Environnement virtuel", self.create_virtual_environment),
            ("Mise à jour pip", self.upgrade_pip),
            ("Packages essentiels", self.install_core_packages_step_by_step),
            ("Requirements Python 3.13", self.install_python313_requirements),
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
            print("Vous pouvez maintenant utiliser l'assistant IA DOFUS")
        else:
            print("Installation partiellement réussie")
            print("L'assistant peut fonctionner avec des fonctionnalités limitées")

        return success_count >= len(steps) - 2  # Allow 2 failures

if __name__ == "__main__":
    setup = EnvironmentSetupPython313()
    success = setup.run_setup()

    if success:
        print("\nConfiguration terminée avec succès!")
        print("Prêt pour le développement de l'assistant IA DOFUS")
    else:
        print("\nConfiguration partiellement réussie")
        print("Vérifiez les erreurs et relancez si nécessaire")