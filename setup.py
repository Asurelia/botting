"""
Script d'installation et configuration automatisée
=================================================

Ce script automatise l'installation complète du système de bot DOFUS,
incluant l'installation des dépendances, la configuration de la base
de données et la validation de l'environnement.

Usage:
    python setup.py [options]

Options:
    --quick         Installation rapide (dépendances core uniquement)
    --full          Installation complète avec toutes les dépendances
    --dev           Installation avec outils de développement
    --reinstall     Réinstallation complète
    --validate      Validation de l'installation existante
    --config        Configuration interactive
    
Créé le: 2025-08-31
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile
import tempfile


class DofusBotSetup:
    """Gestionnaire d'installation du bot DOFUS"""
    
    def __init__(self):
        """Initialise le gestionnaire d'installation"""
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        self.platform = platform.system().lower()
        self.architecture = platform.architecture()[0]
        
        # Couleurs pour l'affichage console
        self.colors = {
            'GREEN': '\033[92m',
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'PURPLE': '\033[95m',
            'CYAN': '\033[96m',
            'RESET': '\033[0m',
            'BOLD': '\033[1m'
        }
        
        # Désactiver les couleurs sur Windows (sauf si terminal moderne)
        if self.platform == 'windows' and not os.environ.get('FORCE_COLOR'):
            self.colors = {key: '' for key in self.colors}
    
    def print_colored(self, message: str, color: str = 'RESET', bold: bool = False):
        """Affiche un message coloré"""
        style = self.colors.get(color, '') + (self.colors.get('BOLD', '') if bold else '')
        reset = self.colors.get('RESET', '')
        print(f"{style}{message}{reset}")
    
    def print_header(self, title: str):
        """Affiche un en-tête de section"""
        self.print_colored("=" * 60, 'BLUE', bold=True)
        self.print_colored(f"  {title}", 'BLUE', bold=True)
        self.print_colored("=" * 60, 'BLUE', bold=True)
    
    def print_success(self, message: str):
        """Affiche un message de succès"""
        self.print_colored(f"✅ {message}", 'GREEN')
    
    def print_warning(self, message: str):
        """Affiche un message d'avertissement"""
        self.print_colored(f"⚠️  {message}", 'YELLOW')
    
    def print_error(self, message: str):
        """Affiche un message d'erreur"""
        self.print_colored(f"❌ {message}", 'RED')
    
    def print_info(self, message: str):
        """Affiche un message d'information"""
        self.print_colored(f"ℹ️  {message}", 'CYAN')
    
    def check_python_version(self) -> bool:
        """Vérifie la version de Python"""
        self.print_info("Vérification de la version Python...")
        
        if self.python_version < (3, 8):
            self.print_error(f"Python 3.8+ requis, version détectée: {sys.version}")
            self.print_info("Veuillez mettre à jour Python: https://www.python.org/downloads/")
            return False
        
        if self.python_version >= (3, 11):
            self.print_success(f"Python {sys.version.split()[0]} - Version optimale")
        elif self.python_version >= (3, 9):
            self.print_success(f"Python {sys.version.split()[0]} - Version compatible")
        else:
            self.print_warning(f"Python {sys.version.split()[0]} - Fonctionne mais 3.11+ recommandé")
        
        return True
    
    def check_pip(self) -> bool:
        """Vérifie et met à jour pip"""
        self.print_info("Vérification de pip...")
        
        try:
            import pip
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.print_success("pip détecté")
                
                # Mettre à jour pip
                self.print_info("Mise à jour de pip...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                             check=False)
                return True
            else:
                self.print_error("pip non fonctionnel")
                return False
        except ImportError:
            self.print_error("pip non installé")
            self.print_info("Installation de pip...")
            return self._install_pip()
    
    def _install_pip(self) -> bool:
        """Installe pip si manquant"""
        try:
            # Télécharger get-pip.py
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp:
                urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', tmp.name)
                result = subprocess.run([sys.executable, tmp.name], capture_output=True)
                os.unlink(tmp.name)
                
                if result.returncode == 0:
                    self.print_success("pip installé avec succès")
                    return True
                else:
                    self.print_error("Échec de l'installation de pip")
                    return False
        except Exception as e:
            self.print_error(f"Erreur lors de l'installation de pip: {e}")
            return False
    
    def install_dependencies(self, mode: str = 'full') -> bool:
        """
        Installe les dépendances Python
        
        Args:
            mode: 'quick', 'full', ou 'dev'
        """
        self.print_header("INSTALLATION DES DÉPENDANCES")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.print_error("Fichier requirements.txt non trouvé")
            return False
        
        # Définir les dépendances selon le mode
        if mode == 'quick':
            deps = self._get_core_dependencies()
            self.print_info("Installation rapide - Dépendances core uniquement")
        elif mode == 'dev':
            deps = ['pip install -r requirements.txt', 'pip install pytest black flake8 mypy']
            self.print_info("Installation développement - Toutes dépendances + outils dev")
        else:  # full
            deps = ['pip install -r requirements.txt']
            self.print_info("Installation complète - Toutes les dépendances")
        
        success = True
        for dep_command in deps:
            if isinstance(dep_command, str):
                cmd = dep_command.split()
            else:
                cmd = [sys.executable, '-m', 'pip', 'install'] + dep_command
            
            self.print_info(f"Installation: {' '.join(cmd[3:]) if len(cmd) > 3 else dep_command}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                if result.returncode == 0:
                    self.print_success("✓ Installé")
                else:
                    self.print_error(f"Erreur: {result.stderr}")
                    success = False
            except Exception as e:
                self.print_error(f"Erreur d'installation: {e}")
                success = False
        
        return success
    
    def _get_core_dependencies(self) -> List[List[str]]:
        """Retourne la liste des dépendances core"""
        return [
            ['pyyaml>=6.0.1'],
            ['opencv-python>=4.8.0'],
            ['pillow>=10.0.0'],
            ['mss>=9.0.1'],
            ['pytesseract>=0.3.10'],
            ['numpy>=1.24.0'],
            ['pyautogui>=0.9.54'],
            ['requests>=2.31.0']
        ]
    
    def check_system_dependencies(self) -> bool:
        """Vérifie les dépendances système"""
        self.print_header("VÉRIFICATION DÉPENDANCES SYSTÈME")
        
        success = True
        
        # Vérifier Tesseract OCR
        if not self._check_tesseract():
            success = False
            
        # Vérifier les outils spécifiques à la plateforme
        if self.platform == 'windows':
            success &= self._check_windows_deps()
        elif self.platform == 'linux':
            success &= self._check_linux_deps()
        elif self.platform == 'darwin':  # macOS
            success &= self._check_macos_deps()
        
        return success
    
    def _check_tesseract(self) -> bool:
        """Vérifie l'installation de Tesseract OCR"""
        self.print_info("Vérification de Tesseract OCR...")
        
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                self.print_success(f"Tesseract détecté: {version}")
                return True
        except FileNotFoundError:
            pass
        
        self.print_warning("Tesseract OCR non trouvé")
        self._provide_tesseract_instructions()
        return False
    
    def _provide_tesseract_instructions(self):
        """Fournit les instructions d'installation de Tesseract"""
        if self.platform == 'windows':
            self.print_info("Installation Windows:")
            self.print_info("1. Téléchargez: https://github.com/UB-Mannheim/tesseract/wiki")
            self.print_info("2. Installez et ajoutez au PATH")
            self.print_info("3. Ou utilisez: winget install UB-Mannheim.TesseractOCR")
        elif self.platform == 'linux':
            self.print_info("Installation Linux:")
            self.print_info("sudo apt-get update")
            self.print_info("sudo apt-get install tesseract-ocr tesseract-ocr-fra")
        elif self.platform == 'darwin':
            self.print_info("Installation macOS:")
            self.print_info("brew install tesseract tesseract-lang")
    
    def _check_windows_deps(self) -> bool:
        """Vérifie les dépendances Windows"""
        success = True
        
        # Vérifier Visual C++ Redistributable
        self.print_info("Vérification Visual C++ Redistributable...")
        # Tentative de vérification basique
        vc_paths = [
            r"C:\Program Files\Microsoft Visual Studio",
            r"C:\Program Files (x86)\Microsoft Visual Studio"
        ]
        
        vc_found = any(Path(path).exists() for path in vc_paths)
        if vc_found:
            self.print_success("Visual C++ Redistributable probablement installé")
        else:
            self.print_warning("Visual C++ Redistributable peut être manquant")
            self.print_info("Téléchargez: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        
        return success
    
    def _check_linux_deps(self) -> bool:
        """Vérifie les dépendances Linux"""
        self.print_info("Vérification des dépendances Linux...")
        
        # Vérifier les packages système courants
        packages = ['python3-dev', 'python3-tk', 'libopencv-dev']
        missing = []
        
        for package in packages:
            try:
                result = subprocess.run(['dpkg', '-s', package], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(package)
            except FileNotFoundError:
                # dpkg non disponible, probablement pas Debian/Ubuntu
                break
        
        if missing:
            self.print_warning(f"Packages manquants: {', '.join(missing)}")
            self.print_info(f"sudo apt-get install {' '.join(missing)}")
        else:
            self.print_success("Dépendances Linux vérifiées")
        
        return True
    
    def _check_macos_deps(self) -> bool:
        """Vérifie les dépendances macOS"""
        self.print_info("Vérification des dépendances macOS...")
        
        # Vérifier Homebrew
        try:
            result = subprocess.run(['brew', '--version'], capture_output=True)
            if result.returncode == 0:
                self.print_success("Homebrew détecté")
            else:
                self.print_warning("Homebrew recommandé pour l'installation")
                self.print_info("Installation: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        except FileNotFoundError:
            self.print_warning("Homebrew non détecté")
        
        return True
    
    def setup_directories(self) -> bool:
        """Crée la structure de répertoires"""
        self.print_header("CONFIGURATION RÉPERTOIRES")
        
        directories = [
            'data/databases',
            'data/backups',
            'logs',
            'screenshots',
            'temp',
            'config/user_configs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_success(f"Répertoire créé/vérifié: {directory}")
            except Exception as e:
                self.print_error(f"Erreur création {directory}: {e}")
                return False
        
        return True
    
    def initialize_database(self) -> bool:
        """Initialise la base de données"""
        self.print_header("INITIALISATION BASE DE DONNÉES")
        
        try:
            # Import et exécution du setup de base de données
            config_path = self.project_root / 'config'
            sys.path.insert(0, str(config_path))
            
            from database_setup import init_database
            
            if init_database():
                self.print_success("Base de données initialisée")
                
                # Initialiser les données DOFUS
                self.print_info("Chargement des données DOFUS...")
                
                data_path = self.project_root / 'data'
                sys.path.insert(0, str(data_path))
                
                from init_database import initialize_dofus_data
                
                if initialize_dofus_data():
                    self.print_success("Données DOFUS chargées")
                    return True
                else:
                    self.print_error("Erreur chargement données DOFUS")
                    return False
            else:
                self.print_error("Erreur initialisation base de données")
                return False
                
        except Exception as e:
            self.print_error(f"Erreur base de données: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Crée les fichiers de configuration par défaut"""
        self.print_header("CONFIGURATION PAR DÉFAUT")
        
        # Créer le fichier .env
        env_file = self.project_root / '.env'
        if not env_file.exists():
            env_content = """# Configuration du bot DOFUS
# ============================

# Paramètres de connexion (À MODIFIER)
DOFUS_USERNAME=votre_nom_utilisateur
DOFUS_PASSWORD=votre_mot_de_passe

# Paramètres de sécurité
DATABASE_ENCRYPTION_KEY=changez_cette_cle_de_chiffrement

# Webhooks (optionnel)
WEBHOOK_URL=
EMAIL_SMTP_SERVER=
EMAIL_USERNAME=
EMAIL_PASSWORD=

# Debug et développement
DEBUG_MODE=false
LOG_LEVEL=INFO

# Sauvegarde automatique
AUTO_BACKUP=true
BACKUP_INTERVAL=3600

# Configuration réseau (optionnel)
PROXY_HOST=
PROXY_PORT=
PROXY_USERNAME=
PROXY_PASSWORD=
"""
            env_file.write_text(env_content, encoding='utf-8')
            self.print_success("Fichier .env créé")
            self.print_warning("⚠️ IMPORTANT: Modifiez .env avec vos paramètres!")
        else:
            self.print_info("Fichier .env existant conservé")
        
        # Créer un exemple de configuration utilisateur
        user_config = self.project_root / 'config' / 'user_configs' / 'example_user.yaml'
        if not user_config.exists():
            user_content = """# Configuration utilisateur exemple
# ==================================

character:
  name: "MonPersonnage"
  class: "Iop"
  server: "Dofus"
  level_target: 200

automation:
  daily_routine: true
  farming_enabled: true
  combat_enabled: true
  
professions:
  enabled:
    - farmer
    - miner
  
combat:
  retreat_threshold: 30
  use_potions: true

safety:
  break_duration: 1800  # 30 minutes
  session_max: 14400    # 4 heures
"""
            user_config.parent.mkdir(parents=True, exist_ok=True)
            user_config.write_text(user_content, encoding='utf-8')
            self.print_success("Configuration exemple créée")
        
        return True
    
    def validate_installation(self) -> bool:
        """Valide l'installation complète"""
        self.print_header("VALIDATION INSTALLATION")
        
        tests = [
            ("Python version", self.check_python_version),
            ("Dépendances Python", self._test_python_imports),
            ("Base de données", self._test_database),
            ("Configuration", self._test_configuration),
            ("Vision système", self._test_vision_system),
        ]
        
        success_count = 0
        for test_name, test_func in tests:
            self.print_info(f"Test: {test_name}...")
            try:
                if test_func():
                    self.print_success(f"✓ {test_name}")
                    success_count += 1
                else:
                    self.print_error(f"✗ {test_name}")
            except Exception as e:
                self.print_error(f"✗ {test_name} - Erreur: {e}")
        
        if success_count == len(tests):
            self.print_colored("\n🎉 Installation validée avec succès!", 'GREEN', bold=True)
            self.print_info("Le bot est prêt à être utilisé!")
            return True
        else:
            self.print_error(f"\n❌ {len(tests) - success_count} tests échoués")
            return False
    
    def _test_python_imports(self) -> bool:
        """Teste l'import des modules principaux"""
        required_modules = [
            'yaml', 'cv2', 'PIL', 'mss', 'pytesseract', 
            'numpy', 'pyautogui', 'requests'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                self.print_error(f"Module manquant: {module} - {e}")
                return False
        
        return True
    
    def _test_database(self) -> bool:
        """Teste la base de données"""
        db_path = self.project_root / 'data' / 'databases' / 'dofus_bot.db'
        if not db_path.exists():
            return False
        
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                return table_count > 10  # Vérifier qu'il y a des tables
        except Exception:
            return False
    
    def _test_configuration(self) -> bool:
        """Teste la configuration"""
        config_path = self.project_root / 'config' / 'bot_config.yaml'
        if not config_path.exists():
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return 'app' in config and 'database' in config
        except Exception:
            return False
    
    def _test_vision_system(self) -> bool:
        """Teste le système de vision"""
        try:
            import cv2
            import mss
            import pytesseract
            
            # Test de capture d'écran
            with mss.mss() as sct:
                screenshot = sct.grab({"top": 0, "left": 0, "width": 100, "height": 100})
                
            # Test basique d'OpenCV
            import numpy as np
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return True
        except Exception:
            return False
    
    def run_interactive_config(self) -> bool:
        """Lance la configuration interactive"""
        self.print_header("CONFIGURATION INTERACTIVE")
        
        try:
            # Demander les informations de base
            self.print_info("Configuration de votre personnage principal:")
            
            char_name = input("Nom du personnage: ").strip()
            if not char_name:
                self.print_warning("Configuration annulée")
                return False
            
            print("\nClasses disponibles:")
            classes = ["Feca", "Osamodas", "Enutrof", "Sram", "Xelor", "Ecaflip", 
                      "Eniripsa", "Iop", "Cra", "Sadida", "Sacrieur", "Pandawa",
                      "Roublard", "Zobal", "Steamer", "Eliotrope", "Huppermage", "Ouginak"]
            
            for i, cls in enumerate(classes, 1):
                print(f"{i:2}. {cls}")
            
            try:
                class_choice = int(input("\nChoisissez votre classe (numéro): ")) - 1
                char_class = classes[class_choice]
            except (ValueError, IndexError):
                self.print_error("Choix invalide")
                return False
            
            server = input("Nom du serveur: ").strip() or "Dofus"
            
            # Créer la configuration utilisateur
            user_config = {
                'character': {
                    'name': char_name,
                    'class': char_class,
                    'server': server,
                    'level_target': 200
                },
                'automation': {
                    'daily_routine': True,
                    'farming_enabled': True,
                    'combat_enabled': True
                },
                'professions': {
                    'enabled': ['farmer', 'miner', 'lumberjack']
                },
                'safety': {
                    'break_duration': 1800,
                    'session_max': 14400
                }
            }
            
            # Sauvegarder la configuration
            config_path = self.project_root / 'config' / 'user_configs' / f'{char_name.lower()}.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(user_config, f, default_flow_style=False, allow_unicode=True)
            
            self.print_success(f"Configuration sauvegardée: {config_path}")
            
            # Rappel de sécurité
            self.print_colored("\n⚠️ RAPPELS DE SÉCURITÉ ⚠️", 'YELLOW', bold=True)
            self.print_warning("1. Modifiez le fichier .env avec vos vrais identifiants")
            self.print_warning("2. Utilisez le bot de manière responsable")
            self.print_warning("3. Respectez les ToS du jeu")
            self.print_warning("4. Surveillez les sessions pour éviter la détection")
            
            return True
            
        except KeyboardInterrupt:
            self.print_info("\nConfiguration annulée par l'utilisateur")
            return False
        except Exception as e:
            self.print_error(f"Erreur configuration: {e}")
            return False


def main():
    """Fonction principale d'installation"""
    parser = argparse.ArgumentParser(
        description="Installation automatisée du bot DOFUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python setup.py --full                 Installation complète
  python setup.py --quick                Installation rapide (core seulement)
  python setup.py --dev                  Installation développement
  python setup.py --validate             Validation de l'installation
  python setup.py --config               Configuration interactive
  python setup.py --reinstall --full     Réinstallation complète
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Installation rapide (dépendances core uniquement)')
    parser.add_argument('--full', action='store_true', 
                       help='Installation complète (par défaut)')
    parser.add_argument('--dev', action='store_true', 
                       help='Installation avec outils de développement')
    parser.add_argument('--reinstall', action='store_true', 
                       help='Réinstallation complète')
    parser.add_argument('--validate', action='store_true', 
                       help='Validation de l\'installation existante')
    parser.add_argument('--config', action='store_true', 
                       help='Configuration interactive')
    parser.add_argument('--no-color', action='store_true', 
                       help='Désactiver les couleurs')
    
    args = parser.parse_args()
    
    # Initialiser le setup
    setup = DofusBotSetup()
    
    if args.no_color:
        setup.colors = {key: '' for key in setup.colors}
    
    # Affichage d'en-tête
    setup.print_colored("🤖 DOFUS Bot Setup - Installation Automatisée", 'PURPLE', bold=True)
    setup.print_colored(f"Version: 1.0.0 | Python: {sys.version.split()[0]} | OS: {platform.system()}", 'CYAN')
    print()
    
    # Mode validation uniquement
    if args.validate:
        success = setup.validate_installation()
        sys.exit(0 if success else 1)
    
    # Mode configuration uniquement
    if args.config:
        success = setup.run_interactive_config()
        sys.exit(0 if success else 1)
    
    # Déterminer le mode d'installation
    if args.dev:
        install_mode = 'dev'
    elif args.quick:
        install_mode = 'quick'
    else:
        install_mode = 'full'
    
    setup.print_info(f"Mode d'installation: {install_mode}")
    
    if args.reinstall:
        setup.print_warning("Réinstallation demandée - suppression des données existantes")
        # Note: ici on pourrait ajouter la logique de nettoyage
    
    # Processus d'installation
    steps = [
        ("Vérification Python", setup.check_python_version),
        ("Vérification pip", setup.check_pip),
        ("Dépendances système", setup.check_system_dependencies),
        ("Installation dépendances", lambda: setup.install_dependencies(install_mode)),
        ("Structure répertoires", setup.setup_directories),
        ("Base de données", setup.initialize_database),
        ("Configuration", setup.create_default_config),
        ("Validation finale", setup.validate_installation)
    ]
    
    print()
    setup.print_info(f"Début de l'installation ({len(steps)} étapes)...")
    print()
    
    failed_steps = []
    for i, (step_name, step_func) in enumerate(steps, 1):
        setup.print_info(f"[{i}/{len(steps)}] {step_name}...")
        
        try:
            if step_func():
                setup.print_success(f"✓ {step_name}")
            else:
                setup.print_error(f"✗ {step_name}")
                failed_steps.append(step_name)
        except Exception as e:
            setup.print_error(f"✗ {step_name} - Erreur: {e}")
            failed_steps.append(step_name)
        
        print()
    
    # Résumé final
    if failed_steps:
        setup.print_colored("❌ Installation terminée avec des erreurs", 'RED', bold=True)
        setup.print_error(f"Étapes échouées: {', '.join(failed_steps)}")
        setup.print_info("Consultez les messages ci-dessus pour résoudre les problèmes")
        sys.exit(1)
    else:
        setup.print_colored("🎉 Installation terminée avec succès!", 'GREEN', bold=True)
        setup.print_info("Prochaines étapes:")
        setup.print_info("1. Modifiez le fichier .env avec vos identifiants")
        setup.print_info("2. Lancez: python setup.py --config (pour configuration interactive)")
        setup.print_info("3. Testez: python -c \"from engine.core import DofusBot; print('✅ Bot prêt!')\"")
        
        # Proposer la configuration interactive
        if not args.quick:
            try:
                response = input("\nLancer la configuration interactive maintenant? (o/N): ").strip().lower()
                if response in ['o', 'y', 'oui', 'yes']:
                    setup.run_interactive_config()
            except KeyboardInterrupt:
                pass
        
        sys.exit(0)


if __name__ == "__main__":
    main()