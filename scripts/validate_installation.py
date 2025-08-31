"""
Script de validation de l'installation complète
==============================================

Ce script valide que tous les composants du bot DOFUS sont
correctement installés et configurés.

Usage:
    python scripts/validate_installation.py [options]

Options:
    --verbose       Affichage détaillé
    --fix           Tenter de corriger les problèmes détectés
    --quick         Validation rapide (tests de base uniquement)

Créé le: 2025-08-31
Version: 1.0.0
"""

import sys
import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


class InstallationValidator:
    """Validateur de l'installation du bot DOFUS"""
    
    def __init__(self, verbose: bool = False, fix_mode: bool = False):
        """
        Initialise le validateur
        
        Args:
            verbose: Mode verbeux
            fix_mode: Tenter de corriger les problèmes
        """
        self.verbose = verbose
        self.fix_mode = fix_mode
        self.project_root = Path(__file__).parent.parent
        self.issues_found = []
        self.tests_passed = 0
        self.tests_total = 0
        
        # Configuration du logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Enregistre le résultat d'un test"""
        self.tests_total += 1
        if passed:
            self.tests_passed += 1
            self.logger.info(f"[OK] {test_name}")
            if message and self.verbose:
                self.logger.debug(f"    {message}")
        else:
            self.logger.error(f"[FAIL] {test_name}")
            if message:
                self.logger.error(f"    {message}")
            self.issues_found.append((test_name, message))
    
    def validate_python_environment(self) -> bool:
        """Valide l'environnement Python"""
        self.logger.info("=== VALIDATION ENVIRONNEMENT PYTHON ===")
        
        # Version Python
        python_version = sys.version_info
        version_ok = python_version >= (3, 8)
        self.log_test(
            "Version Python",
            version_ok,
            f"Version {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        
        # Modules de base requis
        required_modules = [
            'pathlib', 'json', 'sqlite3', 'hashlib', 'datetime',
            'logging', 'shutil', 'os', 'sys', 'threading'
        ]
        
        modules_ok = True
        for module in required_modules:
            try:
                __import__(module)
                if self.verbose:
                    self.log_test(f"Module {module}", True)
            except ImportError as e:
                self.log_test(f"Module {module}", False, str(e))
                modules_ok = False
        
        if not self.verbose:
            self.log_test("Modules Python de base", modules_ok)
        
        return version_ok and modules_ok
    
    def validate_directory_structure(self) -> bool:
        """Valide la structure de répertoires"""
        self.logger.info("=== VALIDATION STRUCTURE RÉPERTOIRES ===")
        
        required_dirs = [
            'config',
            'data/databases',
            'data/backups',
            'logs',
            'engine',
            'modules',
            'state'
        ]
        
        all_dirs_ok = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            
            if not exists and self.fix_mode:
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    exists = True
                    self.logger.info(f"    Répertoire créé: {dir_path}")
                except Exception as e:
                    self.logger.error(f"    Impossible de créer {dir_path}: {e}")
            
            self.log_test(f"Répertoire {dir_path}", exists)
            if not exists:
                all_dirs_ok = False
        
        return all_dirs_ok
    
    def validate_config_files(self) -> bool:
        """Valide les fichiers de configuration"""
        self.logger.info("=== VALIDATION FICHIERS CONFIGURATION ===")
        
        config_files = [
            ('config/bot_config.yaml', 'Configuration principale'),
            ('.env.example', 'Exemple de configuration environnement'),
            ('requirements.txt', 'Dépendances Python'),
            ('setup.py', 'Script d\'installation')
        ]
        
        all_configs_ok = True
        for file_path, description in config_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            self.log_test(f"{description}", exists, str(full_path) if exists else "Fichier manquant")
            if not exists:
                all_configs_ok = False
        
        # Validation du contenu YAML
        yaml_path = self.project_root / 'config' / 'bot_config.yaml'
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                required_sections = ['app', 'database', 'logging', 'security']
                yaml_valid = all(section in config for section in required_sections)
                self.log_test("Structure YAML valide", yaml_valid)
                if not yaml_valid:
                    all_configs_ok = False
                    
            except Exception as e:
                self.log_test("Fichier YAML valide", False, str(e))
                all_configs_ok = False
        
        return all_configs_ok
    
    def validate_database(self) -> bool:
        """Valide la base de données"""
        self.logger.info("=== VALIDATION BASE DE DONNÉES ===")
        
        db_path = self.project_root / 'data' / 'databases' / 'dofus_bot.db'
        
        # Existence du fichier
        db_exists = db_path.exists()
        self.log_test("Fichier base de données", db_exists)
        
        if not db_exists:
            if self.fix_mode:
                self.logger.info("    Tentative d'initialisation de la base de données...")
                try:
                    from config.database_setup import init_database
                    if init_database(str(self.project_root / 'config' / 'bot_config.yaml')):
                        self.logger.info("    Base de données initialisée")
                        db_exists = True
                    else:
                        self.logger.error("    Échec initialisation base de données")
                except Exception as e:
                    self.logger.error(f"    Erreur initialisation: {e}")
            return db_exists
        
        # Test de connexion
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Vérifier les tables principales
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                'system_info', 'characters', 'spells', 'items', 
                'combats', 'market_prices', 'maps'
            ]
            
            tables_ok = all(table in tables for table in required_tables)
            self.log_test("Tables requises présentes", tables_ok, f"{len(tables)} tables trouvées")
            
            # Vérifier la version
            try:
                cursor = conn.execute("SELECT version FROM system_info WHERE id = 1")
                version_row = cursor.fetchone()
                version_ok = version_row is not None
                version = version_row['version'] if version_row else "inconnue"
                self.log_test("Version base de données", version_ok, f"Version: {version}")
            except sqlite3.Error:
                self.log_test("Version base de données", False, "Table system_info invalide")
                tables_ok = False
            
            # Test des données
            cursor = conn.execute("SELECT COUNT(*) as count FROM spells")
            spell_count = cursor.fetchone()['count']
            data_ok = spell_count > 0
            self.log_test("Données de jeu présentes", data_ok, f"{spell_count} sorts chargés")
            
            conn.close()
            return tables_ok and data_ok
            
        except sqlite3.Error as e:
            self.log_test("Connexion base de données", False, str(e))
            return False
    
    def validate_modules(self) -> bool:
        """Valide les modules du bot"""
        self.logger.info("=== VALIDATION MODULES ===")
        
        module_dirs = [
            ('engine', 'Moteur principal'),
            ('modules/combat', 'Module combat'),
            ('modules/navigation', 'Module navigation'),
            ('modules/professions', 'Module professions'),
            ('modules/economy', 'Module économie'),
            ('modules/safety', 'Module sécurité'),
            ('state', 'Gestion état')
        ]
        
        all_modules_ok = True
        for module_path, description in module_dirs:
            full_path = self.project_root / module_path
            exists = full_path.exists() and full_path.is_dir()
            
            # Vérifier qu'il y a des fichiers Python
            if exists:
                py_files = list(full_path.glob('*.py'))
                has_python = len(py_files) > 0
                self.log_test(f"{description}", has_python, f"{len(py_files)} fichiers Python")
                if not has_python:
                    all_modules_ok = False
            else:
                self.log_test(f"{description}", False, "Répertoire manquant")
                all_modules_ok = False
        
        return all_modules_ok
    
    def validate_dependencies(self) -> bool:
        """Valide les dépendances optionnelles"""
        self.logger.info("=== VALIDATION DÉPENDANCES OPTIONNELLES ===")
        
        optional_deps = [
            ('yaml', 'PyYAML - Configuration'),
            ('cv2', 'OpenCV - Vision'),
            ('PIL', 'Pillow - Images'),
            ('mss', 'MSS - Screenshots'),
            ('pytesseract', 'Tesseract - OCR'),
            ('numpy', 'NumPy - Calculs'),
            ('pyautogui', 'PyAutoGUI - Automation'),
            ('requests', 'Requests - HTTP')
        ]
        
        available_deps = 0
        for module, description in optional_deps:
            try:
                __import__(module)
                self.log_test(description, True, "Disponible")
                available_deps += 1
            except ImportError:
                self.log_test(description, False, "Non installé")
        
        # Au moins 50% des dépendances optionnelles doivent être disponibles
        deps_sufficient = available_deps >= len(optional_deps) * 0.5
        self.log_test(
            "Dépendances suffisantes", 
            deps_sufficient, 
            f"{available_deps}/{len(optional_deps)} disponibles"
        )
        
        return deps_sufficient
    
    def validate_permissions(self) -> bool:
        """Valide les permissions de fichiers"""
        self.logger.info("=== VALIDATION PERMISSIONS ===")
        
        # Test d'écriture dans les répertoires importants
        test_dirs = [
            ('logs', 'Répertoire logs'),
            ('data/databases', 'Base de données'),
            ('data/backups', 'Sauvegardes'),
        ]
        
        all_perms_ok = True
        for dir_path, description in test_dirs:
            full_path = self.project_root / dir_path
            
            if not full_path.exists():
                self.log_test(f"Écriture {description}", False, "Répertoire manquant")
                all_perms_ok = False
                continue
            
            # Test d'écriture
            test_file = full_path / '.write_test'
            try:
                test_file.write_text("test")
                test_file.unlink()
                self.log_test(f"Écriture {description}", True)
            except Exception as e:
                self.log_test(f"Écriture {description}", False, str(e))
                all_perms_ok = False
        
        return all_perms_ok
    
    def run_full_validation(self, quick_mode: bool = False) -> bool:
        """
        Lance la validation complète
        
        Args:
            quick_mode: Mode validation rapide
            
        Returns:
            True si toutes les validations passent
        """
        self.logger.info("VALIDATION INSTALLATION DOFUS BOT")
        self.logger.info(f"Mode: {'Rapide' if quick_mode else 'Complet'}")
        self.logger.info(f"Correction automatique: {'Activée' if self.fix_mode else 'Désactivée'}")
        print()
        
        # Tests de base (toujours exécutés)
        validations = [
            ("Environnement Python", self.validate_python_environment),
            ("Structure répertoires", self.validate_directory_structure),
            ("Fichiers configuration", self.validate_config_files),
            ("Base de données", self.validate_database),
        ]
        
        # Tests supplémentaires en mode complet
        if not quick_mode:
            validations.extend([
                ("Modules bot", self.validate_modules),
                ("Dépendances", self.validate_dependencies),
                ("Permissions", self.validate_permissions),
            ])
        
        # Exécuter les validations
        all_passed = True
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.logger.error(f"Erreur dans {validation_name}: {e}")
                all_passed = False
            print()  # Espacement entre les sections
        
        return all_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Génère un rapport de validation"""
        return {
            'timestamp': str(Path(__file__).parent.parent),
            'tests_total': self.tests_total,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_total - self.tests_passed,
            'success_rate': (self.tests_passed / self.tests_total * 100) if self.tests_total > 0 else 0,
            'issues_found': self.issues_found,
            'fix_mode_used': self.fix_mode
        }
    
    def print_summary(self):
        """Affiche le résumé de validation"""
        success_rate = (self.tests_passed / self.tests_total * 100) if self.tests_total > 0 else 0
        
        print("=" * 60)
        print("RÉSUMÉ DE VALIDATION")
        print("=" * 60)
        print(f"Tests exécutés: {self.tests_total}")
        print(f"Tests réussis: {self.tests_passed}")
        print(f"Tests échoués: {self.tests_total - self.tests_passed}")
        print(f"Taux de réussite: {success_rate:.1f}%")
        
        if self.issues_found:
            print(f"\n[ERROR] PROBLÈMES DÉTECTÉS ({len(self.issues_found)}):")
            for issue_name, issue_desc in self.issues_found:
                print(f"  - {issue_name}: {issue_desc}")
            
            if not self.fix_mode:
                print("\n[INFO] Utilisez --fix pour tenter de corriger automatiquement")
        
        if success_rate >= 90:
            print("\n[OK] Installation validée - Bot prêt à utiliser!")
        elif success_rate >= 70:
            print("\n[WARN] Installation partiellement validée - Quelques problèmes à résoudre")
        else:
            print("\n[ERROR] Installation incomplète - Problèmes critiques détectés")
        
        print("=" * 60)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Validation de l'installation du bot DOFUS",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Affichage détaillé')
    parser.add_argument('--fix', action='store_true',
                       help='Tenter de corriger les problèmes détectés')
    parser.add_argument('--quick', action='store_true',
                       help='Validation rapide (tests de base uniquement)')
    parser.add_argument('--report', metavar='FILE',
                       help='Générer un rapport JSON')
    
    args = parser.parse_args()
    
    # Initialiser le validateur
    validator = InstallationValidator(verbose=args.verbose, fix_mode=args.fix)
    
    try:
        # Lancer la validation
        success = validator.run_full_validation(quick_mode=args.quick)
        
        # Afficher le résumé
        validator.print_summary()
        
        # Générer le rapport si demandé
        if args.report:
            report = validator.generate_report()
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] Rapport généré: {args.report}")
        
        # Code de sortie
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n[INFO] Validation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Erreur critique: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()