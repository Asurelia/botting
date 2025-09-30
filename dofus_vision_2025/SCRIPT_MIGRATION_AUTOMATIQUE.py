#!/usr/bin/env python3
"""
🤖 SCRIPT DE MIGRATION AUTOMATIQUE - DOFUS VISION 2025
Créé par Claude Code - Projet Maintenance Specialist

Ce script automatise la restructuration complète du projet
selon les meilleures pratiques de maintenance.
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
import re

class DofusVisionRestructurer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.backup_created = False
        self.changes_log = []

    def log_change(self, action, details):
        """Enregistre les changements pour rollback possible"""
        self.changes_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })
        print(f"✅ {action}: {details}")

    def create_backup(self):
        """Crée une sauvegarde complète avant modifications"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.base_path / 'data' / 'backups' / f'pre_restructuration_{timestamp}'
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copie tout sauf les sauvegardes existantes
        for item in self.base_path.iterdir():
            if item.name != 'data' or not item.is_dir():
                if item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                else:
                    shutil.copy2(item, backup_dir / item.name)

        # Copie data sans les backups
        data_source = self.base_path / 'data'
        data_dest = backup_dir / 'data'
        data_dest.mkdir(exist_ok=True)

        for item in data_source.iterdir():
            if item.name != 'backups':
                if item.is_dir():
                    shutil.copytree(item, data_dest / item.name)
                else:
                    shutil.copy2(item, data_dest / item.name)

        self.backup_created = True
        self.log_change("BACKUP_CREATED", str(backup_dir))
        return backup_dir

    def create_new_structure(self):
        """Crée la nouvelle structure de dossiers"""
        new_dirs = [
            'core',
            'core/vision_engine',
            'tests',
            'tests/integration'
        ]

        for dir_path in new_dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.log_change("DIR_CREATED", dir_path)

    def move_files_to_vision_engine(self):
        """Déplace les fichiers vers core/vision_engine/"""
        files_to_move = [
            'combat_grid_analyzer.py',
            'screenshot_capture.py',
            'unity_interface_reader.py'
        ]

        vision_engine_dir = self.base_path / 'core' / 'vision_engine'

        for file_name in files_to_move:
            source = self.base_path / file_name
            destination = vision_engine_dir / file_name

            if source.exists():
                shutil.move(str(source), str(destination))
                self.log_change("FILE_MOVED", f"{file_name} → core/vision_engine/")

    def move_test_files(self):
        """Déplace les fichiers de test vers tests/"""
        test_files = [f for f in os.listdir(self.base_path) if f.startswith('test_') and f.endswith('.py')]

        tests_dir = self.base_path / 'tests'

        for test_file in test_files:
            source = self.base_path / test_file
            destination = tests_dir / test_file

            if source.exists():
                shutil.move(str(source), str(destination))
                self.log_change("TEST_MOVED", f"{test_file} → tests/")

    def move_modules_to_core(self):
        """Déplace les modules existants vers core/"""
        modules_to_move = [
            'knowledge_base',
            'learning_engine',
            'human_simulation',
            'world_model'
        ]

        core_dir = self.base_path / 'core'

        for module_name in modules_to_move:
            source = self.base_path / module_name
            destination = core_dir / module_name

            if source.exists() and source.is_dir():
                shutil.move(str(source), str(destination))
                self.log_change("MODULE_MOVED", f"{module_name}/ → core/")

    def create_init_files(self):
        """Crée les fichiers __init__.py nécessaires"""
        init_locations = [
            'core/__init__.py',
            'core/vision_engine/__init__.py',
            'tests/__init__.py',
            'tests/integration/__init__.py'
        ]

        # Contenu pour core/__init__.py
        core_init_content = '''"""
🧠 DOFUS VISION 2025 - CORE MODULES
Modules principaux du système d'IA pour DOFUS Unity World Model
"""

__version__ = "2025.1.0"
__author__ = "Claude Code - Project Maintenance Specialist"

# Imports principaux pour faciliter l'utilisation
try:
    from .vision_engine.combat_grid_analyzer import CombatGridAnalyzer
    from .vision_engine.screenshot_capture import ScreenshotCapture
    from .vision_engine.unity_interface_reader import UnityInterfaceReader

    from .knowledge_base.knowledge_integration import KnowledgeIntegration
    from .learning_engine.adaptive_learning_engine import AdaptiveLearningEngine
    from .human_simulation.advanced_human_simulation import AdvancedHumanSimulation
    from .world_model.hrm_dofus_integration import HRMDofusIntegration

    __all__ = [
        'CombatGridAnalyzer',
        'ScreenshotCapture',
        'UnityInterfaceReader',
        'KnowledgeIntegration',
        'AdaptiveLearningEngine',
        'AdvancedHumanSimulation',
        'HRMDofusIntegration'
    ]

except ImportError as e:
    print(f"⚠️ Erreur d'import dans core: {e}")
    __all__ = []
'''

        # Contenu pour vision_engine/__init__.py
        vision_init_content = '''"""
👁️ VISION ENGINE - Moteur de vision Unity pour DOFUS
Gestion de la capture d'écran, analyse d'interface et grille de combat
"""

try:
    from .combat_grid_analyzer import CombatGridAnalyzer
    from .screenshot_capture import ScreenshotCapture
    from .unity_interface_reader import UnityInterfaceReader

    __all__ = [
        'CombatGridAnalyzer',
        'ScreenshotCapture',
        'UnityInterfaceReader'
    ]

except ImportError as e:
    print(f"⚠️ Erreur d'import dans vision_engine: {e}")
    __all__ = []
'''

        # Contenu pour tests/__init__.py
        tests_init_content = '''"""
🧪 TESTS - Suite de tests pour DOFUS Vision 2025
Tests système, d'intégration et unitaires
"""

__version__ = "1.0.0"
'''

        init_contents = {
            'core/__init__.py': core_init_content,
            'core/vision_engine/__init__.py': vision_init_content,
            'tests/__init__.py': tests_init_content,
            'tests/integration/__init__.py': '# Tests d\'intégration\n'
        }

        for init_path, content in init_contents.items():
            full_path = self.base_path / init_path
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.log_change("INIT_CREATED", init_path)

    def update_imports_in_file(self, file_path):
        """Met à jour les imports dans un fichier donné"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Patterns d'imports à remplacer
            import_replacements = {
                # Imports directs depuis la racine
                r'from combat_grid_analyzer import': 'from core.vision_engine.combat_grid_analyzer import',
                r'from screenshot_capture import': 'from core.vision_engine.screenshot_capture import',
                r'from unity_interface_reader import': 'from core.vision_engine.unity_interface_reader import',
                r'import combat_grid_analyzer': 'import core.vision_engine.combat_grid_analyzer as combat_grid_analyzer',
                r'import screenshot_capture': 'import core.vision_engine.screenshot_capture as screenshot_capture',
                r'import unity_interface_reader': 'import core.vision_engine.unity_interface_reader as unity_interface_reader',

                # Imports des modules core
                r'from knowledge_base': 'from core.knowledge_base',
                r'from learning_engine': 'from core.learning_engine',
                r'from human_simulation': 'from core.human_simulation',
                r'from world_model': 'from core.world_model',
                r'import knowledge_base': 'import core.knowledge_base',
                r'import learning_engine': 'import core.learning_engine',
                r'import human_simulation': 'import core.human_simulation',
                r'import world_model': 'import core.world_model',
            }

            # Appliquer les remplacements
            for pattern, replacement in import_replacements.items():
                content = re.sub(pattern, replacement, content)

            # Écrire seulement si changements
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_change("IMPORTS_UPDATED", str(file_path))
                return True

        except Exception as e:
            print(f"❌ Erreur mise à jour imports {file_path}: {e}")
            return False

        return False

    def update_all_imports(self):
        """Met à jour tous les imports dans le projet"""
        # Trouver tous les fichiers .py
        for root, dirs, files in os.walk(self.base_path):
            # Exclure les dossiers de cache et backup
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__') and d != 'backups']

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self.update_imports_in_file(file_path)

    def clean_pycache(self):
        """Supprime tous les dossiers __pycache__"""
        for root, dirs, files in os.walk(self.base_path):
            for dir_name in dirs[:]:  # Copie pour modification sécurisée
                if dir_name == '__pycache__':
                    pycache_path = Path(root) / dir_name
                    shutil.rmtree(pycache_path)
                    dirs.remove(dir_name)  # Évite de parcourir le dossier supprimé
                    self.log_change("PYCACHE_CLEANED", str(pycache_path))

    def apply_gitignore(self):
        """Applique le .gitignore optimisé"""
        gitignore_recommended = self.base_path / '.gitignore_recommended'
        gitignore_final = self.base_path / '.gitignore'

        if gitignore_recommended.exists():
            shutil.copy2(gitignore_recommended, gitignore_final)
            self.log_change("GITIGNORE_APPLIED", ".gitignore créé depuis .gitignore_recommended")

    def run_tests(self):
        """Exécute les tests pour validation"""
        test_results = {}
        tests_dir = self.base_path / 'tests'

        if tests_dir.exists():
            for test_file in tests_dir.glob('test_*.py'):
                try:
                    # Tentative d'import pour vérifier la syntaxe
                    result = subprocess.run([
                        'python', '-m', 'py_compile', str(test_file)
                    ], capture_output=True, text=True, cwd=self.base_path)

                    test_results[test_file.name] = {
                        'syntax_ok': result.returncode == 0,
                        'error': result.stderr if result.returncode != 0 else None
                    }

                except Exception as e:
                    test_results[test_file.name] = {
                        'syntax_ok': False,
                        'error': str(e)
                    }

        return test_results

    def save_migration_report(self):
        """Sauvegarde un rapport de migration"""
        report = {
            'migration_date': datetime.now().isoformat(),
            'backup_created': self.backup_created,
            'changes_count': len(self.changes_log),
            'changes_log': self.changes_log,
            'structure_validation': self.validate_new_structure(),
            'test_results': self.run_tests()
        }

        report_path = self.base_path / 'MIGRATION_REPORT.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_change("REPORT_SAVED", str(report_path))
        return report

    def validate_new_structure(self):
        """Valide que la nouvelle structure est correcte"""
        expected_paths = [
            'core',
            'core/vision_engine',
            'core/vision_engine/combat_grid_analyzer.py',
            'core/vision_engine/screenshot_capture.py',
            'core/vision_engine/unity_interface_reader.py',
            'core/knowledge_base',
            'core/learning_engine',
            'core/human_simulation',
            'core/world_model',
            'tests',
            'data',
            'scripts',
            'assistant_interface'
        ]

        validation_results = {}
        for path in expected_paths:
            full_path = self.base_path / path
            validation_results[path] = full_path.exists()

        return validation_results

    def execute_migration(self):
        """Exécute la migration complète"""
        print("🚀 DÉBUT DE LA MIGRATION AUTOMATIQUE - DOFUS VISION 2025")
        print("=" * 60)

        try:
            # 1. Sauvegarde
            print("\n📦 ÉTAPE 1: Création de la sauvegarde...")
            backup_dir = self.create_backup()

            # 2. Nouvelle structure
            print("\n🏗️ ÉTAPE 2: Création de la nouvelle structure...")
            self.create_new_structure()

            # 3. Migration des fichiers
            print("\n📁 ÉTAPE 3: Migration des fichiers...")
            self.move_files_to_vision_engine()
            self.move_test_files()
            self.move_modules_to_core()

            # 4. Création des __init__.py
            print("\n📝 ÉTAPE 4: Création des fichiers __init__.py...")
            self.create_init_files()

            # 5. Mise à jour des imports
            print("\n🔗 ÉTAPE 5: Mise à jour des imports...")
            self.update_all_imports()

            # 6. Nettoyage
            print("\n🧹 ÉTAPE 6: Nettoyage...")
            self.clean_pycache()
            self.apply_gitignore()

            # 7. Validation et rapport
            print("\n✅ ÉTAPE 7: Validation et rapport...")
            report = self.save_migration_report()

            print("\n🎉 MIGRATION TERMINÉE AVEC SUCCÈS!")
            print(f"📊 {len(self.changes_log)} changements effectués")
            print(f"💾 Sauvegarde: {backup_dir}")
            print(f"📋 Rapport: MIGRATION_REPORT.json")

            return True, report

        except Exception as e:
            print(f"\n❌ ERREUR DURANTE LA MIGRATION: {e}")
            print(f"💾 Restauration possible depuis: {backup_dir if self.backup_created else 'Aucune sauvegarde'}")
            return False, str(e)

if __name__ == "__main__":
    # Exécution du script
    base_path = Path(__file__).parent
    migrator = DofusVisionRestructurer(base_path)
    success, result = migrator.execute_migration()

    if success:
        print(f"\n✅ Migration réussie! Rapport: {result}")
    else:
        print(f"\n❌ Migration échouée: {result}")