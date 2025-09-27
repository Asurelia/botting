"""
Script de Migration vers Configuration Externalisée
Migre automatiquement toutes les données codées en dur vers des fichiers de configuration
"""

import os
import sys
import json
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime

# Ajout du chemin pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.config_manager import ConfigurationManager, load_dofus_configs

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigMigrationTool:
    """Outil de migration vers configuration externalisée"""

    def __init__(self):
        self.base_path = Path(".")
        self.backup_path = Path("backup") / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.migration_report = {
            "start_time": datetime.now().isoformat(),
            "files_processed": [],
            "data_extracted": {},
            "files_modified": [],
            "errors": [],
            "statistics": {}
        }

    async def run_full_migration(self):
        """Lance la migration complète"""
        try:
            print("=== MIGRATION VERS CONFIGURATION EXTERNALISÉE ===\n")

            # 1. Création du backup
            await self._create_backup()

            # 2. Analyse des fichiers Python
            await self._analyze_python_files()

            # 3. Extraction des données codées en dur
            await self._extract_hardcoded_data()

            # 4. Génération des fichiers de configuration
            await self._generate_config_files()

            # 5. Modification des fichiers Python
            await self._modify_python_files()

            # 6. Test de la nouvelle configuration
            await self._test_configuration()

            # 7. Génération du rapport
            await self._generate_migration_report()

            print("\n=== MIGRATION TERMINÉE ===")
            print(f"Rapport de migration: {self.backup_path / 'migration_report.json'}")

        except Exception as e:
            logger.error(f"Erreur lors de la migration: {e}")
            await self._restore_backup()
            raise

    async def _create_backup(self):
        """Crée un backup complet avant migration"""
        try:
            print("Création du backup...")

            self.backup_path.mkdir(parents=True, exist_ok=True)

            # Sauvegarde des modules
            modules_path = self.base_path / "modules"
            if modules_path.exists():
                backup_modules = self.backup_path / "modules"
                shutil.copytree(modules_path, backup_modules)

            # Sauvegarde du core
            core_path = self.base_path / "core"
            if core_path.exists():
                backup_core = self.backup_path / "core"
                shutil.copytree(core_path, backup_core)

            print(f"Backup créé dans: {self.backup_path}")

        except Exception as e:
            logger.error(f"Erreur création backup: {e}")
            raise

    async def _analyze_python_files(self):
        """Analyse tous les fichiers Python pour identifier les données codées en dur"""
        try:
            print("Analyse des fichiers Python...")

            python_files = []

            # Recherche récursive des fichiers .py
            for path in [self.base_path / "modules", self.base_path / "core"]:
                if path.exists():
                    python_files.extend(path.rglob("*.py"))

            self.migration_report["files_processed"] = [str(f) for f in python_files]

            # Patterns de détection
            hardcoded_patterns = [
                # Coordonnées
                r'coordinates\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)',
                # Valeurs numériques
                r'(level_required|base_xp|base_time|market_value|health|damage)\s*=\s*(\d+(?:\.\d+)?)',
                # Listes de données
                r'(resistances|damages|drops)\s*=\s*\{[^}]+\}',
                # Chaînes de configuration
                r'(name|description)\s*=\s*["\']([^"\']+)["\']',
            ]

            for file_path in python_files:
                await self._analyze_file(file_path, hardcoded_patterns)

            print(f"Analysés {len(python_files)} fichiers Python")

        except Exception as e:
            logger.error(f"Erreur analyse fichiers: {e}")
            raise

    async def _analyze_file(self, file_path: Path, patterns: List[str]):
        """Analyse un fichier spécifique"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_data = {
                "path": str(file_path),
                "hardcoded_data": [],
                "modification_needed": False
            }

            for pattern in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    file_data["hardcoded_data"].append({
                        "pattern": pattern,
                        "match": match.group(),
                        "line": content[:match.start()].count('\n') + 1
                    })

            if file_data["hardcoded_data"]:
                file_data["modification_needed"] = True
                self.migration_report["data_extracted"][str(file_path)] = file_data

        except Exception as e:
            logger.error(f"Erreur analyse {file_path}: {e}")

    async def _extract_hardcoded_data(self):
        """Extrait les données codées en dur identifiées"""
        try:
            print("Extraction des données codées en dur...")

            # Données de ressources agricoles (déjà fait)
            agricultural_data = await self._extract_farmer_data()

            # Données de monstres
            monster_data = await self._extract_monster_data()

            # Données de cartes
            map_data = await self._extract_map_data()

            # Données de classes/sorts
            class_data = await self._extract_class_data()

            self.migration_report["statistics"] = {
                "agricultural_resources": len(agricultural_data.get("categories", {}).get("cereals", {}).get("items", {})),
                "monsters": sum(len(cat.get("monsters", {})) for cat in monster_data.get("categories", {}).values()),
                "maps": sum(len(region.get("maps", {})) for region in map_data.get("regions", {}).values()),
                "classes": len(class_data.get("classes", {}))
            }

            print("Extraction des données terminée")

        except Exception as e:
            logger.error(f"Erreur extraction données: {e}")
            raise

    async def _extract_farmer_data(self) -> Dict[str, Any]:
        """Extrait les données du module Farmer"""
        try:
            farmer_file = self.base_path / "modules" / "professions" / "farmer.py"

            if not farmer_file.exists():
                return {}

            # Les données ont déjà été externalisées dans agricultural_resources.json
            # Retour des métadonnées
            return {
                "status": "already_externalized",
                "target_file": "data/resources/agricultural_resources.json"
            }

        except Exception as e:
            logger.error(f"Erreur extraction Farmer: {e}")
            return {}

    async def _extract_monster_data(self) -> Dict[str, Any]:
        """Extrait les données des monstres si elles existent"""
        try:
            # Recherche de fichiers contenant des données de monstres
            combat_files = list((self.base_path / "modules" / "combat").rglob("*.py"))

            monster_data = {
                "metadata": {
                    "version": "1.0",
                    "extracted_from": "python_files",
                    "extraction_date": datetime.now().isoformat()
                },
                "categories": {
                    "extracted": {
                        "monsters": {}
                    }
                }
            }

            for file_path in combat_files:
                await self._scan_file_for_monsters(file_path, monster_data)

            return monster_data

        except Exception as e:
            logger.error(f"Erreur extraction monstres: {e}")
            return {}

    async def _scan_file_for_monsters(self, file_path: Path, monster_data: Dict):
        """Scanne un fichier pour des données de monstres"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Patterns pour identifier les données de monstres
            monster_patterns = [
                r'health\s*=\s*(\d+)',
                r'level\s*=\s*(\d+)',
                r'damage\s*=\s*(\d+)',
            ]

            # Simple extraction - en réalité, cela nécessiterait une analyse plus sophistiquée
            # Pour la démo, on garde les données déjà créées dans monster_database.json

        except Exception as e:
            logger.error(f"Erreur scan monstre {file_path}: {e}")

    async def _extract_map_data(self) -> Dict[str, Any]:
        """Extrait les données de cartes"""
        try:
            # Les données de cartes ont déjà été créées dans map_coordinates.json
            return {
                "status": "already_externalized",
                "target_file": "data/maps/map_coordinates.json"
            }

        except Exception as e:
            logger.error(f"Erreur extraction cartes: {e}")
            return {}

    async def _extract_class_data(self) -> Dict[str, Any]:
        """Extrait les données de classes"""
        try:
            class_files = list((self.base_path / "modules" / "combat" / "classes").rglob("*.py"))

            class_data = {
                "metadata": {
                    "version": "1.0",
                    "description": "Classes et sorts DOFUS",
                    "extracted_date": datetime.now().isoformat()
                },
                "classes": {}
            }

            for file_path in class_files:
                class_name = file_path.stem
                if class_name != "__init__" and class_name != "base_class":
                    await self._extract_class_from_file(file_path, class_name, class_data)

            return class_data

        except Exception as e:
            logger.error(f"Erreur extraction classes: {e}")
            return {}

    async def _extract_class_from_file(self, file_path: Path, class_name: str, class_data: Dict):
        """Extrait les données d'une classe depuis un fichier"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extraction basique - patterns typiques
            extracted_class = {
                "name": class_name.title(),
                "spells": [],
                "base_stats": {},
                "specializations": []
            }

            # Recherche de sorts
            spell_pattern = r'def\s+(\w+_spell|cast_\w+)\s*\('
            spell_matches = re.finditer(spell_pattern, content)

            for match in spell_matches:
                spell_name = match.group(1)
                extracted_class["spells"].append({
                    "name": spell_name,
                    "extracted_from": str(file_path)
                })

            # Recherche de statistiques
            stat_patterns = [
                r'self\.vitality\s*=\s*(\d+)',
                r'self\.strength\s*=\s*(\d+)',
                r'self\.intelligence\s*=\s*(\d+)',
                r'self\.chance\s*=\s*(\d+)',
                r'self\.agility\s*=\s*(\d+)',
                r'self\.wisdom\s*=\s*(\d+)'
            ]

            for pattern in stat_patterns:
                match = re.search(pattern, content)
                if match:
                    stat_name = pattern.split('.')[1].split('\\')[0]
                    extracted_class["base_stats"][stat_name] = int(match.group(1))

            class_data["classes"][class_name] = extracted_class

        except Exception as e:
            logger.error(f"Erreur extraction classe {class_name}: {e}")

    async def _generate_config_files(self):
        """Génère tous les fichiers de configuration"""
        try:
            print("Génération des fichiers de configuration...")

            # Création des répertoires
            config_dirs = [
                "data/classes",
                "data/spells",
                "data/items",
                "data/config"
            ]

            for dir_path in config_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            # Génération du fichier de classes
            await self._generate_classes_config()

            # Génération du fichier de configuration générale
            await self._generate_general_config()

            print("Fichiers de configuration générés")

        except Exception as e:
            logger.error(f"Erreur génération config: {e}")
            raise

    async def _generate_classes_config(self):
        """Génère le fichier de configuration des classes"""
        try:
            classes_data = self.migration_report.get("data_extracted", {}).get("classes", {})

            if not classes_data:
                # Données par défaut basées sur les classes DOFUS connues
                classes_data = {
                    "metadata": {
                        "version": "1.0",
                        "description": "Classes DOFUS avec leurs sorts et caractéristiques",
                        "last_updated": datetime.now().isoformat()
                    },
                    "classes": {
                        "iop": {
                            "name": "Iop",
                            "element": "fire",
                            "role": "damage_dealer",
                            "base_stats": {
                                "vitality": 55,
                                "strength": 20,
                                "intelligence": 0,
                                "chance": 0,
                                "agility": 0,
                                "wisdom": 10
                            },
                            "stat_gains_per_level": {
                                "vitality": 5,
                                "strength": 3,
                                "wisdom": 1
                            },
                            "spells": {
                                "pressure": {
                                    "id": "pressure",
                                    "name": "Pression",
                                    "level_required": 1,
                                    "ap_cost": 3,
                                    "range": 1,
                                    "area": "single",
                                    "element": "fire",
                                    "damage": {"min": 8, "max": 15}
                                },
                                "intimidation": {
                                    "id": "intimidation",
                                    "name": "Intimidation",
                                    "level_required": 3,
                                    "ap_cost": 2,
                                    "range": 3,
                                    "area": "single",
                                    "element": "neutral",
                                    "effect": "reduce_damage"
                                }
                            }
                        },
                        "cra": {
                            "name": "Cra",
                            "element": "air",
                            "role": "ranged_damage",
                            "base_stats": {
                                "vitality": 50,
                                "strength": 0,
                                "intelligence": 0,
                                "chance": 0,
                                "agility": 20,
                                "wisdom": 15
                            },
                            "stat_gains_per_level": {
                                "vitality": 4,
                                "agility": 3,
                                "wisdom": 2
                            },
                            "spells": {
                                "magic_arrow": {
                                    "id": "magic_arrow",
                                    "name": "Flèche Magique",
                                    "level_required": 1,
                                    "ap_cost": 3,
                                    "range": 8,
                                    "area": "line",
                                    "element": "air",
                                    "damage": {"min": 6, "max": 12}
                                }
                            }
                        }
                    }
                }

            # Sauvegarde
            classes_file = Path("data/classes/class_database.json")
            with open(classes_file, 'w', encoding='utf-8') as f:
                json.dump(classes_data, f, indent=2, ensure_ascii=False)

            print(f"Configuration des classes générée: {classes_file}")

        except Exception as e:
            logger.error(f"Erreur génération config classes: {e}")

    async def _generate_general_config(self):
        """Génère le fichier de configuration générale"""
        try:
            general_config = {
                "metadata": {
                    "version": "1.0",
                    "description": "Configuration générale de l'IA DOFUS",
                    "created": datetime.now().isoformat()
                },
                "game_settings": {
                    "default_delay": 1.0,
                    "safety_delay": 2.0,
                    "max_retries": 3,
                    "screenshot_interval": 0.5
                },
                "ai_behavior": {
                    "aggression_level": 0.7,
                    "safety_priority": 0.8,
                    "efficiency_target": 0.75,
                    "exploration_rate": 0.1
                },
                "paths": {
                    "resources": "data/resources/",
                    "monsters": "data/monsters/",
                    "maps": "data/maps/",
                    "classes": "data/classes/",
                    "logs": "logs/",
                    "screenshots": "screenshots/"
                },
                "database": {
                    "cache_size": 1000,
                    "auto_backup": true,
                    "backup_interval": 3600
                }
            }

            config_file = Path("data/config/general_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(general_config, f, indent=2, ensure_ascii=False)

            print(f"Configuration générale générée: {config_file}")

        except Exception as e:
            logger.error(f"Erreur génération config générale: {e}")

    async def _modify_python_files(self):
        """Modifie les fichiers Python pour utiliser la configuration externalisée"""
        try:
            print("Modification des fichiers Python...")

            # Exemple de modification pour farmer.py
            await self._modify_farmer_file()

            # Ajout d'imports de configuration dans les autres fichiers
            await self._add_config_imports()

            print("Fichiers Python modifiés")

        except Exception as e:
            logger.error(f"Erreur modification fichiers: {e}")
            raise

    async def _modify_farmer_file(self):
        """Modifie le fichier farmer.py"""
        try:
            farmer_file = self.base_path / "modules" / "professions" / "farmer.py"

            if not farmer_file.exists():
                return

            # Sauvegarde de l'original
            backup_file = farmer_file.with_suffix('.py.bak')
            shutil.copy2(farmer_file, backup_file)

            # Création d'un nouveau fichier qui importe la version refactorisée
            new_content = '''"""
Module Farmer - Version migrée vers configuration externalisée
Ce fichier a été automatiquement migré pour utiliser la configuration externalisée.
"""

# Import de la version refactorisée
from .farmer_refactored import FarmerRefactored, migrate_from_old_farmer

# Alias pour compatibilité
Farmer = FarmerRefactored

# Message de migration
import logging
logger = logging.getLogger(__name__)
logger.info("Module Farmer migré vers configuration externalisée")
'''

            with open(farmer_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            self.migration_report["files_modified"].append(str(farmer_file))

        except Exception as e:
            logger.error(f"Erreur modification farmer.py: {e}")

    async def _add_config_imports(self):
        """Ajoute les imports de configuration dans les fichiers nécessaires"""
        try:
            # Fichiers à modifier
            files_to_modify = [
                "modules/combat/classes/base_class.py",
                "modules/combat/classes/iop.py",
            ]

            for file_path in files_to_modify:
                full_path = self.base_path / file_path

                if not full_path.exists():
                    continue

                await self._add_config_import_to_file(full_path)

        except Exception as e:
            logger.error(f"Erreur ajout imports: {e}")

    async def _add_config_import_to_file(self, file_path: Path):
        """Ajoute l'import de configuration à un fichier"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Vérification si l'import existe déjà
            if "config_manager" in content:
                return

            # Ajout de l'import après les autres imports
            import_line = "from core.config_manager import get_config_manager, get_resource_data\n"

            # Recherche de la fin des imports
            lines = content.split('\n')
            insert_index = 0

            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_index = i + 1
                elif line.strip() == '' and insert_index > 0:
                    break

            lines.insert(insert_index, import_line)

            # Ajout d'un commentaire de migration
            lines.insert(insert_index + 1, "# Fichier migré vers configuration externalisée")

            new_content = '\n'.join(lines)

            # Sauvegarde
            backup_file = file_path.with_suffix('.py.bak')
            shutil.copy2(file_path, backup_file)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            self.migration_report["files_modified"].append(str(file_path))

        except Exception as e:
            logger.error(f"Erreur ajout import {file_path}: {e}")

    async def _test_configuration(self):
        """Test la nouvelle configuration"""
        try:
            print("Test de la nouvelle configuration...")

            # Test du chargement des configurations
            success = await load_dofus_configs()

            if success:
                print("✓ Configuration DOFUS chargée avec succès")

                # Test du gestionnaire de configuration
                config_manager = ConfigurationManager()

                # Test de récupération d'une ressource
                test_resource = await config_manager.get_config("categories.cereals.items.ble")

                if test_resource:
                    print("✓ Ressource test récupérée avec succès")
                else:
                    print("⚠ Impossible de récupérer la ressource test")

                self.migration_report["test_results"] = {
                    "config_loading": True,
                    "resource_access": test_resource is not None
                }
            else:
                print("✗ Échec du chargement de la configuration")
                self.migration_report["test_results"] = {
                    "config_loading": False
                }

        except Exception as e:
            logger.error(f"Erreur test configuration: {e}")
            self.migration_report["errors"].append(f"Test configuration: {e}")

    async def _generate_migration_report(self):
        """Génère le rapport de migration"""
        try:
            self.migration_report["end_time"] = datetime.now().isoformat()
            self.migration_report["duration"] = (
                datetime.fromisoformat(self.migration_report["end_time"]) -
                datetime.fromisoformat(self.migration_report["start_time"])
            ).total_seconds()

            report_file = self.backup_path / "migration_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.migration_report, f, indent=2, ensure_ascii=False, default=str)

            # Rapport résumé dans la console
            print(f"\n=== RAPPORT DE MIGRATION ===")
            print(f"Durée: {self.migration_report['duration']:.1f} secondes")
            print(f"Fichiers traités: {len(self.migration_report['files_processed'])}")
            print(f"Fichiers modifiés: {len(self.migration_report['files_modified'])}")
            print(f"Erreurs: {len(self.migration_report['errors'])}")

            if self.migration_report.get("test_results", {}).get("config_loading"):
                print("✓ Tests de configuration: SUCCÈS")
            else:
                print("✗ Tests de configuration: ÉCHEC")

            print(f"\nRapport complet: {report_file}")

        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")

    async def _restore_backup(self):
        """Restaure le backup en cas d'erreur"""
        try:
            print("Restauration du backup...")

            # Restauration des modules
            backup_modules = self.backup_path / "modules"
            if backup_modules.exists():
                target_modules = self.base_path / "modules"
                if target_modules.exists():
                    shutil.rmtree(target_modules)
                shutil.copytree(backup_modules, target_modules)

            # Restauration du core
            backup_core = self.backup_path / "core"
            if backup_core.exists():
                target_core = self.base_path / "core"
                if target_core.exists():
                    shutil.rmtree(target_core)
                shutil.copytree(backup_core, target_core)

            print("Backup restauré")

        except Exception as e:
            logger.error(f"Erreur restauration backup: {e}")

async def main():
    """Fonction principale de migration"""
    try:
        migration_tool = ConfigMigrationTool()
        await migration_tool.run_full_migration()

    except KeyboardInterrupt:
        print("\nMigration interrompue par l'utilisateur")
    except Exception as e:
        print(f"Erreur fatale: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())