"""
Tool Integration - Intégration avec outils externes DOFUS
Support pour Dofus Guide, Ganymede et autres outils communautaires

Fonctionnalités:
- Détection automatique des outils installés
- Communication inter-processus
- Synchronisation données quêtes/crafts
- Import/Export configurations
- Automation coordonnée
"""

import os
import sys
import time
import json
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import tempfile
from collections import defaultdict

import psutil
import win32gui
import win32api
import win32process
import win32con
from PIL import Image
import requests
import sqlite3

@dataclass
class ExternalTool:
    """Outil externe détecté"""
    name: str
    executable_path: Path
    process_id: Optional[int]
    window_handle: Optional[int]
    version: str
    capabilities: List[str]
    data_path: Optional[Path]
    config_path: Optional[Path]
    api_port: Optional[int]
    is_running: bool

@dataclass
class QuestData:
    """Données de quête depuis outils externes"""
    quest_id: int
    name: str
    description: str
    objectives: List[str]
    rewards: List[str]
    npc_locations: List[Tuple[str, int, int]]  # (map, x, y)
    requirements: Dict[str, Any]
    completion_steps: List[str]
    estimated_time: int  # minutes

@dataclass
class CraftingRecipe:
    """Recette de craft depuis outils externes"""
    recipe_id: int
    item_name: str
    profession: str
    level_required: int
    ingredients: List[Dict[str, int]]  # {"item": "quantité"}
    tools_required: List[str]
    success_rate: float
    xp_gain: int

class DofusGuideIntegration:
    """Intégration avec Dofus Guide"""

    def __init__(self):
        self.tool_info: Optional[ExternalTool] = None
        self.data_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

        # Chemins possibles d'installation
        self.possible_paths = [
            Path(os.path.expanduser("~/AppData/Local/DofusGuide")),
            Path(os.path.expanduser("~/Documents/DofusGuide")),
            Path("C:/Program Files/DofusGuide"),
            Path("C:/Program Files (x86)/DofusGuide"),
            Path("C:/DofusGuide")
        ]

    def detect_installation(self) -> bool:
        """Détecte l'installation de Dofus Guide"""
        for path in self.possible_paths:
            if self._check_installation_path(path):
                return True

        # Recherche dans les processus actifs
        return self._detect_running_process()

    def _check_installation_path(self, path: Path) -> bool:
        """Vérifie un chemin d'installation"""
        try:
            if not path.exists():
                return False

            # Rechercher executable
            exe_files = list(path.glob("*.exe"))
            guide_exe = [f for f in exe_files if "dofus" in f.name.lower() and "guide" in f.name.lower()]

            if guide_exe:
                exe_path = guide_exe[0]

                # Rechercher données
                data_path = path / "data"
                config_path = path / "config"

                self.tool_info = ExternalTool(
                    name="Dofus Guide",
                    executable_path=exe_path,
                    process_id=None,
                    window_handle=None,
                    version=self._get_version(exe_path),
                    capabilities=["quests", "maps", "npcs", "monsters"],
                    data_path=data_path if data_path.exists() else None,
                    config_path=config_path if config_path.exists() else None,
                    api_port=None,
                    is_running=False
                )

                self.logger.info(f"Dofus Guide détecté: {exe_path}")
                return True

        except Exception as e:
            self.logger.debug(f"Erreur vérification {path}: {e}")

        return False

    def _detect_running_process(self) -> bool:
        """Détecte le processus en cours d'exécution"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                proc_info = proc.info
                if proc_info['name'] and "dofus" in proc_info['name'].lower() and "guide" in proc_info['name'].lower():
                    exe_path = Path(proc_info['exe']) if proc_info['exe'] else None

                    if exe_path:
                        self.tool_info = ExternalTool(
                            name="Dofus Guide",
                            executable_path=exe_path,
                            process_id=proc_info['pid'],
                            window_handle=self._find_window_handle(proc_info['pid']),
                            version=self._get_version(exe_path),
                            capabilities=["quests", "maps", "npcs", "monsters"],
                            data_path=exe_path.parent / "data",
                            config_path=exe_path.parent / "config",
                            api_port=None,
                            is_running=True
                        )

                        self.logger.info(f"Dofus Guide en cours: PID {proc_info['pid']}")
                        return True

        except Exception as e:
            self.logger.error(f"Erreur détection processus: {e}")

        return False

    def _find_window_handle(self, process_id: int) -> Optional[int]:
        """Trouve le handle de fenêtre pour un processus"""
        def enum_windows_callback(hwnd, windows_list):
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == process_id and win32gui.IsWindowVisible(hwnd):
                    windows_list.append(hwnd)
            except:
                pass
            return True

        windows_list = []
        win32gui.EnumWindows(enum_windows_callback, windows_list)
        return windows_list[0] if windows_list else None

    def _get_version(self, exe_path: Path) -> str:
        """Obtient la version de l'exécutable"""
        try:
            # Tentative d'obtenir version depuis propriétés fichier
            info = win32api.GetFileVersionInfo(str(exe_path), "\\")
            ms = info['FileVersionMS']
            ls = info['FileVersionLS']
            version = f"{win32api.HIWORD(ms)}.{win32api.LOWORD(ms)}.{win32api.HIWORD(ls)}.{win32api.LOWORD(ls)}"
            return version
        except:
            return "unknown"

    def get_quest_data(self, quest_name: str = None) -> List[QuestData]:
        """Récupère données de quêtes"""
        if not self.tool_info or not self.tool_info.data_path:
            return []

        quest_data = []

        try:
            # Rechercher base de données de quêtes
            db_files = list(self.tool_info.data_path.glob("*.db"))
            json_files = list(self.tool_info.data_path.glob("*quest*.json"))

            # Traiter fichiers SQLite
            for db_file in db_files:
                quest_data.extend(self._parse_quest_database(db_file, quest_name))

            # Traiter fichiers JSON
            for json_file in json_files:
                quest_data.extend(self._parse_quest_json(json_file, quest_name))

        except Exception as e:
            self.logger.error(f"Erreur récupération quêtes: {e}")

        return quest_data

    def _parse_quest_database(self, db_path: Path, quest_name: str = None) -> List[QuestData]:
        """Parse base de données SQLite de quêtes"""
        quests = []

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Rechercher tables de quêtes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%quest%'")
            tables = cursor.fetchall()

            for table_name, in tables:
                query = f"SELECT * FROM {table_name}"
                if quest_name:
                    query += f" WHERE name LIKE '%{quest_name}%'"

                cursor.execute(query)
                rows = cursor.fetchall()

                # Obtenir noms des colonnes
                column_names = [description[0] for description in cursor.description]

                for row in rows:
                    row_dict = dict(zip(column_names, row))
                    quest = self._create_quest_from_dict(row_dict)
                    if quest:
                        quests.append(quest)

            conn.close()

        except Exception as e:
            self.logger.debug(f"Erreur parsing DB {db_path}: {e}")

        return quests

    def _parse_quest_json(self, json_path: Path, quest_name: str = None) -> List[QuestData]:
        """Parse fichier JSON de quêtes"""
        quests = []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Format peut varier selon l'outil
            if isinstance(data, list):
                quest_list = data
            elif isinstance(data, dict) and 'quests' in data:
                quest_list = data['quests']
            else:
                quest_list = [data]

            for quest_dict in quest_list:
                if quest_name and quest_name.lower() not in quest_dict.get('name', '').lower():
                    continue

                quest = self._create_quest_from_dict(quest_dict)
                if quest:
                    quests.append(quest)

        except Exception as e:
            self.logger.debug(f"Erreur parsing JSON {json_path}: {e}")

        return quests

    def _create_quest_from_dict(self, data: Dict[str, Any]) -> Optional[QuestData]:
        """Crée QuestData depuis dictionnaire"""
        try:
            return QuestData(
                quest_id=data.get('id', 0),
                name=data.get('name', ''),
                description=data.get('description', ''),
                objectives=data.get('objectives', []),
                rewards=data.get('rewards', []),
                npc_locations=data.get('npc_locations', []),
                requirements=data.get('requirements', {}),
                completion_steps=data.get('steps', []),
                estimated_time=data.get('time', 0)
            )
        except Exception as e:
            self.logger.debug(f"Erreur création quest: {e}")
            return None

class GanymedeIntegration:
    """Intégration avec Ganymede (calculateur/optimiseur)"""

    def __init__(self):
        self.tool_info: Optional[ExternalTool] = None
        self.logger = logging.getLogger(__name__)

        self.possible_paths = [
            Path(os.path.expanduser("~/AppData/Local/Ganymede")),
            Path(os.path.expanduser("~/Documents/Ganymede")),
            Path("C:/Program Files/Ganymede"),
            Path("C:/Program Files (x86)/Ganymede")
        ]

    def detect_installation(self) -> bool:
        """Détecte Ganymede"""
        # Logique similaire à DofusGuide mais pour Ganymede
        for path in self.possible_paths:
            if self._check_installation_path(path):
                return True
        return self._detect_running_process()

    def _check_installation_path(self, path: Path) -> bool:
        """Vérifie installation Ganymede"""
        try:
            if not path.exists():
                return False

            exe_files = list(path.glob("*.exe"))
            ganymede_exe = [f for f in exe_files if "ganymede" in f.name.lower()]

            if ganymede_exe:
                self.tool_info = ExternalTool(
                    name="Ganymede",
                    executable_path=ganymede_exe[0],
                    process_id=None,
                    window_handle=None,
                    version="unknown",
                    capabilities=["crafting", "optimization", "economy"],
                    data_path=path / "data",
                    config_path=path / "config",
                    api_port=None,
                    is_running=False
                )

                self.logger.info(f"Ganymede détecté: {ganymede_exe[0]}")
                return True

        except Exception as e:
            self.logger.debug(f"Erreur vérification Ganymede {path}: {e}")

        return False

    def _detect_running_process(self) -> bool:
        """Détecte processus Ganymede en cours"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                proc_info = proc.info
                if proc_info['name'] and "ganymede" in proc_info['name'].lower():
                    exe_path = Path(proc_info['exe']) if proc_info['exe'] else None

                    if exe_path:
                        self.tool_info = ExternalTool(
                            name="Ganymede",
                            executable_path=exe_path,
                            process_id=proc_info['pid'],
                            window_handle=None,
                            version="unknown",
                            capabilities=["crafting", "optimization", "economy"],
                            data_path=exe_path.parent / "data",
                            config_path=exe_path.parent / "config",
                            api_port=None,
                            is_running=True
                        )

                        self.logger.info(f"Ganymede en cours: PID {proc_info['pid']}")
                        return True

        except Exception as e:
            self.logger.error(f"Erreur détection Ganymede: {e}")

        return False

    def get_crafting_recipes(self, profession: str = None) -> List[CraftingRecipe]:
        """Récupère recettes de craft"""
        if not self.tool_info or not self.tool_info.data_path:
            return []

        recipes = []

        try:
            # Rechercher fichiers de recettes
            recipe_files = list(self.tool_info.data_path.glob("*recipe*.json"))
            recipe_files.extend(list(self.tool_info.data_path.glob("*craft*.json")))

            for recipe_file in recipe_files:
                recipes.extend(self._parse_recipe_file(recipe_file, profession))

        except Exception as e:
            self.logger.error(f"Erreur récupération recettes: {e}")

        return recipes

    def _parse_recipe_file(self, file_path: Path, profession: str = None) -> List[CraftingRecipe]:
        """Parse fichier de recettes"""
        recipes = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            recipe_list = data if isinstance(data, list) else data.get('recipes', [])

            for recipe_dict in recipe_list:
                if profession and recipe_dict.get('profession', '').lower() != profession.lower():
                    continue

                recipe = CraftingRecipe(
                    recipe_id=recipe_dict.get('id', 0),
                    item_name=recipe_dict.get('name', ''),
                    profession=recipe_dict.get('profession', ''),
                    level_required=recipe_dict.get('level', 0),
                    ingredients=recipe_dict.get('ingredients', []),
                    tools_required=recipe_dict.get('tools', []),
                    success_rate=recipe_dict.get('success_rate', 1.0),
                    xp_gain=recipe_dict.get('xp', 0)
                )
                recipes.append(recipe)

        except Exception as e:
            self.logger.debug(f"Erreur parsing recettes {file_path}: {e}")

        return recipes

class ToolIntegrationManager:
    """Gestionnaire principal d'intégration des outils"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Intégrations
        self.dofus_guide = DofusGuideIntegration()
        self.ganymede = GanymedeIntegration()

        # État
        self.detected_tools: List[ExternalTool] = []
        self.sync_active = False
        self.sync_thread: Optional[threading.Thread] = None

        # Cache données
        self.quest_cache: Dict[str, QuestData] = {}
        self.recipe_cache: Dict[str, CraftingRecipe] = {}
        self.last_sync_time = 0.0

        self.logger.info("ToolIntegrationManager initialisé")

    def initialize(self) -> bool:
        """Initialise le gestionnaire d'intégration"""
        try:
            # Détecter outils installés
            self._detect_all_tools()

            # Charger cache existant
            self._load_cache()

            self.logger.info(f"Outils détectés: {len(self.detected_tools)}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation ToolIntegrationManager: {e}")
            return False

    def _detect_all_tools(self):
        """Détecte tous les outils supportés"""
        self.detected_tools.clear()

        # Dofus Guide
        if self.dofus_guide.detect_installation():
            self.detected_tools.append(self.dofus_guide.tool_info)

        # Ganymede
        if self.ganymede.detect_installation():
            self.detected_tools.append(self.ganymede.tool_info)

        self.logger.info(f"Détectés: {[tool.name for tool in self.detected_tools]}")

    def start_sync(self) -> bool:
        """Démarre synchronisation des données"""
        if self.sync_active:
            return False

        self.sync_active = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

        self.logger.info("Synchronisation démarrée")
        return True

    def stop_sync(self):
        """Arrête synchronisation"""
        self.sync_active = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)

        self.logger.info("Synchronisation arrêtée")

    def _sync_loop(self):
        """Boucle de synchronisation"""
        while self.sync_active:
            current_time = time.time()

            # Synchroniser toutes les 60 secondes
            if current_time - self.last_sync_time > 60.0:
                self._sync_data()
                self.last_sync_time = current_time

            time.sleep(10.0)

    def _sync_data(self):
        """Synchronise données avec outils externes"""
        try:
            # Synchroniser quêtes
            if self.dofus_guide.tool_info:
                quests = self.dofus_guide.get_quest_data()
                for quest in quests:
                    self.quest_cache[quest.name] = quest

            # Synchroniser recettes
            if self.ganymede.tool_info:
                recipes = self.ganymede.get_crafting_recipes()
                for recipe in recipes:
                    self.recipe_cache[recipe.item_name] = recipe

            # Sauvegarder cache
            self._save_cache()

            self.logger.debug(f"Sync: {len(self.quest_cache)} quêtes, {len(self.recipe_cache)} recettes")

        except Exception as e:
            self.logger.error(f"Erreur synchronisation: {e}")

    def get_quest_guidance(self, quest_name: str) -> Optional[QuestData]:
        """Obtient guidage pour une quête"""
        # Recherche dans cache
        quest = self.quest_cache.get(quest_name)

        if not quest and self.dofus_guide.tool_info:
            # Recherche directe si pas en cache
            quests = self.dofus_guide.get_quest_data(quest_name)
            if quests:
                quest = quests[0]
                self.quest_cache[quest_name] = quest

        return quest

    def get_crafting_optimization(self, item_name: str) -> Optional[CraftingRecipe]:
        """Obtient optimisation de craft"""
        recipe = self.recipe_cache.get(item_name)

        if not recipe and self.ganymede.tool_info:
            # Recherche directe
            recipes = self.ganymede.get_crafting_recipes()
            for r in recipes:
                if r.item_name.lower() == item_name.lower():
                    recipe = r
                    self.recipe_cache[item_name] = recipe
                    break

        return recipe

    def launch_tool(self, tool_name: str) -> bool:
        """Lance un outil externe"""
        for tool in self.detected_tools:
            if tool.name.lower() == tool_name.lower():
                if tool.is_running:
                    self.logger.info(f"{tool_name} déjà en cours")
                    return True

                try:
                    subprocess.Popen([str(tool.executable_path)], cwd=tool.executable_path.parent)
                    time.sleep(2.0)  # Attendre lancement

                    # Mettre à jour statut
                    tool.is_running = True
                    self.logger.info(f"{tool_name} lancé")
                    return True

                except Exception as e:
                    self.logger.error(f"Erreur lancement {tool_name}: {e}")

        return False

    def _load_cache(self):
        """Charge cache depuis fichier"""
        cache_file = self.data_dir / "tools_cache.json"

        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Restaurer quêtes
                for quest_data in data.get('quests', []):
                    quest = QuestData(**quest_data)
                    self.quest_cache[quest.name] = quest

                # Restaurer recettes
                for recipe_data in data.get('recipes', []):
                    recipe = CraftingRecipe(**recipe_data)
                    self.recipe_cache[recipe.item_name] = recipe

                self.logger.info(f"Cache chargé: {len(self.quest_cache)} quêtes, {len(self.recipe_cache)} recettes")

        except Exception as e:
            self.logger.error(f"Erreur chargement cache: {e}")

    def _save_cache(self):
        """Sauvegarde cache"""
        cache_file = self.data_dir / "tools_cache.json"

        try:
            data = {
                'quests': [asdict(quest) for quest in self.quest_cache.values()],
                'recipes': [asdict(recipe) for recipe in self.recipe_cache.values()],
                'timestamp': time.time()
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cache: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Retourne statut des intégrations"""
        return {
            "detected_tools": [tool.name for tool in self.detected_tools],
            "running_tools": [tool.name for tool in self.detected_tools if tool.is_running],
            "quest_cache_size": len(self.quest_cache),
            "recipe_cache_size": len(self.recipe_cache),
            "sync_active": self.sync_active
        }

    def cleanup(self):
        """Nettoyage ressources"""
        self.stop_sync()
        self._save_cache()
        self.quest_cache.clear()
        self.recipe_cache.clear()
        self.logger.info("ToolIntegrationManager nettoyé")

# Factory function
def create_tool_integration_manager(data_dir: Path) -> ToolIntegrationManager:
    """Crée instance ToolIntegrationManager configurée"""
    manager = ToolIntegrationManager(data_dir)
    if manager.initialize():
        return manager
    else:
        raise RuntimeError("Impossible d'initialiser ToolIntegrationManager")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        data_dir = Path("data/tools")
        manager = create_tool_integration_manager(data_dir)

        print("Test ToolIntegration...")

        # Afficher outils détectés
        status = manager.get_integration_status()
        print(f"Status: {status}")

        # Test recherche quête
        quest = manager.get_quest_guidance("Tutoriel")
        if quest:
            print(f"Quête trouvée: {quest.name}")

        # Test recette
        recipe = manager.get_crafting_optimization("Pain")
        if recipe:
            print(f"Recette trouvée: {recipe.item_name}")

    except Exception as e:
        print(f"Erreur test: {e}")
    finally:
        if 'manager' in locals():
            manager.cleanup()