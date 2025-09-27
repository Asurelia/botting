"""
Gestionnaire de Configuration Dynamique pour l'IA DOFUS Évolutive
Gère le chargement, la mise à jour et la synchronisation des configurations externalisées
"""

import json
import sqlite3
import yaml
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import hashlib
import time

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    SQLITE = "sqlite"
    INI = "ini"

class ConfigScope(Enum):
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    TEMPORARY = "temporary"

@dataclass
class ConfigEntry:
    """Entrée de configuration"""
    key: str
    value: Any
    format: ConfigFormat
    scope: ConfigScope
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    source_file: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class ConfigWatcher:
    """Surveillant de fichier de configuration"""
    file_path: str
    callback: Callable
    last_modified: float = 0.0
    active: bool = True

class ConfigurationManager:
    """Gestionnaire principal de configuration"""

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.configs: Dict[str, ConfigEntry] = {}
        self.watchers: Dict[str, ConfigWatcher] = {}
        self.cache: Dict[str, Any] = {}

        # Database pour persistance
        self.db_path = self.base_path / "config" / "config_cache.db"
        self.db_lock = threading.Lock()

        # Configuration du monitoring
        self.watch_interval = 1.0  # secondes
        self.auto_reload = True
        self.cache_ttl = 300  # 5 minutes

        # Thread de surveillance
        self._watch_thread = None
        self._stop_watching = False

        self._initialize_database()

    def _initialize_database(self):
        """Initialise la base de données de cache"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        format TEXT,
                        scope TEXT,
                        last_modified TIMESTAMP,
                        version TEXT,
                        source_file TEXT,
                        checksum TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT,
                        old_value TEXT,
                        new_value TEXT,
                        changed_at TIMESTAMP,
                        reason TEXT
                    )
                """)

                conn.commit()

            logger.info("Base de données de configuration initialisée")

        except Exception as e:
            logger.error(f"Erreur initialisation DB config: {e}")

    async def load_config_file(self, file_path: str, scope: ConfigScope = ConfigScope.GLOBAL,
                              auto_watch: bool = True) -> bool:
        """Charge un fichier de configuration"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Fichier de configuration introuvable: {file_path}")
                return False

            # Détermination du format
            format_map = {
                '.json': ConfigFormat.JSON,
                '.yaml': ConfigFormat.YAML,
                '.yml': ConfigFormat.YAML,
                '.db': ConfigFormat.SQLITE
            }

            config_format = format_map.get(file_path.suffix.lower(), ConfigFormat.JSON)

            # Chargement selon le format
            if config_format == ConfigFormat.JSON:
                data = await self._load_json(file_path)
            elif config_format == ConfigFormat.YAML:
                data = await self._load_yaml(file_path)
            elif config_format == ConfigFormat.SQLITE:
                data = await self._load_sqlite(file_path)
            else:
                logger.error(f"Format non supporté: {config_format}")
                return False

            if data is None:
                return False

            # Calcul du checksum
            checksum = self._calculate_checksum(str(data))

            # Enregistrement des configurations
            await self._register_config_data(data, config_format, scope, str(file_path), checksum)

            # Surveillance automatique
            if auto_watch and self.auto_reload:
                await self._start_watching_file(str(file_path))

            logger.info(f"Configuration chargée: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur chargement config {file_path}: {e}")
            return False

    async def _load_json(self, file_path: Path) -> Optional[Dict]:
        """Charge un fichier JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lecture JSON {file_path}: {e}")
            return None

    async def _load_yaml(self, file_path: Path) -> Optional[Dict]:
        """Charge un fichier YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur lecture YAML {file_path}: {e}")
            return None

    async def _load_sqlite(self, file_path: Path) -> Optional[Dict]:
        """Charge une base de données SQLite"""
        try:
            data = {}
            with sqlite3.connect(str(file_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Récupération de toutes les tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table['name']
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()

                    data[table_name] = [dict(row) for row in rows]

            return data

        except Exception as e:
            logger.error(f"Erreur lecture SQLite {file_path}: {e}")
            return None

    def _calculate_checksum(self, content: str) -> str:
        """Calcule le checksum d'un contenu"""
        return hashlib.md5(content.encode()).hexdigest()

    async def _register_config_data(self, data: Dict, format: ConfigFormat,
                                   scope: ConfigScope, source_file: str, checksum: str):
        """Enregistre les données de configuration"""
        try:
            def register_recursive(obj: Any, prefix: str = ""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        register_recursive(value, full_key)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        full_key = f"{prefix}[{i}]"
                        register_recursive(item, full_key)
                else:
                    # Valeur finale
                    config_entry = ConfigEntry(
                        key=prefix,
                        value=obj,
                        format=format,
                        scope=scope,
                        source_file=source_file,
                        checksum=checksum
                    )

                    self.configs[prefix] = config_entry

                    # Cache
                    self.cache[prefix] = obj

            register_recursive(data)

            # Persistance en base
            await self._persist_to_database()

        except Exception as e:
            logger.error(f"Erreur enregistrement config: {e}")

    async def _persist_to_database(self):
        """Persiste les configurations en base"""
        try:
            with self.db_lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    for key, config in self.configs.items():
                        conn.execute("""
                            INSERT OR REPLACE INTO config_cache
                            (key, value, format, scope, last_modified, version, source_file, checksum)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            key,
                            json.dumps(config.value) if not isinstance(config.value, str) else config.value,
                            config.format.value,
                            config.scope.value,
                            config.last_modified.isoformat(),
                            config.version,
                            config.source_file,
                            config.checksum
                        ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Erreur persistance DB: {e}")

    async def get_config(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration"""
        try:
            # Cache d'abord
            if key in self.cache:
                return self.cache[key]

            # Configuration en mémoire
            if key in self.configs:
                value = self.configs[key].value
                self.cache[key] = value
                return value

            # Base de données
            value = await self._get_from_database(key)
            if value is not None:
                self.cache[key] = value
                return value

            # Valeur par défaut
            return default

        except Exception as e:
            logger.error(f"Erreur récupération config {key}: {e}")
            return default

    async def _get_from_database(self, key: str) -> Optional[Any]:
        """Récupère une valeur depuis la base"""
        try:
            with self.db_lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    cursor.execute("SELECT value FROM config_cache WHERE key = ?", (key,))
                    row = cursor.fetchone()

                    if row:
                        value_str = row['value']
                        try:
                            return json.loads(value_str)
                        except:
                            return value_str

            return None

        except Exception as e:
            logger.error(f"Erreur lecture DB {key}: {e}")
            return None

    async def set_config(self, key: str, value: Any, scope: ConfigScope = ConfigScope.SESSION,
                        persist: bool = True) -> bool:
        """Définit une valeur de configuration"""
        try:
            # Historique du changement
            old_value = await self.get_config(key)
            if old_value != value:
                await self._log_config_change(key, old_value, value, "manual_update")

            # Création de l'entrée
            config_entry = ConfigEntry(
                key=key,
                value=value,
                format=ConfigFormat.JSON,
                scope=scope,
                last_modified=datetime.now()
            )

            # Mise à jour
            self.configs[key] = config_entry
            self.cache[key] = value

            # Persistance
            if persist:
                await self._persist_to_database()

            logger.debug(f"Configuration mise à jour: {key} = {value}")
            return True

        except Exception as e:
            logger.error(f"Erreur mise à jour config {key}: {e}")
            return False

    async def _log_config_change(self, key: str, old_value: Any, new_value: Any, reason: str):
        """Enregistre un changement de configuration"""
        try:
            with self.db_lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("""
                        INSERT INTO config_history
                        (key, old_value, new_value, changed_at, reason)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        key,
                        json.dumps(old_value) if old_value is not None else None,
                        json.dumps(new_value) if new_value is not None else None,
                        datetime.now().isoformat(),
                        reason
                    ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Erreur log changement: {e}")

    async def get_config_section(self, section_prefix: str) -> Dict[str, Any]:
        """Récupère toutes les configurations d'une section"""
        try:
            result = {}

            for key in self.configs.keys():
                if key.startswith(section_prefix):
                    relative_key = key[len(section_prefix):].lstrip('.')
                    value = await self.get_config(key)

                    # Reconstruction de la structure
                    self._set_nested_value(result, relative_key, value)

            return result

        except Exception as e:
            logger.error(f"Erreur récupération section {section_prefix}: {e}")
            return {}

    def _set_nested_value(self, obj: Dict, path: str, value: Any):
        """Définit une valeur imbriquée dans un dictionnaire"""
        parts = path.split('.')
        current = obj

        for part in parts[:-1]:
            if '[' in part and ']' in part:
                # Gestion des listes
                key, index = part.split('[')
                index = int(index.rstrip(']'))

                if key not in current:
                    current[key] = []

                while len(current[key]) <= index:
                    current[key].append({})

                current = current[key][index]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Dernière partie
        final_part = parts[-1]
        if '[' in final_part and ']' in final_part:
            key, index = final_part.split('[')
            index = int(index.rstrip(']'))

            if key not in current:
                current[key] = []

            while len(current[key]) <= index:
                current[key].append(None)

            current[key][index] = value
        else:
            current[final_part] = value

    async def _start_watching_file(self, file_path: str):
        """Démarre la surveillance d'un fichier"""
        try:
            if file_path in self.watchers:
                return  # Déjà surveillé

            callback = lambda: asyncio.create_task(self._reload_file(file_path))

            watcher = ConfigWatcher(
                file_path=file_path,
                callback=callback,
                last_modified=os.path.getmtime(file_path)
            )

            self.watchers[file_path] = watcher

            # Démarrage du thread de surveillance si nécessaire
            if self._watch_thread is None or not self._watch_thread.is_alive():
                self._start_watch_thread()

            logger.debug(f"Surveillance démarrée: {file_path}")

        except Exception as e:
            logger.error(f"Erreur démarrage surveillance {file_path}: {e}")

    def _start_watch_thread(self):
        """Démarre le thread de surveillance des fichiers"""
        try:
            self._stop_watching = False
            self._watch_thread = threading.Thread(target=self._watch_files_thread, daemon=True)
            self._watch_thread.start()

        except Exception as e:
            logger.error(f"Erreur démarrage thread surveillance: {e}")

    def _watch_files_thread(self):
        """Thread de surveillance des fichiers"""
        while not self._stop_watching:
            try:
                for file_path, watcher in list(self.watchers.items()):
                    if not watcher.active:
                        continue

                    try:
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime > watcher.last_modified:
                            watcher.last_modified = current_mtime
                            watcher.callback()
                            logger.info(f"Fichier modifié détecté: {file_path}")

                    except FileNotFoundError:
                        logger.warning(f"Fichier surveillé supprimé: {file_path}")
                        watcher.active = False
                    except Exception as e:
                        logger.error(f"Erreur surveillance {file_path}: {e}")

                time.sleep(self.watch_interval)

            except Exception as e:
                logger.error(f"Erreur dans thread surveillance: {e}")
                time.sleep(5)

    async def _reload_file(self, file_path: str):
        """Recharge un fichier de configuration"""
        try:
            logger.info(f"Rechargement de {file_path}")

            # Suppression des anciennes configurations de ce fichier
            keys_to_remove = [key for key, config in self.configs.items()
                            if config.source_file == file_path]

            for key in keys_to_remove:
                del self.configs[key]
                if key in self.cache:
                    del self.cache[key]

            # Rechargement
            await self.load_config_file(file_path, auto_watch=False)

        except Exception as e:
            logger.error(f"Erreur rechargement {file_path}: {e}")

    async def reload_all_configs(self):
        """Recharge toutes les configurations"""
        try:
            source_files = set(config.source_file for config in self.configs.values()
                             if config.source_file)

            for file_path in source_files:
                await self._reload_file(file_path)

            logger.info("Toutes les configurations rechargées")

        except Exception as e:
            logger.error(f"Erreur rechargement global: {e}")

    async def export_config(self, output_path: str, sections: Optional[List[str]] = None,
                           format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Exporte la configuration vers un fichier"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Préparation des données
            if sections:
                data = {}
                for section in sections:
                    data[section] = await self.get_config_section(section)
            else:
                data = {key: config.value for key, config in self.configs.items()}

            # Export selon le format
            if format == ConfigFormat.JSON:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            elif format == ConfigFormat.YAML:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Configuration exportée vers {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur export config: {e}")
            return False

    async def get_config_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de configuration"""
        try:
            stats = {
                "total_configs": len(self.configs),
                "cache_size": len(self.cache),
                "watched_files": len([w for w in self.watchers.values() if w.active]),
                "scopes": {},
                "formats": {},
                "last_reload": None
            }

            # Statistiques par scope
            for config in self.configs.values():
                scope = config.scope.value
                stats["scopes"][scope] = stats["scopes"].get(scope, 0) + 1

            # Statistiques par format
            for config in self.configs.values():
                format = config.format.value
                stats["formats"][format] = stats["formats"].get(format, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Erreur statistiques config: {e}")
            return {}

    def stop_watching(self):
        """Arrête la surveillance des fichiers"""
        try:
            self._stop_watching = True

            for watcher in self.watchers.values():
                watcher.active = False

            if self._watch_thread and self._watch_thread.is_alive():
                self._watch_thread.join(timeout=2)

            logger.info("Surveillance des fichiers arrêtée")

        except Exception as e:
            logger.error(f"Erreur arrêt surveillance: {e}")

# Instance globale
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """Récupère l'instance globale du gestionnaire de configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

# Fonctions utilitaires
async def load_dofus_configs(base_path: str = "data") -> bool:
    """Charge toutes les configurations DOFUS"""
    try:
        manager = get_config_manager()

        # Liste des fichiers de configuration
        config_files = [
            "resources/agricultural_resources.json",
            "monsters/monster_database.json",
            "maps/map_coordinates.json"
        ]

        success_count = 0
        for config_file in config_files:
            file_path = Path(base_path) / config_file
            if await manager.load_config_file(str(file_path)):
                success_count += 1

        logger.info(f"Configurations DOFUS chargées: {success_count}/{len(config_files)}")
        return success_count == len(config_files)

    except Exception as e:
        logger.error(f"Erreur chargement configs DOFUS: {e}")
        return False

async def get_resource_data(resource_id: str) -> Optional[Dict[str, Any]]:
    """Récupère les données d'une ressource"""
    manager = get_config_manager()

    # Recherche dans toutes les catégories agricoles
    categories = ["cereals", "vegetables", "fruits"]

    for category in categories:
        resource_data = await manager.get_config(
            f"categories.{category}.items.{resource_id}"
        )
        if resource_data:
            return resource_data

    return None

async def get_monster_data(monster_id: str) -> Optional[Dict[str, Any]]:
    """Récupère les données d'un monstre"""
    manager = get_config_manager()

    # Recherche dans toutes les catégories
    categories = ["low_level", "medium_level", "high_level"]

    for category in categories:
        monster_data = await manager.get_config(
            f"categories.{category}.monsters.{monster_id}"
        )
        if monster_data:
            return monster_data

    return None

async def get_map_data(map_coordinates: str) -> Optional[Dict[str, Any]]:
    """Récupère les données d'une carte"""
    manager = get_config_manager()

    # Recherche dans toutes les régions
    regions = await manager.get_config("regions", {})

    for region_data in regions.values():
        maps = region_data.get("maps", {})
        for map_data in maps.values():
            if str(map_data.get("coordinates")) == map_coordinates:
                return map_data

    return None