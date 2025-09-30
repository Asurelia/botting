"""
Connecteurs API Externes - DOFUS Unity World Model AI
Intégration avec Dofapi, Doduapi et autres sources de données DOFUS
"""

import time
import json
import requests
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3

@dataclass
class APIResponse:
    """Réponse d'API standardisée"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    api_source: str = ""
    timestamp: float = 0.0
    cache_hit: bool = False

@dataclass
class DofusItem:
    """Item DOFUS standardisé"""
    id: int
    name: str
    type: str
    level: int
    description: str
    stats: Dict[str, Any]
    image_url: Optional[str] = None
    market_price: Optional[int] = None

@dataclass
class DofusSpell:
    """Sort DOFUS standardisé"""
    id: int
    name: str
    class_name: str
    level: int
    ap_cost: int
    range_min: int
    range_max: int
    effects: List[str]
    cooldown: int
    description: str

class DofapiConnector:
    """Connecteur pour Dofapi.fr (API non-officielle DOFUS)"""

    def __init__(self):
        self.base_url = "https://api.dofapi.fr"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DOFUS-Unity-World-Model-AI/1.0',
            'Accept': 'application/json'
        })

        # Cache local
        self.cache = {}
        self.cache_duration = 3600  # 1 heure

    def get_items_by_category(self, category: str) -> APIResponse:
        """Récupère les items d'une catégorie"""
        try:
            cache_key = f"items_{category}"

            # Vérifier cache
            if self._is_cache_valid(cache_key):
                return APIResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    api_source="dofapi",
                    timestamp=time.time(),
                    cache_hit=True
                )

            # Requête API
            url = f"{self.base_url}/dofus2/fr/items/{category}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Normaliser les données
                items = []
                for item_data in data.get("data", []):
                    item = DofusItem(
                        id=item_data.get("id", 0),
                        name=item_data.get("name", ""),
                        type=category,
                        level=item_data.get("level", 0),
                        description=item_data.get("description", ""),
                        stats=item_data.get("stats", {}),
                        image_url=item_data.get("image_url")
                    )
                    items.append(asdict(item))

                # Mettre en cache
                self._cache_data(cache_key, items)

                return APIResponse(
                    success=True,
                    data=items,
                    api_source="dofapi",
                    timestamp=time.time()
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    api_source="dofapi",
                    timestamp=time.time()
                )

        except requests.RequestException as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Erreur réseau: {str(e)}",
                api_source="dofapi",
                timestamp=time.time()
            )

    def get_spells_by_class(self, class_name: str) -> APIResponse:
        """Récupère les sorts d'une classe"""
        try:
            cache_key = f"spells_{class_name.lower()}"

            if self._is_cache_valid(cache_key):
                return APIResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    api_source="dofapi",
                    timestamp=time.time(),
                    cache_hit=True
                )

            url = f"{self.base_url}/dofus2/fr/classes/{class_name.lower()}/spells"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                spells = []
                for spell_data in data.get("data", []):
                    spell = DofusSpell(
                        id=spell_data.get("id", 0),
                        name=spell_data.get("name", ""),
                        class_name=class_name,
                        level=spell_data.get("level", 1),
                        ap_cost=spell_data.get("apCost", 0),
                        range_min=spell_data.get("minRange", 0),
                        range_max=spell_data.get("maxRange", 0),
                        effects=spell_data.get("effects", []),
                        cooldown=spell_data.get("cooldown", 0),
                        description=spell_data.get("description", "")
                    )
                    spells.append(asdict(spell))

                self._cache_data(cache_key, spells)

                return APIResponse(
                    success=True,
                    data=spells,
                    api_source="dofapi",
                    timestamp=time.time()
                )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="dofapi",
                timestamp=time.time()
            )

    def search_items(self, query: str) -> APIResponse:
        """Recherche d'items par nom"""
        try:
            cache_key = f"search_{query.lower()}"

            if self._is_cache_valid(cache_key):
                return APIResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    api_source="dofapi",
                    timestamp=time.time(),
                    cache_hit=True
                )

            url = f"{self.base_url}/dofus2/fr/items/search"
            params = {"name": query}
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                items = []
                for item_data in data.get("data", []):
                    item = DofusItem(
                        id=item_data.get("id", 0),
                        name=item_data.get("name", ""),
                        type=item_data.get("type", "unknown"),
                        level=item_data.get("level", 0),
                        description=item_data.get("description", ""),
                        stats=item_data.get("stats", {}),
                        image_url=item_data.get("image_url")
                    )
                    items.append(asdict(item))

                self._cache_data(cache_key, items)

                return APIResponse(
                    success=True,
                    data=items,
                    api_source="dofapi",
                    timestamp=time.time()
                )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="dofapi",
                timestamp=time.time()
            )

    def _is_cache_valid(self, key: str) -> bool:
        """Vérifie si les données en cache sont encore valides"""
        if key not in self.cache:
            return False

        cache_time = self.cache[key]["timestamp"]
        return (time.time() - cache_time) < self.cache_duration

    def _cache_data(self, key: str, data: Any):
        """Met en cache les données"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

class DoduapiConnector:
    """Connecteur pour Doduapi (api.dofusdu.de)"""

    def __init__(self):
        self.base_url = "https://api.dofusdu.de"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DOFUS-Unity-World-Model-AI/1.0',
            'Accept': 'application/json'
        })

        self.cache = {}
        self.cache_duration = 3600

    def get_all_items(self, language: str = "en") -> APIResponse:
        """Récupère tous les items"""
        try:
            cache_key = f"all_items_{language}"

            if self._is_cache_valid(cache_key):
                return APIResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    api_source="doduapi",
                    timestamp=time.time(),
                    cache_hit=True
                )

            url = f"{self.base_url}/dofus2/{language}/items/equipment/all"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()

                # Transformer au format standardisé
                items = []
                for item_data in data:
                    item = DofusItem(
                        id=item_data.get("ankama_id", 0),
                        name=item_data.get("name", ""),
                        type=item_data.get("type", {}).get("name", "unknown"),
                        level=item_data.get("level", 0),
                        description=item_data.get("description", ""),
                        stats=item_data.get("stats", []),
                        image_url=item_data.get("image_urls", {}).get("icon")
                    )
                    items.append(asdict(item))

                self._cache_data(cache_key, items)

                return APIResponse(
                    success=True,
                    data=items,
                    api_source="doduapi",
                    timestamp=time.time()
                )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="doduapi",
                timestamp=time.time()
            )

    def get_sets(self, language: str = "en") -> APIResponse:
        """Récupère tous les sets d'équipements"""
        try:
            cache_key = f"sets_{language}"

            if self._is_cache_valid(cache_key):
                return APIResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    api_source="doduapi",
                    timestamp=time.time(),
                    cache_hit=True
                )

            url = f"{self.base_url}/dofus2/{language}/sets/all"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                self._cache_data(cache_key, data)

                return APIResponse(
                    success=True,
                    data=data,
                    api_source="doduapi",
                    timestamp=time.time()
                )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="doduapi",
                timestamp=time.time()
            )

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self.cache:
            return False
        cache_time = self.cache[key]["timestamp"]
        return (time.time() - cache_time) < self.cache_duration

    def _cache_data(self, key: str, data: Any):
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

class UnifiedDofusAPIManager:
    """Gestionnaire unifié des APIs DOFUS externes"""

    def __init__(self, database_path: str = "external_apis_cache.db"):
        self.database_path = database_path
        self.dofapi = DofapiConnector()
        self.doduapi = DoduapiConnector()

        # Cache persistant SQLite
        self._init_database()

        # Statistiques
        self.stats = {
            "dofapi_requests": 0,
            "doduapi_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }

    def _init_database(self):
        """Initialise la base de données cache"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    api_source TEXT,
                    timestamp REAL,
                    expires_at REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_stats (
                    date TEXT,
                    api_source TEXT,
                    requests INTEGER,
                    cache_hits INTEGER,
                    errors INTEGER,
                    PRIMARY KEY (date, api_source)
                )
            """)

    def get_comprehensive_item_data(self, item_name: str) -> APIResponse:
        """Récupère des données d'item complètes depuis plusieurs sources"""
        try:
            # Essayer Dofapi en premier
            dofapi_response = self.dofapi.search_items(item_name)
            self.stats["dofapi_requests"] += 1

            if dofapi_response.success and dofapi_response.data:
                # Enrichir avec données Doduapi si possible
                try:
                    doduapi_response = self.doduapi.get_all_items()
                    self.stats["doduapi_requests"] += 1

                    if doduapi_response.success:
                        # Fusionner les données
                        dofapi_items = dofapi_response.data
                        doduapi_items = doduapi_response.data

                        # Enrichir avec informations complémentaires
                        enriched_items = self._merge_item_data(dofapi_items, doduapi_items, item_name)

                        return APIResponse(
                            success=True,
                            data=enriched_items,
                            api_source="unified",
                            timestamp=time.time()
                        )

                except Exception:
                    pass  # Fallback sur Dofapi seulement

                return dofapi_response

            # Fallback sur Doduapi
            doduapi_response = self.doduapi.get_all_items()
            self.stats["doduapi_requests"] += 1

            if doduapi_response.success:
                # Filtrer par nom
                filtered_items = [
                    item for item in doduapi_response.data
                    if item_name.lower() in item.get("name", "").lower()
                ]

                return APIResponse(
                    success=True,
                    data=filtered_items,
                    api_source="doduapi",
                    timestamp=time.time()
                )

            return APIResponse(
                success=False,
                data=None,
                error_message="Toutes les APIs ont échoué",
                api_source="unified",
                timestamp=time.time()
            )

        except Exception as e:
            self.stats["errors"] += 1
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="unified",
                timestamp=time.time()
            )

    def get_class_spells_comprehensive(self, class_name: str) -> APIResponse:
        """Récupère les sorts d'une classe avec données complètes"""
        try:
            # Priorité à Dofapi pour les sorts
            response = self.dofapi.get_spells_by_class(class_name)
            self.stats["dofapi_requests"] += 1

            if response.success:
                return response

            # TODO: Ajouter d'autres sources pour les sorts si nécessaire

            return APIResponse(
                success=False,
                data=None,
                error_message=f"Impossible de récupérer les sorts pour {class_name}",
                api_source="unified",
                timestamp=time.time()
            )

        except Exception as e:
            self.stats["errors"] += 1
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                api_source="unified",
                timestamp=time.time()
            )

    def _merge_item_data(self, dofapi_items: List[Dict],
                        doduapi_items: List[Dict], search_term: str) -> List[Dict]:
        """Fusionne les données d'items de différentes sources"""
        merged_items = []

        for dofapi_item in dofapi_items:
            # Chercher correspondance dans doduapi
            matching_doduapi = None
            for doduapi_item in doduapi_items:
                if (dofapi_item.get("name", "").lower() ==
                    doduapi_item.get("name", "").lower()):
                    matching_doduapi = doduapi_item
                    break

            # Fusionner les données
            merged_item = dofapi_item.copy()

            if matching_doduapi:
                # Ajouter informations supplémentaires de doduapi
                if "image_urls" in matching_doduapi:
                    merged_item["image_url"] = matching_doduapi["image_urls"].get("icon")

                if "stats" in matching_doduapi and isinstance(matching_doduapi["stats"], list):
                    merged_item["detailed_stats"] = matching_doduapi["stats"]

            merged_items.append(merged_item)

        return merged_items

    def update_cache_from_apis(self) -> Dict[str, Any]:
        """Met à jour le cache depuis toutes les APIs"""
        print("🔄 Mise à jour du cache depuis les APIs externes...")

        update_stats = {
            "items_updated": 0,
            "spells_updated": 0,
            "errors": 0,
            "duration": 0
        }

        start_time = time.time()

        try:
            # Classes DOFUS principales
            classes = ["iop", "cra", "eniripsa", "enutrof", "sram", "xelor",
                      "ecaflip", "sacrieur", "sadida", "osamodas", "pandawa", "roublard"]

            # Mettre à jour sorts par classe
            for class_name in classes:
                try:
                    response = self.dofapi.get_spells_by_class(class_name)
                    if response.success:
                        update_stats["spells_updated"] += len(response.data)
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    update_stats["errors"] += 1
                    print(f"❌ Erreur classe {class_name}: {e}")

            # Mettre à jour items populaires
            item_categories = ["equipment", "weapons", "consumables", "resources"]
            for category in item_categories:
                try:
                    response = self.dofapi.get_items_by_category(category)
                    if response.success:
                        update_stats["items_updated"] += len(response.data)
                    time.sleep(0.5)
                except Exception as e:
                    update_stats["errors"] += 1
                    print(f"❌ Erreur catégorie {category}: {e}")

            update_stats["duration"] = time.time() - start_time

            print(f"✅ Mise à jour cache terminée en {update_stats['duration']:.1f}s")
            print(f"📊 Items: {update_stats['items_updated']}, Sorts: {update_stats['spells_updated']}")

            return update_stats

        except Exception as e:
            update_stats["errors"] += 1
            update_stats["duration"] = time.time() - start_time
            print(f"❌ Erreur mise à jour cache: {e}")
            return update_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation des APIs"""
        return {
            "current_session": self.stats,
            "cache_size": len(self.dofapi.cache) + len(self.doduapi.cache),
            "last_update": datetime.now().isoformat()
        }

# Factory functions
def get_dofapi_connector() -> DofapiConnector:
    """Factory pour Dofapi"""
    return DofapiConnector()

def get_doduapi_connector() -> DoduapiConnector:
    """Factory pour Doduapi"""
    return DoduapiConnector()

def get_unified_api_manager(database_path: str = "external_apis_cache.db") -> UnifiedDofusAPIManager:
    """Factory pour le gestionnaire unifié"""
    return UnifiedDofusAPIManager(database_path)

if __name__ == "__main__":
    # Test des connecteurs
    print("🧪 Test des Connecteurs API DOFUS")
    print("=" * 50)

    # Test Dofapi
    print("\n📡 Test Dofapi...")
    dofapi = get_dofapi_connector()

    response = dofapi.get_spells_by_class("iop")
    if response.success:
        print(f"✅ Sorts Iop récupérés: {len(response.data)} sorts")
    else:
        print(f"❌ Erreur Dofapi: {response.error_message}")

    # Test Doduapi
    print("\n📡 Test Doduapi...")
    doduapi = get_doduapi_connector()

    response = doduapi.get_sets()
    if response.success:
        print(f"✅ Sets récupérés: {len(response.data)} sets")
    else:
        print(f"❌ Erreur Doduapi: {response.error_message}")

    # Test gestionnaire unifié
    print("\n🔗 Test Gestionnaire Unifié...")
    manager = get_unified_api_manager()

    response = manager.get_comprehensive_item_data("Dofus")
    if response.success:
        print(f"✅ Items 'Dofus' trouvés: {len(response.data)} items")
    else:
        print(f"❌ Erreur unified: {response.error_message}")

    print(f"\n📊 Statistiques: {manager.get_statistics()}")