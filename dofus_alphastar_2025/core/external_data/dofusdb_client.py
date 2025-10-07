#!/usr/bin/env python3
"""
DofusDBClient - Client API pour DofusDB
Accès à la base de données complète du jeu (items, sorts, monstres, etc.)

API: https://api.dofusdb.fr/
Documentation: https://docs.dofusdb.fr/
"""

import requests
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

@dataclass
class ItemData:
    """Données d'un item"""
    id: int
    name: str
    type: str
    level: int
    description: Optional[str] = None
    effects: Optional[List[Dict]] = None
    conditions: Optional[List[str]] = None
    recipe: Optional[Dict] = None
    image_url: Optional[str] = None

@dataclass
class SpellData:
    """Données d'un sort"""
    id: int
    name: str
    class_name: str
    type: str
    element: str
    levels: List[Dict]  # Données par niveau
    description: Optional[str] = None
    image_url: Optional[str] = None

@dataclass
class MonsterData:
    """Données d'un monstre"""
    id: int
    name: str
    level: int
    hp: int
    resistances: Dict[str, int]  # {element: resistance}
    drops: List[Dict]
    areas: List[str]
    image_url: Optional[str] = None

class DofusDBClient:
    """
    Client pour l'API DofusDB

    Features:
    - Cache intelligent (évite requêtes répétées)
    - Rate limiting (respecte les limites API)
    - Fallback sur cache si API indisponible
    - Support offline avec cache local
    """

    def __init__(self, cache_dir: str = "cache/dofusdb", rate_limit_delay: float = 0.1):
        self.logger = logging.getLogger(__name__)

        # Configuration API
        self.base_url = "https://api.dofusdb.fr"
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}

        # Statistiques
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

    def get_item(self, item_id: int) -> Optional[ItemData]:
        """
        Récupère les informations complètes d'un item

        Args:
            item_id: ID de l'item dans Dofus

        Returns:
            ItemData ou None si non trouvé
        """
        cache_key = f"item_{item_id}"

        # Vérifie cache mémoire
        if cache_key in self.memory_cache:
            self.stats['cache_hits'] += 1
            return self.memory_cache[cache_key]

        # Vérifie cache disque
        cached = self._load_from_cache(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            item = ItemData(**cached)
            self.memory_cache[cache_key] = item
            return item

        # Requête API
        self.stats['cache_misses'] += 1
        try:
            self._respect_rate_limit()

            response = requests.get(
                f"{self.base_url}/items/{item_id}",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.stats['requests'] += 1

            # Parse et crée ItemData
            item = ItemData(
                id=data.get('id'),
                name=data.get('name'),
                type=data.get('type'),
                level=data.get('level', 0),
                description=data.get('description'),
                effects=data.get('effects'),
                conditions=data.get('conditions'),
                recipe=data.get('recipe'),
                image_url=data.get('imageUrl')
            )

            # Sauvegarde en cache
            self._save_to_cache(cache_key, asdict(item))
            self.memory_cache[cache_key] = item

            self.logger.debug(f"Item {item_id} récupéré: {item.name}")

            return item

        except Exception as e:
            self.logger.error(f"Erreur récupération item {item_id}: {e}")
            self.stats['errors'] += 1
            return None

    def get_spell(self, spell_id: int) -> Optional[SpellData]:
        """Récupère les informations d'un sort"""
        cache_key = f"spell_{spell_id}"

        # Cache
        if cache_key in self.memory_cache:
            self.stats['cache_hits'] += 1
            return self.memory_cache[cache_key]

        cached = self._load_from_cache(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            spell = SpellData(**cached)
            self.memory_cache[cache_key] = spell
            return spell

        # API
        self.stats['cache_misses'] += 1
        try:
            self._respect_rate_limit()

            response = requests.get(
                f"{self.base_url}/spells/{spell_id}",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.stats['requests'] += 1

            spell = SpellData(
                id=data.get('id'),
                name=data.get('name'),
                class_name=data.get('class'),
                type=data.get('type'),
                element=data.get('element'),
                levels=data.get('levels', []),
                description=data.get('description'),
                image_url=data.get('imageUrl')
            )

            self._save_to_cache(cache_key, asdict(spell))
            self.memory_cache[cache_key] = spell

            self.logger.debug(f"Sort {spell_id} récupéré: {spell.name}")

            return spell

        except Exception as e:
            self.logger.error(f"Erreur récupération sort {spell_id}: {e}")
            self.stats['errors'] += 1
            return None

    def get_monster(self, monster_id: int) -> Optional[MonsterData]:
        """Récupère les informations d'un monstre"""
        cache_key = f"monster_{monster_id}"

        # Cache
        if cache_key in self.memory_cache:
            self.stats['cache_hits'] += 1
            return self.memory_cache[cache_key]

        cached = self._load_from_cache(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            monster = MonsterData(**cached)
            self.memory_cache[cache_key] = monster
            return monster

        # API
        self.stats['cache_misses'] += 1
        try:
            self._respect_rate_limit()

            response = requests.get(
                f"{self.base_url}/monsters/{monster_id}",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.stats['requests'] += 1

            monster = MonsterData(
                id=data.get('id'),
                name=data.get('name'),
                level=data.get('level', 1),
                hp=data.get('hp', 0),
                resistances=data.get('resistances', {}),
                drops=data.get('drops', []),
                areas=data.get('areas', []),
                image_url=data.get('imageUrl')
            )

            self._save_to_cache(cache_key, asdict(monster))
            self.memory_cache[cache_key] = monster

            self.logger.debug(f"Monstre {monster_id} récupéré: {monster.name}")

            return monster

        except Exception as e:
            self.logger.error(f"Erreur récupération monstre {monster_id}: {e}")
            self.stats['errors'] += 1
            return None

    def search_items(self, query: str, item_type: Optional[str] = None,
                     limit: int = 20) -> List[ItemData]:
        """
        Recherche d'items par nom

        Args:
            query: Texte à rechercher
            item_type: Type d'item (arme, armure, etc.)
            limit: Nombre max de résultats

        Returns:
            Liste d'ItemData
        """
        try:
            self._respect_rate_limit()

            params = {
                'q': query,
                'limit': limit
            }

            if item_type:
                params['type'] = item_type

            response = requests.get(
                f"{self.base_url}/items/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            results = response.json()
            self.stats['requests'] += 1

            items = []
            for data in results.get('results', []):
                item = ItemData(
                    id=data.get('id'),
                    name=data.get('name'),
                    type=data.get('type'),
                    level=data.get('level', 0),
                    description=data.get('description'),
                    image_url=data.get('imageUrl')
                )
                items.append(item)

            self.logger.info(f"Recherche '{query}': {len(items)} résultats")

            return items

        except Exception as e:
            self.logger.error(f"Erreur recherche items: {e}")
            return []

    def get_spell_damage(self, spell_id: int, level: int, target_resistances: Dict[str, int]) -> float:
        """
        Calcule les dégâts effectifs d'un sort

        Args:
            spell_id: ID du sort
            level: Niveau du sort
            target_resistances: Résistances de la cible {element: %}

        Returns:
            Dégâts effectifs estimés
        """
        spell = self.get_spell(spell_id)
        if not spell or level >= len(spell.levels):
            return 0.0

        spell_level_data = spell.levels[level]
        base_damage = spell_level_data.get('damage', 0)

        # Applique résistances
        element = spell.element
        resistance = target_resistances.get(element, 0)

        # Formule simplifiée: damage * (1 - resistance/100)
        effective_damage = base_damage * (1 - resistance / 100)

        return max(0, effective_damage)

    def get_recipe(self, item_id: int) -> Optional[Dict]:
        """Récupère la recette de craft d'un item"""
        item = self.get_item(item_id)
        if item and item.recipe:
            return item.recipe
        return None

    def _respect_rate_limit(self):
        """Respecte le rate limit de l'API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        self.last_request_time = time.time()

    def _save_to_cache(self, key: str, data: Dict):
        """Sauvegarde dans le cache disque"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.debug(f"Erreur sauvegarde cache {key}: {e}")

    def _load_from_cache(self, key: str) -> Optional[Dict]:
        """Charge depuis le cache disque"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        except Exception as e:
            self.logger.debug(f"Erreur chargement cache {key}: {e}")

        return None

    def clear_cache(self):
        """Efface tout le cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory_cache.clear()
        self.logger.info("Cache effacé")

    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques d'utilisation"""
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_ratio = (self.stats['cache_hits'] / total * 100) if total > 0 else 0

        return {
            **self.stats,
            'cache_ratio': f"{cache_ratio:.1f}%"
        }

def create_dofusdb_client(cache_dir: str = "cache/dofusdb",
                          rate_limit_delay: float = 0.1) -> DofusDBClient:
    """Factory function pour créer un client DofusDB"""
    return DofusDBClient(cache_dir, rate_limit_delay)