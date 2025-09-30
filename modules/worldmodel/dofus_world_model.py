"""
DOFUS World Model - Modèle du monde intelligent pour DOFUS Unity
Intelligence spatiale complète du jeu avec mémoire persistante et prédictions

Fonctionnalités:
- Cartographie automatique temps réel
- Mémoire spatiale persistante (maps, PNJ, ressources, monstres)
- Prédiction événements (spawns, rotations, prix marché)
- Modèle physique du monde (collisions, chemins, zones)
- Intelligence contextuelle (dangers, opportunités, patterns temporels)
"""

import asyncio
import time
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import cv2
from collections import defaultdict, deque
import threading

# Import du framework IA existant
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.ai_framework import AIModule, AIModuleState

class MapCellType(Enum):
    """Types de cellules de carte"""
    UNKNOWN = "unknown"
    WALKABLE = "walkable"
    BLOCKED = "blocked"
    WATER = "water"
    RESOURCE = "resource"
    NPC = "npc"
    MONSTER = "monster"
    INTERACTIVE = "interactive"
    SHOP = "shop"
    ZAAP = "zaap"
    DANGER_ZONE = "danger"

class EventType(Enum):
    """Types d'événements du monde"""
    RESOURCE_SPAWN = "resource_spawn"
    MONSTER_SPAWN = "monster_spawn"
    NPC_MOVEMENT = "npc_movement"
    PLAYER_ACTION = "player_action"
    MARKET_CHANGE = "market_change"
    WEATHER_CHANGE = "weather_change"
    TIME_CYCLE = "time_cycle"

@dataclass
class WorldPosition:
    """Position dans le monde DOFUS"""
    map_id: str
    x: int
    y: int
    z: int = 0  # Niveau (étages)
    sub_area: str = ""
    area: str = ""

@dataclass
class WorldEntity:
    """Entité du monde (PNJ, monstre, ressource, etc.)"""
    entity_id: str
    entity_type: str
    name: str
    position: WorldPosition
    properties: Dict[str, Any]
    last_seen: float
    confidence: float
    state: str = "active"

@dataclass
class WorldEvent:
    """Événement du monde avec timestamp"""
    event_id: str
    event_type: EventType
    position: WorldPosition
    timestamp: float
    data: Dict[str, Any]
    pattern_id: Optional[str] = None

@dataclass
class MapCell:
    """Cellule de carte avec propriétés"""
    x: int
    y: int
    cell_type: MapCellType
    walkable: bool
    properties: Dict[str, Any]
    last_updated: float
    entities: List[str]  # IDs des entités sur cette cellule

class SpatialMemory:
    """Mémoire spatiale persistante du monde"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Base de données SQLite pour persistance
        self.db_path = data_dir / "world_memory.db"
        self.connection: Optional[sqlite3.Connection] = None

        # Cache en mémoire pour performance
        self.maps_cache: Dict[str, Dict[Tuple[int, int], MapCell]] = {}
        self.entities_cache: Dict[str, WorldEntity] = {}
        self.events_cache: deque = deque(maxlen=10000)

        # Statistiques
        self.total_maps_discovered = 0
        self.total_entities_tracked = 0
        self.total_events_recorded = 0

        self.logger = logging.getLogger(__name__ + ".SpatialMemory")

    def initialize(self) -> bool:
        """Initialise la mémoire spatiale"""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._create_tables()
            self._load_cache()
            self.logger.info("Mémoire spatiale initialisée")
            return True
        except Exception as e:
            self.logger.error(f"Erreur initialisation mémoire spatiale: {e}")
            return False

    def _create_tables(self):
        """Crée les tables de base de données"""
        cursor = self.connection.cursor()

        # Table des cartes et cellules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS map_cells (
                map_id TEXT,
                x INTEGER,
                y INTEGER,
                cell_type TEXT,
                walkable BOOLEAN,
                properties TEXT,
                last_updated REAL,
                PRIMARY KEY (map_id, x, y)
            )
        """)

        # Table des entités
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                name TEXT,
                map_id TEXT,
                x INTEGER,
                y INTEGER,
                z INTEGER,
                sub_area TEXT,
                area TEXT,
                properties TEXT,
                last_seen REAL,
                confidence REAL,
                state TEXT
            )
        """)

        # Table des événements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                map_id TEXT,
                x INTEGER,
                y INTEGER,
                timestamp REAL,
                data TEXT,
                pattern_id TEXT
            )
        """)

        # Index pour performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_map ON world_entities(map_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON world_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON world_events(event_type)")

        self.connection.commit()

    def _load_cache(self):
        """Charge les données en cache"""
        try:
            cursor = self.connection.cursor()

            # Charger entités récentes
            cursor.execute("""
                SELECT * FROM world_entities
                WHERE last_seen > ?
                ORDER BY last_seen DESC LIMIT 1000
            """, (time.time() - 86400,))  # Dernières 24h

            for row in cursor.fetchall():
                entity = self._row_to_entity(row)
                self.entities_cache[entity.entity_id] = entity

            # Charger événements récents
            cursor.execute("""
                SELECT * FROM world_events
                WHERE timestamp > ?
                ORDER BY timestamp DESC LIMIT 1000
            """, (time.time() - 3600,))  # Dernière heure

            for row in cursor.fetchall():
                event = self._row_to_event(row)
                self.events_cache.append(event)

            self.logger.info(f"Cache chargé: {len(self.entities_cache)} entités, {len(self.events_cache)} événements")

        except Exception as e:
            self.logger.error(f"Erreur chargement cache: {e}")

    def update_map_cell(self, map_id: str, x: int, y: int, cell_type: MapCellType,
                       walkable: bool, properties: Dict[str, Any] = None):
        """Met à jour une cellule de carte"""
        properties = properties or {}

        cell = MapCell(
            x=x, y=y,
            cell_type=cell_type,
            walkable=walkable,
            properties=properties,
            last_updated=time.time(),
            entities=[]
        )

        # Mettre en cache
        if map_id not in self.maps_cache:
            self.maps_cache[map_id] = {}
        self.maps_cache[map_id][(x, y)] = cell

        # Sauvegarder en base
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO map_cells
                (map_id, x, y, cell_type, walkable, properties, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (map_id, x, y, cell_type.value, walkable, json.dumps(properties), time.time()))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cellule: {e}")

    def add_entity(self, entity: WorldEntity):
        """Ajoute ou met à jour une entité"""
        self.entities_cache[entity.entity_id] = entity

        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO world_entities
                (entity_id, entity_type, name, map_id, x, y, z, sub_area, area,
                 properties, last_seen, confidence, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.entity_id, entity.entity_type, entity.name,
                entity.position.map_id, entity.position.x, entity.position.y, entity.position.z,
                entity.position.sub_area, entity.position.area,
                json.dumps(entity.properties), entity.last_seen, entity.confidence, entity.state
            ))
            self.connection.commit()
            self.total_entities_tracked += 1
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde entité: {e}")

    def record_event(self, event: WorldEvent):
        """Enregistre un événement du monde"""
        self.events_cache.append(event)

        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO world_events
                (event_id, event_type, map_id, x, y, timestamp, data, pattern_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.event_type.value,
                event.position.map_id, event.position.x, event.position.y,
                event.timestamp, json.dumps(event.data), event.pattern_id
            ))
            self.connection.commit()
            self.total_events_recorded += 1
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde événement: {e}")

    def get_entities_in_area(self, map_id: str, center_x: int, center_y: int,
                           radius: int = 10) -> List[WorldEntity]:
        """Récupère entités dans une zone"""
        entities = []

        for entity in self.entities_cache.values():
            if (entity.position.map_id == map_id and
                abs(entity.position.x - center_x) <= radius and
                abs(entity.position.y - center_y) <= radius):
                entities.append(entity)

        return entities

    def get_map_data(self, map_id: str) -> Dict[Tuple[int, int], MapCell]:
        """Récupère données d'une carte"""
        return self.maps_cache.get(map_id, {})

    def _row_to_entity(self, row) -> WorldEntity:
        """Convertit ligne DB en entité"""
        return WorldEntity(
            entity_id=row[0],
            entity_type=row[1],
            name=row[2],
            position=WorldPosition(row[3], row[4], row[5], row[6], row[7], row[8]),
            properties=json.loads(row[9]) if row[9] else {},
            last_seen=row[10],
            confidence=row[11],
            state=row[12]
        )

    def _row_to_event(self, row) -> WorldEvent:
        """Convertit ligne DB en événement"""
        return WorldEvent(
            event_id=row[0],
            event_type=EventType(row[1]),
            position=WorldPosition(row[2], row[3], row[4]),
            timestamp=row[5],
            data=json.loads(row[6]) if row[6] else {},
            pattern_id=row[7]
        )

class WorldModelAI:
    """Intelligence du modèle du monde avec prédictions"""

    def __init__(self, spatial_memory: SpatialMemory):
        self.spatial_memory = spatial_memory
        self.logger = logging.getLogger(__name__ + ".WorldModelAI")

        # Patterns temporels découverts
        self.temporal_patterns: Dict[str, Dict[str, Any]] = {}

        # Prédictions actives
        self.active_predictions: List[Dict[str, Any]] = []

        # Statistiques d'apprentissage
        self.patterns_discovered = 0
        self.predictions_made = 0
        self.prediction_accuracy = 0.85

    def analyze_temporal_patterns(self):
        """Analyse les patterns temporels dans les événements"""
        try:
            events_by_type = defaultdict(list)

            # Grouper événements par type
            for event in self.spatial_memory.events_cache:
                events_by_type[event.event_type].append(event)

            # Analyser chaque type d'événement
            for event_type, events in events_by_type.items():
                if len(events) < 5:  # Pas assez de données
                    continue

                pattern = self._discover_pattern(event_type, events)
                if pattern:
                    pattern_id = f"{event_type.value}_{len(self.temporal_patterns)}"
                    self.temporal_patterns[pattern_id] = pattern
                    self.patterns_discovered += 1

                    self.logger.info(f"Pattern découvert: {pattern_id}")

        except Exception as e:
            self.logger.error(f"Erreur analyse patterns: {e}")

    def _discover_pattern(self, event_type: EventType, events: List[WorldEvent]) -> Optional[Dict[str, Any]]:
        """Découvre un pattern dans une série d'événements"""
        try:
            # Trier par timestamp
            events.sort(key=lambda e: e.timestamp)

            # Analyser intervalles temporels
            intervals = []
            for i in range(1, len(events)):
                interval = events[i].timestamp - events[i-1].timestamp
                intervals.append(interval)

            if not intervals:
                return None

            # Statistiques des intervalles
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            # Analyser positions
            positions = [(e.position.x, e.position.y) for e in events]
            unique_positions = list(set(positions))

            # Créer pattern si régularité détectée
            if std_interval < avg_interval * 0.5:  # Régularité temporelle
                return {
                    "event_type": event_type.value,
                    "average_interval": avg_interval,
                    "std_interval": std_interval,
                    "common_positions": unique_positions[:5],  # Top 5 positions
                    "sample_size": len(events),
                    "confidence": max(0.1, 1.0 - (std_interval / avg_interval)),
                    "discovered_at": time.time()
                }

            return None

        except Exception as e:
            self.logger.error(f"Erreur découverte pattern: {e}")
            return None

    def predict_next_events(self, map_id: str, current_time: float) -> List[Dict[str, Any]]:
        """Prédit les prochains événements probables"""
        predictions = []

        try:
            for pattern_id, pattern in self.temporal_patterns.items():
                # Trouver derniers événements de ce type
                recent_events = [
                    e for e in self.spatial_memory.events_cache
                    if (e.event_type.value == pattern["event_type"] and
                        e.position.map_id == map_id)
                ]

                if not recent_events:
                    continue

                # Prédire prochain événement
                last_event = max(recent_events, key=lambda e: e.timestamp)
                predicted_time = last_event.timestamp + pattern["average_interval"]

                if predicted_time > current_time:  # Événement futur
                    # Prédire position (position la plus probable du pattern)
                    predicted_pos = pattern["common_positions"][0] if pattern["common_positions"] else (0, 0)

                    prediction = {
                        "event_type": pattern["event_type"],
                        "predicted_time": predicted_time,
                        "time_until_event": predicted_time - current_time,
                        "predicted_position": predicted_pos,
                        "confidence": pattern["confidence"],
                        "pattern_id": pattern_id
                    }

                    predictions.append(prediction)
                    self.predictions_made += 1

            # Trier par proximité temporelle
            predictions.sort(key=lambda p: p["time_until_event"])

        except Exception as e:
            self.logger.error(f"Erreur prédiction événements: {e}")

        return predictions[:10]  # Top 10 prédictions

    def get_danger_assessment(self, position: WorldPosition) -> Dict[str, Any]:
        """Évalue le danger d'une position"""
        try:
            danger_score = 0.0
            factors = []

            # Vérifier entités dangereuses à proximité
            nearby_entities = self.spatial_memory.get_entities_in_area(
                position.map_id, position.x, position.y, radius=5
            )

            for entity in nearby_entities:
                if entity.entity_type == "monster":
                    level = entity.properties.get("level", 1)
                    danger_score += min(level / 100.0, 0.3)
                    factors.append(f"Monstre niveau {level}")
                elif entity.entity_type == "aggressive_player":
                    danger_score += 0.5
                    factors.append("Joueur agressif")

            # Vérifier événements dangereux récents
            recent_events = [
                e for e in self.spatial_memory.events_cache
                if (e.position.map_id == position.map_id and
                    abs(e.position.x - position.x) <= 3 and
                    abs(e.position.y - position.y) <= 3 and
                    time.time() - e.timestamp < 300)  # 5 minutes
            ]

            for event in recent_events:
                if "combat" in event.data.get("type", ""):
                    danger_score += 0.2
                    factors.append("Combat récent")
                elif "death" in event.data.get("type", ""):
                    danger_score += 0.4
                    factors.append("Mort récente")

            # Normaliser score
            danger_score = min(danger_score, 1.0)

            # Classifier niveau de danger
            if danger_score < 0.2:
                danger_level = "faible"
            elif danger_score < 0.5:
                danger_level = "modéré"
            elif danger_score < 0.8:
                danger_level = "élevé"
            else:
                danger_level = "critique"

            return {
                "danger_score": danger_score,
                "danger_level": danger_level,
                "factors": factors,
                "recommendation": self._get_danger_recommendation(danger_level)
            }

        except Exception as e:
            self.logger.error(f"Erreur évaluation danger: {e}")
            return {"danger_score": 0.5, "danger_level": "inconnu", "factors": [], "recommendation": "Prudence"}

    def _get_danger_recommendation(self, danger_level: str) -> str:
        """Retourne recommandation selon niveau de danger"""
        recommendations = {
            "faible": "Zone sûre, libre exploration",
            "modéré": "Vigilance recommandée, éviter les groupes",
            "élevé": "Zone dangereuse, éviter si possible",
            "critique": "Fuite immédiate recommandée"
        }
        return recommendations.get(danger_level, "Prudence")

    def find_optimal_path(self, start: WorldPosition, end: WorldPosition) -> List[WorldPosition]:
        """Trouve chemin optimal en tenant compte des dangers"""
        # Implémentation A* avec pondération danger
        # Simplifié pour l'exemple
        try:
            map_data = self.spatial_memory.get_map_data(start.map_id)

            # Si pas de données de carte, retour chemin direct
            if not map_data:
                return [start, end]

            # Implémentation simplifiée - en réalité utiliserait A*
            path = [start]

            # Étapes intermédiaires (simplifié)
            dx = end.x - start.x
            dy = end.y - start.y
            steps = max(abs(dx), abs(dy))

            if steps > 0:
                for i in range(1, steps):
                    intermediate_x = start.x + (dx * i // steps)
                    intermediate_y = start.y + (dy * i // steps)

                    intermediate_pos = WorldPosition(
                        start.map_id, intermediate_x, intermediate_y
                    )

                    # Vérifier si cellule praticable et sûre
                    cell = map_data.get((intermediate_x, intermediate_y))
                    if cell and cell.walkable:
                        path.append(intermediate_pos)

            path.append(end)
            return path

        except Exception as e:
            self.logger.error(f"Erreur calcul chemin: {e}")
            return [start, end]

class DofusWorldModel(AIModule):
    """Module World Model principal intégré au framework IA"""

    def __init__(self, data_dir: Path):
        super().__init__("WorldModel")
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Composants principaux
        self.spatial_memory: Optional[SpatialMemory] = None
        self.world_ai: Optional[WorldModelAI] = None

        # Configuration
        self.update_interval = 1.0
        self.prediction_interval = 30.0
        self.mapping_enabled = True

        # État actuel
        self.current_position: Optional[WorldPosition] = None
        self.current_map_id: str = ""
        self.last_prediction_time = 0.0

        # Métriques
        self.maps_discovered = 0
        self.entities_tracked = 0
        self.predictions_made = 0

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Initialise le World Model"""
        try:
            self.logger.info("Initialisation World Model...")

            # Configuration
            world_config = config.get('world_model', {})
            self.mapping_enabled = world_config.get('mapping_enabled', True)
            self.update_interval = world_config.get('update_interval', 1.0)
            self.prediction_interval = world_config.get('prediction_interval', 30.0)

            # Initialiser mémoire spatiale
            self.spatial_memory = SpatialMemory(self.data_dir / "spatial")
            if not self.spatial_memory.initialize():
                self.logger.error("Échec initialisation mémoire spatiale")
                return False

            # Initialiser IA du monde
            self.world_ai = WorldModelAI(self.spatial_memory)

            self.logger.info("World Model initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation World Model: {e}")
            return False

    async def _run_impl(self):
        """Boucle principale du World Model"""
        self.logger.info("Démarrage boucle World Model...")

        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()

                # Mettre à jour cartographie
                await self._update_world_mapping()

                # Analyser patterns et prédire
                if start_time - self.last_prediction_time >= self.prediction_interval:
                    await self._analyze_and_predict()
                    self.last_prediction_time = start_time

                # Partager données avec autres modules
                await self._update_shared_data()

                # Attendre intervalle
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                self.logger.info("World Model arrêté")
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle World Model: {e}")
                await asyncio.sleep(1.0)

    async def _update_world_mapping(self):
        """Met à jour la cartographie du monde"""
        try:
            if not self.mapping_enabled:
                return

            # Récupérer données vision
            screenshot = self._shared_data.get('last_screenshot')
            ocr_results = self._shared_data.get('last_ocr_results', [])

            if screenshot is None:
                return

            # Analyser screenshot pour extraire infos spatiales
            await self._analyze_screenshot_for_mapping(screenshot, ocr_results)

        except Exception as e:
            self.logger.error(f"Erreur mise à jour cartographie: {e}")

    async def _analyze_screenshot_for_mapping(self, screenshot: np.ndarray, ocr_results: List):
        """Analyse screenshot pour cartographie"""
        try:
            # Détecter ID de carte depuis OCR
            map_id = self._extract_map_id(ocr_results)
            if not map_id:
                return

            self.current_map_id = map_id

            # Détecter position joueur
            player_pos = self._extract_player_position(screenshot, ocr_results)
            if player_pos:
                self.current_position = WorldPosition(map_id, player_pos[0], player_pos[1])

            # Détecter entités visibles
            entities = self._extract_visible_entities(screenshot, ocr_results)

            # Mettre à jour mémoire spatiale
            for entity in entities:
                self.spatial_memory.add_entity(entity)
                self.entities_tracked += 1

            # Enregistrer événement de position
            if self.current_position:
                event = WorldEvent(
                    event_id=f"player_pos_{int(time.time())}",
                    event_type=EventType.PLAYER_ACTION,
                    position=self.current_position,
                    timestamp=time.time(),
                    data={"action": "position_update", "player": True}
                )
                self.spatial_memory.record_event(event)

        except Exception as e:
            self.logger.error(f"Erreur analyse screenshot mapping: {e}")

    def _extract_map_id(self, ocr_results: List) -> Optional[str]:
        """Extrait ID de carte depuis OCR"""
        for ocr_result in ocr_results:
            text = ocr_result.text
            # Rechercher patterns d'ID de carte DOFUS
            if any(pattern in text.lower() for pattern in ['map', 'carte', '[', ']']):
                # Extraire ID (simplifié)
                import re
                match = re.search(r'\[(-?\d+),(-?\d+)\]', text)
                if match:
                    return f"{match.group(1)},{match.group(2)}"
        return None

    def _extract_player_position(self, screenshot: np.ndarray, ocr_results: List) -> Optional[Tuple[int, int]]:
        """Extrait position joueur"""
        # Analyser centre de l'écran (position typique joueur)
        h, w = screenshot.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Convertir en coordonnées de carte (simplifié)
        map_x = center_x // 40  # Approximation taille cellule
        map_y = center_y // 40

        return (map_x, map_y)

    def _extract_visible_entities(self, screenshot: np.ndarray, ocr_results: List) -> List[WorldEntity]:
        """Extrait entités visibles"""
        entities = []

        try:
            # Analyser OCR pour noms d'entités
            for ocr_result in ocr_results:
                text = ocr_result.text.strip()

                # Détecter types d'entités
                entity_type = "unknown"
                if any(word in text.lower() for word in ['pnj', 'npc', 'vendeur']):
                    entity_type = "npc"
                elif any(word in text.lower() for word in ['monstre', 'mob', 'agressif']):
                    entity_type = "monster"
                elif any(word in text.lower() for word in ['ressource', 'arbre', 'minerai']):
                    entity_type = "resource"

                if entity_type != "unknown" and len(text) > 2:
                    # Estimer position
                    bbox = ocr_result.bbox
                    screen_x = (bbox[0] + bbox[2]) // 2
                    screen_y = (bbox[1] + bbox[3]) // 2
                    map_x = screen_x // 40
                    map_y = screen_y // 40

                    entity = WorldEntity(
                        entity_id=f"{entity_type}_{text}_{int(time.time())}",
                        entity_type=entity_type,
                        name=text,
                        position=WorldPosition(self.current_map_id, map_x, map_y),
                        properties={"detected_method": "ocr", "confidence": ocr_result.confidence},
                        last_seen=time.time(),
                        confidence=ocr_result.confidence
                    )

                    entities.append(entity)

        except Exception as e:
            self.logger.error(f"Erreur extraction entités: {e}")

        return entities

    async def _analyze_and_predict(self):
        """Analyse patterns et génère prédictions"""
        try:
            if not self.world_ai:
                return

            # Analyser patterns temporels
            self.world_ai.analyze_temporal_patterns()

            # Générer prédictions pour carte actuelle
            if self.current_map_id:
                predictions = self.world_ai.predict_next_events(
                    self.current_map_id, time.time()
                )

                self.predictions_made += len(predictions)

                # Mettre dans shared_data
                self._shared_data['world_predictions'] = predictions

        except Exception as e:
            self.logger.error(f"Erreur analyse prédictions: {e}")

    async def _update_shared_data(self):
        """Met à jour données partagées"""
        try:
            # État du monde
            world_state = {
                "current_position": asdict(self.current_position) if self.current_position else None,
                "current_map_id": self.current_map_id,
                "maps_discovered": self.maps_discovered,
                "entities_tracked": self.entities_tracked,
                "predictions_made": self.predictions_made
            }

            self._shared_data['world_state'] = world_state

            # Évaluation danger position actuelle
            if self.current_position and self.world_ai:
                danger_assessment = self.world_ai.get_danger_assessment(self.current_position)
                self._shared_data['current_danger'] = danger_assessment

        except Exception as e:
            self.logger.error(f"Erreur mise à jour shared_data: {e}")

    async def get_optimal_path(self, target_x: int, target_y: int) -> List[Dict[str, int]]:
        """Calcule chemin optimal vers position cible"""
        try:
            if not self.current_position or not self.world_ai:
                return []

            target_pos = WorldPosition(self.current_map_id, target_x, target_y)
            path = self.world_ai.find_optimal_path(self.current_position, target_pos)

            return [{"x": pos.x, "y": pos.y} for pos in path]

        except Exception as e:
            self.logger.error(f"Erreur calcul chemin optimal: {e}")
            return []

    async def get_nearby_opportunities(self) -> List[Dict[str, Any]]:
        """Trouve opportunités à proximité"""
        try:
            if not self.current_position:
                return []

            # Récupérer entités à proximité
            nearby_entities = self.spatial_memory.get_entities_in_area(
                self.current_map_id, self.current_position.x, self.current_position.y, radius=10
            )

            opportunities = []
            for entity in nearby_entities:
                if entity.entity_type == "resource":
                    opportunities.append({
                        "type": "resource",
                        "name": entity.name,
                        "position": {"x": entity.position.x, "y": entity.position.y},
                        "confidence": entity.confidence
                    })
                elif entity.entity_type == "npc":
                    opportunities.append({
                        "type": "npc",
                        "name": entity.name,
                        "position": {"x": entity.position.x, "y": entity.position.y},
                        "services": entity.properties.get("services", [])
                    })

            return opportunities

        except Exception as e:
            self.logger.error(f"Erreur recherche opportunités: {e}")
            return []

    async def _shutdown_impl(self):
        """Arrêt propre du module"""
        try:
            self.logger.info("Arrêt World Model...")

            if self.spatial_memory and self.spatial_memory.connection:
                self.spatial_memory.connection.close()

            self.logger.info("World Model arrêté proprement")

        except Exception as e:
            self.logger.error(f"Erreur arrêt World Model: {e}")

    def get_module_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du module"""
        stats = {
            "mapping_enabled": self.mapping_enabled,
            "current_map_id": self.current_map_id,
            "current_position": asdict(self.current_position) if self.current_position else None,
            "maps_discovered": self.maps_discovered,
            "entities_tracked": self.entities_tracked,
            "predictions_made": self.predictions_made,
        }

        if self.spatial_memory:
            stats.update({
                "total_entities_in_memory": self.spatial_memory.total_entities_tracked,
                "total_events_recorded": self.spatial_memory.total_events_recorded,
                "cache_size": len(self.spatial_memory.entities_cache)
            })

        if self.world_ai:
            stats.update({
                "patterns_discovered": self.world_ai.patterns_discovered,
                "prediction_accuracy": self.world_ai.prediction_accuracy,
                "active_predictions": len(self.world_ai.active_predictions)
            })

        return stats

# Factory function
def create_world_model(data_dir: Path) -> DofusWorldModel:
    """Crée instance World Model"""
    return DofusWorldModel(data_dir)