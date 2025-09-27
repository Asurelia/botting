"""
DOFUS Knowledge Graph Engine
Système de représentation sémantique du monde DOFUS
Base de connaissances évolutive pour l'IA autonome
"""

import json
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import asyncio
import hashlib
import pickle
from enum import Enum, auto

# Configuration du logging
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types d'entités dans le monde DOFUS"""
    # Objets de jeu
    ITEM = "item"
    EQUIPMENT = "equipment"
    RESOURCE = "resource"
    CONSUMABLE = "consumable"

    # Créatures
    MONSTER = "monster"
    ARCHMONSTER = "archmonster"
    NPC = "npc"
    PLAYER = "player"

    # Lieux
    ZONE = "zone"
    SUBAREA = "subarea"
    DUNGEON = "dungeon"
    HOUSE = "house"

    # Mécaniques
    SPELL = "spell"
    PROFESSION = "profession"
    QUEST = "quest"
    ACHIEVEMENT = "achievement"

    # Économie
    SHOP = "shop"
    MARKET = "market"
    BANK = "bank"

class RelationType(Enum):
    """Types de relations entre entités"""
    # Relations spatiales
    LOCATED_IN = "located_in"
    CONNECTED_TO = "connected_to"
    ACCESSIBLE_FROM = "accessible_from"

    # Relations fonctionnelles
    CRAFTED_FROM = "crafted_from"
    DROPPED_BY = "dropped_by"
    HARVESTED_FROM = "harvested_from"
    SOLD_BY = "sold_by"
    REQUIRED_FOR = "required_for"

    # Relations temporelles
    SPAWNS_AT = "spawns_at"
    AVAILABLE_DURING = "available_during"
    COOLDOWN_AFTER = "cooldown_after"

    # Relations logiques
    COUNTERS = "counters"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    PREREQUISITE_FOR = "prerequisite_for"

@dataclass
class KnowledgeEntity:
    """Entité de base dans le graphe de connaissances"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)

    # Métadonnées
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "manual"
    verified: bool = False

    # Données de jeu spécifiques
    level: Optional[int] = None
    coordinates: Optional[Tuple[int, int]] = None
    value: Optional[int] = None
    rarity: Optional[str] = None

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation"""
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type.value,
            'properties': self.properties,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat(),
            'source': self.source,
            'verified': self.verified,
            'level': self.level,
            'coordinates': self.coordinates,
            'value': self.value,
            'rarity': self.rarity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntity':
        """Création depuis dictionnaire"""
        entity = cls(
            id=data['id'],
            name=data['name'],
            entity_type=EntityType(data['entity_type']),
            properties=data.get('properties', {}),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'manual'),
            verified=data.get('verified', False),
            level=data.get('level'),
            coordinates=data.get('coordinates'),
            value=data.get('value'),
            rarity=data.get('rarity')
        )

        if 'last_updated' in data:
            entity.last_updated = datetime.fromisoformat(data['last_updated'])

        return entity

@dataclass
class KnowledgeRelation:
    """Relation entre entités dans le graphe"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)

    # Métadonnées
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "manual"

    # Contraintes temporelles
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    def is_valid_at(self, timestamp: datetime = None) -> bool:
        """Vérifie si la relation est valide à un moment donné"""
        if timestamp is None:
            timestamp = datetime.now()

        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False

        return True

class DofusKnowledgeGraph:
    """
    Graphe de connaissances DOFUS avec capacités d'inférence
    Représentation sémantique complète du monde du jeu
    """

    def __init__(self, data_path: str = "data/knowledge"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Graphe principal (NetworkX pour performance)
        self.graph = nx.MultiDiGraph()

        # Index des entités pour accès rapide
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.entities_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self.entities_by_name: Dict[str, List[str]] = defaultdict(list)

        # Cache d'inférence
        self.inference_cache = {}
        self.cache_ttl = timedelta(minutes=10)
        self.last_cache_clear = datetime.now()

        # Statistiques
        self.stats = {
            'total_entities': 0,
            'total_relations': 0,
            'inference_queries': 0,
            'cache_hits': 0,
            'last_updated': datetime.now()
        }

    def add_entity(self, entity: KnowledgeEntity) -> bool:
        """Ajoute une entité au graphe"""
        try:
            # Ajout au graphe
            self.graph.add_node(entity.id, entity=entity)

            # Mise à jour des index
            self.entities[entity.id] = entity
            self.entities_by_type[entity.entity_type].add(entity.id)
            self.entities_by_name[entity.name.lower()].append(entity.id)

            # Mise à jour statistiques
            self.stats['total_entities'] = len(self.entities)
            self.stats['last_updated'] = datetime.now()

            logger.debug(f"Entité ajoutée: {entity.name} ({entity.entity_type.value})")
            return True

        except Exception as e:
            logger.error(f"Erreur ajout entité {entity.id}: {e}")
            return False

    def add_relation(self, relation: KnowledgeRelation) -> bool:
        """Ajoute une relation au graphe"""
        try:
            # Vérification existence des entités
            if relation.source_id not in self.entities:
                logger.warning(f"Entité source inconnue: {relation.source_id}")
                return False
            if relation.target_id not in self.entities:
                logger.warning(f"Entité cible inconnue: {relation.target_id}")
                return False

            # Ajout de l'arête
            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.relation_type.value,
                relation=relation
            )

            # Invalidation du cache
            self._clear_inference_cache()

            # Mise à jour statistiques
            self.stats['total_relations'] = self.graph.number_of_edges()
            self.stats['last_updated'] = datetime.now()

            logger.debug(f"Relation ajoutée: {relation.source_id} --{relation.relation_type.value}--> {relation.target_id}")
            return True

        except Exception as e:
            logger.error(f"Erreur ajout relation: {e}")
            return False

    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Récupère une entité par ID"""
        return self.entities.get(entity_id)

    def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[KnowledgeEntity]:
        """Trouve des entités par nom"""
        results = []
        name_lower = name.lower()

        # Recherche exacte
        if name_lower in self.entities_by_name:
            for entity_id in self.entities_by_name[name_lower]:
                results.append(self.entities[entity_id])

        # Recherche floue si activée
        if fuzzy and not results:
            for stored_name, entity_ids in self.entities_by_name.items():
                if name_lower in stored_name or stored_name in name_lower:
                    for entity_id in entity_ids:
                        results.append(self.entities[entity_id])

        return results

    def find_entities_by_type(self, entity_type: EntityType) -> List[KnowledgeEntity]:
        """Trouve toutes les entités d'un type donné"""
        entity_ids = self.entities_by_type.get(entity_type, set())
        return [self.entities[entity_id] for entity_id in entity_ids]

    def get_relations(self, source_id: str, relation_type: RelationType = None,
                     target_id: str = None) -> List[KnowledgeRelation]:
        """Récupère les relations d'une entité"""
        relations = []

        try:
            if target_id:
                # Relation spécifique
                edges = self.graph.get_edge_data(source_id, target_id, {})
                for key, data in edges.items():
                    if relation_type is None or key == relation_type.value:
                        relations.append(data['relation'])
            else:
                # Toutes les relations sortantes
                for successor in self.graph.successors(source_id):
                    edges = self.graph.get_edge_data(source_id, successor, {})
                    for key, data in edges.items():
                        if relation_type is None or key == relation_type.value:
                            relations.append(data['relation'])

            # Filtrage par validité temporelle
            current_time = datetime.now()
            valid_relations = [r for r in relations if r.is_valid_at(current_time)]

            return valid_relations

        except Exception as e:
            logger.error(f"Erreur récupération relations: {e}")
            return []

    def infer_shortest_path(self, source_id: str, target_id: str,
                          max_length: int = 5) -> Optional[List[str]]:
        """Trouve le chemin le plus court entre deux entités"""
        cache_key = f"path_{source_id}_{target_id}_{max_length}"

        # Vérification cache
        if self._is_cache_valid() and cache_key in self.inference_cache:
            self.stats['cache_hits'] += 1
            return self.inference_cache[cache_key]

        try:
            path = nx.shortest_path(
                self.graph, source_id, target_id, weight=None
            )

            if len(path) <= max_length + 1:  # +1 car le chemin inclut les nœuds
                self.inference_cache[cache_key] = path
                self.stats['inference_queries'] += 1
                return path
            else:
                return None

        except nx.NetworkXNoPath:
            self.inference_cache[cache_key] = None
            return None
        except Exception as e:
            logger.error(f"Erreur inférence chemin: {e}")
            return None

    def infer_related_entities(self, entity_id: str, relation_types: List[RelationType],
                             max_depth: int = 2) -> Dict[str, float]:
        """Infère les entités liées avec scores de pertinence"""
        cache_key = f"related_{entity_id}_{hash(tuple(relation_types))}_{max_depth}"

        if self._is_cache_valid() and cache_key in self.inference_cache:
            self.stats['cache_hits'] += 1
            return self.inference_cache[cache_key]

        related_entities = {}

        try:
            # BFS avec limitation de profondeur
            visited = set()
            queue = deque([(entity_id, 0, 1.0)])  # (entity_id, depth, score)

            while queue:
                current_id, depth, score = queue.popleft()

                if current_id in visited or depth >= max_depth:
                    continue

                visited.add(current_id)

                # Exploration des relations
                for successor in self.graph.successors(current_id):
                    edges = self.graph.get_edge_data(current_id, successor, {})

                    for key, data in edges.items():
                        relation = data['relation']

                        # Vérification type de relation
                        if relation.relation_type in relation_types:
                            # Calcul du score (dégradation avec la distance)
                            new_score = score * relation.confidence * (0.7 ** depth)

                            if successor != entity_id:  # Éviter auto-référence
                                if successor in related_entities:
                                    related_entities[successor] = max(
                                        related_entities[successor], new_score
                                    )
                                else:
                                    related_entities[successor] = new_score

                                queue.append((successor, depth + 1, new_score))

            # Tri par score décroissant
            sorted_entities = dict(
                sorted(related_entities.items(), key=lambda x: x[1], reverse=True)
            )

            self.inference_cache[cache_key] = sorted_entities
            self.stats['inference_queries'] += 1
            return sorted_entities

        except Exception as e:
            logger.error(f"Erreur inférence entités liées: {e}")
            return {}

    def find_optimal_strategy(self, goal_entity_id: str,
                            available_resources: List[str]) -> Optional[Dict[str, Any]]:
        """Trouve une stratégie optimale pour atteindre un objectif"""
        try:
            goal_entity = self.get_entity(goal_entity_id)
            if not goal_entity:
                return None

            strategy = {
                'goal': goal_entity.name,
                'steps': [],
                'requirements': [],
                'estimated_cost': 0,
                'estimated_time': 0,
                'confidence': 1.0
            }

            # Analyse des moyens d'obtention
            incoming_relations = []
            for predecessor in self.graph.predecessors(goal_entity_id):
                edges = self.graph.get_edge_data(predecessor, goal_entity_id, {})
                for key, data in edges.items():
                    relation = data['relation']
                    incoming_relations.append((predecessor, relation))

            # Priorisation des stratégies
            best_strategy = None
            best_score = 0

            for source_id, relation in incoming_relations:
                source_entity = self.get_entity(source_id)
                if not source_entity:
                    continue

                # Score basé sur le type de relation et les ressources disponibles
                score = relation.confidence

                # Bonus si on a déjà les ressources nécessaires
                if source_id in available_resources:
                    score *= 2.0

                # Bonus selon le type de relation
                if relation.relation_type == RelationType.CRAFTED_FROM:
                    score *= 1.5  # Craft souvent efficace
                elif relation.relation_type == RelationType.DROPPED_BY:
                    score *= 1.2  # Drop peut être aléatoire
                elif relation.relation_type == RelationType.SOLD_BY:
                    score *= 1.8  # Achat direct souvent optimal

                if score > best_score:
                    best_score = score
                    best_strategy = {
                        'method': relation.relation_type.value,
                        'source': source_entity.name,
                        'source_id': source_id,
                        'confidence': relation.confidence
                    }

            if best_strategy:
                strategy['steps'].append(best_strategy)
                strategy['confidence'] = best_strategy['confidence']

            return strategy

        except Exception as e:
            logger.error(f"Erreur recherche stratégie optimale: {e}")
            return None

    def analyze_market_trends(self, item_id: str,
                            time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Analyse les tendances de marché pour un item"""
        # Placeholder pour l'analyse de marché
        # Sera étendu avec des données réelles de prix
        item = self.get_entity(item_id)
        if not item:
            return {}

        analysis = {
            'item_name': item.name,
            'base_value': item.value or 0,
            'trend': 'stable',  # stable, rising, falling
            'confidence': 0.5,
            'last_analysis': datetime.now().isoformat()
        }

        # TODO: Intégration avec données de marché réelles
        return analysis

    def bootstrap_dofus_knowledge(self) -> bool:
        """Bootstrap avec les connaissances de base DOFUS"""
        try:
            logger.info("🚀 Bootstrap des connaissances DOFUS de base...")

            # Zones principales
            zones_data = [
                ("zone_astrub", "Astrub", EntityType.ZONE, {"level_range": [1, 20]}),
                ("zone_amakna", "Amakna", EntityType.ZONE, {"level_range": [1, 50]}),
                ("zone_brakmar", "Brakmar", EntityType.ZONE, {"level_range": [30, 100]}),
                ("zone_bonta", "Bonta", EntityType.ZONE, {"level_range": [30, 100]}),
            ]

            for zone_id, zone_name, zone_type, props in zones_data:
                entity = KnowledgeEntity(
                    id=zone_id,
                    name=zone_name,
                    entity_type=zone_type,
                    properties=props,
                    source="bootstrap"
                )
                self.add_entity(entity)

            # Ressources de base
            resources_data = [
                ("resource_wheat", "Blé", EntityType.RESOURCE, {"level": 1, "profession": "farmer"}),
                ("resource_ash", "Frêne", EntityType.RESOURCE, {"level": 1, "profession": "lumberjack"}),
                ("resource_iron", "Fer", EntityType.RESOURCE, {"level": 1, "profession": "miner"}),
            ]

            for res_id, res_name, res_type, props in resources_data:
                entity = KnowledgeEntity(
                    id=res_id,
                    name=res_name,
                    entity_type=res_type,
                    properties=props,
                    source="bootstrap"
                )
                self.add_entity(entity)

            # Relations de base
            basic_relations = [
                ("resource_wheat", "zone_astrub", RelationType.LOCATED_IN),
                ("resource_ash", "zone_amakna", RelationType.LOCATED_IN),
                ("resource_iron", "zone_amakna", RelationType.LOCATED_IN),
            ]

            for source, target, rel_type in basic_relations:
                relation = KnowledgeRelation(
                    source_id=source,
                    target_id=target,
                    relation_type=rel_type,
                    source="bootstrap"
                )
                self.add_relation(relation)

            logger.info(f"✅ Bootstrap terminé: {len(self.entities)} entités, {self.graph.number_of_edges()} relations")
            return True

        except Exception as e:
            logger.error(f"Erreur bootstrap: {e}")
            return False

    def save_to_file(self, filename: str = "knowledge_graph.json") -> bool:
        """Sauvegarde le graphe dans un fichier"""
        try:
            save_path = self.data_path / filename

            # Préparation des données
            data = {
                'metadata': {
                    'version': '1.0',
                    'created': datetime.now().isoformat(),
                    'stats': self.stats
                },
                'entities': [entity.to_dict() for entity in self.entities.values()],
                'relations': []
            }

            # Extraction des relations
            for source, target, key, edge_data in self.graph.edges(keys=True, data=True):
                relation = edge_data['relation']
                relation_dict = {
                    'source_id': relation.source_id,
                    'target_id': relation.target_id,
                    'relation_type': relation.relation_type.value,
                    'properties': relation.properties,
                    'confidence': relation.confidence,
                    'last_updated': relation.last_updated.isoformat(),
                    'source': relation.source
                }
                data['relations'].append(relation_dict)

            # Sauvegarde
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Graphe sauvegardé: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return False

    def load_from_file(self, filename: str = "knowledge_graph.json") -> bool:
        """Charge le graphe depuis un fichier"""
        try:
            load_path = self.data_path / filename

            if not load_path.exists():
                logger.warning(f"Fichier inexistant: {load_path}")
                return False

            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Chargement des entités
            for entity_dict in data.get('entities', []):
                entity = KnowledgeEntity.from_dict(entity_dict)
                self.add_entity(entity)

            # Chargement des relations
            for relation_dict in data.get('relations', []):
                relation = KnowledgeRelation(
                    source_id=relation_dict['source_id'],
                    target_id=relation_dict['target_id'],
                    relation_type=RelationType(relation_dict['relation_type']),
                    properties=relation_dict.get('properties', {}),
                    confidence=relation_dict.get('confidence', 1.0),
                    source=relation_dict.get('source', 'file')
                )

                if 'last_updated' in relation_dict:
                    relation.last_updated = datetime.fromisoformat(relation_dict['last_updated'])

                self.add_relation(relation)

            # Mise à jour des stats
            if 'metadata' in data and 'stats' in data['metadata']:
                self.stats.update(data['metadata']['stats'])

            logger.info(f"✅ Graphe chargé: {len(self.entities)} entités, {self.graph.number_of_edges()} relations")
            return True

        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
            return False

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache d'inférence est valide"""
        return (datetime.now() - self.last_cache_clear) < self.cache_ttl

    def _clear_inference_cache(self):
        """Vide le cache d'inférence"""
        self.inference_cache.clear()
        self.last_cache_clear = datetime.now()

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du graphe"""
        self.stats.update({
            'total_entities': len(self.entities),
            'total_relations': self.graph.number_of_edges(),
            'entities_by_type': {
                entity_type.value: len(entity_ids)
                for entity_type, entity_ids in self.entities_by_type.items()
            },
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['inference_queries'])
                if self.stats['inference_queries'] > 0 else 0
            )
        })
        return self.stats.copy()

# Fonctions utilitaires

async def create_dofus_knowledge_graph(data_path: str = "data/knowledge") -> DofusKnowledgeGraph:
    """Crée et initialise un graphe de connaissances DOFUS"""
    graph = DofusKnowledgeGraph(data_path)

    # Tentative de chargement depuis fichier
    if not graph.load_from_file():
        # Bootstrap si pas de fichier existant
        graph.bootstrap_dofus_knowledge()
        graph.save_to_file()

    return graph

async def main():
    """Test du Knowledge Graph Engine"""
    print("🧠 Test DOFUS Knowledge Graph Engine...")

    # Création du graphe
    graph = await create_dofus_knowledge_graph()

    # Test recherche d'entités
    wheat_entities = graph.find_entities_by_name("blé")
    print(f"✅ Entités 'blé' trouvées: {len(wheat_entities)}")

    if wheat_entities:
        wheat = wheat_entities[0]
        print(f"  - {wheat.name} ({wheat.entity_type.value})")

        # Test relations
        relations = graph.get_relations(wheat.id)
        print(f"  - Relations: {len(relations)}")

        for relation in relations:
            target = graph.get_entity(relation.target_id)
            if target:
                print(f"    → {relation.relation_type.value} → {target.name}")

    # Test inférence
    zones = graph.find_entities_by_type(EntityType.ZONE)
    if len(zones) >= 2:
        path = graph.infer_shortest_path(zones[0].id, zones[1].id)
        print(f"✅ Chemin {zones[0].name} → {zones[1].name}: {path}")

    # Statistiques
    stats = graph.get_statistics()
    print(f"📊 Statistiques:")
    print(f"  - Entités: {stats['total_entities']}")
    print(f"  - Relations: {stats['total_relations']}")
    print(f"  - Requêtes d'inférence: {stats['inference_queries']}")

    print("✅ Test Knowledge Graph terminé !")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())