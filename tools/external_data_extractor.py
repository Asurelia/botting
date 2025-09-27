"""
External Data Extractor pour DOFUS
Système d'extraction intelligent des données depuis Dofus Guide, Ganymede et autres sources
"""

import os
import json
import logging
import requests
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
import hashlib

# Import modules internes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.knowledge_graph import DofusKnowledgeGraph, KnowledgeEntity, EntityType, KnowledgeRelation, RelationType

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Source de données externe"""
    name: str
    type: str  # 'file', 'api', 'database', 'web'
    path: str
    format: str  # 'json', 'xml', 'sqlite', 'csv'
    priority: int = 1
    enabled: bool = True
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DofusGuideExtractor:
    """Extracteur spécialisé pour Dofus Guide"""

    def __init__(self, guide_path: str):
        self.guide_path = Path(guide_path)
        self.logger = logging.getLogger(f"{__name__}.DofusGuideExtractor")

    def detect_dofus_guide_installation(self) -> Optional[Path]:
        """Détecte l'installation de Dofus Guide"""
        possible_paths = [
            Path("C:/Program Files/Dofus Guide"),
            Path("C:/Program Files (x86)/Dofus Guide"),
            Path(os.path.expanduser("~/AppData/Local/Dofus Guide")),
            Path(os.path.expanduser("~/Documents/Dofus Guide")),
            self.guide_path
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"✅ Dofus Guide détecté: {path}")
                return path

        self.logger.warning("❌ Dofus Guide non détecté")
        return None

    def extract_items_data(self, guide_path: Path) -> List[Dict[str, Any]]:
        """Extrait les données d'items depuis Dofus Guide"""
        items_data = []

        try:
            # Recherche de fichiers de données d'items
            data_patterns = [
                "data/items.json",
                "data/items.xml",
                "resources/items.db",
                "cache/items.json",
                "items.sqlite"
            ]

            for pattern in data_patterns:
                data_file = guide_path / pattern
                if data_file.exists():
                    self.logger.info(f"📄 Fichier d'items trouvé: {data_file}")

                    if data_file.suffix.lower() == '.json':
                        items_data.extend(self._parse_json_items(data_file))
                    elif data_file.suffix.lower() == '.xml':
                        items_data.extend(self._parse_xml_items(data_file))
                    elif data_file.suffix.lower() in ['.db', '.sqlite']:
                        items_data.extend(self._parse_sqlite_items(data_file))

            self.logger.info(f"✅ {len(items_data)} items extraits de Dofus Guide")
            return items_data

        except Exception as e:
            self.logger.error(f"Erreur extraction items Dofus Guide: {e}")
            return []

    def _parse_json_items(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse les items depuis un fichier JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Différents formats possibles
                if 'items' in data:
                    items = data['items']
                elif 'data' in data:
                    items = data['data']
                else:
                    items = list(data.values())

            processed_items = []
            for item in items:
                if isinstance(item, dict) and 'name' in item:
                    processed_item = self._normalize_item_data(item, 'dofus_guide_json')
                    if processed_item:
                        processed_items.append(processed_item)

            return processed_items

        except Exception as e:
            self.logger.error(f"Erreur parsing JSON {file_path}: {e}")
            return []

    def _parse_xml_items(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse les items depuis un fichier XML"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            items = []
            # Différents formats XML possibles
            for item_elem in root.findall('.//item') or root.findall('.//object'):
                item_data = {}

                # Attributs
                item_data.update(item_elem.attrib)

                # Éléments enfants
                for child in item_elem:
                    item_data[child.tag] = child.text or child.attrib

                if 'name' in item_data or 'nom' in item_data:
                    processed_item = self._normalize_item_data(item_data, 'dofus_guide_xml')
                    if processed_item:
                        items.append(processed_item)

            return items

        except Exception as e:
            self.logger.error(f"Erreur parsing XML {file_path}: {e}")
            return []

    def _parse_sqlite_items(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse les items depuis une base SQLite"""
        try:
            conn = sqlite3.connect(file_path)
            conn.row_factory = sqlite3.Row

            # Détection des tables d'items
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            items = []
            for table in tables:
                if any(keyword in table.lower() for keyword in ['item', 'object', 'equipment']):
                    try:
                        cursor.execute(f"SELECT * FROM {table} LIMIT 1000")
                        rows = cursor.fetchall()

                        for row in rows:
                            item_data = dict(row)
                            if any(key in item_data for key in ['name', 'nom', 'title']):
                                processed_item = self._normalize_item_data(item_data, 'dofus_guide_sqlite')
                                if processed_item:
                                    items.append(processed_item)

                    except Exception as e:
                        self.logger.debug(f"Erreur table {table}: {e}")

            conn.close()
            return items

        except Exception as e:
            self.logger.error(f"Erreur parsing SQLite {file_path}: {e}")
            return []

    def _normalize_item_data(self, raw_data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Normalise les données d'item vers un format standard"""
        try:
            # Extraction du nom
            name = (
                raw_data.get('name') or
                raw_data.get('nom') or
                raw_data.get('title') or
                raw_data.get('libelle')
            )

            if not name or not isinstance(name, str):
                return None

            # ID unique
            item_id = (
                raw_data.get('id') or
                raw_data.get('itemId') or
                hashlib.md5(name.encode()).hexdigest()[:12]
            )

            # Type d'entité
            entity_type = EntityType.ITEM
            if any(keyword in name.lower() for keyword in ['équipement', 'equipment', 'arme', 'weapon']):
                entity_type = EntityType.EQUIPMENT
            elif any(keyword in name.lower() for keyword in ['ressource', 'resource']):
                entity_type = EntityType.RESOURCE

            # Normalisation des propriétés
            properties = {}

            # Niveau
            level = (
                raw_data.get('level') or
                raw_data.get('niveau') or
                raw_data.get('lvl')
            )
            if level:
                try:
                    properties['level'] = int(level)
                except (ValueError, TypeError):
                    pass

            # Type/catégorie
            item_type = (
                raw_data.get('type') or
                raw_data.get('category') or
                raw_data.get('categorie')
            )
            if item_type:
                properties['type'] = str(item_type)

            # Description
            description = (
                raw_data.get('description') or
                raw_data.get('desc') or
                raw_data.get('tooltip')
            )
            if description:
                properties['description'] = str(description)[:500]  # Limitation

            # Statistiques/effets
            for key in raw_data:
                if key.lower() in ['effects', 'effets', 'stats', 'statistiques']:
                    properties[key] = raw_data[key]

            return {
                'id': f"item_{item_id}",
                'name': name,
                'entity_type': entity_type,
                'properties': properties,
                'source': source,
                'raw_data': raw_data
            }

        except Exception as e:
            self.logger.debug(f"Erreur normalisation item: {e}")
            return None

class GanymedeExtractor:
    """Extracteur spécialisé pour Ganymede"""

    def __init__(self, ganymede_path: str):
        self.ganymede_path = Path(ganymede_path)
        self.logger = logging.getLogger(f"{__name__}.GanymedeExtractor")

    def detect_ganymede_installation(self) -> Optional[Path]:
        """Détecte l'installation de Ganymede"""
        possible_paths = [
            Path("C:/Program Files/Ganymede"),
            Path("C:/Program Files (x86)/Ganymede"),
            Path(os.path.expanduser("~/AppData/Local/Ganymede")),
            Path(os.path.expanduser("~/Documents/Ganymede")),
            self.ganymede_path
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"✅ Ganymede détecté: {path}")
                return path

        self.logger.warning("❌ Ganymede non détecté")
        return None

    def extract_maps_data(self, ganymede_path: Path) -> List[Dict[str, Any]]:
        """Extrait les données de cartes/zones depuis Ganymede"""
        maps_data = []

        try:
            # Recherche de fichiers de cartes
            map_patterns = [
                "data/maps.json",
                "maps/*.json",
                "world/areas.xml",
                "cache/maps.db"
            ]

            for pattern in map_patterns:
                if '*' in pattern:
                    # Pattern avec wildcard
                    base_path = ganymede_path / pattern.split('*')[0]
                    if base_path.exists():
                        for file_path in base_path.glob(pattern.split('/')[-1]):
                            maps_data.extend(self._parse_map_file(file_path))
                else:
                    # Fichier simple
                    map_file = ganymede_path / pattern
                    if map_file.exists():
                        maps_data.extend(self._parse_map_file(map_file))

            self.logger.info(f"✅ {len(maps_data)} zones extraites de Ganymede")
            return maps_data

        except Exception as e:
            self.logger.error(f"Erreur extraction cartes Ganymede: {e}")
            return []

    def _parse_map_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse un fichier de cartes"""
        try:
            if file_path.suffix.lower() == '.json':
                return self._parse_json_maps(file_path)
            elif file_path.suffix.lower() == '.xml':
                return self._parse_xml_maps(file_path)
            elif file_path.suffix.lower() in ['.db', '.sqlite']:
                return self._parse_sqlite_maps(file_path)
            else:
                return []

        except Exception as e:
            self.logger.error(f"Erreur parsing {file_path}: {e}")
            return []

    def _parse_json_maps(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse les cartes JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            maps = []
            if isinstance(data, list):
                maps = data
            elif isinstance(data, dict):
                if 'maps' in data:
                    maps = data['maps']
                elif 'areas' in data:
                    maps = data['areas']

            processed_maps = []
            for map_data in maps:
                if isinstance(map_data, dict):
                    processed_map = self._normalize_map_data(map_data, 'ganymede_json')
                    if processed_map:
                        processed_maps.append(processed_map)

            return processed_maps

        except Exception as e:
            self.logger.error(f"Erreur parsing JSON maps {file_path}: {e}")
            return []

    def _normalize_map_data(self, raw_data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Normalise les données de carte"""
        try:
            # Extraction du nom
            name = (
                raw_data.get('name') or
                raw_data.get('nom') or
                raw_data.get('area_name') or
                raw_data.get('zone_name')
            )

            if not name:
                return None

            # ID unique
            map_id = (
                raw_data.get('id') or
                raw_data.get('mapId') or
                hashlib.md5(name.encode()).hexdigest()[:12]
            )

            # Propriétés
            properties = {}

            # Coordonnées
            x = raw_data.get('x') or raw_data.get('coord_x')
            y = raw_data.get('y') or raw_data.get('coord_y')
            if x is not None and y is not None:
                properties['coordinates'] = [int(x), int(y)]

            # Niveau recommandé
            level = raw_data.get('level') or raw_data.get('suggested_level')
            if level:
                properties['suggested_level'] = int(level)

            # Ressources disponibles
            resources = raw_data.get('resources') or raw_data.get('harvestables')
            if resources:
                properties['resources'] = resources

            return {
                'id': f"zone_{map_id}",
                'name': name,
                'entity_type': EntityType.ZONE,
                'properties': properties,
                'source': source,
                'raw_data': raw_data
            }

        except Exception as e:
            self.logger.debug(f"Erreur normalisation carte: {e}")
            return None

class ExternalDataManager:
    """Gestionnaire principal des données externes"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(f"{__name__}.ExternalDataManager")

        # Extracteurs spécialisés
        self.dofus_guide_extractor = DofusGuideExtractor("")
        self.ganymede_extractor = GanymedeExtractor("")

        # Sources de données
        self.data_sources: List[DataSource] = []

    def discover_data_sources(self) -> List[DataSource]:
        """Découvre automatiquement les sources de données disponibles"""
        sources = []

        # Détection Dofus Guide
        dofus_guide_path = self.dofus_guide_extractor.detect_dofus_guide_installation()
        if dofus_guide_path:
            sources.append(DataSource(
                name="Dofus Guide",
                type="file",
                path=str(dofus_guide_path),
                format="mixed",
                priority=1,
                metadata={"installation_path": str(dofus_guide_path)}
            ))

        # Détection Ganymede
        ganymede_path = self.ganymede_extractor.detect_ganymede_installation()
        if ganymede_path:
            sources.append(DataSource(
                name="Ganymede",
                type="file",
                path=str(ganymede_path),
                format="mixed",
                priority=1,
                metadata={"installation_path": str(ganymede_path)}
            ))

        # Sources web publiques
        web_sources = [
            DataSource(
                name="Dofus Wiki",
                type="web",
                path="https://dofuswiki.fandom.com",
                format="html",
                priority=2,
                enabled=False  # Désactivé par défaut (scraping complexe)
            ),
            DataSource(
                name="DofusDB",
                type="api",
                path="https://api.dofusdb.fr",
                format="json",
                priority=2,
                enabled=False  # Désactivé par défaut (nécessite clé API)
            )
        ]

        sources.extend(web_sources)
        self.data_sources = sources

        self.logger.info(f"✅ {len(sources)} sources de données découvertes")
        return sources

    async def extract_all_data(self) -> Dict[str, int]:
        """Extrait les données de toutes les sources disponibles"""
        results = {
            'items_extracted': 0,
            'zones_extracted': 0,
            'relations_created': 0,
            'sources_processed': 0
        }

        for source in self.data_sources:
            if not source.enabled:
                continue

            try:
                self.logger.info(f"🔄 Extraction depuis {source.name}...")

                if source.name == "Dofus Guide":
                    items_data = self.dofus_guide_extractor.extract_items_data(Path(source.path))
                    items_count = await self._integrate_items_data(items_data, source.name)
                    results['items_extracted'] += items_count

                elif source.name == "Ganymede":
                    maps_data = self.ganymede_extractor.extract_maps_data(Path(source.path))
                    zones_count = await self._integrate_maps_data(maps_data, source.name)
                    results['zones_extracted'] += zones_count

                results['sources_processed'] += 1
                source.last_updated = datetime.now()

            except Exception as e:
                self.logger.error(f"Erreur extraction {source.name}: {e}")

        # Création des relations déduites
        relations_count = await self._create_inferred_relations()
        results['relations_created'] = relations_count

        return results

    async def _integrate_items_data(self, items_data: List[Dict[str, Any]], source_name: str) -> int:
        """Intègre les données d'items dans le graphe de connaissances"""
        integrated_count = 0

        for item_data in items_data:
            try:
                entity = KnowledgeEntity(
                    id=item_data['id'],
                    name=item_data['name'],
                    entity_type=item_data['entity_type'],
                    properties=item_data['properties'],
                    source=source_name.lower().replace(' ', '_'),
                    confidence=0.9  # Confiance élevée pour données officielles
                )

                if self.knowledge_graph.add_entity(entity):
                    integrated_count += 1

            except Exception as e:
                self.logger.debug(f"Erreur intégration item: {e}")

        self.logger.info(f"✅ {integrated_count} items intégrés depuis {source_name}")
        return integrated_count

    async def _integrate_maps_data(self, maps_data: List[Dict[str, Any]], source_name: str) -> int:
        """Intègre les données de cartes dans le graphe de connaissances"""
        integrated_count = 0

        for map_data in maps_data:
            try:
                entity = KnowledgeEntity(
                    id=map_data['id'],
                    name=map_data['name'],
                    entity_type=map_data['entity_type'],
                    properties=map_data['properties'],
                    source=source_name.lower().replace(' ', '_'),
                    confidence=0.9
                )

                if self.knowledge_graph.add_entity(entity):
                    integrated_count += 1

            except Exception as e:
                self.logger.debug(f"Erreur intégration carte: {e}")

        self.logger.info(f"✅ {integrated_count} zones intégrées depuis {source_name}")
        return integrated_count

    async def _create_inferred_relations(self) -> int:
        """Crée des relations inférées entre entités"""
        relations_created = 0

        try:
            # Relation zones ↔ ressources basée sur les noms
            zones = self.knowledge_graph.find_entities_by_type(EntityType.ZONE)
            resources = self.knowledge_graph.find_entities_by_type(EntityType.RESOURCE)

            for zone in zones:
                for resource in resources:
                    # Logique d'inférence simple basée sur les noms
                    if self._should_relate_zone_resource(zone, resource):
                        relation = KnowledgeRelation(
                            source_id=resource.id,
                            target_id=zone.id,
                            relation_type=RelationType.LOCATED_IN,
                            confidence=0.7,  # Confiance modérée pour inférence
                            source="auto_inference"
                        )

                        if self.knowledge_graph.add_relation(relation):
                            relations_created += 1

            self.logger.info(f"✅ {relations_created} relations inférées créées")

        except Exception as e:
            self.logger.error(f"Erreur création relations inférées: {e}")

        return relations_created

    def _should_relate_zone_resource(self, zone: KnowledgeEntity, resource: KnowledgeEntity) -> bool:
        """Détermine si une zone et une ressource doivent être liées"""
        zone_name = zone.name.lower()
        resource_name = resource.name.lower()

        # Correspondances par mots-clés
        correspondences = {
            'astrub': ['blé', 'orge', 'avoine'],
            'amakna': ['frêne', 'châtaignier', 'fer'],
            'brakmar': ['charbon', 'cuivre'],
            'forêt': ['bois', 'arbre', 'frêne'],
            'mine': ['fer', 'cuivre', 'charbon', 'or'],
            'champ': ['blé', 'orge', 'avoine', 'houblon']
        }

        for zone_keyword, resource_keywords in correspondences.items():
            if zone_keyword in zone_name:
                for resource_keyword in resource_keywords:
                    if resource_keyword in resource_name:
                        return True

        return False

    def get_extraction_report(self) -> Dict[str, Any]:
        """Génère un rapport d'extraction"""
        return {
            'sources_discovered': len(self.data_sources),
            'sources_enabled': len([s for s in self.data_sources if s.enabled]),
            'last_extraction': max(
                (s.last_updated for s in self.data_sources if s.last_updated),
                default=None
            ),
            'knowledge_graph_stats': self.knowledge_graph.get_statistics()
        }

# Interface CLI
async def main():
    """Test du système d'extraction"""
    print("🔍 Test External Data Extractor...")

    # Import du knowledge graph
    from core.knowledge_graph import create_dofus_knowledge_graph

    # Création du graphe de connaissances
    knowledge_graph = await create_dofus_knowledge_graph()

    # Création du gestionnaire d'extraction
    data_manager = ExternalDataManager(knowledge_graph)

    # Découverte des sources
    sources = data_manager.discover_data_sources()
    print(f"✅ Sources découvertes: {len(sources)}")

    for source in sources:
        status = "✅ Activé" if source.enabled else "⚠️ Désactivé"
        print(f"  - {source.name} ({source.type}): {status}")

    # Test extraction (sources activées uniquement)
    if any(s.enabled for s in sources):
        print("\n🔄 Test extraction...")
        results = await data_manager.extract_all_data()

        print(f"📊 Résultats extraction:")
        print(f"  - Items extraits: {results['items_extracted']}")
        print(f"  - Zones extraites: {results['zones_extracted']}")
        print(f"  - Relations créées: {results['relations_created']}")
        print(f"  - Sources traitées: {results['sources_processed']}")

        # Sauvegarde du graphe enrichi
        knowledge_graph.save_to_file("knowledge_graph_enriched.json")
        print("✅ Graphe enrichi sauvegardé")

    else:
        print("⚠️ Aucune source activée pour l'extraction")

    print("✅ Test extraction terminé !")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())