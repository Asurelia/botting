"""
Module Fermier Refactorisé - Utilise la configuration externalisée
Plus de données codées en dur, tout est chargé depuis les fichiers de configuration
"""

from typing import Dict, List, Tuple, Optional, Any
from .base import BaseProfession, ResourceData, ResourceType, QualityLevel
import asyncio
import logging
from core.config_manager import get_config_manager, get_resource_data

logger = logging.getLogger(__name__)

class FarmerRefactored(BaseProfession):
    """Classe Fermier refactorisée avec configuration externalisée"""

    def __init__(self):
        super().__init__("Agriculteur", "farmer")
        self.config_manager = get_config_manager()
        self.resources = {}
        self.farming_patterns = {
            'linear': self._linear_pattern,
            'spiral': self._spiral_pattern,
            'zigzag': self._zigzag_pattern,
            'cluster': self._cluster_pattern
        }
        self.current_pattern = 'cluster'

        # Configuration chargée dynamiquement
        self.weather_effects = {}
        self.tools_config = {}
        self.global_modifiers = {}

    async def initialize(self, config: Dict = None) -> bool:
        """Initialise le module Farmer avec configuration externalisée"""
        try:
            # Chargement des configurations
            await self._load_agricultural_config()

            # Chargement des ressources depuis la config
            await self._load_resources_from_config()

            logger.info(f"Farmer initialisé avec {len(self.resources)} ressources")
            return True

        except Exception as e:
            logger.error(f"Erreur initialisation Farmer: {e}")
            return False

    async def _load_agricultural_config(self):
        """Charge la configuration agricole"""
        try:
            # Effets météorologiques
            self.weather_effects = await self.config_manager.get_config(
                "weather_effects", {}
            )

            # Configuration des outils
            self.tools_config = await self.config_manager.get_config(
                "tools", {}
            )

            # Modificateurs globaux
            self.global_modifiers = await self.config_manager.get_config(
                "global_modifiers", {}
            )

            logger.debug("Configuration agricole chargée")

        except Exception as e:
            logger.error(f"Erreur chargement config agricole: {e}")

    async def _load_resources_from_config(self):
        """Charge toutes les ressources depuis la configuration"""
        try:
            # Récupération des catégories
            categories = await self.config_manager.get_config("categories", {})

            for category_name, category_data in categories.items():
                items = category_data.get("items", {})

                for resource_id, resource_config in items.items():
                    # Conversion vers ResourceData
                    resource_data = await self._convert_config_to_resource_data(
                        resource_id, resource_config, category_name
                    )

                    if resource_data:
                        self.resources[resource_id] = resource_data

            logger.info(f"Chargées {len(self.resources)} ressources depuis la configuration")

        except Exception as e:
            logger.error(f"Erreur chargement ressources: {e}")

    async def _convert_config_to_resource_data(self, resource_id: str,
                                             config: Dict[str, Any],
                                             category: str) -> Optional[ResourceData]:
        """Convertit une configuration en ResourceData"""
        try:
            # Mapping des qualités
            quality_map = {
                "common": QualityLevel.COMMON,
                "uncommon": QualityLevel.UNCOMMON,
                "rare": QualityLevel.RARE,
                "epic": QualityLevel.EPIC,
                "legendary": QualityLevel.LEGENDARY
            }

            # Coordonnées principales (première location)
            locations = config.get("locations", [])
            main_coordinates = (0, 0)
            if locations:
                coords = locations[0].get("coordinates", [0, 0])
                main_coordinates = (coords[0], coords[1])

            # Création de ResourceData
            resource_data = ResourceData(
                id=resource_id,
                name=config.get("name", resource_id),
                type=ResourceType.AGRICULTURAL,
                level_required=config.get("level_required", 1),
                base_xp=config.get("base_xp", 1),
                base_time=config.get("base_time", 1.0),
                quality=quality_map.get(config.get("quality", "common"), QualityLevel.COMMON),
                market_value=config.get("market_value", 1),
                coordinates=main_coordinates,
                success_rate=config.get("success_rate", 0.95)
            )

            return resource_data

        except Exception as e:
            logger.error(f"Erreur conversion ressource {resource_id}: {e}")
            return None

    async def get_optimal_resources(self, player_level: int,
                                  current_weather: str = "sunny",
                                  current_season: str = "spring") -> List[ResourceData]:
        """Récupère les ressources optimales selon les conditions"""
        try:
            optimal_resources = []

            for resource in self.resources.values():
                if resource.level_required <= player_level:
                    # Calcul du score avec bonus météo et saisonnier
                    score = await self._calculate_resource_score(
                        resource, current_weather, current_season
                    )

                    if score > 0:
                        optimal_resources.append((resource, score))

            # Tri par score décroissant
            optimal_resources.sort(key=lambda x: x[1], reverse=True)

            return [resource for resource, score in optimal_resources[:10]]

        except Exception as e:
            logger.error(f"Erreur calcul ressources optimales: {e}")
            return []

    async def _calculate_resource_score(self, resource: ResourceData,
                                      weather: str, season: str) -> float:
        """Calcule le score d'une ressource selon les conditions"""
        try:
            base_score = resource.base_xp * resource.market_value * resource.success_rate

            # Récupération des données de configuration de la ressource
            resource_config = await get_resource_data(resource.id)
            if not resource_config:
                return base_score

            # Bonus météorologique
            weather_bonus = 1.0
            if weather in self.weather_effects:
                category = await self._get_resource_category(resource.id)
                if category in self.weather_effects[weather]:
                    weather_bonus = self.weather_effects[weather][category]

            # Bonus saisonnier
            seasonal_bonus = 1.0
            seasonal_bonuses = resource_config.get("seasonal_bonus", {})
            if season in seasonal_bonuses:
                seasonal_bonus = seasonal_bonuses[season]

            # Bonus globaux
            global_bonus = 1.0
            for modifier_name, modifier_value in self.global_modifiers.items():
                if modifier_name == "weekend_bonus":
                    # Simulation : bonus weekend aléatoire
                    import random
                    if random.random() < 0.3:  # 30% chance weekend
                        global_bonus *= modifier_value

            final_score = base_score * weather_bonus * seasonal_bonus * global_bonus

            return final_score

        except Exception as e:
            logger.error(f"Erreur calcul score ressource {resource.id}: {e}")
            return 0.0

    async def _get_resource_category(self, resource_id: str) -> str:
        """Détermine la catégorie d'une ressource"""
        try:
            categories = await self.config_manager.get_config("categories", {})

            for category_name, category_data in categories.items():
                items = category_data.get("items", {})
                if resource_id in items:
                    return category_name

            return "unknown"

        except Exception as e:
            logger.error(f"Erreur détermination catégorie {resource_id}: {e}")
            return "unknown"

    async def get_resource_locations(self, resource_id: str) -> List[Dict[str, Any]]:
        """Récupère toutes les locations d'une ressource"""
        try:
            resource_config = await get_resource_data(resource_id)
            if not resource_config:
                return []

            locations = resource_config.get("locations", [])

            # Enrichissement avec données de carte
            enriched_locations = []
            for location in locations:
                enriched_location = location.copy()

                # Ajout d'informations de la carte si disponible
                map_name = location.get("map")
                if map_name:
                    # Recherche des données de carte
                    map_data = await self._find_map_by_name(map_name)
                    if map_data:
                        enriched_location["map_data"] = map_data

                enriched_locations.append(enriched_location)

            return enriched_locations

        except Exception as e:
            logger.error(f"Erreur récupération locations {resource_id}: {e}")
            return []

    async def _find_map_by_name(self, map_name: str) -> Optional[Dict[str, Any]]:
        """Trouve une carte par son nom"""
        try:
            regions = await self.config_manager.get_config("regions", {})

            for region_data in regions.values():
                maps = region_data.get("maps", {})
                for map_id, map_data in maps.items():
                    if map_data.get("name") == map_name:
                        return map_data

            return None

        except Exception as e:
            logger.error(f"Erreur recherche carte {map_name}: {e}")
            return None

    async def get_farming_efficiency(self, resource_id: str,
                                   player_stats: Dict[str, Any] = None) -> Dict[str, float]:
        """Calcule l'efficacité de farm d'une ressource"""
        try:
            resource_config = await get_resource_data(resource_id)
            if not resource_config:
                return {"efficiency": 0.0}

            base_time = resource_config.get("base_time", 1.0)
            base_xp = resource_config.get("base_xp", 1)
            success_rate = resource_config.get("success_rate", 0.95)
            market_value = resource_config.get("market_value", 1)

            # Facteurs d'efficacité
            player_level = player_stats.get("level", 1) if player_stats else 1
            level_bonus = min(player_level / resource_config.get("level_required", 1), 2.0)

            # Bonus d'outil
            tool_bonus = 1.0
            required_tools = resource_config.get("tools_required", [])
            if required_tools and player_stats:
                equipped_tools = player_stats.get("equipped_tools", [])
                if any(tool.lower() in [t.lower() for t in equipped_tools] for tool in required_tools):
                    tool_config = self.tools_config.get(required_tools[0].lower(), {})
                    tool_bonus = tool_config.get("efficiency_bonus", 1.0)

            # Calculs finaux
            effective_time = base_time / (level_bonus * tool_bonus)
            xp_per_hour = (base_xp * success_rate * 3600) / effective_time
            kamas_per_hour = (market_value * success_rate * 3600) / effective_time

            return {
                "efficiency": xp_per_hour + kamas_per_hour * 0.1,  # Score composite
                "xp_per_hour": xp_per_hour,
                "kamas_per_hour": kamas_per_hour,
                "success_rate": success_rate,
                "time_per_gather": effective_time,
                "level_bonus": level_bonus,
                "tool_bonus": tool_bonus
            }

        except Exception as e:
            logger.error(f"Erreur calcul efficacité {resource_id}: {e}")
            return {"efficiency": 0.0}

    async def update_resource_config(self, resource_id: str,
                                   updates: Dict[str, Any]) -> bool:
        """Met à jour la configuration d'une ressource"""
        try:
            # Trouve la catégorie de la ressource
            category = await self._get_resource_category(resource_id)
            if category == "unknown":
                return False

            # Mise à jour dans la configuration
            for key, value in updates.items():
                config_key = f"categories.{category}.items.{resource_id}.{key}"
                await self.config_manager.set_config(config_key, value)

            # Rechargement de cette ressource
            resource_config = await get_resource_data(resource_id)
            if resource_config:
                updated_resource = await self._convert_config_to_resource_data(
                    resource_id, resource_config, category
                )
                if updated_resource:
                    self.resources[resource_id] = updated_resource

            logger.info(f"Configuration de {resource_id} mise à jour")
            return True

        except Exception as e:
            logger.error(f"Erreur mise à jour config {resource_id}: {e}")
            return False

    # Les patterns de farm restent inchangés car ils ne contiennent pas de données codées en dur
    def _linear_pattern(self, start_pos: Tuple[int, int], resources_count: int) -> List[Tuple[int, int]]:
        """Pattern linéaire de collecte"""
        positions = []
        x, y = start_pos

        for i in range(resources_count):
            positions.append((x + i, y))

        return positions

    def _spiral_pattern(self, center_pos: Tuple[int, int], resources_count: int) -> List[Tuple[int, int]]:
        """Pattern en spirale de collecte"""
        positions = []
        x, y = center_pos
        positions.append((x, y))

        direction = 0  # 0=right, 1=down, 2=left, 3=up
        steps = 1

        while len(positions) < resources_count:
            for _ in range(2):
                for _ in range(steps):
                    if direction == 0:
                        x += 1
                    elif direction == 1:
                        y += 1
                    elif direction == 2:
                        x -= 1
                    elif direction == 3:
                        y -= 1

                    positions.append((x, y))
                    if len(positions) >= resources_count:
                        break

                direction = (direction + 1) % 4
                if len(positions) >= resources_count:
                    break

            steps += 1

        return positions[:resources_count]

    def _zigzag_pattern(self, start_pos: Tuple[int, int], resources_count: int) -> List[Tuple[int, int]]:
        """Pattern en zigzag de collecte"""
        positions = []
        x, y = start_pos

        row = 0
        while len(positions) < resources_count:
            if row % 2 == 0:
                # Ligne de gauche à droite
                for i in range(5):
                    if len(positions) >= resources_count:
                        break
                    positions.append((x + i, y + row))
            else:
                # Ligne de droite à gauche
                for i in range(4, -1, -1):
                    if len(positions) >= resources_count:
                        break
                    positions.append((x + i, y + row))

            row += 1

        return positions[:resources_count]

    def _cluster_pattern(self, center_pos: Tuple[int, int], resources_count: int) -> List[Tuple[int, int]]:
        """Pattern en cluster de collecte"""
        positions = []
        x, y = center_pos

        # Centre
        positions.append((x, y))

        # Cercles concentriques
        radius = 1
        while len(positions) < resources_count:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        positions.append((x + dx, y + dy))
                        if len(positions) >= resources_count:
                            break
                if len(positions) >= resources_count:
                    break
            radius += 1

        return positions[:resources_count]

# Fonction utilitaire pour migration
async def migrate_from_old_farmer(old_farmer_instance) -> FarmerRefactored:
    """Migre une ancienne instance de Farmer vers la nouvelle version"""
    try:
        new_farmer = FarmerRefactored()
        await new_farmer.initialize()

        # Migration des patterns et configurations actuelles
        if hasattr(old_farmer_instance, 'current_pattern'):
            new_farmer.current_pattern = old_farmer_instance.current_pattern

        logger.info("Migration vers Farmer refactorisé réussie")
        return new_farmer

    except Exception as e:
        logger.error(f"Erreur migration Farmer: {e}")
        raise