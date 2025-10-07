"""
Module de base pour tous les métiers du système de botting.
Définit l'interface commune et les méthodes partagées pour tous les métiers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import math
from pathlib import Path

class ResourceType(Enum):
    """Types de ressources disponibles"""
    AGRICULTURAL = "agricultural"
    WOOD = "wood"  
    MINERAL = "mineral"
    POTION = "potion"
    CRAFT_MATERIAL = "craft_material"

class QualityLevel(Enum):
    """Niveaux de qualité des ressources"""
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5

@dataclass
class ResourceData:
    """Données d'une ressource"""
    id: str
    name: str
    type: ResourceType
    level_required: int
    base_xp: int
    base_time: float  # secondes
    quality: QualityLevel
    market_value: int  # prix moyen en kamas
    coordinates: Tuple[int, int]  # position sur la carte
    respawn_time: float = 0.0  # temps de réapparition
    success_rate: float = 1.0  # taux de succès de récolte
    tools_required: List[str] = None

    def __post_init__(self):
        if self.tools_required is None:
            self.tools_required = []

@dataclass
class ProfessionStats:
    """Statistiques d'un métier"""
    level: int = 1
    experience: int = 0
    total_harvested: int = 0
    total_crafted: int = 0
    kamas_earned: int = 0
    time_spent: float = 0.0

class BaseProfession(ABC):
    """Classe abstraite de base pour tous les métiers"""
    
    def __init__(self, name: str, profession_id: str):
        self.name = name
        self.profession_id = profession_id
        self.stats = ProfessionStats()
        self.resources: Dict[str, ResourceData] = {}
        self.inventory_capacity = 1000  # pods
        self.current_inventory = 0
        self.auto_bank = True
        self.auto_craft = False
        self.preferred_resources: List[str] = []
        self.blacklisted_resources: List[str] = []
        
    @abstractmethod
    def load_resources(self) -> None:
        """Charge toutes les ressources du métier"""
        pass
    
    @abstractmethod
    def get_optimal_route(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Calcule la route optimale de farming"""
        pass
    
    @abstractmethod
    def calculate_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité d'une ressource"""
        pass
    
    def get_available_resources(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Retourne les ressources disponibles selon le niveau"""
        available = []
        min_level = level_range[0] if level_range else 1
        max_level = level_range[1] if level_range else 200
        
        for resource in self.resources.values():
            if (min_level <= resource.level_required <= max_level and 
                resource.id not in self.blacklisted_resources):
                available.append(resource)
        
        return sorted(available, key=lambda r: r.level_required)
    
    def calculate_xp_per_hour(self, resource_id: str) -> float:
        """Calcule l'XP par heure pour une ressource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return 0.0
        
        # Temps total incluant déplacements et gestion inventaire
        total_time = resource.base_time + self._get_movement_time(resource) + self._get_inventory_time()
        xp_per_second = (resource.base_xp * resource.success_rate) / total_time
        
        return xp_per_second * 3600
    
    def calculate_kamas_per_hour(self, resource_id: str) -> float:
        """Calcule les kamas par heure pour une ressource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return 0.0
        
        total_time = resource.base_time + self._get_movement_time(resource) + self._get_inventory_time()
        kamas_per_second = (resource.market_value * resource.success_rate) / total_time
        
        return kamas_per_second * 3600
    
    def get_next_level_resources(self) -> List[ResourceData]:
        """Retourne les ressources du niveau suivant"""
        next_level = self.stats.level + 1
        return [r for r in self.resources.values() if r.level_required == next_level]
    
    def update_stats(self, resource_id: str, quantity: int = 1, time_spent: float = 0.0):
        """Met à jour les statistiques du métier"""
        resource = self.resources.get(resource_id)
        if resource:
            self.stats.total_harvested += quantity
            self.stats.experience += resource.base_xp * quantity
            self.stats.kamas_earned += resource.market_value * quantity
            self.stats.time_spent += time_spent
            
            # Calcul du niveau basé sur l'XP
            self.stats.level = self._calculate_level_from_xp(self.stats.experience)
    
    def _calculate_level_from_xp(self, xp: int) -> int:
        """Calcule le niveau basé sur l'expérience (formule Dofus)"""
        if xp < 100:
            return 1
        
        # Formule approximative Dofus: niveau = sqrt(xp/100)
        level = int(math.sqrt(xp / 100))
        return min(level, 200)  # Cap niveau 200
    
    def _get_movement_time(self, resource: ResourceData) -> float:
        """Estime le temps de déplacement vers une ressource"""
        # Temps de base + distance euclidienne * facteur
        base_time = 2.0  # secondes
        distance_factor = 0.1
        distance = math.sqrt(resource.coordinates[0]**2 + resource.coordinates[1]**2)
        return base_time + (distance * distance_factor)
    
    def _get_inventory_time(self) -> float:
        """Estime le temps de gestion de l'inventaire"""
        if self.current_inventory >= self.inventory_capacity * 0.9:
            return 30.0 if self.auto_bank else 0.0  # Temps pour aller à la banque
        return 0.0
    
    def can_harvest(self, resource_id: str) -> bool:
        """Vérifie si une ressource peut être récoltée"""
        resource = self.resources.get(resource_id)
        if not resource:
            return False
        
        return (self.stats.level >= resource.level_required and 
                resource.id not in self.blacklisted_resources and
                self.current_inventory < self.inventory_capacity)
    
    def get_profession_info(self) -> Dict[str, Any]:
        """Retourne les informations du métier"""
        return {
            'name': self.name,
            'id': self.profession_id,
            'level': self.stats.level,
            'experience': self.stats.experience,
            'total_resources': len(self.resources),
            'available_resources': len(self.get_available_resources()),
            'total_harvested': self.stats.total_harvested,
            'kamas_earned': self.stats.kamas_earned,
            'time_spent': round(self.stats.time_spent / 3600, 2),  # heures
            'avg_xp_per_hour': round(self.stats.experience / (self.stats.time_spent / 3600), 2) if self.stats.time_spent > 0 else 0
        }
    
    def save_configuration(self, filepath: str) -> None:
        """Sauvegarde la configuration du métier"""
        import json
        config = {
            'profession_id': self.profession_id,
            'stats': {
                'level': self.stats.level,
                'experience': self.stats.experience,
                'total_harvested': self.stats.total_harvested,
                'total_crafted': self.stats.total_crafted,
                'kamas_earned': self.stats.kamas_earned,
                'time_spent': self.stats.time_spent
            },
            'settings': {
                'inventory_capacity': self.inventory_capacity,
                'auto_bank': self.auto_bank,
                'auto_craft': self.auto_craft,
                'preferred_resources': self.preferred_resources,
                'blacklisted_resources': self.blacklisted_resources
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_configuration(self, filepath: str) -> None:
        """Charge la configuration du métier"""
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Charger les stats
            stats_data = config.get('stats', {})
            self.stats.level = stats_data.get('level', 1)
            self.stats.experience = stats_data.get('experience', 0)
            self.stats.total_harvested = stats_data.get('total_harvested', 0)
            self.stats.total_crafted = stats_data.get('total_crafted', 0)
            self.stats.kamas_earned = stats_data.get('kamas_earned', 0)
            self.stats.time_spent = stats_data.get('time_spent', 0.0)
            
            # Charger les paramètres
            settings = config.get('settings', {})
            self.inventory_capacity = settings.get('inventory_capacity', 1000)
            self.auto_bank = settings.get('auto_bank', True)
            self.auto_craft = settings.get('auto_craft', False)
            self.preferred_resources = settings.get('preferred_resources', [])
            self.blacklisted_resources = settings.get('blacklisted_resources', [])
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erreur lors du chargement de la configuration: {e}")

    def __str__(self) -> str:
        return f"{self.name} (Niveau {self.stats.level})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', level={self.stats.level})>"