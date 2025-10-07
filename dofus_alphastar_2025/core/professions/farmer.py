"""
Module Fermier - Gestion complète du métier d'Agriculteur.
Inclut toutes les ressources agricoles, optimisation de routes et calculs de rentabilité.
"""

from typing import Dict, List, Tuple, Optional
from .base import BaseProfession, ResourceData, ResourceType, QualityLevel
import math
import random

class Farmer(BaseProfession):
    """Classe Fermier avec toutes les ressources agricoles"""
    
    def __init__(self):
        super().__init__("Agriculteur", "farmer")
        self.load_resources()
        self.farming_patterns = {
            'linear': self._linear_pattern,
            'spiral': self._spiral_pattern,
            'zigzag': self._zigzag_pattern,
            'cluster': self._cluster_pattern
        }
        self.current_pattern = 'cluster'
    
    def initialize(self, config: Dict) -> bool:
        """Initialise le module Farmer"""
        try:
            # Initialisation basique, peut être étendue selon les besoins
            return True
        except Exception as e:
            print(f"[ERROR] Erreur initialisation Farmer: {e}")
            return False
        
    def load_resources(self) -> None:
        """Charge toutes les ressources agricoles"""
        self.resources = {
            # Céréales niveau 1-20
            'ble': ResourceData(
                id='ble', name='Blé', type=ResourceType.AGRICULTURAL,
                level_required=1, base_xp=3, base_time=3.0, quality=QualityLevel.COMMON,
                market_value=2, coordinates=(0, 0), success_rate=0.95
            ),
            'orge': ResourceData(
                id='orge', name='Orge', type=ResourceType.AGRICULTURAL,
                level_required=5, base_xp=5, base_time=3.2, quality=QualityLevel.COMMON,
                market_value=3, coordinates=(15, 10), success_rate=0.95
            ),
            'avoine': ResourceData(
                id='avoine', name='Avoine', type=ResourceType.AGRICULTURAL,
                level_required=10, base_xp=8, base_time=3.5, quality=QualityLevel.COMMON,
                market_value=5, coordinates=(25, 15), success_rate=0.93
            ),
            'seigle': ResourceData(
                id='seigle', name='Seigle', type=ResourceType.AGRICULTURAL,
                level_required=15, base_xp=12, base_time=3.8, quality=QualityLevel.COMMON,
                market_value=8, coordinates=(35, 20), success_rate=0.93
            ),
            'riz': ResourceData(
                id='riz', name='Riz', type=ResourceType.AGRICULTURAL,
                level_required=20, base_xp=16, base_time=4.0, quality=QualityLevel.UNCOMMON,
                market_value=12, coordinates=(45, 25), success_rate=0.90
            ),
            
            # Légumes niveau 1-30
            'navet': ResourceData(
                id='navet', name='Navet', type=ResourceType.AGRICULTURAL,
                level_required=1, base_xp=2, base_time=2.8, quality=QualityLevel.COMMON,
                market_value=1, coordinates=(5, 5), success_rate=0.98
            ),
            'carotte': ResourceData(
                id='carotte', name='Carotte', type=ResourceType.AGRICULTURAL,
                level_required=8, base_xp=7, base_time=3.2, quality=QualityLevel.COMMON,
                market_value=4, coordinates=(20, 12), success_rate=0.95
            ),
            'radis': ResourceData(
                id='radis', name='Radis', type=ResourceType.AGRICULTURAL,
                level_required=12, base_xp=10, base_time=3.5, quality=QualityLevel.COMMON,
                market_value=6, coordinates=(30, 18), success_rate=0.93
            ),
            'poireau': ResourceData(
                id='poireau', name='Poireau', type=ResourceType.AGRICULTURAL,
                level_required=18, base_xp=14, base_time=3.8, quality=QualityLevel.COMMON,
                market_value=10, coordinates=(40, 22), success_rate=0.92
            ),
            'chou': ResourceData(
                id='chou', name='Chou', type=ResourceType.AGRICULTURAL,
                level_required=25, base_xp=20, base_time=4.2, quality=QualityLevel.UNCOMMON,
                market_value=15, coordinates=(50, 30), success_rate=0.90
            ),
            'artichaut': ResourceData(
                id='artichaut', name='Artichaut', type=ResourceType.AGRICULTURAL,
                level_required=30, base_xp=25, base_time=4.5, quality=QualityLevel.UNCOMMON,
                market_value=20, coordinates=(60, 35), success_rate=0.88
            ),
            
            # Fruits niveau 20-50
            'pomme': ResourceData(
                id='pomme', name='Pomme', type=ResourceType.AGRICULTURAL,
                level_required=20, base_xp=18, base_time=4.0, quality=QualityLevel.COMMON,
                market_value=12, coordinates=(25, 40), success_rate=0.92
            ),
            'cerise': ResourceData(
                id='cerise', name='Cerise', type=ResourceType.AGRICULTURAL,
                level_required=25, base_xp=22, base_time=4.3, quality=QualityLevel.UNCOMMON,
                market_value=18, coordinates=(35, 45), success_rate=0.90
            ),
            'fraise': ResourceData(
                id='fraise', name='Fraise', type=ResourceType.AGRICULTURAL,
                level_required=30, base_xp=28, base_time=4.5, quality=QualityLevel.UNCOMMON,
                market_value=25, coordinates=(45, 50), success_rate=0.88
            ),
            'orange': ResourceData(
                id='orange', name='Orange', type=ResourceType.AGRICULTURAL,
                level_required=35, base_xp=32, base_time=4.8, quality=QualityLevel.UNCOMMON,
                market_value=30, coordinates=(55, 55), success_rate=0.87
            ),
            'kiwi': ResourceData(
                id='kiwi', name='Kiwi', type=ResourceType.AGRICULTURAL,
                level_required=40, base_xp=38, base_time=5.0, quality=QualityLevel.RARE,
                market_value=40, coordinates=(65, 60), success_rate=0.85
            ),
            'banane': ResourceData(
                id='banane', name='Banane', type=ResourceType.AGRICULTURAL,
                level_required=45, base_xp=42, base_time=5.2, quality=QualityLevel.RARE,
                market_value=45, coordinates=(75, 65), success_rate=0.83
            ),
            'noix_coco': ResourceData(
                id='noix_coco', name='Noix de Coco', type=ResourceType.AGRICULTURAL,
                level_required=50, base_xp=48, base_time=5.5, quality=QualityLevel.RARE,
                market_value=55, coordinates=(85, 70), success_rate=0.80
            ),
            
            # Plantes médicinales niveau 30-80
            'menthe': ResourceData(
                id='menthe', name='Menthe', type=ResourceType.AGRICULTURAL,
                level_required=30, base_xp=30, base_time=4.5, quality=QualityLevel.UNCOMMON,
                market_value=25, coordinates=(40, 35), success_rate=0.88
            ),
            'orchidee': ResourceData(
                id='orchidee', name='Orchidée', type=ResourceType.AGRICULTURAL,
                level_required=40, base_xp=40, base_time=5.0, quality=QualityLevel.RARE,
                market_value=50, coordinates=(70, 45), success_rate=0.85
            ),
            'ginseng': ResourceData(
                id='ginseng', name='Ginseng', type=ResourceType.AGRICULTURAL,
                level_required=50, base_xp=55, base_time=5.5, quality=QualityLevel.RARE,
                market_value=80, coordinates=(90, 55), success_rate=0.80
            ),
            'belladone': ResourceData(
                id='belladone', name='Belladone', type=ResourceType.AGRICULTURAL,
                level_required=60, base_xp=70, base_time=6.0, quality=QualityLevel.EPIC,
                market_value=120, coordinates=(100, 65), success_rate=0.75
            ),
            'mandragore': ResourceData(
                id='mandragore', name='Mandragore', type=ResourceType.AGRICULTURAL,
                level_required=70, base_xp=85, base_time=6.5, quality=QualityLevel.EPIC,
                market_value=180, coordinates=(120, 75), success_rate=0.70
            ),
            'fleuraison': ResourceData(
                id='fleuraison', name='Fleuraison', type=ResourceType.AGRICULTURAL,
                level_required=80, base_xp=105, base_time=7.0, quality=QualityLevel.EPIC,
                market_value=250, coordinates=(140, 85), success_rate=0.65
            ),
            
            # Ressources rares niveau 60-100
            'lin': ResourceData(
                id='lin', name='Lin', type=ResourceType.AGRICULTURAL,
                level_required=60, base_xp=75, base_time=6.0, quality=QualityLevel.EPIC,
                market_value=150, coordinates=(110, 70), success_rate=0.75
            ),
            'chanvre': ResourceData(
                id='chanvre', name='Chanvre', type=ResourceType.AGRICULTURAL,
                level_required=70, base_xp=90, base_time=6.5, quality=QualityLevel.EPIC,
                market_value=200, coordinates=(130, 80), success_rate=0.70
            ),
            'houblon': ResourceData(
                id='houblon', name='Houblon', type=ResourceType.AGRICULTURAL,
                level_required=80, base_xp=110, base_time=7.0, quality=QualityLevel.LEGENDARY,
                market_value=300, coordinates=(150, 90), success_rate=0.65
            ),
            'bambou': ResourceData(
                id='bambou', name='Bambou', type=ResourceType.AGRICULTURAL,
                level_required=90, base_xp=135, base_time=7.5, quality=QualityLevel.LEGENDARY,
                market_value=400, coordinates=(170, 100), success_rate=0.60
            ),
            'bambou_sacre': ResourceData(
                id='bambou_sacre', name='Bambou Sacré', type=ResourceType.AGRICULTURAL,
                level_required=100, base_xp=160, base_time=8.0, quality=QualityLevel.LEGENDARY,
                market_value=500, coordinates=(200, 110), success_rate=0.55
            )
        }
    
    def get_optimal_route(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Calcule la route optimale de farming basée sur XP/h et rentabilité"""
        available_resources = self.get_available_resources(level_range)
        
        if not available_resources:
            return []
        
        # Calcul du score pour chaque ressource
        scored_resources = []
        for resource in available_resources:
            xp_per_hour = self.calculate_xp_per_hour(resource.id)
            kamas_per_hour = self.calculate_kamas_per_hour(resource.id)
            profitability = self.calculate_profitability(resource.id)
            
            # Score composite basé sur XP, kamas et facilité d'accès
            score = (
                xp_per_hour * 0.4 +
                kamas_per_hour * 0.3 +
                profitability['efficiency'] * 0.2 +
                resource.success_rate * 100 * 0.1
            )
            
            scored_resources.append((resource, score))
        
        # Trier par score décroissant
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Optimiser l'ordre basé sur la proximité géographique
        optimized_route = self._optimize_geographic_route([r[0] for r in scored_resources[:10]])
        
        return optimized_route
    
    def calculate_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité détaillée d'une ressource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return {}
        
        # Temps total par récolte
        harvest_time = resource.base_time
        movement_time = self._get_movement_time(resource)
        inventory_time = self._get_inventory_time()
        total_time_per_harvest = harvest_time + movement_time + inventory_time
        
        # Calculs par heure
        harvests_per_hour = 3600 / total_time_per_harvest
        successful_harvests = harvests_per_hour * resource.success_rate
        
        xp_per_hour = successful_harvests * resource.base_xp
        kamas_per_hour = successful_harvests * resource.market_value
        
        # Efficacité (rapport XP/temps)
        efficiency = resource.base_xp / total_time_per_harvest
        
        # ROI (Return on Investment) basé sur le niveau requis
        roi = kamas_per_hour / max(resource.level_required, 1)
        
        return {
            'xp_per_hour': round(xp_per_hour, 2),
            'kamas_per_hour': round(kamas_per_hour, 2),
            'efficiency': round(efficiency, 2),
            'roi': round(roi, 2),
            'harvests_per_hour': round(successful_harvests, 2),
            'success_rate': resource.success_rate,
            'time_per_harvest': round(total_time_per_harvest, 2)
        }
    
    def get_best_resources_by_level(self, level: int, count: int = 3) -> List[ResourceData]:
        """Retourne les meilleures ressources pour un niveau donné"""
        available = [r for r in self.resources.values() if r.level_required <= level]
        
        # Tri par rentabilité
        best_resources = []
        for resource in available:
            profit = self.calculate_profitability(resource.id)
            best_resources.append((resource, profit.get('efficiency', 0)))
        
        best_resources.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in best_resources[:count]]
    
    def _optimize_geographic_route(self, resources: List[ResourceData]) -> List[ResourceData]:
        """Optimise la route basée sur la proximité géographique (TSP simplifié)"""
        if len(resources) <= 1:
            return resources
        
        # Algorithme du plus proche voisin
        unvisited = resources.copy()
        route = [unvisited.pop(0)]  # Commencer par la première ressource
        
        while unvisited:
            current = route[-1]
            closest_resource = min(unvisited, key=lambda r: self._calculate_distance(current, r))
            route.append(closest_resource)
            unvisited.remove(closest_resource)
        
        return route
    
    def _calculate_distance(self, resource1: ResourceData, resource2: ResourceData) -> float:
        """Calcule la distance euclidienne entre deux ressources"""
        x1, y1 = resource1.coordinates
        x2, y2 = resource2.coordinates
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_farming_pattern(self, resources: List[ResourceData], pattern_name: str = None) -> List[Tuple[int, int]]:
        """Génère un pattern de farming pour les ressources données"""
        if not resources:
            return []
        
        pattern_name = pattern_name or self.current_pattern
        pattern_func = self.farming_patterns.get(pattern_name, self._cluster_pattern)
        
        return pattern_func(resources)
    
    def _linear_pattern(self, resources: List[ResourceData]) -> List[Tuple[int, int]]:
        """Pattern linéaire - suit l'ordre des ressources"""
        return [r.coordinates for r in resources]
    
    def _spiral_pattern(self, resources: List[ResourceData]) -> List[Tuple[int, int]]:
        """Pattern spiral - crée une spirale autour du centre"""
        if not resources:
            return []
        
        # Calculer le centre
        center_x = sum(r.coordinates[0] for r in resources) / len(resources)
        center_y = sum(r.coordinates[1] for r in resources) / len(resources)
        
        # Trier par angle depuis le centre
        def angle_from_center(resource):
            x, y = resource.coordinates
            return math.atan2(y - center_y, x - center_x)
        
        sorted_resources = sorted(resources, key=angle_from_center)
        return [r.coordinates for r in sorted_resources]
    
    def _zigzag_pattern(self, resources: List[ResourceData]) -> List[Tuple[int, int]]:
        """Pattern zigzag - alterne directions"""
        sorted_resources = sorted(resources, key=lambda r: (r.coordinates[1], r.coordinates[0]))
        coordinates = []
        
        for i, resource in enumerate(sorted_resources):
            if i % 2 == 0:
                coordinates.append(resource.coordinates)
            else:
                # Insérer au début pour effet zigzag
                coordinates.insert(-i//2 if i > 1 else 0, resource.coordinates)
        
        return coordinates
    
    def _cluster_pattern(self, resources: List[ResourceData]) -> List[Tuple[int, int]]:
        """Pattern cluster - groupe par proximité"""
        if not resources:
            return []
        
        clusters = []
        unprocessed = resources.copy()
        
        while unprocessed:
            # Commencer un nouveau cluster
            cluster_center = unprocessed.pop(0)
            current_cluster = [cluster_center]
            
            # Trouver les ressources proches
            to_remove = []
            for resource in unprocessed:
                if self._calculate_distance(cluster_center, resource) < 30:  # Seuil de proximité
                    current_cluster.append(resource)
                    to_remove.append(resource)
            
            # Retirer les ressources ajoutées au cluster
            for resource in to_remove:
                unprocessed.remove(resource)
            
            clusters.append(current_cluster)
        
        # Optimiser chaque cluster et les connecter
        optimized_coordinates = []
        for cluster in clusters:
            optimized_cluster = self._optimize_geographic_route(cluster)
            optimized_coordinates.extend([r.coordinates for r in optimized_cluster])
        
        return optimized_coordinates
    
    def estimate_farming_time(self, resource_ids: List[str], target_quantity: int = 100) -> Dict[str, float]:
        """Estime le temps nécessaire pour farmer une quantité donnée"""
        estimates = {}
        
        for resource_id in resource_ids:
            resource = self.resources.get(resource_id)
            if not resource:
                continue
            
            time_per_harvest = (resource.base_time + 
                              self._get_movement_time(resource) + 
                              self._get_inventory_time())
            
            expected_harvests = target_quantity / resource.success_rate
            total_time = expected_harvests * time_per_harvest
            
            estimates[resource_id] = {
                'total_time_hours': round(total_time / 3600, 2),
                'harvests_needed': round(expected_harvests),
                'success_rate': resource.success_rate,
                'estimated_xp': round(target_quantity * resource.base_xp),
                'estimated_kamas': round(target_quantity * resource.market_value)
            }
        
        return estimates
    
    def get_resource_by_quality(self, quality: QualityLevel) -> List[ResourceData]:
        """Retourne les ressources d'une qualité donnée"""
        return [r for r in self.resources.values() if r.quality == quality]
    
    def get_seasonal_bonuses(self) -> Dict[str, float]:
        """Simule des bonus saisonniers (à adapter selon le jeu)"""
        import datetime
        month = datetime.datetime.now().month
        
        seasonal_bonuses = {
            'spring': ['carotte', 'radis', 'menthe'],  # Mars-Mai
            'summer': ['ble', 'orge', 'cerise', 'fraise'],  # Juin-Août
            'autumn': ['pomme', 'noix_coco', 'houblon'],  # Sept-Nov
            'winter': ['chou', 'poireau', 'ginseng']  # Déc-Fév
        }
        
        season = 'spring' if 3 <= month <= 5 else 'summer' if 6 <= month <= 8 else 'autumn' if 9 <= month <= 11 else 'winter'
        
        bonuses = {}
        for resource_id in seasonal_bonuses.get(season, []):
            if resource_id in self.resources:
                bonuses[resource_id] = 1.2  # 20% de bonus
        
        return bonuses
    
    def __str__(self) -> str:
        available_resources = len(self.get_available_resources())
        return f"Fermier (Niveau {self.stats.level}) - {available_resources} ressources disponibles"