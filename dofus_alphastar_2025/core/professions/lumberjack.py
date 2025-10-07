"""
Module Bûcheron - Gestion complète du métier de Bûcheron.
Inclut tous les types d'arbres, optimisation de routes et calculs de rentabilité.
"""

from typing import Dict, List, Tuple, Optional
from .base import BaseProfession, ResourceData, ResourceType, QualityLevel
import math
import random

class Lumberjack(BaseProfession):
    """Classe Bûcheron avec tous les types d'arbres"""
    
    def __init__(self):
        super().__init__("Bûcheron", "lumberjack")
        self.load_resources()
        self.cutting_patterns = {
            'efficient': self._efficient_cutting,
            'sustainable': self._sustainable_cutting,
            'aggressive': self._aggressive_cutting,
            'balanced': self._balanced_cutting
        }
        self.current_pattern = 'balanced'
        self.tools_efficiency = {
            'hache_debutant': 1.0,
            'hache_fer': 1.2,
            'hache_bronze': 1.4,
            'hache_argent': 1.6,
            'hache_or': 1.8,
            'hache_platine': 2.0
        }
        self.current_tool = 'hache_debutant'
        
    def load_resources(self) -> None:
        """Charge tous les types d'arbres"""
        self.resources = {
            # Arbres niveau 1-20
            'chene': ResourceData(
                id='chene', name='Chêne', type=ResourceType.WOOD,
                level_required=1, base_xp=5, base_time=4.0, quality=QualityLevel.COMMON,
                market_value=3, coordinates=(10, 5), success_rate=0.95, respawn_time=120.0,
                tools_required=['hache_debutant']
            ),
            'frene': ResourceData(
                id='frene', name='Frêne', type=ResourceType.WOOD,
                level_required=5, base_xp=8, base_time=4.2, quality=QualityLevel.COMMON,
                market_value=5, coordinates=(20, 15), success_rate=0.93, respawn_time=150.0,
                tools_required=['hache_debutant']
            ),
            'noyer': ResourceData(
                id='noyer', name='Noyer', type=ResourceType.WOOD,
                level_required=10, base_xp=12, base_time=4.5, quality=QualityLevel.COMMON,
                market_value=8, coordinates=(30, 25), success_rate=0.92, respawn_time=180.0,
                tools_required=['hache_fer']
            ),
            'chateignier': ResourceData(
                id='chateignier', name='Châtaignier', type=ResourceType.WOOD,
                level_required=15, base_xp=16, base_time=4.8, quality=QualityLevel.COMMON,
                market_value=12, coordinates=(40, 35), success_rate=0.90, respawn_time=210.0,
                tools_required=['hache_fer']
            ),
            'hetre': ResourceData(
                id='hetre', name='Hêtre', type=ResourceType.WOOD,
                level_required=20, base_xp=20, base_time=5.0, quality=QualityLevel.UNCOMMON,
                market_value=18, coordinates=(50, 45), success_rate=0.88, respawn_time=240.0,
                tools_required=['hache_bronze']
            ),
            
            # Arbres niveau 20-40
            'bouleau': ResourceData(
                id='bouleau', name='Bouleau', type=ResourceType.WOOD,
                level_required=20, base_xp=22, base_time=5.2, quality=QualityLevel.UNCOMMON,
                market_value=20, coordinates=(25, 50), success_rate=0.87, respawn_time=270.0,
                tools_required=['hache_bronze']
            ),
            'merisier': ResourceData(
                id='merisier', name='Merisier', type=ResourceType.WOOD,
                level_required=25, base_xp=26, base_time=5.5, quality=QualityLevel.UNCOMMON,
                market_value=25, coordinates=(35, 55), success_rate=0.85, respawn_time=300.0,
                tools_required=['hache_bronze']
            ),
            'orme': ResourceData(
                id='orme', name='Orme', type=ResourceType.WOOD,
                level_required=30, base_xp=32, base_time=5.8, quality=QualityLevel.UNCOMMON,
                market_value=30, coordinates=(45, 60), success_rate=0.83, respawn_time=330.0,
                tools_required=['hache_argent']
            ),
            'erable': ResourceData(
                id='erable', name='Érable', type=ResourceType.WOOD,
                level_required=35, base_xp=38, base_time=6.0, quality=QualityLevel.RARE,
                market_value=40, coordinates=(55, 65), success_rate=0.80, respawn_time=360.0,
                tools_required=['hache_argent']
            ),
            'charme': ResourceData(
                id='charme', name='Charme', type=ResourceType.WOOD,
                level_required=40, base_xp=45, base_time=6.2, quality=QualityLevel.RARE,
                market_value=50, coordinates=(65, 70), success_rate=0.78, respawn_time=390.0,
                tools_required=['hache_argent']
            ),
            
            # Arbres niveau 40-60
            'chene_rouge': ResourceData(
                id='chene_rouge', name='Chêne Rouge', type=ResourceType.WOOD,
                level_required=40, base_xp=48, base_time=6.5, quality=QualityLevel.RARE,
                market_value=55, coordinates=(75, 50), success_rate=0.76, respawn_time=420.0,
                tools_required=['hache_or']
            ),
            'pin': ResourceData(
                id='pin', name='Pin', type=ResourceType.WOOD,
                level_required=45, base_xp=52, base_time=6.8, quality=QualityLevel.RARE,
                market_value=60, coordinates=(85, 55), success_rate=0.74, respawn_time=450.0,
                tools_required=['hache_or']
            ),
            'sapin': ResourceData(
                id='sapin', name='Sapin', type=ResourceType.WOOD,
                level_required=50, base_xp=58, base_time=7.0, quality=QualityLevel.RARE,
                market_value=70, coordinates=(95, 60), success_rate=0.72, respawn_time=480.0,
                tools_required=['hache_or']
            ),
            'epicea': ResourceData(
                id='epicea', name='Épicéa', type=ResourceType.WOOD,
                level_required=55, base_xp=65, base_time=7.2, quality=QualityLevel.EPIC,
                market_value=80, coordinates=(105, 65), success_rate=0.70, respawn_time=510.0,
                tools_required=['hache_or']
            ),
            'if': ResourceData(
                id='if', name='If', type=ResourceType.WOOD,
                level_required=60, base_xp=72, base_time=7.5, quality=QualityLevel.EPIC,
                market_value=95, coordinates=(115, 70), success_rate=0.68, respawn_time=540.0,
                tools_required=['hache_platine']
            ),
            
            # Arbres niveau 60-80
            'chene_blanc': ResourceData(
                id='chene_blanc', name='Chêne Blanc', type=ResourceType.WOOD,
                level_required=60, base_xp=75, base_time=7.8, quality=QualityLevel.EPIC,
                market_value=100, coordinates=(125, 75), success_rate=0.66, respawn_time=570.0,
                tools_required=['hache_platine']
            ),
            'bambou_geant': ResourceData(
                id='bambou_geant', name='Bambou Géant', type=ResourceType.WOOD,
                level_required=65, base_xp=82, base_time=8.0, quality=QualityLevel.EPIC,
                market_value=120, coordinates=(135, 80), success_rate=0.64, respawn_time=600.0,
                tools_required=['hache_platine']
            ),
            'teck': ResourceData(
                id='teck', name='Teck', type=ResourceType.WOOD,
                level_required=70, base_xp=90, base_time=8.2, quality=QualityLevel.EPIC,
                market_value=140, coordinates=(145, 85), success_rate=0.62, respawn_time=630.0,
                tools_required=['hache_platine']
            ),
            'acajou': ResourceData(
                id='acajou', name='Acajou', type=ResourceType.WOOD,
                level_required=75, base_xp=98, base_time=8.5, quality=QualityLevel.LEGENDARY,
                market_value=160, coordinates=(155, 90), success_rate=0.60, respawn_time=660.0,
                tools_required=['hache_platine']
            ),
            'palissandre': ResourceData(
                id='palissandre', name='Palissandre', type=ResourceType.WOOD,
                level_required=80, base_xp=108, base_time=8.8, quality=QualityLevel.LEGENDARY,
                market_value=200, coordinates=(165, 95), success_rate=0.58, respawn_time=720.0,
                tools_required=['hache_platine']
            ),
            
            # Arbres légendaires niveau 80-100
            'olivier_ancien': ResourceData(
                id='olivier_ancien', name='Olivier Ancien', type=ResourceType.WOOD,
                level_required=80, base_xp=115, base_time=9.0, quality=QualityLevel.LEGENDARY,
                market_value=250, coordinates=(175, 100), success_rate=0.55, respawn_time=900.0,
                tools_required=['hache_platine']
            ),
            'sequoia': ResourceData(
                id='sequoia', name='Séquoia', type=ResourceType.WOOD,
                level_required=85, base_xp=125, base_time=9.5, quality=QualityLevel.LEGENDARY,
                market_value=300, coordinates=(185, 105), success_rate=0.52, respawn_time=1080.0,
                tools_required=['hache_platine']
            ),
            'baobab': ResourceData(
                id='baobab', name='Baobab', type=ResourceType.WOOD,
                level_required=90, base_xp=135, base_time=10.0, quality=QualityLevel.LEGENDARY,
                market_value=350, coordinates=(195, 110), success_rate=0.50, respawn_time=1200.0,
                tools_required=['hache_platine']
            ),
            'arbre_monde': ResourceData(
                id='arbre_monde', name='Arbre-Monde', type=ResourceType.WOOD,
                level_required=95, base_xp=150, base_time=10.5, quality=QualityLevel.LEGENDARY,
                market_value=400, coordinates=(200, 115), success_rate=0.48, respawn_time=1440.0,
                tools_required=['hache_platine']
            ),
            'yggdrasil': ResourceData(
                id='yggdrasil', name='Yggdrasil', type=ResourceType.WOOD,
                level_required=100, base_xp=200, base_time=12.0, quality=QualityLevel.LEGENDARY,
                market_value=500, coordinates=(210, 120), success_rate=0.45, respawn_time=1800.0,
                tools_required=['hache_platine']
            )
        }
    
    def get_optimal_route(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Calcule la route optimale de coupe basée sur respawn et rentabilité"""
        available_resources = self.get_available_resources(level_range)
        
        if not available_resources:
            return []
        
        # Prendre en compte les temps de respawn
        scored_resources = []
        for resource in available_resources:
            # Calculer la rentabilité avec temps de respawn
            profitability = self.calculate_profitability_with_respawn(resource.id)
            
            # Score basé sur XP/h, kamas/h et disponibilité
            xp_score = profitability.get('effective_xp_per_hour', 0)
            kamas_score = profitability.get('effective_kamas_per_hour', 0)
            availability_score = 1.0 / (resource.respawn_time / 60.0)  # Inversement proportionnel au respawn
            
            total_score = (
                xp_score * 0.4 +
                kamas_score * 0.3 +
                availability_score * 0.2 +
                resource.success_rate * 100 * 0.1
            )
            
            scored_resources.append((resource, total_score))
        
        # Trier par score décroissant
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Créer une route qui respecte les temps de respawn
        return self._create_respawn_aware_route([r[0] for r in scored_resources])
    
    def calculate_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité d'un arbre (sans considérer le respawn)"""
        resource = self.resources.get(resource_id)
        if not resource:
            return {}
        
        # Efficacité des outils
        tool_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
        
        # Temps effectif avec l'outil
        effective_time = resource.base_time / tool_efficiency
        movement_time = self._get_movement_time(resource)
        total_time_per_cut = effective_time + movement_time
        
        # Calculs par heure
        cuts_per_hour = 3600 / total_time_per_cut
        successful_cuts = cuts_per_hour * resource.success_rate
        
        xp_per_hour = successful_cuts * resource.base_xp
        kamas_per_hour = successful_cuts * resource.market_value
        efficiency = resource.base_xp / total_time_per_cut
        
        return {
            'xp_per_hour': round(xp_per_hour, 2),
            'kamas_per_hour': round(kamas_per_hour, 2),
            'efficiency': round(efficiency, 2),
            'cuts_per_hour': round(successful_cuts, 2),
            'time_per_cut': round(total_time_per_cut, 2),
            'tool_efficiency': tool_efficiency
        }
    
    def calculate_profitability_with_respawn(self, resource_id: str) -> Dict[str, float]:
        """Calcule la rentabilité en tenant compte du temps de respawn"""
        base_profit = self.calculate_profitability(resource_id)
        resource = self.resources.get(resource_id)
        
        if not resource or not base_profit:
            return {}
        
        # Facteur de réduction basé sur le respawn
        cutting_time = base_profit['time_per_cut']
        respawn_factor = cutting_time / (cutting_time + resource.respawn_time)
        
        # Rentabilité effective
        effective_xp = base_profit['xp_per_hour'] * respawn_factor
        effective_kamas = base_profit['kamas_per_hour'] * respawn_factor
        
        result = base_profit.copy()
        result.update({
            'effective_xp_per_hour': round(effective_xp, 2),
            'effective_kamas_per_hour': round(effective_kamas, 2),
            'respawn_factor': round(respawn_factor, 3),
            'respawn_time_minutes': round(resource.respawn_time / 60, 1)
        })
        
        return result
    
    def _create_respawn_aware_route(self, resources: List[ResourceData]) -> List[ResourceData]:
        """Crée une route qui optimise les temps de respawn"""
        if len(resources) <= 1:
            return resources
        
        # Grouper par temps de respawn similaires
        fast_respawn = [r for r in resources if r.respawn_time <= 300]  # 5 min
        medium_respawn = [r for r in resources if 300 < r.respawn_time <= 600]  # 5-10 min
        slow_respawn = [r for r in resources if r.respawn_time > 600]  # >10 min
        
        # Optimiser chaque groupe géographiquement
        optimized_route = []
        for group in [fast_respawn, medium_respawn, slow_respawn]:
            if group:
                optimized_group = self._optimize_geographic_route(group)
                optimized_route.extend(optimized_group)
        
        return optimized_route
    
    def _optimize_geographic_route(self, resources: List[ResourceData]) -> List[ResourceData]:
        """Optimise la route basée sur la proximité géographique"""
        if len(resources) <= 1:
            return resources
        
        # Algorithme du plus proche voisin
        unvisited = resources.copy()
        route = [unvisited.pop(0)]
        
        while unvisited:
            current = route[-1]
            closest = min(unvisited, key=lambda r: self._calculate_distance(current, r))
            route.append(closest)
            unvisited.remove(closest)
        
        return route
    
    def _calculate_distance(self, resource1: ResourceData, resource2: ResourceData) -> float:
        """Calcule la distance euclidienne entre deux arbres"""
        x1, y1 = resource1.coordinates
        x2, y2 = resource2.coordinates
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_cutting_pattern(self, resources: List[ResourceData], pattern_name: str = None) -> Dict[str, any]:
        """Applique un pattern de coupe spécifique"""
        pattern_name = pattern_name or self.current_pattern
        pattern_func = self.cutting_patterns.get(pattern_name, self._balanced_cutting)
        
        return pattern_func(resources)
    
    def _efficient_cutting(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern efficace - maximise XP/heure"""
        # Prioriser les arbres avec le meilleur ratio XP/temps
        sorted_resources = sorted(resources, 
                                key=lambda r: (r.base_xp / r.base_time) * r.success_rate, 
                                reverse=True)
        
        return {
            'pattern': 'efficient',
            'resources': sorted_resources[:10],  # Top 10
            'strategy': 'Maximiser XP/heure en ignorant la rentabilité',
            'estimated_xp_bonus': 1.2
        }
    
    def _sustainable_cutting(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern durable - respecte les temps de respawn"""
        # Équilibrer entre rentabilité et disponibilité
        balanced_resources = []
        for resource in resources:
            sustainability_score = (
                (resource.base_xp * 0.4) +
                (resource.market_value * 0.3) +
                (600 / max(resource.respawn_time, 60) * 0.3)  # Bonus pour respawn rapide
            )
            balanced_resources.append((resource, sustainability_score))
        
        balanced_resources.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'sustainable',
            'resources': [r[0] for r in balanced_resources[:8]],
            'strategy': 'Équilibrer rentabilité et préservation des ressources',
            'estimated_longevity': 'Élevée'
        }
    
    def _aggressive_cutting(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern agressif - maximise les gains à court terme"""
        # Prioriser la valeur marchande
        valuable_resources = sorted(resources, 
                                  key=lambda r: r.market_value * r.success_rate, 
                                  reverse=True)
        
        return {
            'pattern': 'aggressive',
            'resources': valuable_resources[:12],
            'strategy': 'Maximiser les profits immédiats',
            'estimated_kamas_bonus': 1.3,
            'warning': 'Peut épuiser rapidement les ressources locales'
        }
    
    def _balanced_cutting(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern équilibré - bon compromis général"""
        # Score composite
        balanced_resources = []
        for resource in resources:
            profit = self.calculate_profitability_with_respawn(resource.id)
            balance_score = (
                profit.get('effective_xp_per_hour', 0) * 0.3 +
                profit.get('effective_kamas_per_hour', 0) * 0.3 +
                resource.success_rate * 100 * 0.2 +
                (1000 / max(resource.respawn_time, 60)) * 0.2
            )
            balanced_resources.append((resource, balance_score))
        
        balanced_resources.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'balanced',
            'resources': [r[0] for r in balanced_resources[:10]],
            'strategy': 'Équilibre optimal entre XP, kamas et durabilité',
            'estimated_overall_efficiency': 1.1
        }
    
    def estimate_forest_depletion(self, resources: List[str], cutting_rate: float = 1.0) -> Dict[str, float]:
        """Estime le temps avant épuisement d'une zone forestière"""
        depletion_times = {}
        
        for resource_id in resources:
            resource = self.resources.get(resource_id)
            if not resource:
                continue
            
            # Hypothèse: 10 arbres de chaque type par zone
            trees_per_zone = 10
            cutting_time = resource.base_time / self.tools_efficiency.get(self.current_tool, 1.0)
            
            # Temps pour couper tous les arbres
            total_cutting_time = trees_per_zone * cutting_time * cutting_rate
            
            # Temps avant que le premier arbre respawn
            first_respawn = resource.respawn_time
            
            # Si la coupe est plus rapide que le respawn, dépletion temporaire
            if total_cutting_time < first_respawn:
                depletion_time = first_respawn - total_cutting_time
            else:
                depletion_time = 0  # Pas de dépletion
            
            depletion_times[resource_id] = {
                'depletion_time_minutes': round(depletion_time / 60, 2),
                'sustainable': depletion_time == 0,
                'trees_per_zone': trees_per_zone,
                'cutting_efficiency': self.tools_efficiency.get(self.current_tool, 1.0)
            }
        
        return depletion_times
    
    def upgrade_tool(self, new_tool: str) -> bool:
        """Met à niveau l'outil de coupe"""
        if new_tool in self.tools_efficiency:
            old_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
            new_efficiency = self.tools_efficiency[new_tool]
            
            self.current_tool = new_tool
            
            print(f"Outil mis à niveau: {new_tool}")
            print(f"Efficacité: {old_efficiency:.1f} → {new_efficiency:.1f}")
            print(f"Amélioration: {((new_efficiency/old_efficiency - 1) * 100):.1f}%")
            
            return True
        return False
    
    def get_tool_recommendations(self) -> Dict[str, any]:
        """Recommande des améliorations d'outils basées sur le niveau"""
        recommendations = {}
        current_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
        
        for tool, efficiency in self.tools_efficiency.items():
            if efficiency > current_efficiency:
                level_req = {
                    'hache_fer': 10,
                    'hache_bronze': 20,
                    'hache_argent': 40,
                    'hache_or': 60,
                    'hache_platine': 80
                }.get(tool, 1)
                
                if self.stats.level >= level_req:
                    improvement = ((efficiency / current_efficiency - 1) * 100)
                    recommendations[tool] = {
                        'efficiency': efficiency,
                        'improvement_percent': round(improvement, 1),
                        'level_required': level_req,
                        'recommended': improvement >= 15  # Recommander si >15% d'amélioration
                    }
        
        return recommendations
    
    def __str__(self) -> str:
        available_trees = len(self.get_available_resources())
        tool_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
        return f"Bûcheron (Niveau {self.stats.level}) - {available_trees} arbres disponibles - Efficacité outil: {tool_efficiency:.1f}x"