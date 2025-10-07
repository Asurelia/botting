"""
Module Mineur - Gestion complète du métier de Mineur.
Inclut tous les minerais, optimisation de routes minières et calculs de rentabilité.
"""

from typing import Dict, List, Tuple, Optional
from .base import BaseProfession, ResourceData, ResourceType, QualityLevel
import math
import random

class Miner(BaseProfession):
    """Classe Mineur avec tous les types de minerais"""
    
    def __init__(self):
        super().__init__("Mineur", "miner")
        self.load_resources()
        self.mining_patterns = {
            'surface': self._surface_mining,
            'deep': self._deep_mining,
            'vein_following': self._vein_following,
            'systematic': self._systematic_mining
        }
        self.current_pattern = 'systematic'
        self.tools_efficiency = {
            'pioche_debutant': 1.0,
            'pioche_fer': 1.3,
            'pioche_bronze': 1.5,
            'pioche_argent': 1.8,
            'pioche_or': 2.1,
            'pioche_platine': 2.5,
            'pioche_diamant': 3.0
        }
        self.current_tool = 'pioche_debutant'
        self.mine_levels = {
            'surface': (1, 30),
            'shallow': (20, 50),
            'deep': (40, 70),
            'abyssal': (60, 100)
        }
        
    def load_resources(self) -> None:
        """Charge tous les types de minerais"""
        self.resources = {
            # Minerais de surface niveau 1-30
            'fer': ResourceData(
                id='fer', name='Fer', type=ResourceType.MINERAL,
                level_required=1, base_xp=8, base_time=5.0, quality=QualityLevel.COMMON,
                market_value=5, coordinates=(5, 10), success_rate=0.92, respawn_time=180.0,
                tools_required=['pioche_debutant']
            ),
            'cuivre': ResourceData(
                id='cuivre', name='Cuivre', type=ResourceType.MINERAL,
                level_required=3, base_xp=10, base_time=5.2, quality=QualityLevel.COMMON,
                market_value=7, coordinates=(12, 18), success_rate=0.90, respawn_time=200.0,
                tools_required=['pioche_debutant']
            ),
            'bronze': ResourceData(
                id='bronze', name='Bronze', type=ResourceType.MINERAL,
                level_required=8, base_xp=14, base_time=5.5, quality=QualityLevel.COMMON,
                market_value=12, coordinates=(20, 25), success_rate=0.88, respawn_time=240.0,
                tools_required=['pioche_fer']
            ),
            'etain': ResourceData(
                id='etain', name='Étain', type=ResourceType.MINERAL,
                level_required=12, base_xp=18, base_time=5.8, quality=QualityLevel.COMMON,
                market_value=15, coordinates=(28, 30), success_rate=0.86, respawn_time=270.0,
                tools_required=['pioche_fer']
            ),
            'argent': ResourceData(
                id='argent', name='Argent', type=ResourceType.MINERAL,
                level_required=18, base_xp=24, base_time=6.0, quality=QualityLevel.UNCOMMON,
                market_value=25, coordinates=(35, 35), success_rate=0.83, respawn_time=300.0,
                tools_required=['pioche_bronze']
            ),
            'bauxite': ResourceData(
                id='bauxite', name='Bauxite', type=ResourceType.MINERAL,
                level_required=22, base_xp=28, base_time=6.2, quality=QualityLevel.UNCOMMON,
                market_value=30, coordinates=(42, 40), success_rate=0.81, respawn_time=330.0,
                tools_required=['pioche_bronze']
            ),
            'or': ResourceData(
                id='or', name='Or', type=ResourceType.MINERAL,
                level_required=25, base_xp=35, base_time=6.5, quality=QualityLevel.UNCOMMON,
                market_value=45, coordinates=(50, 45), success_rate=0.78, respawn_time=360.0,
                tools_required=['pioche_bronze']
            ),
            'cobalt': ResourceData(
                id='cobalt', name='Cobalt', type=ResourceType.MINERAL,
                level_required=30, base_xp=42, base_time=7.0, quality=QualityLevel.RARE,
                market_value=60, coordinates=(58, 50), success_rate=0.75, respawn_time=420.0,
                tools_required=['pioche_argent']
            ),
            
            # Minerais des mines peu profondes niveau 20-50
            'manganese': ResourceData(
                id='manganese', name='Manganèse', type=ResourceType.MINERAL,
                level_required=20, base_xp=30, base_time=6.3, quality=QualityLevel.UNCOMMON,
                market_value=35, coordinates=(25, 60), success_rate=0.80, respawn_time=390.0,
                tools_required=['pioche_bronze']
            ),
            'silice': ResourceData(
                id='silice', name='Silice', type=ResourceType.MINERAL,
                level_required=28, base_xp=38, base_time=6.8, quality=QualityLevel.RARE,
                market_value=50, coordinates=(35, 65), success_rate=0.77, respawn_time=450.0,
                tools_required=['pioche_argent']
            ),
            'platine': ResourceData(
                id='platine', name='Platine', type=ResourceType.MINERAL,
                level_required=35, base_xp=48, base_time=7.2, quality=QualityLevel.RARE,
                market_value=80, coordinates=(45, 70), success_rate=0.73, respawn_time=480.0,
                tools_required=['pioche_argent']
            ),
            'palladium': ResourceData(
                id='palladium', name='Palladium', type=ResourceType.MINERAL,
                level_required=40, base_xp=55, base_time=7.5, quality=QualityLevel.RARE,
                market_value=100, coordinates=(55, 75), success_rate=0.70, respawn_time=540.0,
                tools_required=['pioche_or']
            ),
            'titane': ResourceData(
                id='titane', name='Titane', type=ResourceType.MINERAL,
                level_required=45, base_xp=65, base_time=8.0, quality=QualityLevel.EPIC,
                market_value=130, coordinates=(65, 80), success_rate=0.67, respawn_time=600.0,
                tools_required=['pioche_or']
            ),
            'rhodium': ResourceData(
                id='rhodium', name='Rhodium', type=ResourceType.MINERAL,
                level_required=50, base_xp=75, base_time=8.5, quality=QualityLevel.EPIC,
                market_value=170, coordinates=(75, 85), success_rate=0.63, respawn_time=660.0,
                tools_required=['pioche_or']
            ),
            
            # Minerais des mines profondes niveau 40-70
            'mithril': ResourceData(
                id='mithril', name='Mithril', type=ResourceType.MINERAL,
                level_required=40, base_xp=70, base_time=8.2, quality=QualityLevel.EPIC,
                market_value=150, coordinates=(85, 90), success_rate=0.65, respawn_time=720.0,
                tools_required=['pioche_or']
            ),
            'adamantium': ResourceData(
                id='adamantium', name='Adamantium', type=ResourceType.MINERAL,
                level_required=50, base_xp=85, base_time=9.0, quality=QualityLevel.EPIC,
                market_value=200, coordinates=(95, 95), success_rate=0.60, respawn_time=780.0,
                tools_required=['pioche_platine']
            ),
            'orichalque': ResourceData(
                id='orichalque', name='Orichalque', type=ResourceType.MINERAL,
                level_required=55, base_xp=95, base_time=9.5, quality=QualityLevel.LEGENDARY,
                market_value=250, coordinates=(105, 100), success_rate=0.57, respawn_time=840.0,
                tools_required=['pioche_platine']
            ),
            'vibranium': ResourceData(
                id='vibranium', name='Vibranium', type=ResourceType.MINERAL,
                level_required=60, base_xp=110, base_time=10.0, quality=QualityLevel.LEGENDARY,
                market_value=320, coordinates=(115, 105), success_rate=0.53, respawn_time=900.0,
                tools_required=['pioche_platine']
            ),
            'unobtainium': ResourceData(
                id='unobtainium', name='Unobtainium', type=ResourceType.MINERAL,
                level_required=65, base_xp=125, base_time=10.5, quality=QualityLevel.LEGENDARY,
                market_value=400, coordinates=(125, 110), success_rate=0.50, respawn_time=1020.0,
                tools_required=['pioche_diamant']
            ),
            'elementium': ResourceData(
                id='elementium', name='Élémentium', type=ResourceType.MINERAL,
                level_required=70, base_xp=145, base_time=11.0, quality=QualityLevel.LEGENDARY,
                market_value=500, coordinates=(135, 115), success_rate=0.47, respawn_time=1140.0,
                tools_required=['pioche_diamant']
            ),
            
            # Minerais abyssaux niveau 60-100
            'obsidienne': ResourceData(
                id='obsidienne', name='Obsidienne', type=ResourceType.MINERAL,
                level_required=60, base_xp=100, base_time=9.8, quality=QualityLevel.EPIC,
                market_value=280, coordinates=(145, 120), success_rate=0.55, respawn_time=960.0,
                tools_required=['pioche_platine']
            ),
            'cristal_abyssal': ResourceData(
                id='cristal_abyssal', name='Cristal Abyssal', type=ResourceType.MINERAL,
                level_required=70, base_xp=140, base_time=11.5, quality=QualityLevel.LEGENDARY,
                market_value=450, coordinates=(155, 125), success_rate=0.48, respawn_time=1200.0,
                tools_required=['pioche_diamant']
            ),
            'stellarium': ResourceData(
                id='stellarium', name='Stellarium', type=ResourceType.MINERAL,
                level_required=75, base_xp=160, base_time=12.0, quality=QualityLevel.LEGENDARY,
                market_value=550, coordinates=(165, 130), success_rate=0.45, respawn_time=1320.0,
                tools_required=['pioche_diamant']
            ),
            'voidstone': ResourceData(
                id='voidstone', name='Pierre du Vide', type=ResourceType.MINERAL,
                level_required=80, base_xp=180, base_time=12.5, quality=QualityLevel.LEGENDARY,
                market_value=650, coordinates=(175, 135), success_rate=0.42, respawn_time=1440.0,
                tools_required=['pioche_diamant']
            ),
            'chronite': ResourceData(
                id='chronite', name='Chronite', type=ResourceType.MINERAL,
                level_required=85, base_xp=200, base_time=13.0, quality=QualityLevel.LEGENDARY,
                market_value=750, coordinates=(185, 140), success_rate=0.40, respawn_time=1620.0,
                tools_required=['pioche_diamant']
            ),
            'quintessence': ResourceData(
                id='quintessence', name='Quintessence', type=ResourceType.MINERAL,
                level_required=90, base_xp=230, base_time=14.0, quality=QualityLevel.LEGENDARY,
                market_value=900, coordinates=(195, 145), success_rate=0.37, respawn_time=1800.0,
                tools_required=['pioche_diamant']
            ),
            'cosmicium': ResourceData(
                id='cosmicium', name='Cosmicium', type=ResourceType.MINERAL,
                level_required=95, base_xp=260, base_time=15.0, quality=QualityLevel.LEGENDARY,
                market_value=1100, coordinates=(205, 150), success_rate=0.35, respawn_time=2100.0,
                tools_required=['pioche_diamant']
            ),
            'eternium': ResourceData(
                id='eternium', name='Éternium', type=ResourceType.MINERAL,
                level_required=100, base_xp=300, base_time=16.0, quality=QualityLevel.LEGENDARY,
                market_value=1500, coordinates=(215, 155), success_rate=0.32, respawn_time=2400.0,
                tools_required=['pioche_diamant']
            ),
            
            # Gemmes rares (tous niveaux)
            'emeraude': ResourceData(
                id='emeraude', name='Émeraude', type=ResourceType.MINERAL,
                level_required=25, base_xp=40, base_time=7.5, quality=QualityLevel.RARE,
                market_value=80, coordinates=(30, 120), success_rate=0.25, respawn_time=600.0,
                tools_required=['pioche_bronze']
            ),
            'saphir': ResourceData(
                id='saphir', name='Saphir', type=ResourceType.MINERAL,
                level_required=35, base_xp=60, base_time=8.5, quality=QualityLevel.EPIC,
                market_value=150, coordinates=(50, 125), success_rate=0.20, respawn_time=720.0,
                tools_required=['pioche_argent']
            ),
            'rubis': ResourceData(
                id='rubis', name='Rubis', type=ResourceType.MINERAL,
                level_required=45, base_xp=80, base_time=9.5, quality=QualityLevel.EPIC,
                market_value=220, coordinates=(70, 130), success_rate=0.18, respawn_time=900.0,
                tools_required=['pioche_or']
            ),
            'diamant': ResourceData(
                id='diamant', name='Diamant', type=ResourceType.MINERAL,
                level_required=60, base_xp=120, base_time=12.0, quality=QualityLevel.LEGENDARY,
                market_value=400, coordinates=(100, 135), success_rate=0.15, respawn_time=1200.0,
                tools_required=['pioche_platine']
            ),
            'diamant_noir': ResourceData(
                id='diamant_noir', name='Diamant Noir', type=ResourceType.MINERAL,
                level_required=80, base_xp=200, base_time=15.0, quality=QualityLevel.LEGENDARY,
                market_value=800, coordinates=(150, 140), success_rate=0.10, respawn_time=1800.0,
                tools_required=['pioche_diamant']
            )
        }
    
    def get_optimal_route(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Calcule la route optimale de minage basée sur les niveaux de mine"""
        available_resources = self.get_available_resources(level_range)
        
        if not available_resources:
            return []
        
        # Grouper par niveau de mine
        mines_groups = {
            'surface': [],
            'shallow': [],
            'deep': [],
            'abyssal': []
        }
        
        for resource in available_resources:
            level = resource.level_required
            if level <= 30:
                mines_groups['surface'].append(resource)
            elif level <= 50:
                mines_groups['shallow'].append(resource)
            elif level <= 70:
                mines_groups['deep'].append(resource)
            else:
                mines_groups['abyssal'].append(resource)
        
        # Optimiser chaque niveau de mine séparément
        optimized_route = []
        for mine_level, resources in mines_groups.items():
            if resources:
                scored_resources = self._score_resources_by_profitability(resources)
                optimized_mine = self._optimize_mine_route(scored_resources, mine_level)
                optimized_route.extend(optimized_mine)
        
        return optimized_route
    
    def _score_resources_by_profitability(self, resources: List[ResourceData]) -> List[Tuple[ResourceData, float]]:
        """Score les ressources par rentabilité"""
        scored_resources = []
        for resource in resources:
            profitability = self.calculate_mining_profitability(resource.id)
            
            # Score composite avec bonus pour les gemmes
            base_score = (
                profitability.get('effective_xp_per_hour', 0) * 0.3 +
                profitability.get('effective_kamas_per_hour', 0) * 0.4 +
                resource.success_rate * 100 * 0.2 +
                (1 / max(resource.respawn_time / 60, 1)) * 0.1
            )
            
            # Bonus pour les gemmes rares
            gem_bonus = 1.5 if resource.id in ['emeraude', 'saphir', 'rubis', 'diamant', 'diamant_noir'] else 1.0
            final_score = base_score * gem_bonus
            
            scored_resources.append((resource, final_score))
        
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        return scored_resources
    
    def _optimize_mine_route(self, scored_resources: List[Tuple[ResourceData, float]], mine_level: str) -> List[ResourceData]:
        """Optimise la route dans une mine spécifique"""
        # Prendre les meilleures ressources selon la capacité de la mine
        capacity = {
            'surface': 8,
            'shallow': 6,
            'deep': 5,
            'abyssal': 3
        }.get(mine_level, 5)
        
        top_resources = [r[0] for r in scored_resources[:capacity]]
        
        # Optimiser géographiquement
        return self._optimize_geographic_route(top_resources)
    
    def calculate_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité d'un minerai (version de base)"""
        return self.calculate_mining_profitability(resource_id, duration)
    
    def calculate_mining_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité minière avec spécificités du métier"""
        resource = self.resources.get(resource_id)
        if not resource:
            return {}
        
        # Efficacité des outils
        tool_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
        
        # Temps effectif avec l'outil et difficulté de la mine
        mine_difficulty = self._get_mine_difficulty(resource.level_required)
        effective_mining_time = (resource.base_time / tool_efficiency) * mine_difficulty
        
        movement_time = self._get_movement_time(resource)
        inventory_time = self._get_inventory_time()
        
        total_time_per_mining = effective_mining_time + movement_time + inventory_time
        
        # Calculs par heure
        minings_per_hour = 3600 / total_time_per_mining
        successful_minings = minings_per_hour * resource.success_rate
        
        # Facteur de réduction basé sur le respawn
        respawn_factor = total_time_per_mining / (total_time_per_mining + resource.respawn_time)
        
        effective_xp = successful_minings * resource.base_xp * respawn_factor
        effective_kamas = successful_minings * resource.market_value * respawn_factor
        
        return {
            'xp_per_hour': round(successful_minings * resource.base_xp, 2),
            'kamas_per_hour': round(successful_minings * resource.market_value, 2),
            'effective_xp_per_hour': round(effective_xp, 2),
            'effective_kamas_per_hour': round(effective_kamas, 2),
            'minings_per_hour': round(successful_minings, 2),
            'time_per_mining': round(total_time_per_mining, 2),
            'tool_efficiency': tool_efficiency,
            'mine_difficulty': mine_difficulty,
            'respawn_factor': round(respawn_factor, 3),
            'rarity_bonus': 1.0 / max(resource.success_rate, 0.1)  # Bonus inversé à la rareté
        }
    
    def _get_mine_difficulty(self, required_level: int) -> float:
        """Calcule la difficulté de minage selon le niveau de mine"""
        if required_level <= 30:
            return 1.0  # Surface - facile
        elif required_level <= 50:
            return 1.2  # Peu profond - moyen
        elif required_level <= 70:
            return 1.5  # Profond - difficile
        else:
            return 2.0  # Abyssal - très difficile
    
    def _optimize_geographic_route(self, resources: List[ResourceData]) -> List[ResourceData]:
        """Optimise la route basée sur la proximité géographique dans les mines"""
        if len(resources) <= 1:
            return resources
        
        # Dans les mines, on considère aussi la profondeur (coordonnée Z implicite)
        unvisited = resources.copy()
        route = [unvisited.pop(0)]
        
        while unvisited:
            current = route[-1]
            # Distance pondérée par le niveau (profondeur)
            closest = min(unvisited, key=lambda r: self._calculate_mine_distance(current, r))
            route.append(closest)
            unvisited.remove(closest)
        
        return route
    
    def _calculate_mine_distance(self, resource1: ResourceData, resource2: ResourceData) -> float:
        """Calcule la distance dans les mines (incluant la profondeur)"""
        x1, y1 = resource1.coordinates
        x2, y2 = resource2.coordinates
        
        # Distance géographique
        geo_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Distance de niveau (profondeur)
        level_distance = abs(resource2.level_required - resource1.level_required)
        
        # Distance totale pondérée
        return geo_distance + (level_distance * 0.5)
    
    def get_mining_pattern(self, resources: List[ResourceData], pattern_name: str = None) -> Dict[str, any]:
        """Applique un pattern de minage spécifique"""
        pattern_name = pattern_name or self.current_pattern
        pattern_func = self.mining_patterns.get(pattern_name, self._systematic_mining)
        
        return pattern_func(resources)
    
    def _surface_mining(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern de minage en surface - rapide et efficace"""
        surface_resources = [r for r in resources if r.level_required <= 30]
        surface_resources.sort(key=lambda r: r.base_time)  # Plus rapides en premier
        
        return {
            'pattern': 'surface',
            'resources': surface_resources[:10],
            'strategy': 'Exploitation rapide des minerais de surface',
            'estimated_speed_bonus': 1.3,
            'depth_range': '0-30m'
        }
    
    def _deep_mining(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern de minage en profondeur - vise les minerais rares"""
        deep_resources = [r for r in resources if r.level_required >= 50]
        deep_resources.sort(key=lambda r: r.market_value, reverse=True)
        
        return {
            'pattern': 'deep',
            'resources': deep_resources[:8],
            'strategy': 'Exploitation des minerais rares en profondeur',
            'estimated_value_bonus': 1.8,
            'depth_range': '50m+',
            'risk_factor': 'Élevé'
        }
    
    def _vein_following(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern de suivi de filons - suit les concentrations"""
        # Grouper par proximité géographique (filons)
        veins = self._detect_veins(resources)
        best_vein = max(veins, key=lambda v: sum(r.market_value for r in v)) if veins else []
        
        return {
            'pattern': 'vein_following',
            'resources': best_vein,
            'strategy': 'Suivi du filon le plus rentable',
            'vein_count': len(veins),
            'estimated_efficiency': 1.4,
            'note': 'Concentre les efforts sur les zones riches'
        }
    
    def _systematic_mining(self, resources: List[ResourceData]) -> Dict[str, any]:
        """Pattern systématique - équilibre optimal"""
        # Score basé sur tous les facteurs
        scored_resources = []
        for resource in resources:
            profit = self.calculate_mining_profitability(resource.id)
            systematic_score = (
                profit.get('effective_xp_per_hour', 0) * 0.25 +
                profit.get('effective_kamas_per_hour', 0) * 0.35 +
                resource.success_rate * 50 * 0.2 +
                (1 / max(resource.respawn_time / 300, 1)) * 0.2
            )
            scored_resources.append((resource, systematic_score))
        
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'systematic',
            'resources': [r[0] for r in scored_resources[:12]],
            'strategy': 'Approche systématique équilibrée',
            'estimated_overall_efficiency': 1.2,
            'balance': 'Optimal entre tous les facteurs'
        }
    
    def _detect_veins(self, resources: List[ResourceData]) -> List[List[ResourceData]]:
        """Détecte les filons de minerais (groupes proches)"""
        veins = []
        processed = set()
        
        for resource in resources:
            if resource.id in processed:
                continue
            
            # Créer un nouveau filon
            vein = [resource]
            processed.add(resource.id)
            
            # Chercher les minerais proches
            for other_resource in resources:
                if (other_resource.id not in processed and 
                    self._calculate_mine_distance(resource, other_resource) < 25):
                    vein.append(other_resource)
                    processed.add(other_resource.id)
            
            if len(vein) >= 2:  # Filon valide si au moins 2 minerais
                veins.append(vein)
        
        return veins
    
    def estimate_excavation_time(self, resources: List[str], target_quantity: int = 50) -> Dict[str, any]:
        """Estime le temps d'excavation pour une quantité donnée"""
        estimates = {}
        
        for resource_id in resources:
            resource = self.resources.get(resource_id)
            if not resource:
                continue
            
            profitability = self.calculate_mining_profitability(resource_id)
            time_per_mining = profitability.get('time_per_mining', 0)
            
            # Prendre en compte le taux de succès
            expected_attempts = target_quantity / resource.success_rate
            total_time = expected_attempts * time_per_mining
            
            # Temps d'attente pour les respawns
            if resource.respawn_time > time_per_mining:
                wait_cycles = max(0, expected_attempts - 1)
                total_wait_time = wait_cycles * (resource.respawn_time - time_per_mining)
            else:
                total_wait_time = 0
            
            estimates[resource_id] = {
                'mining_time_hours': round(total_time / 3600, 2),
                'wait_time_hours': round(total_wait_time / 3600, 2),
                'total_time_hours': round((total_time + total_wait_time) / 3600, 2),
                'attempts_needed': round(expected_attempts),
                'estimated_xp': round(target_quantity * resource.base_xp),
                'estimated_kamas': round(target_quantity * resource.market_value),
                'mine_level': self._get_mine_level_name(resource.level_required)
            }
        
        return estimates
    
    def _get_mine_level_name(self, required_level: int) -> str:
        """Retourne le nom du niveau de mine"""
        if required_level <= 30:
            return 'Surface'
        elif required_level <= 50:
            return 'Peu profonde'
        elif required_level <= 70:
            return 'Profonde'
        else:
            return 'Abyssale'
    
    def upgrade_tool(self, new_tool: str) -> bool:
        """Met à niveau l'outil de minage"""
        if new_tool in self.tools_efficiency:
            old_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
            new_efficiency = self.tools_efficiency[new_tool]
            
            self.current_tool = new_tool
            
            print(f"Outil mis à niveau: {new_tool}")
            print(f"Efficacité: {old_efficiency:.1f} → {new_efficiency:.1f}")
            print(f"Amélioration: {((new_efficiency/old_efficiency - 1) * 100):.1f}%")
            
            return True
        return False
    
    def get_gem_hunting_strategy(self) -> Dict[str, any]:
        """Stratégie spécialisée pour la chasse aux gemmes"""
        gems = ['emeraude', 'saphir', 'rubis', 'diamant', 'diamant_noir']
        available_gems = [self.resources[gem_id] for gem_id in gems if gem_id in self.resources and self.can_harvest(gem_id)]
        
        if not available_gems:
            return {'strategy': 'Aucune gemme accessible à votre niveau'}
        
        # Calculer le potentiel de profit par gemme
        gem_profits = {}
        for gem in available_gems:
            profit = self.calculate_mining_profitability(gem.id)
            expected_value_per_hour = profit.get('effective_kamas_per_hour', 0) / gem.success_rate
            gem_profits[gem.id] = {
                'resource': gem,
                'expected_hourly_value': round(expected_value_per_hour, 2),
                'rarity_multiplier': round(1 / gem.success_rate, 1)
            }
        
        # Trier par valeur attendue
        sorted_gems = sorted(gem_profits.items(), key=lambda x: x[1]['expected_hourly_value'], reverse=True)
        
        return {
            'strategy': 'Chasse aux gemmes optimisée',
            'recommended_gems': [item[0] for item in sorted_gems[:3]],
            'gem_analysis': dict(sorted_gems),
            'estimated_daily_profit': sum(item[1]['expected_hourly_value'] for item in sorted_gems[:3]) * 8,
            'warning': 'Taux de succès très faibles - patience requise'
        }
    
    def __str__(self) -> str:
        available_minerals = len(self.get_available_resources())
        tool_efficiency = self.tools_efficiency.get(self.current_tool, 1.0)
        return f"Mineur (Niveau {self.stats.level}) - {available_minerals} minerais disponibles - Efficacité outil: {tool_efficiency:.1f}x"