"""
Module d'optimisation des métiers DOFUS.
Calcule les meilleures stratégies XP/Kamas en fonction du marché actuel.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math


class OptimizationGoal(Enum):
    """Objectifs d'optimisation disponibles."""
    XP_MAX = "xp_max"           # Maximiser l'XP
    KAMAS_MAX = "kamas_max"     # Maximiser les Kamas
    BALANCED = "balanced"       # Équilibre XP/Kamas
    TIME_EFFICIENT = "time_efficient"  # Efficacité temporelle
    RESOURCE_EFFICIENT = "resource_efficient"  # Économie de ressources


class ActivityType(Enum):
    """Types d'activités de métier."""
    GATHERING = "gathering"     # Récolte
    CRAFTING = "crafting"      # Artisanat
    TRADING = "trading"        # Commerce


@dataclass
class ActivityRecommendation:
    """Recommandation d'activité optimisée."""
    activity_type: ActivityType
    profession: str
    specific_action: str        # Action spécifique (ex: "Récolter Blé")
    location: str              # Lieu recommandé
    expected_xp_per_hour: float
    expected_kamas_per_hour: float
    resource_cost: Dict[str, int]  # Coût en ressources
    time_estimate: int         # Temps estimé en minutes
    difficulty_score: float    # Score de difficulté (0-10)
    market_opportunity: float  # Score d'opportunité marché (0-10)
    confidence: float         # Confiance dans la prédiction
    requirements: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class MarketData:
    """Données de marché pour optimisation."""
    item_name: str
    current_price: float
    price_trend: str          # "rising", "falling", "stable"
    supply: int              # Quantité disponible
    demand_score: float      # Score de demande
    volatility: float        # Volatilité du prix
    last_updated: datetime


@dataclass
class ProfessionState:
    """État actuel d'un métier."""
    name: str
    level: int
    current_xp: int
    xp_to_next_level: int
    available_recipes: List[str]
    unlocked_areas: List[str]
    current_tools: List[str]
    inventory: Dict[str, int]


class ProfessionOptimizer:
    """
    Optimiseur principal des métiers DOFUS.
    Calcule les meilleures stratégies selon différents critères.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Base de données des métiers et recettes
        self.profession_data = {}
        self.recipe_database = {}
        self.market_cache = {}
        self.optimization_cache = {}
        
        # Configuration d'optimisation
        self.update_interval = config.get('update_interval', 300)  # 5 minutes
        self.market_data_max_age = config.get('market_data_max_age', 1800)  # 30 minutes
        self.cache_max_age = config.get('cache_max_age', 600)  # 10 minutes
        
        # Poids pour l'optimisation multi-objectifs
        self.optimization_weights = {
            OptimizationGoal.XP_MAX: {'xp': 1.0, 'kamas': 0.1, 'time': 0.3},
            OptimizationGoal.KAMAS_MAX: {'xp': 0.1, 'kamas': 1.0, 'time': 0.3},
            OptimizationGoal.BALANCED: {'xp': 0.5, 'kamas': 0.5, 'time': 0.4},
            OptimizationGoal.TIME_EFFICIENT: {'xp': 0.3, 'kamas': 0.3, 'time': 1.0},
            OptimizationGoal.RESOURCE_EFFICIENT: {'xp': 0.4, 'kamas': 0.4, 'time': 0.5}
        }
        
        self._load_profession_data()
    
    def _load_profession_data(self):
        """Charge les données des métiers."""
        try:
            # Données simplifiées pour la démo
            self.profession_data = {
                'Alchimiste': {
                    'base_xp_per_craft': {'1-50': 10, '51-100': 25, '101-150': 50, '151-200': 100},
                    'popular_recipes': ['Potion de Rappel', 'Pain', 'Potion de Soin'],
                    'gathering_spots': ['Champs', 'Forêts', 'Montagnes']
                },
                'Forgeron': {
                    'base_xp_per_craft': {'1-50': 15, '51-100': 30, '101-150': 60, '151-200': 120},
                    'popular_recipes': ['Épée de Boisaille', 'Marteau du Craqueleur', 'Anneau du Bouftou'],
                    'gathering_spots': ['Mines', 'Caves']
                },
                'Tailleur': {
                    'base_xp_per_craft': {'1-50': 12, '51-100': 28, '101-150': 55, '151-200': 110},
                    'popular_recipes': ['Cape de Bontarian', 'Coiffe du Bouftou', 'Ceinture du Sanglier'],
                    'gathering_spots': ['Zones à monstres', 'Donjons']
                }
            }
            
            # Base de recettes simplifiée
            self.recipe_database = {
                'Pain': {
                    'ingredients': {'Blé': 1},
                    'base_xp': 10,
                    'craft_time': 5,
                    'min_level': 1,
                    'profession': 'Alchimiste'
                },
                'Potion de Soin': {
                    'ingredients': {'Ortie': 1, 'Trèfle à 5 Feuilles': 1},
                    'base_xp': 25,
                    'craft_time': 8,
                    'min_level': 20,
                    'profession': 'Alchimiste'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur chargement données métiers: {e}")
    
    def get_optimization_recommendations(self, 
                                       profession_states: List[ProfessionState],
                                       goal: OptimizationGoal,
                                       time_budget: int = 60,
                                       include_market_analysis: bool = True) -> List[ActivityRecommendation]:
        """
        Génère des recommandations d'optimisation.
        
        Args:
            profession_states: États actuels des métiers
            goal: Objectif d'optimisation
            time_budget: Budget temps en minutes
            include_market_analysis: Inclure l'analyse marché
            
        Returns:
            Liste des recommandations triées par score
        """
        try:
            cache_key = f"{goal.value}_{time_budget}_{len(profession_states)}"
            
            # Vérifier le cache
            if cache_key in self.optimization_cache:
                cached_data = self.optimization_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_max_age:
                    return cached_data['recommendations']
            
            recommendations = []
            
            for profession_state in profession_states:
                profession_recs = self._analyze_profession_opportunities(
                    profession_state, goal, time_budget, include_market_analysis
                )
                recommendations.extend(profession_recs)
            
            # Trier par score composite
            recommendations.sort(key=lambda x: self._calculate_composite_score(x, goal), reverse=True)
            
            # Mettre en cache
            self.optimization_cache[cache_key] = {
                'timestamp': datetime.now(),
                'recommendations': recommendations[:10]  # Top 10
            }
            
            return recommendations[:10]
            
        except Exception as e:
            self.logger.error(f"Erreur génération recommandations: {e}")
            return []
    
    def _analyze_profession_opportunities(self, 
                                        profession_state: ProfessionState,
                                        goal: OptimizationGoal,
                                        time_budget: int,
                                        include_market: bool) -> List[ActivityRecommendation]:
        """Analyse les opportunités pour un métier spécifique."""
        recommendations = []
        
        try:
            # Analyser les opportunités de craft
            craft_opportunities = self._analyze_crafting_opportunities(
                profession_state, goal, time_budget, include_market
            )
            recommendations.extend(craft_opportunities)
            
            # Analyser les opportunités de récolte
            gathering_opportunities = self._analyze_gathering_opportunities(
                profession_state, goal, time_budget
            )
            recommendations.extend(gathering_opportunities)
            
            # Analyser les opportunités de trading
            if include_market:
                trading_opportunities = self._analyze_trading_opportunities(
                    profession_state, goal, time_budget
                )
                recommendations.extend(trading_opportunities)
        
        except Exception as e:
            self.logger.error(f"Erreur analyse opportunités {profession_state.name}: {e}")
        
        return recommendations
    
    def _analyze_crafting_opportunities(self, 
                                      profession_state: ProfessionState,
                                      goal: OptimizationGoal,
                                      time_budget: int,
                                      include_market: bool) -> List[ActivityRecommendation]:
        """Analyse les opportunités de craft."""
        opportunities = []
        
        try:
            available_recipes = self._get_available_recipes(profession_state)
            
            for recipe_name, recipe_data in available_recipes.items():
                if recipe_data['min_level'] > profession_state.level:
                    continue
                
                # Calculer les métriques
                xp_per_hour = self._calculate_craft_xp_per_hour(recipe_data, profession_state)
                kamas_per_hour = 0
                
                if include_market:
                    kamas_per_hour = self._calculate_craft_profit_per_hour(recipe_data, recipe_name)
                
                # Estimer le coût en ressources
                resource_cost = recipe_data.get('ingredients', {})
                
                # Calculer la difficulté
                difficulty = self._calculate_craft_difficulty(recipe_data, profession_state)
                
                # Score d'opportunité marché
                market_opportunity = 5.0  # Score par défaut
                if include_market:
                    market_opportunity = self._calculate_market_opportunity(recipe_name)
                
                recommendation = ActivityRecommendation(
                    activity_type=ActivityType.CRAFTING,
                    profession=profession_state.name,
                    specific_action=f"Crafter {recipe_name}",
                    location="Atelier",
                    expected_xp_per_hour=xp_per_hour,
                    expected_kamas_per_hour=kamas_per_hour,
                    resource_cost=resource_cost,
                    time_estimate=min(time_budget, 60),
                    difficulty_score=difficulty,
                    market_opportunity=market_opportunity,
                    confidence=0.8,
                    requirements=[f"Niveau {recipe_data['min_level']} minimum"],
                    notes=[f"XP par craft: {recipe_data['base_xp']}"]
                )
                
                opportunities.append(recommendation)
        
        except Exception as e:
            self.logger.error(f"Erreur analyse craft {profession_state.name}: {e}")
        
        return opportunities
    
    def _analyze_gathering_opportunities(self,
                                       profession_state: ProfessionState,
                                       goal: OptimizationGoal,
                                       time_budget: int) -> List[ActivityRecommendation]:
        """Analyse les opportunités de récolte."""
        opportunities = []
        
        try:
            profession_info = self.profession_data.get(profession_state.name, {})
            gathering_spots = profession_info.get('gathering_spots', [])
            
            for spot in gathering_spots:
                # Simuler des données de récolte
                base_xp = profession_state.level * 2
                xp_per_hour = base_xp * 30  # 30 récoltes par heure
                
                recommendation = ActivityRecommendation(
                    activity_type=ActivityType.GATHERING,
                    profession=profession_state.name,
                    specific_action=f"Récolte en {spot}",
                    location=spot,
                    expected_xp_per_hour=xp_per_hour,
                    expected_kamas_per_hour=0,  # Pas de profit direct
                    resource_cost={},
                    time_estimate=min(time_budget, 120),
                    difficulty_score=3.0,
                    market_opportunity=6.0,
                    confidence=0.7,
                    requirements=[f"Accès à {spot}"],
                    notes=["Récolte de base", "Pas de coût de ressources"]
                )
                
                opportunities.append(recommendation)
        
        except Exception as e:
            self.logger.error(f"Erreur analyse récolte {profession_state.name}: {e}")
        
        return opportunities
    
    def _analyze_trading_opportunities(self,
                                     profession_state: ProfessionState,
                                     goal: OptimizationGoal,
                                     time_budget: int) -> List[ActivityRecommendation]:
        """Analyse les opportunités de trading."""
        opportunities = []
        
        # Simulation d'opportunités de trading
        try:
            trading_items = ['Pain', 'Potion de Soin', 'Épée de Boisaille']
            
            for item in trading_items:
                if item in profession_state.inventory and profession_state.inventory[item] > 0:
                    estimated_profit = profession_state.inventory[item] * 100  # 100 kamas par item
                    
                    recommendation = ActivityRecommendation(
                        activity_type=ActivityType.TRADING,
                        profession=profession_state.name,
                        specific_action=f"Vendre {item}",
                        location="Hôtel de Vente",
                        expected_xp_per_hour=0,
                        expected_kamas_per_hour=estimated_profit,
                        resource_cost={item: profession_state.inventory[item]},
                        time_estimate=15,  # 15 minutes pour aller vendre
                        difficulty_score=1.0,
                        market_opportunity=7.0,
                        confidence=0.9,
                        requirements=["Accès Hôtel de Vente"],
                        notes=[f"Stock disponible: {profession_state.inventory[item]}"]
                    )
                    
                    opportunities.append(recommendation)
        
        except Exception as e:
            self.logger.error(f"Erreur analyse trading {profession_state.name}: {e}")
        
        return opportunities
    
    def _calculate_composite_score(self, recommendation: ActivityRecommendation, goal: OptimizationGoal) -> float:
        """Calcule un score composite pour une recommandation."""
        try:
            weights = self.optimization_weights.get(goal, self.optimization_weights[OptimizationGoal.BALANCED])
            
            # Normaliser les métriques
            xp_score = min(recommendation.expected_xp_per_hour / 1000, 10)
            kamas_score = min(recommendation.expected_kamas_per_hour / 10000, 10)
            time_score = max(0, 10 - recommendation.difficulty_score)
            
            # Calculer le score composite
            composite_score = (
                xp_score * weights['xp'] +
                kamas_score * weights['kamas'] +
                time_score * weights['time'] +
                recommendation.market_opportunity * 0.1
            ) * recommendation.confidence
            
            return composite_score
        
        except Exception as e:
            self.logger.error(f"Erreur calcul score composite: {e}")
            return 0.0
    
    def _get_available_recipes(self, profession_state: ProfessionState) -> Dict:
        """Récupère les recettes disponibles pour un métier."""
        available = {}
        
        for recipe_name, recipe_data in self.recipe_database.items():
            if (recipe_data['profession'] == profession_state.name and 
                recipe_data['min_level'] <= profession_state.level):
                available[recipe_name] = recipe_data
        
        return available
    
    def _calculate_craft_xp_per_hour(self, recipe_data: Dict, profession_state: ProfessionState) -> float:
        """Calcule l'XP par heure pour un craft."""
        base_xp = recipe_data['base_xp']
        craft_time_minutes = recipe_data['craft_time']
        
        # Bonus de niveau (plus haut niveau = plus efficace)
        level_bonus = 1 + (profession_state.level / 200)
        
        crafts_per_hour = 60 / craft_time_minutes
        xp_per_hour = base_xp * crafts_per_hour * level_bonus
        
        return xp_per_hour
    
    def _calculate_craft_profit_per_hour(self, recipe_data: Dict, recipe_name: str) -> float:
        """Calcule le profit par heure pour un craft."""
        # Simulation de calcul de profit
        base_profit = 150  # Profit de base par craft
        craft_time_minutes = recipe_data['craft_time']
        
        crafts_per_hour = 60 / craft_time_minutes
        profit_per_hour = base_profit * crafts_per_hour
        
        return profit_per_hour
    
    def _calculate_craft_difficulty(self, recipe_data: Dict, profession_state: ProfessionState) -> float:
        """Calcule la difficulté d'un craft."""
        level_gap = recipe_data['min_level'] - profession_state.level
        
        if level_gap <= 0:
            return max(1.0, 5.0 + level_gap * 0.1)  # Plus facile si niveau dépassé
        else:
            return min(10.0, 5.0 + level_gap * 0.5)  # Plus difficile si niveau insuffisant
    
    def _calculate_market_opportunity(self, item_name: str) -> float:
        """Calcule le score d'opportunité marché."""
        # Simulation d'analyse marché
        market_data = self.market_cache.get(item_name)
        
        if not market_data:
            return 5.0  # Score neutre par défaut
        
        opportunity_score = 5.0
        
        # Ajuster selon la tendance des prix
        if market_data.price_trend == "rising":
            opportunity_score += 2.0
        elif market_data.price_trend == "falling":
            opportunity_score -= 1.0
        
        # Ajuster selon la demande
        opportunity_score += market_data.demand_score * 0.5
        
        return min(10.0, max(0.0, opportunity_score))
    
    def update_market_data(self, market_data_list: List[MarketData]):
        """Met à jour les données de marché."""
        try:
            for market_data in market_data_list:
                self.market_cache[market_data.item_name] = market_data
            
            self.logger.info(f"Données marché mises à jour: {len(market_data_list)} items")
        
        except Exception as e:
            self.logger.error(f"Erreur mise à jour marché: {e}")
    
    def get_profession_analysis(self, profession_state: ProfessionState) -> Dict:
        """Analyse complète d'un métier."""
        try:
            analysis = {
                'current_state': {
                    'level': profession_state.level,
                    'xp_progress': f"{profession_state.current_xp}/{profession_state.current_xp + profession_state.xp_to_next_level}",
                    'xp_percentage': (profession_state.current_xp / (profession_state.current_xp + profession_state.xp_to_next_level)) * 100
                },
                'available_activities': len(self._get_available_recipes(profession_state)),
                'inventory_value': sum(profession_state.inventory.values()) * 50,  # Valeur estimée
                'recommendations_count': 0,
                'next_milestone': self._get_next_milestone(profession_state),
                'efficiency_rating': self._calculate_efficiency_rating(profession_state)
            }
            
            return analysis
        
        except Exception as e:
            self.logger.error(f"Erreur analyse métier {profession_state.name}: {e}")
            return {}
    
    def _get_next_milestone(self, profession_state: ProfessionState) -> Dict:
        """Calcule le prochain jalon important."""
        next_levels = [50, 100, 150, 200]
        current_level = profession_state.level
        
        for level in next_levels:
            if level > current_level:
                return {
                    'level': level,
                    'xp_needed': (level - current_level) * 1000 + profession_state.xp_to_next_level,
                    'estimated_time_hours': ((level - current_level) * 1000 + profession_state.xp_to_next_level) / 500
                }
        
        return {'level': 'Max', 'xp_needed': 0, 'estimated_time_hours': 0}
    
    def _calculate_efficiency_rating(self, profession_state: ProfessionState) -> str:
        """Calcule une note d'efficacité."""
        score = 0
        
        # Niveau
        if profession_state.level >= 100:
            score += 3
        elif profession_state.level >= 50:
            score += 2
        else:
            score += 1
        
        # Inventaire
        if len(profession_state.inventory) > 10:
            score += 2
        elif len(profession_state.inventory) > 5:
            score += 1
        
        # Outils
        score += min(len(profession_state.current_tools), 2)
        
        if score >= 7:
            return "Excellent"
        elif score >= 5:
            return "Bon"
        elif score >= 3:
            return "Moyen"
        else:
            return "Faible"
    
    def export_analysis_report(self, profession_states: List[ProfessionState], 
                              recommendations: List[ActivityRecommendation],
                              filename: str = None) -> str:
        """Exporte un rapport d'analyse détaillé."""
        try:
            if not filename:
                filename = f"profession_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'professions': [],
                'recommendations': [],
                'summary': {
                    'total_professions': len(profession_states),
                    'total_recommendations': len(recommendations),
                    'avg_confidence': sum(r.confidence for r in recommendations) / len(recommendations) if recommendations else 0
                }
            }
            
            # Analyser chaque métier
            for profession_state in profession_states:
                profession_analysis = self.get_profession_analysis(profession_state)
                profession_analysis['name'] = profession_state.name
                report['professions'].append(profession_analysis)
            
            # Ajouter recommandations
            for rec in recommendations:
                rec_dict = {
                    'activity_type': rec.activity_type.value,
                    'profession': rec.profession,
                    'action': rec.specific_action,
                    'location': rec.location,
                    'xp_per_hour': rec.expected_xp_per_hour,
                    'kamas_per_hour': rec.expected_kamas_per_hour,
                    'time_estimate': rec.time_estimate,
                    'confidence': rec.confidence,
                    'market_opportunity': rec.market_opportunity
                }
                report['recommendations'].append(rec_dict)
            
            # Sauvegarder
            report_path = f"data/reports/{filename}"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Rapport exporté: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"Erreur export rapport: {e}")
            return ""