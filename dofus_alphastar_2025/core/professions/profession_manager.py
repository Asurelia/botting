"""
Gestionnaire de tous les métiers - Coordination et optimisation multi-métiers.
Gère la progression, l'économie et les synergies entre tous les métiers.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import math
import time
from pathlib import Path

from .base import BaseProfession, ResourceData, ProfessionStats
from .farmer import Farmer
from .lumberjack import Lumberjack
from .miner import Miner
from .alchemist import Alchemist

class OptimizationStrategy(Enum):
    """Stratégies d'optimisation globale"""
    BALANCED = "balanced"  # Équilibre entre tous les métiers
    XP_FOCUSED = "xp_focused"  # Maximise l'XP globale
    PROFIT_FOCUSED = "profit_focused"  # Maximise les profits
    LEVELING = "leveling"  # Progression rapide du niveau le plus bas
    SYNERGY = "synergy"  # Exploite les synergies entre métiers

@dataclass
class ProfessionSynergy:
    """Synergie entre deux métiers"""
    profession1: str
    profession2: str
    resource1: str  # Ressource du métier 1
    resource2: str  # Ressource du métier 2 qui l'utilise
    efficiency_bonus: float  # Bonus d'efficacité
    description: str

@dataclass
class GlobalSession:
    """Session de jeu multi-métiers"""
    duration_hours: float
    strategy: OptimizationStrategy
    profession_allocation: Dict[str, float]  # % de temps par métier
    expected_results: Dict[str, Any]
    synergies_used: List[ProfessionSynergy]

class ProfessionManager:
    """Gestionnaire principal de tous les métiers"""
    
    def __init__(self, config_path: str = "G:/Botting/config/professions.json"):
        self.config_path = config_path
        self.professions: Dict[str, BaseProfession] = {}
        self.synergies: List[ProfessionSynergy] = []
        self.global_stats = {
            'total_play_time': 0.0,
            'total_xp_gained': 0,
            'total_kamas_earned': 0,
            'sessions_completed': 0,
            'best_hourly_xp': 0,
            'best_hourly_kamas': 0
        }
        
        self._initialize_professions()
        self._define_synergies()
        self._load_configuration()
        
    def _initialize_professions(self) -> None:
        """Initialise tous les métiers"""
        self.professions = {
            'farmer': Farmer(),
            'lumberjack': Lumberjack(),
            'miner': Miner(),
            'alchemist': Alchemist()
        }
        
    def _define_synergies(self) -> None:
        """Définit les synergies entre métiers"""
        self.synergies = [
            # Fermier -> Alchimiste
            ProfessionSynergy(
                'farmer', 'alchemist', 'ble', 'ble',
                1.2, "Le blé fermé réduit le coût des potions de base"
            ),
            ProfessionSynergy(
                'farmer', 'alchemist', 'menthe', 'menthe',
                1.3, "La menthe fraîche améliore l'efficacité des potions de soin"
            ),
            ProfessionSynergy(
                'farmer', 'alchemist', 'ginseng', 'ginseng',
                1.5, "Le ginseng auto-produit réduit drastiquement les coûts"
            ),
            
            # Mineur -> Alchimiste  
            ProfessionSynergy(
                'miner', 'alchemist', 'fer', 'fer',
                1.2, "Le fer miné réduit le coût des potions de force"
            ),
            ProfessionSynergy(
                'miner', 'alchemist', 'or', 'or',
                1.4, "L'or auto-extrait améliore grandement la rentabilité"
            ),
            
            # Bûcheron -> Fermier (outils)
            ProfessionSynergy(
                'lumberjack', 'farmer', 'chene', 'outils_bois',
                1.1, "Le bois de qualité améliore l'efficacité des outils agricoles"
            ),
            
            # Multi-synergies pour l'alchimie
            ProfessionSynergy(
                'all', 'alchemist', 'multiple', 'ingredients',
                2.0, "Tous les ingrédients auto-produits maximisent la rentabilité alchimique"
            )
        ]
    
    def get_profession(self, profession_id: str) -> Optional[BaseProfession]:
        """Récupère un métier spécifique"""
        return self.professions.get(profession_id)
    
    def get_all_professions(self) -> Dict[str, BaseProfession]:
        """Retourne tous les métiers"""
        return self.professions.copy()
    
    def get_profession_levels(self) -> Dict[str, int]:
        """Retourne les niveaux de tous les métiers"""
        return {name: prof.stats.level for name, prof in self.professions.items()}
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Statistiques globales de tous les métiers"""
        total_levels = sum(prof.stats.level for prof in self.professions.values())
        avg_level = total_levels / len(self.professions)
        
        total_resources = sum(len(prof.resources) for prof in self.professions.values())
        available_resources = sum(len(prof.get_available_resources()) for prof in self.professions.values())
        
        profession_stats = {}
        for name, prof in self.professions.items():
            profession_stats[name] = {
                'level': prof.stats.level,
                'experience': prof.stats.experience,
                'total_harvested': prof.stats.total_harvested,
                'kamas_earned': prof.stats.kamas_earned,
                'time_spent_hours': round(prof.stats.time_spent / 3600, 2),
                'efficiency_rating': self._calculate_efficiency_rating(prof)
            }
        
        return {
            'global_stats': self.global_stats,
            'profession_summary': {
                'total_levels': total_levels,
                'average_level': round(avg_level, 1),
                'highest_level': max(prof.stats.level for prof in self.professions.values()),
                'lowest_level': min(prof.stats.level for prof in self.professions.values()),
                'total_resources': total_resources,
                'available_resources': available_resources
            },
            'profession_details': profession_stats,
            'synergies_available': len(self._get_available_synergies()),
            'recommendations': self._get_global_recommendations()
        }
    
    def _calculate_efficiency_rating(self, profession: BaseProfession) -> str:
        """Calcule une note d'efficacité pour un métier"""
        if profession.stats.time_spent == 0:
            return "Non évalué"
        
        xp_per_hour = profession.stats.experience / (profession.stats.time_spent / 3600)
        kamas_per_hour = profession.stats.kamas_earned / (profession.stats.time_spent / 3600)
        
        # Barèmes approximatifs basés sur le niveau
        expected_xp = profession.stats.level * 50
        expected_kamas = profession.stats.level * 20
        
        xp_ratio = xp_per_hour / max(expected_xp, 1)
        kamas_ratio = kamas_per_hour / max(expected_kamas, 1)
        
        overall_ratio = (xp_ratio + kamas_ratio) / 2
        
        if overall_ratio >= 1.5:
            return "Excellent"
        elif overall_ratio >= 1.2:
            return "Très bon"
        elif overall_ratio >= 1.0:
            return "Bon" 
        elif overall_ratio >= 0.8:
            return "Moyen"
        else:
            return "À améliorer"
    
    def _get_global_recommendations(self) -> List[str]:
        """Génère des recommandations globales"""
        recommendations = []
        levels = self.get_profession_levels()
        
        # Équilibrage des niveaux
        min_level = min(levels.values())
        max_level = max(levels.values())
        if max_level - min_level > 20:
            lowest_prof = min(levels, key=levels.get)
            recommendations.append(f"Rattraper le niveau du {self.professions[lowest_prof].name} (niveau {min_level})")
        
        # Synergies disponibles
        available_synergies = self._get_available_synergies()
        if available_synergies:
            recommendations.append(f"Exploiter {len(available_synergies)} synergies entre métiers disponibles")
        
        # Métiers très rentables
        for name, prof in self.professions.items():
            if prof.stats.time_spent > 0:
                kamas_per_hour = prof.stats.kamas_earned / (prof.stats.time_spent / 3600)
                if kamas_per_hour > 1000:  # Seuil de rentabilité élevée
                    recommendations.append(f"Continuer à exploiter la rentabilité du {prof.name}")
        
        # Outils/ateliers à améliorer
        if hasattr(self.professions['lumberjack'], 'tools_efficiency'):
            lj = self.professions['lumberjack']
            tool_recs = lj.get_tool_recommendations()
            if any(rec['recommended'] for rec in tool_recs.values()):
                recommendations.append("Améliorer les outils de bûcheron pour plus d'efficacité")
        
        if hasattr(self.professions['alchemist'], 'workshop_level'):
            alch = self.professions['alchemist']
            if alch.workshop_level < 5:
                workshop_info = alch.get_workshop_upgrades()
                if workshop_info.get('recommendation') == 'Recommandé':
                    recommendations.append("Améliorer l'atelier d'alchimie (ROI favorable)")
        
        return recommendations[:5]  # Top 5 recommandations
    
    def optimize_global_session(self, duration_hours: float = 4.0, 
                               strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> GlobalSession:
        """Optimise une session multi-métiers"""
        
        if strategy == OptimizationStrategy.BALANCED:
            return self._optimize_balanced_session(duration_hours)
        elif strategy == OptimizationStrategy.XP_FOCUSED:
            return self._optimize_xp_session(duration_hours)
        elif strategy == OptimizationStrategy.PROFIT_FOCUSED:
            return self._optimize_profit_session(duration_hours)
        elif strategy == OptimizationStrategy.LEVELING:
            return self._optimize_leveling_session(duration_hours)
        elif strategy == OptimizationStrategy.SYNERGY:
            return self._optimize_synergy_session(duration_hours)
        else:
            return self._optimize_balanced_session(duration_hours)
    
    def _optimize_balanced_session(self, duration_hours: float) -> GlobalSession:
        """Session équilibrée entre tous les métiers"""
        # Répartition égale du temps
        time_per_profession = duration_hours / len(self.professions)
        allocation = {name: time_per_profession / duration_hours for name in self.professions.keys()}
        
        expected_results = {}
        total_xp = 0
        total_kamas = 0
        
        for name, prof in self.professions.items():
            prof_time = time_per_profession
            best_resources = prof.get_optimal_route()[:3]  # Top 3 ressources
            
            if best_resources:
                # Calculer les gains moyens
                avg_xp_per_hour = sum(prof.calculate_xp_per_hour(r.id) for r in best_resources) / len(best_resources)
                avg_kamas_per_hour = sum(prof.calculate_kamas_per_hour(r.id) for r in best_resources) / len(best_resources)
                
                prof_xp = avg_xp_per_hour * prof_time
                prof_kamas = avg_kamas_per_hour * prof_time
                
                total_xp += prof_xp
                total_kamas += prof_kamas
                
                expected_results[name] = {
                    'time_hours': prof_time,
                    'expected_xp': round(prof_xp),
                    'expected_kamas': round(prof_kamas),
                    'top_resources': [r.name for r in best_resources]
                }
        
        return GlobalSession(
            duration_hours=duration_hours,
            strategy=OptimizationStrategy.BALANCED,
            profession_allocation=allocation,
            expected_results={
                'by_profession': expected_results,
                'totals': {
                    'total_xp': round(total_xp),
                    'total_kamas': round(total_kamas),
                    'avg_xp_per_hour': round(total_xp / duration_hours),
                    'avg_kamas_per_hour': round(total_kamas / duration_hours)
                }
            },
            synergies_used=[]
        )
    
    def _optimize_xp_session(self, duration_hours: float) -> GlobalSession:
        """Session optimisée pour l'XP"""
        # Trouver les meilleures ressources pour XP de chaque métier
        profession_xp_rates = {}
        
        for name, prof in self.professions.items():
            best_resources = prof.get_optimal_route()[:5]
            if best_resources:
                max_xp_rate = max(prof.calculate_xp_per_hour(r.id) for r in best_resources)
                profession_xp_rates[name] = max_xp_rate
        
        # Allouer plus de temps aux métiers avec meilleur taux XP
        total_xp_potential = sum(profession_xp_rates.values())
        allocation = {name: (rate / total_xp_potential) for name, rate in profession_xp_rates.items()}
        
        expected_results = {}
        total_xp = 0
        total_kamas = 0
        
        for name, prof in self.professions.items():
            prof_time = duration_hours * allocation[name]
            max_xp_rate = profession_xp_rates[name]
            
            # Estimer aussi les kamas
            best_resource = max(prof.get_optimal_route()[:5], key=lambda r: prof.calculate_xp_per_hour(r.id))
            kamas_rate = prof.calculate_kamas_per_hour(best_resource.id)
            
            prof_xp = max_xp_rate * prof_time
            prof_kamas = kamas_rate * prof_time
            
            total_xp += prof_xp
            total_kamas += prof_kamas
            
            expected_results[name] = {
                'time_hours': round(prof_time, 2),
                'expected_xp': round(prof_xp),
                'expected_kamas': round(prof_kamas),
                'focus_resource': best_resource.name
            }
        
        return GlobalSession(
            duration_hours=duration_hours,
            strategy=OptimizationStrategy.XP_FOCUSED,
            profession_allocation=allocation,
            expected_results={
                'by_profession': expected_results,
                'totals': {
                    'total_xp': round(total_xp),
                    'total_kamas': round(total_kamas),
                    'xp_optimization_bonus': 1.3  # 30% bonus estimé
                }
            },
            synergies_used=[]
        )
    
    def _optimize_profit_session(self, duration_hours: float) -> GlobalSession:
        """Session optimisée pour les profits"""
        profession_profit_rates = {}
        
        for name, prof in self.professions.items():
            best_resources = prof.get_optimal_route()[:5]
            if best_resources:
                max_profit_rate = max(prof.calculate_kamas_per_hour(r.id) for r in best_resources)
                profession_profit_rates[name] = max_profit_rate
        
        # Allouer plus de temps aux métiers les plus rentables
        total_profit_potential = sum(profession_profit_rates.values())
        allocation = {name: (rate / total_profit_potential) for name, rate in profession_profit_rates.items()}
        
        expected_results = {}
        total_xp = 0
        total_kamas = 0
        
        for name, prof in self.professions.items():
            prof_time = duration_hours * allocation[name]
            max_profit_rate = profession_profit_rates[name]
            
            best_resource = max(prof.get_optimal_route()[:5], key=lambda r: prof.calculate_kamas_per_hour(r.id))
            xp_rate = prof.calculate_xp_per_hour(best_resource.id)
            
            prof_kamas = max_profit_rate * prof_time
            prof_xp = xp_rate * prof_time
            
            total_xp += prof_xp
            total_kamas += prof_kamas
            
            expected_results[name] = {
                'time_hours': round(prof_time, 2),
                'expected_xp': round(prof_xp),
                'expected_kamas': round(prof_kamas),
                'focus_resource': best_resource.name,
                'profit_rate': round(max_profit_rate)
            }
        
        return GlobalSession(
            duration_hours=duration_hours,
            strategy=OptimizationStrategy.PROFIT_FOCUSED,
            profession_allocation=allocation,
            expected_results={
                'by_profession': expected_results,
                'totals': {
                    'total_xp': round(total_xp),
                    'total_kamas': round(total_kamas),
                    'profit_optimization_bonus': 1.5  # 50% bonus estimé
                }
            },
            synergies_used=[]
        )
    
    def _optimize_leveling_session(self, duration_hours: float) -> GlobalSession:
        """Session pour rattraper le métier le plus bas"""
        levels = self.get_profession_levels()
        lowest_profession = min(levels, key=levels.get)
        lowest_level = levels[lowest_profession]
        
        # Allouer 70% du temps au métier le plus bas, 30% aux autres
        allocation = {}
        for name in self.professions.keys():
            if name == lowest_profession:
                allocation[name] = 0.7
            else:
                allocation[name] = 0.3 / (len(self.professions) - 1)
        
        expected_results = {}
        total_xp = 0
        total_kamas = 0
        
        for name, prof in self.professions.items():
            prof_time = duration_hours * allocation[name]
            
            # Pour le métier le plus bas, optimiser pour XP
            if name == lowest_profession:
                best_resources = prof.get_optimal_route()
                if best_resources:
                    best_resource = max(best_resources[:3], key=lambda r: prof.calculate_xp_per_hour(r.id))
                    xp_rate = prof.calculate_xp_per_hour(best_resource.id)
                    kamas_rate = prof.calculate_kamas_per_hour(best_resource.id)
                else:
                    xp_rate = kamas_rate = 0
                    best_resource = None
            else:
                # Pour les autres, maintenir un équilibre
                best_resources = prof.get_optimal_route()
                if best_resources:
                    best_resource = best_resources[0]  # Premier de la liste optimisée
                    xp_rate = prof.calculate_xp_per_hour(best_resource.id)
                    kamas_rate = prof.calculate_kamas_per_hour(best_resource.id)
                else:
                    xp_rate = kamas_rate = 0
                    best_resource = None
            
            prof_xp = xp_rate * prof_time
            prof_kamas = kamas_rate * prof_time
            
            total_xp += prof_xp
            total_kamas += prof_kamas
            
            expected_results[name] = {
                'time_hours': round(prof_time, 2),
                'expected_xp': round(prof_xp),
                'expected_kamas': round(prof_kamas),
                'focus_resource': best_resource.name if best_resource else 'Aucune',
                'is_priority': name == lowest_profession
            }
        
        return GlobalSession(
            duration_hours=duration_hours,
            strategy=OptimizationStrategy.LEVELING,
            profession_allocation=allocation,
            expected_results={
                'by_profession': expected_results,
                'totals': {
                    'total_xp': round(total_xp),
                    'total_kamas': round(total_kamas),
                },
                'priority_profession': lowest_profession,
                'current_level': lowest_level,
                'estimated_level_gain': round(expected_results[lowest_profession]['expected_xp'] / 1000)  # Estimation
            },
            synergies_used=[]
        )
    
    def _optimize_synergy_session(self, duration_hours: float) -> GlobalSession:
        """Session optimisée pour exploiter les synergies"""
        available_synergies = self._get_available_synergies()
        
        if not available_synergies:
            # Pas de synergies disponibles, faire une session équilibrée
            return self._optimize_balanced_session(duration_hours)
        
        # Prioriser les métiers impliqués dans les synergies
        synergy_professions = set()
        for synergy in available_synergies:
            if synergy.profession1 != 'all':
                synergy_professions.add(synergy.profession1)
            synergy_professions.add(synergy.profession2)
        
        # Allocation basée sur les synergies
        total_bonus = sum(s.efficiency_bonus for s in available_synergies)
        allocation = {}
        
        for name in self.professions.keys():
            if name in synergy_professions:
                # Plus de temps pour les métiers avec synergies
                relevant_synergies = [s for s in available_synergies if name in [s.profession1, s.profession2]]
                synergy_weight = sum(s.efficiency_bonus for s in relevant_synergies)
                allocation[name] = synergy_weight / total_bonus * 0.8  # 80% pour les synergies
            else:
                allocation[name] = 0.2 / max(len(self.professions) - len(synergy_professions), 1)
        
        expected_results = {}
        total_xp = 0
        total_kamas = 0
        
        for name, prof in self.professions.items():
            prof_time = duration_hours * allocation[name]
            
            # Calculer avec bonus de synergie
            synergy_bonus = 1.0
            for synergy in available_synergies:
                if name in [synergy.profession1, synergy.profession2]:
                    synergy_bonus *= synergy.efficiency_bonus
            
            best_resources = prof.get_optimal_route()
            if best_resources:
                best_resource = best_resources[0]
                base_xp_rate = prof.calculate_xp_per_hour(best_resource.id)
                base_kamas_rate = prof.calculate_kamas_per_hour(best_resource.id)
                
                # Appliquer les bonus de synergie
                effective_xp_rate = base_xp_rate * synergy_bonus
                effective_kamas_rate = base_kamas_rate * synergy_bonus
            else:
                effective_xp_rate = effective_kamas_rate = 0
                best_resource = None
            
            prof_xp = effective_xp_rate * prof_time
            prof_kamas = effective_kamas_rate * prof_time
            
            total_xp += prof_xp
            total_kamas += prof_kamas
            
            expected_results[name] = {
                'time_hours': round(prof_time, 2),
                'expected_xp': round(prof_xp),
                'expected_kamas': round(prof_kamas),
                'focus_resource': best_resource.name if best_resource else 'Aucune',
                'synergy_bonus': round(synergy_bonus, 2)
            }
        
        return GlobalSession(
            duration_hours=duration_hours,
            strategy=OptimizationStrategy.SYNERGY,
            profession_allocation=allocation,
            expected_results={
                'by_profession': expected_results,
                'totals': {
                    'total_xp': round(total_xp),
                    'total_kamas': round(total_kamas),
                    'synergy_bonus_average': round(sum(r['synergy_bonus'] for r in expected_results.values()) / len(expected_results), 2)
                }
            },
            synergies_used=available_synergies
        )
    
    def _get_available_synergies(self) -> List[ProfessionSynergy]:
        """Retourne les synergies actuellement disponibles"""
        available = []
        levels = self.get_profession_levels()
        
        for synergy in self.synergies:
            # Vérifier si les métiers sont de niveau suffisant pour la synergie
            if synergy.profession1 == 'all':
                # Synergie multi-métiers - tous doivent être niveau 20+
                if all(level >= 20 for level in levels.values()):
                    available.append(synergy)
            else:
                prof1_level = levels.get(synergy.profession1, 0)
                prof2_level = levels.get(synergy.profession2, 0)
                
                # Vérifier que les métiers peuvent produire/utiliser les ressources
                if prof1_level >= 10 and prof2_level >= 10:  # Seuil minimum
                    available.append(synergy)
        
        return available
    
    def execute_session(self, session: GlobalSession) -> Dict[str, Any]:
        """Exécute virtuellement une session optimisée"""
        start_time = time.time()
        execution_results = {}
        
        for profession_name, time_allocation in session.profession_allocation.items():
            profession = self.professions[profession_name]
            session_time = session.duration_hours * time_allocation
            
            # Simuler l'exécution
            expected = session.expected_results['by_profession'].get(profession_name, {})
            
            # Ajouter une variance réaliste (-10% à +20%)
            variance = random.uniform(0.9, 1.2) if hasattr(math, 'random') else 1.0
            actual_xp = expected.get('expected_xp', 0) * variance
            actual_kamas = expected.get('expected_kamas', 0) * variance
            
            # Mettre à jour les stats du métier
            profession.stats.experience += int(actual_xp)
            profession.stats.kamas_earned += int(actual_kamas)
            profession.stats.time_spent += session_time * 3600  # Convertir en secondes
            profession.stats.level = profession._calculate_level_from_xp(profession.stats.experience)
            
            execution_results[profession_name] = {
                'time_spent_hours': session_time,
                'xp_gained': round(actual_xp),
                'kamas_gained': round(actual_kamas),
                'new_level': profession.stats.level,
                'variance_factor': round(variance, 2)
            }
        
        # Mettre à jour les stats globales
        total_xp_gained = sum(r['xp_gained'] for r in execution_results.values())
        total_kamas_gained = sum(r['kamas_gained'] for r in execution_results.values())
        
        self.global_stats['total_play_time'] += session.duration_hours
        self.global_stats['total_xp_gained'] += total_xp_gained
        self.global_stats['total_kamas_earned'] += total_kamas_gained
        self.global_stats['sessions_completed'] += 1
        
        # Records
        hourly_xp = total_xp_gained / session.duration_hours
        hourly_kamas = total_kamas_gained / session.duration_hours
        
        if hourly_xp > self.global_stats['best_hourly_xp']:
            self.global_stats['best_hourly_xp'] = round(hourly_xp)
        if hourly_kamas > self.global_stats['best_hourly_kamas']:
            self.global_stats['best_hourly_kamas'] = round(hourly_kamas)
        
        execution_time = time.time() - start_time
        
        return {
            'session_summary': {
                'strategy': session.strategy.value,
                'duration_hours': session.duration_hours,
                'execution_time_seconds': round(execution_time, 3)
            },
            'results_by_profession': execution_results,
            'totals': {
                'total_xp_gained': total_xp_gained,
                'total_kamas_gained': total_kamas_gained,
                'average_xp_per_hour': round(hourly_xp),
                'average_kamas_per_hour': round(hourly_kamas)
            },
            'performance': {
                'vs_expected_xp': round((total_xp_gained / session.expected_results['totals'].get('total_xp', 1)) * 100, 1),
                'vs_expected_kamas': round((total_kamas_gained / session.expected_results['totals'].get('total_kamas', 1)) * 100, 1),
                'new_records': {
                    'xp': hourly_xp == self.global_stats['best_hourly_xp'],
                    'kamas': hourly_kamas == self.global_stats['best_hourly_kamas']
                }
            },
            'synergies_activated': len(session.synergies_used)
        }
    
    def save_configuration(self) -> None:
        """Sauvegarde la configuration de tous les métiers"""
        config_data = {
            'version': '1.0.0',
            'last_saved': time.strftime('%Y-%m-%d %H:%M:%S'),
            'global_stats': self.global_stats,
            'professions': {}
        }
        
        for name, profession in self.professions.items():
            config_data['professions'][name] = {
                'level': profession.stats.level,
                'experience': profession.stats.experience,
                'total_harvested': profession.stats.total_harvested,
                'total_crafted': profession.stats.total_crafted,
                'kamas_earned': profession.stats.kamas_earned,
                'time_spent': profession.stats.time_spent,
                'inventory_capacity': profession.inventory_capacity,
                'auto_bank': profession.auto_bank,
                'auto_craft': profession.auto_craft,
                'preferred_resources': profession.preferred_resources,
                'blacklisted_resources': profession.blacklisted_resources
            }
            
            # Données spécifiques par métier
            if hasattr(profession, 'current_tool'):
                config_data['professions'][name]['current_tool'] = profession.current_tool
            if hasattr(profession, 'workshop_level'):
                config_data['professions'][name]['workshop_level'] = profession.workshop_level
        
        # Créer le répertoire de config s'il n'existe pas
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _load_configuration(self) -> None:
        """Charge la configuration de tous les métiers"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Charger les stats globales
            self.global_stats.update(config_data.get('global_stats', {}))
            
            # Charger les stats de chaque métier
            professions_config = config_data.get('professions', {})
            for name, profession in self.professions.items():
                prof_config = professions_config.get(name, {})
                
                if prof_config:
                    profession.stats.level = prof_config.get('level', 1)
                    profession.stats.experience = prof_config.get('experience', 0)
                    profession.stats.total_harvested = prof_config.get('total_harvested', 0)
                    profession.stats.total_crafted = prof_config.get('total_crafted', 0)
                    profession.stats.kamas_earned = prof_config.get('kamas_earned', 0)
                    profession.stats.time_spent = prof_config.get('time_spent', 0.0)
                    
                    profession.inventory_capacity = prof_config.get('inventory_capacity', 1000)
                    profession.auto_bank = prof_config.get('auto_bank', True)
                    profession.auto_craft = prof_config.get('auto_craft', False)
                    profession.preferred_resources = prof_config.get('preferred_resources', [])
                    profession.blacklisted_resources = prof_config.get('blacklisted_resources', [])
                    
                    # Données spécifiques
                    if hasattr(profession, 'current_tool') and 'current_tool' in prof_config:
                        profession.current_tool = prof_config['current_tool']
                    if hasattr(profession, 'workshop_level') and 'workshop_level' in prof_config:
                        profession.workshop_level = prof_config['workshop_level']
                        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Impossible de charger la configuration: {e}")
            print("Utilisation des paramètres par défaut")
    
    def compare_strategies(self, duration_hours: float = 4.0) -> Dict[str, Any]:
        """Compare toutes les stratégies d'optimisation"""
        strategies = [
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.XP_FOCUSED,
            OptimizationStrategy.PROFIT_FOCUSED,
            OptimizationStrategy.LEVELING,
            OptimizationStrategy.SYNERGY
        ]
        
        comparison = {}
        
        for strategy in strategies:
            session = self.optimize_global_session(duration_hours, strategy)
            totals = session.expected_results.get('totals', {})
            
            comparison[strategy.value] = {
                'total_xp': totals.get('total_xp', 0),
                'total_kamas': totals.get('total_kamas', 0),
                'avg_xp_per_hour': totals.get('avg_xp_per_hour', totals.get('total_xp', 0) / duration_hours),
                'avg_kamas_per_hour': totals.get('avg_kamas_per_hour', totals.get('total_kamas', 0) / duration_hours),
                'synergies_count': len(session.synergies_used),
                'allocation': session.profession_allocation
            }
        
        # Trouver les meilleures stratégies
        best_xp = max(comparison.values(), key=lambda x: x['total_xp'])['total_xp']
        best_kamas = max(comparison.values(), key=lambda x: x['total_kamas'])['total_kamas']
        
        best_xp_strategy = next(k for k, v in comparison.items() if v['total_xp'] == best_xp)
        best_kamas_strategy = next(k for k, v in comparison.items() if v['total_kamas'] == best_kamas)
        
        return {
            'comparison_duration_hours': duration_hours,
            'strategies_analyzed': len(strategies),
            'detailed_comparison': comparison,
            'recommendations': {
                'best_for_xp': best_xp_strategy,
                'best_for_kamas': best_kamas_strategy,
                'xp_difference_percent': round(((best_xp / min(v['total_xp'] for v in comparison.values() if v['total_xp'] > 0)) - 1) * 100, 1),
                'kamas_difference_percent': round(((best_kamas / min(v['total_kamas'] for v in comparison.values() if v['total_kamas'] > 0)) - 1) * 100, 1)
            }
        }
    
    def __str__(self) -> str:
        levels = self.get_profession_levels()
        total_level = sum(levels.values())
        avg_level = total_level / len(levels)
        return f"Gestionnaire de Métiers - {len(self.professions)} métiers - Niveau moyen: {avg_level:.1f}"