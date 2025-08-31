"""
Sélecteur de stratégies adaptatif pour le système de botting.

Ce module implémente la sélection automatique de stratégies selon la situation
et les objectifs, avec adaptation dynamique aux conditions changeantes.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import math

from .decision_engine import DecisionContext, Priority, ActionType


class StrategyType(Enum):
    """Types de stratégies disponibles."""
    AGGRESSIVE = "aggressive"    # Maximum de risque pour maximum de gain
    DEFENSIVE = "defensive"      # Priorité à la sécurité et survie
    BALANCED = "balanced"        # Équilibre entre risque et sécurité
    EFFICIENT = "efficient"      # Optimisation du temps et des ressources
    STEALTH = "stealth"         # Discrétion et évitement
    SOCIAL = "social"           # Coopération avec autres joueurs


class Situation(Enum):
    """Situations identifiables automatiquement."""
    PEACEFUL_FARMING = "peaceful_farming"      # Farm tranquille sans danger
    DANGEROUS_AREA = "dangerous_area"          # Zone avec risques élevés
    CROWDED_AREA = "crowded_area"             # Zone avec beaucoup de joueurs
    DUNGEON_EXPLORATION = "dungeon_exploration" # Exploration de donjon
    PVP_ZONE = "pvp_zone"                     # Zone de combat PvP
    RESOURCE_COMPETITION = "resource_competition" # Compétition pour ressources
    BOSS_FIGHT = "boss_fight"                 # Combat de boss
    LOW_RESOURCES = "low_resources"           # Ressources (vie, mana) faibles
    INVENTORY_FULL = "inventory_full"         # Inventaire plein
    MISSION_CRITICAL = "mission_critical"     # Mission importante en cours


@dataclass
class StrategyWeights:
    """Poids pour une stratégie donnée."""
    survival: float = 1.0
    efficiency: float = 1.0
    risk_tolerance: float = 0.5
    speed: float = 1.0
    stealth: float = 0.0
    cooperation: float = 0.0
    resource_conservation: float = 1.0
    objective_focus: float = 1.0


@dataclass
class StrategyConfig:
    """Configuration complète d'une stratégie."""
    name: str
    description: str
    weights: StrategyWeights
    preferred_actions: List[ActionType]
    avoided_actions: List[ActionType]
    activation_conditions: Dict[str, Any]
    duration_limits: Dict[str, float]  # Limites de durée par type d'action
    
    def matches_situation(self, context: DecisionContext, situation: Situation) -> float:
        """Calcule le score de correspondance avec la situation."""
        base_score = 0.5
        
        # Vérifier les conditions d'activation
        for condition, threshold in self.activation_conditions.items():
            if condition == "min_health" and context.health_percent >= threshold:
                base_score += 0.2
            elif condition == "max_health" and context.health_percent <= threshold:
                base_score += 0.2
            elif condition == "in_combat" and context.in_combat == threshold:
                base_score += 0.3
            elif condition == "enemy_count_min" and context.enemies_count >= threshold:
                base_score += 0.2
            elif condition == "safe_zone" and context.safe_zone == threshold:
                base_score += 0.2
        
        return min(1.0, max(0.0, base_score))


class StrategySelector:
    """
    Sélecteur de stratégies adaptatif qui analyse la situation
    et sélectionne la stratégie optimale.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Stratégies prédéfinies
        self.strategies = self._initialize_default_strategies()
        
        # Historique des stratégies utilisées
        self.strategy_history: List[Dict[str, Any]] = []
        
        # Métriques de performance par stratégie
        self.strategy_metrics = {
            strategy_type.value: {
                'usage_count': 0,
                'success_rate': 0.5,
                'average_reward': 0.0,
                'average_duration': 0.0,
                'last_used': None
            }
            for strategy_type in StrategyType
        }
        
        # Détecteurs de situations
        self.situation_detectors = {
            Situation.PEACEFUL_FARMING: self._detect_peaceful_farming,
            Situation.DANGEROUS_AREA: self._detect_dangerous_area,
            Situation.CROWDED_AREA: self._detect_crowded_area,
            Situation.DUNGEON_EXPLORATION: self._detect_dungeon_exploration,
            Situation.PVP_ZONE: self._detect_pvp_zone,
            Situation.RESOURCE_COMPETITION: self._detect_resource_competition,
            Situation.BOSS_FIGHT: self._detect_boss_fight,
            Situation.LOW_RESOURCES: self._detect_low_resources,
            Situation.INVENTORY_FULL: self._detect_inventory_full,
            Situation.MISSION_CRITICAL: self._detect_mission_critical
        }
        
        # Stratégie actuellement active
        self.current_strategy: Optional[StrategyType] = None
        self.strategy_start_time: Optional[datetime] = None
        
        # Adaptations apprises
        self.learned_adaptations = {}
        
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("Sélecteur de stratégies initialisé")
    
    def select_strategy(
        self, 
        context: DecisionContext,
        force_reevaluation: bool = False
    ) -> Tuple[StrategyType, StrategyConfig]:
        """
        Sélectionne la stratégie optimale selon le contexte.
        
        Args:
            context: Contexte de décision actuel
            force_reevaluation: Force la réévaluation même si une stratégie est active
            
        Returns:
            Tuple (type de stratégie, configuration de la stratégie)
        """
        # Détecter la situation actuelle
        current_situation = self._detect_current_situation(context)
        
        # Vérifier si on doit changer de stratégie
        if not force_reevaluation and self._should_keep_current_strategy(context, current_situation):
            strategy_config = self.strategies[self.current_strategy]
            return self.current_strategy, strategy_config
        
        # Évaluer toutes les stratégies
        strategy_scores = {}
        for strategy_type, strategy_config in self.strategies.items():
            score = self._evaluate_strategy(strategy_type, strategy_config, context, current_situation)
            strategy_scores[strategy_type] = score
        
        # Sélectionner la meilleure stratégie
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        # Enregistrer le changement de stratégie
        if best_strategy != self.current_strategy:
            self._record_strategy_change(best_strategy, current_situation, context)
        
        self.current_strategy = best_strategy
        self.strategy_start_time = datetime.now()
        
        self.logger.info(f"Stratégie sélectionnée: {best_strategy.value} "
                        f"(situation: {current_situation.value})")
        
        return best_strategy, self.strategies[best_strategy]
    
    def get_strategy_recommendations(
        self, 
        context: DecisionContext,
        top_n: int = 3
    ) -> List[Tuple[StrategyType, float, str]]:
        """
        Génère des recommandations de stratégies avec explications.
        
        Args:
            context: Contexte de décision actuel
            top_n: Nombre de recommandations à retourner
            
        Returns:
            Liste de tuples (stratégie, score, explication)
        """
        current_situation = self._detect_current_situation(context)
        strategy_scores = {}
        
        for strategy_type, strategy_config in self.strategies.items():
            score = self._evaluate_strategy(strategy_type, strategy_config, context, current_situation)
            strategy_scores[strategy_type] = score
        
        # Trier par score décroissant
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i, (strategy_type, score) in enumerate(sorted_strategies[:top_n]):
            explanation = self._generate_strategy_explanation(
                strategy_type, score, context, current_situation, i + 1
            )
            recommendations.append((strategy_type, score, explanation))
        
        return recommendations
    
    def update_strategy_outcome(
        self, 
        strategy_type: StrategyType, 
        success: bool,
        reward: float, 
        duration: float
    ):
        """
        Met à jour les métriques de performance d'une stratégie.
        
        Args:
            strategy_type: Type de stratégie utilisée
            success: Si la stratégie a réussi
            reward: Récompense obtenue
            duration: Durée d'utilisation
        """
        metrics = self.strategy_metrics[strategy_type.value]
        
        # Mettre à jour le nombre d'utilisations
        metrics['usage_count'] += 1
        
        # Mettre à jour le taux de succès (moyenne mobile)
        current_success_rate = metrics['success_rate']
        new_success_rate = (current_success_rate * 0.8) + (0.2 * (1.0 if success else 0.0))
        metrics['success_rate'] = new_success_rate
        
        # Mettre à jour la récompense moyenne
        current_reward = metrics['average_reward']
        metrics['average_reward'] = (current_reward * 0.8) + (reward * 0.2)
        
        # Mettre à jour la durée moyenne
        current_duration = metrics['average_duration']
        metrics['average_duration'] = (current_duration * 0.8) + (duration * 0.2)
        
        # Enregistrer la dernière utilisation
        metrics['last_used'] = datetime.now().isoformat()
        
        self.logger.debug(f"Métriques mises à jour pour {strategy_type.value}: "
                         f"succès={success}, récompense={reward}")
    
    def adapt_strategy_weights(
        self, 
        strategy_type: StrategyType, 
        performance_feedback: Dict[str, float]
    ):
        """
        Adapte les poids d'une stratégie selon les retours de performance.
        
        Args:
            strategy_type: Type de stratégie à adapter
            performance_feedback: Retours de performance par critère
        """
        strategy_config = self.strategies[strategy_type]
        weights = strategy_config.weights
        
        # Adapter les poids selon le feedback
        for criterion, feedback in performance_feedback.items():
            if hasattr(weights, criterion):
                current_value = getattr(weights, criterion)
                # Ajustement graduel basé sur le feedback
                adjustment = (feedback - 0.5) * 0.1  # Feedback entre 0 et 1
                new_value = max(0.0, min(2.0, current_value + adjustment))
                setattr(weights, criterion, new_value)
        
        # Enregistrer l'adaptation
        adaptation_key = f"{strategy_type.value}_{datetime.now().date()}"
        self.learned_adaptations[adaptation_key] = asdict(weights)
        
        self.logger.info(f"Poids adaptés pour {strategy_type.value}")
    
    def get_strategy_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics des stratégies."""
        total_usage = sum(m['usage_count'] for m in self.strategy_metrics.values())
        
        if total_usage == 0:
            return {"total_usage": 0}
        
        # Calculer les statistiques
        strategy_distribution = {}
        for strategy, metrics in self.strategy_metrics.items():
            usage_percent = (metrics['usage_count'] / total_usage) * 100
            strategy_distribution[strategy] = {
                'usage_percentage': usage_percent,
                'success_rate': metrics['success_rate'],
                'average_reward': metrics['average_reward'],
                'average_duration': metrics['average_duration']
            }
        
        # Identifier la stratégie la plus performante
        best_strategy = max(
            self.strategy_metrics.keys(),
            key=lambda s: self.strategy_metrics[s]['success_rate'] * 
                         self.strategy_metrics[s]['average_reward']
        )
        
        return {
            'total_usage': total_usage,
            'strategy_distribution': strategy_distribution,
            'best_performing_strategy': best_strategy,
            'current_strategy': self.current_strategy.value if self.current_strategy else None,
            'learned_adaptations_count': len(self.learned_adaptations)
        }
    
    def save_config(self, config_path: str):
        """Sauvegarde la configuration et les métriques."""
        config_data = {
            'strategy_metrics': self.strategy_metrics,
            'learned_adaptations': self.learned_adaptations,
            'strategy_history': self.strategy_history[-100:]  # Garder seulement les 100 dernières
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration sélecteur sauvegardée: {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def load_config(self, config_path: str):
        """Charge la configuration et les métriques."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.strategy_metrics.update(config_data.get('strategy_metrics', {}))
            self.learned_adaptations.update(config_data.get('learned_adaptations', {}))
            self.strategy_history.extend(config_data.get('strategy_history', []))
            
            self.logger.info(f"Configuration sélecteur chargée: {config_path}")
        except FileNotFoundError:
            self.logger.info(f"Fichier de configuration non trouvé: {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {e}")
    
    # Méthodes privées
    
    def _initialize_default_strategies(self) -> Dict[StrategyType, StrategyConfig]:
        """Initialise les stratégies par défaut."""
        strategies = {}
        
        # Stratégie agressive
        strategies[StrategyType.AGGRESSIVE] = StrategyConfig(
            name="Agressive",
            description="Maximise les gains en acceptant des risques élevés",
            weights=StrategyWeights(
                survival=0.6,
                efficiency=1.4,
                risk_tolerance=0.8,
                speed=1.3,
                stealth=0.2,
                cooperation=0.3,
                resource_conservation=0.7,
                objective_focus=1.5
            ),
            preferred_actions=[ActionType.COMBAT, ActionType.PROFESSION],
            avoided_actions=[ActionType.MAINTENANCE],
            activation_conditions={
                "min_health": 60,
                "safe_zone": True
            },
            duration_limits={
                "combat": 300,  # 5 minutes max en combat
                "movement": 120  # 2 minutes max de déplacement
            }
        )
        
        # Stratégie défensive
        strategies[StrategyType.DEFENSIVE] = StrategyConfig(
            name="Défensive",
            description="Priorité absolue à la survie et à la sécurité",
            weights=StrategyWeights(
                survival=1.8,
                efficiency=0.7,
                risk_tolerance=0.2,
                speed=0.8,
                stealth=1.2,
                cooperation=0.8,
                resource_conservation=1.3,
                objective_focus=0.9
            ),
            preferred_actions=[ActionType.SURVIVAL, ActionType.MOVEMENT],
            avoided_actions=[ActionType.COMBAT],
            activation_conditions={
                "max_health": 50,
                "in_combat": False
            },
            duration_limits={
                "combat": 60,   # 1 minute max en combat
                "profession": 600  # 10 minutes max de métier
            }
        )
        
        # Stratégie équilibrée
        strategies[StrategyType.BALANCED] = StrategyConfig(
            name="Équilibrée",
            description="Équilibre optimal entre tous les aspects",
            weights=StrategyWeights(
                survival=1.0,
                efficiency=1.0,
                risk_tolerance=0.5,
                speed=1.0,
                stealth=0.5,
                cooperation=0.5,
                resource_conservation=1.0,
                objective_focus=1.0
            ),
            preferred_actions=[ActionType.PROFESSION, ActionType.COMBAT, ActionType.MOVEMENT],
            avoided_actions=[],
            activation_conditions={
                "min_health": 30
            },
            duration_limits={
                "combat": 180,     # 3 minutes max en combat
                "profession": 900  # 15 minutes max de métier
            }
        )
        
        # Stratégie efficace
        strategies[StrategyType.EFFICIENT] = StrategyConfig(
            name="Efficace",
            description="Optimisation maximale du temps et des ressources",
            weights=StrategyWeights(
                survival=1.1,
                efficiency=1.8,
                risk_tolerance=0.6,
                speed=1.6,
                stealth=0.3,
                cooperation=0.4,
                resource_conservation=1.4,
                objective_focus=1.7
            ),
            preferred_actions=[ActionType.PROFESSION, ActionType.INVENTORY],
            avoided_actions=[ActionType.SOCIAL],
            activation_conditions={
                "min_health": 40
            },
            duration_limits={
                "movement": 60,    # 1 minute max de déplacement
                "maintenance": 180 # 3 minutes max de maintenance
            }
        )
        
        # Stratégie furtive
        strategies[StrategyType.STEALTH] = StrategyConfig(
            name="Furtive",
            description="Évitement des conflits et discrétion maximale",
            weights=StrategyWeights(
                survival=1.3,
                efficiency=0.9,
                risk_tolerance=0.1,
                speed=0.9,
                stealth=1.8,
                cooperation=0.2,
                resource_conservation=1.2,
                objective_focus=1.1
            ),
            preferred_actions=[ActionType.MOVEMENT, ActionType.PROFESSION],
            avoided_actions=[ActionType.COMBAT, ActionType.SOCIAL],
            activation_conditions={
                "safe_zone": False
            },
            duration_limits={
                "combat": 30,      # 30 secondes max en combat
                "social": 60       # 1 minute max d'interaction sociale
            }
        )
        
        # Stratégie sociale
        strategies[StrategyType.SOCIAL] = StrategyConfig(
            name="Sociale",
            description="Coopération et interaction avec autres joueurs",
            weights=StrategyWeights(
                survival=1.1,
                efficiency=0.8,
                risk_tolerance=0.4,
                speed=0.7,
                stealth=0.2,
                cooperation=1.8,
                resource_conservation=0.9,
                objective_focus=0.8
            ),
            preferred_actions=[ActionType.SOCIAL, ActionType.COMBAT],
            avoided_actions=[ActionType.MAINTENANCE],
            activation_conditions={
                "safe_zone": True
            },
            duration_limits={
                "profession": 300, # 5 minutes max de métier
                "movement": 180    # 3 minutes max de déplacement
            }
        )
        
        return strategies
    
    def _detect_current_situation(self, context: DecisionContext) -> Situation:
        """Détecte la situation actuelle selon le contexte."""
        situation_scores = {}
        
        for situation, detector in self.situation_detectors.items():
            score = detector(context)
            situation_scores[situation] = score
        
        # Retourner la situation avec le score le plus élevé
        return max(situation_scores.keys(), key=lambda s: situation_scores[s])
    
    def _should_keep_current_strategy(
        self, 
        context: DecisionContext, 
        current_situation: Situation
    ) -> bool:
        """Détermine si on doit garder la stratégie actuelle."""
        if not self.current_strategy or not self.strategy_start_time:
            return False
        
        # Vérifier si la stratégie actuelle correspond encore à la situation
        strategy_config = self.strategies[self.current_strategy]
        situation_match = strategy_config.matches_situation(context, current_situation)
        
        # Garder la stratégie si elle correspond encore bien (score > 0.6)
        if situation_match > 0.6:
            return True
        
        # Changer si situation critique
        if current_situation in [Situation.BOSS_FIGHT, Situation.LOW_RESOURCES]:
            return False
        
        # Garder si utilisée depuis moins de 2 minutes (éviter les changements trop fréquents)
        time_since_start = (datetime.now() - self.strategy_start_time).total_seconds()
        if time_since_start < 120:
            return True
        
        return False
    
    def _evaluate_strategy(
        self, 
        strategy_type: StrategyType,
        strategy_config: StrategyConfig, 
        context: DecisionContext,
        current_situation: Situation
    ) -> float:
        """Évalue une stratégie pour le contexte et la situation donnés."""
        # Score de base selon la correspondance avec la situation
        base_score = strategy_config.matches_situation(context, current_situation)
        
        # Ajustements selon les métriques historiques
        metrics = self.strategy_metrics[strategy_type.value]
        performance_multiplier = (metrics['success_rate'] + metrics['average_reward']) / 2.0
        
        # Pénalité pour stratégies récemment utilisées (éviter la répétition)
        recency_penalty = 1.0
        if metrics['last_used']:
            last_used = datetime.fromisoformat(metrics['last_used'])
            hours_since = (datetime.now() - last_used).total_seconds() / 3600
            if hours_since < 1:
                recency_penalty = 0.8
        
        # Ajustement selon les préférences du contexte
        context_multiplier = 1.0
        if context.risk_tolerance > 0.7 and strategy_type == StrategyType.AGGRESSIVE:
            context_multiplier = 1.3
        elif context.risk_tolerance < 0.3 and strategy_type == StrategyType.DEFENSIVE:
            context_multiplier = 1.3
        elif 0.3 <= context.risk_tolerance <= 0.7 and strategy_type == StrategyType.BALANCED:
            context_multiplier = 1.2
        
        final_score = base_score * performance_multiplier * recency_penalty * context_multiplier
        return max(0.0, min(1.0, final_score))
    
    def _record_strategy_change(
        self, 
        new_strategy: StrategyType, 
        situation: Situation,
        context: DecisionContext
    ):
        """Enregistre un changement de stratégie."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'previous_strategy': self.current_strategy.value if self.current_strategy else None,
            'new_strategy': new_strategy.value,
            'situation': situation.value,
            'context_summary': {
                'health_percent': context.health_percent,
                'in_combat': context.in_combat,
                'safe_zone': context.safe_zone,
                'current_objective': context.current_objective
            }
        }
        
        self.strategy_history.append(record)
        
        # Limiter la taille de l'historique
        if len(self.strategy_history) > 500:
            self.strategy_history = self.strategy_history[-250:]
    
    def _generate_strategy_explanation(
        self, 
        strategy_type: StrategyType, 
        score: float,
        context: DecisionContext, 
        situation: Situation, 
        rank: int
    ) -> str:
        """Génère une explication pour une recommandation de stratégie."""
        strategy_config = self.strategies[strategy_type]
        explanations = []
        
        # Explication du rang
        if rank == 1:
            explanations.append("Stratégie recommandée")
        else:
            explanations.append(f"Option #{rank}")
        
        # Explication selon la situation
        if situation == Situation.DANGEROUS_AREA and strategy_type == StrategyType.DEFENSIVE:
            explanations.append("priorité à la sécurité en zone dangereuse")
        elif situation == Situation.PEACEFUL_FARMING and strategy_type == StrategyType.EFFICIENT:
            explanations.append("optimisation pour le farm tranquille")
        elif situation == Situation.BOSS_FIGHT and strategy_type == StrategyType.AGGRESSIVE:
            explanations.append("maximise les dégâts contre le boss")
        
        # Explication selon les métriques
        metrics = self.strategy_metrics[strategy_type.value]
        if metrics['success_rate'] > 0.8:
            explanations.append(f"taux de succès élevé ({metrics['success_rate']:.1%})")
        elif metrics['average_reward'] > 0.7:
            explanations.append("récompenses élevées historiques")
        
        # Explication selon le contexte
        if context.health_percent < 50 and 'survival' in strategy_config.description.lower():
            explanations.append("adapté à la situation de santé faible")
        
        return " - " + ", ".join(explanations)
    
    # Détecteurs de situations
    
    def _detect_peaceful_farming(self, context: DecisionContext) -> float:
        """Détecte une situation de farm tranquille."""
        score = 0.0
        
        if context.safe_zone:
            score += 0.4
        if not context.in_combat and context.enemies_count == 0:
            score += 0.3
        if context.health_percent > 70:
            score += 0.2
        if "farm" in context.current_objective.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_dangerous_area(self, context: DecisionContext) -> float:
        """Détecte une zone dangereuse."""
        score = 0.0
        
        if not context.safe_zone:
            score += 0.4
        if context.enemies_count > 2:
            score += 0.3
        if context.combat_difficulty > 0.7:
            score += 0.2
        if context.health_percent < 50:
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_crowded_area(self, context: DecisionContext) -> float:
        """Détecte une zone avec beaucoup de joueurs."""
        score = 0.0
        
        if context.allies_count > 3:
            score += 0.5
        if len(context.resources_available) == 0:  # Ressources épuisées = compétition
            score += 0.3
        if context.safe_zone:
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_dungeon_exploration(self, context: DecisionContext) -> float:
        """Détecte l'exploration d'un donjon."""
        score = 0.0
        
        if "donjon" in context.current_map.lower() or "dungeon" in context.current_map.lower():
            score += 0.6
        if context.enemies_count > 0 and not context.safe_zone:
            score += 0.2
        if "exploration" in context.current_objective.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_pvp_zone(self, context: DecisionContext) -> float:
        """Détecte une zone PvP."""
        score = 0.0
        
        if not context.safe_zone and context.allies_count > 0 and context.enemies_count > 0:
            score += 0.5
        if "pvp" in context.current_map.lower():
            score += 0.4
        if context.combat_difficulty > 0.5:
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_resource_competition(self, context: DecisionContext) -> float:
        """Détecte une compétition pour les ressources."""
        score = 0.0
        
        if context.allies_count > 1 and len(context.resources_available) > 0:
            score += 0.4
        if context.pod_percent > 70:  # Inventaire se remplit = ressources disponibles
            score += 0.3
        if "farm" in context.current_objective.lower():
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_boss_fight(self, context: DecisionContext) -> float:
        """Détecte un combat de boss."""
        score = 0.0
        
        if context.in_combat and context.combat_difficulty > 0.8:
            score += 0.5
        if context.enemies_count == 1 and context.allies_count > 0:  # Boss unique avec alliés
            score += 0.3
        if "boss" in context.current_objective.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_low_resources(self, context: DecisionContext) -> float:
        """Détecte une situation de ressources faibles."""
        score = 0.0
        
        if context.health_percent < 30:
            score += 0.4
        if context.mana_percent < 20:
            score += 0.3
        if context.health_percent < 50 and context.mana_percent < 40:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_inventory_full(self, context: DecisionContext) -> float:
        """Détecte un inventaire plein."""
        score = 0.0
        
        if context.pod_percent > 90:
            score += 0.6
        elif context.pod_percent > 80:
            score += 0.4
        elif context.pod_percent > 70:
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_mission_critical(self, context: DecisionContext) -> float:
        """Détecte une mission critique."""
        score = 0.0
        
        if "urgent" in context.current_objective.lower() or "critique" in context.current_objective.lower():
            score += 0.5
        if context.objective_progress > 0.8:  # Mission presque terminée
            score += 0.3
        if context.session_time > 7200:  # Plus de 2h de session = objectif important
            score += 0.2
        
        return min(1.0, score)