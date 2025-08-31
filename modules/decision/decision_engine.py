"""
Moteur de décision centralisé pour le système de botting.

Ce module implémente un système de décision multi-critères avec gestion des priorités,
évaluation contextuelle et apprentissage des préférences utilisateur.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import math


class Priority(Enum):
    """Énumération des priorités du système."""
    CRITICAL = 100    # Survie immédiate (fuite, potion critique)
    HIGH = 80         # Sécurité (heal, bouclier)
    MEDIUM = 60       # Objectifs principaux (farm, combat)
    LOW = 40          # Efficacité (optimisation, confort)
    MINIMAL = 20      # Maintenance (tri inventaire, réparation)


class ActionType(Enum):
    """Types d'actions possibles."""
    SURVIVAL = "survival"        # Actions de survie
    COMBAT = "combat"           # Actions de combat
    MOVEMENT = "movement"       # Actions de déplacement
    PROFESSION = "profession"   # Actions de métiers
    INVENTORY = "inventory"     # Gestion d'inventaire
    SOCIAL = "social"          # Interactions sociales
    MAINTENANCE = "maintenance" # Maintenance du bot


@dataclass
class DecisionContext:
    """Contexte de décision contenant toutes les informations nécessaires."""
    # État du personnage
    health_percent: float = 100.0
    mana_percent: float = 100.0
    pod_percent: float = 0.0
    
    # État du combat
    in_combat: bool = False
    enemies_count: int = 0
    allies_count: int = 0
    combat_difficulty: float = 0.0
    
    # État de l'environnement
    current_map: str = ""
    safe_zone: bool = True
    resources_available: List[str] = None
    
    # Objectifs et progression
    current_objective: str = ""
    objective_progress: float = 0.0
    session_time: float = 0.0
    
    # Préférences utilisateur
    risk_tolerance: float = 0.5  # 0.0 = très prudent, 1.0 = très agressif
    efficiency_focus: float = 0.5  # 0.0 = sécurité, 1.0 = rapidité
    
    def __post_init__(self):
        if self.resources_available is None:
            self.resources_available = []


@dataclass
class Decision:
    """Représente une décision possible avec tous ses attributs."""
    action_id: str
    action_type: ActionType
    priority: Priority
    confidence: float  # 0.0 à 1.0
    estimated_duration: float  # en secondes
    success_probability: float  # 0.0 à 1.0
    risk_level: float  # 0.0 à 1.0
    reward_estimate: float  # valeur attendue
    prerequisites: List[str] = None
    consequences: Dict[str, Any] = None
    module_source: str = ""
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.consequences is None:
            self.consequences = {}
    
    def calculate_score(self, context: DecisionContext, weights: Dict[str, float]) -> float:
        """Calcule le score de la décision selon le contexte et les poids."""
        # Score de base basé sur la priorité
        base_score = self.priority.value
        
        # Ajustements contextuels
        contextual_multiplier = 1.0
        
        # Ajustement selon l'état de santé (survie prioritaire)
        if context.health_percent < 30 and self.action_type == ActionType.SURVIVAL:
            contextual_multiplier *= 2.0
        elif context.health_percent < 50 and self.action_type != ActionType.SURVIVAL:
            contextual_multiplier *= 0.7
        
        # Ajustement selon le combat
        if context.in_combat:
            if self.action_type in [ActionType.COMBAT, ActionType.SURVIVAL]:
                contextual_multiplier *= 1.5
            else:
                contextual_multiplier *= 0.3
        
        # Ajustement selon la tolérance au risque
        risk_factor = 1.0 - (self.risk_level * (1.0 - context.risk_tolerance))
        
        # Ajustement selon l'efficacité
        efficiency_factor = (
            self.success_probability * weights.get('success', 1.0) +
            (1.0 / max(self.estimated_duration, 0.1)) * weights.get('speed', 1.0) +
            self.reward_estimate * weights.get('reward', 1.0)
        ) / 3.0
        
        # Score final
        final_score = (
            base_score * 
            contextual_multiplier * 
            self.confidence * 
            risk_factor * 
            efficiency_factor
        )
        
        return max(0.0, min(1000.0, final_score))


class DecisionEngine:
    """
    Moteur de décision centralisé qui évalue les actions possibles 
    et sélectionne la meilleure selon le contexte.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des poids par défaut
        self.default_weights = {
            'priority': 1.0,
            'confidence': 0.8,
            'success': 0.9,
            'speed': 0.6,
            'reward': 0.7,
            'risk': -0.5
        }
        
        # Historique des décisions pour l'apprentissage
        self.decision_history: List[Dict[str, Any]] = []
        
        # Préférences apprises
        self.learned_preferences = {
            'action_success_rates': {},
            'preferred_strategies': {},
            'risk_patterns': {},
            'time_preferences': {}
        }
        
        # Conflits de modules
        self.module_conflicts = {
            'combat_vs_profession': self._resolve_combat_profession_conflict,
            'movement_vs_action': self._resolve_movement_action_conflict,
            'inventory_vs_objective': self._resolve_inventory_objective_conflict
        }
        
        # Charger la configuration si fournie
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("Moteur de décision initialisé")
    
    def evaluate_decisions(
        self, 
        possible_decisions: List[Decision], 
        context: DecisionContext,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Decision, float]]:
        """
        Évalue toutes les décisions possibles et retourne une liste triée par score.
        
        Args:
            possible_decisions: Liste des décisions possibles
            context: Contexte actuel de décision
            custom_weights: Poids personnalisés (optionnel)
            
        Returns:
            Liste de tuples (décision, score) triée par score décroissant
        """
        weights = custom_weights or self.default_weights
        scored_decisions = []
        
        for decision in possible_decisions:
            # Vérifier les prérequis
            if not self._check_prerequisites(decision, context):
                continue
            
            # Calculer le score
            score = decision.calculate_score(context, weights)
            
            # Appliquer l'apprentissage
            score = self._apply_learning_adjustment(decision, score, context)
            
            scored_decisions.append((decision, score))
        
        # Trier par score décroissant
        scored_decisions.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.debug(f"Évalué {len(possible_decisions)} décisions, "
                         f"retenu {len(scored_decisions)} valides")
        
        return scored_decisions
    
    def make_decision(
        self, 
        possible_decisions: List[Decision], 
        context: DecisionContext,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Optional[Decision]:
        """
        Sélectionne la meilleure décision selon le contexte.
        
        Args:
            possible_decisions: Liste des décisions possibles
            context: Contexte actuel de décision
            custom_weights: Poids personnalisés (optionnel)
            
        Returns:
            La meilleure décision ou None si aucune n'est valide
        """
        if not possible_decisions:
            self.logger.warning("Aucune décision possible fournie")
            return None
        
        # Résoudre les conflits entre modules
        filtered_decisions = self._resolve_module_conflicts(possible_decisions, context)
        
        # Évaluer les décisions
        scored_decisions = self.evaluate_decisions(filtered_decisions, context, custom_weights)
        
        if not scored_decisions:
            self.logger.warning("Aucune décision valide après évaluation")
            return None
        
        best_decision, best_score = scored_decisions[0]
        
        # Enregistrer la décision pour l'apprentissage
        self._record_decision(best_decision, best_score, context)
        
        self.logger.info(f"Décision sélectionnée: {best_decision.action_id} "
                        f"(score: {best_score:.2f})")
        
        return best_decision
    
    def get_recommendations(
        self, 
        possible_decisions: List[Decision], 
        context: DecisionContext,
        top_n: int = 3
    ) -> List[Tuple[Decision, float, str]]:
        """
        Génère des recommandations intelligentes avec explications.
        
        Args:
            possible_decisions: Liste des décisions possibles
            context: Contexte actuel
            top_n: Nombre de recommandations à retourner
            
        Returns:
            Liste de tuples (décision, score, explication)
        """
        scored_decisions = self.evaluate_decisions(possible_decisions, context)
        recommendations = []
        
        for i, (decision, score) in enumerate(scored_decisions[:top_n]):
            explanation = self._generate_explanation(decision, score, context, i + 1)
            recommendations.append((decision, score, explanation))
        
        return recommendations
    
    def update_decision_outcome(
        self, 
        decision_id: str, 
        success: bool, 
        actual_duration: float,
        actual_reward: float
    ):
        """
        Met à jour les résultats d'une décision pour l'apprentissage.
        
        Args:
            decision_id: ID de la décision exécutée
            success: Si l'action a réussi
            actual_duration: Durée réelle d'exécution
            actual_reward: Récompense réelle obtenue
        """
        # Trouver la décision dans l'historique
        for record in reversed(self.decision_history):
            if record['decision_id'] == decision_id:
                record['outcome'] = {
                    'success': success,
                    'actual_duration': actual_duration,
                    'actual_reward': actual_reward,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Mettre à jour l'apprentissage
                self._update_learning(record)
                break
        
        self.logger.debug(f"Résultat mis à jour pour la décision {decision_id}: "
                         f"succès={success}")
    
    def configure_priorities(self, priority_config: Dict[str, float]):
        """
        Configure les poids des priorités.
        
        Args:
            priority_config: Dictionnaire des nouveaux poids
        """
        self.default_weights.update(priority_config)
        self.logger.info("Priorités mises à jour")
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de décision."""
        total_decisions = len(self.decision_history)
        
        if total_decisions == 0:
            return {"total_decisions": 0}
        
        successful_decisions = sum(
            1 for record in self.decision_history 
            if record.get('outcome', {}).get('success', False)
        )
        
        action_types_count = {}
        for record in self.decision_history:
            action_type = record.get('action_type', 'unknown')
            action_types_count[action_type] = action_types_count.get(action_type, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'success_rate': successful_decisions / total_decisions,
            'action_types_distribution': action_types_count,
            'learned_preferences': self.learned_preferences
        }
    
    def save_config(self, config_path: str):
        """Sauvegarde la configuration et l'apprentissage."""
        config_data = {
            'weights': self.default_weights,
            'learned_preferences': self.learned_preferences,
            'decision_history': self.decision_history[-100:]  # Garder seulement les 100 dernières
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration sauvegardée: {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def load_config(self, config_path: str):
        """Charge la configuration et l'apprentissage."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.default_weights.update(config_data.get('weights', {}))
            self.learned_preferences.update(config_data.get('learned_preferences', {}))
            self.decision_history.extend(config_data.get('decision_history', []))
            
            self.logger.info(f"Configuration chargée: {config_path}")
        except FileNotFoundError:
            self.logger.info(f"Fichier de configuration non trouvé: {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {e}")
    
    # Méthodes privées
    
    def _check_prerequisites(self, decision: Decision, context: DecisionContext) -> bool:
        """Vérifie si tous les prérequis d'une décision sont remplis."""
        for prereq in decision.prerequisites:
            if prereq == "health_above_50" and context.health_percent < 50:
                return False
            elif prereq == "mana_above_30" and context.mana_percent < 30:
                return False
            elif prereq == "not_in_combat" and context.in_combat:
                return False
            elif prereq == "safe_zone" and not context.safe_zone:
                return False
            # Ajouter d'autres prérequis selon les besoins
        
        return True
    
    def _apply_learning_adjustment(
        self, 
        decision: Decision, 
        base_score: float, 
        context: DecisionContext
    ) -> float:
        """Applique les ajustements basés sur l'apprentissage."""
        adjusted_score = base_score
        
        # Ajustement basé sur le taux de succès historique
        action_key = f"{decision.action_type.value}_{decision.action_id}"
        success_rate = self.learned_preferences['action_success_rates'].get(action_key, 0.5)
        adjusted_score *= (0.5 + success_rate)
        
        # Ajustement basé sur les préférences temporelles
        current_hour = datetime.now().hour
        time_pref = self.learned_preferences['time_preferences'].get(
            f"{decision.action_type.value}_{current_hour}", 1.0
        )
        adjusted_score *= time_pref
        
        return adjusted_score
    
    def _resolve_module_conflicts(
        self, 
        decisions: List[Decision], 
        context: DecisionContext
    ) -> List[Decision]:
        """Résout les conflits entre modules."""
        # Grouper les décisions par module source
        module_decisions = {}
        for decision in decisions:
            module = decision.module_source
            if module not in module_decisions:
                module_decisions[module] = []
            module_decisions[module].append(decision)
        
        # Appliquer les résolveurs de conflits
        filtered_decisions = []
        
        # Conflit combat vs profession
        if 'combat' in module_decisions and 'profession' in module_decisions:
            resolved = self.module_conflicts['combat_vs_profession'](
                module_decisions['combat'], 
                module_decisions['profession'], 
                context
            )
            filtered_decisions.extend(resolved)
        else:
            filtered_decisions.extend(module_decisions.get('combat', []))
            filtered_decisions.extend(module_decisions.get('profession', []))
        
        # Ajouter les autres décisions non conflictuelles
        for module, module_decisions_list in module_decisions.items():
            if module not in ['combat', 'profession']:
                filtered_decisions.extend(module_decisions_list)
        
        return filtered_decisions
    
    def _resolve_combat_profession_conflict(
        self, 
        combat_decisions: List[Decision], 
        profession_decisions: List[Decision], 
        context: DecisionContext
    ) -> List[Decision]:
        """Résout le conflit entre combat et profession."""
        # Priorité au combat si en danger ou déjà en combat
        if context.in_combat or context.health_percent < 50:
            return combat_decisions
        
        # Sinon, permettre les deux avec préférence pour l'objectif actuel
        if "combat" in context.current_objective.lower():
            return combat_decisions + [d for d in profession_decisions if d.priority.value < 60]
        else:
            return profession_decisions + [d for d in combat_decisions if d.priority.value > 80]
    
    def _resolve_movement_action_conflict(
        self, 
        movement_decisions: List[Decision], 
        action_decisions: List[Decision], 
        context: DecisionContext
    ) -> List[Decision]:
        """Résout le conflit entre mouvement et action."""
        # Ne pas bouger si une action importante est en cours
        high_priority_actions = [d for d in action_decisions if d.priority.value > 70]
        if high_priority_actions:
            return high_priority_actions
        
        return movement_decisions + action_decisions
    
    def _resolve_inventory_objective_conflict(
        self, 
        inventory_decisions: List[Decision], 
        objective_decisions: List[Decision], 
        context: DecisionContext
    ) -> List[Decision]:
        """Résout le conflit entre gestion d'inventaire et objectif."""
        # Priorité à la gestion d'inventaire si pods pleins
        if context.pod_percent > 90:
            return inventory_decisions
        
        # Sinon, équilibrer selon les priorités
        return objective_decisions + [d for d in inventory_decisions if d.priority.value > 60]
    
    def _record_decision(self, decision: Decision, score: float, context: DecisionContext):
        """Enregistre une décision dans l'historique."""
        record = {
            'decision_id': f"{decision.action_id}_{datetime.now().timestamp()}",
            'action_id': decision.action_id,
            'action_type': decision.action_type.value,
            'priority': decision.priority.value,
            'score': score,
            'context': asdict(context),
            'timestamp': datetime.now().isoformat(),
            'outcome': None  # Sera rempli plus tard
        }
        
        self.decision_history.append(record)
        
        # Limiter la taille de l'historique
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def _update_learning(self, decision_record: Dict[str, Any]):
        """Met à jour l'apprentissage basé sur le résultat d'une décision."""
        outcome = decision_record.get('outcome')
        if not outcome:
            return
        
        action_key = f"{decision_record['action_type']}_{decision_record['action_id']}"
        
        # Mettre à jour le taux de succès
        current_rate = self.learned_preferences['action_success_rates'].get(action_key, 0.5)
        new_rate = (current_rate * 0.9) + (0.1 * (1.0 if outcome['success'] else 0.0))
        self.learned_preferences['action_success_rates'][action_key] = new_rate
        
        # Mettre à jour les préférences temporelles
        timestamp = datetime.fromisoformat(decision_record['timestamp'])
        hour_key = f"{decision_record['action_type']}_{timestamp.hour}"
        current_pref = self.learned_preferences['time_preferences'].get(hour_key, 1.0)
        
        if outcome['success']:
            new_pref = min(1.5, current_pref * 1.05)
        else:
            new_pref = max(0.5, current_pref * 0.95)
        
        self.learned_preferences['time_preferences'][hour_key] = new_pref
    
    def _generate_explanation(
        self, 
        decision: Decision, 
        score: float, 
        context: DecisionContext, 
        rank: int
    ) -> str:
        """Génère une explication pour une recommandation."""
        explanations = []
        
        # Explication du rang
        if rank == 1:
            explanations.append("Meilleure option")
        else:
            explanations.append(f"Option #{rank}")
        
        # Explication de la priorité
        if decision.priority == Priority.CRITICAL:
            explanations.append("priorité critique")
        elif decision.priority == Priority.HIGH:
            explanations.append("priorité élevée")
        
        # Explication contextuelle
        if context.health_percent < 30 and decision.action_type == ActionType.SURVIVAL:
            explanations.append("nécessaire pour la survie")
        elif context.in_combat and decision.action_type == ActionType.COMBAT:
            explanations.append("adapté au combat en cours")
        
        # Explication de l'efficacité
        if decision.success_probability > 0.8:
            explanations.append("haute probabilité de succès")
        elif decision.estimated_duration < 5:
            explanations.append("action rapide")
        
        return " - " + ", ".join(explanations)