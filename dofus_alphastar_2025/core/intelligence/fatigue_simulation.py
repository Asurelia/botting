"""
Simulation de Fatigue Comportementale pour TacticalBot
Phase 2 du Projet Augmenta

Fonctionnalités:
- Simulation de fatigue basée sur le temps de session
- Dégradation progressive des performances
- Comportements de récupération
- Adaptation des seuils selon l'activité
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# from ...engine.module_interface import IModule, ModuleStatus
# from ...state.realtime_state import GameState, CombatState, Character

# Type stubs temporaires
class IModule:
    """Stub temporaire pour IModule"""
    def __init__(self, name: str = "Module"):
        self.name = name

ModuleStatus = Any  # Stub temporaire

@dataclass
class FatigueState:
    """État de fatigue du bot"""
    session_start_time: float
    current_fatigue_level: float = 0.0  # 0.0 = frais, 1.0 = épuisé
    physical_fatigue: float = 0.0  # Fatigue physique (mouvements, clics)
    mental_fatigue: float = 0.0    # Fatigue mentale (décisions, concentration)
    last_activity_time: float = 0.0
    consecutive_actions: int = 0
    break_count: int = 0
    recovery_rate: float = 0.1  # Taux de récupération pendant les pauses


@dataclass
class FatigueEffect:
    """Effet de la fatigue sur les performances"""
    accuracy_reduction: float = 0.0  # Réduction de précision (0.0-1.0)
    speed_reduction: float = 0.0     # Réduction de vitesse (0.0-1.0)
    error_increase: float = 0.0      # Augmentation des erreurs (0.0-1.0)
    decision_quality: float = 1.0    # Qualité des décisions (0.0-1.0)
    reaction_time: float = 1.0       # Multiplicateur de temps de réaction


class FatigueSimulator:
    """Simulateur de fatigue comportementale"""

    def __init__(self):
        self.fatigue_params = {
            "max_session_hours": 4.0,        # 4h max avant fatigue critique
            "fatigue_buildup_rate": 0.1,     # Taux d'accumulation par heure
            "recovery_rate": 0.2,            # Taux de récupération par heure de pause
            "activity_threshold": 10,        # Actions/min considérées comme actives
            "break_recovery_bonus": 0.3,     # Bonus de récupération après pause
            "consecutive_action_penalty": 0.05  # Pénalité par action consécutive
        }

        self.activity_history = deque(maxlen=1000)
        self.fatigue_history = deque(maxlen=100)

    def update_fatigue(self, current_time: float, actions_performed: int = 0) -> FatigueState:
        """Met à jour l'état de fatigue"""
        # Calcul du temps de session
        session_duration = current_time - self.fatigue_params.get("session_start", current_time)

        # Calcul de l'activité récente (dernières 5 minutes)
        recent_activity = self._calculate_recent_activity(current_time)

        # Accumulation de fatigue
        base_fatigue = min(1.0, session_duration / 3600 * self.fatigue_params["fatigue_buildup_rate"])

        # Fatigue basée sur l'activité
        activity_fatigue = min(0.3, recent_activity / self.fatigue_params["activity_threshold"] * 0.3)

        # Pénalité pour actions consécutives
        consecutive_penalty = min(0.2, self.fatigue_params["consecutive_action_penalty"] * actions_performed)

        # Fatigue totale
        total_fatigue = base_fatigue + activity_fatigue + consecutive_penalty

        # Création de l'état de fatigue
        fatigue_state = FatigueState(
            session_start_time=self.fatigue_params.get("session_start", current_time),
            current_fatigue_level=min(1.0, total_fatigue),
            physical_fatigue=min(1.0, base_fatigue + activity_fatigue),
            mental_fatigue=min(1.0, consecutive_penalty * 2),
            last_activity_time=current_time,
            consecutive_actions=actions_performed
        )

        self.fatigue_history.append(fatigue_state)
        return fatigue_state

    def _calculate_recent_activity(self, current_time: float) -> float:
        """Calcule l'activité récente"""
        cutoff_time = current_time - 300  # Dernières 5 minutes

        recent_activities = [
            activity for activity in self.activity_history
            if activity["timestamp"] > cutoff_time
        ]

        if not recent_activities:
            return 0.0

        # Moyenne des actions par minute
        total_actions = sum(activity["actions"] for activity in recent_activities)
        time_span = current_time - min(activity["timestamp"] for activity in recent_activities)

        if time_span == 0:
            return 0.0

        return (total_actions / time_span) * 60  # Actions par minute

    def record_activity(self, actions: int = 1):
        """Enregistre une activité"""
        self.activity_history.append({
            "timestamp": time.time(),
            "actions": actions
        })

    def simulate_break_recovery(self, break_duration: float) -> float:
        """Simule la récupération pendant une pause"""
        # Récupération basée sur la durée de la pause
        recovery_amount = break_duration / 3600 * self.fatigue_params["recovery_rate"]

        # Bonus pour les pauses plus longues
        if break_duration > 900:  # >15min
            recovery_amount *= self.fatigue_params["break_recovery_bonus"]

        return min(0.5, recovery_amount)  # Max 50% de récupération par pause


class FatigueSimulation(IModule):
    """
    Module de simulation de fatigue comportementale
    """

    def __init__(self, name: str = "fatigue_simulation"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Composants
        self.simulator = FatigueSimulator()
        self.current_fatigue_state = None

        # Configuration
        self.enable_effects = True
        self.fatigue_thresholds = {
            "warning": 0.3,      # Avertissement à 30%
            "degraded": 0.6,     # Performances dégradées à 60%
            "critical": 0.8      # Critique à 80%
        }

        # Historique des effets
        self.effects_history = deque(maxlen=100)

        # Métriques
        self.metrics = {
            "total_session_time": 0.0,
            "fatigue_warnings": 0,
            "performance_degradations": 0,
            "breaks_taken": 0,
            "average_fatigue_level": 0.0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING

            # Configuration
            self.enable_effects = config.get("enable_effects", True)
            self.fatigue_thresholds.update(config.get("fatigue_thresholds", {}))

            # Paramètres du simulateur
            simulator_config = config.get("simulator_params", {})
            self.simulator.fatigue_params.update(simulator_config)

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Module de simulation de fatigue initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour la simulation de fatigue"""
        if not self.is_active():
            return None

        try:
            current_time = time.time()

            # Mise à jour de la fatigue
            actions_this_cycle = 1 if game_state else 0
            self.simulator.record_activity(actions_this_cycle)

            self.current_fatigue_state = self.simulator.update_fatigue(
                current_time,
                actions_this_cycle
            )

            # Mise à jour des métriques
            self._update_metrics()

            # Vérification des seuils
            self._check_fatigue_thresholds()

            # Retour des effets de fatigue
            if self.enable_effects and self.current_fatigue_state:
                effects = self._calculate_fatigue_effects()
                self.effects_history.append(effects)

                return {
                    "fatigue_effects": effects,
                    "fatigue_state": {
                        "level": self.current_fatigue_state.current_fatigue_level,
                        "physical": self.current_fatigue_state.physical_fatigue,
                        "mental": self.current_fatigue_state.mental_fatigue
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Traite les événements"""
        try:
            if hasattr(event, 'type'):
                if event.type == "session_break":
                    self._handle_break_event(event)
                elif event.type == "high_activity":
                    self._handle_high_activity_event(event)
                elif event.type == "session_end":
                    self._handle_session_end_event(event)

            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "metrics": self.metrics,
            "current_fatigue": self.current_fatigue_state.current_fatigue_level if self.current_fatigue_state else 0.0,
            "enable_effects": self.enable_effects,
            "fatigue_thresholds": self.fatigue_thresholds
        }

    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            # Sauvegarde des données de fatigue
            self._save_fatigue_data()

            self.logger.info("Module de simulation de fatigue nettoyé")

        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")

    def apply_fatigue_effects(self, base_accuracy: float = 1.0, base_speed: float = 1.0) -> Tuple[float, float]:
        """Applique les effets de fatigue à des valeurs de base"""
        if not self.enable_effects or not self.current_fatigue_state:
            return base_accuracy, base_speed

        effects = self._calculate_fatigue_effects()

        # Application des effets
        adjusted_accuracy = base_accuracy * (1.0 - effects.accuracy_reduction)
        adjusted_speed = base_speed * (1.0 - effects.speed_reduction)

        return max(0.1, adjusted_accuracy), max(0.1, adjusted_speed)

    def should_take_break(self) -> bool:
        """Détermine si une pause est recommandée"""
        if not self.current_fatigue_state:
            return False

        fatigue_level = self.current_fatigue_state.current_fatigue_level

        # Pause recommandée si fatigue élevée
        if fatigue_level > self.fatigue_thresholds["degraded"]:
            return True

        # Pause probabiliste basée sur la fatigue
        break_probability = fatigue_level * 0.3  # 30% de chance max
        return np.random.random() < break_probability

    def _calculate_fatigue_effects(self) -> FatigueEffect:
        """Calcule les effets de la fatigue actuelle"""
        if not self.current_fatigue_state:
            return FatigueEffect()

        fatigue_level = self.current_fatigue_state.current_fatigue_level

        # Effets progressifs
        effects = FatigueEffect()

        if fatigue_level > 0.2:
            # Réduction de précision
            effects.accuracy_reduction = min(0.3, fatigue_level * 0.4)

        if fatigue_level > 0.4:
            # Réduction de vitesse
            effects.speed_reduction = min(0.4, (fatigue_level - 0.4) * 0.8)

        if fatigue_level > 0.6:
            # Augmentation des erreurs
            effects.error_increase = min(0.5, (fatigue_level - 0.6) * 1.0)

        if fatigue_level > 0.3:
            # Dégradation de la qualité des décisions
            effects.decision_quality = max(0.3, 1.0 - (fatigue_level * 0.7))

        if fatigue_level > 0.5:
            # Augmentation du temps de réaction
            effects.reaction_time = 1.0 + (fatigue_level * 0.5)

        return effects

    def _update_metrics(self):
        """Met à jour les métriques"""
        if self.current_fatigue_state:
            self.metrics["total_session_time"] = time.time() - self.current_fatigue_state.session_start_time
            self.metrics["average_fatigue_level"] = (
                self.metrics["average_fatigue_level"] * 0.9 +
                self.current_fatigue_state.current_fatigue_level * 0.1
            )

    def _check_fatigue_thresholds(self):
        """Vérifie les seuils de fatigue"""
        if not self.current_fatigue_state:
            return

        fatigue_level = self.current_fatigue_state.current_fatigue_level

        if fatigue_level > self.fatigue_thresholds["critical"]:
            self.logger.warning(f"Fatigue critique: {fatigue_level:.2f}")
        elif fatigue_level > self.fatigue_thresholds["degraded"]:
            self.logger.info(f"Performances dégradées: {fatigue_level:.2f}")
            self.metrics["performance_degradations"] += 1
        elif fatigue_level > self.fatigue_thresholds["warning"]:
            self.logger.debug(f"Avertissement fatigue: {fatigue_level:.2f}")
            self.metrics["fatigue_warnings"] += 1

    def _handle_break_event(self, event):
        """Traite un événement de pause"""
        break_duration = event.data.get("duration", 0.0)
        recovery = self.simulator.simulate_break_recovery(break_duration)

        if self.current_fatigue_state:
            # Application de la récupération
            self.current_fatigue_state.current_fatigue_level = max(
                0.0,
                self.current_fatigue_state.current_fatigue_level - recovery
            )
            self.current_fatigue_state.break_count += 1

        self.metrics["breaks_taken"] += 1

    def _handle_high_activity_event(self, event):
        """Traite un événement d'activité élevée"""
        # Augmentation temporaire de la fatigue
        if self.current_fatigue_state:
            self.current_fatigue_state.mental_fatigue = min(
                1.0,
                self.current_fatigue_state.mental_fatigue + 0.1
            )

    def _handle_session_end_event(self, event):
        """Traite la fin de session"""
        # Sauvegarde finale de l'état de fatigue
        pass

    def _save_fatigue_data(self):
        """Sauvegarde les données de fatigue"""
        try:
            # Sauvegarde dans un fichier (placeholder)
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")

    def get_fatigue_report(self) -> Dict[str, Any]:
        """Génère un rapport sur la fatigue"""
        if not self.current_fatigue_state:
            return {"error": "No fatigue data available"}

        effects = self._calculate_fatigue_effects()

        return {
            "current_fatigue_level": self.current_fatigue_state.current_fatigue_level,
            "physical_fatigue": self.current_fatigue_state.physical_fatigue,
            "mental_fatigue": self.current_fatigue_state.mental_fatigue,
            "session_duration_hours": self.metrics["total_session_time"] / 3600,
            "effects": {
                "accuracy_reduction": effects.accuracy_reduction,
                "speed_reduction": effects.speed_reduction,
                "error_increase": effects.error_increase,
                "decision_quality": effects.decision_quality,
                "reaction_time_multiplier": effects.reaction_time
            },
            "metrics": self.metrics,
            "recommendations": self._generate_fatigue_recommendations()
        }

    def _generate_fatigue_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur la fatigue"""
        recommendations = []

        if not self.current_fatigue_state:
            return recommendations

        fatigue_level = self.current_fatigue_state.current_fatigue_level

        if fatigue_level > 0.8:
            recommendations.append("Pause immédiate recommandée - fatigue critique")
        elif fatigue_level > 0.6:
            recommendations.append("Envisagez une pause - performances dégradées")
        elif fatigue_level > 0.3:
            recommendations.append("Fatigue modérée - surveillez les performances")

        if self.current_fatigue_state.mental_fatigue > 0.7:
            recommendations.append("Fatigue mentale élevée - réduisez la fréquence des décisions")

        if self.current_fatigue_state.break_count == 0 and fatigue_level > 0.4:
            recommendations.append("Première pause recommandée pour récupération")

        return recommendations