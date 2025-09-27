"""
Uncertainty Management System
Système de gestion de l'incertitude pour des décisions IA robustes
Implémente confidence scoring, risk assessment et recovery mechanisms
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import asyncio
import statistics
from collections import deque, defaultdict
import math

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Niveaux de confiance standardisés"""
    VERY_LOW = (0.0, 0.2)
    LOW = (0.2, 0.4)
    MEDIUM = (0.4, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """Détermine le niveau depuis un score"""
        for level in cls:
            if level.min_val <= score <= level.max_val:
                return level
        return cls.VERY_LOW

class RiskLevel(Enum):
    """Niveaux de risque pour les actions"""
    MINIMAL = 1      # Actions très sûres
    LOW = 2          # Risque faible
    MODERATE = 3     # Risque modéré
    HIGH = 4         # Risque élevé
    CRITICAL = 5     # Risque critique

class UncertaintySource(Enum):
    """Sources d'incertitude identifiées"""
    SENSOR_NOISE = auto()        # Bruit dans les capteurs/vision
    MODEL_UNCERTAINTY = auto()   # Incertitude du modèle IA
    ENVIRONMENT_CHANGE = auto()  # Changement d'environnement
    INCOMPLETE_INFO = auto()     # Information incomplète
    TEMPORAL_DRIFT = auto()      # Dérive temporelle
    ADVERSARIAL = auto()         # Conditions adverses
    SYSTEM_PERFORMANCE = auto()  # Performance système variable

@dataclass
class UncertaintyMeasurement:
    """Mesure d'incertitude pour une décision/action"""
    confidence_score: float
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    uncertainty_sources: List[UncertaintySource]

    # Métriques détaillées
    model_confidence: float = 0.0
    data_quality: float = 1.0
    temporal_stability: float = 1.0
    context_relevance: float = 1.0

    # Métadonnées
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_context: Dict[str, Any] = field(default_factory=dict)

    def overall_reliability(self) -> float:
        """Score de fiabilité global (0-1)"""
        factors = [
            self.confidence_score,
            self.data_quality,
            self.temporal_stability,
            self.context_relevance
        ]

        # Pondération avec pénalité pour les sources d'incertitude
        uncertainty_penalty = len(self.uncertainty_sources) * 0.05
        base_score = statistics.harmonic_mean([f for f in factors if f > 0])

        return max(0.0, min(1.0, base_score - uncertainty_penalty))

@dataclass
class DecisionCheckpoint:
    """Point de sauvegarde pour rollback"""
    decision_id: str
    timestamp: datetime
    state_snapshot: Dict[str, Any]
    action_taken: str
    confidence_at_decision: float
    context: Dict[str, Any]

    def age_seconds(self) -> float:
        """Âge du checkpoint en secondes"""
        return (datetime.now() - self.timestamp).total_seconds()

class ConfidenceEstimator:
    """Estimateur de confiance multi-sources"""

    def __init__(self):
        self.calibration_data = deque(maxlen=1000)
        self.historical_accuracy = defaultdict(list)

    def estimate_model_confidence(self, model_output: Any,
                                model_metadata: Dict[str, Any] = None) -> float:
        """Estime la confiance d'un modèle"""
        try:
            if isinstance(model_output, dict):
                # Cas YOLO/détection avec scores
                if 'confidence' in model_output:
                    return float(model_output['confidence'])
                elif 'score' in model_output:
                    return float(model_output['score'])
                elif 'probability' in model_output:
                    return float(model_output['probability'])

            elif isinstance(model_output, (list, tuple)):
                # Cas multiple détections
                if len(model_output) > 0:
                    confidences = []
                    for item in model_output:
                        item_confidence = self.estimate_model_confidence(item)
                        if item_confidence > 0:
                            confidences.append(item_confidence)

                    if confidences:
                        # Moyenne pondérée par la qualité
                        return statistics.mean(confidences)

            elif isinstance(model_output, (int, float)):
                # Score direct
                return max(0.0, min(1.0, float(model_output)))

            # Cas par défaut : incertitude élevée
            return 0.5

        except Exception as e:
            logger.warning(f"Erreur estimation confiance modèle: {e}")
            return 0.3

    def estimate_data_quality(self, data: Any,
                            expected_format: Dict[str, Any] = None) -> float:
        """Estime la qualité des données d'entrée"""
        try:
            quality_factors = []

            # Complétude des données
            if isinstance(data, dict):
                if expected_format:
                    expected_keys = set(expected_format.keys())
                    actual_keys = set(data.keys())
                    completeness = len(actual_keys & expected_keys) / len(expected_keys)
                    quality_factors.append(completeness)

                # Valeurs non nulles
                non_null_ratio = sum(1 for v in data.values() if v is not None) / len(data)
                quality_factors.append(non_null_ratio)

            elif isinstance(data, (list, tuple)):
                # Cohérence de la liste
                if len(data) > 0:
                    # Variance des types (moins = mieux)
                    types_variance = len(set(type(item) for item in data)) / len(data)
                    quality_factors.append(1.0 - types_variance)

            elif hasattr(data, 'shape'):  # Array-like
                # Vérification forme et valeurs
                if hasattr(data, 'size') and data.size > 0:
                    # Détection NaN/inf
                    finite_ratio = np.isfinite(data).sum() / data.size
                    quality_factors.append(finite_ratio)

            # Score composite
            if quality_factors:
                return statistics.mean(quality_factors)
            else:
                return 0.8  # Neutre si pas d'analyse possible

        except Exception as e:
            logger.warning(f"Erreur estimation qualité données: {e}")
            return 0.6

    def estimate_temporal_stability(self, current_value: Any,
                                  history: List[Any],
                                  window_size: int = 10) -> float:
        """Estime la stabilité temporelle"""
        try:
            if not history or len(history) < 2:
                return 0.7  # Neutre sans historique

            recent_history = history[-window_size:]

            # Analyse selon le type
            if isinstance(current_value, (int, float)):
                # Valeurs numériques : variance
                values = [float(v) for v in recent_history if isinstance(v, (int, float))]
                if len(values) >= 2:
                    mean_val = statistics.mean(values)
                    if mean_val != 0:
                        coefficient_variation = statistics.stdev(values) / abs(mean_val)
                        stability = max(0.0, 1.0 - coefficient_variation)
                        return min(1.0, stability)

            elif isinstance(current_value, str):
                # Valeurs catégorielles : consistance
                recent_values = [str(v) for v in recent_history]
                if recent_values:
                    most_common = max(set(recent_values), key=recent_values.count)
                    consistency = recent_values.count(most_common) / len(recent_values)
                    return consistency

            elif isinstance(current_value, dict):
                # Objets complexes : similarité structurelle
                key_consistencies = []
                for key in current_value.keys():
                    key_values = [h.get(key) for h in recent_history if isinstance(h, dict)]
                    if key_values:
                        key_stability = self.estimate_temporal_stability(
                            current_value[key], key_values, window_size
                        )
                        key_consistencies.append(key_stability)

                if key_consistencies:
                    return statistics.mean(key_consistencies)

            return 0.7  # Neutre par défaut

        except Exception as e:
            logger.warning(f"Erreur estimation stabilité temporelle: {e}")
            return 0.5

    def calibrate_confidence(self, predicted_confidence: float,
                           actual_outcome: bool):
        """Calibre les estimations de confiance avec les résultats réels"""
        self.calibration_data.append((predicted_confidence, actual_outcome))

        # Mise à jour calibration si suffisamment de données
        if len(self.calibration_data) >= 50:
            self._update_calibration_model()

    def _update_calibration_model(self):
        """Met à jour le modèle de calibration"""
        try:
            # Groupage par bins de confiance
            bins = np.linspace(0, 1, 11)  # 10 bins
            bin_accuracies = {}

            for i in range(len(bins) - 1):
                bin_min, bin_max = bins[i], bins[i + 1]
                bin_data = [
                    outcome for conf, outcome in self.calibration_data
                    if bin_min <= conf < bin_max
                ]

                if bin_data:
                    bin_accuracies[i] = statistics.mean(bin_data)

            self.bin_accuracies = bin_accuracies
            logger.info(f"Modèle de calibration mis à jour avec {len(self.calibration_data)} points")

        except Exception as e:
            logger.error(f"Erreur mise à jour calibration: {e}")

class RiskAssessor:
    """Évaluateur de risque pour les actions"""

    def __init__(self):
        self.risk_models = {}
        self.risk_history = defaultdict(list)

    def assess_action_risk(self, action: str,
                          context: Dict[str, Any],
                          uncertainty: UncertaintyMeasurement) -> RiskLevel:
        """Évalue le risque d'une action dans un contexte"""
        try:
            risk_factors = []

            # Facteur 1: Incertitude globale
            uncertainty_risk = 1.0 - uncertainty.overall_reliability()
            risk_factors.append(uncertainty_risk)

            # Facteur 2: Risque spécifique à l'action
            action_risk = self._assess_action_specific_risk(action, context)
            risk_factors.append(action_risk)

            # Facteur 3: Risque contextuel
            context_risk = self._assess_context_risk(context)
            risk_factors.append(context_risk)

            # Facteur 4: Risque historique
            historical_risk = self._assess_historical_risk(action, context)
            risk_factors.append(historical_risk)

            # Score composite avec pondération
            weights = [0.3, 0.3, 0.2, 0.2]
            composite_risk = sum(w * r for w, r in zip(weights, risk_factors))

            # Conversion en niveau de risque
            if composite_risk < 0.2:
                return RiskLevel.MINIMAL
            elif composite_risk < 0.4:
                return RiskLevel.LOW
            elif composite_risk < 0.6:
                return RiskLevel.MODERATE
            elif composite_risk < 0.8:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL

        except Exception as e:
            logger.error(f"Erreur évaluation risque: {e}")
            return RiskLevel.HIGH  # Conservateur en cas d'erreur

    def _assess_action_specific_risk(self, action: str,
                                   context: Dict[str, Any]) -> float:
        """Évalue le risque spécifique à une action"""
        # Règles de base pour actions DOFUS
        action_risks = {
            'move': 0.1,
            'attack': 0.3,
            'cast_spell': 0.3,
            'interact_npc': 0.2,
            'trade': 0.4,
            'bank_operation': 0.5,
            'send_message': 0.6,  # Risque social
            'pvp_action': 0.8,
            'admin_command': 0.9
        }

        action_lower = action.lower()
        base_risk = 0.3  # Risque par défaut

        for risk_action, risk_value in action_risks.items():
            if risk_action in action_lower:
                base_risk = risk_value
                break

        # Modificateurs contextuels
        if context.get('in_combat', False):
            base_risk *= 1.2

        if context.get('players_nearby', 0) > 5:
            base_risk *= 1.1

        if context.get('hp_percentage', 100) < 30:
            base_risk *= 1.3

        return min(1.0, base_risk)

    def _assess_context_risk(self, context: Dict[str, Any]) -> float:
        """Évalue le risque lié au contexte"""
        context_risk = 0.0

        # Facteurs de risque contextuels
        if context.get('in_pvp_zone', False):
            context_risk += 0.3

        if context.get('server_lag', False):
            context_risk += 0.2

        aggressive_players = context.get('aggressive_players_nearby', 0)
        if aggressive_players > 0:
            context_risk += min(0.4, aggressive_players * 0.1)

        # Niveau de zone
        player_level = context.get('player_level', 100)
        zone_level = context.get('zone_level', player_level)
        if zone_level > player_level + 10:
            level_risk = (zone_level - player_level) / 50
            context_risk += min(0.3, level_risk)

        return min(1.0, context_risk)

    def _assess_historical_risk(self, action: str,
                              context: Dict[str, Any]) -> float:
        """Évalue le risque basé sur l'historique"""
        action_history = self.risk_history.get(action, [])

        if not action_history:
            return 0.3  # Neutre sans historique

        # Analyse des échecs récents
        recent_history = action_history[-20:]  # 20 dernières occurrences
        failure_rate = sum(1 for outcome in recent_history if not outcome) / len(recent_history)

        return failure_rate

class UncertaintyManager:
    """Gestionnaire principal de l'incertitude"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UncertaintyManager")

        # Composants
        self.confidence_estimator = ConfidenceEstimator()
        self.risk_assessor = RiskAssessor()

        # Historique et checkpoints
        self.decision_history = deque(maxlen=1000)
        self.checkpoints = deque(maxlen=50)
        self.uncertainty_trends = defaultdict(list)

        # Configuration
        self.rollback_enabled = self.config.get('rollback_enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.risk_threshold = self.config.get('risk_threshold', RiskLevel.MODERATE)

    async def evaluate_decision(self, decision_data: Dict[str, Any],
                              context: Dict[str, Any] = None) -> UncertaintyMeasurement:
        """Évalue l'incertitude d'une décision"""
        try:
            context = context or {}

            # Estimation de confiance du modèle
            model_confidence = self.confidence_estimator.estimate_model_confidence(
                decision_data.get('model_output'),
                decision_data.get('model_metadata', {})
            )

            # Estimation qualité des données
            data_quality = self.confidence_estimator.estimate_data_quality(
                decision_data.get('input_data'),
                decision_data.get('expected_format')
            )

            # Estimation stabilité temporelle
            temporal_stability = self.confidence_estimator.estimate_temporal_stability(
                decision_data.get('current_value'),
                decision_data.get('history', [])
            )

            # Pertinence contextuelle
            context_relevance = self._assess_context_relevance(decision_data, context)

            # Score de confiance composite
            confidence_factors = [model_confidence, data_quality, temporal_stability, context_relevance]
            confidence_score = statistics.harmonic_mean([f for f in confidence_factors if f > 0])

            # Identification des sources d'incertitude
            uncertainty_sources = self._identify_uncertainty_sources(
                model_confidence, data_quality, temporal_stability, context_relevance, context
            )

            # Niveau de confiance
            confidence_level = ConfidenceLevel.from_score(confidence_score)

            # Évaluation du risque
            measurement = UncertaintyMeasurement(
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                risk_level=RiskLevel.MODERATE,  # Sera mis à jour
                uncertainty_sources=uncertainty_sources,
                model_confidence=model_confidence,
                data_quality=data_quality,
                temporal_stability=temporal_stability,
                context_relevance=context_relevance,
                measurement_context=context.copy()
            )

            # Évaluation du risque avec la mesure d'incertitude
            if 'action' in decision_data:
                measurement.risk_level = self.risk_assessor.assess_action_risk(
                    decision_data['action'], context, measurement
                )

            # Sauvegarde dans l'historique
            self.decision_history.append(measurement)

            return measurement

        except Exception as e:
            self.logger.error(f"Erreur évaluation décision: {e}")
            # Retour conservateur en cas d'erreur
            return UncertaintyMeasurement(
                confidence_score=0.3,
                confidence_level=ConfidenceLevel.LOW,
                risk_level=RiskLevel.HIGH,
                uncertainty_sources=[UncertaintySource.SYSTEM_PERFORMANCE]
            )

    def _assess_context_relevance(self, decision_data: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """Évalue la pertinence du contexte pour la décision"""
        try:
            relevance_factors = []

            # Fraîcheur des données contextuelles
            context_timestamp = context.get('timestamp')
            if context_timestamp:
                if isinstance(context_timestamp, str):
                    context_timestamp = datetime.fromisoformat(context_timestamp)

                age_seconds = (datetime.now() - context_timestamp).total_seconds()
                freshness = max(0.0, 1.0 - age_seconds / 300)  # 5 minutes max
                relevance_factors.append(freshness)

            # Cohérence avec les attentes
            expected_context = decision_data.get('expected_context', {})
            if expected_context:
                matching_keys = set(context.keys()) & set(expected_context.keys())
                if matching_keys:
                    matches = sum(
                        1 for key in matching_keys
                        if context.get(key) == expected_context.get(key)
                    )
                    coherence = matches / len(matching_keys)
                    relevance_factors.append(coherence)

            # Complétude du contexte
            required_fields = decision_data.get('required_context_fields', [])
            if required_fields:
                available_fields = sum(1 for field in required_fields if field in context)
                completeness = available_fields / len(required_fields)
                relevance_factors.append(completeness)

            if relevance_factors:
                return statistics.mean(relevance_factors)
            else:
                return 0.8  # Neutre

        except Exception as e:
            self.logger.warning(f"Erreur évaluation pertinence contexte: {e}")
            return 0.6

    def _identify_uncertainty_sources(self, model_confidence: float,
                                    data_quality: float,
                                    temporal_stability: float,
                                    context_relevance: float,
                                    context: Dict[str, Any]) -> List[UncertaintySource]:
        """Identifie les sources d'incertitude"""
        sources = []

        if model_confidence < 0.6:
            sources.append(UncertaintySource.MODEL_UNCERTAINTY)

        if data_quality < 0.7:
            sources.append(UncertaintySource.SENSOR_NOISE)

        if temporal_stability < 0.6:
            sources.append(UncertaintySource.TEMPORAL_DRIFT)

        if context_relevance < 0.7:
            sources.append(UncertaintySource.INCOMPLETE_INFO)

        # Sources contextuelles
        if context.get('environment_changed', False):
            sources.append(UncertaintySource.ENVIRONMENT_CHANGE)

        if context.get('adversarial_conditions', False):
            sources.append(UncertaintySource.ADVERSARIAL)

        if context.get('system_performance_degraded', False):
            sources.append(UncertaintySource.SYSTEM_PERFORMANCE)

        return sources

    async def should_proceed(self, measurement: UncertaintyMeasurement) -> bool:
        """Détermine si on doit procéder avec une décision incertaine"""
        # Vérification seuil de confiance
        if measurement.confidence_score < self.confidence_threshold:
            self.logger.warning(f"Confiance trop faible: {measurement.confidence_score:.3f}")
            return False

        # Vérification seuil de risque
        if measurement.risk_level.value > self.risk_threshold.value:
            self.logger.warning(f"Risque trop élevé: {measurement.risk_level.name}")
            return False

        return True

    async def create_checkpoint(self, decision_id: str,
                              state_snapshot: Dict[str, Any],
                              action: str,
                              measurement: UncertaintyMeasurement) -> bool:
        """Crée un checkpoint pour rollback potentiel"""
        try:
            if not self.rollback_enabled:
                return True

            checkpoint = DecisionCheckpoint(
                decision_id=decision_id,
                timestamp=datetime.now(),
                state_snapshot=state_snapshot.copy(),
                action_taken=action,
                confidence_at_decision=measurement.confidence_score,
                context=measurement.measurement_context.copy()
            )

            self.checkpoints.append(checkpoint)
            self.logger.debug(f"Checkpoint créé: {decision_id}")

            return True

        except Exception as e:
            self.logger.error(f"Erreur création checkpoint: {e}")
            return False

    async def rollback_to_checkpoint(self, decision_id: Optional[str] = None,
                                   max_age_seconds: float = 60.0) -> Optional[DecisionCheckpoint]:
        """Effectue un rollback vers un checkpoint"""
        try:
            target_checkpoint = None

            if decision_id:
                # Recherche par ID
                target_checkpoint = next(
                    (cp for cp in reversed(self.checkpoints) if cp.decision_id == decision_id),
                    None
                )
            else:
                # Recherche du checkpoint le plus récent valide
                for checkpoint in reversed(self.checkpoints):
                    if checkpoint.age_seconds() <= max_age_seconds:
                        target_checkpoint = checkpoint
                        break

            if target_checkpoint:
                self.logger.info(f"Rollback vers checkpoint: {target_checkpoint.decision_id}")
                return target_checkpoint
            else:
                self.logger.warning("Aucun checkpoint valide trouvé pour rollback")
                return None

        except Exception as e:
            self.logger.error(f"Erreur rollback: {e}")
            return None

    def get_uncertainty_trends(self) -> Dict[str, Any]:
        """Retourne les tendances d'incertitude"""
        try:
            recent_measurements = list(self.decision_history)[-100:]

            if not recent_measurements:
                return {}

            # Calcul des tendances
            confidence_trend = [m.confidence_score for m in recent_measurements]
            risk_trend = [m.risk_level.value for m in recent_measurements]

            # Sources d'incertitude fréquentes
            source_counts = defaultdict(int)
            for measurement in recent_measurements:
                for source in measurement.uncertainty_sources:
                    source_counts[source.name] += 1

            return {
                'average_confidence': statistics.mean(confidence_trend),
                'confidence_trend': confidence_trend[-10:],  # 10 dernières
                'average_risk': statistics.mean(risk_trend),
                'risk_trend': risk_trend[-10:],
                'frequent_uncertainty_sources': dict(source_counts),
                'total_measurements': len(recent_measurements)
            }

        except Exception as e:
            self.logger.error(f"Erreur calcul tendances: {e}")
            return {}

# Fonctions utilitaires

def create_uncertainty_manager(config_path: Optional[str] = None) -> UncertaintyManager:
    """Crée un gestionnaire d'incertitude configuré"""
    config = {}

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f).get('uncertainty', {})
        except Exception as e:
            logger.error(f"Erreur chargement config incertitude: {e}")

    return UncertaintyManager(config)

async def main():
    """Test du système d'incertitude"""
    print("🧪 Test Uncertainty Management System...")

    manager = create_uncertainty_manager()

    # Test de décision simple
    decision_data = {
        'model_output': {'confidence': 0.85, 'action': 'move'},
        'input_data': {'position': (100, 200), 'target': (150, 250)},
        'action': 'move',
        'history': [0.8, 0.82, 0.85, 0.83]
    }

    context = {
        'timestamp': datetime.now(),
        'in_combat': False,
        'players_nearby': 2,
        'hp_percentage': 95
    }

    measurement = await manager.evaluate_decision(decision_data, context)

    print(f"✅ Mesure d'incertitude:")
    print(f"  Confiance: {measurement.confidence_score:.3f} ({measurement.confidence_level.name})")
    print(f"  Risque: {measurement.risk_level.name}")
    print(f"  Fiabilité globale: {measurement.overall_reliability():.3f}")
    print(f"  Sources d'incertitude: {[s.name for s in measurement.uncertainty_sources]}")

    should_proceed = await manager.should_proceed(measurement)
    print(f"  Recommandation: {'Procéder' if should_proceed else 'Arrêter'}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())