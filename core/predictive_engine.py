"""
DOFUS Predictive Analytics Engine
Moteur de prédiction multi-échelle pour l'IA autonome DOFUS
Prédiction des tendances marché, événements serveur et timing optimal
"""

import json
import logging
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum, auto
import pickle
import hashlib
from abc import ABC, abstractmethod

# Import modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.knowledge_graph import DofusKnowledgeGraph, EntityType, RelationType

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types de prédictions disponibles"""
    MARKET_TREND = "market_trend"
    SERVER_EVENT = "server_event"
    OPTIMAL_TIMING = "optimal_timing"
    RESOURCE_AVAILABILITY = "resource_availability"
    PLAYER_BEHAVIOR = "player_behavior"

class TimeWindow(Enum):
    """Fenêtres temporelles pour les prédictions"""
    IMMEDIATE = timedelta(minutes=15)
    SHORT_TERM = timedelta(hours=2)
    MEDIUM_TERM = timedelta(hours=12)
    LONG_TERM = timedelta(days=3)
    STRATEGIC = timedelta(weeks=1)

@dataclass
class PredictionRequest:
    """Requête de prédiction"""
    prediction_type: PredictionType
    target_entity_id: str
    time_window: TimeWindow
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.5

@dataclass
class Prediction:
    """Résultat de prédiction"""
    prediction_id: str
    prediction_type: PredictionType
    target_entity_id: str
    target_name: str

    # Résultats
    predicted_value: Any
    confidence: float
    certainty_range: Tuple[float, float]

    # Métadonnées temporelles
    created_at: datetime
    valid_until: datetime
    time_window: TimeWindow

    # Justification
    reasoning: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    actual_value: Optional[Any] = None
    accuracy_score: Optional[float] = None
    verified_at: Optional[datetime] = None

@dataclass
class MarketForecast:
    """Prévision de marché spécialisée"""
    item_id: str
    item_name: str

    # Prédictions de prix
    current_price: float
    predicted_price: float
    price_change_percent: float
    trend_direction: str  # "rising", "falling", "stable"

    # Facteurs d'influence
    demand_score: float
    supply_score: float
    seasonal_factor: float
    event_impact: float

    # Confiance et validité
    confidence: float
    forecast_horizon: timedelta
    last_updated: datetime

@dataclass
class EventCalendar:
    """Calendrier d'événements prédits"""
    event_type: str
    event_name: str
    predicted_start: datetime
    predicted_duration: timedelta
    probability: float
    impact_zones: List[str]
    recommended_actions: List[str]

@dataclass
class OptimalTimingWindow:
    """Fenêtre de timing optimal pour une action"""
    action_type: str
    optimal_start: datetime
    optimal_end: datetime
    efficiency_score: float
    reasoning: List[str]
    alternative_windows: List[Tuple[datetime, datetime, float]]

class BasePredictionModel(ABC):
    """Modèle de prédiction de base"""

    def __init__(self, name: str, knowledge_graph: DofusKnowledgeGraph):
        self.name = name
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Données d'entraînement et historique
        self.training_data = []
        self.prediction_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'confidence_correlation': 0.0
        }

    @abstractmethod
    async def predict(self, request: PredictionRequest) -> Prediction:
        """Effectue une prédiction"""
        pass

    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Entraîne le modèle"""
        pass

    def update_performance(self, prediction: Prediction, actual_value: Any):
        """Met à jour les métriques de performance"""
        try:
            prediction.actual_value = actual_value
            prediction.verified_at = datetime.now()

            # Calcul de l'accuracy (spécifique au type de prédiction)
            accuracy = self._calculate_accuracy(prediction, actual_value)
            prediction.accuracy_score = accuracy

            # Mise à jour des métriques globales
            self.performance_metrics['total_predictions'] += 1
            if accuracy > 0.7:  # Seuil de "correct"
                self.performance_metrics['correct_predictions'] += 1

            self.performance_metrics['accuracy'] = (
                self.performance_metrics['correct_predictions'] /
                self.performance_metrics['total_predictions']
            )

            self.prediction_history.append(prediction)

        except Exception as e:
            self.logger.error(f"Erreur mise à jour performance: {e}")

    def _calculate_accuracy(self, prediction: Prediction, actual_value: Any) -> float:
        """Calcule l'accuracy d'une prédiction (à implémenter par sous-classe)"""
        return 0.5  # Placeholder

class MarketTrendPredictor(BasePredictionModel):
    """Prédicteur de tendances de marché"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph):
        super().__init__("MarketTrendPredictor", knowledge_graph)

        # Données de marché simulées (à remplacer par vraies données)
        self.price_history = defaultdict(deque)
        self.demand_patterns = {}
        self.seasonal_factors = {}

    async def predict(self, request: PredictionRequest) -> Prediction:
        """Prédit les tendances de marché pour un item"""
        try:
            entity = self.knowledge_graph.get_entity(request.target_entity_id)
            if not entity:
                raise ValueError(f"Entité non trouvée: {request.target_entity_id}")

            # Analyse des données historiques
            historical_data = self._get_historical_data(request.target_entity_id)

            # Calcul de la tendance
            trend_analysis = self._analyze_market_trend(historical_data, request.time_window)

            # Prédiction du prix futur
            predicted_price = self._predict_future_price(
                historical_data,
                request.time_window,
                trend_analysis
            )

            # Calcul de la confiance
            confidence = self._calculate_market_confidence(historical_data, trend_analysis)

            # Création de la prédiction
            prediction = Prediction(
                prediction_id=f"market_{request.target_entity_id}_{int(datetime.now().timestamp())}",
                prediction_type=request.prediction_type,
                target_entity_id=request.target_entity_id,
                target_name=entity.name,
                predicted_value=predicted_price,
                confidence=confidence,
                certainty_range=(predicted_price * 0.8, predicted_price * 1.2),
                created_at=datetime.now(),
                valid_until=datetime.now() + request.time_window.value,
                time_window=request.time_window,
                reasoning=[
                    f"Tendance analysée sur {len(historical_data)} points de données",
                    f"Facteur saisonnier: {trend_analysis.get('seasonal_factor', 1.0):.2f}",
                    f"Volatilité observée: {trend_analysis.get('volatility', 0.1):.2f}"
                ],
                supporting_data={
                    'historical_data': historical_data[-10:],  # Derniers 10 points
                    'trend_analysis': trend_analysis
                }
            )

            return prediction

        except Exception as e:
            self.logger.error(f"Erreur prédiction marché: {e}")
            # Prédiction par défaut en cas d'erreur
            return self._create_fallback_prediction(request)

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Entraîne le modèle avec des données de marché"""
        try:
            for data_point in training_data:
                item_id = data_point.get('item_id')
                price = data_point.get('price')
                timestamp = data_point.get('timestamp', datetime.now())

                if item_id and price:
                    self.price_history[item_id].append({
                        'price': float(price),
                        'timestamp': timestamp
                    })

            self.logger.info(f"Modèle entraîné avec {len(training_data)} points de données")
            return True

        except Exception as e:
            self.logger.error(f"Erreur entraînement modèle marché: {e}")
            return False

    def _get_historical_data(self, item_id: str) -> List[Dict[str, Any]]:
        """Récupère les données historiques d'un item"""
        if item_id in self.price_history:
            return list(self.price_history[item_id])

        # Données simulées si pas d'historique
        base_price = 100
        current_time = datetime.now()

        simulated_data = []
        for i in range(20):  # 20 points de données
            timestamp = current_time - timedelta(hours=i)
            # Prix avec variation aléatoire
            price = base_price * (1 + np.random.normal(0, 0.1))
            simulated_data.append({
                'price': max(10, price),  # Prix minimum de 10
                'timestamp': timestamp
            })

        return simulated_data

    def _analyze_market_trend(self, historical_data: List[Dict[str, Any]],
                            time_window: TimeWindow) -> Dict[str, Any]:
        """Analyse la tendance du marché"""
        if len(historical_data) < 2:
            return {'trend': 'stable', 'volatility': 0.1, 'seasonal_factor': 1.0}

        # Extraction des prix
        prices = [point['price'] for point in historical_data]

        # Calcul de la tendance (simple régression linéaire)
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend_slope = coeffs[0]

        # Détermination de la direction
        if abs(trend_slope) < 0.5:
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'rising'
        else:
            trend_direction = 'falling'

        # Calcul de la volatilité
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1]
                        for i in range(1, len(prices))]
        volatility = np.mean(price_changes) if price_changes else 0.1

        # Facteur saisonnier (simplifié)
        current_hour = datetime.now().hour
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_hour / 24)

        return {
            'trend': trend_direction,
            'slope': trend_slope,
            'volatility': volatility,
            'seasonal_factor': seasonal_factor,
            'r_squared': 0.7  # Simulé
        }

    def _predict_future_price(self, historical_data: List[Dict[str, Any]],
                            time_window: TimeWindow,
                            trend_analysis: Dict[str, Any]) -> float:
        """Prédit le prix futur"""
        if not historical_data:
            return 100.0  # Prix par défaut

        current_price = historical_data[-1]['price']
        trend_slope = trend_analysis.get('slope', 0)
        seasonal_factor = trend_analysis.get('seasonal_factor', 1.0)

        # Projection basée sur la tendance
        hours_ahead = time_window.value.total_seconds() / 3600
        price_change = trend_slope * hours_ahead

        # Application du facteur saisonnier
        predicted_price = (current_price + price_change) * seasonal_factor

        return max(10, predicted_price)  # Prix minimum

    def _calculate_market_confidence(self, historical_data: List[Dict[str, Any]],
                                   trend_analysis: Dict[str, Any]) -> float:
        """Calcule la confiance de la prédiction de marché"""
        base_confidence = 0.6

        # Bonus pour plus de données
        data_bonus = min(0.2, len(historical_data) * 0.01)

        # Malus pour volatilité élevée
        volatility = trend_analysis.get('volatility', 0.1)
        volatility_penalty = min(0.3, volatility * 2)

        # Bonus pour tendance claire
        r_squared = trend_analysis.get('r_squared', 0.5)
        trend_bonus = r_squared * 0.2

        confidence = base_confidence + data_bonus + trend_bonus - volatility_penalty
        return max(0.1, min(0.95, confidence))

    def _create_fallback_prediction(self, request: PredictionRequest) -> Prediction:
        """Crée une prédiction par défaut en cas d'erreur"""
        return Prediction(
            prediction_id=f"fallback_{request.target_entity_id}_{int(datetime.now().timestamp())}",
            prediction_type=request.prediction_type,
            target_entity_id=request.target_entity_id,
            target_name="Unknown Entity",
            predicted_value=100.0,
            confidence=0.3,
            certainty_range=(80.0, 120.0),
            created_at=datetime.now(),
            valid_until=datetime.now() + request.time_window.value,
            time_window=request.time_window,
            reasoning=["Prédiction de fallback - données insuffisantes"],
            supporting_data={}
        )

class ServerEventPredictor(BasePredictionModel):
    """Prédicteur d'événements serveur"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph):
        super().__init__("ServerEventPredictor", knowledge_graph)

        # Patterns d'événements connus
        self.event_patterns = {
            'archmonster_spawn': {
                'frequency': timedelta(hours=6),
                'variance': timedelta(hours=1),
                'zones': ['amakna', 'brakmar', 'bonta']
            },
            'server_maintenance': {
                'frequency': timedelta(days=7),
                'typical_time': 'tuesday_morning',
                'duration': timedelta(hours=2)
            },
            'double_xp_weekend': {
                'frequency': timedelta(days=30),
                'typical_start': 'friday_evening',
                'duration': timedelta(days=2)
            }
        }

        self.event_history = deque(maxlen=500)

    async def predict(self, request: PredictionRequest) -> Prediction:
        """Prédit les événements serveur"""
        try:
            # Analyse des patterns d'événements
            relevant_events = self._find_relevant_events(request.target_entity_id)

            # Prédiction basée sur les patterns
            next_event = self._predict_next_event(relevant_events, request.time_window)

            if next_event:
                confidence = self._calculate_event_confidence(next_event, relevant_events)

                prediction = Prediction(
                    prediction_id=f"event_{request.target_entity_id}_{int(datetime.now().timestamp())}",
                    prediction_type=request.prediction_type,
                    target_entity_id=request.target_entity_id,
                    target_name=next_event['name'],
                    predicted_value=next_event,
                    confidence=confidence,
                    certainty_range=(0.0, 1.0),  # Probabilité d'occurrence
                    created_at=datetime.now(),
                    valid_until=datetime.now() + request.time_window.value,
                    time_window=request.time_window,
                    reasoning=[
                        f"Pattern identifié: {next_event['pattern']}",
                        f"Dernière occurrence: {next_event.get('last_occurrence', 'Inconnue')}",
                        f"Fréquence observée: {next_event.get('frequency', 'Variable')}"
                    ],
                    supporting_data={
                        'event_pattern': next_event['pattern'],
                        'historical_events': relevant_events[-5:]
                    }
                )

                return prediction

            # Aucun événement prédit
            return self._create_no_event_prediction(request)

        except Exception as e:
            self.logger.error(f"Erreur prédiction événement: {e}")
            return self._create_no_event_prediction(request)

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Entraîne le modèle avec l'historique des événements"""
        try:
            for event_data in training_data:
                self.event_history.append({
                    'event_type': event_data.get('event_type'),
                    'timestamp': event_data.get('timestamp', datetime.now()),
                    'duration': event_data.get('duration'),
                    'zone': event_data.get('zone'),
                    'impact': event_data.get('impact', 'medium')
                })

            # Mise à jour des patterns basée sur l'historique
            self._update_patterns_from_history()

            return True

        except Exception as e:
            self.logger.error(f"Erreur entraînement événements: {e}")
            return False

    def _find_relevant_events(self, target_entity_id: str) -> List[Dict[str, Any]]:
        """Trouve les événements pertinents pour l'entité cible"""
        # Pour l'instant, retourne tous les événements
        # TODO: Filtrer par zone/type d'entité
        return list(self.event_history)

    def _predict_next_event(self, historical_events: List[Dict[str, Any]],
                           time_window: TimeWindow) -> Optional[Dict[str, Any]]:
        """Prédit le prochain événement dans la fenêtre temporelle"""
        current_time = datetime.now()
        window_end = current_time + time_window.value

        # Vérification des patterns connus
        for event_type, pattern in self.event_patterns.items():
            last_event = self._find_last_event_of_type(historical_events, event_type)

            if last_event:
                expected_next = last_event['timestamp'] + pattern['frequency']

                # Si l'événement est prévu dans la fenêtre
                if current_time <= expected_next <= window_end:
                    return {
                        'name': event_type.replace('_', ' ').title(),
                        'type': event_type,
                        'predicted_time': expected_next,
                        'pattern': pattern,
                        'last_occurrence': last_event['timestamp']
                    }

        # Simulation d'un événement aléatoire pour démonstration
        if time_window == TimeWindow.MEDIUM_TERM:
            return {
                'name': 'Archmonster Spawn',
                'type': 'archmonster_spawn',
                'predicted_time': current_time + timedelta(hours=4),
                'pattern': self.event_patterns['archmonster_spawn'],
                'probability': 0.6
            }

        return None

    def _find_last_event_of_type(self, events: List[Dict[str, Any]],
                                event_type: str) -> Optional[Dict[str, Any]]:
        """Trouve le dernier événement d'un type donné"""
        for event in reversed(events):
            if event.get('event_type') == event_type:
                return event
        return None

    def _calculate_event_confidence(self, predicted_event: Dict[str, Any],
                                  historical_events: List[Dict[str, Any]]) -> float:
        """Calcule la confiance de la prédiction d'événement"""
        base_confidence = 0.5

        # Bonus pour pattern récurrent
        event_type = predicted_event.get('type')
        type_events = [e for e in historical_events if e.get('event_type') == event_type]
        pattern_bonus = min(0.3, len(type_events) * 0.05)

        # Bonus pour régularité du pattern
        if len(type_events) >= 2:
            intervals = []
            for i in range(1, len(type_events)):
                interval = type_events[i]['timestamp'] - type_events[i-1]['timestamp']
                intervals.append(interval.total_seconds())

            if intervals:
                regularity = 1.0 - (np.std(intervals) / np.mean(intervals))
                regularity_bonus = regularity * 0.2
            else:
                regularity_bonus = 0.0
        else:
            regularity_bonus = 0.0

        confidence = base_confidence + pattern_bonus + regularity_bonus
        return max(0.1, min(0.9, confidence))

    def _update_patterns_from_history(self):
        """Met à jour les patterns basés sur l'historique"""
        # Groupement par type d'événement
        events_by_type = defaultdict(list)
        for event in self.event_history:
            event_type = event.get('event_type')
            if event_type:
                events_by_type[event_type].append(event)

        # Calcul des fréquences observées
        for event_type, events in events_by_type.items():
            if len(events) >= 2:
                intervals = []
                sorted_events = sorted(events, key=lambda x: x['timestamp'])

                for i in range(1, len(sorted_events)):
                    interval = sorted_events[i]['timestamp'] - sorted_events[i-1]['timestamp']
                    intervals.append(interval)

                if intervals:
                    avg_frequency = sum(intervals, timedelta()) / len(intervals)

                    # Mise à jour du pattern si différent
                    if event_type in self.event_patterns:
                        self.event_patterns[event_type]['frequency'] = avg_frequency

    def _create_no_event_prediction(self, request: PredictionRequest) -> Prediction:
        """Crée une prédiction indiquant qu'aucun événement n'est prévu"""
        return Prediction(
            prediction_id=f"no_event_{request.target_entity_id}_{int(datetime.now().timestamp())}",
            prediction_type=request.prediction_type,
            target_entity_id=request.target_entity_id,
            target_name="No Event",
            predicted_value=None,
            confidence=0.7,
            certainty_range=(0.0, 0.0),
            created_at=datetime.now(),
            valid_until=datetime.now() + request.time_window.value,
            time_window=request.time_window,
            reasoning=["Aucun événement prévu dans la fenêtre temporelle"],
            supporting_data={}
        )

class OptimalTimingPredictor(BasePredictionModel):
    """Prédicteur de timing optimal pour les actions"""

    def __init__(self, knowledge_graph: DofusKnowledgeGraph):
        super().__init__("OptimalTimingPredictor", knowledge_graph)

        # Patterns de timing pour différentes activités
        self.activity_patterns = {
            'farming': {
                'peak_hours': [8, 12, 20],  # Heures de pic d'activité
                'low_competition_hours': [2, 5, 14],
                'resource_respawn_rate': timedelta(minutes=30)
            },
            'trading': {
                'peak_hours': [19, 20, 21],  # Heures de pic du marché
                'best_selling_hours': [18, 19, 20, 21, 22],
                'best_buying_hours': [8, 9, 10, 14, 15]
            },
            'hunting': {
                'low_competition_hours': [6, 7, 13, 14],
                'spawn_cycles': timedelta(minutes=45)
            }
        }

    async def predict(self, request: PredictionRequest) -> Prediction:
        """Prédit le timing optimal pour une action"""
        try:
            # Extraction du type d'activité depuis le contexte
            activity_type = request.context.get('activity_type', 'general')

            # Calcul de la fenêtre optimale
            optimal_window = self._calculate_optimal_timing(
                activity_type,
                request.time_window,
                request.context
            )

            confidence = self._calculate_timing_confidence(activity_type, optimal_window)

            prediction = Prediction(
                prediction_id=f"timing_{request.target_entity_id}_{int(datetime.now().timestamp())}",
                prediction_type=request.prediction_type,
                target_entity_id=request.target_entity_id,
                target_name=f"Optimal Timing for {activity_type}",
                predicted_value=optimal_window,
                confidence=confidence,
                certainty_range=(0.0, 1.0),
                created_at=datetime.now(),
                valid_until=datetime.now() + request.time_window.value,
                time_window=request.time_window,
                reasoning=[
                    f"Activité: {activity_type}",
                    f"Fenêtre optimale: {optimal_window['optimal_start']} - {optimal_window['optimal_end']}",
                    f"Score d'efficacité: {optimal_window['efficiency_score']:.2f}"
                ],
                supporting_data={
                    'activity_patterns': self.activity_patterns.get(activity_type, {}),
                    'current_hour': datetime.now().hour
                }
            )

            return prediction

        except Exception as e:
            self.logger.error(f"Erreur prédiction timing: {e}")
            return self._create_default_timing_prediction(request)

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Entraîne le modèle avec des données de performance par timing"""
        try:
            # Groupement par activité et heure
            performance_by_hour = defaultdict(list)

            for data_point in training_data:
                activity = data_point.get('activity_type')
                hour = data_point.get('hour')
                performance = data_point.get('performance_score', 0.5)

                if activity and hour is not None:
                    performance_by_hour[(activity, hour)].append(performance)

            # Mise à jour des patterns
            for (activity, hour), performances in performance_by_hour.items():
                avg_performance = np.mean(performances)

                if activity not in self.activity_patterns:
                    self.activity_patterns[activity] = {}

                if 'hourly_performance' not in self.activity_patterns[activity]:
                    self.activity_patterns[activity]['hourly_performance'] = {}

                self.activity_patterns[activity]['hourly_performance'][hour] = avg_performance

            return True

        except Exception as e:
            self.logger.error(f"Erreur entraînement timing: {e}")
            return False

    def _calculate_optimal_timing(self, activity_type: str, time_window: TimeWindow,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule le timing optimal pour une activité"""
        current_time = datetime.now()
        window_end = current_time + time_window.value

        patterns = self.activity_patterns.get(activity_type, {})

        # Recherche de la meilleure heure dans la fenêtre
        best_hour = current_time.hour
        best_score = 0.5

        # Évaluation de chaque heure dans la fenêtre
        for hours_ahead in range(0, int(time_window.value.total_seconds() // 3600) + 1):
            check_time = current_time + timedelta(hours=hours_ahead)
            if check_time > window_end:
                break

            hour_score = self._evaluate_hour_performance(activity_type, check_time.hour, patterns)

            if hour_score > best_score:
                best_score = hour_score
                best_hour = check_time.hour

        # Calcul de la fenêtre optimale
        optimal_start = current_time.replace(hour=best_hour, minute=0, second=0, microsecond=0)
        if optimal_start < current_time:
            optimal_start += timedelta(days=1)

        optimal_end = optimal_start + timedelta(hours=2)  # Fenêtre de 2 heures

        return {
            'optimal_start': optimal_start,
            'optimal_end': optimal_end,
            'efficiency_score': best_score,
            'reasoning': [
                f"Heure optimale identifiée: {best_hour}h",
                f"Score d'efficacité: {best_score:.2f}",
                f"Basé sur les patterns de {activity_type}"
            ],
            'alternative_windows': self._find_alternative_windows(
                activity_type, current_time, window_end, patterns
            )
        }

    def _evaluate_hour_performance(self, activity_type: str, hour: int,
                                 patterns: Dict[str, Any]) -> float:
        """Évalue la performance d'une heure pour une activité"""
        base_score = 0.5

        # Bonus pour heures de pic
        peak_hours = patterns.get('peak_hours', [])
        if hour in peak_hours:
            base_score += 0.2

        # Bonus pour heures de faible compétition
        low_competition = patterns.get('low_competition_hours', [])
        if hour in low_competition:
            base_score += 0.3

        # Performance historique si disponible
        hourly_perf = patterns.get('hourly_performance', {})
        if hour in hourly_perf:
            base_score = (base_score + hourly_perf[hour]) / 2

        # Malus pour heures creuses générales (3h-6h)
        if 3 <= hour <= 6:
            base_score -= 0.1

        return max(0.1, min(1.0, base_score))

    def _find_alternative_windows(self, activity_type: str, start_time: datetime,
                                end_time: datetime, patterns: Dict[str, Any]) -> List[Tuple[datetime, datetime, float]]:
        """Trouve des fenêtres alternatives de timing"""
        alternatives = []

        for hours_ahead in range(0, int((end_time - start_time).total_seconds() // 3600) + 1, 2):
            alt_start = start_time + timedelta(hours=hours_ahead)
            if alt_start >= end_time:
                break

            alt_end = alt_start + timedelta(hours=1)
            score = self._evaluate_hour_performance(activity_type, alt_start.hour, patterns)

            alternatives.append((alt_start, alt_end, score))

        # Tri par score décroissant
        alternatives.sort(key=lambda x: x[2], reverse=True)

        return alternatives[:3]  # Top 3 alternatives

    def _calculate_timing_confidence(self, activity_type: str, optimal_window: Dict[str, Any]) -> float:
        """Calcule la confiance de la prédiction de timing"""
        base_confidence = 0.6

        # Bonus si on a des données spécifiques à l'activité
        if activity_type in self.activity_patterns:
            base_confidence += 0.2

        # Bonus basé sur le score d'efficacité
        efficiency_score = optimal_window.get('efficiency_score', 0.5)
        efficiency_bonus = (efficiency_score - 0.5) * 0.4

        # Bonus si on a des données de performance historique
        patterns = self.activity_patterns.get(activity_type, {})
        if 'hourly_performance' in patterns:
            base_confidence += 0.1

        confidence = base_confidence + efficiency_bonus
        return max(0.2, min(0.9, confidence))

    def _create_default_timing_prediction(self, request: PredictionRequest) -> Prediction:
        """Crée une prédiction de timing par défaut"""
        current_time = datetime.now()

        return Prediction(
            prediction_id=f"default_timing_{request.target_entity_id}_{int(datetime.now().timestamp())}",
            prediction_type=request.prediction_type,
            target_entity_id=request.target_entity_id,
            target_name="Default Timing",
            predicted_value={
                'optimal_start': current_time + timedelta(hours=1),
                'optimal_end': current_time + timedelta(hours=3),
                'efficiency_score': 0.5,
                'reasoning': ["Timing par défaut - données insuffisantes"]
            },
            confidence=0.3,
            certainty_range=(0.0, 1.0),
            created_at=datetime.now(),
            valid_until=datetime.now() + request.time_window.value,
            time_window=request.time_window,
            reasoning=["Prédiction par défaut faute de données spécifiques"],
            supporting_data={}
        )

class PredictiveAnalyticsEngine:
    """
    Moteur de prédiction principal intégrant tous les modèles spécialisés
    Point d'entrée unique pour toutes les prédictions
    """

    def __init__(self, knowledge_graph: DofusKnowledgeGraph, data_path: str = "data/predictions"):
        self.knowledge_graph = knowledge_graph
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Modèles de prédiction spécialisés
        self.predictors = {
            PredictionType.MARKET_TREND: MarketTrendPredictor(knowledge_graph),
            PredictionType.SERVER_EVENT: ServerEventPredictor(knowledge_graph),
            PredictionType.OPTIMAL_TIMING: OptimalTimingPredictor(knowledge_graph)
        }

        # Cache et historique
        self.prediction_cache = {}
        self.prediction_history = deque(maxlen=1000)

        # Statistiques globales
        self.stats = {
            'total_predictions': 0,
            'predictions_by_type': defaultdict(int),
            'average_confidence': 0.0,
            'accuracy_by_type': defaultdict(float),
            'last_updated': datetime.now()
        }

    async def predict(self, request: PredictionRequest) -> Prediction:
        """Point d'entrée principal pour les prédictions"""
        try:
            # Validation de la requête
            if not await self._validate_request(request):
                raise ValueError("Requête de prédiction invalide")

            # Vérification du cache
            cache_key = self._generate_cache_key(request)
            if cache_key in self.prediction_cache:
                cached_prediction = self.prediction_cache[cache_key]
                if cached_prediction.valid_until > datetime.now():
                    self.logger.debug(f"Prédiction trouvée en cache: {cache_key}")
                    return cached_prediction

            # Sélection du modèle approprié
            predictor = self.predictors.get(request.prediction_type)
            if not predictor:
                raise ValueError(f"Type de prédiction non supporté: {request.prediction_type}")

            # Exécution de la prédiction
            prediction = await predictor.predict(request)

            # Mise en cache si confiance suffisante
            if prediction.confidence >= request.confidence_threshold:
                self.prediction_cache[cache_key] = prediction

            # Mise à jour des statistiques
            self._update_stats(prediction)

            # Sauvegarde dans l'historique
            self.prediction_history.append(prediction)

            self.logger.info(f"Prédiction {prediction.prediction_type.value} créée avec confiance {prediction.confidence:.2f}")

            return prediction

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {e}")
            raise

    async def predict_market_forecast(self, item_id: str,
                                    time_horizon: TimeWindow = TimeWindow.MEDIUM_TERM) -> MarketForecast:
        """Génère une prévision de marché complète pour un item"""
        try:
            # Requête de prédiction
            request = PredictionRequest(
                prediction_type=PredictionType.MARKET_TREND,
                target_entity_id=item_id,
                time_window=time_horizon
            )

            prediction = await self.predict(request)

            # Conversion en MarketForecast
            predicted_price = prediction.predicted_value
            entity = self.knowledge_graph.get_entity(item_id)

            if not entity:
                raise ValueError(f"Item non trouvé: {item_id}")

            current_price = entity.value or 100.0
            price_change = ((predicted_price - current_price) / current_price) * 100

            # Détermination de la direction de tendance
            if abs(price_change) < 5:
                trend_direction = "stable"
            elif price_change > 0:
                trend_direction = "rising"
            else:
                trend_direction = "falling"

            forecast = MarketForecast(
                item_id=item_id,
                item_name=entity.name,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change_percent=price_change,
                trend_direction=trend_direction,
                demand_score=0.7,  # Simulé
                supply_score=0.6,  # Simulé
                seasonal_factor=1.0,  # Simulé
                event_impact=0.0,  # Simulé
                confidence=prediction.confidence,
                forecast_horizon=time_horizon.value,
                last_updated=datetime.now()
            )

            return forecast

        except Exception as e:
            self.logger.error(f"Erreur génération forecast: {e}")
            raise

    async def predict_server_events(self, time_window: TimeWindow = TimeWindow.LONG_TERM) -> List[EventCalendar]:
        """Prédit les événements serveur dans une fenêtre temporelle"""
        try:
            # Prédiction pour différents types d'événements
            event_types = ['archmonster_spawn', 'server_maintenance', 'double_xp_weekend']
            events = []

            for event_type in event_types:
                request = PredictionRequest(
                    prediction_type=PredictionType.SERVER_EVENT,
                    target_entity_id=event_type,
                    time_window=time_window
                )

                prediction = await self.predict(request)

                if prediction.predicted_value:
                    event_data = prediction.predicted_value

                    event_calendar = EventCalendar(
                        event_type=event_type,
                        event_name=event_data.get('name', event_type),
                        predicted_start=event_data.get('predicted_time', datetime.now()),
                        predicted_duration=timedelta(hours=2),  # Durée par défaut
                        probability=prediction.confidence,
                        impact_zones=['all'],  # Simulé
                        recommended_actions=[
                            f"Préparer pour {event_data.get('name', event_type)}",
                            "Surveiller les annonces officielles"
                        ]
                    )

                    events.append(event_calendar)

            return events

        except Exception as e:
            self.logger.error(f"Erreur prédiction événements: {e}")
            return []

    async def find_optimal_timing(self, activity_type: str,
                                time_window: TimeWindow = TimeWindow.SHORT_TERM) -> OptimalTimingWindow:
        """Trouve le timing optimal pour une activité"""
        try:
            request = PredictionRequest(
                prediction_type=PredictionType.OPTIMAL_TIMING,
                target_entity_id=activity_type,
                time_window=time_window,
                context={'activity_type': activity_type}
            )

            prediction = await self.predict(request)
            timing_data = prediction.predicted_value

            optimal_timing = OptimalTimingWindow(
                action_type=activity_type,
                optimal_start=timing_data['optimal_start'],
                optimal_end=timing_data['optimal_end'],
                efficiency_score=timing_data['efficiency_score'],
                reasoning=timing_data['reasoning'],
                alternative_windows=timing_data.get('alternative_windows', [])
            )

            return optimal_timing

        except Exception as e:
            self.logger.error(f"Erreur recherche timing optimal: {e}")
            raise

    async def train_all_models(self, training_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, bool]:
        """Entraîne tous les modèles de prédiction"""
        results = {}

        for prediction_type, predictor in self.predictors.items():
            type_name = prediction_type.value
            if type_name in training_data:
                try:
                    success = await predictor.train(training_data[type_name])
                    results[type_name] = success
                    self.logger.info(f"Entraînement {type_name}: {'✅' if success else '❌'}")
                except Exception as e:
                    self.logger.error(f"Erreur entraînement {type_name}: {e}")
                    results[type_name] = False

        return results

    async def verify_prediction(self, prediction_id: str, actual_value: Any) -> bool:
        """Vérifie une prédiction avec la valeur réelle observée"""
        try:
            # Recherche de la prédiction dans l'historique
            prediction = None
            for p in self.prediction_history:
                if p.prediction_id == prediction_id:
                    prediction = p
                    break

            if not prediction:
                self.logger.warning(f"Prédiction non trouvée: {prediction_id}")
                return False

            # Mise à jour du modèle correspondant
            predictor = self.predictors.get(prediction.prediction_type)
            if predictor:
                predictor.update_performance(prediction, actual_value)
                self.logger.info(f"Prédiction {prediction_id} vérifiée avec accuracy {prediction.accuracy_score:.2f}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Erreur vérification prédiction: {e}")
            return False

    async def _validate_request(self, request: PredictionRequest) -> bool:
        """Valide une requête de prédiction"""
        if not request.target_entity_id:
            return False

        if request.confidence_threshold < 0 or request.confidence_threshold > 1:
            return False

        # Vérification de l'existence de l'entité (sauf pour timing)
        if request.prediction_type != PredictionType.OPTIMAL_TIMING:
            entity = self.knowledge_graph.get_entity(request.target_entity_id)
            if not entity:
                # Permettre les IDs génériques pour certains types
                if request.prediction_type == PredictionType.SERVER_EVENT:
                    return True
                return False

        return True

    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Génère une clé de cache pour une requête"""
        key_data = f"{request.prediction_type.value}_{request.target_entity_id}_{request.time_window.name}"
        if request.context:
            key_data += f"_{hash(str(sorted(request.context.items())))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_stats(self, prediction: Prediction):
        """Met à jour les statistiques globales"""
        self.stats['total_predictions'] += 1
        self.stats['predictions_by_type'][prediction.prediction_type.value] += 1

        # Moyenne pondérée de la confiance
        total_predictions = self.stats['total_predictions']
        current_avg = self.stats['average_confidence']
        new_avg = ((current_avg * (total_predictions - 1)) + prediction.confidence) / total_predictions
        self.stats['average_confidence'] = new_avg

        self.stats['last_updated'] = datetime.now()

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du moteur de prédiction"""
        return self.stats.copy()

    def clear_cache(self):
        """Vide le cache de prédictions"""
        self.prediction_cache.clear()
        self.logger.info("Cache de prédictions vidé")

# Interface utilitaire
async def create_predictive_engine(knowledge_graph: DofusKnowledgeGraph) -> PredictiveAnalyticsEngine:
    """Crée et initialise le moteur de prédiction"""
    engine = PredictiveAnalyticsEngine(knowledge_graph)

    # Données d'entraînement simulées pour démarrage
    sample_training_data = {
        'market_trend': [
            {'item_id': 'item_wheat', 'price': 95, 'timestamp': datetime.now() - timedelta(hours=i)}
            for i in range(1, 25)
        ],
        'server_event': [
            {'event_type': 'archmonster_spawn', 'timestamp': datetime.now() - timedelta(hours=i*6)}
            for i in range(1, 10)
        ],
        'optimal_timing': [
            {'activity_type': 'farming', 'hour': h, 'performance_score': 0.6 + 0.3 * np.sin(h * np.pi / 12)}
            for h in range(24)
        ]
    }

    # Entraînement initial
    await engine.train_all_models(sample_training_data)

    return engine

# Interface CLI pour tests
async def main():
    """Test du Predictive Analytics Engine"""
    print("🔮 Test DOFUS Predictive Analytics Engine...")

    # Import du knowledge graph
    from core.knowledge_graph import create_dofus_knowledge_graph

    # Création du graphe de connaissances
    knowledge_graph = await create_dofus_knowledge_graph()

    # Création du moteur de prédiction
    engine = await create_predictive_engine(knowledge_graph)

    print(f"✅ Moteur de prédiction initialisé avec {len(engine.predictors)} modèles")

    # Test prédiction de marché
    print("\n🔄 Test prédiction de marché...")
    try:
        wheat_entities = knowledge_graph.find_entities_by_name("blé")
        if wheat_entities:
            forecast = await engine.predict_market_forecast(
                wheat_entities[0].id,
                TimeWindow.MEDIUM_TERM
            )
            print(f"📈 Prévision marché pour {forecast.item_name}:")
            print(f"  - Prix actuel: {forecast.current_price:.2f}")
            print(f"  - Prix prédit: {forecast.predicted_price:.2f}")
            print(f"  - Variation: {forecast.price_change_percent:+.1f}%")
            print(f"  - Tendance: {forecast.trend_direction}")
            print(f"  - Confiance: {forecast.confidence:.2f}")
    except Exception as e:
        print(f"❌ Erreur test marché: {e}")

    # Test prédiction d'événements
    print("\n🔄 Test prédiction d'événements...")
    try:
        events = await engine.predict_server_events(TimeWindow.LONG_TERM)
        print(f"📅 {len(events)} événements prédits:")
        for event in events:
            print(f"  - {event.event_name} (probabilité: {event.probability:.2f})")
            print(f"    Début prévu: {event.predicted_start.strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        print(f"❌ Erreur test événements: {e}")

    # Test timing optimal
    print("\n🔄 Test timing optimal...")
    try:
        optimal_timing = await engine.find_optimal_timing(
            'farming',
            TimeWindow.SHORT_TERM
        )
        print(f"⏰ Timing optimal pour farming:")
        print(f"  - Début optimal: {optimal_timing.optimal_start.strftime('%Y-%m-%d %H:%M')}")
        print(f"  - Fin optimale: {optimal_timing.optimal_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"  - Score d'efficacité: {optimal_timing.efficiency_score:.2f}")
        print(f"  - Alternatives: {len(optimal_timing.alternative_windows)}")
    except Exception as e:
        print(f"❌ Erreur test timing: {e}")

    # Statistiques
    print("\n📊 Statistiques du moteur:")
    stats = engine.get_statistics()
    print(f"  - Total prédictions: {stats['total_predictions']}")
    print(f"  - Confiance moyenne: {stats['average_confidence']:.2f}")
    print(f"  - Dernière mise à jour: {stats['last_updated'].strftime('%H:%M:%S')}")

    print("✅ Test Predictive Analytics Engine terminé !")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())