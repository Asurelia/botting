"""
Module de prédiction de respawn des ressources DOFUS.
Utilise des modèles ML pour prédire quand et où les ressources vont réapparaître.
"""

import logging
import time
import math
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Tentative d'importation des modules ML avec fallback
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn non disponible, utilisation du modèle simplifié")


class ResourceType(Enum):
    """Types de ressources."""
    CEREALS = "cereals"         # Céréales
    PLANTS = "plants"          # Plantes
    TREES = "trees"            # Arbres
    ORES = "ores"              # Minerais
    FISH = "fish"              # Poissons
    MONSTER_DROPS = "monster_drops"  # Drops de monstres


class PredictionModel(Enum):
    """Modèles de prédiction disponibles."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SIMPLE_REGRESSION = "simple_regression"  # Fallback


@dataclass
class ResourceSpot:
    """Représente un spot de ressource."""
    spot_id: str
    resource_name: str
    resource_type: ResourceType
    map_id: str
    position: Tuple[int, int]
    base_respawn_time: int      # Temps de base en secondes
    last_harvested: Optional[datetime] = None
    harvest_count: int = 0
    quality_bonus: float = 1.0  # Bonus de qualité du spot


@dataclass
class HarvestEvent:
    """Événement de récolte historique."""
    spot_id: str
    timestamp: datetime
    player_count: int           # Nombre de joueurs dans la zone
    resource_quantity: int      # Quantité récoltée
    time_since_last: int       # Temps depuis dernière récolte (secondes)
    hour_of_day: int
    day_of_week: int
    server_population: int = 100  # Population serveur estimée


@dataclass
class PredictionResult:
    """Résultat d'une prédiction."""
    spot_id: str
    resource_name: str
    predicted_respawn_time: datetime
    confidence: float           # Confiance (0-1)
    expected_quantity: int      # Quantité attendue
    competition_level: float    # Niveau de compétition prévu (0-10)
    prediction_model: str       # Modèle utilisé
    factors: Dict[str, float]   # Facteurs influençant la prédiction
    reliability_score: float    # Fiabilité basée sur l'historique


class ResourcePredictor:
    """
    Prédicteur principal de respawn des ressources.
    Utilise des modèles ML pour optimiser les routes de farming.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Modèles ML
        self.models = {}
        self.scalers = {}
        self.model_performances = {}
        
        # Base de données des ressources et historique
        self.resource_spots = {}
        self.harvest_history = []
        self.server_patterns = {}
        
        # Configuration
        self.default_model = PredictionModel.SIMPLE_REGRESSION if not SKLEARN_AVAILABLE else PredictionModel.RANDOM_FOREST
        self.history_retention_days = config.get('history_retention_days', 30)
        self.min_samples_for_ml = config.get('min_samples_for_ml', 100)
        self.update_interval = config.get('update_interval', 300)  # 5 minutes
        
        # Facteurs de prédiction
        self.prediction_factors = {
            'time_since_harvest': 0.4,
            'player_density': 0.2,
            'hour_of_day': 0.15,
            'day_of_week': 0.1,
            'server_population': 0.1,
            'historical_pattern': 0.05
        }
        
        self._initialize_models()
        self._load_resource_data()
    
    def _initialize_models(self):
        """Initialise les modèles ML."""
        try:
            if SKLEARN_AVAILABLE:
                self.models[PredictionModel.RANDOM_FOREST] = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
                self.models[PredictionModel.GRADIENT_BOOSTING] = GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                )
                self.models[PredictionModel.NEURAL_NETWORK] = MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
                )
                
                # Scalers pour les modèles
                for model_type in [PredictionModel.RANDOM_FOREST, 
                                 PredictionModel.GRADIENT_BOOSTING, 
                                 PredictionModel.NEURAL_NETWORK]:
                    self.scalers[model_type] = StandardScaler()
            
            self.logger.info(f"Modèles initialisés: {list(self.models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation modèles: {e}")
    
    def _load_resource_data(self):
        """Charge les données de ressources."""
        try:
            # Spots de ressources exemple
            example_spots = [
                ResourceSpot("cereals_001", "Blé", ResourceType.CEREALS, "map_001", (100, 200), 180),
                ResourceSpot("cereals_002", "Orge", ResourceType.CEREALS, "map_001", (150, 250), 200),
                ResourceSpot("plants_001", "Ortie", ResourceType.PLANTS, "map_002", (300, 400), 240),
                ResourceSpot("trees_001", "Frêne", ResourceType.TREES, "map_003", (500, 600), 300),
                ResourceSpot("ores_001", "Fer", ResourceType.ORES, "map_004", (700, 800), 420)
            ]
            
            for spot in example_spots:
                self.resource_spots[spot.spot_id] = spot
            
            self.logger.info(f"Spots de ressources chargés: {len(self.resource_spots)}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement données ressources: {e}")
    
    def predict_respawn_times(self, 
                            resource_type: Optional[ResourceType] = None,
                            map_ids: Optional[List[str]] = None,
                            hours_ahead: int = 2) -> List[PredictionResult]:
        """
        Prédit les temps de respawn des ressources.
        
        Args:
            resource_type: Type de ressource à prédire (optionnel)
            map_ids: IDs des cartes à analyser (optionnel)
            hours_ahead: Heures à prédire à l'avance
            
        Returns:
            Liste des prédictions triées par temps de respawn
        """
        try:
            predictions = []
            target_spots = self._filter_spots(resource_type, map_ids)
            
            for spot_id, spot in target_spots.items():
                prediction = self._predict_spot_respawn(spot, hours_ahead)
                if prediction:
                    predictions.append(prediction)
            
            # Trier par temps de respawn prévu
            predictions.sort(key=lambda x: x.predicted_respawn_time)
            
            self.logger.info(f"Prédictions générées: {len(predictions)}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction respawns: {e}")
            return []
    
    def _filter_spots(self, 
                     resource_type: Optional[ResourceType],
                     map_ids: Optional[List[str]]) -> Dict[str, ResourceSpot]:
        """Filtre les spots selon les critères."""
        filtered_spots = {}
        
        for spot_id, spot in self.resource_spots.items():
            # Filtrer par type de ressource
            if resource_type and spot.resource_type != resource_type:
                continue
            
            # Filtrer par carte
            if map_ids and spot.map_id not in map_ids:
                continue
            
            filtered_spots[spot_id] = spot
        
        return filtered_spots
    
    def _predict_spot_respawn(self, spot: ResourceSpot, hours_ahead: int) -> Optional[PredictionResult]:
        """Prédit le respawn pour un spot spécifique."""
        try:
            # Obtenir les features pour la prédiction
            features = self._extract_features(spot)
            
            # Choisir le meilleur modèle
            best_model = self._get_best_model_for_spot(spot.spot_id)
            
            # Faire la prédiction
            predicted_seconds = self._make_prediction(features, best_model, spot)
            
            if predicted_seconds <= 0:
                return None
            
            # Calculer le temps de respawn prévu
            base_time = spot.last_harvested or datetime.now()
            predicted_time = base_time + timedelta(seconds=predicted_seconds)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(spot, best_model)
            
            # Calculer la quantité attendue
            expected_quantity = self._predict_quantity(spot, features)
            
            # Calculer le niveau de compétition
            competition_level = self._predict_competition(spot, predicted_time)
            
            # Facteurs influençant la prédiction
            factors = self._analyze_prediction_factors(features, spot)
            
            # Score de fiabilité
            reliability_score = self._calculate_reliability(spot)
            
            return PredictionResult(
                spot_id=spot.spot_id,
                resource_name=spot.resource_name,
                predicted_respawn_time=predicted_time,
                confidence=confidence,
                expected_quantity=expected_quantity,
                competition_level=competition_level,
                prediction_model=best_model.value,
                factors=factors,
                reliability_score=reliability_score
            )
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction spot {spot.spot_id}: {e}")
            return None
    
    def _extract_features(self, spot: ResourceSpot) -> List[float]:
        """Extrait les features pour la prédiction ML."""
        now = datetime.now()
        
        # Temps depuis dernière récolte
        time_since_harvest = 0
        if spot.last_harvested:
            time_since_harvest = (now - spot.last_harvested).total_seconds()
        
        # Heure de la journée (0-23)
        hour_of_day = now.hour
        
        # Jour de la semaine (0-6)
        day_of_week = now.weekday()
        
        # Densité de joueurs estimée (simulation)
        player_density = self._estimate_player_density(spot.map_id, now)
        
        # Population serveur (simulation)
        server_population = self._get_server_population()
        
        # Bonus qualité du spot
        quality_bonus = spot.quality_bonus
        
        # Nombre de récoltes historiques
        harvest_frequency = self._get_harvest_frequency(spot.spot_id)
        
        features = [
            time_since_harvest / 3600,  # Convertir en heures
            hour_of_day / 24,           # Normaliser
            day_of_week / 7,            # Normaliser
            player_density / 100,       # Normaliser
            server_population / 1000,   # Normaliser
            quality_bonus,
            harvest_frequency / 10      # Normaliser
        ]
        
        return features
    
    def _get_best_model_for_spot(self, spot_id: str) -> PredictionModel:
        """Détermine le meilleur modèle pour un spot."""
        # Vérifier si on a assez de données pour ML
        spot_history = [event for event in self.harvest_history 
                       if event.spot_id == spot_id]
        
        if len(spot_history) < self.min_samples_for_ml or not SKLEARN_AVAILABLE:
            return PredictionModel.SIMPLE_REGRESSION
        
        # Retourner le modèle avec les meilleures performances
        best_model = PredictionModel.RANDOM_FOREST
        best_score = float('inf')
        
        for model_type, score in self.model_performances.items():
            if score < best_score:
                best_score = score
                best_model = model_type
        
        return best_model
    
    def _make_prediction(self, features: List[float], model_type: PredictionModel, spot: ResourceSpot) -> float:
        """Fait une prédiction de temps de respawn."""
        try:
            if model_type == PredictionModel.SIMPLE_REGRESSION:
                # Modèle simple basé sur le temps de base et les facteurs
                base_time = spot.base_respawn_time
                
                # Ajustements basés sur les features
                time_adjustment = 1.0
                
                # Ajustement selon l'heure (plus lent la nuit)
                hour_factor = features[1] * 24  # Récupérer l'heure originale
                if 0 <= hour_factor <= 6 or 22 <= hour_factor <= 24:
                    time_adjustment *= 1.2  # 20% plus lent la nuit
                
                # Ajustement selon la densité de joueurs
                player_density = features[3] * 100
                time_adjustment *= (1 + player_density / 200)  # Plus de joueurs = plus lent
                
                predicted_time = base_time * time_adjustment
                return predicted_time
                
            elif model_type in self.models and SKLEARN_AVAILABLE:
                # Utiliser le modèle ML
                model = self.models[model_type]
                scaler = self.scalers[model_type]
                
                # Normaliser les features
                features_scaled = scaler.transform([features])
                
                # Faire la prédiction
                prediction = model.predict(features_scaled)[0]
                
                return max(0, prediction)  # Pas de temps négatif
            
            else:
                # Fallback au modèle simple
                return self._make_prediction(features, PredictionModel.SIMPLE_REGRESSION, spot)
                
        except Exception as e:
            self.logger.error(f"Erreur prédiction modèle {model_type}: {e}")
            return spot.base_respawn_time
    
    def _calculate_confidence(self, spot: ResourceSpot, model_type: PredictionModel) -> float:
        """Calcule la confiance dans la prédiction."""
        base_confidence = 0.7
        
        # Ajuster selon l'historique disponible
        spot_history = [event for event in self.harvest_history 
                       if event.spot_id == spot.spot_id]
        
        if len(spot_history) >= 50:
            base_confidence += 0.2
        elif len(spot_history) >= 20:
            base_confidence += 0.1
        elif len(spot_history) < 5:
            base_confidence -= 0.2
        
        # Ajuster selon le type de modèle
        if model_type == PredictionModel.SIMPLE_REGRESSION:
            base_confidence -= 0.1
        elif model_type in [PredictionModel.RANDOM_FOREST, PredictionModel.GRADIENT_BOOSTING]:
            base_confidence += 0.1
        
        # Ajuster selon les performances du modèle
        if model_type in self.model_performances:
            performance = self.model_performances[model_type]
            if performance < 100:  # Bonne performance (erreur faible)
                base_confidence += 0.1
            elif performance > 300:  # Mauvaise performance
                base_confidence -= 0.1
        
        return min(1.0, max(0.1, base_confidence))
    
    def _predict_quantity(self, spot: ResourceSpot, features: List[float]) -> int:
        """Prédit la quantité de ressource disponible."""
        base_quantity = 1
        
        # Ajustement selon la qualité du spot
        quantity_multiplier = spot.quality_bonus
        
        # Ajustement selon l'heure (plus de ressources le matin)
        hour_factor = features[1] * 24
        if 6 <= hour_factor <= 10:
            quantity_multiplier *= 1.3
        elif 18 <= hour_factor <= 22:
            quantity_multiplier *= 0.8
        
        # Ajustement selon la densité de joueurs (moins si beaucoup de concurrence)
        player_density = features[3] * 100
        quantity_multiplier *= max(0.5, 1 - player_density / 150)
        
        predicted_quantity = int(base_quantity * quantity_multiplier)
        return max(1, predicted_quantity)
    
    def _predict_competition(self, spot: ResourceSpot, predicted_time: datetime) -> float:
        """Prédit le niveau de compétition."""
        base_competition = 5.0  # Niveau moyen
        
        # Ajuster selon l'heure
        hour = predicted_time.hour
        if 18 <= hour <= 22:  # Prime time
            base_competition += 2.0
        elif 2 <= hour <= 6:   # Nuit
            base_competition -= 2.0
        
        # Ajuster selon le type de ressource
        if spot.resource_type in [ResourceType.ORES, ResourceType.TREES]:
            base_competition += 1.0  # Plus prisées
        elif spot.resource_type == ResourceType.CEREALS:
            base_competition -= 0.5  # Moins prisées
        
        return min(10.0, max(0.0, base_competition))
    
    def _analyze_prediction_factors(self, features: List[float], spot: ResourceSpot) -> Dict[str, float]:
        """Analyse les facteurs influençant la prédiction."""
        factors = {}
        
        # Temps depuis récolte
        time_since_harvest = features[0] * 3600  # Reconvertir en secondes
        if time_since_harvest > spot.base_respawn_time:
            factors['overdue_respawn'] = min(1.0, time_since_harvest / spot.base_respawn_time - 1)
        
        # Heure de la journée
        hour = features[1] * 24
        if hour in [19, 20, 21]:  # Prime time
            factors['prime_time_bonus'] = 0.8
        elif hour in [3, 4, 5]:   # Nuit calme
            factors['night_penalty'] = -0.3
        
        # Densité de joueurs
        player_density = features[3] * 100
        if player_density > 50:
            factors['high_competition'] = -0.5
        elif player_density < 20:
            factors['low_competition'] = 0.3
        
        return factors
    
    def _calculate_reliability(self, spot: ResourceSpot) -> float:
        """Calcule un score de fiabilité basé sur l'historique."""
        spot_events = [event for event in self.harvest_history 
                      if event.spot_id == spot.spot_id]
        
        if len(spot_events) < 5:
            return 0.3  # Faible fiabilité si peu de données
        
        # Calculer la régularité des respawns
        respawn_times = []
        for i in range(1, len(spot_events)):
            time_diff = (spot_events[i].timestamp - spot_events[i-1].timestamp).total_seconds()
            respawn_times.append(time_diff)
        
        if not respawn_times:
            return 0.5
        
        # Calculer la variance
        mean_time = sum(respawn_times) / len(respawn_times)
        variance = sum((t - mean_time) ** 2 for t in respawn_times) / len(respawn_times)
        coefficient_of_variation = math.sqrt(variance) / mean_time if mean_time > 0 else 1.0
        
        # Score de fiabilité inversement proportionnel à la variation
        reliability = max(0.1, 1.0 - coefficient_of_variation)
        
        return min(1.0, reliability)
    
    def _estimate_player_density(self, map_id: str, timestamp: datetime) -> float:
        """Estime la densité de joueurs sur une carte."""
        # Simulation basique
        base_density = 30  # Densité de base
        
        # Variation selon l'heure
        hour = timestamp.hour
        if 18 <= hour <= 22:  # Prime time
            base_density *= 1.5
        elif 2 <= hour <= 6:   # Nuit
            base_density *= 0.3
        
        return base_density
    
    def _get_server_population(self) -> int:
        """Obtient la population serveur estimée."""
        # Simulation - pourrait être connecté à l'API du serveur
        return 800  # Population moyenne
    
    def _get_harvest_frequency(self, spot_id: str) -> float:
        """Calcule la fréquence de récolte pour un spot."""
        spot_events = [event for event in self.harvest_history 
                      if event.spot_id == spot_id]
        
        if len(spot_events) < 2:
            return 1.0  # Fréquence par défaut
        
        # Calculer les récoltes par jour
        time_span = (spot_events[-1].timestamp - spot_events[0].timestamp).total_seconds()
        if time_span <= 0:
            return 1.0
        
        harvests_per_day = len(spot_events) / (time_span / 86400)  # 86400 = secondes par jour
        
        return harvests_per_day
    
    def add_harvest_event(self, event: HarvestEvent):
        """Ajoute un événement de récolte à l'historique."""
        try:
            self.harvest_history.append(event)
            
            # Mettre à jour le spot
            if event.spot_id in self.resource_spots:
                spot = self.resource_spots[event.spot_id]
                spot.last_harvested = event.timestamp
                spot.harvest_count += 1
            
            # Nettoyer l'historique ancien
            cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
            self.harvest_history = [e for e in self.harvest_history 
                                  if e.timestamp > cutoff_date]
            
            self.logger.debug(f"Événement de récolte ajouté: {event.spot_id}")
            
        except Exception as e:
            self.logger.error(f"Erreur ajout événement récolte: {e}")
    
    def train_models(self) -> Dict[PredictionModel, float]:
        """Entraîne les modèles ML avec l'historique disponible."""
        try:
            if not SKLEARN_AVAILABLE or len(self.harvest_history) < self.min_samples_for_ml:
                self.logger.warning("Pas assez de données pour entraînement ML")
                return {}
            
            # Préparer les données d'entraînement
            X, y = self._prepare_training_data()
            
            if len(X) < 10:
                self.logger.warning("Données d'entraînement insuffisantes")
                return {}
            
            # Séparer train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            performances = {}
            
            # Entraîner chaque modèle
            for model_type, model in self.models.items():
                try:
                    # Normaliser les données
                    scaler = self.scalers[model_type]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Entraîner
                    model.fit(X_train_scaled, y_train)
                    
                    # Évaluer
                    predictions = model.predict(X_test_scaled)
                    mae = mean_absolute_error(y_test, predictions)
                    
                    performances[model_type] = mae
                    self.model_performances[model_type] = mae
                    
                    self.logger.info(f"Modèle {model_type.value} entraîné - MAE: {mae:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur entraînement {model_type.value}: {e}")
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement modèles: {e}")
            return {}
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prépare les données pour l'entraînement ML."""
        X = []
        y = []
        
        # Grouper par spot et calculer les temps entre récoltes
        spots_events = {}
        for event in self.harvest_history:
            if event.spot_id not in spots_events:
                spots_events[event.spot_id] = []
            spots_events[event.spot_id].append(event)
        
        # Pour chaque spot, créer des échantillons d'entraînement
        for spot_id, events in spots_events.items():
            if spot_id not in self.resource_spots:
                continue
                
            spot = self.resource_spots[spot_id]
            events.sort(key=lambda x: x.timestamp)
            
            for i in range(1, len(events)):
                prev_event = events[i-1]
                curr_event = events[i]
                
                # Calculer le temps de respawn réel
                respawn_time = (curr_event.timestamp - prev_event.timestamp).total_seconds()
                
                # Extraire les features au moment de la prédiction
                features = self._extract_features_for_timestamp(spot, prev_event.timestamp)
                
                X.append(features)
                y.append(respawn_time)
        
        return X, y
    
    def _extract_features_for_timestamp(self, spot: ResourceSpot, timestamp: datetime) -> List[float]:
        """Extrait les features pour un timestamp donné."""
        # Similaire à _extract_features mais pour un moment spécifique
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Estimation des autres facteurs pour ce moment
        player_density = self._estimate_player_density(spot.map_id, timestamp)
        server_population = self._get_server_population()  # Pourrait être historisé
        
        features = [
            0,  # time_since_harvest (sera calculé différemment dans le contexte d'entraînement)
            hour_of_day / 24,
            day_of_week / 7,
            player_density / 100,
            server_population / 1000,
            spot.quality_bonus,
            1.0  # harvest_frequency (pourrait être calculé historiquement)
        ]
        
        return features
    
    def get_optimal_farming_route(self, 
                                resource_types: List[ResourceType],
                                max_travel_time: int = 5,
                                hours_ahead: int = 2) -> List[PredictionResult]:
        """
        Calcule une route de farming optimale.
        
        Args:
            resource_types: Types de ressources recherchés
            max_travel_time: Temps de voyage max entre spots (minutes)
            hours_ahead: Horizon de prédiction
            
        Returns:
            Route optimisée triée par priorité
        """
        try:
            all_predictions = []
            
            for resource_type in resource_types:
                predictions = self.predict_respawn_times(
                    resource_type=resource_type,
                    hours_ahead=hours_ahead
                )
                all_predictions.extend(predictions)
            
            # Filtrer les prédictions dans la fenêtre de temps
            now = datetime.now()
            future_limit = now + timedelta(hours=hours_ahead)
            
            valid_predictions = [
                p for p in all_predictions 
                if now <= p.predicted_respawn_time <= future_limit
            ]
            
            # Trier par score de priorité (temps + confiance + compétition)
            def priority_score(pred: PredictionResult) -> float:
                time_factor = (pred.predicted_respawn_time - now).total_seconds() / 3600
                competition_penalty = pred.competition_level / 10
                confidence_bonus = pred.confidence
                
                return confidence_bonus - time_factor * 0.5 - competition_penalty * 0.3
            
            valid_predictions.sort(key=priority_score, reverse=True)
            
            self.logger.info(f"Route optimale calculée: {len(valid_predictions)} spots")
            return valid_predictions[:20]  # Top 20 spots
            
        except Exception as e:
            self.logger.error(f"Erreur calcul route optimale: {e}")
            return []
    
    def export_predictions_report(self, predictions: List[PredictionResult], 
                                filename: str = None) -> str:
        """Exporte un rapport de prédictions."""
        try:
            if not filename:
                filename = f"resource_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'predictions': [],
                'summary': {
                    'total_predictions': len(predictions),
                    'avg_confidence': sum(p.confidence for p in predictions) / len(predictions) if predictions else 0,
                    'resource_types': list(set(p.resource_name for p in predictions))
                }
            }
            
            for pred in predictions:
                pred_dict = {
                    'spot_id': pred.spot_id,
                    'resource_name': pred.resource_name,
                    'predicted_time': pred.predicted_respawn_time.isoformat(),
                    'confidence': pred.confidence,
                    'expected_quantity': pred.expected_quantity,
                    'competition_level': pred.competition_level,
                    'model_used': pred.prediction_model,
                    'reliability_score': pred.reliability_score,
                    'factors': pred.factors
                }
                report['predictions'].append(pred_dict)
            
            # Sauvegarder
            report_path = f"data/reports/{filename}"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Rapport prédictions exporté: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Erreur export rapport: {e}")
            return ""