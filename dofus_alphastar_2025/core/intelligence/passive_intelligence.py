"""
Module d'Intelligence Passive pour TacticalBot
Observe et analyse l'environnement sans intervention active
Phase 1 du Projet Augmenta

Fonctionnalités:
- Analyse de patterns comportementaux
- Évaluation des risques environnementaux
- Détection d'opportunités passives
- Surveillance des entités et ressources
- Apprentissage des patterns de jeu
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

# Type stubs temporaires (à remplacer par les vraies classes plus tard)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any as GameState, Any as IModule, Any as ModuleStatus
else:
    GameState = Any  # Stub temporaire

    # Stub IModule compatible
    class IModule:
        def __init__(self, name: str = "Module"):
            self.name = name

    ModuleStatus = Any  # Stub temporaire

@dataclass
class PatternObservation:
    """Observation d'un pattern comportemental"""
    pattern_type: str  # "enemy_spawn", "resource_availability", "player_behavior"
    location: Tuple[int, int]
    frequency: float  # observations par heure
    confidence: float  # 0.0 à 1.0
    last_seen: float
    duration: float  # durée moyenne d'observation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Évaluation des risques environnementaux"""
    location: Tuple[int, int]
    risk_level: float  # 0.0 = sûr, 1.0 = dangereux
    risk_factors: List[str]  # ["high_enemy_density", "resource_scarcity"]
    assessment_time: float
    validity_duration: float  # durée de validité de l'évaluation


@dataclass
class OpportunityDetection:
    """Détection d'opportunités passives"""
    opportunity_type: str  # "resource_node", "safe_area", "trading_post"
    location: Tuple[int, int]
    value_estimate: float
    accessibility: float  # 0.0 = inaccessible, 1.0 = facilement accessible
    competition_level: float  # 0.0 = pas de compétition, 1.0 = très concurrentiel
    discovery_time: float


class PatternAnalyzer:
    """Analyseur de patterns comportementaux"""

    def __init__(self):
        self.observations = deque(maxlen=1000)
        self.patterns = defaultdict(list)
        self.analysis_cache = {}

    def add_observation(self, observation: PatternObservation):
        """Ajoute une nouvelle observation"""
        self.observations.append(observation)
        self._update_patterns(observation)

    def _update_patterns(self, observation: PatternObservation):
        """Met à jour les patterns basés sur les observations"""
        pattern_key = f"{observation.pattern_type}_{observation.location}"

        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = []

        self.patterns[pattern_key].append(observation)

        # Nettoyer les vieilles observations (>1h)
        cutoff_time = time.time() - 3600
        self.patterns[pattern_key] = [
            obs for obs in self.patterns[pattern_key]
            if obs.last_seen > cutoff_time
        ]

    def analyze_patterns(self, location: Tuple[int, int], radius: int = 5) -> Dict[str, Any]:
        """Analyse les patterns autour d'une location"""
        cache_key = f"{location}_{radius}_{time.time() // 300}"  # Cache 5min

        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        analysis = {
            "enemy_patterns": self._analyze_enemy_patterns(location, radius),
            "resource_patterns": self._analyze_resource_patterns(location, radius),
            "player_patterns": self._analyze_player_patterns(location, radius),
            "temporal_patterns": self._analyze_temporal_patterns(location, radius)
        }

        self.analysis_cache[cache_key] = analysis
        return analysis

    def _analyze_enemy_patterns(self, location: Tuple[int, int], radius: int) -> Dict[str, Any]:
        """Analyse les patterns d'ennemis"""
        enemy_obs = [
            obs for obs in self.observations
            if obs.pattern_type == "enemy_spawn" and
            self._distance(obs.location, location) <= radius
        ]

        if not enemy_obs:
            return {"density": 0.0, "frequency": 0.0, "threat_level": 0.0}

        # Calcul de la densité et fréquence
        density = len(enemy_obs) / max(1, radius ** 2)
        frequency = sum(obs.frequency for obs in enemy_obs) / len(enemy_obs)
        avg_confidence = sum(obs.confidence for obs in enemy_obs) / len(enemy_obs)

        return {
            "density": min(1.0, density),
            "frequency": frequency,
            "threat_level": (density * frequency * avg_confidence),
            "observations_count": len(enemy_obs)
        }

    def _analyze_resource_patterns(self, location: Tuple[int, int], radius: int) -> Dict[str, Any]:
        """Analyse les patterns de ressources"""
        resource_obs = [
            obs for obs in self.observations
            if obs.pattern_type == "resource_availability" and
            self._distance(obs.location, location) <= radius
        ]

        if not resource_obs:
            return {"availability": 0.0, "richness": 0.0, "competition": 0.0}

        availability = sum(obs.confidence for obs in resource_obs) / len(resource_obs)
        richness = sum(obs.metadata.get("richness", 0.5) for obs in resource_obs) / len(resource_obs)
        competition = sum(obs.metadata.get("competition", 0.0) for obs in resource_obs) / len(resource_obs)

        return {
            "availability": availability,
            "richness": richness,
            "competition": competition,
            "nodes_count": len(resource_obs)
        }

    def _analyze_player_patterns(self, location: Tuple[int, int], radius: int) -> Dict[str, Any]:
        """Analyse les patterns de joueurs"""
        player_obs = [
            obs for obs in self.observations
            if obs.pattern_type == "player_behavior" and
            self._distance(obs.location, location) <= radius
        ]

        if not player_obs:
            return {"activity_level": 0.0, "friendliness": 0.5, "trade_opportunities": 0.0}

        activity = sum(obs.frequency for obs in player_obs) / len(player_obs)
        friendliness = sum(obs.metadata.get("friendliness", 0.5) for obs in player_obs) / len(player_obs)
        trade_opps = sum(obs.metadata.get("trade_interest", 0.0) for obs in player_obs) / len(player_obs)

        return {
            "activity_level": activity,
            "friendliness": friendliness,
            "trade_opportunities": trade_opps,
            "player_count": len(player_obs)
        }

    def _analyze_temporal_patterns(self, location: Tuple[int, int], radius: int) -> Dict[str, Any]:
        """Analyse les patterns temporels"""
        recent_obs = [
            obs for obs in self.observations
            if self._distance(obs.location, location) <= radius and
            obs.last_seen > time.time() - 1800  # Dernières 30min
        ]

        if not recent_obs:
            return {"peak_hours": [], "quiet_hours": [], "trend": "stable"}

        # Analyse par heure
        hourly_activity = defaultdict(int)
        for obs in recent_obs:
            hour = int((obs.last_seen % 86400) / 3600)  # Heure du jour
            hourly_activity[hour] += 1

        peak_hours = [h for h, count in hourly_activity.items() if count > np.mean(list(hourly_activity.values()))]
        quiet_hours = [h for h, count in hourly_activity.items() if count < np.mean(list(hourly_activity.values())) * 0.5]

        # Tendance globale
        recent_trend = self._calculate_trend(recent_obs)

        return {
            "peak_hours": peak_hours,
            "quiet_hours": quiet_hours,
            "trend": recent_trend,
            "activity_variance": np.var(list(hourly_activity.values()))
        }

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance euclidienne"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _calculate_trend(self, observations: List[PatternObservation]) -> str:
        """Calcule la tendance d'activité"""
        if len(observations) < 10:
            return "insufficient_data"

        # Trier par timestamp
        sorted_obs = sorted(observations, key=lambda x: x.last_seen)

        # Diviser en deux moitiés
        mid = len(sorted_obs) // 2
        first_half = sorted_obs[:mid]
        second_half = sorted_obs[mid:]

        first_activity = sum(obs.frequency for obs in first_half)
        second_activity = sum(obs.frequency for obs in second_half)

        if second_activity > first_activity * 1.2:
            return "increasing"
        elif second_activity < first_activity * 0.8:
            return "decreasing"
        else:
            return "stable"


class RiskEvaluator:
    """Évaluateur de risques environnementaux"""

    def __init__(self):
        self.risk_cache = {}
        self.risk_history = deque(maxlen=500)

    def evaluate_risk(self, location: Tuple[int, int], game_state: GameState) -> RiskAssessment:
        """Évalue le niveau de risque d'une location"""
        cache_key = f"{location}_{time.time() // 60}"  # Cache 1min

        if cache_key in self.risk_cache:
            return self.risk_cache[cache_key]

        risk_factors = []
        risk_score = 0.0

        # Facteur 1: Densité d'ennemis
        enemy_risk = self._evaluate_enemy_risk(location, game_state)
        if enemy_risk > 0.3:
            risk_factors.append("high_enemy_density")
            risk_score += enemy_risk * 0.4

        # Facteur 2: Activité des joueurs
        player_risk = self._evaluate_player_risk(location, game_state)
        if player_risk > 0.5:
            risk_factors.append("high_player_activity")
            risk_score += player_risk * 0.3

        # Facteur 3: Ressources disponibles
        resource_risk = self._evaluate_resource_risk(location, game_state)
        if resource_risk > 0.6:
            risk_factors.append("resource_scarcity")
            risk_score += resource_risk * 0.2

        # Facteur 4: Conditions environnementales
        env_risk = self._evaluate_environmental_risk(location, game_state)
        risk_score += env_risk * 0.1

        # Normalisation
        risk_score = min(1.0, risk_score)

        assessment = RiskAssessment(
            location=location,
            risk_level=risk_score,
            risk_factors=risk_factors,
            assessment_time=time.time(),
            validity_duration=300  # 5 minutes
        )

        self.risk_cache[cache_key] = assessment
        self.risk_history.append(assessment)

        return assessment

    def _evaluate_enemy_risk(self, location: Tuple[int, int], game_state: GameState) -> float:
        """Évalue le risque lié aux ennemis"""
        if not game_state.combat.enemies:
            return 0.0

        enemy_count = 0
        total_threat = 0.0

        for enemy in game_state.combat.enemies:
            if enemy.position:
                distance = np.sqrt((enemy.position.x - location[0])**2 + (enemy.position.y - location[1])**2)
                if distance <= 10:  # Dans un rayon de 10 cases
                    enemy_count += 1
                    # Menace basée sur le niveau et les HP
                    threat = (enemy.level / 100.0) * (enemy.hp_percentage() / 100.0)
                    total_threat += threat

        if enemy_count == 0:
            return 0.0

        # Risque = densité * menace moyenne
        density = min(1.0, enemy_count / 5.0)  # Max 5 ennemis considérés
        avg_threat = total_threat / enemy_count

        return density * avg_threat

    def _evaluate_player_risk(self, location: Tuple[int, int], game_state: GameState) -> float:
        """Évalue le risque lié aux autres joueurs"""
        # Pour l'instant, estimation simple basée sur la proximité
        # Dans un vrai cas, il faudrait analyser les intentions des joueurs
        player_count = len([p for p in game_state.combat.allies if p.entity_id != game_state.character.name])

        if player_count == 0:
            return 0.0

        # Risque plus élevé avec plus de joueurs (concurrence potentielle)
        return min(1.0, player_count / 3.0 * 0.5)

    def _evaluate_resource_risk(self, location: Tuple[int, int], game_state: GameState) -> float:
        """Évalue le risque lié à la disponibilité des ressources"""
        # Risque élevé si peu de ressources disponibles
        # (nécessiterait une analyse plus poussée des ressources)
        return 0.3  # Placeholder

    def _evaluate_environmental_risk(self, location: Tuple[int, int], game_state: GameState) -> float:
        """Évalue les risques environnementaux"""
        # Facteurs comme la météo, l'heure, les événements spéciaux
        current_hour = datetime.now().hour

        # Risque plus élevé la nuit
        if current_hour < 6 or current_hour > 22:
            return 0.3
        else:
            return 0.1


class OpportunityDetector:
    """Détecteur d'opportunités passives"""

    def __init__(self):
        self.opportunities = deque(maxlen=200)
        self.detection_history = defaultdict(list)

    def scan_for_opportunities(self, game_state: GameState) -> List[OpportunityDetection]:
        """Scanne l'environnement pour détecter des opportunités"""
        opportunities = []

        # Détection de nœuds de ressources
        resource_opportunities = self._detect_resource_opportunities(game_state)
        opportunities.extend(resource_opportunities)

        # Détection de zones sûres
        safe_zones = self._detect_safe_zones(game_state)
        opportunities.extend(safe_zones)

        # Détection d'opportunités sociales
        social_opportunities = self._detect_social_opportunities(game_state)
        opportunities.extend(social_opportunities)

        # Enregistrement des détections
        for opp in opportunities:
            self.opportunities.append(opp)
            self.detection_history[opp.opportunity_type].append(time.time())

        return opportunities

    def _detect_resource_opportunities(self, game_state: GameState) -> List[OpportunityDetection]:
        """Détecte les opportunités de ressources"""
        opportunities = []

        # Simulation de détection de ressources (placeholder)
        # Dans un vrai cas, utiliser la vision pour détecter les nodes

        # Exemple: Ressources agricoles
        if game_state.current_map in ["Astrub", "Village"]:
            opportunities.append(OpportunityDetection(
                opportunity_type="resource_node",
                location=(100, 150),  # Position exemple
                value_estimate=75.0,
                accessibility=0.8,
                competition_level=0.3,
                discovery_time=time.time()
            ))

        return opportunities

    def _detect_safe_zones(self, game_state: GameState) -> List[OpportunityDetection]:
        """Détecte les zones sûres"""
        opportunities = []

        # Zones avec peu d'ennemis et bonne accessibilité
        if len(game_state.combat.enemies) < 2:
            opportunities.append(OpportunityDetection(
                opportunity_type="safe_area",
                location=(game_state.character.position.x, game_state.character.position.y),
                value_estimate=50.0,
                accessibility=0.9,
                competition_level=0.1,
                discovery_time=time.time()
            ))

        return opportunities

    def _detect_social_opportunities(self, game_state: GameState) -> List[OpportunityDetection]:
        """Détecte les opportunités sociales"""
        opportunities = []

        # Opportunités de commerce ou de groupe
        player_count = len([p for p in game_state.combat.allies if p.entity_id != game_state.character.name])

        if player_count > 0:
            opportunities.append(OpportunityDetection(
                opportunity_type="trading_post",
                location=(game_state.character.position.x + 5, game_state.character.position.y + 5),
                value_estimate=60.0,
                accessibility=0.7,
                competition_level=0.4,
                discovery_time=time.time()
            ))

        return opportunities


class PassiveIntelligence(IModule):
    """
    Module d'intelligence passive - observe sans agir
    Collecte des données pour améliorer les décisions futures
    """

    def __init__(self, name: str = "passive_intelligence"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Composants d'analyse
        self.pattern_analyzer = PatternAnalyzer()
        self.risk_evaluator = RiskEvaluator()
        self.opportunity_detector = OpportunityDetector()

        # Données collectées
        self.collected_data = {
            "observations": [],
            "risk_assessments": [],
            "opportunities": [],
            "analysis_results": []
        }

        # Configuration
        self.scan_interval = 30.0  # Scan toutes les 30 secondes
        self.last_scan_time = 0.0
        self.enable_learning = True

        # Métriques
        self.metrics = {
            "scans_performed": 0,
            "patterns_detected": 0,
            "risks_assessed": 0,
            "opportunities_found": 0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING

            # Configuration
            self.scan_interval = config.get("scan_interval", 30.0)
            self.enable_learning = config.get("enable_learning", True)

            # Chargement des données précédentes si disponibles
            self._load_previous_data()

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Module d'intelligence passive initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour l'analyse passive"""
        if not self.is_active():
            return None

        try:
            current_time = time.time()

            # Scan périodique
            if current_time - self.last_scan_time >= self.scan_interval:
                self._perform_passive_scan(game_state)
                self.last_scan_time = current_time
                self.metrics["scans_performed"] += 1

            # Retour des données collectées pour partage
            return {
                "passive_data": {
                    "observations_count": len(self.collected_data["observations"]),
                    "risk_assessments_count": len(self.collected_data["risk_assessments"]),
                    "opportunities_count": len(self.collected_data["opportunities"]),
                    "last_scan": self.last_scan_time
                }
            }

        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Traite les événements"""
        try:
            # Réagir aux événements sans intervention active
            if hasattr(event, 'type'):
                if event.type == "combat_started":
                    self._observe_combat_start(event)
                elif event.type == "resource_found":
                    self._observe_resource_event(event)
                elif event.type == "player_interaction":
                    self._observe_player_interaction(event)

            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "metrics": self.metrics,
            "collected_data_summary": {
                key: len(value) for key, value in self.collected_data.items()
            },
            "scan_interval": self.scan_interval,
            "enable_learning": self.enable_learning
        }

    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            # Sauvegarde des données collectées
            self._save_collected_data()

            self.logger.info("Module d'intelligence passive nettoyé")

        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")

    def _perform_passive_scan(self, game_state: GameState):
        """Effectue un scan passif de l'environnement"""
        try:
            # 1. Analyse des patterns
            if game_state.character.position:
                patterns = self.pattern_analyzer.analyze_patterns(
                    (game_state.character.position.x, game_state.character.position.y)
                )

                # Enregistrement des observations
                for pattern_type, analysis in patterns.items():
                    if analysis.get("observations_count", 0) > 0:
                        observation = PatternObservation(
                            pattern_type=pattern_type,
                            location=(game_state.character.position.x, game_state.character.position.y),
                            frequency=analysis.get("frequency", 0.0),
                            confidence=analysis.get("threat_level", 0.0),
                            last_seen=time.time(),
                            duration=30.0,
                            metadata=analysis
                        )
                        self.pattern_analyzer.add_observation(observation)
                        self.collected_data["observations"].append(observation)
                        self.metrics["patterns_detected"] += 1

            # 2. Évaluation des risques
            if game_state.character.position:
                risk_assessment = self.risk_evaluator.evaluate_risk(
                    (game_state.character.position.x, game_state.character.position.y),
                    game_state
                )

                if risk_assessment.risk_level > 0.1:  # Seulement les risques notables
                    self.collected_data["risk_assessments"].append(risk_assessment)
                    self.metrics["risks_assessed"] += 1

            # 3. Détection d'opportunités
            opportunities = self.opportunity_detector.scan_for_opportunities(game_state)

            for opportunity in opportunities:
                if opportunity.value_estimate > 30:  # Seulement les opportunités intéressantes
                    self.collected_data["opportunities"].append(opportunity)
                    self.metrics["opportunities_found"] += 1

        except Exception as e:
            self.logger.error(f"Erreur scan passif: {e}")

    def _observe_combat_start(self, event):
        """Observe le début d'un combat"""
        observation = PatternObservation(
            pattern_type="combat_start",
            location=(event.data.get("x", 0), event.data.get("y", 0)),
            frequency=1.0,
            confidence=0.8,
            last_seen=time.time(),
            duration=0.0,
            metadata={"enemy_count": event.data.get("enemy_count", 1)}
        )
        self.pattern_analyzer.add_observation(observation)

    def _observe_resource_event(self, event):
        """Observe un événement de ressource"""
        observation = PatternObservation(
            pattern_type="resource_availability",
            location=(event.data.get("x", 0), event.data.get("y", 0)),
            frequency=0.5,
            confidence=0.9,
            last_seen=time.time(),
            duration=60.0,
            metadata={
                "resource_type": event.data.get("resource_type", "unknown"),
                "richness": event.data.get("richness", 0.5)
            }
        )
        self.pattern_analyzer.add_observation(observation)

    def _observe_player_interaction(self, event):
        """Observe une interaction avec un joueur"""
        observation = PatternObservation(
            pattern_type="player_behavior",
            location=(event.data.get("x", 0), event.data.get("y", 0)),
            frequency=0.3,
            confidence=0.7,
            last_seen=time.time(),
            duration=30.0,
            metadata={
                "interaction_type": event.data.get("interaction_type", "unknown"),
                "friendliness": event.data.get("friendliness", 0.5)
            }
        )
        self.pattern_analyzer.add_observation(observation)

    def _load_previous_data(self):
        """Charge les données précédentes"""
        try:
            # Chargement depuis un fichier (placeholder)
            # Dans un vrai cas, charger depuis un fichier JSON ou base de données
            pass
        except Exception as e:
            self.logger.warning(f"Impossible de charger les données précédentes: {e}")

    def _save_collected_data(self):
        """Sauvegarde les données collectées"""
        try:
            # Sauvegarde dans un fichier (placeholder)
            # Dans un vrai cas, sauvegarder dans un fichier JSON ou base de données
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde données: {e}")

    def get_analysis_report(self) -> Dict[str, Any]:
        """Génère un rapport d'analyse"""
        return {
            "metrics": self.metrics,
            "patterns_summary": {
                "total_observations": len(self.collected_data["observations"]),
                "pattern_types": list(set(obs.pattern_type for obs in self.collected_data["observations"]))
            },
            "risk_summary": {
                "total_assessments": len(self.collected_data["risk_assessments"]),
                "average_risk": np.mean([r.risk_level for r in self.collected_data["risk_assessments"]]) if self.collected_data["risk_assessments"] else 0.0
            },
            "opportunities_summary": {
                "total_opportunities": len(self.collected_data["opportunities"]),
                "average_value": np.mean([o.value_estimate for o in self.collected_data["opportunities"]]) if self.collected_data["opportunities"] else 0.0
            }
        }