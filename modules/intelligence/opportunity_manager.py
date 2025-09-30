"""
Gestionnaire d'Opportunités pour TacticalBot
Phase 1 du Projet Augmenta

Fonctionnalités:
- Détection d'opportunités en temps réel
- Évaluation de la valeur et de la faisabilité
- Priorisation des opportunités
- Intégration avec l'intelligence passive
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from ...engine.module_interface import IModule, ModuleStatus
from ...state.realtime_state import GameState, CombatState, Character
from .passive_intelligence import OpportunityDetection, RiskAssessment


@dataclass
class Opportunity:
    """Représente une opportunité détectée"""
    id: str
    opportunity_type: str
    location: Tuple[int, int]
    value_estimate: float
    accessibility: float  # 0.0 = inaccessible, 1.0 = facilement accessible
    competition_level: float  # 0.0 = pas de compétition, 1.0 = très concurrentiel
    risk_level: float  # 0.0 = sûr, 1.0 = dangereux
    time_to_reach: float  # secondes estimées
    duration: float  # durée de disponibilité estimée
    discovery_time: float
    last_updated: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    _priority_score: float = 0.0  # Score de priorité calculé

    def calculate_priority_score(self) -> float:
        """Calcule un score de priorité pour cette opportunité"""
        # Facteurs: valeur, accessibilité, risque, compétition
        value_weight = 0.4
        accessibility_weight = 0.3
        risk_penalty = 0.2
        competition_penalty = 0.1

        # Pénalité pour le risque et la compétition
        risk_factor = 1.0 - (self.risk_level * risk_penalty)
        competition_factor = 1.0 - (self.competition_level * competition_penalty)

        # Bonus pour la durée restante
        time_factor = min(1.0, self.duration / 300.0)  # Bonus si >5min

        score = (
            self.value_estimate * value_weight +
            self.accessibility * accessibility_weight * risk_factor * competition_factor * time_factor
        )

        return max(0.0, score)


@dataclass
class OpportunityFilter:
    """Filtres pour les opportunités"""
    min_value: float = 0.0
    max_risk: float = 1.0
    max_competition: float = 1.0
    max_time_to_reach: float = 300.0  # 5 minutes
    allowed_types: List[str] = field(default_factory=list)
    required_accessibility: float = 0.0


class OpportunityTracker:
    """Suit et met à jour les opportunités"""

    def __init__(self):
        self.opportunities = {}
        self.opportunity_history = deque(maxlen=1000)
        self.location_index = defaultdict(list)  # Index par location

    def add_opportunity(self, opportunity: Opportunity):
        """Ajoute une nouvelle opportunité"""
        self.opportunities[opportunity.id] = opportunity
        self.opportunity_history.append(opportunity)

        # Mise à jour de l'index de location
        location_key = f"{opportunity.location[0]}_{opportunity.location[1]}"
        self.location_index[location_key].append(opportunity.id)

    def update_opportunity(self, opportunity_id: str, updates: Dict[str, Any]):
        """Met à jour une opportunité existante"""
        if opportunity_id in self.opportunities:
            opp = self.opportunities[opportunity_id]
            for key, value in updates.items():
                if hasattr(opp, key):
                    setattr(opp, key, value)
            opp.last_updated = time.time()

    def remove_expired_opportunities(self):
        """Supprime les opportunités expirées"""
        current_time = time.time()
        expired_ids = []

        for opp_id, opp in self.opportunities.items():
            # Expiration basée sur la durée ou le temps depuis la découverte
            if (current_time - opp.discovery_time > opp.duration or
                current_time - opp.last_updated > 600):  # 10min sans mise à jour
                expired_ids.append(opp_id)

        for opp_id in expired_ids:
            del self.opportunities[opp_id]

        return len(expired_ids)

    def get_opportunities_by_location(self, location: Tuple[int, int], radius: int = 5) -> List[Opportunity]:
        """Récupère les opportunités autour d'une location"""
        opportunities = []

        # Recherche dans un rayon
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue

                x, y = location[0] + dx, location[1] + dy
                location_key = f"{x}_{y}"

                for opp_id in self.location_index.get(location_key, []):
                    if opp_id in self.opportunities:
                        opportunities.append(self.opportunities[opp_id])

        return opportunities

    def get_prioritized_opportunities(self, limit: int = 10) -> List[Opportunity]:
        """Retourne les opportunités triées par priorité"""
        active_opportunities = list(self.opportunities.values())

        # Calcul des scores de priorité
        for opp in active_opportunities:
            opp._priority_score = opp.calculate_priority_score()

        # Tri par score décroissant
        active_opportunities.sort(key=lambda x: x._priority_score, reverse=True)

        return active_opportunities[:limit]


class OpportunityManager(IModule):
    """
    Gestionnaire d'opportunités - détecte et priorise les opportunités
    """

    def __init__(self, name: str = "opportunity_manager"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Composants
        self.tracker = OpportunityTracker()

        # Configuration
        self.scan_radius = 15
        self.update_interval = 60.0  # Mise à jour toutes les minutes
        self.last_update = 0.0

        # Filtres par défaut
        self.default_filter = OpportunityFilter(
            min_value=30.0,
            max_risk=0.7,
            max_competition=0.8,
            max_time_to_reach=180.0,
            allowed_types=["resource_node", "safe_area", "trading_post"]
        )

        # Métriques
        self.metrics = {
            "opportunities_detected": 0,
            "opportunities_expired": 0,
            "opportunities_pursued": 0,
            "average_value": 0.0,
            "total_value_pursued": 0.0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING

            # Configuration
            self.scan_radius = config.get("scan_radius", 15)
            self.update_interval = config.get("update_interval", 60.0)

            # Filtres personnalisés
            filter_config = config.get("filters", {})
            self.default_filter = OpportunityFilter(
                min_value=filter_config.get("min_value", 30.0),
                max_risk=filter_config.get("max_risk", 0.7),
                max_competition=filter_config.get("max_competition", 0.8),
                max_time_to_reach=filter_config.get("max_time_to_reach", 180.0),
                allowed_types=filter_config.get("allowed_types", ["resource_node", "safe_area", "trading_post"])
            )

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Gestionnaire d'opportunités initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour la gestion des opportunités"""
        if not self.is_active():
            return None

        try:
            current_time = time.time()

            # Mise à jour périodique
            if current_time - self.last_update >= self.update_interval:
                self._update_opportunities(game_state)
                self.last_update = current_time

            # Nettoyage des opportunités expirées
            expired_count = self.tracker.remove_expired_opportunities()
            self.metrics["opportunities_expired"] += expired_count

            # Retour des données pour partage
            return {
                "opportunities": {
                    "total": len(self.tracker.opportunities),
                    "prioritized": [opp.id for opp in self.tracker.get_prioritized_opportunities(5)],
                    "average_priority": np.mean([opp._priority_score for opp in self.tracker.opportunities.values()]) if self.tracker.opportunities else 0.0
                }
            }

        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Traite les événements"""
        try:
            if hasattr(event, 'type'):
                if event.type == "opportunity_detected":
                    self._handle_opportunity_event(event)
                elif event.type == "opportunity_expired":
                    self._handle_expiration_event(event)
                elif event.type == "opportunity_pursued":
                    self._handle_pursuit_event(event)

            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "metrics": self.metrics,
            "opportunities_count": len(self.tracker.opportunities),
            "scan_radius": self.scan_radius,
            "update_interval": self.update_interval
        }

    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            # Sauvegarde des données
            self._save_opportunity_data()

            self.logger.info("Gestionnaire d'opportunités nettoyé")

        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")

    def detect_opportunities(self, game_state: GameState, location: Optional[Tuple[int, int]] = None) -> List[Opportunity]:
        """Détecte les opportunités autour d'une location"""
        if location is None and game_state.character.position:
            location = (game_state.character.position.x, game_state.character.position.y)

        if not location:
            return []

        # Récupération des opportunités existantes
        existing_opportunities = self.tracker.get_opportunities_by_location(location, self.scan_radius)

        # Mise à jour des opportunités existantes
        updated_opportunities = []
        for opp in existing_opportunities:
            # Recalcul de la valeur et accessibilité basée sur l'état actuel
            updates = self._reassess_opportunity(opp, game_state)
            if updates:
                self.tracker.update_opportunity(opp.id, updates)
                updated_opportunities.append(self.tracker.opportunities[opp.id])

        # Détection de nouvelles opportunités
        new_opportunities = self._scan_for_new_opportunities(game_state, location)

        all_opportunities = updated_opportunities + new_opportunities

        # Filtrage
        filtered_opportunities = self._filter_opportunities(all_opportunities, self.default_filter)

        return filtered_opportunities

    def get_best_opportunities(self, game_state: GameState, limit: int = 5) -> List[Opportunity]:
        """Retourne les meilleures opportunités disponibles"""
        all_opportunities = list(self.tracker.opportunities.values())

        # Filtrage contextuel
        filtered = self._filter_opportunities(all_opportunities, self.default_filter)

        # Tri par priorité
        prioritized = self.tracker.get_prioritized_opportunities(limit * 2)

        # Sélection finale basée sur le contexte
        best_opportunities = self._select_contextual_opportunities(prioritized, game_state, limit)

        return best_opportunities

    def _update_opportunities(self, game_state: GameState):
        """Met à jour toutes les opportunités"""
        # Mise à jour des scores et validité
        for opp_id, opp in list(self.tracker.opportunities.items()):
            updates = self._reassess_opportunity(opp, game_state)
            if updates:
                self.tracker.update_opportunity(opp_id, updates)

    def _scan_for_new_opportunities(self, game_state: GameState, location: Tuple[int, int]) -> List[Opportunity]:
        """Scanne pour de nouvelles opportunités"""
        new_opportunities = []

        # 1. Opportunités de ressources
        resource_opportunities = self._detect_resource_opportunities(game_state, location)
        new_opportunities.extend(resource_opportunities)

        # 2. Opportunités de combat (ennemis faibles)
        combat_opportunities = self._detect_combat_opportunities(game_state, location)
        new_opportunities.extend(combat_opportunities)

        # 3. Opportunités sociales
        social_opportunities = self._detect_social_opportunities(game_state, location)
        new_opportunities.extend(social_opportunities)

        # 4. Opportunités de quêtes
        quest_opportunities = self._detect_quest_opportunities(game_state, location)
        new_opportunities.extend(quest_opportunities)

        return new_opportunities

    def _detect_resource_opportunities(self, game_state: GameState, location: Tuple[int, int]) -> List[Opportunity]:
        """Détecte les opportunités de ressources"""
        opportunities = []

        # Simulation basée sur la carte et les ressources connues
        if game_state.current_map in ["Astrub Fields", "Village"]:
            # Ressources agricoles
            for i in range(3):  # 3 nodes potentiels
                opp_location = (location[0] + np.random.randint(-10, 10),
                               location[1] + np.random.randint(-10, 10))

                opportunity = Opportunity(
                    id=f"resource_{int(time.time())}_{i}",
                    opportunity_type="resource_node",
                    location=opp_location,
                    value_estimate=50.0 + np.random.random() * 30.0,
                    accessibility=0.7 + np.random.random() * 0.3,
                    competition_level=np.random.random() * 0.5,
                    risk_level=np.random.random() * 0.3,
                    time_to_reach=30.0 + np.random.random() * 60.0,
                    duration=300.0 + np.random.random() * 600.0,  # 5-15min
                    discovery_time=time.time(),
                    last_updated=time.time(),
                    metadata={"resource_type": "agricultural", "richness": 0.6 + np.random.random() * 0.4}
                )
                opportunities.append(opportunity)

        return opportunities

    def _detect_combat_opportunities(self, game_state: GameState, location: Tuple[int, int]) -> List[Opportunity]:
        """Détecte les opportunités de combat"""
        opportunities = []

        # Ennemis faibles comme opportunités
        for enemy in game_state.combat.enemies:
            if enemy.is_dead() or not enemy.position:
                continue

            # Calcul de la faiblesse
            hp_percentage = enemy.hp_percentage()
            if hp_percentage < 30:  # Ennemi très faible
                distance = np.sqrt((enemy.position.x - location[0])**2 + (enemy.position.y - location[1])**2)

                opportunity = Opportunity(
                    id=f"combat_{enemy.entity_id}_{int(time.time())}",
                    opportunity_type="combat_opportunity",
                    location=(enemy.position.x, enemy.position.y),
                    value_estimate=40.0 + (100 - hp_percentage) * 0.5,  # Plus faible = plus de valeur
                    accessibility=1.0 if distance <= 5 else 0.8,
                    competition_level=0.2,  # Peu de compétition pour les kills
                    risk_level=min(0.8, hp_percentage / 100.0),  # Moins de risque si faible
                    time_to_reach=distance * 2.0,  # 2s par case
                    duration=120.0,  # 2min pour tuer
                    discovery_time=time.time(),
                    last_updated=time.time(),
                    metadata={"enemy_level": enemy.level, "hp_percentage": hp_percentage}
                )
                opportunities.append(opportunity)

        return opportunities

    def _detect_social_opportunities(self, game_state: GameState, location: Tuple[int, int]) -> List[Opportunity]:
        """Détecte les opportunités sociales"""
        opportunities = []

        # Joueurs comme opportunités de commerce
        for player in game_state.combat.allies:
            if player.entity_id == game_state.character.name or not player.position:
                continue

            distance = np.sqrt((player.position.x - location[0])**2 + (player.position.y - location[1])**2)

            opportunity = Opportunity(
                id=f"social_{player.entity_id}_{int(time.time())}",
                opportunity_type="trading_opportunity",
                location=(player.position.x, player.position.y),
                value_estimate=60.0,
                accessibility=0.9 if distance <= 3 else 0.6,
                competition_level=0.3,
                risk_level=0.1,  # Très peu de risque
                time_to_reach=distance * 1.5,
                duration=600.0,  # 10min pour interagir
                discovery_time=time.time(),
                last_updated=time.time(),
                metadata={"player_name": player.name, "trade_interest": 0.7}
            )
            opportunities.append(opportunity)

        return opportunities

    def _detect_quest_opportunities(self, game_state: GameState, location: Tuple[int, int]) -> List[Opportunity]:
        """Détecte les opportunités de quêtes"""
        opportunities = []

        # Placeholder pour les quêtes
        if game_state.current_quest:
            # Objectif de quête comme opportunité
            quest_location = (location[0] + 10, location[1] + 10)  # Position fictive

            opportunity = Opportunity(
                id=f"quest_{game_state.current_quest}_{int(time.time())}",
                opportunity_type="quest_objective",
                location=quest_location,
                value_estimate=100.0,  # Haute valeur pour les quêtes
                accessibility=0.8,
                competition_level=0.1,
                risk_level=0.3,
                time_to_reach=45.0,
                duration=1800.0,  # 30min pour compléter
                discovery_time=time.time(),
                last_updated=time.time(),
                metadata={"quest_name": game_state.current_quest, "progress": 0.5}
            )
            opportunities.append(opportunity)

        return opportunities

    def _reassess_opportunity(self, opportunity: Opportunity, game_state: GameState) -> Dict[str, Any]:
        """Réévalue une opportunité basée sur l'état actuel"""
        updates = {}

        # Mise à jour du risque
        if game_state.character.position:
            # Calcul de la distance actuelle
            current_distance = np.sqrt(
                (opportunity.location[0] - game_state.character.position.x)**2 +
                (opportunity.location[1] - game_state.character.position.y)**2
            )

            # Mise à jour du temps pour atteindre
            updates["time_to_reach"] = current_distance * 2.0

            # Mise à jour de l'accessibilité basée sur les obstacles
            updates["accessibility"] = max(0.1, opportunity.accessibility - (current_distance / 100.0))

        # Mise à jour de la compétition (simplifiée)
        # Dans un vrai cas, compter les joueurs proches

        if updates:
            updates["last_updated"] = time.time()

        return updates

    def _filter_opportunities(self, opportunities: List[Opportunity], filter_config: OpportunityFilter) -> List[Opportunity]:
        """Filtre les opportunités selon les critères"""
        filtered = []

        for opp in opportunities:
            # Filtre de valeur minimale
            if opp.value_estimate < filter_config.min_value:
                continue

            # Filtre de risque maximum
            if opp.risk_level > filter_config.max_risk:
                continue

            # Filtre de compétition
            if opp.competition_level > filter_config.max_competition:
                continue

            # Filtre de temps pour atteindre
            if opp.time_to_reach > filter_config.max_time_to_reach:
                continue

            # Filtre de types autorisés
            if filter_config.allowed_types and opp.opportunity_type not in filter_config.allowed_types:
                continue

            # Filtre d'accessibilité
            if opp.accessibility < filter_config.required_accessibility:
                continue

            filtered.append(opp)

        return filtered

    def _select_contextual_opportunities(self, opportunities: List[Opportunity], game_state: GameState, limit: int) -> List[Opportunity]:
        """Sélectionne les opportunités les plus appropriées au contexte"""
        if not opportunities:
            return []

        # Ajustement basé sur l'état du personnage
        adjusted_opportunities = []

        for opp in opportunities:
            adjusted_opp = opp  # Copie pour modification

            # Bonus si le personnage a peu de HP et que l'opportunité est sûre
            if game_state.character.hp_percentage() < 50 and opp.risk_level < 0.3:
                adjusted_opp._priority_score *= 1.5

            # Bonus si l'opportunité est proche et facile d'accès
            if opp.time_to_reach < 30 and opp.accessibility > 0.8:
                adjusted_opp._priority_score *= 1.3

            # Pénalité si compétition élevée et personnage faible
            if opp.competition_level > 0.7 and game_state.character.hp_percentage() < 70:
                adjusted_opp._priority_score *= 0.7

            adjusted_opportunities.append(adjusted_opp)

        # Tri et sélection
        adjusted_opportunities.sort(key=lambda x: x._priority_score, reverse=True)

        return adjusted_opportunities[:limit]

    def _handle_opportunity_event(self, event):
        """Traite un événement d'opportunité détectée"""
        # Création d'une opportunité à partir de l'événement
        pass

    def _handle_expiration_event(self, event):
        """Traite un événement d'expiration"""
        # Suppression de l'opportunité
        pass

    def _handle_pursuit_event(self, event):
        """Traite un événement de poursuite d'opportunité"""
        self.metrics["opportunities_pursued"] += 1
        self.metrics["total_value_pursued"] += event.data.get("value", 0.0)

    def _save_opportunity_data(self):
        """Sauvegarde les données d'opportunités"""
        try:
            # Sauvegarde dans un fichier (placeholder)
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")

    def get_opportunity_report(self) -> Dict[str, Any]:
        """Génère un rapport sur les opportunités"""
        if not self.tracker.opportunities:
            return {"total_opportunities": 0}

        opportunities = list(self.tracker.opportunities.values())

        # Calcul des statistiques
        total_value = sum(opp.value_estimate for opp in opportunities)
        avg_risk = np.mean([opp.risk_level for opp in opportunities])
        avg_competition = np.mean([opp.competition_level for opp in opportunities])

        # Répartition par type
        type_distribution = defaultdict(int)
        for opp in opportunities:
            type_distribution[opp.opportunity_type] += 1

        return {
            "total_opportunities": len(opportunities),
            "average_value": total_value / len(opportunities),
            "average_risk": avg_risk,
            "average_competition": avg_competition,
            "type_distribution": dict(type_distribution),
            "most_valuable": max(opportunities, key=lambda x: x.value_estimate).id if opportunities else None,
            "safest": min(opportunities, key=lambda x: x.risk_level).id if opportunities else None
        }