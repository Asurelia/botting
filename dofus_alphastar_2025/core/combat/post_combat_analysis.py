"""
Analyse Post-Combat (After Action Report) pour TacticalBot
Phase 3 du Projet Augmenta

Fonctionnalités:
- Analyse des performances de combat
- Identification des erreurs et succès
- Recommandations d'amélioration
- Apprentissage des patterns de victoire/défaite
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
# from ...state.realtime_state import GameState, CombatState, Character, CombatEntity


@dataclass
class CombatReport:
    """Rapport d'analyse post-combat"""
    combat_id: str
    start_time: float
    end_time: float
    duration: float
    victory: bool
    character_class: str
    initial_hp: float
    final_hp: float
    damage_dealt: float
    damage_received: float
    spells_used: List[Dict[str, Any]]
    enemies_faced: List[Dict[str, Any]]
    allies_present: List[Dict[str, Any]]
    key_events: List[Dict[str, Any]]
    performance_score: float
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Métriques de performance du combat"""
    accuracy: float = 0.0          # Précision des sorts
    efficiency: float = 0.0        # Efficacité des actions
    survival: float = 0.0          # Capacité de survie
    damage_output: float = 0.0     # Production de dégâts
    resource_management: float = 0.0  # Gestion des ressources
    tactical_awareness: float = 0.0   # Conscience tactique


class CombatAnalyzer:
    """Analyseur de performances de combat"""

    def __init__(self):
        self.combat_history = deque(maxlen=1000)
        self.performance_patterns = defaultdict(list)
        self.error_patterns = defaultdict(list)

    def analyze_combat(self, game_state_before: GameState, game_state_after: GameState,
                      combat_events: List[Dict[str, Any]]) -> CombatReport:
        """Analyse un combat terminé"""
        combat_id = f"combat_{int(time.time())}_{np.random.randint(1000, 9999)}"

        # Calcul de la durée
        duration = game_state_after.timestamp - game_state_before.timestamp

        # Détermination de la victoire
        victory = self._determine_victory(game_state_after)

        # Collecte des données
        initial_hp = game_state_before.character.hp_percentage()
        final_hp = game_state_after.character.hp_percentage()

        damage_dealt = self._calculate_damage_dealt(combat_events)
        damage_received = initial_hp - final_hp

        spells_used = self._extract_spell_usage(combat_events)
        enemies_faced = self._extract_enemy_data(game_state_before, game_state_after)
        allies_present = self._extract_ally_data(game_state_before)

        # Calcul des métriques de performance
        metrics = self._calculate_performance_metrics(
            game_state_before, game_state_after, combat_events
        )

        # Génération des recommandations
        recommendations = self._generate_recommendations(
            metrics, spells_used, enemies_faced, victory
        )

        # Calcul du score de performance
        performance_score = self._calculate_performance_score(metrics, victory)

        # Création du rapport
        report = CombatReport(
            combat_id=combat_id,
            start_time=game_state_before.timestamp,
            end_time=game_state_after.timestamp,
            duration=duration,
            victory=victory,
            character_class=game_state_before.character.character_class.value,
            initial_hp=initial_hp,
            final_hp=final_hp,
            damage_dealt=damage_dealt,
            damage_received=damage_received,
            spells_used=spells_used,
            enemies_faced=enemies_faced,
            allies_present=allies_present,
            key_events=combat_events,
            performance_score=performance_score,
            recommendations=recommendations
        )

        # Enregistrement pour analyse de patterns
        self.combat_history.append(report)
        self._update_performance_patterns(report)

        return report

    def _determine_victory(self, game_state_after: GameState) -> bool:
        """Détermine si le combat a été gagné"""
        # Victoire si tous les ennemis sont morts et le personnage est vivant
        all_enemies_dead = all(enemy.is_dead() for enemy in game_state_after.combat.enemies)
        character_alive = not game_state_after.character.is_dead

        return all_enemies_dead and character_alive

    def _calculate_damage_dealt(self, combat_events: List[Dict[str, Any]]) -> float:
        """Calcule les dégâts infligés"""
        total_damage = 0.0

        for event in combat_events:
            if event.get("type") == "spell_cast" and event.get("damage_dealt"):
                total_damage += event["damage_dealt"]

        return total_damage

    def _extract_spell_usage(self, combat_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extrait l'utilisation des sorts"""
        spell_usage = []

        for event in combat_events:
            if event.get("type") == "spell_cast":
                spell_usage.append({
                    "spell_id": event.get("spell_id"),
                    "target": event.get("target"),
                    "damage": event.get("damage_dealt", 0),
                    "heal": event.get("heal_amount", 0),
                    "timestamp": event.get("timestamp"),
                    "success": event.get("success", True)
                })

        return spell_usage

    def _extract_enemy_data(self, game_state_before: GameState, game_state_after: GameState) -> List[Dict[str, Any]]:
        """Extrait les données des ennemis"""
        enemies = []

        for enemy in game_state_before.combat.enemies:
            enemy_after = self._find_entity_after(enemy.entity_id, game_state_after)

            enemies.append({
                "entity_id": enemy.entity_id,
                "level": enemy.level,
                "initial_hp": enemy.current_hp,
                "final_hp": enemy_after.current_hp if enemy_after else 0,
                "damage_taken": enemy.current_hp - (enemy_after.current_hp if enemy_after else 0),
                "is_dead": enemy_after.is_dead() if enemy_after else True
            })

        return enemies

    def _extract_ally_data(self, game_state_before: GameState) -> List[Dict[str, Any]]:
        """Extrait les données des alliés"""
        allies = []

        for ally in game_state_before.combat.allies:
            allies.append({
                "entity_id": ally.entity_id,
                "level": ally.level,
                "initial_hp": ally.current_hp,
                "is_player": ally.entity_id == game_state_before.character.name
            })

        return allies

    def _find_entity_after(self, entity_id: int, game_state_after: GameState) -> Optional[CombatEntity]:
        """Trouve une entité dans l'état après combat"""
        for entity in game_state_after.combat.get_all_entities():
            if entity.entity_id == entity_id:
                return entity
        return None

    def _calculate_performance_metrics(self, game_state_before: GameState,
                                     game_state_after: GameState,
                                     combat_events: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calcule les métriques de performance"""
        metrics = PerformanceMetrics()

        # Précision des sorts
        spell_casts = [e for e in combat_events if e.get("type") == "spell_cast"]
        successful_casts = [e for e in spell_casts if e.get("success", True)]
        metrics.accuracy = len(successful_casts) / max(1, len(spell_casts))

        # Efficacité (dégâts par PA dépensé)
        total_pa_used = sum(e.get("pa_cost", 0) for e in spell_casts)
        total_damage = sum(e.get("damage_dealt", 0) for e in spell_casts)
        metrics.efficiency = total_damage / max(1, total_pa_used)

        # Survie (HP conservés)
        hp_conserved = game_state_after.character.hp_percentage()
        metrics.survival = hp_conserved / 100.0

        # Production de dégâts
        metrics.damage_output = total_damage / max(1, game_state_after.timestamp - game_state_before.timestamp)

        # Gestion des ressources (PA/PM utilisés efficacement)
        total_resources = game_state_before.character.current_pa + game_state_before.character.current_pm
        resources_used = total_pa_used + sum(e.get("pm_cost", 0) for e in spell_casts)
        metrics.resource_management = resources_used / max(1, total_resources)

        # Conscience tactique (basée sur les événements clés)
        key_events = [e for e in combat_events if e.get("importance", 0) > 0.5]
        metrics.tactical_awareness = len(key_events) / max(1, len(combat_events))

        return metrics

    def _generate_recommendations(self, metrics: PerformanceMetrics,
                                spells_used: List[Dict[str, Any]],
                                enemies_faced: List[Dict[str, Any]],
                                victory: bool) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []

        # Recommandations basées sur les métriques
        if metrics.accuracy < 0.7:
            recommendations.append("Améliorez la précision des sorts - visez des cibles plus faibles")

        if metrics.efficiency < 50:
            recommendations.append("Optimisez l'utilisation des PA - priorisez les sorts à haut rendement")

        if metrics.survival < 0.3:
            recommendations.append("Améliorez la survie - utilisez plus de sorts défensifs")

        if metrics.resource_management > 0.9:
            recommendations.append("Économisez les ressources - évitez les sorts inutiles")

        # Recommandations basées sur les sorts utilisés
        if spells_used:
            failed_spells = [s for s in spells_used if not s["success"]]
            if len(failed_spells) > len(spells_used) * 0.3:
                recommendations.append("Réduisez les échecs de sorts - vérifiez les prérequis")

        # Recommandations basées sur les ennemis
        tough_enemies = [e for e in enemies_faced if e["damage_taken"] < e["initial_hp"] * 0.5]
        if tough_enemies:
            recommendations.append("Adaptez la stratégie contre les ennemis résistants")

        # Recommandations générales
        if not victory:
            recommendations.append("Analysez les causes de défaite pour améliorer la stratégie")

        return recommendations

    def _calculate_performance_score(self, metrics: PerformanceMetrics, victory: bool) -> float:
        """Calcule un score de performance global"""
        # Pondération des métriques
        weights = {
            "accuracy": 0.2,
            "efficiency": 0.25,
            "survival": 0.3,
            "damage_output": 0.15,
            "resource_management": 0.05,
            "tactical_awareness": 0.05
        }

        score = (
            metrics.accuracy * weights["accuracy"] +
            min(1.0, metrics.efficiency / 100) * weights["efficiency"] +
            metrics.survival * weights["survival"] +
            min(1.0, metrics.damage_output / 200) * weights["damage_output"] +
            (1.0 - metrics.resource_management) * weights["resource_management"] +
            metrics.tactical_awareness * weights["tactical_awareness"]
        )

        # Bonus de victoire
        if victory:
            score *= 1.2

        return min(100.0, score * 100)

    def _update_performance_patterns(self, report: CombatReport):
        """Met à jour les patterns de performance"""
        # Patterns par classe
        class_key = report.character_class
        self.performance_patterns[class_key].append(report.performance_score)

        # Patterns d'erreurs
        if not report.victory:
            self.error_patterns[class_key].append({
                "combat_id": report.combat_id,
                "recommendations": report.recommendations,
                "damage_received": report.damage_received,
                "spells_failed": len([s for s in report.spells_used if not s["success"]])
            })


class PostCombatAnalysis(IModule):
    """
    Module d'analyse post-combat
    """

    def __init__(self, name: str = "post_combat_analysis"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Composants
        self.analyzer = CombatAnalyzer()

        # État du combat actuel
        self.current_combat_start = None
        self.current_combat_events = []

        # Configuration
        self.auto_analyze = True
        self.min_combat_duration = 10.0  # 10 secondes minimum

        # Métriques
        self.metrics = {
            "combats_analyzed": 0,
            "average_performance": 0.0,
            "victory_rate": 0.0,
            "total_recommendations": 0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING

            # Configuration
            self.auto_analyze = config.get("auto_analyze", True)
            self.min_combat_duration = config.get("min_combat_duration", 10.0)

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Module d'analyse post-combat initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour l'analyse"""
        if not self.is_active():
            return None

        try:
            # Détection de fin de combat
            if self._is_combat_ended(game_state):
                self._analyze_current_combat(game_state)

            # Mise à jour des métriques
            self._update_metrics()

            # Retour des données pour partage
            return {
                "analysis_data": {
                    "combats_analyzed": self.metrics["combats_analyzed"],
                    "current_combat_active": self.current_combat_start is not None,
                    "average_performance": self.metrics["average_performance"]
                }
            }

        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Traite les événements"""
        try:
            if hasattr(event, 'type'):
                if event.type == "combat_started":
                    self._handle_combat_start_event(event)
                elif event.type == "spell_cast":
                    self._handle_spell_cast_event(event)
                elif event.type == "damage_dealt":
                    self._handle_damage_event(event)
                elif event.type == "combat_ended":
                    self._handle_combat_end_event(event)

            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "metrics": self.metrics,
            "current_combat_active": self.current_combat_start is not None,
            "auto_analyze": self.auto_analyze,
            "min_combat_duration": self.min_combat_duration
        }

    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            # Sauvegarde des données d'analyse
            self._save_analysis_data()

            self.logger.info("Module d'analyse post-combat nettoyé")

        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")

    def get_latest_reports(self, limit: int = 5) -> List[CombatReport]:
        """Retourne les derniers rapports d'analyse"""
        return list(self.analyzer.combat_history)[-limit:]

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyse les tendances de performance"""
        if not self.analyzer.combat_history:
            return {"error": "No combat data available"}

        recent_reports = list(self.analyzer.combat_history)[-10:]  # 10 derniers combats

        # Calcul des tendances
        performance_scores = [r.performance_score for r in recent_reports]
        victory_count = sum(1 for r in recent_reports if r.victory)

        # Tendances par métrique
        trends = {
            "performance_trend": self._calculate_trend(performance_scores),
            "victory_rate": victory_count / len(recent_reports),
            "average_performance": np.mean(performance_scores),
            "performance_volatility": np.std(performance_scores)
        }

        return trends

    def _is_combat_ended(self, game_state: GameState) -> bool:
        """Détermine si le combat est terminé"""
        if not self.current_combat_start:
            return False

        # Combat terminé si plus d'ennemis actifs ou personnage mort
        active_enemies = [e for e in game_state.combat.enemies if not e.is_dead()]
        character_alive = not game_state.character.is_dead

        return len(active_enemies) == 0 or not character_alive

    def _analyze_current_combat(self, game_state: GameState):
        """Analyse le combat actuel"""
        if not self.current_combat_start:
            return

        # Création d'un état "avant" fictif (approximation)
        game_state_before = game_state  # Simplification

        # Analyse du combat
        report = self.analyzer.analyze_combat(
            game_state_before, game_state, self.current_combat_events
        )

        self.metrics["combats_analyzed"] += 1

        # Log du rapport
        self.logger.info(f"Combat analysé: {report.combat_id} - Victoire: {report.victory} - Score: {report.performance_score:.1f}")

        # Réinitialisation
        self.current_combat_start = None
        self.current_combat_events = []

    def _handle_combat_start_event(self, event):
        """Traite le début d'un combat"""
        self.current_combat_start = time.time()
        self.current_combat_events = []

        self.current_combat_events.append({
            "type": "combat_started",
            "timestamp": time.time(),
            "data": event.data
        })

    def _handle_spell_cast_event(self, event):
        """Traite un événement de sort lancé"""
        if self.current_combat_start:
            self.current_combat_events.append({
                "type": "spell_cast",
                "timestamp": time.time(),
                "spell_id": event.data.get("spell_id"),
                "target": event.data.get("target"),
                "damage_dealt": event.data.get("damage", 0),
                "heal_amount": event.data.get("heal", 0),
                "pa_cost": event.data.get("pa_cost", 0),
                "pm_cost": event.data.get("pm_cost", 0),
                "success": event.data.get("success", True)
            })

    def _handle_damage_event(self, event):
        """Traite un événement de dégâts"""
        if self.current_combat_start:
            self.current_combat_events.append({
                "type": "damage_dealt",
                "timestamp": time.time(),
                "damage": event.data.get("damage", 0),
                "target": event.data.get("target"),
                "source": event.data.get("source")
            })

    def _handle_combat_end_event(self, event):
        """Traite la fin d'un combat"""
        if self.current_combat_start:
            self.current_combat_events.append({
                "type": "combat_ended",
                "timestamp": time.time(),
                "victory": event.data.get("victory", False),
                "duration": time.time() - self.current_combat_start
            })

    def _update_metrics(self):
        """Met à jour les métriques"""
        if self.analyzer.combat_history:
            recent_reports = list(self.analyzer.combat_history)[-10:]
            self.metrics["average_performance"] = np.mean([r.performance_score for r in recent_reports])
            self.metrics["victory_rate"] = sum(1 for r in recent_reports if r.victory) / len(recent_reports)
            self.metrics["total_recommendations"] = sum(len(r.recommendations) for r in recent_reports)

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcule la tendance d'une série de valeurs"""
        if len(values) < 3:
            return "insufficient_data"

        # Calcul de la pente
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 5:
            return "improving"
        elif slope < -5:
            return "declining"
        else:
            return "stable"

    def _save_analysis_data(self):
        """Sauvegarde les données d'analyse"""
        try:
            # Sauvegarde dans un fichier (placeholder)
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")

    def get_analysis_report(self) -> Dict[str, Any]:
        """Génère un rapport d'analyse complet"""
        trends = self.get_performance_trends()

        return {
            "metrics": self.metrics,
            "trends": trends,
            "recent_reports": [
                {
                    "combat_id": r.combat_id,
                    "victory": r.victory,
                    "performance_score": r.performance_score,
                    "recommendations_count": len(r.recommendations)
                }
                for r in self.get_latest_reports(5)
            ],
            "error_patterns": dict(self.analyzer.error_patterns),
            "performance_patterns": dict(self.analyzer.performance_patterns)
        }