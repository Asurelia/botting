"""
After Action Report (AAR) pour TacticalBot
Phase 3 du Projet Augmenta - Apprentissage Actif

Fonctionnalités:
- Analyse détaillée post-combat
- Identification erreurs critiques
- Comparaison avec combats réussis
- Recommandations d'amélioration
- Apprentissage patterns victoire/défaite
- Optimisation stratégies de combat
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
from ...state.realtime_state import GameState, CombatState


class CombatOutcome(Enum):
    """Résultats possibles d'un combat"""
    VICTORY = "victory"
    DEFEAT = "defeat"
    FLEE = "flee"
    TIMEOUT = "timeout"


class ErrorType(Enum):
    """Types d'erreurs en combat"""
    POSITIONING = "positioning"
    SPELL_CHOICE = "spell_choice"
    RESOURCE_MANAGEMENT = "resource_management"
    TIMING = "timing"
    TARGET_SELECTION = "target_selection"
    DEFENSIVE_FAILURE = "defensive_failure"


@dataclass
class CombatAction:
    """Action effectuée pendant le combat"""
    turn: int
    action_type: str  # "spell", "movement", "pass"
    spell_name: Optional[str] = None
    target: Optional[str] = None
    position: Optional[Tuple[int, int]] = None
    ap_cost: int = 0
    mp_cost: int = 0
    damage_dealt: int = 0
    damage_received: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CombatSnapshot:
    """État du combat à un moment donné"""
    turn: int
    player_hp: int
    player_ap: int
    player_mp: int
    player_position: Tuple[int, int]
    enemies: List[Dict[str, Any]] = field(default_factory=list)
    allies: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CombatRecord:
    """Enregistrement complet d'un combat"""
    combat_id: str
    outcome: CombatOutcome
    
    # Participants
    player_level: int
    enemy_types: List[str]
    enemy_levels: List[int]
    
    # Déroulement
    actions: List[CombatAction] = field(default_factory=list)
    snapshots: List[CombatSnapshot] = field(default_factory=list)
    
    # Métriques
    duration: float = 0.0
    total_damage_dealt: int = 0
    total_damage_received: int = 0
    turns_count: int = 0
    
    # Contexte
    location: str = "Unknown"
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    # Résultats
    xp_gained: int = 0
    kamas_gained: int = 0
    items_looted: List[str] = field(default_factory=list)


@dataclass
class CombatError:
    """Erreur identifiée dans un combat"""
    error_type: ErrorType
    turn: int
    description: str
    severity: float  # 0.0 = mineur, 1.0 = critique
    action_involved: Optional[CombatAction] = None
    recommendation: str = ""


@dataclass
class AfterActionReport:
    """Rapport d'analyse post-combat"""
    combat_id: str
    outcome: CombatOutcome
    
    # Analyse
    errors_identified: List[CombatError] = field(default_factory=list)
    critical_moments: List[int] = field(default_factory=list)  # Turns critiques
    
    # Comparaison
    similar_victories: List[str] = field(default_factory=list)
    similar_defeats: List[str] = field(default_factory=list)
    
    # Recommandations
    recommendations: List[str] = field(default_factory=list)
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Métriques de performance
    efficiency_score: float = 0.0
    positioning_score: float = 0.0
    spell_usage_score: float = 0.0
    resource_management_score: float = 0.0
    
    # Apprentissage
    lessons_learned: List[str] = field(default_factory=list)
    patterns_identified: List[str] = field(default_factory=list)
    
    # Métadonnées
    generated_at: datetime = field(default_factory=datetime.now)


class CombatAnalyzer:
    """Analyseur de combats"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CombatAnalyzer")
    
    def analyze_combat(self, combat_record: CombatRecord) -> AfterActionReport:
        """Analyse complète d'un combat"""
        report = AfterActionReport(
            combat_id=combat_record.combat_id,
            outcome=combat_record.outcome
        )
        
        # 1. Identification des erreurs
        report.errors_identified = self._identify_errors(combat_record)
        
        # 2. Détection moments critiques
        report.critical_moments = self._find_critical_moments(combat_record)
        
        # 3. Calcul scores de performance
        report.efficiency_score = self._calculate_efficiency(combat_record)
        report.positioning_score = self._analyze_positioning(combat_record)
        report.spell_usage_score = self._analyze_spell_usage(combat_record)
        report.resource_management_score = self._analyze_resource_management(combat_record)
        
        # 4. Génération recommandations
        report.recommendations = self._generate_recommendations(combat_record, report.errors_identified)
        
        # 5. Identification patterns
        report.patterns_identified = self._identify_patterns(combat_record)
        
        # 6. Leçons apprises
        report.lessons_learned = self._extract_lessons(combat_record, report)
        
        return report
    
    def _identify_errors(self, combat: CombatRecord) -> List[CombatError]:
        """Identifie les erreurs commises"""
        errors = []
        
        # Analyse turn par turn
        for i, action in enumerate(combat.actions):
            # Erreur 1 : Mauvais positionnement
            if action.action_type == "movement":
                positioning_error = self._check_positioning_error(action, combat, i)
                if positioning_error:
                    errors.append(positioning_error)
            
            # Erreur 2 : Mauvais choix de sort
            if action.action_type == "spell":
                spell_error = self._check_spell_choice_error(action, combat, i)
                if spell_error:
                    errors.append(spell_error)
            
            # Erreur 3 : Gaspillage de ressources
            resource_error = self._check_resource_waste(action, combat, i)
            if resource_error:
                errors.append(resource_error)
        
        # Erreur 4 : Défense insuffisante
        if combat.outcome == CombatOutcome.DEFEAT:
            defensive_errors = self._check_defensive_failures(combat)
            errors.extend(defensive_errors)
        
        return errors
    
    def _check_positioning_error(self, action: CombatAction, combat: CombatRecord, 
                                 action_index: int) -> Optional[CombatError]:
        """Vérifie les erreurs de positionnement"""
        # Récupération du snapshot correspondant
        snapshot = self._get_snapshot_at_turn(combat, action.turn)
        if not snapshot:
            return None
        
        # Vérification : trop proche des ennemis
        if action.position:
            for enemy in snapshot.enemies:
                if enemy.get("position"):
                    distance = self._calculate_distance(action.position, enemy["position"])
                    if distance <= 2 and enemy.get("hp_percentage", 100) > 50:
                        return CombatError(
                            error_type=ErrorType.POSITIONING,
                            turn=action.turn,
                            description=f"Trop proche d'un ennemi fort (distance: {distance})",
                            severity=0.7,
                            action_involved=action,
                            recommendation="Maintenir distance de sécurité (3-5 cases)"
                        )
        
        return None
    
    def _check_spell_choice_error(self, action: CombatAction, combat: CombatRecord,
                                  action_index: int) -> Optional[CombatError]:
        """Vérifie les erreurs de choix de sort"""
        snapshot = self._get_snapshot_at_turn(combat, action.turn)
        if not snapshot:
            return None
        
        # Erreur : Sort faible contre ennemi résistant
        if action.damage_dealt < action.ap_cost * 10:  # Ratio dégâts/PA faible
            return CombatError(
                error_type=ErrorType.SPELL_CHOICE,
                turn=action.turn,
                description=f"Sort inefficace : {action.damage_dealt} dégâts pour {action.ap_cost} PA",
                severity=0.5,
                action_involved=action,
                recommendation="Utiliser sorts avec meilleur ratio dégâts/PA"
            )
        
        # Erreur : Sort de zone sans cibles multiples
        if "zone" in action.spell_name.lower() if action.spell_name else False:
            if len(snapshot.enemies) == 1:
                return CombatError(
                    error_type=ErrorType.SPELL_CHOICE,
                    turn=action.turn,
                    description="Sort de zone utilisé sur cible unique",
                    severity=0.4,
                    action_involved=action,
                    recommendation="Réserver sorts de zone pour groupes d'ennemis"
                )
        
        return None
    
    def _check_resource_waste(self, action: CombatAction, combat: CombatRecord,
                             action_index: int) -> Optional[CombatError]:
        """Vérifie le gaspillage de ressources"""
        snapshot = self._get_snapshot_at_turn(combat, action.turn)
        if not snapshot:
            return None
        
        # Erreur : Overkill (trop de dégâts sur ennemi presque mort)
        if action.target and action.damage_dealt > 0:
            target_enemy = next((e for e in snapshot.enemies if e.get("id") == action.target), None)
            if target_enemy:
                target_hp = target_enemy.get("hp", 100)
                if action.damage_dealt > target_hp * 2:
                    return CombatError(
                        error_type=ErrorType.RESOURCE_MANAGEMENT,
                        turn=action.turn,
                        description=f"Overkill : {action.damage_dealt} dégâts sur ennemi à {target_hp} HP",
                        severity=0.3,
                        action_involved=action,
                        recommendation="Utiliser sorts faibles sur ennemis presque morts"
                    )
        
        return None
    
    def _check_defensive_failures(self, combat: CombatRecord) -> List[CombatError]:
        """Vérifie les échecs défensifs"""
        errors = []
        
        # Analyse de la perte de HP
        if combat.snapshots:
            initial_hp = combat.snapshots[0].player_hp
            final_hp = combat.snapshots[-1].player_hp
            hp_lost = initial_hp - final_hp
            
            # Trop de dégâts reçus
            if hp_lost > initial_hp * 0.7:  # Plus de 70% HP perdus
                errors.append(CombatError(
                    error_type=ErrorType.DEFENSIVE_FAILURE,
                    turn=len(combat.actions),
                    description=f"Dégâts excessifs reçus : {hp_lost} HP perdus",
                    severity=0.9,
                    recommendation="Améliorer positionnement et utiliser sorts défensifs"
                ))
        
        return errors
    
    def _find_critical_moments(self, combat: CombatRecord) -> List[int]:
        """Identifie les moments critiques du combat"""
        critical_turns = []
        
        for i, snapshot in enumerate(combat.snapshots):
            # Moment critique : HP bas
            if snapshot.player_hp < 30:
                critical_turns.append(snapshot.turn)
            
            # Moment critique : Ennemi éliminé
            if i > 0:
                prev_enemies = len(combat.snapshots[i-1].enemies)
                curr_enemies = len(snapshot.enemies)
                if curr_enemies < prev_enemies:
                    critical_turns.append(snapshot.turn)
        
        return critical_turns
    
    def _calculate_efficiency(self, combat: CombatRecord) -> float:
        """Calcule l'efficacité globale du combat"""
        if combat.outcome != CombatOutcome.VICTORY:
            return 0.3  # Défaite = faible efficacité
        
        # Facteurs d'efficacité
        duration_score = 1.0 - min(1.0, combat.duration / 300)  # Pénalité si > 5min
        damage_efficiency = combat.total_damage_dealt / max(1, combat.total_damage_received)
        damage_efficiency_score = min(1.0, damage_efficiency / 3)  # Ratio 3:1 = parfait
        
        # Score final
        efficiency = (duration_score * 0.4 + damage_efficiency_score * 0.6)
        return round(efficiency, 2)
    
    def _analyze_positioning(self, combat: CombatRecord) -> float:
        """Analyse la qualité du positionnement"""
        if not combat.snapshots:
            return 0.5
        
        good_positions = 0
        total_positions = 0
        
        for snapshot in combat.snapshots:
            for enemy in snapshot.enemies:
                if enemy.get("position"):
                    distance = self._calculate_distance(
                        snapshot.player_position,
                        enemy["position"]
                    )
                    
                    # Distance optimale : 3-5 cases
                    if 3 <= distance <= 5:
                        good_positions += 1
                    total_positions += 1
        
        if total_positions == 0:
            return 0.5
        
        return round(good_positions / total_positions, 2)
    
    def _analyze_spell_usage(self, combat: CombatRecord) -> float:
        """Analyse l'utilisation des sorts"""
        spell_actions = [a for a in combat.actions if a.action_type == "spell"]
        
        if not spell_actions:
            return 0.0
        
        # Calcul ratio dégâts/PA moyen
        total_damage = sum(a.damage_dealt for a in spell_actions)
        total_ap = sum(a.ap_cost for a in spell_actions)
        
        if total_ap == 0:
            return 0.0
        
        avg_ratio = total_damage / total_ap
        
        # Score basé sur ratio (20+ = excellent)
        score = min(1.0, avg_ratio / 20)
        return round(score, 2)
    
    def _analyze_resource_management(self, combat: CombatRecord) -> float:
        """Analyse la gestion des ressources (PA/PM)"""
        if not combat.snapshots:
            return 0.5
        
        # Calcul utilisation moyenne des PA
        ap_usage = []
        for snapshot in combat.snapshots:
            if snapshot.player_ap < 6:  # PA max typique
                ap_used = 6 - snapshot.player_ap
                ap_usage.append(ap_used / 6)
        
        if not ap_usage:
            return 0.5
        
        # Score basé sur utilisation (>80% = bon)
        avg_usage = np.mean(ap_usage)
        return round(avg_usage, 2)
    
    def _generate_recommendations(self, combat: CombatRecord, 
                                 errors: List[CombatError]) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Recommandations basées sur les erreurs
        error_types = [e.error_type for e in errors]
        
        if ErrorType.POSITIONING in error_types:
            recommendations.append("🎯 Améliorer positionnement : maintenir distance 3-5 cases")
        
        if ErrorType.SPELL_CHOICE in error_types:
            recommendations.append("⚔️ Optimiser choix sorts : privilégier ratio dégâts/PA élevé")
        
        if ErrorType.RESOURCE_MANAGEMENT in error_types:
            recommendations.append("💎 Éviter overkill : adapter puissance sorts aux HP ennemis")
        
        if ErrorType.DEFENSIVE_FAILURE in error_types:
            recommendations.append("🛡️ Renforcer défense : utiliser sorts défensifs et meilleures positions")
        
        # Recommandations basées sur l'outcome
        if combat.outcome == CombatOutcome.DEFEAT:
            recommendations.append("⚠️ Combat trop difficile : augmenter niveau ou améliorer équipement")
        
        elif combat.outcome == CombatOutcome.VICTORY:
            if combat.duration > 180:  # > 3 minutes
                recommendations.append("⏱️ Combat trop long : utiliser sorts plus puissants")
        
        return recommendations
    
    def _identify_patterns(self, combat: CombatRecord) -> List[str]:
        """Identifie les patterns de comportement"""
        patterns = []
        
        # Pattern : Sorts favoris
        spell_usage = defaultdict(int)
        for action in combat.actions:
            if action.action_type == "spell" and action.spell_name:
                spell_usage[action.spell_name] += 1
        
        if spell_usage:
            most_used = max(spell_usage, key=spell_usage.get)
            patterns.append(f"Sort le plus utilisé : {most_used} ({spell_usage[most_used]}x)")
        
        # Pattern : Style de jeu
        movement_count = sum(1 for a in combat.actions if a.action_type == "movement")
        spell_count = sum(1 for a in combat.actions if a.action_type == "spell")
        
        if movement_count > spell_count:
            patterns.append("Style : Mobile (beaucoup de déplacements)")
        else:
            patterns.append("Style : Statique (peu de déplacements)")
        
        return patterns
    
    def _extract_lessons(self, combat: CombatRecord, report: AfterActionReport) -> List[str]:
        """Extrait les leçons apprises"""
        lessons = []
        
        if combat.outcome == CombatOutcome.VICTORY:
            if report.efficiency_score > 0.8:
                lessons.append("✅ Excellente exécution : reproduire cette stratégie")
            
            if report.positioning_score > 0.8:
                lessons.append("✅ Bon positionnement : maintenir distances optimales")
        
        else:  # Défaite
            critical_errors = [e for e in report.errors_identified if e.severity > 0.7]
            if critical_errors:
                lessons.append(f"❌ Erreur critique : {critical_errors[0].description}")
        
        return lessons
    
    def _get_snapshot_at_turn(self, combat: CombatRecord, turn: int) -> Optional[CombatSnapshot]:
        """Récupère le snapshot à un tour donné"""
        for snapshot in combat.snapshots:
            if snapshot.turn == turn:
                return snapshot
        return None
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance entre deux positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5


class CombatLearner:
    """Système d'apprentissage des combats"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CombatLearner")
        self.victory_patterns = defaultdict(list)
        self.defeat_patterns = defaultdict(list)
    
    def learn_from_combat(self, combat: CombatRecord, report: AfterActionReport):
        """Apprend d'un combat"""
        # Classification par type d'ennemi
        enemy_key = "_".join(sorted(combat.enemy_types))
        
        if combat.outcome == CombatOutcome.VICTORY:
            self.victory_patterns[enemy_key].append({
                "combat_id": combat.combat_id,
                "efficiency": report.efficiency_score,
                "strategy": self._extract_strategy(combat),
                "duration": combat.duration
            })
        
        else:
            self.defeat_patterns[enemy_key].append({
                "combat_id": combat.combat_id,
                "errors": [e.error_type.value for e in report.errors_identified],
                "critical_moment": report.critical_moments[0] if report.critical_moments else None
            })
    
    def get_best_strategy(self, enemy_types: List[str]) -> Optional[Dict]:
        """Récupère la meilleure stratégie pour un type d'ennemi"""
        enemy_key = "_".join(sorted(enemy_types))
        
        victories = self.victory_patterns.get(enemy_key, [])
        if not victories:
            return None
        
        # Tri par efficacité
        best_victory = max(victories, key=lambda x: x["efficiency"])
        return best_victory["strategy"]
    
    def _extract_strategy(self, combat: CombatRecord) -> Dict:
        """Extrait la stratégie utilisée"""
        spell_sequence = [
            a.spell_name for a in combat.actions 
            if a.action_type == "spell" and a.spell_name
        ]
        
        return {
            "spell_sequence": spell_sequence[:5],  # 5 premiers sorts
            "avg_distance": self._calculate_avg_distance(combat),
            "aggressive": combat.total_damage_dealt > combat.total_damage_received * 2
        }
    
    def _calculate_avg_distance(self, combat: CombatRecord) -> float:
        """Calcule la distance moyenne aux ennemis"""
        distances = []
        for snapshot in combat.snapshots:
            for enemy in snapshot.enemies:
                if enemy.get("position"):
                    dist = ((snapshot.player_position[0] - enemy["position"][0])**2 + 
                           (snapshot.player_position[1] - enemy["position"][1])**2) ** 0.5
                    distances.append(dist)
        
        return np.mean(distances) if distances else 0.0


class AfterActionReportModule(IModule):
    """
    Module After Action Report
    Analyse et apprentissage des combats
    """
    
    def __init__(self, name: str = "after_action_report"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Composants
        self.analyzer = CombatAnalyzer()
        self.learner = CombatLearner()
        
        # Historique
        self.combat_records: deque = deque(maxlen=1000)
        self.reports: Dict[str, AfterActionReport] = {}
        
        # Métriques
        self.metrics = {
            "combats_analyzed": 0,
            "victories": 0,
            "defeats": 0,
            "avg_efficiency": 0.0,
            "lessons_learned": 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING
            
            # Chargement historique
            self._load_combat_history()
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info("Module After Action Report initialisé")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    def analyze_combat(self, combat_record: CombatRecord) -> AfterActionReport:
        """Analyse un combat et génère un rapport"""
        try:
            # Analyse
            report = self.analyzer.analyze_combat(combat_record)
            
            # Apprentissage
            self.learner.learn_from_combat(combat_record, report)
            
            # Sauvegarde
            self.combat_records.append(combat_record)
            self.reports[combat_record.combat_id] = report
            
            # Mise à jour métriques
            self.metrics["combats_analyzed"] += 1
            if combat_record.outcome == CombatOutcome.VICTORY:
                self.metrics["victories"] += 1
            else:
                self.metrics["defeats"] += 1
            
            self.metrics["lessons_learned"] += len(report.lessons_learned)
            
            # Log rapport
            self._log_report(combat_record, report)
            
            return report
        
        except Exception as e:
            self.logger.error(f"Erreur analyse combat: {e}")
            return AfterActionReport(
                combat_id=combat_record.combat_id,
                outcome=combat_record.outcome
            )
    
    def _log_report(self, combat: CombatRecord, report: AfterActionReport):
        """Log le rapport d'analyse"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 AFTER ACTION REPORT - Combat {combat.combat_id}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Résultat : {combat.outcome.value.upper()}")
        self.logger.info(f"Durée : {combat.duration:.1f}s | Tours : {combat.turns_count}")
        self.logger.info(f"\n📈 SCORES DE PERFORMANCE:")
        self.logger.info(f"  Efficacité : {report.efficiency_score:.0%}")
        self.logger.info(f"  Positionnement : {report.positioning_score:.0%}")
        self.logger.info(f"  Utilisation sorts : {report.spell_usage_score:.0%}")
        self.logger.info(f"  Gestion ressources : {report.resource_management_score:.0%}")
        
        if report.errors_identified:
            self.logger.info(f"\n⚠️ ERREURS IDENTIFIÉES ({len(report.errors_identified)}):")
            for error in report.errors_identified[:3]:  # Top 3
                self.logger.info(f"  • {error.description} (Sévérité: {error.severity:.0%})")
        
        if report.recommendations:
            self.logger.info(f"\n💡 RECOMMANDATIONS:")
            for rec in report.recommendations:
                self.logger.info(f"  {rec}")
        
        if report.lessons_learned:
            self.logger.info(f"\n🎓 LEÇONS APPRISES:")
            for lesson in report.lessons_learned:
                self.logger.info(f"  {lesson}")
        
        self.logger.info(f"{'='*60}\n")
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "combats_analyzed": len(self.combat_records),
            "reports_generated": len(self.reports),
            "metrics": self.metrics
        }
    
    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            self._save_combat_history()
            self.logger.info("Module After Action Report nettoyé")
        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")
    
    def _load_combat_history(self):
        """Charge l'historique des combats"""
        # Placeholder
        pass
    
    def _save_combat_history(self):
        """Sauvegarde l'historique"""
        # Placeholder
        pass
    
    def get_combat_statistics(self) -> Dict[str, Any]:
        """Génère des statistiques sur les combats"""
        if not self.combat_records:
            return {}
        
        victories = [c for c in self.combat_records if c.outcome == CombatOutcome.VICTORY]
        defeats = [c for c in self.combat_records if c.outcome == CombatOutcome.DEFEAT]
        
        return {
            "total_combats": len(self.combat_records),
            "win_rate": len(victories) / len(self.combat_records) if self.combat_records else 0,
            "avg_duration": np.mean([c.duration for c in self.combat_records]),
            "avg_efficiency": np.mean([
                self.reports[c.combat_id].efficiency_score 
                for c in self.combat_records 
                if c.combat_id in self.reports
            ]) if self.reports else 0,
            "most_common_errors": self._get_most_common_errors(),
            "best_strategies": self._get_best_strategies()
        }
    
    def _get_most_common_errors(self) -> List[str]:
        """Récupère les erreurs les plus fréquentes"""
        error_counts = defaultdict(int)
        for report in self.reports.values():
            for error in report.errors_identified:
                error_counts[error.error_type.value] += 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _get_best_strategies(self) -> List[Dict]:
        """Récupère les meilleures stratégies"""
        # Placeholder
        return []
