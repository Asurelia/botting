"""
Bibliothèque de Combos de Sorts pour TacticalBot
Phase 2 du Projet Augmenta

Fonctionnalités:
- Stockage et gestion des combos de sorts
- Génération de séquences optimales
- Évaluation de l'efficacité des combos
- Adaptation par classe de personnage
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
from ...state.realtime_state import GameState, CombatState, Character, Spell, CharacterClass


@dataclass
class SpellCombo:
    """Représente un combo de sorts"""
    id: str
    name: str
    character_class: CharacterClass
    spell_sequence: List[str]  # Liste des IDs de sorts
    conditions: Dict[str, Any]  # Conditions d'activation
    effects: Dict[str, float]   # Effets du combo
    cooldown: float             # Temps de rechargement en secondes
    last_used: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    average_damage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComboExecution:
    """Exécution d'un combo"""
    combo_id: str
    start_time: float
    spell_order: List[str]
    current_spell_index: int = 0
    total_damage_dealt: float = 0.0
    targets_hit: int = 0
    success: bool = False
    execution_time: float = 0.0


class ComboGenerator:
    """Générateur de combos de sorts"""

    def __init__(self):
        self.base_combos = self._initialize_base_combos()
        self.generated_combos = {}
        self.combo_performance = defaultdict(list)

    def _initialize_base_combos(self) -> Dict[CharacterClass, List[SpellCombo]]:
        """Initialise les combos de base par classe"""
        return {
            CharacterClass.IOP: [
                SpellCombo(
                    id="iop_sword_celestial",
                    name="Épée Céleste",
                    character_class=CharacterClass.IOP,
                    spell_sequence=["sword_celestial", "colere_iop"],
                    conditions={"pa_available": 8, "target_distance": {"min": 1, "max": 3}},
                    effects={"damage_bonus": 25, "crit_chance": 0.15},
                    cooldown=30.0
                ),
                SpellCombo(
                    id="iop_vitality_combo",
                    name="Vitalité Iop",
                    character_class=CharacterClass.IOP,
                    spell_sequence=["vitality", "sword_celestial"],
                    conditions={"pa_available": 6, "ally_low_hp": True},
                    effects={"heal_bonus": 30, "damage_reduction": 0.2},
                    cooldown=45.0
                )
            ],
            CharacterClass.CRA: [
                SpellCombo(
                    id="cra_explosive_arrow",
                    name="Flèche Explosive",
                    character_class=CharacterClass.CRA,
                    spell_sequence=["fleche_enflammee", "fleche_explosive"],
                    conditions={"pa_available": 7, "target_distance": {"min": 4, "max": 8}},
                    effects={"damage_bonus": 20, "aoe_damage": 15},
                    cooldown=25.0
                ),
                SpellCombo(
                    id="cra_precision_shot",
                    name="Tir de Précision",
                    character_class=CharacterClass.CRA,
                    spell_sequence=["fleche_magique", "fleche_perforante"],
                    conditions={"pa_available": 6, "target_distance": {"min": 3, "max": 6}},
                    effects={"crit_chance": 0.3, "accuracy_bonus": 0.2},
                    cooldown=20.0
                )
            ],
            CharacterClass.ENIRIPSA: [
                SpellCombo(
                    id="eniripsa_healing_word",
                    name="Mot Soignant",
                    character_class=CharacterClass.ENIRIPSA,
                    spell_sequence=["mot_soignant", "mot_regenerant"],
                    conditions={"pa_available": 6, "allies_low_hp": 2},
                    effects={"heal_bonus": 30, "regen_effect": 20},
                    cooldown=35.0
                ),
                SpellCombo(
                    id="eniripsa_protection_combo",
                    name="Protection Sacrée",
                    character_class=CharacterClass.ENIRIPSA,
                    spell_sequence=["protection_sacree", "mot_soignant"],
                    conditions={"pa_available": 8, "allies_in_danger": True},
                    effects={"damage_reduction": 0.4, "heal_bonus": 25},
                    cooldown=60.0
                )
            ],
            CharacterClass.SRAM: [
                SpellCombo(
                    id="sram_invisibility_combo",
                    name="Attaque Invisible",
                    character_class=CharacterClass.SRAM,
                    spell_sequence=["invisibilite", "piege_insidieux"],
                    conditions={"pa_available": 6, "target_distance": {"min": 1, "max": 4}},
                    effects={"damage_bonus": 35, "crit_chance": 0.25},
                    cooldown=40.0
                )
            ],
            CharacterClass.SACRIEUR: [
                SpellCombo(
                    id="sacrieur_punishment_combo",
                    name="Punition Sacrée",
                    character_class=CharacterClass.SACRIEUR,
                    spell_sequence=["punition", "chatiment"],
                    conditions={"pa_available": 8, "low_hp": True},
                    effects={"damage_bonus": 40, "self_damage": 0.3},
                    cooldown=50.0
                )
            ],
            CharacterClass.FECA: [
                SpellCombo(
                    id="feca_shield_combo",
                    name="Bouclier Runique",
                    character_class=CharacterClass.FECA,
                    spell_sequence=["bouclier", "armure_runique"],
                    conditions={"pa_available": 7, "allies_count": {"min": 2}},
                    effects={"damage_reduction": 0.5, "armor_bonus": 0.3},
                    cooldown=45.0
                )
            ]
        }

    def generate_combo(self, character_class: CharacterClass, available_spells: List[str],
                      game_context: Dict[str, Any]) -> Optional[SpellCombo]:
        """Génère un combo adapté au contexte"""
        base_combos = self.base_combos.get(character_class, [])

        # Filtrage des combos possibles
        viable_combos = []
        for combo in base_combos:
            if self._is_combo_viable(combo, available_spells, game_context):
                viable_combos.append(combo)

        if not viable_combos:
            return None

        # Sélection du meilleur combo
        best_combo = max(viable_combos, key=lambda c: self._evaluate_combo_effectiveness(c, game_context))

        # Création d'une variante si nécessaire
        if len(viable_combos) > 1:
            best_combo = self._create_combo_variant(best_combo, game_context)

        return best_combo

    def _is_combo_viable(self, combo: SpellCombo, available_spells: List[str],
                        game_context: Dict[str, Any]) -> bool:
        """Vérifie si un combo est viable"""
        # Vérification des sorts disponibles
        for spell_id in combo.spell_sequence:
            if spell_id not in available_spells:
                return False

        # Vérification des conditions
        pa_available = game_context.get("pa_available", 0)
        if pa_available < combo.conditions.get("pa_available", 0):
            return False

        # Vérification de la distance
        target_distance = game_context.get("target_distance", 0)
        distance_condition = combo.conditions.get("target_distance", {})
        if distance_condition:
            min_dist = distance_condition.get("min", 0)
            max_dist = distance_condition.get("max", 99)
            if not (min_dist <= target_distance <= max_dist):
                return False

        # Vérification des alliés en danger
        if combo.conditions.get("allies_low_hp", False):
            allies_low_hp = game_context.get("allies_low_hp", 0)
            if allies_low_hp < 2:
                return False

        return True

    def _evaluate_combo_effectiveness(self, combo: SpellCombo, game_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un combo"""
        score = 0.0

        # Score basé sur les effets
        damage_bonus = combo.effects.get("damage_bonus", 0)
        heal_bonus = combo.effects.get("heal_bonus", 0)
        score += damage_bonus * 0.3 + heal_bonus * 0.4

        # Score basé sur les conditions favorables
        if game_context.get("target_weak", False):
            score += 20

        if game_context.get("allies_in_danger", False) and heal_bonus > 0:
            score += 15

        # Pénalité pour le cooldown
        cooldown_penalty = min(10, combo.cooldown / 10)
        score -= cooldown_penalty

        return score

    def _create_combo_variant(self, base_combo: SpellCombo, game_context: Dict[str, Any]) -> SpellCombo:
        """Crée une variante d'un combo"""
        variant_id = f"{base_combo.id}_variant_{int(time.time())}"

        # Modification légère des effets
        variant_effects = base_combo.effects.copy()
        if game_context.get("high_damage_needed", False):
            variant_effects["damage_bonus"] = variant_effects.get("damage_bonus", 0) * 1.2

        return SpellCombo(
            id=variant_id,
            name=f"{base_combo.name} (Variant)",
            character_class=base_combo.character_class,
            spell_sequence=base_combo.spell_sequence,
            conditions=base_combo.conditions,
            effects=variant_effects,
            cooldown=base_combo.cooldown,
            metadata={"base_combo": base_combo.id, "variant_reason": "contextual"}
        )


class ComboExecutor:
    """Exécuteur de combos de sorts"""

    def __init__(self):
        self.active_combos = {}
        self.execution_history = deque(maxlen=500)

    def start_combo_execution(self, combo: SpellCombo, game_state: GameState) -> Optional[ComboExecution]:
        """Démarre l'exécution d'un combo"""
        if combo.id in self.active_combos:
            return None  # Combo déjà en cours

        execution = ComboExecution(
            combo_id=combo.id,
            start_time=time.time(),
            spell_order=combo.spell_sequence.copy()
        )

        self.active_combos[combo.id] = execution
        return execution

    def execute_next_spell(self, combo_id: str, game_state: GameState) -> Optional[str]:
        """Exécute le prochain sort du combo"""
        if combo_id not in self.active_combos:
            return None

        execution = self.active_combos[combo_id]

        if execution.current_spell_index >= len(execution.spell_order):
            self._finish_combo_execution(combo_id, True)
            return None

        next_spell_id = execution.spell_order[execution.current_spell_index]

        # Vérification de la disponibilité du sort
        if not self._can_cast_spell(next_spell_id, game_state):
            self._finish_combo_execution(combo_id, False)
            return None

        execution.current_spell_index += 1
        return next_spell_id

    def _can_cast_spell(self, spell_id: str, game_state: GameState) -> bool:
        """Vérifie si un sort peut être lancé"""
        if spell_id not in game_state.character.spells:
            return False

        spell = game_state.character.spells[spell_id]
        return game_state.character.can_cast_spell(spell.spell_id)

    def _finish_combo_execution(self, combo_id: str, success: bool):
        """Termine l'exécution d'un combo"""
        if combo_id not in self.active_combos:
            return

        execution = self.active_combos[combo_id]
        execution.success = success
        execution.execution_time = time.time() - execution.start_time

        self.execution_history.append(execution)
        del self.active_combos[combo_id]

    def get_active_combo_progress(self, combo_id: str) -> Optional[Dict[str, Any]]:
        """Retourne la progression d'un combo actif"""
        if combo_id not in self.active_combos:
            return None

        execution = self.active_combos[combo_id]
        return {
            "combo_id": combo_id,
            "current_spell_index": execution.current_spell_index,
            "total_spells": len(execution.spell_order),
            "progress_percentage": (execution.current_spell_index / len(execution.spell_order)) * 100,
            "elapsed_time": time.time() - execution.start_time
        }


class ComboLibrary(IModule):
    """
    Bibliothèque de combos de sorts
    """

    def __init__(self, name: str = "combo_library"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Composants
        self.generator = ComboGenerator()
        self.executor = ComboExecutor()

        # Configuration
        self.auto_generate_combos = True
        self.max_active_combos = 3
        self.combo_success_threshold = 0.7

        # Métriques
        self.metrics = {
            "combos_generated": 0,
            "combos_executed": 0,
            "successful_combos": 0,
            "average_combo_damage": 0.0,
            "combo_library_size": 0
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING

            # Configuration
            self.auto_generate_combos = config.get("auto_generate_combos", True)
            self.max_active_combos = config.get("max_active_combos", 3)
            self.combo_success_threshold = config.get("combo_success_threshold", 0.7)

            # Mise à jour des métriques
            self.metrics["combo_library_size"] = sum(
                len(combos) for combos in self.generator.base_combos.values()
            )

            self.status = ModuleStatus.ACTIVE
            self.logger.info("Bibliothèque de combos initialisée")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False

    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour la bibliothèque de combos"""
        if not self.is_active():
            return None

        try:
            # Mise à jour des combos actifs
            self._update_active_combos(game_state)

            # Génération automatique de combos si activée
            if self.auto_generate_combos and game_state.character:
                self._generate_contextual_combos(game_state)

            # Mise à jour des métriques
            self._update_metrics()

            # Retour des données pour partage
            return {
                "combo_data": {
                    "active_combos": len(self.executor.active_combos),
                    "library_size": self.metrics["combo_library_size"],
                    "success_rate": self._calculate_overall_success_rate()
                }
            }

        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None

    def handle_event(self, event: Any) -> bool:
        """Traite les événements"""
        try:
            if hasattr(event, 'type'):
                if event.type == "spell_cast":
                    self._handle_spell_cast_event(event)
                elif event.type == "combo_completed":
                    self._handle_combo_completed_event(event)
                elif event.type == "combat_ended":
                    self._handle_combat_ended_event(event)

            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "metrics": self.metrics,
            "active_combos": list(self.executor.active_combos.keys()),
            "auto_generate": self.auto_generate_combos,
            "max_active": self.max_active_combos
        }

    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            # Sauvegarde des données de combos
            self._save_combo_data()

            self.logger.info("Bibliothèque de combos nettoyée")

        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")

    def generate_combo_for_situation(self, game_state: GameState) -> Optional[SpellCombo]:
        """Génère un combo adapté à la situation actuelle"""
        if not game_state.character:
            return None

        # Préparation du contexte
        context = self._build_game_context(game_state)

        # Génération du combo
        combo = self.generator.generate_combo(
            game_state.character.character_class,
            list(game_state.character.spells.keys()),
            context
        )

        if combo:
            self.metrics["combos_generated"] += 1

        return combo

    def execute_combo(self, combo: SpellCombo, game_state: GameState) -> bool:
        """Démarre l'exécution d'un combo"""
        if len(self.executor.active_combos) >= self.max_active_combos:
            return False

        execution = self.executor.start_combo_execution(combo, game_state)
        if execution:
            self.metrics["combos_executed"] += 1
            return True

        return False

    def get_next_spell_in_combo(self, combo_id: str, game_state: GameState) -> Optional[str]:
        """Retourne le prochain sort à lancer dans un combo"""
        return self.executor.execute_next_spell(combo_id, game_state)

    def _update_active_combos(self, game_state: GameState):
        """Met à jour les combos actifs"""
        # Vérification des timeouts
        current_time = time.time()
        expired_combos = []

        for combo_id, execution in self.executor.active_combos.items():
            if current_time - execution.start_time > 30.0:  # Timeout 30s
                expired_combos.append(combo_id)

        for combo_id in expired_combos:
            self.executor._finish_combo_execution(combo_id, False)

    def _generate_contextual_combos(self, game_state: GameState):
        """Génère des combos contextuels"""
        if not game_state.character or not game_state.combat:
            return

        # Génération basée sur la situation
        context = self._build_game_context(game_state)

        # Génération pour la classe actuelle
        new_combo = self.generator.generate_combo(
            game_state.character.character_class,
            list(game_state.character.spells.keys()),
            context
        )

        if new_combo:
            # Ajout à la bibliothèque générée
            self.generator.generated_combos[new_combo.id] = new_combo

    def _build_game_context(self, game_state: GameState) -> Dict[str, Any]:
        """Construit le contexte du jeu"""
        context = {
            "pa_available": game_state.character.current_pa,
            "pm_available": game_state.character.current_pm,
            "hp_percentage": game_state.character.hp_percentage(),
            "target_distance": 0.0,
            "allies_low_hp": 0,
            "enemies_count": len(game_state.combat.enemies),
            "allies_count": len(game_state.combat.allies)
        }

        # Calcul de la distance à la cible la plus proche
        if game_state.character.position and game_state.combat.enemies:
            min_distance = float('inf')
            for enemy in game_state.combat.enemies:
                if enemy.position:
                    distance = np.sqrt(
                        (enemy.position.x - game_state.character.position.x)**2 +
                        (enemy.position.y - game_state.character.position.y)**2
                    )
                    min_distance = min(min_distance, distance)

            context["target_distance"] = min_distance

        # Comptage des alliés en danger
        for ally in game_state.combat.allies:
            if ally.hp_percentage() < 50:
                context["allies_low_hp"] += 1

        # Contexte additionnel
        context["high_damage_needed"] = context["enemies_count"] > 2
        context["allies_in_danger"] = context["allies_low_hp"] > 0

        return context

    def _update_metrics(self):
        """Met à jour les métriques"""
        # Calcul du taux de succès moyen
        if self.executor.execution_history:
            successful_executions = [e for e in self.executor.execution_history if e.success]
            if successful_executions:
                self.metrics["successful_combos"] = len(successful_executions)
                self.metrics["average_combo_damage"] = np.mean([e.total_damage_dealt for e in successful_executions])

    def _calculate_overall_success_rate(self) -> float:
        """Calcule le taux de succès global"""
        if not self.executor.execution_history:
            return 0.0

        successful = sum(1 for e in self.executor.execution_history if e.success)
        return successful / len(self.executor.execution_history)

    def _handle_spell_cast_event(self, event):
        """Traite un événement de sort lancé"""
        # Mise à jour des combos actifs
        for combo_id in list(self.executor.active_combos.keys()):
            next_spell = self.executor.execute_next_spell(combo_id, event.data.get("game_state"))
            if next_spell:
                # Le sort suivant est prêt à être lancé
                pass

    def _handle_combo_completed_event(self, event):
        """Traite un événement de combo terminé"""
        combo_id = event.data.get("combo_id")
        success = event.data.get("success", False)
        damage = event.data.get("damage", 0.0)

        if combo_id in self.executor.active_combos:
            execution = self.executor.active_combos[combo_id]
            execution.success = success
            execution.total_damage_dealt = damage
            self.executor._finish_combo_execution(combo_id, success)

    def _handle_combat_ended_event(self, event):
        """Traite la fin d'un combat"""
        # Nettoyage des combos actifs
        for combo_id in list(self.executor.active_combos.keys()):
            self.executor._finish_combo_execution(combo_id, False)

    def _save_combo_data(self):
        """Sauvegarde les données de combos"""
        try:
            # Sauvegarde dans un fichier (placeholder)
            pass
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")

    def get_combo_report(self) -> Dict[str, Any]:
        """Génère un rapport sur les combos"""
        return {
            "metrics": self.metrics,
            "active_combos": len(self.executor.active_combos),
            "generated_combos": len(self.generator.generated_combos),
            "execution_history_size": len(self.executor.execution_history),
            "success_rate": self._calculate_overall_success_rate(),
            "most_used_combo": self._get_most_used_combo()
        }

    def _get_most_used_combo(self) -> Optional[str]:
        """Retourne le combo le plus utilisé"""
        if not self.generator.combo_performance:
            return None

        most_used = max(self.generator.combo_performance.keys(),
                       key=lambda k: len(self.generator.combo_performance[k]))
        return most_used