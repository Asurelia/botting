"""
Combat Engine - Moteur de combat intelligent
Gère le combat tactique au tour par tour avec IA de décision
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .combo_library import ComboLibrary, CharacterClass, Spell, SpellCombo
from .after_action_report import AARManager, create_aar_manager

logger = logging.getLogger(__name__)


class CombatPhase(Enum):
    """Phases du combat"""
    PREPARATION = "preparation"  # Début du tour
    POSITIONING = "positioning"  # Déplacements tactiques
    BUFFING = "buffing"  # Application de buffs
    ATTACKING = "attacking"  # Attaques
    FINISHING = "finishing"  # Fin de tour


class TargetPriority(Enum):
    """Priorité des cibles"""
    LOWEST_HP = "lowest_hp"  # Cible avec HP les plus bas
    HIGHEST_THREAT = "highest_threat"  # Cible la plus menaçante
    NEAREST = "nearest"  # Cible la plus proche
    WEAKEST_DEFENSE = "weakest_defense"  # Défense la plus faible


@dataclass
class CombatEntity:
    """Entité en combat (joueur ou ennemi)"""
    id: str
    name: str
    hp: int
    max_hp: int
    pa: int
    max_pa: int
    pm: int
    max_pm: int
    position: Tuple[int, int]  # (x, y) sur la grille
    is_ally: bool
    level: int = 1

    # Stats
    strength: int = 0
    intelligence: int = 0
    agility: int = 0
    resistance: Dict[str, int] = None

    # État
    is_alive: bool = True
    buffs: List[str] = None
    debuffs: List[str] = None

    def __post_init__(self):
        if self.resistance is None:
            self.resistance = {}
        if self.buffs is None:
            self.buffs = []
        if self.debuffs is None:
            self.debuffs = []

    @property
    def hp_percent(self) -> float:
        """Pourcentage de HP"""
        if self.max_hp == 0:
            return 0.0
        return (self.hp / self.max_hp) * 100.0

    @property
    def threat_score(self) -> float:
        """Score de menace (pour priorisation)"""
        # Threat = HP% * Level * (1 + buffs count)
        return self.hp_percent * self.level * (1 + len(self.buffs))


@dataclass
class CombatState:
    """État complet du combat"""
    turn_number: int = 0
    my_turn: bool = False
    phase: CombatPhase = CombatPhase.PREPARATION

    # Entités
    player: Optional[CombatEntity] = None
    allies: List[CombatEntity] = None
    enemies: List[CombatEntity] = None

    # Historique
    actions_history: List[Dict] = None
    damage_dealt_total: int = 0
    damage_taken_total: int = 0

    def __post_init__(self):
        if self.allies is None:
            self.allies = []
        if self.enemies is None:
            self.enemies = []
        if self.actions_history is None:
            self.actions_history = []

    @property
    def enemies_alive(self) -> List[CombatEntity]:
        """Ennemis vivants"""
        return [e for e in self.enemies if e.is_alive]

    @property
    def allies_alive(self) -> List[CombatEntity]:
        """Alliés vivants"""
        return [a for a in self.allies if a.is_alive]


class CombatEngine:
    """
    Moteur de combat principal

    Responsabilités:
    - Analyse de l'état du combat
    - Sélection de cibles intelligente
    - Choix de combos optimaux
    - Gestion des phases de combat
    - Logging des actions
    """

    def __init__(self, character_class: CharacterClass, aar_manager: Optional[AARManager] = None):
        self.character_class = character_class
        self.combo_library = ComboLibrary(character_class)
        self.aar_manager = aar_manager or create_aar_manager()

        # État du combat
        self.current_combat: Optional[CombatState] = None
        self.combat_start_time: float = 0.0

        # Configuration
        self.config = {
            'target_priority': TargetPriority.LOWEST_HP,
            'min_hp_threshold': 30.0,  # Seuil HP pour panique
            'save_pa_for_escape': 2,  # PA à garder pour fuite
            'prefer_combos': True,  # Préférer combos vs sorts simples
            'max_combo_length': 4,  # Longueur max des combos
        }

        logger.info(f"Combat Engine initialisé (classe: {character_class.value})")

    def start_combat(self, combat_state: CombatState):
        """Démarre un nouveau combat"""
        self.current_combat = combat_state
        self.combat_start_time = time.time()

        logger.info(f"Combat démarré: {len(combat_state.enemies)} ennemis")
        logger.info(f"Player: {combat_state.player.name} (HP: {combat_state.player.hp}/{combat_state.player.max_hp})")

    def end_combat(self, victory: bool):
        """Termine le combat"""
        if not self.current_combat:
            return

        duration = time.time() - self.combat_start_time

        # Créer rapport AAR
        self.aar_manager.create_report(
            combat_duration=duration,
            player_level=self.current_combat.player.level,
            enemies_defeated=len([e for e in self.current_combat.enemies if not e.is_alive]),
            damage_dealt=self.current_combat.damage_dealt_total,
            damage_taken=self.current_combat.damage_taken_total,
            victory=victory,
            character_class=self.character_class.value
        )

        logger.info(f"Combat terminé: {'VICTOIRE' if victory else 'DÉFAITE'} (durée: {duration:.1f}s)")

        self.current_combat = None

    def decide_action(self, combat_state: CombatState) -> Optional[Dict[str, Any]]:
        """
        Décide de l'action à effectuer ce tour

        Returns:
            Dict avec:
                - action_type: 'spell', 'move', 'pass', 'flee'
                - spell_id: ID du sort à lancer
                - target_id: ID de la cible
                - position: Position pour déplacement
                - combo_name: Nom du combo si applicable
                - reason: Raison de la décision
        """
        self.current_combat = combat_state

        # === PHASE 1: SURVIE ===
        if combat_state.player.hp_percent < self.config['min_hp_threshold']:
            logger.warning(f"HP critique: {combat_state.player.hp_percent:.1f}%")
            return self._decide_survival_action(combat_state)

        # === PHASE 2: POSITIONNEMENT ===
        if combat_state.phase == CombatPhase.POSITIONING:
            return self._decide_movement_action(combat_state)

        # === PHASE 3: COMBAT OFFENSIF ===
        if combat_state.my_turn and combat_state.player.pa >= 2:
            return self._decide_offensive_action(combat_state)

        # === PHASE 4: PASSER LE TOUR ===
        return {
            'action_type': 'pass',
            'reason': 'Pas assez de PA ou pas mon tour'
        }

    def _decide_survival_action(self, combat_state: CombatState) -> Dict[str, Any]:
        """Décision en mode survie (HP bas)"""
        player = combat_state.player

        # Option 1: Fuite si possible
        if player.pm >= 3 and len(combat_state.enemies_alive) > 2:
            return {
                'action_type': 'flee',
                'reason': f'HP critique ({player.hp_percent:.1f}%), fuite tactique'
            }

        # Option 2: Sort de soin si disponible
        healing_spells = self.combo_library.get_available_healing_spells()
        if healing_spells and player.pa >= healing_spells[0].pa_cost:
            return {
                'action_type': 'spell',
                'spell_id': healing_spells[0].spell_id,
                'target_id': player.id,
                'reason': 'Soin d\'urgence'
            }

        # Option 3: Attaque defensive (tuer la plus grande menace)
        target = self._select_target(combat_state, TargetPriority.HIGHEST_THREAT)
        if target:
            spell = self._select_best_spell(player.pa, target)
            if spell:
                return {
                    'action_type': 'spell',
                    'spell_id': spell.spell_id,
                    'target_id': target.id,
                    'reason': 'Éliminer la menace principale'
                }

        return {'action_type': 'pass', 'reason': 'Aucune action de survie possible'}

    def _decide_movement_action(self, combat_state: CombatState) -> Dict[str, Any]:
        """Décision de déplacement tactique"""
        player = combat_state.player

        # Trouver position optimale
        optimal_position = self._find_optimal_position(combat_state)

        if optimal_position and optimal_position != player.position:
            pm_needed = self._calculate_pm_needed(player.position, optimal_position)

            if pm_needed <= player.pm:
                return {
                    'action_type': 'move',
                    'position': optimal_position,
                    'reason': 'Positionnement tactique optimal'
                }

        return {
            'action_type': 'pass',
            'phase_transition': CombatPhase.ATTACKING,
            'reason': 'Position actuelle acceptable'
        }

    def _decide_offensive_action(self, combat_state: CombatState) -> Dict[str, Any]:
        """Décision d'attaque offensive"""
        player = combat_state.player

        # Sélectionner la cible
        target = self._select_target(combat_state, self.config['target_priority'])

        if not target:
            return {'action_type': 'pass', 'reason': 'Aucune cible valide'}

        # === OPTION 1: COMBO OPTIMAL ===
        if self.config['prefer_combos']:
            combo = self._select_best_combo(player.pa, target, combat_state)

            if combo:
                return {
                    'action_type': 'combo',
                    'combo_name': combo.name,
                    'spells': [s.spell_id for s in combo.spells],
                    'target_id': target.id,
                    'estimated_damage': combo.total_damage,
                    'reason': f'Combo optimal: {combo.name} (damage: {combo.total_damage})'
                }

        # === OPTION 2: SORT SIMPLE ===
        spell = self._select_best_spell(player.pa, target)

        if spell:
            return {
                'action_type': 'spell',
                'spell_id': spell.spell_id,
                'target_id': target.id,
                'estimated_damage': spell.damage,
                'reason': f'Sort optimal: {spell.name} (damage: {spell.damage})'
            }

        return {'action_type': 'pass', 'reason': 'Pas de sort disponible'}

    def _select_target(self, combat_state: CombatState, priority: TargetPriority) -> Optional[CombatEntity]:
        """Sélectionne la meilleure cible selon priorité"""
        enemies = combat_state.enemies_alive

        if not enemies:
            return None

        if priority == TargetPriority.LOWEST_HP:
            return min(enemies, key=lambda e: e.hp_percent)

        elif priority == TargetPriority.HIGHEST_THREAT:
            return max(enemies, key=lambda e: e.threat_score)

        elif priority == TargetPriority.NEAREST:
            player_pos = combat_state.player.position
            return min(enemies, key=lambda e: self._distance(player_pos, e.position))

        elif priority == TargetPriority.WEAKEST_DEFENSE:
            return min(enemies, key=lambda e: sum(e.resistance.values()))

        return enemies[0]

    def _select_best_combo(self, available_pa: int, target: CombatEntity, combat_state: CombatState) -> Optional[SpellCombo]:
        """Sélectionne le meilleur combo disponible"""
        available_combos = self.combo_library.get_available_combos(
            available_pa=available_pa,
            max_length=self.config['max_combo_length']
        )

        if not available_combos:
            return None

        # Trier par damage total
        available_combos.sort(key=lambda c: c.total_damage, reverse=True)

        # Prendre le combo avec le meilleur damage
        best_combo = available_combos[0]

        logger.debug(f"Combo sélectionné: {best_combo.name} (damage: {best_combo.total_damage})")

        return best_combo

    def _select_best_spell(self, available_pa: int, target: CombatEntity) -> Optional[Spell]:
        """Sélectionne le meilleur sort simple"""
        all_spells = self.combo_library.get_all_spells()

        # Filtrer sorts disponibles avec PA disponible
        available_spells = [s for s in all_spells if s.pa_cost <= available_pa]

        if not available_spells:
            return None

        # Trier par damage
        available_spells.sort(key=lambda s: s.damage, reverse=True)

        best_spell = available_spells[0]

        logger.debug(f"Sort sélectionné: {best_spell.name} (damage: {best_spell.damage})")

        return best_spell

    def _find_optimal_position(self, combat_state: CombatState) -> Optional[Tuple[int, int]]:
        """Trouve la position optimale sur la grille"""
        player_pos = combat_state.player.position
        enemies = combat_state.enemies_alive

        if not enemies:
            return None

        # Calcul position médiane des ennemis
        enemy_positions = [e.position for e in enemies]
        median_x = int(sum(p[0] for p in enemy_positions) / len(enemy_positions))
        median_y = int(sum(p[1] for p in enemy_positions) / len(enemy_positions))

        # Position optimale = distance moyenne des ennemis (portée moyenne)
        optimal_range = 5  # Portée moyenne des sorts

        # Trouver position à ~5 cases du centre des ennemis
        # (simplification - dans le vrai jeu, utiliser pathfinding)
        dx = 1 if median_x > player_pos[0] else -1
        dy = 1 if median_y > player_pos[1] else -1

        optimal_pos = (
            player_pos[0] + dx * 2,
            player_pos[1] + dy * 2
        )

        return optimal_pos

    def _calculate_pm_needed(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Calcule PM nécessaire (distance Manhattan)"""
        return abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distance euclidienne"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def log_action(self, action: Dict[str, Any], success: bool, result: Optional[Dict] = None):
        """Log une action de combat"""
        if not self.current_combat:
            return

        action_log = {
            'turn': self.current_combat.turn_number,
            'timestamp': time.time(),
            'action': action,
            'success': success,
            'result': result or {}
        }

        self.current_combat.actions_history.append(action_log)

        # Update damage totals
        if result:
            if 'damage_dealt' in result:
                self.current_combat.damage_dealt_total += result['damage_dealt']
            if 'damage_taken' in result:
                self.current_combat.damage_taken_total += result['damage_taken']

        logger.info(f"[Turn {self.current_combat.turn_number}] {action.get('action_type')} - {action.get('reason')} - {'SUCCESS' if success else 'FAIL'}")

    def get_combat_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du combat en cours"""
        if not self.current_combat:
            return {}

        return {
            'turn': self.current_combat.turn_number,
            'duration': time.time() - self.combat_start_time,
            'player_hp': self.current_combat.player.hp_percent,
            'enemies_alive': len(self.current_combat.enemies_alive),
            'damage_dealt': self.current_combat.damage_dealt_total,
            'damage_taken': self.current_combat.damage_taken_total,
            'actions_count': len(self.current_combat.actions_history)
        }


def create_combat_engine(character_class: CharacterClass) -> CombatEngine:
    """Factory function"""
    return CombatEngine(character_class)


# Helpers pour créer entités de combat
def create_player_entity(name: str, hp: int, max_hp: int, pa: int, pm: int, position: Tuple[int, int], level: int = 1) -> CombatEntity:
    """Crée une entité joueur"""
    return CombatEntity(
        id="player",
        name=name,
        hp=hp,
        max_hp=max_hp,
        pa=pa,
        max_pa=pa,
        pm=pm,
        max_pm=pm,
        position=position,
        is_ally=True,
        level=level
    )


def create_enemy_entity(id: str, name: str, hp: int, max_hp: int, position: Tuple[int, int], level: int = 1) -> CombatEntity:
    """Crée une entité ennemie"""
    return CombatEntity(
        id=id,
        name=name,
        hp=hp,
        max_hp=max_hp,
        pa=6,
        max_pa=6,
        pm=3,
        max_pm=3,
        position=position,
        is_ally=False,
        level=level
    )


def create_combat_state(player: CombatEntity, enemies: List[CombatEntity]) -> CombatState:
    """Crée un état de combat"""
    return CombatState(
        turn_number=1,
        my_turn=True,
        phase=CombatPhase.PREPARATION,
        player=player,
        enemies=enemies
    )
