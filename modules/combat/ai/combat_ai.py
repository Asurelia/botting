"""
Système d'IA de Combat Avancé pour DOFUS
Implémente l'algorithme MinMax avec élagage alpha-beta pour des décisions optimales

Fonctionnalités principales:
- Algorithme MinMax avec élagage alpha-beta
- Évaluation multi-critères des positions
- Génération d'arbres d'actions avec simulation
- Scoring avancé avec poids configurables
- Prédiction des actions ennemies
- Gestion des combos et synergies
- Adaptation par classe
- Système d'apprentissage
"""

import time
import copy
import json
import math
import random
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

# Imports des structures du projet
from ...state.realtime_state import (
    GameState, CombatState, Character, CombatEntity, 
    GridPosition, Spell, CharacterClass
)


class ActionType(Enum):
    """Types d'actions possibles en combat"""
    MOVE = "move"
    CAST_SPELL = "cast_spell"
    PASS_TURN = "pass_turn"
    USE_ITEM = "use_item"


class ThreatLevel(Enum):
    """Niveaux de menace"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CombatAction:
    """Représente une action de combat"""
    action_type: ActionType
    target_position: Optional[GridPosition] = None
    target_entity: Optional[CombatEntity] = None
    spell_id: Optional[int] = None
    item_id: Optional[int] = None
    pa_cost: int = 0
    pm_cost: int = 0
    expected_damage: int = 0
    expected_heal: int = 0
    success_probability: float = 1.0
    combo_potential: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.action_type.value}({self.target_position}, spell={self.spell_id})"


@dataclass
class GameStateEvaluation:
    """Évaluation d'un état de jeu"""
    score: float = 0.0
    hp_score: float = 0.0
    position_score: float = 0.0
    threat_score: float = 0.0
    opportunity_score: float = 0.0
    synergy_score: float = 0.0
    reasoning: List[str] = field(default_factory=list)


@dataclass
class MinMaxNode:
    """Nœud de l'arbre MinMax"""
    state: GameState
    action: Optional[CombatAction] = None
    depth: int = 0
    evaluation: Optional[GameStateEvaluation] = None
    children: List['MinMaxNode'] = field(default_factory=list)
    is_maximizing: bool = True
    alpha: float = float('-inf')
    beta: float = float('inf')


class CombatEvaluator:
    """Évalue les états de combat selon différents critères"""
    
    def __init__(self, character_class: CharacterClass):
        self.character_class = character_class
        self.logger = logging.getLogger(f"{__name__}.CombatEvaluator")
        
        # Poids configurables selon la classe
        self._configure_weights_for_class()
        
    def _configure_weights_for_class(self):
        """Configure les poids d'évaluation selon la classe"""
        base_weights = {
            'hp': 0.3,
            'position': 0.2,
            'threat': 0.25,
            'opportunity': 0.15,
            'synergy': 0.1
        }
        
        # Ajustements par classe
        class_adjustments = {
            CharacterClass.IOP: {'threat': 0.3, 'position': 0.25},
            CharacterClass.CRA: {'position': 0.35, 'opportunity': 0.2},
            CharacterClass.ENIRIPSA: {'hp': 0.4, 'synergy': 0.15},
            CharacterClass.SRAM: {'position': 0.3, 'opportunity': 0.2},
            CharacterClass.SACRIEUR: {'hp': 0.4, 'threat': 0.2},
            CharacterClass.FECA: {'hp': 0.35, 'synergy': 0.15}
        }
        
        self.weights = base_weights.copy()
        if self.character_class in class_adjustments:
            self.weights.update(class_adjustments[self.character_class])
    
    def evaluate_state(self, state: GameState) -> GameStateEvaluation:
        """
        Évalue un état de combat de manière exhaustive
        
        Args:
            state: État du jeu à évaluer
            
        Returns:
            GameStateEvaluation: Évaluation complète de l'état
        """
        evaluation = GameStateEvaluation()
        
        try:
            # Évaluations par composantes
            evaluation.hp_score = self._evaluate_hp_situation(state)
            evaluation.position_score = self._evaluate_positions(state)
            evaluation.threat_score = self._evaluate_threats(state)
            evaluation.opportunity_score = self._evaluate_opportunities(state)
            evaluation.synergy_score = self._evaluate_synergies(state)
            
            # Score global pondéré
            evaluation.score = (
                self.weights['hp'] * evaluation.hp_score +
                self.weights['position'] * evaluation.position_score +
                self.weights['threat'] * evaluation.threat_score +
                self.weights['opportunity'] * evaluation.opportunity_score +
                self.weights['synergy'] * evaluation.synergy_score
            )
            
            # Normalisation du score entre -100 et 100
            evaluation.score = max(-100, min(100, evaluation.score))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation: {e}")
            evaluation.score = -50  # Score pessimiste en cas d'erreur
        
        return evaluation
    
    def _evaluate_hp_situation(self, state: GameState) -> float:
        """Évalue la situation des HP (alliés vs ennemis)"""
        try:
            if not state.combat.allies or not state.combat.enemies:
                return 0.0
            
            # HP totaux des alliés et ennemis
            allies_hp = sum(ally.current_hp for ally in state.combat.allies)
            allies_max_hp = sum(ally.max_hp for ally in state.combat.allies)
            
            enemies_hp = sum(enemy.current_hp for enemy in state.combat.enemies)
            enemies_max_hp = sum(enemy.max_hp for enemy in state.combat.enemies)
            
            if allies_max_hp == 0 or enemies_max_hp == 0:
                return 0.0
            
            # Ratios de HP
            allies_hp_ratio = allies_hp / allies_max_hp
            enemies_hp_ratio = enemies_hp / enemies_max_hp
            
            # Score basé sur la différence
            hp_advantage = allies_hp_ratio - enemies_hp_ratio
            
            # Bonus si le joueur principal est en bonne santé
            player_hp_ratio = state.character.hp_percentage() / 100.0
            player_bonus = (player_hp_ratio - 0.5) * 20
            
            return (hp_advantage * 50) + player_bonus
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation HP: {e}")
            return 0.0
    
    def _evaluate_positions(self, state: GameState) -> float:
        """Évalue l'avantage positionnel"""
        try:
            if not state.character.position:
                return 0.0
            
            position_score = 0.0
            char_pos = state.character.position
            
            # Distance aux ennemis (adaptation selon la classe)
            for enemy in state.combat.enemies:
                if not enemy.position or enemy.is_dead():
                    continue
                
                distance = char_pos.distance_to(enemy.position)
                
                # Score selon la classe et la distance optimale
                if self.character_class in [CharacterClass.CRA, CharacterClass.ENUTROF]:
                    # Classes à distance: préfèrent être loin
                    if distance > 4:
                        position_score += 10
                    elif distance < 2:
                        position_score -= 15
                elif self.character_class in [CharacterClass.IOP, CharacterClass.SACRIEUR]:
                    # Classes de mêlée: préfèrent être proches
                    if distance <= 2:
                        position_score += 15
                    elif distance > 5:
                        position_score -= 10
                else:
                    # Classes hybrides: distance moyenne optimale
                    if 2 <= distance <= 4:
                        position_score += 8
            
            # Évaluation de la mobilité (cellules accessibles)
            mobility_score = self._calculate_mobility_score(state, char_pos)
            position_score += mobility_score
            
            # Bonus si position près des alliés (soutien)
            ally_proximity = self._calculate_ally_proximity(state, char_pos)
            position_score += ally_proximity
            
            return position_score
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation position: {e}")
            return 0.0
    
    def _evaluate_threats(self, state: GameState) -> float:
        """Évalue les menaces immédiates et potentielles"""
        try:
            threat_score = 0.0
            char_pos = state.character.position
            
            if not char_pos:
                return -20.0  # Pénalité si position inconnue
            
            for enemy in state.combat.enemies:
                if not enemy.position or enemy.is_dead():
                    continue
                
                # Niveau de menace selon les HP et niveau de l'ennemi
                threat_level = self._calculate_threat_level(enemy)
                distance = char_pos.distance_to(enemy.position)
                
                # Menace selon la distance
                if distance <= 1:
                    threat_score -= threat_level.value * 15  # Menace immédiate
                elif distance <= 3:
                    threat_score -= threat_level.value * 8   # Menace proche
                elif distance <= 6:
                    threat_score -= threat_level.value * 3   # Menace modérée
                
                # Bonus si l'ennemi est très affaibli
                if enemy.hp_percentage() < 25:
                    threat_score += 10
            
            # Pénalité si encerclé
            if self._is_surrounded(state, char_pos):
                threat_score -= 25
            
            return threat_score
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation menaces: {e}")
            return 0.0
    
    def _evaluate_opportunities(self, state: GameState) -> float:
        """Évalue les opportunités d'attaque et de soutien"""
        try:
            opportunity_score = 0.0
            
            # Opportunités d'élimination d'ennemis faibles
            for enemy in state.combat.enemies:
                if enemy.is_dead():
                    continue
                
                hp_percentage = enemy.hp_percentage()
                if hp_percentage < 30:
                    # Grosse opportunité d'élimination
                    opportunity_score += 20 * (1 - hp_percentage / 100)
                elif hp_percentage < 50:
                    # Opportunité modérée
                    opportunity_score += 10 * (1 - hp_percentage / 100)
            
            # Opportunités de soutien aux alliés
            for ally in state.combat.allies:
                if ally.is_dead() or ally.entity_id == state.character.name:
                    continue
                
                if ally.hp_percentage() < 40:
                    # Allié en danger - opportunité de soin
                    if self.character_class in [CharacterClass.ENIRIPSA]:
                        opportunity_score += 15
                    else:
                        opportunity_score += 5
            
            # Opportunités de combos (évaluation basique)
            combo_opportunities = self._evaluate_combo_opportunities(state)
            opportunity_score += combo_opportunities
            
            return opportunity_score
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation opportunités: {e}")
            return 0.0
    
    def _evaluate_synergies(self, state: GameState) -> float:
        """Évalue les synergies possibles avec les alliés"""
        try:
            synergy_score = 0.0
            
            if len(state.combat.allies) <= 1:
                return 0.0  # Pas de synergie possible seul
            
            # Synergies basées sur les positions relatives
            char_pos = state.character.position
            if not char_pos:
                return 0.0
            
            for ally in state.combat.allies:
                if (ally.entity_id == state.character.name or 
                    ally.is_dead() or not ally.position):
                    continue
                
                distance = char_pos.distance_to(ally.position)
                
                # Bonus pour formation tactique
                if 2 <= distance <= 4:
                    synergy_score += 5
                
                # Synergie de classe (exemple simplifié)
                if self.character_class == CharacterClass.FECA:
                    # Feca protège mieux à courte distance
                    if distance <= 3:
                        synergy_score += 8
            
            return synergy_score
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation synergies: {e}")
            return 0.0
    
    def _calculate_mobility_score(self, state: GameState, position: GridPosition) -> float:
        """Calcule le score de mobilité depuis une position"""
        # Simulation simplifiée - dans un vrai cas, il faudrait analyser la grille
        mobility = 0.0
        
        # Nombre approximatif de cases accessibles
        accessible_cells = 8  # Estimation de base
        
        # Réduction si près des bords (estimation)
        if position.x < 2 or position.x > 12 or position.y < 2 or position.y > 12:
            accessible_cells -= 2
        
        # Bonus de mobilité
        mobility = accessible_cells * 1.5
        
        return mobility
    
    def _calculate_ally_proximity(self, state: GameState, position: GridPosition) -> float:
        """Calcule le bonus de proximité avec les alliés"""
        proximity_score = 0.0
        
        for ally in state.combat.allies:
            if (ally.entity_id == state.character.name or 
                ally.is_dead() or not ally.position):
                continue
            
            distance = position.distance_to(ally.position)
            
            # Bonus pour soutien mutuel
            if distance <= 3:
                proximity_score += 3
            elif distance <= 5:
                proximity_score += 1
        
        return proximity_score
    
    def _calculate_threat_level(self, enemy: CombatEntity) -> ThreatLevel:
        """Détermine le niveau de menace d'un ennemi"""
        if enemy.is_dead():
            return ThreatLevel.LOW
        
        hp_ratio = enemy.hp_percentage() / 100.0
        level_factor = enemy.level / 100.0  # Normalisation approximative
        
        threat_score = hp_ratio * 0.7 + level_factor * 0.3
        
        if threat_score > 0.8:
            return ThreatLevel.CRITICAL
        elif threat_score > 0.6:
            return ThreatLevel.HIGH
        elif threat_score > 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _is_surrounded(self, state: GameState, position: GridPosition) -> bool:
        """Vérifie si le personnage est encerclé"""
        enemy_count_nearby = 0
        
        for enemy in state.combat.enemies:
            if not enemy.position or enemy.is_dead():
                continue
            
            if position.distance_to(enemy.position) <= 2:
                enemy_count_nearby += 1
        
        return enemy_count_nearby >= 3
    
    def _evaluate_combo_opportunities(self, state: GameState) -> float:
        """Évalue les opportunités de combos"""
        combo_score = 0.0
        
        # Évaluation basique des combos selon les sorts disponibles
        available_spells = [
            spell for spell in state.character.spells.values()
            if state.character.can_cast_spell(spell.spell_id)
        ]
        
        # Bonus si plusieurs sorts disponibles (potentiel de combo)
        if len(available_spells) >= 3:
            combo_score += 5
        
        # Bonus si PA suffisants pour plusieurs actions
        if state.character.current_pa >= 8:
            combo_score += 8
        
        return combo_score


class ActionGenerator:
    """Génère les actions possibles dans une situation donnée"""
    
    def __init__(self, character_class: CharacterClass):
        self.character_class = character_class
        self.logger = logging.getLogger(f"{__name__}.ActionGenerator")
    
    def generate_possible_actions(self, state: GameState) -> List[CombatAction]:
        """
        Génère toutes les actions possibles dans l'état actuel
        
        Args:
            state: État actuel du jeu
            
        Returns:
            List[CombatAction]: Liste des actions possibles
        """
        actions = []
        
        try:
            if not state.combat.is_my_turn():
                return actions
            
            # Actions de déplacement
            movement_actions = self._generate_movement_actions(state)
            actions.extend(movement_actions)
            
            # Actions de sorts
            spell_actions = self._generate_spell_actions(state)
            actions.extend(spell_actions)
            
            # Action de passer le tour
            actions.append(CombatAction(action_type=ActionType.PASS_TURN))
            
            # Tri par priorité estimée
            actions.sort(key=lambda a: self._estimate_action_priority(a, state), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Erreur génération d'actions: {e}")
        
        return actions
    
    def _generate_movement_actions(self, state: GameState) -> List[CombatAction]:
        """Génère les actions de déplacement possibles"""
        movement_actions = []
        
        if state.character.current_pm <= 0 or not state.character.position:
            return movement_actions
        
        current_pos = state.character.position
        pm_available = state.character.current_pm
        
        # Génération des positions accessibles (simulation simplifiée)
        for dx in range(-pm_available, pm_available + 1):
            for dy in range(-pm_available, pm_available + 1):
                if abs(dx) + abs(dy) > pm_available:
                    continue  # Distance Manhattan trop grande
                
                new_x = current_pos.x + dx
                new_y = current_pos.y + dy
                
                # Vérification des limites de la grille (estimation)
                if new_x < 0 or new_x > 14 or new_y < 0 or new_y > 14:
                    continue
                
                new_pos = GridPosition(new_x, new_y)
                
                # Vérification que la cellule n'est pas occupée
                if self._is_cell_occupied(state, new_pos):
                    continue
                
                pm_cost = abs(dx) + abs(dy)
                action = CombatAction(
                    action_type=ActionType.MOVE,
                    target_position=new_pos,
                    pm_cost=pm_cost
                )
                
                movement_actions.append(action)
        
        return movement_actions
    
    def _generate_spell_actions(self, state: GameState) -> List[CombatAction]:
        """Génère les actions de sorts possibles"""
        spell_actions = []
        
        if state.character.current_pa <= 0:
            return spell_actions
        
        for spell_id, spell in state.character.spells.items():
            if not state.character.can_cast_spell(spell_id):
                continue
            
            # Génération des cibles possibles pour ce sort
            target_actions = self._generate_spell_targets(state, spell)
            spell_actions.extend(target_actions)
        
        return spell_actions
    
    def _generate_spell_targets(self, state: GameState, spell: Spell) -> List[CombatAction]:
        """Génère les cibles possibles pour un sort"""
        target_actions = []
        
        if not state.character.position:
            return target_actions
        
        char_pos = state.character.position
        
        # Cibles ennemies
        for enemy in state.combat.enemies:
            if enemy.is_dead() or not enemy.position:
                continue
            
            distance = char_pos.distance_to(enemy.position)
            
            if (distance >= spell.range_min and distance <= spell.range_max):
                action = CombatAction(
                    action_type=ActionType.CAST_SPELL,
                    target_entity=enemy,
                    target_position=enemy.position,
                    spell_id=spell.spell_id,
                    pa_cost=spell.pa_cost,
                    expected_damage=self._estimate_spell_damage(spell, enemy)
                )
                target_actions.append(action)
        
        # Cibles alliées (si sort de soin)
        if "heal" in spell.effects:
            for ally in state.combat.allies:
                if ally.is_dead() or not ally.position:
                    continue
                
                distance = char_pos.distance_to(ally.position)
                
                if (distance >= spell.range_min and distance <= spell.range_max):
                    action = CombatAction(
                        action_type=ActionType.CAST_SPELL,
                        target_entity=ally,
                        target_position=ally.position,
                        spell_id=spell.spell_id,
                        pa_cost=spell.pa_cost,
                        expected_heal=self._estimate_spell_heal(spell, ally)
                    )
                    target_actions.append(action)
        
        return target_actions
    
    def _is_cell_occupied(self, state: GameState, position: GridPosition) -> bool:
        """Vérifie si une cellule est occupée"""
        # Vérification avec les entités alliées
        for ally in state.combat.allies:
            if ally.position and ally.position.x == position.x and ally.position.y == position.y:
                return True
        
        # Vérification avec les entités ennemies
        for enemy in state.combat.enemies:
            if enemy.position and enemy.position.x == position.x and enemy.position.y == position.y:
                return True
        
        # Vérification avec les cellules bloquées
        if hasattr(state.combat, 'blocked_cells'):
            cell_id = position.x * 15 + position.y  # Approximation de l'ID de cellule
            return cell_id in state.combat.blocked_cells
        
        return False
    
    def _estimate_spell_damage(self, spell: Spell, target: CombatEntity) -> int:
        """Estime les dégâts d'un sort sur une cible"""
        if not spell.damage_range or spell.damage_range == (0, 0):
            return 0
        
        # Dégâts moyens du sort
        min_dmg, max_dmg = spell.damage_range
        base_damage = (min_dmg + max_dmg) // 2
        
        # Facteurs de modification (simplifiés)
        # Dans un vrai cas, il faudrait considérer les résistances, statistiques, etc.
        level_factor = 1.0 + (spell.level - 1) * 0.1
        
        estimated_damage = int(base_damage * level_factor)
        return min(estimated_damage, target.current_hp)  # Ne peut pas faire plus que les HP restants
    
    def _estimate_spell_heal(self, spell: Spell, target: CombatEntity) -> int:
        """Estime les soins d'un sort sur une cible"""
        # Estimation basique des soins
        base_heal = 50 + spell.level * 10
        
        # Ne peut pas soigner plus que les HP manquants
        missing_hp = target.max_hp - target.current_hp
        return min(base_heal, missing_hp)
    
    def _estimate_action_priority(self, action: CombatAction, state: GameState) -> float:
        """Estime la priorité d'une action"""
        priority = 0.0
        
        if action.action_type == ActionType.CAST_SPELL:
            # Priorité basée sur les dégâts/soins potentiels
            priority += action.expected_damage * 2
            priority += action.expected_heal * 1.5
            
            # Bonus pour élimination potentielle
            if (action.target_entity and 
                action.expected_damage >= action.target_entity.current_hp):
                priority += 50
        
        elif action.action_type == ActionType.MOVE:
            # Priorité de mouvement basée sur l'amélioration positionnelle
            priority += 10  # Base
            
        elif action.action_type == ActionType.PASS_TURN:
            priority = -10  # Faible priorité par défaut
        
        return priority


class MinMaxAlgorithm:
    """Implémentation de l'algorithme MinMax avec élagage alpha-beta"""
    
    def __init__(self, evaluator: CombatEvaluator, max_depth: int = 4):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.logger = logging.getLogger(f"{__name__}.MinMaxAlgorithm")
    
    def find_best_action(self, state: GameState, actions: List[CombatAction], 
                        time_limit: float = 5.0) -> Optional[CombatAction]:
        """
        Trouve la meilleure action en utilisant MinMax avec élagage alpha-beta
        
        Args:
            state: État actuel du jeu
            actions: Actions possibles
            time_limit: Limite de temps en secondes
            
        Returns:
            CombatAction: Meilleure action trouvée
        """
        if not actions:
            return None
        
        start_time = time.time()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
        best_action = None
        best_score = float('-inf')
        
        try:
            for action in actions:
                # Vérification du temps limite
                if time.time() - start_time > time_limit:
                    self.logger.warning(f"Limite de temps atteinte, évaluation partielle")
                    break
                
                # Simulation de l'état après l'action
                new_state = self._simulate_action(state, action)
                if not new_state:
                    continue
                
                # Évaluation MinMax
                score = self._minmax(
                    new_state, 
                    self.max_depth - 1, 
                    float('-inf'), 
                    float('inf'), 
                    False,  # Tour de l'ennemi après notre action
                    start_time + time_limit
                )
                
                self.logger.debug(f"Action {action} -> Score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"MinMax terminé: {self.nodes_evaluated} nœuds évalués, "
                f"{self.pruned_branches} branches élaguées en {elapsed_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Erreur MinMax: {e}")
            # Retourner la première action viable en cas d'erreur
            return actions[0] if actions else None
        
        return best_action or (actions[0] if actions else None)
    
    def _minmax(self, state: GameState, depth: int, alpha: float, beta: float, 
               maximizing: bool, time_limit: float) -> float:
        """
        Algorithme MinMax récursif avec élagage alpha-beta
        
        Args:
            state: État du jeu
            depth: Profondeur restante
            alpha: Valeur alpha pour élagage
            beta: Valeur beta pour élagage
            maximizing: True si c'est le tour du joueur (maximisant)
            time_limit: Limite de temps absolue
            
        Returns:
            float: Score évalué de l'état
        """
        self.nodes_evaluated += 1
        
        # Vérifications de terminaison
        if time.time() > time_limit:
            return self._evaluate_state_quickly(state)
        
        if depth == 0 or self._is_terminal_state(state):
            return self.evaluator.evaluate_state(state).score
        
        if maximizing:
            # Tour du joueur - maximisation
            max_eval = float('-inf')
            actions = self._get_player_actions(state)
            
            for action in actions:
                new_state = self._simulate_action(state, action)
                if not new_state:
                    continue
                
                eval_score = self._minmax(new_state, depth - 1, alpha, beta, False, time_limit)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Élagage alpha-beta
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
            
            return max_eval
        
        else:
            # Tour de l'ennemi - minimisation
            min_eval = float('inf')
            enemy_actions = self._get_enemy_actions(state)
            
            for action in enemy_actions:
                new_state = self._simulate_enemy_action(state, action)
                if not new_state:
                    continue
                
                eval_score = self._minmax(new_state, depth - 1, alpha, beta, True, time_limit)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Élagage alpha-beta
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
            
            return min_eval
    
    def _simulate_action(self, state: GameState, action: CombatAction) -> Optional[GameState]:
        """
        Simule l'exécution d'une action et retourne le nouvel état
        
        Args:
            state: État actuel
            action: Action à simuler
            
        Returns:
            GameState: Nouvel état après l'action
        """
        try:
            # Copie profonde de l'état pour simulation
            new_state = copy.deepcopy(state)
            
            if action.action_type == ActionType.MOVE:
                self._apply_movement(new_state, action)
            elif action.action_type == ActionType.CAST_SPELL:
                self._apply_spell(new_state, action)
            elif action.action_type == ActionType.PASS_TURN:
                self._apply_pass_turn(new_state)
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Erreur simulation action {action}: {e}")
            return None
    
    def _simulate_enemy_action(self, state: GameState, action: CombatAction) -> Optional[GameState]:
        """Simule une action ennemie"""
        # Implémentation similaire mais adaptée aux actions ennemies
        return self._simulate_action(state, action)
    
    def _apply_movement(self, state: GameState, action: CombatAction):
        """Applique un mouvement dans l'état simulé"""
        if action.target_position:
            state.character.position = action.target_position
            state.character.current_pm -= action.pm_cost
    
    def _apply_spell(self, state: GameState, action: CombatAction):
        """Applique un sort dans l'état simulé"""
        # Réduction des PA
        state.character.current_pa -= action.pa_cost
        
        # Application des effets du sort
        if action.target_entity and action.expected_damage > 0:
            # Recherche de l'entité cible dans le nouvel état
            target = self._find_entity_in_state(state, action.target_entity.entity_id)
            if target:
                target.current_hp = max(0, target.current_hp - action.expected_damage)
        
        if action.target_entity and action.expected_heal > 0:
            target = self._find_entity_in_state(state, action.target_entity.entity_id)
            if target:
                target.current_hp = min(target.max_hp, target.current_hp + action.expected_heal)
    
    def _apply_pass_turn(self, state: GameState):
        """Applique l'action de passer le tour"""
        # Réinitialisation des PA/PM (simplifiée)
        state.character.current_pa = 0
        state.character.current_pm = 0
    
    def _find_entity_in_state(self, state: GameState, entity_id: int) -> Optional[CombatEntity]:
        """Trouve une entité par son ID dans un état"""
        for entity in state.combat.get_all_entities():
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def _get_player_actions(self, state: GameState) -> List[CombatAction]:
        """Génère les actions possibles pour le joueur"""
        # Version simplifiée - dans un vrai cas, utiliser ActionGenerator
        actions = []
        
        # Action de base: passer le tour
        actions.append(CombatAction(action_type=ActionType.PASS_TURN))
        
        # Quelques actions de sort si possible
        for spell_id, spell in list(state.character.spells.items())[:3]:  # Limite pour performance
            if state.character.can_cast_spell(spell_id):
                for enemy in state.combat.enemies[:2]:  # Limite les cibles
                    if not enemy.is_dead():
                        action = CombatAction(
                            action_type=ActionType.CAST_SPELL,
                            target_entity=enemy,
                            spell_id=spell_id,
                            pa_cost=spell.pa_cost,
                            expected_damage=30  # Valeur approximative
                        )
                        actions.append(action)
        
        return actions
    
    def _get_enemy_actions(self, state: GameState) -> List[CombatAction]:
        """Génère les actions probables de l'ennemi"""
        # Prédiction simplifiée des actions ennemies
        enemy_actions = []
        
        # Actions basiques des ennemis
        enemy_actions.append(CombatAction(action_type=ActionType.PASS_TURN))
        
        # Attaque sur le joueur si possible
        for enemy in state.combat.enemies:
            if enemy.is_dead():
                continue
            
            action = CombatAction(
                action_type=ActionType.CAST_SPELL,
                target_entity=next(iter(state.combat.allies), None),
                expected_damage=25  # Dégâts moyens estimés
            )
            enemy_actions.append(action)
            break  # Limite à un ennemi pour performance
        
        return enemy_actions
    
    def _is_terminal_state(self, state: GameState) -> bool:
        """Vérifie si l'état est terminal (combat terminé)"""
        # Combat terminé si tous les alliés ou tous les ennemis sont morts
        all_allies_dead = all(ally.is_dead() for ally in state.combat.allies)
        all_enemies_dead = all(enemy.is_dead() for enemy in state.combat.enemies)
        
        return all_allies_dead or all_enemies_dead
    
    def _evaluate_state_quickly(self, state: GameState) -> float:
        """Évaluation rapide pour les limitations de temps"""
        # Évaluation simplifiée basée sur les HP
        if not state.combat.allies or not state.combat.enemies:
            return 0.0
        
        allies_hp = sum(max(0, ally.current_hp) for ally in state.combat.allies)
        enemies_hp = sum(max(0, enemy.current_hp) for enemy in state.combat.enemies)
        
        if enemies_hp == 0:
            return 100.0  # Victoire
        if allies_hp == 0:
            return -100.0  # Défaite
        
        # Score basé sur la différence de HP
        return (allies_hp - enemies_hp) / max(1, (allies_hp + enemies_hp)) * 100


class ComboSystem:
    """Système de gestion des combos et synergies"""
    
    def __init__(self, character_class: CharacterClass):
        self.character_class = character_class
        self.logger = logging.getLogger(f"{__name__}.ComboSystem")
        
        # Base de données des combos par classe
        self.combo_database = self._initialize_combo_database()
    
    def _initialize_combo_database(self) -> Dict[CharacterClass, List[Dict]]:
        """Initialise la base de données des combos par classe"""
        return {
            CharacterClass.IOP: [
                {
                    "name": "Combo Épée Céleste",
                    "spells": ["Épée Céleste", "Colère de Iop"],
                    "conditions": {"pa_min": 8, "target_distance": {"min": 1, "max": 3}},
                    "bonus_damage": 25,
                    "priority": 3
                }
            ],
            CharacterClass.CRA: [
                {
                    "name": "Combo Flèche Explosive",
                    "spells": ["Flèche Enflammée", "Flèche Explosive"],
                    "conditions": {"pa_min": 7, "target_distance": {"min": 4, "max": 8}},
                    "bonus_damage": 20,
                    "priority": 2
                }
            ],
            CharacterClass.ENIRIPSA: [
                {
                    "name": "Combo Soin Zone",
                    "spells": ["Mot Soignant", "Mot Régénérant"],
                    "conditions": {"pa_min": 6, "allies_low_hp": 2},
                    "bonus_heal": 30,
                    "priority": 4
                }
            ]
            # Autres classes à compléter...
        }
    
    def identify_possible_combos(self, state: GameState) -> List[Dict]:
        """
        Identifie les combos possibles dans l'état actuel
        
        Args:
            state: État actuel du combat
            
        Returns:
            List[Dict]: Liste des combos possibles avec leurs détails
        """
        possible_combos = []
        
        try:
            class_combos = self.combo_database.get(self.character_class, [])
            
            for combo in class_combos:
                if self._is_combo_feasible(state, combo):
                    combo_info = combo.copy()
                    combo_info["feasibility_score"] = self._calculate_combo_feasibility(state, combo)
                    possible_combos.append(combo_info)
            
            # Tri par priorité et faisabilité
            possible_combos.sort(
                key=lambda c: (c["priority"], c["feasibility_score"]), 
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Erreur identification combos: {e}")
        
        return possible_combos
    
    def _is_combo_feasible(self, state: GameState, combo: Dict) -> bool:
        """Vérifie si un combo est réalisable"""
        # Vérification PA minimum
        if state.character.current_pa < combo["conditions"].get("pa_min", 0):
            return False
        
        # Vérification disponibilité des sorts
        required_spells = combo.get("spells", [])
        available_spells = [spell.name for spell in state.character.spells.values() 
                          if state.character.can_cast_spell(spell.spell_id)]
        
        for required_spell in required_spells:
            if required_spell not in available_spells:
                return False
        
        # Vérification conditions spéciales
        conditions = combo["conditions"]
        
        # Distance à la cible
        if "target_distance" in conditions:
            if not self._check_target_distance_condition(state, conditions["target_distance"]):
                return False
        
        # Alliés avec HP faibles
        if "allies_low_hp" in conditions:
            low_hp_allies = sum(1 for ally in state.combat.allies 
                              if ally.hp_percentage() < 50 and not ally.is_dead())
            if low_hp_allies < conditions["allies_low_hp"]:
                return False
        
        return True
    
    def _check_target_distance_condition(self, state: GameState, distance_condition: Dict) -> bool:
        """Vérifie les conditions de distance pour un combo"""
        if not state.character.position:
            return False
        
        char_pos = state.character.position
        min_dist = distance_condition.get("min", 0)
        max_dist = distance_condition.get("max", 99)
        
        # Vérifie s'il y a au moins une cible dans la bonne distance
        for enemy in state.combat.enemies:
            if enemy.is_dead() or not enemy.position:
                continue
            
            distance = char_pos.distance_to(enemy.position)
            if min_dist <= distance <= max_dist:
                return True
        
        return False
    
    def _calculate_combo_feasibility(self, state: GameState, combo: Dict) -> float:
        """Calcule un score de faisabilité pour un combo"""
        feasibility = 0.0
        
        # Score basé sur les PA disponibles
        pa_ratio = state.character.current_pa / max(1, combo["conditions"].get("pa_min", 1))
        feasibility += min(1.0, pa_ratio) * 30
        
        # Score basé sur les cibles appropriées
        target_score = self._calculate_target_availability_score(state, combo)
        feasibility += target_score * 40
        
        # Score basé sur la situation tactique
        tactical_score = self._calculate_tactical_appropriateness(state, combo)
        feasibility += tactical_score * 30
        
        return feasibility
    
    def _calculate_target_availability_score(self, state: GameState, combo: Dict) -> float:
        """Calcule la disponibilité des cibles pour un combo"""
        if not state.character.position:
            return 0.0
        
        conditions = combo["conditions"]
        target_distance = conditions.get("target_distance", {"min": 0, "max": 99})
        
        valid_targets = 0
        total_targets = 0
        
        for enemy in state.combat.enemies:
            if enemy.is_dead():
                continue
            
            total_targets += 1
            
            if enemy.position:
                distance = state.character.position.distance_to(enemy.position)
                if target_distance["min"] <= distance <= target_distance["max"]:
                    valid_targets += 1
        
        return valid_targets / max(1, total_targets)
    
    def _calculate_tactical_appropriateness(self, state: GameState, combo: Dict) -> float:
        """Calcule la pertinence tactique d'un combo"""
        appropriateness = 0.5  # Score de base
        
        # Bonus si beaucoup d'ennemis (combos de dégâts de zone)
        if "zone" in combo.get("name", "").lower():
            if len(state.combat.enemies) >= 3:
                appropriateness += 0.3
        
        # Bonus si alliés en danger (combos de soin)
        if "soin" in combo.get("name", "").lower() or combo.get("bonus_heal", 0) > 0:
            endangered_allies = sum(1 for ally in state.combat.allies 
                                  if ally.hp_percentage() < 40 and not ally.is_dead())
            if endangered_allies > 0:
                appropriateness += 0.4
        
        return min(1.0, appropriateness)
    
    def generate_combo_actions(self, state: GameState, combo: Dict) -> List[CombatAction]:
        """Génère la séquence d'actions pour exécuter un combo"""
        actions = []
        
        try:
            spell_names = combo.get("spells", [])
            
            for spell_name in spell_names:
                # Recherche du sort par nom
                spell = self._find_spell_by_name(state, spell_name)
                if not spell:
                    continue
                
                # Recherche de la meilleure cible pour ce sort
                target = self._find_best_target_for_spell(state, spell)
                if not target:
                    continue
                
                action = CombatAction(
                    action_type=ActionType.CAST_SPELL,
                    target_entity=target,
                    target_position=target.position,
                    spell_id=spell.spell_id,
                    pa_cost=spell.pa_cost,
                    expected_damage=combo.get("bonus_damage", 0),
                    expected_heal=combo.get("bonus_heal", 0),
                    combo_potential=combo.get("priority", 1) * 10
                )
                
                actions.append(action)
        
        except Exception as e:
            self.logger.error(f"Erreur génération actions combo: {e}")
        
        return actions
    
    def _find_spell_by_name(self, state: GameState, spell_name: str) -> Optional[Spell]:
        """Trouve un sort par son nom"""
        for spell in state.character.spells.values():
            if spell.name == spell_name:
                return spell
        return None
    
    def _find_best_target_for_spell(self, state: GameState, spell: Spell) -> Optional[CombatEntity]:
        """Trouve la meilleure cible pour un sort"""
        if not state.character.position:
            return None
        
        char_pos = state.character.position
        best_target = None
        best_score = -1
        
        # Cibles ennemies pour sorts offensifs
        if spell.damage_range and spell.damage_range != (0, 0):
            for enemy in state.combat.enemies:
                if enemy.is_dead() or not enemy.position:
                    continue
                
                distance = char_pos.distance_to(enemy.position)
                if spell.range_min <= distance <= spell.range_max:
                    # Score basé sur les HP (priorité aux faibles HP)
                    score = (100 - enemy.hp_percentage()) / 100
                    if score > best_score:
                        best_score = score
                        best_target = enemy
        
        # Cibles alliées pour sorts de soin
        elif "heal" in spell.effects:
            for ally in state.combat.allies:
                if ally.is_dead() or not ally.position:
                    continue
                
                distance = char_pos.distance_to(ally.position)
                if spell.range_min <= distance <= spell.range_max:
                    # Score basé sur les HP manquants
                    missing_hp_ratio = 1 - (ally.hp_percentage() / 100)
                    if missing_hp_ratio > best_score:
                        best_score = missing_hp_ratio
                        best_target = ally
        
        return best_target


class LearningSystem:
    """Système d'apprentissage pour améliorer les décisions au fil du temps"""
    
    def __init__(self, character_class: CharacterClass, learning_file: str = "combat_learning.json"):
        self.character_class = character_class
        self.learning_file = learning_file
        self.logger = logging.getLogger(f"{__name__}.LearningSystem")
        
        # Données d'apprentissage
        self.action_history = []
        self.performance_stats = {
            "combats_won": 0,
            "combats_lost": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "avg_combat_duration": 0.0
        }
        
        # Poids adaptatifs pour l'évaluation
        self.adaptive_weights = {
            'hp': 0.3,
            'position': 0.2,
            'threat': 0.25,
            'opportunity': 0.15,
            'synergy': 0.1
        }
        
        # Charger les données précédentes
        self.load_learning_data()
    
    def record_action_outcome(self, action: CombatAction, state_before: GameState, 
                            state_after: GameState, success: bool):
        """
        Enregistre le résultat d'une action pour apprentissage
        
        Args:
            action: Action exécutée
            state_before: État avant l'action
            state_after: État après l'action
            success: True si l'action a eu le résultat escompté
        """
        try:
            action_record = {
                "timestamp": datetime.now().isoformat(),
                "action_type": action.action_type.value,
                "spell_id": action.spell_id,
                "target_position": {
                    "x": action.target_position.x if action.target_position else -1,
                    "y": action.target_position.y if action.target_position else -1
                },
                "expected_damage": action.expected_damage,
                "expected_heal": action.expected_heal,
                "success": success,
                "context": {
                    "hp_before": state_before.character.hp_percentage(),
                    "hp_after": state_after.character.hp_percentage(),
                    "enemies_count": len(state_before.combat.enemies),
                    "pa_used": action.pa_cost,
                    "pm_used": action.pm_cost
                }
            }
            
            self.action_history.append(action_record)
            self.performance_stats["total_actions"] += 1
            
            if success:
                self.performance_stats["successful_actions"] += 1
            
            # Limitation de l'historique
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-800:]  # Garde les 800 plus récents
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement action: {e}")
    
    def record_combat_result(self, victory: bool, combat_duration: float, 
                           final_hp_percentage: float):
        """
        Enregistre le résultat d'un combat
        
        Args:
            victory: True si victoire
            combat_duration: Durée du combat en secondes
            final_hp_percentage: HP restants en fin de combat
        """
        try:
            if victory:
                self.performance_stats["combats_won"] += 1
            else:
                self.performance_stats["combats_lost"] += 1
            
            # Mise à jour de la durée moyenne
            total_combats = (self.performance_stats["combats_won"] + 
                           self.performance_stats["combats_lost"])
            
            old_avg = self.performance_stats["avg_combat_duration"]
            self.performance_stats["avg_combat_duration"] = (
                (old_avg * (total_combats - 1) + combat_duration) / total_combats
            )
            
            # Ajustement adaptatif des poids selon les résultats
            self._adjust_adaptive_weights(victory, final_hp_percentage)
            
            # Sauvegarde périodique
            if total_combats % 10 == 0:
                self.save_learning_data()
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement résultat combat: {e}")
    
    def get_adaptive_evaluation_weights(self) -> Dict[str, float]:
        """Retourne les poids adaptatifs actuels pour l'évaluation"""
        return self.adaptive_weights.copy()
    
    def analyze_action_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns d'actions pour identifier les stratégies efficaces
        
        Returns:
            Dict contenant l'analyse des patterns
        """
        if not self.action_history:
            return {"error": "Pas de données d'historique"}
        
        try:
            analysis = {
                "most_successful_actions": self._analyze_successful_actions(),
                "spell_effectiveness": self._analyze_spell_effectiveness(),
                "positional_preferences": self._analyze_positional_patterns(),
                "timing_patterns": self._analyze_timing_patterns()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erreur analyse patterns: {e}")
            return {"error": str(e)}
    
    def _adjust_adaptive_weights(self, victory: bool, final_hp_percentage: float):
        """Ajuste les poids selon les résultats du combat"""
        adjustment_factor = 0.05  # Taux d'apprentissage
        
        if victory:
            # Renforcement des stratégies gagnantes
            if final_hp_percentage > 70:
                # Victoire avec beaucoup de HP -> bonne stratégie défensive
                self.adaptive_weights['hp'] += adjustment_factor
                self.adaptive_weights['threat'] += adjustment_factor
            else:
                # Victoire serrée -> importance de l'opportunisme
                self.adaptive_weights['opportunity'] += adjustment_factor
        else:
            # Ajustement après défaite
            if final_hp_percentage < 30:
                # Défaite avec peu de HP -> manque de défense
                self.adaptive_weights['hp'] += adjustment_factor * 2
                self.adaptive_weights['threat'] += adjustment_factor * 2
            else:
                # Défaite avec HP correct -> manque d'efficacité offensive
                self.adaptive_weights['opportunity'] += adjustment_factor * 1.5
        
        # Normalisation des poids
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight
    
    def _analyze_successful_actions(self) -> Dict[str, Any]:
        """Analyse les actions les plus réussies"""
        action_success_rate = {}
        action_counts = {}
        
        for record in self.action_history:
            action_type = record["action_type"]
            
            if action_type not in action_counts:
                action_counts[action_type] = 0
                action_success_rate[action_type] = 0
            
            action_counts[action_type] += 1
            if record["success"]:
                action_success_rate[action_type] += 1
        
        # Calcul des taux de succès
        success_rates = {}
        for action_type in action_counts:
            if action_counts[action_type] > 5:  # Minimum d'échantillons
                success_rates[action_type] = (
                    action_success_rate[action_type] / action_counts[action_type]
                )
        
        return {
            "success_rates": success_rates,
            "most_used_actions": sorted(action_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_spell_effectiveness(self) -> Dict[str, Any]:
        """Analyse l'efficacité des sorts utilisés"""
        spell_stats = {}
        
        for record in self.action_history:
            if record["spell_id"] and record["action_type"] == "cast_spell":
                spell_id = record["spell_id"]
                
                if spell_id not in spell_stats:
                    spell_stats[spell_id] = {
                        "uses": 0,
                        "successes": 0,
                        "total_damage": 0,
                        "total_heal": 0
                    }
                
                stats = spell_stats[spell_id]
                stats["uses"] += 1
                
                if record["success"]:
                    stats["successes"] += 1
                
                stats["total_damage"] += record.get("expected_damage", 0)
                stats["total_heal"] += record.get("expected_heal", 0)
        
        # Calcul des métriques d'efficacité
        effectiveness = {}
        for spell_id, stats in spell_stats.items():
            if stats["uses"] >= 3:  # Minimum d'utilisations
                effectiveness[spell_id] = {
                    "success_rate": stats["successes"] / stats["uses"],
                    "avg_damage": stats["total_damage"] / stats["uses"],
                    "avg_heal": stats["total_heal"] / stats["uses"],
                    "total_uses": stats["uses"]
                }
        
        return effectiveness
    
    def _analyze_positional_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns de positionnement"""
        # Analyse simplifiée des positions préférées
        position_outcomes = {
            "close_range": {"success": 0, "total": 0},
            "medium_range": {"success": 0, "total": 0},
            "long_range": {"success": 0, "total": 0}
        }
        
        for record in self.action_history:
            if record["target_position"]["x"] != -1:
                # Classification approximative de la distance
                # (nécessiterait la position du personnage pour être précise)
                distance_category = "medium_range"  # Valeur par défaut
                
                position_outcomes[distance_category]["total"] += 1
                if record["success"]:
                    position_outcomes[distance_category]["success"] += 1
        
        # Calcul des taux de succès par position
        position_success_rates = {}
        for category, data in position_outcomes.items():
            if data["total"] > 0:
                position_success_rates[category] = data["success"] / data["total"]
        
        return position_success_rates
    
    def _analyze_timing_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns temporels"""
        # Analyse du timing des actions (début vs fin de tour)
        timing_analysis = {
            "early_turn_success": 0,
            "late_turn_success": 0,
            "early_turn_total": 0,
            "late_turn_total": 0
        }
        
        # Cette analyse nécessiterait des informations sur le timing dans le tour
        # Pour l'instant, analyse basique basée sur l'ordre des actions
        
        return {
            "pattern": "Analyse temporelle nécessite plus de données contextuelles"
        }
    
    def recommend_strategy_adjustments(self) -> List[str]:
        """Recommande des ajustements stratégiques basés sur l'apprentissage"""
        recommendations = []
        
        try:
            # Analyse du taux de victoire
            total_combats = (self.performance_stats["combats_won"] + 
                           self.performance_stats["combats_lost"])
            
            if total_combats > 0:
                win_rate = self.performance_stats["combats_won"] / total_combats
                
                if win_rate < 0.4:
                    recommendations.append(
                        "Taux de victoire faible: Privilégier les stratégies défensives"
                    )
                elif win_rate > 0.8:
                    recommendations.append(
                        "Excellent taux de victoire: Continuer la stratégie actuelle"
                    )
            
            # Analyse de l'efficacité des actions
            if self.performance_stats["total_actions"] > 0:
                action_success_rate = (self.performance_stats["successful_actions"] / 
                                     self.performance_stats["total_actions"])
                
                if action_success_rate < 0.6:
                    recommendations.append(
                        "Efficacité des actions faible: Revoir la sélection des cibles"
                    )
            
            # Recommandations basées sur les poids adaptatifs
            max_weight = max(self.adaptive_weights.values())
            dominant_factor = max(self.adaptive_weights.items(), key=lambda x: x[1])[0]
            
            if max_weight > 0.4:
                recommendations.append(
                    f"Stratégie focalisée sur: {dominant_factor}. "
                    f"Considérer plus d'équilibre."
                )
            
        except Exception as e:
            self.logger.error(f"Erreur génération recommandations: {e}")
            recommendations.append("Erreur dans l'analyse - données insuffisantes")
        
        return recommendations
    
    def save_learning_data(self):
        """Sauvegarde les données d'apprentissage"""
        try:
            learning_data = {
                "character_class": self.character_class.value,
                "performance_stats": self.performance_stats,
                "adaptive_weights": self.adaptive_weights,
                "action_history": self.action_history[-500:],  # Garde les 500 plus récents
                "last_save": datetime.now().isoformat()
            }
            
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Données d'apprentissage sauvegardées: {self.learning_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")
    
    def load_learning_data(self):
        """Charge les données d'apprentissage précédentes"""
        try:
            with open(self.learning_file, 'r', encoding='utf-8') as f:
                learning_data = json.load(f)
            
            # Vérification de la classe de personnage
            if learning_data.get("character_class") != self.character_class.value:
                self.logger.warning("Classe de personnage différente, données partiellement ignorées")
                return
            
            # Chargement des données
            self.performance_stats.update(learning_data.get("performance_stats", {}))
            self.adaptive_weights.update(learning_data.get("adaptive_weights", {}))
            self.action_history = learning_data.get("action_history", [])
            
            self.logger.info(f"Données d'apprentissage chargées: {len(self.action_history)} actions")
            
        except FileNotFoundError:
            self.logger.info("Pas de données d'apprentissage précédentes, démarrage fresh")
        except Exception as e:
            self.logger.error(f"Erreur chargement données d'apprentissage: {e}")


class CombatAI:
    """
    Système d'IA de Combat Principal
    Coordonne tous les composants pour des décisions optimales
    """
    
    def __init__(self, character_class: CharacterClass, config: Optional[Dict] = None):
        self.character_class = character_class
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.CombatAI")
        
        # Initialisation des composants
        self.evaluator = CombatEvaluator(character_class)
        self.action_generator = ActionGenerator(character_class)
        self.minmax_algorithm = MinMaxAlgorithm(
            self.evaluator, 
            max_depth=self.config.get("max_search_depth", 4)
        )
        self.combo_system = ComboSystem(character_class)
        self.learning_system = LearningSystem(character_class)
        
        # État du système
        self.last_action = None
        self.current_strategy = "balanced"
        self.decision_time_limit = self.config.get("decision_time_limit", 5.0)
        
        # Intégration des poids adaptatifs
        self._update_evaluator_weights()
        
        self.logger.info(f"IA de combat initialisée pour {character_class.value}")
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut du système d'IA"""
        return {
            "max_search_depth": 4,
            "decision_time_limit": 5.0,
            "enable_learning": True,
            "enable_combos": True,
            "aggressive_mode": False,
            "defensive_threshold": 30  # % HP en dessous duquel être défensif
        }
    
    def decide_action(self, game_state: GameState) -> Optional[CombatAction]:
        """
        Décide de la meilleure action à effectuer
        
        Args:
            game_state: État actuel du jeu
            
        Returns:
            CombatAction: Meilleure action à effectuer
        """
        start_time = time.time()
        
        try:
            # Vérification des pré-conditions
            if not self._can_make_decision(game_state):
                return None
            
            self.logger.info("=== Début de la décision d'action ===")
            
            # Génération des actions possibles
            possible_actions = self.action_generator.generate_possible_actions(game_state)
            self.logger.info(f"Actions possibles générées: {len(possible_actions)}")
            
            if not possible_actions:
                return CombatAction(action_type=ActionType.PASS_TURN)
            
            # Vérification des combos prioritaires
            best_combo_action = self._check_for_priority_combos(game_state)
            if best_combo_action:
                self.logger.info("Combo prioritaire détecté")
                return best_combo_action
            
            # Décision MinMax pour les actions standard
            best_action = self.minmax_algorithm.find_best_action(
                game_state, 
                possible_actions, 
                self.decision_time_limit
            )
            
            # Application de la logique stratégique finale
            final_action = self._apply_strategic_override(game_state, best_action, possible_actions)
            
            decision_time = time.time() - start_time
            self.logger.info(
                f"Décision prise en {decision_time:.2f}s: {final_action}"
            )
            
            # Enregistrement pour apprentissage
            self.last_action = final_action
            
            return final_action
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la décision: {e}")
            # Action de sécurité
            return CombatAction(action_type=ActionType.PASS_TURN)
    
    def _can_make_decision(self, game_state: GameState) -> bool:
        """Vérifie si on peut prendre une décision"""
        if game_state.combat.state == CombatState.NO_COMBAT:
            self.logger.debug("Pas en combat")
            return False
        
        if not game_state.combat.is_my_turn():
            self.logger.debug("Pas notre tour")
            return False
        
        if game_state.character.is_dead:
            self.logger.debug("Personnage mort")
            return False
        
        return True
    
    def _check_for_priority_combos(self, game_state: GameState) -> Optional[CombatAction]:
        """Vérifie s'il y a des combos prioritaires à exécuter"""
        if not self.config.get("enable_combos", True):
            return None
        
        try:
            possible_combos = self.combo_system.identify_possible_combos(game_state)
            
            for combo in possible_combos:
                # Seuls les combos de haute priorité remplacent MinMax
                if combo.get("priority", 0) >= 4 and combo.get("feasibility_score", 0) > 70:
                    combo_actions = self.combo_system.generate_combo_actions(game_state, combo)
                    
                    if combo_actions:
                        self.logger.info(f"Combo prioritaire sélectionné: {combo['name']}")
                        return combo_actions[0]  # Première action du combo
            
        except Exception as e:
            self.logger.error(f"Erreur vérification combos: {e}")
        
        return None
    
    def _apply_strategic_override(self, game_state: GameState, 
                                best_action: Optional[CombatAction],
                                all_actions: List[CombatAction]) -> Optional[CombatAction]:
        """
        Applique des règles stratégiques de haut niveau qui peuvent remplacer MinMax
        
        Args:
            game_state: État du jeu
            best_action: Action recommandée par MinMax
            all_actions: Toutes les actions possibles
            
        Returns:
            CombatAction: Action finale après application des règles stratégiques
        """
        try:
            current_hp_percentage = game_state.character.hp_percentage()
            
            # Mode défensif si HP faibles
            if current_hp_percentage < self.config.get("defensive_threshold", 30):
                self.logger.info("Mode défensif activé (HP faibles)")
                defensive_action = self._find_defensive_action(game_state, all_actions)
                if defensive_action:
                    return defensive_action
            
            # Mode agressif si configuré et HP élevés
            if (self.config.get("aggressive_mode", False) and 
                current_hp_percentage > 70):
                self.logger.info("Mode agressif activé")
                aggressive_action = self._find_aggressive_action(game_state, all_actions)
                if aggressive_action:
                    return aggressive_action
            
            # Règle d'élimination: Priorité aux ennemis très faibles
            elimination_action = self._find_elimination_opportunity(game_state, all_actions)
            if elimination_action:
                self.logger.info("Opportunité d'élimination détectée")
                return elimination_action
            
            # Action par défaut
            return best_action
            
        except Exception as e:
            self.logger.error(f"Erreur règles stratégiques: {e}")
            return best_action
    
    def _find_defensive_action(self, game_state: GameState, 
                              actions: List[CombatAction]) -> Optional[CombatAction]:
        """Trouve la meilleure action défensive"""
        # Priorité 1: Soins si disponibles
        heal_actions = [a for a in actions 
                       if a.action_type == ActionType.CAST_SPELL and a.expected_heal > 0]
        
        if heal_actions:
            # Soin avec le meilleur ratio heal/PA
            best_heal = max(heal_actions, key=lambda a: a.expected_heal / max(1, a.pa_cost))
            return best_heal
        
        # Priorité 2: Mouvement pour s'éloigner des ennemis
        movement_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        
        if movement_actions and game_state.character.position:
            char_pos = game_state.character.position
            
            # Trouver le mouvement qui maximise la distance aux ennemis
            best_movement = None
            best_safety_score = -1
            
            for move_action in movement_actions:
                if not move_action.target_position:
                    continue
                
                safety_score = 0
                for enemy in game_state.combat.enemies:
                    if enemy.position and not enemy.is_dead():
                        distance = move_action.target_position.distance_to(enemy.position)
                        safety_score += distance
                
                if safety_score > best_safety_score:
                    best_safety_score = safety_score
                    best_movement = move_action
            
            return best_movement
        
        return None
    
    def _find_aggressive_action(self, game_state: GameState,
                               actions: List[CombatAction]) -> Optional[CombatAction]:
        """Trouve la meilleure action agressive"""
        # Priorité aux sorts de dégâts élevés
        damage_actions = [a for a in actions 
                         if a.action_type == ActionType.CAST_SPELL and a.expected_damage > 0]
        
        if damage_actions:
            # Sort avec les meilleurs dégâts
            best_damage = max(damage_actions, key=lambda a: a.expected_damage)
            return best_damage
        
        return None
    
    def _find_elimination_opportunity(self, game_state: GameState,
                                    actions: List[CombatAction]) -> Optional[CombatAction]:
        """Cherche des opportunités d'élimination d'ennemis faibles"""
        elimination_actions = []
        
        for action in actions:
            if (action.action_type == ActionType.CAST_SPELL and 
                action.target_entity and 
                not action.target_entity.is_ally and
                action.expected_damage > 0):
                
                # Vérifier si cette action peut éliminer l'ennemi
                if action.expected_damage >= action.target_entity.current_hp:
                    elimination_actions.append(action)
        
        if elimination_actions:
            # Priorité à l'élimination qui coûte le moins de PA
            return min(elimination_actions, key=lambda a: a.pa_cost)
        
        return None
    
    def post_action_learning(self, action_taken: CombatAction, 
                           state_before: GameState, state_after: GameState):
        """
        Apprentissage après exécution d'une action
        
        Args:
            action_taken: Action qui a été exécutée
            state_before: État avant l'action
            state_after: État après l'action
        """
        if not self.config.get("enable_learning", True):
            return
        
        try:
            # Évaluation du succès de l'action
            success = self._evaluate_action_success(action_taken, state_before, state_after)
            
            # Enregistrement pour apprentissage
            self.learning_system.record_action_outcome(
                action_taken, state_before, state_after, success
            )
            
            # Mise à jour des poids adaptatifs
            self._update_evaluator_weights()
            
        except Exception as e:
            self.logger.error(f"Erreur apprentissage post-action: {e}")
    
    def post_combat_learning(self, victory: bool, combat_duration: float, 
                           final_hp_percentage: float):
        """
        Apprentissage après un combat complet
        
        Args:
            victory: True si victoire
            combat_duration: Durée totale du combat
            final_hp_percentage: HP restants en fin de combat
        """
        if not self.config.get("enable_learning", True):
            return
        
        try:
            self.learning_system.record_combat_result(
                victory, combat_duration, final_hp_percentage
            )
            
            # Analyse et recommandations
            recommendations = self.learning_system.recommend_strategy_adjustments()
            if recommendations:
                self.logger.info("Recommandations stratégiques:")
                for rec in recommendations:
                    self.logger.info(f"  - {rec}")
            
        except Exception as e:
            self.logger.error(f"Erreur apprentissage post-combat: {e}")
    
    def _evaluate_action_success(self, action: CombatAction, 
                                state_before: GameState, state_after: GameState) -> bool:
        """Évalue si une action a été réussie"""
        try:
            if action.action_type == ActionType.CAST_SPELL:
                # Succès basé sur l'atteinte des objectifs
                if action.expected_damage > 0:
                    # Vérifier si l'ennemi ciblé a perdu des HP
                    target_damaged = False
                    if action.target_entity:
                        # Dans un vrai cas, il faudrait suivre l'entité spécifique
                        # Ici, vérification générale que des ennemis ont perdu des HP
                        enemies_before = sum(e.current_hp for e in state_before.combat.enemies)
                        enemies_after = sum(e.current_hp for e in state_after.combat.enemies)
                        target_damaged = enemies_after < enemies_before
                    
                    return target_damaged
                
                elif action.expected_heal > 0:
                    # Vérifier si des alliés ont gagné des HP
                    allies_before = sum(a.current_hp for a in state_before.combat.allies)
                    allies_after = sum(a.current_hp for a in state_after.combat.allies)
                    return allies_after > allies_before
            
            elif action.action_type == ActionType.MOVE:
                # Succès si la position a changé comme prévu
                return (state_after.character.position is not None and
                        state_after.character.position == action.target_position)
            
            # Autres types d'actions considérés comme réussis par défaut
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation succès action: {e}")
            return False
    
    def _update_evaluator_weights(self):
        """Met à jour les poids de l'évaluateur avec l'apprentissage adaptatif"""
        try:
            adaptive_weights = self.learning_system.get_adaptive_evaluation_weights()
            self.evaluator.weights.update(adaptive_weights)
        except Exception as e:
            self.logger.error(f"Erreur mise à jour poids: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance de l'IA"""
        try:
            base_stats = self.learning_system.performance_stats.copy()
            
            # Ajout de statistiques supplémentaires
            base_stats.update({
                "character_class": self.character_class.value,
                "current_strategy": self.current_strategy,
                "minmax_nodes_last_decision": self.minmax_algorithm.nodes_evaluated,
                "adaptive_weights": self.learning_system.adaptive_weights.copy(),
                "decision_time_limit": self.decision_time_limit
            })
            
            return base_stats
            
        except Exception as e:
            self.logger.error(f"Erreur récupération statistiques: {e}")
            return {"error": str(e)}
    
    def export_learning_data(self, filepath: str) -> bool:
        """
        Exporte toutes les données d'apprentissage
        
        Args:
            filepath: Chemin du fichier d'export
            
        Returns:
            bool: True si succès
        """
        try:
            export_data = {
                "metadata": {
                    "character_class": self.character_class.value,
                    "export_date": datetime.now().isoformat(),
                    "config": self.config
                },
                "performance_stats": self.learning_system.performance_stats,
                "adaptive_weights": self.learning_system.adaptive_weights,
                "action_patterns": self.learning_system.analyze_action_patterns(),
                "strategy_recommendations": self.learning_system.recommend_strategy_adjustments()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Données d'apprentissage exportées: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur export: {e}")
            return False
    
    def configure_strategy(self, strategy_name: str, parameters: Dict[str, Any]):
        """
        Configure la stratégie de combat
        
        Args:
            strategy_name: Nom de la stratégie ("aggressive", "defensive", "balanced")
            parameters: Paramètres spécifiques à la stratégie
        """
        try:
            self.current_strategy = strategy_name
            
            if strategy_name == "aggressive":
                self.config["aggressive_mode"] = True
                self.config["defensive_threshold"] = 20
                # Ajustement des poids pour favoriser l'offense
                self.evaluator.weights.update({
                    'opportunity': 0.3,
                    'threat': 0.2,
                    'hp': 0.2,
                    'position': 0.2,
                    'synergy': 0.1
                })
            
            elif strategy_name == "defensive":
                self.config["aggressive_mode"] = False
                self.config["defensive_threshold"] = 50
                # Ajustement des poids pour favoriser la défense
                self.evaluator.weights.update({
                    'hp': 0.4,
                    'threat': 0.3,
                    'position': 0.15,
                    'opportunity': 0.1,
                    'synergy': 0.05
                })
            
            elif strategy_name == "balanced":
                self.config["aggressive_mode"] = False
                self.config["defensive_threshold"] = 30
                # Poids équilibrés
                self.evaluator.weights.update({
                    'hp': 0.25,
                    'position': 0.2,
                    'threat': 0.25,
                    'opportunity': 0.2,
                    'synergy': 0.1
                })
            
            # Application des paramètres personnalisés
            self.config.update(parameters)
            
            self.logger.info(f"Stratégie configurée: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Erreur configuration stratégie: {e}")


# Fonction d'utilisation principale
def create_combat_ai(character_class: CharacterClass, 
                    config: Optional[Dict] = None) -> CombatAI:
    """
    Crée une instance de l'IA de combat configurée
    
    Args:
        character_class: Classe du personnage
        config: Configuration optionnelle
        
    Returns:
        CombatAI: Instance configurée de l'IA
    """
    return CombatAI(character_class, config)


# Configuration de logging pour le module
def setup_combat_ai_logging(level=logging.INFO):
    """Configure le logging pour le module d'IA de combat"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


if __name__ == "__main__":
    # Exemple d'utilisation et tests
    setup_combat_ai_logging(logging.DEBUG)
    
    # Création d'une IA pour un Iop
    ai = create_combat_ai(CharacterClass.IOP, {
        "max_search_depth": 3,
        "decision_time_limit": 3.0,
        "aggressive_mode": True
    })
    
    print("=== IA de Combat DOFUS Initialisée ===")
    print(f"Classe: {ai.character_class.value}")
    print(f"Statistiques: {ai.get_performance_stats()}")
    print("Système prêt pour l'intégration dans le bot DOFUS!")