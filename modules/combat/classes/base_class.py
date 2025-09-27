"""
Classe abstraite de base pour tous les personnages dans le système de combat DOFUS
Définit l'interface commune et les comportements de base pour toutes les classes de personnage
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

from state.realtime_state import (
    GridPosition, Spell, CombatEntity, CombatState, Character
)


class SpellCategory(Enum):
    """Catégories de sorts"""
    DAMAGE = "damage"
    HEAL = "heal"
    BUFF = "buff"
    DEBUFF = "debuff"
    UTILITY = "utility"
    MOVEMENT = "movement"
    SUMMON = "summon"


class TargetType(Enum):
    """Types de cibles"""
    ENEMY = "enemy"
    ALLY = "ally"
    SELF = "self"
    EMPTY_CELL = "empty_cell"
    ANY = "any"


@dataclass
class SpellInfo:
    """Informations détaillées sur un sort"""
    spell_id: int
    name: str
    level: int
    pa_cost: int
    range_min: int
    range_max: int
    category: SpellCategory
    target_type: TargetType
    
    # Caractéristiques avancées
    line_of_sight: bool = True
    is_modifiable_range: bool = False
    cooldown_turns: int = 0
    cast_per_turn: int = 1
    cast_per_target: int = 1
    
    # Dégâts/soins
    damage_min: int = 0
    damage_max: int = 0
    heal_min: int = 0
    heal_max: int = 0
    
    # Effets spéciaux
    effects: List[str] = field(default_factory=list)
    aoe_size: int = 0  # Taille de la zone d'effet (0 = mono-cible)
    aoe_pattern: str = "circle"  # circle, cross, line, etc.
    
    # Positionnement
    push_damage: int = 0
    pull_damage: int = 0
    teleport_range: int = 0
    
    def get_average_damage(self) -> float:
        """Retourne les dégâts moyens du sort"""
        if self.damage_min == 0 and self.damage_max == 0:
            return 0.0
        return (self.damage_min + self.damage_max) / 2
    
    def get_average_heal(self) -> float:
        """Retourne les soins moyens du sort"""
        if self.heal_min == 0 and self.heal_max == 0:
            return 0.0
        return (self.heal_min + self.heal_max) / 2


@dataclass
class CombatAction:
    """Action de combat à exécuter"""
    action_type: str  # "cast_spell", "move", "skip_turn", "end_turn"
    spell_id: Optional[int] = None
    target_position: Optional[GridPosition] = None
    target_entity_id: Optional[int] = None
    move_to_position: Optional[GridPosition] = None
    priority: int = 0  # Plus élevé = plus prioritaire
    reasoning: str = ""  # Explication de l'action pour debug


@dataclass
class TargetEvaluation:
    """Évaluation d'une cible potentielle"""
    entity: CombatEntity
    priority_score: float
    distance: int
    threat_level: float
    vulnerability: float
    strategic_value: float
    can_be_targeted: bool = True
    reasoning: str = ""


class BaseCharacterClass(ABC):
    """
    Classe abstraite de base pour tous les personnages
    Définit l'interface commune pour toutes les classes de DOFUS
    """
    
    def __init__(self, character: Character, logger: Optional[logging.Logger] = None):
        """
        Initialise la classe de personnage
        
        Args:
            character: Données du personnage
            logger: Logger pour les messages de debug
        """
        self.character = character
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Dictionnaire des sorts disponibles avec leurs informations détaillées
        self.spells_info: Dict[int, SpellInfo] = {}
        
        # Stratégies de combat
        self.combat_strategies: List[str] = []
        self.current_strategy: Optional[str] = None
        
        # Cache pour optimisation
        self._target_cache: Dict[str, List[TargetEvaluation]] = {}
        self._last_cache_update = 0
        
        # Statistiques de combat
        self.combat_stats = {
            "spells_cast": {},
            "damage_dealt": 0,
            "damage_taken": 0,
            "heals_given": 0,
            "turns_played": 0,
            "enemies_killed": 0
        }
        
        # Configuration personnalisable
        self.config = {
            "aggressive_threshold": 0.7,  # Seuil pour jouer agressif
            "defensive_threshold": 0.3,   # Seuil pour jouer défensif
            "heal_threshold": 0.5,        # Seuil HP pour se soigner
            "retreat_threshold": 0.2,     # Seuil HP pour fuir
            "preserve_pa_ratio": 0.3      # Ratio de PA à conserver
        }
        
        # Initialisation des sorts
        self._initialize_spells()
        self._initialize_strategies()
    
    @abstractmethod
    def _initialize_spells(self) -> None:
        """Initialise les sorts spécifiques à la classe"""
        pass
    
    @abstractmethod
    def _initialize_strategies(self) -> None:
        """Initialise les stratégies de combat spécifiques à la classe"""
        pass
    
    @abstractmethod
    def get_class_role(self) -> str:
        """
        Retourne le rôle principal de la classe
        
        Returns:
            str: "damage", "tank", "support", "hybrid"
        """
        pass
    
    @abstractmethod
    def evaluate_spell_effectiveness(self, spell_id: int, target: CombatEntity, 
                                   combat_context: Dict[str, Any]) -> float:
        """
        Évalue l'efficacité d'un sort sur une cible dans un contexte donné
        
        Args:
            spell_id: ID du sort
            target: Entité cible
            combat_context: Contexte du combat
            
        Returns:
            float: Score d'efficacité (0-1)
        """
        pass
    
    def get_best_action(self, combat_context: Dict[str, Any]) -> Optional[CombatAction]:
        """
        Détermine la meilleure action à effectuer
        
        Args:
            combat_context: Contexte complet du combat
            
        Returns:
            CombatAction: Meilleure action à effectuer ou None
        """
        try:
            # Validation du contexte
            if not self._validate_combat_context(combat_context):
                self.logger.error("Contexte de combat invalide")
                return None
            
            # Mise à jour de la stratégie si nécessaire
            self._update_strategy(combat_context)
            
            # Génération des actions possibles
            possible_actions = self._generate_possible_actions(combat_context)
            
            if not possible_actions:
                self.logger.warning("Aucune action possible trouvée")
                return CombatAction(action_type="skip_turn", reasoning="Aucune action disponible")
            
            # Évaluation et sélection de la meilleure action
            best_action = self._select_best_action(possible_actions, combat_context)
            
            if best_action:
                self._log_action_selection(best_action)
                self._update_combat_stats(best_action)
            
            return best_action
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sélection d'action: {e}")
            return CombatAction(action_type="skip_turn", reasoning=f"Erreur: {e}")
    
    def evaluate_targets(self, spell_id: int, combat_context: Dict[str, Any]) -> List[TargetEvaluation]:
        """
        Évalue toutes les cibles possibles pour un sort donné
        
        Args:
            spell_id: ID du sort à lancer
            combat_context: Contexte du combat
            
        Returns:
            List[TargetEvaluation]: Liste des cibles évaluées, triées par priorité
        """
        if spell_id not in self.spells_info:
            return []
        
        spell_info = self.spells_info[spell_id]
        all_entities = combat_context.get("all_entities", [])
        player_position = combat_context.get("player_position")
        
        if not player_position:
            return []
        
        evaluations = []
        
        for entity in all_entities:
            if not entity.position:
                continue
            
            # Vérification de la portée
            distance = player_position.distance_to(entity.position)
            if distance < spell_info.range_min or distance > spell_info.range_max:
                continue
            
            # Vérification du type de cible
            if not self._is_valid_target_type(entity, spell_info.target_type, combat_context):
                continue
            
            # Vérification de la ligne de vue
            if spell_info.line_of_sight and not self._has_line_of_sight(
                player_position, entity.position, combat_context
            ):
                continue
            
            # Évaluation de la cible
            evaluation = self._evaluate_target(entity, spell_info, combat_context)
            evaluations.append(evaluation)
        
        # Tri par score de priorité décroissant
        evaluations.sort(key=lambda x: x.priority_score, reverse=True)
        return evaluations
    
    def get_optimal_positioning(self, combat_context: Dict[str, Any]) -> Optional[GridPosition]:
        """
        Détermine la position optimale pour le personnage
        
        Args:
            combat_context: Contexte du combat
            
        Returns:
            GridPosition: Position optimale ou None
        """
        player_position = combat_context.get("player_position")
        available_moves = combat_context.get("available_moves", [])
        
        if not player_position or not available_moves:
            return None
        
        best_position = None
        best_score = -1.0
        
        for position in available_moves:
            score = self._evaluate_position(position, combat_context)
            if score > best_score:
                best_score = score
                best_position = position
        
        return best_position
    
    def can_cast_spell(self, spell_id: int) -> bool:
        """
        Vérifie si un sort peut être lancé
        
        Args:
            spell_id: ID du sort
            
        Returns:
            bool: True si le sort peut être lancé
        """
        if spell_id not in self.spells_info:
            return False
        
        spell_info = self.spells_info[spell_id]
        
        # Vérification des PA
        if self.character.current_pa < spell_info.pa_cost:
            return False
        
        # Vérification du cooldown
        if spell_id in self.character.spells:
            spell_state = self.character.spells[spell_id]
            if spell_state.current_cooldown > 0:
                return False
        
        return True
    
    def get_spell_combo_sequences(self) -> List[List[int]]:
        """
        Retourne les séquences de sorts recommandées (combos)
        
        Returns:
            List[List[int]]: Liste des séquences de sorts (IDs)
        """
        # Implémentation de base - à surcharger dans les classes filles
        return []
    
    def _validate_combat_context(self, combat_context: Dict[str, Any]) -> bool:
        """Valide que le contexte de combat contient les données nécessaires"""
        required_keys = [
            "player_position", "current_pa", "current_pm", "allies", "enemies"
        ]
        
        for key in required_keys:
            if key not in combat_context:
                self.logger.error(f"Clé manquante dans le contexte: {key}")
                return False
        
        return True
    
    def _update_strategy(self, combat_context: Dict[str, Any]) -> None:
        """Met à jour la stratégie de combat selon le contexte"""
        # Analyse de la situation
        allies = combat_context.get("allies", [])
        enemies = combat_context.get("enemies", [])
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        
        # Logique de base pour changer de stratégie
        if player_hp_ratio < self.config["defensive_threshold"]:
            self.current_strategy = "defensive"
        elif player_hp_ratio > self.config["aggressive_threshold"] and len(enemies) <= len(allies):
            self.current_strategy = "aggressive"
        else:
            self.current_strategy = "balanced"
        
        self.logger.debug(f"Stratégie mise à jour: {self.current_strategy}")
    
    def _generate_possible_actions(self, combat_context: Dict[str, Any]) -> List[CombatAction]:
        """Génère toutes les actions possibles dans le contexte actuel"""
        actions = []
        current_pa = combat_context.get("current_pa", 0)
        current_pm = combat_context.get("current_pm", 0)
        
        # Actions de sorts
        for spell_id, spell_info in self.spells_info.items():
            if not self.can_cast_spell(spell_id):
                continue
            
            # Évaluation des cibles pour ce sort
            target_evaluations = self.evaluate_targets(spell_id, combat_context)
            
            for target_eval in target_evaluations[:3]:  # Top 3 cibles
                if not target_eval.can_be_targeted:
                    continue
                
                action = CombatAction(
                    action_type="cast_spell",
                    spell_id=spell_id,
                    target_entity_id=target_eval.entity.entity_id,
                    target_position=target_eval.entity.position,
                    priority=int(target_eval.priority_score * 100),
                    reasoning=f"Sort {spell_info.name} sur {target_eval.entity.name}"
                )
                actions.append(action)
        
        # Actions de mouvement
        if current_pm > 0:
            optimal_position = self.get_optimal_positioning(combat_context)
            if optimal_position:
                action = CombatAction(
                    action_type="move",
                    move_to_position=optimal_position,
                    priority=50,
                    reasoning="Repositionnement tactique"
                )
                actions.append(action)
        
        # Action de passer le tour
        actions.append(CombatAction(
            action_type="skip_turn",
            priority=1,
            reasoning="Passer le tour"
        ))
        
        return actions
    
    def _select_best_action(self, actions: List[CombatAction], 
                          combat_context: Dict[str, Any]) -> Optional[CombatAction]:
        """Sélectionne la meilleure action parmi les possibles"""
        if not actions:
            return None
        
        # Filtrage selon la stratégie actuelle
        filtered_actions = self._filter_actions_by_strategy(actions, combat_context)
        
        # Tri par priorité décroissante
        filtered_actions.sort(key=lambda x: x.priority, reverse=True)
        
        return filtered_actions[0] if filtered_actions else actions[0]
    
    def _filter_actions_by_strategy(self, actions: List[CombatAction], 
                                  combat_context: Dict[str, Any]) -> List[CombatAction]:
        """Filtre les actions selon la stratégie actuelle"""
        if not self.current_strategy:
            return actions
        
        filtered = []
        
        for action in actions:
            if self._action_fits_strategy(action, self.current_strategy, combat_context):
                filtered.append(action)
        
        return filtered if filtered else actions  # Fallback sur toutes les actions
    
    def _action_fits_strategy(self, action: CombatAction, strategy: str, 
                            combat_context: Dict[str, Any]) -> bool:
        """Vérifie si une action correspond à la stratégie"""
        if action.action_type == "cast_spell" and action.spell_id:
            spell_info = self.spells_info.get(action.spell_id)
            if not spell_info:
                return True
            
            if strategy == "aggressive":
                return spell_info.category in [SpellCategory.DAMAGE, SpellCategory.DEBUFF]
            elif strategy == "defensive":
                return spell_info.category in [SpellCategory.HEAL, SpellCategory.BUFF, SpellCategory.UTILITY]
            else:  # balanced
                return True
        
        return True
    
    def _evaluate_target(self, entity: CombatEntity, spell_info: SpellInfo, 
                        combat_context: Dict[str, Any]) -> TargetEvaluation:
        """Évalue une cible pour un sort donné"""
        player_position = combat_context.get("player_position")
        
        # Calcul de la distance
        distance = player_position.distance_to(entity.position) if player_position else 99
        
        # Évaluation de base selon le type de sort
        base_score = self._calculate_base_target_score(entity, spell_info, combat_context)
        
        # Facteurs de modification
        distance_penalty = min(0.1 * distance, 0.5)  # Pénalité de distance
        hp_factor = self._calculate_hp_factor(entity, spell_info)
        threat_factor = self._calculate_threat_factor(entity, combat_context)
        
        # Score final
        final_score = base_score * hp_factor * threat_factor - distance_penalty
        final_score = max(0.0, min(1.0, final_score))  # Clamp entre 0 et 1
        
        return TargetEvaluation(
            entity=entity,
            priority_score=final_score,
            distance=distance,
            threat_level=threat_factor,
            vulnerability=hp_factor,
            strategic_value=base_score,
            reasoning=f"Score: {final_score:.2f} (base: {base_score:.2f}, HP: {hp_factor:.2f}, menace: {threat_factor:.2f})"
        )
    
    def _calculate_base_target_score(self, entity: CombatEntity, spell_info: SpellInfo, 
                                   combat_context: Dict[str, Any]) -> float:
        """Calcule le score de base pour une cible"""
        if spell_info.category == SpellCategory.DAMAGE:
            return 0.8 if not entity.is_ally else 0.0
        elif spell_info.category == SpellCategory.HEAL:
            return 0.8 if entity.is_ally else 0.0
        elif spell_info.category == SpellCategory.BUFF:
            return 0.7 if entity.is_ally else 0.0
        elif spell_info.category == SpellCategory.DEBUFF:
            return 0.7 if not entity.is_ally else 0.0
        else:
            return 0.5
    
    def _calculate_hp_factor(self, entity: CombatEntity, spell_info: SpellInfo) -> float:
        """Calcule le facteur HP pour l'évaluation de cible"""
        hp_ratio = entity.hp_percentage() / 100.0
        
        if spell_info.category == SpellCategory.DAMAGE:
            # Préférer les cibles avec peu de HP (faciles à tuer)
            return 2.0 - hp_ratio
        elif spell_info.category == SpellCategory.HEAL:
            # Préférer les alliés avec peu de HP
            return 2.0 - hp_ratio
        else:
            return 1.0
    
    def _calculate_threat_factor(self, entity: CombatEntity, combat_context: Dict[str, Any]) -> float:
        """Calcule le facteur de menace d'une entité"""
        # Facteur de base selon le type
        if entity.is_monster:
            base_threat = 0.8
        elif entity.is_player and not entity.is_ally:
            base_threat = 0.9
        elif entity.is_ally:
            base_threat = 0.3  # Faible menace mais importante à protéger
        else:
            base_threat = 0.5
        
        # Modification selon le niveau
        level_factor = min(entity.level / 100.0, 1.5)
        
        return base_threat * level_factor
    
    def _evaluate_position(self, position: GridPosition, combat_context: Dict[str, Any]) -> float:
        """Évalue une position selon les critères tactiques"""
        score = 0.0
        
        # Distance aux ennemis
        enemies = combat_context.get("enemies", [])
        for enemy in enemies:
            if enemy.position:
                distance = position.distance_to(enemy.position)
                # Ni trop près ni trop loin
                optimal_distance = 4
                distance_score = 1.0 - abs(distance - optimal_distance) / 10.0
                score += max(0, distance_score)
        
        # Distance aux alliés (rester groupé)
        allies = combat_context.get("allies", [])
        if allies:
            total_ally_distance = sum(
                position.distance_to(ally.position) 
                for ally in allies if ally.position
            )
            avg_ally_distance = total_ally_distance / len(allies) if allies else 0
            # Préférer rester proche des alliés
            score += max(0, 1.0 - avg_ally_distance / 8.0)
        
        return max(0.0, min(1.0, score))
    
    def _is_valid_target_type(self, entity: CombatEntity, target_type: TargetType, 
                            combat_context: Dict[str, Any]) -> bool:
        """Vérifie si une entité correspond au type de cible requis"""
        if target_type == TargetType.ANY:
            return True
        elif target_type == TargetType.ENEMY:
            return not entity.is_ally
        elif target_type == TargetType.ALLY:
            return entity.is_ally
        elif target_type == TargetType.SELF:
            return entity.is_player and entity.is_ally
        else:
            return False
    
    def _has_line_of_sight(self, from_pos: GridPosition, to_pos: GridPosition, 
                          combat_context: Dict[str, Any]) -> bool:
        """Vérifie s'il y a une ligne de vue entre deux positions"""
        # Implémentation simplifiée - à améliorer avec la vraie grille de combat
        blocked_cells = combat_context.get("blocked_cells", set())
        
        # Algorithme de Bresenham simplifié
        dx = abs(to_pos.x - from_pos.x)
        dy = abs(to_pos.y - from_pos.y)
        
        if dx == 0 and dy == 0:
            return True
        
        # Pour l'instant, on considère qu'il y a ligne de vue si pas d'obstacle direct
        return to_pos.cell_id not in blocked_cells
    
    def _log_action_selection(self, action: CombatAction) -> None:
        """Log l'action sélectionnée pour debug"""
        self.logger.debug(f"Action sélectionnée: {action.action_type} - {action.reasoning}")
    
    def _update_combat_stats(self, action: CombatAction) -> None:
        """Met à jour les statistiques de combat"""
        if action.action_type == "cast_spell" and action.spell_id:
            spell_name = self.spells_info.get(action.spell_id, {}).get("name", "Unknown")
            self.combat_stats["spells_cast"][spell_name] = \
                self.combat_stats["spells_cast"].get(spell_name, 0) + 1
        
        self.combat_stats["turns_played"] += 1
    
    def get_spell_by_name(self, name: str) -> Optional[SpellInfo]:
        """Trouve un sort par son nom"""
        for spell_info in self.spells_info.values():
            if spell_info.name.lower() == name.lower():
                return spell_info
        return None
    
    def get_spells_by_category(self, category: SpellCategory) -> List[SpellInfo]:
        """Retourne tous les sorts d'une catégorie"""
        return [spell for spell in self.spells_info.values() if spell.category == category]
    
    def get_combat_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de combat"""
        return dict(self.combat_stats)