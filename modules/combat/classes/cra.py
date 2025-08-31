"""
Classe Cra - Archer expert en combat à distance et contrôle de zone
Rôle: Damage dealer à distance avec capacités de contrôle et positionnement
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

from .base_class import (
    BaseCharacterClass, SpellInfo, SpellCategory, TargetType, 
    CombatAction, TargetEvaluation
)
from ....state.realtime_state import Character, CombatEntity, GridPosition


class CraClass(BaseCharacterClass):
    """
    Classe Cra - Maître de l'arc et du tir de précision
    
    Spécialités:
    - Dégâts à distance élevés
    - Contrôle de zone avec pièges et recul
    - Tir en diagonale et lignes de vue
    - Excellence en kite et positionnement
    """
    
    def __init__(self, character: Character, logger: Optional[logging.Logger] = None):
        super().__init__(character, logger)
        
        # Configuration spécifique Cra
        self.config.update({
            "optimal_range": 6,        # Distance optimale de combat
            "min_safe_distance": 3,    # Distance de sécurité minimale
            "max_effective_range": 10, # Portée maximale efficace
            "kiting_threshold": 2,     # Distance pour commencer à kiter
            "trap_usage_priority": 0.7, # Priorité d'utilisation des pièges
            "diagonal_bonus": 1.2      # Bonus pour tirs en diagonale
        })
        
        # États spéciaux Cra
        self.cra_states = {
            "kiting_mode": False,      # Mode fuite/harcèlement
            "trap_control": False,     # Utilisation active des pièges
            "precision_mode": False,   # Mode tir de précision
            "beacon_placed": False,    # Balise placée
            "elevation_bonus": False   # Bonus de hauteur
        }
        
        # Tracking des pièges posés
        self.active_traps = {}  # {position: trap_info}
        self.trap_cooldowns = {}  # {trap_spell_id: remaining_turns}
    
    def get_class_role(self) -> str:
        """Cra = Damage dealer à distance"""
        return "damage"
    
    def _initialize_spells(self) -> None:
        """Initialise tous les sorts Cra avec leurs caractéristiques complètes"""
        
        # SORTS DE TIR DE BASE
        self.spells_info[161] = SpellInfo(
            spell_id=161, name="Tir Critique", level=1, pa_cost=3,
            range_min=2, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=15, damage_max=20, effects=["Dommages Terre", "+10% CC"],
            cast_per_turn=3, cast_per_target=2
        )
        
        self.spells_info[162] = SpellInfo(
            spell_id=162, name="Tir Puissant", level=3, pa_cost=4,
            range_min=3, range_max=10, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=22, damage_max=28, effects=["Dommages Terre", "Recul 1"],
            push_damage=1, cast_per_turn=2, cast_per_target=1
        )
        
        self.spells_info[163] = SpellInfo(
            spell_id=163, name="Tir Éloigné", level=6, pa_cost=3,
            range_min=4, range_max=12, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=18, damage_max=24, effects=["Dommages Air", "+dégâts par case de distance"],
            cast_per_turn=2, cast_per_target=2
        )
        
        self.spells_info[164] = SpellInfo(
            spell_id=164, name="Tir Percant", level=9, pa_cost=4,
            range_min=2, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=25, damage_max=32, effects=["Dommages Air", "Ignore armure", "Traverse"],
            cast_per_turn=2, cast_per_target=1
        )
        
        # SORTS DE TIR SPÉCIALISÉ
        self.spells_info[165] = SpellInfo(
            spell_id=165, name="Flèche Explosive", level=13, pa_cost=5,
            range_min=3, range_max=9, category=SpellCategory.DAMAGE,
            target_type=TargetType.ANY, line_of_sight=True,
            damage_min=30, damage_max=38, effects=["Dommages Feu", "Zone cercle 1"],
            aoe_size=1, aoe_pattern="circle", cast_per_turn=2, cooldown_turns=2
        )
        
        self.spells_info[166] = SpellInfo(
            spell_id=166, name="Flèche Glacée", level=17, pa_cost=4,
            range_min=2, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=20, damage_max=26, effects=["Dommages Eau", "Ralentissement", "-2 PM"],
            cast_per_turn=2, cast_per_target=2
        )
        
        self.spells_info[167] = SpellInfo(
            spell_id=167, name="Flèche Empoisonnée", level=21, pa_cost=3,
            range_min=2, range_max=10, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=12, damage_max=16, effects=["Dommages Terre", "Poison 3 tours", "DoT 8-12"],
            cast_per_turn=3, cast_per_target=1
        )
        
        self.spells_info[168] = SpellInfo(
            spell_id=168, name="Tir Critique", level=26, pa_cost=6,
            range_min=3, range_max=12, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=45, damage_max=55, effects=["Dommages Air", "100% CC", "Recul 2"],
            push_damage=2, cast_per_turn=1, cooldown_turns=3
        )
        
        # SORTS DE CONTRÔLE ET PIÈGES
        self.spells_info[169] = SpellInfo(
            spell_id=169, name="Piège Sournois", level=31, pa_cost=3,
            range_min=1, range_max=6, category=SpellCategory.UTILITY,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            damage_min=20, damage_max=28, effects=["Piège invisible", "Dommages Terre", "Recul 3"],
            push_damage=3, cast_per_turn=2, cooldown_turns=1
        )
        
        self.spells_info[170] = SpellInfo(
            spell_id=170, name="Piège de Masse", level=36, pa_cost=4,
            range_min=1, range_max=8, category=SpellCategory.UTILITY,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            damage_min=25, damage_max=35, effects=["Piège", "Zone cercle 2", "Recul 2"],
            aoe_size=2, aoe_pattern="circle", push_damage=2,
            cast_per_turn=2, cooldown_turns=2
        )
        
        self.spells_info[171] = SpellInfo(
            spell_id=171, name="Piège Répulsif", level=42, pa_cost=3,
            range_min=1, range_max=5, category=SpellCategory.UTILITY,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            effects=["Piège", "Recul maximum", "Dommages selon distance"],
            push_damage=6, cast_per_turn=1, cooldown_turns=3
        )
        
        self.spells_info[172] = SpellInfo(
            spell_id=172, name="Piège Empoisonné", level=48, pa_cost=4,
            range_min=1, range_max=7, category=SpellCategory.UTILITY,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            damage_min=15, damage_max=20, effects=["Piège", "Poison 5 tours", "Zone croix"],
            aoe_pattern="cross", aoe_size=1, cast_per_turn=1, cooldown_turns=4
        )
        
        # SORTS DE MOBILITÉ ET POSITIONNEMENT
        self.spells_info[173] = SpellInfo(
            spell_id=173, name="Tir Repoussant", level=54, pa_cost=4,
            range_min=2, range_max=6, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=22, damage_max=30, effects=["Dommages Air", "Recul 4", "Attire le Cra"],
            push_damage=4, teleport_range=4, cast_per_turn=1, cooldown_turns=3
        )
        
        self.spells_info[174] = SpellInfo(
            spell_id=174, name="Balise Tactique", level=60, pa_cost=2,
            range_min=1, range_max=8, category=SpellCategory.UTILITY,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            effects=["Place une balise", "Téléportation vers balise", "Bonus portée +2"],
            teleport_range=99, cast_per_turn=1, cooldown_turns=5
        )
        
        self.spells_info[175] = SpellInfo(
            spell_id=175, name="Flèche de Dispersion", level=70, pa_cost=5,
            range_min=3, range_max=10, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=35, damage_max=45, effects=["Dommages Air", "Frappe 3 ennemis proches"],
            aoe_size=2, cast_per_turn=1, cooldown_turns=4
        )
        
        # SORTS AVANCÉS ET MAÎTRISES
        self.spells_info[176] = SpellInfo(
            spell_id=176, name="Maîtrise Arc", level=80, pa_cost=3,
            range_min=0, range_max=0, category=SpellCategory.BUFF,
            target_type=TargetType.SELF, line_of_sight=False,
            effects=["+3 Portée tous sorts", "+50% CC", "+1 PA", "4 tours"],
            cast_per_turn=1, cooldown_turns=8
        )
        
        self.spells_info[177] = SpellInfo(
            spell_id=177, name="Pluie de Flèches", level=90, pa_cost=7,
            range_min=4, range_max=12, category=SpellCategory.DAMAGE,
            target_type=TargetType.ANY, line_of_sight=False,
            damage_min=40, damage_max=55, effects=["Dommages Air", "Zone cercle 4", "5 impacts"],
            aoe_size=4, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=6
        )
        
        self.spells_info[178] = SpellInfo(
            spell_id=178, name="Flèche Divine", level=100, pa_cost=8,
            range_min=5, range_max=15, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=80, damage_max=120, effects=["Dommages Neutre", "Ignore résistances", "Traverse tout"],
            cast_per_turn=1, cooldown_turns=10
        )
        
        # SORTS DE SUPPORT
        self.spells_info[179] = SpellInfo(
            spell_id=179, name="Oeil de Taupe", level=25, pa_cost=2,
            range_min=1, range_max=8, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            effects=["-4 Portée", "-50 Agilité", "État Aveugle", "3 tours"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=1
        )
        
        self.spells_info[180] = SpellInfo(
            spell_id=180, name="Flèche Ralentissante", level=35, pa_cost=3,
            range_min=2, range_max=8, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=10, damage_max=15, effects=["Dommages Eau", "-3 PM", "-25% dommages", "2 tours"],
            cast_per_turn=3, cast_per_target=1
        )
        
        # Ajout des sorts au dictionnaire du personnage pour compatibilité
        for spell_id, spell_info in self.spells_info.items():
            if spell_id not in self.character.spells:
                from ....state.realtime_state import Spell
                self.character.spells[spell_id] = Spell(
                    spell_id=spell_id, name=spell_info.name, level=spell_info.level,
                    pa_cost=spell_info.pa_cost, range_min=spell_info.range_min,
                    range_max=spell_info.range_max, line_of_sight=spell_info.line_of_sight,
                    cooldown_turns=spell_info.cooldown_turns, current_cooldown=0,
                    damage_range=(spell_info.damage_min, spell_info.damage_max),
                    effects=spell_info.effects
                )
    
    def _initialize_strategies(self) -> None:
        """Initialise les stratégies de combat Cra"""
        self.combat_strategies = [
            "sniper",        # Tir de précision longue distance
            "kiting",        # Harcèlement et fuite
            "trap_control",  # Contrôle de zone avec pièges
            "support_fire",  # Tir de soutien pour les alliés
            "area_denial"    # Déni de zone avec AoE
        ]
        self.current_strategy = "sniper"
    
    def evaluate_spell_effectiveness(self, spell_id: int, target: CombatEntity, 
                                   combat_context: Dict[str, Any]) -> float:
        """
        Évalue l'efficacité spécifique d'un sort Cra
        Prend en compte la distance, ligne de vue, position, etc.
        """
        if spell_id not in self.spells_info:
            return 0.0
        
        spell_info = self.spells_info[spell_id]
        player_pos = combat_context.get("player_position")
        
        if not player_pos:
            return 0.0
        
        # Calcul de distance et évaluation de base
        if target and target.position:
            distance = player_pos.distance_to(target.position)
        else:
            distance = 0
        
        base_effectiveness = 0.0
        
        # Évaluation selon le type de sort
        if spell_info.category == SpellCategory.DAMAGE:
            base_effectiveness = self._evaluate_damage_spell_cra(spell_info, target, distance, combat_context)
        elif spell_info.category == SpellCategory.UTILITY:  # Pièges
            base_effectiveness = self._evaluate_trap_spell(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.BUFF:
            base_effectiveness = self._evaluate_buff_spell_cra(spell_info, combat_context)
        elif spell_info.category == SpellCategory.DEBUFF:
            base_effectiveness = self._evaluate_debuff_spell_cra(spell_info, target, distance, combat_context)
        
        # Modificateurs Cra spécifiques
        effectiveness = self._apply_cra_modifiers(base_effectiveness, spell_info, target, distance, combat_context)
        
        return max(0.0, min(1.0, effectiveness))
    
    def _evaluate_damage_spell_cra(self, spell_info: SpellInfo, target: CombatEntity, 
                                  distance: int, combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de dégâts Cra"""
        if target and target.is_ally:
            return 0.0  # Ne pas attaquer les alliés
        
        # Score de base selon les dégâts
        avg_damage = spell_info.get_average_damage()
        if target:
            damage_ratio = avg_damage / target.max_hp if target.max_hp > 0 else 0
        else:
            damage_ratio = 0.5  # Score neutre pour les AoE
        damage_score = min(damage_ratio * 2, 1.0)
        
        # Bonus distance optimale (Cra préfère les distances moyennes-longues)
        distance_bonus = self._calculate_distance_bonus_cra(distance, spell_info)
        
        # Bonus ligne de vue (très important pour Cra)
        los_bonus = 1.2 if spell_info.line_of_sight else 1.0
        
        # Bonus diagonal (Cra excelle en tir diagonal)
        diagonal_bonus = self._calculate_diagonal_bonus(combat_context, target)
        
        # Bonus HP de la cible
        if target:
            hp_ratio = target.hp_percentage() / 100.0
            hp_bonus = 1.3 - (hp_ratio * 0.3)  # Préférer les cibles basses
        else:
            hp_bonus = 1.0
        
        # Bonus zone d'effet
        aoe_bonus = 1.0
        if spell_info.aoe_size > 0:
            enemies_in_range = self._count_enemies_in_aoe_cra(target.position if target else None, spell_info, combat_context)
            aoe_bonus = 1.0 + (enemies_in_range * 0.25)
        
        # Évaluation spéciale pour certains sorts
        spell_bonus = self._get_spell_specific_bonus_cra(spell_info, target, combat_context)
        
        final_score = (damage_score * distance_bonus * los_bonus * 
                      diagonal_bonus * hp_bonus * aoe_bonus * spell_bonus)
        
        self.logger.debug(f"Évaluation Cra {spell_info.name}: {final_score:.2f} "
                         f"(dist:{distance_bonus:.2f}, diag:{diagonal_bonus:.2f}, "
                         f"AoE:{aoe_bonus:.2f}, bonus:{spell_bonus:.2f})")
        
        return final_score
    
    def _calculate_distance_bonus_cra(self, distance: int, spell_info: SpellInfo) -> float:
        """Calcule le bonus de distance pour un Cra"""
        optimal_range = self.config["optimal_range"]
        
        # Différents optimums selon le sort
        if "Tir Éloigné" in spell_info.name or "Flèche Divine" in spell_info.name:
            # Sorts longue distance : plus c'est loin, mieux c'est
            if distance >= 8:
                return 1.3
            elif distance >= 5:
                return 1.1
            else:
                return 0.8
        
        elif "Explosif" in spell_info.name or "Pluie" in spell_info.name:
            # Sorts AoE : distance moyenne optimale
            if 4 <= distance <= 7:
                return 1.2
            elif distance >= 3:
                return 1.0
            else:
                return 0.7
        
        else:
            # Sorts normaux : autour de la distance optimale
            distance_diff = abs(distance - optimal_range)
            if distance_diff == 0:
                return 1.2
            elif distance_diff <= 2:
                return 1.0
            elif distance_diff <= 4:
                return 0.8
            else:
                return 0.6
    
    def _calculate_diagonal_bonus(self, combat_context: Dict[str, Any], target: Optional[CombatEntity]) -> float:
        """Calcule le bonus de tir diagonal caractéristique des Cras"""
        if not target or not target.position:
            return 1.0
        
        player_pos = combat_context.get("player_position")
        if not player_pos:
            return 1.0
        
        # Vérifier si c'est un tir diagonal
        dx = abs(target.position.x - player_pos.x)
        dy = abs(target.position.y - player_pos.y)
        
        # Tir diagonal pur (45°)
        if dx == dy and dx > 0:
            return self.config["diagonal_bonus"]
        # Tir presque diagonal
        elif abs(dx - dy) == 1 and min(dx, dy) > 0:
            return 1.1
        else:
            return 1.0
    
    def _evaluate_trap_spell(self, spell_info: SpellInfo, target: CombatEntity, 
                           combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un piège"""
        # Les pièges sont évalués différemment
        base_score = 0.6
        
        # Bonus selon la stratégie
        if self.current_strategy == "trap_control":
            base_score *= 1.4
        elif self.current_strategy == "kiting":
            base_score *= 1.2
        
        # Bonus selon le nombre d'ennemis proches
        enemies = combat_context.get("enemies", [])
        player_pos = combat_context.get("player_position")
        
        if player_pos:
            close_enemies = sum(
                1 for enemy in enemies 
                if enemy.position and player_pos.distance_to(enemy.position) <= 4
            )
            base_score *= (1.0 + close_enemies * 0.2)
        
        # Malus si trop de pièges déjà actifs
        active_trap_count = len(self.active_traps)
        if active_trap_count >= 3:
            base_score *= 0.7
        
        # Bonus spécifique selon le type de piège
        if "Répulsif" in spell_info.name and self.cra_states["kiting_mode"]:
            base_score *= 1.3
        elif "Masse" in spell_info.name and len(enemies) >= 3:
            base_score *= 1.2
        
        return base_score
    
    def _evaluate_buff_spell_cra(self, spell_info: SpellInfo, combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de buff Cra"""
        if spell_info.name == "Maîtrise Arc":
            # Excellent en début de combat ou avec beaucoup d'ennemis
            turn_number = combat_context.get("turn_number", 1)
            enemy_count = len(combat_context.get("enemies", []))
            
            timing_bonus = 1.5 if turn_number <= 2 else 1.0
            enemy_bonus = 1.0 + (enemy_count * 0.1)
            
            return 0.8 * timing_bonus * enemy_bonus
        
        elif spell_info.name == "Balise Tactique":
            # Utile pour le positionnement tactique
            if self.current_strategy in ["kiting", "sniper"]:
                return 0.9
            else:
                return 0.6
        
        return 0.5
    
    def _evaluate_debuff_spell_cra(self, spell_info: SpellInfo, target: CombatEntity, 
                                 distance: int, combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de debuff Cra"""
        if not target or target.is_ally:
            return 0.0
        
        base_score = 0.6
        
        if spell_info.name == "Oeil de Taupe":
            # Très utile contre les ennemis à distance
            if distance >= 4:
                base_score *= 1.4
            # Excellent contre d'autres Cras ou archers
            if "archer" in target.name.lower() or "cra" in target.name.lower():
                base_score *= 1.3
        
        elif spell_info.name == "Flèche Ralentissante":
            # Utile pour le kiting
            if self.cra_states["kiting_mode"]:
                base_score *= 1.3
            # Priorité sur les ennemis rapides
            if distance <= 3:  # Ennemi proche = dangereux
                base_score *= 1.2
        
        return base_score
    
    def _apply_cra_modifiers(self, base_effectiveness: float, spell_info: SpellInfo, 
                           target: Optional[CombatEntity], distance: int, combat_context: Dict[str, Any]) -> float:
        """Applique les modificateurs spécifiques à la classe Cra"""
        modified_effectiveness = base_effectiveness
        
        # Mode kiting : bonus aux sorts de contrôle et recul
        if self.cra_states["kiting_mode"]:
            if spell_info.push_damage > 0 or spell_info.category == SpellCategory.UTILITY:
                modified_effectiveness *= 1.3
            elif spell_info.range_min <= 2:  # Sorts courte portée
                modified_effectiveness *= 0.7
        
        # Mode précision : bonus aux sorts longue portée avec LoS
        if self.cra_states["precision_mode"]:
            if spell_info.line_of_sight and spell_info.range_max >= 8:
                modified_effectiveness *= 1.2
            if "Critique" in spell_info.name:
                modified_effectiveness *= 1.3
        
        # Bonus élévation si disponible
        if self.cra_states["elevation_bonus"]:
            if spell_info.category == SpellCategory.DAMAGE and spell_info.line_of_sight:
                modified_effectiveness *= 1.15
        
        # Malus si ennemis trop proches (sauf pièges de défense)
        if distance <= self.config["kiting_threshold"]:
            if spell_info.category == SpellCategory.DAMAGE and spell_info.range_min > 1:
                modified_effectiveness *= 0.8
            elif spell_info.category == SpellCategory.UTILITY and spell_info.push_damage > 0:
                modified_effectiveness *= 1.2  # Bonus aux pièges repoussants
        
        # Bonus stratégie actuelle
        strategy_bonus = self._get_strategy_bonus_cra(spell_info)
        modified_effectiveness *= strategy_bonus
        
        # Gestion des cooldowns longs
        if spell_info.cooldown_turns >= 5:
            enemy_count = len(combat_context.get("enemies", []))
            if enemy_count >= 3:  # Utiliser les cooldowns longs contre plusieurs ennemis
                modified_effectiveness *= 1.1
            else:
                modified_effectiveness *= 0.9
        
        return modified_effectiveness
    
    def _get_strategy_bonus_cra(self, spell_info: SpellInfo) -> float:
        """Retourne le bonus de stratégie pour un sort"""
        if self.current_strategy == "sniper":
            if spell_info.range_max >= 8 and spell_info.line_of_sight:
                return 1.2
            elif "Critique" in spell_info.name or "Divine" in spell_info.name:
                return 1.3
        
        elif self.current_strategy == "kiting":
            if spell_info.push_damage > 0 or spell_info.category == SpellCategory.UTILITY:
                return 1.2
            elif "Ralentissant" in spell_info.name or "Glacée" in spell_info.name:
                return 1.3
        
        elif self.current_strategy == "trap_control":
            if spell_info.category == SpellCategory.UTILITY:
                return 1.4
            elif spell_info.aoe_size > 0:
                return 1.1
        
        elif self.current_strategy == "area_denial":
            if spell_info.aoe_size >= 2:
                return 1.3
            elif "Explosif" in spell_info.name or "Pluie" in spell_info.name:
                return 1.4
        
        elif self.current_strategy == "support_fire":
            if spell_info.category == SpellCategory.DEBUFF:
                return 1.2
            elif spell_info.push_damage > 0:
                return 1.1
        
        return 1.0
    
    def _get_spell_specific_bonus_cra(self, spell_info: SpellInfo, target: Optional[CombatEntity], 
                                    combat_context: Dict[str, Any]) -> float:
        """Bonus spécifiques à certains sorts Cra"""
        bonus = 1.0
        
        # Tir Éloigné : bonus distance
        if spell_info.name == "Tir Éloigné" and target:
            player_pos = combat_context.get("player_position")
            if player_pos and target.position:
                distance = player_pos.distance_to(target.position)
                bonus = 1.0 + (distance * 0.05)  # +5% par case
        
        # Flèche Empoisonnée : bonus sur cibles haute HP
        elif spell_info.name == "Flèche Empoisonnée" and target:
            hp_ratio = target.hp_percentage() / 100.0
            if hp_ratio > 0.7:
                bonus = 1.3  # Excellent pour les DoT sur grosses HP
        
        # Tir Percant : bonus contre tanks/armure
        elif spell_info.name == "Tir Percant" and target:
            if target.level >= 50 or target.max_hp >= 500:  # Cibles tankantes
                bonus = 1.2
        
        # Flèche Explosive : bonus contre groupes
        elif spell_info.name == "Flèche Explosive":
            enemies_nearby = self._count_enemies_in_aoe_cra(target.position if target else None, spell_info, combat_context)
            bonus = 1.0 + (enemies_nearby * 0.2)
        
        return bonus
    
    def _count_enemies_in_aoe_cra(self, center_position: Optional[GridPosition], 
                                spell_info: SpellInfo, combat_context: Dict[str, Any]) -> int:
        """Compte les ennemis dans la zone d'effet d'un sort Cra"""
        if not center_position or spell_info.aoe_size == 0:
            return 0
        
        enemies = combat_context.get("enemies", [])
        count = 0
        
        for enemy in enemies:
            if not enemy.position:
                continue
            
            distance = center_position.distance_to(enemy.position)
            
            if spell_info.aoe_pattern == "circle":
                if distance <= spell_info.aoe_size:
                    count += 1
            elif spell_info.aoe_pattern == "cross":
                if ((enemy.position.x == center_position.x and 
                     abs(enemy.position.y - center_position.y) <= spell_info.aoe_size) or
                    (enemy.position.y == center_position.y and 
                     abs(enemy.position.x - center_position.x) <= spell_info.aoe_size)):
                    count += 1
        
        return count
    
    def get_spell_combo_sequences(self) -> List[List[int]]:
        """Retourne les combos optimaux pour Cra selon la situation"""
        combos = []
        
        # Combo sniper longue distance
        combos.append([
            176,  # Maîtrise Arc
            168,  # Tir Critique
            178,  # Flèche Divine
        ])
        
        # Combo kiting/fuite
        combos.append([
            171,  # Piège Répulsif
            166,  # Flèche Glacée
            173,  # Tir Repoussant
            163   # Tir Éloigné
        ])
        
        # Combo contrôle de zone
        combos.append([
            169,  # Piège Sournois
            170,  # Piège de Masse
            165,  # Flèche Explosive
            177   # Pluie de Flèches
        ])
        
        # Combo débuff/support
        combos.append([
            179,  # Oeil de Taupe
            180,  # Flèche Ralentissante
            164,  # Tir Percant
            167   # Flèche Empoisonnée
        ])
        
        # Combo burst AoE
        combos.append([
            174,  # Balise Tactique (repositionnement)
            177,  # Pluie de Flèches
            175,  # Flèche de Dispersion
            165   # Flèche Explosive
        ])
        
        return combos
    
    def _update_strategy(self, combat_context: Dict[str, Any]) -> None:
        """Met à jour la stratégie spécifique Cra selon le contexte"""
        super()._update_strategy(combat_context)
        
        # Analyse de la situation tactique
        player_pos = combat_context.get("player_position")
        enemies = combat_context.get("enemies", [])
        allies = combat_context.get("allies", [])
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        
        # Calcul des distances aux ennemis
        if player_pos and enemies:
            enemy_distances = [
                player_pos.distance_to(enemy.position) 
                for enemy in enemies if enemy.position
            ]
            min_distance = min(enemy_distances) if enemy_distances else 99
            avg_distance = sum(enemy_distances) / len(enemy_distances) if enemy_distances else 99
        else:
            min_distance = avg_distance = 99
        
        # États spéciaux Cra
        self.cra_states["kiting_mode"] = (
            min_distance <= self.config["kiting_threshold"] or 
            player_hp_ratio < 0.4
        )
        
        self.cra_states["precision_mode"] = (
            avg_distance >= self.config["optimal_range"] and
            len(enemies) <= 2
        )
        
        self.cra_states["trap_control"] = (
            len(enemies) >= 3 or 
            (min_distance <= 3 and len(enemies) >= 2)
        )
        
        # Sélection de stratégie
        if self.cra_states["kiting_mode"]:
            self.current_strategy = "kiting"
        elif self.cra_states["trap_control"]:
            self.current_strategy = "trap_control"
        elif len(enemies) >= 3:
            self.current_strategy = "area_denial"
        elif self.cra_states["precision_mode"]:
            self.current_strategy = "sniper"
        else:
            self.current_strategy = "support_fire"
        
        self.logger.debug(f"États Cra: kiting={self.cra_states['kiting_mode']}, "
                         f"précision={self.cra_states['precision_mode']}, "
                         f"pièges={self.cra_states['trap_control']}, "
                         f"stratégie={self.current_strategy}")
    
    def get_optimal_positioning_cra(self, combat_context: Dict[str, Any]) -> Optional[GridPosition]:
        """Détermine la position optimale spécifique pour un Cra"""
        player_pos = combat_context.get("player_position")
        enemies = combat_context.get("enemies", [])
        available_moves = combat_context.get("available_moves", [])
        
        if not player_pos or not enemies or not available_moves:
            return None
        
        best_position = None
        best_score = -1.0
        
        for position in available_moves:
            score = self._evaluate_position_cra(position, enemies, combat_context)
            if score > best_score:
                best_score = score
                best_position = position
        
        return best_position
    
    def _evaluate_position_cra(self, position: GridPosition, enemies: List[CombatEntity], 
                             combat_context: Dict[str, Any]) -> float:
        """Évalue une position selon les critères tactiques Cra"""
        score = 0.0
        
        # Distance optimale aux ennemis
        for enemy in enemies:
            if not enemy.position:
                continue
            
            distance = position.distance_to(enemy.position)
            
            # Distance optimale selon la stratégie
            if self.current_strategy == "sniper":
                # Préférer les longues distances
                if distance >= 8:
                    score += 1.0
                elif distance >= 5:
                    score += 0.8
                else:
                    score += 0.3
            
            elif self.current_strategy == "kiting":
                # Maintenir distance de sécurité
                if distance >= self.config["min_safe_distance"]:
                    score += 1.0
                else:
                    score -= 0.5
            
            else:  # Stratégies équilibrées
                optimal_dist = self.config["optimal_range"]
                distance_diff = abs(distance - optimal_dist)
                score += max(0, 1.0 - distance_diff * 0.1)
        
        # Bonus ligne de vue sur plusieurs ennemis
        enemies_with_los = sum(
            1 for enemy in enemies
            if enemy.position and self._has_line_of_sight_cra(position, enemy.position, combat_context)
        )
        score += enemies_with_los * 0.3
        
        # Bonus position diagonale
        diagonal_enemies = sum(
            1 for enemy in enemies
            if enemy.position and self._is_diagonal_position(position, enemy.position)
        )
        score += diagonal_enemies * 0.2
        
        # Malus si position trop près de pièges alliés (éviter le friendly fire)
        trap_penalty = sum(
            0.2 for trap_pos in self.active_traps.keys()
            if position.distance_to(trap_pos) <= 2
        )
        score -= trap_penalty
        
        # Bonus élévation (si information disponible)
        if combat_context.get("elevation_bonus"):
            score += 0.5
        
        return max(0.0, score)
    
    def _has_line_of_sight_cra(self, from_pos: GridPosition, to_pos: GridPosition, 
                             combat_context: Dict[str, Any]) -> bool:
        """Vérifie la ligne de vue avec considérations Cra spécifiques"""
        # Utilise la méthode de base mais avec des considérations Cra
        blocked_cells = combat_context.get("blocked_cells", set())
        
        # Les Cras peuvent parfois tirer par-dessus les obstacles bas
        if combat_context.get("elevation_bonus"):
            # Ligne de vue améliorée en hauteur
            return to_pos.cell_id not in blocked_cells
        
        return self._has_line_of_sight(from_pos, to_pos, combat_context)
    
    def _is_diagonal_position(self, pos1: GridPosition, pos2: GridPosition) -> bool:
        """Vérifie si deux positions sont en diagonale"""
        dx = abs(pos2.x - pos1.x)
        dy = abs(pos2.y - pos1.y)
        return dx == dy and dx > 0
    
    def get_trap_placement_suggestions(self, combat_context: Dict[str, Any]) -> List[Tuple[GridPosition, int]]:
        """
        Suggère les meilleures positions pour placer des pièges
        
        Returns:
            List[Tuple[GridPosition, spell_id]]: Positions et sorts de pièges recommandés
        """
        suggestions = []
        player_pos = combat_context.get("player_position")
        enemies = combat_context.get("enemies", [])
        available_cells = combat_context.get("available_trap_cells", [])
        
        if not player_pos or not enemies or not available_cells:
            return suggestions
        
        # Stratégies de placement selon le contexte
        if self.current_strategy == "kiting":
            # Pièges défensifs entre le joueur et les ennemis
            suggestions.extend(self._get_defensive_trap_positions(player_pos, enemies, available_cells))
        
        elif self.current_strategy == "trap_control":
            # Pièges de contrôle sur les chemins d'approche
            suggestions.extend(self._get_control_trap_positions(enemies, available_cells))
        
        elif self.current_strategy == "area_denial":
            # Pièges de déni de zone dans les zones stratégiques
            suggestions.extend(self._get_denial_trap_positions(enemies, available_cells))
        
        # Tri par priorité
        suggestions.sort(key=lambda x: self._evaluate_trap_position(x[0], x[1], combat_context), reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _get_defensive_trap_positions(self, player_pos: GridPosition, enemies: List[CombatEntity], 
                                    available_cells: List[GridPosition]) -> List[Tuple[GridPosition, int]]:
        """Positions de pièges défensifs pour le kiting"""
        suggestions = []
        
        for enemy in enemies:
            if not enemy.position:
                continue
            
            # Cellules entre le joueur et l'ennemi
            for cell in available_cells:
                distance_to_player = cell.distance_to(player_pos)
                distance_to_enemy = cell.distance_to(enemy.position)
                
                # Position idéale : proche de l'ennemi, pas trop proche du joueur
                if 2 <= distance_to_player <= 4 and distance_to_enemy <= 3:
                    # Piège Répulsif pour repousser
                    suggestions.append((cell, 171))
                    # Piège Sournois pour les dégâts
                    suggestions.append((cell, 169))
        
        return suggestions
    
    def _get_control_trap_positions(self, enemies: List[CombatEntity], 
                                  available_cells: List[GridPosition]) -> List[Tuple[GridPosition, int]]:
        """Positions de pièges pour contrôler la zone"""
        suggestions = []
        
        if len(enemies) >= 3:
            # Centre géométrique des ennemis
            avg_x = sum(e.position.x for e in enemies if e.position) / len(enemies)
            avg_y = sum(e.position.y for e in enemies if e.position) / len(enemies)
            center = GridPosition(int(avg_x), int(avg_y))
            
            # Pièges de masse autour du centre
            for cell in available_cells:
                if cell.distance_to(center) <= 3:
                    suggestions.append((cell, 170))  # Piège de Masse
        
        # Pièges sur les positions à forte densité d'ennemis
        for cell in available_cells:
            nearby_enemies = sum(
                1 for enemy in enemies
                if enemy.position and cell.distance_to(enemy.position) <= 2
            )
            if nearby_enemies >= 2:
                suggestions.append((cell, 172))  # Piège Empoisonné
        
        return suggestions
    
    def _get_denial_trap_positions(self, enemies: List[CombatEntity], 
                                 available_cells: List[GridPosition]) -> List[Tuple[GridPosition, int]]:
        """Positions de pièges pour déni de zone"""
        suggestions = []
        
        # Identifier les corridors et passages étroits
        high_traffic_cells = []
        
        for cell in available_cells:
            # Compter combien d'ennemis doivent passer près de cette cellule
            blocking_potential = sum(
                1 for enemy in enemies
                if enemy.position and cell.distance_to(enemy.position) <= 4
            )
            
            if blocking_potential >= 2:
                high_traffic_cells.append(cell)
        
        # Placer des pièges de masse dans les zones à fort trafic
        for cell in high_traffic_cells:
            suggestions.append((cell, 170))  # Piège de Masse
            if len(enemies) >= 4:
                suggestions.append((cell, 172))  # Piège Empoisonné pour DoT
        
        return suggestions
    
    def _evaluate_trap_position(self, position: GridPosition, spell_id: int, 
                              combat_context: Dict[str, Any]) -> float:
        """Évalue la qualité d'une position de piège"""
        enemies = combat_context.get("enemies", [])
        player_pos = combat_context.get("player_position")
        
        score = 0.0
        
        # Nombre d'ennemis affectés potentiellement
        affected_enemies = sum(
            1 for enemy in enemies
            if enemy.position and position.distance_to(enemy.position) <= 3
        )
        score += affected_enemies * 0.4
        
        # Distance au joueur (ni trop près ni trop loin)
        if player_pos:
            distance_to_player = position.distance_to(player_pos)
            if 2 <= distance_to_player <= 6:
                score += 0.3
            elif distance_to_player < 2:
                score -= 0.2  # Trop près, risque de friendly fire
        
        # Bonus selon le type de piège
        spell_info = self.spells_info.get(spell_id)
        if spell_info:
            if spell_info.aoe_size > 0:
                score += spell_info.aoe_size * 0.1
            if spell_info.push_damage > 0:
                score += 0.2
        
        return score
    
    def analyze_shooting_lanes(self, combat_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse les couloirs de tir disponibles et leur qualité
        Spécifique aux besoins tactiques d'un Cra
        """
        player_pos = combat_context.get("player_position")
        enemies = combat_context.get("enemies", [])
        blocked_cells = combat_context.get("blocked_cells", set())
        
        if not player_pos or not enemies:
            return {"lanes": [], "best_lane": None, "coverage": 0.0}
        
        shooting_lanes = []
        
        # Analyser chaque direction possible
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),     # Cardinales
            (1, 1), (-1, -1), (1, -1), (-1, 1)   # Diagonales
        ]
        
        for dx, dy in directions:
            lane_info = self._analyze_lane_direction(player_pos, dx, dy, enemies, blocked_cells)
            if lane_info["targets"] > 0:
                shooting_lanes.append(lane_info)
        
        # Tri par qualité de couloir
        shooting_lanes.sort(key=lambda x: x["quality"], reverse=True)
        
        # Calcul de la couverture globale
        total_enemies = len(enemies)
        covered_enemies = len(set().union(*[lane["enemy_ids"] for lane in shooting_lanes]))
        coverage = covered_enemies / total_enemies if total_enemies > 0 else 0.0
        
        return {
            "lanes": shooting_lanes,
            "best_lane": shooting_lanes[0] if shooting_lanes else None,
            "coverage": coverage,
            "diagonal_lanes": [lane for lane in shooting_lanes if lane["is_diagonal"]],
            "clear_lanes": [lane for lane in shooting_lanes if lane["obstructions"] == 0]
        }