"""
Classe Iop - Guerrier corps à corps spécialisé dans les dégâts de mêlée
Rôle: Damage dealer/Tank hybride avec focus sur le combat rapproché
"""

from typing import Dict, List, Optional, Any
import logging

from .base_class import (
    BaseCharacterClass, SpellInfo, SpellCategory, TargetType, 
    CombatAction, TargetEvaluation
)
from ....state.realtime_state import Character, CombatEntity, GridPosition


class IopClass(BaseCharacterClass):
    """
    Classe Iop - Maître du combat rapproché
    
    Spécialités:
    - Dégâts de mêlée élevés
    - Capacités de charge et de mobilité
    - Sorts de boost et d'invulnérabilité
    - Excellent en 1v1 et contre les groupes compacts
    """
    
    def __init__(self, character: Character, logger: Optional[logging.Logger] = None):
        super().__init__(character, logger)
        
        # Configuration spécifique Iop
        self.config.update({
            "preferred_range": 1,      # Combat au corps à corps
            "charge_threshold": 3,      # Distance pour utiliser une charge
            "berserker_hp_threshold": 0.3,  # Seuil HP pour mode berserker
            "combo_pa_reserve": 2,      # PA à garder pour les combos
            "defensive_stance_threshold": 0.4  # Seuil HP pour jouer défensif
        })
        
        # États spéciaux Iop
        self.iop_states = {
            "berserker_mode": False,
            "defensive_stance": False,
            "combo_ready": False,
            "charge_available": True
        }
    
    def get_class_role(self) -> str:
        """Iop = Damage dealer hybride"""
        return "damage"
    
    def _initialize_spells(self) -> None:
        """Initialise tous les sorts Iop avec leurs caractéristiques complètes"""
        
        # SORTS DE DÉGÂTS PRINCIPAUX
        self.spells_info[143] = SpellInfo(
            spell_id=143, name="Pression", level=1, pa_cost=3,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=18, damage_max=22, effects=["Dommages Terre"],
            aoe_size=0, cast_per_turn=3, cast_per_target=2
        )
        
        self.spells_info[144] = SpellInfo(
            spell_id=144, name="Compulsion", level=3, pa_cost=4,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=25, damage_max=29, effects=["Dommages Terre", "Bonus CC"],
            aoe_size=0, cast_per_turn=2, cast_per_target=1
        )
        
        self.spells_info[145] = SpellInfo(
            spell_id=145, name="Intimidation", level=6, pa_cost=3,
            range_min=1, range_max=2, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=15, damage_max=19, effects=["Dommages Terre", "Recul 2"],
            aoe_size=0, push_damage=2, cast_per_turn=2
        )
        
        self.spells_info[146] = SpellInfo(
            spell_id=146, name="Surpuissance", level=9, pa_cost=4,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=35, damage_max=40, effects=["Dommages Terre", "+20% dommages finaux"],
            aoe_size=0, cast_per_turn=2, cast_per_target=1
        )
        
        self.spells_info[147] = SpellInfo(
            spell_id=147, name="Puissance", level=13, pa_cost=5,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=45, damage_max=52, effects=["Dommages Terre", "Critique facilité"],
            aoe_size=0, cast_per_turn=1, cast_per_target=1, cooldown_turns=3
        )
        
        # SORTS DE CHARGE ET MOBILITÉ
        self.spells_info[148] = SpellInfo(
            spell_id=148, name="Bond", level=17, pa_cost=3,
            range_min=2, range_max=6, category=SpellCategory.MOVEMENT,
            target_type=TargetType.EMPTY_CELL, line_of_sight=False,
            effects=["Téléportation", "Dommages aux ennemis adjacents"],
            damage_min=12, damage_max=16, teleport_range=6,
            cast_per_turn=2, cooldown_turns=2
        )
        
        self.spells_info[149] = SpellInfo(
            spell_id=149, name="Charge Héroïque", level=21, pa_cost=4,
            range_min=2, range_max=8, category=SpellCategory.MOVEMENT,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=22, damage_max=28, effects=["Charge vers l'ennemi", "Dommages Terre"],
            teleport_range=8, cast_per_turn=1, cooldown_turns=4
        )
        
        self.spells_info[150] = SpellInfo(
            spell_id=150, name="Épée Divine", level=26, pa_cost=5,
            range_min=1, range_max=6, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=35, damage_max=42, effects=["Dommages Terre", "Ligne"],
            aoe_pattern="line", cast_per_turn=1, cooldown_turns=3
        )
        
        # SORTS DE ZONE
        self.spells_info[151] = SpellInfo(
            spell_id=151, name="Souffle", level=31, pa_cost=3,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=20, damage_max=24, effects=["Dommages Terre", "Zone croix"],
            aoe_size=1, aoe_pattern="cross", cast_per_turn=2
        )
        
        self.spells_info[152] = SpellInfo(
            spell_id=152, name="Concentration", level=36, pa_cost=6,
            range_min=0, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ANY, line_of_sight=False,
            damage_min=50, damage_max=60, effects=["Dommages Terre", "Zone cercle 2"],
            aoe_size=2, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=5
        )
        
        self.spells_info[153] = SpellInfo(
            spell_id=153, name="Colère de Iop", level=42, pa_cost=6,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=65, damage_max=75, effects=["Dommages Terre", "Zone cercle 3", "État Colère"],
            aoe_size=3, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=6
        )
        
        # SORTS DE BUFF ET UTILITÉ
        self.spells_info[154] = SpellInfo(
            spell_id=154, name="Vitalité", level=48, pa_cost=2,
            range_min=1, range_max=4, category=SpellCategory.BUFF,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=25, heal_max=35, effects=["+100 Vitalité", "Régénération"],
            cast_per_turn=3, cast_per_target=1
        )
        
        self.spells_info[155] = SpellInfo(
            spell_id=155, name="Immobilisation", level=54, pa_cost=3,
            range_min=1, range_max=6, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=15, damage_max=20, effects=["Dommages Terre", "État Enraciné 3 tours"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=2
        )
        
        self.spells_info[156] = SpellInfo(
            spell_id=156, name="Invulnérabilité", level=60, pa_cost=4,
            range_min=0, range_max=0, category=SpellCategory.BUFF,
            target_type=TargetType.SELF, line_of_sight=False,
            effects=["Invulnérable 1 tour", "-50% dommages infligés"],
            cast_per_turn=1, cooldown_turns=8
        )
        
        # SORTS AVANCÉS
        self.spells_info[157] = SpellInfo(
            spell_id=157, name="Tempête de Feu", level=70, pa_cost=7,
            range_min=1, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ANY, line_of_sight=False,
            damage_min=80, damage_max=95, effects=["Dommages Feu", "Zone cercle 4"],
            aoe_size=4, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=7
        )
        
        self.spells_info[158] = SpellInfo(
            spell_id=158, name="Maîtrise", level=80, pa_cost=8,
            range_min=0, range_max=0, category=SpellCategory.BUFF,
            target_type=TargetType.SELF, line_of_sight=False,
            effects=["+150 Force", "+2 PA", "+30% CC", "3 tours"],
            cast_per_turn=1, cooldown_turns=10
        )
        
        self.spells_info[159] = SpellInfo(
            spell_id=159, name="Châtiment Osmosien", level=90, pa_cost=6,
            range_min=1, range_max=1, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=100, damage_max=120, effects=["Dommages = % HP manquant", "Soins équivalents"],
            heal_min=50, heal_max=60, cast_per_turn=1, cooldown_turns=5
        )
        
        # SORTS ÉPIQUES
        self.spells_info[160] = SpellInfo(
            spell_id=160, name="Wrath of Iop", level=100, pa_cost=10,
            range_min=0, range_max=12, category=SpellCategory.DAMAGE,
            target_type=TargetType.ANY, line_of_sight=False,
            damage_min=150, damage_max=200, effects=["Dommages Terre", "Zone cercle 6", "Ignore résistances"],
            aoe_size=6, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=12
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
        """Initialise les stratégies de combat Iop"""
        self.combat_strategies = [
            "berserker",     # Mode agressif maximum
            "balanced_melee", # Équilibre entre attaque et défense
            "defensive",     # Mode défensif avec contrôle
            "mobility",      # Focus sur mobilité et repositionnement
            "group_damage"   # Optimisé pour les combats multi-cibles
        ]
        self.current_strategy = "balanced_melee"
    
    def evaluate_spell_effectiveness(self, spell_id: int, target: CombatEntity, 
                                   combat_context: Dict[str, Any]) -> float:
        """
        Évalue l'efficacité spécifique d'un sort Iop
        Prend en compte la distance, les HP, les buffs, etc.
        """
        if spell_id not in self.spells_info:
            return 0.0
        
        spell_info = self.spells_info[spell_id]
        player_pos = combat_context.get("player_position")
        
        if not player_pos or not target.position:
            return 0.0
        
        distance = player_pos.distance_to(target.position)
        base_effectiveness = 0.0
        
        # Évaluation selon le type de sort
        if spell_info.category == SpellCategory.DAMAGE:
            base_effectiveness = self._evaluate_damage_spell(spell_info, target, distance, combat_context)
        elif spell_info.category == SpellCategory.BUFF:
            base_effectiveness = self._evaluate_buff_spell(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.MOVEMENT:
            base_effectiveness = self._evaluate_movement_spell(spell_info, target, distance, combat_context)
        elif spell_info.category == SpellCategory.DEBUFF:
            base_effectiveness = self._evaluate_debuff_spell(spell_info, target, combat_context)
        
        # Modificateurs selon l'état du combat
        effectiveness = self._apply_iop_modifiers(base_effectiveness, spell_info, target, combat_context)
        
        return max(0.0, min(1.0, effectiveness))
    
    def _evaluate_damage_spell(self, spell_info: SpellInfo, target: CombatEntity, 
                              distance: int, combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de dégâts"""
        if target.is_ally:
            return 0.0  # Ne pas attaquer les alliés
        
        # Score de base selon les dégâts
        avg_damage = spell_info.get_average_damage()
        damage_ratio = avg_damage / target.max_hp if target.max_hp > 0 else 0
        damage_score = min(damage_ratio * 2, 1.0)  # Cap à 1.0
        
        # Bonus distance optimale (Iop préfère le corps à corps)
        distance_bonus = 1.0
        if spell_info.range_max == 1:  # Sort de mêlée
            distance_bonus = 1.2 if distance == 1 else 0.8
        elif spell_info.name in ["Charge Héroïque", "Bond"]:  # Sorts de mobilité
            distance_bonus = 1.1 if distance > 3 else 0.9
        
        # Bonus HP de la cible (préférer les cibles basses)
        hp_ratio = target.hp_percentage() / 100.0
        hp_bonus = 1.5 - (hp_ratio * 0.5)  # Bonus si HP bas
        
        # Bonus zone d'effet
        aoe_bonus = 1.0
        if spell_info.aoe_size > 0:
            enemies_in_range = self._count_enemies_in_aoe(target.position, spell_info, combat_context)
            aoe_bonus = 1.0 + (enemies_in_range * 0.2)  # +20% par ennemi supplémentaire
        
        # Malus cooldown
        cooldown_penalty = 0.9 if spell_info.cooldown_turns > 3 else 1.0
        
        final_score = damage_score * distance_bonus * hp_bonus * aoe_bonus * cooldown_penalty
        
        self.logger.debug(f"Évaluation sort {spell_info.name}: {final_score:.2f} "
                         f"(dégâts:{damage_score:.2f}, distance:{distance_bonus:.2f}, "
                         f"HP:{hp_bonus:.2f}, AoE:{aoe_bonus:.2f})")
        
        return final_score
    
    def _evaluate_buff_spell(self, spell_info: SpellInfo, target: CombatEntity, 
                           combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de buff"""
        if not target.is_ally:
            return 0.0  # Ne buffer que les alliés
        
        base_score = 0.7  # Score de base pour les buffs
        
        # Bonus selon l'état de l'allié
        hp_ratio = target.hp_percentage() / 100.0
        
        if spell_info.name == "Vitalité":
            # Plus l'allié a peu de HP, plus c'est prioritaire
            hp_urgency = 2.0 - hp_ratio
            return base_score * hp_urgency
        elif spell_info.name == "Invulnérabilité":
            # Très utile si HP très bas ou beaucoup d'ennemis
            enemy_count = len(combat_context.get("enemies", []))
            urgency = (2.0 - hp_ratio) * (1.0 + enemy_count * 0.1)
            return base_score * urgency
        elif spell_info.name == "Maîtrise":
            # Meilleur en début de combat avec beaucoup d'ennemis
            turn_number = combat_context.get("turn_number", 1)
            enemy_count = len(combat_context.get("enemies", []))
            timing_bonus = 2.0 if turn_number <= 2 else 1.0
            enemy_bonus = 1.0 + (enemy_count * 0.1)
            return base_score * timing_bonus * enemy_bonus
        
        return base_score
    
    def _evaluate_movement_spell(self, spell_info: SpellInfo, target: CombatEntity, 
                               distance: int, combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de mobilité"""
        if spell_info.name == "Bond":
            # Bond est bon pour se repositionner ou fuir
            player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
            if player_hp_ratio < 0.4:  # Fuir si HP bas
                return 0.9
            # Sinon, bon pour se rapprocher des ennemis distants
            return 0.7 if distance > 3 else 0.4
            
        elif spell_info.name == "Charge Héroïque":
            # Charge vers un ennemi distant
            if target.is_ally or distance < 3:
                return 0.2
            # Excellent pour initier le combat
            return 0.8 + min((distance - 3) * 0.1, 0.2)
        
        return 0.5
    
    def _evaluate_debuff_spell(self, spell_info: SpellInfo, target: CombatEntity, 
                             combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de debuff"""
        if target.is_ally:
            return 0.0
        
        if spell_info.name == "Immobilisation":
            # Très utile contre les ennemis mobiles ou à distance
            enemy_distance = combat_context.get("player_position", GridPosition(0,0)).distance_to(target.position)
            if enemy_distance > 3:  # Ennemi distant
                return 0.8
            # Utile aussi pour empêcher la fuite
            return 0.6
        
        return 0.5
    
    def _apply_iop_modifiers(self, base_effectiveness: float, spell_info: SpellInfo, 
                           target: CombatEntity, combat_context: Dict[str, Any]) -> float:
        """Applique les modificateurs spécifiques à la classe Iop"""
        modified_effectiveness = base_effectiveness
        
        # Mode berserker : bonus aux sorts de dégâts, malus aux défensifs
        if self.iop_states["berserker_mode"]:
            if spell_info.category == SpellCategory.DAMAGE:
                modified_effectiveness *= 1.3  # +30% efficacité
            elif spell_info.category in [SpellCategory.BUFF, SpellCategory.HEAL]:
                modified_effectiveness *= 0.7  # -30% efficacité
        
        # Mode défensif : l'inverse
        if self.iop_states["defensive_stance"]:
            if spell_info.category in [SpellCategory.BUFF, SpellCategory.HEAL]:
                modified_effectiveness *= 1.2
            elif spell_info.category == SpellCategory.DAMAGE and spell_info.range_max == 1:
                modified_effectiveness *= 0.8  # Moins de mêlée en défensif
        
        # Bonus combo si PA suffisants
        current_pa = combat_context.get("current_pa", 0)
        if current_pa >= 6 and spell_info.pa_cost <= current_pa - 3:
            # Bonus si on peut encore faire un autre sort après
            modified_effectiveness *= 1.1
        
        # Malus si c'est le dernier tour (préserver PA)
        time_remaining = combat_context.get("time_remaining", 30.0)
        if time_remaining < 5.0 and spell_info.pa_cost > 3:
            modified_effectiveness *= 0.8
        
        return modified_effectiveness
    
    def _count_enemies_in_aoe(self, center_position: GridPosition, spell_info: SpellInfo, 
                            combat_context: Dict[str, Any]) -> int:
        """Compte le nombre d'ennemis dans la zone d'effet d'un sort"""
        if spell_info.aoe_size == 0:
            return 0
        
        enemies = combat_context.get("enemies", [])
        count = 0
        
        for enemy in enemies:
            if not enemy.position:
                continue
            
            distance = center_position.distance_to(enemy.position)
            
            if spell_info.aoe_pattern == "circle" and distance <= spell_info.aoe_size:
                count += 1
            elif spell_info.aoe_pattern == "cross" and (
                (enemy.position.x == center_position.x and abs(enemy.position.y - center_position.y) <= spell_info.aoe_size) or
                (enemy.position.y == center_position.y and abs(enemy.position.x - center_position.x) <= spell_info.aoe_size)
            ):
                count += 1
            elif spell_info.aoe_pattern == "line":
                # Implémentation simplifiée pour ligne
                if distance <= spell_info.aoe_size:
                    count += 1
        
        return count
    
    def get_spell_combo_sequences(self) -> List[List[int]]:
        """
        Retourne les combos optimaux pour Iop selon la situation
        """
        combos = []
        
        # Combo burst mêlée (PA élevés requis)
        combos.append([
            154,  # Vitalité (si HP bas)
            158,  # Maîtrise (si début combat) 
            147,  # Puissance
            144   # Compulsion
        ])
        
        # Combo charge + dégâts
        combos.append([
            149,  # Charge Héroïque
            153,  # Colère de Iop
            143   # Pression (finition)
        ])
        
        # Combo zone pour multiples ennemis
        combos.append([
            152,  # Concentration 
            151,  # Souffle
            157   # Tempête de Feu
        ])
        
        # Combo défensif
        combos.append([
            156,  # Invulnérabilité
            154,  # Vitalité
            155   # Immobilisation (contrôle)
        ])
        
        # Combo mobilité + finition
        combos.append([
            148,  # Bond
            150,  # Épée Divine
            159   # Châtiment Osmosien
        ])
        
        return combos
    
    def _update_strategy(self, combat_context: Dict[str, Any]) -> None:
        """Met à jour la stratégie spécifique Iop selon le contexte"""
        super()._update_strategy(combat_context)
        
        # Mise à jour des états spéciaux Iop
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        enemies = combat_context.get("enemies", [])
        allies = combat_context.get("allies", [])
        
        # Mode berserker si HP très bas
        self.iop_states["berserker_mode"] = player_hp_ratio < self.config["berserker_hp_threshold"]
        
        # Mode défensif si HP modérément bas ou en infériorité numérique
        self.iop_states["defensive_stance"] = (
            player_hp_ratio < self.config["defensive_stance_threshold"] or
            len(enemies) > len(allies) + 1
        )
        
        # Combo disponible si assez de PA
        current_pa = combat_context.get("current_pa", 0)
        self.iop_states["combo_ready"] = current_pa >= 6
        
        # Stratégie spécifique selon l'état
        if self.iop_states["berserker_mode"]:
            self.current_strategy = "berserker"
        elif self.iop_states["defensive_stance"]:
            self.current_strategy = "defensive"
        elif len(enemies) > 2:
            self.current_strategy = "group_damage"
        else:
            self.current_strategy = "balanced_melee"
        
        self.logger.debug(f"États Iop: berserker={self.iop_states['berserker_mode']}, "
                         f"défensif={self.iop_states['defensive_stance']}, "
                         f"stratégie={self.current_strategy}")
    
    def get_preferred_positioning(self, combat_context: Dict[str, Any]) -> List[GridPosition]:
        """
        Retourne les positions préférées pour un Iop selon la stratégie
        """
        preferred_positions = []
        enemies = combat_context.get("enemies", [])
        
        if not enemies:
            return preferred_positions
        
        if self.current_strategy == "berserker":
            # Au corps à corps avec l'ennemi le plus faible
            weakest_enemy = min(enemies, key=lambda e: e.current_hp)
            if weakest_enemy.position:
                # Positions adjacentes à l'ennemi le plus faible
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    pos = GridPosition(
                        weakest_enemy.position.x + dx,
                        weakest_enemy.position.y + dy
                    )
                    preferred_positions.append(pos)
        
        elif self.current_strategy == "defensive":
            # Près des alliés, loin des ennemis
            allies = combat_context.get("allies", [])
            if allies:
                # Position centrale par rapport aux alliés
                avg_x = sum(a.position.x for a in allies if a.position) / len(allies)
                avg_y = sum(a.position.y for a in allies if a.position) / len(allies)
                preferred_positions.append(GridPosition(int(avg_x), int(avg_y)))
        
        elif self.current_strategy == "group_damage":
            # Position centrale pour maximiser les dégâts de zone
            if len(enemies) >= 2:
                # Centre géométrique des ennemis
                avg_x = sum(e.position.x for e in enemies if e.position) / len(enemies)
                avg_y = sum(e.position.y for e in enemies if e.position) / len(enemies)
                
                # Positions autour du centre pour les sorts de zone
                center = GridPosition(int(avg_x), int(avg_y))
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    pos = GridPosition(center.x + dx, center.y + dy)
                    preferred_positions.append(pos)
        
        else:  # balanced_melee
            # Mélange équilibré : accessible à plusieurs ennemis
            for enemy in enemies[:3]:  # Top 3 des ennemis
                if enemy.position:
                    # Positions à distance 1-2 de l'ennemi
                    for distance in [1, 2]:
                        for dx in range(-distance, distance + 1):
                            for dy in range(-distance, distance + 1):
                                if abs(dx) + abs(dy) == distance:  # Distance Manhattan
                                    pos = GridPosition(
                                        enemy.position.x + dx,
                                        enemy.position.y + dy
                                    )
                                    preferred_positions.append(pos)
        
        return preferred_positions[:10]  # Limiter à 10 positions max
    
    def get_emergency_actions(self, combat_context: Dict[str, Any]) -> List[CombatAction]:
        """
        Retourne les actions d'urgence quand les HP sont critiques
        """
        emergency_actions = []
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        
        if player_hp_ratio > 0.3:  # Pas d'urgence
            return emergency_actions
        
        # 1. Invulnérabilité si disponible
        if self.can_cast_spell(156):  # Invulnérabilité
            emergency_actions.append(CombatAction(
                action_type="cast_spell",
                spell_id=156,
                target_entity_id=None,  # Sort sur soi
                priority=100,
                reasoning="Invulnérabilité d'urgence - HP critiques"
            ))
        
        # 2. Vitalité pour se soigner
        if self.can_cast_spell(154):  # Vitalité
            emergency_actions.append(CombatAction(
                action_type="cast_spell",
                spell_id=154,
                target_entity_id=None,  # Sort sur soi
                priority=90,
                reasoning="Vitalité d'urgence - HP critiques"
            ))
        
        # 3. Bond pour fuir si possible
        if self.can_cast_spell(148):  # Bond
            # Chercher une position éloignée des ennemis
            optimal_position = self._find_escape_position(combat_context)
            if optimal_position:
                emergency_actions.append(CombatAction(
                    action_type="cast_spell",
                    spell_id=148,
                    target_position=optimal_position,
                    priority=80,
                    reasoning="Bond de fuite - HP critiques"
                ))
        
        return emergency_actions
    
    def _find_escape_position(self, combat_context: Dict[str, Any]) -> Optional[GridPosition]:
        """Trouve une position d'évasion loin des ennemis"""
        player_pos = combat_context.get("player_position")
        enemies = combat_context.get("enemies", [])
        available_moves = combat_context.get("available_moves", [])
        
        if not player_pos or not enemies or not available_moves:
            return None
        
        best_position = None
        best_distance = 0
        
        for position in available_moves:
            # Calculer la distance minimale aux ennemis
            min_enemy_distance = min(
                position.distance_to(enemy.position) 
                for enemy in enemies 
                if enemy.position
            )
            
            # Préférer les positions les plus éloignées des ennemis
            if min_enemy_distance > best_distance:
                best_distance = min_enemy_distance
                best_position = position
        
        return best_position if best_distance >= 3 else None
    
    def analyze_threat_level(self, combat_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyse le niveau de menace global et par ennemi
        Spécifique aux considérations tactiques Iop
        """
        enemies = combat_context.get("enemies", [])
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        
        threat_analysis = {
            "global_threat": 0.0,
            "immediate_threat": 0.0,  # Menace au tour suivant
            "individual_threats": {}
        }
        
        if not enemies:
            return threat_analysis
        
        total_threat = 0.0
        immediate_threat = 0.0
        
        for enemy in enemies:
            # Facteurs de menace pour un Iop
            enemy_threat = 0.0
            
            # Menace de base selon le niveau et HP
            base_threat = min(enemy.level / 100.0, 1.0)
            hp_threat = enemy.hp_percentage() / 100.0
            enemy_threat = base_threat * hp_threat
            
            # Modificateur distance (plus dangereux si proche pour un Iop)
            if enemy.position:
                player_pos = combat_context.get("player_position")
                if player_pos:
                    distance = player_pos.distance_to(enemy.position)
                    if distance <= 1:
                        enemy_threat *= 1.5  # Très dangereux en mêlée
                        immediate_threat += enemy_threat
                    elif distance <= 3:
                        enemy_threat *= 1.2  # Moyennement dangereux
                    # Les ennemis distants sont moins menaçants pour un Iop
            
            # Modificateur selon le type d'ennemi (si on a l'info)
            if enemy.is_monster:
                enemy_threat *= 1.1  # Les monstres sont plus prévisibles
            
            threat_analysis["individual_threats"][enemy.entity_id] = enemy_threat
            total_threat += enemy_threat
        
        # Calcul de la menace globale
        threat_analysis["global_threat"] = min(total_threat / len(enemies), 1.0)
        threat_analysis["immediate_threat"] = min(immediate_threat, 1.0)
        
        # Ajustement selon les HP du joueur
        if player_hp_ratio < 0.5:
            threat_analysis["global_threat"] *= (2.0 - player_hp_ratio)
            threat_analysis["immediate_threat"] *= (2.0 - player_hp_ratio)
        
        return threat_analysis