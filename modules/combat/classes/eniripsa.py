"""
Classe Eniripsa - Guérisseur et support spécialisé dans les soins et le soutien
Rôle: Support/Healer avec capacités de buff, debuff et contrôle
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

from .base_class import (
    BaseCharacterClass, SpellInfo, SpellCategory, TargetType, 
    CombatAction, TargetEvaluation
)
from ....state.realtime_state import Character, CombatEntity, GridPosition


class EniClass(BaseCharacterClass):
    """
    Classe Eniripsa - Maître des soins et du support tactique
    
    Spécialités:
    - Soins puissants mono-cible et AoE
    - Buffs de groupe et protection
    - Résurrection et immunités
    - Contrôle via debuffs et sorts d'état
    """
    
    def __init__(self, character: Character, logger: Optional[logging.Logger] = None):
        super().__init__(character, logger)
        
        # Configuration spécifique Eniripsa
        self.config.update({
            "heal_priority_threshold": 0.6,    # Seuil HP pour priorité soins
            "emergency_heal_threshold": 0.3,   # Seuil HP urgence
            "group_heal_threshold": 0.5,       # Seuil pour soins de groupe
            "buff_optimal_timing": 2,          # Tours optimaux pour buffer
            "mana_conservation_ratio": 0.3,    # Ratio PA à conserver
            "support_range": 8                 # Portée optimale support
        })
        
        # États spéciaux Eniripsa
        self.eni_states = {
            "healing_mode": False,       # Mode soins prioritaires
            "support_mode": False,       # Mode support/buff
            "emergency_mode": False,     # Mode urgence
            "resurrection_ready": False, # Résurrection disponible
            "group_buff_active": False   # Buffs de groupe actifs
        }
        
        # Tracking des états appliqués
        self.active_buffs = {}      # {entity_id: {buff_name: turns_remaining}}
        self.active_debuffs = {}    # {entity_id: {debuff_name: turns_remaining}}
        self.heal_history = []      # Historique des soins pour optimisation
    
    def get_class_role(self) -> str:
        """Eniripsa = Support/Healer"""
        return "support"
    
    def _initialize_spells(self) -> None:
        """Initialise tous les sorts Eniripsa avec leurs caractéristiques complètes"""
        
        # SORTS DE SOINS DE BASE
        self.spells_info[201] = SpellInfo(
            spell_id=201, name="Mot Curatif", level=1, pa_cost=2,
            range_min=1, range_max=6, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=15, heal_max=20, effects=["Soins Eau"],
            cast_per_turn=4, cast_per_target=3
        )
        
        self.spells_info[202] = SpellInfo(
            spell_id=202, name="Mot Soignant", level=3, pa_cost=3,
            range_min=1, range_max=8, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=25, heal_max=32, effects=["Soins Eau", "Enlève poison"],
            cast_per_turn=3, cast_per_target=2
        )
        
        self.spells_info[203] = SpellInfo(
            spell_id=203, name="Mot Régénérant", level=6, pa_cost=4,
            range_min=1, range_max=8, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=30, heal_max=40, effects=["Soins Eau", "Régénération 3 tours"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=1
        )
        
        self.spells_info[204] = SpellInfo(
            spell_id=204, name="Mot Reconstituant", level=9, pa_cost=5,
            range_min=1, range_max=10, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=45, heal_max=58, effects=["Soins Eau", "Restaure PM"],
            cast_per_turn=2, cast_per_target=1
        )
        
        # SORTS DE SOINS AVANCÉS
        self.spells_info[205] = SpellInfo(
            spell_id=205, name="Mot Revitalisant", level=13, pa_cost=6,
            range_min=1, range_max=10, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=65, heal_max=80, effects=["Soins Eau", "Bonus Vitalité temporaire"],
            cast_per_turn=1, cast_per_target=1, cooldown_turns=2
        )
        
        self.spells_info[206] = SpellInfo(
            spell_id=206, name="Soin Collectif", level=17, pa_cost=5,
            range_min=1, range_max=6, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=35, heal_max=45, effects=["Soins Eau", "Zone cercle 2"],
            aoe_size=2, aoe_pattern="circle", cast_per_turn=2, cooldown_turns=3
        )
        
        self.spells_info[207] = SpellInfo(
            spell_id=207, name="Résurrection", level=21, pa_cost=8,
            range_min=1, range_max=4, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=100, heal_max=150, effects=["Ressuscite", "Restaure 50% HP"],
            cast_per_turn=1, cooldown_turns=10
        )
        
        # SORTS D'ATTAQUE (Dégâts pour Eniripsa)
        self.spells_info[208] = SpellInfo(
            spell_id=208, name="Mot Blessant", level=26, pa_cost=3,
            range_min=1, range_max=8, category=SpellCategory.DAMAGE,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=20, damage_max=26, effects=["Dommages Eau", "Vole HP"],
            heal_min=10, heal_max=13, cast_per_turn=3, cast_per_target=2
        )
        
        self.spells_info[209] = SpellInfo(
            spell_id=209, name="Mot Paralysant", level=31, pa_cost=4,
            range_min=1, range_max=6, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            damage_min=15, damage_max=20, effects=["Dommages Eau", "Paralysie 2 tours"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=2
        )
        
        self.spells_info[210] = SpellInfo(
            spell_id=210, name="Mot d'Épouvante", level=36, pa_cost=3,
            range_min=1, range_max=8, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            effects=["Fuite forcée", "État Peur 3 tours", "-50% dommages"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=4
        )
        
        # SORTS DE BUFF ET PROTECTION
        self.spells_info[211] = SpellInfo(
            spell_id=211, name="Mot de Prévention", level=42, pa_cost=4,
            range_min=1, range_max=8, category=SpellCategory.BUFF,
            target_type=TargetType.ALLY, line_of_sight=True,
            effects=["Immunité débuffs", "Résistance +100", "3 tours"],
            cast_per_turn=2, cast_per_target=1, cooldown_turns=5
        )
        
        self.spells_info[212] = SpellInfo(
            spell_id=212, name="Mot de Jouvence", level=48, pa_cost=3,
            range_min=1, range_max=6, category=SpellCategory.BUFF,
            target_type=TargetType.ALLY, line_of_sight=True,
            effects=["+2 PM", "+1 PA", "Enlève malédictions", "4 tours"],
            cast_per_turn=3, cast_per_target=1, cooldown_turns=3
        )
        
        self.spells_info[213] = SpellInfo(
            spell_id=213, name="Mot de Sacrifice", level=54, pa_cost=2,
            range_min=1, range_max=4, category=SpellCategory.UTILITY,
            target_type=TargetType.ALLY, line_of_sight=True,
            effects=["Échange HP", "Transfert états", "Sacrifice partiel"],
            cast_per_turn=1, cast_per_target=1, cooldown_turns=6
        )
        
        # SORTS DE CONTRÔLE AVANCÉ
        self.spells_info[214] = SpellInfo(
            spell_id=214, name="Cercle Curatif", level=60, pa_cost=6,
            range_min=0, range_max=0, category=SpellCategory.HEAL,
            target_type=TargetType.SELF, line_of_sight=False,
            heal_min=40, heal_max=55, effects=["Zone cercle 4", "Soins tous alliés"],
            aoe_size=4, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=8
        )
        
        self.spells_info[215] = SpellInfo(
            spell_id=215, name="Immunité", level=70, pa_cost=5,
            range_min=1, range_max=6, category=SpellCategory.BUFF,
            target_type=TargetType.ALLY, line_of_sight=True,
            effects=["Immunité complète", "Invulnérable 1 tour", "Enlève tous débuffs"],
            cast_per_turn=1, cooldown_turns=12
        )
        
        self.spells_info[216] = SpellInfo(
            spell_id=216, name="Mot de Régénération Ultime", level=80, pa_cost=7,
            range_min=1, range_max=8, category=SpellCategory.HEAL,
            target_type=TargetType.ALLY, line_of_sight=True,
            heal_min=80, heal_max=120, effects=["Soins Eau", "Régénération 5 tours", "Immunité poison"],
            cast_per_turn=1, cooldown_turns=6
        )
        
        # SORTS ÉPIQUES ET MAÎTRISES
        self.spells_info[217] = SpellInfo(
            spell_id=217, name="Bénédiction Divine", level=90, pa_cost=8,
            range_min=0, range_max=12, category=SpellCategory.BUFF,
            target_type=TargetType.ANY, line_of_sight=False,
            effects=["Zone cercle 6", "Tous alliés", "+200 tous stats", "Immunités", "5 tours"],
            aoe_size=6, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=15
        )
        
        self.spells_info[218] = SpellInfo(
            spell_id=218, name="Mot de Résurrection de Masse", level=100, pa_cost=12,
            range_min=0, range_max=0, category=SpellCategory.HEAL,
            target_type=TargetType.SELF, line_of_sight=False,
            heal_min=200, heal_max=300, effects=["Zone cercle 8", "Ressuscite tous", "Soins complets"],
            aoe_size=8, aoe_pattern="circle", cast_per_turn=1, cooldown_turns=20
        )
        
        # SORTS UTILITAIRES SPÉCIALISÉS
        self.spells_info[219] = SpellInfo(
            spell_id=219, name="Mot d'Amitié", level=25, pa_cost=2,
            range_min=1, range_max=8, category=SpellCategory.UTILITY,
            target_type=TargetType.ANY, line_of_sight=True,
            effects=["Change camp temporaire", "Contrôle ennemi 1 tour"],
            cast_per_turn=1, cooldown_turns=8
        )
        
        self.spells_info[220] = SpellInfo(
            spell_id=220, name="Mot de Silence", level=45, pa_cost=3,
            range_min=1, range_max=6, category=SpellCategory.DEBUFF,
            target_type=TargetType.ENEMY, line_of_sight=True,
            effects=["Silence", "Pas de sorts 2 tours", "Zone cercle 1"],
            aoe_size=1, aoe_pattern="circle", cast_per_turn=2, cooldown_turns=5
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
        """Initialise les stratégies de combat Eniripsa"""
        self.combat_strategies = [
            "pure_healer",      # Soins purs, support passif
            "combat_medic",     # Soins + dégâts tactiques
            "group_support",    # Focus buffs et soins de groupe
            "control_support",  # Contrôle via debuffs et états
            "emergency_medic"   # Mode urgence, soins prioritaires
        ]
        self.current_strategy = "combat_medic"
    
    def evaluate_spell_effectiveness(self, spell_id: int, target: CombatEntity, 
                                   combat_context: Dict[str, Any]) -> float:
        """
        Évalue l'efficacité spécifique d'un sort Eniripsa
        Prend en compte l'urgence des soins, l'état des alliés, etc.
        """
        if spell_id not in self.spells_info:
            return 0.0
        
        spell_info = self.spells_info[spell_id]
        
        base_effectiveness = 0.0
        
        # Évaluation selon le type de sort
        if spell_info.category == SpellCategory.HEAL:
            base_effectiveness = self._evaluate_heal_spell_eni(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.BUFF:
            base_effectiveness = self._evaluate_buff_spell_eni(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.DEBUFF:
            base_effectiveness = self._evaluate_debuff_spell_eni(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.DAMAGE:
            base_effectiveness = self._evaluate_damage_spell_eni(spell_info, target, combat_context)
        elif spell_info.category == SpellCategory.UTILITY:
            base_effectiveness = self._evaluate_utility_spell_eni(spell_info, target, combat_context)
        
        # Modificateurs Eniripsa spécifiques
        effectiveness = self._apply_eni_modifiers(base_effectiveness, spell_info, target, combat_context)
        
        return max(0.0, min(1.0, effectiveness))
    
    def _evaluate_heal_spell_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                               combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de soin"""
        if not target or not target.is_ally:
            return 0.0  # Ne soigner que les alliés
        
        # Urgence selon les HP de la cible
        hp_ratio = target.hp_percentage() / 100.0
        urgency_factor = self._calculate_heal_urgency(hp_ratio)
        
        # Efficacité du sort selon les HP manquants
        hp_missing = target.max_hp - target.current_hp
        avg_heal = spell_info.get_average_heal()
        
        if hp_missing <= 0:
            return 0.1  # Cible déjà pleine, très faible priorité
        
        # Ratio efficacité = soins / HP manquants (idéal = 1.0)
        efficiency_ratio = min(avg_heal / hp_missing, 2.0) if hp_missing > 0 else 0.1
        
        # Score de base
        base_score = urgency_factor * efficiency_ratio
        
        # Bonus sorts spécifiques
        spell_bonus = self._get_heal_spell_bonus_eni(spell_info, target, combat_context)
        
        # Bonus AoE si plusieurs alliés blessés
        aoe_bonus = 1.0
        if spell_info.aoe_size > 0:
            wounded_allies = self._count_wounded_allies_in_aoe(target.position, spell_info, combat_context)
            aoe_bonus = 1.0 + (wounded_allies * 0.3)
        
        # Malus coût PA élevé
        pa_efficiency = 1.0 - (spell_info.pa_cost - 2) * 0.05  # Référence = 2 PA
        pa_efficiency = max(0.5, pa_efficiency)
        
        final_score = base_score * spell_bonus * aoe_bonus * pa_efficiency
        
        self.logger.debug(f"Évaluation soin {spell_info.name} sur {target.name}: {final_score:.2f} "
                         f"(urgence:{urgency_factor:.2f}, efficacité:{efficiency_ratio:.2f}, "
                         f"AoE:{aoe_bonus:.2f})")
        
        return final_score
    
    def _calculate_heal_urgency(self, hp_ratio: float) -> float:
        """Calcule l'urgence d'un soin selon le ratio de HP"""
        if hp_ratio <= 0.2:
            return 2.0      # Critique
        elif hp_ratio <= 0.4:
            return 1.5      # Urgent
        elif hp_ratio <= 0.6:
            return 1.2      # Important
        elif hp_ratio <= 0.8:
            return 0.8      # Normal
        else:
            return 0.3      # Faible priorité
    
    def _get_heal_spell_bonus_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                                combat_context: Dict[str, Any]) -> float:
        """Bonus spécifiques aux sorts de soin Eni"""
        bonus = 1.0
        
        # Mot Curatif : rapide et efficace
        if spell_info.name == "Mot Curatif":
            bonus = 1.1  # Bon sort de base
        
        # Mot Régénérant : excellent pour soins over time
        elif spell_info.name == "Mot Régénérant":
            hp_ratio = target.hp_percentage() / 100.0
            if hp_ratio > 0.3:  # Plus efficace sur cibles pas trop basses
                bonus = 1.3
        
        # Résurrection : priorité absolue sur alliés morts
        elif spell_info.name == "Résurrection":
            if target.current_hp <= 0:
                bonus = 3.0  # Priorité maximale
            else:
                bonus = 0.1  # Inutile sur vivants
        
        # Soin Collectif : bonus selon nombre d'alliés blessés
        elif spell_info.name == "Soin Collectif":
            allies = combat_context.get("allies", [])
            wounded_count = sum(1 for ally in allies if ally.hp_percentage() < 80)
            bonus = 1.0 + (wounded_count * 0.2)
        
        # Cercle Curatif : excellent pour soins de groupe
        elif spell_info.name == "Cercle Curatif":
            allies = combat_context.get("allies", [])
            wounded_count = sum(1 for ally in allies if ally.hp_percentage() < 70)
            if wounded_count >= 3:
                bonus = 1.5
        
        return bonus
    
    def _evaluate_buff_spell_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                               combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de buff"""
        if not target or not target.is_ally:
            return 0.0
        
        base_score = 0.6  # Score de base pour buffs
        
        # Timing du buff (meilleur en début de combat)
        turn_number = combat_context.get("turn_number", 1)
        timing_bonus = 1.5 if turn_number <= 3 else 1.0
        
        # Bonus selon l'état de la cible
        target_bonus = self._calculate_buff_target_bonus_eni(spell_info, target, combat_context)
        
        # Bonus selon la situation tactique
        tactical_bonus = self._calculate_buff_tactical_bonus_eni(spell_info, combat_context)
        
        final_score = base_score * timing_bonus * target_bonus * tactical_bonus
        
        return final_score
    
    def _calculate_buff_target_bonus_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                                       combat_context: Dict[str, Any]) -> float:
        """Calcule le bonus selon la cible du buff"""
        bonus = 1.0
        
        # Vérifier si la cible a déjà ce buff
        if target.entity_id in self.active_buffs:
            active_target_buffs = self.active_buffs[target.entity_id]
            if spell_info.name in active_target_buffs:
                return 0.2  # Déjà présent, faible priorité
        
        # Bonus selon le type de buff
        if spell_info.name == "Mot de Prévention":
            # Excellent si la cible a déjà des débuffs
            if target.entity_id in self.active_debuffs:
                bonus = 1.5
            # Bon sur les tanks exposés
            player_pos = combat_context.get("player_position")
            if player_pos and target.position:
                enemies = combat_context.get("enemies", [])
                nearby_enemies = sum(
                    1 for enemy in enemies
                    if enemy.position and target.position.distance_to(enemy.position) <= 3
                )
                bonus = 1.0 + (nearby_enemies * 0.2)
        
        elif spell_info.name == "Mot de Jouvence":
            # Excellent sur les classes qui consomment beaucoup de PA/PM
            bonus = 1.2  # Toujours utile
        
        elif spell_info.name == "Immunité":
            # Priorité sur les cibles critiques ou avec beaucoup de débuffs
            hp_ratio = target.hp_percentage() / 100.0
            if hp_ratio < 0.4:
                bonus = 1.8  # Très utile sur cibles fragiles
            if target.entity_id in self.active_debuffs:
                debuff_count = len(self.active_debuffs[target.entity_id])
                bonus *= (1.0 + debuff_count * 0.3)
        
        return bonus
    
    def _calculate_buff_tactical_bonus_eni(self, spell_info: SpellInfo, 
                                         combat_context: Dict[str, Any]) -> float:
        """Calcule le bonus tactique selon la situation"""
        bonus = 1.0
        
        enemies = combat_context.get("enemies", [])
        allies = combat_context.get("allies", [])
        
        # Bénédiction Divine : excellent avec beaucoup d'alliés
        if spell_info.name == "Bénédiction Divine":
            bonus = 1.0 + (len(allies) * 0.3)
            if len(enemies) >= 4:  # Combat difficile
                bonus *= 1.5
        
        # Buffs défensifs plus importants si beaucoup d'ennemis
        if "Prévention" in spell_info.name or "Immunité" in spell_info.name:
            enemy_threat = min(len(enemies) * 0.2, 1.0)
            bonus += enemy_threat
        
        return bonus
    
    def _evaluate_debuff_spell_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                                 combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de debuff"""
        if not target or target.is_ally:
            return 0.0
        
        base_score = 0.6
        
        # Bonus selon le type de débuff
        if spell_info.name == "Mot Paralysant":
            # Très efficace contre ennemis mobiles
            base_score = 0.8
            
        elif spell_info.name == "Mot d'Épouvante":
            # Excellent pour contrôler ennemis dangereux
            hp_ratio = target.hp_percentage() / 100.0
            threat_bonus = 2.0 - hp_ratio  # Plus l'ennemi a de HP, plus c'est utile
            base_score *= threat_bonus
            
        elif spell_info.name == "Mot de Silence":
            # Très efficace contre les casters
            enemy_name = target.name.lower()
            if any(keyword in enemy_name for keyword in ["mage", "eni", "xelor", "sadida"]):
                base_score = 1.2  # Excellent contre casters
        
        # Malus si la cible a déjà ce débuff
        if target.entity_id in self.active_debuffs:
            if spell_info.name in self.active_debuffs[target.entity_id]:
                base_score *= 0.3
        
        return base_score
    
    def _evaluate_damage_spell_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                                 combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort de dégâts Eni (limité)"""
        if not target or target.is_ally:
            return 0.0
        
        # Les Eni ne sont pas des DD, score modéré
        base_score = 0.4
        
        if spell_info.name == "Mot Blessant":
            # Soigne le lanceur, double utilité
            player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
            if player_hp_ratio < 0.7:
                base_score = 0.7  # Plus utile si l'Eni a besoin de HP
        
        # Bonus si stratégie combat_medic
        if self.current_strategy == "combat_medic":
            base_score *= 1.3
        
        return base_score
    
    def _evaluate_utility_spell_eni(self, spell_info: SpellInfo, target: CombatEntity, 
                                  combat_context: Dict[str, Any]) -> float:
        """Évalue l'efficacité d'un sort utilitaire"""
        base_score = 0.5
        
        if spell_info.name == "Mot d'Amitié":
            # Très situationnel, excellent contre ennemis forts isolés
            enemies = combat_context.get("enemies", [])
            if target and not target.is_ally:
                threat_level = self._calculate_enemy_threat_for_control(target, combat_context)
                base_score = 0.3 + (threat_level * 0.7)
        
        elif spell_info.name == "Mot de Sacrifice":
            # Utile pour sauver des alliés critiques
            if target and target.is_ally:
                hp_ratio = target.hp_percentage() / 100.0
                player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
                # Utile si allié critique et Eni en meilleur état
                if hp_ratio < 0.3 and player_hp_ratio > 0.5:
                    base_score = 1.2
        
        return base_score
    
    def _apply_eni_modifiers(self, base_effectiveness: float, spell_info: SpellInfo, 
                           target: Optional[CombatEntity], combat_context: Dict[str, Any]) -> float:
        """Applique les modificateurs spécifiques à la classe Eniripsa"""
        modified_effectiveness = base_effectiveness
        
        # Mode soins d'urgence : bonus massif aux soins
        if self.eni_states["emergency_mode"]:
            if spell_info.category == SpellCategory.HEAL:
                modified_effectiveness *= 1.5
            else:
                modified_effectiveness *= 0.7  # Malus aux autres sorts
        
        # Mode support pur : bonus aux buffs et soins
        if self.current_strategy == "pure_healer":
            if spell_info.category in [SpellCategory.HEAL, SpellCategory.BUFF]:
                modified_effectiveness *= 1.2
            elif spell_info.category in [SpellCategory.DAMAGE, SpellCategory.DEBUFF]:
                modified_effectiveness *= 0.8
        
        # Mode contrôle : bonus aux débuffs
        if self.current_strategy == "control_support":
            if spell_info.category == SpellCategory.DEBUFF:
                modified_effectiveness *= 1.3
        
        # Conservation PA : malus aux sorts coûteux en fin de tour
        current_pa = combat_context.get("current_pa", 0)
        if spell_info.pa_cost > current_pa * 0.7:  # Plus de 70% des PA
            modified_effectiveness *= 0.9
        
        # Bonus cooldown si situation critique
        if spell_info.cooldown_turns >= 5:
            allies = combat_context.get("allies", [])
            critical_allies = sum(1 for ally in allies if ally.hp_percentage() < 30)
            if critical_allies >= 2:
                modified_effectiveness *= 1.2  # Utiliser les gros cooldowns
        
        return modified_effectiveness
    
    def _count_wounded_allies_in_aoe(self, center_position: Optional[GridPosition], 
                                   spell_info: SpellInfo, combat_context: Dict[str, Any]) -> int:
        """Compte les alliés blessés dans la zone d'effet"""
        if not center_position or spell_info.aoe_size == 0:
            return 0
        
        allies = combat_context.get("allies", [])
        count = 0
        
        for ally in allies:
            if not ally.position or ally.hp_percentage() >= 90:  # Pas blessé
                continue
            
            distance = center_position.distance_to(ally.position)
            if distance <= spell_info.aoe_size:
                count += 1
        
        return count
    
    def _calculate_enemy_threat_for_control(self, enemy: CombatEntity, 
                                          combat_context: Dict[str, Any]) -> float:
        """Calcule la menace d'un ennemi pour les sorts de contrôle"""
        threat = 0.3  # Base
        
        # Facteur HP
        hp_factor = enemy.hp_percentage() / 100.0
        threat += hp_factor * 0.4
        
        # Facteur niveau
        level_factor = min(enemy.level / 50.0, 2.0)
        threat += level_factor * 0.3
        
        return min(threat, 1.0)
    
    def get_spell_combo_sequences(self) -> List[List[int]]:
        """Retourne les combos optimaux pour Eniripsa selon la situation"""
        combos = []
        
        # Combo soins d'urgence
        combos.append([
            215,  # Immunité
            216,  # Mot de Régénération Ultime  
            205,  # Mot Revitalisant
            206   # Soin Collectif
        ])
        
        # Combo support de groupe
        combos.append([
            217,  # Bénédiction Divine
            212,  # Mot de Jouvence
            211,  # Mot de Prévention
            214   # Cercle Curatif
        ])
        
        # Combo résurrection et stabilisation
        combos.append([
            207,  # Résurrection
            216,  # Mot de Régénération Ultime
            211,  # Mot de Prévention
            204   # Mot Reconstituant
        ])
        
        # Combo contrôle ennemi
        combos.append([
            220,  # Mot de Silence
            209,  # Mot Paralysant
            210,  # Mot d'Épouvante
            219   # Mot d'Amitié
        ])
        
        # Combo combat médic
        combos.append([
            208,  # Mot Blessant (dégâts + soins)
            203,  # Mot Régénérant
            202,  # Mot Soignant
            201   # Mot Curatif
        ])
        
        return combos
    
    def _update_strategy(self, combat_context: Dict[str, Any]) -> None:
        """Met à jour la stratégie spécifique Eniripsa selon le contexte"""
        super()._update_strategy(combat_context)
        
        # Analyse de l'état des alliés
        allies = combat_context.get("allies", [])
        enemies = combat_context.get("enemies", [])
        player_hp_ratio = combat_context.get("player_hp_ratio", 1.0)
        
        # Calcul des statistiques d'alliés
        if allies:
            ally_hp_ratios = [ally.hp_percentage() / 100.0 for ally in allies]
            avg_ally_hp = sum(ally_hp_ratios) / len(ally_hp_ratios)
            critical_allies = sum(1 for hp in ally_hp_ratios if hp < 0.3)
            wounded_allies = sum(1 for hp in ally_hp_ratios if hp < 0.7)
        else:
            avg_ally_hp = 1.0
            critical_allies = 0
            wounded_allies = 0
        
        # États spéciaux Eniripsa
        self.eni_states["emergency_mode"] = (
            critical_allies >= 2 or 
            (critical_allies >= 1 and player_hp_ratio < 0.4)
        )
        
        self.eni_states["healing_mode"] = (
            avg_ally_hp < self.config["heal_priority_threshold"] or
            wounded_allies >= len(allies) // 2
        )
        
        self.eni_states["support_mode"] = (
            avg_ally_hp > 0.7 and len(enemies) >= 3
        )
        
        # Sélection de stratégie
        if self.eni_states["emergency_mode"]:
            self.current_strategy = "emergency_medic"
        elif self.eni_states["healing_mode"]:
            self.current_strategy = "pure_healer"
        elif self.eni_states["support_mode"]:
            self.current_strategy = "group_support"
        elif len(enemies) >= 4:
            self.current_strategy = "control_support"
        else:
            self.current_strategy = "combat_medic"
        
        self.logger.debug(f"États Eni: urgence={self.eni_states['emergency_mode']}, "
                         f"soins={self.eni_states['healing_mode']}, "
                         f"support={self.eni_states['support_mode']}, "
                         f"stratégie={self.current_strategy}")
    
    def get_heal_priority_list(self, combat_context: Dict[str, Any]) -> List[TargetEvaluation]:
        """
        Retourne la liste des cibles prioritaires pour les soins
        Trié par urgence décroissante
        """
        allies = combat_context.get("allies", [])
        if not allies:
            return []
        
        heal_priorities = []
        
        for ally in allies:
            if not ally.position:
                continue
            
            # Calcul du score de priorité de soin
            hp_ratio = ally.hp_percentage() / 100.0
            urgency = self._calculate_heal_urgency(hp_ratio)
            
            # Facteurs modificateurs
            distance_factor = self._calculate_distance_priority_factor(ally, combat_context)
            role_factor = self._calculate_role_priority_factor(ally, combat_context)
            threat_factor = self._calculate_threat_exposure_factor(ally, combat_context)
            
            # Score final
            priority_score = urgency * distance_factor * role_factor * threat_factor
            
            evaluation = TargetEvaluation(
                entity=ally,
                priority_score=priority_score,
                distance=combat_context.get("player_position", GridPosition(0,0)).distance_to(ally.position),
                threat_level=threat_factor,
                vulnerability=2.0 - hp_ratio,  # Plus les HP sont bas, plus vulnérable
                strategic_value=role_factor,
                reasoning=f"Priorité soins: {priority_score:.2f} (HP:{hp_ratio:.1%}, urgence:{urgency:.1f})"
            )
            
            heal_priorities.append(evaluation)
        
        # Tri par priorité décroissante
        heal_priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return heal_priorities
    
    def _calculate_distance_priority_factor(self, ally: CombatEntity, 
                                          combat_context: Dict[str, Any]) -> float:
        """Facteur de priorité selon la distance"""
        player_pos = combat_context.get("player_position")
        if not player_pos or not ally.position:
            return 1.0
        
        distance = player_pos.distance_to(ally.position)
        
        # Préférence pour les alliés à portée optimale
        if distance <= 6:
            return 1.0
        elif distance <= 10:
            return 0.9
        else:
            return 0.7  # Pénalité distance élevée
    
    def _calculate_role_priority_factor(self, ally: CombatEntity, 
                                      combat_context: Dict[str, Any]) -> float:
        """Facteur de priorité selon le rôle supposé de l'allié"""
        # Analyse basée sur le nom (simplifiée)
        ally_name = ally.name.lower()
        
        # Priorité élevée : autres supports et healers
        if any(keyword in ally_name for keyword in ["eni", "pandawa"]):
            return 1.3
        
        # Priorité moyenne-haute : damage dealers
        elif any(keyword in ally_name for keyword in ["iop", "cra", "sram"]):
            return 1.1
        
        # Priorité moyenne : tanks (ont plus de HP)
        elif any(keyword in ally_name for keyword in ["feca", "sacrieur"]):
            return 0.9
        
        # Par défaut
        return 1.0
    
    def _calculate_threat_exposure_factor(self, ally: CombatEntity, 
                                        combat_context: Dict[str, Any]) -> float:
        """Facteur de priorité selon l'exposition aux menaces"""
        if not ally.position:
            return 1.0
        
        enemies = combat_context.get("enemies", [])
        exposure = 1.0
        
        # Compter les ennemis proches de l'allié
        nearby_enemies = sum(
            1 for enemy in enemies
            if enemy.position and ally.position.distance_to(enemy.position) <= 3
        )
        
        # Plus il y a d'ennemis proches, plus c'est prioritaire de soigner
        exposure += nearby_enemies * 0.2
        
        return min(exposure, 2.0)  # Cap à 2.0
    
    def analyze_group_health_status(self, combat_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse l'état de santé global du groupe
        Retourne des statistiques et recommandations
        """
        allies = combat_context.get("allies", [])
        
        if not allies:
            return {
                "total_allies": 0,
                "health_status": "unknown",
                "recommendations": []
            }
        
        # Statistiques de base
        hp_ratios = [ally.hp_percentage() / 100.0 for ally in allies]
        
        stats = {
            "total_allies": len(allies),
            "average_hp": sum(hp_ratios) / len(hp_ratios),
            "min_hp": min(hp_ratios),
            "max_hp": max(hp_ratios),
            "critical_count": sum(1 for hp in hp_ratios if hp < 0.3),
            "wounded_count": sum(1 for hp in hp_ratios if hp < 0.7),
            "healthy_count": sum(1 for hp in hp_ratios if hp >= 0.8)
        }
        
        # Détermination du statut global
        if stats["critical_count"] >= 2:
            health_status = "critical"
        elif stats["average_hp"] < 0.4:
            health_status = "poor"
        elif stats["wounded_count"] >= len(allies) // 2:
            health_status = "wounded"
        elif stats["average_hp"] > 0.8:
            health_status = "excellent"
        else:
            health_status = "good"
        
        # Recommandations tactiques
        recommendations = []
        
        if health_status == "critical":
            recommendations.extend([
                "Utiliser sorts de soins de groupe (Cercle Curatif, Soin Collectif)",
                "Prioriser Résurrection si alliés morts",
                "Considérer Mot de Sacrifice pour redistribuer HP",
                "Activer mode soins d'urgence"
            ])
        elif health_status == "poor":
            recommendations.extend([
                "Focus sur soins mono-cible puissants",
                "Utiliser buffs défensifs (Immunité, Prévention)",
                "Éviter les sorts offensifs",
                "Maintenir distance de sécurité"
            ])
        elif health_status == "wounded":
            recommendations.extend([
                "Soins régénératifs pour optimiser les PA",
                "Buffs préventifs avant aggravation",
                "Surveiller les cibles prioritaires"
            ])
        else:  # good ou excellent
            recommendations.extend([
                "Opportunité pour buffs offensifs",
                "Sorts de soutien tactique disponibles",
                "Peut participer aux dégâts si nécessaire"
            ])
        
        return {
            **stats,
            "health_status": health_status,
            "recommendations": recommendations,
            "requires_immediate_attention": stats["critical_count"] > 0,
            "group_sustainability": self._calculate_group_sustainability(stats, combat_context)
        }
    
    def _calculate_group_sustainability(self, stats: Dict[str, Any], 
                                      combat_context: Dict[str, Any]) -> float:
        """Calcule la durabilité estimée du groupe (0.0 à 1.0)"""
        base_sustainability = stats["average_hp"]
        
        # Facteur distribution HP
        hp_variance = (stats["max_hp"] - stats["min_hp"])
        distribution_factor = 1.0 - (hp_variance * 0.3)  # Malus si HP très inégaux
        
        # Facteur pression ennemie
        enemies = combat_context.get("enemies", [])
        enemy_pressure = min(len(enemies) * 0.1, 0.5)
        pressure_factor = 1.0 - enemy_pressure
        
        # Facteur ressources Eni
        current_pa = combat_context.get("current_pa", 0)
        resource_factor = min(current_pa / 8.0, 1.0)  # 8 PA = optimal
        
        sustainability = (base_sustainability * distribution_factor * 
                         pressure_factor * resource_factor)
        
        return max(0.0, min(1.0, sustainability))