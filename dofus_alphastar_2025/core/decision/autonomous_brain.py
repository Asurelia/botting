"""
Autonomous Brain - Cerveau décisionnel autonome
Prend des décisions basées sur le game state
"""

import time
import logging
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """Objectif du bot"""
    name: str
    priority: int  # 1 = max priorité, 10 = min
    condition: str  # Condition pour activer l'objectif
    tasks: List[str]  # Tâches à accomplir


class AutonomousBrain:
    """
    Cerveau autonome du bot
    Architecture hiérarchique:
    1. Survie (priorité max)
    2. Combat
    3. Objectifs (farming, questing, etc.)
    4. Idle/Exploration
    """
    
    def __init__(self):
        # Objectifs du bot (sera configurable)
        self.current_objective = "idle"
        self.objectives = self._init_objectives()
        
        # Historique décisions
        self.decision_history = []
        
        # Cooldowns
        self.last_action_time = 0
        self.action_cooldown = 0.5  # secondes entre actions
        
        # Stats
        self.stats = {
            'decisions_made': 0,
            'combat_decisions': 0,
            'navigation_decisions': 0,
            'idle_decisions': 0
        }
        
        logger.info("AutonomousBrain initialisé")
    
    def _init_objectives(self) -> List[Objective]:
        """Initialise les objectifs disponibles"""
        return [
            Objective(
                name="survival",
                priority=1,
                condition="hp_low",
                tasks=["heal", "flee"]
            ),
            Objective(
                name="combat",
                priority=2,
                condition="in_combat",
                tasks=["attack", "move_tactical", "defend"]
            ),
            Objective(
                name="farming",
                priority=5,
                condition="farming_mode",
                tasks=["find_resource", "harvest", "move_to_resource"]
            ),
            Objective(
                name="idle",
                priority=10,
                condition="no_objective",
                tasks=["wait", "explore"]
            )
        ]
    
    def decide(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """
        Prend une décision basée sur l'état du jeu
        
        Args:
            game_state: GameState object
        
        Returns:
            Decision dict ou None si pas d'action
        """
        # Cooldown entre actions
        if time.time() - self.last_action_time < self.action_cooldown:
            return None
        
        self.stats['decisions_made'] += 1
        
        decision = None
        
        # === PRIORITÉ 1: SURVIE ===
        if game_state.character.hp_percent < 30:
            decision = self._decide_survival(game_state)
            if decision:
                self.last_action_time = time.time()
                return decision
        
        # === PRIORITÉ 2: COMBAT ===
        if game_state.combat.in_combat:
            decision = self._decide_combat(game_state)
            self.stats['combat_decisions'] += 1
        
        # === PRIORITÉ 3: OBJECTIF ACTUEL ===
        elif game_state.current_objective == "farming":
            decision = self._decide_farming(game_state)
        
        elif game_state.current_objective == "questing":
            decision = self._decide_questing(game_state)
        
        # === PRIORITÉ 4: IDLE/EXPLORATION ===
        else:
            decision = self._decide_idle(game_state)
            self.stats['idle_decisions'] += 1
        
        # Enregistrer décision
        if decision:
            self.decision_history.append({
                'time': time.time(),
                'state': game_state.bot_state.value,
                'decision': decision
            })
            
            # Limiter historique
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            self.last_action_time = time.time()
        
        return decision
    
    def _decide_survival(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions de survie (HP bas)"""
        logger.warning(f"HP BAS: {game_state.character.hp_percent}%")
        
        # Si en combat, essayer de fuir
        if game_state.combat.in_combat:
            return {
                'action_type': 'flee',
                'details': {},
                'reason': 'hp_critical'
            }
        
        # Sinon, utiliser pain/potion (TODO)
        return {
            'action_type': 'heal',
            'details': {},
            'reason': 'hp_low'
        }
    
    def _decide_combat(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions de combat"""
        # Pas notre tour = attendre
        if not game_state.combat.my_turn:
            return None
        
        # Pas d'ennemis = fin combat
        enemies = game_state.combat.get_alive_enemies()
        if not enemies:
            return None
        
        # Stratégie simple: attaquer l'ennemi le plus faible
        target = game_state.combat.get_weakest_enemy()
        
        if target:
            # Vérifier si on a les PA pour attaquer
            if game_state.character.pa >= 3:
                return {
                    'action_type': 'spell',
                    'details': {
                        'spell_key': '1',  # Sort slot 1
                        'target_x': target.position[0],
                        'target_y': target.position[1]
                    },
                    'reason': 'attack_weakest_enemy'
                }
        
        # Pas de cible ou pas assez de PA = passer tour
        return {
            'action_type': 'shortcut',
            'details': {'keys': 'space'},  # Passer tour
            'reason': 'no_action_available'
        }
    
    def _decide_farming(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions de farming"""
        # Chercher ressources visibles
        resources = game_state.environment.get_harvestable_resources()
        
        if resources:
            # Aller vers la ressource la plus proche
            nearest = game_state.environment.get_nearest_resource()
            
            if nearest:
                return {
                    'action_type': 'interact',
                    'details': {
                        'x': nearest.position[0],
                        'y': nearest.position[1]
                    },
                    'reason': 'harvest_resource'
                }
        
        # Pas de ressources = se déplacer aléatoirement
        return self._decide_exploration(game_state)
    
    def _decide_questing(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions de quête"""
        # TODO: Intégrer avec Ganymede
        # Pour l'instant, idle
        return None
    
    def _decide_idle(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions en idle"""
        # Aléatoirement: explorer ou rester immobile
        if random.random() < 0.3:  # 30% chance d'explorer
            return self._decide_exploration(game_state)
        
        # Sinon, rester idle (click aléatoire pour exploration)
        return self._decide_exploration(game_state)
    
    def _decide_exploration(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Décisions d'exploration"""
        # Mouvement aléatoire sur la map
        # TODO: Pathfinding intelligent
        
        # Coordonnées aléatoires (relative à la fenêtre)
        rand_x = random.randint(200, 800)
        rand_y = random.randint(200, 500)
        
        return {
            'action_type': 'move',
            'details': {
                'x': rand_x,
                'y': rand_y
            },
            'reason': 'exploration'
        }
    
    def set_objective(self, objective: str):
        """Change l'objectif actuel"""
        self.current_objective = objective
        logger.info(f"Objectif changé: {objective}")
    
    def set_spells(self, spells: list):
        """Configure les sorts disponibles"""
        self.spells = spells
        logger.info(f"{len(spells)} sorts configurés")
    
    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """Retourne historique décisions"""
        return self.decision_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques"""
        return self.stats.copy()
    
    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Explique une décision (conscience)"""
        action_type = decision.get('action_type', 'unknown')
        reason = decision.get('reason', 'no_reason')
        
        explanations = {
            'attack_weakest_enemy': "J'attaque l'ennemi le plus faible pour optimiser les dégâts",
            'hp_critical': "Mes HP sont critiques, je dois fuir le combat",
            'hp_low': "Mes HP sont bas, je dois me soigner",
            'harvest_resource': "J'ai détecté une ressource à récolter",
            'exploration': "Je me déplace pour explorer la map",
            'no_action_available': "Aucune action possible, je passe mon tour"
        }
        
        return f"Action: {action_type} - Raison: {explanations.get(reason, reason)}"


def create_autonomous_brain() -> AutonomousBrain:
    """Factory function"""
    return AutonomousBrain()
