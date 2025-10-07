"""
Action System - Orchestre l'exécution des actions
"""

import time
import logging
from typing import Dict, Any, Optional
from enum import Enum

from .input_controller import InputController
from .humanizer import ActionHumanizer

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types d'actions possibles"""
    IDLE = "idle"
    CLICK = "click"
    MOVE = "move"
    SPELL = "spell"
    CHAT = "chat"
    INTERACT = "interact"
    SHORTCUT = "shortcut"
    UI_ACTION = "ui_action"


class ActionSystem:
    """Système d'exécution des actions"""
    
    def __init__(self, window_title: str = "Dofus"):
        self.input_ctrl = InputController(window_title=window_title)
        self.humanizer = ActionHumanizer()
        
        # Statistiques
        self.stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'actions_by_type': {}
        }
        
        logger.info("ActionSystem initialisé")
    
    def initialize(self) -> bool:
        """Initialise le système"""
        if not self.input_ctrl.find_window():
            logger.error("Impossible de trouver la fenêtre Dofus")
            return False
        
        logger.info("ActionSystem prêt")
        return True
    
    def execute(self, action_type: str, details: Dict[str, Any]) -> bool:
        """
        Exécute une action
        
        Args:
            action_type: Type d'action (click, spell, move, etc.)
            details: Détails spécifiques à l'action
        
        Returns:
            True si succès, False sinon
        """
        self.stats['total_actions'] += 1
        
        # Tracker par type
        if action_type not in self.stats['actions_by_type']:
            self.stats['actions_by_type'][action_type] = 0
        self.stats['actions_by_type'][action_type] += 1
        
        # S'assurer que la fenêtre est active
        if not self.input_ctrl.is_window_active():
            if not self.input_ctrl.activate_window():
                logger.warning("Fenêtre Dofus non active")
                return False
        
        try:
            success = False
            
            # Router vers la bonne méthode
            if action_type == ActionType.CLICK.value:
                success = self._execute_click(details)
            
            elif action_type == ActionType.MOVE.value:
                success = self._execute_move(details)
            
            elif action_type == ActionType.SPELL.value:
                success = self._execute_spell(details)
            
            elif action_type == ActionType.CHAT.value:
                success = self._execute_chat(details)
            
            elif action_type == ActionType.INTERACT.value:
                success = self._execute_interact(details)
            
            elif action_type == ActionType.SHORTCUT.value:
                success = self._execute_shortcut(details)
            
            elif action_type == ActionType.UI_ACTION.value:
                success = self._execute_ui_action(details)
            
            elif action_type == ActionType.IDLE.value:
                success = True  # Idle = toujours succès
            
            elif action_type == 'flee':
                # Fuite = ESC pour quitter combat
                success = self._execute_shortcut({'keys': 'escape'})
            
            else:
                logger.warning(f"Type d'action inconnu: {action_type}")
                success = False
            
            # Stats
            if success:
                self.stats['successful_actions'] += 1
            else:
                self.stats['failed_actions'] += 1
            
            # Pause humanisée
            if success and action_type != ActionType.IDLE.value:
                pause = self.humanizer.humanize_pause(0.3)
                time.sleep(pause)
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur exécution action {action_type}: {e}")
            self.stats['failed_actions'] += 1
            return False
    
    def _execute_click(self, details: Dict[str, Any]) -> bool:
        """Exécute un clic"""
        x = details.get('x')
        y = details.get('y')
        button = details.get('button', 'left')
        double = details.get('double', False)
        
        if x is None or y is None:
            logger.error("Position manquante pour clic")
            return False
        
        if double:
            return self.input_ctrl.double_click(x, y)
        else:
            return self.input_ctrl.click(x, y, button=button)
    
    def _execute_move(self, details: Dict[str, Any]) -> bool:
        """Exécute un déplacement (clic sur case)"""
        x = details.get('x')
        y = details.get('y')
        
        if x is None or y is None:
            logger.error("Position manquante pour déplacement")
            return False
        
        # Double-clic pour se déplacer dans Dofus
        return self.input_ctrl.double_click(x, y)
    
    def _execute_spell(self, details: Dict[str, Any]) -> bool:
        """Exécute un sort"""
        spell_key = details.get('spell_key', '1')
        target_x = details.get('target_x')
        target_y = details.get('target_y')
        
        # Appuyer sur le raccourci du sort
        if not self.input_ctrl.press_key(spell_key):
            return False
        
        # Pause courte
        time.sleep(0.15)
        
        # Cliquer sur la cible si fournie
        if target_x is not None and target_y is not None:
            return self.input_ctrl.click(target_x, target_y)
        
        return True
    
    def _execute_chat(self, details: Dict[str, Any]) -> bool:
        """Envoie un message chat"""
        message = details.get('message', '')
        
        if not message:
            return False
        
        # Enter pour ouvrir chat
        self.input_ctrl.press_key('enter')
        time.sleep(0.1)
        
        # Taper le message
        self.input_ctrl.type_text(message, interval=0.05)
        time.sleep(0.1)
        
        # Enter pour envoyer
        self.input_ctrl.press_key('enter')
        
        return True
    
    def _execute_interact(self, details: Dict[str, Any]) -> bool:
        """Interagit avec un élément (NPC, ressource, etc.)"""
        x = details.get('x')
        y = details.get('y')
        
        if x is None or y is None:
            return False
        
        # Clic pour sélectionner
        if not self.input_ctrl.click(x, y):
            return False
        
        time.sleep(0.2)
        
        # Raccourci interaction (souvent 'space' ou 'enter')
        interaction_key = details.get('key', 'space')
        return self.input_ctrl.press_key(interaction_key)
    
    def _execute_shortcut(self, details: Dict[str, Any]) -> bool:
        """Exécute un raccourci clavier"""
        keys = details.get('keys', [])
        
        if not keys:
            return False
        
        if isinstance(keys, str):
            return self.input_ctrl.press_key(keys)
        elif isinstance(keys, list):
            return self.input_ctrl.press_keys(*keys)
        
        return False
    
    def _execute_ui_action(self, details: Dict[str, Any]) -> bool:
        """Action UI (ouvrir inventaire, sorts, etc.)"""
        ui_element = details.get('element')
        
        # Mapping raccourcis UI Dofus
        ui_shortcuts = {
            'inventory': 'i',
            'spells': 's',
            'quests': 'q',
            'map': 'm',
            'friends': 'f',
            'guild': 'g',
            'alignment': 'a',
            'bestiary': 'b',
            'achievements': 'u'
        }
        
        if ui_element in ui_shortcuts:
            key = ui_shortcuts[ui_element]
            return self.input_ctrl.press_key(key)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return {
            **self.stats,
            'input_controller_stats': self.input_ctrl.get_stats()
        }


def create_action_system(window_title: str = "Dofus") -> ActionSystem:
    """Factory function"""
    return ActionSystem(window_title=window_title)
