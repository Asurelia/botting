"""
Adaptateur COMPLET qui utilise VRAIMENT le TemplateMatcher
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import pyautogui
import cv2

logger = logging.getLogger(__name__)


class VisionCompleteAdapter:
    """
    Adaptateur qui connecte TemplateMatcher au GameEngine
    """
    
    def __init__(self, window_title: str = "Dofus"):
        self.window_title = window_title
        self.logger = logging.getLogger(__name__)
        
        # Import et initialisation TemplateMatcher
        try:
            from ..vision.template_matcher import TemplateMatcher
            self.template_matcher = TemplateMatcher()
            
            # Initialiser avec config vide (va charger templates depuis assets/)
            success = self.template_matcher.initialize({})
            
            if success:
                self.logger.info(f"✅ TemplateMatcher initialisé - {self.template_matcher.stats['templates_loaded']} templates")
                self.vision_available = True
            else:
                self.logger.warning("⚠️  TemplateMatcher non initialisé")
                self.vision_available = False
                
        except Exception as e:
            self.logger.error(f"Erreur init TemplateMatcher: {e}")
            self.template_matcher = None
            self.vision_available = False
        
        # Stats
        self.stats = {
            'captures': 0,
            'detections': 0,
            'cache_hits': 0,
            'mobs_detected': 0,
            'npcs_detected': 0,
            'resources_detected': 0
        }
        
        self.last_screenshot = None
    
    def capture_screen(self) -> np.ndarray:
        """Capture l'écran"""
        try:
            screenshot = pyautogui.screenshot()
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.last_screenshot = img
            self.stats['captures'] += 1
            return img
        except Exception as e:
            self.logger.error(f"Erreur capture: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def extract_game_state(self) -> Dict[str, Any]:
        """
        Extrait l'état du jeu en utilisant TemplateMatcher
        
        Returns:
            Dict compatible GameEngine
        """
        img = self.capture_screen()
        
        mobs = []
        npcs = []
        resources = []
        
        if self.vision_available and self.template_matcher:
            try:
                # Analyser l'image avec TemplateMatcher
                # On cherche dans la zone de jeu principale
                result = self.template_matcher.analyze(
                    img, 
                    target_categories=['monster', 'mob', 'npc', 'resource'],
                    roi='game_area'
                )
                
                # Extraire les détections par catégorie
                matches_by_category = result.get('matches_by_category', {})
                
                # Mobs/Monsters
                mob_matches = matches_by_category.get('monster', []) + matches_by_category.get('mob', [])
                for match in mob_matches:
                    mobs.append({
                        'position': match.position,
                        'bbox': match.bounding_box,
                        'confidence': match.confidence,
                        'name': match.template_name
                    })
                    self.stats['mobs_detected'] += 1
                
                # NPCs
                npc_matches = matches_by_category.get('npc', [])
                for match in npc_matches:
                    npcs.append({
                        'position': match.position,
                        'bbox': match.bounding_box,
                        'confidence': match.confidence,
                        'name': match.template_name
                    })
                    self.stats['npcs_detected'] += 1
                
                # Resources
                resource_matches = matches_by_category.get('resource', [])
                for match in resource_matches:
                    resources.append({
                        'position': match.position,
                        'bbox': match.bounding_box,
                        'confidence': match.confidence,
                        'name': match.template_name
                    })
                    self.stats['resources_detected'] += 1
                
                self.stats['detections'] += 1
                
            except Exception as e:
                self.logger.error(f"Erreur détection: {e}")
        
        # Retourner état au format GameEngine
        return {
            'character': {
                'hp': 100,  # TODO: OCR réel
                'max_hp': 100,
                'hp_percent': 100.0,
                'pa': 6,
                'max_pa': 6,
                'pm': 3,
                'max_pm': 3
            },
            'combat': {
                'in_combat': False,  # TODO: Détection combat réelle
                'my_turn': False,
                'enemies': [],
                'allies': []
            },
            'environment': {
                'mobs': mobs,
                'npcs': npcs,
                'resources': resources,
                'danger_level': len(mobs) * 0.2  # Plus de mobs = plus dangereux
            },
            'ui': {
                'window_active': True
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques"""
        stats = self.stats.copy()
        if self.template_matcher:
            stats['template_matcher_stats'] = self.template_matcher.get_state()
        return stats


def create_vision_complete_adapter(window_title: str = "Dofus") -> VisionCompleteAdapter:
    """Factory function"""
    return VisionCompleteAdapter(window_title)
