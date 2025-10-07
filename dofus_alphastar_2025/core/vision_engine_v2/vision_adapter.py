"""
Adaptateur pour connecter l'ancien système de vision au nouveau GameEngine
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
import pyautogui

logger = logging.getLogger(__name__)


class VisionSystemAdapter:
    """
    Adapte l'ancien VisionOrchestrator au format attendu par GameEngine
    """
    
    def __init__(self, window_title: str = "Dofus"):
        self.window_title = window_title
        self.last_screenshot = None
        self.stats = {
            'captures': 0,
            'detections': 0,
            'cache_hits': 0,
            'ocr_enabled': False
        }
        
        # Essayer d'importer VisionOrchestrator
        try:
            import sys
            from pathlib import Path
            # Remove engine dependencies
            sys.modules['engine'] = type(sys)('engine')
            sys.modules['engine.module_interface'] = type(sys)('engine.module_interface')
            sys.modules['engine.event_bus'] = type(sys)('engine.event_bus')
            
            # Mock classes
            class MockModule:
                def __init__(self, name):
                    self.name = name
                    self.status = "active"
                def set_error(self, msg):
                    pass
                def is_active(self):
                    return True
            
            class ModuleStatus:
                ACTIVE = "active"
                INACTIVE = "inactive"
            
            sys.modules['engine.module_interface'].IAnalysisModule = MockModule
            sys.modules['engine.module_interface'].ModuleStatus = ModuleStatus
            sys.modules['engine.event_bus'].EventType = type('EventType', (), {})
            sys.modules['engine.event_bus'].EventPriority = type('EventPriority', (), {})
            
            from ..vision.vision_orchestrator import VisionOrchestrator, VisionConfig
            from ..vision.template_matcher import TemplateMatcher
            
            self.vision_orchestrator = VisionOrchestrator(config=VisionConfig())
            self.vision_available = True
            logger.info("✅ VisionOrchestrator chargé avec succès")
        except Exception as e:
            logger.warning(f"⚠️  VisionOrchestrator non disponible: {e}")
            self.vision_orchestrator = None
            self.vision_available = False
        
        logger.info(f"VisionSystemAdapter initialisé (vision: {self.vision_available})")
    
    def capture_screen(self) -> np.ndarray:
        """Capture l'écran"""
        try:
            screenshot = pyautogui.screenshot()
            img = np.array(screenshot)
            img = img[:, :, ::-1]  # RGB -> BGR
            self.last_screenshot = img
            self.stats['captures'] += 1
            return img
        except Exception as e:
            logger.error(f"Erreur capture: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def extract_game_state(self) -> Dict[str, Any]:
        """
        Extrait l'état du jeu (compatible avec GameEngine)
        
        Returns:
            Dict avec: character, combat, environment, ui
        """
        img = self.capture_screen()
        
        # Détection avec VisionOrchestrator si disponible
        mobs = []
        npcs = []
        resources = []
        
        if self.vision_available and self.vision_orchestrator:
            try:
                # Analyser l'image
                result = self.vision_orchestrator.analyze(img, target_categories=['mob', 'npc', 'resource'])
                
                # Extraire détections
                detections_by_class = result.get('detections_by_class', {})
                
                # Mobs
                if 'mob' in detections_by_class or 'monster' in detections_by_class:
                    mob_detections = detections_by_class.get('mob', []) + detections_by_class.get('monster', [])
                    for det in mob_detections:
                        mobs.append({
                            'position': det['position'],
                            'bbox': det['bounding_box'],
                            'confidence': det['confidence']
                        })
                
                # NPCs
                if 'npc' in detections_by_class:
                    for det in detections_by_class['npc']:
                        npcs.append({
                            'position': det['position'],
                            'bbox': det['bounding_box'],
                            'confidence': det['confidence']
                        })
                
                # Resources
                if 'resource' in detections_by_class:
                    for det in detections_by_class['resource']:
                        resources.append({
                            'position': det['position'],
                            'bbox': det['bounding_box'],
                            'confidence': det['confidence']
                        })
                
                self.stats['detections'] += 1
                
            except Exception as e:
                logger.error(f"Erreur détection: {e}")
        
        # Retourner état au format GameEngine
        return {
            'character': {
                'hp': 100,
                'max_hp': 100,
                'hp_percent': 100.0,
                'pa': 6,
                'max_pa': 6,
                'pm': 3,
                'max_pm': 3
            },
            'combat': {
                'in_combat': False,  # Désactivé pour éviter faux positifs
                'my_turn': False,
                'enemies': [],
                'allies': []
            },
            'environment': {
                'mobs': mobs,
                'npcs': npcs,
                'resources': resources,
                'danger_level': 0.0
            },
            'ui': {
                'window_active': True
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques"""
        return self.stats.copy()


def create_vision_adapter(window_title: str = "Dofus") -> VisionSystemAdapter:
    """Factory function"""
    return VisionSystemAdapter(window_title)
