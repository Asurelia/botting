"""
AI Vision Module - Module de vision intégré au framework IA existant
Adaptation des systèmes de vision pour l'architecture AI Framework
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Import du framework IA existant
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.ai_framework import AIModule, AIModuleState

# Import des modules de vision développés
from .game_capture import GameCapture
from .ocr_engine import OCREngine

class VisionModule(AIModule):
    """Module de vision intégré au framework IA"""

    def __init__(self):
        super().__init__("Vision")
        self.game_capture: Optional[GameCapture] = None
        self.ocr_engine: Optional[OCREngine] = None

        # Configuration
        self.capture_fps = 30
        self.analysis_interval = 1.0

        # État
        self.last_screenshot = None
        self.last_ocr_results = []
        self.last_analysis_time = 0.0

        # Métriques
        self.screenshots_count = 0
        self.ocr_analysis_count = 0
        self.avg_capture_time = 0.0
        self.avg_ocr_time = 0.0

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Implémentation de l'initialisation du module vision"""
        try:
            self.logger.info("Initialisation du module Vision...")

            # Configuration depuis le config
            vision_config = config.get('vision', {})
            self.capture_fps = vision_config.get('capture_fps', 30)
            self.analysis_interval = vision_config.get('analysis_interval', 1.0)

            # Initialiser GameCapture
            self.logger.info("Initialisation GameCapture...")
            self.game_capture = GameCapture()
            if not self.game_capture.initialize():
                self.logger.error("Échec initialisation GameCapture")
                return False

            # Configurer qualité capture
            quality = vision_config.get('quality', 'high')
            self.game_capture.set_capture_quality(quality)

            # Initialiser OCR Engine
            self.logger.info("Initialisation OCR Engine...")
            self.ocr_engine = OCREngine()
            if not self.ocr_engine.initialize():
                self.logger.error("Échec initialisation OCR Engine")
                return False

            # Démarrer capture continue
            if not self.game_capture.start_continuous_capture(self.capture_fps):
                self.logger.error("Échec démarrage capture continue")
                return False

            self.logger.info("Module Vision initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation module Vision: {e}")
            return False

    async def _run_impl(self):
        """Boucle principale du module vision"""
        self.logger.info("Démarrage boucle Vision...")

        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()

                # Capturer screenshot
                screenshot = await self._capture_screenshot()
                if screenshot is not None:
                    self.last_screenshot = screenshot
                    self.screenshots_count += 1

                    # Analyser périodiquement avec OCR
                    if start_time - self.last_analysis_time >= self.analysis_interval:
                        await self._analyze_screenshot(screenshot)
                        self.last_analysis_time = start_time

                # Maintenir FPS
                elapsed = time.time() - start_time
                target_interval = 1.0 / self.capture_fps
                sleep_time = max(0, target_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self.logger.info("Module Vision arrêté")
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle Vision: {e}")
                await asyncio.sleep(1.0)

    async def _capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot de manière asynchrone"""
        try:
            if not self.game_capture:
                return None

            start_time = time.time()

            # Capturer dans un thread pour éviter blocage
            loop = asyncio.get_event_loop()
            screenshot = await loop.run_in_executor(
                None, self.game_capture.capture_screenshot_fast
            )

            # Mettre à jour métriques
            capture_time = time.time() - start_time
            if self.screenshots_count > 0:
                self.avg_capture_time = (self.avg_capture_time * (self.screenshots_count - 1) + capture_time) / self.screenshots_count
            else:
                self.avg_capture_time = capture_time

            return screenshot

        except Exception as e:
            self.logger.error(f"Erreur capture screenshot: {e}")
            return None

    async def _analyze_screenshot(self, screenshot: np.ndarray):
        """Analyse screenshot avec OCR"""
        try:
            if not self.ocr_engine:
                return

            start_time = time.time()

            # OCR dans un thread
            loop = asyncio.get_event_loop()
            ocr_results = await loop.run_in_executor(
                None, self.ocr_engine.extract_text_multi_engine, screenshot
            )

            self.last_ocr_results = ocr_results
            self.ocr_analysis_count += 1

            # Mettre à jour métriques
            ocr_time = time.time() - start_time
            if self.ocr_analysis_count > 0:
                self.avg_ocr_time = (self.avg_ocr_time * (self.ocr_analysis_count - 1) + ocr_time) / self.ocr_analysis_count
            else:
                self.avg_ocr_time = ocr_time

            # Mettre dans shared_data pour autres modules
            self._shared_data['last_screenshot'] = screenshot
            self._shared_data['last_ocr_results'] = ocr_results
            self._shared_data['vision_timestamp'] = time.time()

        except Exception as e:
            self.logger.error(f"Erreur analyse OCR: {e}")

    async def _shutdown_impl(self):
        """Arrêt propre du module"""
        try:
            self.logger.info("Arrêt module Vision...")

            if self.game_capture:
                self.game_capture.stop_continuous_capture()
                self.game_capture.cleanup()

            if self.ocr_engine:
                self.ocr_engine.cleanup()

            self.logger.info("Module Vision arrêté proprement")

        except Exception as e:
            self.logger.error(f"Erreur arrêt module Vision: {e}")

    def get_latest_screenshot(self) -> Optional[np.ndarray]:
        """Retourne le dernier screenshot capturé"""
        return self.last_screenshot

    def get_latest_ocr_results(self) -> List:
        """Retourne les derniers résultats OCR"""
        return self.last_ocr_results

    def get_game_elements(self) -> Dict[str, Any]:
        """Extrait les éléments de jeu détectés"""
        elements = {
            'timestamp': time.time(),
            'text_elements': [],
            'ui_elements': {},
            'game_state': 'unknown'
        }

        # Analyser résultats OCR
        for ocr_result in self.last_ocr_results:
            elements['text_elements'].append({
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'position': ocr_result.bbox,
                'region_type': ocr_result.region_type
            })

        # Détecter éléments UI spécifiques
        elements['ui_elements'] = self._detect_ui_elements()

        # Inférer état du jeu
        elements['game_state'] = self._infer_game_state()

        return elements

    def _detect_ui_elements(self) -> Dict[str, Any]:
        """Détecte les éléments d'interface utilisateur"""
        ui_elements = {
            'health_visible': False,
            'mana_visible': False,
            'combat_mode': False,
            'menu_open': False,
            'dialog_open': False
        }

        # Analyser texte OCR pour détecter UI
        for ocr_result in self.last_ocr_results:
            text = ocr_result.text.lower()

            if any(word in text for word in ['hp', 'vie', 'health']):
                ui_elements['health_visible'] = True
            elif any(word in text for word in ['mp', 'mana', 'pa']):
                ui_elements['mana_visible'] = True
            elif any(word in text for word in ['combat', 'attaque', 'sort']):
                ui_elements['combat_mode'] = True
            elif any(word in text for word in ['menu', 'option', 'config']):
                ui_elements['menu_open'] = True
            elif any(word in text for word in ['dialog', 'parler', 'discussion']):
                ui_elements['dialog_open'] = True

        return ui_elements

    def _infer_game_state(self) -> str:
        """Infère l'état actuel du jeu"""
        ui_elements = self._detect_ui_elements()

        if ui_elements['combat_mode']:
            return 'combat'
        elif ui_elements['dialog_open']:
            return 'dialog'
        elif ui_elements['menu_open']:
            return 'menu'
        else:
            return 'exploration'

    def get_module_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du module"""
        stats = {
            'screenshots_captured': self.screenshots_count,
            'ocr_analyses': self.ocr_analysis_count,
            'avg_capture_time_ms': round(self.avg_capture_time * 1000, 2),
            'avg_ocr_time_ms': round(self.avg_ocr_time * 1000, 2),
            'current_fps': round(1.0 / self.avg_capture_time, 1) if self.avg_capture_time > 0 else 0,
            'has_screenshot': self.last_screenshot is not None,
            'ocr_results_count': len(self.last_ocr_results),
            'game_state': self._infer_game_state()
        }

        # Ajouter stats de GameCapture si disponible
        if self.game_capture:
            capture_stats = self.game_capture.get_performance_stats()
            stats.update({
                'capture_fps': capture_stats.get('actual_fps', 0),
                'window_detected': capture_stats.get('window_detected', False),
                'capture_quality': capture_stats.get('capture_quality', 'unknown')
            })

        return stats

# Factory function pour intégration avec le framework
def create_vision_module() -> VisionModule:
    """Crée une instance du module Vision"""
    return VisionModule()