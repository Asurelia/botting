"""
Module de détection des archimonstres DOFUS.
Détection via analyse d'écran et monitoring du chat système.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ArchmonsterDetection:
    """Représente une détection d'archimonstre."""
    name: str
    zone: str
    position: Tuple[int, int]
    confidence: float
    detection_time: datetime = field(default_factory=datetime.now)
    screenshot_path: Optional[str] = None
    detection_method: str = "unknown"  # "chat", "visual", "system"


class ChatPatternMatcher:
    """Matcher de patterns pour messages du chat système."""
    
    def __init__(self):
        self.archmonster_patterns = {
            "archimonstre": [
                r"Un Archimonstre (.+?) vient d'apparaître en (.+?)!",
                r"L'Archimonstre (.+?) est apparu en (.+?)!",
                r"(.+?) \(Archimonstre\) vient d'apparaître en (.+?)!",
            ]
        }
        
        self.compiled_patterns = {}
        for category, patterns in self.archmonster_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_archmonster_info(self, message: str) -> Optional[Tuple[str, str]]:
        """Extrait nom archimonstre et zone depuis message chat."""
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(message)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        return (groups[0].strip(), groups[1].strip())
        return None


class VisualDetector:
    """Détection visuelle d'archimonstres via analyse d'écran."""
    
    def __init__(self):
        self.archmonster_templates = {}
        self.detection_threshold = 0.8
    
    def detect_archmonster_on_screen(self, screenshot: np.ndarray) -> List[ArchmonsterDetection]:
        """Détecte archimonstres sur capture d'écran."""
        # Simulation pour demo
        return []


class ArchmonsterDetector:
    """Détecteur principal d'archimonstres."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chat_matcher = ChatPatternMatcher()
        self.visual_detector = VisualDetector()
        
        self.is_running = False
        self.detection_thread = None
        self.detection_callbacks: List[Callable] = []
        
        self.scan_interval = config.get('scan_interval', 2.0)
        self.watched_archmonsters = set(config.get('watched_archmonsters', []))
        self.watched_zones = set(config.get('watched_zones', []))
        
        self.logger = logging.getLogger(__name__)
    
    def add_detection_callback(self, callback: Callable[[ArchmonsterDetection], None]):
        """Ajoute callback appelé lors des détections."""
        self.detection_callbacks.append(callback)
    
    def start_detection(self):
        """Démarre la détection en arrière-plan."""
        if self.is_running:
            return
        
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.logger.info("Détection d'archimonstres démarrée")
    
    def stop_detection(self):
        """Arrête la détection."""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        self.logger.info("Détection d'archimonstres arrêtée")
    
    def _detection_loop(self):
        """Boucle principale de détection."""
        while self.is_running:
            try:
                # Simulation détection
                time.sleep(self.scan_interval)
            except Exception as e:
                self.logger.error(f"Erreur dans boucle détection: {e}")
                time.sleep(5)
    
    def get_detection_stats(self) -> Dict:
        """Récupère statistiques de détection."""
        return {
            'is_running': self.is_running,
            'watched_archmonsters': len(self.watched_archmonsters),
            'watched_zones': len(self.watched_zones)
        }