"""
Vision Capture Adapter - Capture écran cross-platform
Support Windows + Linux avec détection automatique

Author: Claude Code  
Date: 2025-10-06
"""

import cv2
import numpy as np
import mss
from typing import Optional, Tuple
from dataclasses import dataclass
import logging
import time

from core.platform_adapter import PlatformAdapter, WindowInfo

logger = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    """Statistiques de capture"""
    fps: float
    avg_latency_ms: float
    frames_captured: int
    errors: int


class VisionCaptureAdapter:
    """
    Adaptateur capture écran cross-platform
    
    Features:
    - Auto-détection OS (Windows/Linux)
    - Capture mss (rapide, cross-platform)
    - Détection fenêtre Dofus automatique
    - Stats performance temps réel
    
    Usage:
        adapter = VisionCaptureAdapter()
        adapter.find_game_window("Dofus")
        img = adapter.capture()
    """

    def __init__(self):
        self.platform_adapter = PlatformAdapter()
        self.game_window: Optional[WindowInfo] = None
        self.sct = mss.mss()
        
        # Stats
        self.frames_captured = 0
        self.capture_times = []
        self.errors = 0
        
        # Cache
        self.last_capture_time = 0
        self.cache_duration = 0.016  # 16ms (~60 FPS max)
        self.cached_frame = None
        
        logger.info(f"VisionCaptureAdapter initialized - OS: {self.platform_adapter.get_system()}")

    def find_game_window(self, title_pattern: str = "Dofus") -> bool:
        """
        Trouve fenêtre du jeu
        
        Args:
            title_pattern: Pattern titre fenêtre (défaut "Dofus")
            
        Returns:
            True si fenêtre trouvée
        """
        self.game_window = self.platform_adapter.find_window(title_pattern)
        
        if self.game_window:
            logger.info(f"Game window found: {self.game_window.title}")
            logger.info(f"  Position: ({self.game_window.x}, {self.game_window.y})")
            logger.info(f"  Size: {self.game_window.width}x{self.game_window.height}")
            return True
        else:
            logger.warning(f"Game window not found: {title_pattern}")
            return False

    def capture(self, use_cache: bool = False) -> Optional[np.ndarray]:
        """
        Capture screenshot
        
        Args:
            use_cache: Utiliser cache si capture récente (<16ms)
            
        Returns:
            Image numpy array (BGR) ou None si erreur
        """
        # Check cache
        if use_cache and self.cached_frame is not None:
            if (time.time() - self.last_capture_time) < self.cache_duration:
                return self.cached_frame
        
        try:
            start = time.time()
            
            if self.game_window:
                # Capture fenêtre spécifique
                region = {
                    "top": self.game_window.y,
                    "left": self.game_window.x,
                    "width": self.game_window.width,
                    "height": self.game_window.height
                }
            else:
                # Capture écran complet
                region = self.sct.monitors[1]  # Primary monitor
            
            # Capture avec mss
            screenshot = self.sct.grab(region)
            
            # Convert to numpy array (BGR for OpenCV)
            img = np.array(screenshot)
            
            # mss donne BGRA, convertir en BGR
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Stats
            elapsed = time.time() - start
            self.capture_times.append(elapsed)
            if len(self.capture_times) > 100:
                self.capture_times.pop(0)
            
            self.frames_captured += 1
            
            # Cache
            self.cached_frame = img
            self.last_capture_time = time.time()
            
            return img
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self.errors += 1
            return None

    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        Capture région spécifique de l'écran
        
        Args:
            x, y: Position coin supérieur gauche
            width, height: Dimensions
            
        Returns:
            Image numpy array ou None
        """
        try:
            region = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            
            screenshot = self.sct.grab(region)
            img = np.array(screenshot)
            
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
        except Exception as e:
            logger.error(f"Region capture error: {e}")
            return None

    def get_stats(self) -> CaptureStats:
        """Retourne statistiques de capture"""
        if not self.capture_times:
            return CaptureStats(0, 0, self.frames_captured, self.errors)
        
        avg_time = sum(self.capture_times) / len(self.capture_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return CaptureStats(
            fps=fps,
            avg_latency_ms=avg_time * 1000,
            frames_captured=self.frames_captured,
            errors=self.errors
        )

    def reset_stats(self):
        """Reset statistiques"""
        self.frames_captured = 0
        self.capture_times.clear()
        self.errors = 0

    def __del__(self):
        """Cleanup"""
        try:
            self.sct.close()
        except:
            pass


# ===== Tests rapides =====

def test_capture_adapter():
    """Test rapide de l'adaptateur"""
    print("🧪 Test VisionCaptureAdapter")
    
    adapter = VisionCaptureAdapter()
    
    # Test capture écran complet
    print("\n📸 Capture écran complet...")
    img = adapter.capture()
    
    if img is not None:
        print(f"   ✅ Capture OK: {img.shape}")
        print(f"   Type: {img.dtype}")
    else:
        print("   ❌ Capture échouée")
        return
    
    # Benchmark
    print("\n⚡ Benchmark (100 frames)...")
    for _ in range(100):
        _ = adapter.capture()
    
    stats = adapter.get_stats()
    print(f"   FPS: {stats.fps:.1f}")
    print(f"   Latency: {stats.avg_latency_ms:.2f}ms")
    print(f"   Frames: {stats.frames_captured}")
    print(f"   Errors: {stats.errors}")
    
    # Test recherche fenêtre
    print("\n🔍 Recherche fenêtre Dofus...")
    if adapter.find_game_window("Dofus"):
        print(f"   ✅ Trouvé: {adapter.game_window.title}")
    else:
        print("   ⚠️  Dofus non trouvé (normal si non lancé)")
    
    print("\n✅ Tests terminés")


if __name__ == "__main__":
    test_capture_adapter()
