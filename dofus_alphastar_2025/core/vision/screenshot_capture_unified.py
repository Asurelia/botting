"""
Screenshot Capture Unifié DOFUS Unity - Cross-Platform
Capture d'écran haute performance optimisée pour DOFUS Unity
Support Linux + Windows via Platform Adapter
Approche 100% vision - Détection automatique fenêtre DOFUS
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time
import logging
import sys
import os
from dataclasses import dataclass

# Add dofus_alphastar_2025 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.platform_adapter import PlatformAdapter, WindowInfo as PlatformWindowInfo
from core.vision_capture_adapter import VisionCaptureAdapter

logger = logging.getLogger(__name__)

@dataclass
class WindowInfo:
    """
    Informations sur la fenêtre DOFUS (format legacy pour compatibilité)
    Compatible avec l'ancienne interface Windows-only
    """
    hwnd: int  # Sur Linux, c'est l'ID window en int
    title: str
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    width: int
    height: int
    is_visible: bool

    @classmethod
    def from_platform_window(cls, pw: PlatformWindowInfo) -> 'WindowInfo':
        """Convertit PlatformWindowInfo vers WindowInfo legacy"""
        hwnd = int(pw.id) if isinstance(pw.id, str) else pw.id
        return cls(
            hwnd=hwnd,
            title=pw.title,
            rect=(pw.x, pw.y, pw.x + pw.width, pw.y + pw.height),
            width=pw.width,
            height=pw.height,
            is_visible=True
        )


class DofusWindowCapture:
    """
    Capture de fenêtre DOFUS Unity optimisée - Cross-Platform

    Architecture:
    - Module A (Perception): Capture visuelle haute performance
    - Support Linux (xdotool/mss) + Windows (win32gui/mss)
    - Compatible avec ancienne interface pour rétrocompatibilité

    Performance Targets:
    - >60 FPS capture écran complet
    - >100 FPS capture région fenêtre
    - <10ms latency moyenne
    """

    def __init__(self):
        """Initialise le système de capture cross-platform"""

        # Adaptateurs platform
        self.platform_adapter = PlatformAdapter()
        self.vision_adapter = VisionCaptureAdapter()

        # State
        self.dofus_window: Optional[WindowInfo] = None
        self.capture_method = "auto"  # auto, mss (legacy: win32, pil)
        self.last_capture_time = 0
        self.capture_cache = None
        self.cache_duration = 0.05  # 50ms cache

        # Patterns de titre pour DOFUS Unity (tous OS)
        self.dofus_patterns = [
            "DOFUS",
            "Dofus Unity",
            "DOFUS Unity",
            "dofus.exe",
            "Ankama",
            "dofus",  # Linux (minuscule)
        ]

        # Détection OS
        self.is_linux = self.platform_adapter.is_linux()
        self.is_windows = self.platform_adapter.is_windows()

        logger.info(f"DofusWindowCapture initialisé - OS: {'Linux' if self.is_linux else 'Windows'}")

    def find_dofus_window(self) -> Optional[WindowInfo]:
        """
        Trouve la fenêtre DOFUS Unity active (cross-platform)

        Returns:
            WindowInfo si trouvée, None sinon
        """

        # Chercher parmi tous les patterns
        for pattern in self.dofus_patterns:
            platform_window = self.platform_adapter.find_window(pattern)

            if platform_window:
                # Convertir vers format legacy
                self.dofus_window = WindowInfo.from_platform_window(platform_window)

                # Mettre à jour l'adapter vision
                self.vision_adapter.game_window = platform_window

                logger.info(f"✅ Fenêtre DOFUS trouvée: {self.dofus_window.title}")
                logger.info(f"   Taille: {self.dofus_window.width}x{self.dofus_window.height}")
                logger.info(f"   Position: ({platform_window.x}, {platform_window.y})")

                return self.dofus_window

        logger.warning("⚠️  Aucune fenêtre DOFUS trouvée")
        return None

    def capture_mss(self) -> Optional[np.ndarray]:
        """
        Capture via MSS - Cross-platform haute performance
        Utilisé par défaut sur Linux et Windows
        """
        if not self.dofus_window:
            return None

        try:
            # Utiliser l'adapter vision unifié
            screenshot = self.vision_adapter.capture(use_cache=False)
            return screenshot

        except Exception as e:
            logger.error(f"Erreur capture MSS: {e}")
            return None

    def capture_win32(self) -> Optional[np.ndarray]:
        """
        Capture via Win32 API - Windows uniquement (legacy)
        Gardé pour compatibilité mais mss est recommandé
        """
        if not self.is_windows:
            logger.warning("capture_win32 appelé sur Linux - utiliser capture_mss")
            return self.capture_mss()

        # Sur Windows, fallback vers mss (plus fiable)
        logger.debug("Win32 capture redirigé vers MSS pour performance")
        return self.capture_mss()

    def capture_pil_backup(self) -> Optional[np.ndarray]:
        """
        Capture PIL en backup - Plus lent mais fiable
        Fallback si MSS échoue
        """
        if not self.dofus_window:
            return None

        try:
            from PIL import ImageGrab

            left, top, right, bottom = self.dofus_window.rect
            screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        except Exception as e:
            logger.error(f"Erreur capture PIL: {e}")
            return None

    def capture_screenshot(self, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Capture screenshot avec méthode optimale (cross-platform)
        Cache intelligent pour performance

        Args:
            use_cache: Utiliser le cache si disponible (défaut: True)

        Returns:
            np.ndarray: Screenshot en format BGR OpenCV, ou None
        """

        # Vérification cache
        current_time = time.time()
        if (use_cache and self.capture_cache is not None and
            current_time - self.last_capture_time < self.cache_duration):
            return self.capture_cache

        # Recherche fenêtre si nécessaire
        if not self.dofus_window:
            self.find_dofus_window()
            if not self.dofus_window:
                logger.debug("Fenêtre DOFUS non trouvée - capture écran complet")
                # Fallback: capture écran complet
                try:
                    screenshot = self.vision_adapter.capture(use_cache=False)
                    if screenshot is not None:
                        self.capture_cache = screenshot.copy()
                        self.last_capture_time = current_time
                    return screenshot
                except Exception as e:
                    logger.error(f"Erreur capture fallback: {e}")
                    return None

        # Tentative de capture selon méthode
        screenshot = None

        if self.capture_method == "auto":
            # MSS en priorité (meilleure performance cross-platform)
            screenshot = self.capture_mss()
            if screenshot is None:
                screenshot = self.capture_pil_backup()

        elif self.capture_method == "win32":
            # Legacy Windows - redirigé vers mss
            screenshot = self.capture_win32()

        elif self.capture_method == "mss":
            screenshot = self.capture_mss()

        elif self.capture_method == "pil":
            screenshot = self.capture_pil_backup()

        # Mise à jour cache
        if screenshot is not None:
            self.capture_cache = screenshot.copy()
            self.last_capture_time = current_time

        return screenshot

    def get_window_info(self) -> Optional[WindowInfo]:
        """Retourne les informations de la fenêtre DOFUS"""
        return self.dofus_window

    def set_capture_method(self, method: str):
        """
        Définit la méthode de capture

        Args:
            method: "auto" (recommandé), "mss", "win32" (legacy), "pil" (backup)
        """
        if method in ["auto", "win32", "mss", "pil"]:
            self.capture_method = method
            logger.info(f"Méthode de capture: {method}")
        else:
            logger.warning(f"Méthode inconnue: {method}")

    def benchmark_capture_methods(self) -> dict:
        """
        Benchmark des différentes méthodes de capture

        Returns:
            dict: {method: {avg_time_ms, fps, success_rate}}
        """
        methods = ["mss", "pil"]
        if self.is_windows:
            methods.insert(0, "win32")

        results = {}

        for method in methods:
            self.set_capture_method(method)
            times = []

            for _ in range(10):
                start = time.time()
                screenshot = self.capture_screenshot(use_cache=False)
                end = time.time()

                if screenshot is not None:
                    times.append(end - start)

            if times:
                avg_time = sum(times) / len(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                results[method] = {
                    "avg_time_ms": avg_time * 1000,
                    "fps": fps,
                    "success_rate": len(times) / 10.0
                }

        # Remettre en auto
        self.set_capture_method("auto")

        return results

    def get_capture_stats(self) -> dict:
        """
        Statistiques de capture depuis l'adapter vision

        Returns:
            dict: Statistiques de performance
        """
        return {
            "frames_captured": self.vision_adapter.frames_captured,
            "avg_capture_time": (
                sum(self.vision_adapter.capture_times) / len(self.vision_adapter.capture_times)
                if self.vision_adapter.capture_times else 0
            ),
            "fps": (
                1.0 / (sum(self.vision_adapter.capture_times) / len(self.vision_adapter.capture_times))
                if self.vision_adapter.capture_times and
                   sum(self.vision_adapter.capture_times) > 0 else 0
            ),
            "cache_hits": getattr(self.vision_adapter, 'cache_hits', 0),
            "os": "Linux" if self.is_linux else "Windows"
        }


# Instance globale pour capture (singleton pattern)
_capture_instance = None

def get_capture_instance() -> DofusWindowCapture:
    """
    Retourne l'instance singleton de capture

    Returns:
        DofusWindowCapture: Instance unique de capture
    """
    global _capture_instance
    if _capture_instance is None:
        _capture_instance = DofusWindowCapture()
    return _capture_instance

def capture_dofus_window(use_cache: bool = True) -> Optional[np.ndarray]:
    """
    Fonction principale de capture DOFUS (interface simple)
    Compatible avec ancienne API

    Args:
        use_cache: Utiliser le cache si disponible

    Returns:
        np.ndarray: Screenshot BGR, ou None
    """
    capture = get_capture_instance()
    return capture.capture_screenshot(use_cache)

def find_dofus_window() -> Optional[WindowInfo]:
    """
    Trouve et retourne info fenêtre DOFUS
    Compatible avec ancienne API

    Returns:
        WindowInfo: Informations fenêtre, ou None
    """
    capture = get_capture_instance()
    return capture.find_dofus_window()

def benchmark_capture_performance() -> dict:
    """
    Benchmark performance des méthodes de capture
    Compatible avec ancienne API

    Returns:
        dict: Résultats benchmark par méthode
    """
    capture = get_capture_instance()
    return capture.benchmark_capture_methods()

def get_capture_stats() -> dict:
    """
    Statistiques de capture globales

    Returns:
        dict: Statistiques de performance
    """
    capture = get_capture_instance()
    return capture.get_capture_stats()


# Test du module
if __name__ == "__main__":
    print("🎮 Test Capture DOFUS Unity - Cross-Platform")
    print(f"OS: {'Linux' if PlatformAdapter.is_linux() else 'Windows'}")
    print()

    # Recherche fenêtre
    print("🔍 Recherche fenêtre DOFUS...")
    window_info = find_dofus_window()

    if window_info:
        print(f"✅ Fenêtre trouvée: {window_info.title}")
        print(f"   Taille: {window_info.width}x{window_info.height}")
        print(f"   Position: {window_info.rect}")
        print()

        # Test capture
        print("📸 Test capture...")
        screenshot = capture_dofus_window()

        if screenshot is not None:
            print(f"✅ Capture réussie: {screenshot.shape}")
            print(f"   Type: {screenshot.dtype}")
            print()

            # Sauvegarde test
            output_path = "test_capture_unified.png"
            cv2.imwrite(output_path, screenshot)
            print(f"💾 Capture sauvée: {output_path}")
            print()

            # Benchmark
            print("⚡ Benchmark performance:")
            results = benchmark_capture_performance()
            for method, stats in results.items():
                print(f"   {method:8s}: {stats['fps']:6.1f} FPS ({stats['avg_time_ms']:5.1f}ms) - Success: {stats['success_rate']*100:.0f}%")
            print()

            # Stats globales
            print("📊 Statistiques globales:")
            stats = get_capture_stats()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")

        else:
            print("❌ Échec capture")

    else:
        print("❌ Fenêtre DOFUS non trouvée")
        print()
        print("💡 Solutions:")
        print("   1. Lancez DOFUS Unity")
        print("   2. Vérifiez que la fenêtre est visible")
        if PlatformAdapter.is_linux():
            print("   3. Vérifiez que xdotool est installé: sudo apt install xdotool")
        print()
        print("📸 Test capture écran complet...")

        # Test capture écran complet en fallback
        capture = get_capture_instance()
        screenshot = capture.vision_adapter.capture(use_cache=False)

        if screenshot is not None:
            print(f"✅ Capture écran complet réussie: {screenshot.shape}")
            cv2.imwrite("test_capture_fullscreen.png", screenshot)
            print("💾 Sauvegardé: test_capture_fullscreen.png")
