"""
GameCapture - Système de capture d'écran optimisé pour DOFUS Unity
Inspiré des techniques DofuBot et Inkybot 2025

Fonctionnalités:
- Capture DirectX optimisée 60fps
- Détection automatique fenêtre DOFUS
- Multiple méthodes de capture (fallback)
- Cache intelligent pour performance
"""

import time
import threading
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import logging

# Standard libraries
import numpy as np
from PIL import Image, ImageGrab
import cv2

# Windows API (conditionnel)
try:
    import win32gui
    import win32api
    import win32con
    import win32ui
    from ctypes import windll
    WINDOWS_API_AVAILABLE = True
except ImportError:
    WINDOWS_API_AVAILABLE = False
    print("Windows API non disponible - fonctionnalités limitées")

@dataclass
class CaptureRegion:
    """Région de capture définie"""
    x: int
    y: int
    width: int
    height: int
    name: str = "default"

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

@dataclass
class WindowInfo:
    """Informations fenêtre DOFUS"""
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]
    client_rect: Tuple[int, int, int, int]
    pid: int
    is_active: bool = False
    is_dofus: bool = False

class GameCapture:
    """Capture d'écran optimisée pour DOFUS Unity"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.target_fps = 60
        self.capture_quality = "high"  # high, medium, low
        self.auto_detect_window = True

        # État capture
        self.dofus_window: Optional[WindowInfo] = None
        self.capture_region: Optional[CaptureRegion] = None
        self.last_screenshot: Optional[np.ndarray] = None
        self.last_capture_time = 0.0

        # Cache et optimisation
        self.enable_cache = True
        self.cache_duration = 0.016  # ~60fps cache
        self.frame_buffer: List[np.ndarray] = []
        self.max_buffer_size = 5

        # Threading
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

        # Métriques performance
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.actual_fps = 0.0

        self.logger.info("GameCapture initialisé")

    def initialize(self) -> bool:
        """Initialise le système de capture"""
        try:
            if not WINDOWS_API_AVAILABLE:
                self.logger.warning("API Windows non disponible - mode compatibilité")

            # Détection fenêtre DOFUS
            if self.auto_detect_window:
                if not self.detect_dofus_window():
                    self.logger.warning("Fenêtre DOFUS non détectée")
                    return False

            self.logger.info("GameCapture initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation GameCapture: {e}")
            return False

    def detect_dofus_window(self) -> bool:
        """Détecte automatiquement la fenêtre DOFUS Unity"""
        if not WINDOWS_API_AVAILABLE:
            return False

        def enum_windows_callback(hwnd, windows_list):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and ("dofus" in title.lower() or "ankama" in title.lower()):
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        client_rect = win32gui.GetClientRect(hwnd)
                        pid = win32gui.GetWindowThreadProcessId(hwnd)[1]

                        window_info = WindowInfo(
                            hwnd=hwnd,
                            title=title,
                            rect=rect,
                            client_rect=client_rect,
                            pid=pid,
                            is_active=hwnd == win32gui.GetForegroundWindow(),
                            is_dofus=True
                        )
                        windows_list.append(window_info)
                    except Exception as e:
                        self.logger.debug(f"Erreur info fenêtre {hwnd}: {e}")
            return True

        windows_list = []
        try:
            win32gui.EnumWindows(enum_windows_callback, windows_list)

            if windows_list:
                # Prioriser fenêtre active
                active_windows = [w for w in windows_list if w.is_active]
                if active_windows:
                    self.dofus_window = active_windows[0]
                else:
                    self.dofus_window = windows_list[0]

                self.logger.info(f"Fenêtre DOFUS détectée: {self.dofus_window.title}")

                # Définir région de capture
                rect = self.dofus_window.client_rect
                self.capture_region = CaptureRegion(
                    x=rect[0], y=rect[1],
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1],
                    name="dofus_client"
                )

                return True
            else:
                self.logger.warning("Aucune fenêtre DOFUS trouvée")
                return False

        except Exception as e:
            self.logger.error(f"Erreur détection fenêtre: {e}")
            return False

    def capture_screenshot_fast(self, region: Optional[CaptureRegion] = None) -> Optional[np.ndarray]:
        """Capture rapide optimisée performance"""
        try:
            # Vérifier cache
            current_time = time.time()
            if (self.enable_cache and
                self.last_screenshot is not None and
                current_time - self.last_capture_time < self.cache_duration):
                return self.last_screenshot.copy()

            # Région de capture
            if region is None:
                region = self.capture_region

            if region is None:
                # Capture écran complet en fallback
                screenshot = self._capture_fullscreen()
            else:
                # Capture région spécifique
                screenshot = self._capture_region(region)

            if screenshot is not None:
                # Mise à jour cache
                with self.lock:
                    self.last_screenshot = screenshot
                    self.last_capture_time = current_time

                    # Buffer frames
                    if len(self.frame_buffer) >= self.max_buffer_size:
                        self.frame_buffer.pop(0)
                    self.frame_buffer.append(screenshot.copy())

                # Mise à jour FPS
                self._update_fps_counter()

                return screenshot
            else:
                self.logger.warning("Capture échouée")
                return None

        except Exception as e:
            self.logger.error(f"Erreur capture screenshot: {e}")
            return None

    def _capture_region(self, region: CaptureRegion) -> Optional[np.ndarray]:
        """Capture une région spécifique avec multiple méthodes"""

        # Méthode 1: Win32 DirectX (plus rapide)
        if WINDOWS_API_AVAILABLE and self.dofus_window:
            screenshot = self._capture_win32_dx(region)
            if screenshot is not None:
                return screenshot

        # Méthode 2: PIL ImageGrab (compatible)
        screenshot = self._capture_pil(region)
        if screenshot is not None:
            return screenshot

        # Méthode 3: OpenCV (fallback)
        return self._capture_opencv(region)

    def _capture_win32_dx(self, region: CaptureRegion) -> Optional[np.ndarray]:
        """Capture DirectX optimisée Windows"""
        try:
            hwnd = self.dofus_window.hwnd

            # Obtenir device context
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            # Créer bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, region.width, region.height)

            saveDC.SelectObject(saveBitMap)

            # Copier pixels
            result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

            if result:
                # Convertir en numpy array
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)

                img = np.frombuffer(bmpstr, dtype='uint8')
                img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
                img = img[..., :3]  # Supprimer canal alpha
                img = img[..., ::-1]  # BGR vers RGB

                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)

                return img
            else:
                # Cleanup en cas d'échec
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                return None

        except Exception as e:
            self.logger.debug(f"Capture Win32 échouée: {e}")
            return None

    def _capture_pil(self, region: CaptureRegion) -> Optional[np.ndarray]:
        """Capture PIL ImageGrab"""
        try:
            screenshot = ImageGrab.grab(bbox=region.bounds)
            return np.array(screenshot)
        except Exception as e:
            self.logger.debug(f"Capture PIL échouée: {e}")
            return None

    def _capture_opencv(self, region: CaptureRegion) -> Optional[np.ndarray]:
        """Capture OpenCV (méthode fallback)"""
        try:
            # OpenCV n'a pas de capture écran native
            # Utiliser PIL puis convertir
            screenshot = self._capture_pil(region)
            if screenshot is not None:
                return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            self.logger.debug(f"Capture OpenCV échouée: {e}")
            return None

    def _capture_fullscreen(self) -> Optional[np.ndarray]:
        """Capture écran complet en fallback"""
        try:
            screenshot = ImageGrab.grab()
            return np.array(screenshot)
        except Exception as e:
            self.logger.error(f"Capture fullscreen échouée: {e}")
            return None

    def start_continuous_capture(self, fps: int = 60) -> bool:
        """Démarre capture continue en thread"""
        if self.running:
            self.logger.warning("Capture continue déjà active")
            return False

        self.target_fps = fps
        self.running = True

        self.capture_thread = threading.Thread(
            target=self._continuous_capture_loop,
            daemon=True
        )
        self.capture_thread.start()

        self.logger.info(f"Capture continue démarrée ({fps} fps)")
        return True

    def stop_continuous_capture(self):
        """Arrête capture continue"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.logger.info("Capture continue arrêtée")

    def _continuous_capture_loop(self):
        """Boucle capture continue"""
        frame_duration = 1.0 / self.target_fps

        while self.running:
            start_time = time.time()

            # Capture frame
            screenshot = self.capture_screenshot_fast()

            if screenshot is not None:
                # Process frame si nécessaire
                pass

            # Attendre pour maintenir FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_fps_counter(self):
        """Met à jour compteur FPS"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_last_time >= 1.0:
            self.actual_fps = self.fps_counter / (current_time - self.fps_last_time)
            self.fps_counter = 0
            self.fps_last_time = current_time

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Retourne dernière frame capturée"""
        with self.lock:
            if self.last_screenshot is not None:
                return self.last_screenshot.copy()
            return None

    def get_frame_buffer(self) -> List[np.ndarray]:
        """Retourne buffer des dernières frames"""
        with self.lock:
            return [frame.copy() for frame in self.frame_buffer]

    def set_capture_quality(self, quality: str):
        """Configure qualité capture (high/medium/low)"""
        self.capture_quality = quality

        if quality == "high":
            self.target_fps = 60
            self.cache_duration = 0.016
        elif quality == "medium":
            self.target_fps = 30
            self.cache_duration = 0.033
        else:  # low
            self.target_fps = 15
            self.cache_duration = 0.066

        self.logger.info(f"Qualité capture: {quality} ({self.target_fps} fps)")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne statistiques performance"""
        return {
            "actual_fps": round(self.actual_fps, 2),
            "target_fps": self.target_fps,
            "buffer_size": len(self.frame_buffer),
            "cache_enabled": self.enable_cache,
            "capture_quality": self.capture_quality,
            "window_detected": self.dofus_window is not None,
            "window_title": self.dofus_window.title if self.dofus_window else None
        }

    def save_screenshot(self, filepath: str, region: Optional[CaptureRegion] = None) -> bool:
        """Sauvegarde screenshot"""
        try:
            screenshot = self.capture_screenshot_fast(region)
            if screenshot is not None:
                # Convertir numpy vers PIL
                if len(screenshot.shape) == 3:
                    image = Image.fromarray(screenshot, 'RGB')
                else:
                    image = Image.fromarray(screenshot)

                image.save(filepath)
                self.logger.info(f"Screenshot sauvegardé: {filepath}")
                return True
            else:
                self.logger.error("Impossible de capturer screenshot")
                return False
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde screenshot: {e}")
            return False

    def cleanup(self):
        """Nettoyage ressources"""
        self.stop_continuous_capture()
        with self.lock:
            self.frame_buffer.clear()
            self.last_screenshot = None
        self.logger.info("GameCapture nettoyé")

    def __del__(self):
        """Destructeur"""
        self.cleanup()

# Factory function
def create_game_capture() -> GameCapture:
    """Crée instance GameCapture configurée"""
    capture = GameCapture()
    if capture.initialize():
        return capture
    else:
        raise RuntimeError("Impossible d'initialiser GameCapture")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        capture = create_game_capture()

        print("Test capture...")
        screenshot = capture.capture_screenshot_fast()

        if screenshot is not None:
            print(f"Capture réussie: {screenshot.shape}")

            # Sauvegarde test
            test_path = "test_capture.png"
            if capture.save_screenshot(test_path):
                print(f"Screenshot sauvegardé: {test_path}")
        else:
            print("Capture échouée")

        # Stats
        stats = capture.get_performance_stats()
        print(f"Stats: {stats}")

    except Exception as e:
        print(f"Erreur test: {e}")
    finally:
        if 'capture' in locals():
            capture.cleanup()