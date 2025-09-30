"""
Screenshot Capture Optimisé DOFUS Unity
Capture d'écran haute performance spécialement optimisée pour DOFUS Unity
Approche 100% vision - Détection automatique fenêtre DOFUS
"""

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from typing import Optional, Tuple, List
import time
import logging
from dataclasses import dataclass
from PIL import Image
import mss

logger = logging.getLogger(__name__)

@dataclass
class WindowInfo:
    """Informations sur la fenêtre DOFUS"""
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    width: int
    height: int
    is_visible: bool

class DofusWindowCapture:
    """
    Capture de fenêtre DOFUS Unity optimisée
    Méthodes multiples pour performance maximale
    """

    def __init__(self):
        self.dofus_window: Optional[WindowInfo] = None
        self.capture_method = "auto"  # auto, win32, mss, pil
        self.last_capture_time = 0
        self.capture_cache = None
        self.cache_duration = 0.05  # 50ms cache

        # Patterns de titre pour DOFUS Unity
        self.dofus_patterns = [
            "DOFUS",
            "Dofus Unity",
            "DOFUS Unity",
            "dofus.exe",
            "Ankama"
        ]

        logger.info("DofusWindowCapture initialisé")

    def find_dofus_window(self) -> Optional[WindowInfo]:
        """Trouve la fenêtre DOFUS Unity active"""

        def enum_window_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                for pattern in self.dofus_patterns:
                    if pattern.lower() in window_title.lower():
                        rect = win32gui.GetWindowRect(hwnd)
                        windows.append(WindowInfo(
                            hwnd=hwnd,
                            title=window_title,
                            rect=rect,
                            width=rect[2] - rect[0],
                            height=rect[3] - rect[1],
                            is_visible=True
                        ))
            return True

        windows = []
        win32gui.EnumWindows(enum_window_callback, windows)

        if windows:
            # Prendre la première fenêtre DOFUS trouvée
            self.dofus_window = windows[0]
            logger.info(f"Fenêtre DOFUS trouvée: {self.dofus_window.title}")
            return self.dofus_window

        logger.warning("Aucune fenêtre DOFUS trouvée")
        return None

    def capture_win32(self) -> Optional[np.ndarray]:
        """Capture via Win32 API - Haute performance"""
        if not self.dofus_window:
            return None

        try:
            hwnd = self.dofus_window.hwnd
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top

            # Contexte device
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # Bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Capture
            result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

            if result:
                # Conversion en numpy array
                bmp_info = save_bitmap.GetInfo()
                bmp_str = save_bitmap.GetBitmapBits(True)

                img = np.frombuffer(bmp_str, dtype=np.uint8)
                img = img.reshape((height, width, 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Cleanup
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)

                return img

        except Exception as e:
            logger.error(f"Erreur capture Win32: {e}")

        return None

    def capture_mss(self) -> Optional[np.ndarray]:
        """Capture via MSS - Alternative rapide"""
        if not self.dofus_window:
            return None

        try:
            with mss.mss() as sct:
                left, top, right, bottom = self.dofus_window.rect
                monitor = {
                    "top": top,
                    "left": left,
                    "width": right - left,
                    "height": bottom - top
                }

                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img

        except Exception as e:
            logger.error(f"Erreur capture MSS: {e}")

        return None

    def capture_pil_backup(self) -> Optional[np.ndarray]:
        """Capture PIL en backup - Plus lent mais fiable"""
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
        Capture screenshot avec méthode optimale
        Cache intelligent pour performance
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
                return None

        # Tentative de capture selon méthode
        screenshot = None

        if self.capture_method == "auto":
            # Essai Win32 en premier (plus rapide)
            screenshot = self.capture_win32()
            if screenshot is None:
                screenshot = self.capture_mss()
            if screenshot is None:
                screenshot = self.capture_pil_backup()

        elif self.capture_method == "win32":
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
        """Définit la méthode de capture (auto, win32, mss, pil)"""
        if method in ["auto", "win32", "mss", "pil"]:
            self.capture_method = method
            logger.info(f"Méthode de capture: {method}")
        else:
            logger.warning(f"Méthode inconnue: {method}")

    def benchmark_capture_methods(self) -> dict:
        """Benchmark des différentes méthodes de capture"""
        methods = ["win32", "mss", "pil"]
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

# Instance globale pour capture
_capture_instance = None

def get_capture_instance() -> DofusWindowCapture:
    """Retourne l'instance singleton de capture"""
    global _capture_instance
    if _capture_instance is None:
        _capture_instance = DofusWindowCapture()
    return _capture_instance

def capture_dofus_window(use_cache: bool = True) -> Optional[np.ndarray]:
    """
    Fonction principale de capture DOFUS
    Interface simple pour autres modules
    """
    capture = get_capture_instance()
    return capture.capture_screenshot(use_cache)

def find_dofus_window() -> Optional[WindowInfo]:
    """Trouve et retourne info fenêtre DOFUS"""
    capture = get_capture_instance()
    return capture.find_dofus_window()

def benchmark_capture_performance() -> dict:
    """Benchmark performance des méthodes de capture"""
    capture = get_capture_instance()
    return capture.benchmark_capture_methods()

# Test du module
if __name__ == "__main__":
    print("Test capture DOFUS Unity")

    # Recherche fenêtre
    window_info = find_dofus_window()
    if window_info:
        print(f"Fenêtre trouvée: {window_info.title}")
        print(f"Taille: {window_info.width}x{window_info.height}")

        # Test capture
        screenshot = capture_dofus_window()
        if screenshot is not None:
            print(f"Capture réussie: {screenshot.shape}")

            # Sauvegarde test
            cv2.imwrite("test_capture.png", screenshot)
            print("Capture sauvée: test_capture.png")

            # Benchmark
            print("\nBenchmark performance:")
            results = benchmark_capture_performance()
            for method, stats in results.items():
                print(f"{method}: {stats['fps']:.1f} FPS ({stats['avg_time_ms']:.1f}ms)")

        else:
            print("Échec capture")
    else:
        print("Fenêtre DOFUS non trouvée - Lancez DOFUS Unity")