"""
Input Controller - Contrôle clavier/souris avec humanisation
Compatible Windows + Linux
"""

import time
import random
import logging
import platform
import pyautogui
from typing import Tuple, Optional
from dataclasses import dataclass

# Import conditionnel Windows
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

if IS_WINDOWS:
    try:
        import win32gui
        import win32con
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
else:
    WIN32_AVAILABLE = False
    # Import couche d'abstraction pour Linux
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from core.platform_adapter import WindowManager as PlatformWindowManager

logger = logging.getLogger(__name__)

# Configuration PyAutoGUI pour sécurité
pyautogui.FAILSAFE = True  # Curseur coin haut-gauche = arrêt
pyautogui.PAUSE = 0.01  # Pause minimale entre actions


@dataclass
class MouseMovement:
    """Configuration mouvement souris"""
    duration: float = 0.3
    tween: any = pyautogui.easeInOutQuad
    human_variation: bool = True


class InputController:
    """Contrôle bas-niveau des entrées (souris, clavier) - Multiplateforme"""

    def __init__(self, window_title: str = "Dofus"):
        self.window_title = window_title
        self.window_handle = None
        self.window_rect = None
        self.is_windows = IS_WINDOWS
        self.is_linux = IS_LINUX

        # Statistiques
        self.stats = {
            'clicks': 0,
            'moves': 0,
            'keypress': 0,
            'failures': 0
        }

        # Platform manager pour Linux
        if self.is_linux:
            self.platform_window_mgr = PlatformWindowManager()

        logger.info(f"InputController initialisé (OS: {platform.system()})")
    
    def find_window(self) -> bool:
        """Trouve la fenêtre Dofus (multiplateforme)"""
        try:
            if self.is_windows and WIN32_AVAILABLE:
                # Windows: win32gui
                self.window_handle = win32gui.FindWindow(None, self.window_title)

                if self.window_handle:
                    self.window_rect = win32gui.GetWindowRect(self.window_handle)
                    logger.info(f"Fenêtre trouvée (Windows): {self.window_title}")
                    return True
                else:
                    logger.warning(f"Fenêtre non trouvée: {self.window_title}")
                    return False

            elif self.is_linux:
                # Linux: Xlib via platform_adapter
                window_info = self.platform_window_mgr.find_window(self.window_title)

                if window_info:
                    self.window_handle = window_info.handle
                    self.window_rect = (window_info.x, window_info.y,
                                      window_info.x + window_info.width,
                                      window_info.y + window_info.height)
                    logger.info(f"Fenêtre trouvée (Linux): {window_info.title}")
                    return True
                else:
                    logger.warning(f"Fenêtre non trouvée: {self.window_title}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Erreur recherche fenêtre: {e}")
            return False
    
    def activate_window(self) -> bool:
        """Active la fenêtre Dofus (multiplateforme)"""
        if not self.window_handle:
            if not self.find_window():
                return False

        try:
            if self.is_windows and WIN32_AVAILABLE:
                win32gui.SetForegroundWindow(self.window_handle)
                time.sleep(0.1)
                return True

            elif self.is_linux:
                # Linux: utiliser wmctrl ou xdotool
                import subprocess
                try:
                    subprocess.run(['wmctrl', '-i', '-a', str(self.window_handle)],
                                 check=True, capture_output=True)
                    time.sleep(0.1)
                    return True
                except subprocess.CalledProcessError:
                    # Fallback: xdotool
                    try:
                        subprocess.run(['xdotool', 'windowactivate', str(self.window_handle)],
                                     check=True, capture_output=True)
                        time.sleep(0.1)
                        return True
                    except:
                        logger.error("wmctrl et xdotool échoués")
                        return False

            return False
        except Exception as e:
            logger.error(f"Erreur activation fenêtre: {e}")
            return False
    
    def is_window_active(self) -> bool:
        """Vérifie si la fenêtre est active (multiplateforme)"""
        if not self.window_handle:
            return False

        try:
            if self.is_windows and WIN32_AVAILABLE:
                return win32gui.GetForegroundWindow() == self.window_handle

            elif self.is_linux:
                # Linux: vérifier fenêtre active via xdotool
                import subprocess
                try:
                    result = subprocess.run(['xdotool', 'getactivewindow'],
                                          capture_output=True, text=True, check=True)
                    active_id = result.stdout.strip()
                    return str(self.window_handle) == active_id
                except:
                    return False

            return False
        except:
            return False
    
    def get_window_pos(self) -> Optional[Tuple[int, int, int, int]]:
        """Retourne position fenêtre (left, top, right, bottom) - multiplateforme"""
        if not self.window_handle:
            if not self.find_window():
                return None

        try:
            if self.is_windows and WIN32_AVAILABLE:
                return win32gui.GetWindowRect(self.window_handle)

            elif self.is_linux:
                # Déjà stocké dans self.window_rect lors de find_window
                return self.window_rect

            return None
        except:
            return None
    
    def to_screen_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convertit coordonnées relatives fenêtre → écran"""
        if not self.window_rect:
            self.find_window()
        
        if self.window_rect:
            left, top, _, _ = self.window_rect
            return (left + x, top + y)
        
        return (x, y)
    
    def move_mouse(
        self, 
        x: int, 
        y: int, 
        relative: bool = True,
        humanize: bool = True,
        duration: float = 0.3
    ) -> bool:
        """
        Déplace la souris
        
        Args:
            x, y: Coordonnées destination
            relative: Si True, x/y relatifs à la fenêtre
            humanize: Ajoute variation humaine
            duration: Durée du mouvement
        """
        try:
            # Convertir en coordonnées écran si nécessaire
            if relative:
                screen_x, screen_y = self.to_screen_coords(x, y)
            else:
                screen_x, screen_y = x, y
            
            # Variation humaine
            if humanize:
                screen_x += random.randint(-2, 2)
                screen_y += random.randint(-2, 2)
                duration += random.uniform(-0.05, 0.10)
            
            # Mouvement
            pyautogui.moveTo(
                screen_x, 
                screen_y, 
                duration=max(0.1, duration),
                tween=pyautogui.easeInOutQuad
            )
            
            self.stats['moves'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Erreur mouvement souris: {e}")
            self.stats['failures'] += 1
            return False
    
    def click(
        self, 
        x: Optional[int] = None, 
        y: Optional[int] = None,
        button: str = 'left',
        clicks: int = 1,
        relative: bool = True,
        humanize: bool = True
    ) -> bool:
        """
        Clic souris
        
        Args:
            x, y: Position (None = position actuelle)
            button: 'left', 'right', 'middle'
            clicks: Nombre de clics
            relative: Coordonnées relatives à la fenêtre
            humanize: Variation humaine
        """
        try:
            # Déplacer si position spécifiée
            if x is not None and y is not None:
                if not self.move_mouse(x, y, relative, humanize):
                    return False
                
                # Petite pause après mouvement
                time.sleep(random.uniform(0.05, 0.15) if humanize else 0.05)
            
            # Clic
            pyautogui.click(button=button, clicks=clicks)
            
            self.stats['clicks'] += 1
            
            # Pause post-clic
            if humanize:
                time.sleep(random.uniform(0.05, 0.15))
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur clic: {e}")
            self.stats['failures'] += 1
            return False
    
    def double_click(self, x: int, y: int, relative: bool = True) -> bool:
        """Double-clic"""
        return self.click(x, y, clicks=2, relative=relative)
    
    def right_click(self, x: int, y: int, relative: bool = True) -> bool:
        """Clic droit"""
        return self.click(x, y, button='right', relative=relative)
    
    def press_key(self, key: str, duration: float = 0.1) -> bool:
        """
        Appuie sur une touche
        
        Args:
            key: Touche ('a', 'enter', 'space', 'ctrl', etc.)
            duration: Durée appui
        """
        try:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            
            self.stats['keypress'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Erreur touche: {e}")
            self.stats['failures'] += 1
            return False
    
    def press_keys(self, *keys: str) -> bool:
        """Appuie sur plusieurs touches (raccourci)"""
        try:
            pyautogui.hotkey(*keys)
            self.stats['keypress'] += 1
            return True
        except Exception as e:
            logger.error(f"Erreur raccourci: {e}")
            self.stats['failures'] += 1
            return False
    
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Tape du texte"""
        try:
            pyautogui.write(text, interval=interval)
            self.stats['keypress'] += len(text)
            return True
        except Exception as e:
            logger.error(f"Erreur saisie texte: {e}")
            return False
    
    def drag(
        self, 
        start_x: int, 
        start_y: int, 
        end_x: int, 
        end_y: int,
        duration: float = 0.5,
        relative: bool = True
    ) -> bool:
        """Drag & drop"""
        try:
            # Déplacer au point de départ
            if not self.move_mouse(start_x, start_y, relative):
                return False
            
            time.sleep(0.1)
            
            # Convertir destination
            if relative:
                screen_end_x, screen_end_y = self.to_screen_coords(end_x, end_y)
            else:
                screen_end_x, screen_end_y = end_x, end_y
            
            # Drag
            pyautogui.drag(
                screen_end_x - pyautogui.position()[0],
                screen_end_y - pyautogui.position()[1],
                duration=duration,
                button='left'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur drag: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Retourne statistiques"""
        return self.stats.copy()
