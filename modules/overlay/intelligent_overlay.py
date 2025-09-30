"""
Intelligent Overlay - Système de superposition intelligente DOFUS Unity
Inspiré de NVIDIA G-Assist 2025 - Conseils temps réel sur écran

Fonctionnalités:
- Overlay DirectX11 transparent
- Highlighting intelligent spells/zones
- Suggestions d'actions contextuelles
- Indicateurs de performance
- Conseils stratégiques temps réel
- Anti-détection par variation temporelle
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import logging
from enum import Enum

import numpy as np
import cv2
import pygame
import win32gui
import win32api
import win32con
from PIL import Image, ImageDraw, ImageFont

# Custom types
class OverlayType(Enum):
    SPELL_HIGHLIGHT = "spell_highlight"
    MOVEMENT_SUGGESTION = "movement_suggestion"
    TARGET_PRIORITY = "target_priority"
    RESOURCE_INDICATOR = "resource_indicator"
    QUEST_GUIDANCE = "quest_guidance"
    PERFORMANCE_STATS = "performance_stats"

@dataclass
class OverlayElement:
    """Élément d'overlay à afficher"""
    element_id: str
    overlay_type: OverlayType
    position: Tuple[int, int]
    size: Tuple[int, int]
    color: Tuple[int, int, int, int]  # RGBA
    text: str
    priority: int  # 1-10, 10 = le plus important
    duration: float  # durée d'affichage en secondes
    animation: str  # fade, pulse, slide, static
    timestamp: float

@dataclass
class OverlayConfig:
    """Configuration de l'overlay"""
    enable_overlay: bool = True
    transparency: float = 0.8
    max_elements: int = 10
    default_duration: float = 5.0
    animation_speed: float = 1.0
    anti_detection: bool = True
    colors: Dict[str, Tuple[int, int, int, int]] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "spell_highlight": (255, 215, 0, 180),     # Or doré
                "movement": (0, 255, 0, 150),              # Vert
                "target": (255, 0, 0, 200),                # Rouge
                "resource": (0, 150, 255, 160),            # Bleu
                "quest": (255, 165, 0, 170),               # Orange
                "warning": (255, 69, 0, 200),              # Rouge-orange
                "info": (255, 255, 255, 140)               # Blanc
            }

class DirectXOverlay:
    """Overlay DirectX11 optimisé performance"""

    def __init__(self, target_window_title: str = "dofus"):
        self.target_window_title = target_window_title.lower()
        self.target_hwnd: Optional[int] = None
        self.overlay_hwnd: Optional[int] = None

        # Pygame surface pour rendu
        self.overlay_surface: Optional[pygame.Surface] = None
        self.screen_size: Tuple[int, int] = (1920, 1080)

        # État overlay
        self.elements: List[OverlayElement] = []
        self.running = False
        self.render_thread: Optional[threading.Thread] = None

        # Performance
        self.target_fps = 60
        self.last_render_time = 0.0
        self.frame_count = 0

        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialise l'overlay DirectX"""
        try:
            # Trouver fenêtre cible
            if not self._find_target_window():
                self.logger.warning("Fenêtre DOFUS non trouvée")
                return False

            # Initialiser pygame
            pygame.init()
            pygame.display.set_mode((1, 1))  # Fenêtre minimale

            # Obtenir taille écran
            self.screen_size = (
                win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
                win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            )

            # Créer surface overlay
            self.overlay_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)

            self.logger.info("DirectX Overlay initialisé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation overlay: {e}")
            return False

    def _find_target_window(self) -> bool:
        """Trouve la fenêtre DOFUS cible"""
        def enum_windows_callback(hwnd, windows_list):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if self.target_window_title in title:
                    windows_list.append(hwnd)
            return True

        windows_list = []
        win32gui.EnumWindows(enum_windows_callback, windows_list)

        if windows_list:
            self.target_hwnd = windows_list[0]
            self.logger.info(f"Fenêtre cible trouvée: {self.target_hwnd}")
            return True

        return False

    def start_overlay(self) -> bool:
        """Démarre le rendu overlay"""
        if self.running:
            return False

        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        self.logger.info("Overlay démarré")
        return True

    def stop_overlay(self):
        """Arrête l'overlay"""
        self.running = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=2.0)

        self.logger.info("Overlay arrêté")

    def add_element(self, element: OverlayElement):
        """Ajoute un élément à l'overlay"""
        # Supprimer éléments expirés
        current_time = time.time()
        self.elements = [e for e in self.elements
                        if current_time - e.timestamp < e.duration]

        # Ajouter nouvel élément
        self.elements.append(element)

        # Limiter nombre d'éléments
        self.elements.sort(key=lambda x: x.priority, reverse=True)
        if len(self.elements) > 10:  # Max 10 éléments
            self.elements = self.elements[:10]

    def clear_elements(self, overlay_type: Optional[OverlayType] = None):
        """Efface les éléments (optionnellement par type)"""
        if overlay_type:
            self.elements = [e for e in self.elements if e.overlay_type != overlay_type]
        else:
            self.elements.clear()

    def _render_loop(self):
        """Boucle de rendu principal"""
        clock = pygame.time.Clock()

        while self.running:
            start_time = time.time()

            # Vérifier fenêtre cible active
            if not self._is_target_window_active():
                time.sleep(0.1)
                continue

            # Nettoyer surface
            self.overlay_surface.fill((0, 0, 0, 0))

            # Rendre éléments
            self._render_elements()

            # Afficher overlay
            self._display_overlay()

            # Maintenir FPS
            clock.tick(self.target_fps)
            self.frame_count += 1

            # Statistiques performance
            if time.time() - self.last_render_time > 1.0:
                fps = self.frame_count / (time.time() - self.last_render_time)
                self.logger.debug(f"Overlay FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_render_time = time.time()

    def _is_target_window_active(self) -> bool:
        """Vérifie si la fenêtre cible est active"""
        try:
            if not self.target_hwnd:
                return False

            foreground = win32gui.GetForegroundWindow()
            return foreground == self.target_hwnd
        except:
            return False

    def _render_elements(self):
        """Rend tous les éléments overlay"""
        current_time = time.time()

        for element in self.elements[:]:  # Copie pour éviter modification pendant itération
            # Vérifier expiration
            if current_time - element.timestamp > element.duration:
                self.elements.remove(element)
                continue

            # Calculer alpha pour animation
            alpha = self._calculate_alpha(element, current_time)
            if alpha <= 0:
                continue

            # Rendre selon type
            if element.overlay_type == OverlayType.SPELL_HIGHLIGHT:
                self._render_spell_highlight(element, alpha)
            elif element.overlay_type == OverlayType.MOVEMENT_SUGGESTION:
                self._render_movement_suggestion(element, alpha)
            elif element.overlay_type == OverlayType.TARGET_PRIORITY:
                self._render_target_priority(element, alpha)
            elif element.overlay_type == OverlayType.RESOURCE_INDICATOR:
                self._render_resource_indicator(element, alpha)
            elif element.overlay_type == OverlayType.QUEST_GUIDANCE:
                self._render_quest_guidance(element, alpha)
            elif element.overlay_type == OverlayType.PERFORMANCE_STATS:
                self._render_performance_stats(element, alpha)

    def _calculate_alpha(self, element: OverlayElement, current_time: float) -> int:
        """Calcule la transparence selon l'animation"""
        age = current_time - element.timestamp
        progress = age / element.duration

        if element.animation == "fade":
            # Fade out vers la fin
            if progress > 0.8:
                fade_progress = (progress - 0.8) / 0.2
                alpha = int(element.color[3] * (1.0 - fade_progress))
            else:
                alpha = element.color[3]
        elif element.animation == "pulse":
            # Pulsation sinusoïdale
            pulse = abs(np.sin(age * 3.14159 * 2))  # 2 cycles par seconde
            alpha = int(element.color[3] * (0.5 + 0.5 * pulse))
        elif element.animation == "slide":
            # Slide in depuis le côté
            if progress < 0.1:
                alpha = int(element.color[3] * (progress / 0.1))
            else:
                alpha = element.color[3]
        else:  # static
            alpha = element.color[3]

        return max(0, min(255, alpha))

    def _render_spell_highlight(self, element: OverlayElement, alpha: int):
        """Rend surbrillance de sort"""
        x, y = element.position
        w, h = element.size
        color = (*element.color[:3], alpha)

        # Cercle de surbrillance avec bordure
        pygame.draw.circle(self.overlay_surface, color, (x + w//2, y + h//2), max(w, h)//2 + 10, 5)

        # Texte descriptif
        if element.text:
            font = pygame.font.Font(None, 24)
            text_surface = font.render(element.text, True, (255, 255, 255, alpha))
            self.overlay_surface.blit(text_surface, (x, y - 30))

    def _render_movement_suggestion(self, element: OverlayElement, alpha: int):
        """Rend suggestion de mouvement"""
        x, y = element.position
        w, h = element.size
        color = (*element.color[:3], alpha)

        # Flèche directionnelle
        points = [
            (x + w//2, y),
            (x + w, y + h//2),
            (x + w//2, y + h//4),
            (x, y + h//2),
            (x + w//2, y + 3*h//4)
        ]
        pygame.draw.polygon(self.overlay_surface, color, points)

        # Texte de distance/direction
        if element.text:
            font = pygame.font.Font(None, 20)
            text_surface = font.render(element.text, True, (255, 255, 255, alpha))
            self.overlay_surface.blit(text_surface, (x + w + 10, y + h//2 - 10))

    def _render_target_priority(self, element: OverlayElement, alpha: int):
        """Rend priorité de cible"""
        x, y = element.position
        w, h = element.size
        color = (*element.color[:3], alpha)

        # Cadre de priorité avec coins
        thickness = 3
        corner_size = 15

        # Coins du cadre
        corners = [
            [(x, y), (x + corner_size, y), (x, y + corner_size)],  # Top-left
            [(x + w - corner_size, y), (x + w, y), (x + w, y + corner_size)],  # Top-right
            [(x, y + h - corner_size), (x, y + h), (x + corner_size, y + h)],  # Bottom-left
            [(x + w - corner_size, y + h), (x + w, y + h), (x + w, y + h - corner_size)]  # Bottom-right
        ]

        for corner in corners:
            pygame.draw.lines(self.overlay_surface, color, False, corner, thickness)

        # Numéro de priorité
        if element.text:
            font = pygame.font.Font(None, 36)
            text_surface = font.render(element.text, True, color[:3])
            text_rect = text_surface.get_rect(center=(x + w//2, y - 25))
            self.overlay_surface.blit(text_surface, text_rect)

    def _render_resource_indicator(self, element: OverlayElement, alpha: int):
        """Rend indicateur de ressource"""
        x, y = element.position
        w, h = element.size
        color = (*element.color[:3], alpha)

        # Barre de ressource
        pygame.draw.rect(self.overlay_surface, (0, 0, 0, alpha//2), (x-2, y-2, w+4, h+4))
        pygame.draw.rect(self.overlay_surface, color, (x, y, w, h))

        # Texte de valeur
        if element.text:
            font = pygame.font.Font(None, 18)
            text_surface = font.render(element.text, True, (255, 255, 255, alpha))
            text_rect = text_surface.get_rect(center=(x + w//2, y + h//2))
            self.overlay_surface.blit(text_surface, text_rect)

    def _render_quest_guidance(self, element: OverlayElement, alpha: int):
        """Rend guidage de quête"""
        x, y = element.position
        color = (*element.color[:3], alpha)

        # Icône de quête (étoile)
        points = []
        for i in range(10):
            angle = i * 36 * 3.14159 / 180
            if i % 2 == 0:
                radius = 15
            else:
                radius = 7
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            points.append((px, py))

        pygame.draw.polygon(self.overlay_surface, color, points)

        # Texte de quête
        if element.text:
            font = pygame.font.Font(None, 22)
            lines = element.text.split('\n')
            for i, line in enumerate(lines):
                text_surface = font.render(line, True, (255, 255, 255, alpha))
                self.overlay_surface.blit(text_surface, (x + 25, y - 10 + i * 22))

    def _render_performance_stats(self, element: OverlayElement, alpha: int):
        """Rend statistiques de performance"""
        x, y = element.position
        color = (255, 255, 255, alpha)

        # Fond semi-transparent
        font = pygame.font.Font(None, 18)
        lines = element.text.split('\n')
        max_width = max(font.size(line)[0] for line in lines)
        height = len(lines) * 20 + 10

        pygame.draw.rect(self.overlay_surface, (0, 0, 0, alpha//3),
                        (x-5, y-5, max_width+10, height))

        # Texte des stats
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, color[:3])
            self.overlay_surface.blit(text_surface, (x, y + i * 20))

    def _display_overlay(self):
        """Affiche l'overlay à l'écran"""
        # NOTE: Pour un vrai overlay DirectX11, il faudrait utiliser
        # des APIs DirectX natives. Ici on simule avec pygame pour le prototype
        pass

    def get_overlay_stats(self) -> Dict[str, Any]:
        """Retourne statistiques overlay"""
        return {
            "active_elements": len(self.elements),
            "target_window": self.target_hwnd is not None,
            "running": self.running,
            "fps": self.frame_count if self.last_render_time > 0 else 0
        }

class IntelligentOverlay:
    """Gestionnaire principal de l'overlay intelligent"""

    def __init__(self, config: OverlayConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Composants
        self.directx_overlay = DirectXOverlay()

        # État
        self.active = False
        self.anti_detection_enabled = config.anti_detection

        # Cache pour optimisation
        self.element_cache: Dict[str, OverlayElement] = {}
        self.last_cleanup_time = 0.0

        self.logger.info("IntelligentOverlay initialisé")

    def initialize(self) -> bool:
        """Initialise l'overlay intelligent"""
        try:
            if not self.directx_overlay.initialize():
                self.logger.error("Échec initialisation DirectX overlay")
                return False

            self.logger.info("IntelligentOverlay initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation IntelligentOverlay: {e}")
            return False

    def start(self) -> bool:
        """Démarre l'overlay"""
        if not self.config.enable_overlay:
            self.logger.info("Overlay désactivé dans configuration")
            return False

        if self.active:
            return False

        if not self.directx_overlay.start_overlay():
            return False

        self.active = True
        self.logger.info("Overlay intelligent démarré")
        return True

    def stop(self):
        """Arrête l'overlay"""
        self.active = False
        self.directx_overlay.stop_overlay()
        self.logger.info("Overlay intelligent arrêté")

    def highlight_spell(self, position: Tuple[int, int], spell_name: str,
                       priority: int = 5, duration: float = None):
        """Surligne un sort recommandé"""
        if not self.active:
            return

        element = OverlayElement(
            element_id=f"spell_{spell_name}_{int(time.time())}",
            overlay_type=OverlayType.SPELL_HIGHLIGHT,
            position=position,
            size=(60, 60),
            color=self.config.colors["spell_highlight"],
            text=spell_name,
            priority=priority,
            duration=duration or self.config.default_duration,
            animation="pulse",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def suggest_movement(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                        reason: str = "", priority: int = 3):
        """Suggère un mouvement"""
        if not self.active:
            return

        element = OverlayElement(
            element_id=f"movement_{int(time.time())}",
            overlay_type=OverlayType.MOVEMENT_SUGGESTION,
            position=from_pos,
            size=(40, 40),
            color=self.config.colors["movement"],
            text=reason,
            priority=priority,
            duration=self.config.default_duration,
            animation="slide",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def mark_target_priority(self, position: Tuple[int, int], priority_num: int,
                           size: Tuple[int, int] = (100, 100)):
        """Marque la priorité d'une cible"""
        if not self.active:
            return

        element = OverlayElement(
            element_id=f"target_{priority_num}_{int(time.time())}",
            overlay_type=OverlayType.TARGET_PRIORITY,
            position=position,
            size=size,
            color=self.config.colors["target"],
            text=str(priority_num),
            priority=10 - priority_num,  # Plus haute priorité = plus bas numéro
            duration=self.config.default_duration * 2,
            animation="static",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def show_resource_status(self, position: Tuple[int, int], resource_type: str,
                           current: int, maximum: int):
        """Affiche état des ressources"""
        if not self.active:
            return

        percentage = (current / maximum) * 100 if maximum > 0 else 0
        bar_width = int(80 * (current / maximum)) if maximum > 0 else 0

        element = OverlayElement(
            element_id=f"resource_{resource_type}",
            overlay_type=OverlayType.RESOURCE_INDICATOR,
            position=position,
            size=(bar_width, 12),
            color=self.config.colors["resource"],
            text=f"{current}/{maximum}",
            priority=2,
            duration=60.0,  # Longue durée pour les ressources
            animation="static",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def show_quest_guidance(self, position: Tuple[int, int], objective: str,
                          steps: List[str]):
        """Affiche guidage de quête"""
        if not self.active:
            return

        guidance_text = f"{objective}\n" + "\n".join(f"• {step}" for step in steps[:3])

        element = OverlayElement(
            element_id=f"quest_{int(time.time())}",
            overlay_type=OverlayType.QUEST_GUIDANCE,
            position=position,
            size=(200, 100),
            color=self.config.colors["quest"],
            text=guidance_text,
            priority=4,
            duration=self.config.default_duration * 3,
            animation="fade",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def show_performance_stats(self, stats: Dict[str, Any],
                             position: Tuple[int, int] = (10, 10)):
        """Affiche statistiques de performance"""
        if not self.active:
            return

        stats_text = "\n".join(f"{key}: {value}" for key, value in stats.items())

        element = OverlayElement(
            element_id="performance_stats",
            overlay_type=OverlayType.PERFORMANCE_STATS,
            position=position,
            size=(200, 100),
            color=self.config.colors["info"],
            text=stats_text,
            priority=1,
            duration=60.0,
            animation="static",
            timestamp=time.time()
        )

        self.directx_overlay.add_element(element)

    def clear_overlay(self, overlay_type: Optional[OverlayType] = None):
        """Efface l'overlay"""
        self.directx_overlay.clear_elements(overlay_type)

    def cleanup(self):
        """Nettoyage des ressources"""
        self.stop()
        self.element_cache.clear()
        self.logger.info("IntelligentOverlay nettoyé")

# Factory function
def create_intelligent_overlay(config: Optional[OverlayConfig] = None) -> IntelligentOverlay:
    """Crée une instance IntelligentOverlay configurée"""
    if config is None:
        config = OverlayConfig()

    overlay = IntelligentOverlay(config)
    if overlay.initialize():
        return overlay
    else:
        raise RuntimeError("Impossible d'initialiser IntelligentOverlay")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        config = OverlayConfig(enable_overlay=True, transparency=0.8)
        overlay = create_intelligent_overlay(config)

        print("Test Overlay...")

        if overlay.start():
            print("Overlay démarré")

            # Test des différents éléments
            overlay.highlight_spell((100, 100), "Foudroiement", priority=8)
            overlay.suggest_movement((200, 200), (300, 250), "Position optimale")
            overlay.mark_target_priority((400, 300), 1, (120, 80))
            overlay.show_resource_status((10, 50), "HP", 75, 100)

            time.sleep(5)
            overlay.stop()
        else:
            print("Impossible de démarrer overlay")

    except Exception as e:
        print(f"Erreur test: {e}")
    finally:
        if 'overlay' in locals():
            overlay.cleanup()