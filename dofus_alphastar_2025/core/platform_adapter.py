"""
Platform Adapter - Support cross-platform pour Windows et Linux
Gère les différences de capture écran et détection de fenêtre

Author: Claude Code
Date: 2025-10-06
"""

import platform
import subprocess
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Informations sur une fenêtre applicative"""
    id: str  # HWND (Windows) ou window ID (Linux)
    title: str
    x: int
    y: int
    width: int
    height: int
    is_visible: bool


class PlatformAdapter:
    """
    Adaptateur cross-platform pour fonctions dépendantes de l'OS

    Usage:
        adapter = PlatformAdapter()
        if adapter.is_linux():
            window = adapter.find_window_linux("Dofus")
        else:
            window = adapter.find_window_windows("Dofus")
    """

    @staticmethod
    def get_system() -> str:
        """Retourne système d'exploitation"""
        return platform.system()  # 'Windows', 'Linux', 'Darwin'

    @staticmethod
    def is_windows() -> bool:
        """Vérifie si Windows"""
        return platform.system() == "Windows"

    @staticmethod
    def is_linux() -> bool:
        """Vérifie si Linux"""
        return platform.system() == "Linux"

    @staticmethod
    def is_macos() -> bool:
        """Vérifie si macOS"""
        return platform.system() == "Darwin"

    # ===== LINUX METHODS =====

    @staticmethod
    def find_window_linux(title_pattern: str) -> Optional[WindowInfo]:
        """
        Trouve fenêtre sur Linux via xdotool

        Args:
            title_pattern: Pattern du titre (ex: "Dofus")

        Returns:
            WindowInfo si trouvé, None sinon
        """
        try:
            # Chercher fenêtres avec xdotool
            result = subprocess.run(
                ['xdotool', 'search', '--name', title_pattern],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0 or not result.stdout.strip():
                logger.warning(f"Aucune fenêtre trouvée pour '{title_pattern}'")
                return None

            # Prendre première fenêtre trouvée
            window_id = result.stdout.strip().split('\n')[0]

            # Obtenir infos géométrie
            geom_result = subprocess.run(
                ['xdotool', 'getwindowgeometry', '--shell', window_id],
                capture_output=True,
                text=True,
                check=True
            )

            # Parser output
            geom = {}
            for line in geom_result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    geom[key] = value

            # Obtenir titre
            title_result = subprocess.run(
                ['xdotool', 'getwindowname', window_id],
                capture_output=True,
                text=True,
                check=True
            )

            return WindowInfo(
                id=window_id,
                title=title_result.stdout.strip(),
                x=int(geom.get('X', 0)),
                y=int(geom.get('Y', 0)),
                width=int(geom.get('WIDTH', 0)),
                height=int(geom.get('HEIGHT', 0)),
                is_visible=True
            )

        except FileNotFoundError:
            logger.error("xdotool non trouvé - Installer avec: sudo apt install xdotool")
            return None
        except Exception as e:
            logger.error(f"Erreur recherche fenêtre Linux: {e}")
            return None

    # ===== GENERIC METHODS =====

    @classmethod
    def find_window(cls, title_pattern: str) -> Optional[WindowInfo]:
        """
        Trouve fenêtre (auto-détection OS)

        Args:
            title_pattern: Pattern du titre

        Returns:
            WindowInfo si trouvé, None sinon
        """
        if cls.is_linux():
            return cls.find_window_linux(title_pattern)
        else:
            logger.error(f"OS non supporté: {cls.get_system()}")
            return None


if __name__ == "__main__":
    print(f"🖥️  OS: {PlatformAdapter.get_system()}")
