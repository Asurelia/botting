"""
Module d'automatisation des chasses aux trésors DOFUS
Système complet avec IA, navigation automatique et interface GUI

Composants principaux:
- HintDatabase: Base de données d'indices avec reconnaissance visuelle
- TreasureSolver: Solveur intelligent d'indices avec IA
- MapNavigator: Navigation automatique entre étapes
- TreasureHuntAutomation: Automatisation complète des chasses
- TreasureHuntGUI: Interface graphique avec monitoring

Auteur: DofuBot System
Version: 1.0.0
"""

from .hint_database import (
    HintDatabase,
    HintData,
    HintType,
    HintDifficulty
)

from .treasure_solver import (
    TreasureSolver,
    SolutionCandidate,
    SolutionType,
    PatternMatcher,
    VisualAnalyzer
)

from .map_navigator import (
    MapNavigator,
    MapPosition,
    NavigationState,
    NavigationStep,
    NavigationPath,
    MovementType
)

from .treasure_automation import (
    TreasureHuntAutomation,
    TreasureHuntState,
    TreasureHuntType,
    TreasureHuntSession,
    TreasureHuntStep
)

from .treasure_gui import (
    TreasureHuntGUI
)

__version__ = "1.0.0"
__author__ = "DofuBot System"

__all__ = [
    # Base de données
    'HintDatabase',
    'HintData', 
    'HintType',
    'HintDifficulty',
    
    # Solveur
    'TreasureSolver',
    'SolutionCandidate',
    'SolutionType',
    'PatternMatcher',
    'VisualAnalyzer',
    
    # Navigation
    'MapNavigator',
    'MapPosition',
    'NavigationState', 
    'NavigationStep',
    'NavigationPath',
    'MovementType',
    
    # Automatisation
    'TreasureHuntAutomation',
    'TreasureHuntState',
    'TreasureHuntType',
    'TreasureHuntSession',
    'TreasureHuntStep',
    
    # Interface graphique
    'TreasureHuntGUI'
]


def create_treasure_hunt_system(click_handler, screen_capture_handler):
    """
    Factory function pour créer un système complet de chasse aux trésors
    
    Args:
        click_handler: Function(x, y) pour effectuer des clics
        screen_capture_handler: Function() -> np.ndarray pour capturer l'écran
    
    Returns:
        TreasureHuntAutomation: Système d'automatisation prêt à l'emploi
    """
    return TreasureHuntAutomation(click_handler, screen_capture_handler)


def create_treasure_hunt_gui(automation_system):
    """
    Factory function pour créer l'interface graphique
    
    Args:
        automation_system: Instance de TreasureHuntAutomation
    
    Returns:
        TreasureHuntGUI: Interface graphique prête à l'emploi
    """
    return TreasureHuntGUI(automation_system)


# Configuration par défaut du module
DEFAULT_CONFIG = {
    'database_path': 'treasure_hunt_hints.db',
    'session_database_path': 'treasure_hunt_sessions.db',
    'screenshots_path': 'screenshots/treasure_hunts/',
    'max_attempts_per_step': 3,
    'step_timeout': 300,
    'combat_timeout': 180,
    'auto_fight': True,
    'auto_collect_rewards': True,
    'save_screenshots': True,
    'learning_mode': True,
    'debug_mode': False
}

# Métadonnées du module
MODULE_INFO = {
    'name': 'DOFUS Treasure Hunt Automation',
    'description': 'Système complet d\'automatisation des chasses aux trésors DOFUS',
    'version': __version__,
    'author': __author__,
    'features': [
        'Base de données complète d\'indices',
        'Reconnaissance visuelle d\'indices',
        'Solveur intelligent avec IA',
        'Navigation automatique optimisée',
        'Automatisation complète multi-chasses',
        'Interface GUI avec monitoring temps réel',
        'Statistiques détaillées et historique',
        'Import/export de bases communautaires',
        'Mode apprentissage pour nouveaux indices'
    ],
    'requirements': [
        'opencv-python>=4.5.0',
        'numpy>=1.20.0',
        'pillow>=8.0.0',
        'matplotlib>=3.3.0',
        'sqlite3 (built-in)',
        'tkinter (built-in)',
        'threading (built-in)'
    ]
}