"""
DOFUS VISION 2025 - CORE MODULES
Modules principaux du système d'IA pour DOFUS Unity World Model
"""

__version__ = "2025.1.0"
__author__ = "Claude Code - Project Maintenance Specialist"

# Imports principaux pour faciliter l'utilisation
try:
    from .vision_engine.combat_grid_analyzer import DofusCombatGridAnalyzer
    from .vision_engine.screenshot_capture import DofusWindowCapture
    from .vision_engine.unity_interface_reader import DofusUnityInterfaceReader

    from .knowledge_base.knowledge_integration import DofusKnowledgeBase
    from .learning_engine.adaptive_learning_engine import AdaptiveLearningEngine
    from .human_simulation.advanced_human_simulation import AdvancedHumanSimulator

    # HRM Integration temporairement désactivé (problèmes de dépendances)
    # from .world_model.hrm_dofus_integration import DofusIntelligentDecisionMaker

    __all__ = [
        'DofusCombatGridAnalyzer',
        'DofusWindowCapture',
        'DofusUnityInterfaceReader',
        'DofusKnowledgeBase',
        'AdaptiveLearningEngine',
        'AdvancedHumanSimulator'
        # 'DofusIntelligentDecisionMaker'  # Temporairement désactivé
    ]

except ImportError as e:
    print(f"Erreur d'import dans core: {e}")
    __all__ = []