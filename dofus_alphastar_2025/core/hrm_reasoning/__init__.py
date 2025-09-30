"""
HRM Reasoning Module - Raisonnement hiérarchique pour DOFUS AlphaStar
Intégration du Hierarchical Reasoning Model avec optimisations AMD
"""

from .hrm_amd_core import (
    HRMAMDModel,
    HRMSystemOne,
    HRMSystemTwo,
    AMDDeviceManager,
    AMDOptimizationConfig
)

from .hrm_dofus_adapter import (
    DofusHRMAgent,
    DofusGameState,
    DofusAction,
    HRMDecisionMaker
)

__all__ = [
    "HRMAMDModel",
    "HRMSystemOne",
    "HRMSystemTwo",
    "AMDDeviceManager",
    "AMDOptimizationConfig",
    "DofusHRMAgent",
    "DofusGameState",
    "DofusAction",
    "HRMDecisionMaker"
]