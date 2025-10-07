"""
Calibration Module - Auto-découverte et configuration Dofus
Scanne automatiquement l'interface, les raccourcis et les éléments du jeu
"""

from .dofus_calibrator import (
    DofusCalibrator,
    CalibrationResult,
    create_calibrator
)

__all__ = [
    "DofusCalibrator",
    "CalibrationResult",
    "create_calibrator"
]