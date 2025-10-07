"""
Vision Engine V2 - Vision avancée pour DOFUS AlphaStar
SAM 2 + TrOCR + Analyse contextuelle optimisée AMD + Realtime Vision MVP
"""

# Realtime Vision (MVP - fonctionnel)
from .realtime_vision import (
    RealtimeVision,
    create_realtime_vision
)

# Complete Vision Adapter
from .vision_complete_adapter import (
    VisionCompleteAdapter,
    create_vision_complete_adapter
)

# TrOCR Integration
from .trocr_integration import (
    TextDetection,
    TrOCRProcessor,
    DofusTextRecognizer
)

# SAM Integration
from .sam_integration import (
    SAMSegment,
    SAMProcessor
)

# Alias pour compatibilité
create_vision_engine = create_vision_complete_adapter

__all__ = [
    "RealtimeVision",
    "create_realtime_vision",
    "VisionCompleteAdapter",
    "create_vision_complete_adapter",
    "create_vision_engine",
    "TextDetection",
    "TrOCRProcessor",
    "DofusTextRecognizer",
    "SAMSegment",
    "SAMProcessor"
]