"""
Engine stub - Pour compatibilité avec ancien code
"""

from .module_interface import IAnalysisModule, ModuleStatus
from .event_bus import EventType, EventPriority

__all__ = ['IAnalysisModule', 'ModuleStatus', 'EventType', 'EventPriority']
