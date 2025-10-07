"""
Configuration module pour DOFUS AlphaStar 2025
"""

from .alphastar_config import (
    MasterConfig,
    config,
    get_device,
    get_dtype,
    update_config,
    save_config,
    load_config
)

__all__ = [
    "MasterConfig",
    "config",
    "get_device",
    "get_dtype",
    "update_config",
    "save_config",
    "load_config"
]