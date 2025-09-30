"""
Environment module - Interface DOFUS Unity pour RL
Compatible avec OpenAI Gym et RLlib
"""

from .dofus_env import (
    DofusUnityEnvironment,
    DofusEnvConfig,
    create_dofus_env
)

from .env_wrappers import (
    DofusObservationWrapper,
    DofusActionWrapper,
    DofusRewardWrapper,
    DofusFrameStack
)

__all__ = [
    "DofusUnityEnvironment",
    "DofusEnvConfig",
    "create_dofus_env",
    "DofusObservationWrapper",
    "DofusActionWrapper",
    "DofusRewardWrapper",
    "DofusFrameStack"
]