"""
Networks Module - Architectures de réseaux de neurones pour AlphaStar DOFUS
Transformer + LSTM + Pointer Networks + HRM Integration
"""

from .alphastar_model import (
    AlphaStarModel,
    AlphaStarHRMModel,
    create_alphastar_model
)

from .transformer_core import (
    TransformerEncoder,
    SpatialTransformer,
    EntityTransformer
)

from .lstm_core import (
    AdvancedLSTM,
    BidirectionalLSTM,
    StackedLSTM
)

from .pointer_network import (
    PointerNetwork,
    AttentionPointer,
    SpatialPointer
)

from .value_networks import (
    ValueNetwork,
    AdvantageNetwork,
    QNetwork
)

__all__ = [
    "AlphaStarModel",
    "AlphaStarHRMModel",
    "create_alphastar_model",
    "TransformerEncoder",
    "SpatialTransformer",
    "EntityTransformer",
    "AdvancedLSTM",
    "BidirectionalLSTM",
    "StackedLSTM",
    "PointerNetwork",
    "AttentionPointer",
    "SpatialPointer",
    "ValueNetwork",
    "AdvantageNetwork",
    "QNetwork"
]