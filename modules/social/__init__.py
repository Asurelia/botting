"""
Module social - Gestion des interactions sociales du bot
Inclut le chat, la guilde et les groupes avec intelligence artificielle
"""

from .chat_manager import ChatManager, MessageType, ResponseType
from .guild_manager import GuildManager, GuildRank, EventType, PerceptorStatus
from .group_manager import GroupManager, GroupRole, GroupFormation, CombatPhase

__all__ = [
    'ChatManager', 'MessageType', 'ResponseType',
    'GuildManager', 'GuildRank', 'EventType', 'PerceptorStatus',
    'GroupManager', 'GroupRole', 'GroupFormation', 'CombatPhase'
]