"""
Module d'Intelligence Sociale pour l'IA DOFUS Évolutive
Phase 3 : Exécution Adaptative & Sociale

Ce module gère l'interaction sociale, la négociation,
et l'adaptation comportementale dans DOFUS.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import json
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    ALLY = "ally"
    NEUTRAL = "neutral"
    COMPETITOR = "competitor"
    ENEMY = "enemy"
    GUILD_MEMBER = "guild_member"
    FRIEND = "friend"
    UNKNOWN = "unknown"

class SocialContext(Enum):
    SOLO_PLAY = "solo_play"
    GROUP_DUNGEON = "group_dungeon"
    PVP_COMBAT = "pvp_combat"
    GUILD_ACTIVITY = "guild_activity"
    MARKET_TRADE = "market_trade"
    SOCIAL_CHAT = "social_chat"
    ALLIANCE_WAR = "alliance_war"

class CommunicationStyle(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    AGGRESSIVE = "aggressive"
    DIPLOMATIC = "diplomatic"

@dataclass
class PlayerProfile:
    """Profil d'un joueur observé"""
    name: str
    level: int = 0
    class_type: str = ""
    guild: str = ""
    last_seen: datetime = field(default_factory=datetime.now)
    relationship: RelationshipType = RelationshipType.UNKNOWN
    trust_score: float = 0.5  # 0-1
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    trading_history: List[Dict[str, Any]] = field(default_factory=list)
    social_influence: float = 0.0  # Impact social estimé

@dataclass
class SocialAction:
    """Action sociale à effectuer"""
    action_type: str
    target_player: Optional[str] = None
    message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    expected_outcome: str = ""
    context: SocialContext = SocialContext.SOLO_PLAY

@dataclass
class NegotiationStrategy:
    """Stratégie de négociation"""
    opening_offer: float
    minimum_acceptable: float
    maximum_offer: float
    concession_rate: float = 0.1
    time_pressure: float = 0.5
    relationship_weight: float = 0.3
    style: CommunicationStyle = CommunicationStyle.PROFESSIONAL

class SocialIntelligenceEngine:
    """Moteur principal d'intelligence sociale"""

    def __init__(self):
        self.known_players: Dict[str, PlayerProfile] = {}
        self.guild_members: Set[str] = set()
        self.current_context: SocialContext = SocialContext.SOLO_PLAY
        self.personality_traits = {
            "friendliness": 0.7,
            "assertiveness": 0.6,
            "trustworthiness": 0.8,
            "competitiveness": 0.5,
            "helpfulness": 0.7
        }

        # Configuration adaptative
        self.config = {
            "auto_respond_to_trades": True,
            "auto_group_invites": False,
            "auto_guild_activities": True,
            "social_learning_enabled": True,
            "reputation_tracking": True
        }

        # Métriques sociales
        self.social_metrics = {
            "successful_trades": 0,
            "failed_negotiations": 0,
            "reputation_score": 0.5,
            "social_connections": 0,
            "guild_contribution": 0.0
        }

    async def analyze_social_context(self, game_state: Dict[str, Any]) -> SocialContext:
        """Analyse le contexte social actuel"""
        try:
            # Détection du contexte basée sur l'état du jeu
            if game_state.get("in_combat") and game_state.get("pvp_mode"):
                return SocialContext.PVP_COMBAT
            elif game_state.get("in_group"):
                return SocialContext.GROUP_DUNGEON
            elif game_state.get("in_guild_area"):
                return SocialContext.GUILD_ACTIVITY
            elif game_state.get("market_interface_open"):
                return SocialContext.MARKET_TRADE
            elif game_state.get("chat_active"):
                return SocialContext.SOCIAL_CHAT
            else:
                return SocialContext.SOLO_PLAY

        except Exception as e:
            logger.error(f"Erreur analyse contexte social: {e}")
            return SocialContext.SOLO_PLAY

    async def update_player_profile(self, player_name: str, observations: Dict[str, Any]):
        """Met à jour le profil d'un joueur basé sur les observations"""
        try:
            if player_name not in self.known_players:
                self.known_players[player_name] = PlayerProfile(name=player_name)

            profile = self.known_players[player_name]
            profile.last_seen = datetime.now()

            # Mise à jour des informations observées
            if "level" in observations:
                profile.level = observations["level"]
            if "class" in observations:
                profile.class_type = observations["class"]
            if "guild" in observations:
                profile.guild = observations["guild"]

            # Analyse comportementale
            if "behavior" in observations:
                await self._analyze_behavior_patterns(profile, observations["behavior"])

            # Historique d'interaction
            if "interaction" in observations:
                profile.interaction_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": observations["interaction"]["type"],
                    "outcome": observations["interaction"].get("outcome", "unknown"),
                    "context": self.current_context.value
                })

            logger.info(f"Profil mis à jour pour {player_name}")

        except Exception as e:
            logger.error(f"Erreur mise à jour profil {player_name}: {e}")

    async def _analyze_behavior_patterns(self, profile: PlayerProfile, behavior_data: Dict[str, Any]):
        """Analyse les patterns comportementaux d'un joueur"""
        try:
            patterns = profile.behavioral_patterns

            # Analyse de l'agressivité
            if "combat_style" in behavior_data:
                patterns["aggression_level"] = behavior_data["combat_style"].get("aggression", 0.5)

            # Analyse de la coopération
            if "group_behavior" in behavior_data:
                patterns["cooperation_score"] = behavior_data["group_behavior"].get("helpful", 0.5)

            # Analyse des patterns de communication
            if "communication" in behavior_data:
                patterns["communication_frequency"] = behavior_data["communication"].get("frequency", 0.5)
                patterns["politeness_level"] = behavior_data["communication"].get("politeness", 0.5)

            # Mise à jour du score de confiance
            profile.trust_score = self._calculate_trust_score(profile)

        except Exception as e:
            logger.error(f"Erreur analyse comportementale: {e}")

    def _calculate_trust_score(self, profile: PlayerProfile) -> float:
        """Calcule le score de confiance d'un joueur"""
        try:
            base_score = 0.5

            # Facteurs positifs
            if profile.relationship == RelationshipType.ALLY:
                base_score += 0.3
            elif profile.relationship == RelationshipType.GUILD_MEMBER:
                base_score += 0.2
            elif profile.relationship == RelationshipType.FRIEND:
                base_score += 0.4

            # Historique d'interactions
            positive_interactions = sum(1 for interaction in profile.interaction_history
                                      if interaction.get("outcome") == "positive")
            total_interactions = len(profile.interaction_history)

            if total_interactions > 0:
                interaction_ratio = positive_interactions / total_interactions
                base_score = (base_score + interaction_ratio) / 2

            # Patterns comportementaux
            patterns = profile.behavioral_patterns
            if "cooperation_score" in patterns:
                base_score = (base_score + patterns["cooperation_score"]) / 2

            return max(0.0, min(1.0, base_score))

        except Exception as e:
            logger.error(f"Erreur calcul score confiance: {e}")
            return 0.5

    async def generate_social_action(self, context: SocialContext,
                                   available_players: List[str]) -> Optional[SocialAction]:
        """Génère une action sociale appropriée au contexte"""
        try:
            if context == SocialContext.MARKET_TRADE:
                return await self._generate_trade_action(available_players)
            elif context == SocialContext.GROUP_DUNGEON:
                return await self._generate_group_action(available_players)
            elif context == SocialContext.GUILD_ACTIVITY:
                return await self._generate_guild_action(available_players)
            elif context == SocialContext.SOCIAL_CHAT:
                return await self._generate_chat_action(available_players)
            else:
                return None

        except Exception as e:
            logger.error(f"Erreur génération action sociale: {e}")
            return None

    async def _generate_trade_action(self, available_players: List[str]) -> Optional[SocialAction]:
        """Génère une action de commerce"""
        try:
            # Trouver le meilleur partenaire commercial
            best_trader = None
            best_score = 0.0

            for player_name in available_players:
                if player_name in self.known_players:
                    profile = self.known_players[player_name]
                    trading_score = profile.trust_score * len(profile.trading_history)
                    if trading_score > best_score:
                        best_score = trading_score
                        best_trader = player_name

            if best_trader:
                return SocialAction(
                    action_type="initiate_trade",
                    target_player=best_trader,
                    message="Salut ! Tu veux faire du commerce ?",
                    priority=0.7,
                    expected_outcome="trade_agreement",
                    context=SocialContext.MARKET_TRADE
                )

            return None

        except Exception as e:
            logger.error(f"Erreur génération action commerce: {e}")
            return None

    async def _generate_group_action(self, available_players: List[str]) -> Optional[SocialAction]:
        """Génère une action de groupe/donjon"""
        try:
            # Chercher des alliés ou membres de guilde
            allies = [name for name in available_players
                     if name in self.known_players and
                     self.known_players[name].relationship in [RelationshipType.ALLY,
                                                              RelationshipType.GUILD_MEMBER]]

            if allies:
                target = allies[0]  # Prendre le premier allié disponible
                return SocialAction(
                    action_type="invite_to_group",
                    target_player=target,
                    message="Salut ! Tu veux faire un donjon ensemble ?",
                    priority=0.8,
                    expected_outcome="group_formation",
                    context=SocialContext.GROUP_DUNGEON
                )

            return None

        except Exception as e:
            logger.error(f"Erreur génération action groupe: {e}")
            return None

    async def _generate_guild_action(self, available_players: List[str]) -> Optional[SocialAction]:
        """Génère une action de guilde"""
        try:
            guild_members = [name for name in available_players if name in self.guild_members]

            if guild_members:
                return SocialAction(
                    action_type="guild_coordination",
                    target_player=guild_members[0],
                    message="Salut ! Des activités de guilde prévues ?",
                    priority=0.6,
                    expected_outcome="guild_coordination",
                    context=SocialContext.GUILD_ACTIVITY
                )

            return None

        except Exception as e:
            logger.error(f"Erreur génération action guilde: {e}")
            return None

    async def _generate_chat_action(self, available_players: List[str]) -> Optional[SocialAction]:
        """Génère une action de chat social"""
        try:
            # Interaction sociale basée sur la personnalité
            if self.personality_traits["friendliness"] > 0.6:
                return SocialAction(
                    action_type="social_greeting",
                    message="Salut tout le monde ! Comment ça va ?",
                    priority=0.4,
                    expected_outcome="social_bonding",
                    context=SocialContext.SOCIAL_CHAT
                )

            return None

        except Exception as e:
            logger.error(f"Erreur génération action chat: {e}")
            return None

class NegotiationEngine:
    """Moteur de négociation automatique"""

    def __init__(self, social_intelligence: SocialIntelligenceEngine):
        self.social_intelligence = social_intelligence
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}

    async def start_negotiation(self, player_name: str, item: str,
                              initial_offer: float) -> NegotiationStrategy:
        """Démarre une négociation avec un joueur"""
        try:
            profile = self.social_intelligence.known_players.get(player_name)

            # Stratégie basée sur la relation et l'historique
            if profile:
                trust_multiplier = 0.8 + (profile.trust_score * 0.4)

                strategy = NegotiationStrategy(
                    opening_offer=initial_offer,
                    minimum_acceptable=initial_offer * 0.7,
                    maximum_offer=initial_offer * trust_multiplier,
                    concession_rate=0.05 if profile.trust_score > 0.7 else 0.15,
                    relationship_weight=profile.trust_score
                )
            else:
                # Stratégie conservatrice pour joueurs inconnus
                strategy = NegotiationStrategy(
                    opening_offer=initial_offer,
                    minimum_acceptable=initial_offer * 0.8,
                    maximum_offer=initial_offer * 1.1,
                    concession_rate=0.1
                )

            # Enregistrer la négociation
            self.active_negotiations[player_name] = {
                "item": item,
                "strategy": strategy,
                "current_offer": strategy.opening_offer,
                "rounds": 0,
                "start_time": datetime.now()
            }

            logger.info(f"Négociation démarrée avec {player_name} pour {item}")
            return strategy

        except Exception as e:
            logger.error(f"Erreur démarrage négociation: {e}")
            return NegotiationStrategy(opening_offer=initial_offer,
                                     minimum_acceptable=initial_offer * 0.8,
                                     maximum_offer=initial_offer * 1.2)

    async def process_counter_offer(self, player_name: str,
                                  counter_offer: float) -> Tuple[bool, float]:
        """Traite une contre-offre et décide de la réponse"""
        try:
            if player_name not in self.active_negotiations:
                return False, 0.0

            negotiation = self.active_negotiations[player_name]
            strategy = negotiation["strategy"]

            # Accepter si l'offre est dans la fourchette acceptable
            if counter_offer >= strategy.minimum_acceptable:
                logger.info(f"Contre-offre acceptée: {counter_offer}")
                return True, counter_offer

            # Sinon, calculer une nouvelle offre
            current_offer = negotiation["current_offer"]
            new_offer = current_offer - (current_offer * strategy.concession_rate)

            # Vérifier si on peut encore négocier
            if new_offer < strategy.minimum_acceptable:
                logger.info(f"Négociation échouée avec {player_name}")
                del self.active_negotiations[player_name]
                return False, 0.0

            # Mettre à jour la négociation
            negotiation["current_offer"] = new_offer
            negotiation["rounds"] += 1

            logger.info(f"Nouvelle offre: {new_offer}")
            return False, new_offer

        except Exception as e:
            logger.error(f"Erreur traitement contre-offre: {e}")
            return False, 0.0

class MultiAgentCoordinator:
    """Coordinateur pour la gestion multi-comptes"""

    def __init__(self):
        self.managed_accounts: Dict[str, Dict[str, Any]] = {}
        self.coordination_strategies = {
            "resource_sharing": True,
            "synchronized_activities": True,
            "complementary_roles": True,
            "risk_distribution": True
        }

    async def register_account(self, account_id: str, capabilities: List[str]):
        """Enregistre un compte géré"""
        try:
            self.managed_accounts[account_id] = {
                "capabilities": capabilities,
                "current_activity": "idle",
                "last_coordination": datetime.now(),
                "performance_metrics": {
                    "efficiency": 0.5,
                    "reliability": 0.5,
                    "cooperation": 0.5
                }
            }
            logger.info(f"Compte {account_id} enregistré")

        except Exception as e:
            logger.error(f"Erreur enregistrement compte: {e}")

    async def coordinate_activities(self, target_objective: str) -> Dict[str, str]:
        """Coordonne les activités entre comptes pour un objectif"""
        try:
            coordination_plan = {}

            # Répartition des rôles selon les capacités
            for account_id, account_data in self.managed_accounts.items():
                capabilities = account_data["capabilities"]

                if target_objective == "dungeon_farming":
                    if "tank" in capabilities:
                        coordination_plan[account_id] = "lead_group"
                    elif "healer" in capabilities:
                        coordination_plan[account_id] = "support_group"
                    else:
                        coordination_plan[account_id] = "damage_dealer"

                elif target_objective == "resource_gathering":
                    if "farming" in capabilities:
                        coordination_plan[account_id] = "gather_resources"
                    elif "crafting" in capabilities:
                        coordination_plan[account_id] = "process_materials"
                    else:
                        coordination_plan[account_id] = "market_trading"

            logger.info(f"Plan de coordination créé: {coordination_plan}")
            return coordination_plan

        except Exception as e:
            logger.error(f"Erreur coordination activités: {e}")
            return {}

# Intégration principale
class SocialModule:
    """Module principal d'intelligence sociale"""

    def __init__(self):
        self.social_engine = SocialIntelligenceEngine()
        self.negotiation_engine = NegotiationEngine(self.social_engine)
        self.multi_agent_coordinator = MultiAgentCoordinator()
        self.is_active = False

    async def initialize(self):
        """Initialise le module social"""
        try:
            self.is_active = True
            logger.info("Module d'intelligence sociale initialisé")

        except Exception as e:
            logger.error(f"Erreur initialisation module social: {e}")

    async def process_social_frame(self, game_state: Dict[str, Any]) -> List[SocialAction]:
        """Traite un frame de données sociales"""
        try:
            if not self.is_active:
                return []

            # Analyser le contexte social
            context = await self.social_engine.analyze_social_context(game_state)
            self.social_engine.current_context = context

            # Mettre à jour les profils des joueurs observés
            if "observed_players" in game_state:
                for player_data in game_state["observed_players"]:
                    await self.social_engine.update_player_profile(
                        player_data["name"],
                        player_data
                    )

            # Générer des actions sociales
            actions = []
            available_players = game_state.get("available_players", [])

            if available_players:
                action = await self.social_engine.generate_social_action(
                    context, available_players
                )
                if action:
                    actions.append(action)

            return actions

        except Exception as e:
            logger.error(f"Erreur traitement frame sociale: {e}")
            return []

    async def get_social_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques sociales actuelles"""
        try:
            return {
                "known_players": len(self.social_engine.known_players),
                "guild_members": len(self.social_engine.guild_members),
                "reputation_score": self.social_engine.social_metrics["reputation_score"],
                "successful_trades": self.social_engine.social_metrics["successful_trades"],
                "active_negotiations": len(self.negotiation_engine.active_negotiations),
                "managed_accounts": len(self.multi_agent_coordinator.managed_accounts)
            }

        except Exception as e:
            logger.error(f"Erreur récupération métriques sociales: {e}")
            return {}