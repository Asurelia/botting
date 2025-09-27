"""
Emotional State Management System
Système de simulation d'états émotionnels et de personnalité pour l'IA DOFUS
Implémente humeur, motivation, tolérance au risque et réponses comportementales
"""

import numpy as np
import asyncio
import logging
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import json

# Import modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Types d'émotions de base"""
    # Émotions positives
    JOY = "joy"
    SATISFACTION = "satisfaction"
    EXCITEMENT = "excitement"
    CONFIDENCE = "confidence"
    CURIOSITY = "curiosity"

    # Émotions négatives
    FRUSTRATION = "frustration"
    BOREDOM = "boredom"
    ANXIETY = "anxiety"
    DISAPPOINTMENT = "disappointment"
    IMPATIENCE = "impatience"

    # Émotions neutres
    CALM = "calm"
    FOCUSED = "focused"
    NEUTRAL = "neutral"

class MoodState(Enum):
    """États d'humeur globaux"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class PersonalityTrait(Enum):
    """Traits de personnalité"""
    # Big Five
    OPENNESS = "openness"          # Ouverture à l'expérience
    CONSCIENTIOUSNESS = "conscientiousness"  # Conscienciosité
    EXTRAVERSION = "extraversion"   # Extraversion
    AGREEABLENESS = "agreeableness" # Agréabilité
    NEUROTICISM = "neuroticism"     # Névrosisme

    # Traits spécifiques au jeu
    RISK_TAKING = "risk_taking"     # Prise de risque
    COMPETITIVENESS = "competitiveness"  # Compétitivité
    PATIENCE = "patience"           # Patience
    CURIOSITY = "curiosity"         # Curiosité
    SOCIAL_TENDENCY = "social_tendency"  # Tendance sociale

class GameEvent(Enum):
    """Événements de jeu affectant l'état émotionnel"""
    # Événements positifs
    LEVEL_UP = "level_up"
    RARE_DROP = "rare_drop"
    QUEST_COMPLETED = "quest_completed"
    ACHIEVEMENT_UNLOCKED = "achievement_unlocked"
    SUCCESSFUL_TRADE = "successful_trade"
    TEAM_VICTORY = "team_victory"

    # Événements négatifs
    DEATH = "death"
    ITEM_LOST = "item_lost"
    QUEST_FAILED = "quest_failed"
    SCAM_ATTEMPT = "scam_attempt"
    TEAM_DEFEAT = "team_defeat"
    SERVER_LAG = "server_lag"

    # Événements neutres
    RESOURCE_GATHERED = "resource_gathered"
    TRAVEL_COMPLETED = "travel_completed"
    SHOP_VISIT = "shop_visit"

@dataclass
class Emotion:
    """Émotion individuelle avec intensité et durée"""
    type: EmotionType
    intensity: float  # 0.0 à 1.0
    duration: timedelta
    created_at: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1  # Taux de décroissance par minute

    def current_intensity(self) -> float:
        """Calcule l'intensité actuelle avec décroissance temporelle"""
        elapsed = datetime.now() - self.created_at
        elapsed_minutes = elapsed.total_seconds() / 60.0

        # Décroissance exponentielle
        current = self.intensity * math.exp(-self.decay_rate * elapsed_minutes)
        return max(0.0, current)

    def is_active(self) -> bool:
        """Vérifie si l'émotion est encore active"""
        return self.current_intensity() > 0.01

@dataclass
class PersonalityProfile:
    """Profil de personnalité complet"""
    # Traits Big Five (0.0 à 1.0)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Traits spécifiques jeu
    risk_taking: float = 0.5
    competitiveness: float = 0.5
    patience: float = 0.5
    curiosity: float = 0.5
    social_tendency: float = 0.5

    # Préférences comportementales
    preferred_activities: List[str] = field(default_factory=list)
    avoided_activities: List[str] = field(default_factory=list)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_trait_value(self, trait: PersonalityTrait) -> float:
        """Récupère la valeur d'un trait spécifique"""
        trait_map = {
            PersonalityTrait.OPENNESS: self.openness,
            PersonalityTrait.CONSCIENTIOUSNESS: self.conscientiousness,
            PersonalityTrait.EXTRAVERSION: self.extraversion,
            PersonalityTrait.AGREEABLENESS: self.agreeableness,
            PersonalityTrait.NEUROTICISM: self.neuroticism,
            PersonalityTrait.RISK_TAKING: self.risk_taking,
            PersonalityTrait.COMPETITIVENESS: self.competitiveness,
            PersonalityTrait.PATIENCE: self.patience,
            PersonalityTrait.CURIOSITY: self.curiosity,
            PersonalityTrait.SOCIAL_TENDENCY: self.social_tendency
        }
        return trait_map.get(trait, 0.5)

    def evolve_trait(self, trait: PersonalityTrait, change: float, reason: str = ""):
        """Fait évoluer un trait de personnalité"""
        current_value = self.get_trait_value(trait)
        new_value = max(0.0, min(1.0, current_value + change))

        # Application du changement
        if trait == PersonalityTrait.OPENNESS:
            self.openness = new_value
        elif trait == PersonalityTrait.CONSCIENTIOUSNESS:
            self.conscientiousness = new_value
        elif trait == PersonalityTrait.EXTRAVERSION:
            self.extraversion = new_value
        elif trait == PersonalityTrait.AGREEABLENESS:
            self.agreeableness = new_value
        elif trait == PersonalityTrait.NEUROTICISM:
            self.neuroticism = new_value
        elif trait == PersonalityTrait.RISK_TAKING:
            self.risk_taking = new_value
        elif trait == PersonalityTrait.COMPETITIVENESS:
            self.competitiveness = new_value
        elif trait == PersonalityTrait.PATIENCE:
            self.patience = new_value
        elif trait == PersonalityTrait.CURIOSITY:
            self.curiosity = new_value
        elif trait == PersonalityTrait.SOCIAL_TENDENCY:
            self.social_tendency = new_value

        # Enregistrement de l'évolution
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'trait': trait.value,
            'old_value': current_value,
            'new_value': new_value,
            'change': change,
            'reason': reason
        })

@dataclass
class RiskProfile:
    """Profil de tolérance au risque"""
    base_tolerance: float = 0.5  # Tolérance de base (0.0 = très conservateur, 1.0 = très risqué)
    mood_adjustment: float = 0.0  # Ajustement selon l'humeur
    situational_adjustment: float = 0.0  # Ajustement selon la situation

    # Contextes spécifiques
    financial_risk: float = 0.5
    combat_risk: float = 0.5
    social_risk: float = 0.5
    exploration_risk: float = 0.5

    def current_tolerance(self) -> float:
        """Calcule la tolérance actuelle au risque"""
        adjusted = self.base_tolerance + self.mood_adjustment + self.situational_adjustment
        return max(0.0, min(1.0, adjusted))

    def get_context_tolerance(self, context: str) -> float:
        """Récupère la tolérance pour un contexte spécifique"""
        context_map = {
            'financial': self.financial_risk,
            'combat': self.combat_risk,
            'social': self.social_risk,
            'exploration': self.exploration_risk
        }
        return context_map.get(context, self.current_tolerance())

@dataclass
class PersonalityResponse:
    """Réponse comportementale selon la personnalité"""
    response_type: str
    intensity: float  # Intensité de la réponse
    duration: timedelta  # Durée de la réponse
    actions: List[str] = field(default_factory=list)  # Actions suggérées
    dialogue: Optional[str] = None  # Dialogue généré
    behavior_changes: Dict[str, float] = field(default_factory=dict)  # Changements comportementaux

@dataclass
class EmotionalMemory:
    """Mémoire émotionnelle d'un événement"""
    event_type: GameEvent
    emotional_impact: float  # Impact émotionnel (-1.0 à 1.0)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Apprentissage
    learning_value: float = 0.0  # Valeur d'apprentissage de l'événement
    pattern_recognition: Dict[str, Any] = field(default_factory=dict)

class EmotionalStateManager:
    """
    Gestionnaire principal des états émotionnels et de la personnalité
    Simule humeur, motivation et comportements adaptatifs
    """

    def __init__(self, initial_personality: Optional[PersonalityProfile] = None):
        self.logger = logging.getLogger(__name__)

        # État émotionnel actuel
        self.active_emotions: List[Emotion] = []
        self.current_mood: MoodState = MoodState.NEUTRAL
        self.mood_stability: float = 0.8  # Stabilité de l'humeur (0.0 = très instable)

        # Profil de personnalité
        self.personality = initial_personality or self._generate_random_personality()
        self.risk_profile = RiskProfile()

        # Historique et apprentissage
        self.emotional_memory: deque = deque(maxlen=1000)
        self.behavior_patterns: Dict[str, Any] = {}
        self.adaptation_learning: Dict[str, float] = defaultdict(float)

        # Métriques de performance
        self.mood_history: deque = deque(maxlen=100)
        self.decision_impacts: List[Dict[str, Any]] = []

        # Configuration des réactions émotionnelles
        self.emotion_triggers = self._initialize_emotion_triggers()
        self.personality_responses = self._initialize_personality_responses()

    def simulate_player_mood(self) -> MoodState:
        """Simule l'humeur actuelle du joueur basée sur les émotions"""
        try:
            # Nettoyage des émotions expirées
            self.active_emotions = [e for e in self.active_emotions if e.is_active()]

            if not self.active_emotions:
                return MoodState.NEUTRAL

            # Calcul du score émotionnel global
            positive_score = 0.0
            negative_score = 0.0

            for emotion in self.active_emotions:
                intensity = emotion.current_intensity()

                if emotion.type in [EmotionType.JOY, EmotionType.SATISFACTION,
                                  EmotionType.EXCITEMENT, EmotionType.CONFIDENCE,
                                  EmotionType.CURIOSITY]:
                    positive_score += intensity
                elif emotion.type in [EmotionType.FRUSTRATION, EmotionType.BOREDOM,
                                    EmotionType.ANXIETY, EmotionType.DISAPPOINTMENT,
                                    EmotionType.IMPATIENCE]:
                    negative_score += intensity

            # Détermination de l'humeur
            net_score = positive_score - negative_score

            # Ajustement par stabilité de l'humeur
            if abs(net_score) < (1.0 - self.mood_stability):
                self.current_mood = MoodState.NEUTRAL
            elif net_score >= 0.6:
                self.current_mood = MoodState.VERY_POSITIVE
            elif net_score >= 0.2:
                self.current_mood = MoodState.POSITIVE
            elif net_score <= -0.6:
                self.current_mood = MoodState.VERY_NEGATIVE
            elif net_score <= -0.2:
                self.current_mood = MoodState.NEGATIVE
            else:
                self.current_mood = MoodState.NEUTRAL

            # Enregistrement dans l'historique
            self.mood_history.append({
                'timestamp': datetime.now(),
                'mood': self.current_mood,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'net_score': net_score
            })

            return self.current_mood

        except Exception as e:
            self.logger.error(f"Erreur simulation humeur: {e}")
            return MoodState.NEUTRAL

    def adjust_risk_tolerance(self, mood: MoodState) -> RiskProfile:
        """Ajuste la tolérance au risque selon l'humeur"""
        try:
            # Réinitialisation des ajustements
            self.risk_profile.mood_adjustment = 0.0

            # Ajustements selon l'humeur
            mood_adjustments = {
                MoodState.VERY_POSITIVE: 0.3,    # Plus de prise de risque
                MoodState.POSITIVE: 0.15,
                MoodState.NEUTRAL: 0.0,
                MoodState.NEGATIVE: -0.15,       # Moins de prise de risque
                MoodState.VERY_NEGATIVE: -0.3
            }

            self.risk_profile.mood_adjustment = mood_adjustments.get(mood, 0.0)

            # Ajustement par traits de personnalité
            neuroticism_penalty = self.personality.neuroticism * 0.2
            risk_taking_bonus = self.personality.risk_taking * 0.1

            self.risk_profile.mood_adjustment += risk_taking_bonus - neuroticism_penalty

            # Ajustements contextuels basés sur l'expérience
            self._adjust_contextual_risk_tolerance()

            return self.risk_profile

        except Exception as e:
            self.logger.error(f"Erreur ajustement tolérance risque: {e}")
            return self.risk_profile

    def generate_personality_response(self, situation: Dict[str, Any]) -> PersonalityResponse:
        """Génère une réponse comportementale selon la personnalité"""
        try:
            situation_type = situation.get('type', 'unknown')
            context = situation.get('context', {})

            # Sélection du type de réponse basé sur la personnalité
            response_type = self._determine_response_type(situation_type, context)

            # Calcul de l'intensité de la réponse
            intensity = self._calculate_response_intensity(situation_type, context)

            # Génération d'actions suggérées
            actions = self._generate_response_actions(situation_type, response_type, context)

            # Génération de dialogue si approprié
            dialogue = self._generate_dialogue(situation_type, response_type, context)

            # Changements comportementaux temporaires
            behavior_changes = self._calculate_behavior_changes(response_type, intensity)

            response = PersonalityResponse(
                response_type=response_type,
                intensity=intensity,
                duration=timedelta(minutes=max(5, int(intensity * 30))),
                actions=actions,
                dialogue=dialogue,
                behavior_changes=behavior_changes
            )

            return response

        except Exception as e:
            self.logger.error(f"Erreur génération réponse personnalité: {e}")
            return PersonalityResponse(
                response_type="neutral",
                intensity=0.5,
                duration=timedelta(minutes=5)
            )

    async def process_game_event(self, event: GameEvent, context: Dict[str, Any] = None) -> bool:
        """Traite un événement de jeu et met à jour l'état émotionnel"""
        try:
            context = context or {}

            # Détermination de l'impact émotionnel
            emotional_impact = self._calculate_emotional_impact(event, context)

            # Génération d'émotions
            triggered_emotions = self._generate_emotions_from_event(event, emotional_impact)

            # Ajout des nouvelles émotions
            self.active_emotions.extend(triggered_emotions)

            # Création de la mémoire émotionnelle
            memory = EmotionalMemory(
                event_type=event,
                emotional_impact=emotional_impact,
                context=context,
                learning_value=abs(emotional_impact)
            )

            self.emotional_memory.append(memory)

            # Apprentissage et adaptation
            await self._learn_from_event(event, emotional_impact, context)

            # Mise à jour de l'humeur
            self.simulate_player_mood()

            self.logger.debug(f"Événement {event.value} traité, impact: {emotional_impact:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur traitement événement {event}: {e}")
            return False

    async def evolve_personality(self, experiences: List[Dict[str, Any]]) -> bool:
        """Fait évoluer la personnalité basée sur les expériences"""
        try:
            for experience in experiences:
                event_type = experience.get('event_type')
                outcome = experience.get('outcome', 'neutral')
                context = experience.get('context', {})

                # Évolution basée sur le type d'événement et l'outcome
                trait_changes = self._calculate_personality_evolution(event_type, outcome, context)

                for trait, change in trait_changes.items():
                    if isinstance(trait, PersonalityTrait) and abs(change) > 0.01:
                        self.personality.evolve_trait(
                            trait, change, f"Expérience: {event_type} -> {outcome}"
                        )

            return True

        except Exception as e:
            self.logger.error(f"Erreur évolution personnalité: {e}")
            return False

    def get_current_motivation(self) -> Dict[str, float]:
        """Calcule la motivation actuelle pour différentes activités"""
        try:
            # Motivation de base selon la personnalité
            base_motivations = {
                'exploration': self.personality.curiosity * 0.5 + self.personality.openness * 0.3,
                'social': self.personality.social_tendency * 0.6 + self.personality.extraversion * 0.4,
                'achievement': self.personality.competitiveness * 0.5 + self.personality.conscientiousness * 0.3,
                'relaxation': self.personality.patience * 0.4 + (1.0 - self.personality.neuroticism) * 0.4,
                'risk_taking': self.personality.risk_taking * 0.6 + self.risk_profile.current_tolerance() * 0.4
            }

            # Ajustements selon l'humeur
            mood_multipliers = {
                MoodState.VERY_POSITIVE: 1.3,
                MoodState.POSITIVE: 1.1,
                MoodState.NEUTRAL: 1.0,
                MoodState.NEGATIVE: 0.8,
                MoodState.VERY_NEGATIVE: 0.6
            }

            mood_multiplier = mood_multipliers.get(self.current_mood, 1.0)

            # Application des ajustements
            adjusted_motivations = {
                activity: min(1.0, motivation * mood_multiplier)
                for activity, motivation in base_motivations.items()
            }

            return adjusted_motivations

        except Exception as e:
            self.logger.error(f"Erreur calcul motivation: {e}")
            return {'exploration': 0.5, 'social': 0.5, 'achievement': 0.5, 'relaxation': 0.5, 'risk_taking': 0.5}

    def _generate_random_personality(self) -> PersonalityProfile:
        """Génère un profil de personnalité aléatoire mais cohérent"""
        # Génération avec distribution normale centrée sur 0.5
        traits = {}
        for trait in PersonalityTrait:
            # Distribution normale avec écart-type 0.15
            value = np.random.normal(0.5, 0.15)
            traits[trait.value] = max(0.0, min(1.0, value))

        return PersonalityProfile(**traits)

    def _initialize_emotion_triggers(self) -> Dict[GameEvent, List[Tuple[EmotionType, float]]]:
        """Initialise les déclencheurs d'émotions pour chaque événement"""
        return {
            GameEvent.LEVEL_UP: [
                (EmotionType.JOY, 0.8),
                (EmotionType.SATISFACTION, 0.7),
                (EmotionType.CONFIDENCE, 0.6)
            ],
            GameEvent.RARE_DROP: [
                (EmotionType.EXCITEMENT, 0.9),
                (EmotionType.JOY, 0.8)
            ],
            GameEvent.QUEST_COMPLETED: [
                (EmotionType.SATISFACTION, 0.7),
                (EmotionType.CONFIDENCE, 0.5)
            ],
            GameEvent.DEATH: [
                (EmotionType.FRUSTRATION, 0.8),
                (EmotionType.DISAPPOINTMENT, 0.6)
            ],
            GameEvent.ITEM_LOST: [
                (EmotionType.FRUSTRATION, 0.9),
                (EmotionType.ANXIETY, 0.5)
            ],
            GameEvent.SERVER_LAG: [
                (EmotionType.IMPATIENCE, 0.6),
                (EmotionType.FRUSTRATION, 0.4)
            ]
        }

    def _initialize_personality_responses(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les types de réponses comportementales"""
        return {
            'aggressive': {
                'dialogue_style': 'direct',
                'risk_adjustment': 0.2,
                'patience_penalty': -0.3
            },
            'cautious': {
                'dialogue_style': 'careful',
                'risk_adjustment': -0.3,
                'planning_bonus': 0.2
            },
            'social': {
                'dialogue_style': 'friendly',
                'team_preference': 0.4,
                'communication_bonus': 0.3
            },
            'analytical': {
                'dialogue_style': 'logical',
                'information_seeking': 0.5,
                'decision_delay': 0.2
            }
        }

    def _calculate_emotional_impact(self, event: GameEvent, context: Dict[str, Any]) -> float:
        """Calcule l'impact émotionnel d'un événement"""
        # Impact de base selon l'événement
        base_impacts = {
            GameEvent.LEVEL_UP: 0.8,
            GameEvent.RARE_DROP: 0.9,
            GameEvent.QUEST_COMPLETED: 0.6,
            GameEvent.ACHIEVEMENT_UNLOCKED: 0.7,
            GameEvent.SUCCESSFUL_TRADE: 0.4,
            GameEvent.TEAM_VICTORY: 0.7,
            GameEvent.DEATH: -0.8,
            GameEvent.ITEM_LOST: -0.9,
            GameEvent.QUEST_FAILED: -0.6,
            GameEvent.SCAM_ATTEMPT: -0.8,
            GameEvent.TEAM_DEFEAT: -0.6,
            GameEvent.SERVER_LAG: -0.3,
            GameEvent.RESOURCE_GATHERED: 0.2,
            GameEvent.TRAVEL_COMPLETED: 0.1,
            GameEvent.SHOP_VISIT: 0.0
        }

        base_impact = base_impacts.get(event, 0.0)

        # Ajustements selon la personnalité
        if event in [GameEvent.RARE_DROP, GameEvent.ACHIEVEMENT_UNLOCKED]:
            # Les personnalités compétitives sont plus affectées par les réussites
            base_impact *= (1.0 + self.personality.competitiveness * 0.3)

        if event in [GameEvent.DEATH, GameEvent.QUEST_FAILED]:
            # Les personnalités névrotiques sont plus affectées par les échecs
            base_impact *= (1.0 + self.personality.neuroticism * 0.4)

        # Ajustements contextuels
        importance = context.get('importance', 1.0)
        surprise_factor = context.get('surprise', 1.0)

        adjusted_impact = base_impact * importance * surprise_factor

        return max(-1.0, min(1.0, adjusted_impact))

    def _generate_emotions_from_event(self, event: GameEvent, impact: float) -> List[Emotion]:
        """Génère les émotions appropriées pour un événement"""
        emotions = []

        if event in self.emotion_triggers:
            for emotion_type, base_intensity in self.emotion_triggers[event]:
                # Ajustement de l'intensité par l'impact
                adjusted_intensity = base_intensity * abs(impact)

                # Durée basée sur l'intensité et le type d'émotion
                duration_minutes = max(5, int(adjusted_intensity * 30))

                emotion = Emotion(
                    type=emotion_type,
                    intensity=adjusted_intensity,
                    duration=timedelta(minutes=duration_minutes),
                    decay_rate=0.05 + random.uniform(0, 0.1)
                )

                emotions.append(emotion)

        return emotions

    async def _learn_from_event(self, event: GameEvent, impact: float, context: Dict[str, Any]):
        """Apprentissage depuis un événement"""
        # Mise à jour des patterns comportementaux
        event_key = event.value

        if event_key not in self.behavior_patterns:
            self.behavior_patterns[event_key] = {
                'frequency': 0,
                'average_impact': 0.0,
                'best_response': None,
                'learning_rate': 0.1
            }

        pattern = self.behavior_patterns[event_key]
        pattern['frequency'] += 1

        # Mise à jour de l'impact moyen
        learning_rate = pattern['learning_rate']
        pattern['average_impact'] = (
            pattern['average_impact'] * (1.0 - learning_rate) +
            impact * learning_rate
        )

        # Adaptation de l'apprentissage
        adaptation_key = f"{event.value}_{context.get('context_type', 'general')}"
        self.adaptation_learning[adaptation_key] += abs(impact) * 0.1

    def _determine_response_type(self, situation_type: str, context: Dict[str, Any]) -> str:
        """Détermine le type de réponse basé sur la personnalité"""
        # Mapping des traits vers les types de réponse
        if self.personality.agreeableness > 0.7:
            return 'social'
        elif self.personality.conscientiousness > 0.7:
            return 'analytical'
        elif self.personality.neuroticism > 0.7:
            return 'cautious'
        elif self.personality.extraversion > 0.7:
            return 'aggressive'
        else:
            return 'balanced'

    def _calculate_response_intensity(self, situation_type: str, context: Dict[str, Any]) -> float:
        """Calcule l'intensité de la réponse"""
        # Intensité de base selon l'humeur
        mood_intensities = {
            MoodState.VERY_POSITIVE: 0.9,
            MoodState.POSITIVE: 0.7,
            MoodState.NEUTRAL: 0.5,
            MoodState.NEGATIVE: 0.6,
            MoodState.VERY_NEGATIVE: 0.8
        }

        base_intensity = mood_intensities.get(self.current_mood, 0.5)

        # Ajustement par traits de personnalité
        if self.personality.neuroticism > 0.6:
            base_intensity *= 1.3  # Plus de réactivité

        if self.personality.conscientiousness > 0.6:
            base_intensity *= 0.8  # Plus de modération

        return max(0.1, min(1.0, base_intensity))

    def _generate_response_actions(self, situation_type: str, response_type: str,
                                 context: Dict[str, Any]) -> List[str]:
        """Génère les actions suggérées pour la réponse"""
        actions = []

        if response_type == 'aggressive':
            actions = ['take_immediate_action', 'confront_challenge', 'increase_activity']
        elif response_type == 'cautious':
            actions = ['assess_situation', 'seek_information', 'plan_carefully']
        elif response_type == 'social':
            actions = ['seek_team', 'communicate', 'collaborate']
        elif response_type == 'analytical':
            actions = ['analyze_data', 'calculate_odds', 'optimize_strategy']
        else:
            actions = ['continue_current_activity', 'monitor_situation']

        return actions

    def _generate_dialogue(self, situation_type: str, response_type: str,
                         context: Dict[str, Any]) -> Optional[str]:
        """Génère un dialogue approprié"""
        if response_type == 'social' and random.random() < 0.3:
            return "Quelqu'un veut faire équipe ?"
        elif response_type == 'aggressive' and random.random() < 0.2:
            return "Allons-y, on n'a pas toute la journée !"
        elif response_type == 'cautious' and random.random() < 0.25:
            return "Attendez, vérifions d'abord..."

        return None

    def _calculate_behavior_changes(self, response_type: str, intensity: float) -> Dict[str, float]:
        """Calcule les changements comportementaux temporaires"""
        changes = {}

        if response_type == 'aggressive':
            changes['speed_multiplier'] = 1.0 + intensity * 0.3
            changes['risk_tolerance'] = intensity * 0.2
        elif response_type == 'cautious':
            changes['speed_multiplier'] = 1.0 - intensity * 0.2
            changes['risk_tolerance'] = -intensity * 0.3
        elif response_type == 'social':
            changes['team_preference'] = intensity * 0.4
            changes['communication_frequency'] = intensity * 0.5

        return changes

    def _adjust_contextual_risk_tolerance(self):
        """Ajuste la tolérance au risque selon les contextes"""
        # Apprentissage basé sur l'historique
        for memory in list(self.emotional_memory)[-50:]:  # Derniers 50 événements
            if memory.event_type in [GameEvent.DEATH, GameEvent.ITEM_LOST]:
                # Réduction de la tolérance au risque de combat/financier
                self.risk_profile.combat_risk *= 0.99
                self.risk_profile.financial_risk *= 0.99

            elif memory.event_type in [GameEvent.RARE_DROP, GameEvent.SUCCESSFUL_TRADE]:
                # Augmentation légère de la tolérance
                self.risk_profile.combat_risk = min(1.0, self.risk_profile.combat_risk * 1.001)
                self.risk_profile.financial_risk = min(1.0, self.risk_profile.financial_risk * 1.001)

    def _calculate_personality_evolution(self, event_type: str, outcome: str,
                                       context: Dict[str, Any]) -> Dict[PersonalityTrait, float]:
        """Calcule l'évolution de la personnalité basée sur une expérience"""
        changes = {}

        # Taux d'apprentissage (très lent pour la personnalité)
        learning_rate = 0.001

        if event_type == 'social_interaction':
            if outcome == 'positive':
                changes[PersonalityTrait.EXTRAVERSION] = learning_rate
                changes[PersonalityTrait.SOCIAL_TENDENCY] = learning_rate
            elif outcome == 'negative':
                changes[PersonalityTrait.EXTRAVERSION] = -learning_rate * 0.5

        elif event_type == 'risk_taking':
            if outcome == 'success':
                changes[PersonalityTrait.RISK_TAKING] = learning_rate
            elif outcome == 'failure':
                changes[PersonalityTrait.RISK_TAKING] = -learning_rate
                changes[PersonalityTrait.NEUROTICISM] = learning_rate * 0.5

        elif event_type == 'achievement':
            if outcome == 'success':
                changes[PersonalityTrait.CONSCIENTIOUSNESS] = learning_rate
                changes[PersonalityTrait.COMPETITIVENESS] = learning_rate * 0.5

        return changes

    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état émotionnel actuel"""
        return {
            'current_mood': self.current_mood.value,
            'active_emotions': [
                {
                    'type': emotion.type.value,
                    'intensity': emotion.current_intensity(),
                    'remaining_duration': str(emotion.duration - (datetime.now() - emotion.created_at))
                }
                for emotion in self.active_emotions if emotion.is_active()
            ],
            'risk_tolerance': self.risk_profile.current_tolerance(),
            'motivation': self.get_current_motivation(),
            'personality_summary': {
                'dominant_traits': self._get_dominant_traits(),
                'recent_evolution': self.personality.evolution_history[-5:] if self.personality.evolution_history else []
            },
            'emotional_memory_count': len(self.emotional_memory)
        }

    def _get_dominant_traits(self) -> List[str]:
        """Identifie les traits de personnalité dominants"""
        trait_values = {
            'openness': self.personality.openness,
            'conscientiousness': self.personality.conscientiousness,
            'extraversion': self.personality.extraversion,
            'agreeableness': self.personality.agreeableness,
            'neuroticism': self.personality.neuroticism,
            'risk_taking': self.personality.risk_taking,
            'competitiveness': self.personality.competitiveness,
            'patience': self.personality.patience,
            'curiosity': self.personality.curiosity,
            'social_tendency': self.personality.social_tendency
        }

        # Tri par valeur décroissante
        sorted_traits = sorted(trait_values.items(), key=lambda x: x[1], reverse=True)

        # Retourne les 3 traits les plus élevés (>0.6)
        dominant = [trait for trait, value in sorted_traits[:3] if value > 0.6]

        return dominant

# Interface utilitaire
async def create_emotional_manager(personality_config: Optional[Dict[str, float]] = None) -> EmotionalStateManager:
    """Crée et initialise le gestionnaire d'état émotionnel"""

    # Création du profil de personnalité
    if personality_config:
        personality = PersonalityProfile(**personality_config)
    else:
        personality = None  # Sera généré aléatoirement

    manager = EmotionalStateManager(personality)

    # Simulation d'événements initiaux pour établir un état de base
    initial_events = [
        (GameEvent.RESOURCE_GATHERED, {'importance': 0.5}),
        (GameEvent.TRAVEL_COMPLETED, {'importance': 0.3}),
        (GameEvent.SHOP_VISIT, {'importance': 0.2})
    ]

    for event, context in initial_events:
        await manager.process_game_event(event, context)

    return manager

# Interface CLI pour tests
async def main():
    """Test du système d'état émotionnel"""
    print("Test Emotional State Management...")

    # Création du gestionnaire
    manager = await create_emotional_manager()

    print(f"Gestionnaire créé avec personnalité:")
    dominant_traits = manager._get_dominant_traits()
    print(f"  - Traits dominants: {dominant_traits}")

    # Simulation d'événements
    test_events = [
        (GameEvent.LEVEL_UP, {'importance': 1.0, 'surprise': 0.8}),
        (GameEvent.RARE_DROP, {'importance': 1.2, 'surprise': 1.5}),
        (GameEvent.DEATH, {'importance': 0.8, 'surprise': 0.5}),
        (GameEvent.QUEST_COMPLETED, {'importance': 0.7, 'surprise': 0.3})
    ]

    for event, context in test_events:
        await manager.process_game_event(event, context)
        mood = manager.simulate_player_mood()
        print(f"Événement {event.value}: humeur = {mood.value}")

    # Test de génération de réponse
    situation = {
        'type': 'combat_challenge',
        'context': {'difficulty': 'high', 'reward': 'valuable'}
    }

    response = manager.generate_personality_response(situation)
    print(f"Réponse générée:")
    print(f"  - Type: {response.response_type}")
    print(f"  - Intensité: {response.intensity:.2f}")
    print(f"  - Actions: {response.actions}")
    if response.dialogue:
        print(f"  - Dialogue: {response.dialogue}")

    # Ajustement de la tolérance au risque
    risk_profile = manager.adjust_risk_tolerance(manager.current_mood)
    print(f"Tolérance au risque: {risk_profile.current_tolerance():.2f}")

    # Motivation actuelle
    motivation = manager.get_current_motivation()
    print(f"Motivation actuelle:")
    for activity, level in motivation.items():
        print(f"  - {activity}: {level:.2f}")

    # Résumé de l'état émotionnel
    summary = manager.get_emotional_state_summary()
    print(f"Résumé émotionnel:")
    print(f"  - Humeur: {summary['current_mood']}")
    print(f"  - Émotions actives: {len(summary['active_emotions'])}")
    print(f"  - Mémoires émotionnelles: {summary['emotional_memory_count']}")

    print("Test Emotional State terminé !")

if __name__ == "__main__":
    asyncio.run(main())