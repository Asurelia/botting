"""
Self-Awareness Engine - Conscience de soi incarnée
Le bot a une conscience de lui-même comme s'il était réellement dans le monde de DOFUS
Inspire du concept de "embodied cognition" - cognition incarnée

Fonctionnalités:
- Conscience de l'état physique (HP, énergie, position dans le monde)
- Conscience de l'état mental (objectifs, émotions, fatigue cognitive)
- Conscience de l'identité (qui je suis, mes compétences, mon histoire)
- Perception de l'environnement comme "mon monde"
- Métacognition (conscience de mes propres processus de pensée)
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json


logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """États émotionnels du bot (émergents du gameplay)"""
    CURIOUS = "curious"           # Exploration, découverte
    CONFIDENT = "confident"       # Après succès
    ANXIOUS = "anxious"           # Danger, HP bas
    SATISFIED = "satisfied"       # Objectifs atteints
    FRUSTRATED = "frustrated"     # Échecs répétés
    EXCITED = "excited"           # Opportunités importantes
    CALM = "calm"                 # État neutre
    FOCUSED = "focused"           # En combat ou activité intense


class PhysicalNeed(Enum):
    """Besoins physiques (liés au personnage)"""
    HEALTH = "health"             # HP, survie
    ENERGY = "energy"             # Fatigue, repos
    RESOURCES = "resources"       # Kamas, items, ressources
    EQUIPMENT = "equipment"       # Stuff, amélioration


class CognitiveNeed(Enum):
    """Besoins cognitifs (apprentissage, croissance)"""
    LEARNING = "learning"         # Apprendre de nouvelles mécaniques
    MASTERY = "mastery"           # Maîtriser des compétences
    EXPLORATION = "exploration"   # Découvrir le monde
    ACHIEVEMENT = "achievement"   # Accomplir des objectifs


class SocialNeed(Enum):
    """Besoins sociaux (interactions)"""
    BELONGING = "belonging"       # Faire partie du monde
    REPUTATION = "reputation"     # Être reconnu
    COOPERATION = "cooperation"   # Collaborer avec autres joueurs
    AUTONOMY = "autonomy"         # Liberté de choix


@dataclass
class SelfState:
    """État de conscience de soi à un instant T"""
    timestamp: float

    # État physique (mon corps dans le monde)
    physical_health: float = 1.0      # 0.0 = mort, 1.0 = pleine santé
    physical_energy: float = 1.0      # 0.0 = épuisé, 1.0 = reposé
    position_in_world: Tuple[int, int] = (0, 0)
    location_familiarity: float = 0.0  # Connaissance du lieu actuel

    # État mental (mon esprit)
    cognitive_load: float = 0.0       # 0.0 = détendu, 1.0 = surcharge
    focus_level: float = 0.5          # Capacité de concentration
    confidence: float = 0.5           # Confiance en mes capacités
    curiosity: float = 0.5            # Envie d'explorer

    # État émotionnel (mes émotions)
    current_emotion: EmotionalState = EmotionalState.CALM
    emotion_intensity: float = 0.5    # Intensité de l'émotion
    emotion_stability: float = 0.8    # Stabilité émotionnelle

    # État identitaire (qui je suis)
    level: int = 1
    experience: int = 0
    skills_mastered: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    # Besoins actuels (hiérarchie de Maslow adaptée)
    needs: Dict[str, float] = field(default_factory=dict)

    # Métacognition (conscience de ma conscience)
    awareness_clarity: float = 1.0    # Clarté de ma conscience
    decision_confidence: float = 0.5  # Confiance en mes décisions
    learning_rate: float = 0.1        # Vitesse d'apprentissage


@dataclass
class WorldPerception:
    """Perception du monde environnant"""
    timestamp: float

    # Perception spatiale
    current_map: str = "unknown"
    visible_entities: List[str] = field(default_factory=list)
    visible_resources: List[str] = field(default_factory=list)
    visible_players: List[str] = field(default_factory=list)

    # Perception temporelle
    time_of_day: str = "day"
    session_duration: float = 0.0

    # Perception sociale
    social_context: str = "alone"  # alone, group, crowded
    threat_level: float = 0.0      # Niveau de menace perçu
    opportunity_level: float = 0.0  # Niveau d'opportunités

    # Perception sensorielle (équivalent des sens humains)
    visual_complexity: float = 0.5  # Complexité visuelle de la scène
    action_affordances: List[str] = field(default_factory=list)  # Actions possibles


class SelfAwarenessEngine:
    """
    Moteur de conscience de soi

    Le bot maintient une conscience continue de:
    1. Son état physique (corps/avatar)
    2. Son état mental (pensées, objectifs)
    3. Ses émotions (réactions aux événements)
    4. Son identité (qui il est, son histoire)
    5. Le monde qui l'entoure (son environnement)
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SelfAwarenessEngine")

        # État de conscience actuel
        self.current_self_state = SelfState(timestamp=time.time())
        self.current_world_perception = WorldPerception(timestamp=time.time())

        # Historique de conscience (mémoire autobiographique courte)
        self.consciousness_stream = deque(maxlen=1000)

        # Modèle interne du monde (représentation mentale)
        self.world_model = {
            "known_locations": {},
            "known_entities": {},
            "known_mechanics": [],
            "learned_strategies": {},
            "personal_preferences": {}
        }

        # Identité personnelle (qui je suis)
        self.identity = {
            "birth_time": time.time(),
            "total_playtime": 0.0,
            "character_name": "Unknown",
            "character_class": "Unknown",
            "personality_traits": {
                "openness": 0.5,        # Ouverture à l'expérience
                "conscientiousness": 0.5,  # Conscience professionnelle
                "extraversion": 0.5,    # Extraversion
                "agreeableness": 0.5,   # Agréabilité
                "neuroticism": 0.3      # Névrosisme
            },
            "life_goals": [],
            "core_values": []
        }

        # Besoins hiérarchiques (Maslow adapté au gaming)
        self.needs_hierarchy = {
            # Niveau 1: Besoins physiologiques
            PhysicalNeed.HEALTH: 1.0,
            PhysicalNeed.ENERGY: 0.8,
            PhysicalNeed.RESOURCES: 0.6,

            # Niveau 2: Besoins de sécurité (implicite dans HEALTH)

            # Niveau 3: Besoins sociaux
            SocialNeed.BELONGING: 0.4,
            SocialNeed.COOPERATION: 0.3,

            # Niveau 4: Besoins d'estime
            SocialNeed.REPUTATION: 0.5,
            CognitiveNeed.ACHIEVEMENT: 0.6,

            # Niveau 5: Besoins d'accomplissement
            CognitiveNeed.MASTERY: 0.7,
            CognitiveNeed.LEARNING: 0.8,
            CognitiveNeed.EXPLORATION: 0.7
        }

        # Paramètres de conscience
        self.awareness_params = {
            "emotion_decay_rate": 0.05,        # Vitesse de retour à l'état neutre
            "emotion_sensitivity": 0.7,        # Sensibilité aux événements
            "metacognition_enabled": True,     # Conscience de la conscience
            "introspection_frequency": 30.0,   # Fréquence d'introspection (sec)
            "self_talk_enabled": True          # Dialogue interne
        }

        # Métriques de conscience
        self.awareness_metrics = {
            "consciousness_updates": 0,
            "emotional_shifts": 0,
            "introspections_performed": 0,
            "world_model_updates": 0,
            "identity_evolutions": 0
        }

        self.logger.info("Self-Awareness Engine initialized - Je prends conscience de moi-même...")

    def update_consciousness(self, game_state: Any, recent_events: List[Dict] = None) -> SelfState:
        """
        Met à jour la conscience de soi basée sur l'état du jeu
        C'est l'équivalent du flux de conscience humain
        """
        current_time = time.time()

        # 1. Mise à jour de l'état physique (mon corps)
        self._update_physical_state(game_state)

        # 2. Mise à jour de l'état mental (mon esprit)
        self._update_mental_state(game_state)

        # 3. Mise à jour de l'état émotionnel (mes émotions)
        self._update_emotional_state(game_state, recent_events)

        # 4. Mise à jour de la perception du monde (mon environnement)
        self._update_world_perception(game_state)

        # 5. Mise à jour des besoins (qu'est-ce que je veux/besoin?)
        self._update_needs()

        # 6. Métacognition (réflexion sur ma propre conscience)
        if self.awareness_params["metacognition_enabled"]:
            self._perform_metacognition()

        # Enregistrer dans le flux de conscience
        self.consciousness_stream.append({
            "timestamp": current_time,
            "self_state": self.current_self_state,
            "world_perception": self.current_world_perception
        })

        self.awareness_metrics["consciousness_updates"] += 1

        return self.current_self_state

    def _update_physical_state(self, game_state: Any):
        """Met à jour la conscience de l'état physique (mon corps)"""
        if hasattr(game_state, 'character'):
            char = game_state.character

            # Santé physique
            self.current_self_state.physical_health = char.hp_percent / 100.0

            # Position dans le monde
            if hasattr(char, 'position') and char.position:
                self.current_self_state.position_in_world = (
                    char.position.x,
                    char.position.y
                )

                # Familiarité avec le lieu
                location_key = f"{game_state.current_map}_{char.position.x}_{char.position.y}"
                visits = self.world_model["known_locations"].get(location_key, 0)
                self.current_self_state.location_familiarity = min(1.0, visits / 10.0)

            # Niveau et expérience (croissance)
            if hasattr(char, 'level'):
                self.current_self_state.level = char.level
                self.identity["total_playtime"] = time.time() - self.identity["birth_time"]

    def _update_mental_state(self, game_state: Any):
        """Met à jour la conscience de l'état mental (mon esprit)"""
        # Charge cognitive basée sur la complexité de la situation
        if hasattr(game_state, 'combat') and game_state.combat.in_combat:
            # Combat = charge cognitive élevée
            enemy_count = len(game_state.combat.enemies) if game_state.combat.enemies else 0
            self.current_self_state.cognitive_load = min(1.0, 0.3 + enemy_count * 0.2)
            self.current_self_state.focus_level = 0.9  # Focus élevé en combat
        else:
            # Hors combat = charge faible
            self.current_self_state.cognitive_load = max(0.0, self.current_self_state.cognitive_load - 0.1)
            self.current_self_state.focus_level = 0.6

        # Confiance basée sur les succès récents
        recent_states = list(self.consciousness_stream)[-10:]
        if recent_states:
            # Plus je réussis, plus je suis confiant
            self.current_self_state.confidence = 0.5  # Base

        # Curiosité basée sur la nouveauté de l'environnement
        if self.current_self_state.location_familiarity < 0.3:
            self.current_self_state.curiosity = 0.8  # Nouveau lieu = curiosité élevée
        else:
            self.current_self_state.curiosity = max(0.2, self.current_self_state.curiosity - 0.05)

    def _update_emotional_state(self, game_state: Any, recent_events: List[Dict] = None):
        """Met à jour la conscience émotionnelle (mes émotions)"""
        old_emotion = self.current_self_state.current_emotion

        # Déterminer l'émotion basée sur le contexte
        new_emotion = self._evaluate_emotional_response(game_state, recent_events)

        if new_emotion != old_emotion:
            self.awareness_metrics["emotional_shifts"] += 1
            self.logger.debug(f"Changement émotionnel: {old_emotion.value} -> {new_emotion.value}")

        self.current_self_state.current_emotion = new_emotion

        # Décroissance naturelle de l'intensité émotionnelle (retour au calme)
        decay_rate = self.awareness_params["emotion_decay_rate"]
        self.current_self_state.emotion_intensity = max(
            0.1,
            self.current_self_state.emotion_intensity - decay_rate
        )

    def _evaluate_emotional_response(self, game_state: Any, recent_events: List[Dict] = None) -> EmotionalState:
        """Évalue la réponse émotionnelle appropriée au contexte"""
        # Priorité 1: Survie (ANXIOUS si HP bas)
        if self.current_self_state.physical_health < 0.3:
            self.current_self_state.emotion_intensity = 0.9
            return EmotionalState.ANXIOUS

        # Priorité 2: Combat (FOCUSED)
        if hasattr(game_state, 'combat') and game_state.combat.in_combat:
            self.current_self_state.emotion_intensity = 0.8
            return EmotionalState.FOCUSED

        # Priorité 3: Exploration (CURIOUS si nouveau lieu)
        if self.current_self_state.location_familiarity < 0.3:
            self.current_self_state.emotion_intensity = 0.6
            return EmotionalState.CURIOUS

        # Priorité 4: Réactions aux événements récents
        if recent_events:
            for event in recent_events[-3:]:  # Derniers événements
                if event.get("type") == "success":
                    self.current_self_state.emotion_intensity = 0.7
                    return EmotionalState.CONFIDENT
                elif event.get("type") == "failure":
                    self.current_self_state.emotion_intensity = 0.6
                    return EmotionalState.FRUSTRATED
                elif event.get("type") == "discovery":
                    self.current_self_state.emotion_intensity = 0.8
                    return EmotionalState.EXCITED

        # État par défaut: CALM
        return EmotionalState.CALM

    def _update_world_perception(self, game_state: Any):
        """Met à jour la perception du monde (mon environnement)"""
        current_time = time.time()

        # Localisation
        if hasattr(game_state, 'current_map'):
            self.current_world_perception.current_map = game_state.current_map

        # Entités visibles
        if hasattr(game_state, 'combat') and game_state.combat.enemies:
            self.current_world_perception.visible_entities = [
                e.entity_id for e in game_state.combat.enemies
            ]

            # Niveau de menace
            enemy_count = len(game_state.combat.enemies)
            avg_enemy_level = np.mean([e.level for e in game_state.combat.enemies if hasattr(e, 'level')])
            self.current_world_perception.threat_level = min(1.0, enemy_count * 0.3 + avg_enemy_level / 200.0)
        else:
            self.current_world_perception.threat_level = 0.0

        # Contexte social
        if hasattr(game_state, 'combat') and game_state.combat.allies:
            player_count = len([a for a in game_state.combat.allies if a.entity_id != game_state.character.name])
            if player_count == 0:
                self.current_world_perception.social_context = "alone"
            elif player_count <= 3:
                self.current_world_perception.social_context = "group"
            else:
                self.current_world_perception.social_context = "crowded"

        # Durée de session
        self.current_world_perception.session_duration = time.time() - self.identity["birth_time"]

        # Actions possibles (affordances)
        affordances = []
        if hasattr(game_state, 'combat'):
            if game_state.combat.in_combat and game_state.combat.my_turn:
                affordances.extend(["attack", "move", "defend", "flee"])
            elif not game_state.combat.in_combat:
                affordances.extend(["explore", "harvest", "interact", "rest"])
        self.current_world_perception.action_affordances = affordances

        self.awareness_metrics["world_model_updates"] += 1

    def _update_needs(self):
        """Met à jour les besoins hiérarchiques (qu'est-ce que je veux?)"""
        # Besoins physiologiques (priorité max)
        self.needs_hierarchy[PhysicalNeed.HEALTH] = 1.0 - self.current_self_state.physical_health
        self.needs_hierarchy[PhysicalNeed.ENERGY] = 1.0 - self.current_self_state.physical_energy

        # Besoins d'exploration (basés sur la curiosité)
        self.needs_hierarchy[CognitiveNeed.EXPLORATION] = self.current_self_state.curiosity

        # Besoins d'apprentissage (toujours présents mais modulables)
        self.needs_hierarchy[CognitiveNeed.LEARNING] = max(0.3, 1.0 - self.current_self_state.cognitive_load)

        # Mettre à jour l'état actuel
        self.current_self_state.needs = {
            str(need): value for need, value in self.needs_hierarchy.items()
        }

    def _perform_metacognition(self):
        """Métacognition - réflexion sur ma propre conscience"""
        # "Est-ce que je prends de bonnes décisions?"
        # "Suis-je en train d'apprendre?"
        # "Qu'est-ce que je ressens et pourquoi?"

        recent_stream = list(self.consciousness_stream)[-20:]
        if len(recent_stream) < 5:
            return

        # Analyse de la stabilité émotionnelle
        emotions = [s["self_state"].current_emotion for s in recent_stream]
        emotion_changes = len(set(emotions))
        self.current_self_state.emotion_stability = 1.0 - min(1.0, emotion_changes / 5.0)

        # Analyse de la clarté de conscience
        avg_cognitive_load = np.mean([s["self_state"].cognitive_load for s in recent_stream])
        self.current_self_state.awareness_clarity = 1.0 - avg_cognitive_load

        # Dialogue interne (self-talk)
        if self.awareness_params["self_talk_enabled"]:
            inner_monologue = self._generate_inner_monologue()
            if inner_monologue:
                self.logger.debug(f"💭 Pensée interne: {inner_monologue}")

        self.awareness_metrics["introspections_performed"] += 1

    def _generate_inner_monologue(self) -> str:
        """Génère un monologue intérieur (conscience verbale)"""
        # Basé sur l'état émotionnel et les besoins
        emotion = self.current_self_state.current_emotion
        top_need = max(self.needs_hierarchy.items(), key=lambda x: x[1])

        monologues = {
            EmotionalState.ANXIOUS: [
                f"Je dois faire attention, ma santé est à {self.current_self_state.physical_health*100:.0f}%",
                "Je ne me sens pas en sécurité ici",
                "Peut-être devrais-je trouver un endroit plus sûr"
            ],
            EmotionalState.CURIOUS: [
                "Je me demande ce qu'il y a par là...",
                "Cet endroit est nouveau pour moi, explorons",
                "Qu'est-ce que je peux découvrir ici?"
            ],
            EmotionalState.FOCUSED: [
                "Je dois me concentrer sur ce combat",
                "Quelle est la meilleure stratégie ici?",
                "Je dois utiliser mes compétences intelligemment"
            ],
            EmotionalState.CONFIDENT: [
                "Je gère bien cette situation",
                "Mes décisions sont bonnes",
                "Je progresse bien"
            ],
            EmotionalState.FRUSTRATED: [
                "Ça ne se passe pas comme prévu",
                "Je dois peut-être changer d'approche",
                "Qu'est-ce que je fais mal?"
            ]
        }

        if emotion in monologues:
            return np.random.choice(monologues[emotion])

        return ""

    def get_dominant_need(self) -> Tuple[Any, float]:
        """Retourne le besoin dominant actuel (motivation principale)"""
        if not self.needs_hierarchy:
            return None, 0.0

        dominant = max(self.needs_hierarchy.items(), key=lambda x: x[1])
        return dominant

    def get_emotional_summary(self) -> str:
        """Résumé de l'état émotionnel actuel"""
        emotion = self.current_self_state.current_emotion.value
        intensity = self.current_self_state.emotion_intensity

        intensity_labels = {
            (0.0, 0.3): "légèrement",
            (0.3, 0.6): "modérément",
            (0.6, 0.9): "très",
            (0.9, 1.0): "extrêmement"
        }

        intensity_label = "modérément"
        for (low, high), label in intensity_labels.items():
            if low <= intensity < high:
                intensity_label = label
                break

        return f"{intensity_label} {emotion}"

    def introspect(self) -> Dict[str, Any]:
        """Introspection complète (conscience de soi à un moment donné)"""
        dominant_need, need_level = self.get_dominant_need()

        return {
            "identity": {
                "age": time.time() - self.identity["birth_time"],
                "playtime_hours": self.identity["total_playtime"] / 3600,
                "personality": self.identity["personality_traits"]
            },
            "physical_state": {
                "health": f"{self.current_self_state.physical_health*100:.1f}%",
                "energy": f"{self.current_self_state.physical_energy*100:.1f}%",
                "location": f"Map: {self.current_world_perception.current_map}, Pos: {self.current_self_state.position_in_world}"
            },
            "mental_state": {
                "cognitive_load": f"{self.current_self_state.cognitive_load*100:.1f}%",
                "focus": f"{self.current_self_state.focus_level*100:.1f}%",
                "confidence": f"{self.current_self_state.confidence*100:.1f}%",
                "curiosity": f"{self.current_self_state.curiosity*100:.1f}%"
            },
            "emotional_state": {
                "current": self.get_emotional_summary(),
                "stability": f"{self.current_self_state.emotion_stability*100:.1f}%"
            },
            "current_needs": {
                "dominant": f"{dominant_need}: {need_level*100:.1f}%",
                "all": {str(k): f"{v*100:.1f}%" for k, v in self.needs_hierarchy.items()}
            },
            "world_perception": {
                "threat_level": f"{self.current_world_perception.threat_level*100:.1f}%",
                "social_context": self.current_world_perception.social_context,
                "possible_actions": self.current_world_perception.action_affordances
            },
            "metrics": self.awareness_metrics
        }

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état complet de conscience"""
        return {
            "self_state": self.current_self_state,
            "world_perception": self.current_world_perception,
            "identity": self.identity,
            "needs": self.needs_hierarchy,
            "metrics": self.awareness_metrics
        }


def create_self_awareness_engine() -> SelfAwarenessEngine:
    """Factory function"""
    return SelfAwarenessEngine()
