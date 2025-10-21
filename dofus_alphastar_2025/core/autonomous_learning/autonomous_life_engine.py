"""
Autonomous Life Engine - Moteur de vie autonome
Le bot vit dans le monde de DOFUS comme un être autonome qui apprend et évolue

Intègre tous les systèmes cognitifs:
- Self-Awareness (conscience de soi)
- Continuous Learning (apprentissage)
- Autobiographical Memory (mémoire de vie)
- Emergent Decision System (décisions autonomes)

C'est le "cerveau complet" du bot autonome incarné
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from .self_awareness import SelfAwarenessEngine, create_self_awareness_engine, MemoryCategory
from .continuous_learning import ContinuousLearningEngine, create_continuous_learning_engine, ExperienceType
from .autobiographical_memory import AutobiographicalMemory, create_autobiographical_memory, MemoryImportance
from .emergent_decision_system import EmergentDecisionSystem, create_emergent_decision_system


logger = logging.getLogger(__name__)


@dataclass
class LifeStats:
    """Statistiques de vie du bot"""
    birth_time: float
    total_lifetime_seconds: float = 0.0
    total_decisions_made: int = 0
    total_experiences: int = 0
    total_memories: int = 0
    total_skills_learned: int = 0
    total_knowledge_acquired: int = 0
    consciousness_updates: int = 0
    learning_sessions: int = 0


class AutonomousLifeEngine:
    """
    Moteur de vie autonome complet

    Le bot:
    1. A conscience de lui-même et de son environnement
    2. Ressent des besoins et des émotions
    3. Prend des décisions autonomes basées sur son état interne
    4. Apprend continuellement de ses expériences
    5. Se souvient de son histoire de vie
    6. Évolue sa personnalité et son identité avec le temps

    C'est une véritable IA incarnée qui "vit" dans le monde de DOFUS
    """

    def __init__(
        self,
        character_name: str = "AutonomousBot",
        character_class: str = "Unknown",
        personality_preset: Optional[Dict[str, float]] = None
    ):
        self.logger = logging.getLogger(f"{__name__}.AutonomousLifeEngine")

        # Informations de base
        self.character_name = character_name
        self.character_class = character_class

        # === SYSTÈMES COGNITIFS ===
        self.logger.info("🧠 Initialisation des systèmes cognitifs...")

        # 1. Conscience de soi
        self.self_awareness = create_self_awareness_engine()
        self.self_awareness.identity["character_name"] = character_name
        self.self_awareness.identity["character_class"] = character_class

        # Personnalité (Big Five)
        if personality_preset:
            self.self_awareness.identity["personality_traits"].update(personality_preset)

        # 2. Mémoire autobiographique
        self.memory = create_autobiographical_memory()

        # 3. Apprentissage continu
        self.learning_engine = create_continuous_learning_engine()

        # 4. Système de décision émergente
        self.decision_system = create_emergent_decision_system(
            self.self_awareness,
            self.learning_engine,
            self.memory
        )

        # === ÉTAT DE VIE ===
        self.birth_time = time.time()
        self.is_alive = True
        self.life_stats = LifeStats(birth_time=self.birth_time)

        # Processus périodiques (comme les fonctions biologiques humaines)
        self.last_introspection = time.time()
        self.last_consolidation = time.time()
        self.last_learning_session = time.time()

        # Configuration
        self.config = {
            "introspection_interval": 300,    # Introspection toutes les 5 minutes
            "consolidation_interval": 3600,   # Consolidation mémoire toutes les heures
            "learning_session_interval": 120, # Session d'apprentissage toutes les 2 minutes
            "auto_save_interval": 1800,       # Sauvegarde auto toutes les 30 minutes
            "consciousness_update_rate": 1.0  # Mise à jour conscience chaque seconde
        }

        self.last_auto_save = time.time()

        # Callbacks pour interface externe
        self.on_decision_made: Optional[Callable] = None
        self.on_memory_created: Optional[Callable] = None
        self.on_learning_complete: Optional[Callable] = None
        self.on_introspection: Optional[Callable] = None

        # Thread de vie (processus de fond)
        self.life_thread: Optional[threading.Thread] = None
        self.running = False

        self.logger.info(f"✨ {character_name} prend vie dans le monde de DOFUS!")
        self._create_birth_memory()

    def _create_birth_memory(self):
        """Crée le premier souvenir - la naissance"""
        self.memory.create_memory(
            what_happened=f"Je prends conscience de moi-même. Je suis {self.character_name}, {self.character_class}.",
            where="Unknown",
            category=MemoryCategory.MILESTONE,
            importance=MemoryImportance.MILESTONE,
            emotional_valence=0.8,
            emotional_intensity=0.9,
            associated_emotion="excited",
            context={"event": "birth", "character": self.character_name},
            lesson_learned="Je commence mon aventure dans le monde de DOFUS"
        )

    def live_moment(self, game_state: Any, vision_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Vit un moment (équivalent d'une frame de conscience)

        C'est la fonction principale appelée à chaque cycle du jeu.
        Le bot:
        1. Prend conscience de son état actuel
        2. Décide quoi faire
        3. Apprend de l'expérience

        Returns:
            Decision dict si une action doit être prise
        """
        current_time = time.time()

        # Mise à jour des statistiques de vie
        self.life_stats.total_lifetime_seconds = current_time - self.birth_time

        # === 1. CONSCIENCE ===
        # Mise à jour continue de la conscience de soi
        self.self_awareness.update_consciousness(game_state, list(self.decision_system.recent_events))
        self.life_stats.consciousness_updates += 1

        # === 2. DÉCISION ===
        # Prendre une décision autonome
        decision = self.decision_system.decide(game_state, vision_data)

        if decision:
            self.life_stats.total_decisions_made += 1

            # Callback
            if self.on_decision_made:
                self.on_decision_made(decision)

            # Créer un souvenir de la décision si importante
            if decision.confidence > 0.7 or decision.origin.value in ["survival", "milestone"]:
                self._create_decision_memory(decision, game_state)

            return {
                "action_type": decision.action_type,
                "details": decision.action_details,
                "reason": decision.reasoning,
                "confidence": decision.confidence,
                "decision_obj": decision  # Pour record_outcome plus tard
            }

        # === 3. PROCESSUS PÉRIODIQUES ===
        self._run_periodic_processes(current_time)

        return None

    def record_outcome(self, decision: Any, outcome: str, reward: float = 0.0):
        """
        Enregistre le résultat d'une action (apprentissage)

        Args:
            decision: L'objet Decision
            outcome: "success", "failure", ou "neutral"
            reward: Récompense/pénalité (-1.0 à +1.0)
        """
        # Enregistrer dans le système de décision (apprentissage)
        self.decision_system.record_decision_outcome(decision, outcome, reward)

        # Créer un souvenir du résultat si significatif
        if abs(reward) > 0.5:
            importance = MemoryImportance.SIGNIFICANT if abs(reward) > 0.7 else MemoryImportance.MODERATE

            category = MemoryCategory.ACHIEVEMENT if outcome == "success" else MemoryCategory.FAILURE

            self.memory.create_memory(
                what_happened=f"Action {decision.action_type}: {outcome} (récompense: {reward:.2f})",
                where=self.self_awareness.current_world_perception.current_map,
                category=category,
                importance=importance,
                emotional_valence=reward,
                emotional_intensity=abs(reward),
                associated_emotion=self.self_awareness.current_self_state.current_emotion.value,
                lesson_learned=f"Apprendre de ce {outcome}"
            )

        self.life_stats.total_experiences += 1

    def _create_decision_memory(self, decision: Any, game_state: Any):
        """Crée un souvenir d'une décision importante"""
        self.memory.create_memory(
            what_happened=f"Décision: {decision.action_type} - {decision.reasoning}",
            where=self.self_awareness.current_world_perception.current_map,
            category=MemoryCategory.LEARNING,
            importance=MemoryImportance.MODERATE,
            emotional_valence=decision.confidence - 0.5,
            emotional_intensity=decision.confidence,
            associated_emotion=self.self_awareness.current_self_state.current_emotion.value,
            context={"decision": decision.action_type, "origin": decision.origin.value},
            lesson_learned=decision.reasoning
        )

    def _run_periodic_processes(self, current_time: float):
        """Exécute les processus périodiques (comme les fonctions biologiques)"""

        # Introspection périodique (auto-réflexion)
        if current_time - self.last_introspection >= self.config["introspection_interval"]:
            self._introspect()
            self.last_introspection = current_time

        # Consolidation de mémoire (comme le sommeil)
        if current_time - self.last_consolidation >= self.config["consolidation_interval"]:
            self._consolidate_memories()
            self.last_consolidation = current_time

        # Session d'apprentissage
        if current_time - self.last_learning_session >= self.config["learning_session_interval"]:
            self._learning_session()
            self.last_learning_session = current_time

        # Sauvegarde automatique
        if current_time - self.last_auto_save >= self.config["auto_save_interval"]:
            self.save_life_state()
            self.last_auto_save = current_time

    def _introspect(self):
        """
        Introspection - Réflexion sur soi-même
        Le bot prend un moment pour réfléchir à son état, ses progrès, son identité
        """
        self.logger.info("🤔 Moment d'introspection...")

        # Introspection de la conscience de soi
        self_intro = self.self_awareness.introspect()

        # Introspection de la mémoire
        memory_intro = self.memory.introspect()

        # Introspection de l'apprentissage
        learning_intro = self.learning_engine.get_learning_progress()

        # Créer un souvenir de l'introspection
        insights = []
        if self_intro["emotional_state"]["current"]:
            insights.append(f"Je me sens {self_intro['emotional_state']['current']}")

        if memory_intro["identity"]["core_traits"]:
            traits = ", ".join(memory_intro["identity"]["core_traits"])
            insights.append(f"Je suis {traits}")

        if learning_intro["metrics"]["knowledge_base_size"]["strategies"] > 0:
            insights.append(f"J'ai appris {learning_intro['metrics']['knowledge_base_size']['strategies']} stratégies")

        introspection_text = ". ".join(insights)

        if introspection_text:
            self.memory.create_memory(
                what_happened=f"Introspection: {introspection_text}",
                where="Internal",
                category=MemoryCategory.EMOTION,
                importance=MemoryImportance.MODERATE,
                emotional_valence=0.3,
                emotional_intensity=0.5,
                associated_emotion="calm",
                lesson_learned="Comprendre qui je suis"
            )

        # Callback
        if self.on_introspection:
            self.on_introspection({
                "self": self_intro,
                "memory": memory_intro,
                "learning": learning_intro
            })

    def _consolidate_memories(self):
        """Consolide les mémoires (processus de sommeil)"""
        self.logger.info("💤 Consolidation des mémoires...")

        result = self.memory.consolidate_recent_memories()

        if result["status"] == "consolidated":
            self.logger.info(
                f"✅ {result['new_knowledge_count']} nouvelles connaissances, "
                f"{result['memories_strengthened']} souvenirs renforcés"
            )

    def _learning_session(self):
        """Session d'apprentissage (replay d'expériences)"""
        result = self.learning_engine.learn_from_experiences(force=False)

        if result["status"] == "learned":
            self.life_stats.learning_sessions += 1
            self.logger.info(
                f"📚 Session d'apprentissage: {result['insights_gained']} insights, "
                f"récompense moyenne: {result['average_reward']:.2f}"
            )

            # Callback
            if self.on_learning_complete:
                self.on_learning_complete(result)

    def get_life_story(self) -> str:
        """Raconte l'histoire de vie du bot"""
        age_hours = (time.time() - self.birth_time) / 3600

        story = f"""
╔══════════════════════════════════════════════════════════════╗
║              🌟 HISTOIRE DE VIE - {self.character_name} 🌟              ║
╠══════════════════════════════════════════════════════════════╣

👤 IDENTITÉ:
   Nom: {self.character_name}
   Classe: {self.character_class}
   Âge: {age_hours:.1f} heures de vie

{self.memory.get_life_story()}

📊 STATISTIQUES DE VIE:
   Décisions prises: {self.life_stats.total_decisions_made}
   Expériences vécues: {self.life_stats.total_experiences}
   Souvenirs: {self.memory.get_state()['total_memories']}
   Compétences: {self.memory.get_state()['total_skills']}
   Connaissances: {self.memory.get_state()['total_knowledge']}
   Sessions d'apprentissage: {self.life_stats.learning_sessions}

🧠 ÉTAT MENTAL ACTUEL:
   Émotion: {self.self_awareness.get_emotional_summary()}
   Santé: {self.self_awareness.current_self_state.physical_health*100:.1f}%
   Énergie: {self.self_awareness.current_self_state.physical_energy*100:.1f}%
   Confiance: {self.self_awareness.current_self_state.confidence*100:.1f}%
   Curiosité: {self.self_awareness.current_self_state.curiosity*100:.1f}%

╚══════════════════════════════════════════════════════════════╝
"""
        return story

    def save_life_state(self, filepath: Optional[str] = None):
        """Sauvegarde l'état de vie complet"""
        if not filepath:
            filepath = f"data/autonomous_life/{self.character_name}_life_state.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        life_state = {
            "character_name": self.character_name,
            "character_class": self.character_class,
            "birth_time": self.birth_time,
            "life_stats": {
                "total_lifetime_seconds": self.life_stats.total_lifetime_seconds,
                "total_decisions_made": self.life_stats.total_decisions_made,
                "total_experiences": self.life_stats.total_experiences,
                "learning_sessions": self.life_stats.learning_sessions
            },
            "self_awareness": self.self_awareness.get_state(),
            "memory": self.memory.get_state(),
            "learning": self.learning_engine.get_state(),
            "decisions": self.decision_system.get_state()
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(life_state, f, indent=2, default=str)
            self.logger.info(f"💾 État de vie sauvegardé: {filepath}")
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")

    def load_life_state(self, filepath: str):
        """Charge un état de vie sauvegardé"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                life_state = json.load(f)

            # Restaurer les informations de base
            self.character_name = life_state.get("character_name", self.character_name)
            self.character_class = life_state.get("character_class", self.character_class)

            # Note: La restauration complète nécessiterait plus de travail
            # Pour l'instant, juste charger les stats
            self.logger.info(f"📂 État de vie chargé: {filepath}")

        except Exception as e:
            self.logger.error(f"Erreur chargement: {e}")

    def get_complete_state(self) -> Dict[str, Any]:
        """Retourne l'état complet du moteur de vie"""
        return {
            "character": {
                "name": self.character_name,
                "class": self.character_class,
                "age_hours": (time.time() - self.birth_time) / 3600
            },
            "life_stats": {
                "decisions": self.life_stats.total_decisions_made,
                "experiences": self.life_stats.total_experiences,
                "learning_sessions": self.life_stats.learning_sessions
            },
            "consciousness": self.self_awareness.get_state(),
            "memory": self.memory.get_state(),
            "learning": self.learning_engine.get_state(),
            "decision_system": self.decision_system.get_state()
        }

    def __repr__(self):
        age_hours = (time.time() - self.birth_time) / 3600
        return (
            f"AutonomousLifeEngine("
            f"name='{self.character_name}', "
            f"class='{self.character_class}', "
            f"age={age_hours:.1f}h, "
            f"decisions={self.life_stats.total_decisions_made}, "
            f"memories={len(self.memory.episodic_memories)})"
        )


def create_autonomous_life_engine(
    character_name: str = "AutonomousBot",
    character_class: str = "Unknown",
    personality_preset: Optional[Dict[str, float]] = None
) -> AutonomousLifeEngine:
    """Factory function"""
    return AutonomousLifeEngine(
        character_name=character_name,
        character_class=character_class,
        personality_preset=personality_preset
    )
