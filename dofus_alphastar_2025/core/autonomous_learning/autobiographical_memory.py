"""
Autobiographical Memory - Mémoire autobiographique
Le bot se souvient de son histoire de vie dans le monde de DOFUS

Inspiré de la mémoire épisodique humaine:
- Mémoire des événements vécus
- Mémoire sémantique (connaissances générales)
- Mémoire procédurale (compétences apprises)
- Construction de l'identité à travers les souvenirs

Fonctionnalités:
- Stockage des souvenirs importants (moments marquants)
- Organisation temporelle des souvenirs (timeline de vie)
- Rappel de souvenirs similaires (associations)
- Construction de narratifs personnels
- Évolution de l'identité basée sur l'histoire
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
import pickle


logger = logging.getLogger(__name__)


class MemoryImportance(Enum):
    """Importance d'un souvenir"""
    TRIVIAL = 1      # Routine quotidienne
    MINOR = 2        # Petit événement
    MODERATE = 3     # Événement notable
    SIGNIFICANT = 4  # Événement important
    MILESTONE = 5    # Moment marquant de la vie


class MemoryCategory(Enum):
    """Catégories de souvenirs"""
    COMBAT = "combat"              # Combats mémorables
    EXPLORATION = "exploration"    # Découvertes
    SOCIAL = "social"              # Interactions sociales
    ACHIEVEMENT = "achievement"    # Accomplissements
    FAILURE = "failure"            # Échecs marquants
    LEARNING = "learning"          # Apprentissages
    EMOTION = "emotion"            # Moments émotionnels forts
    MILESTONE = "milestone"        # Jalons de progression


@dataclass
class EpisodicMemory:
    """
    Mémoire épisodique - Un souvenir d'événement vécu
    Équivalent d'un souvenir humain avec contexte émotionnel
    """
    memory_id: str
    timestamp: float
    category: MemoryCategory
    importance: MemoryImportance

    # Contenu du souvenir (l'événement)
    what_happened: str            # Description textuelle
    where: str                    # Lieu (map)
    who: List[str]                # Entités impliquées
    context: Dict[str, Any]       # Contexte complet

    # Dimension émotionnelle
    emotional_valence: float      # -1.0 (négatif) à +1.0 (positif)
    emotional_intensity: float    # 0.0 à 1.0
    associated_emotion: str       # Émotion ressentie

    # Dimension cognitive
    lesson_learned: str = ""      # Ce qui a été appris
    impact_on_identity: float = 0.0  # Impact sur l'identité

    # Métadonnées
    recall_count: int = 0         # Nombre de fois rappelé
    last_recalled: float = 0.0
    memory_strength: float = 1.0  # Décroît avec le temps
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """
    Mémoire sémantique - Connaissance générale
    Ce que le bot "sait" sur le monde
    """
    concept: str
    definition: str
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.5
    learned_from: List[str] = field(default_factory=list)  # IDs de mémoires épisodiques
    last_updated: float = field(default_factory=time.time)


@dataclass
class ProceduralMemory:
    """
    Mémoire procédurale - Compétences et habitudes
    "Comment faire" quelque chose
    """
    skill_name: str
    description: str
    mastery_level: float = 0.0    # 0.0 = débutant, 1.0 = expert
    practice_count: int = 0
    success_rate: float = 0.0
    last_practiced: float = field(default_factory=time.time)
    improvement_rate: float = 0.0


@dataclass
class LifeChapter:
    """
    Chapitre de vie - Organisation temporelle des souvenirs
    Comme les humains organisent leur vie en périodes
    """
    chapter_name: str
    start_time: float
    end_time: Optional[float] = None
    theme: str = ""                # Thème du chapitre (ex: "apprentissage du combat")
    key_memories: List[str] = field(default_factory=list)  # IDs des souvenirs marquants
    achievements: List[str] = field(default_factory=list)
    challenges_faced: List[str] = field(default_factory=list)
    growth_summary: str = ""


class MemoryConsolidation:
    """
    Consolidation de mémoire - Processus de renforcement des souvenirs importants
    Similaire au sommeil humain qui consolide les mémoires
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MemoryConsolidation")

    def consolidate_memories(
        self,
        recent_memories: List[EpisodicMemory],
        existing_knowledge: Dict[str, SemanticMemory]
    ) -> Tuple[List[SemanticMemory], List[str]]:
        """
        Consolide les mémoires récentes en connaissances générales
        Transforme les expériences en apprentissages
        """
        new_knowledge = []
        memories_to_strengthen = []

        # Regrouper les mémoires similaires
        memory_clusters = self._cluster_similar_memories(recent_memories)

        for cluster in memory_clusters:
            # Extraire un pattern général
            pattern = self._extract_pattern(cluster)

            if pattern:
                # Créer une connaissance sémantique
                semantic_memory = SemanticMemory(
                    concept=pattern["concept"],
                    definition=pattern["definition"],
                    related_concepts=pattern["related"],
                    confidence=pattern["confidence"],
                    learned_from=[m.memory_id for m in cluster]
                )
                new_knowledge.append(semantic_memory)

                # Renforcer les mémoires du cluster
                for memory in cluster:
                    if memory.importance.value >= MemoryImportance.MODERATE.value:
                        memories_to_strengthen.append(memory.memory_id)

        return new_knowledge, memories_to_strengthen

    def _cluster_similar_memories(self, memories: List[EpisodicMemory]) -> List[List[EpisodicMemory]]:
        """Regroupe les mémoires similaires"""
        clusters = defaultdict(list)

        for memory in memories:
            # Clé de clustering: catégorie + lieu
            cluster_key = f"{memory.category.value}_{memory.where}"
            clusters[cluster_key].append(memory)

        return [cluster for cluster in clusters.values() if len(cluster) >= 2]

    def _extract_pattern(self, memories: List[EpisodicMemory]) -> Optional[Dict[str, Any]]:
        """Extrait un pattern général d'un groupe de mémoires"""
        if not memories:
            return None

        # Exemple: Si plusieurs combats dans le même lieu ont été gagnés
        category = memories[0].category
        location = memories[0].where

        positive_outcomes = sum(1 for m in memories if m.emotional_valence > 0)
        total = len(memories)

        if positive_outcomes / total > 0.7:
            return {
                "concept": f"{category.value}_strategy_{location}",
                "definition": f"Stratégie efficace pour {category.value} dans {location}",
                "related": [category.value, location],
                "confidence": positive_outcomes / total
            }

        return None


class AutobiographicalMemory:
    """
    Système de mémoire autobiographique complet

    Le bot construit son histoire de vie à travers:
    1. Mémoires épisodiques (événements vécus)
    2. Mémoires sémantiques (connaissances)
    3. Mémoires procédurales (compétences)
    4. Organisation narrative (chapitres de vie)
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AutobiographicalMemory")

        # Mémoires épisodiques (événements vécus)
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.episodic_timeline = []  # Timeline chronologique

        # Mémoires sémantiques (connaissances)
        self.semantic_memories: Dict[str, SemanticMemory] = {}

        # Mémoires procédurales (compétences)
        self.procedural_memories: Dict[str, ProceduralMemory] = {}

        # Organisation narrative
        self.life_chapters: List[LifeChapter] = []
        self.current_chapter: Optional[LifeChapter] = None

        # Système de consolidation
        self.consolidation_system = MemoryConsolidation()

        # Identité construite à partir des souvenirs
        self.identity_narrative = {
            "self_concept": "Je commence mon aventure dans le monde de DOFUS",
            "core_traits": [],
            "defining_moments": [],
            "life_goals_evolved": []
        }

        # Statistiques
        self.memory_stats = {
            "total_memories": 0,
            "memories_by_category": defaultdict(int),
            "memories_by_importance": defaultdict(int),
            "consolidations_performed": 0,
            "identity_updates": 0
        }

        # Configuration
        self.config = {
            "memory_decay_rate": 0.001,      # Décroissance de la force des souvenirs
            "consolidation_interval": 3600,  # Consolidation toutes les heures
            "max_episodic_memories": 5000,
            "importance_threshold": MemoryImportance.MODERATE
        }

        self.last_consolidation = time.time()

        # Démarrer le premier chapitre
        self._start_new_chapter("Début de l'aventure", "Premiers pas dans le monde de DOFUS")

        self.logger.info("Autobiographical Memory initialized - Je commence à écrire mon histoire...")

    def create_memory(
        self,
        what_happened: str,
        where: str,
        category: MemoryCategory,
        importance: MemoryImportance,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.5,
        associated_emotion: str = "neutral",
        context: Dict[str, Any] = None,
        who: List[str] = None,
        lesson_learned: str = ""
    ) -> EpisodicMemory:
        """
        Crée un nouveau souvenir épisodique
        Équivalent de "vivre un moment" qui sera retenu
        """
        memory_id = f"mem_{int(time.time()*1000)}_{len(self.episodic_memories)}"

        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=time.time(),
            category=category,
            importance=importance,
            what_happened=what_happened,
            where=where,
            who=who or [],
            context=context or {},
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            associated_emotion=associated_emotion,
            lesson_learned=lesson_learned
        )

        # Stocker le souvenir
        self.episodic_memories[memory_id] = memory
        self.episodic_timeline.append(memory_id)

        # Ajouter au chapitre actuel
        if self.current_chapter and importance.value >= MemoryImportance.MODERATE.value:
            self.current_chapter.key_memories.append(memory_id)

        # Mise à jour des statistiques
        self.memory_stats["total_memories"] += 1
        self.memory_stats["memories_by_category"][category.value] += 1
        self.memory_stats["memories_by_importance"][importance.value] += 1

        # Log des souvenirs importants
        if importance.value >= MemoryImportance.SIGNIFICANT.value:
            self.logger.info(f"📝 Souvenir important créé: {what_happened} ({category.value})")

            # Impact sur l'identité pour les jalons
            if importance == MemoryImportance.MILESTONE:
                self._update_identity_from_memory(memory)

        return memory

    def recall_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """
        Rappelle un souvenir (accès conscient)
        Le fait de rappeler renforce le souvenir
        """
        if memory_id not in self.episodic_memories:
            return None

        memory = self.episodic_memories[memory_id]

        # Renforcer le souvenir en le rappelant
        memory.recall_count += 1
        memory.last_recalled = time.time()
        memory.memory_strength = min(1.0, memory.memory_strength + 0.1)

        return memory

    def recall_similar_memories(
        self,
        current_context: Dict[str, Any],
        limit: int = 5
    ) -> List[EpisodicMemory]:
        """
        Rappelle des souvenirs similaires au contexte actuel
        Équivalent de "ça me rappelle quand..."
        """
        current_map = current_context.get("current_map", "")
        current_category = current_context.get("category", "")

        similar_memories = []

        for memory in self.episodic_memories.values():
            similarity_score = 0.0

            # Similitude de lieu
            if memory.where == current_map:
                similarity_score += 0.5

            # Similitude de catégorie
            if memory.category.value == current_category:
                similarity_score += 0.3

            # Force du souvenir
            similarity_score *= memory.memory_strength

            if similarity_score > 0.3:
                similar_memories.append((similarity_score, memory))

        # Trier par similarité
        similar_memories.sort(key=lambda x: x[0], reverse=True)

        return [mem for _, mem in similar_memories[:limit]]

    def learn_skill(self, skill_name: str, description: str):
        """Apprend une nouvelle compétence (mémoire procédurale)"""
        if skill_name not in self.procedural_memories:
            self.procedural_memories[skill_name] = ProceduralMemory(
                skill_name=skill_name,
                description=description
            )
            self.logger.info(f"🎓 Nouvelle compétence apprise: {skill_name}")

    def practice_skill(self, skill_name: str, success: bool):
        """Pratique une compétence (amélioration)"""
        if skill_name not in self.procedural_memories:
            return

        skill = self.procedural_memories[skill_name]
        skill.practice_count += 1
        skill.last_practiced = time.time()

        # Mise à jour du taux de succès
        old_success_rate = skill.success_rate
        skill.success_rate = (skill.success_rate * (skill.practice_count - 1) + (1.0 if success else 0.0)) / skill.practice_count

        # Amélioration de la maîtrise
        if success:
            improvement = 0.01 * (1.0 - skill.mastery_level)  # Plus difficile d'améliorer quand on est bon
            skill.mastery_level = min(1.0, skill.mastery_level + improvement)
            skill.improvement_rate = skill.mastery_level - old_success_rate

    def add_knowledge(self, concept: str, definition: str, related: List[str] = None):
        """Ajoute une connaissance sémantique"""
        self.semantic_memories[concept] = SemanticMemory(
            concept=concept,
            definition=definition,
            related_concepts=related or []
        )

    def consolidate_recent_memories(self) -> Dict[str, Any]:
        """
        Consolide les mémoires récentes (comme le sommeil humain)
        Transforme les expériences en connaissances durables
        """
        current_time = time.time()

        # Vérifier l'intervalle
        if current_time - self.last_consolidation < self.config["consolidation_interval"]:
            return {"status": "not_needed"}

        self.logger.info("🧠 Consolidation des mémoires en cours...")

        # Récupérer les mémoires récentes (dernière heure)
        recent_cutoff = current_time - 3600
        recent_memories = [
            mem for mem in self.episodic_memories.values()
            if mem.timestamp > recent_cutoff
        ]

        if not recent_memories:
            return {"status": "no_recent_memories"}

        # Consolider
        new_knowledge, strengthened = self.consolidation_system.consolidate_memories(
            recent_memories,
            self.semantic_memories
        )

        # Ajouter les nouvelles connaissances
        for knowledge in new_knowledge:
            self.semantic_memories[knowledge.concept] = knowledge

        # Renforcer les mémoires importantes
        for mem_id in strengthened:
            if mem_id in self.episodic_memories:
                self.episodic_memories[mem_id].memory_strength = min(
                    1.0,
                    self.episodic_memories[mem_id].memory_strength + 0.2
                )

        # Décroissance naturelle des autres mémoires
        self._apply_memory_decay()

        self.memory_stats["consolidations_performed"] += 1
        self.last_consolidation = current_time

        return {
            "status": "consolidated",
            "new_knowledge_count": len(new_knowledge),
            "memories_strengthened": len(strengthened),
            "memories_processed": len(recent_memories)
        }

    def _apply_memory_decay(self):
        """Applique la décroissance naturelle des souvenirs (oubli)"""
        for memory in self.episodic_memories.values():
            # Les souvenirs importants résistent mieux à l'oubli
            resistance = memory.importance.value / 5.0
            decay = self.config["memory_decay_rate"] * (1.0 - resistance)

            memory.memory_strength = max(0.0, memory.memory_strength - decay)

    def _start_new_chapter(self, chapter_name: str, theme: str):
        """Démarre un nouveau chapitre de vie"""
        # Clôturer le chapitre actuel si existe
        if self.current_chapter:
            self.current_chapter.end_time = time.time()
            self.current_chapter.growth_summary = self._summarize_chapter_growth(self.current_chapter)

        # Nouveau chapitre
        new_chapter = LifeChapter(
            chapter_name=chapter_name,
            start_time=time.time(),
            theme=theme
        )

        self.life_chapters.append(new_chapter)
        self.current_chapter = new_chapter

        self.logger.info(f"📖 Nouveau chapitre de vie: {chapter_name}")

    def _summarize_chapter_growth(self, chapter: LifeChapter) -> str:
        """Résume la croissance pendant un chapitre"""
        # Compter les accomplissements
        achievements = len(chapter.achievements)
        challenges = len(chapter.challenges_faced)
        key_moments = len(chapter.key_memories)

        return f"Chapitre avec {achievements} accomplissements, {challenges} défis, et {key_moments} moments marquants"

    def _update_identity_from_memory(self, memory: EpisodicMemory):
        """Met à jour l'identité basée sur un souvenir marquant"""
        # Ajouter aux moments déterminants
        if memory.memory_id not in self.identity_narrative["defining_moments"]:
            self.identity_narrative["defining_moments"].append(memory.memory_id)

        # Extraire un trait de personnalité si pertinent
        if memory.category == MemoryCategory.ACHIEVEMENT and memory.emotional_valence > 0.5:
            if "accomplished" not in self.identity_narrative["core_traits"]:
                self.identity_narrative["core_traits"].append("accomplished")

        elif memory.category == MemoryCategory.EXPLORATION and memory.emotional_intensity > 0.6:
            if "curious" not in self.identity_narrative["core_traits"]:
                self.identity_narrative["core_traits"].append("curious")

        elif memory.category == MemoryCategory.SOCIAL and memory.emotional_valence > 0.5:
            if "social" not in self.identity_narrative["core_traits"]:
                self.identity_narrative["core_traits"].append("social")

        # Mise à jour du concept de soi
        traits_text = ", ".join(self.identity_narrative["core_traits"][:3])
        if traits_text:
            self.identity_narrative["self_concept"] = f"Je suis quelqu'un de {traits_text}"

        self.memory_stats["identity_updates"] += 1

    def get_life_story(self) -> str:
        """Génère un récit de vie basé sur les souvenirs"""
        story_parts = []

        story_parts.append(f"Mon concept de moi: {self.identity_narrative['self_concept']}")
        story_parts.append(f"\nJ'ai vécu {len(self.life_chapters)} chapitres de vie:")

        for chapter in self.life_chapters:
            duration = (chapter.end_time or time.time()) - chapter.start_time
            duration_hours = duration / 3600

            story_parts.append(
                f"\n- {chapter.chapter_name} ({duration_hours:.1f}h): {chapter.theme}"
            )

            if chapter.key_memories:
                story_parts.append(f"  Moments marquants: {len(chapter.key_memories)}")

        story_parts.append(f"\nCompétences maîtrisées: {len(self.procedural_memories)}")
        story_parts.append(f"Connaissances acquises: {len(self.semantic_memories)}")
        story_parts.append(f"Souvenirs totaux: {len(self.episodic_memories)}")

        return "\n".join(story_parts)

    def introspect(self) -> Dict[str, Any]:
        """Introspection de la mémoire autobiographique"""
        return {
            "identity": self.identity_narrative,
            "life_chapters": len(self.life_chapters),
            "current_chapter": self.current_chapter.chapter_name if self.current_chapter else None,
            "episodic_memories": len(self.episodic_memories),
            "semantic_knowledge": len(self.semantic_memories),
            "skills_learned": len(self.procedural_memories),
            "memory_stats": dict(self.memory_stats),
            "recent_important_memories": [
                mem.what_happened for mem in sorted(
                    self.episodic_memories.values(),
                    key=lambda m: m.timestamp,
                    reverse=True
                )[:5] if mem.importance.value >= MemoryImportance.SIGNIFICANT.value
            ]
        }

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état complet de la mémoire"""
        return {
            "total_memories": len(self.episodic_memories),
            "total_knowledge": len(self.semantic_memories),
            "total_skills": len(self.procedural_memories),
            "chapters": len(self.life_chapters),
            "identity_traits": self.identity_narrative["core_traits"],
            "stats": dict(self.memory_stats)
        }


def create_autobiographical_memory() -> AutobiographicalMemory:
    """Factory function"""
    return AutobiographicalMemory()
