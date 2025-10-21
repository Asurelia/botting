"""
Tests pour le système d'apprentissage autonome
"""

import pytest
import time
from pathlib import Path
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.autonomous_learning import (
    create_self_awareness_engine,
    create_continuous_learning_engine,
    create_autobiographical_memory,
    create_emergent_decision_system,
    create_autonomous_life_engine,
    ExperienceType,
    MemoryCategory,
    MemoryImportance
)
from core.game_loop.game_state import create_game_state


class TestSelfAwareness:
    """Tests pour le moteur de conscience de soi"""

    def test_creation(self):
        """Test de création du moteur"""
        engine = create_self_awareness_engine()
        assert engine is not None
        assert engine.current_self_state is not None

    def test_consciousness_update(self):
        """Test de mise à jour de la conscience"""
        engine = create_self_awareness_engine()
        game_state = create_game_state()

        # Mettre à jour la conscience
        self_state = engine.update_consciousness(game_state, [])

        assert self_state is not None
        assert 0.0 <= self_state.physical_health <= 1.0
        assert 0.0 <= self_state.cognitive_load <= 1.0

    def test_emotional_state(self):
        """Test des états émotionnels"""
        engine = create_self_awareness_engine()
        game_state = create_game_state()

        # Simuler HP bas
        game_state.character.hp_percent = 25.0

        self_state = engine.update_consciousness(game_state, [])

        # Devrait être anxieux avec HP bas
        assert self_state.current_emotion.value in ["anxious", "focused"]

    def test_introspection(self):
        """Test d'introspection"""
        engine = create_self_awareness_engine()

        intro = engine.introspect()

        assert "identity" in intro
        assert "physical_state" in intro
        assert "mental_state" in intro
        assert "emotional_state" in intro


class TestContinuousLearning:
    """Tests pour le moteur d'apprentissage continu"""

    def test_creation(self):
        """Test de création"""
        engine = create_continuous_learning_engine()
        assert engine is not None

    def test_experience_recording(self):
        """Test d'enregistrement d'expérience"""
        engine = create_continuous_learning_engine()

        experience = engine.record_experience(
            state_before={"hp": 100},
            action="attack",
            state_after={"hp": 95},
            reward=0.5,
            experience_type=ExperienceType.SUCCESS
        )

        assert experience is not None
        assert experience.action_taken == "attack"
        assert experience.reward >= 0.5  # Peut inclure curiosity reward

    def test_learning_session(self):
        """Test de session d'apprentissage"""
        engine = create_continuous_learning_engine()

        # Ajouter plusieurs expériences
        for i in range(15):
            engine.record_experience(
                state_before={"hp": 100},
                action=f"action_{i}",
                state_after={"hp": 95},
                reward=0.5,
                experience_type=ExperienceType.SUCCESS
            )

        # Session d'apprentissage
        result = engine.learn_from_experiences(force=True)

        assert result["status"] == "learned"
        assert result["experiences_processed"] > 0


class TestAutobiographicalMemory:
    """Tests pour la mémoire autobiographique"""

    def test_creation(self):
        """Test de création"""
        memory = create_autobiographical_memory()
        assert memory is not None

    def test_memory_creation(self):
        """Test de création de souvenir"""
        memory = create_autobiographical_memory()

        mem = memory.create_memory(
            what_happened="Premier combat gagné!",
            where="Astrub",
            category=MemoryCategory.COMBAT,
            importance=MemoryImportance.SIGNIFICANT,
            emotional_valence=0.8,
            emotional_intensity=0.9,
            associated_emotion="confident"
        )

        assert mem is not None
        assert mem.what_happened == "Premier combat gagné!"
        assert len(memory.episodic_memories) > 0  # Inclut le souvenir de naissance

    def test_memory_recall(self):
        """Test de rappel de souvenir"""
        memory = create_autobiographical_memory()

        mem = memory.create_memory(
            what_happened="Test memory",
            where="Test location",
            category=MemoryCategory.LEARNING,
            importance=MemoryImportance.MODERATE
        )

        # Rappeler le souvenir
        recalled = memory.recall_memory(mem.memory_id)

        assert recalled is not None
        assert recalled.recall_count == 1

    def test_skill_learning(self):
        """Test d'apprentissage de compétence"""
        memory = create_autobiographical_memory()

        memory.learn_skill("Combat", "Skill de combat")

        assert "Combat" in memory.procedural_memories

        # Pratiquer
        memory.practice_skill("Combat", success=True)
        memory.practice_skill("Combat", success=True)

        skill = memory.procedural_memories["Combat"]
        assert skill.practice_count == 2
        assert skill.mastery_level > 0


class TestEmergentDecisionSystem:
    """Tests pour le système de décision émergente"""

    def test_creation(self):
        """Test de création"""
        self_awareness = create_self_awareness_engine()
        learning = create_continuous_learning_engine()
        memory = create_autobiographical_memory()

        decision_system = create_emergent_decision_system(
            self_awareness,
            learning,
            memory
        )

        assert decision_system is not None

    def test_decision_making(self):
        """Test de prise de décision"""
        self_awareness = create_self_awareness_engine()
        learning = create_continuous_learning_engine()
        memory = create_autobiographical_memory()

        decision_system = create_emergent_decision_system(
            self_awareness,
            learning,
            memory
        )

        game_state = create_game_state()

        # Prendre une décision
        decision = decision_system.decide(game_state)

        # Peut être None si aucune action nécessaire
        if decision:
            assert decision.action_type is not None
            assert decision.reasoning is not None
            assert 0.0 <= decision.confidence <= 1.0


class TestAutonomousLifeEngine:
    """Tests pour le moteur de vie autonome"""

    def test_creation(self):
        """Test de création"""
        life_engine = create_autonomous_life_engine(
            character_name="TestBot",
            character_class="Iop"
        )

        assert life_engine is not None
        assert life_engine.character_name == "TestBot"
        assert life_engine.character_class == "Iop"

    def test_live_moment(self):
        """Test d'un moment de vie"""
        life_engine = create_autonomous_life_engine()
        game_state = create_game_state()

        # Vivre un moment
        decision = life_engine.live_moment(game_state)

        # Peut retourner None ou une décision
        assert decision is None or "action_type" in decision

    def test_record_outcome(self):
        """Test d'enregistrement de résultat"""
        life_engine = create_autonomous_life_engine()
        game_state = create_game_state()

        # Vivre un moment
        decision = life_engine.live_moment(game_state)

        if decision and "decision_obj" in decision:
            # Enregistrer un résultat
            life_engine.record_outcome(
                decision["decision_obj"],
                "success",
                reward=0.8
            )

            assert life_engine.life_stats.total_experiences >= 1

    def test_life_story(self):
        """Test de génération d'histoire de vie"""
        life_engine = create_autonomous_life_engine(
            character_name="StoryBot"
        )

        story = life_engine.get_life_story()

        assert "StoryBot" in story
        assert "HISTOIRE DE VIE" in story

    def test_state_persistence(self):
        """Test de sauvegarde d'état"""
        life_engine = create_autonomous_life_engine()

        # Sauvegarder
        test_path = "data/autonomous_life/test_state.json"
        life_engine.save_life_state(test_path)

        # Vérifier que le fichier existe
        assert Path(test_path).exists()

        # Nettoyer
        Path(test_path).unlink()


def test_integration_scenario():
    """Test d'un scénario complet d'intégration"""

    # Créer le bot
    life_engine = create_autonomous_life_engine(
        character_name="IntegrationBot",
        character_class="Iop",
        personality_preset={
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.7,
            "neuroticism": 0.3
        }
    )

    game_state = create_game_state()

    # Simuler plusieurs moments de vie
    for i in range(10):
        decision = life_engine.live_moment(game_state)

        if decision and "decision_obj" in decision:
            # Simuler un résultat
            outcome = "success" if i % 2 == 0 else "failure"
            reward = 0.7 if outcome == "success" else -0.3

            life_engine.record_outcome(
                decision["decision_obj"],
                outcome,
                reward
            )

        time.sleep(0.1)

    # Vérifier les statistiques
    assert life_engine.life_stats.total_decisions_made > 0
    assert life_engine.life_stats.consciousness_updates >= 10

    # Vérifier l'histoire
    story = life_engine.get_life_story()
    assert "IntegrationBot" in story

    # Vérifier l'état
    state = life_engine.get_complete_state()
    assert "character" in state
    assert "life_stats" in state
    assert "consciousness" in state
    assert "memory" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
