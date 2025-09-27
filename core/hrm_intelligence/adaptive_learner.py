"""
Adaptive Learning System pour TacticalBot + HRM
Système d'apprentissage continu inspiré par AlphaGo/MuZero

Auteur: Claude Code
Basé sur: Architecture Serpent.AI + HRM + Self-Play
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import pickle
import threading
from queue import Queue

try:
    from hrm_core import GameState, HRMDecision, HRMBot
except ImportError:
    # Fallback pour imports relatifs
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from hrm_core import GameState, HRMDecision, HRMBot

logger = logging.getLogger(__name__)

@dataclass
class LearningExperience:
    """Expérience d'apprentissage stockée"""
    game_state: GameState
    action_taken: str
    decision_confidence: float
    reward_received: float
    next_state: Optional[GameState]
    outcome_success: bool
    timestamp: float
    session_id: str

@dataclass
class PerformanceMetrics:
    """Métriques de performance du bot"""
    session_start: float
    total_actions: int
    successful_actions: int
    failed_actions: int
    average_reward: float
    learning_rate: float
    exploration_rate: float
    current_strategy: str
    games_won: int
    games_lost: int
    quests_completed: int

class ReplayBuffer:
    """Buffer de replay pour l'apprentissage par expérience"""

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add_experience(self, experience: LearningExperience):
        """Ajoute une expérience au buffer"""
        self.buffer.append(experience)

    def sample_batch(self, batch_size=32) -> List[LearningExperience]:
        """Échantillonne un batch d'expériences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def get_recent_experiences(self, n=100) -> List[LearningExperience]:
        """Récupère les n dernières expériences"""
        return list(self.buffer)[-n:]

    def save_to_disk(self, path: str):
        """Sauvegarde le buffer sur disque"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load_from_disk(self, path: str):
        """Charge le buffer depuis le disque"""
        if Path(path).exists():
            with open(path, 'rb') as f:
                experiences = pickle.load(f)
                self.buffer.extend(experiences)

class HumanLikeBehavior:
    """Module pour rendre le comportement plus humain"""

    def __init__(self):
        self.reaction_times = deque(maxlen=100)
        self.error_patterns = []
        self.fatigue_level = 0.0
        self.skill_progression = {}

    def add_human_delay(self, base_delay=0.2) -> float:
        """Ajoute un délai humain variable"""
        # Délai de base + variation + fatigue
        human_delay = base_delay + np.random.normal(0, 0.1) + (self.fatigue_level * 0.1)
        return max(0.05, human_delay)

    def should_make_error(self) -> bool:
        """Détermine si le bot doit faire une erreur 'humaine'"""
        error_probability = 0.02 + (self.fatigue_level * 0.03)
        return np.random.random() < error_probability

    def update_fatigue(self, session_duration: float):
        """Met à jour le niveau de fatigue"""
        # Fatigue augmente avec le temps, se reset avec les pauses
        self.fatigue_level = min(1.0, session_duration / 7200)  # 2h = fatigue max

    def simulate_human_mouse_movement(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Simule un mouvement de souris humain"""
        path = []
        steps = max(10, int(np.linalg.norm(np.array(end_pos) - np.array(start_pos)) / 20))

        for i in range(steps):
            t = i / (steps - 1)
            # Courbe de Bézier avec variation
            noise_x = np.random.normal(0, 2)
            noise_y = np.random.normal(0, 2)

            x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * t + noise_x)
            y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * t + noise_y)

            path.append((x, y))

        return path

class StrategyEvolution:
    """Évolution des stratégies du bot"""

    def __init__(self):
        self.strategies = {
            "aggressive": {"attack_priority": 0.8, "explore_rate": 0.3, "risk_tolerance": 0.7},
            "defensive": {"attack_priority": 0.3, "explore_rate": 0.1, "risk_tolerance": 0.2},
            "balanced": {"attack_priority": 0.5, "explore_rate": 0.5, "risk_tolerance": 0.5},
            "explorer": {"attack_priority": 0.2, "explore_rate": 0.9, "risk_tolerance": 0.6},
            "questfocused": {"attack_priority": 0.4, "explore_rate": 0.2, "risk_tolerance": 0.3}
        }
        self.current_strategy = "balanced"
        self.strategy_performance = defaultdict(list)

    def evaluate_strategy_performance(self, metrics: PerformanceMetrics) -> float:
        """Évalue la performance de la stratégie actuelle"""
        if metrics.total_actions == 0:
            return 0.0

        success_rate = metrics.successful_actions / metrics.total_actions
        reward_factor = max(0, metrics.average_reward)
        efficiency = metrics.quests_completed / max(1, metrics.total_actions / 100)

        performance_score = (success_rate * 0.4) + (reward_factor * 0.3) + (efficiency * 0.3)
        return performance_score

    def should_switch_strategy(self, current_performance: float) -> bool:
        """Détermine s'il faut changer de stratégie"""
        recent_scores = self.strategy_performance[self.current_strategy][-5:]
        if len(recent_scores) < 3:
            return False

        # Change si performance déclinante ou bloquée
        avg_recent = np.mean(recent_scores)
        return current_performance < avg_recent * 0.8

    def select_new_strategy(self) -> str:
        """Sélectionne une nouvelle stratégie"""
        # Éviter la stratégie actuelle et prioriser les moins testées
        available = [s for s in self.strategies.keys() if s != self.current_strategy]

        # Stratégie avec le moins d'évaluations ou la meilleure performance
        if not any(self.strategy_performance.values()):
            return np.random.choice(available)

        strategy_scores = {}
        for strategy in available:
            recent_perf = self.strategy_performance[strategy][-3:] if strategy in self.strategy_performance else []
            if recent_perf:
                strategy_scores[strategy] = np.mean(recent_perf)
            else:
                strategy_scores[strategy] = 0.5  # Score neutre pour les non-testées

        return max(strategy_scores, key=strategy_scores.get)

class AdaptiveLearner:
    """Système d'apprentissage adaptatif principal"""

    def __init__(self, hrm_bot: HRMBot, learning_config: Dict = None):
        self.hrm_bot = hrm_bot
        self.config = learning_config or self._default_config()

        # Composants d'apprentissage
        self.replay_buffer = ReplayBuffer(max_size=self.config["buffer_size"])
        self.human_behavior = HumanLikeBehavior()
        self.strategy_evolution = StrategyEvolution()

        # Métriques et tracking
        self.performance_metrics = PerformanceMetrics(
            session_start=time.time(),
            total_actions=0,
            successful_actions=0,
            failed_actions=0,
            average_reward=0.0,
            learning_rate=self.config["learning_rate"],
            exploration_rate=self.config["exploration_rate"],
            current_strategy="balanced",
            games_won=0,
            games_lost=0,
            quests_completed=0
        )

        # Threading pour apprentissage asynchrone
        self.learning_queue = Queue()
        self.learning_thread = None
        self.is_learning = False

        # Sauvegarde automatique
        self.last_save_time = time.time()
        self.save_directory = Path(self.config["save_directory"])
        self.save_directory.mkdir(exist_ok=True)

        logger.info("Adaptive Learner initialisé")

    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "learning_rate": 0.001,
            "exploration_rate": 0.15,
            "buffer_size": 10000,
            "batch_size": 32,
            "save_interval": 300,  # 5 minutes
            "strategy_eval_interval": 100,  # 100 actions
            "save_directory": "G:/Botting/data/learning",
            "human_behavior_enabled": True,
            "self_play_enabled": False
        }

    def start_learning_session(self):
        """Démarre une session d'apprentissage"""
        self.is_learning = True
        self.performance_metrics.session_start = time.time()

        # Charger les expériences précédentes
        self._load_previous_session()

        # Démarrer le thread d'apprentissage asynchrone
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()

        logger.info("Session d'apprentissage démarrée")

    def stop_learning_session(self):
        """Arrête la session d'apprentissage"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)

        # Sauvegarde finale
        self._save_session()
        logger.info("Session d'apprentissage terminée")

    def process_game_action(self, game_state: GameState, decision: HRMDecision,
                          outcome_success: bool, reward: float = 0.0,
                          next_state: Optional[GameState] = None) -> HRMDecision:
        """Traite une action du jeu et apprend de l'expérience"""

        # Créer l'expérience d'apprentissage
        experience = LearningExperience(
            game_state=game_state,
            action_taken=decision.action,
            decision_confidence=decision.confidence,
            reward_received=reward,
            next_state=next_state,
            outcome_success=outcome_success,
            timestamp=time.time(),
            session_id=f"session_{int(self.performance_metrics.session_start)}"
        )

        # Ajouter au buffer de replay
        self.replay_buffer.add_experience(experience)

        # Mettre à jour les métriques
        self._update_performance_metrics(outcome_success, reward)

        # Ajouter à la queue d'apprentissage
        self.learning_queue.put(experience)

        # Comportement humain
        modified_decision = self._apply_human_behavior(decision)

        # Évaluation périodique de stratégie
        if self.performance_metrics.total_actions % self.config["strategy_eval_interval"] == 0:
            self._evaluate_and_adapt_strategy()

        # Sauvegarde périodique
        if time.time() - self.last_save_time > self.config["save_interval"]:
            self._save_session()

        return modified_decision

    def _apply_human_behavior(self, decision: HRMDecision) -> HRMDecision:
        """Applique des comportements humains à la décision"""
        if not self.config["human_behavior_enabled"]:
            return decision

        # Mise à jour de la fatigue
        session_duration = time.time() - self.performance_metrics.session_start
        self.human_behavior.update_fatigue(session_duration)

        # Ajout de délai humain
        human_delay = self.human_behavior.add_human_delay()
        time.sleep(human_delay)

        # Erreur humaine occasionnelle
        if self.human_behavior.should_make_error():
            # Modifier légèrement la décision ou ajouter une action "d'erreur"
            modified_decision = HRMDecision(
                action="wait",  # Action neutre en cas d'erreur
                confidence=decision.confidence * 0.7,
                reasoning_path=decision.reasoning_path + ["Erreur humaine simulée"],
                expected_outcome="Correction d'erreur",
                priority=1,
                execution_time=decision.execution_time + human_delay
            )
            return modified_decision

        return decision

    def _update_performance_metrics(self, success: bool, reward: float):
        """Met à jour les métriques de performance"""
        self.performance_metrics.total_actions += 1

        if success:
            self.performance_metrics.successful_actions += 1
        else:
            self.performance_metrics.failed_actions += 1

        # Moyenne mobile des rewards
        old_avg = self.performance_metrics.average_reward
        new_avg = (old_avg * 0.95) + (reward * 0.05)
        self.performance_metrics.average_reward = new_avg

    def _evaluate_and_adapt_strategy(self):
        """Évalue et adapte la stratégie actuelle"""
        current_performance = self.strategy_evolution.evaluate_strategy_performance(self.performance_metrics)

        # Enregistrer la performance de la stratégie actuelle
        current_strategy = self.strategy_evolution.current_strategy
        self.strategy_evolution.strategy_performance[current_strategy].append(current_performance)

        # Décider si changer de stratégie
        if self.strategy_evolution.should_switch_strategy(current_performance):
            new_strategy = self.strategy_evolution.select_new_strategy()
            self.strategy_evolution.current_strategy = new_strategy
            self.performance_metrics.current_strategy = new_strategy

            logger.info(f"Changement de stratégie: {current_strategy} -> {new_strategy} (perf: {current_performance:.3f})")

    def _learning_loop(self):
        """Boucle d'apprentissage asynchrone"""
        while self.is_learning:
            try:
                # Traiter les expériences en attente
                experiences_to_process = []
                while not self.learning_queue.empty() and len(experiences_to_process) < self.config["batch_size"]:
                    experiences_to_process.append(self.learning_queue.get_nowait())

                if experiences_to_process:
                    self._process_learning_batch(experiences_to_process)

                time.sleep(0.1)  # Éviter la surcharge CPU

            except Exception as e:
                logger.error(f"Erreur dans la boucle d'apprentissage: {e}")

    def _process_learning_batch(self, experiences: List[LearningExperience]):
        """Traite un batch d'expériences pour l'apprentissage"""
        # Analyser les patterns de succès/échec
        successful_experiences = [exp for exp in experiences if exp.outcome_success]
        failed_experiences = [exp for exp in experiences if not exp.outcome_success]

        # TODO: Implémenter l'apprentissage par renforcement avec HRM
        # Ici on pourrait:
        # 1. Réentraîner les couches du HRM sur les expériences positives
        # 2. Ajuster les paramètres de décision
        # 3. Modifier les poids d'attention selon les patterns découverts

        logger.debug(f"Batch traité: {len(successful_experiences)} succès, {len(failed_experiences)} échecs")

    def _save_session(self):
        """Sauvegarde la session d'apprentissage"""
        timestamp = int(time.time())

        # Sauvegarder le buffer de replay
        buffer_path = self.save_directory / f"replay_buffer_{timestamp}.pkl"
        self.replay_buffer.save_to_disk(str(buffer_path))

        # Sauvegarder les métriques
        metrics_path = self.save_directory / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(asdict(self.performance_metrics), f, indent=2)

        # Sauvegarder le modèle HRM
        model_path = self.save_directory / f"hrm_model_{timestamp}.pth"
        self.hrm_bot.save_model(str(model_path))

        # Sauvegarder les stratégies
        strategy_path = self.save_directory / f"strategies_{timestamp}.json"
        with open(strategy_path, 'w') as f:
            json.dump({
                "current_strategy": self.strategy_evolution.current_strategy,
                "strategy_performance": dict(self.strategy_evolution.strategy_performance),
                "strategies": self.strategy_evolution.strategies
            }, f, indent=2)

        self.last_save_time = time.time()
        logger.info(f"Session sauvegardée: {timestamp}")

    def _load_previous_session(self):
        """Charge la session précédente si disponible"""
        if not self.save_directory.exists():
            return

        # Trouver les fichiers les plus récents
        buffer_files = list(self.save_directory.glob("replay_buffer_*.pkl"))
        metrics_files = list(self.save_directory.glob("metrics_*.json"))
        model_files = list(self.save_directory.glob("hrm_model_*.pth"))
        strategy_files = list(self.save_directory.glob("strategies_*.json"))

        if buffer_files:
            latest_buffer = max(buffer_files, key=lambda p: p.stat().st_mtime)
            self.replay_buffer.load_from_disk(str(latest_buffer))
            logger.info(f"Buffer de replay chargé: {len(self.replay_buffer.buffer)} expériences")

        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self.hrm_bot.load_model(str(latest_model))
            logger.info(f"Modèle HRM chargé: {latest_model}")

        if strategy_files:
            latest_strategy = max(strategy_files, key=lambda p: p.stat().st_mtime)
            with open(latest_strategy, 'r') as f:
                strategy_data = json.load(f)
                self.strategy_evolution.current_strategy = strategy_data["current_strategy"]
                self.strategy_evolution.strategy_performance = defaultdict(list, strategy_data["strategy_performance"])
            logger.info(f"Stratégies chargées: {latest_strategy}")

    def get_learning_report(self) -> Dict:
        """Génère un rapport d'apprentissage"""
        session_duration = time.time() - self.performance_metrics.session_start

        return {
            "session_duration_hours": session_duration / 3600,
            "performance_metrics": asdict(self.performance_metrics),
            "current_strategy": self.strategy_evolution.current_strategy,
            "strategy_performance": dict(self.strategy_evolution.strategy_performance),
            "replay_buffer_size": len(self.replay_buffer.buffer),
            "learning_queue_size": self.learning_queue.qsize(),
            "fatigue_level": self.human_behavior.fatigue_level,
            "recent_experiences": len(self.replay_buffer.get_recent_experiences(50))
        }