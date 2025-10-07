"""
Ray RLlib Trainer pour DOFUS AlphaStar
Entraînement distribué optimisé AMD avec support multi-agent
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.air import RunConfig, ScalingConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper

import torch
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import asdict

from .rl_config import RLlibConfig, get_training_config, create_league_config
from ..networks.alphastar_model import AlphaStarModel, AlphaStarHRMModel
from ..hrm_reasoning import DofusHRMAgent
from config import config

logger = logging.getLogger(__name__)

class RLlibTrainer:
    """Trainer Ray RLlib pour DOFUS AlphaStar avec optimisations AMD"""

    def __init__(self,
                 training_config: Optional[RLlibConfig] = None,
                 experiment_name: str = "dofus_alphastar",
                 results_dir: str = "results"):

        self.config = training_config or get_training_config("rllib")
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Ray initialization
        self._init_ray()

        # Register custom models
        self._register_models()

        # Algorithm instance
        self.algorithm = None

        # Training state
        self.training_iteration = 0
        self.total_timesteps = 0
        self.best_reward = -np.inf

        # Metrics tracking
        self.training_metrics = []

        logger.info(f"RLlib Trainer initialisé: {experiment_name}")

    def _init_ray(self):
        """Initialise Ray avec optimisations AMD"""
        if not ray.is_initialized():
            # Configuration Ray pour AMD
            ray_config = {
                "num_cpus": config.rl.num_workers + 2,
                "num_gpus": 1 if config.amd.use_directml else 0,
                "object_store_memory": int(8 * 1024**3),  # 8GB
                "dashboard_host": "0.0.0.0",
                "dashboard_port": 8265,
                "include_dashboard": True
            }

            ray.init(**ray_config)
            logger.info("Ray initialisé avec configuration AMD")

    def _register_models(self):
        """Enregistre les modèles personnalisés AlphaStar"""

        # Modèle AlphaStar de base
        ModelCatalog.register_custom_model("alphastar_model", AlphaStarModel)

        # Modèle AlphaStar + HRM
        ModelCatalog.register_custom_model("alphastar_hrm_model", AlphaStarHRMModel)

        # Modèles spécialisés pour league
        ModelCatalog.register_custom_model("alphastar_exploiter_model", AlphaStarModel)
        ModelCatalog.register_custom_model("alphastar_league_model", AlphaStarModel)

        logger.info("Modèles personnalisés enregistrés dans Ray RLlib")

    def setup_algorithm(self, algorithm_name: str = "PPO") -> None:
        """Configure l'algorithme d'entraînement"""

        if algorithm_name == "PPO":
            algo_config = PPOConfig()
        elif algorithm_name == "IMPALA":
            algo_config = IMPALAConfig()
        elif algorithm_name == "SAC":
            algo_config = SACConfig()
        else:
            raise ValueError(f"Algorithme non supporté: {algorithm_name}")

        # Configuration de base
        algo_config = algo_config.environment(
            env=self.config.env,
            env_config=self.config.env_config
        )

        # Configuration modèle
        algo_config = algo_config.training(
            model=self.config.model,
            lr=self.config.ppo_config["lr"] if algorithm_name == "PPO" else 3e-4,
            gamma=self.config.ppo_config["gamma"]
        )

        # Configuration ressources avec optimisations AMD
        algo_config = algo_config.resources(
            num_cpus_for_main_process=2,
            num_gpus=1 if config.amd.use_directml else 0
        )

        # Configuration rollouts
        algo_config = algo_config.rollouts(
            num_workers=self.config.num_workers,
            num_envs_per_worker=self.config.num_envs_per_worker,
            num_cpus_per_worker=self.config.num_cpus_per_worker,
            num_gpus_per_worker=self.config.num_gpus_per_worker
        )

        # Configuration évaluation
        algo_config = algo_config.evaluation(
            evaluation_interval=self.config.evaluation_config["evaluation_interval"],
            evaluation_duration=self.config.evaluation_config["evaluation_duration"],
            evaluation_num_workers=self.config.evaluation_config["evaluation_num_workers"],
            evaluation_parallel_to_training=self.config.evaluation_config["evaluation_parallel_to_training"]
        )

        # Configuration spécifique PPO
        if algorithm_name == "PPO":
            algo_config = algo_config.training(
                train_batch_size=self.config.ppo_config["train_batch_size"],
                sgd_minibatch_size=self.config.ppo_config["sgd_minibatch_size"],
                num_sgd_iter=self.config.ppo_config["num_sgd_iter"],
                clip_param=self.config.ppo_config["clip_param"],
                vf_loss_coeff=self.config.ppo_config["vf_loss_coeff"],
                entropy_coeff=self.config.ppo_config["entropy_coeff"]
            )

        # Multi-agent si configuré
        if "policies" in self.config.multiagent and self.config.multiagent["policies"]:
            algo_config = algo_config.multi_agent(
                policies=self.config.multiagent["policies"],
                policy_mapping_fn=self._get_policy_mapping_fn(),
                policies_to_train=self.config.multiagent["policies_to_train"]
            )

        # Créer l'algorithme
        if algorithm_name == "PPO":
            self.algorithm = algo_config.build()
        elif algorithm_name == "IMPALA":
            self.algorithm = algo_config.build()
        elif algorithm_name == "SAC":
            self.algorithm = algo_config.build()

        logger.info(f"Algorithme {algorithm_name} configuré avec succès")

    def _get_policy_mapping_fn(self) -> Callable:
        """Retourne la fonction de mapping des politiques"""

        def policy_mapping_fn(agent_id, episode, **kwargs):
            # Implémentation simple pour démonstration
            import random

            policies = list(self.config.multiagent["policies"].keys())

            # 35% agent principal
            if random.random() < 0.35:
                return "main_agent"

            # 25% exploiters
            exploiters = [p for p in policies if "exploiter_" in p and "league_" not in p]
            if exploiters and random.random() < 0.25:
                return random.choice(exploiters)

            # 40% league exploiters
            league_exploiters = [p for p in policies if "league_exploiter_" in p]
            if league_exploiters:
                return random.choice(league_exploiters)

            return "main_agent"

        return policy_mapping_fn

    def train_single_agent(self,
                          max_iterations: int = 1000,
                          checkpoint_freq: int = 100) -> Dict[str, Any]:
        """Entraîne un agent unique"""

        if self.algorithm is None:
            self.setup_algorithm("PPO")

        training_start = time.time()
        best_reward = -np.inf

        for iteration in range(max_iterations):
            try:
                # Étape d'entraînement
                result = self.algorithm.train()

                self.training_iteration = iteration + 1
                self.total_timesteps = result["timesteps_total"]

                # Métriques
                episode_reward_mean = result["episode_reward_mean"]
                episode_len_mean = result["episode_len_mean"]

                # Logging
                if iteration % 10 == 0:
                    logger.info(
                        f"Iteration {iteration}: "
                        f"Reward={episode_reward_mean:.2f}, "
                        f"Length={episode_len_mean:.1f}, "
                        f"Timesteps={self.total_timesteps}"
                    )

                # Sauvegarde si amélioration
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                    self._save_checkpoint("best_model")

                # Checkpoint périodique
                if iteration % checkpoint_freq == 0:
                    self._save_checkpoint(f"checkpoint_{iteration}")

                # Enregistrer métriques
                self.training_metrics.append({
                    "iteration": iteration,
                    "reward": episode_reward_mean,
                    "length": episode_len_mean,
                    "timesteps": self.total_timesteps,
                    "training_time": time.time() - training_start
                })

            except KeyboardInterrupt:
                logger.info("Entraînement interrompu par l'utilisateur")
                break
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement: {e}")
                break

        training_time = time.time() - training_start

        return {
            "final_iteration": self.training_iteration,
            "final_reward": episode_reward_mean,
            "best_reward": best_reward,
            "total_timesteps": self.total_timesteps,
            "training_time": training_time,
            "metrics": self.training_metrics
        }

    def train_league(self,
                     league_size: int = 8,
                     max_iterations: int = 2000) -> Dict[str, Any]:
        """Entraîne avec système de league multi-agent"""

        # Configuration league
        league_config = create_league_config(league_size)
        self.config.multiagent = league_config

        # Setup algorithme avec multi-agent
        self.setup_algorithm("PPO")

        logger.info(f"Démarrage entraînement league avec {league_size} agents")

        # Métriques league
        league_metrics = {
            "win_rates": {},
            "elo_scores": {},
            "matchup_results": {}
        }

        # Initialize policies ELO
        for policy_name in league_config["policies"].keys():
            league_metrics["elo_scores"][policy_name] = 1200  # ELO initial

        training_start = time.time()

        for iteration in range(max_iterations):
            try:
                result = self.algorithm.train()

                # Mise à jour métriques league
                self._update_league_metrics(result, league_metrics)

                # Logging league
                if iteration % 50 == 0:
                    self._log_league_progress(iteration, result, league_metrics)

                # Checkpoint
                if iteration % 100 == 0:
                    self._save_league_checkpoint(iteration, league_metrics)

            except KeyboardInterrupt:
                logger.info("Entraînement league interrompu")
                break
            except Exception as e:
                logger.error(f"Erreur entraînement league: {e}")
                break

        training_time = time.time() - training_start

        return {
            "final_iteration": iteration + 1,
            "training_time": training_time,
            "league_metrics": league_metrics,
            "total_timesteps": result["timesteps_total"] if 'result' in locals() else 0
        }

    def _update_league_metrics(self, result: Dict[str, Any], league_metrics: Dict[str, Any]):
        """Met à jour les métriques de la league"""

        # Extraction des métriques par politique
        if "policy_reward_mean" in result:
            for policy_name, reward in result["policy_reward_mean"].items():
                # Win rate approximation basée sur reward
                win_rate = max(0.0, min(1.0, (reward + 100) / 200))
                league_metrics["win_rates"][policy_name] = win_rate

                # Update ELO basé sur performance relative
                if policy_name in league_metrics["elo_scores"]:
                    current_elo = league_metrics["elo_scores"][policy_name]
                    # Ajustement ELO simple
                    delta_elo = (win_rate - 0.5) * 32  # K-factor = 32
                    league_metrics["elo_scores"][policy_name] = current_elo + delta_elo

    def _log_league_progress(self, iteration: int, result: Dict[str, Any], league_metrics: Dict[str, Any]):
        """Log du progrès de la league"""

        logger.info(f"\n=== League Progress - Iteration {iteration} ===")

        # ELO scores
        if league_metrics["elo_scores"]:
            sorted_elo = sorted(league_metrics["elo_scores"].items(),
                               key=lambda x: x[1], reverse=True)
            logger.info("ELO Rankings:")
            for i, (policy, elo) in enumerate(sorted_elo[:5]):
                logger.info(f"  #{i+1} {policy}: {elo:.0f}")

        # Win rates
        if league_metrics["win_rates"]:
            logger.info("Win Rates:")
            for policy, win_rate in league_metrics["win_rates"].items():
                logger.info(f"  {policy}: {win_rate:.3f}")

        # Overall metrics
        if "episode_reward_mean" in result:
            logger.info(f"Overall Reward: {result['episode_reward_mean']:.2f}")

    def _save_checkpoint(self, checkpoint_name: str):
        """Sauvegarde un checkpoint"""
        if self.algorithm is None:
            return

        checkpoint_path = self.results_dir / "checkpoints" / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder algorithme
        self.algorithm.save(str(checkpoint_path))

        # Sauvegarder configuration
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")

    def _save_league_checkpoint(self, iteration: int, league_metrics: Dict[str, Any]):
        """Sauvegarde checkpoint de league"""
        checkpoint_name = f"league_checkpoint_{iteration}"
        self._save_checkpoint(checkpoint_name)

        # Sauvegarder métriques league
        metrics_path = self.results_dir / "checkpoints" / checkpoint_name / "league_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(league_metrics, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint"""
        if self.algorithm is None:
            self.setup_algorithm("PPO")

        self.algorithm.restore(checkpoint_path)
        logger.info(f"Checkpoint chargé: {checkpoint_path}")

    def evaluate_agent(self,
                      num_episodes: int = 100,
                      render: bool = False) -> Dict[str, Any]:
        """Évalue l'agent entraîné"""

        if self.algorithm is None:
            raise ValueError("Aucun algorithme configuré pour l'évaluation")

        evaluation_results = {
            "episodes": [],
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_length": 0.0,
            "success_rate": 0.0
        }

        episode_rewards = []
        episode_lengths = []
        successes = 0

        for episode in range(num_episodes):
            # Créer environnement d'évaluation
            env = gym.make(self.config.env, **self.config.env_config)

            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Politique d'action
                action = self.algorithm.compute_single_action(obs)
                obs, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Critère de succès (à adapter selon le jeu)
            if episode_reward > 0:
                successes += 1

            env.close()

        # Calcul statistiques
        evaluation_results["mean_reward"] = np.mean(episode_rewards)
        evaluation_results["std_reward"] = np.std(episode_rewards)
        evaluation_results["mean_length"] = np.mean(episode_lengths)
        evaluation_results["success_rate"] = successes / num_episodes

        logger.info(f"Évaluation terminée: Reward={evaluation_results['mean_reward']:.2f}±{evaluation_results['std_reward']:.2f}")

        return evaluation_results

    def stop_training(self):
        """Arrête l'entraînement et nettoie les ressources"""
        if self.algorithm:
            self.algorithm.stop()
            self.algorithm = None

        logger.info("Entraînement arrêté")

    def __del__(self):
        """Nettoyage automatique"""
        self.stop_training()

# Factory functions
def create_rllib_trainer(config: Optional[RLlibConfig] = None,
                        experiment_name: str = "dofus_alphastar") -> RLlibTrainer:
    """Crée un trainer RLlib configuré"""
    return RLlibTrainer(config, experiment_name)

def train_with_tune(config: RLlibConfig,
                   num_samples: int = 1,
                   max_iterations: int = 1000) -> None:
    """Lance l'entraînement avec Ray Tune pour hyperparameter search"""

    # Configuration Tune
    tune_config = {
        "env": config.env,
        "env_config": config.env_config,
        "model": config.model,
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.999),
        "clip_param": tune.uniform(0.1, 0.3),
        "num_workers": config.num_workers,
    }

    # Scheduler pour early stopping
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=max_iterations,
        grace_period=50,
        reduction_factor=2
    )

    # Stopper
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iterations),
        tune.stopper.TrialPlateauStopper(
            metric="episode_reward_mean",
            std=0.1,
            num_results=10,
            grace_period=100
        )
    )

    # Launch tuning
    tuner = tune.Tuner(
        PPO,
        tune_config=tune_config,
        run_config=RunConfig(
            name="dofus_alphastar_tune",
            stop=stopper,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=100,
                checkpoint_at_end=True
            )
        ),
        param_space=tune_config
    )

    results = tuner.fit()

    # Best result
    best_result = results.get_best_result("episode_reward_mean", "max")
    logger.info(f"Meilleure configuration trouvée: {best_result.config}")

    return results