"""
Configuration pour Ray RLlib et Stable Baselines3
Optimisée pour AMD 7800XT et architecture AlphaStar
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import torch
from pathlib import Path

from config import config

@dataclass
class RLlibConfig:
    """Configuration Ray RLlib pour AlphaStar DOFUS"""

    # Framework settings
    framework: str = "torch"
    env: str = "dofus_unity_env"

    # Algorithm configuration
    algorithm: str = "PPO"  # PPO, IMPALA, APPO, SAC

    # Model architecture
    model: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "alphastar_model",
        "custom_model_config": {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_transformer_layers": 6,
            "num_lstm_layers": 2,
            "use_hrm": True,
            "hrm_config": {
                "system_one_layers": 6,
                "system_two_layers": 12,
                "max_reasoning_steps": 8
            }
        },
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
        "use_attention": True,
        "attention_num_transformer_units": 4,
        "attention_dim": 64,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "max_seq_len": 50
    })

    # Training settings
    num_workers: int = field(default=4)
    num_envs_per_worker: int = field(default=1)
    num_cpus_per_worker: int = field(default=1)
    num_gpus_per_worker: float = field(default=0.25)  # Partage GPU AMD

    # Environment settings
    env_config: Dict[str, Any] = field(default_factory=lambda: {
        "character_class": "iop",
        "target_level": 200,
        "max_episode_steps": 2000,
        "reward_shaping": True,
        "human_like_behavior": True,
        "anti_detection": True
    })

    # PPO specific
    ppo_config: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "kl_coeff": 0.2,
        "clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes"
    })

    # Exploration
    exploration_config: Dict[str, Any] = field(default_factory=lambda: {
        "type": "StochasticSampling"
    })

    # Multi-agent settings
    multiagent: Dict[str, Any] = field(default_factory=lambda: {
        "policies": {
            "main_agent": (None, None, None, {
                "gamma": 0.99,
                "lr": 3e-4
            }),
            "exploiter": (None, None, None, {
                "gamma": 0.99,
                "lr": 5e-4
            }),
            "league_exploiter": (None, None, None, {
                "gamma": 0.99,
                "lr": 2e-4
            })
        },
        "policy_mapping_fn": "select_policy_randomly",
        "policies_to_train": ["main_agent"]
    })

    # Resources
    resources: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "num_cpus_for_driver": 2,
        "num_gpus": 1,  # AMD 7800XT
    })

    # Evaluation
    evaluation_config: Dict[str, Any] = field(default_factory=lambda: {
        "evaluation_interval": 10,
        "evaluation_duration": 10,
        "evaluation_num_workers": 1,
        "evaluation_parallel_to_training": True
    })

@dataclass
class SB3Config:
    """Configuration Stable Baselines3"""

    # Algorithm
    algorithm: str = "PPO"  # PPO, A2C, SAC, TD3, DQN

    # Policy network
    policy: str = "MultiInputPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": {
            "pi": [512, 512],
            "vf": [512, 512]
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": False,
        "use_sde": False,
        "log_std_init": 0.0,
        "full_std": True,
        "sde_net_arch": None,
        "use_expln": False,
        "squash_output": False,
        "features_extractor_class": None,
        "features_extractor_kwargs": {},
        "normalize_images": True,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {}
    })

    # Training hyperparameters
    learning_rate: Union[float, str] = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, str] = 0.2
    clip_range_vf: Optional[Union[float, str]] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Tensorboard logging
    tensorboard_log: Optional[str] = None
    create_eval_env: bool = False
    policy_base = None  # Auto-detected
    verbose: int = 1
    seed: Optional[int] = None

@dataclass
class TrainingConfig:
    """Configuration générale d'entraînement"""

    # Framework choice
    framework: str = "rllib"  # "rllib", "sb3", "both"

    # Training phases
    phases: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "supervised_learning",
            "duration_steps": 1_000_000,
            "description": "Apprentissage supervisé sur replays humains",
            "data_source": "human_replays"
        },
        {
            "name": "self_play_basic",
            "duration_steps": 5_000_000,
            "description": "Self-play basique",
            "opponents": ["random", "rule_based"]
        },
        {
            "name": "league_training",
            "duration_steps": 20_000_000,
            "description": "Entraînement multi-agent league",
            "league_size": 8,
            "exploiter_ratio": 0.25
        },
        {
            "name": "fine_tuning",
            "duration_steps": 2_000_000,
            "description": "Fine-tuning final",
            "learning_rate_decay": 0.1
        }
    ])

    # Checkpointing
    checkpoint_frequency: int = 100_000  # steps
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 5
    checkpoint_score_attr: str = "episode_reward_mean"

    # Stopping criteria
    stop: Dict[str, Any] = field(default_factory=lambda: {
        "training_iteration": 1000,
        "timesteps_total": 50_000_000,
        "episode_reward_mean": 500.0,
        "custom_metrics/win_rate_mean": 0.8
    })

    # Monitoring
    metrics: List[str] = field(default_factory=lambda: [
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "episode_len_mean",
        "episodes_this_iter",
        "policy_loss",
        "value_loss",
        "entropy",
        "custom_metrics/win_rate",
        "custom_metrics/survival_time",
        "custom_metrics/damage_dealt",
        "custom_metrics/xp_gained"
    ])

    # Hardware optimization
    hardware_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_amd_optimizations": True,
        "mixed_precision": True,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "non_blocking": True
    })

def get_training_config(framework: str = "rllib") -> Union[RLlibConfig, SB3Config]:
    """Factory function pour obtenir la configuration d'entraînement"""

    if framework == "rllib":
        rl_config = RLlibConfig()

        # Ajustements basés sur config globale
        rl_config.num_workers = min(rl_config.num_workers, config.rl.num_workers)
        rl_config.ppo_config["train_batch_size"] = config.rl.train_batch_size
        rl_config.ppo_config["lr"] = config.rl.learning_rate

        # Optimisations AMD
        if config.amd.use_mixed_precision:
            rl_config.model["custom_model_config"]["mixed_precision"] = True

        return rl_config

    elif framework == "sb3":
        sb3_config = SB3Config()

        # Ajustements basés sur config globale
        sb3_config.learning_rate = config.rl.learning_rate
        sb3_config.batch_size = min(sb3_config.batch_size, config.rl.train_batch_size // 32)
        sb3_config.n_steps = config.rl.rollout_fragment_length

        # Device AMD
        if config.amd.use_directml:
            sb3_config.device = "cuda"  # DirectML maps to cuda interface
        else:
            sb3_config.device = "cpu"

        return sb3_config

    else:
        raise ValueError(f"Framework non supporté: {framework}")

def create_league_config(league_size: int = 8) -> Dict[str, Any]:
    """Crée une configuration de league multi-agent"""

    policies = {}

    # Agent principal (main)
    policies["main_agent"] = (
        None, None, None,
        {
            "lr": 3e-4,
            "gamma": 0.99,
            "model": {
                "custom_model": "alphastar_hrm_model",
                "custom_model_config": {
                    "use_hrm": True,
                    "hrm_max_steps": 8
                }
            }
        }
    )

    # Agents exploiters
    for i in range(int(league_size * 0.25)):
        policies[f"exploiter_{i}"] = (
            None, None, None,
            {
                "lr": 5e-4,
                "gamma": 0.95,
                "model": {
                    "custom_model": "alphastar_exploiter_model"
                }
            }
        )

    # League exploiters
    for i in range(int(league_size * 0.4)):
        policies[f"league_exploiter_{i}"] = (
            None, None, None,
            {
                "lr": 2e-4,
                "gamma": 0.99,
                "model": {
                    "custom_model": "alphastar_league_model"
                }
            }
        )

    return {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, episode, **kwargs:
            select_league_policy(agent_id, episode, list(policies.keys())),
        "policies_to_train": list(policies.keys())
    }

def select_league_policy(agent_id: str, episode, available_policies: List[str]) -> str:
    """Sélectionne la politique pour l'agent dans la league"""
    import random

    # 35% main agent
    if random.random() < 0.35:
        return "main_agent"

    # 25% exploiters
    exploiters = [p for p in available_policies if "exploiter_" in p and "league_" not in p]
    if exploiters and random.random() < 0.25:
        return random.choice(exploiters)

    # 40% league exploiters
    league_exploiters = [p for p in available_policies if "league_exploiter_" in p]
    if league_exploiters:
        return random.choice(league_exploiters)

    # Fallback
    return "main_agent"

# Configuration par défaut
default_training_config = TrainingConfig()

# Export des configurations
__all__ = [
    "RLlibConfig",
    "SB3Config",
    "TrainingConfig",
    "get_training_config",
    "create_league_config",
    "default_training_config"
]