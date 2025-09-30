"""
Configuration AlphaStar pour DOFUS Unity
Architecture hybride avec HRM et optimisations AMD
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path

@dataclass
class AMDConfig:
    """Configuration optimisations AMD 7800XT"""
    # GPU Settings
    device_name: str = "AMD Radeon RX 7800 XT"
    compute_units: int = 60
    vram_gb: int = 16
    memory_bandwidth_gbps: float = 624.0

    # Performance Settings
    use_directml: bool = True
    use_rocm: bool = True
    use_mixed_precision: bool = True
    memory_fraction: float = 0.9

    # Optimization Flags
    enable_flash_attention: bool = False  # Not available on AMD
    enable_sdp_attention: bool = True     # PyTorch 2.0 fallback
    use_gradient_checkpointing: bool = True
    use_torch_compile: bool = False       # Experimental for AMD

@dataclass
class VisionConfig:
    """Configuration vision avancée"""
    # SAM 2 Settings
    sam_model_size: str = "sam2_hiera_large"  # sam2_hiera_tiny, small, base, large
    sam_checkpoint_path: str = "models/sam2_hiera_large.pt"

    # TrOCR Settings
    trocr_model: str = "microsoft/trocr-large-printed"
    trocr_device: str = "auto"

    # Vision Pipeline
    screenshot_resolution: Tuple[int, int] = (1920, 1080)
    processing_resolution: Tuple[int, int] = (1024, 768)
    fps_target: int = 30

    # DOFUS Specific
    ui_elements: List[str] = field(default_factory=lambda: [
        "health_bar", "mana_bar", "action_points", "movement_points",
        "combat_grid", "spells_bar", "inventory", "chat", "minimap"
    ])

    # OCR Settings
    ocr_confidence_threshold: float = 0.8
    text_recognition_languages: List[str] = field(default_factory=lambda: ["fr", "en"])

@dataclass
class HRMConfig:
    """Configuration HRM (Hierarchical Reasoning Model)"""
    # System Architecture
    system_one_layers: int = 6      # Fast, intuitive reasoning
    system_two_layers: int = 12     # Slow, deliberate reasoning
    hidden_size: int = 512
    num_attention_heads: int = 8

    # Adaptive Computation
    max_reasoning_steps: int = 8
    halting_threshold: float = 0.5
    halting_penalty: float = 0.01

    # Memory & Context
    context_window: int = 2048
    memory_size: int = 1024
    episodic_memory_capacity: int = 10000

@dataclass
class AlphaStarConfig:
    """Configuration agent AlphaStar"""
    # Network Architecture
    transformer_layers: int = 8
    lstm_layers: int = 2
    hidden_dim: int = 512
    attention_heads: int = 8

    # Action Space
    action_space_size: int = 200  # Actions DOFUS possibles
    pointer_network_dim: int = 128

    # Observation Processing
    spatial_features: int = 32
    nonspatial_features: int = 64
    entity_features: int = 16
    max_entities: int = 100

    # Training Settings
    learning_rate: float = 3e-4
    batch_size: int = 32
    sequence_length: int = 64
    replay_buffer_size: int = 100000

    # Multi-Agent League
    league_size: int = 8
    exploiter_ratio: float = 0.25
    main_agent_ratio: float = 0.35
    league_exploiter_ratio: float = 0.4

@dataclass
class RLConfig:
    """Configuration apprentissage par renforcement"""
    # Algorithms
    algorithm: str = "PPO"  # PPO, SAC, TD3, IMPALA
    framework: str = "rllib"  # rllib, stable_baselines3

    # Training Parameters
    total_timesteps: int = 10_000_000
    num_envs: int = 8
    num_workers: int = 4
    rollout_fragment_length: int = 256
    train_batch_size: int = 2048

    # PPO Specific
    ppo_epochs: int = 10
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coef: float = 0.01

    # Exploration
    exploration_config: Dict = field(default_factory=lambda: {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 100000
    })

@dataclass
class DofusConfig:
    """Configuration spécifique DOFUS"""
    # Game Settings
    server_type: str = "unity"  # unity, retro
    character_class: str = "iop"
    starting_level: int = 1
    target_level: int = 200

    # Gameplay Objectives
    primary_objectives: List[str] = field(default_factory=lambda: [
        "leveling", "quests", "dungeon_farming", "resource_gathering"
    ])

    # Anti-Detection
    human_behavior_variance: float = 0.15
    action_delay_range: Tuple[float, float] = (0.1, 0.5)
    typing_speed_wpm: int = 45
    mouse_movement_smoothness: float = 0.8

    # Safety & Ethics
    max_session_duration: int = 7200  # 2 hours
    mandatory_break_duration: int = 1800  # 30 minutes
    respect_other_players: bool = True
    no_exploitation: bool = True

@dataclass
class SystemConfig:
    """Configuration système générale"""
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    # Logging
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    # Performance Monitoring
    enable_profiling: bool = False
    metrics_collection: bool = True
    telemetry_endpoint: Optional[str] = None

    # Development
    debug_mode: bool = False
    seed: Optional[int] = 42
    deterministic: bool = False

@dataclass
class MasterConfig:
    """Configuration maître regroupant tous les modules"""
    amd: AMDConfig = field(default_factory=AMDConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    hrm: HRMConfig = field(default_factory=HRMConfig)
    alphastar: AlphaStarConfig = field(default_factory=AlphaStarConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    dofus: DofusConfig = field(default_factory=DofusConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Validation et ajustements post-initialisation"""
        # Ajuster batch size selon VRAM disponible
        estimated_model_size = self.hrm.hidden_size * self.alphastar.hidden_dim * 4 / (1024**3)
        if estimated_model_size > self.amd.vram_gb * 0.6:
            self.rl.batch_size = min(self.rl.batch_size, 16)
            self.alphastar.batch_size = min(self.alphastar.batch_size, 16)

        # Optimiser pour AMD
        if not self.amd.use_directml:
            # Fallback CPU settings
            self.rl.num_workers = min(self.rl.num_workers, 2)
            self.vision.fps_target = min(self.vision.fps_target, 15)

        # Créer dossiers nécessaires
        for directory in [self.system.data_dir, self.system.models_dir, self.system.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

# Instance globale de configuration
config = MasterConfig()

# Fonctions utilitaires
def get_device() -> torch.device:
    """Retourne le device optimal pour les calculs"""
    if config.amd.use_directml:
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def get_dtype() -> torch.dtype:
    """Retourne le type de données optimal"""
    if config.amd.use_mixed_precision:
        return torch.bfloat16
    return torch.float32

def update_config(**kwargs):
    """Met à jour la configuration globale"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                current_value = getattr(config, key)
                for sub_key, sub_value in value.items():
                    setattr(current_value, sub_key, sub_value)
            else:
                setattr(config, key, value)

def save_config(path: str):
    """Sauvegarde la configuration dans un fichier"""
    import json
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    config_dict = to_dict(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(path: str):
    """Charge la configuration depuis un fichier"""
    import json
    global config

    with open(path, 'r') as f:
        config_dict = json.load(f)

    # TODO: Reconstruire l'objet config depuis le dictionnaire
    # Implementation complexe, à faire si nécessaire
    pass