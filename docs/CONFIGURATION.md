# ⚙️ Guide de Configuration Avancée - TacticalBot

## 📋 Vue d'Ensemble

TacticalBot utilise un système de configuration multicouche permettant une personnalisation fine de tous les aspects du bot. La configuration combine fichiers JSON, variables d'environnement, profils prédéfinis et interface de configuration dynamique.

## 📁 Structure de Configuration

```
config/
├── 🔧 Core Configuration
│   ├── engine.json              # Moteur principal
│   ├── modules.json             # Configuration des modules
│   └── logging.json             # Système de logs
│
├── 🎮 Game Configuration  
│   ├── professions.json         # Métiers et progression
│   ├── combat.json              # Configuration de combat
│   └── navigation.json          # Paramètres de navigation
│
├── 🛡️ Security Configuration
│   ├── safety.json              # Paramètres de sécurité
│   ├── detection_avoidance.json # Évitement détection
│   └── session_limits.json      # Limites de session
│
├── 🎯 AI Configuration
│   ├── decision_engine.json     # Moteur de décision
│   ├── strategy_selector.json   # Sélection stratégies
│   └── learning.json            # Paramètres apprentissage
│
├── 👁️ Vision Configuration
│   ├── screen_analysis.json     # Analyse d'écran
│   ├── templates.json           # Configuration templates
│   └── ocr.json                 # Reconnaissance texte
│
└── 📊 Profiles/                 # Profils prédéfinis
    ├── farmer_safe.json
    ├── farmer_efficient.json
    ├── combat_aggressive.json
    ├── combat_defensive.json
    ├── explorer_balanced.json
    └── social_cooperative.json
```

---

## 🔧 Configuration du Moteur Principal

### engine.json
Configuration du moteur central et des performances système.

```json
{
  "engine": {
    "target_fps": 30,
    "decision_fps": 10,
    "max_modules": 50,
    "enable_logging": true,
    "log_level": "INFO",
    "performance_monitoring": true,
    "auto_recovery": true,
    "safety_checks": true
  },
  "threading": {
    "main_thread_priority": "normal",
    "worker_threads": 4,
    "thread_pool_size": 8,
    "async_event_processing": true
  },
  "memory": {
    "max_memory_mb": 512,
    "garbage_collection_interval": 300,
    "cache_cleanup_frequency": 600,
    "enable_memory_profiling": false
  },
  "performance": {
    "enable_performance_monitoring": true,
    "performance_log_interval": 60,
    "fps_warning_threshold": 25,
    "memory_warning_threshold": 400,
    "cpu_warning_threshold": 80
  },
  "error_handling": {
    "max_errors_per_module": 10,
    "error_reset_interval": 300,
    "auto_restart_failed_modules": true,
    "emergency_shutdown_threshold": 50
  }
}
```

#### Paramètres détaillés

**Performance**
- `target_fps` : FPS cible pour la boucle principale (recommandé: 30)
- `decision_fps` : FPS pour les décisions IA (recommandé: 10)
- `max_modules` : Nombre maximum de modules actifs

**Threading**
- `worker_threads` : Nombre de threads de travail
- `thread_pool_size` : Taille du pool de threads
- `async_event_processing` : Traitement asynchrone des événements

**Mémoire**
- `max_memory_mb` : Limite mémoire en MB
- `garbage_collection_interval` : Intervalle nettoyage mémoire (secondes)
- `cache_cleanup_frequency` : Fréquence nettoyage cache (secondes)

### modules.json
Configuration des modules système et leur priorité.

```json
{
  "core_modules": {
    "event_bus": {
      "enabled": true,
      "priority": 1,
      "auto_start": true,
      "config": {
        "max_events_per_second": 1000,
        "event_queue_size": 5000,
        "enable_event_logging": false
      }
    },
    "state_manager": {
      "enabled": true,
      "priority": 2,
      "dependencies": ["vision_system"],
      "config": {
        "update_frequency": 30,
        "state_history_size": 100,
        "auto_save_interval": 60
      }
    }
  },
  "game_modules": {
    "profession_manager": {
      "enabled": true,
      "priority": 5,
      "config": {
        "auto_optimize_routes": true,
        "enable_synergies": true,
        "max_session_duration": 4.0
      }
    },
    "combat_ai": {
      "enabled": true,
      "priority": 3,
      "config": {
        "character_class": "auto_detect",
        "combat_strategy": "balanced",
        "retreat_health_threshold": 30.0
      }
    }
  }
}
```

---

## 🎮 Configuration Gameplay

### professions.json
Configuration complète des métiers et progression.

```json
{
  "version": "1.0.0",
  "last_saved": "2025-08-31 10:00:49",
  "global_settings": {
    "auto_bank_enabled": true,
    "auto_craft_enabled": false,
    "inventory_threshold": 85,
    "prefer_safe_routes": true,
    "enable_synergies": true
  },
  "optimization": {
    "default_strategy": "BALANCED",
    "rebalance_frequency": 300,
    "min_efficiency_threshold": 0.6,
    "enable_adaptive_learning": true
  },
  "professions": {
    "farmer": {
      "enabled": true,
      "level": 4,
      "experience": 2436,
      "config": {
        "preferred_resources": ["ble", "orge", "avoine"],
        "blacklisted_resources": ["bambou_sacre"],
        "farming_pattern": "cluster",
        "auto_replant": false,
        "harvest_quality_threshold": 0.8
      },
      "routes": {
        "custom_routes_enabled": true,
        "avoid_crowded_areas": true,
        "max_travel_distance": 500,
        "prioritize_efficiency": true
      }
    },
    "lumberjack": {
      "enabled": true,
      "level": 1,
      "experience": 259,
      "config": {
        "current_tool": "hache_debutant",
        "auto_upgrade_tool": true,
        "preferred_trees": ["chene", "frene", "noyer"],
        "respect_regrowth_time": true,
        "cut_pattern": "spiral"
      }
    },
    "miner": {
      "enabled": true,
      "level": 2,
      "experience": 554,
      "config": {
        "current_tool": "pioche_debutant",
        "gem_hunting_enabled": true,
        "deep_mining_threshold": 40,
        "ore_quality_minimum": 0.7,
        "follow_veins": true
      }
    },
    "alchemist": {
      "enabled": true,
      "level": 2,
      "experience": 651,
      "config": {
        "workshop_level": 1,
        "auto_upgrade_workshop": true,
        "preferred_recipes": ["potion_soin_mineure", "potion_mana"],
        "ingredient_threshold": 20,
        "craft_in_batches": true
      }
    }
  }
}
```

### combat.json
Configuration spécialisée pour le combat.

```json
{
  "combat_ai": {
    "enabled": true,
    "character_class": "auto_detect",
    "combat_strategy": "balanced",
    "difficulty": "normal"
  },
  "classes": {
    "cra": {
      "combat_range": "long",
      "preferred_position": "back",
      "spell_priorities": ["fleche_magique", "fleche_explosive", "tir_eloigne"],
      "escape_threshold": 25.0,
      "use_environmental_advantages": true
    },
    "iop": {
      "combat_range": "melee",
      "preferred_position": "front",
      "spell_priorities": ["epee_divine", "concentration", "mutilation"],
      "berserker_threshold": 40.0,
      "defensive_stance_health": 30.0
    },
    "eniripsa": {
      "combat_range": "support",
      "preferred_position": "middle",
      "spell_priorities": ["mot_soignant", "mot_revitalisant", "mot_stimulant"],
      "heal_threshold": 60.0,
      "support_priority": "health_critical"
    }
  },
  "tactics": {
    "target_selection": "weakest_first",
    "spell_rotation": "adaptive",
    "movement_strategy": "maintain_optimal_range",
    "team_coordination": true,
    "environment_usage": true
  },
  "safety": {
    "retreat_conditions": {
      "health_below": 20.0,
      "mana_below": 10.0,
      "outnumbered_ratio": 3.0,
      "high_level_enemy": true
    },
    "emergency_actions": ["potion_soin", "sort_fuite", "recall_item"],
    "safe_zones": ["temple", "banque", "maison"]
  }
}
```

---

## 🛡️ Configuration Sécurité

### safety.json
Paramètres de sécurité et évitement de détection.

```json
{
  "detection_avoidance": {
    "enabled": true,
    "randomization_level": 0.7,
    "human_behavior_simulation": true,
    "anti_pattern_detection": true
  },
  "timing_randomization": {
    "action_delay_range": [100, 500],
    "keystroke_interval": [80, 200],
    "mouse_movement_speed": [0.5, 1.5],
    "pause_probability": 0.1
  },
  "human_behavior": {
    "typing_speed_range": [80, 120],
    "reaction_time_range": [150, 400],
    "error_probability": 0.02,
    "correction_delay": [200, 800],
    "idle_movements": true,
    "random_clicks": 0.05
  },
  "session_limits": {
    "max_session_duration": 4.0,
    "mandatory_break_interval": 1.5,
    "min_break_duration": 0.25,
    "max_daily_playtime": 8.0,
    "weekly_limit": 40.0
  },
  "pattern_breaking": {
    "route_variations": true,
    "activity_mixing": true,
    "schedule_randomization": true,
    "social_interactions": 0.15
  },
  "monitoring": {
    "track_behavior_patterns": true,
    "alert_on_anomalies": true,
    "log_security_events": true,
    "performance_degradation_detection": true
  }
}
```

### detection_avoidance.json
Configuration avancée pour l'évitement de détection.

```json
{
  "behavioral_patterns": {
    "movement_patterns": {
      "path_deviation_max": 20,
      "speed_variation_range": [0.8, 1.3],
      "pause_insertion_probability": 0.15,
      "backtrack_probability": 0.05,
      "exploration_detours": 0.1
    },
    "action_patterns": {
      "sequence_randomization": true,
      "action_clustering_avoidance": true,
      "perfect_timing_avoidance": true,
      "mistake_simulation": 0.03,
      "hesitation_simulation": 0.08
    },
    "interaction_patterns": {
      "ui_interaction_delay": [200, 800],
      "menu_navigation_humanization": true,
      "scroll_wheel_variation": true,
      "window_management_realistic": true
    }
  },
  "statistical_camouflage": {
    "performance_variation": {
      "efficiency_fluctuation": [0.85, 1.15],
      "skill_degradation_simulation": true,
      "fatigue_effect_modeling": true,
      "learning_curve_simulation": true
    },
    "temporal_patterns": {
      "daily_activity_variation": true,
      "weekly_schedule_changes": true,
      "break_pattern_randomization": true,
      "login_time_variation": [300, 1800]
    }
  },
  "advanced_techniques": {
    "mouse_trajectory_humanization": true,
    "keyboard_rhythm_variation": true,
    "screen_reading_simulation": true,
    "attention_span_modeling": true,
    "multitasking_simulation": false
  }
}
```

---

## 🎯 Configuration IA et Décision

### decision_engine.json
Configuration du moteur de décision intelligent.

```json
{
  "decision_engine": {
    "enabled": true,
    "decision_frequency": 10,
    "learning_enabled": true,
    "explanation_level": "detailed"
  },
  "criteria_weights": {
    "priority": 2.0,
    "confidence": 1.5,
    "efficiency": 1.8,
    "safety": 2.5,
    "reward": 1.2,
    "duration": 0.8,
    "risk": -1.5,
    "context_match": 1.3
  },
  "priority_scaling": {
    "critical": 100,
    "high": 80,
    "medium": 60,
    "low": 40,
    "minimal": 20
  },
  "context_factors": {
    "health_weight": 2.0,
    "mana_weight": 1.5,
    "combat_multiplier": 1.8,
    "safe_zone_bonus": 0.3,
    "time_pressure_factor": 1.4
  },
  "learning_parameters": {
    "learning_rate": 0.1,
    "memory_size": 1000,
    "adaptation_threshold": 0.15,
    "success_reward": 1.0,
    "failure_penalty": -0.5,
    "exploration_rate": 0.2
  },
  "conflict_resolution": {
    "enable_conflict_detection": true,
    "resolution_strategy": "priority_weighted",
    "timeout_resolution": "highest_priority",
    "deadlock_prevention": true
  }
}
```

### strategy_selector.json
Configuration du sélecteur de stratégies adaptatif.

```json
{
  "strategy_selector": {
    "enabled": true,
    "adaptation_frequency": 30,
    "situation_analysis_depth": "detailed",
    "performance_tracking": true
  },
  "strategies": {
    "aggressive": {
      "risk_tolerance": 0.8,
      "efficiency_weight": 1.5,
      "safety_weight": 0.3,
      "reward_focus": 2.0,
      "speed_priority": 1.8
    },
    "defensive": {
      "risk_tolerance": 0.2,
      "efficiency_weight": 0.7,
      "safety_weight": 2.0,
      "reward_focus": 0.8,
      "speed_priority": 0.6
    },
    "balanced": {
      "risk_tolerance": 0.5,
      "efficiency_weight": 1.0,
      "safety_weight": 1.0,
      "reward_focus": 1.0,
      "speed_priority": 1.0
    },
    "efficient": {
      "risk_tolerance": 0.6,
      "efficiency_weight": 2.0,
      "safety_weight": 0.8,
      "reward_focus": 1.2,
      "speed_priority": 1.8
    },
    "stealth": {
      "risk_tolerance": 0.3,
      "efficiency_weight": 0.6,
      "safety_weight": 1.8,
      "reward_focus": 0.7,
      "speed_priority": 0.4
    },
    "social": {
      "risk_tolerance": 0.4,
      "efficiency_weight": 0.8,
      "safety_weight": 1.2,
      "reward_focus": 1.1,
      "speed_priority": 0.9
    }
  },
  "situation_detection": {
    "analysis_window": 60,
    "confidence_threshold": 0.7,
    "adaptation_cooldown": 120,
    "learning_from_outcomes": true
  },
  "performance_metrics": {
    "track_success_rates": true,
    "measure_efficiency": true,
    "calculate_roi": true,
    "monitor_safety_incidents": true,
    "adaptation_effectiveness": true
  }
}
```

---

## 👁️ Configuration Vision

### screen_analysis.json
Configuration de l'analyse d'écran et reconnaissance visuelle.

```json
{
  "screen_capture": {
    "capture_method": "windows_api",
    "capture_frequency": 30,
    "capture_region": "full_screen",
    "optimize_for_speed": true,
    "enable_caching": true
  },
  "image_processing": {
    "preprocessing_enabled": true,
    "noise_reduction": true,
    "contrast_enhancement": true,
    "brightness_normalization": true,
    "edge_detection": false
  },
  "template_matching": {
    "matching_method": "cv2.TM_CCOEFF_NORMED",
    "confidence_threshold": 0.8,
    "multi_scale_matching": true,
    "rotation_tolerance": 5,
    "adaptive_templates": true
  },
  "ocr_settings": {
    "ocr_engine": "tesseract",
    "language": "fra",
    "confidence_threshold": 0.7,
    "preprocessing": "adaptive_threshold",
    "character_whitelist": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz%/- ",
    "enable_spell_check": false
  },
  "ui_detection": {
    "health_bar_region": [50, 50, 200, 20],
    "mana_bar_region": [50, 80, 200, 20],
    "inventory_region": [600, 200, 400, 300],
    "chat_region": [10, 400, 500, 200],
    "minimap_region": [1700, 50, 200, 200]
  },
  "performance": {
    "max_processing_time": 33,
    "enable_multithreading": true,
    "cache_processed_images": true,
    "cleanup_frequency": 300
  }
}
```

---

## 📊 Profils Prédéfinis

TacticalBot inclut plusieurs profils préconfigurés pour différents styles de jeu.

### Profil Farmer Efficient

```json
{
  "profile_name": "farmer_efficient",
  "description": "Farming efficace avec optimisation du temps",
  "version": "1.0",
  "engine": {
    "target_fps": 30,
    "decision_fps": 15,
    "performance_monitoring": true
  },
  "decision_engine": {
    "criteria_weights": {
      "efficiency": 2.0,
      "reward": 1.8,
      "safety": 1.2,
      "duration": 0.6
    }
  },
  "strategy_selector": {
    "default_strategy": "efficient",
    "adaptation_enabled": true
  },
  "professions": {
    "farmer": {
      "priority": 1,
      "optimization_strategy": "PROFIT_FOCUSED",
      "routes": "optimized_efficiency"
    }
  },
  "safety": {
    "randomization_level": 0.6,
    "session_duration": 3.5,
    "break_frequency": 1.0
  }
}
```

### Profil Combat Defensive

```json
{
  "profile_name": "combat_defensive",
  "description": "Combat défensif avec priorité à la sécurité",
  "version": "1.0",
  "engine": {
    "target_fps": 30,
    "decision_fps": 20,
    "safety_checks": true
  },
  "decision_engine": {
    "criteria_weights": {
      "safety": 3.0,
      "risk": -2.0,
      "priority": 2.0,
      "efficiency": 1.0
    }
  },
  "combat": {
    "strategy": "defensive",
    "retreat_threshold": 40.0,
    "potion_usage": "conservative",
    "target_selection": "safest_first"
  },
  "safety": {
    "randomization_level": 0.8,
    "human_behavior": true,
    "emergency_protocols": "strict"
  }
}
```

---

## 🔧 Configuration Dynamique

### Interface de Configuration

```python
from config.config_manager import ConfigManager

# Initialisation
config = ConfigManager()

# Chargement d'un profil
config.load_profile("farmer_efficient")

# Modification dynamique
config.update_setting("decision_engine.criteria_weights.efficiency", 2.5)
config.update_setting("professions.farmer.preferred_resources", ["ble", "orge"])

# Application des changements
config.apply_changes()

# Sauvegarde
config.save_current_config("my_custom_profile")
```

### API de Configuration

```python
class ConfigAPI:
    def get_config(self, path: str) -> Any:
        """Récupère une valeur de configuration"""
        
    def set_config(self, path: str, value: Any) -> bool:
        """Définit une valeur de configuration"""
        
    def reset_config(self, path: str = None) -> bool:
        """Remet la configuration par défaut"""
        
    def validate_config(self, config_data: dict) -> bool:
        """Valide une configuration"""
        
    def export_config(self, filepath: str) -> bool:
        """Exporte la configuration actuelle"""
        
    def import_config(self, filepath: str) -> bool:
        """Importe une configuration"""
```

### Variables d'Environnement

```bash
# .env - Configuration d'environnement
GAME_WINDOW_TITLE="Dofus 2.0"
GAME_EXECUTABLE_PATH="C:/Program Files/Dofus/Dofus.exe"

# Performance
MAX_CPU_USAGE=80
MAX_MEMORY_MB=512
TARGET_FPS=30
DECISION_FPS=10

# Sécurité
ENABLE_SAFETY_CHECKS=true
HUMAN_BEHAVIOR_ENABLED=true
RANDOMIZATION_LEVEL=0.7
MAX_SESSION_DURATION=4.0

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_DIRECTORY=logs
LOG_MAX_SIZE=100MB

# Modules
ENABLE_PROFESSIONS=true
ENABLE_COMBAT=true
ENABLE_VISION=true
ENABLE_NAVIGATION=true

# Interface
ENABLE_WEB_INTERFACE=false
WEB_PORT=8080
API_ENABLED=false
```

---

## 🛠️ Configuration Avancée

### Custom Hooks et Callbacks

```python
# Configuration des hooks personnalisés
{
  "hooks": {
    "on_module_error": "custom_error_handler",
    "on_level_up": "celebration_routine",
    "on_resource_found": "priority_targeting",
    "on_combat_start": "combat_preparation",
    "on_death": "revival_protocol"
  },
  "callbacks": {
    "decision_made": ["log_decision", "update_statistics"],
    "action_completed": ["measure_performance", "update_learning"],
    "error_occurred": ["error_recovery", "notify_user"]
  }
}
```

### Configuration par Conditions

```python
# Configuration conditionnelle
{
  "conditional_config": {
    "conditions": {
      "low_health": {
        "trigger": "character.health_percent < 30",
        "config_overrides": {
          "decision_engine.criteria_weights.safety": 5.0,
          "strategy_selector.force_strategy": "defensive"
        }
      },
      "high_level_area": {
        "trigger": "current_map.danger_level > 7",
        "config_overrides": {
          "safety.randomization_level": 0.9,
          "combat.retreat_threshold": 60.0
        }
      }
    }
  }
}
```

### Configuration Multi-Environnement

```json
{
  "environments": {
    "development": {
      "debug_mode": true,
      "verbose_logging": true,
      "safety_disabled": true,
      "fast_execution": false
    },
    "testing": {
      "mock_game_interactions": true,
      "accelerated_time": true,
      "test_scenarios_enabled": true
    },
    "production": {
      "debug_mode": false,
      "safety_enabled": true,
      "performance_optimized": true,
      "error_reporting": true
    }
  }
}
```

---

Cette documentation couvre tous les aspects de la configuration de TacticalBot, permettant une personnalisation complète selon vos besoins spécifiques. La configuration modulaire permet d'ajuster finement chaque aspect du comportement du bot tout en maintenant la simplicité d'utilisation.