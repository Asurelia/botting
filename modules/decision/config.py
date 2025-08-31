"""
Configuration et interface pour la gestion des priorités du module de décision.

Ce module fournit une interface pour configurer les priorités et paramètres
du système de décision, avec sauvegarde automatique des préférences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .decision_engine import Priority, ActionType, DecisionEngine
from .strategy_selector import StrategyType, StrategySelector


@dataclass
class DecisionConfig:
    """Configuration complète du module de décision."""
    # Poids globaux
    priority_weights: Dict[str, float]
    
    # Préférences de risque par situation
    risk_profiles: Dict[str, float]
    
    # Limites de temps par type d'action
    time_limits: Dict[str, float]
    
    # Seuils d'activation automatique
    activation_thresholds: Dict[str, float]
    
    # Paramètres d'apprentissage
    learning_rate: float
    history_limit: int
    
    # Stratégies préférées par situation
    preferred_strategies: Dict[str, str]


class DecisionConfigManager:
    """
    Gestionnaire de configuration pour le module de décision.
    Fournit une interface simple pour configurer les priorités et préférences.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Répertoire de configuration
        self.config_dir = Path(config_dir) if config_dir else Path("config/decision")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichiers de configuration
        self.config_file = self.config_dir / "decision_config.json"
        self.engine_config_file = self.config_dir / "engine_config.json"
        self.strategy_config_file = self.config_dir / "strategy_config.json"
        
        # Configuration par défaut
        self.default_config = DecisionConfig(
            priority_weights={
                'survival': 2.0,      # Survie = priorité maximale
                'objective': 1.5,     # Objectif principal
                'efficiency': 1.0,    # Efficacité standard
                'maintenance': 0.5,   # Maintenance = priorité faible
                'social': 0.3        # Social = priorité minimale
            },
            risk_profiles={
                'very_safe': 0.1,     # Très prudent
                'safe': 0.3,          # Prudent
                'balanced': 0.5,      # Équilibré
                'aggressive': 0.7,    # Agressif
                'very_aggressive': 0.9 # Très agressif
            },
            time_limits={
                'combat': 300.0,      # 5 minutes max en combat
                'movement': 180.0,    # 3 minutes max de déplacement
                'profession': 1800.0, # 30 minutes max de métier
                'inventory': 120.0,   # 2 minutes max de gestion inventaire
                'maintenance': 300.0, # 5 minutes max de maintenance
                'social': 600.0       # 10 minutes max d'interaction sociale
            },
            activation_thresholds={
                'critical_health': 20.0,    # Santé critique à 20%
                'low_health': 40.0,         # Santé faible à 40%
                'critical_mana': 10.0,      # Mana critique à 10%
                'low_mana': 30.0,           # Mana faible à 30%
                'full_inventory': 90.0,     # Inventaire plein à 90%
                'high_inventory': 75.0,     # Inventaire élevé à 75%
                'high_risk_enemy_count': 3, # 3+ ennemis = haut risque
                'max_session_time': 14400.0 # 4h max de session
            },
            learning_rate=0.1,              # Apprentissage modéré
            history_limit=1000,             # Garder 1000 décisions en historique
            preferred_strategies={
                'peaceful_farming': StrategyType.EFFICIENT.value,
                'dangerous_area': StrategyType.DEFENSIVE.value,
                'boss_fight': StrategyType.AGGRESSIVE.value,
                'crowded_area': StrategyType.SOCIAL.value,
                'low_resources': StrategyType.DEFENSIVE.value,
                'inventory_full': StrategyType.EFFICIENT.value
            }
        )
        
        # Configuration actuelle
        self.config = self._load_config()
        
        # Profils de configuration prédéfinis
        self.predefined_profiles = {
            'farmer_safe': self._create_farmer_safe_profile(),
            'farmer_efficient': self._create_farmer_efficient_profile(),
            'combat_aggressive': self._create_combat_aggressive_profile(),
            'combat_defensive': self._create_combat_defensive_profile(),
            'explorer_balanced': self._create_explorer_balanced_profile(),
            'social_cooperative': self._create_social_cooperative_profile()
        }
        
        self.logger.info("Gestionnaire de configuration initialisé")
    
    def get_config(self) -> DecisionConfig:
        """Retourne la configuration actuelle."""
        return self.config
    
    def update_priority_weights(self, new_weights: Dict[str, float]):
        """
        Met à jour les poids des priorités.
        
        Args:
            new_weights: Dictionnaire des nouveaux poids (nom: poids)
        """
        self.config.priority_weights.update(new_weights)
        self._save_config()
        self.logger.info(f"Poids des priorités mis à jour: {new_weights}")
    
    def set_risk_profile(self, profile_name: str, custom_value: Optional[float] = None):
        """
        Définit le profil de risque.
        
        Args:
            profile_name: Nom du profil ('very_safe', 'safe', 'balanced', etc.)
            custom_value: Valeur personnalisée (0.0 à 1.0) si non dans les profils prédéfinis
        """
        if profile_name in self.config.risk_profiles:
            risk_value = self.config.risk_profiles[profile_name]
        elif custom_value is not None:
            risk_value = max(0.0, min(1.0, custom_value))
        else:
            raise ValueError(f"Profil de risque '{profile_name}' non reconnu. "
                           f"Profils disponibles: {list(self.config.risk_profiles.keys())}")
        
        # Appliquer le profil de risque à toutes les situations
        for situation in self.config.preferred_strategies.keys():
            # Ajuster la tolérance au risque selon le profil
            pass  # Sera utilisé par DecisionEngine
        
        self._save_config()
        self.logger.info(f"Profil de risque défini: {profile_name} (valeur: {risk_value})")
    
    def update_time_limits(self, new_limits: Dict[str, float]):
        """
        Met à jour les limites de temps par action.
        
        Args:
            new_limits: Dictionnaire des nouvelles limites (action: secondes)
        """
        self.config.time_limits.update(new_limits)
        self._save_config()
        self.logger.info(f"Limites de temps mises à jour: {new_limits}")
    
    def update_activation_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Met à jour les seuils d'activation automatique.
        
        Args:
            new_thresholds: Dictionnaire des nouveaux seuils
        """
        self.config.activation_thresholds.update(new_thresholds)
        self._save_config()
        self.logger.info(f"Seuils d'activation mis à jour: {new_thresholds}")
    
    def set_preferred_strategies(self, strategy_preferences: Dict[str, str]):
        """
        Définit les stratégies préférées par situation.
        
        Args:
            strategy_preferences: Dictionnaire situation -> stratégie
        """
        # Valider que les stratégies existent
        valid_strategies = [s.value for s in StrategyType]
        for situation, strategy in strategy_preferences.items():
            if strategy not in valid_strategies:
                raise ValueError(f"Stratégie '{strategy}' non valide. "
                               f"Stratégies disponibles: {valid_strategies}")
        
        self.config.preferred_strategies.update(strategy_preferences)
        self._save_config()
        self.logger.info(f"Préférences de stratégies mises à jour: {strategy_preferences}")
    
    def apply_profile(self, profile_name: str):
        """
        Applique un profil de configuration prédéfini.
        
        Args:
            profile_name: Nom du profil à appliquer
        """
        if profile_name not in self.predefined_profiles:
            raise ValueError(f"Profil '{profile_name}' non trouvé. "
                           f"Profils disponibles: {list(self.predefined_profiles.keys())}")
        
        profile_config = self.predefined_profiles[profile_name]
        
        # Appliquer la configuration du profil
        self.config.priority_weights.update(profile_config.priority_weights)
        self.config.time_limits.update(profile_config.time_limits)
        self.config.activation_thresholds.update(profile_config.activation_thresholds)
        self.config.preferred_strategies.update(profile_config.preferred_strategies)
        
        self._save_config()
        self.logger.info(f"Profil '{profile_name}' appliqué avec succès")
    
    def create_custom_profile(self, profile_name: str, config: DecisionConfig):
        """
        Crée un profil personnalisé.
        
        Args:
            profile_name: Nom du nouveau profil
            config: Configuration du profil
        """
        self.predefined_profiles[profile_name] = config
        
        # Sauvegarder le profil personnalisé
        profiles_file = self.config_dir / "custom_profiles.json"
        try:
            if profiles_file.exists():
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    custom_profiles = json.load(f)
            else:
                custom_profiles = {}
            
            custom_profiles[profile_name] = asdict(config)
            
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(custom_profiles, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Profil personnalisé '{profile_name}' créé et sauvegardé")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du profil: {e}")
    
    def get_available_profiles(self) -> List[str]:
        """Retourne la liste des profils disponibles."""
        return list(self.predefined_profiles.keys())
    
    def get_profile_description(self, profile_name: str) -> str:
        """
        Retourne la description d'un profil.
        
        Args:
            profile_name: Nom du profil
            
        Returns:
            Description du profil
        """
        descriptions = {
            'farmer_safe': "Farming sécurisé avec priorité à la survie",
            'farmer_efficient': "Farming efficace avec optimisation du temps",
            'combat_aggressive': "Combat agressif pour maximiser les gains",
            'combat_defensive': "Combat défensif avec priorité à la sécurité",
            'explorer_balanced': "Exploration équilibrée entre risque et efficacité",
            'social_cooperative': "Jeu coopératif avec interactions sociales"
        }
        
        return descriptions.get(profile_name, "Profil personnalisé")
    
    def configure_decision_engine(self, engine: DecisionEngine):
        """
        Configure un moteur de décision avec la configuration actuelle.
        
        Args:
            engine: Instance du moteur de décision à configurer
        """
        # Configurer les poids
        engine.configure_priorities(self.config.priority_weights)
        
        # Charger la configuration sauvegardée
        if self.engine_config_file.exists():
            engine.load_config(str(self.engine_config_file))
        
        self.logger.info("Moteur de décision configuré")
    
    def configure_strategy_selector(self, selector: StrategySelector):
        """
        Configure un sélecteur de stratégies avec la configuration actuelle.
        
        Args:
            selector: Instance du sélecteur de stratégies à configurer
        """
        # Charger la configuration sauvegardée
        if self.strategy_config_file.exists():
            selector.load_config(str(self.strategy_config_file))
        
        self.logger.info("Sélecteur de stratégies configuré")
    
    def save_engine_state(self, engine: DecisionEngine):
        """Sauvegarde l'état d'un moteur de décision."""
        engine.save_config(str(self.engine_config_file))
    
    def save_strategy_state(self, selector: StrategySelector):
        """Sauvegarde l'état d'un sélecteur de stratégies."""
        selector.save_config(str(self.strategy_config_file))
    
    def export_config(self, export_path: str):
        """
        Exporte la configuration complète vers un fichier.
        
        Args:
            export_path: Chemin du fichier d'export
        """
        export_data = {
            'config': asdict(self.config),
            'profiles': {name: asdict(profile) for name, profile in self.predefined_profiles.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration exportée vers: {export_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export: {e}")
    
    def import_config(self, import_path: str):
        """
        Importe une configuration depuis un fichier.
        
        Args:
            import_path: Chemin du fichier à importer
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Importer la configuration principale
            config_dict = import_data.get('config', {})
            self.config = DecisionConfig(**config_dict)
            
            # Importer les profils
            profiles_dict = import_data.get('profiles', {})
            for name, profile_data in profiles_dict.items():
                self.predefined_profiles[name] = DecisionConfig(**profile_data)
            
            self._save_config()
            self.logger.info(f"Configuration importée depuis: {import_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'import: {e}")
    
    # Méthodes privées
    
    def _load_config(self) -> DecisionConfig:
        """Charge la configuration depuis le fichier ou utilise les défauts."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                return DecisionConfig(**config_dict)
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de la configuration: {e}")
                return self.default_config
        else:
            return self.default_config
    
    def _save_config(self):
        """Sauvegarde la configuration actuelle."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def _create_farmer_safe_profile(self) -> DecisionConfig:
        """Crée le profil farming sécurisé."""
        return DecisionConfig(
            priority_weights={
                'survival': 2.5,
                'objective': 1.2,
                'efficiency': 0.8,
                'maintenance': 0.7,
                'social': 0.2
            },
            risk_profiles={'default': 0.2},  # Très prudent
            time_limits={
                'combat': 60.0,      # Combat minimal
                'movement': 300.0,   # Déplacement sécurisé
                'profession': 3600.0, # Farm long
                'inventory': 180.0,
                'maintenance': 600.0,
                'social': 120.0
            },
            activation_thresholds={
                'critical_health': 30.0,
                'low_health': 60.0,
                'critical_mana': 20.0,
                'low_mana': 50.0,
                'full_inventory': 85.0,
                'high_inventory': 70.0,
                'high_risk_enemy_count': 2,
                'max_session_time': 18000.0
            },
            learning_rate=0.05,  # Apprentissage conservateur
            history_limit=500,
            preferred_strategies={
                'peaceful_farming': StrategyType.DEFENSIVE.value,
                'dangerous_area': StrategyType.STEALTH.value,
                'boss_fight': StrategyType.DEFENSIVE.value,
                'crowded_area': StrategyType.STEALTH.value,
                'low_resources': StrategyType.DEFENSIVE.value,
                'inventory_full': StrategyType.EFFICIENT.value
            }
        )
    
    def _create_farmer_efficient_profile(self) -> DecisionConfig:
        """Crée le profil farming efficace."""
        return DecisionConfig(
            priority_weights={
                'survival': 1.5,
                'objective': 2.0,
                'efficiency': 2.2,
                'maintenance': 0.8,
                'social': 0.1
            },
            risk_profiles={'default': 0.6},  # Modérément agressif
            time_limits={
                'combat': 120.0,
                'movement': 90.0,     # Déplacement rapide
                'profession': 2400.0,
                'inventory': 60.0,    # Gestion rapide
                'maintenance': 180.0,
                'social': 30.0
            },
            activation_thresholds={
                'critical_health': 15.0,
                'low_health': 35.0,
                'critical_mana': 5.0,
                'low_mana': 25.0,
                'full_inventory': 95.0,
                'high_inventory': 80.0,
                'high_risk_enemy_count': 4,
                'max_session_time': 12000.0
            },
            learning_rate=0.15,  # Apprentissage rapide
            history_limit=1500,
            preferred_strategies={
                'peaceful_farming': StrategyType.EFFICIENT.value,
                'dangerous_area': StrategyType.EFFICIENT.value,
                'boss_fight': StrategyType.BALANCED.value,
                'crowded_area': StrategyType.EFFICIENT.value,
                'low_resources': StrategyType.EFFICIENT.value,
                'inventory_full': StrategyType.EFFICIENT.value
            }
        )
    
    def _create_combat_aggressive_profile(self) -> DecisionConfig:
        """Crée le profil combat agressif."""
        return DecisionConfig(
            priority_weights={
                'survival': 1.3,
                'objective': 2.2,
                'efficiency': 1.8,
                'maintenance': 0.5,
                'social': 0.8
            },
            risk_profiles={'default': 0.8},  # Très agressif
            time_limits={
                'combat': 600.0,     # Combat prolongé
                'movement': 120.0,
                'profession': 600.0,
                'inventory': 90.0,
                'maintenance': 120.0,
                'social': 300.0
            },
            activation_thresholds={
                'critical_health': 10.0,
                'low_health': 25.0,
                'critical_mana': 5.0,
                'low_mana': 15.0,
                'full_inventory': 95.0,
                'high_inventory': 85.0,
                'high_risk_enemy_count': 6,
                'max_session_time': 10800.0
            },
            learning_rate=0.2,   # Apprentissage agressif
            history_limit=2000,
            preferred_strategies={
                'peaceful_farming': StrategyType.AGGRESSIVE.value,
                'dangerous_area': StrategyType.AGGRESSIVE.value,
                'boss_fight': StrategyType.AGGRESSIVE.value,
                'crowded_area': StrategyType.AGGRESSIVE.value,
                'low_resources': StrategyType.BALANCED.value,
                'inventory_full': StrategyType.EFFICIENT.value
            }
        )
    
    def _create_combat_defensive_profile(self) -> DecisionConfig:
        """Crée le profil combat défensif."""
        return DecisionConfig(
            priority_weights={
                'survival': 2.8,
                'objective': 1.0,
                'efficiency': 0.7,
                'maintenance': 1.2,
                'social': 0.5
            },
            risk_profiles={'default': 0.25},  # Très défensif
            time_limits={
                'combat': 180.0,     # Combat court
                'movement': 600.0,   # Déplacement sûr
                'profession': 1800.0,
                'inventory': 300.0,
                'maintenance': 900.0,
                'social': 180.0
            },
            activation_thresholds={
                'critical_health': 40.0,
                'low_health': 70.0,
                'critical_mana': 25.0,
                'low_mana': 60.0,
                'full_inventory': 80.0,
                'high_inventory': 65.0,
                'high_risk_enemy_count': 2,
                'max_session_time': 21600.0
            },
            learning_rate=0.08,  # Apprentissage prudent
            history_limit=800,
            preferred_strategies={
                'peaceful_farming': StrategyType.DEFENSIVE.value,
                'dangerous_area': StrategyType.DEFENSIVE.value,
                'boss_fight': StrategyType.DEFENSIVE.value,
                'crowded_area': StrategyType.STEALTH.value,
                'low_resources': StrategyType.DEFENSIVE.value,
                'inventory_full': StrategyType.DEFENSIVE.value
            }
        )
    
    def _create_explorer_balanced_profile(self) -> DecisionConfig:
        """Crée le profil exploration équilibré."""
        return DecisionConfig(
            priority_weights={
                'survival': 1.8,
                'objective': 1.5,
                'efficiency': 1.2,
                'maintenance': 0.8,
                'social': 0.4
            },
            risk_profiles={'default': 0.5},  # Équilibré
            time_limits={
                'combat': 240.0,
                'movement': 600.0,    # Exploration longue
                'profession': 900.0,
                'inventory': 120.0,
                'maintenance': 300.0,
                'social': 240.0
            },
            activation_thresholds={
                'critical_health': 25.0,
                'low_health': 45.0,
                'critical_mana': 15.0,
                'low_mana': 35.0,
                'full_inventory': 88.0,
                'high_inventory': 72.0,
                'high_risk_enemy_count': 3,
                'max_session_time': 16200.0
            },
            learning_rate=0.12,
            history_limit=1200,
            preferred_strategies={
                'peaceful_farming': StrategyType.BALANCED.value,
                'dangerous_area': StrategyType.BALANCED.value,
                'boss_fight': StrategyType.BALANCED.value,
                'crowded_area': StrategyType.SOCIAL.value,
                'low_resources': StrategyType.DEFENSIVE.value,
                'inventory_full': StrategyType.EFFICIENT.value
            }
        )
    
    def _create_social_cooperative_profile(self) -> DecisionConfig:
        """Crée le profil social coopératif."""
        return DecisionConfig(
            priority_weights={
                'survival': 1.5,
                'objective': 1.3,
                'efficiency': 1.0,
                'maintenance': 0.7,
                'social': 1.8
            },
            risk_profiles={'default': 0.4},  # Modérément prudent
            time_limits={
                'combat': 300.0,
                'movement': 240.0,
                'profession': 1200.0,
                'inventory': 180.0,
                'maintenance': 240.0,
                'social': 1800.0     # Interactions longues
            },
            activation_thresholds={
                'critical_health': 20.0,
                'low_health': 40.0,
                'critical_mana': 10.0,
                'low_mana': 30.0,
                'full_inventory': 85.0,
                'high_inventory': 70.0,
                'high_risk_enemy_count': 3,
                'max_session_time': 18000.0
            },
            learning_rate=0.1,
            history_limit=1000,
            preferred_strategies={
                'peaceful_farming': StrategyType.SOCIAL.value,
                'dangerous_area': StrategyType.SOCIAL.value,
                'boss_fight': StrategyType.SOCIAL.value,
                'crowded_area': StrategyType.SOCIAL.value,
                'low_resources': StrategyType.BALANCED.value,
                'inventory_full': StrategyType.BALANCED.value
            }
        )