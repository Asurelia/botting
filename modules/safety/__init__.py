"""
Module Safety - Système de sécurité et simulation comportementale
Fournit des outils pour simuler un comportement humain naturel et éviter la détection
"""

from .human_behavior import (
    HumanBehaviorSimulator,
    HumanProfile, 
    ActionType,
    BEGINNER_PROFILE,
    EXPERIENCED_PROFILE,
    CASUAL_PROFILE
)

from .session_manager import (
    SessionManager,
    SessionLimits,
    SessionState,
    BreakReason,
    SessionStats
)

from .detection_avoidance import (
    AntiDetectionSystem,
    DetectionRisk,
    BehaviorCategory,
    BehaviorPattern,
    DetectionMetrics,
    STEALTH_CONFIGS
)

__version__ = "1.0.0"
__author__ = "Claude AI"

# Configuration par défaut recommandée
DEFAULT_SAFETY_CONFIG = {
    "human_behavior": {
        "profile": "casual",
        "enable_errors": True,
        "enable_pauses": True,
        "fatigue_simulation": True
    },
    "session_limits": {
        "max_continuous_hours": 2.0,
        "max_daily_hours": 8.0,
        "mandatory_break_minutes": 15,
        "short_break_frequency_minutes": 30
    },
    "anti_detection": {
        "stealth_level": "careful",
        "analysis_frequency_minutes": 5,
        "auto_stealth_mode": True,
        "risk_threshold": "medium"
    }
}

# Profils de sécurité prédéfinis
SAFETY_PROFILES = {
    "conservative": {
        "human_profile": BEGINNER_PROFILE,
        "session_limits": SessionLimits(
            max_continuous_minutes=90,
            max_daily_minutes=360,
            short_break_frequency=20
        ),
        "stealth_config": "paranoid"
    },
    "balanced": {
        "human_profile": CASUAL_PROFILE,
        "session_limits": SessionLimits(
            max_continuous_minutes=120,
            max_daily_minutes=480,
            short_break_frequency=30
        ),
        "stealth_config": "careful"
    },
    "aggressive": {
        "human_profile": EXPERIENCED_PROFILE,
        "session_limits": SessionLimits(
            max_continuous_minutes=180,
            max_daily_minutes=600,
            short_break_frequency=45
        ),
        "stealth_config": "bold"
    }
}


class SafetyManager:
    """
    Gestionnaire principal du système de sécurité
    Coordonne tous les modules de sécurité pour un comportement cohérent
    """
    
    def __init__(self, profile: str = "balanced", config_file: str = None):
        """
        Initialise le gestionnaire de sécurité avec un profil prédéfini
        
        Args:
            profile: Profil de sécurité ('conservative', 'balanced', 'aggressive')
            config_file: Fichier de configuration personnalisée (optionnel)
        """
        if profile not in SAFETY_PROFILES:
            raise ValueError(f"Profil inconnu: {profile}. Profils disponibles: {list(SAFETY_PROFILES.keys())}")
        
        profile_config = SAFETY_PROFILES[profile]
        
        # Initialisation des composants
        self.human_simulator = HumanBehaviorSimulator(profile_config["human_profile"])
        self.session_manager = SessionManager(profile_config["session_limits"])
        self.detection_system = AntiDetectionSystem(config_file)
        
        # Configuration du niveau de furtivité
        stealth_level = profile_config["stealth_config"]
        if stealth_level in STEALTH_CONFIGS:
            stealth_config = STEALTH_CONFIGS[stealth_level]
            self.human_simulator.adjust_profile(
                error_rate=stealth_config["base_error_rate"],
                reaction_variance=stealth_config["timing_variance"],
                pause_frequency=stealth_config["pause_frequency"]
            )
        
        # Intégration des callbacks
        self._setup_integration()
        
        self.active = False
        self.profile_name = profile
    
    def start_safe_session(self) -> bool:
        """
        Démarre une session sécurisée avec tous les systèmes de protection
        """
        if not self.session_manager.start_session():
            return False
        
        self.active = True
        
        # Analyse initiale du comportement
        self.detection_system.analyze_current_behavior()
        
        return True
    
    def perform_safe_action(self, action_type: str, action_func, *args, **kwargs):
        """
        Exécute une action avec toutes les protections de sécurité
        """
        if not self.active:
            raise RuntimeError("Aucune session sécurisée active")
        
        # Conversion du type d'action
        if hasattr(ActionType, action_type.upper()):
            action_enum = getattr(ActionType, action_type.upper())
        else:
            action_enum = ActionType.CLICK  # Par défaut
        
        # Enregistrement pour la gestion de session
        self.session_manager.record_action(action_type)
        
        # Exécution avec simulation comportementale
        result = self.human_simulator.perform_action(action_enum, action_func, *args, **kwargs)
        
        # Enregistrement pour l'anti-détection
        self.detection_system.record_game_event(action_type, {
            'success': result['success'],
            'error_simulated': result['error_simulated'],
            'timing': result['delay_used'],
            'fatigue': result['fatigue_level']
        })
        
        return result
    
    def check_safety_status(self) -> dict:
        """
        Retourne l'état complet du système de sécurité
        """
        return {
            "profile": self.profile_name,
            "active": self.active,
            "session_info": self.session_manager.get_session_info(),
            "behavior_stats": self.human_simulator.get_session_stats(),
            "detection_risk": self.detection_system.analyze_current_behavior().value,
            "stealth_recommendations": self.detection_system.get_stealth_recommendations()
        }
    
    def end_safe_session(self):
        """
        Termine proprement la session sécurisée
        """
        if self.active:
            self.session_manager.end_session()
            self.active = False
    
    def _setup_integration(self):
        """Configure l'intégration entre les modules"""
        
        # Callback pour les pauses de session
        def on_session_break(reason, duration):
            # Réinitialiser la fatigue pendant les longues pauses
            if duration > 10:
                self.human_simulator.reset_fatigue()
        
        # Callback pour la reprise de session  
        def on_session_resume(reason):
            # Réanalyse du comportement après reprise
            self.detection_system.analyze_current_behavior()
        
        self.session_manager.add_break_callback(on_session_break)
        self.session_manager.add_resume_callback(on_session_resume)


__all__ = [
    "SafetyManager",
    "HumanBehaviorSimulator", 
    "SessionManager",
    "AntiDetectionSystem",
    "HumanProfile",
    "SessionLimits", 
    "ActionType",
    "SessionState",
    "BreakReason",
    "DetectionRisk",
    "BehaviorCategory",
    "SAFETY_PROFILES",
    "DEFAULT_SAFETY_CONFIG"
]