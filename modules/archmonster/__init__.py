"""
Module d'alertes archimonstres DOFUS.
Système complet de détection, tracking et notification multi-canaux.

Architecture:
- ArchmonsterDetector: Détection via analyse écran/chat
- AlertSystem: Alertes multi-canaux (Discord, SMS, email, son)
- ArchmonsterTracker: Tracking spawns et prédictions temporelles
- ArchmonsterDatabase: Base de données SQLite avec statistiques
- NotificationManager: Orchestrateur principal avec IA

Fonctionnalités principales:
- Détection automatique archimonstres (visuelle + chat)
- Alertes instantanées configurables par canal
- Tracking historique et prédictions spawns
- Intelligence artificielle adaptive
- Dashboard statistiques complet
- Mode veille économe en ressources
"""

from .archmonster_detector import (
    ArchmonsterDetector,
    ArchmonsterDetection,
    ChatPatternMatcher,
    VisualDetector
)

from .alert_system import (
    AlertSystem,
    AlertPriority,
    AlertChannel,
    AlertConfig,
    AlertMessage,
    DiscordAlerter,
    TelegramAlerter,
    EmailAlerter,
    SoundAlerter,
    SystemAlerter
)

from .archmonster_tracker import (
    ArchmonsterTracker,
    SpawnPattern,
    SpawnPrediction,
    PatternAnalyzer,
    SpawnPredictor
)

from .archmonster_database import (
    ArchmonsterDatabase,
    ArchmonsterInfo,
    ZoneStats
)

from .notification_manager import (
    NotificationManager,
    NotificationMode,
    NotificationFilter,
    NotificationRule,
    FilterType,
    SmartNotificationEngine
)


__version__ = "1.0.0"
__author__ = "DOFUS Bot Team"


class ArchmonsterSystem:
    """
    Système principal d'alertes archimonstres.
    Interface simplifiée pour utilisation dans le bot principal.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise le système d'alertes archimonstres.
        
        Args:
            config: Configuration du système
        """
        self.config = config or self._get_default_config()
        self.notification_manager = NotificationManager(self.config)
        
        # Accès aux composants
        self.detector = self.notification_manager.detector
        self.alert_system = self.notification_manager.alert_system
        self.tracker = self.notification_manager.tracker
        self.database = self.notification_manager.database
    
    def start(self):
        """Démarre le système complet."""
        self.notification_manager.start()
    
    def stop(self):
        """Arrête le système complet."""
        self.notification_manager.stop()
    
    def set_watched_archmonsters(self, archmonsters: list):
        """
        Configure les archimonstres à surveiller.
        
        Args:
            archmonsters: Liste des noms d'archimonstres
        """
        self.detector.watched_archmonsters = set(archmonsters)
    
    def set_watched_zones(self, zones: list):
        """
        Configure les zones à surveiller.
        
        Args:
            zones: Liste des noms de zones
        """
        self.detector.watched_zones = set(zones)
    
    def add_discord_webhook(self, webhook_url: str, mention_roles: list = None):
        """
        Ajoute webhook Discord pour alertes.
        
        Args:
            webhook_url: URL du webhook Discord
            mention_roles: IDs des rôles à mentionner
        """
        # Configuration Discord dynamique
        discord_config = {
            'webhook_url': webhook_url,
            'username': 'DOFUS Archimonstres',
            'mention_roles': mention_roles or []
        }
        
        # Ajouter ou mettre à jour alerter Discord
        from .alert_system import DiscordAlerter, AlertChannel
        discord_alerter = DiscordAlerter(discord_config)
        self.alert_system.alerters[AlertChannel.DISCORD] = discord_alerter
    
    def set_notification_mode(self, mode: str):
        """
        Change le mode de notification.
        
        Args:
            mode: 'silent', 'normal', 'aggressive', 'ultra'
        """
        from .notification_manager import NotificationMode
        self.notification_manager.set_mode(NotificationMode(mode))
    
    def get_predictions(self, hours_ahead: int = 24) -> list:
        """
        Récupère prédictions de spawns.
        
        Args:
            hours_ahead: Heures à anticiper
            
        Returns:
            Liste des prédictions
        """
        return self.tracker.get_predictions(hours_ahead=hours_ahead)
    
    def get_statistics(self) -> dict:
        """
        Récupère statistiques complètes du système.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'notification_manager': self.notification_manager.get_statistics(),
            'detector': self.detector.get_detection_stats(),
            'alert_system': self.alert_system.get_stats(),
            'tracker': self.tracker.get_statistics(),
            'database': self.database.get_database_stats()
        }
        return stats
    
    def test_alerts(self):
        """Test tous les systèmes d'alerte."""
        self.notification_manager.test_notification("Test Archimonstre", "Zone Test")
        self.alert_system.test_alerters()
    
    def _get_default_config(self) -> dict:
        """Configuration par défaut du système."""
        return {
            'detector': {
                'database_path': 'data/archmonsters.db',
                'scan_interval': 2.0,
                'chat_buffer_size': 50,
                'enable_visual_detection': True,
                'enable_chat_detection': True,
                'templates_path': 'data/templates/archmonsters',
                'watched_archmonsters': [],
                'watched_zones': []
            },
            'alerts': {
                'alert_channels': [
                    {
                        'channel': 'sound',
                        'enabled': True,
                        'priority_threshold': 'MEDIUM',
                        'config': {
                            'sounds_path': 'data/sounds',
                            'default_sound': 'archmonster_alert.wav',
                            'archmonster_sounds': {}
                        }
                    },
                    {
                        'channel': 'system',
                        'enabled': True,
                        'priority_threshold': 'HIGH',
                        'config': {
                            'timeout': 5000
                        }
                    }
                ]
            },
            'tracker': {
                'database_path': 'data/archmonsters.db',
                'analysis_interval': 3600,  # 1 heure
                'patterns_file': 'data/spawn_patterns.json',
                'min_data_days': 7
            },
            'database_path': 'data/archmonsters.db',
            'mode': 'normal',
            'enabled': True,
            'rules_file': 'config/notification_rules.json'
        }


def create_archmonster_system(config: dict = None) -> ArchmonsterSystem:
    """
    Factory function pour créer système d'alertes archimonstres.
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance du système configuré
    """
    return ArchmonsterSystem(config)


# Configuration rapide
def quick_setup_discord(webhook_url: str, watched_archmonsters: list = None, 
                       watched_zones: list = None) -> ArchmonsterSystem:
    """
    Configuration rapide avec Discord.
    
    Args:
        webhook_url: URL webhook Discord
        watched_archmonsters: Archimonstres à surveiller
        watched_zones: Zones à surveiller
        
    Returns:
        Système configuré et prêt
    """
    system = create_archmonster_system()
    
    # Configurer Discord
    system.add_discord_webhook(webhook_url)
    
    # Configurer surveillance
    if watched_archmonsters:
        system.set_watched_archmonsters(watched_archmonsters)
    
    if watched_zones:
        system.set_watched_zones(watched_zones)
    
    return system


def quick_setup_basic(sound_enabled: bool = True) -> ArchmonsterSystem:
    """
    Configuration basique avec son système.
    
    Args:
        sound_enabled: Activer alertes sonores
        
    Returns:
        Système configuré
    """
    config = {
        'alerts': {
            'alert_channels': [
                {
                    'channel': 'sound',
                    'enabled': sound_enabled,
                    'priority_threshold': 'MEDIUM',
                    'config': {'sounds_path': 'data/sounds'}
                },
                {
                    'channel': 'system',
                    'enabled': True,
                    'priority_threshold': 'HIGH',
                    'config': {'timeout': 3000}
                }
            ]
        }
    }
    
    return create_archmonster_system(config)


# Exports principaux
__all__ = [
    # Classes principales
    'ArchmonsterSystem',
    'ArchmonsterDetector',
    'AlertSystem', 
    'ArchmonsterTracker',
    'ArchmonsterDatabase',
    'NotificationManager',
    
    # Types et enums
    'ArchmonsterDetection',
    'AlertPriority',
    'AlertChannel',
    'NotificationMode',
    'SpawnPattern',
    'SpawnPrediction',
    'ArchmonsterInfo',
    'ZoneStats',
    
    # Factory functions
    'create_archmonster_system',
    'quick_setup_discord',
    'quick_setup_basic',
    
    # Version
    '__version__'
]