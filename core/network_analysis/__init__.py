#!/usr/bin/env python3
"""
Module d'analyse réseau sécurisé pour DOFUS
Système complet de capture, analyse et monitoring du trafic réseau

Ce module fournit un ensemble d'outils pour l'analyse éthique du trafic réseau
DOFUS, incluant la capture de paquets, l'analyse de protocole, le monitoring
de sécurité et l'injection contrôlée.

Modules inclus:
- packet_sniffer: Capture passive de paquets avec Scapy/WinDivert
- mitm_proxy: Proxy MITM transparent pour interception SSL/TLS
- protocol_analyzer: Analyse et désobfuscation du protocole DOFUS Unity
- network_monitor_safe: Monitoring temps réel éthique et sécurisé
- packet_injector: Injection contrôlée pour tests et automatisation

Fonctionnalités principales:
- Capture passive sans modification client
- Analyse de protocole respectueuse des ToS
- Monitoring de sécurité avec alertes
- Interface plugins extensible
- Logging sécurisé et chiffrement
- Rate limiting et contrôles de sécurité

Usage éthique:
Ce module est conçu pour l'analyse locale et éducative uniquement.
Respect strict des ToS d'Ankama et des réglementations applicables.

Auteur: Système d'automatisation DOFUS
Date: 2025-08-31
Licence: Usage éducatif et recherche uniquement
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configuration du logging du module
logger = logging.getLogger(__name__)

# Version du module
__version__ = "1.0.0"
__author__ = "Système d'automatisation DOFUS"
__license__ = "Usage éducatif et recherche uniquement"

# Vérification des dépendances
DEPENDENCIES_STATUS = {}

# Scapy pour la capture de paquets
try:
    import scapy
    DEPENDENCIES_STATUS['scapy'] = True
except ImportError:
    DEPENDENCIES_STATUS['scapy'] = False
    logger.warning("Scapy non disponible - capture de paquets désactivée")

# mitmproxy pour le proxy MITM
try:
    import mitmproxy
    DEPENDENCIES_STATUS['mitmproxy'] = True
except ImportError:
    DEPENDENCIES_STATUS['mitmproxy'] = False
    logger.warning("mitmproxy non disponible - proxy MITM désactivé")

# psutil pour le monitoring système
try:
    import psutil
    DEPENDENCIES_STATUS['psutil'] = True
except ImportError:
    DEPENDENCIES_STATUS['psutil'] = False
    logger.warning("psutil non disponible - monitoring système limité")

# cryptography pour le chiffrement
try:
    import cryptography
    DEPENDENCIES_STATUS['cryptography'] = True
except ImportError:
    DEPENDENCIES_STATUS['cryptography'] = False
    logger.warning("cryptography non disponible - chiffrement désactivé")

# numpy pour les analyses statistiques
try:
    import numpy
    DEPENDENCIES_STATUS['numpy'] = True
except ImportError:
    DEPENDENCIES_STATUS['numpy'] = False
    logger.warning("numpy non disponible - analyses statistiques limitées")

# flask pour les interfaces web
try:
    import flask
    DEPENDENCIES_STATUS['flask'] = True
except ImportError:
    DEPENDENCIES_STATUS['flask'] = False
    logger.warning("flask non disponible - interfaces web désactivées")


def check_dependencies() -> Dict[str, bool]:
    """
    Vérifie le statut des dépendances du module
    
    Returns:
        Dict contenant le statut de chaque dépendance
    """
    return DEPENDENCIES_STATUS.copy()


def get_missing_dependencies() -> List[str]:
    """
    Retourne la liste des dépendances manquantes
    
    Returns:
        Liste des dépendances non installées
    """
    return [dep for dep, status in DEPENDENCIES_STATUS.items() if not status]


def install_requirements_message() -> str:
    """
    Génère un message d'installation pour les dépendances manquantes
    
    Returns:
        Message avec les commandes d'installation
    """
    missing = get_missing_dependencies()
    
    if not missing:
        return "Toutes les dépendances sont installées."
    
    requirements_map = {
        'scapy': 'scapy',
        'mitmproxy': 'mitmproxy',
        'psutil': 'psutil',
        'cryptography': 'cryptography',
        'numpy': 'numpy',
        'flask': 'flask flask-socketio'
    }
    
    packages = []
    for dep in missing:
        if dep in requirements_map:
            packages.append(requirements_map[dep])
    
    message = f"Dépendances manquantes: {', '.join(missing)}\n"
    message += f"Installation: pip install {' '.join(packages)}"
    
    return message


# Import conditionnel des modules selon les dépendances disponibles
_modules_loaded = {}

# PacketSniffer (nécessite Scapy)
if DEPENDENCIES_STATUS.get('scapy', False):
    try:
        from .packet_sniffer import PacketSniffer, PacketLogger, ProcessTracker
        from .packet_sniffer import CaptureMode, PacketInfo
        from .packet_sniffer import dofus_only_filter, large_packets_filter, tcp_only_filter
        from .packet_sniffer import packet_counter_callback, suspicious_pattern_callback
        _modules_loaded['packet_sniffer'] = True
        logger.info("Module packet_sniffer chargé")
    except ImportError as e:
        _modules_loaded['packet_sniffer'] = False
        logger.error(f"Erreur chargement packet_sniffer: {e}")
else:
    _modules_loaded['packet_sniffer'] = False

# MITMProxy (nécessite mitmproxy)
if DEPENDENCIES_STATUS.get('mitmproxy', False):
    try:
        from .mitm_proxy import MITMProxy, DOFUSProxyAddon, CertificateManager
        from .mitm_proxy import ProxyMode, InterceptionType
        from .mitm_proxy import InterceptedRequest, InterceptedResponse
        from .mitm_proxy import log_dofus_requests, log_dofus_responses, save_dofus_traffic
        _modules_loaded['mitm_proxy'] = True
        logger.info("Module mitm_proxy chargé")
    except ImportError as e:
        _modules_loaded['mitm_proxy'] = False
        logger.error(f"Erreur chargement mitm_proxy: {e}")
else:
    _modules_loaded['mitm_proxy'] = False

# ProtocolAnalyzer (toujours disponible)
try:
    from .protocol_analyzer import ProtocolAnalyzer, PatternRecognizer, DataDeobfuscator
    from .protocol_analyzer import MessageType, DataFormat, CompressionType
    from .protocol_analyzer import MessagePattern, AnalyzedMessage
    from .protocol_analyzer import log_interesting_messages, detect_anomalies, learn_from_patterns
    _modules_loaded['protocol_analyzer'] = True
    logger.info("Module protocol_analyzer chargé")
except ImportError as e:
    _modules_loaded['protocol_analyzer'] = False
    logger.error(f"Erreur chargement protocol_analyzer: {e}")

# NetworkMonitorSafe (nécessite psutil de préférence)
try:
    from .network_monitor_safe import NetworkMonitorSafe, DOFUSProcessMonitor
    from .network_monitor_safe import NetworkStatsCollector, SecurityAnalyzer
    from .network_monitor_safe import AlertLevel, ConnectionState, MetricType
    from .network_monitor_safe import NetworkMetric, ConnectionInfo, SecurityAlert
    from .network_monitor_safe import log_high_bandwidth_usage, alert_security_issues
    from .network_monitor_safe import monitor_dofus_connections
    _modules_loaded['network_monitor_safe'] = True
    logger.info("Module network_monitor_safe chargé")
except ImportError as e:
    _modules_loaded['network_monitor_safe'] = False
    logger.error(f"Erreur chargement network_monitor_safe: {e}")

# PacketInjectorSafe (nécessite Scapy de préférence)
try:
    from .packet_injector import PacketInjectorSafe, PacketBuilder, SecurityValidator
    from .packet_injector import RateLimiter, InjectionTarget, PacketTemplate, InjectionJob
    from .packet_injector import InjectionMode, PacketType, InjectionResult
    from .packet_injector import log_injection_results, monitor_job_completion
    from .packet_injector import security_alert_on_block
    _modules_loaded['packet_injector'] = True
    logger.info("Module packet_injector chargé")
except ImportError as e:
    _modules_loaded['packet_injector'] = False
    logger.error(f"Erreur chargement packet_injector: {e}")


def get_loaded_modules() -> Dict[str, bool]:
    """
    Retourne le statut de chargement des modules
    
    Returns:
        Dict indiquant quels modules ont été chargés avec succès
    """
    return _modules_loaded.copy()


def create_network_analysis_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Crée un système d'analyse réseau complet
    
    Args:
        config: Configuration optionnelle du système
        
    Returns:
        Dict contenant les composants initialisés
    """
    if config is None:
        config = {}
    
    system = {
        'components': {},
        'status': {},
        'config': config
    }
    
    # Configuration par défaut
    default_config = {
        'capture_mode': 'passive',
        'mitm_port': 8080,
        'monitoring_interval': 1.0,
        'injection_mode': 'simulation',
        'log_dir': 'logs/network_analysis',
        'cache_dir': 'data/network_analysis'
    }
    
    # Fusion avec la configuration fournie
    effective_config = {**default_config, **config}
    system['config'] = effective_config
    
    # Créer les répertoires nécessaires
    log_dir = Path(effective_config['log_dir'])
    cache_dir = Path(effective_config['cache_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialiser les composants disponibles
    try:
        # PacketSniffer
        if _modules_loaded.get('packet_sniffer', False):
            from .packet_sniffer import PacketSniffer, CaptureMode
            mode = CaptureMode.PASSIVE if effective_config['capture_mode'] == 'passive' else CaptureMode.MONITOR
            
            system['components']['packet_sniffer'] = PacketSniffer(
                mode=mode,
                log_dir=str(log_dir / 'sniffer')
            )
            system['status']['packet_sniffer'] = 'initialized'
        else:
            system['status']['packet_sniffer'] = 'unavailable'
        
        # MITMProxy
        if _modules_loaded.get('mitm_proxy', False):
            from .mitm_proxy import MITMProxy, ProxyMode
            
            system['components']['mitm_proxy'] = MITMProxy(
                listen_port=effective_config['mitm_port'],
                mode=ProxyMode.EXPLICIT,
                log_dir=str(log_dir / 'mitm')
            )
            system['status']['mitm_proxy'] = 'initialized'
        else:
            system['status']['mitm_proxy'] = 'unavailable'
        
        # ProtocolAnalyzer
        if _modules_loaded.get('protocol_analyzer', False):
            from .protocol_analyzer import ProtocolAnalyzer
            
            system['components']['protocol_analyzer'] = ProtocolAnalyzer(
                cache_dir=str(cache_dir / 'protocol'),
                log_dir=str(log_dir / 'protocol')
            )
            system['status']['protocol_analyzer'] = 'initialized'
        else:
            system['status']['protocol_analyzer'] = 'unavailable'
        
        # NetworkMonitorSafe
        if _modules_loaded.get('network_monitor_safe', False):
            from .network_monitor_safe import NetworkMonitorSafe
            
            system['components']['network_monitor'] = NetworkMonitorSafe(
                monitoring_interval=effective_config['monitoring_interval'],
                log_dir=str(log_dir / 'monitor')
            )
            system['status']['network_monitor'] = 'initialized'
        else:
            system['status']['network_monitor'] = 'unavailable'
        
        # PacketInjectorSafe
        if _modules_loaded.get('packet_injector', False):
            from .packet_injector import PacketInjectorSafe, InjectionMode
            
            mode = InjectionMode.SIMULATION
            if effective_config['injection_mode'] == 'controlled':
                mode = InjectionMode.CONTROLLED
            elif effective_config['injection_mode'] == 'local_only':
                mode = InjectionMode.LOCAL_ONLY
            
            system['components']['packet_injector'] = PacketInjectorSafe(
                mode=mode,
                log_dir=str(log_dir / 'injector')
            )
            system['status']['packet_injector'] = 'initialized'
        else:
            system['status']['packet_injector'] = 'unavailable'
        
    except Exception as e:
        logger.error(f"Erreur initialisation système: {e}")
        system['status']['system'] = f'error: {e}'
    else:
        system['status']['system'] = 'ready'
    
    # Compteurs de composants
    available_count = sum(1 for status in system['status'].values() 
                         if status == 'initialized')
    system['components_available'] = available_count
    system['components_total'] = len(_modules_loaded)
    
    logger.info(f"Système d'analyse réseau créé: {available_count}/{len(_modules_loaded)} composants disponibles")
    
    return system


def print_system_info():
    """Affiche les informations du système d'analyse réseau"""
    print("="*60)
    print("SYSTÈME D'ANALYSE RÉSEAU DOFUS")
    print("="*60)
    print(f"Version: {__version__}")
    print(f"Auteur: {__author__}")
    print(f"Licence: {__license__}")
    print()
    
    print("DÉPENDANCES:")
    for dep, status in DEPENDENCIES_STATUS.items():
        status_str = "✓ Installé" if status else "✗ Manquant"
        print(f"  {dep:15} {status_str}")
    
    missing = get_missing_dependencies()
    if missing:
        print(f"\nPour installer les dépendances manquantes:")
        print(install_requirements_message())
    
    print("\nMODULES CHARGÉS:")
    for module, status in _modules_loaded.items():
        status_str = "✓ Chargé" if status else "✗ Erreur"
        print(f"  {module:20} {status_str}")
    
    print("\nFONCTIONNALITÉS DISPONIBLES:")
    
    if _modules_loaded.get('packet_sniffer'):
        print("  ✓ Capture de paquets passive avec Scapy")
        print("  ✓ Filtrage et analyse du trafic DOFUS")
        print("  ✓ Logging sécurisé et chiffré")
    
    if _modules_loaded.get('mitm_proxy'):
        print("  ✓ Proxy MITM transparent SSL/TLS")
        print("  ✓ Interception HTTP/HTTPS")
        print("  ✓ Certificats auto-générés")
    
    if _modules_loaded.get('protocol_analyzer'):
        print("  ✓ Analyse de protocole DOFUS Unity")
        print("  ✓ Désobfuscation basique")
        print("  ✓ Reconnaissance de patterns")
    
    if _modules_loaded.get('network_monitor_safe'):
        print("  ✓ Monitoring temps réel éthique")
        print("  ✓ Alertes de sécurité")
        print("  ✓ Dashboard web (si Flask installé)")
    
    if _modules_loaded.get('packet_injector'):
        print("  ✓ Injection contrôlée de paquets")
        print("  ✓ Mode simulation sécurisé")
        print("  ✓ Rate limiting et validation")
    
    print("\nUSAGE ÉTHIQUE:")
    print("  • Analyse locale uniquement")
    print("  • Respect strict des ToS Ankama")
    print("  • Pas de reverse engineering illégal")
    print("  • Usage éducatif et recherche")
    print("="*60)


# Exposition des principales classes et fonctions
__all__ = [
    # Informations du module
    '__version__',
    '__author__',
    '__license__',
    
    # Fonctions utilitaires
    'check_dependencies',
    'get_missing_dependencies',
    'install_requirements_message',
    'get_loaded_modules',
    'create_network_analysis_system',
    'print_system_info',
    
    # Classes principales (si disponibles)
    'PacketSniffer', 'MITMProxy', 'ProtocolAnalyzer',
    'NetworkMonitorSafe', 'PacketInjectorSafe',
    
    # Énumérations
    'CaptureMode', 'ProxyMode', 'MessageType', 'AlertLevel',
    'InjectionMode', 'PacketType',
    
    # Structures de données
    'PacketInfo', 'InterceptedRequest', 'AnalyzedMessage',
    'NetworkMetric', 'SecurityAlert', 'InjectionJob'
]

# Filtrer __all__ selon les modules réellement chargés
_available_exports = []
for name in __all__:
    try:
        # Vérifier si l'objet existe dans l'espace de noms local
        if name in locals() or name in globals():
            _available_exports.append(name)
        else:
            # Vérifier si c'est une classe/fonction importée
            try:
                eval(name)
                _available_exports.append(name)
            except NameError:
                pass
    except:
        pass

__all__ = _available_exports

# Message de bienvenue au chargement du module
logger.info(f"Module network_analysis v{__version__} chargé")
logger.info(f"Modules disponibles: {sum(1 for loaded in _modules_loaded.values() if loaded)}/{len(_modules_loaded)}")

if get_missing_dependencies():
    logger.warning(f"Dépendances manquantes: {', '.join(get_missing_dependencies())}")
    logger.info("Utilisez install_requirements_message() pour les instructions d'installation")

# Configuration du logging par défaut pour le module
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)