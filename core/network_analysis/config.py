#!/usr/bin/env python3
"""
Configuration centralisée pour le module d'analyse réseau
Paramètres par défaut et validation pour tous les composants

Auteur: Système d'automatisation DOFUS
Date: 2025-08-31
Licence: Usage éducatif et recherche uniquement
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(Enum):
    """Niveaux de sécurité pour l'analyse réseau"""
    STRICT = "strict"          # Maximum sécurité, local seulement
    MODERATE = "moderate"      # Sécurité équilibrée
    PERMISSIVE = "permissive"  # Moins restrictif (développement)


@dataclass
class PacketSnifferConfig:
    """Configuration du packet sniffer"""
    # Mode de capture
    capture_mode: str = "passive"  # passive, monitor, debug
    interface: Optional[str] = None
    
    # Filtres
    capture_dofus_only: bool = True
    min_packet_size: int = 4
    max_packet_size: int = 10000
    
    # Performance
    buffer_size: int = 1000
    processing_threads: int = 2
    
    # Sécurité
    encrypt_logs: bool = True
    local_traffic_only: bool = True
    
    # Répertoires
    log_directory: str = "logs/network/sniffer"
    cache_directory: str = "data/network/sniffer"


@dataclass
class MITMProxyConfig:
    """Configuration du proxy MITM"""
    # Réseau
    listen_port: int = 8080
    listen_host: str = "127.0.0.1"
    mode: str = "explicit"  # explicit, transparent, reverse
    
    # SSL/TLS
    generate_ca_cert: bool = True
    ca_cert_path: Optional[str] = None
    ca_key_path: Optional[str] = None
    
    # Sécurité
    local_connections_only: bool = True
    max_connections: int = 100
    connection_timeout: int = 30
    
    # Logging
    log_requests: bool = True
    log_responses: bool = True
    encrypt_sensitive_data: bool = True
    
    # Répertoires
    cert_directory: str = "config/certificates"
    log_directory: str = "logs/network/mitm"


@dataclass
class ProtocolAnalyzerConfig:
    """Configuration de l'analyseur de protocole"""
    # Cache
    cache_patterns: bool = True
    max_cache_size: int = 10000
    cache_retention_hours: int = 24
    
    # Analyse
    enable_deobfuscation: bool = True
    enable_compression_detection: bool = True
    enable_format_detection: bool = True
    
    # Apprentissage
    learn_new_patterns: bool = True
    pattern_confidence_threshold: float = 0.5
    
    # Performance
    processing_timeout: float = 5.0
    max_payload_size: int = 1024 * 1024  # 1MB
    
    # Répertoires
    cache_directory: str = "data/network/protocol"
    log_directory: str = "logs/network/protocol"


@dataclass
class NetworkMonitorConfig:
    """Configuration du monitoring réseau"""
    # Collecte
    monitoring_interval: float = 1.0
    metrics_retention_hours: int = 24
    
    # Alertes
    enable_security_alerts: bool = True
    alert_bandwidth_threshold_mbps: float = 10.0
    alert_connection_threshold: int = 100
    alert_cpu_threshold: float = 80.0
    alert_memory_threshold: float = 85.0
    
    # Processus
    monitor_dofus_processes: bool = True
    process_update_interval: float = 5.0
    
    # Interface web
    enable_web_interface: bool = True
    web_port: int = 5000
    web_host: str = "127.0.0.1"
    
    # Répertoires
    log_directory: str = "logs/network/monitor"
    data_directory: str = "data/network/monitor"


@dataclass
class PacketInjectorConfig:
    """Configuration de l'injecteur de paquets"""
    # Mode de sécurité
    injection_mode: str = "simulation"  # simulation, local_only, controlled
    
    # Rate limiting
    max_packets_per_second: int = 10
    burst_size: int = 20
    
    # Validation
    validate_targets: bool = True
    validate_payloads: bool = True
    allow_external_targets: bool = False
    max_payload_size: int = 10000
    
    # Workers
    worker_threads: int = 2
    job_timeout_seconds: int = 300
    
    # Templates
    load_default_templates: bool = True
    custom_templates_directory: str = "config/injection_templates"
    
    # Répertoires
    log_directory: str = "logs/network/injector"
    template_directory: str = "data/network/templates"


@dataclass
class NetworkAnalysisConfig:
    """Configuration globale du système d'analyse réseau"""
    # Niveau de sécurité global
    security_level: SecurityLevel = SecurityLevel.STRICT
    
    # Composants
    packet_sniffer: PacketSnifferConfig = field(default_factory=PacketSnifferConfig)
    mitm_proxy: MITMProxyConfig = field(default_factory=MITMProxyConfig)
    protocol_analyzer: ProtocolAnalyzerConfig = field(default_factory=ProtocolAnalyzerConfig)
    network_monitor: NetworkMonitorConfig = field(default_factory=NetworkMonitorConfig)
    packet_injector: PacketInjectorConfig = field(default_factory=PacketInjectorConfig)
    
    # Répertoires globaux
    base_log_directory: str = "logs/network_analysis"
    base_data_directory: str = "data/network_analysis"
    base_config_directory: str = "config/network_analysis"
    
    # Logging global
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_to_console: bool = True
    rotate_logs: bool = True
    max_log_size_mb: int = 100
    
    def __post_init__(self):
        """Post-traitement de la configuration"""
        self._apply_security_level()
        self._create_directories()
    
    def _apply_security_level(self):
        """Applique le niveau de sécurité à tous les composants"""
        if self.security_level == SecurityLevel.STRICT:
            # Configuration sécurité maximale
            self.packet_sniffer.local_traffic_only = True
            self.packet_sniffer.encrypt_logs = True
            
            self.mitm_proxy.local_connections_only = True
            self.mitm_proxy.encrypt_sensitive_data = True
            
            self.protocol_analyzer.max_payload_size = 512 * 1024  # 512KB max
            
            self.network_monitor.enable_security_alerts = True
            
            self.packet_injector.injection_mode = "simulation"
            self.packet_injector.allow_external_targets = False
            self.packet_injector.validate_targets = True
            self.packet_injector.validate_payloads = True
            
        elif self.security_level == SecurityLevel.MODERATE:
            # Configuration équilibrée
            self.packet_sniffer.local_traffic_only = True
            self.mitm_proxy.local_connections_only = True
            self.packet_injector.injection_mode = "local_only"
            
        elif self.security_level == SecurityLevel.PERMISSIVE:
            # Configuration développement
            self.packet_sniffer.local_traffic_only = False
            self.mitm_proxy.local_connections_only = False
            self.packet_injector.injection_mode = "controlled"
    
    def _create_directories(self):
        """Crée les répertoires nécessaires"""
        directories = [
            self.base_log_directory,
            self.base_data_directory,
            self.base_config_directory,
            self.packet_sniffer.log_directory,
            self.packet_sniffer.cache_directory,
            self.mitm_proxy.cert_directory,
            self.mitm_proxy.log_directory,
            self.protocol_analyzer.cache_directory,
            self.protocol_analyzer.log_directory,
            self.network_monitor.log_directory,
            self.network_monitor.data_directory,
            self.packet_injector.log_directory,
            self.packet_injector.template_directory,
            self.packet_injector.custom_templates_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            'security_level': self.security_level.value,
            'packet_sniffer': {
                'capture_mode': self.packet_sniffer.capture_mode,
                'interface': self.packet_sniffer.interface,
                'capture_dofus_only': self.packet_sniffer.capture_dofus_only,
                'min_packet_size': self.packet_sniffer.min_packet_size,
                'max_packet_size': self.packet_sniffer.max_packet_size,
                'buffer_size': self.packet_sniffer.buffer_size,
                'processing_threads': self.packet_sniffer.processing_threads,
                'encrypt_logs': self.packet_sniffer.encrypt_logs,
                'local_traffic_only': self.packet_sniffer.local_traffic_only,
                'log_directory': self.packet_sniffer.log_directory,
                'cache_directory': self.packet_sniffer.cache_directory
            },
            'mitm_proxy': {
                'listen_port': self.mitm_proxy.listen_port,
                'listen_host': self.mitm_proxy.listen_host,
                'mode': self.mitm_proxy.mode,
                'generate_ca_cert': self.mitm_proxy.generate_ca_cert,
                'local_connections_only': self.mitm_proxy.local_connections_only,
                'max_connections': self.mitm_proxy.max_connections,
                'connection_timeout': self.mitm_proxy.connection_timeout,
                'log_requests': self.mitm_proxy.log_requests,
                'log_responses': self.mitm_proxy.log_responses,
                'encrypt_sensitive_data': self.mitm_proxy.encrypt_sensitive_data,
                'cert_directory': self.mitm_proxy.cert_directory,
                'log_directory': self.mitm_proxy.log_directory
            },
            'protocol_analyzer': {
                'cache_patterns': self.protocol_analyzer.cache_patterns,
                'max_cache_size': self.protocol_analyzer.max_cache_size,
                'cache_retention_hours': self.protocol_analyzer.cache_retention_hours,
                'enable_deobfuscation': self.protocol_analyzer.enable_deobfuscation,
                'enable_compression_detection': self.protocol_analyzer.enable_compression_detection,
                'enable_format_detection': self.protocol_analyzer.enable_format_detection,
                'learn_new_patterns': self.protocol_analyzer.learn_new_patterns,
                'pattern_confidence_threshold': self.protocol_analyzer.pattern_confidence_threshold,
                'processing_timeout': self.protocol_analyzer.processing_timeout,
                'max_payload_size': self.protocol_analyzer.max_payload_size,
                'cache_directory': self.protocol_analyzer.cache_directory,
                'log_directory': self.protocol_analyzer.log_directory
            },
            'network_monitor': {
                'monitoring_interval': self.network_monitor.monitoring_interval,
                'metrics_retention_hours': self.network_monitor.metrics_retention_hours,
                'enable_security_alerts': self.network_monitor.enable_security_alerts,
                'alert_bandwidth_threshold_mbps': self.network_monitor.alert_bandwidth_threshold_mbps,
                'alert_connection_threshold': self.network_monitor.alert_connection_threshold,
                'alert_cpu_threshold': self.network_monitor.alert_cpu_threshold,
                'alert_memory_threshold': self.network_monitor.alert_memory_threshold,
                'monitor_dofus_processes': self.network_monitor.monitor_dofus_processes,
                'process_update_interval': self.network_monitor.process_update_interval,
                'enable_web_interface': self.network_monitor.enable_web_interface,
                'web_port': self.network_monitor.web_port,
                'web_host': self.network_monitor.web_host,
                'log_directory': self.network_monitor.log_directory,
                'data_directory': self.network_monitor.data_directory
            },
            'packet_injector': {
                'injection_mode': self.packet_injector.injection_mode,
                'max_packets_per_second': self.packet_injector.max_packets_per_second,
                'burst_size': self.packet_injector.burst_size,
                'validate_targets': self.packet_injector.validate_targets,
                'validate_payloads': self.packet_injector.validate_payloads,
                'allow_external_targets': self.packet_injector.allow_external_targets,
                'max_payload_size': self.packet_injector.max_payload_size,
                'worker_threads': self.packet_injector.worker_threads,
                'job_timeout_seconds': self.packet_injector.job_timeout_seconds,
                'load_default_templates': self.packet_injector.load_default_templates,
                'custom_templates_directory': self.packet_injector.custom_templates_directory,
                'log_directory': self.packet_injector.log_directory,
                'template_directory': self.packet_injector.template_directory
            },
            'global': {
                'base_log_directory': self.base_log_directory,
                'base_data_directory': self.base_data_directory,
                'base_config_directory': self.base_config_directory,
                'log_level': self.log_level,
                'log_to_file': self.log_to_file,
                'log_to_console': self.log_to_console,
                'rotate_logs': self.rotate_logs,
                'max_log_size_mb': self.max_log_size_mb
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NetworkAnalysisConfig':
        """Crée une configuration depuis un dictionnaire"""
        config = cls()
        
        # Niveau de sécurité global
        if 'security_level' in config_dict:
            config.security_level = SecurityLevel(config_dict['security_level'])
        
        # Configuration des composants
        for component_name in ['packet_sniffer', 'mitm_proxy', 'protocol_analyzer', 
                              'network_monitor', 'packet_injector']:
            if component_name in config_dict:
                component_config = getattr(config, component_name)
                for key, value in config_dict[component_name].items():
                    if hasattr(component_config, key):
                        setattr(component_config, key, value)
        
        # Configuration globale
        if 'global' in config_dict:
            for key, value in config_dict['global'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def save_to_file(self, filepath: str):
        """Sauvegarde la configuration dans un fichier JSON"""
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NetworkAnalysisConfig':
        """Charge la configuration depuis un fichier JSON"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


def create_default_config(security_level: SecurityLevel = SecurityLevel.STRICT) -> NetworkAnalysisConfig:
    """
    Crée une configuration par défaut
    
    Args:
        security_level: Niveau de sécurité souhaité
        
    Returns:
        Configuration par défaut avec le niveau de sécurité spécifié
    """
    config = NetworkAnalysisConfig(security_level=security_level)
    return config


def create_development_config() -> NetworkAnalysisConfig:
    """Crée une configuration pour le développement"""
    config = NetworkAnalysisConfig(security_level=SecurityLevel.PERMISSIVE)
    
    # Ajustements pour le développement
    config.packet_sniffer.capture_mode = "debug"
    config.protocol_analyzer.learn_new_patterns = True
    config.network_monitor.monitoring_interval = 0.5
    config.packet_injector.injection_mode = "controlled"
    
    config.log_level = "DEBUG"
    config.log_to_console = True
    
    return config


def create_production_config() -> NetworkAnalysisConfig:
    """Crée une configuration pour la production"""
    config = NetworkAnalysisConfig(security_level=SecurityLevel.STRICT)
    
    # Ajustements pour la production
    config.packet_sniffer.encrypt_logs = True
    config.mitm_proxy.encrypt_sensitive_data = True
    config.network_monitor.enable_security_alerts = True
    config.packet_injector.injection_mode = "simulation"
    
    config.log_level = "WARNING"
    config.log_to_console = False
    config.rotate_logs = True
    
    return config


def validate_config(config: NetworkAnalysisConfig) -> Dict[str, Any]:
    """
    Valide une configuration et retourne les erreurs/avertissements
    
    Args:
        config: Configuration à valider
        
    Returns:
        Dict contenant les erreurs et avertissements
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validation des ports
    ports_used = []
    
    if config.mitm_proxy.listen_port:
        if config.mitm_proxy.listen_port in ports_used:
            validation_result['errors'].append(f"Port {config.mitm_proxy.listen_port} déjà utilisé")
        else:
            ports_used.append(config.mitm_proxy.listen_port)
    
    if config.network_monitor.web_port:
        if config.network_monitor.web_port in ports_used:
            validation_result['errors'].append(f"Port {config.network_monitor.web_port} déjà utilisé")
        else:
            ports_used.append(config.network_monitor.web_port)
    
    # Validation des seuils
    if config.network_monitor.monitoring_interval <= 0:
        validation_result['errors'].append("L'intervalle de monitoring doit être positif")
    
    if config.packet_injector.max_packets_per_second <= 0:
        validation_result['errors'].append("Le taux d'injection doit être positif")
    
    # Validation de cohérence sécurité
    if config.security_level == SecurityLevel.STRICT:
        if not config.packet_sniffer.local_traffic_only:
            validation_result['warnings'].append("Trafic externe autorisé en mode strict")
        
        if config.packet_injector.injection_mode != "simulation":
            validation_result['warnings'].append("Injection réelle autorisée en mode strict")
        
        if not config.mitm_proxy.local_connections_only:
            validation_result['warnings'].append("Connexions externes MITM autorisées en mode strict")
    
    # Validation des répertoires
    base_dirs = [
        config.base_log_directory,
        config.base_data_directory,
        config.base_config_directory
    ]
    
    for directory in base_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            validation_result['errors'].append(f"Impossible de créer {directory}: {e}")
    
    # Résultat final
    if validation_result['errors']:
        validation_result['valid'] = False
    
    return validation_result


def main():
    """Fonction principale pour test de la configuration"""
    print("=== Test du Module de Configuration ===")
    
    # Créer des configurations de test
    configs = {
        'strict': create_default_config(SecurityLevel.STRICT),
        'moderate': create_default_config(SecurityLevel.MODERATE),
        'permissive': create_default_config(SecurityLevel.PERMISSIVE),
        'development': create_development_config(),
        'production': create_production_config()
    }
    
    for name, config in configs.items():
        print(f"\n--- Configuration {name.upper()} ---")
        print(f"Niveau de sécurité: {config.security_level.value}")
        print(f"Mode injection: {config.packet_injector.injection_mode}")
        print(f"Trafic local seulement: {config.packet_sniffer.local_traffic_only}")
        print(f"Chiffrement logs: {config.packet_sniffer.encrypt_logs}")
        print(f"Niveau de log: {config.log_level}")
        
        # Validation
        validation = validate_config(config)
        if validation['valid']:
            print("✓ Configuration valide")
        else:
            print("✗ Configuration invalide:")
            for error in validation['errors']:
                print(f"  Erreur: {error}")
        
        if validation['warnings']:
            print("Avertissements:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
    
    # Test de sauvegarde/chargement
    print("\n--- Test Sauvegarde/Chargement ---")
    
    test_config = create_development_config()
    config_file = "config/test_network_config.json"
    
    try:
        # Sauvegarder
        test_config.save_to_file(config_file)
        print(f"✓ Configuration sauvegardée: {config_file}")
        
        # Charger
        loaded_config = NetworkAnalysisConfig.load_from_file(config_file)
        print(f"✓ Configuration chargée: {loaded_config.security_level.value}")
        
        # Vérifier l'équivalence
        if test_config.to_dict() == loaded_config.to_dict():
            print("✓ Sauvegarde/chargement cohérent")
        else:
            print("✗ Incohérence sauvegarde/chargement")
            
    except Exception as e:
        print(f"✗ Erreur test sauvegarde/chargement: {e}")
    
    print("\n=== Fin du Test ===")


if __name__ == "__main__":
    main()