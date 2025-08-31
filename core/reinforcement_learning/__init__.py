"""
Module d'Apprentissage par Renforcement Avancé pour DOFUS
========================================================

Ce module fournit un système complet d'apprentissage par renforcement
state-of-the-art pour l'automatisation intelligente du jeu DOFUS.

Auteur: Système RL DOFUS
Version: 1.0.0
"""

__version__ = "1.0.0"

# Configuration par défaut
DEFAULT_CONFIG = {
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "save_dir": "G:/Botting/models",
    "data_dir": "G:/Botting/data",
    "log_level": "INFO"
}

def get_default_config():
    """Retourne la configuration par défaut"""
    return DEFAULT_CONFIG.copy()

def setup_logging(level: str = "INFO"):
    """Configure le logging pour le module"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Module RL DOFUS initialisé (version {__version__})")

# Configuration automatique du logging
setup_logging(DEFAULT_CONFIG["log_level"])