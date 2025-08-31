"""
Interface de base pour tous les modules du bot
Définit le contrat que chaque module doit respecter
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class ModuleStatus(Enum):
    """États possibles d'un module"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ModuleInfo:
    """Informations sur un module"""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str]
    priority: int  # 1 = haute priorité, 10 = basse priorité


class IModule(ABC):
    """
    Interface abstraite que tous les modules doivent implémenter.
    Cette interface garantit la cohérence et la compatibilité entre modules.
    """
    
    def __init__(self, name: str, engine_core=None):
        """
        Constructeur de base pour tous les modules
        
        Args:
            name: Nom unique du module
            engine_core: Référence vers le moteur central
        """
        self.name = name
        self.status = ModuleStatus.INACTIVE
        self.engine = engine_core
        self._error_count = 0
        self._last_error = None
        self._config = {}
        
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialise le module avec sa configuration
        
        Args:
            config: Configuration spécifique au module
            
        Returns:
            bool: True si l'initialisation réussit
        """
        pass
    
    @abstractmethod
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """
        Met à jour le module avec l'état actuel du jeu
        Appelé à chaque cycle principal (30 FPS)
        
        Args:
            game_state: État complet du jeu
            
        Returns:
            Dict optionnel avec les données à partager avec autres modules
        """
        pass
    
    @abstractmethod
    def handle_event(self, event: Any) -> bool:
        """
        Traite un événement reçu du système
        
        Args:
            event: Événement à traiter
            
        Returns:
            bool: True si l'événement a été traité
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel du module
        
        Returns:
            Dict contenant l'état interne du module
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Nettoie les ressources utilisées par le module
        Appelé avant l'arrêt du bot
        """
        pass
    
    def get_info(self) -> ModuleInfo:
        """
        Retourne les informations sur le module
        
        Returns:
            ModuleInfo: Informations détaillées du module
        """
        return ModuleInfo(
            name=self.name,
            version="1.0.0",
            author="TacticalBot",
            description="Module de base",
            dependencies=[],
            priority=5
        )
    
    def pause(self) -> None:
        """Met le module en pause"""
        if self.status == ModuleStatus.ACTIVE:
            self.status = ModuleStatus.PAUSED
    
    def resume(self) -> None:
        """Reprend l'exécution du module"""
        if self.status == ModuleStatus.PAUSED:
            self.status = ModuleStatus.ACTIVE
    
    def set_error(self, error: str) -> None:
        """
        Marque le module en erreur
        
        Args:
            error: Message d'erreur
        """
        self.status = ModuleStatus.ERROR
        self._error_count += 1
        self._last_error = error
    
    def get_last_error(self) -> Optional[str]:
        """
        Retourne la dernière erreur survenue
        
        Returns:
            str: Message de la dernière erreur ou None
        """
        return self._last_error
    
    def reset_errors(self) -> None:
        """Remet à zéro le compteur d'erreurs"""
        self._error_count = 0
        self._last_error = None
        if self.status == ModuleStatus.ERROR:
            self.status = ModuleStatus.INACTIVE
    
    def is_active(self) -> bool:
        """Vérifie si le module est actif"""
        return self.status == ModuleStatus.ACTIVE
    
    def is_critical(self) -> bool:
        """
        Indique si ce module est critique pour le fonctionnement
        Les modules critiques ne peuvent pas être désactivés
        
        Returns:
            bool: True si le module est critique
        """
        return False
    
    def can_execute_action(self, action_type: str) -> bool:
        """
        Vérifie si le module peut exécuter un type d'action donné
        
        Args:
            action_type: Type d'action à vérifier
            
        Returns:
            bool: True si l'action peut être exécutée
        """
        return self.is_active()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de performance du module
        
        Returns:
            Dict contenant les métriques
        """
        return {
            "status": self.status.value,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "name": self.name
        }


class IGameModule(IModule):
    """
    Interface étendue pour les modules qui interagissent directement avec le jeu
    """
    
    @abstractmethod
    def execute_action(self, action: Any) -> bool:
        """
        Exécute une action dans le jeu
        
        Args:
            action: Action à exécuter
            
        Returns:
            bool: True si l'action a été exécutée avec succès
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, game_state: Any) -> List[Any]:
        """
        Retourne la liste des actions disponibles selon l'état du jeu
        
        Args:
            game_state: État actuel du jeu
            
        Returns:
            List des actions possibles
        """
        pass


class IAnalysisModule(IModule):
    """
    Interface pour les modules d'analyse (pas d'actions directes)
    """
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyse des données et retourne des informations
        
        Args:
            data: Données à analyser
            
        Returns:
            Dict avec les résultats de l'analyse
        """
        pass