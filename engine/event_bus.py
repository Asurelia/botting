"""
Système de bus d'événements pour la communication inter-modules
Permet une architecture découplée où les modules communiquent via des événements
"""

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
import queue
import logging
from collections import defaultdict


class EventPriority(Enum):
    """Priorités des événements"""
    CRITICAL = 0    # Événements critiques (mort du personnage, déconnexion)
    HIGH = 1        # Événements importants (combat démarré, objectif atteint)
    NORMAL = 2      # Événements normaux (changement de map, ressource détectée)
    LOW = 3         # Événements informatifs (message reçu, statistiques)


class EventType(Enum):
    """Types d'événements du système"""
    # Combat
    COMBAT_STARTED = "combat_started"
    COMBAT_ENDED = "combat_ended"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    SPELL_CAST = "spell_cast"
    DAMAGE_RECEIVED = "damage_received"
    DEATH_OCCURRED = "death_occurred"
    
    # Navigation et Monde
    MAP_CHANGED = "map_changed"
    POSITION_CHANGED = "position_changed"
    STUCK_DETECTED = "stuck_detected"
    
    # Ressources et Métiers
    RESOURCE_DETECTED = "resource_detected"
    RESOURCE_HARVESTED = "resource_harvested"
    INVENTORY_FULL = "inventory_full"
    LEVEL_UP = "level_up"
    
    # Économie
    ITEM_SOLD = "item_sold"
    ITEM_BOUGHT = "item_bought"
    MARKET_PRICE_UPDATED = "market_price_updated"
    
    # Social
    MESSAGE_RECEIVED = "message_received"
    PLAYER_DETECTED = "player_detected"
    GUILD_INVITE = "guild_invite"
    
    # Système
    MODULE_ERROR = "module_error"
    CONFIG_CHANGED = "config_changed"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    PERFORMANCE_WARNING = "performance_warning"


@dataclass
class Event:
    """
    Représente un événement dans le système
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    source_module: Optional[str] = None
    target_modules: Optional[Set[str]] = None
    processed_by: Set[str] = None
    
    def __post_init__(self):
        """Initialisation après création"""
        if self.processed_by is None:
            self.processed_by = set()
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventHandler:
    """
    Gestionnaire d'événement pour un module
    """
    def __init__(self, callback: Callable[[Event], bool], 
                 event_types: Set[EventType], 
                 module_name: str):
        self.callback = callback
        self.event_types = event_types
        self.module_name = module_name
        self.is_active = True
        self.call_count = 0
        self.last_called = None


class EventBus:
    """
    Bus d'événements central pour la communication inter-modules
    Implémente le pattern Publisher-Subscriber avec gestion des priorités
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialise le bus d'événements
        
        Args:
            max_queue_size: Taille maximum de la queue d'événements
        """
        self._handlers = defaultdict(list)  # event_type -> list of handlers
        self._event_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread = None
        self._lock = threading.RLock()
        self._event_history = []
        self._max_history = 1000
        self._statistics = defaultdict(int)
        
        # Configuration du logging
        self.logger = logging.getLogger(f"{__name__}.EventBus")
        
    def start(self) -> None:
        """Démarre le traitement des événements en arrière-plan"""
        with self._lock:
            if not self._running:
                self._running = True
                self._worker_thread = threading.Thread(
                    target=self._process_events,
                    name="EventBus-Worker",
                    daemon=True
                )
                self._worker_thread.start()
                self.logger.info("Event bus démarré")
    
    def stop(self) -> None:
        """Arrête le traitement des événements"""
        with self._lock:
            self._running = False
            
        # Signaler l'arrêt avec un événement spécial
        if self._worker_thread and self._worker_thread.is_alive():
            try:
                self._event_queue.put((0, datetime.now(), None), timeout=1.0)
                self._worker_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'arrêt du bus: {e}")
        
        self.logger.info("Event bus arrêté")
    
    def subscribe(self, module_name: str, event_types: Set[EventType], 
                  callback: Callable[[Event], bool]) -> bool:
        """
        Abonne un module à des types d'événements
        
        Args:
            module_name: Nom du module qui s'abonne
            event_types: Types d'événements à écouter
            callback: Fonction à appeler lors de la réception d'un événement
            
        Returns:
            bool: True si l'abonnement a réussi
        """
        try:
            handler = EventHandler(callback, event_types, module_name)
            
            with self._lock:
                for event_type in event_types:
                    self._handlers[event_type].append(handler)
            
            self.logger.debug(f"Module {module_name} abonné aux événements {event_types}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de l'abonnement de {module_name}: {e}")
            return False
    
    def unsubscribe(self, module_name: str, event_type: EventType = None) -> None:
        """
        Désabonne un module d'un type d'événement ou de tous
        
        Args:
            module_name: Nom du module à désabonner
            event_type: Type d'événement spécifique (None = tous)
        """
        with self._lock:
            if event_type:
                # Désabonner d'un type spécifique
                handlers = self._handlers[event_type]
                self._handlers[event_type] = [h for h in handlers 
                                            if h.module_name != module_name]
            else:
                # Désabonner de tous les types
                for event_type in list(self._handlers.keys()):
                    handlers = self._handlers[event_type]
                    self._handlers[event_type] = [h for h in handlers 
                                                if h.module_name != module_name]
        
        self.logger.debug(f"Module {module_name} désabonné")
    
    def publish(self, event: Event) -> bool:
        """
        Publie un événement sur le bus
        
        Args:
            event: Événement à publier
            
        Returns:
            bool: True si l'événement a été ajouté à la queue
        """
        try:
            if not self._running:
                return False
            
            # Ajout à la queue avec priorité
            priority_value = event.priority.value
            self._event_queue.put((priority_value, event.timestamp, event), block=False)
            
            self._statistics["events_published"] += 1
            self._statistics[f"events_{event.type.value}"] += 1
            
            return True
        except queue.Full:
            self.logger.warning("Queue d'événements pleine, événement ignoré")
            self._statistics["events_dropped"] += 1
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors de la publication: {e}")
            return False
    
    def publish_immediate(self, event_type: EventType, 
                         data: Dict[str, Any], 
                         source_module: str = None,
                         priority: EventPriority = EventPriority.NORMAL) -> bool:
        """
        Publie immédiatement un événement (méthode de convenance)
        
        Args:
            event_type: Type d'événement
            data: Données de l'événement
            source_module: Module source
            priority: Priorité de l'événement
            
        Returns:
            bool: True si publié avec succès
        """
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            priority=priority,
            source_module=source_module
        )
        return self.publish(event)
    
    def _process_events(self) -> None:
        """
        Thread worker qui traite les événements de la queue
        """
        self.logger.info("Démarrage du traitement des événements")
        
        while self._running:
            try:
                # Récupération de l'événement avec timeout
                try:
                    priority, timestamp, event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Signal d'arrêt
                if event is None:
                    break
                
                # Traitement de l'événement
                self._dispatch_event(event)
                
                # Historique
                self._add_to_history(event)
                
                self._event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement d'événement: {e}")
                continue
        
        self.logger.info("Arrêt du traitement des événements")
    
    def _dispatch_event(self, event: Event) -> None:
        """
        Distribue un événement à tous les handlers concernés
        
        Args:
            event: Événement à distribuer
        """
        handlers = self._handlers.get(event.type, [])
        
        for handler in handlers[:]:  # Copie pour éviter les modifications concurrentes
            if not handler.is_active:
                continue
            
            # Vérification des modules cibles
            if event.target_modules and handler.module_name not in event.target_modules:
                continue
            
            # Éviter le traitement multiple
            if handler.module_name in event.processed_by:
                continue
            
            try:
                # Appel du callback
                success = handler.callback(event)
                
                # Mise à jour des statistiques du handler
                handler.call_count += 1
                handler.last_called = datetime.now()
                
                if success:
                    event.processed_by.add(handler.module_name)
                    self._statistics[f"handler_success_{handler.module_name}"] += 1
                else:
                    self._statistics[f"handler_failed_{handler.module_name}"] += 1
                
            except Exception as e:
                self.logger.error(f"Erreur dans le handler {handler.module_name}: {e}")
                self._statistics[f"handler_error_{handler.module_name}"] += 1
    
    def _add_to_history(self, event: Event) -> None:
        """
        Ajoute un événement à l'historique
        
        Args:
            event: Événement à ajouter
        """
        self._event_history.append(event)
        
        # Limitation de la taille de l'historique
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du bus d'événements
        
        Returns:
            Dict contenant les statistiques
        """
        with self._lock:
            stats = dict(self._statistics)
            stats.update({
                "queue_size": self._event_queue.qsize(),
                "handlers_count": sum(len(handlers) for handlers in self._handlers.values()),
                "history_size": len(self._event_history),
                "is_running": self._running
            })
            return stats
    
    def get_recent_events(self, count: int = 10, 
                         event_type: EventType = None) -> List[Event]:
        """
        Retourne les événements récents
        
        Args:
            count: Nombre d'événements à retourner
            event_type: Filtre par type d'événement
            
        Returns:
            List des événements récents
        """
        events = self._event_history[-count:]
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def clear_history(self) -> None:
        """Vide l'historique des événements"""
        self._event_history.clear()
        self.logger.info("Historique des événements vidé")
    
    def get_handler_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retourne des informations sur tous les handlers
        
        Returns:
            Dict organisé par type d'événement
        """
        info = {}
        for event_type, handlers in self._handlers.items():
            info[event_type.value] = [
                {
                    "module_name": h.module_name,
                    "is_active": h.is_active,
                    "call_count": h.call_count,
                    "last_called": h.last_called
                }
                for h in handlers
            ]
        return info