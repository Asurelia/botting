"""
Short Term Memory - Mémoire court-terme pour la session actuelle
Stocke les événements récents, patterns, et décisions
"""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryEvent:
    """Événement mémorisé"""
    timestamp: float
    event_type: str  # "decision", "combat", "death", "level_up", etc.
    data: Dict[str, Any]
    importance: int = 1  # 1-10, 10 = très important


class ShortTermMemory:
    """
    Mémoire court-terme - Session actuelle seulement
    
    Stocke:
    - Actions récentes
    - Combats
    - Erreurs
    - Patterns détectés
    """
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        
        # Files d'événements (FIFO limité)
        self.events: deque = deque(maxlen=max_events)
        self.decisions: deque = deque(maxlen=200)
        self.combats: deque = deque(maxlen=50)
        self.errors: deque = deque(maxlen=100)
        
        # Statistiques session
        self.session_stats = {
            'start_time': time.time(),
            'total_decisions': 0,
            'total_combats': 0,
            'total_kills': 0,
            'total_deaths': 0,
            'total_xp_gained': 0,
            'maps_visited': set()
        }
        
        # Patterns récents
        self.recent_patterns = {}
        
        logger.info("ShortTermMemory initialisée")
    
    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        importance: int = 1
    ):
        """Ajoute un événement en mémoire"""
        event = MemoryEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            importance=importance
        )
        
        self.events.append(event)
        
        # Router vers files spécifiques
        if event_type == "decision":
            self.decisions.append(event)
            self.session_stats['total_decisions'] += 1
        
        elif event_type in ["combat_start", "combat_end"]:
            self.combats.append(event)
            self.session_stats['total_combats'] += 1
        
        elif event_type == "kill":
            self.session_stats['total_kills'] += 1
        
        elif event_type == "death":
            self.session_stats['total_deaths'] += 1
        
        elif event_type == "error":
            self.errors.append(event)
    
    def add_decision(
        self,
        action_type: str,
        details: Dict[str, Any],
        reason: str,
        success: bool
    ):
        """Ajoute une décision"""
        self.add_event("decision", {
            'action_type': action_type,
            'details': details,
            'reason': reason,
            'success': success
        }, importance=2)
    
    def add_combat(
        self,
        enemies: int,
        allies: int,
        result: str,  # "won", "lost", "fled"
        duration: float
    ):
        """Ajoute un combat"""
        self.add_event("combat_end", {
            'enemies': enemies,
            'allies': allies,
            'result': result,
            'duration': duration
        }, importance=5 if result == "lost" else 3)
    
    def add_error(self, error_type: str, message: str):
        """Ajoute une erreur"""
        self.add_event("error", {
            'error_type': error_type,
            'message': message
        }, importance=4)
    
    def get_recent_events(self, count: int = 10) -> List[MemoryEvent]:
        """Retourne les N événements les plus récents"""
        return list(self.events)[-count:]
    
    def get_recent_decisions(self, count: int = 10) -> List[MemoryEvent]:
        """Retourne les N décisions les plus récentes"""
        return list(self.decisions)[-count:]
    
    def get_recent_combats(self, count: int = 5) -> List[MemoryEvent]:
        """Retourne les N combats les plus récents"""
        return list(self.combats)[-count:]
    
    def get_pattern_frequency(self, pattern_name: str, time_window: float = 300) -> int:
        """
        Compte combien de fois un pattern s'est produit récemment
        
        Args:
            pattern_name: Nom du pattern
            time_window: Fenêtre de temps en secondes
        
        Returns:
            Nombre d'occurrences
        """
        cutoff_time = time.time() - time_window
        count = 0
        
        for event in self.events:
            if event.timestamp < cutoff_time:
                continue
            
            if event.data.get('pattern') == pattern_name:
                count += 1
        
        return count
    
    def get_decision_success_rate(self, action_type: Optional[str] = None) -> float:
        """
        Calcule le taux de succès des décisions
        
        Args:
            action_type: Type d'action spécifique ou None pour toutes
        
        Returns:
            Taux de succès (0.0 à 1.0)
        """
        relevant_decisions = [
            d for d in self.decisions
            if action_type is None or d.data.get('action_type') == action_type
        ]
        
        if not relevant_decisions:
            return 0.0
        
        successes = sum(1 for d in relevant_decisions if d.data.get('success', False))
        return successes / len(relevant_decisions)
    
    def get_combat_win_rate(self) -> float:
        """Calcule le taux de victoire en combat"""
        if not self.combats:
            return 0.0
        
        wins = sum(1 for c in self.combats if c.data.get('result') == 'won')
        return wins / len(self.combats)
    
    def should_avoid_pattern(self, pattern_name: str, threshold: int = 3) -> bool:
        """
        Détermine si un pattern devrait être évité (trop d'échecs récents)
        
        Args:
            pattern_name: Nom du pattern
            threshold: Seuil d'échecs
        
        Returns:
            True si devrait éviter
        """
        recent_failures = 0
        
        for event in list(self.events)[-50:]:  # 50 derniers événements
            if (event.event_type == "decision" and
                event.data.get('pattern') == pattern_name and
                not event.data.get('success', True)):
                recent_failures += 1
        
        return recent_failures >= threshold
    
    def get_session_duration(self) -> float:
        """Durée de la session en secondes"""
        return time.time() - self.session_stats['start_time']
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Retourne statistiques de session"""
        stats = self.session_stats.copy()
        # Convertir set en list pour JSON
        if 'maps_visited' in stats and isinstance(stats['maps_visited'], set):
            stats['maps_visited'] = list(stats['maps_visited'])
        stats['duration_seconds'] = self.get_session_duration()
        stats['events_count'] = len(self.events)
        stats['combat_win_rate'] = self.get_combat_win_rate()
        stats['decision_success_rate'] = self.get_decision_success_rate()
        return stats
    
    def clear(self):
        """Efface toute la mémoire"""
        self.events.clear()
        self.decisions.clear()
        self.combats.clear()
        self.errors.clear()
        self.recent_patterns.clear()
        logger.info("Mémoire effacée")


def create_short_term_memory(max_events: int = 1000) -> ShortTermMemory:
    """Factory function"""
    return ShortTermMemory(max_events=max_events)
