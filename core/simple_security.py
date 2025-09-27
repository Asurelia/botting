"""
Système de sécurité simplifié pour usage personnel
Version streamlined sans complexité inutile
"""

import random
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
from pathlib import Path
import numpy as np


class SimpleHumanBehavior:
    """
    Simulation comportement humain basique mais efficace
    Juste ce qu'il faut pour éviter la détection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Paramètres simples ajustables
        self.click_delay_base = 150  # 150ms de base
        self.click_delay_variation = 50  # ±50ms de variation
        self.mouse_variation = 3  # ±3 pixels de variation
        
        # État de fatigue simple
        self.start_time = time.time()
        self.actions_count = 0
        
    def get_human_click_delay(self) -> float:
        """
        Délai de clic avec variation naturelle
        Entre 100ms et 250ms, centré sur 150ms
        """
        delay = random.gauss(self.click_delay_base, self.click_delay_variation)
        
        # Augmentation légère avec la fatigue (plus on joue, plus on ralentit)
        fatigue_factor = 1 + (self.actions_count / 10000) * 0.1  # +10% après 10k actions
        delay *= fatigue_factor
        
        # Contraintes réalistes
        return max(80, min(400, delay)) / 1000  # Convertir en secondes
    
    def get_human_mouse_offset(self, target_x: int, target_y: int) -> Tuple[int, int]:
        """
        Ajoute une petite variation naturelle à la position de clic
        Évite de cliquer toujours exactement au même pixel
        """
        offset_x = random.randint(-self.mouse_variation, self.mouse_variation)
        offset_y = random.randint(-self.mouse_variation, self.mouse_variation)
        
        return target_x + offset_x, target_y + offset_y
    
    def should_take_break(self) -> bool:
        """
        Détermine si il faut prendre une pause
        Simple : pause de 5-15s toutes les 500-800 actions
        """
        if self.actions_count > 0 and self.actions_count % random.randint(500, 800) == 0:
            return True
        return False
    
    def get_break_duration(self) -> float:
        """
        Durée de pause naturelle (5 à 15 secondes)
        """
        return random.uniform(5, 15)
    
    def simulate_human_action(self, x: int, y: int, action_type: str = "click"):
        """
        Simule une action humaine complète
        
        Args:
            x, y: Position du clic
            action_type: Type d'action pour les logs
        
        Returns:
            Tuple[int, int]: Position finale avec variation humaine
        """
        # Délai avant action
        delay = self.get_human_click_delay()
        time.sleep(delay)
        
        # Position avec variation
        final_x, final_y = self.get_human_mouse_offset(x, y)
        
        # Comptage des actions
        self.actions_count += 1
        
        # Pause si nécessaire
        if self.should_take_break():
            break_time = self.get_break_duration()
            self.logger.info(f"Pause humaine de {break_time:.1f}s après {self.actions_count} actions")
            time.sleep(break_time)
        
        return final_x, final_y


class SimpleSessionManager:
    """
    Gestionnaire de session basique
    Évite les sessions trop longues qui paraissent suspectes
    """
    
    def __init__(self, max_session_hours: float = 3.0):
        """
        Args:
            max_session_hours: Durée max d'une session (défaut 3h)
        """
        self.max_session_duration = max_session_hours * 3600  # En secondes
        self.session_start = time.time()
        self.logger = logging.getLogger(__name__)
        
    def is_session_too_long(self) -> bool:
        """Vérifie si la session dure depuis trop longtemps"""
        current_duration = time.time() - self.session_start
        return current_duration >= self.max_session_duration
    
    def get_session_info(self) -> dict:
        """Infos sur la session actuelle"""
        duration = time.time() - self.session_start
        remaining = max(0, self.max_session_duration - duration)
        
        return {
            "duration_minutes": duration / 60,
            "remaining_minutes": remaining / 60,
            "is_too_long": self.is_session_too_long()
        }
    
    def suggest_break(self) -> Optional[str]:
        """Suggère une pause si nécessaire"""
        info = self.get_session_info()
        
        if info["remaining_minutes"] < 30:
            return f"⚠️ Session longue ! Plus que {info['remaining_minutes']:.0f} min recommandées"
        elif info["is_too_long"]:
            return "🛑 Session trop longue ! Pause fortement recommandée"
        
        return None


class SimpleLogger:
    """
    Système de logs simplifié mais efficace
    Juste ce qu'il faut pour débugger et suivre l'activité
    """
    
    def __init__(self, log_file: str = "logs/bot_simple.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Configuration logging Python standard
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # Affichage console aussi
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistiques simples en mémoire
        self.stats = {
            "actions_total": 0,
            "session_start": datetime.now(),
            "errors": 0,
            "last_action_time": None
        }
    
    def log_action(self, action: str, details: dict = None):
        """
        Log une action simple
        
        Args:
            action: Type d'action (ex: "click", "farm_wheat", "combat_won")
            details: Détails optionnels
        """
        self.stats["actions_total"] += 1
        self.stats["last_action_time"] = datetime.now()
        
        details_str = f" - {details}" if details else ""
        self.logger.info(f"Action: {action}{details_str}")
    
    def log_error(self, error: str, exception: Exception = None):
        """Log une erreur"""
        self.stats["errors"] += 1
        self.logger.error(f"Erreur: {error}")
        if exception:
            self.logger.exception("Détails de l'exception:")
    
    def get_session_stats(self) -> dict:
        """Statistiques de la session actuelle"""
        duration = datetime.now() - self.stats["session_start"]
        
        return {
            "session_duration_minutes": duration.total_seconds() / 60,
            "total_actions": self.stats["actions_total"],
            "actions_per_minute": self.stats["actions_total"] / max(1, duration.total_seconds() / 60),
            "errors": self.stats["errors"],
            "last_action": self.stats["last_action_time"]
        }
    
    def print_stats(self):
        """Affiche les statistiques en console"""
        stats = self.get_session_stats()
        print("\n" + "="*50)
        print("[STATS] STATISTIQUES SESSION")
        print("="*50)
        print(f"[TIME] Durée: {stats['session_duration_minutes']:.1f} minutes")
        print(f"[ACTIONS] Total: {stats['total_actions']}")
        print(f"[RATE] Actions/min: {stats['actions_per_minute']:.1f}")
        print(f"[ERRORS] Total: {stats['errors']}")
        print("="*50)


class SimpleSecurity:
    """
    Système de sécurité tout-en-un simplifié
    Combine comportement humain + gestion session + logs
    """
    
    def __init__(self, max_session_hours: float = 3.0):
        self.behavior = SimpleHumanBehavior()
        self.session = SimpleSessionManager(max_session_hours)
        self.logger = SimpleLogger()
        
        self.logger.log_action("security_system_started", {
            "max_session_hours": max_session_hours,
            "version": "simple"
        })
    
    def safe_click(self, x: int, y: int, action_name: str = "click") -> Tuple[int, int]:
        """
        Clic sécurisé avec simulation humaine
        
        Args:
            x, y: Position du clic
            action_name: Nom de l'action pour les logs
            
        Returns:
            Position finale du clic (avec variation humaine)
        """
        # Vérification session
        warning = self.session.suggest_break()
        if warning:
            self.logger.logger.warning(warning)
        
        # Action humaine simulée
        final_x, final_y = self.behavior.simulate_human_action(x, y, action_name)
        
        # Log de l'action
        self.logger.log_action(action_name, {
            "target": (x, y),
            "actual": (final_x, final_y),
            "variation": (final_x - x, final_y - y)
        })
        
        return final_x, final_y
    
    def safe_wait(self, base_seconds: float, variation: float = 0.5) -> None:
        """
        Attente avec variation humaine
        
        Args:
            base_seconds: Temps de base
            variation: Variation max en secondes
        """
        actual_wait = base_seconds + random.uniform(-variation, variation)
        actual_wait = max(0.1, actual_wait)  # Minimum 0.1s
        
        time.sleep(actual_wait)
        self.logger.log_action("wait", {
            "requested": base_seconds,
            "actual": actual_wait
        })
    
    def should_stop(self) -> bool:
        """Vérifie si on devrait arrêter le bot"""
        return self.session.is_session_too_long()
    
    def get_status(self) -> dict:
        """État complet du système de sécurité"""
        return {
            "session": self.session.get_session_info(),
            "stats": self.logger.get_session_stats(),
            "actions_count": self.behavior.actions_count,
            "should_stop": self.should_stop()
        }
    
    def print_status(self):
        """Affiche l'état du système"""
        self.logger.print_stats()
        
        session_info = self.session.get_session_info()
        print(f"[TIME] Temps restant recommandé: {session_info['remaining_minutes']:.0f} min")
        
        if self.should_stop():
            print("[STOP] ARRÊT RECOMMANDÉ - Session trop longue")


# Fonctions utilitaires pour utilisation simple
def create_simple_security(max_hours: float = 3.0) -> SimpleSecurity:
    """
    Crée un système de sécurité simple prêt à l'emploi
    
    Args:
        max_hours: Durée max de session recommandée
        
    Returns:
        Système de sécurité configuré
    """
    return SimpleSecurity(max_hours)


def safe_bot_click(security: SimpleSecurity, x: int, y: int, action: str = "click"):
    """
    Fonction utilitaire pour clic sécurisé
    """
    return security.safe_click(x, y, action)


if __name__ == "__main__":
    # Test du système simplifié
    print("[TEST] Test du système de sécurité simplifié...")
    
    security = create_simple_security(max_hours=0.1)  # 6 minutes pour test
    
    # Simulation de quelques actions
    for i in range(10):
        # Clic simulé
        final_pos = security.safe_click(100 + i, 200 + i, f"test_action_{i}")
        print(f"Clic {i+1}: position finale {final_pos}")
        
        # Attente simulée
        security.safe_wait(1.0, 0.3)
        
        # État toutes les 3 actions
        if i % 3 == 0:
            status = security.get_status()
            print(f"Actions: {status['actions_count']}, Session: {status['session']['duration_minutes']:.1f}min")
    
    # Statistiques finales
    security.print_status()
    
    print("\n[OK] Test terminé ! Le système fonctionne.")