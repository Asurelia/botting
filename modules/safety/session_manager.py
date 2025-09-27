"""
Gestionnaire de sessions avec limites de temps et pauses obligatoires
Simule des sessions de jeu réalistes avec pauses naturelles et limites saines
"""

import time
import random
import logging
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """États possibles d'une session"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    MANDATORY_BREAK = "mandatory_break"
    OPTIONAL_BREAK = "optional_break"
    SUSPENDED = "suspended"
    ENDED = "ended"


class BreakReason(Enum):
    """Raisons des pauses"""
    FATIGUE = "fatigue"
    TIME_LIMIT = "time_limit"
    RANDOM_BREAK = "random_break"
    MEAL_TIME = "meal_time"
    NATURAL_PAUSE = "natural_pause"
    SAFETY_LIMIT = "safety_limit"


@dataclass
class SessionLimits:
    """Limites de session configurables"""
    max_continuous_minutes: int = 120      # 2h max en continu
    max_daily_minutes: int = 480           # 8h max par jour
    mandatory_break_minutes: int = 15      # 15min de pause obligatoire
    short_break_frequency: int = 30        # Pause courte toutes les 30min
    short_break_duration: int = 5          # 5min de pause courte
    meal_break_times: List[str] = None     # Heures des repas ["12:00", "19:00"]
    night_mode_start: str = "23:00"        # Début mode nuit
    night_mode_end: str = "07:00"          # Fin mode nuit
    weekend_factor: float = 1.2            # Facteur weekend (+20% temps)
    random_break_chance: float = 0.1       # 10% chance pause aléatoire/heure


@dataclass
class SessionStats:
    """Statistiques de session"""
    start_time: float
    end_time: Optional[float] = None
    total_active_time: float = 0.0
    total_break_time: float = 0.0
    breaks_taken: int = 0
    actions_performed: int = 0
    experience_gained: int = 0
    levels_gained: int = 0
    deaths_count: int = 0
    errors_made: int = 0


class SessionManager:
    """
    Gestionnaire de sessions avec comportement humain réaliste
    Gère les limites de temps, pauses obligatoires et patterns naturels
    """
    
    def __init__(self, limits: Optional[SessionLimits] = None, save_file: Optional[str] = None):
        self.limits = limits or SessionLimits()
        if self.limits.meal_break_times is None:
            self.limits.meal_break_times = ["12:00", "19:00"]
        
        self.save_file = save_file
        self.state = SessionState.INACTIVE
        self.current_stats = None
        self.daily_stats = {}
        self.break_callbacks = []
        self.resume_callbacks = []
        
        # État interne
        self.session_start_time = None
        self.last_action_time = None
        self.last_break_time = None
        self.break_reason = None
        self.mandatory_break_end_time = None
        
        # Surveillance continue
        self.monitor_thread = None
        self.monitor_active = False
        
        # Chargement des données sauvegardées
        self._load_session_data()
    
    def start_session(self) -> bool:
        """
        Démarre une nouvelle session avec vérifications de sécurité
        """
        current_time = time.time()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Vérification des limites quotidiennes
        daily_time = self.get_daily_playtime(current_date)
        if daily_time >= self.limits.max_daily_minutes * 60:
            logger.warning(f"Limite quotidienne atteinte ({daily_time/3600:.1f}h)")
            return False
        
        # Vérification mode nuit
        if self._is_night_time():
            logger.warning("Mode nuit actif - session non recommandée")
            if not self._confirm_night_session():
                return False
        
        # Initialisation de la session
        self.session_start_time = current_time
        self.last_action_time = current_time
        self.last_break_time = current_time
        self.state = SessionState.ACTIVE
        
        self.current_stats = SessionStats(start_time=current_time)
        
        # Démarrage du monitoring
        self._start_monitoring()
        
        logger.info(f"Session démarrée à {datetime.now().strftime('%H:%M:%S')}")
        
        # Planification de la première pause courte
        self._schedule_next_break()
        
        return True
    
    def end_session(self):
        """
        Termine la session actuelle et sauvegarde les statistiques
        """
        if self.state == SessionState.INACTIVE:
            return
        
        current_time = time.time()
        
        # Arrêt du monitoring
        self._stop_monitoring()
        
        # Finalisation des statistiques
        if self.current_stats:
            self.current_stats.end_time = current_time
            session_duration = current_time - self.current_stats.start_time
            self.current_stats.total_active_time = session_duration - self.current_stats.total_break_time
        
        # Sauvegarde des statistiques quotidiennes
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = []
        self.daily_stats[current_date].append(asdict(self.current_stats))
        
        # Sauvegarde sur disque
        self._save_session_data()
        
        self.state = SessionState.ENDED
        
        duration_hours = (current_time - self.session_start_time) / 3600
        logger.info(f"Session terminée après {duration_hours:.1f}h")
        
        # Rapport final
        self._print_session_report()
    
    def record_action(self, action_type: str = "generic"):
        """
        Enregistre une action utilisateur pour le suivi d'activité
        """
        if self.state != SessionState.ACTIVE:
            return
        
        current_time = time.time()
        self.last_action_time = current_time
        
        if self.current_stats:
            self.current_stats.actions_performed += 1
        
        # Vérification des limites en temps réel
        self._check_session_limits()
    
    def take_break(self, reason: BreakReason, duration_minutes: Optional[int] = None) -> bool:
        """
        Déclenche une pause avec durée spécifiée
        """
        if self.state != SessionState.ACTIVE:
            return False
        
        # Détermination de la durée de pause
        if duration_minutes is None:
            duration_minutes = self._get_break_duration(reason)
        
        logger.info(f"Pause déclenchée: {reason.value} ({duration_minutes}min)")
        
        # Changement d'état
        if reason in [BreakReason.TIME_LIMIT, BreakReason.SAFETY_LIMIT]:
            self.state = SessionState.MANDATORY_BREAK
            self.mandatory_break_end_time = time.time() + (duration_minutes * 60)
        else:
            self.state = SessionState.OPTIONAL_BREAK
        
        self.break_reason = reason
        self.last_break_time = time.time()
        
        # Mise à jour des statistiques
        if self.current_stats:
            self.current_stats.breaks_taken += 1
        
        # Exécution des callbacks de pause
        for callback in self.break_callbacks:
            try:
                callback(reason, duration_minutes)
            except Exception as e:
                logger.error(f"Erreur callback pause: {e}")
        
        # Pause automatique si obligatoire
        if self.state == SessionState.MANDATORY_BREAK:
            self._execute_mandatory_break(duration_minutes)
        
        return True
    
    def resume_session(self) -> bool:
        """
        Reprend la session après une pause
        """
        if self.state not in [SessionState.OPTIONAL_BREAK, SessionState.MANDATORY_BREAK]:
            return False
        
        # Vérification pause obligatoire
        if self.state == SessionState.MANDATORY_BREAK:
            if time.time() < self.mandatory_break_end_time:
                remaining = (self.mandatory_break_end_time - time.time()) / 60
                logger.warning(f"Pause obligatoire non terminée ({remaining:.1f}min restantes)")
                return False
        
        # Calcul du temps de pause
        if self.last_break_time:
            break_duration = time.time() - self.last_break_time
            if self.current_stats:
                self.current_stats.total_break_time += break_duration
        
        # Reprise de la session
        self.state = SessionState.ACTIVE
        self.last_action_time = time.time()
        
        logger.info("Session reprise après pause")
        
        # Exécution des callbacks de reprise
        for callback in self.resume_callbacks:
            try:
                callback(self.break_reason)
            except Exception as e:
                logger.error(f"Erreur callback reprise: {e}")
        
        # Replanification des pauses
        self._schedule_next_break()
        
        return True
    
    def get_session_info(self) -> Dict:
        """
        Retourne les informations de la session actuelle
        """
        if not self.current_stats:
            return {"status": "no_active_session"}
        
        current_time = time.time()
        session_duration = current_time - self.current_stats.start_time
        active_duration = session_duration - self.current_stats.total_break_time
        
        # Calcul du temps restant avant limite
        remaining_continuous = max(0, (self.limits.max_continuous_minutes * 60) - active_duration)
        
        # Temps quotidien utilisé
        current_date = datetime.now().strftime("%Y-%m-%d")
        daily_used = self.get_daily_playtime(current_date)
        remaining_daily = max(0, (self.limits.max_daily_minutes * 60) - daily_used)
        
        return {
            "status": self.state.value,
            "session_duration": session_duration,
            "active_duration": active_duration,
            "break_duration": self.current_stats.total_break_time,
            "actions_performed": self.current_stats.actions_performed,
            "breaks_taken": self.current_stats.breaks_taken,
            "remaining_continuous": remaining_continuous,
            "remaining_daily": remaining_daily,
            "break_reason": self.break_reason.value if self.break_reason else None,
            "mandatory_break_remaining": max(0, self.mandatory_break_end_time - current_time) if self.mandatory_break_end_time else 0
        }
    
    def get_daily_playtime(self, date: str) -> float:
        """
        Retourne le temps de jeu total pour une date donnée
        """
        if date not in self.daily_stats:
            return 0.0
        
        total_time = 0.0
        for session in self.daily_stats[date]:
            if session.get('end_time'):
                total_time += session['total_active_time']
        
        # Ajout de la session actuelle si applicable
        if self.current_stats and date == datetime.now().strftime("%Y-%m-%d"):
            current_active = time.time() - self.current_stats.start_time - self.current_stats.total_break_time
            total_time += current_active
        
        return total_time
    
    def add_break_callback(self, callback: Callable):
        """Ajoute un callback appelé lors des pauses"""
        self.break_callbacks.append(callback)
    
    def add_resume_callback(self, callback: Callable):
        """Ajoute un callback appelé lors de la reprise"""
        self.resume_callbacks.append(callback)
    
    def _start_monitoring(self):
        """Démarre le thread de surveillance de session"""
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _stop_monitoring(self):
        """Arrête le thread de surveillance"""
        self.monitor_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Boucle de surveillance continue"""
        while self.monitor_active and self.state == SessionState.ACTIVE:
            try:
                self._check_session_limits()
                self._check_scheduled_breaks()
                time.sleep(60)  # Vérification chaque minute
            except Exception as e:
                logger.error(f"Erreur monitoring session: {e}")
                time.sleep(60)
    
    def _check_session_limits(self):
        """Vérification des limites de session"""
        if self.state != SessionState.ACTIVE:
            return
        
        current_time = time.time()
        
        # Limite de session continue
        active_duration = current_time - self.session_start_time - (self.current_stats.total_break_time if self.current_stats else 0)
        max_continuous = self.limits.max_continuous_minutes * 60
        
        if active_duration >= max_continuous:
            logger.warning("Limite de session continue atteinte")
            self.take_break(BreakReason.TIME_LIMIT, self.limits.mandatory_break_minutes)
            return
        
        # Limite quotidienne
        current_date = datetime.now().strftime("%Y-%m-%d")
        daily_time = self.get_daily_playtime(current_date)
        max_daily = self.limits.max_daily_minutes * 60
        
        if daily_time >= max_daily * 0.9:  # Avertissement à 90%
            logger.warning(f"Limite quotidienne bientôt atteinte ({daily_time/3600:.1f}h/{max_daily/3600:.1f}h)")
        
        if daily_time >= max_daily:
            logger.warning("Limite quotidienne atteinte")
            self.take_break(BreakReason.SAFETY_LIMIT, 60)  # 1h de pause obligatoire
    
    def _check_scheduled_breaks(self):
        """Vérification des pauses programmées"""
        if self.state != SessionState.ACTIVE:
            return
        
        current_time = time.time()
        
        # Pause courte programmée
        if self.last_break_time:
            time_since_break = current_time - self.last_break_time
            if time_since_break >= self.limits.short_break_frequency * 60:
                self.take_break(BreakReason.NATURAL_PAUSE, self.limits.short_break_duration)
                return
        
        # Pauses repas
        current_hour_min = datetime.now().strftime("%H:%M")
        for meal_time in self.limits.meal_break_times:
            if self._is_time_close(current_hour_min, meal_time, 5):  # 5min de marge
                self.take_break(BreakReason.MEAL_TIME, 30)
                return
        
        # Pause aléatoire
        if random.random() < self.limits.random_break_chance / 60:  # Par minute
            self.take_break(BreakReason.RANDOM_BREAK, random.randint(2, 10))
    
    def _schedule_next_break(self):
        """Programme la prochaine pause"""
        # Implémentation de la planification des pauses
        pass
    
    def _get_break_duration(self, reason: BreakReason) -> int:
        """Calcule la durée de pause selon la raison"""
        durations = {
            BreakReason.FATIGUE: random.randint(10, 20),
            BreakReason.TIME_LIMIT: self.limits.mandatory_break_minutes,
            BreakReason.RANDOM_BREAK: random.randint(2, 8),
            BreakReason.MEAL_TIME: random.randint(20, 45),
            BreakReason.NATURAL_PAUSE: self.limits.short_break_duration,
            BreakReason.SAFETY_LIMIT: 60
        }
        return durations.get(reason, 5)
    
    def _execute_mandatory_break(self, duration_minutes: int):
        """Exécute une pause obligatoire"""
        logger.info(f"Début pause obligatoire: {duration_minutes} minutes")
        
        # Ici pourrait être ajoutée la logique pour suspendre le bot
        # Par exemple: self._pause_all_activities()
        
        # Surveillance de la pause obligatoire
        end_time = time.time() + (duration_minutes * 60)
        while time.time() < end_time and self.state == SessionState.MANDATORY_BREAK:
            remaining = (end_time - time.time()) / 60
            if remaining > 0:
                logger.debug(f"Pause obligatoire: {remaining:.1f}min restantes")
                time.sleep(min(60, remaining * 60))
            else:
                break
        
        logger.info("Pause obligatoire terminée")
    
    def _is_night_time(self) -> bool:
        """Vérifie si on est en période nocturne"""
        current_time = datetime.now().strftime("%H:%M")
        night_start = self.limits.night_mode_start
        night_end = self.limits.night_mode_end
        
        return self._is_time_in_range(current_time, night_start, night_end)
    
    def _is_time_in_range(self, current: str, start: str, end: str) -> bool:
        """Vérifie si une heure est dans une plage"""
        current_mins = self._time_to_minutes(current)
        start_mins = self._time_to_minutes(start)
        end_mins = self._time_to_minutes(end)
        
        if start_mins <= end_mins:  # Même jour
            return start_mins <= current_mins <= end_mins
        else:  # Chevauchement minuit
            return current_mins >= start_mins or current_mins <= end_mins
    
    def _is_time_close(self, time1: str, time2: str, margin_minutes: int) -> bool:
        """Vérifie si deux heures sont proches"""
        mins1 = self._time_to_minutes(time1)
        mins2 = self._time_to_minutes(time2)
        return abs(mins1 - mins2) <= margin_minutes
    
    def _time_to_minutes(self, time_str: str) -> int:
        """Convertit HH:MM en minutes depuis minuit"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def _confirm_night_session(self) -> bool:
        """Demande confirmation pour une session nocturne"""
        # En mode automatique, on refuse les sessions nocturnes
        return False
    
    def _save_session_data(self):
        """Sauvegarde chiffrée des données de session"""
        if not self.save_file:
            return
        
        try:
            # Sanitisation et validation des données avant sauvegarde
            sanitized_stats = self._sanitize_session_data(self.daily_stats)
            
            data = {
                'daily_stats': sanitized_stats,
                'limits': asdict(self.limits),
                'last_save': time.time(),
                'user_hash': self.user_hash,
                'version': "1.0",
                'integrity_check': self._generate_data_integrity_hash(sanitized_stats)
            }
            
            # Chiffrement des données sensibles
            json_data = json.dumps(data, ensure_ascii=False)
            encrypted_data = self._cipher.encrypt(json_data.encode('utf-8'))
            
            # Sauvegarde sécurisée avec permissions restreintes
            with open(self.save_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Permissions de fichier restrictives (lecture seule pour le propriétaire)
            import os
            os.chmod(self.save_file, 0o600)
            
            self._log_security_event("session_data_saved", {"file_size": len(encrypted_data)})
                
        except Exception as e:
            self._log_security_event("save_session_error", {"error": str(e)[:100]})
            logger.error(f"Erreur sauvegarde session sécurisée: {e}")
    
    def _sanitize_session_data(self, data) -> Dict:
        """Nettoie et valide les données de session"""
        if not isinstance(data, dict):
            return {}
        
        sanitized = {}
        for date, sessions in data.items():
            # Validation du format de date
            if not isinstance(date, str) or len(date) != 10:
                continue
            
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                continue
            
            # Sanitisation des sessions
            sanitized_sessions = []
            for session in sessions:
                if isinstance(session, dict):
                    clean_session = {}
                    # Validation et nettoyage des champs numériques
                    for key in ['start_time', 'end_time', 'total_active_time', 'total_break_time']:
                        if key in session and isinstance(session[key], (int, float)) and session[key] >= 0:
                            clean_session[key] = min(session[key], 86400)  # Max 24h
                    
                    # Validation des compteurs
                    for key in ['breaks_taken', 'actions_performed', 'experience_gained', 'levels_gained']:
                        if key in session and isinstance(session[key], int) and session[key] >= 0:
                            clean_session[key] = min(session[key], 1000000)  # Limite raisonnable
                    
                    if clean_session:  # Seulement si des données valides
                        sanitized_sessions.append(clean_session)
            
            if sanitized_sessions:
                sanitized[date] = sanitized_sessions
        
        return sanitized
    
    def _generate_data_integrity_hash(self, data) -> str:
        """Génère un hash d'intégrité pour les données"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _load_session_data(self):
        """Charge les données de session sauvegardées"""
        if not self.save_file:
            return
        
        try:
            with open(self.save_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.daily_stats = data.get('daily_stats', {})
            
            # Chargement des limites si disponibles
            if 'limits' in data:
                saved_limits = data['limits']
                for key, value in saved_limits.items():
                    if hasattr(self.limits, key):
                        setattr(self.limits, key, value)
                        
        except FileNotFoundError:
            logger.info("Aucune donnée de session sauvegardée trouvée")
        except Exception as e:
            logger.error(f"Erreur chargement session: {e}")
    
    def _print_session_report(self):
        """Affiche un rapport de fin de session"""
        if not self.current_stats:
            return
        
        stats = self.current_stats
        duration_hours = (stats.end_time - stats.start_time) / 3600 if stats.end_time else 0
        active_hours = stats.total_active_time / 3600
        break_hours = stats.total_break_time / 3600
        
        logger.info("=== RAPPORT DE SESSION ===")
        logger.info(f"Durée totale: {duration_hours:.1f}h")
        logger.info(f"Temps actif: {active_hours:.1f}h ({active_hours/duration_hours*100:.1f}%)")
        logger.info(f"Temps de pause: {break_hours:.1f}h ({break_hours/duration_hours*100:.1f}%)")
        logger.info(f"Actions effectuées: {stats.actions_performed}")
        logger.info(f"Pauses prises: {stats.breaks_taken}")
        logger.info(f"Actions/heure: {stats.actions_performed/active_hours:.1f}")
        logger.info("========================")


if __name__ == "__main__":
    # Test du gestionnaire de session
    logging.basicConfig(level=logging.INFO)
    
    # Configuration de test avec limites courtes
    test_limits = SessionLimits(
        max_continuous_minutes=5,  # 5min pour test
        short_break_frequency=2,   # Pause toutes les 2min
        short_break_duration=1     # 1min de pause
    )
    
    manager = SessionManager(test_limits)
    
    # Test de session
    if manager.start_session():
        print("Session démarrée")
        
        # Simulation d'activité
        for i in range(10):
            manager.record_action("test_action")
            print(f"Action {i+1} - État: {manager.get_session_info()}")
            time.sleep(30)  # 30s entre actions
        
        manager.end_session()
    else:
        print("Impossible de démarrer la session")