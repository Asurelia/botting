"""
Système de suivi et d'analyse de l'historique des états du jeu
Permet la prédiction, l'analyse de patterns et les statistiques avancées
"""

import time
import pickle
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import logging
import statistics
import json

from .realtime_state import GameState, CombatState, MapType


@dataclass
class StateSnapshot:
    """Snapshot d'un état du jeu avec métadonnées"""
    timestamp: datetime
    character_level: int
    character_hp_percentage: float
    current_map: Tuple[int, int]
    combat_state: str
    pa_remaining: int
    pm_remaining: int
    kamas: int
    energy_percentage: float
    
    # Contexte additionnel
    actions_taken: List[str] = field(default_factory=list)
    events_occurred: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Pattern:
    """Pattern comportemental détecté"""
    pattern_id: str
    pattern_type: str  # "movement", "combat", "farming", "economic"
    description: str
    occurrences: int
    success_rate: float
    last_seen: datetime
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Prédiction d'état futur"""
    predicted_state: Dict[str, Any]
    confidence: float
    time_horizon: timedelta
    based_on_patterns: List[str]
    created_at: datetime


class StateTracker:
    """
    Système de suivi et d'analyse de l'état du jeu
    
    Fonctionnalités:
    - Historique complet des états avec compression intelligente
    - Détection automatique de patterns comportementaux
    - Prédictions basées sur l'historique
    - Statistiques de performance et d'efficacité
    - Base de données persistante pour l'apprentissage
    """
    
    def __init__(self, max_memory_states: int = 10000, db_path: str = "data/state_history.db"):
        """
        Initialise le tracker d'état
        
        Args:
            max_memory_states: Nombre maximum d'états en mémoire
            db_path: Chemin vers la base de données SQLite
        """
        self.max_memory_states = max_memory_states
        self.db_path = db_path
        
        # Configuration du logging
        self.logger = logging.getLogger(f"{__name__}.StateTracker")
        
        # Stockage en mémoire (ring buffer pour performance)
        self.state_history = deque(maxlen=max_memory_states)
        self.compressed_history = []  # Historique ancien compressé
        
        # Patterns détectés
        self.detected_patterns: Dict[str, Pattern] = {}
        self.pattern_matchers: List[Callable] = []
        
        # Prédictions en cours
        self.active_predictions: List[Prediction] = []
        
        # Statistiques de session
        self.session_stats = {
            "session_start": datetime.now(),
            "total_states_recorded": 0,
            "patterns_detected": 0,
            "successful_predictions": 0,
            "failed_predictions": 0
        }
        
        # Métriques de performance
        self.performance_metrics = {
            "xp_per_hour": 0.0,
            "kamas_per_hour": 0.0,
            "maps_per_hour": 0.0,
            "combat_efficiency": 0.0,
            "death_rate": 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialisation de la base de données
        self._init_database()
        
        # Configuration des pattern matchers
        self._setup_pattern_matchers()
        
        self.logger.info("StateTracker initialisé")
    
    def _init_database(self) -> None:
        """Initialise la base de données SQLite"""
        try:
            # Création du dossier data s'il n'existe pas
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table des états historiques
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS state_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        character_level INTEGER,
                        hp_percentage REAL,
                        map_x INTEGER,
                        map_y INTEGER,
                        combat_state TEXT,
                        pa_remaining INTEGER,
                        pm_remaining INTEGER,
                        kamas INTEGER,
                        energy_percentage REAL,
                        actions_taken TEXT,
                        events_occurred TEXT,
                        performance_data TEXT
                    )
                """)
                
                # Table des patterns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        description TEXT,
                        occurrences INTEGER DEFAULT 1,
                        success_rate REAL DEFAULT 0.0,
                        last_seen REAL,
                        conditions TEXT,
                        outcomes TEXT
                    )
                """)
                
                # Table des prédictions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL,
                        predicted_state TEXT,
                        confidence REAL,
                        time_horizon REAL,
                        actual_outcome TEXT,
                        was_successful BOOLEAN
                    )
                """)
                
                # Index pour performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON state_history(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la DB: {e}")
    
    def _setup_pattern_matchers(self) -> None:
        """Configure les détecteurs de patterns"""
        self.pattern_matchers = [
            self._detect_farming_patterns,
            self._detect_combat_patterns,
            self._detect_movement_patterns,
            self._detect_economic_patterns,
            self._detect_death_patterns
        ]
    
    def record_state(self, game_state: GameState, actions_taken: List[str] = None, 
                    events_occurred: List[str] = None) -> None:
        """
        Enregistre un nouvel état du jeu
        
        Args:
            game_state: État actuel du jeu
            actions_taken: Actions prises depuis le dernier état
            events_occurred: Événements survenus
        """
        try:
            with self._lock:
                # Création du snapshot
                snapshot = StateSnapshot(
                    timestamp=datetime.now(),
                    character_level=game_state.character.level,
                    character_hp_percentage=game_state.character.hp_percentage(),
                    current_map=game_state.map_state.coordinates,
                    combat_state=game_state.combat.state.value,
                    pa_remaining=game_state.character.current_pa,
                    pm_remaining=game_state.character.current_pm,
                    kamas=game_state.character.kamas,
                    energy_percentage=game_state.character.energy_percentage(),
                    actions_taken=actions_taken or [],
                    events_occurred=events_occurred or []
                )
                
                # Ajout à l'historique mémoire
                self.state_history.append(snapshot)
                self.session_stats["total_states_recorded"] += 1
                
                # Sauvegarde périodique en base (tous les 50 états)
                if self.session_stats["total_states_recorded"] % 50 == 0:
                    self._save_to_database(snapshot)
                
                # Détection de patterns
                self._analyze_patterns()
                
                # Mise à jour des métriques
                self._update_performance_metrics()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement d'état: {e}")
    
    def _save_to_database(self, snapshot: StateSnapshot) -> None:
        """Sauvegarde un état en base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO state_history 
                    (timestamp, character_level, hp_percentage, map_x, map_y, 
                     combat_state, pa_remaining, pm_remaining, kamas, energy_percentage,
                     actions_taken, events_occurred, performance_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.timestamp.timestamp(),
                    snapshot.character_level,
                    snapshot.character_hp_percentage,
                    snapshot.current_map[0],
                    snapshot.current_map[1],
                    snapshot.combat_state,
                    snapshot.pa_remaining,
                    snapshot.pm_remaining,
                    snapshot.kamas,
                    snapshot.energy_percentage,
                    json.dumps(snapshot.actions_taken),
                    json.dumps(snapshot.events_occurred),
                    json.dumps(snapshot.performance_metrics)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur de sauvegarde DB: {e}")
    
    def predict_next_state(self, time_horizon: timedelta = timedelta(minutes=5)) -> Optional[Prediction]:
        """
        Prédit l'état futur basé sur les patterns historiques
        
        Args:
            time_horizon: Horizon temporel de la prédiction
            
        Returns:
            Prediction: Prédiction ou None si impossible
        """
        if len(self.state_history) < 10:
            return None  # Pas assez de données
        
        try:
            with self._lock:
                current_state = self.state_history[-1]
                
                # Analyse des trends récents
                recent_states = list(self.state_history)[-20:]  # 20 derniers états
                
                # Prédiction basée sur les patterns
                predicted_state = self._calculate_state_prediction(recent_states, time_horizon)
                
                # Calcul de la confiance basée sur la stabilité des patterns
                confidence = self._calculate_prediction_confidence(recent_states)
                
                # Patterns utilisés pour cette prédiction
                relevant_patterns = self._find_relevant_patterns(current_state)
                
                prediction = Prediction(
                    predicted_state=predicted_state,
                    confidence=confidence,
                    time_horizon=time_horizon,
                    based_on_patterns=[p.pattern_id for p in relevant_patterns],
                    created_at=datetime.now()
                )
                
                self.active_predictions.append(prediction)
                
                # Nettoyage des anciennes prédictions
                self._cleanup_old_predictions()
                
                return prediction
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {e}")
            return None
    
    def _calculate_state_prediction(self, recent_states: List[StateSnapshot], 
                                   time_horizon: timedelta) -> Dict[str, Any]:
        """Calcule la prédiction d'état basée sur les tendances"""
        if len(recent_states) < 5:
            return {}
        
        # Calcul des tendances
        time_diffs = [(recent_states[i].timestamp - recent_states[0].timestamp).total_seconds() 
                     for i in range(len(recent_states))]
        
        # Tendance HP
        hp_values = [s.character_hp_percentage for s in recent_states]
        hp_trend = self._calculate_linear_trend(time_diffs, hp_values)
        
        # Tendance Kamas
        kamas_values = [s.kamas for s in recent_states]
        kamas_trend = self._calculate_linear_trend(time_diffs, kamas_values)
        
        # Tendance Énergie
        energy_values = [s.energy_percentage for s in recent_states]
        energy_trend = self._calculate_linear_trend(time_diffs, energy_values)
        
        # Projection dans le futur
        future_seconds = time_horizon.total_seconds()
        current_state = recent_states[-1]
        
        predicted_hp = max(0, min(100, current_state.character_hp_percentage + hp_trend * future_seconds))
        predicted_kamas = max(0, current_state.kamas + kamas_trend * future_seconds)
        predicted_energy = max(0, min(100, current_state.energy_percentage + energy_trend * future_seconds))
        
        return {
            "hp_percentage": predicted_hp,
            "kamas": int(predicted_kamas),
            "energy_percentage": predicted_energy,
            "map_coordinates": current_state.current_map,  # Supposé stable
            "confidence_factors": {
                "hp_trend": hp_trend,
                "kamas_trend": kamas_trend,
                "energy_trend": energy_trend
            }
        }
    
    def _calculate_linear_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """Calcule la tendance linéaire simple"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Éviter division par zéro
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        # Calcul de la pente
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_prediction_confidence(self, recent_states: List[StateSnapshot]) -> float:
        """Calcule la confiance de prédiction basée sur la stabilité"""
        if len(recent_states) < 5:
            return 0.1
        
        # Facteurs de confiance
        factors = []
        
        # Stabilité HP
        hp_values = [s.character_hp_percentage for s in recent_states]
        hp_stability = 1.0 - (statistics.stdev(hp_values) / 100.0) if len(set(hp_values)) > 1 else 1.0
        factors.append(hp_stability)
        
        # Stabilité de position
        map_changes = sum(1 for i in range(1, len(recent_states)) 
                         if recent_states[i].current_map != recent_states[i-1].current_map)
        position_stability = max(0.0, 1.0 - (map_changes / len(recent_states)))
        factors.append(position_stability)
        
        # Stabilité d'état de combat
        combat_changes = sum(1 for i in range(1, len(recent_states))
                           if recent_states[i].combat_state != recent_states[i-1].combat_state)
        combat_stability = max(0.0, 1.0 - (combat_changes / len(recent_states)))
        factors.append(combat_stability)
        
        # Moyenne pondérée
        confidence = statistics.mean(factors)
        return min(0.95, max(0.05, confidence))  # Entre 5% et 95%
    
    def _analyze_patterns(self) -> None:
        """Analyse les derniers états pour détecter des patterns"""
        if len(self.state_history) < 5:
            return
        
        # Application de tous les détecteurs de patterns
        for matcher in self.pattern_matchers:
            try:
                patterns = matcher()
                for pattern in patterns:
                    self._update_pattern(pattern)
            except Exception as e:
                self.logger.error(f"Erreur dans le détecteur de pattern: {e}")
    
    def _detect_farming_patterns(self) -> List[Pattern]:
        """Détecte les patterns de farming"""
        patterns = []
        
        if len(self.state_history) < 10:
            return patterns
        
        recent_states = list(self.state_history)[-10:]
        
        # Pattern: Récolte répétitive sur même carte
        current_map = recent_states[-1].current_map
        same_map_states = [s for s in recent_states if s.current_map == current_map]
        
        if len(same_map_states) >= 7:  # Majoritairement sur la même carte
            # Vérifier les actions de farming
            farming_actions = []
            for state in same_map_states:
                farming_actions.extend([a for a in state.actions_taken 
                                      if any(keyword in a.lower() 
                                           for keyword in ['harvest', 'mine', 'cut', 'collect'])])
            
            if len(farming_actions) >= 3:
                pattern = Pattern(
                    pattern_id=f"farming_{current_map[0]}_{current_map[1]}",
                    pattern_type="farming",
                    description=f"Farming répétitif sur carte {current_map}",
                    occurrences=len(farming_actions),
                    success_rate=0.8,  # Sera mis à jour avec les données réelles
                    last_seen=datetime.now(),
                    conditions={"map": current_map, "min_actions": 3},
                    outcomes={"resources_gained": len(farming_actions)}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_combat_patterns(self) -> List[Pattern]:
        """Détecte les patterns de combat"""
        patterns = []
        
        if len(self.state_history) < 15:
            return patterns
        
        recent_states = list(self.state_history)[-15:]
        combat_states = [s for s in recent_states if s.combat_state != "no_combat"]
        
        if len(combat_states) >= 3:
            # Pattern: Séquence de combats
            combat_durations = []
            combat_hp_loss = []
            
            current_combat_start = None
            pre_combat_hp = 100
            
            for i, state in enumerate(recent_states):
                if state.combat_state != "no_combat" and current_combat_start is None:
                    current_combat_start = i
                    pre_combat_hp = recent_states[max(0, i-1)].character_hp_percentage
                
                elif state.combat_state == "no_combat" and current_combat_start is not None:
                    # Fin du combat
                    duration = i - current_combat_start
                    hp_loss = pre_combat_hp - state.character_hp_percentage
                    
                    combat_durations.append(duration)
                    combat_hp_loss.append(hp_loss)
                    current_combat_start = None
            
            if combat_durations:
                avg_duration = statistics.mean(combat_durations)
                avg_hp_loss = statistics.mean(combat_hp_loss)
                
                pattern = Pattern(
                    pattern_id="combat_efficiency",
                    pattern_type="combat",
                    description="Efficacité de combat basée sur durée et HP perdus",
                    occurrences=len(combat_durations),
                    success_rate=max(0.1, 1.0 - (avg_hp_loss / 100)),
                    last_seen=datetime.now(),
                    conditions={"min_combats": 2},
                    outcomes={"avg_duration": avg_duration, "avg_hp_loss": avg_hp_loss}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_movement_patterns(self) -> List[Pattern]:
        """Détecte les patterns de mouvement"""
        patterns = []
        
        if len(self.state_history) < 8:
            return patterns
        
        recent_states = list(self.state_history)[-8:]
        maps_visited = [s.current_map for s in recent_states]
        
        # Détection de boucles de cartes
        if len(set(maps_visited)) <= 4 and len(maps_visited) >= 6:
            # Possible pattern de rotation
            pattern = Pattern(
                pattern_id="map_rotation",
                pattern_type="movement",
                description=f"Rotation entre {len(set(maps_visited))} cartes",
                occurrences=1,
                success_rate=0.9,
                last_seen=datetime.now(),
                conditions={"maps_in_rotation": list(set(maps_visited))},
                outcomes={"efficiency": "high"}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_economic_patterns(self) -> List[Pattern]:
        """Détecte les patterns économiques"""
        patterns = []
        
        if len(self.state_history) < 10:
            return patterns
        
        recent_states = list(self.state_history)[-10:]
        kamas_values = [s.kamas for s in recent_states]
        
        # Détection de gains réguliers
        kamas_gains = []
        for i in range(1, len(kamas_values)):
            gain = kamas_values[i] - kamas_values[i-1]
            if gain > 0:
                kamas_gains.append(gain)
        
        if len(kamas_gains) >= 3:
            avg_gain = statistics.mean(kamas_gains)
            pattern = Pattern(
                pattern_id="kamas_generation",
                pattern_type="economic",
                description=f"Génération régulière de kamas: {avg_gain:.0f} par action",
                occurrences=len(kamas_gains),
                success_rate=0.85,
                last_seen=datetime.now(),
                conditions={"min_gains": 3},
                outcomes={"avg_gain_per_action": avg_gain}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_death_patterns(self) -> List[Pattern]:
        """Détecte les patterns de mort/danger"""
        patterns = []
        
        if len(self.state_history) < 20:
            return patterns
        
        recent_states = list(self.state_history)[-20:]
        low_hp_incidents = [s for s in recent_states if s.character_hp_percentage <= 20]
        
        if len(low_hp_incidents) >= 2:
            pattern = Pattern(
                pattern_id="danger_situations",
                pattern_type="safety",
                description=f"{len(low_hp_incidents)} situations dangereuses détectées",
                occurrences=len(low_hp_incidents),
                success_rate=0.3,  # Faible car dangereux
                last_seen=datetime.now(),
                conditions={"hp_threshold": 20},
                outcomes={"incidents_count": len(low_hp_incidents)}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _update_pattern(self, pattern: Pattern) -> None:
        """Met à jour ou ajoute un pattern"""
        if pattern.pattern_id in self.detected_patterns:
            existing = self.detected_patterns[pattern.pattern_id]
            existing.occurrences += 1
            existing.last_seen = pattern.last_seen
            # Moyenne mobile pour success_rate
            existing.success_rate = (existing.success_rate * 0.8 + pattern.success_rate * 0.2)
        else:
            self.detected_patterns[pattern.pattern_id] = pattern
            self.session_stats["patterns_detected"] += 1
        
        # Sauvegarde périodique des patterns
        if self.session_stats["patterns_detected"] % 10 == 0:
            self._save_patterns_to_db()
    
    def _save_patterns_to_db(self) -> None:
        """Sauvegarde les patterns en base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for pattern in self.detected_patterns.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO patterns 
                        (pattern_id, pattern_type, description, occurrences, success_rate, 
                         last_seen, conditions, outcomes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.description,
                        pattern.occurrences,
                        pattern.success_rate,
                        pattern.last_seen.timestamp(),
                        json.dumps(pattern.conditions),
                        json.dumps(pattern.outcomes)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde patterns: {e}")
    
    def _find_relevant_patterns(self, current_state: StateSnapshot) -> List[Pattern]:
        """Trouve les patterns pertinents pour l'état actuel"""
        relevant = []
        
        for pattern in self.detected_patterns.values():
            # Vérification des conditions du pattern
            conditions = pattern.conditions
            
            if pattern.pattern_type == "farming":
                if "map" in conditions and conditions["map"] == current_state.current_map:
                    relevant.append(pattern)
            
            elif pattern.pattern_type == "combat":
                if current_state.combat_state != "no_combat":
                    relevant.append(pattern)
            
            elif pattern.pattern_type == "movement":
                if "maps_in_rotation" in conditions:
                    if current_state.current_map in conditions["maps_in_rotation"]:
                        relevant.append(pattern)
        
        return relevant
    
    def _update_performance_metrics(self) -> None:
        """Met à jour les métriques de performance"""
        if len(self.state_history) < 10:
            return
        
        # Calcul sur les dernières heures
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_states = [s for s in self.state_history if s.timestamp > one_hour_ago]
        
        if len(recent_states) < 2:
            return
        
        time_span_hours = (recent_states[-1].timestamp - recent_states[0].timestamp).total_seconds() / 3600
        
        if time_span_hours > 0:
            # Kamas par heure
            kamas_gain = recent_states[-1].kamas - recent_states[0].kamas
            self.performance_metrics["kamas_per_hour"] = kamas_gain / time_span_hours
            
            # Changements de carte par heure (mobilité)
            map_changes = len(set(s.current_map for s in recent_states))
            self.performance_metrics["maps_per_hour"] = map_changes / time_span_hours
            
            # Taux de mort (incidents HP = 0)
            deaths = sum(1 for s in recent_states if s.character_hp_percentage <= 0)
            self.performance_metrics["death_rate"] = deaths / time_span_hours
    
    def _cleanup_old_predictions(self) -> None:
        """Nettoie les anciennes prédictions"""
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.active_predictions = [p for p in self.active_predictions 
                                 if p.created_at > cutoff_time]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du tracker"""
        with self._lock:
            return {
                "session": dict(self.session_stats),
                "patterns": {
                    "total_detected": len(self.detected_patterns),
                    "by_type": self._group_patterns_by_type(),
                    "most_successful": self._get_most_successful_patterns()
                },
                "performance": dict(self.performance_metrics),
                "predictions": {
                    "active_count": len(self.active_predictions),
                    "success_rate": self._calculate_prediction_success_rate()
                },
                "data": {
                    "states_in_memory": len(self.state_history),
                    "database_path": self.db_path
                }
            }
    
    def _group_patterns_by_type(self) -> Dict[str, int]:
        """Groupe les patterns par type"""
        groups = defaultdict(int)
        for pattern in self.detected_patterns.values():
            groups[pattern.pattern_type] += 1
        return dict(groups)
    
    def _get_most_successful_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retourne les patterns les plus réussis"""
        patterns = list(self.detected_patterns.values())
        patterns.sort(key=lambda p: p.success_rate * p.occurrences, reverse=True)
        
        return [
            {
                "pattern_id": p.pattern_id,
                "type": p.pattern_type,
                "description": p.description,
                "success_rate": p.success_rate,
                "occurrences": p.occurrences
            }
            for p in patterns[:limit]
        ]
    
    def _calculate_prediction_success_rate(self) -> float:
        """Calcule le taux de succès des prédictions"""
        total = self.session_stats["successful_predictions"] + self.session_stats["failed_predictions"]
        if total == 0:
            return 0.0
        return self.session_stats["successful_predictions"] / total
    
    def analyze_patterns(self, pattern_type: str = None, days: int = 7) -> Dict[str, Any]:
        """
        Analyse approfondie des patterns sur une période donnée
        
        Args:
            pattern_type: Type de pattern à analyser (None = tous)
            days: Nombre de jours à analyser
            
        Returns:
            Dict avec l'analyse détaillée
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filtrage des patterns
            patterns_to_analyze = []
            for pattern in self.detected_patterns.values():
                if pattern.last_seen > cutoff_date:
                    if pattern_type is None or pattern.pattern_type == pattern_type:
                        patterns_to_analyze.append(pattern)
            
            if not patterns_to_analyze:
                return {"message": "Aucun pattern trouvé pour les critères donnés"}
            
            # Analyse statistique
            success_rates = [p.success_rate for p in patterns_to_analyze]
            occurrences = [p.occurrences for p in patterns_to_analyze]
            
            analysis = {
                "period_days": days,
                "pattern_type": pattern_type or "all",
                "total_patterns": len(patterns_to_analyze),
                "success_rate_stats": {
                    "average": statistics.mean(success_rates),
                    "median": statistics.median(success_rates),
                    "min": min(success_rates),
                    "max": max(success_rates),
                    "std_dev": statistics.stdev(success_rates) if len(success_rates) > 1 else 0
                },
                "occurrence_stats": {
                    "total": sum(occurrences),
                    "average": statistics.mean(occurrences),
                    "median": statistics.median(occurrences),
                    "max": max(occurrences)
                },
                "top_patterns": self._get_most_successful_patterns(limit=10)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de patterns: {e}")
            return {"error": str(e)}
    
    def export_data(self, filepath: str, format: str = "json") -> bool:
        """
        Exporte les données du tracker
        
        Args:
            filepath: Chemin du fichier d'export
            format: Format d'export ("json", "csv")
            
        Returns:
            bool: True si l'export réussit
        """
        try:
            if format.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "session_stats": self.session_stats,
                    "performance_metrics": self.performance_metrics,
                    "patterns": {
                        pid: {
                            "pattern_id": p.pattern_id,
                            "pattern_type": p.pattern_type,
                            "description": p.description,
                            "occurrences": p.occurrences,
                            "success_rate": p.success_rate,
                            "last_seen": p.last_seen.isoformat(),
                            "conditions": p.conditions,
                            "outcomes": p.outcomes
                        }
                        for pid, p in self.detected_patterns.items()
                    },
                    "recent_states": [
                        {
                            "timestamp": s.timestamp.isoformat(),
                            "character_level": s.character_level,
                            "hp_percentage": s.character_hp_percentage,
                            "map": s.current_map,
                            "combat_state": s.combat_state,
                            "kamas": s.kamas
                        }
                        for s in list(self.state_history)[-100:]  # 100 derniers états
                    ]
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                return True
            
            else:
                self.logger.error(f"Format d'export non supporté: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export: {e}")
            return False