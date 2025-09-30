"""
AI Learning Module - Module d'apprentissage intégré au framework IA existant
Adaptation du système d'apprentissage pour l'architecture AI Framework
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from collections import deque
import json
from pathlib import Path

# Import du framework IA existant
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.ai_framework import AIModule, AIModuleState

# Import des modules d'apprentissage développés
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_core'))
from learning_engine import LearningEngine, GameAction, GameSituation, BehaviorAnalyzer

class LearningModule(AIModule):
    """Module d'apprentissage intégré au framework IA"""

    def __init__(self, data_dir: Path):
        super().__init__("Learning")
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Composants d'apprentissage
        self.learning_engine: Optional[LearningEngine] = None
        self.behavior_analyzer: Optional[BehaviorAnalyzer] = None

        # Configuration
        self.learning_active = True
        self.observation_interval = 0.5
        self.pattern_analysis_interval = 30.0

        # État d'apprentissage
        self.current_situation: Optional[GameSituation] = None
        self.action_buffer: deque = deque(maxlen=100)
        self.learning_sessions: List[Dict[str, Any]] = []

        # Métriques
        self.actions_observed = 0
        self.patterns_discovered = 0
        self.recommendations_generated = 0
        self.accuracy_score = 0.0
        self.last_pattern_analysis = 0.0

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Implémentation de l'initialisation du module d'apprentissage"""
        try:
            self.logger.info("Initialisation du module Learning...")

            # Configuration depuis le config
            learning_config = config.get('learning', {})
            self.learning_active = learning_config.get('enabled', True)
            self.observation_interval = learning_config.get('observation_interval', 0.5)
            self.pattern_analysis_interval = learning_config.get('pattern_analysis_interval', 30.0)

            if not self.learning_active:
                self.logger.info("Apprentissage désactivé dans la configuration")
                return True

            # Initialiser Learning Engine
            self.logger.info("Initialisation Learning Engine...")
            self.learning_engine = LearningEngine(self.data_dir)
            if not self.learning_engine.initialize():
                self.logger.error("Échec initialisation Learning Engine")
                return False

            # Démarrer apprentissage
            if not self.learning_engine.start_learning():
                self.logger.error("Échec démarrage Learning Engine")
                return False

            # Initialiser Behavior Analyzer
            self.behavior_analyzer = BehaviorAnalyzer()

            self.logger.info("Module Learning initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation module Learning: {e}")
            return False

    async def _run_impl(self):
        """Boucle principale du module d'apprentissage"""
        if not self.learning_active:
            self.logger.info("Apprentissage désactivé - module en veille")
            await self._shutdown_event.wait()
            return

        self.logger.info("Démarrage boucle Learning...")

        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()

                # Observer l'environnement
                await self._observe_environment()

                # Analyser patterns périodiquement
                if start_time - self.last_pattern_analysis >= self.pattern_analysis_interval:
                    await self._analyze_patterns()
                    self.last_pattern_analysis = start_time

                # Générer recommandations si demandé
                await self._update_recommendations()

                # Attendre intervalle d'observation
                await asyncio.sleep(self.observation_interval)

            except asyncio.CancelledError:
                self.logger.info("Module Learning arrêté")
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle Learning: {e}")
                await asyncio.sleep(1.0)

    async def _observe_environment(self):
        """Observe l'environnement de jeu pour apprentissage"""
        try:
            # Récupérer données de vision depuis shared_data
            vision_data = self._shared_data.get('last_screenshot')
            ocr_results = self._shared_data.get('last_ocr_results', [])

            if vision_data is not None and self.learning_engine:
                # Analyser situation actuelle
                loop = asyncio.get_event_loop()
                situation = await loop.run_in_executor(
                    None, self.learning_engine.observe_situation, vision_data, ocr_results
                )

                self.current_situation = situation

                # Mettre à jour shared_data pour autres modules
                self._shared_data['current_situation'] = situation
                self._shared_data['learning_active'] = True

        except Exception as e:
            self.logger.error(f"Erreur observation environnement: {e}")

    async def _analyze_patterns(self):
        """Analyse les patterns d'actions pour découvrir de nouveaux comportements"""
        try:
            if not self.behavior_analyzer or len(self.action_buffer) < 5:
                return

            self.logger.debug("Analyse des patterns comportementaux...")

            # Analyser actions récentes
            recent_actions = list(self.action_buffer)[-20:]  # 20 dernières actions

            loop = asyncio.get_event_loop()
            patterns = await loop.run_in_executor(
                None, self.behavior_analyzer.analyze_action_sequence, recent_actions
            )

            # Compter nouveaux patterns
            new_patterns = 0
            for pattern in patterns:
                if pattern.pattern_id not in self.behavior_analyzer.patterns:
                    new_patterns += 1
                    self.patterns_discovered += 1

            if new_patterns > 0:
                self.logger.info(f"Découverte de {new_patterns} nouveaux patterns")

            # Mettre à jour métriques
            await self._update_learning_metrics()

        except Exception as e:
            self.logger.error(f"Erreur analyse patterns: {e}")

    async def _update_recommendations(self):
        """Met à jour les recommandations d'actions"""
        try:
            if not self.current_situation or not self.learning_engine:
                return

            context = self.current_situation.situation_type

            # Générer recommandations
            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(
                None, self.learning_engine.get_action_recommendations, context
            )

            if recommendations:
                self.recommendations_generated += len(recommendations)

                # Mettre dans shared_data pour autres modules
                self._shared_data['action_recommendations'] = recommendations
                self._shared_data['recommendations_timestamp'] = time.time()

        except Exception as e:
            self.logger.error(f"Erreur génération recommandations: {e}")

    async def _update_learning_metrics(self):
        """Met à jour les métriques d'apprentissage"""
        try:
            if not self.learning_engine:
                return

            # Obtenir statistiques du learning engine
            stats = self.learning_engine.get_learning_statistics()

            # Calculer score de précision
            total_patterns = stats.get('patterns_count', 0)
            if total_patterns > 0:
                # Score basé sur la diversité et qualité des patterns
                pattern_quality = sum(
                    p.confidence_score for p in self.behavior_analyzer.patterns.values()
                ) / total_patterns if self.behavior_analyzer and self.behavior_analyzer.patterns else 0.5

                self.accuracy_score = min(1.0, pattern_quality)

            # Mettre à jour health du module
            self.health.performance_score = self.accuracy_score

        except Exception as e:
            self.logger.error(f"Erreur mise à jour métriques: {e}")

    async def observe_user_action(self, action_type: str, coordinates: Tuple[int, int],
                                success: bool = True, target_info: Dict[str, Any] = None):
        """Observe une action utilisateur pour apprentissage"""
        try:
            if not self.learning_active or not self.learning_engine:
                return

            # Créer action de jeu
            action = GameAction(
                timestamp=time.time(),
                action_type=action_type,
                coordinates=coordinates,
                context=self.current_situation.situation_type if self.current_situation else "unknown",
                success=success,
                target_info=target_info or {},
                screen_hash=str(hash(str(self._shared_data.get('last_screenshot', b''))))
            )

            # Observer action
            self.learning_engine.observe_action(action)
            self.action_buffer.append(action)
            self.actions_observed += 1

            self.logger.debug(f"Action observée: {action_type} à {coordinates}")

        except Exception as e:
            self.logger.error(f"Erreur observation action: {e}")

    async def get_contextual_recommendations(self, context: str = None) -> List[Dict[str, Any]]:
        """Obtient recommandations contextuelles"""
        try:
            if not self.learning_engine:
                return []

            if context is None and self.current_situation:
                context = self.current_situation.situation_type

            if context is None:
                return []

            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(
                None, self.learning_engine.get_action_recommendations, context
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Erreur recommandations contextuelles: {e}")
            return []

    async def start_learning_session(self, session_name: str = None):
        """Démarre une nouvelle session d'apprentissage"""
        try:
            session = {
                'name': session_name or f"session_{len(self.learning_sessions) + 1}",
                'start_time': time.time(),
                'actions_start': self.actions_observed,
                'patterns_start': self.patterns_discovered
            }

            self.learning_sessions.append(session)
            self.logger.info(f"Session d'apprentissage démarrée: {session['name']}")

        except Exception as e:
            self.logger.error(f"Erreur démarrage session: {e}")

    async def end_learning_session(self):
        """Termine la session d'apprentissage actuelle"""
        try:
            if not self.learning_sessions:
                return

            session = self.learning_sessions[-1]
            session['end_time'] = time.time()
            session['duration'] = session['end_time'] - session['start_time']
            session['actions_learned'] = self.actions_observed - session['actions_start']
            session['patterns_discovered'] = self.patterns_discovered - session['patterns_start']

            self.logger.info(f"Session terminée: {session['name']} - "
                           f"{session['actions_learned']} actions, "
                           f"{session['patterns_discovered']} patterns")

            # Sauvegarder session
            await self._save_session(session)

        except Exception as e:
            self.logger.error(f"Erreur fin session: {e}")

    async def _save_session(self, session: Dict[str, Any]):
        """Sauvegarde une session d'apprentissage"""
        try:
            sessions_dir = self.data_dir / "sessions"
            sessions_dir.mkdir(exist_ok=True)

            session_file = sessions_dir / f"{session['name']}_{int(session['start_time'])}.json"

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde session: {e}")

    async def _shutdown_impl(self):
        """Arrêt propre du module"""
        try:
            self.logger.info("Arrêt module Learning...")

            # Terminer session active
            if self.learning_sessions:
                await self.end_learning_session()

            # Arrêter learning engine
            if self.learning_engine:
                self.learning_engine.stop_learning()
                self.learning_engine.cleanup()

            self.logger.info("Module Learning arrêté proprement")

        except Exception as e:
            self.logger.error(f"Erreur arrêt module Learning: {e}")

    def get_module_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du module"""
        stats = {
            'learning_active': self.learning_active,
            'actions_observed': self.actions_observed,
            'patterns_discovered': self.patterns_discovered,
            'recommendations_generated': self.recommendations_generated,
            'accuracy_score': round(self.accuracy_score, 3),
            'current_situation': self.current_situation.situation_type if self.current_situation else None,
            'action_buffer_size': len(self.action_buffer),
            'learning_sessions': len(self.learning_sessions)
        }

        # Ajouter stats du learning engine si disponible
        if self.learning_engine:
            engine_stats = self.learning_engine.get_learning_statistics()
            stats.update({
                'engine_patterns_count': engine_stats.get('patterns_count', 0),
                'engine_uptime_minutes': round(engine_stats.get('uptime', 0) / 60, 1),
                'engine_accuracy': engine_stats.get('accuracy_score', 0.0)
            })

        return stats

# Factory function pour intégration avec le framework
def create_learning_module(data_dir: Path) -> LearningModule:
    """Crée une instance du module Learning"""
    return LearningModule(data_dir)