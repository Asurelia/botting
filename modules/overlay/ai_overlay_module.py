"""
AI Overlay Module - Module d'overlay intégré au framework IA existant
Adaptation du système d'overlay intelligent pour l'architecture AI Framework
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

# Import du framework IA existant
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.ai_framework import AIModule, AIModuleState

# Import des modules d'overlay développés
from .intelligent_overlay import IntelligentOverlay, OverlayConfig, OverlayType

class OverlayModule(AIModule):
    """Module d'overlay intégré au framework IA"""

    def __init__(self):
        super().__init__("Overlay")

        # Composants overlay
        self.overlay: Optional[IntelligentOverlay] = None
        self.config: Optional[OverlayConfig] = None

        # Configuration
        self.enabled = True
        self.transparency = 0.8
        self.max_elements = 10
        self.update_interval = 0.1

        # État
        self.active_recommendations = []
        self.last_update_time = 0.0

        # Métriques
        self.overlays_displayed = 0
        self.recommendations_shown = 0
        self.user_interactions = 0

    async def _initialize_impl(self, config: Dict[str, Any]) -> bool:
        """Implémentation de l'initialisation du module overlay"""
        try:
            self.logger.info("Initialisation du module Overlay...")

            # Configuration depuis le config
            overlay_config = config.get('overlay', {})
            self.enabled = overlay_config.get('enabled', True)
            self.transparency = overlay_config.get('transparency', 0.8)
            self.max_elements = overlay_config.get('max_elements', 10)
            self.update_interval = overlay_config.get('update_interval', 0.1)

            if not self.enabled:
                self.logger.info("Overlay désactivé dans la configuration")
                return True

            # Créer configuration overlay
            self.config = OverlayConfig(
                enable_overlay=True,
                transparency=self.transparency,
                max_elements=self.max_elements,
                default_duration=overlay_config.get('default_duration', 5.0),
                animation_speed=overlay_config.get('animation_speed', 1.0),
                anti_detection=overlay_config.get('anti_detection', True)
            )

            # Initialiser overlay intelligent
            self.logger.info("Initialisation IntelligentOverlay...")
            self.overlay = IntelligentOverlay(self.config)
            if not self.overlay.initialize():
                self.logger.error("Échec initialisation IntelligentOverlay")
                return False

            # Démarrer overlay
            if not self.overlay.start():
                self.logger.error("Échec démarrage IntelligentOverlay")
                return False

            self.logger.info("Module Overlay initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation module Overlay: {e}")
            return False

    async def _run_impl(self):
        """Boucle principale du module overlay"""
        if not self.enabled:
            self.logger.info("Overlay désactivé - module en veille")
            await self._shutdown_event.wait()
            return

        self.logger.info("Démarrage boucle Overlay...")

        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()

                # Mettre à jour overlay avec nouvelles recommandations
                await self._update_overlay_content()

                # Afficher recommandations d'apprentissage
                await self._display_learning_recommendations()

                # Afficher informations de contexte
                await self._display_context_information()

                # Attendre intervalle de mise à jour
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                self.logger.info("Module Overlay arrêté")
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle Overlay: {e}")
                await asyncio.sleep(1.0)

    async def _update_overlay_content(self):
        """Met à jour le contenu de l'overlay"""
        try:
            if not self.overlay:
                return

            current_time = time.time()

            # Récupérer nouvelles recommandations depuis shared_data
            recommendations = self._shared_data.get('action_recommendations', [])
            recommendations_timestamp = self._shared_data.get('recommendations_timestamp', 0)

            # Afficher seulement si nouvelles recommandations
            if recommendations_timestamp > self.last_update_time:
                await self._display_action_recommendations(recommendations)
                self.last_update_time = current_time

        except Exception as e:
            self.logger.error(f"Erreur mise à jour overlay: {e}")

    async def _display_action_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Affiche les recommandations d'actions"""
        try:
            if not recommendations or not self.overlay:
                return

            # Effacer anciennes recommandations
            self.overlay.clear_overlay(OverlayType.SPELL_HIGHLIGHT)
            self.overlay.clear_overlay(OverlayType.MOVEMENT_SUGGESTION)

            # Afficher top 3 recommandations
            for i, rec in enumerate(recommendations[:3]):
                confidence = rec.get('confidence', 0.5)
                priority = int(confidence * 10)  # Convertir confidence en priorité

                if 'actions' in rec and rec['actions']:
                    # Recommandation de séquence d'actions
                    action_sequence = rec['actions']

                    if 'spell' in action_sequence[0].lower():
                        # Surligner sort recommandé
                        self.overlay.highlight_spell(
                            position=(400 + i * 100, 300),
                            spell_name=action_sequence[0],
                            priority=priority
                        )
                    elif 'movement' in action_sequence[0].lower():
                        # Suggérer mouvement
                        self.overlay.suggest_movement(
                            from_pos=(300, 300),
                            to_pos=(400 + i * 50, 250),
                            reason=f"Stratégie optimale ({confidence:.1%})"
                        )

            self.recommendations_shown += len(recommendations[:3])
            self.overlays_displayed += 1

        except Exception as e:
            self.logger.error(f"Erreur affichage recommandations: {e}")

    async def _display_learning_recommendations(self):
        """Affiche recommandations basées sur l'apprentissage"""
        try:
            if not self.overlay:
                return

            # Récupérer situation actuelle
            current_situation = self._shared_data.get('current_situation')
            if not current_situation:
                return

            situation_type = current_situation.situation_type

            # Affichage contextuel selon situation
            if situation_type == 'combat':
                await self._display_combat_overlay()
            elif situation_type == 'quest':
                await self._display_quest_overlay()
            elif situation_type == 'exploration':
                await self._display_exploration_overlay()

        except Exception as e:
            self.logger.error(f"Erreur recommandations apprentissage: {e}")

    async def _display_combat_overlay(self):
        """Affiche overlay spécifique au combat"""
        try:
            # Analyser résultats OCR pour détecter ennemis
            ocr_results = self._shared_data.get('last_ocr_results', [])

            enemy_positions = []
            for ocr_result in ocr_results:
                text = ocr_result.text.lower()
                if any(word in text for word in ['ennemi', 'enemy', 'monster', 'mob']):
                    # Estimer position enemy
                    bbox = ocr_result.bbox
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    enemy_positions.append((center_x, center_y))

            # Marquer priorités d'attaque
            for i, pos in enumerate(enemy_positions[:3]):  # Max 3 cibles
                self.overlay.mark_target_priority(
                    position=pos,
                    priority_num=i + 1,
                    size=(80, 60)
                )

            # Suggérer position optimale
            if enemy_positions:
                optimal_pos = self._calculate_optimal_position(enemy_positions)
                self.overlay.suggest_movement(
                    from_pos=(300, 300),
                    to_pos=optimal_pos,
                    reason="Position tactique optimale"
                )

        except Exception as e:
            self.logger.error(f"Erreur overlay combat: {e}")

    async def _display_quest_overlay(self):
        """Affiche overlay spécifique aux quêtes"""
        try:
            # Récupérer données de quête depuis shared_data ou outils externes
            quest_info = self._shared_data.get('current_quest')

            if quest_info:
                # Afficher objectif actuel
                objective = quest_info.get('current_objective', 'Objectif inconnu')
                steps = quest_info.get('next_steps', [])

                self.overlay.show_quest_guidance(
                    position=(50, 100),
                    objective=objective,
                    steps=steps
                )

        except Exception as e:
            self.logger.error(f"Erreur overlay quête: {e}")

    async def _display_exploration_overlay(self):
        """Affiche overlay spécifique à l'exploration"""
        try:
            # Identifier points d'intérêt
            ocr_results = self._shared_data.get('last_ocr_results', [])

            interesting_points = []
            for ocr_result in ocr_results:
                text = ocr_result.text.lower()
                if any(word in text for word in ['ressource', 'coffre', 'pnj', 'npc', 'zaap']):
                    bbox = ocr_result.bbox
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    interesting_points.append((center_x, center_y, text))

            # Indiquer points d'intérêt
            for x, y, description in interesting_points[:2]:  # Max 2 points
                self.overlay.suggest_movement(
                    from_pos=(300, 300),
                    to_pos=(x, y),
                    reason=f"Point d'intérêt: {description}"
                )

        except Exception as e:
            self.logger.error(f"Erreur overlay exploration: {e}")

    async def _display_context_information(self):
        """Affiche informations contextuelles"""
        try:
            if not self.overlay:
                return

            # Récupérer statistiques des modules
            vision_stats = self._shared_data.get('vision_stats', {})
            learning_stats = self._shared_data.get('learning_stats', {})

            # Construire texte d'information
            info_text = []

            if 'game_state' in vision_stats:
                info_text.append(f"État: {vision_stats['game_state']}")

            if 'current_fps' in vision_stats:
                info_text.append(f"FPS: {vision_stats['current_fps']}")

            if 'accuracy_score' in learning_stats:
                accuracy = learning_stats['accuracy_score']
                info_text.append(f"IA: {accuracy:.1%}")

            if info_text:
                self.overlay.show_performance_stats(
                    stats={f"Info_{i}": text for i, text in enumerate(info_text)},
                    position=(10, 10)
                )

        except Exception as e:
            self.logger.error(f"Erreur affichage contexte: {e}")

    def _calculate_optimal_position(self, enemy_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calcule position optimale pour le combat"""
        if not enemy_positions:
            return (300, 300)

        # Centroid des ennemis
        avg_x = sum(pos[0] for pos in enemy_positions) / len(enemy_positions)
        avg_y = sum(pos[1] for pos in enemy_positions) / len(enemy_positions)

        # Position à distance optimale (environ 150 pixels)
        optimal_distance = 150
        angle = 0  # Simplification - position à droite

        optimal_x = int(avg_x + optimal_distance)
        optimal_y = int(avg_y)

        return (optimal_x, optimal_y)

    async def display_manual_recommendation(self, recommendation_type: str,
                                          position: Tuple[int, int],
                                          text: str, priority: int = 5):
        """Affiche une recommandation manuelle"""
        try:
            if not self.overlay:
                return

            if recommendation_type == "spell":
                self.overlay.highlight_spell(position, text, priority)
            elif recommendation_type == "movement":
                self.overlay.suggest_movement(
                    from_pos=(300, 300),
                    to_pos=position,
                    reason=text
                )
            elif recommendation_type == "target":
                self.overlay.mark_target_priority(position, priority)

            self.user_interactions += 1

        except Exception as e:
            self.logger.error(f"Erreur recommandation manuelle: {e}")

    async def clear_all_overlays(self):
        """Efface tous les overlays"""
        try:
            if self.overlay:
                self.overlay.clear_overlay()

        except Exception as e:
            self.logger.error(f"Erreur effacement overlays: {e}")

    async def _shutdown_impl(self):
        """Arrêt propre du module"""
        try:
            self.logger.info("Arrêt module Overlay...")

            if self.overlay:
                self.overlay.stop()
                self.overlay.cleanup()

            self.logger.info("Module Overlay arrêté proprement")

        except Exception as e:
            self.logger.error(f"Erreur arrêt module Overlay: {e}")

    def get_module_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du module"""
        stats = {
            'enabled': self.enabled,
            'overlays_displayed': self.overlays_displayed,
            'recommendations_shown': self.recommendations_shown,
            'user_interactions': self.user_interactions,
            'transparency': self.transparency,
            'max_elements': self.max_elements,
            'active_recommendations_count': len(self.active_recommendations)
        }

        # Ajouter stats de l'overlay si disponible
        if self.overlay:
            overlay_stats = self.overlay.directx_overlay.get_overlay_stats()
            stats.update({
                'overlay_active': overlay_stats.get('running', False),
                'active_elements': overlay_stats.get('active_elements', 0),
                'target_window': overlay_stats.get('target_window', False)
            })

        return stats

# Factory function pour intégration avec le framework
def create_overlay_module() -> OverlayModule:
    """Crée une instance du module Overlay"""
    return OverlayModule()