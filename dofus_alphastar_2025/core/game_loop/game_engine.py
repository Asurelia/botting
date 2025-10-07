"""
Game Engine - Boucle de jeu principale
Orchestre vision, décision, actions
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import traceback

from .game_state import GameState, BotState, create_game_state
from ..safety.observation_mode import ObservationMode, create_observation_mode
from ..memory import ShortTermMemory, create_short_term_memory

# Import Vision complète
try:
    from ..vision_engine_v2 import create_vision_engine
    VISION_V2_AVAILABLE = True
except ImportError:
    VISION_V2_AVAILABLE = False
    logger.warning("Vision Engine V2 non disponible")

# Import Brain intégré
try:
    from ..decision.autonomous_brain_integrated import create_integrated_brain
    from ..combat.combo_library import CharacterClass
    INTEGRATED_BRAIN_AVAILABLE = True
except ImportError:
    INTEGRATED_BRAIN_AVAILABLE = False
    logger.warning("Brain intégré non disponible")

# Import Action System
try:
    from ..actions.action_system import create_action_system
    ACTION_SYSTEM_AVAILABLE = True
except ImportError:
    ACTION_SYSTEM_AVAILABLE = False
    logger.warning("Action System non disponible")

logger = logging.getLogger(__name__)


class GameEngine:
    """
    Moteur de jeu principal - Boucle autonome
    
    Architecture:
    1. Vision → Extrait Game State
    2. Decision → Prend décision basée sur état
    3. Action → Exécute l'action choisie
    4. Repeat
    """
    
    def __init__(
        self,
        target_fps: int = 10,
        observation_mode: bool = True,
        character_class: CharacterClass = None
    ):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.character_class = character_class or CharacterClass.IOP

        # État du jeu
        self.game_state = create_game_state()

        # Composants (seront initialisés plus tard)
        self.vision_system = None
        self.brain = None  # Brain intégré (remplace decision_engine)
        self.action_system = None
        self.observation_mode = None
        self.memory = create_short_term_memory()
        
        # Contrôle de la boucle
        self.running = False
        self.paused = False
        self.main_thread: Optional[threading.Thread] = None
        
        # Statistiques
        self.stats = {
            'frames_processed': 0,
            'actions_executed': 0,
            'errors': 0,
            'avg_fps': 0.0,
            'start_time': 0.0,
            'uptime': 0.0
        }
        
        # Callbacks
        self.on_state_update: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Safety
        if observation_mode:
            self.observation_mode = create_observation_mode()
            self.observation_mode.enable()
        
        logger.info(f"GameEngine initialisé (target FPS: {target_fps})")
    
    def initialize_systems(self):
        """Initialise tous les systèmes nécessaires"""
        logger.info("Initialisation des systèmes...")

        try:
            # ===  VISION ENGINE V2 ===
            if VISION_V2_AVAILABLE:
                logger.info("Initialisation Vision Engine V2...")
                self.vision_system = create_vision_engine()
                logger.info("✓ Vision Engine V2 initialisé")
            else:
                logger.warning("✗ Vision Engine V2 non disponible")

            # === BRAIN INTÉGRÉ ===
            if INTEGRATED_BRAIN_AVAILABLE:
                logger.info(f"Initialisation Brain Intégré (classe: {self.character_class.value})...")
                self.brain = create_integrated_brain(self.character_class)
                logger.info("✓ Brain Intégré initialisé avec 17 systèmes")
            else:
                logger.warning("✗ Brain Intégré non disponible")

            # === ACTION SYSTEM ===
            if ACTION_SYSTEM_AVAILABLE:
                logger.info("Initialisation Action System...")
                self.action_system = create_action_system()
                logger.info("✓ Action System initialisé")
            else:
                logger.warning("✗ Action System non disponible")

            systems_ok = (
                (self.vision_system is not None) and
                (self.brain is not None) and
                (self.action_system is not None)
            )

            if systems_ok:
                logger.info("✅ Tous les systèmes initialisés avec succès!")
            else:
                logger.warning("⚠️ Certains systèmes manquants, fonctionnalité limitée")

            return systems_ok

        except Exception as e:
            logger.error(f"Erreur initialisation systèmes: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def start(self):
        """Démarre la boucle de jeu"""
        if self.running:
            logger.warning("Game engine déjà démarré")
            return False
        
        logger.info("Démarrage du game engine...")
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Démarre dans un thread séparé
        self.main_thread = threading.Thread(
            target=self._main_loop,
            name="GameEngine",
            daemon=True
        )
        self.main_thread.start()
        
        logger.info("Game engine démarré")
        return True
    
    def stop(self):
        """Arrête la boucle de jeu"""
        if not self.running:
            logger.warning("Game engine n'est pas démarré")
            return False
        
        logger.info("Arrêt du game engine...")
        
        self.running = False
        
        # Attend la fin du thread
        if self.main_thread:
            self.main_thread.join(timeout=5.0)
        
        # Sauvegarde observations si mode actif
        if self.observation_mode:
            self.observation_mode.save_observations()
        
        logger.info("Game engine arrêté")
        return True
    
    def pause(self):
        """Met en pause"""
        self.paused = True
        logger.info("Game engine en pause")
    
    def resume(self):
        """Reprend"""
        self.paused = False
        logger.info("Game engine repris")
    
    def _main_loop(self):
        """Boucle principale du jeu"""
        logger.info("Boucle principale démarrée")
        
        frame_count = 0
        fps_samples = []
        last_fps_update = time.time()
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Pause check
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    # === 1. UPDATE GAME STATE ===
                    self._update_game_state()

                    # === 2. VISION → Analyse frame ===
                    vision_data = None
                    if self.vision_system:
                        try:
                            # Capturer et analyser l'écran
                            frame = self.vision_system.capture_screen()
                            vision_data = self.vision_system.analyze_frame(frame)
                            self._apply_vision_to_state(vision_data)
                        except Exception as e:
                            logger.error(f"Erreur vision: {e}")

                    # === 3. BRAIN → Décision intelligente ===
                    decision = None
                    if self.brain and self.game_state.can_act():
                        try:
                            # Brain décide avec tous les systèmes intégrés
                            decision = self.brain.decide(self.game_state, vision_data)
                        except Exception as e:
                            logger.error(f"Erreur brain: {e}")

                        # === 4. ACTION → Exécution ===
                        if decision:
                            self._execute_decision(decision)
                    
                    # === 5. STATS & CALLBACKS ===
                    frame_count += 1
                    self.stats['frames_processed'] = frame_count
                    
                    if self.on_state_update:
                        self.on_state_update(self.game_state)
                    
                    # FPS calculation
                    if time.time() - last_fps_update > 1.0:
                        if fps_samples:
                            self.stats['avg_fps'] = sum(fps_samples) / len(fps_samples)
                        fps_samples = []
                        last_fps_update = time.time()
                    
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"Erreur dans la boucle: {e}")
                    logger.debug(traceback.format_exc())
                    
                    if self.on_error:
                        self.on_error(e)
                    
                    # Évite boucle infinie d'erreurs
                    time.sleep(1.0)
                
                # === FRAME TIMING ===
                frame_duration = time.time() - frame_start
                fps_samples.append(1.0 / frame_duration if frame_duration > 0 else 0)
                
                # Attendre pour respecter target FPS
                sleep_time = self.frame_time - frame_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Erreur fatale dans la boucle principale: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            logger.info(f"Boucle principale terminée ({frame_count} frames)")
    
    def _update_game_state(self):
        """Met à jour le game state"""
        self.game_state.update_timestamp()
        self.stats['uptime'] = time.time() - self.stats['start_time']
    
    def _apply_vision_to_state(self, vision_data: Dict[str, Any]):
        """Applique les données de vision au game state"""
        if not vision_data:
            return
        
        # Mettre à jour character
        if 'character' in vision_data:
            char_data = vision_data['character']
            self.game_state.character.hp = char_data.get('hp', 100)
            self.game_state.character.max_hp = char_data.get('max_hp', 100)
            self.game_state.character.hp_percent = char_data.get('hp_percent', 100.0)
            self.game_state.character.pa = char_data.get('pa', 6)
            self.game_state.character.max_pa = char_data.get('max_pa', 6)
            self.game_state.character.pm = char_data.get('pm', 3)
            self.game_state.character.max_pm = char_data.get('max_pm', 3)
        
        # Mettre à jour combat
        if 'combat' in vision_data:
            combat_data = vision_data['combat']
            was_in_combat = self.game_state.combat.in_combat
            is_in_combat = combat_data.get('in_combat', False)
            
            if is_in_combat and not was_in_combat:
                self.game_state.enter_combat()
            elif not is_in_combat and was_in_combat:
                self.game_state.exit_combat()
            
            self.game_state.combat.my_turn = combat_data.get('my_turn', False)
        
        # Mettre à jour UI
        if 'ui' in vision_data:
            ui_data = vision_data['ui']
            self.game_state.ui.window_active = ui_data.get('window_active', True)
    
    def _execute_decision(self, decision: Dict[str, Any]):
        """Exécute une décision"""
        action_type = decision.get('action_type')
        action_details = decision.get('details', {})
        
        # Mode observation: bloquer toutes les actions
        if self.observation_mode and self.observation_mode.is_enabled():
            blocked = self.observation_mode.intercept_action(
                action_type=action_type,
                action_details=action_details,
                game_state=self.game_state.to_dict(),
                reason="Game engine decision"
            )
            
            if blocked:
                logger.debug(f"Action bloquée par observation mode: {action_type}")
                return
        
        # Exécuter l'action
        if self.action_system:
            success = self.action_system.execute(action_type, action_details)
            
            if success:
                self.stats['actions_executed'] += 1
                self.game_state.last_action = action_type
                self.game_state.last_action_time = time.time()
                self.game_state.last_action_success = True
            else:
                self.game_state.last_action_success = False
            
            # Mémoriser la décision
            if self.memory:
                self.memory.add_decision(
                    action_type=action_type,
                    details=action_details,
                    reason=decision.get('reason', 'unknown'),
                    success=success
                )
    
    def set_state_callback(self, callback: Callable):
        """Définit callback pour updates d'état"""
        self.on_state_update = callback
    
    def set_error_callback(self, callback: Callable):
        """Définit callback pour erreurs"""
        self.on_error = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return self.stats.copy()
    
    def get_state(self) -> GameState:
        """Retourne l'état actuel"""
        return self.game_state
    
    def is_running(self) -> bool:
        """Engine en cours?"""
        return self.running
    
    def is_paused(self) -> bool:
        """Engine en pause?"""
        return self.paused


def create_game_engine(
    target_fps: int = 10,
    observation_mode: bool = True
) -> GameEngine:
    """Factory function"""
    return GameEngine(
        target_fps=target_fps,
        observation_mode=observation_mode
    )
