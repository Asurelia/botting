"""
Session Recorder - Dataset Collection pour AGA
Enregistre gameplay Dofus: vidéo 60 FPS + actions clavier/souris synchronisées

Architecture:
- Thread 1: Video capture (mss) @ 60 FPS
- Thread 2: Input monitoring (pynput) @ event-driven
- Thread 3: Game state extraction (OCR/detection) @ 5 FPS
- Storage: HDF5 format optimisé

Format Dataset:
{
    'frames': [frame_0, frame_1, ...],  # Video @ 60 FPS
    'actions': [(t, type, data), ...],   # Actions synchronisées
    'states': [(t, state_dict), ...],    # Game states @ 5 FPS
    'metadata': {session_id, player, duration, ...}
}
"""

import cv2
import numpy as np
import h5py
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Imports locaux
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.vision_capture_adapter import VisionCaptureAdapter
from core.platform_adapter import PlatformAdapter

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logging.warning("pynput non disponible - installer: pip install pynput")

logger = logging.getLogger(__name__)

@dataclass
class ActionEvent:
    """Événement action (clavier/souris)"""
    timestamp: float  # Temps depuis début session
    action_type: str  # 'key_press', 'key_release', 'mouse_move', 'mouse_click'
    data: Dict[str, Any]  # Données spécifiques à l'action

@dataclass
class GameState:
    """État du jeu extrait de la vision"""
    timestamp: float
    hp_ratio: float = 0.0
    mp_ratio: float = 0.0
    position: Tuple[int, int] = (0, 0)
    entities_detected: int = 0
    combat_active: bool = False
    ui_visible: bool = True

@dataclass
class SessionMetadata:
    """Métadonnées session d'enregistrement"""
    session_id: str
    start_time: str
    player_name: str
    character_level: int
    resolution: Tuple[int, int]
    fps_target: int
    os_platform: str

class SessionRecorder:
    """
    Enregistreur de sessions de gameplay Dofus

    Capture simultanée:
    - Vidéo @ 60 FPS (format optimisé)
    - Actions clavier/souris (event-driven)
    - États du jeu (extraction vision @ 5 FPS)

    Usage:
        recorder = SessionRecorder(output_dir="./datasets/sessions")
        recorder.start()
        # ... joueur joue ...
        recorder.stop()
        recorder.save()
    """

    def __init__(
        self,
        output_dir: str = "./datasets/sessions",
        fps_target: int = 60,
        state_extraction_fps: int = 5,
        max_duration: int = 3600,  # 1h max par session
        compression: str = "lzf"  # Compression HDF5
    ):
        """
        Initialise le session recorder

        Args:
            output_dir: Répertoire de sauvegarde
            fps_target: FPS cible vidéo (60 recommandé)
            state_extraction_fps: FPS extraction game state (5 recommandé)
            max_duration: Durée max session (secondes)
            compression: Compression HDF5 ('lzf', 'gzip', None)
        """

        # Configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps_target = fps_target
        self.state_extraction_fps = state_extraction_fps
        self.max_duration = max_duration
        self.compression = compression

        # Session state
        self.session_id = self._generate_session_id()
        self.is_recording = False
        self.start_timestamp = 0.0
        self.paused = False

        # Data buffers (thread-safe queues)
        self.frame_queue = queue.Queue(maxsize=fps_target * 5)  # 5s buffer
        self.action_queue = queue.Queue(maxsize=1000)
        self.state_queue = queue.Queue(maxsize=100)

        # Storage lists
        self.frames: List[np.ndarray] = []
        self.actions: List[ActionEvent] = []
        self.states: List[GameState] = []

        # Threads
        self.video_thread: Optional[threading.Thread] = None
        self.input_thread: Optional[threading.Thread] = None
        self.state_thread: Optional[threading.Thread] = None
        self.storage_thread: Optional[threading.Thread] = None

        # Vision system
        self.vision_adapter = VisionCaptureAdapter()
        self.platform_adapter = PlatformAdapter()

        # Input monitoring
        if PYNPUT_AVAILABLE:
            self.keyboard_listener: Optional[keyboard.Listener] = None
            self.mouse_listener: Optional[mouse.Listener] = None

        # Metadata
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            start_time="",
            player_name="unknown",
            character_level=0,
            resolution=(0, 0),
            fps_target=fps_target,
            os_platform="Linux" if self.platform_adapter.is_linux() else "Windows"
        )

        logger.info(f"SessionRecorder initialisé - Session ID: {self.session_id}")

    def _generate_session_id(self) -> str:
        """Génère ID unique pour la session"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_current_time(self) -> float:
        """Retourne temps depuis début session"""
        if self.start_timestamp == 0:
            return 0.0
        return time.time() - self.start_timestamp

    # ===== VIDEO CAPTURE THREAD =====

    def _video_capture_loop(self):
        """
        Thread de capture vidéo @ 60 FPS
        Enregistre frames dans frame_queue
        """
        logger.info(f"Video capture thread démarré (target: {self.fps_target} FPS)")

        frame_interval = 1.0 / self.fps_target
        frame_count = 0

        while self.is_recording:
            if self.paused:
                time.sleep(0.1)
                continue

            start_time = time.time()

            # Capture frame
            frame = self.vision_adapter.capture(use_cache=False)

            if frame is not None:
                timestamp = self._get_current_time()

                try:
                    # Enqueue frame (non-blocking)
                    self.frame_queue.put((timestamp, frame), block=False)
                    frame_count += 1
                except queue.Full:
                    logger.warning("Frame queue full - dropping frame")

            # Sleep pour maintenir FPS target
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Check max duration
            if self._get_current_time() > self.max_duration:
                logger.warning(f"Max duration atteinte ({self.max_duration}s) - arrêt capture")
                self.stop()
                break

        logger.info(f"Video capture thread terminé - {frame_count} frames capturées")

    # ===== INPUT MONITORING THREADS =====

    def _on_key_press(self, key):
        """Callback pynput - touche pressée"""
        if not self.is_recording or self.paused:
            return

        timestamp = self._get_current_time()

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        action = ActionEvent(
            timestamp=timestamp,
            action_type="key_press",
            data={"key": key_char}
        )

        try:
            self.action_queue.put(action, block=False)
        except queue.Full:
            logger.warning("Action queue full - dropping key press")

    def _on_key_release(self, key):
        """Callback pynput - touche relâchée"""
        if not self.is_recording or self.paused:
            return

        timestamp = self._get_current_time()

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        action = ActionEvent(
            timestamp=timestamp,
            action_type="key_release",
            data={"key": key_char}
        )

        try:
            self.action_queue.put(action, block=False)
        except queue.Full:
            pass

    def _on_mouse_move(self, x, y):
        """Callback pynput - souris déplacée"""
        if not self.is_recording or self.paused:
            return

        timestamp = self._get_current_time()

        action = ActionEvent(
            timestamp=timestamp,
            action_type="mouse_move",
            data={"x": x, "y": y}
        )

        # Throttle mouse move events (trop nombreux)
        if len(self.actions) == 0 or timestamp - self.actions[-1].timestamp > 0.05:
            try:
                self.action_queue.put(action, block=False)
            except queue.Full:
                pass

    def _on_mouse_click(self, x, y, button, pressed):
        """Callback pynput - clic souris"""
        if not self.is_recording or self.paused:
            return

        timestamp = self._get_current_time()

        action = ActionEvent(
            timestamp=timestamp,
            action_type="mouse_click" if pressed else "mouse_release",
            data={"x": x, "y": y, "button": str(button)}
        )

        try:
            self.action_queue.put(action, block=False)
        except queue.Full:
            pass

    # ===== GAME STATE EXTRACTION THREAD =====

    def _state_extraction_loop(self):
        """
        Thread d'extraction game state @ 5 FPS
        Analyse frames pour extraire HP, MP, position, etc.
        """
        logger.info(f"State extraction thread démarré ({self.state_extraction_fps} FPS)")

        state_interval = 1.0 / self.state_extraction_fps
        state_count = 0

        while self.is_recording:
            if self.paused:
                time.sleep(0.1)
                continue

            start_time = time.time()

            # Capture frame pour analyse
            frame = self.vision_adapter.capture(use_cache=True)

            if frame is not None:
                timestamp = self._get_current_time()

                # Extraction basique (à améliorer avec DL)
                state = self._extract_game_state(frame, timestamp)

                try:
                    self.state_queue.put(state, block=False)
                    state_count += 1
                except queue.Full:
                    logger.warning("State queue full - dropping state")

            # Sleep pour maintenir FPS target
            elapsed = time.time() - start_time
            sleep_time = max(0, state_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"State extraction thread terminé - {state_count} states extraits")

    def _extract_game_state(self, frame: np.ndarray, timestamp: float) -> GameState:
        """
        Extrait game state depuis frame (vision basique)
        TODO: Remplacer par Deep Learning (Phase 5+)
        """

        # Detection basique HP/MP via HSV
        height, width = frame.shape[:2]
        hp_region = frame[0:50, 0:250]

        hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)

        # HP bar (vert)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        hp_mask = cv2.inRange(hsv, lower_green, upper_green)
        hp_ratio = float(np.sum(hp_mask > 0) / hp_mask.size)

        # MP bar (bleu)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mp_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mp_ratio = float(np.sum(mp_mask > 0) / mp_mask.size)

        # Combat detection (rouge dans frame)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        full_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(full_hsv, lower_red, upper_red)
        combat_active = np.sum(red_mask > 0) > (height * width * 0.01)

        return GameState(
            timestamp=timestamp,
            hp_ratio=min(1.0, hp_ratio * 5),  # Calibration approximative
            mp_ratio=min(1.0, mp_ratio * 5),
            position=(0, 0),  # TODO: extraire de minimap
            entities_detected=0,  # TODO: YOLO detection
            combat_active=bool(combat_active),
            ui_visible=True
        )

    # ===== STORAGE THREAD =====

    def _storage_loop(self):
        """
        Thread de stockage - déqueue et stocke dans memory
        Évite de bloquer les threads de capture
        """
        logger.info("Storage thread démarré")

        while self.is_recording or not self.frame_queue.empty():
            # Process frames
            try:
                timestamp, frame = self.frame_queue.get(timeout=0.1)
                self.frames.append((timestamp, frame))
            except queue.Empty:
                pass

            # Process actions
            try:
                action = self.action_queue.get(timeout=0.01)
                self.actions.append(action)
            except queue.Empty:
                pass

            # Process states
            try:
                state = self.state_queue.get(timeout=0.01)
                self.states.append(state)
            except queue.Empty:
                pass

        logger.info(f"Storage thread terminé - {len(self.frames)} frames, {len(self.actions)} actions, {len(self.states)} states stockés")

    # ===== PUBLIC API =====

    def start(self, player_name: str = "unknown", character_level: int = 0):
        """
        Démarre l'enregistrement de la session

        Args:
            player_name: Nom du joueur
            character_level: Niveau du personnage
        """

        if self.is_recording:
            logger.warning("Recording déjà en cours")
            return

        # Update metadata
        self.metadata.start_time = datetime.now().isoformat()
        self.metadata.player_name = player_name
        self.metadata.character_level = character_level

        # Detect resolution
        test_frame = self.vision_adapter.capture(use_cache=False)
        if test_frame is not None:
            self.metadata.resolution = (test_frame.shape[1], test_frame.shape[0])

        # Clear buffers
        self.frames = []
        self.actions = []
        self.states = []

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # Start recording
        self.is_recording = True
        self.start_timestamp = time.time()

        # Start threads
        self.video_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
        self.state_thread = threading.Thread(target=self._state_extraction_loop, daemon=True)
        self.storage_thread = threading.Thread(target=self._storage_loop, daemon=True)

        self.video_thread.start()
        self.state_thread.start()
        self.storage_thread.start()

        # Start input monitoring
        if PYNPUT_AVAILABLE:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click
            )

            self.keyboard_listener.start()
            self.mouse_listener.start()

        logger.info(f"🎬 Recording démarré - Session: {self.session_id}")
        logger.info(f"   Player: {player_name} (Level {character_level})")
        logger.info(f"   Resolution: {self.metadata.resolution}")
        logger.info(f"   FPS Target: {self.fps_target}")

    def stop(self):
        """Arrête l'enregistrement"""

        if not self.is_recording:
            logger.warning("Aucun recording en cours")
            return

        logger.info("⏹️  Arrêt recording...")

        # Stop recording flag
        self.is_recording = False

        # Stop input listeners
        if PYNPUT_AVAILABLE:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            if self.mouse_listener:
                self.mouse_listener.stop()

        # Wait for threads
        if self.video_thread:
            self.video_thread.join(timeout=5)
        if self.state_thread:
            self.state_thread.join(timeout=5)
        if self.storage_thread:
            self.storage_thread.join(timeout=5)

        duration = time.time() - self.start_timestamp

        logger.info(f"✅ Recording arrêté - Durée: {duration:.1f}s")
        logger.info(f"   Frames: {len(self.frames)}")
        logger.info(f"   Actions: {len(self.actions)}")
        logger.info(f"   States: {len(self.states)}")

    def pause(self):
        """Met en pause l'enregistrement"""
        self.paused = True
        logger.info("⏸️  Recording en pause")

    def resume(self):
        """Reprend l'enregistrement"""
        self.paused = False
        logger.info("▶️  Recording repris")

    def save(self, format: str = "hdf5") -> Path:
        """
        Sauvegarde la session enregistrée

        Args:
            format: Format de sauvegarde ('hdf5', 'npz', 'pkl')

        Returns:
            Path: Chemin du fichier sauvegardé
        """

        if len(self.frames) == 0:
            logger.error("Aucune donnée à sauvegarder")
            return None

        if format == "hdf5":
            return self._save_hdf5()
        elif format == "npz":
            return self._save_npz()
        else:
            raise ValueError(f"Format non supporté: {format}")

    def _save_hdf5(self) -> Path:
        """Sauvegarde au format HDF5 optimisé"""

        output_path = self.output_dir / f"{self.session_id}.h5"

        logger.info(f"💾 Sauvegarde HDF5: {output_path}")

        with h5py.File(output_path, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            meta_dict = asdict(self.metadata)
            for key, value in meta_dict.items():
                if isinstance(value, tuple):
                    meta_group.attrs[key] = list(value)
                else:
                    meta_group.attrs[key] = value

            # Frames (compression importante)
            if len(self.frames) > 0:
                timestamps = np.array([t for t, _ in self.frames], dtype=np.float32)
                frames_array = np.array([f for _, f in self.frames], dtype=np.uint8)

                f.create_dataset('frame_timestamps', data=timestamps, compression=self.compression)
                f.create_dataset('frames', data=frames_array, compression=self.compression, chunks=True)

            # Actions
            if len(self.actions) > 0:
                action_group = f.create_group('actions')
                timestamps = np.array([a.timestamp for a in self.actions], dtype=np.float32)
                action_types = np.array([a.action_type.encode('utf-8') for a in self.actions])

                action_group.create_dataset('timestamps', data=timestamps, compression=self.compression)
                action_group.create_dataset('types', data=action_types, compression=self.compression)

                # Data (JSON serialized)
                action_data_json = json.dumps([a.data for a in self.actions])
                action_group.attrs['data_json'] = action_data_json

            # States
            if len(self.states) > 0:
                state_group = f.create_group('states')
                timestamps = np.array([s.timestamp for s in self.states], dtype=np.float32)
                hp_ratios = np.array([s.hp_ratio for s in self.states], dtype=np.float32)
                mp_ratios = np.array([s.mp_ratio for s in self.states], dtype=np.float32)
                combat_flags = np.array([s.combat_active for s in self.states], dtype=bool)

                state_group.create_dataset('timestamps', data=timestamps, compression=self.compression)
                state_group.create_dataset('hp_ratios', data=hp_ratios, compression=self.compression)
                state_group.create_dataset('mp_ratios', data=mp_ratios, compression=self.compression)
                state_group.create_dataset('combat_active', data=combat_flags, compression=self.compression)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Session sauvegardée: {output_path.name} ({file_size:.1f} MB)")

        return output_path

    def _save_npz(self) -> Path:
        """Sauvegarde au format NPZ (numpy compressed)"""

        output_path = self.output_dir / f"{self.session_id}.npz"

        logger.info(f"💾 Sauvegarde NPZ: {output_path}")

        # Prepare data
        save_dict = {
            'metadata': json.dumps(asdict(self.metadata)),
            'frame_timestamps': np.array([t for t, _ in self.frames]),
            'frames': np.array([f for _, f in self.frames]),
            'action_timestamps': np.array([a.timestamp for a in self.actions]),
            'action_types': np.array([a.action_type for a in self.actions]),
            'action_data': json.dumps([a.data for a in self.actions]),
            'state_timestamps': np.array([s.timestamp for s in self.states]),
            'state_hp': np.array([s.hp_ratio for s in self.states]),
            'state_mp': np.array([s.mp_ratio for s in self.states]),
        }

        np.savez_compressed(output_path, **save_dict)

        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Session sauvegardée: {output_path.name} ({file_size:.1f} MB)")

        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques de la session"""

        duration = self._get_current_time() if self.is_recording else (
            self.frames[-1][0] if len(self.frames) > 0 else 0
        )

        actual_fps = len(self.frames) / duration if duration > 0 else 0

        return {
            'session_id': self.session_id,
            'duration': duration,
            'frames_captured': len(self.frames),
            'actions_recorded': len(self.actions),
            'states_extracted': len(self.states),
            'fps_target': self.fps_target,
            'fps_actual': actual_fps,
            'is_recording': self.is_recording,
            'resolution': self.metadata.resolution,
        }


# Helper function
def create_recorder(
    output_dir: str = "./datasets/sessions",
    fps: int = 60,
    player_name: str = "unknown",
    character_level: int = 0
) -> SessionRecorder:
    """
    Helper pour créer et démarrer un recorder rapidement

    Usage:
        recorder = create_recorder(player_name="Sylvain", character_level=50)
        # ... jouer ...
        recorder.stop()
        recorder.save()
    """
    recorder = SessionRecorder(output_dir=output_dir, fps_target=fps)
    recorder.start(player_name=player_name, character_level=character_level)
    return recorder


# Test du module
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🎬 Test Session Recorder")
    print()

    if not PYNPUT_AVAILABLE:
        print("⚠️  pynput non disponible - installer: pip install pynput")
        print()

    # Créer recorder
    recorder = SessionRecorder(
        output_dir="./datasets/test_sessions",
        fps_target=30,  # 30 FPS pour test
        max_duration=10  # 10 secondes max
    )

    print(f"Session ID: {recorder.session_id}")
    print()

    # Démarrer
    print("▶️  Démarrage recording (10 secondes)...")
    recorder.start(player_name="TestPlayer", character_level=1)

    # Wait
    time.sleep(10)

    # Arrêter
    print()
    print("⏹️  Arrêt recording...")
    recorder.stop()

    # Stats
    print()
    print("📊 Statistiques:")
    stats = recorder.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Save
    print()
    output_path = recorder.save(format="hdf5")

    if output_path:
        print(f"✅ Session sauvegardée: {output_path}")

    print()
    print("✅ Test terminé!")
