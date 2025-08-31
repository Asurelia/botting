"""Gestionnaire de notifications."""
import logging
import threading
import queue
from enum import Enum
from .archmonster_detector import ArchmonsterDetector, ArchmonsterDetection
from .alert_system import AlertSystem, AlertPriority
from .archmonster_tracker import ArchmonsterTracker
from .archmonster_database import ArchmonsterDatabase

class NotificationMode(Enum):
    SILENT = "silent"
    NORMAL = "normal" 
    AGGRESSIVE = "aggressive"
    ULTRA = "ultra"

class NotificationManager:
    def __init__(self, config):
        self.config = config
        self.detector = ArchmonsterDetector(config.get('detector', {}))
        self.alert_system = AlertSystem(config.get('alerts', {}))
        self.tracker = ArchmonsterTracker(config.get('tracker', {}))
        self.database = ArchmonsterDatabase(config.get('database_path', 'data/archmonsters.db'))
        
        self.mode = NotificationMode(config.get('mode', 'normal'))
        self.is_running = False
        self.detection_queue = queue.Queue()
        self.stats = {'total_notifications': 0}
        self.logger = logging.getLogger(__name__)
        self.detector.add_detection_callback(self._on_detection)
    
    def start(self):
        self.is_running = True
        self.detector.start_detection()
        self.alert_system.start()
        self.tracker.start_tracking()
        self.logger.info("Gestionnaire démarré")
    
    def stop(self):
        self.is_running = False
        self.detector.stop_detection()
        self.alert_system.stop()
        self.tracker.stop_tracking()
    
    def _on_detection(self, detection):
        self.detection_queue.put(detection)
    
    def set_mode(self, mode):
        self.mode = mode
    
    def test_notification(self, name="Test", zone="Test"):
        test = ArchmonsterDetection(name, zone, (0,0), 1.0, detection_method="test")
        self.alert_system.send_alert(test)
    
    def get_statistics(self):
        return {'is_running': self.is_running, 'current_mode': self.mode.value}