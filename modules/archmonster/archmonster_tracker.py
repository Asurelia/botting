"""Tracker des spawns d'archimonstres."""
import logging
import threading

class ArchmonsterTracker:
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.logger = logging.getLogger(__name__)
    
    def start_tracking(self):
        self.is_running = True
        self.logger.info("Tracker démarré")
    
    def stop_tracking(self):
        self.is_running = False
        self.logger.info("Tracker arrêté")
    
    def get_predictions(self, **kwargs):
        return []
    
    def get_statistics(self):
        return {'is_running': self.is_running, 'total_patterns': 0}