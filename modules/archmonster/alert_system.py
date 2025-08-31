"""
Système d'alertes multi-canaux pour archimonstres DOFUS.
"""

import asyncio
import aiohttp
import smtplib
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from email.mime.text import MimeText
import threading
import queue
import os

from .archmonster_detector import ArchmonsterDetection


class AlertPriority(Enum):
    """Priorités d'alerte."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertChannel(Enum):
    """Canaux d'alerte disponibles."""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SMS = "sms"
    EMAIL = "email"
    SOUND = "sound"
    SYSTEM = "system"


@dataclass
class AlertMessage:
    """Message d'alerte à envoyer."""
    detection: ArchmonsterDetection
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime


class DiscordAlerter:
    """Envoi d'alertes Discord via webhooks."""
    
    def __init__(self, config: Dict):
        self.webhook_url = config.get('webhook_url')
        self.username = config.get('username', 'DOFUS Bot')
    
    async def send_alert(self, alert: AlertMessage) -> bool:
        """Envoie alerte Discord."""
        if not self.webhook_url:
            return False
        
        try:
            embed = {
                "title": f"🐉 {alert.title}",
                "description": alert.message,
                "color": 0x00ff00,
                "fields": [
                    {"name": "Archimonstre", "value": alert.detection.name, "inline": True},
                    {"name": "Zone", "value": alert.detection.zone, "inline": True}
                ]
            }
            
            payload = {"username": self.username, "embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 204
        except Exception as e:
            logging.error(f"Erreur envoi Discord: {e}")
            return False


class TelegramAlerter:
    """Envoi d'alertes Telegram."""
    
    def __init__(self, config: Dict):
        self.bot_token = config.get('bot_token')
        self.chat_ids = config.get('chat_ids', [])
    
    async def send_alert(self, alert: AlertMessage) -> bool:
        """Envoie alerte Telegram."""
        # Simulation
        return True


class EmailAlerter:
    """Envoi d'alertes par email."""
    
    def __init__(self, config: Dict):
        self.smtp_server = config.get('smtp_server')
        self.username = config.get('username')
        self.password = config.get('password')
        self.to_emails = config.get('to_emails', [])
    
    async def send_alert(self, alert: AlertMessage) -> bool:
        """Envoie alerte par email."""
        # Simulation
        return True


class SoundAlerter:
    """Alertes sonores."""
    
    def __init__(self, config: Dict):
        self.sounds_path = config.get('sounds_path', 'data/sounds')
        self.default_sound = config.get('default_sound', 'alert.wav')
    
    async def send_alert(self, alert: AlertMessage) -> bool:
        """Joue son d'alerte."""
        try:
            sound_file = os.path.join(self.sounds_path, self.default_sound)
            if os.path.exists(sound_file):
                # Simulation son
                logging.info(f"Son d'alerte: {sound_file}")
            return True
        except Exception as e:
            logging.error(f"Erreur son: {e}")
            return False


class SystemAlerter:
    """Notifications système."""
    
    def __init__(self, config: Dict):
        self.timeout = config.get('timeout', 5000)
    
    async def send_alert(self, alert: AlertMessage) -> bool:
        """Envoie notification système."""
        try:
            title = f"🐉 {alert.title}"
            message = f"{alert.detection.name} en {alert.detection.zone}"
            logging.info(f"Notification système: {title} - {message}")
            return True
        except Exception as e:
            logging.error(f"Erreur notification système: {e}")
            return False


class AlertSystem:
    """Système principal de gestion des alertes."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerters = {}
        self.is_running = False
        self.alert_queue = queue.Queue()
        self.worker_thread = None
        
        self.stats = {
            'total_alerts': 0,
            'successful_alerts': 0,
            'failed_alerts': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_alerters()
    
    def _initialize_alerters(self):
        """Initialise les alerteurs."""
        alerter_classes = {
            AlertChannel.DISCORD: DiscordAlerter,
            AlertChannel.TELEGRAM: TelegramAlerter,
            AlertChannel.EMAIL: EmailAlerter,
            AlertChannel.SOUND: SoundAlerter,
            AlertChannel.SYSTEM: SystemAlerter
        }
        
        for channel_config in self.config.get('alert_channels', []):
            try:
                channel = AlertChannel(channel_config['channel'])
                if channel in alerter_classes:
                    alerter_class = alerter_classes[channel]
                    alerter = alerter_class(channel_config.get('config', {}))
                    self.alerters[channel] = alerter
                    self.logger.info(f"Alerteur {channel.value} initialisé")
            except Exception as e:
                self.logger.error(f"Erreur initialisation alerteur: {e}")
    
    def start(self):
        """Démarre le système d'alertes."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._alert_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.logger.info("Système d'alertes démarré")
    
    def stop(self):
        """Arrête le système d'alertes."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("Système d'alertes arrêté")
    
    def send_alert(self, detection: ArchmonsterDetection, 
                  priority: AlertPriority = AlertPriority.MEDIUM,
                  custom_message: str = None):
        """Envoie alerte pour détection."""
        if custom_message is None:
            custom_message = f"Archimonstre {detection.name} détecté en {detection.zone}!"
        
        title = f"Alerte Archimonstre"
        
        alert = AlertMessage(
            detection=detection,
            priority=priority,
            title=title,
            message=custom_message,
            timestamp=datetime.now()
        )
        
        self.alert_queue.put(alert)
        self.logger.info(f"Alerte mise en queue: {detection.name}")
    
    def _alert_worker(self):
        """Worker thread pour traitement des alertes."""
        while self.is_running:
            try:
                try:
                    alert = self.alert_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                self._process_alert(alert)
                self.alert_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erreur worker alertes: {e}")
    
    def _process_alert(self, alert: AlertMessage):
        """Traite une alerte."""
        self.stats['total_alerts'] += 1
        
        # Envoyer via tous les canaux
        for channel, alerter in self.alerters.items():
            try:
                asyncio.create_task(self._send_alert_async(alerter, alert))
            except Exception as e:
                self.logger.error(f"Erreur envoi alerte: {e}")
    
    async def _send_alert_async(self, alerter, alert: AlertMessage):
        """Envoie alerte de façon asynchrone."""
        try:
            success = await alerter.send_alert(alert)
            if success:
                self.stats['successful_alerts'] += 1
            else:
                self.stats['failed_alerts'] += 1
        except Exception as e:
            self.stats['failed_alerts'] += 1
            self.logger.error(f"Erreur envoi alerte async: {e}")
    
    def get_stats(self) -> Dict:
        """Récupère statistiques."""
        return dict(self.stats)
    
    def test_alerters(self):
        """Test tous les alerteurs."""
        from .archmonster_detector import ArchmonsterDetection
        
        test_detection = ArchmonsterDetection(
            name="Test Archimonstre",
            zone="Zone Test",
            position=(100, 200),
            confidence=0.95,
            detection_method="test"
        )
        
        self.send_alert(test_detection, AlertPriority.MEDIUM, "Test du système d'alertes")
        self.logger.info("Test des alerteurs lancé")