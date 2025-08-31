#!/usr/bin/env python3
"""
Système d'injection de paquets sécurisée pour automatisation DOFUS
Injection contrôlée et éthique pour tests et automatisation

Fonctionnalités:
- Injection de paquets TCP/UDP contrôlée et sécurisée
- Support protocoles DOFUS Unity avec validation
- Mode simulation pour tests sans impact réseau
- Limitation de débit et contrôles de sécurité
- Logging détaillé de toutes les opérations
- Interface pour plugins d'injection spécialisés
- Respect strict des ToS et usage éthique

Auteur: Système d'automatisation DOFUS
Date: 2025-08-31
Licence: Usage éducatif et recherche uniquement
"""

import os
import sys
import time
import json
import struct
import socket
import threading
import logging
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum, IntEnum
import hashlib
import ipaddress
from collections import defaultdict, deque
import statistics

# Importations avec gestion des erreurs
try:
    from scapy.all import send, sendp, Raw, IP, TCP, UDP, Ether
    from scapy.layers.inet import ICMP
    from scapy.error import Scapy_Exception
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy non disponible. Injection de paquets désactivée.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil non disponible. Vérifications processus limitées.")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Cryptography non disponible. Chiffrement des payloads désactivé.")


class InjectionMode(Enum):
    """Modes d'injection de paquets"""
    SIMULATION = "simulation"      # Simulation sans envoi réel
    LOCAL_ONLY = "local_only"     # Injection locale seulement
    CONTROLLED = "controlled"      # Injection contrôlée avec limites
    DIRECT = "direct"             # Injection directe (use with caution)


class PacketType(Enum):
    """Types de paquets supportés"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    RAW = "raw"
    DOFUS_GAME = "dofus_game"
    DOFUS_AUTH = "dofus_auth"


class InjectionResult(Enum):
    """Résultats d'injection"""
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SIMULATED = "simulated"
    RATE_LIMITED = "rate_limited"
    VALIDATION_ERROR = "validation_error"


@dataclass
class InjectionTarget:
    """Cible d'injection de paquets"""
    host: str
    port: int
    protocol: str = "tcp"
    interface: Optional[str] = None
    
    def __post_init__(self):
        # Validation de l'adresse IP
        try:
            self.ip_obj = ipaddress.ip_address(self.host)
        except ValueError:
            # Peut-être un hostname
            try:
                self.host = socket.gethostbyname(self.host)
                self.ip_obj = ipaddress.ip_address(self.host)
            except socket.gaierror:
                raise ValueError(f"Adresse invalide: {self.host}")
    
    @property
    def is_local(self) -> bool:
        """Vérifie si la cible est locale"""
        return self.ip_obj.is_loopback or self.ip_obj.is_private
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'interface': self.interface,
            'is_local': self.is_local
        }


@dataclass
class PacketTemplate:
    """Template de paquet pour injection"""
    id: str
    name: str
    packet_type: PacketType
    payload: bytes
    description: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    rate_limit_per_second: int = 10
    max_size: int = 1500
    
    def validate_payload(self) -> Tuple[bool, str]:
        """Valide le payload du template"""
        if len(self.payload) > self.max_size:
            return False, f"Payload trop volumineux: {len(self.payload)} > {self.max_size}"
        
        if len(self.payload) == 0:
            return False, "Payload vide"
        
        # Validation spécifique au type
        if self.packet_type == PacketType.DOFUS_GAME:
            # Vérifications basiques pour packets DOFUS
            if len(self.payload) < 4:
                return False, "Payload DOFUS trop court"
            
            # Vérifier que ce n'est pas du contenu sensible
            sensitive_patterns = [b'password', b'login', b'auth', b'token']
            payload_lower = self.payload.lower()
            for pattern in sensitive_patterns:
                if pattern in payload_lower:
                    return False, f"Contenu sensible détecté: {pattern.decode()}"
        
        return True, "Validation réussie"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'packet_type': self.packet_type.value,
            'payload_size': len(self.payload),
            'payload_hash': hashlib.sha256(self.payload).hexdigest(),
            'description': self.description,
            'headers': self.headers,
            'validation_rules': self.validation_rules,
            'rate_limit_per_second': self.rate_limit_per_second,
            'max_size': self.max_size
        }


@dataclass
class InjectionJob:
    """Job d'injection de paquets"""
    id: str
    target: InjectionTarget
    template: PacketTemplate
    count: int = 1
    interval: float = 1.0
    timeout: Optional[float] = None
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    status: str = "pending"
    results: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def is_completed(self) -> bool:
        return self.status in ['completed', 'failed', 'cancelled']
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.get('result') == InjectionResult.SUCCESS.value)
        return successful / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'target': self.target.to_dict(),
            'template': self.template.to_dict(),
            'count': self.count,
            'interval': self.interval,
            'timeout': self.timeout,
            'created_time': self.created_time,
            'started_time': self.started_time,
            'completed_time': self.completed_time,
            'status': self.status,
            'packets_sent': len(self.results),
            'success_rate': self.success_rate,
            'error_message': self.error_message
        }


class RateLimiter:
    """Limiteur de débit pour contrôler les injections"""
    
    def __init__(self, max_packets_per_second: int = 10, burst_size: int = 20):
        self.max_packets_per_second = max_packets_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()
        
        # Historique des injections
        self.injection_history = deque(maxlen=1000)
    
    def can_inject(self, packet_count: int = 1) -> bool:
        """Vérifie si l'injection est autorisée"""
        with self.lock:
            current_time = time.time()
            
            # Remplir le bucket avec les tokens
            time_passed = current_time - self.last_update
            self.tokens = min(self.burst_size, self.tokens + time_passed * self.max_packets_per_second)
            self.last_update = current_time
            
            # Vérifier si on a assez de tokens
            if self.tokens >= packet_count:
                self.tokens -= packet_count
                
                # Enregistrer l'injection
                self.injection_history.append({
                    'timestamp': current_time,
                    'packet_count': packet_count
                })
                
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du limiteur"""
        with self.lock:
            current_time = time.time()
            
            # Calculer le taux d'injection sur la dernière minute
            one_minute_ago = current_time - 60
            recent_injections = [inj for inj in self.injection_history if inj['timestamp'] >= one_minute_ago]
            
            packets_per_minute = sum(inj['packet_count'] for inj in recent_injections)
            
            return {
                'max_packets_per_second': self.max_packets_per_second,
                'current_tokens': self.tokens,
                'burst_size': self.burst_size,
                'packets_last_minute': packets_per_minute,
                'total_injections': len(self.injection_history)
            }


class SecurityValidator:
    """Validateur de sécurité pour les injections"""
    
    def __init__(self):
        self.blocked_ips = set()
        self.allowed_local_only = True
        self.max_payload_size = 10000  # 10KB max
        self.sensitive_patterns = [
            b'password', b'passwd', b'login', b'auth', b'token',
            b'session', b'cookie', b'secret', b'key', b'private'
        ]
        self.logger = logging.getLogger('SecurityValidator')
    
    def validate_target(self, target: InjectionTarget) -> Tuple[bool, str]:
        """Valide la cible d'injection"""
        # Vérification des IPs bloquées
        if target.host in self.blocked_ips:
            return False, f"IP bloquée: {target.host}"
        
        # Restriction locale si configurée
        if self.allowed_local_only and not target.is_local:
            return False, f"Injection vers IP externe interdite: {target.host}"
        
        # Vérification des ports sensibles
        sensitive_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        if target.port in sensitive_ports:
            self.logger.warning(f"Injection vers port sensible: {target.port}")
        
        return True, "Cible validée"
    
    def validate_payload(self, payload: bytes, packet_type: PacketType) -> Tuple[bool, str]:
        """Valide le payload d'injection"""
        # Vérification de la taille
        if len(payload) > self.max_payload_size:
            return False, f"Payload trop volumineux: {len(payload)} bytes"
        
        if len(payload) == 0:
            return False, "Payload vide"
        
        # Vérification du contenu sensible
        payload_lower = payload.lower()
        for pattern in self.sensitive_patterns:
            if pattern in payload_lower:
                return False, f"Contenu sensible détecté: {pattern.decode()}"
        
        # Vérifications spécifiques au type
        if packet_type == PacketType.DOFUS_GAME:
            # Vérifier que c'est bien du contenu DOFUS
            if not self._looks_like_dofus_packet(payload):
                return False, "Payload ne ressemble pas à un paquet DOFUS valide"
        
        return True, "Payload validé"
    
    def _looks_like_dofus_packet(self, payload: bytes) -> bool:
        """Heuristique pour vérifier si c'est un paquet DOFUS"""
        if len(payload) < 4:
            return False
        
        # Vérifications basiques de structure
        # (Basé sur l'observation publique du protocole)
        
        # Vérifier les premiers bytes pour des patterns typiques
        first_bytes = payload[:4]
        
        # Patterns connus (non-sensibles) du protocole DOFUS
        common_patterns = [
            b'\x00\x01',  # Messages de contrôle
            b'\x00\x02',  # Messages de chat
            b'\x00\x03',  # Messages de mouvement
        ]
        
        for pattern in common_patterns:
            if payload.startswith(pattern):
                return True
        
        # Vérifier la structure générale (longueur + type + données)
        if len(payload) >= 6:
            try:
                # Format typique: longueur (2 bytes) + type (2 bytes) + données
                length = struct.unpack('>H', first_bytes[:2])[0]
                if 4 <= length <= len(payload):
                    return True
            except:
                pass
        
        return False
    
    def block_ip(self, ip: str):
        """Bloque une IP"""
        self.blocked_ips.add(ip)
        self.logger.info(f"IP bloquée: {ip}")
    
    def unblock_ip(self, ip: str):
        """Débloque une IP"""
        self.blocked_ips.discard(ip)
        self.logger.info(f"IP débloquée: {ip}")


class PacketBuilder:
    """Constructeur de paquets pour injection"""
    
    def __init__(self):
        self.logger = logging.getLogger('PacketBuilder')
    
    def build_tcp_packet(self, target: InjectionTarget, payload: bytes, **kwargs) -> Optional[Any]:
        """Construit un paquet TCP"""
        if not SCAPY_AVAILABLE:
            return None
        
        try:
            # Source IP (utiliser l'interface locale)
            src_ip = kwargs.get('src_ip', '127.0.0.1')
            src_port = kwargs.get('src_port', 12345)
            
            # Construire le paquet
            packet = IP(dst=target.host) / TCP(
                sport=src_port,
                dport=target.port,
                flags=kwargs.get('tcp_flags', 'A'),  # ACK par défaut
                seq=kwargs.get('seq', 0),
                ack=kwargs.get('ack', 0)
            ) / Raw(load=payload)
            
            return packet
            
        except Exception as e:
            self.logger.error(f"Erreur construction paquet TCP: {e}")
            return None
    
    def build_udp_packet(self, target: InjectionTarget, payload: bytes, **kwargs) -> Optional[Any]:
        """Construit un paquet UDP"""
        if not SCAPY_AVAILABLE:
            return None
        
        try:
            src_ip = kwargs.get('src_ip', '127.0.0.1')
            src_port = kwargs.get('src_port', 12345)
            
            packet = IP(dst=target.host) / UDP(
                sport=src_port,
                dport=target.port
            ) / Raw(load=payload)
            
            return packet
            
        except Exception as e:
            self.logger.error(f"Erreur construction paquet UDP: {e}")
            return None
    
    def build_dofus_packet(self, target: InjectionTarget, template: PacketTemplate) -> Optional[Any]:
        """Construit un paquet spécifique DOFUS"""
        if template.packet_type == PacketType.DOFUS_GAME:
            # Les paquets DOFUS utilisent généralement TCP
            return self.build_tcp_packet(
                target, 
                template.payload,
                tcp_flags='PA',  # PUSH + ACK pour données de jeu
                **template.headers
            )
        elif template.packet_type == PacketType.DOFUS_AUTH:
            # Authentification également en TCP
            return self.build_tcp_packet(
                target,
                template.payload,
                tcp_flags='PA',
                **template.headers
            )
        
        return None


class PacketInjectorSafe:
    """
    Système d'injection de paquets sécurisé et contrôlé
    
    Fonctionnalités:
    - Injection contrôlée avec validation de sécurité
    - Support multi-protocoles (TCP, UDP, DOFUS)
    - Rate limiting et burst control
    - Mode simulation pour tests
    - Logging détaillé de toutes les opérations
    - Interface plugins extensible
    - Respect strict des ToS
    """
    
    def __init__(self,
                 mode: InjectionMode = InjectionMode.SIMULATION,
                 max_rate_per_second: int = 10,
                 log_dir: str = "logs/injection"):
        
        self.mode = mode
        self.running = False
        
        # Composants de sécurité
        self.rate_limiter = RateLimiter(max_rate_per_second)
        self.security_validator = SecurityValidator()
        self.packet_builder = PacketBuilder()
        
        # Files de travail
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[str, InjectionJob] = {}
        self.completed_jobs: Dict[str, InjectionJob] = {}
        
        # Templates de paquets
        self.packet_templates: Dict[str, PacketTemplate] = {}
        
        # Workers
        self.worker_threads: List[threading.Thread] = []
        self.num_workers = 2
        
        # Callbacks et plugins
        self.injection_callbacks: List[Callable[[str, InjectionResult, Dict], None]] = []
        self.job_callbacks: List[Callable[[InjectionJob], None]] = []
        
        # Statistiques
        self.stats = {
            'jobs_created': 0,
            'jobs_completed': 0,
            'packets_sent': 0,
            'packets_blocked': 0,
            'packets_simulated': 0,
            'start_time': None
        }
        
        # Configuration du logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Charger les templates par défaut
        self.load_default_templates()
    
    def setup_logging(self):
        """Configuration du logging détaillé"""
        self.logger = logging.getLogger('PacketInjectorSafe')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Handler fichier pour les injections
            injection_handler = logging.FileHandler(
                self.log_dir / f"packet_injector_{datetime.now().strftime('%Y%m%d')}.log"
            )
            injection_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            injection_handler.setFormatter(injection_formatter)
            self.logger.addHandler(injection_handler)
            
            # Handler sécurité pour les violations
            security_handler = logging.FileHandler(
                self.log_dir / f"injection_security_{datetime.now().strftime('%Y%m%d')}.log"
            )
            security_formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            security_handler.setFormatter(security_formatter)
            
            # Logger séparé pour la sécurité
            security_logger = logging.getLogger('InjectionSecurity')
            security_logger.addHandler(security_handler)
            security_logger.setLevel(logging.WARNING)
    
    def load_default_templates(self):
        """Charge les templates de paquets par défaut"""
        # Template de test simple
        test_template = PacketTemplate(
            id="test_ping",
            name="Test Ping",
            packet_type=PacketType.ICMP,
            payload=b"test_ping_payload",
            description="Paquet de test ICMP",
            rate_limit_per_second=5
        )
        self.packet_templates[test_template.id] = test_template
        
        # Template DOFUS heartbeat (simulé)
        dofus_heartbeat = PacketTemplate(
            id="dofus_heartbeat",
            name="DOFUS Heartbeat",
            packet_type=PacketType.DOFUS_GAME,
            payload=b'\x00\x01\x00\x04ping',  # Format simulé
            description="Message de maintien de connexion DOFUS",
            headers={'tcp_flags': 'PA'},
            rate_limit_per_second=1
        )
        self.packet_templates[dofus_heartbeat.id] = dofus_heartbeat
        
        # Template UDP test
        udp_test = PacketTemplate(
            id="udp_test",
            name="Test UDP",
            packet_type=PacketType.UDP,
            payload=b"udp_test_message",
            description="Paquet de test UDP",
            rate_limit_per_second=10
        )
        self.packet_templates[udp_test.id] = udp_test
        
        self.logger.info(f"Chargé {len(self.packet_templates)} templates par défaut")
    
    def add_template(self, template: PacketTemplate) -> bool:
        """Ajoute un template de paquet"""
        # Validation du template
        is_valid, error_msg = template.validate_payload()
        if not is_valid:
            self.logger.error(f"Template invalide {template.id}: {error_msg}")
            return False
        
        # Validation sécurité
        is_secure, security_msg = self.security_validator.validate_payload(
            template.payload, template.packet_type
        )
        if not is_secure:
            self.logger.error(f"Template non sécurisé {template.id}: {security_msg}")
            return False
        
        self.packet_templates[template.id] = template
        self.logger.info(f"Template ajouté: {template.id}")
        return True
    
    def add_injection_callback(self, callback: Callable[[str, InjectionResult, Dict], None]):
        """Ajoute un callback pour les injections"""
        self.injection_callbacks.append(callback)
    
    def add_job_callback(self, callback: Callable[[InjectionJob], None]):
        """Ajoute un callback pour les jobs"""
        self.job_callbacks.append(callback)
    
    def create_injection_job(self,
                           target_host: str,
                           target_port: int,
                           template_id: str,
                           count: int = 1,
                           interval: float = 1.0,
                           timeout: Optional[float] = None) -> Optional[str]:
        """Crée un job d'injection"""
        try:
            # Vérifier que le template existe
            if template_id not in self.packet_templates:
                self.logger.error(f"Template introuvable: {template_id}")
                return None
            
            template = self.packet_templates[template_id]
            
            # Créer la cible
            target = InjectionTarget(
                host=target_host,
                port=target_port,
                protocol=template.packet_type.value
            )
            
            # Validation sécurité de la cible
            is_valid, error_msg = self.security_validator.validate_target(target)
            if not is_valid:
                self.logger.error(f"Cible invalide: {error_msg}")
                return None
            
            # Créer le job
            job_id = f"job_{int(time.time() * 1000)}_{hash((target_host, target_port, template_id)) & 0xFFFF:04x}"
            job = InjectionJob(
                id=job_id,
                target=target,
                template=template,
                count=count,
                interval=interval,
                timeout=timeout
            )
            
            # Ajouter à la file
            self.job_queue.put(job)
            self.active_jobs[job_id] = job
            self.stats['jobs_created'] += 1
            
            self.logger.info(
                f"Job créé: {job_id} - {count} paquets vers {target_host}:{target_port}"
            )
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Erreur création job: {e}")
            return None
    
    def start_injector(self):
        """Démarre le système d'injection"""
        if self.running:
            self.logger.warning("Injecteur déjà en cours")
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        self.logger.info(f"Démarrage injecteur - Mode: {self.mode.value}")
        
        if self.mode == InjectionMode.SIMULATION:
            self.logger.info("MODE SIMULATION - Aucun paquet ne sera réellement envoyé")
        
        # Démarrer les workers
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        self.logger.info(f"Démarré {self.num_workers} workers d'injection")
    
    def _worker_loop(self, worker_id: int):
        """Boucle de traitement des workers"""
        self.logger.info(f"Worker {worker_id} démarré")
        
        try:
            while self.running:
                try:
                    # Récupérer un job (timeout pour vérification périodique)
                    job = self.job_queue.get(timeout=1.0)
                    
                    if job:
                        self._process_job(job, worker_id)
                        self.job_queue.task_done()
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Erreur worker {worker_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erreur critique worker {worker_id}: {e}")
        
        self.logger.info(f"Worker {worker_id} arrêté")
    
    def _process_job(self, job: InjectionJob, worker_id: int):
        """Traite un job d'injection"""
        job.started_time = time.time()
        job.status = "running"
        
        self.logger.info(f"Worker {worker_id} traite job {job.id}")
        
        try:
            for i in range(job.count):
                if not self.running:
                    job.status = "cancelled"
                    break
                
                # Vérifier le timeout
                if job.timeout and (time.time() - job.started_time) > job.timeout:
                    job.status = "timeout"
                    job.error_message = "Timeout atteint"
                    break
                
                # Vérifier rate limiting
                if not self.rate_limiter.can_inject():
                    result = {
                        'packet_index': i,
                        'timestamp': time.time(),
                        'result': InjectionResult.RATE_LIMITED.value,
                        'message': "Rate limit atteint"
                    }
                    job.results.append(result)
                    self.stats['packets_blocked'] += 1
                    
                    # Attendre avant de réessayer
                    time.sleep(0.1)
                    continue
                
                # Injecter le paquet
                result = self._inject_packet(job)
                job.results.append(result)
                
                # Callbacks
                for callback in self.injection_callbacks:
                    try:
                        callback(job.id, InjectionResult(result['result']), result)
                    except Exception as e:
                        self.logger.error(f"Erreur callback injection: {e}")
                
                # Attendre l'intervalle
                if i < job.count - 1:  # Pas d'attente après le dernier paquet
                    time.sleep(job.interval)
            
            job.completed_time = time.time()
            if job.status == "running":
                job.status = "completed"
            
            # Déplacer vers les jobs complétés
            self.completed_jobs[job.id] = job
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
            self.stats['jobs_completed'] += 1
            
            # Callback job completé
            for callback in self.job_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    self.logger.error(f"Erreur callback job: {e}")
            
            self.logger.info(
                f"Job {job.id} complété - {len(job.results)} paquets - "
                f"Taux de succès: {job.success_rate:.1%}"
            )
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_time = time.time()
            
            self.completed_jobs[job.id] = job
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
            self.logger.error(f"Erreur traitement job {job.id}: {e}")
    
    def _inject_packet(self, job: InjectionJob) -> Dict[str, Any]:
        """Injecte un paquet individuel"""
        start_time = time.time()
        
        result = {
            'timestamp': start_time,
            'job_id': job.id,
            'target': f"{job.target.host}:{job.target.port}",
            'template_id': job.template.id,
            'result': InjectionResult.FAILED.value,
            'message': '',
            'processing_time': 0.0
        }
        
        try:
            # Mode simulation
            if self.mode == InjectionMode.SIMULATION:
                # Simuler un délai d'envoi
                time.sleep(0.001)
                result['result'] = InjectionResult.SIMULATED.value
                result['message'] = 'Paquet simulé avec succès'
                self.stats['packets_simulated'] += 1
                
                self.logger.debug(f"Paquet simulé: {job.template.id} -> {job.target.host}:{job.target.port}")
                
            else:
                # Injection réelle
                if not SCAPY_AVAILABLE:
                    result['result'] = InjectionResult.FAILED.value
                    result['message'] = 'Scapy non disponible'
                    return result
                
                # Construire le paquet
                packet = None
                if job.template.packet_type in [PacketType.DOFUS_GAME, PacketType.DOFUS_AUTH]:
                    packet = self.packet_builder.build_dofus_packet(job.target, job.template)
                elif job.template.packet_type == PacketType.TCP:
                    packet = self.packet_builder.build_tcp_packet(job.target, job.template.payload)
                elif job.template.packet_type == PacketType.UDP:
                    packet = self.packet_builder.build_udp_packet(job.target, job.template.payload)
                
                if not packet:
                    result['result'] = InjectionResult.FAILED.value
                    result['message'] = 'Échec construction paquet'
                    return result
                
                # Envoyer le paquet
                try:
                    send(packet, verbose=False)
                    result['result'] = InjectionResult.SUCCESS.value
                    result['message'] = 'Paquet envoyé avec succès'
                    self.stats['packets_sent'] += 1
                    
                    self.logger.debug(f"Paquet envoyé: {job.template.id} -> {job.target.host}:{job.target.port}")
                    
                except Scapy_Exception as e:
                    result['result'] = InjectionResult.FAILED.value
                    result['message'] = f'Erreur Scapy: {str(e)}'
                    
                except Exception as e:
                    result['result'] = InjectionResult.FAILED.value
                    result['message'] = f'Erreur envoi: {str(e)}'
            
        except Exception as e:
            result['result'] = InjectionResult.FAILED.value
            result['message'] = f'Erreur injection: {str(e)}'
            self.logger.error(f"Erreur injection paquet: {e}")
        
        finally:
            result['processing_time'] = time.time() - start_time
        
        return result
    
    def stop_injector(self):
        """Arrête le système d'injection"""
        if not self.running:
            return
        
        self.logger.info("Arrêt de l'injecteur demandé")
        self.running = False
        
        # Attendre que tous les workers se terminent
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        # Statistiques finales
        uptime = time.time() - (self.stats['start_time'] or time.time())
        self.logger.info(
            f"Injecteur arrêté - Uptime: {uptime:.1f}s - "
            f"Jobs: {self.stats['jobs_completed']}/{self.stats['jobs_created']} - "
            f"Paquets: {self.stats['packets_sent']} envoyés, "
            f"{self.stats['packets_simulated']} simulés, "
            f"{self.stats['packets_blocked']} bloqués"
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut d'un job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id].to_dict()
        return None
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """Liste les templates disponibles"""
        return [template.to_dict() for template in self.packet_templates.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
        
        stats['active_jobs'] = len(self.active_jobs)
        stats['completed_jobs'] = len(self.completed_jobs)
        stats['templates_available'] = len(self.packet_templates)
        stats['rate_limiter'] = self.rate_limiter.get_stats()
        stats['mode'] = self.mode.value
        
        return stats
    
    def is_running(self) -> bool:
        """Vérifie si l'injecteur est actif"""
        return self.running


# Callbacks d'exemple
def log_injection_results(job_id: str, result: InjectionResult, details: Dict[str, Any]):
    """Callback pour logger les résultats d'injection"""
    if result == InjectionResult.SUCCESS:
        print(f"[INJECTION] {job_id}: Succès vers {details.get('target')}")
    elif result == InjectionResult.FAILED:
        print(f"[INJECTION] {job_id}: Échec - {details.get('message')}")

def monitor_job_completion(job: InjectionJob):
    """Callback pour surveiller la completion des jobs"""
    print(f"[JOB] {job.id} terminé - Statut: {job.status} - "
          f"Taux succès: {job.success_rate:.1%}")

def security_alert_on_block(job_id: str, result: InjectionResult, details: Dict[str, Any]):
    """Callback d'alerte sécurité pour les injections bloquées"""
    if result in [InjectionResult.BLOCKED, InjectionResult.VALIDATION_ERROR]:
        print(f"[SÉCURITÉ] Injection bloquée: {job_id} - {details.get('message')}")


def main():
    """Fonction principale pour test du module"""
    print("=== Test du Module PacketInjectorSafe ===")
    
    # Vérification des dépendances
    print(f"Scapy disponible: {SCAPY_AVAILABLE}")
    print(f"psutil disponible: {PSUTIL_AVAILABLE}")
    
    # Création de l'injecteur en mode simulation
    injector = PacketInjectorSafe(
        mode=InjectionMode.SIMULATION,
        max_rate_per_second=5
    )
    
    # Ajout des callbacks
    injector.add_injection_callback(log_injection_results)
    injector.add_injection_callback(security_alert_on_block)
    injector.add_job_callback(monitor_job_completion)
    
    try:
        # Démarrage de l'injecteur
        injector.start_injector()
        
        print("\nTemplates disponibles:")
        for template in injector.list_templates():
            print(f"- {template['id']}: {template['name']} ({template['packet_type']})")
        
        # Création de jobs de test
        print("\nCréation de jobs de test...")
        
        job1 = injector.create_injection_job(
            target_host="127.0.0.1",
            target_port=8080,
            template_id="test_ping",
            count=3,
            interval=1.0
        )
        
        job2 = injector.create_injection_job(
            target_host="127.0.0.1",
            target_port=5555,
            template_id="dofus_heartbeat",
            count=2,
            interval=2.0
        )
        
        if job1:
            print(f"Job créé: {job1}")
        if job2:
            print(f"Job créé: {job2}")
        
        # Monitoring des jobs
        print("\nMonitoring des jobs...")
        active_jobs = set()
        
        while True:
            time.sleep(2)
            
            # Vérifier les jobs actifs
            current_active = set(injector.active_jobs.keys())
            new_jobs = current_active - active_jobs
            completed_jobs = active_jobs - current_active
            
            for job_id in new_jobs:
                print(f"Job démarré: {job_id}")
            
            for job_id in completed_jobs:
                status = injector.get_job_status(job_id)
                if status:
                    print(f"Job terminé: {job_id} - {status['status']}")
            
            active_jobs = current_active
            
            # Arrêter quand tous les jobs sont terminés
            if not active_jobs and injector.stats['jobs_completed'] > 0:
                break
        
        # Statistiques finales
        print("\n=== Statistiques Finales ===")
        stats = injector.get_statistics()
        for key, value in stats.items():
            if key != 'rate_limiter':
                print(f"{key}: {value}")
        
        # Statistiques rate limiter
        print("\nRate Limiter:")
        for key, value in stats['rate_limiter'].items():
            print(f"  {key}: {value}")
            
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    finally:
        injector.stop_injector()


if __name__ == "__main__":
    main()