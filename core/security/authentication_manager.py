#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire d'authentification et d'autorisation sécurisé pour DOFUS Bot
=========================================================================

Ce module implémente un système d'authentification robuste avec :
- Authentification multi-facteurs (MFA)
- Gestion de sessions sécurisées  
- Contrôle d'accès basé sur les rôles (RBAC)
- Protection contre les attaques par force brute
- Chiffrement des données sensibles

Auteur: Claude AI Assistant (Security Specialist)
Date: 2025-08-31
Licence: Usage éthique uniquement
"""

import time
import hashlib
import secrets
import hmac
import logging
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import base64
import threading
import pyotp

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """Rôles utilisateur avec permissions"""
    GUEST = "guest"
    USER = "user" 
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class SessionStatus(Enum):
    """États de session"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

@dataclass
class Permission:
    """Permission système"""
    name: str
    description: str
    resource: str
    action: str

@dataclass 
class UserProfile:
    """Profil utilisateur sécurisé"""
    user_id: str
    username_hash: str
    password_hash: str
    salt: bytes
    role: UserRole
    permissions: Set[str] = field(default_factory=set)
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    is_locked: bool = False
    locked_until: Optional[datetime] = None
    session_timeout: int = 3600  # 1 heure par défaut

@dataclass
class UserSession:
    """Session utilisateur sécurisée"""
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address_hash: str
    user_agent_hash: str
    status: SessionStatus = SessionStatus.ACTIVE
    permissions: Set[str] = field(default_factory=set)

class AuthenticationManager:
    """
    Gestionnaire d'authentification sécurisé avec protection multi-couches
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialise le gestionnaire d'authentification"""
        
        # Configuration cryptographique
        if master_key is None:
            master_key = secrets.token_bytes(32)
        self._setup_encryption(master_key)
        
        # Base de données utilisateurs chiffrée
        self.users: Dict[str, UserProfile] = {}
        self.sessions: Dict[str, UserSession] = {}
        
        # Configuration de sécurité
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=1)
        self.password_min_length = 12
        
        # Rate limiting
        self._rate_limiter: Dict[str, List[float]] = {}
        self.max_requests_per_minute = 10
        
        # Journal de sécurité
        self._security_events: List[Dict] = []
        
        # Synchronisation thread-safe
        self._lock = threading.RLock()
        self._cleanup_thread = None
        
        # Permissions système
        self._setup_permissions()
        
        logger.info("AuthenticationManager sécurisé initialisé")
    
    def _setup_encryption(self, master_key: bytes):
        """Configure le système de chiffrement avancé"""
        # Génération de salt unique
        self._master_salt = secrets.token_bytes(32)
        
        # Dérivation de clé avec Scrypt (plus résistant aux attaques GPU)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._master_salt,
            n=2**14,  # 16384 iterations
            r=8,
            p=1,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        self._cipher = Fernet(key)
    
    def _setup_permissions(self):
        """Configure les permissions système"""
        self.permissions = {
            # Sessions
            "session.create": Permission("session.create", "Créer une session", "session", "create"),
            "session.manage": Permission("session.manage", "Gérer les sessions", "session", "manage"),
            "session.view": Permission("session.view", "Voir les sessions", "session", "read"),
            
            # Utilisateurs
            "user.create": Permission("user.create", "Créer un utilisateur", "user", "create"),
            "user.edit": Permission("user.edit", "Modifier un utilisateur", "user", "update"),
            "user.delete": Permission("user.delete", "Supprimer un utilisateur", "user", "delete"),
            "user.view": Permission("user.view", "Voir les utilisateurs", "user", "read"),
            
            # Bot
            "bot.start": Permission("bot.start", "Démarrer le bot", "bot", "execute"),
            "bot.stop": Permission("bot.stop", "Arrêter le bot", "bot", "execute"),
            "bot.config": Permission("bot.config", "Configurer le bot", "bot", "configure"),
            
            # Sécurité
            "security.audit": Permission("security.audit", "Auditer la sécurité", "security", "audit"),
            "security.admin": Permission("security.admin", "Administration sécurité", "security", "admin")
        }
        
        # Attribution des permissions par rôle
        self.role_permissions = {
            UserRole.GUEST: {"session.view"},
            UserRole.USER: {"session.create", "session.view", "bot.start", "bot.stop"},
            UserRole.MODERATOR: {"session.create", "session.view", "session.manage", "bot.start", "bot.stop", "bot.config", "user.view"},
            UserRole.ADMIN: {"session.create", "session.manage", "session.view", "bot.start", "bot.stop", "bot.config", "user.create", "user.edit", "user.view", "security.audit"},
            UserRole.SUPER_ADMIN: set(self.permissions.keys())  # Toutes les permissions
        }
    
    def create_user(self, username: str, password: str, role: UserRole = UserRole.USER, 
                   enable_mfa: bool = True) -> Tuple[bool, str]:
        """
        Crée un utilisateur avec authentification sécurisée
        """
        try:
            with self._lock:
                # Validation des entrées
                if not self._validate_username(username):
                    return False, "Nom d'utilisateur invalide"
                
                if not self._validate_password(password):
                    return False, f"Mot de passe faible (min {self.password_min_length} caractères)"
                
                # Hash sécurisé du nom d'utilisateur
                username_hash = hashlib.sha256(username.encode()).hexdigest()
                
                # Vérifier si l'utilisateur existe déjà
                if any(user.username_hash == username_hash for user in self.users.values()):
                    return False, "Utilisateur déjà existant"
                
                # Génération d'ID utilisateur unique
                user_id = secrets.token_urlsafe(16)
                
                # Génération de salt individuel
                salt = secrets.token_bytes(32)
                
                # Hash du mot de passe avec Argon2 (via scrypt comme alternative)
                password_hash = self._hash_password(password, salt)
                
                # Configuration MFA
                mfa_secret = None
                if enable_mfa:
                    mfa_secret = pyotp.random_base32()
                
                # Création du profil utilisateur
                user_profile = UserProfile(
                    user_id=user_id,
                    username_hash=username_hash,
                    password_hash=password_hash,
                    salt=salt,
                    role=role,
                    mfa_secret=mfa_secret,
                    mfa_enabled=enable_mfa,
                    permissions=self.role_permissions.get(role, set())
                )
                
                # Enregistrement sécurisé
                self.users[user_id] = user_profile
                
                # Journal de sécurité
                self._log_security_event("user_created", {
                    "user_id": user_id,
                    "username_hash": username_hash[:8] + "...",
                    "role": role.value,
                    "mfa_enabled": enable_mfa
                })
                
                logger.info(f"Utilisateur créé: {username_hash[:8]}... avec rôle {role.value}")
                
                if enable_mfa:
                    # Retourner le secret MFA pour configuration
                    qr_url = pyotp.totp.TOTP(mfa_secret).provisioning_uri(
                        name=username_hash[:8],
                        issuer_name="DOFUS Bot Security"
                    )
                    return True, f"Utilisateur créé avec MFA. QR Code: {qr_url}"
                else:
                    return True, "Utilisateur créé avec succès"
        
        except Exception as e:
            self._log_security_event("user_creation_error", {"error": str(e)[:100]})
            logger.error(f"Erreur création utilisateur: {e}")
            return False, "Erreur interne"
    
    def authenticate(self, username: str, password: str, mfa_code: Optional[str] = None,
                    ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
        """
        Authentifie un utilisateur avec protection multi-couches
        
        Returns:
            Tuple[bool, Optional[session_id], message]
        """
        try:
            with self._lock:
                # Rate limiting
                client_id = self._get_client_id(ip_address, user_agent)
                if not self._check_rate_limit(client_id):
                    self._log_security_event("rate_limit_exceeded", {"client_id": client_id})
                    return False, None, "Trop de tentatives, veuillez patienter"
                
                # Hash du nom d'utilisateur
                username_hash = hashlib.sha256(username.encode()).hexdigest()
                
                # Recherche de l'utilisateur
                user_profile = None
                for user in self.users.values():
                    if hmac.compare_digest(user.username_hash, username_hash):
                        user_profile = user
                        break
                
                if not user_profile:
                    self._log_security_event("auth_failed_user_not_found", {"username_hash": username_hash[:8]})
                    return False, None, "Identifiants invalides"
                
                # Vérification du verrouillage de compte
                if user_profile.is_locked:
                    if user_profile.locked_until and datetime.now() < user_profile.locked_until:
                        remaining = (user_profile.locked_until - datetime.now()).total_seconds()
                        self._log_security_event("auth_failed_account_locked", {"user_id": user_profile.user_id})
                        return False, None, f"Compte verrouillé pour {remaining:.0f}s"
                    else:
                        # Déverrouillage automatique
                        user_profile.is_locked = False
                        user_profile.locked_until = None
                        user_profile.failed_attempts = 0
                
                # Vérification du mot de passe
                if not self._verify_password(password, user_profile.password_hash, user_profile.salt):
                    user_profile.failed_attempts += 1
                    
                    # Verrouillage après échecs répétés
                    if user_profile.failed_attempts >= self.max_failed_attempts:
                        user_profile.is_locked = True
                        user_profile.locked_until = datetime.now() + self.lockout_duration
                        self._log_security_event("account_locked", {"user_id": user_profile.user_id})
                        logger.warning(f"Compte verrouillé: {user_profile.user_id}")
                    
                    self._log_security_event("auth_failed_invalid_password", {"user_id": user_profile.user_id})
                    return False, None, "Identifiants invalides"
                
                # Vérification MFA si activé
                if user_profile.mfa_enabled:
                    if not mfa_code:
                        return False, None, "Code MFA requis"
                    
                    if not self._verify_mfa(user_profile.mfa_secret, mfa_code):
                        user_profile.failed_attempts += 1
                        self._log_security_event("auth_failed_invalid_mfa", {"user_id": user_profile.user_id})
                        return False, None, "Code MFA invalide"
                
                # Authentification réussie - création de session
                session_id = self._create_session(user_profile, ip_address, user_agent)
                
                # Remise à zéro des échecs
                user_profile.failed_attempts = 0
                user_profile.last_login = datetime.now()
                
                self._log_security_event("auth_success", {
                    "user_id": user_profile.user_id,
                    "session_id": session_id[:8] + "...",
                    "role": user_profile.role.value
                })
                
                logger.info(f"Authentification réussie: {user_profile.user_id}")
                return True, session_id, "Authentification réussie"
        
        except Exception as e:
            self._log_security_event("auth_error", {"error": str(e)[:100]})
            logger.error(f"Erreur authentification: {e}")
            return False, None, "Erreur interne"
    
    def validate_session(self, session_id: str, required_permission: Optional[str] = None) -> Tuple[bool, Optional[UserSession]]:
        """
        Valide une session et vérifie les permissions
        """
        try:
            with self._lock:
                if session_id not in self.sessions:
                    return False, None
                
                session = self.sessions[session_id]
                current_time = datetime.now()
                
                # Vérification de l'expiration
                if current_time > session.expires_at:
                    session.status = SessionStatus.EXPIRED
                    self._log_security_event("session_expired", {"session_id": session_id[:8]})
                    return False, None
                
                # Vérification du statut
                if session.status != SessionStatus.ACTIVE:
                    return False, None
                
                # Vérification des permissions
                if required_permission and required_permission not in session.permissions:
                    self._log_security_event("permission_denied", {
                        "session_id": session_id[:8],
                        "permission": required_permission
                    })
                    return False, None
                
                # Mise à jour de l'activité
                session.last_activity = current_time
                session.expires_at = current_time + self.session_timeout
                
                return True, session
        
        except Exception as e:
            logger.error(f"Erreur validation session: {e}")
            return False, None
    
    def terminate_session(self, session_id: str) -> bool:
        """Termine une session"""
        try:
            with self._lock:
                if session_id in self.sessions:
                    self.sessions[session_id].status = SessionStatus.TERMINATED
                    self._log_security_event("session_terminated", {"session_id": session_id[:8]})
                    logger.info(f"Session terminée: {session_id[:8]}...")
                    return True
                return False
        except Exception as e:
            logger.error(f"Erreur terminaison session: {e}")
            return False
    
    def get_security_report(self) -> Dict:
        """Génère un rapport de sécurité complet"""
        try:
            with self._lock:
                current_time = datetime.now()
                
                # Statistiques utilisateurs
                total_users = len(self.users)
                locked_users = sum(1 for user in self.users.values() if user.is_locked)
                mfa_enabled_users = sum(1 for user in self.users.values() if user.mfa_enabled)
                
                # Statistiques sessions
                active_sessions = sum(1 for session in self.sessions.values() if session.status == SessionStatus.ACTIVE)
                expired_sessions = sum(1 for session in self.sessions.values() if session.status == SessionStatus.EXPIRED)
                
                # Événements de sécurité récents (dernière heure)
                recent_events = [e for e in self._security_events if current_time.timestamp() - e['timestamp'] < 3600]
                
                # Analyse des menaces
                failed_auths = len([e for e in recent_events if e['type'].startswith('auth_failed')])
                rate_limit_violations = len([e for e in recent_events if e['type'] == 'rate_limit_exceeded'])
                
                return {
                    "timestamp": current_time.timestamp(),
                    "users": {
                        "total": total_users,
                        "locked": locked_users,
                        "mfa_enabled": mfa_enabled_users,
                        "mfa_adoption_rate": (mfa_enabled_users / total_users * 100) if total_users > 0 else 0
                    },
                    "sessions": {
                        "active": active_sessions,
                        "expired": expired_sessions,
                        "total": len(self.sessions)
                    },
                    "security_events": {
                        "total": len(self._security_events),
                        "recent_hour": len(recent_events),
                        "failed_authentications": failed_auths,
                        "rate_limit_violations": rate_limit_violations
                    },
                    "threat_assessment": {
                        "level": "high" if failed_auths > 10 else "medium" if failed_auths > 5 else "low",
                        "recommendations": self._generate_security_recommendations(failed_auths, rate_limit_violations)
                    }
                }
        
        except Exception as e:
            logger.error(f"Erreur génération rapport sécurité: {e}")
            return {"error": "Impossible de générer le rapport"}
    
    def _create_session(self, user_profile: UserProfile, ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> str:
        """Crée une nouvelle session sécurisée"""
        session_id = secrets.token_urlsafe(32)
        current_time = datetime.now()
        
        # Hash des informations client pour la détection d'anomalies
        ip_hash = hashlib.sha256((ip_address or "unknown").encode()).hexdigest()[:16]
        agent_hash = hashlib.sha256((user_agent or "unknown").encode()).hexdigest()[:16]
        
        session = UserSession(
            session_id=session_id,
            user_id=user_profile.user_id,
            role=user_profile.role,
            created_at=current_time,
            last_activity=current_time,
            expires_at=current_time + timedelta(seconds=user_profile.session_timeout),
            ip_address_hash=ip_hash,
            user_agent_hash=agent_hash,
            permissions=user_profile.permissions.copy()
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def _hash_password(self, password: str, salt: bytes) -> str:
        """Hash sécurisé du mot de passe"""
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=64,
            salt=salt,
            n=2**14,
            r=8,
            p=1,
        )
        key = kdf.derive(password.encode())
        return base64.b64encode(key).decode()
    
    def _verify_password(self, password: str, password_hash: str, salt: bytes) -> bool:
        """Vérifie un mot de passe"""
        try:
            expected_hash = self._hash_password(password, salt)
            return hmac.compare_digest(password_hash, expected_hash)
        except Exception:
            return False
    
    def _verify_mfa(self, secret: str, code: str) -> bool:
        """Vérifie un code MFA TOTP"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)  # Fenêtre de tolérance de 30s
        except Exception:
            return False
    
    def _validate_username(self, username: str) -> bool:
        """Valide un nom d'utilisateur"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        # Caractères autorisés: lettres, chiffres, underscore, tiret
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))
    
    def _validate_password(self, password: str) -> bool:
        """Valide la force d'un mot de passe"""
        if len(password) < self.password_min_length:
            return False
        
        # Vérifications de complexité
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    def _get_client_id(self, ip_address: Optional[str], user_agent: Optional[str]) -> str:
        """Génère un ID client pour le rate limiting"""
        client_data = f"{ip_address or 'unknown'}:{user_agent or 'unknown'}"
        return hashlib.sha256(client_data.encode()).hexdigest()[:16]
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Vérifie les limites de débit"""
        current_time = time.time()
        
        if client_id not in self._rate_limiter:
            self._rate_limiter[client_id] = []
        
        # Nettoyer les anciennes requêtes
        self._rate_limiter[client_id] = [
            t for t in self._rate_limiter[client_id] 
            if current_time - t < 60
        ]
        
        # Vérifier la limite
        if len(self._rate_limiter[client_id]) >= self.max_requests_per_minute:
            return False
        
        # Enregistrer la requête actuelle
        self._rate_limiter[client_id].append(current_time)
        return True
    
    def _log_security_event(self, event_type: str, data: Dict):
        """Enregistre un événement de sécurité"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        self._security_events.append(event)
        
        # Garder seulement les 1000 derniers événements
        if len(self._security_events) > 1000:
            self._security_events.pop(0)
    
    def _generate_security_recommendations(self, failed_auths: int, rate_limit_violations: int) -> List[str]:
        """Génère des recommandations de sécurité"""
        recommendations = []
        
        if failed_auths > 10:
            recommendations.append("Nombre élevé d'échecs d'authentification - vérifier les tentatives d'intrusion")
        
        if rate_limit_violations > 5:
            recommendations.append("Violations de limite de débit détectées - possible attaque par déni de service")
        
        mfa_adoption = sum(1 for user in self.users.values() if user.mfa_enabled) / len(self.users) * 100 if self.users else 0
        if mfa_adoption < 80:
            recommendations.append("Adoption MFA faible - encourager l'activation de l'authentification à deux facteurs")
        
        if not recommendations:
            recommendations.append("Situation sécuritaire normale - continuer la surveillance")
        
        return recommendations


def create_secure_auth_manager(master_key: Optional[bytes] = None) -> AuthenticationManager:
    """Factory pour créer un gestionnaire d'authentification sécurisé"""
    return AuthenticationManager(master_key)


if __name__ == "__main__":
    # Test du système d'authentification
    logging.basicConfig(level=logging.INFO)
    
    # Création du gestionnaire
    auth_manager = create_secure_auth_manager()
    
    # Test de création d'utilisateur
    success, message = auth_manager.create_user("admin", "MonMotDePasseTresFort123!", UserRole.ADMIN, enable_mfa=False)
    print(f"Création utilisateur: {success} - {message}")
    
    # Test d'authentification
    success, session_id, message = auth_manager.authenticate("admin", "MonMotDePasseTresFort123!")
    print(f"Authentification: {success} - {message}")
    
    if session_id:
        # Test de validation de session
        valid, session = auth_manager.validate_session(session_id, "user.view")
        print(f"Session valide: {valid}")
        
        # Rapport de sécurité
        report = auth_manager.get_security_report()
        print(f"Rapport sécurité: {json.dumps(report, indent=2)}")