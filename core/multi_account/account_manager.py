"""
Gestionnaire centralisé des comptes DOFUS.

Ce module fournit un gestionnaire centralisé pour tous les comptes DOFUS,
incluant le stockage sécurisé des credentials, la gestion des sessions,
et la coordination des actions multi-comptes.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import sqlite3
import threading
from enum import Enum

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    """Statuts possibles d'un compte."""
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ONLINE = "online"
    IN_COMBAT = "in_combat"
    FARMING = "farming"
    TRADING = "trading"
    ERROR = "error"
    BANNED = "banned"

class AccountRole(Enum):
    """Rôles possibles dans un groupe."""
    LEADER = "leader"
    FOLLOWER = "follower"
    INDEPENDENT = "independent"

@dataclass
class AccountCredentials:
    """Credentials d'un compte DOFUS."""
    username: str
    password: str  # Sera chiffré
    character_name: str
    server: str

@dataclass
class AccountConfig:
    """Configuration d'un compte."""
    max_session_duration: int = 14400  # 4 heures en secondes
    auto_reconnect: bool = True
    farming_priority: List[str] = None
    combat_role: str = "dps"
    rest_intervals: Dict[str, int] = None
    
    def __post_init__(self):
        if self.farming_priority is None:
            self.farming_priority = []
        if self.rest_intervals is None:
            self.rest_intervals = {"min": 300, "max": 900}  # 5-15 minutes

@dataclass
class Account:
    """Représentation d'un compte DOFUS."""
    id: str
    credentials: AccountCredentials
    config: AccountConfig
    status: AccountStatus = AccountStatus.INACTIVE
    role: AccountRole = AccountRole.INDEPENDENT
    group_id: Optional[str] = None
    last_login: Optional[datetime] = None
    session_start: Optional[datetime] = None
    process_id: Optional[int] = None
    window_handle: Optional[int] = None
    current_map: Optional[str] = None
    level: int = 1
    kamas: int = 0
    stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stats is None:
            self.stats = {
                "total_playtime": 0,
                "sessions_count": 0,
                "kamas_earned": 0,
                "exp_gained": 0,
                "fights_won": 0,
                "fights_lost": 0
            }

class CryptoManager:
    """Gestionnaire de chiffrement pour les credentials."""
    
    def __init__(self, master_password: str):
        """
        Initialise le gestionnaire de chiffrement.
        
        Args:
            master_password: Mot de passe maître pour le chiffrement
        """
        self.salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Chiffre une chaîne de caractères."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Déchiffre une chaîne de caractères."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

class AccountManager:
    """
    Gestionnaire centralisé des comptes DOFUS.
    
    Gère tous les aspects des comptes multiples incluant:
    - Stockage sécurisé des credentials
    - Gestion des sessions
    - Coordination des groupes
    - Surveillance du statut
    """
    
    def __init__(self, master_password: str, max_concurrent_accounts: int = 8):
        """
        Initialise le gestionnaire de comptes.
        
        Args:
            master_password: Mot de passe maître pour le chiffrement
            max_concurrent_accounts: Nombre maximum de comptes simultanés
        """
        self.crypto = CryptoManager(master_password)
        self.max_concurrent_accounts = max_concurrent_accounts
        
        # État en mémoire
        self._accounts: Dict[str, Account] = {}
        self._groups: Dict[str, List[str]] = {}
        self._active_sessions: Dict[str, datetime] = {}
        
        # Synchronisation
        self._lock = threading.RLock()
        
        logger.info(f"AccountManager initialisé avec max {max_concurrent_accounts} comptes")
    
    def add_account(
        self, 
        username: str, 
        password: str, 
        character_name: str, 
        server: str,
        config: Optional[AccountConfig] = None
    ) -> str:
        """
        Ajoute un nouveau compte.
        
        Args:
            username: Nom d'utilisateur DOFUS
            password: Mot de passe (sera chiffré)
            character_name: Nom du personnage
            server: Serveur DOFUS
            config: Configuration personnalisée
            
        Returns:
            ID du compte créé
        """
        with self._lock:
            # Génération d'un ID unique
            account_id = hashlib.md5(f"{username}_{character_name}_{server}".encode()).hexdigest()
            
            # Vérification d'unicité
            if account_id in self._accounts:
                raise ValueError(f"Compte déjà existant: {username}")
            
            # Chiffrement du mot de passe
            encrypted_password = self.crypto.encrypt(password)
            
            # Création des credentials
            credentials = AccountCredentials(
                username=username,
                password=encrypted_password,
                character_name=character_name,
                server=server
            )
            
            # Configuration par défaut
            if config is None:
                config = AccountConfig()
            
            # Création du compte
            account = Account(
                id=account_id,
                credentials=credentials,
                config=config
            )
            
            self._accounts[account_id] = account
            logger.info(f"Compte ajouté: {username} ({character_name})")
            return account_id
    
    def get_account(self, account_id: str) -> Optional[Account]:
        """
        Récupère un compte par son ID.
        
        Args:
            account_id: ID du compte
            
        Returns:
            Compte ou None si non trouvé
        """
        return self._accounts.get(account_id)
    
    def get_all_accounts(self) -> List[Account]:
        """
        Récupère tous les comptes.
        
        Returns:
            Liste de tous les comptes
        """
        with self._lock:
            return list(self._accounts.values())
    
    def get_active_accounts(self) -> List[Account]:
        """
        Récupère les comptes actuellement actifs.
        
        Returns:
            Liste des comptes avec un statut actif
        """
        active_statuses = {AccountStatus.ONLINE, AccountStatus.IN_COMBAT, 
                          AccountStatus.FARMING, AccountStatus.TRADING}
        return [acc for acc in self._accounts.values() if acc.status in active_statuses]
    
    def update_account_status(self, account_id: str, status: AccountStatus):
        """
        Met à jour le statut d'un compte.
        
        Args:
            account_id: ID du compte
            status: Nouveau statut
        """
        with self._lock:
            if account_id in self._accounts:
                account = self._accounts[account_id]
                old_status = account.status
                account.status = status
                
                # Gestion des transitions spéciales
                if old_status == AccountStatus.INACTIVE and status != AccountStatus.INACTIVE:
                    account.session_start = datetime.now()
                    self._active_sessions[account_id] = account.session_start
                elif old_status != AccountStatus.INACTIVE and status == AccountStatus.INACTIVE:
                    if account_id in self._active_sessions:
                        del self._active_sessions[account_id]
                    account.session_start = None
                
                logger.info(f"Statut mis à jour pour {account.credentials.username}: {old_status.value} -> {status.value}")
    
    def can_start_new_session(self) -> bool:
        """
        Vérifie si une nouvelle session peut être démarrée.
        
        Returns:
            True si possible de démarrer une nouvelle session
        """
        active_count = len(self.get_active_accounts())
        return active_count < self.max_concurrent_accounts
    
    def get_account_credentials(self, account_id: str) -> Optional[Dict[str, str]]:
        """
        Récupère les credentials déchiffrés d'un compte.
        
        Args:
            account_id: ID du compte
            
        Returns:
            Dictionnaire avec username et password déchiffré
        """
        account = self.get_account(account_id)
        if not account:
            return None
        
        try:
            decrypted_password = self.crypto.decrypt(account.credentials.password)
            return {
                "username": account.credentials.username,
                "password": decrypted_password,
                "character_name": account.credentials.character_name,
                "server": account.credentials.server
            }
        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement des credentials: {e}")
            return None
    
    def create_group(self, group_name: str, leader_id: str, member_ids: List[str]) -> str:
        """
        Crée un nouveau groupe de comptes.
        
        Args:
            group_name: Nom du groupe
            leader_id: ID du compte leader
            member_ids: IDs des comptes membres
            
        Returns:
            ID du groupe créé
        """
        with self._lock:
            # Génération de l'ID du groupe
            group_id = hashlib.md5(f"{group_name}_{leader_id}_{datetime.now()}".encode()).hexdigest()
            
            # Validation des comptes
            all_members = [leader_id] + member_ids
            for member_id in all_members:
                if member_id not in self._accounts:
                    raise ValueError(f"Compte inexistant: {member_id}")
            
            # Configuration des rôles
            self._accounts[leader_id].role = AccountRole.LEADER
            self._accounts[leader_id].group_id = group_id
            
            for member_id in member_ids:
                self._accounts[member_id].role = AccountRole.FOLLOWER
                self._accounts[member_id].group_id = group_id
            
            # Enregistrement du groupe
            self._groups[group_id] = all_members
            
            logger.info(f"Groupe créé: {group_name} avec {len(all_members)} membres")
            return group_id
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Récupère les statistiques des sessions actives.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        active_accounts = self.get_active_accounts()
        
        stats = {
            "total_accounts": len(self._accounts),
            "active_accounts": len(active_accounts),
            "inactive_accounts": len(self._accounts) - len(active_accounts),
            "max_concurrent": self.max_concurrent_accounts,
            "groups_count": len(self._groups),
            "total_session_time": 0,
            "account_details": []
        }
        
        for account in active_accounts:
            if account.session_start:
                session_duration = (datetime.now() - account.session_start).total_seconds()
                stats["total_session_time"] += session_duration
                
                stats["account_details"].append({
                    "id": account.id,
                    "username": account.credentials.username,
                    "character_name": account.credentials.character_name,
                    "status": account.status.value,
                    "session_duration": int(session_duration),
                    "level": account.level,
                    "kamas": account.kamas
                })
        
        return stats