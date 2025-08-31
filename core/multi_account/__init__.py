"""
Système de gestion multi-comptes DOFUS.

Ce module fournit un système complet de gestion multi-comptes pour DOFUS,
incluant la gestion centralisée des comptes, des fenêtres, la synchronisation
des actions et la planification des sessions.

Classes principales:
    - AccountManager: Gestionnaire centralisé des comptes
    - WindowManager: Gestion des fenêtres multiples
    - AccountSynchronizer: Synchronisation des actions entre comptes
    - SessionScheduler: Planification automatique des sessions
    - MultiAccountMainWindow: Interface graphique complète

Usage basique:
    from core.multi_account import MultiAccountSystem
    
    system = MultiAccountSystem()
    system.start()
    
    # Ajouter un compte
    system.add_account("username", "password", "character", "server")
    
    # Lancer l'interface graphique
    system.launch_gui()
"""

from .account_manager import (
    AccountManager, Account, AccountStatus, AccountRole, AccountConfig,
    AccountCredentials, CryptoManager, AccountDatabase
)

from .window_manager import (
    WindowManager, WindowInfo, WindowState, WindowPosition,
    DisplayManager, ProcessManager
)

from .account_synchronizer import (
    AccountSynchronizer, SyncAction, SyncActionType, SyncStatus,
    SyncPriority, GroupCoordinator, IPCManager
)

from .session_scheduler import (
    SessionScheduler, SessionSchedule, SessionType, SessionStatus,
    RotationType, RotationPlan, SafetyManager
)

from .multi_account_gui import (
    MultiAccountMainWindow, AccountDialog, ScheduleDialog, UpdateThread
)

__version__ = "1.0.0"
__author__ = "DOFUS Bot Development Team"
__email__ = "dev@dofusbot.com"

__all__ = [
    # Account Management
    "AccountManager", "Account", "AccountStatus", "AccountRole", 
    "AccountConfig", "AccountCredentials", "CryptoManager", "AccountDatabase",
    
    # Window Management
    "WindowManager", "WindowInfo", "WindowState", "WindowPosition",
    "DisplayManager", "ProcessManager",
    
    # Synchronization
    "AccountSynchronizer", "SyncAction", "SyncActionType", "SyncStatus",
    "SyncPriority", "GroupCoordinator", "IPCManager",
    
    # Scheduling
    "SessionScheduler", "SessionSchedule", "SessionType", "SessionStatus",
    "RotationType", "RotationPlan", "SafetyManager",
    
    # GUI
    "MultiAccountMainWindow", "AccountDialog", "ScheduleDialog", "UpdateThread",
    
    # Main System
    "MultiAccountSystem"
]


class MultiAccountSystem:
    """
    Système principal de gestion multi-comptes.
    
    Cette classe orchestre tous les composants du système multi-comptes
    et fournit une interface simplifiée pour l'utilisation.
    """
    
    def __init__(self, master_password: str = "DofusBotMaster2024", max_accounts: int = 8):
        """
        Initialise le système multi-comptes.
        
        Args:
            master_password: Mot de passe maître pour le chiffrement
            max_accounts: Nombre maximum de comptes simultanés
        """
        self.master_password = master_password
        self.max_accounts = max_accounts
        
        # Initialisation des gestionnaires
        self.account_manager = None
        self.window_manager = None
        self.synchronizer = None
        self.session_scheduler = None
        
        self.running = False
    
    def start(self) -> bool:
        """
        Démarre tous les composants du système.
        
        Returns:
            True si le démarrage a réussi
        """
        try:
            # Initialisation des gestionnaires
            self.account_manager = AccountManager(self.master_password, self.max_accounts)
            self.window_manager = WindowManager(max_instances=self.max_accounts)
            self.synchronizer = AccountSynchronizer()
            self.session_scheduler = SessionScheduler()
            
            # Démarrage des services
            self.window_manager.start_monitoring()
            self.synchronizer.start()
            self.session_scheduler.start()
            
            self.running = True
            return True
            
        except Exception as e:
            print(f"Erreur lors du démarrage du système: {e}")
            return False
    
    def stop(self):
        """Arrête tous les composants du système."""
        if not self.running:
            return
        
        try:
            if self.window_manager:
                self.window_manager.stop_monitoring()
            
            if self.synchronizer:
                self.synchronizer.stop()
            
            if self.session_scheduler:
                self.session_scheduler.stop()
            
            self.running = False
            
        except Exception as e:
            print(f"Erreur lors de l'arrêt du système: {e}")
    
    def add_account(
        self, 
        username: str, 
        password: str, 
        character_name: str, 
        server: str,
        config: dict = None
    ) -> str:
        """
        Ajoute un nouveau compte au système.
        
        Args:
            username: Nom d'utilisateur DOFUS
            password: Mot de passe
            character_name: Nom du personnage
            server: Serveur DOFUS
            config: Configuration personnalisée
            
        Returns:
            ID du compte créé
        """
        if not self.account_manager:
            raise RuntimeError("Système non démarré")
        
        account_config = None
        if config:
            account_config = AccountConfig(**config)
        
        return self.account_manager.add_account(
            username, password, character_name, server, account_config
        )
    
    def launch_account(self, account_id: str) -> bool:
        """
        Lance une instance DOFUS pour un compte.
        
        Args:
            account_id: ID du compte à lancer
            
        Returns:
            True si le lancement a réussi
        """
        if not self.running:
            return False
        
        # Vérifier si le compte peut démarrer
        if not self.account_manager.can_start_new_session():
            return False
        
        # Lancer la fenêtre
        if not self.window_manager.launch_account_window(account_id):
            return False
        
        # Mettre à jour le statut
        self.account_manager.update_account_status(account_id, AccountStatus.CONNECTING)
        
        return True
    
    def create_group(self, name: str, leader_id: str, member_ids: List[str]) -> str:
        """
        Crée un groupe de comptes synchronisés.
        
        Args:
            name: Nom du groupe
            leader_id: ID du compte leader
            member_ids: Liste des IDs des membres
            
        Returns:
            ID du groupe créé
        """
        if not self.running:
            raise RuntimeError("Système non démarré")
        
        # Créer le groupe dans le gestionnaire de comptes
        group_id = self.account_manager.create_group(name, leader_id, member_ids)
        
        # Créer le groupe dans le synchroniseur
        self.synchronizer.create_group(group_id, leader_id, member_ids)
        
        return group_id
    
    def schedule_session(
        self, 
        account_id: str, 
        session_type: str,
        start_time: datetime,
        duration: timedelta,
        **kwargs
    ) -> str:
        """
        Planifie une session pour un compte.
        
        Args:
            account_id: ID du compte
            session_type: Type de session
            start_time: Heure de début
            duration: Durée de la session
            **kwargs: Paramètres additionnels
            
        Returns:
            ID de la planification
        """
        if not self.running:
            raise RuntimeError("Système non démarré")
        
        schedule = SessionSchedule(
            id=f"schedule_{account_id}_{int(start_time.timestamp())}",
            account_id=account_id,
            session_type=SessionType(session_type),
            start_time=start_time,
            duration=duration,
            **kwargs
        )
        
        if self.session_scheduler.schedule_session(schedule):
            return schedule.id
        else:
            raise RuntimeError("Échec de la planification")
    
    def get_statistics(self) -> dict:
        """
        Récupère les statistiques complètes du système.
        
        Returns:
            Dictionnaire avec toutes les statistiques
        """
        if not self.running:
            return {}
        
        return {
            "accounts": self.account_manager.get_session_statistics(),
            "windows": self.window_manager.get_memory_usage_summary(),
            "synchronization": self.synchronizer.get_statistics(),
            "scheduling": self.session_scheduler.get_statistics()
        }
    
    def launch_gui(self):
        """Lance l'interface graphique du système."""
        try:
            import sys
            from PySide6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            app.setApplicationName("DOFUS Multi-Account Manager")
            
            # Création de la fenêtre principale avec les gestionnaires existants
            from .multi_account_gui import MultiAccountMainWindow
            
            # Injection des gestionnaires dans la GUI
            window = MultiAccountMainWindow()
            if self.running:
                window.account_manager = self.account_manager
                window.window_manager = self.window_manager
                window.synchronizer = self.synchronizer
                window.session_scheduler = self.session_scheduler
            
            window.show()
            
            return app.exec()
            
        except ImportError:
            print("Interface graphique non disponible. Installez PySide6: pip install PySide6")
            return 1
        except Exception as e:
            print(f"Erreur lors du lancement de l'interface: {e}")
            return 1
    
    def __enter__(self):
        """Support du context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager."""
        self.stop()


# Utilitaires de compatibilité
def create_system(master_password: str = None, max_accounts: int = 8) -> MultiAccountSystem:
    """
    Crée une nouvelle instance du système multi-comptes.
    
    Args:
        master_password: Mot de passe maître
        max_accounts: Nombre maximum de comptes
        
    Returns:
        Instance du système
    """
    if master_password is None:
        master_password = "DofusBotMaster2024"
    
    return MultiAccountSystem(master_password, max_accounts)


def launch_gui_standalone():
    """Lance l'interface graphique de manière autonome."""
    system = create_system()
    if system.start():
        try:
            return system.launch_gui()
        finally:
            system.stop()
    else:
        print("Échec du démarrage du système")
        return 1


# Point d'entrée pour l'exécution directe du module
if __name__ == "__main__":
    import sys
    sys.exit(launch_gui_standalone())