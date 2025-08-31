#!/usr/bin/env python3
"""
Point d'entrée principal du bot DOFUS avec CLI et GUI intégrées.

Ce module fournit l'interface principale pour lancer le bot avec différents modes
d'opération : CLI, GUI, ou service en arrière-plan.

Usage:
    python main.py --mode cli --profile default
    python main.py --mode gui
    python main.py --mode service --daemon
    python main.py --calibrate
    python main.py --monitor
"""

import sys
import os
import argparse
import logging
import asyncio
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

from engine.core import BotCore
from engine.event_bus import EventBus
from state.realtime_state import RealtimeState
from state.state_tracker import StateTracker

# Import conditionnel pour l'interface graphique
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Interface graphique non disponible - mode CLI uniquement")


class MainApplication:
    """Application principale du bot DOFUS."""
    
    def __init__(self):
        self.bot_core: Optional[BotCore] = None
        self.event_bus = EventBus()
        self.state_tracker = StateTracker()
        self.realtime_state = RealtimeState()
        self.running = False
        self.mode = "cli"
        self.profile_name = "default"
        self.config_dir = Path("config")
        self.logs_dir = Path("logs")
        
        # Créer les dossiers nécessaires
        self._create_directories()
        
        # Configurer les signaux système
        self._setup_signal_handlers()
        
        # Configurer le logging
        self._setup_logging()
        
    def _create_directories(self):
        """Créer les dossiers nécessaires."""
        for directory in [self.config_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
            
    def _setup_signal_handlers(self):
        """Configurer les gestionnaires de signaux pour un arrêt propre."""
        def signal_handler(signum, frame):
            logging.info(f"Signal {signum} reçu, arrêt du bot...")
            self.shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def _setup_logging(self):
        """Configurer le système de logging."""
        log_file = self.logs_dir / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Application principale initialisée")
        
    async def initialize_bot(self, profile_name: str = "default"):
        """Initialiser le core du bot avec le profil spécifié."""
        try:
            profile_path = self.config_dir / f"profiles/{profile_name}.json"
            
            if not profile_path.exists():
                self.logger.warning(f"Profil {profile_name} non trouvé, création du profil par défaut")
                await self._create_default_profile(profile_name)
                
            self.bot_core = BotCore(
                event_bus=self.event_bus,
                state_tracker=self.state_tracker,
                realtime_state=self.realtime_state,
                profile_path=str(profile_path)
            )
            
            await self.bot_core.initialize()
            self.logger.info(f"Bot initialisé avec le profil '{profile_name}'")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation : {e}")
            raise
            
    async def _create_default_profile(self, profile_name: str):
        """Créer un profil par défaut."""
        import json
        
        profile_dir = self.config_dir / "profiles"
        profile_dir.mkdir(exist_ok=True)
        
        default_profile = {
            "name": profile_name,
            "created": datetime.now().isoformat(),
            "character": {
                "name": "BotCharacter",
                "class": "iop",
                "level": 1,
                "server": "Ily"
            },
            "modules": {
                "combat": {"enabled": True, "priority": 1},
                "professions": {"enabled": True, "priority": 2},
                "navigation": {"enabled": True, "priority": 3},
                "economy": {"enabled": False, "priority": 4},
                "social": {"enabled": False, "priority": 5}
            },
            "safety": {
                "detection_avoidance": True,
                "human_behavior": True,
                "max_session_time": 14400,  # 4 heures
                "break_intervals": [3600, 7200]  # Pauses toutes les heures
            },
            "automation": {
                "auto_start": False,
                "auto_shutdown": True,
                "daily_routine": True
            }
        }
        
        profile_path = profile_dir / f"{profile_name}.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(default_profile, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Profil par défaut créé : {profile_path}")

    async def run_cli_mode(self, args):
        """Exécuter en mode ligne de commande."""
        self.logger.info("Démarrage en mode CLI")
        
        try:
            await self.initialize_bot(args.profile)
            
            print("\n=== BOT DOFUS - MODE CLI ===")
            print(f"Profil: {args.profile}")
            print(f"Mode debug: {'Oui' si args.debug else 'Non'}")
            print("Commandes disponibles:")
            print("  start    - Démarrer le bot")
            print("  stop     - Arrêter le bot")  
            print("  status   - Afficher le statut")
            print("  modules  - Lister les modules")
            print("  profile  - Changer de profil")
            print("  exit     - Quitter l'application")
            print("-" * 40)
            
            if args.auto_start:
                print("Démarrage automatique...")
                await self._start_bot()
                
            # Boucle interactive CLI
            await self._cli_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Interruption clavier détectée")
        except Exception as e:
            self.logger.error(f"Erreur en mode CLI : {e}")
        finally:
            await self.shutdown()

    async def _cli_loop(self):
        """Boucle interactive de la ligne de commande."""
        while self.running or not hasattr(self, '_shutdown_requested'):
            try:
                command = input("\nbot> ").strip().lower()
                
                if command == "exit":
                    break
                elif command == "start":
                    await self._start_bot()
                elif command == "stop":
                    await self._stop_bot()
                elif command == "status":
                    self._show_status()
                elif command == "modules":
                    self._show_modules()
                elif command == "profile":
                    await self._change_profile_cli()
                elif command == "help" or command == "?":
                    self._show_help()
                elif command == "":
                    continue
                else:
                    print(f"Commande inconnue: {command}")
                    print("Tapez 'help' pour voir les commandes disponibles")
                    
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle CLI : {e}")
                
    async def _start_bot(self):
        """Démarrer le bot."""
        if self.running:
            print("Le bot est déjà en cours d'exécution")
            return
            
        try:
            self.running = True
            await self.bot_core.start()
            print("✓ Bot démarré avec succès")
        except Exception as e:
            self.running = False
            print(f"✗ Erreur lors du démarrage : {e}")
            
    async def _stop_bot(self):
        """Arrêter le bot."""
        if not self.running:
            print("Le bot n'est pas en cours d'exécution")
            return
            
        try:
            await self.bot_core.stop()
            self.running = False
            print("✓ Bot arrêté")
        except Exception as e:
            print(f"✗ Erreur lors de l'arrêt : {e}")
            
    def _show_status(self):
        """Afficher le statut du bot."""
        status = "En cours" if self.running else "Arrêté"
        print(f"\nStatut du bot: {status}")
        print(f"Profil actuel: {self.profile_name}")
        
        if self.bot_core and hasattr(self.bot_core, 'get_stats'):
            stats = self.bot_core.get_stats()
            print(f"Temps d'exécution: {stats.get('runtime', 'N/A')}")
            print(f"Actions effectuées: {stats.get('actions_count', 0)}")
            print(f"Modules actifs: {stats.get('active_modules', 0)}")
            
    def _show_modules(self):
        """Afficher la liste des modules."""
        print("\nModules disponibles:")
        modules = ["combat", "professions", "navigation", "economy", "social", "automation", "safety"]
        for i, module in enumerate(modules, 1):
            status = "✓" if self.bot_core and hasattr(self.bot_core, 'is_module_enabled') and self.bot_core.is_module_enabled(module) else "✗"
            print(f"  {i}. {module} {status}")
            
    async def _change_profile_cli(self):
        """Changer de profil en mode CLI."""
        profiles_dir = self.config_dir / "profiles"
        if not profiles_dir.exists():
            print("Aucun profil trouvé")
            return
            
        profiles = list(profiles_dir.glob("*.json"))
        if not profiles:
            print("Aucun profil disponible")
            return
            
        print("\nProfils disponibles:")
        for i, profile in enumerate(profiles, 1):
            name = profile.stem
            current = " (actuel)" if name == self.profile_name else ""
            print(f"  {i}. {name}{current}")
            
        try:
            choice = input("\nChoisir un profil (numéro): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(profiles):
                    new_profile = profiles[idx].stem
                    if new_profile != self.profile_name:
                        await self._stop_bot()
                        self.profile_name = new_profile
                        await self.initialize_bot(new_profile)
                        print(f"✓ Profil changé pour: {new_profile}")
                    else:
                        print("Profil déjà sélectionné")
                else:
                    print("Choix invalide")
            else:
                print("Veuillez entrer un numéro")
        except ValueError:
            print("Entrée invalide")
            
    def _show_help(self):
        """Afficher l'aide des commandes."""
        print("\nCommandes disponibles:")
        print("  start    - Démarrer le bot")
        print("  stop     - Arrêter le bot")
        print("  status   - Afficher le statut du bot")
        print("  modules  - Lister tous les modules")
        print("  profile  - Changer de profil utilisateur")
        print("  help     - Afficher cette aide")
        print("  exit     - Quitter l'application")

    def run_gui_mode(self, args):
        """Exécuter en mode interface graphique."""
        if not GUI_AVAILABLE:
            print("Interface graphique non disponible")
            sys.exit(1)
            
        self.logger.info("Démarrage en mode GUI")
        
        root = tk.Tk()
        root.title("Bot DOFUS - Interface Principale")
        root.geometry("800x600")
        
        # Créer l'interface
        self._create_gui(root, args)
        
        # Démarrer la boucle GUI
        root.mainloop()
        
    def _create_gui(self, root, args):
        """Créer l'interface graphique."""
        # Frame principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration des colonnes/lignes extensibles
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Section statut
        status_frame = ttk.LabelFrame(main_frame, text="Statut du Bot", padding="5")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Statut: Arrêté", font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.profile_label = ttk.Label(status_frame, text=f"Profil: {args.profile}")
        self.profile_label.grid(row=1, column=0, sticky=tk.W)
        
        # Boutons de contrôle
        control_frame = ttk.LabelFrame(main_frame, text="Contrôles", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Démarrer", command=self._gui_start_bot)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Arrêter", command=self._gui_stop_bot, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        self.calibrate_button = ttk.Button(control_frame, text="Calibrer", command=self._gui_calibrate)
        self.calibrate_button.grid(row=0, column=2, padx=(0, 5))
        
        self.monitor_button = ttk.Button(control_frame, text="Monitoring", command=self._gui_monitor)
        self.monitor_button.grid(row=0, column=3)
        
        # Zone de logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configuration pour que les logs s'étendent
        main_frame.rowconfigure(2, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=15, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Menu
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouveau profil...", command=self._gui_new_profile)
        file_menu.add_command(label="Charger profil...", command=self._gui_load_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=root.quit)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(label="Calibration", command=self._gui_calibrate)
        tools_menu.add_command(label="Monitoring", command=self._gui_monitor)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="À propos", command=self._gui_about)
        
    def _gui_start_bot(self):
        """Démarrer le bot depuis l'interface graphique."""
        # Implémentation simplifiée pour l'exemple
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Statut: En cours d'exécution")
        self._log_to_gui("Bot démarré")
        
    def _gui_stop_bot(self):
        """Arrêter le bot depuis l'interface graphique."""
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Statut: Arrêté")
        self._log_to_gui("Bot arrêté")
        
    def _gui_calibrate(self):
        """Ouvrir l'outil de calibration."""
        import subprocess
        try:
            subprocess.Popen([sys.executable, "calibrate.py"])
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir la calibration : {e}")
            
    def _gui_monitor(self):
        """Ouvrir le monitoring."""
        import subprocess
        try:
            subprocess.Popen([sys.executable, "monitor.py"])
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le monitoring : {e}")
            
    def _gui_new_profile(self):
        """Créer un nouveau profil."""
        # Dialog simple pour l'exemple
        name = tk.simpledialog.askstring("Nouveau profil", "Nom du profil:")
        if name:
            # Créer le profil
            asyncio.run(self._create_default_profile(name))
            self._log_to_gui(f"Profil '{name}' créé")
            
    def _gui_load_profile(self):
        """Charger un profil existant."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Charger un profil",
            initialdir=self.config_dir / "profiles",
            filetypes=[("Profils JSON", "*.json")]
        )
        if file_path:
            profile_name = Path(file_path).stem
            self.profile_name = profile_name
            self.profile_label.config(text=f"Profil: {profile_name}")
            self._log_to_gui(f"Profil '{profile_name}' chargé")
            
    def _gui_about(self):
        """Afficher les informations à propos."""
        messagebox.showinfo("À propos", "Bot DOFUS v1.0\nInterface principale du bot")
        
    def _log_to_gui(self, message):
        """Ajouter un message aux logs de l'interface."""
        if hasattr(self, 'log_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")

    async def run_service_mode(self, args):
        """Exécuter en mode service/daemon."""
        self.logger.info("Démarrage en mode service")
        
        try:
            await self.initialize_bot(args.profile)
            
            if args.daemon:
                self.logger.info("Mode daemon activé")
                # En mode daemon, le bot tourne indéfiniment
                await self._start_bot()
                
                # Maintenir le service actif
                while self.running:
                    await asyncio.sleep(1)
            else:
                # Mode service interactif
                await self._start_bot()
                print("Bot en cours d'exécution en mode service...")
                print("Appuyez sur Ctrl+C pour arrêter")
                
                try:
                    while self.running:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nArrêt du service demandé...")
                    
        except Exception as e:
            self.logger.error(f"Erreur en mode service : {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Arrêt propre de l'application."""
        self.logger.info("Arrêt de l'application...")
        self._shutdown_requested = True
        
        if self.bot_core and self.running:
            await self._stop_bot()
            
        self.logger.info("Application fermée")

def create_argument_parser():
    """Créer le parser d'arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Bot DOFUS - Point d'entrée principal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --mode cli --profile default
  %(prog)s --mode gui
  %(prog)s --mode service --daemon
  %(prog)s --calibrate
  %(prog)s --monitor
        """
    )
    
    # Mode d'opération
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "gui", "service"],
        default="cli",
        help="Mode d'opération (défaut: cli)"
    )
    
    # Profil utilisateur
    parser.add_argument(
        "--profile", "-p",
        default="default",
        help="Nom du profil à utiliser (défaut: default)"
    )
    
    # Options spécifiques
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Activer le mode debug"
    )
    
    parser.add_argument(
        "--auto-start", "-a",
        action="store_true",
        help="Démarrage automatique du bot"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Mode daemon (service en arrière-plan)"
    )
    
    # Outils rapides
    parser.add_argument(
        "--calibrate", "-c",
        action="store_true",
        help="Lancer l'outil de calibration"
    )
    
    parser.add_argument(
        "--monitor",
        action="store_true", 
        help="Lancer le monitoring"
    )
    
    # Configuration
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Répertoire de configuration (défaut: config)"
    )
    
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Répertoire des logs (défaut: logs)"
    )
    
    return parser

async def main():
    """Fonction principale."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Outils rapides
    if args.calibrate:
        import subprocess
        subprocess.run([sys.executable, "calibrate.py"])
        return
        
    if args.monitor:
        import subprocess
        subprocess.run([sys.executable, "monitor.py"])
        return
    
    # Configuration du niveau de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Créer et lancer l'application principale
    app = MainApplication()
    
    # Configurer les répertoires personnalisés
    if args.config_dir:
        app.config_dir = Path(args.config_dir)
    if args.logs_dir:
        app.logs_dir = Path(args.logs_dir)
        
    try:
        if args.mode == "cli":
            await app.run_cli_mode(args)
        elif args.mode == "gui":
            app.run_gui_mode(args)  # Synchrone
        elif args.mode == "service":
            await app.run_service_mode(args)
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        logging.error(f"Erreur critique : {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nAu revoir!")
        sys.exit(0)