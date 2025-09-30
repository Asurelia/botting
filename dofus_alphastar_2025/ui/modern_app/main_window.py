#!/usr/bin/env python3
"""
MainWindow - Fenêtre principale de l'application DOFUS AlphaStar
Interface moderne avec onglets et monitoring temps réel
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Any, Optional

from .app_controller import AppController, BotState
from .theme_manager import ThemeManager
from .dashboard_panel import DashboardPanel
from .control_panel import ControlPanel
from .monitoring_panel import MonitoringPanel
from .config_panel import ConfigPanel
from .analytics_panel import AnalyticsPanel

class StatusBar:
    """Barre de statut en bas de l'application"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.theme = theme_manager
        self.colors = theme_manager.get_colors()

        # Frame principal
        self.frame = tk.Frame(parent,
                             bg=self.colors.bg_secondary,
                             height=30,
                             relief="flat",
                             bd=1)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.frame.pack_propagate(False)

        # Labels de statut
        self.status_label = tk.Label(self.frame,
                                   text="🔴 Arrêté",
                                   bg=self.colors.bg_secondary,
                                   fg=self.colors.text_secondary,
                                   font=self.theme.get_fonts()["status"],
                                   anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Performance
        self.perf_label = tk.Label(self.frame,
                                 text="CPU: 0% | RAM: 0%",
                                 bg=self.colors.bg_secondary,
                                 fg=self.colors.text_secondary,
                                 font=self.theme.get_fonts()["status"])
        self.perf_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Temps d'activité
        self.uptime_label = tk.Label(self.frame,
                                   text="Temps: 00:00:00",
                                   bg=self.colors.bg_secondary,
                                   fg=self.colors.text_secondary,
                                   font=self.theme.get_fonts()["status"])
        self.uptime_label.pack(side=tk.RIGHT, padx=10, pady=5)

    def update_status(self, bot_state: BotState, action: str = None):
        """Met à jour le statut du bot"""
        status_icons = {
            BotState.STOPPED: "🔴",
            BotState.STARTING: "🟡",
            BotState.RUNNING: "🟢",
            BotState.PAUSED: "⏸️",
            BotState.STOPPING: "🟡",
            BotState.ERROR: "❌"
        }

        status_texts = {
            BotState.STOPPED: "Arrêté",
            BotState.STARTING: "Démarrage...",
            BotState.RUNNING: "En cours",
            BotState.PAUSED: "En pause",
            BotState.STOPPING: "Arrêt...",
            BotState.ERROR: "Erreur"
        }

        icon = status_icons.get(bot_state, "⚪")
        text = status_texts.get(bot_state, "Inconnu")

        if action and bot_state == BotState.RUNNING:
            text += f" - {action}"

        self.status_label.config(text=f"{icon} {text}")

    def update_performance(self, cpu_percent: float, memory_percent: float):
        """Met à jour les métriques de performance"""
        self.perf_label.config(text=f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}%")

    def update_uptime(self, uptime_seconds: float):
        """Met à jour le temps d'activité"""
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        self.uptime_label.config(text=f"Temps: {hours:02d}:{minutes:02d}:{seconds:02d}")

class MenuBar:
    """Barre de menu principale"""

    def __init__(self, parent, app_controller: AppController, main_window):
        self.app_controller = app_controller
        self.main_window = main_window

        # Créer menu
        self.menubar = tk.Menu(parent)
        parent.config(menu=self.menubar)

        # Menu Fichier
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouveau profil", command=self.new_profile)
        file_menu.add_command(label="Charger profil", command=self.load_profile)
        file_menu.add_command(label="Sauvegarder profil", command=self.save_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.quit_app)

        # Menu Bot
        bot_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Bot", menu=bot_menu)
        bot_menu.add_command(label="Démarrer", command=self.start_bot)
        bot_menu.add_command(label="Arrêter", command=self.stop_bot)
        bot_menu.add_command(label="Pause", command=self.pause_bot)
        bot_menu.add_command(label="Reprendre", command=self.resume_bot)
        bot_menu.add_separator()
        bot_menu.add_command(label="Redémarrer", command=self.restart_bot)

        # Menu Outils
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(label="Test systèmes", command=self.test_systems)
        tools_menu.add_command(label="Calibrage vision", command=self.calibrate_vision)
        tools_menu.add_command(label="Analyse performance", command=self.analyze_performance)

        # Menu Affichage
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Affichage", menu=view_menu)
        view_menu.add_command(label="Thème sombre", command=lambda: self.change_theme("dark"))
        view_menu.add_command(label="Thème clair", command=lambda: self.change_theme("light"))
        view_menu.add_separator()
        view_menu.add_command(label="Plein écran", command=self.toggle_fullscreen)

        # Menu Aide
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="À propos", command=self.show_about)

    def new_profile(self):
        messagebox.showinfo("Nouveau profil", "Fonctionnalité à implémenter")

    def load_profile(self):
        messagebox.showinfo("Charger profil", "Fonctionnalité à implémenter")

    def save_profile(self):
        messagebox.showinfo("Sauvegarder profil", "Fonctionnalité à implémenter")

    def quit_app(self):
        if messagebox.askokcancel("Quitter", "Êtes-vous sûr de vouloir quitter?"):
            self.main_window.quit_application()

    def start_bot(self):
        self.app_controller.start_bot()

    def stop_bot(self):
        self.app_controller.stop_bot()

    def pause_bot(self):
        self.app_controller.pause_bot()

    def resume_bot(self):
        self.app_controller.resume_bot()

    def restart_bot(self):
        self.app_controller.stop_bot()
        threading.Timer(2.0, self.app_controller.start_bot).start()

    def test_systems(self):
        success = self.app_controller.initialize_systems()
        if success:
            messagebox.showinfo("Test", "Tous les systèmes fonctionnent correctement")
        else:
            messagebox.showerror("Test", "Erreur dans l'initialisation des systèmes")

    def calibrate_vision(self):
        messagebox.showinfo("Calibrage", "Lancement calibrage vision...")

    def analyze_performance(self):
        messagebox.showinfo("Performance", "Analyse des performances...")

    def change_theme(self, theme_name: str):
        self.main_window.change_theme(theme_name)

    def toggle_fullscreen(self):
        self.main_window.toggle_fullscreen()

    def show_docs(self):
        messagebox.showinfo("Documentation", "Documentation disponible dans le README.md")

    def show_about(self):
        about_text = """DOFUS AlphaStar 2025

Version: 1.0.0
Architecture: AlphaStar + HRM
Vision: SAM 2 + TrOCR
GPU: AMD 7800XT optimisé

Développé avec Claude Code"""
        messagebox.showinfo("À propos", about_text)

class MainWindow:
    """Fenêtre principale de l'application"""

    def __init__(self):
        # Contrôleur principal
        self.app_controller = AppController()

        # Interface
        self.root = tk.Tk()
        self.theme_manager = ThemeManager()
        self.theme_manager.register_root(self.root)

        # État
        self.is_fullscreen = False
        self.last_update_time = 0

        # Initialiser interface
        self.setup_window()
        self.create_widgets()
        self.setup_event_handlers()
        self.start_update_loop()

    def setup_window(self):
        """Configure la fenêtre principale"""
        self.root.title("DOFUS AlphaStar 2025 - Bot IA Avancé")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Icône (si disponible)
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass

        # Centrer la fenêtre
        self.center_window()

        # Gestionnaire de fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def center_window(self):
        """Centre la fenêtre sur l'écran"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        """Crée les widgets de l'interface"""

        # Menu bar
        self.menu_bar = MenuBar(self.root, self.app_controller, self)

        # Frame principal
        main_frame = self.theme_manager.create_frame(self.root, "primary")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Header avec titre et contrôles rapides
        self.create_header(main_frame)

        # Notebook (onglets)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Créer les onglets
        self.create_tabs()

        # Status bar
        self.status_bar = StatusBar(self.root, self.theme_manager)

    def create_header(self, parent):
        """Crée le header avec titre et contrôles"""
        header_frame = self.theme_manager.create_frame(parent, "secondary")
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Titre principal
        title_frame = tk.Frame(header_frame, bg=self.theme_manager.get_colors().bg_secondary)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)

        title_label = self.theme_manager.create_title_label(
            title_frame,
            "🎮 DOFUS AlphaStar 2025"
        )
        title_label.pack(side=tk.TOP, anchor="w")

        subtitle_label = self.theme_manager.create_subtitle_label(
            title_frame,
            "Bot d'Intelligence Artificielle Avancé"
        )
        subtitle_label.pack(side=tk.TOP, anchor="w")

        # Contrôles rapides
        controls_frame = tk.Frame(header_frame, bg=self.theme_manager.get_colors().bg_secondary)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20)

        # Boutons de contrôle
        self.start_button = self.theme_manager.create_primary_button(
            controls_frame,
            "▶ Démarrer",
            command=self.start_bot
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = self.theme_manager.create_secondary_button(
            controls_frame,
            "⏹ Arrêter",
            command=self.stop_bot,
            state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = self.theme_manager.create_secondary_button(
            controls_frame,
            "⏸ Pause",
            command=self.pause_bot,
            state="disabled"
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

    def create_tabs(self):
        """Crée les onglets principaux"""

        # Dashboard principal
        self.dashboard_panel = DashboardPanel(
            self.notebook,
            self.app_controller,
            self.theme_manager
        )
        self.notebook.add(self.dashboard_panel.frame, text="📊 Dashboard")

        # Contrôles
        self.control_panel = ControlPanel(
            self.notebook,
            self.app_controller,
            self.theme_manager
        )
        self.notebook.add(self.control_panel.frame, text="🎮 Contrôles")

        # Monitoring
        self.monitoring_panel = MonitoringPanel(
            self.notebook,
            self.app_controller,
            self.theme_manager
        )
        self.notebook.add(self.monitoring_panel.frame, text="📈 Monitoring")

        # Configuration
        self.config_panel = ConfigPanel(
            self.notebook,
            self.app_controller,
            self.theme_manager
        )
        self.notebook.add(self.config_panel.frame, text="⚙️ Configuration")

        # Analytics
        self.analytics_panel = AnalyticsPanel(
            self.notebook,
            self.app_controller,
            self.theme_manager
        )
        self.notebook.add(self.analytics_panel.frame, text="📊 Analytics")

    def setup_event_handlers(self):
        """Configure les gestionnaires d'événements"""

        # Événements du contrôleur
        self.app_controller.event_manager.subscribe("bot_started", self.on_bot_started)
        self.app_controller.event_manager.subscribe("bot_stopped", self.on_bot_stopped)
        self.app_controller.event_manager.subscribe("bot_paused", self.on_bot_paused)
        self.app_controller.event_manager.subscribe("bot_resumed", self.on_bot_resumed)
        self.app_controller.event_manager.subscribe("bot_error", self.on_bot_error)
        self.app_controller.event_manager.subscribe("bot_status_updated", self.on_status_updated)
        self.app_controller.event_manager.subscribe("performance_metrics", self.on_performance_updated)

        # Événements fenêtre
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Control-q>", lambda e: self.quit_application())

    def start_update_loop(self):
        """Démarre la boucle de mise à jour de l'interface"""
        self.update_interface()

    def update_interface(self):
        """Met à jour l'interface périodiquement"""
        current_time = time.time()

        # Mettre à jour toutes les 500ms
        if current_time - self.last_update_time >= 0.5:
            self.last_update_time = current_time

            # Mettre à jour les panels
            self.dashboard_panel.update()
            self.monitoring_panel.update()
            self.analytics_panel.update()

        # Programmer prochaine mise à jour
        self.root.after(100, self.update_interface)

    def on_bot_started(self, data):
        """Gestionnaire démarrage bot"""
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal")
        self.status_bar.update_status(BotState.RUNNING)

    def on_bot_stopped(self, data):
        """Gestionnaire arrêt bot"""
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.pause_button.config(state="disabled")
        self.status_bar.update_status(BotState.STOPPED)

    def on_bot_paused(self, data):
        """Gestionnaire pause bot"""
        self.pause_button.config(text="▶ Reprendre")
        self.status_bar.update_status(BotState.PAUSED)

    def on_bot_resumed(self, data):
        """Gestionnaire reprise bot"""
        self.pause_button.config(text="⏸ Pause")
        self.status_bar.update_status(BotState.RUNNING)

    def on_bot_error(self, data):
        """Gestionnaire erreur bot"""
        error_msg = data.get("error", "Erreur inconnue")
        self.status_bar.update_status(BotState.ERROR)
        messagebox.showerror("Erreur Bot", f"Erreur du bot:\n{error_msg}")

    def on_status_updated(self, data):
        """Gestionnaire mise à jour statut"""
        current_action = data.get("current_action", "")
        if current_action:
            bot_state = BotState(data.get("state", "stopped"))
            self.status_bar.update_status(bot_state, current_action)

    def on_performance_updated(self, data):
        """Gestionnaire mise à jour performance"""
        cpu = data.get("cpu_percent", 0)
        memory = data.get("memory_percent", 0)
        self.status_bar.update_performance(cpu, memory)

        # Calculer uptime
        if self.app_controller.bot_status.session_start_time:
            uptime = time.time() - self.app_controller.bot_status.session_start_time
            self.status_bar.update_uptime(uptime)

    def start_bot(self):
        """Démarre le bot"""
        self.app_controller.start_bot("auto")

    def stop_bot(self):
        """Arrête le bot"""
        self.app_controller.stop_bot()

    def pause_bot(self):
        """Pause/reprend le bot"""
        if self.app_controller.bot_status.state == BotState.RUNNING:
            self.app_controller.pause_bot()
        elif self.app_controller.bot_status.state == BotState.PAUSED:
            self.app_controller.resume_bot()

    def change_theme(self, theme_name: str):
        """Change le thème de l'interface"""
        self.theme_manager.set_theme(theme_name)
        # Redessiner interface si nécessaire
        self.root.update()

    def toggle_fullscreen(self):
        """Bascule le mode plein écran"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def on_closing(self):
        """Gestionnaire fermeture application"""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application?"):
            self.quit_application()

    def quit_application(self):
        """Quitte proprement l'application"""
        # Arrêter le bot
        if self.app_controller.bot_status.state in [BotState.RUNNING, BotState.PAUSED]:
            self.app_controller.stop_bot()

        # Shutdown contrôleur
        self.app_controller.shutdown()

        # Fermer fenêtre
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Lance l'application"""
        try:
            # Initialiser systèmes en arrière-plan
            threading.Thread(
                target=self.app_controller.initialize_systems,
                daemon=True
            ).start()

            # Démarrer interface
            self.root.mainloop()

        except KeyboardInterrupt:
            self.quit_application()
        except Exception as e:
            messagebox.showerror("Erreur Fatale", f"Erreur fatale:\n{e}")
            self.quit_application()

def create_main_window() -> MainWindow:
    """Factory function pour créer MainWindow"""
    return MainWindow()