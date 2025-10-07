"""
AlphaStar Dashboard - Interface principale inspirée de DeepMind
Dashboard de monitoring et contrôle en temps réel
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import queue

from ..core.alphastar_engine import LeagueManager, create_league_system
from ..core.rl_training import RLlibTrainer, create_rllib_trainer
from ..core.hrm_reasoning import DofusHRMAgent
from ..config import config

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Métriques du dashboard"""
    timestamp: float
    training_iteration: int = 0
    total_timesteps: int = 0
    episode_reward_mean: float = 0.0
    win_rate: float = 0.0
    league_diversity: float = 0.0
    agents_active: int = 0
    matches_completed: int = 0
    avg_reasoning_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0

class AlphaStarDashboard:
    """Dashboard principal AlphaStar pour DOFUS"""

    def __init__(self, master=None):
        # Interface principale
        if master is None:
            self.root = tk.Tk()
            self.root.title("DOFUS AlphaStar 2025 - Dashboard")
            self.root.geometry("1400x900")
            self.root.configure(bg="#1e1e1e")  # Theme sombre
        else:
            self.root = master

        # Composants système
        self.league_manager: Optional[LeagueManager] = None
        self.trainer: Optional[RLlibTrainer] = None

        # État du dashboard
        self.is_training = False
        self.is_monitoring = False
        self.metrics_history: List[DashboardMetrics] = []
        self.update_interval = 1000  # ms

        # Threading pour UI non-bloquante
        self.update_thread = None
        self.metrics_queue = queue.Queue()

        # Variables UI
        self.setup_variables()

        # Interface
        self.setup_ui()

        # Démarrer monitoring
        self.start_monitoring()

        logger.info("AlphaStar Dashboard initialisé")

    def setup_variables(self):
        """Configure les variables tkinter"""
        # Status variables
        self.var_status = tk.StringVar(value="Prêt")
        self.var_training_status = tk.StringVar(value="Arrêté")
        self.var_league_status = tk.StringVar(value="Inactif")

        # Metrics variables
        self.var_total_timesteps = tk.StringVar(value="0")
        self.var_episode_reward = tk.StringVar(value="0.00")
        self.var_win_rate = tk.StringVar(value="0.0%")
        self.var_agents_count = tk.StringVar(value="0")
        self.var_matches_completed = tk.StringVar(value="0")
        self.var_memory_usage = tk.StringVar(value="0 MB")

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Style
        self.setup_styles()

        # Menu principal
        self.setup_menu()

        # Toolbar
        self.setup_toolbar()

        # Layout principal (PanedWindow)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panneau gauche - Contrôles
        self.setup_control_panel()

        # Panneau central - Graphiques
        self.setup_charts_panel()

        # Panneau droit - League & Agents
        self.setup_league_panel()

        # Status bar
        self.setup_status_bar()

    def setup_styles(self):
        """Configure le style sombre"""
        style = ttk.Style()

        # Configurer thème sombre
        style.theme_use('clam')

        # Couleurs du thème
        bg_dark = "#2d2d2d"
        bg_lighter = "#3d3d3d"
        fg_light = "#ffffff"
        accent = "#0078d4"

        # Configurer styles
        style.configure("Dark.TFrame", background=bg_dark)
        style.configure("Dark.TLabel", background=bg_dark, foreground=fg_light)
        style.configure("Dark.TButton", background=bg_lighter, foreground=fg_light)
        style.configure("Accent.TButton", background=accent, foreground=fg_light)

    def setup_menu(self):
        """Configure le menu principal"""
        menubar = tk.Menu(self.root, bg="#2d2d2d", fg="white")
        self.root.config(menu=menubar)

        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2d2d2d", fg="white")
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouvelle configuration...", command=self.new_config)
        file_menu.add_command(label="Charger configuration...", command=self.load_config)
        file_menu.add_command(label="Sauvegarder configuration...", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exporter métriques...", command=self.export_metrics)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.quit_dashboard)

        # Menu Entraînement
        train_menu = tk.Menu(menubar, tearoff=0, bg="#2d2d2d", fg="white")
        menubar.add_cascade(label="Entraînement", menu=train_menu)
        train_menu.add_command(label="Démarrer entraînement", command=self.start_training)
        train_menu.add_command(label="Arrêter entraînement", command=self.stop_training)
        train_menu.add_command(label="Pause/Resume", command=self.toggle_training)
        train_menu.add_separator()
        train_menu.add_command(label="Reset league", command=self.reset_league)

        # Menu Vue
        view_menu = tk.Menu(menubar, tearoff=0, bg="#2d2d2d", fg="white")
        menubar.add_cascade(label="Vue", menu=view_menu)
        view_menu.add_command(label="Plein écran", command=self.toggle_fullscreen)
        view_menu.add_command(label="Reset layout", command=self.reset_layout)

    def setup_toolbar(self):
        """Configure la barre d'outils"""
        toolbar = ttk.Frame(self.root, style="Dark.TFrame")
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # Boutons principaux
        ttk.Button(toolbar, text="> Start", style="Accent.TButton",
                  command=self.start_training).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="|| Pause", command=self.toggle_training).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="[] Stop", command=self.stop_training).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Contrôles league
        ttk.Button(toolbar, text=" New League", command=self.create_league).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="[TARGET] Manual Match", command=self.create_manual_match).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Status en temps réel
        ttk.Label(toolbar, text="Status:", style="Dark.TLabel").pack(side=tk.LEFT, padx=5)
        ttk.Label(toolbar, textvariable=self.var_status, style="Dark.TLabel").pack(side=tk.LEFT)

    def setup_control_panel(self):
        """Configure le panneau de contrôles"""
        # Frame principal pour contrôles
        control_frame = ttk.Frame(self.main_paned, style="Dark.TFrame", width=350)
        self.main_paned.add(control_frame, weight=1)

        # Notebook pour organiser les contrôles
        control_notebook = ttk.Notebook(control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === ONGLET TRAINING ===
        training_frame = ttk.Frame(control_notebook, style="Dark.TFrame")
        control_notebook.add(training_frame, text="[START] Training")

        # Status training
        status_group = ttk.LabelFrame(training_frame, text="Training Status", style="Dark.TFrame")
        status_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(status_group, text="Status:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(status_group, textvariable=self.var_training_status, style="Dark.TLabel").pack(anchor=tk.W)

        ttk.Label(status_group, text="Total Steps:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(status_group, textvariable=self.var_total_timesteps, style="Dark.TLabel").pack(anchor=tk.W)

        ttk.Label(status_group, text="Episode Reward:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(status_group, textvariable=self.var_episode_reward, style="Dark.TLabel").pack(anchor=tk.W)

        # Configuration training
        config_group = ttk.LabelFrame(training_frame, text="Configuration", style="Dark.TFrame")
        config_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(config_group, text="Algorithm:", style="Dark.TLabel").pack(anchor=tk.W)
        self.algorithm_var = tk.StringVar(value="PPO")
        algorithm_combo = ttk.Combobox(config_group, textvariable=self.algorithm_var,
                                     values=["PPO", "IMPALA", "SAC"], state="readonly")
        algorithm_combo.pack(fill=tk.X, pady=2)

        ttk.Label(config_group, text="Learning Rate:", style="Dark.TLabel").pack(anchor=tk.W)
        self.lr_var = tk.StringVar(value="3e-4")
        lr_entry = ttk.Entry(config_group, textvariable=self.lr_var)
        lr_entry.pack(fill=tk.X, pady=2)

        # === ONGLET LEAGUE ===
        league_frame = ttk.Frame(control_notebook, style="Dark.TFrame")
        control_notebook.add(league_frame, text=" League")

        # Status league
        league_status_group = ttk.LabelFrame(league_frame, text="League Status", style="Dark.TFrame")
        league_status_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(league_status_group, text="Active Agents:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(league_status_group, textvariable=self.var_agents_count, style="Dark.TLabel").pack(anchor=tk.W)

        ttk.Label(league_status_group, text="Matches:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(league_status_group, textvariable=self.var_matches_completed, style="Dark.TLabel").pack(anchor=tk.W)

        ttk.Label(league_status_group, text="Win Rate:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(league_status_group, textvariable=self.var_win_rate, style="Dark.TLabel").pack(anchor=tk.W)

        # === ONGLET SYSTEM ===
        system_frame = ttk.Frame(control_notebook, style="Dark.TFrame")
        control_notebook.add(system_frame, text=" System")

        # Métriques système
        system_group = ttk.LabelFrame(system_frame, text="Performance", style="Dark.TFrame")
        system_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(system_group, text="Memory Usage:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Label(system_group, textvariable=self.var_memory_usage, style="Dark.TLabel").pack(anchor=tk.W)

        # Contrôles système
        ttk.Button(system_group, text="Clear Memory Cache",
                  command=self.clear_memory_cache).pack(fill=tk.X, pady=2)
        ttk.Button(system_group, text="Garbage Collect",
                  command=self.force_garbage_collect).pack(fill=tk.X, pady=2)

    def setup_charts_panel(self):
        """Configure le panneau des graphiques"""
        charts_frame = ttk.Frame(self.main_paned, style="Dark.TFrame")
        self.main_paned.add(charts_frame, weight=3)

        # Notebook pour les graphiques
        self.charts_notebook = ttk.Notebook(charts_frame)
        self.charts_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === GRAPHIQUE TRAINING ===
        self.setup_training_charts()

        # === GRAPHIQUE LEAGUE ===
        self.setup_league_charts()

        # === GRAPHIQUE PERFORMANCE ===
        self.setup_performance_charts()

    def setup_training_charts(self):
        """Configure les graphiques d'entraînement"""
        training_chart_frame = ttk.Frame(self.charts_notebook, style="Dark.TFrame")
        self.charts_notebook.add(training_chart_frame, text="[CHART] Training")

        # Figure matplotlib
        self.training_fig, ((self.ax_reward, self.ax_steps),
                           (self.ax_loss, self.ax_win_rate)) = plt.subplots(2, 2, figsize=(12, 8))
        self.training_fig.patch.set_facecolor('#2d2d2d')

        # Style des graphiques
        for ax in [self.ax_reward, self.ax_steps, self.ax_loss, self.ax_win_rate]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        # Titres et labels
        self.ax_reward.set_title("Episode Reward", color='white')
        self.ax_steps.set_title("Training Steps", color='white')
        self.ax_loss.set_title("Policy Loss", color='white')
        self.ax_win_rate.set_title("Win Rate", color='white')

        # Canvas
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, training_chart_frame)
        self.training_canvas.draw()
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_league_charts(self):
        """Configure les graphiques de league"""
        league_chart_frame = ttk.Frame(self.charts_notebook, style="Dark.TFrame")
        self.charts_notebook.add(league_chart_frame, text=" League")

        # Figure league
        self.league_fig, ((self.ax_elo, self.ax_diversity),
                         (self.ax_matches, self.ax_agents)) = plt.subplots(2, 2, figsize=(12, 8))
        self.league_fig.patch.set_facecolor('#2d2d2d')

        # Style
        for ax in [self.ax_elo, self.ax_diversity, self.ax_matches, self.ax_agents]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        self.ax_elo.set_title("ELO Distribution", color='white')
        self.ax_diversity.set_title("League Diversity", color='white')
        self.ax_matches.set_title("Match Results", color='white')
        self.ax_agents.set_title("Active Agents", color='white')

        self.league_canvas = FigureCanvasTkAgg(self.league_fig, league_chart_frame)
        self.league_canvas.draw()
        self.league_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_performance_charts(self):
        """Configure les graphiques de performance"""
        perf_chart_frame = ttk.Frame(self.charts_notebook, style="Dark.TFrame")
        self.charts_notebook.add(perf_chart_frame, text=" Performance")

        self.perf_fig, ((self.ax_memory, self.ax_gpu),
                        (self.ax_timing, self.ax_throughput)) = plt.subplots(2, 2, figsize=(12, 8))
        self.perf_fig.patch.set_facecolor('#2d2d2d')

        for ax in [self.ax_memory, self.ax_gpu, self.ax_timing, self.ax_throughput]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        self.ax_memory.set_title("Memory Usage", color='white')
        self.ax_gpu.set_title("GPU Utilization", color='white')
        self.ax_timing.set_title("Reasoning Time", color='white')
        self.ax_throughput.set_title("Training Throughput", color='white')

        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_chart_frame)
        self.perf_canvas.draw()
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_league_panel(self):
        """Configure le panneau de league"""
        league_panel = ttk.Frame(self.main_paned, style="Dark.TFrame", width=300)
        self.main_paned.add(league_panel, weight=1)

        # === AGENT RANKING ===
        ranking_group = ttk.LabelFrame(league_panel, text=" Agent Ranking", style="Dark.TFrame")
        ranking_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview pour le ranking
        columns = ("Rank", "Agent", "ELO", "W/L", "Games")
        self.ranking_tree = ttk.Treeview(ranking_group, columns=columns, show="headings", height=15)

        for col in columns:
            self.ranking_tree.heading(col, text=col)
            self.ranking_tree.column(col, width=60)

        scrollbar_ranking = ttk.Scrollbar(ranking_group, orient=tk.VERTICAL, command=self.ranking_tree.yview)
        self.ranking_tree.configure(yscrollcommand=scrollbar_ranking.set)

        self.ranking_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ranking.pack(side=tk.RIGHT, fill=tk.Y)

        # === MATCH HISTORY ===
        history_group = ttk.LabelFrame(league_panel, text="[STATS] Recent Matches", style="Dark.TFrame")
        history_group.pack(fill=tk.X, padx=5, pady=5)

        # Liste des matchs récents
        self.match_listbox = tk.Listbox(history_group, height=6, bg="#1e1e1e", fg="white")
        scrollbar_matches = ttk.Scrollbar(history_group, orient=tk.VERTICAL, command=self.match_listbox.yview)
        self.match_listbox.configure(yscrollcommand=scrollbar_matches.set)

        self.match_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_matches.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_status_bar(self):
        """Configure la barre de statut"""
        status_frame = ttk.Frame(self.root, style="Dark.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Status items
        ttk.Label(status_frame, text="DOFUS AlphaStar 2025", style="Dark.TLabel").pack(side=tk.LEFT, padx=5)

        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.status_time = ttk.Label(status_frame, text="", style="Dark.TLabel")
        self.status_time.pack(side=tk.RIGHT, padx=5)

    # === ACTIONS ET CONTRÔLES ===

    def start_training(self):
        """Démarre l'entraînement"""
        if self.is_training:
            messagebox.showinfo("Info", "Entraînement déjà en cours")
            return

        try:
            # Créer trainer si nécessaire
            if self.trainer is None:
                self.trainer = create_rllib_trainer()
                self.trainer.setup_algorithm(self.algorithm_var.get())

            # Démarrer en arrière-plan
            self.is_training = True
            self.var_training_status.set("En cours")
            self.var_status.set("Entraînement actif")

            # Thread d'entraînement
            training_thread = threading.Thread(target=self._training_loop, daemon=True)
            training_thread.start()

            logger.info("Entraînement démarré")

        except Exception as e:
            logger.error(f"Erreur démarrage entraînement: {e}")
            messagebox.showerror("Erreur", f"Impossible de démarrer l'entraînement:\n{e}")
            self.is_training = False

    def stop_training(self):
        """Arrête l'entraînement"""
        if not self.is_training:
            return

        self.is_training = False
        self.var_training_status.set("Arrêté")
        self.var_status.set("Prêt")

        if self.trainer:
            self.trainer.stop_training()

        logger.info("Entraînement arrêté")

    def toggle_training(self):
        """Bascule pause/resume entraînement"""
        # Pour simplification, juste arrêter/démarrer
        if self.is_training:
            self.stop_training()
        else:
            self.start_training()

    def create_league(self):
        """Crée une nouvelle league"""
        try:
            if self.league_manager is None:
                self.league_manager = create_league_system(league_size=config.alphastar.league_size)
                logger.info("Nouvelle league créée")
                messagebox.showinfo("Succès", "Nouvelle league créée avec succès")
            else:
                result = messagebox.askyesno("Confirmation", "Une league existe déjà. La remplacer?")
                if result:
                    self.league_manager = create_league_system(league_size=config.alphastar.league_size)
                    logger.info("League remplacée")

        except Exception as e:
            logger.error(f"Erreur création league: {e}")
            messagebox.showerror("Erreur", f"Impossible de créer la league:\n{e}")

    def create_manual_match(self):
        """Crée un match manuel entre agents"""
        if not self.league_manager:
            messagebox.showwarning("Warning", "Aucune league active")
            return

        # Dialog pour sélectionner agents
        # Pour simplification, juste un message
        messagebox.showinfo("Manual Match", "Fonctionnalité à implémenter: sélection d'agents")

    # === ACTIONS MENU ===

    def new_config(self):
        """Nouvelle configuration"""
        messagebox.showinfo("Info", "Nouvelle configuration - à implémenter")

    def load_config(self):
        """Charge une configuration"""
        file_path = filedialog.askopenfilename(
            title="Charger configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"Configuration chargée: {file_path}")
                messagebox.showinfo("Succès", "Configuration chargée avec succès")
            except Exception as e:
                logger.error(f"Erreur chargement config: {e}")
                messagebox.showerror("Erreur", f"Impossible de charger la configuration:\n{e}")

    def save_config(self):
        """Sauvegarde la configuration"""
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json"
        )
        if file_path:
            try:
                config_data = {
                    "algorithm": self.algorithm_var.get(),
                    "learning_rate": self.lr_var.get(),
                    "timestamp": time.time()
                }
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Configuration sauvegardée: {file_path}")
                messagebox.showinfo("Succès", "Configuration sauvegardée avec succès")
            except Exception as e:
                logger.error(f"Erreur sauvegarde config: {e}")
                messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{e}")

    def export_metrics(self):
        """Exporte les métriques"""
        if not self.metrics_history:
            messagebox.showwarning("Warning", "Aucune métrique à exporter")
            return

        file_path = filedialog.asksaveasfilename(
            title="Exporter métriques",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")],
            defaultextension=".csv"
        )
        if file_path:
            try:
                # Conversion en DataFrame pandas
                df = pd.DataFrame([
                    {
                        "timestamp": m.timestamp,
                        "iteration": m.training_iteration,
                        "timesteps": m.total_timesteps,
                        "reward": m.episode_reward_mean,
                        "win_rate": m.win_rate,
                        "agents": m.agents_active
                    }
                    for m in self.metrics_history
                ])

                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_json(file_path, indent=2)

                logger.info(f"Métriques exportées: {file_path}")
                messagebox.showinfo("Succès", "Métriques exportées avec succès")

            except Exception as e:
                logger.error(f"Erreur export: {e}")
                messagebox.showerror("Erreur", f"Impossible d'exporter:\n{e}")

    def quit_dashboard(self):
        """Ferme le dashboard"""
        if self.is_training:
            result = messagebox.askyesno("Confirmation",
                                       "Entraînement en cours. Vraiment quitter?")
            if not result:
                return

        self.stop_training()
        self.is_monitoring = False

        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()

        logger.info("Dashboard fermé")

    def toggle_fullscreen(self):
        """Bascule plein écran"""
        current_state = self.root.wm_attributes('-fullscreen')
        self.root.wm_attributes('-fullscreen', not current_state)

    def reset_layout(self):
        """Reset le layout"""
        self.root.geometry("1400x900")
        messagebox.showinfo("Info", "Layout réinitialisé")

    def reset_league(self):
        """Reset la league"""
        if self.league_manager:
            result = messagebox.askyesno("Confirmation", "Vraiment reset la league?")
            if result:
                self.league_manager = None
                logger.info("League reset")
                messagebox.showinfo("Info", "League reset avec succès")

    def clear_memory_cache(self):
        """Nettoie le cache mémoire"""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        messagebox.showinfo("Info", "Cache mémoire nettoyé")

    def force_garbage_collect(self):
        """Force le garbage collection"""
        import gc
        collected = gc.collect()
        messagebox.showinfo("Info", f"Garbage collection: {collected} objets collectés")

    # === MONITORING ET UPDATES ===

    def start_monitoring(self):
        """Démarre le monitoring en temps réel"""
        self.is_monitoring = True

        # Thread de monitoring
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()

        # Démarrer les updates UI
        self.update_ui()

    def _monitoring_loop(self):
        """Boucle de monitoring en arrière-plan"""
        while self.is_monitoring:
            try:
                # Collecter métriques
                metrics = self._collect_metrics()

                # Ajouter à la queue pour UI thread
                self.metrics_queue.put(metrics)

                time.sleep(1.0)  # Update chaque seconde

            except Exception as e:
                logger.error(f"Erreur monitoring: {e}")
                time.sleep(5.0)

    def _training_loop(self):
        """Boucle d'entraînement en arrière-plan"""
        try:
            while self.is_training and self.trainer:
                # Étape d'entraînement
                result = self.trainer.algorithm.train()

                # Mettre à jour métriques
                metrics = DashboardMetrics(
                    timestamp=time.time(),
                    training_iteration=result.get("training_iteration", 0),
                    total_timesteps=result.get("timesteps_total", 0),
                    episode_reward_mean=result.get("episode_reward_mean", 0.0),
                    win_rate=result.get("custom_metrics/win_rate_mean", 0.0),
                )

                self.metrics_queue.put(metrics)

                # Pause entre itérations
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Erreur training loop: {e}")
            self.is_training = False

    def _collect_metrics(self) -> DashboardMetrics:
        """Collecte les métriques actuelles"""
        import psutil

        # Métriques système
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB

        # Métriques league
        agents_active = 0
        matches_completed = 0
        if self.league_manager:
            agents_active = len(self.league_manager.agent_pool.agents)
            matches_completed = self.league_manager.total_games_played

        return DashboardMetrics(
            timestamp=time.time(),
            agents_active=agents_active,
            matches_completed=matches_completed,
            memory_usage_mb=memory_usage
        )

    def update_ui(self):
        """Met à jour l'interface utilisateur"""
        try:
            # Traiter métriques de la queue
            while not self.metrics_queue.empty():
                metrics = self.metrics_queue.get_nowait()
                self.metrics_history.append(metrics)

                # Limiter historique
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]

                # Mettre à jour variables UI
                self.var_total_timesteps.set(f"{metrics.total_timesteps:,}")
                self.var_episode_reward.set(f"{metrics.episode_reward_mean:.2f}")
                self.var_win_rate.set(f"{metrics.win_rate:.1%}")
                self.var_agents_count.set(str(metrics.agents_active))
                self.var_matches_completed.set(str(metrics.matches_completed))
                self.var_memory_usage.set(f"{metrics.memory_usage_mb:.0f} MB")

            # Mettre à jour graphiques
            self._update_charts()

            # Mettre à jour ranking
            self._update_ranking()

            # Mettre à jour status bar
            self.status_time.config(text=time.strftime("%H:%M:%S"))

        except Exception as e:
            logger.error(f"Erreur update UI: {e}")

        # Programmer prochaine mise à jour
        if self.is_monitoring:
            self.root.after(self.update_interval, self.update_ui)

    def _update_charts(self):
        """Met à jour les graphiques"""
        if not self.metrics_history:
            return

        try:
            # Données récentes (50 derniers points)
            recent_metrics = self.metrics_history[-50:]
            times = [m.timestamp - self.metrics_history[0].timestamp for m in recent_metrics]
            rewards = [m.episode_reward_mean for m in recent_metrics]
            timesteps = [m.total_timesteps for m in recent_metrics]
            win_rates = [m.win_rate for m in recent_metrics]
            memory = [m.memory_usage_mb for m in recent_metrics]

            # Update training charts
            self.ax_reward.clear()
            self.ax_reward.plot(times, rewards, 'g-', linewidth=2)
            self.ax_reward.set_title("Episode Reward", color='white')
            self.ax_reward.tick_params(colors='white')

            self.ax_steps.clear()
            self.ax_steps.plot(times, timesteps, 'b-', linewidth=2)
            self.ax_steps.set_title("Training Steps", color='white')
            self.ax_steps.tick_params(colors='white')

            self.ax_win_rate.clear()
            self.ax_win_rate.plot(times, win_rates, 'r-', linewidth=2)
            self.ax_win_rate.set_title("Win Rate", color='white')
            self.ax_win_rate.tick_params(colors='white')

            # Update performance charts
            self.ax_memory.clear()
            self.ax_memory.plot(times, memory, 'orange', linewidth=2)
            self.ax_memory.set_title("Memory Usage (MB)", color='white')
            self.ax_memory.tick_params(colors='white')

            # Redraw canvases
            self.training_canvas.draw()
            self.perf_canvas.draw()

        except Exception as e:
            logger.error(f"Erreur update charts: {e}")

    def _update_ranking(self):
        """Met à jour le ranking des agents"""
        if not self.league_manager:
            return

        try:
            # Clear existing items
            for item in self.ranking_tree.get_children():
                self.ranking_tree.delete(item)

            # Get ranking
            ranking = self.league_manager.get_league_ranking()

            # Insert new data
            for entry in ranking[:20]:  # Top 20
                values = (
                    entry["rank"],
                    entry["agent_id"][:15],  # Truncate long names
                    f"{entry['elo']:.0f}",
                    f"{entry['wins']}/{entry['losses']}",
                    entry["games_played"]
                )
                self.ranking_tree.insert("", tk.END, values=values)

        except Exception as e:
            logger.error(f"Erreur update ranking: {e}")

    def run(self):
        """Lance le dashboard"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Dashboard interrompu par Ctrl+C")
            self.quit_dashboard()

# Factory function
def create_dashboard() -> AlphaStarDashboard:
    """Crée un dashboard AlphaStar"""
    return AlphaStarDashboard()

# Lancement direct
if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.run()