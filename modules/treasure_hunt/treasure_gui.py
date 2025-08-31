"""
Interface graphique pour le système d'automatisation des chasses aux trésors DOFUS
GUI complète avec monitoring en temps réel et contrôles avancés
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import cv2
from PIL import Image, ImageTk

from .treasure_automation import TreasureHuntAutomation, TreasureHuntState, TreasureHuntType
from .hint_database import HintDatabase, HintType, HintDifficulty
from .map_navigator import NavigationState

logger = logging.getLogger(__name__)

class TreasureHuntGUI:
    """Interface graphique principale pour les chasses aux trésors"""
    
    def __init__(self, automation_system: TreasureHuntAutomation):
        self.automation = automation_system
        self.root = tk.Tk()
        self.root.title("DOFUS Treasure Hunt Automation - DofuBot")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variables d'interface
        self.is_running = False
        self.current_screenshot = None
        self.log_buffer = []
        self.max_log_lines = 1000
        
        # Configuration des couleurs et styles
        self.colors = {
            'bg_primary': '#2b2b2b',
            'bg_secondary': '#3c3c3c',
            'bg_tertiary': '#4d4d4d',
            'fg_primary': '#ffffff',
            'fg_secondary': '#cccccc',
            'accent_blue': '#4a9eff',
            'accent_green': '#4ade80',
            'accent_red': '#ef4444',
            'accent_orange': '#f97316'
        }
        
        self._setup_styles()
        self._create_widgets()
        self._setup_automation_callbacks()
        
        # Thread de mise à jour de l'interface
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.should_update = True
        self.update_thread.start()
        
        logger.info("Interface GUI initialisée")
    
    def _setup_styles(self):
        """Configure les styles d'interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configuration des styles personnalisés
        style.configure('Title.TLabel', 
                       background=self.colors['bg_primary'], 
                       foreground=self.colors['fg_primary'], 
                       font=('Arial', 16, 'bold'))
        
        style.configure('Status.TLabel', 
                       background=self.colors['bg_secondary'], 
                       foreground=self.colors['fg_secondary'], 
                       font=('Arial', 10))
        
        style.configure('Success.TLabel', 
                       background=self.colors['bg_secondary'], 
                       foreground=self.colors['accent_green'], 
                       font=('Arial', 10, 'bold'))
        
        style.configure('Error.TLabel', 
                       background=self.colors['bg_secondary'], 
                       foreground=self.colors['accent_red'], 
                       font=('Arial', 10, 'bold'))
    
    def _create_widgets(self):
        """Crée tous les widgets de l'interface"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre principal
        title_label = ttk.Label(main_frame, text="🏴‍☠️ DOFUS Treasure Hunt Automation", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 15))
        
        # Création du notebook (onglets)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglets
        self._create_control_tab()
        self._create_monitoring_tab()
        self._create_statistics_tab()
        self._create_database_tab()
        self._create_settings_tab()
    
    def _create_control_tab(self):
        """Onglet de contrôle principal"""
        control_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(control_frame, text="🎮 Contrôle")
        
        # Section état actuel
        status_frame = tk.LabelFrame(control_frame, text="📊 État Actuel", 
                                   bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Variables d'état
        self.status_var = tk.StringVar(value="Inactif")
        self.session_var = tk.StringVar(value="Aucune session")
        self.step_var = tk.StringVar(value="0/0")
        self.time_var = tk.StringVar(value="00:00:00")
        
        # Labels d'état
        ttk.Label(status_frame, text="État:", style='Status.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel')
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_frame, text="Session:", style='Status.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.session_var, style='Status.TLabel').grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_frame, text="Étape:", style='Status.TLabel').grid(row=0, column=2, sticky=tk.W, padx=15)
        ttk.Label(status_frame, textvariable=self.step_var, style='Status.TLabel').grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(status_frame, text="Temps:", style='Status.TLabel').grid(row=1, column=2, sticky=tk.W, padx=15)
        ttk.Label(status_frame, textvariable=self.time_var, style='Status.TLabel').grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Section contrôles
        control_buttons_frame = tk.LabelFrame(control_frame, text="🎯 Contrôles", 
                                            bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Type de chasse
        tk.Label(control_buttons_frame, text="Type de chasse:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.hunt_type_var = tk.StringVar(value="CLASSIC")
        hunt_type_combo = ttk.Combobox(control_buttons_frame, textvariable=self.hunt_type_var,
                                     values=["CLASSIC", "LEGENDARY", "WEEKLY", "DAILY", "EVENT"],
                                     state="readonly", width=15)
        hunt_type_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Nom du personnage
        tk.Label(control_buttons_frame, text="Personnage:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=0, column=2, sticky=tk.W, padx=15)
        
        self.character_var = tk.StringVar(value="MonPersonnage")
        character_entry = tk.Entry(control_buttons_frame, textvariable=self.character_var, width=20)
        character_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Boutons de contrôle
        button_frame = tk.Frame(control_buttons_frame, bg=self.colors['bg_secondary'])
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        self.start_btn = tk.Button(button_frame, text="▶️ Démarrer", command=self._start_hunt,
                                  bg=self.colors['accent_green'], fg='white', font=('Arial', 10, 'bold'))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(button_frame, text="⏸️ Pause", command=self._pause_hunt,
                                  bg=self.colors['accent_orange'], fg='white', font=('Arial', 10, 'bold'))
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ Arrêter", command=self._stop_hunt,
                                 bg=self.colors['accent_red'], fg='white', font=('Arial', 10, 'bold'))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Section indice actuel
        hint_frame = tk.LabelFrame(control_frame, text="🔍 Indice Actuel", 
                                 bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        hint_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Texte de l'indice
        self.hint_text = scrolledtext.ScrolledText(hint_frame, height=4, width=60,
                                                  bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.hint_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Solutions proposées
        solutions_frame = tk.Frame(hint_frame, bg=self.colors['bg_secondary'])
        solutions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(solutions_frame, text="Solutions:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).pack(anchor=tk.W)
        
        self.solutions_listbox = tk.Listbox(solutions_frame, height=3,
                                           bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.solutions_listbox.pack(fill=tk.X, pady=2)
        
        # Section logs
        log_frame = tk.LabelFrame(control_frame, text="📋 Logs", 
                                bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                 bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_monitoring_tab(self):
        """Onglet de monitoring en temps réel"""
        monitor_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(monitor_frame, text="📈 Monitoring")
        
        # Section aperçu écran
        screen_frame = tk.LabelFrame(monitor_frame, text="🖥️ Aperçu Écran", 
                                   bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        screen_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas pour l'aperçu d'écran
        self.screen_canvas = tk.Canvas(screen_frame, width=400, height=300, 
                                      bg=self.colors['bg_tertiary'])
        self.screen_canvas.pack(padx=5, pady=5)
        
        # Section progression
        progress_frame = tk.LabelFrame(monitor_frame, text="⏱️ Progression", 
                                     bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        progress_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Barres de progression
        tk.Label(progress_frame, text="Progression globale:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).pack(anchor=tk.W, padx=5)
        
        self.global_progress = ttk.Progressbar(progress_frame, mode='determinate', length=200)
        self.global_progress.pack(padx=5, pady=2)
        
        tk.Label(progress_frame, text="Navigation:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).pack(anchor=tk.W, padx=5, pady=(10,0))
        
        self.nav_progress = ttk.Progressbar(progress_frame, mode='determinate', length=200)
        self.nav_progress.pack(padx=5, pady=2)
        
        # Métriques temps réel
        metrics_frame = tk.LabelFrame(progress_frame, text="📊 Métriques", 
                                    bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        metrics_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.metrics_text = tk.Text(metrics_frame, height=15, width=25,
                                   bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.metrics_text.pack(padx=5, pady=5)
    
    def _create_statistics_tab(self):
        """Onglet des statistiques et graphiques"""
        stats_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(stats_frame, text="📊 Statistiques")
        
        # Section statistiques générales
        general_stats_frame = tk.LabelFrame(stats_frame, text="📈 Statistiques Générales", 
                                          bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        general_stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Création des graphiques
        self.stats_figure = Figure(figsize=(12, 6), facecolor=self.colors['bg_secondary'])
        self.stats_canvas = FigureCanvasTkAgg(self.stats_figure, stats_frame)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Sous-graphiques
        self.success_rate_ax = self.stats_figure.add_subplot(221)
        self.time_distribution_ax = self.stats_figure.add_subplot(222)
        self.reward_breakdown_ax = self.stats_figure.add_subplot(223)
        self.difficulty_analysis_ax = self.stats_figure.add_subplot(224)
        
        self._setup_statistics_plots()
        
        # Section historique des sessions
        history_frame = tk.LabelFrame(stats_frame, text="📜 Historique des Sessions", 
                                    bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tableau des sessions
        columns = ('ID', 'Type', 'Personnage', 'Début', 'Durée', 'Étapes', 'Succès', 'Récompenses')
        self.sessions_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.sessions_tree.heading(col, text=col)
            self.sessions_tree.column(col, width=100)
        
        sessions_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.sessions_tree.yview)
        self.sessions_tree.configure(yscrollcommand=sessions_scrollbar.set)
        
        self.sessions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sessions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def _create_database_tab(self):
        """Onglet de gestion de la base de données d'indices"""
        db_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(db_frame, text="🗄️ Base de Données")
        
        # Section gestion des indices
        hints_mgmt_frame = tk.LabelFrame(db_frame, text="🔍 Gestion des Indices", 
                                       bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        hints_mgmt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Boutons de gestion
        db_buttons_frame = tk.Frame(hints_mgmt_frame, bg=self.colors['bg_secondary'])
        db_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(db_buttons_frame, text="📥 Importer Base", command=self._import_hints_database,
                 bg=self.colors['accent_blue'], fg='white').pack(side=tk.LEFT, padx=2)
        
        tk.Button(db_buttons_frame, text="📤 Exporter Base", command=self._export_hints_database,
                 bg=self.colors['accent_blue'], fg='white').pack(side=tk.LEFT, padx=2)
        
        tk.Button(db_buttons_frame, text="🔄 Actualiser", command=self._refresh_hints_list,
                 bg=self.colors['accent_green'], fg='white').pack(side=tk.LEFT, padx=2)
        
        tk.Button(db_buttons_frame, text="➕ Nouvel Indice", command=self._add_new_hint,
                 bg=self.colors['accent_orange'], fg='white').pack(side=tk.LEFT, padx=2)
        
        # Liste des indices
        hints_columns = ('ID', 'Texte', 'Type', 'Difficulté', 'Zone', 'Taux Succès', 'Utilisations')
        self.hints_tree = ttk.Treeview(hints_mgmt_frame, columns=hints_columns, show='headings', height=15)
        
        for col in hints_columns:
            self.hints_tree.heading(col, text=col)
            self.hints_tree.column(col, width=120)
        
        hints_scrollbar = ttk.Scrollbar(hints_mgmt_frame, orient=tk.VERTICAL, command=self.hints_tree.yview)
        self.hints_tree.configure(yscrollcommand=hints_scrollbar.set)
        
        self.hints_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        hints_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Section statistiques de la base
        db_stats_frame = tk.LabelFrame(db_frame, text="📊 Statistiques de la Base", 
                                     bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        db_stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.db_stats_text = tk.Text(db_stats_frame, height=6, width=80,
                                    bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.db_stats_text.pack(padx=5, pady=5)
    
    def _create_settings_tab(self):
        """Onglet de paramètres et configuration"""
        settings_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(settings_frame, text="⚙️ Paramètres")
        
        # Section paramètres généraux
        general_settings_frame = tk.LabelFrame(settings_frame, text="🔧 Paramètres Généraux", 
                                             bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        general_settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Variables de paramètres
        self.auto_fight_var = tk.BooleanVar(value=True)
        self.auto_collect_var = tk.BooleanVar(value=True)
        self.save_screenshots_var = tk.BooleanVar(value=True)
        self.max_attempts_var = tk.IntVar(value=3)
        self.step_timeout_var = tk.IntVar(value=300)
        
        # Checkboxes
        tk.Checkbutton(general_settings_frame, text="Combat automatique", 
                      variable=self.auto_fight_var, bg=self.colors['bg_secondary'], 
                      fg=self.colors['fg_secondary']).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        tk.Checkbutton(general_settings_frame, text="Collecte automatique des récompenses", 
                      variable=self.auto_collect_var, bg=self.colors['bg_secondary'], 
                      fg=self.colors['fg_secondary']).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        tk.Checkbutton(general_settings_frame, text="Sauvegarder les captures d'écran", 
                      variable=self.save_screenshots_var, bg=self.colors['bg_secondary'], 
                      fg=self.colors['fg_secondary']).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Paramètres numériques
        tk.Label(general_settings_frame, text="Tentatives max par étape:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=0, column=1, sticky=tk.W, padx=15)
        tk.Spinbox(general_settings_frame, from_=1, to=10, textvariable=self.max_attempts_var, width=5).grid(row=0, column=2, padx=5)
        
        tk.Label(general_settings_frame, text="Timeout étape (secondes):", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=1, column=1, sticky=tk.W, padx=15)
        tk.Spinbox(general_settings_frame, from_=60, to=600, textvariable=self.step_timeout_var, width=5).grid(row=1, column=2, padx=5)
        
        # Bouton sauvegarder paramètres
        tk.Button(general_settings_frame, text="💾 Sauvegarder Paramètres", command=self._save_settings,
                 bg=self.colors['accent_green'], fg='white').grid(row=2, column=2, padx=5, pady=5)
        
        # Section chemins et dossiers
        paths_frame = tk.LabelFrame(settings_frame, text="📁 Chemins et Dossiers", 
                                  bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        paths_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Variables de chemins
        self.screenshots_path_var = tk.StringVar(value="screenshots/treasure_hunts/")
        self.database_path_var = tk.StringVar(value="treasure_hunt_hints.db")
        
        tk.Label(paths_frame, text="Dossier captures:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=0, column=0, sticky=tk.W, padx=5)
        tk.Entry(paths_frame, textvariable=self.screenshots_path_var, width=40).grid(row=0, column=1, padx=5, pady=2)
        tk.Button(paths_frame, text="📁", command=lambda: self._browse_folder(self.screenshots_path_var)).grid(row=0, column=2, padx=2)
        
        tk.Label(paths_frame, text="Base de données:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_secondary']).grid(row=1, column=0, sticky=tk.W, padx=5)
        tk.Entry(paths_frame, textvariable=self.database_path_var, width=40).grid(row=1, column=1, padx=5, pady=2)
        tk.Button(paths_frame, text="📄", command=lambda: self._browse_file(self.database_path_var)).grid(row=1, column=2, padx=2)
        
        # Section informations système
        system_info_frame = tk.LabelFrame(settings_frame, text="ℹ️ Informations Système", 
                                        bg=self.colors['bg_secondary'], fg=self.colors['fg_primary'])
        system_info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.system_info_text = tk.Text(system_info_frame, height=10,
                                       bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'])
        self.system_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._update_system_info()
    
    def _setup_statistics_plots(self):
        """Configure les graphiques de statistiques"""
        # Style des graphiques
        for ax in [self.success_rate_ax, self.time_distribution_ax, 
                   self.reward_breakdown_ax, self.difficulty_analysis_ax]:
            ax.set_facecolor(self.colors['bg_tertiary'])
            ax.tick_params(colors=self.colors['fg_secondary'])
            ax.xaxis.label.set_color(self.colors['fg_secondary'])
            ax.yaxis.label.set_color(self.colors['fg_secondary'])
            ax.title.set_color(self.colors['fg_primary'])
        
        # Titres des graphiques
        self.success_rate_ax.set_title('Taux de Réussite')
        self.time_distribution_ax.set_title('Distribution des Temps')
        self.reward_breakdown_ax.set_title('Répartition des Récompenses')
        self.difficulty_analysis_ax.set_title('Analyse par Difficulté')
        
        self.stats_figure.tight_layout()
    
    def _setup_automation_callbacks(self):
        """Configure les callbacks du système d'automatisation"""
        self.automation.register_callback('on_hunt_started', self._on_hunt_started)
        self.automation.register_callback('on_step_completed', self._on_step_completed)
        self.automation.register_callback('on_hint_solved', self._on_hint_solved)
        self.automation.register_callback('on_hunt_completed', self._on_hunt_completed)
        self.automation.register_callback('on_error', self._on_error)
        self.automation.register_callback('on_state_changed', self._on_state_changed)
    
    def _update_loop(self):
        """Boucle de mise à jour de l'interface"""
        while self.should_update:
            try:
                self._update_interface()
                time.sleep(1.0)  # Mise à jour chaque seconde
            except Exception as e:
                logger.error(f"Erreur mise à jour interface: {e}")
                time.sleep(1.0)
    
    def _update_interface(self):
        """Met à jour tous les éléments de l'interface"""
        # Mise à jour de l'état
        status = self.automation.get_current_status()
        
        # État général
        self.root.after(0, lambda: self.status_var.set(status['state'].title()))
        
        if status['session_active'] and status['current_session']:
            session = status['current_session']
            self.root.after(0, lambda: self.session_var.set(session['session_id']))
            self.root.after(0, lambda: self.step_var.set(f"{session['current_step'] + 1}/{len(session['steps'])}"))
            
            # Temps écoulé
            if session['start_time']:
                start_time = datetime.fromisoformat(session['start_time'])
                elapsed = datetime.now() - start_time
                time_str = str(elapsed).split('.')[0]  # Enlever les microsecondes
                self.root.after(0, lambda: self.time_var.set(time_str))
        else:
            self.root.after(0, lambda: self.session_var.set("Aucune session"))
            self.root.after(0, lambda: self.step_var.set("0/0"))
            self.root.after(0, lambda: self.time_var.set("00:00:00"))
        
        # Mise à jour des barres de progression
        nav_progress = status.get('navigation_progress', {})
        if nav_progress:
            progress_value = nav_progress.get('progress', 0.0) * 100
            self.root.after(0, lambda: self.nav_progress.configure(value=progress_value))
        
        # Mise à jour des métriques
        self._update_metrics_display(status)
    
    def _update_metrics_display(self, status: Dict[str, Any]):
        """Met à jour l'affichage des métriques"""
        def update_text():
            self.metrics_text.delete(1.0, tk.END)
            
            # Statistiques globales
            global_stats = status.get('global_statistics', {})
            
            metrics = [
                f"📊 MÉTRIQUES TEMPS RÉEL\n" + "="*25,
                f"🎯 Chasses démarrées: {global_stats.get('total_hunts_started', 0)}",
                f"✅ Chasses complétées: {global_stats.get('total_hunts_completed', 0)}",
                f"❌ Chasses échouées: {global_stats.get('total_hunts_failed', 0)}",
                f"⏱️ Temps moyen: {global_stats.get('average_completion_time', 0.0):.1f}s",
                "",
                "🏆 RÉCOMPENSES TOTALES",
                "-"*20
            ]
            
            total_rewards = global_stats.get('total_rewards_collected', {})
            for item, quantity in total_rewards.items():
                metrics.append(f"💰 {item}: {quantity}")
            
            if not total_rewards:
                metrics.append("Aucune récompense collectée")
            
            self.metrics_text.insert(tk.END, "\n".join(metrics))
        
        self.root.after(0, update_text)
    
    # Callbacks du système d'automatisation
    
    def _on_hunt_started(self, session):
        """Callback appelé au démarrage d'une chasse"""
        self.root.after(0, lambda: self._add_log(f"🏁 Chasse démarrée: {session.session_id}", "INFO"))
        self.is_running = True
        self._update_button_states()
    
    def _on_step_completed(self, step):
        """Callback appelé à la completion d'une étape"""
        self.root.after(0, lambda: self._add_log(f"✅ Étape {step.step_number} complétée", "SUCCESS"))
    
    def _on_hint_solved(self, step):
        """Callback appelé quand un indice est résolu"""
        def update_hint_display():
            self.hint_text.delete(1.0, tk.END)
            self.hint_text.insert(tk.END, f"Étape {step.step_number}: {step.hint_text}")
            
            self.solutions_listbox.delete(0, tk.END)
            if step.solution:
                solution_text = f"{step.solution.reasoning} (confiance: {step.solution.confidence:.2f})"
                self.solutions_listbox.insert(tk.END, solution_text)
        
        self.root.after(0, update_hint_display)
        self.root.after(0, lambda: self._add_log(f"🔍 Indice résolu: {step.solution.reasoning}", "INFO"))
    
    def _on_hunt_completed(self, session):
        """Callback appelé à la fin d'une chasse"""
        duration = session.statistics.get('duration', 0)
        self.root.after(0, lambda: self._add_log(f"🎉 Chasse terminée en {duration:.1f}s", "SUCCESS"))
        self.is_running = False
        self._update_button_states()
        self.root.after(0, self._refresh_statistics)
    
    def _on_error(self, error_msg):
        """Callback appelé en cas d'erreur"""
        self.root.after(0, lambda: self._add_log(f"❌ Erreur: {error_msg}", "ERROR"))
    
    def _on_state_changed(self, new_state):
        """Callback appelé lors d'un changement d'état"""
        state_display = {
            TreasureHuntState.INACTIVE: "Inactif",
            TreasureHuntState.STARTING: "Démarrage...",
            TreasureHuntState.READING_HINT: "Lecture indice",
            TreasureHuntState.SOLVING_HINT: "Résolution indice",
            TreasureHuntState.NAVIGATING: "Navigation",
            TreasureHuntState.VALIDATING_POSITION: "Validation position",
            TreasureHuntState.DIGGING: "Creusage",
            TreasureHuntState.FIGHTING: "Combat",
            TreasureHuntState.COLLECTING_REWARD: "Collecte récompense",
            TreasureHuntState.COMPLETED: "Terminé",
            TreasureHuntState.FAILED: "Échoué",
            TreasureHuntState.PAUSED: "En pause",
            TreasureHuntState.ERROR: "Erreur"
        }
        
        display_text = state_display.get(new_state, str(new_state))
        self.root.after(0, lambda: self.status_var.set(display_text))
    
    # Méthodes de contrôle
    
    def _start_hunt(self):
        """Démarre une chasse aux trésors"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une chasse est déjà en cours!")
            return
        
        hunt_type = TreasureHuntType(self.hunt_type_var.get())
        character_name = self.character_var.get()
        
        # Appliquer les paramètres
        self._apply_settings()
        
        success = self.automation.start_treasure_hunt(hunt_type, character_name)
        if success:
            self._add_log(f"Démarrage chasse {hunt_type.value} pour {character_name}", "INFO")
        else:
            messagebox.showerror("Erreur", "Impossible de démarrer la chasse!")
    
    def _pause_hunt(self):
        """Met en pause ou reprend la chasse"""
        if not self.is_running:
            return
        
        current_status = self.automation.get_current_status()
        if current_status['state'] == 'paused':
            self.automation.resume_automation()
            self.pause_btn.configure(text="⏸️ Pause")
            self._add_log("Chasse reprise", "INFO")
        else:
            self.automation.pause_automation()
            self.pause_btn.configure(text="▶️ Reprendre")
            self._add_log("Chasse mise en pause", "INFO")
    
    def _stop_hunt(self):
        """Arrête la chasse en cours"""
        if not self.is_running:
            return
        
        if messagebox.askyesno("Confirmation", "Voulez-vous vraiment arrêter la chasse en cours?"):
            self.automation.stop_automation()
            self.is_running = False
            self._update_button_states()
            self._add_log("Chasse arrêtée par l'utilisateur", "WARNING")
    
    def _update_button_states(self):
        """Met à jour l'état des boutons"""
        if self.is_running:
            self.start_btn.configure(state=tk.DISABLED)
            self.pause_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.NORMAL)
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self.pause_btn.configure(state=tk.DISABLED, text="⏸️ Pause")
            self.stop_btn.configure(state=tk.DISABLED)
    
    def _add_log(self, message: str, level: str = "INFO"):
        """Ajoute un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Couleurs selon le niveau
        color_map = {
            "INFO": self.colors['fg_secondary'],
            "SUCCESS": self.colors['accent_green'],
            "WARNING": self.colors['accent_orange'],
            "ERROR": self.colors['accent_red']
        }
        
        log_entry = f"[{timestamp}] {message}"
        self.log_buffer.append((log_entry, color_map.get(level, self.colors['fg_secondary'])))
        
        # Limiter le buffer
        if len(self.log_buffer) > self.max_log_lines:
            self.log_buffer = self.log_buffer[-self.max_log_lines:]
        
        # Mise à jour du widget de log
        def update_log():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            
            for entry, color in self.log_buffer:
                self.log_text.insert(tk.END, entry + "\n")
            
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)
        
        self.root.after(0, update_log)
    
    # Méthodes de gestion de la base de données
    
    def _import_hints_database(self):
        """Importe une base de données d'indices"""
        filename = filedialog.askopenfilename(
            title="Importer base d'indices",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                success = self.automation.hint_database.import_hints(filename)
                if success:
                    messagebox.showinfo("Succès", "Base d'indices importée avec succès!")
                    self._refresh_hints_list()
                else:
                    messagebox.showerror("Erreur", "Échec de l'importation")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'importation: {str(e)}")
    
    def _export_hints_database(self):
        """Exporte la base de données d'indices"""
        filename = filedialog.asksaveasfilename(
            title="Exporter base d'indices",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                success = self.automation.hint_database.export_hints(filename)
                if success:
                    messagebox.showinfo("Succès", "Base d'indices exportée avec succès!")
                else:
                    messagebox.showerror("Erreur", "Échec de l'exportation")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'exportation: {str(e)}")
    
    def _refresh_hints_list(self):
        """Actualise la liste des indices"""
        # Vider l'arbre actuel
        for item in self.hints_tree.get_children():
            self.hints_tree.delete(item)
        
        # Récupérer tous les indices
        try:
            # Simulation - en production, récupérer depuis la vraie base
            hints_stats = self.automation.hint_database.get_database_stats()
            
            # Afficher les statistiques
            stats_text = f"""📊 STATISTIQUES DE LA BASE D'INDICES

Total d'indices: {hints_stats.get('total_hints', 0)}

Répartition par type:
"""
            for hint_type, count in hints_stats.get('by_type', {}).items():
                stats_text += f"  • {hint_type}: {count}\n"
            
            stats_text += "\nIndices les plus utilisés:\n"
            for hint in hints_stats.get('most_used', []):
                stats_text += f"  • {hint['text'][:50]}... (utilisé {hint['usage_count']} fois)\n"
            
            self.db_stats_text.delete(1.0, tk.END)
            self.db_stats_text.insert(tk.END, stats_text)
            
        except Exception as e:
            logger.error(f"Erreur actualisation liste indices: {e}")
    
    def _add_new_hint(self):
        """Ouvre une fenêtre pour ajouter un nouvel indice"""
        # Fenêtre modale pour ajouter un indice
        hint_window = tk.Toplevel(self.root)
        hint_window.title("Nouvel Indice")
        hint_window.geometry("500x400")
        hint_window.configure(bg=self.colors['bg_secondary'])
        
        # Champs de saisie
        tk.Label(hint_window, text="Texte de l'indice:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_primary']).pack(anchor=tk.W, padx=10, pady=5)
        
        hint_text_entry = tk.Text(hint_window, height=3, width=50)
        hint_text_entry.pack(padx=10, pady=5)
        
        tk.Label(hint_window, text="Type d'indice:", 
                bg=self.colors['bg_secondary'], fg=self.colors['fg_primary']).pack(anchor=tk.W, padx=10, pady=5)
        
        hint_type_var = tk.StringVar(value="ELEMENT")
        hint_type_combo = ttk.Combobox(hint_window, textvariable=hint_type_var,
                                     values=["DIRECTION", "BUILDING", "NPC", "ELEMENT", "ZONE", "MONSTER", "OBJECT"])
        hint_type_combo.pack(padx=10, pady=5)
        
        # Boutons
        button_frame = tk.Frame(hint_window, bg=self.colors['bg_secondary'])
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="💾 Sauvegarder", 
                 command=lambda: self._save_new_hint(hint_window, hint_text_entry.get(1.0, tk.END).strip(), hint_type_var.get()),
                 bg=self.colors['accent_green'], fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Annuler", 
                 command=hint_window.destroy,
                 bg=self.colors['accent_red'], fg='white').pack(side=tk.LEFT, padx=5)
    
    def _save_new_hint(self, window, hint_text, hint_type):
        """Sauvegarde un nouvel indice"""
        if not hint_text.strip():
            messagebox.showerror("Erreur", "Le texte de l'indice ne peut pas être vide!")
            return
        
        try:
            from .hint_database import HintData, HintType, HintDifficulty
            import hashlib
            
            # Créer le nouvel indice
            hint_id = hashlib.md5(hint_text.encode('utf-8')).hexdigest()[:12]
            
            new_hint = HintData(
                id=hint_id,
                text=hint_text,
                hint_type=HintType(hint_type.lower()),
                difficulty=HintDifficulty.MEDIUM,
                map_coordinates=None,
                area_name="Zone inconnue",
                sub_area_name="",
                cell_id=None,
                description=f"Indice ajouté via GUI: {hint_text[:50]}",
                keywords=[],
                image_hash=None,
                image_data=None,
                success_rate=0.0,
                usage_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                validated=False,
                community_rating=0.0
            )
            
            # Ajouter à la base
            success = self.automation.hint_database.add_hint(new_hint)
            
            if success:
                messagebox.showinfo("Succès", "Nouvel indice ajouté avec succès!")
                window.destroy()
                self._refresh_hints_list()
            else:
                messagebox.showerror("Erreur", "Échec de l'ajout de l'indice")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    # Méthodes des paramètres
    
    def _apply_settings(self):
        """Applique les paramètres à l'automatisation"""
        self.automation.config.update({
            'auto_fight': self.auto_fight_var.get(),
            'auto_collect_rewards': self.auto_collect_var.get(),
            'save_screenshots': self.save_screenshots_var.get(),
            'max_attempts_per_step': self.max_attempts_var.get(),
            'step_timeout': self.step_timeout_var.get(),
            'screenshot_path': Path(self.screenshots_path_var.get())
        })
        
        self._add_log("Paramètres appliqués", "INFO")
    
    def _save_settings(self):
        """Sauvegarde les paramètres dans un fichier"""
        settings = {
            'auto_fight': self.auto_fight_var.get(),
            'auto_collect_rewards': self.auto_collect_var.get(),
            'save_screenshots': self.save_screenshots_var.get(),
            'max_attempts_per_step': self.max_attempts_var.get(),
            'step_timeout': self.step_timeout_var.get(),
            'screenshots_path': self.screenshots_path_var.get(),
            'database_path': self.database_path_var.get()
        }
        
        try:
            with open('treasure_hunt_settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Succès", "Paramètres sauvegardés!")
            self._apply_settings()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur sauvegarde paramètres: {str(e)}")
    
    def _browse_folder(self, var):
        """Ouvre un dialogue de sélection de dossier"""
        folder = filedialog.askdirectory(title="Sélectionner un dossier")
        if folder:
            var.set(folder)
    
    def _browse_file(self, var):
        """Ouvre un dialogue de sélection de fichier"""
        file = filedialog.askopenfilename(title="Sélectionner un fichier")
        if file:
            var.set(file)
    
    def _update_system_info(self):
        """Met à jour les informations système"""
        import platform
        import psutil
        
        system_info = f"""🖥️ INFORMATIONS SYSTÈME

Système d'exploitation: {platform.system()} {platform.release()}
Architecture: {platform.architecture()[0]}
Processeur: {platform.processor()}
RAM totale: {psutil.virtual_memory().total / (1024**3):.1f} GB
RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB

📊 ÉTAT DU SYSTÈME D'AUTOMATISATION

Version: 1.0.0
Modules chargés:
  • Base de données d'indices: ✅
  • Solveur intelligent: ✅
  • Navigateur de carte: ✅
  • Système d'automatisation: ✅
  • Interface graphique: ✅

🔧 CONFIGURATION ACTUELLE

Chemin base de données: {self.automation.hint_database.db_path}
Threads actifs: {threading.active_count()}
Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.system_info_text.delete(1.0, tk.END)
        self.system_info_text.insert(tk.END, system_info)
    
    def _refresh_statistics(self):
        """Actualise les graphiques de statistiques"""
        try:
            # Récupération des données
            global_stats = self.automation.global_stats
            session_history = self.automation.get_session_history(100)
            
            # Graphique du taux de succès
            self.success_rate_ax.clear()
            if global_stats['total_hunts_started'] > 0:
                success_rate = global_stats['total_hunts_completed'] / global_stats['total_hunts_started']
                self.success_rate_ax.pie([success_rate, 1-success_rate], 
                                       labels=['Succès', 'Échecs'],
                                       colors=[self.colors['accent_green'], self.colors['accent_red']])
            
            # Graphique de distribution des temps
            self.time_distribution_ax.clear()
            if session_history:
                completion_times = [s.get('statistics_data', {}).get('duration', 0) for s in session_history if s.get('success')]
                if completion_times:
                    self.time_distribution_ax.hist(completion_times, bins=10, 
                                                  color=self.colors['accent_blue'], alpha=0.7)
                    self.time_distribution_ax.set_xlabel('Temps (secondes)')
                    self.time_distribution_ax.set_ylabel('Fréquence')
            
            # Actualiser l'affichage
            self.stats_canvas.draw()
            
            # Mise à jour du tableau des sessions
            for item in self.sessions_tree.get_children():
                self.sessions_tree.delete(item)
            
            for session in session_history[:20]:  # Limiter aux 20 dernières
                start_time = datetime.fromisoformat(session['start_time']).strftime('%H:%M')
                duration = session.get('statistics_data', {}).get('duration', 0)
                success_icon = "✅" if session.get('success') else "❌"
                
                self.sessions_tree.insert('', tk.END, values=(
                    session['session_id'][:8],
                    session['hunt_type'],
                    session['character_name'],
                    start_time,
                    f"{duration:.0f}s",
                    len(session.get('steps_data', [])),
                    success_icon,
                    str(session.get('rewards_data', {}))
                ))
                
        except Exception as e:
            logger.error(f"Erreur actualisation statistiques: {e}")
    
    def run(self):
        """Lance l'interface graphique"""
        try:
            self._add_log("Interface graphique démarrée", "INFO")
            self._update_button_states()
            self._refresh_hints_list()
            self._refresh_statistics()
            
            # Boucle principale
            self.root.mainloop()
            
        except KeyboardInterrupt:
            logger.info("Interruption clavier détectée")
        except Exception as e:
            logger.error(f"Erreur interface graphique: {e}")
            messagebox.showerror("Erreur Critique", f"Erreur dans l'interface: {str(e)}")
        finally:
            self.close()
    
    def close(self):
        """Ferme l'interface et nettoie les ressources"""
        self.should_update = False
        
        if self.is_running:
            self.automation.stop_automation()
        
        logger.info("Interface graphique fermée")


# Point d'entrée principal
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Mock handlers pour test
    def mock_click_handler(x: int, y: int):
        print(f"Clic simulé à ({x}, {y})")
    
    def mock_screen_capture() -> np.ndarray:
        return np.zeros((800, 600, 3), dtype=np.uint8)
    
    try:
        # Initialisation du système complet
        automation_system = TreasureHuntAutomation(mock_click_handler, mock_screen_capture)
        
        # Lancement de l'interface
        gui = TreasureHuntGUI(automation_system)
        gui.run()
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        if 'automation_system' in locals():
            automation_system.close()
    
    logger.info("Application fermée")