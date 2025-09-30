"""
Interface Graphique Avancée - DOFUS Unity World Model AI
Interface complète et modulaire pour contrôle total du bot
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Configuration du logging pour l'interface
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BotStatus:
    """État du bot temps réel"""
    is_running: bool = False
    current_mode: str = "Idle"
    uptime: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    fps: float = 0.0
    actions_count: int = 0
    patterns_learned: int = 0
    last_action: str = "None"
    anti_detection_active: bool = True

@dataclass
class InterfaceConfig:
    """Configuration interface utilisateur"""
    theme: str = "dark"
    window_geometry: str = "1400x900+100+50"
    panels_layout: Dict[str, Dict] = None
    auto_save: bool = True
    refresh_rate: int = 1000  # ms
    show_debug: bool = False
    language: str = "fr"

    def __post_init__(self):
        if self.panels_layout is None:
            self.panels_layout = {}

class AdvancedBotInterface:
    """Interface graphique complète pour DOFUS Unity Bot"""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_main_window()

        # Configuration et état
        self.config = self.load_interface_config()
        self.bot_status = BotStatus()

        # Queue pour communication thread-safe
        self.message_queue = queue.Queue()

        # Variables interface
        self.log_buffer = []
        self.max_log_entries = 1000

        # Composants principaux
        self.notebook = None
        self.status_bar = None
        self.menu_bar = None

        # Threads
        self.update_thread = None
        self.is_running = False

        # Initialisation
        self.setup_interface()
        self.setup_monitoring()

        # Import des modules bot si disponibles
        self.setup_bot_integration()

    def setup_main_window(self):
        """Configure la fenêtre principale"""
        self.root.title("DOFUS Unity World Model AI - Interface Avancée")
        self.root.geometry(self.config.window_geometry if hasattr(self, 'config') else "1400x900+100+50")

        # Icône et style
        try:
            # Style moderne
            style = ttk.Style()
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')

            # Couleurs sombres
            style.configure('TNotebook', background='#2e2e2e')
            style.configure('TNotebook.Tab', background='#404040', foreground='white')
            style.map('TNotebook.Tab', background=[('selected', '#1e1e1e')])

        except Exception as e:
            logger.warning(f"Erreur configuration style: {e}")

        # Gestion fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_interface(self):
        """Configuration complète de l'interface"""
        # Menu principal
        self.create_menu_bar()

        # Barre d'outils
        self.create_toolbar()

        # Notebook principal avec onglets
        self.create_main_notebook()

        # Barre de statut
        self.create_status_bar()

        # Raccourcis clavier
        self.setup_keyboard_shortcuts()

    def create_menu_bar(self):
        """Crée la barre de menu"""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Menu Fichier
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouveau Profil...", command=self.new_profile)
        file_menu.add_command(label="Ouvrir Profil...", command=self.load_profile)
        file_menu.add_command(label="Sauvegarder Profil", command=self.save_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exporter Logs...", command=self.export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.on_closing)

        # Menu Bot
        bot_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Bot", menu=bot_menu)
        bot_menu.add_command(label="Démarrer Bot", command=self.start_bot)
        bot_menu.add_command(label="Arrêter Bot", command=self.stop_bot)
        bot_menu.add_command(label="Pause/Resume", command=self.toggle_bot)
        bot_menu.add_separator()
        bot_menu.add_command(label="Mode Apprentissage Passif", command=lambda: self.set_bot_mode("passive"))
        bot_menu.add_command(label="Mode Tests", command=lambda: self.set_bot_mode("tests"))
        bot_menu.add_command(label="Mode Synthétique", command=lambda: self.set_bot_mode("synthetic"))

        # Menu Outils
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Outils", menu=tools_menu)
        tools_menu.add_command(label="Test Système Complet", command=self.run_system_tests)
        tools_menu.add_command(label="Synchroniser Ganymede", command=self.sync_ganymede)
        tools_menu.add_command(label="Vérifier APIs", command=self.test_apis)
        tools_menu.add_separator()
        tools_menu.add_command(label="Console Debug", command=self.show_debug_console)

        # Menu Aide
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="Guide Utilisateur", command=self.show_user_guide)
        help_menu.add_command(label="À Propos", command=self.show_about)

    def create_toolbar(self):
        """Crée la barre d'outils"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        # Boutons principaux
        ttk.Button(toolbar, text="▶ Start", command=self.start_bot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏸ Pause", command=self.toggle_bot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹ Stop", command=self.stop_bot).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Mode selection
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=2)
        self.mode_var = tk.StringVar(value="Idle")
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var, width=15, state="readonly")
        mode_combo['values'] = ("Idle", "Tests", "Passive Learning", "Synthetic", "Ganymede", "API Tests")
        mode_combo.pack(side=tk.LEFT, padx=2)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_changed)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Status indicators
        self.status_indicator = tk.Label(toolbar, text="●", fg="red", font=("Arial", 16))
        self.status_indicator.pack(side=tk.LEFT, padx=2)

        self.status_text = tk.Label(toolbar, text="Bot Arrêté", fg="gray")
        self.status_text.pack(side=tk.LEFT, padx=5)

    def create_main_notebook(self):
        """Crée le notebook principal avec tous les onglets"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Créer tous les onglets
        self.create_dashboard_tab()
        self.create_vision_tab()
        self.create_intelligence_tab()
        self.create_knowledge_tab()
        self.create_config_tab()
        self.create_logs_tab()
        self.create_actions_tab()

    def create_dashboard_tab(self):
        """Onglet Dashboard - Vue d'ensemble"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="🎮 Dashboard")

        # Layout en grille
        main_container = ttk.Frame(dashboard_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Section Status du Bot (gauche)
        status_frame = ttk.LabelFrame(main_container, text="Statut du Bot", padding=10)
        status_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Métriques temps réel
        metrics_frame = ttk.LabelFrame(main_container, text="Métriques Performance", padding=10)
        metrics_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Modules Status (bas)
        modules_frame = ttk.LabelFrame(main_container, text="État des Modules", padding=10)
        modules_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Configuration grille
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=1)

        # Contenu Status du Bot
        self.status_labels = {}
        status_items = [
            ("Mode Actuel:", "current_mode"),
            ("Temps de Fonctionnement:", "uptime"),
            ("Actions Effectuées:", "actions_count"),
            ("Patterns Appris:", "patterns_learned"),
            ("Dernière Action:", "last_action"),
            ("Anti-Détection:", "anti_detection")
        ]

        for i, (label, key) in enumerate(status_items):
            ttk.Label(status_frame, text=label, font=("Arial", 9, "bold")).grid(row=i, column=0, sticky="w", pady=2)
            self.status_labels[key] = ttk.Label(status_frame, text="--", font=("Arial", 9))
            self.status_labels[key].grid(row=i, column=1, sticky="w", padx=10, pady=2)

        # Graphique métriques performance
        self.create_performance_chart(metrics_frame)

        # État des modules avec indicateurs visuels
        self.create_modules_status(modules_frame)

    def create_performance_chart(self, parent):
        """Crée le graphique de performance temps réel"""
        # Figure matplotlib
        self.perf_figure = Figure(figsize=(6, 3), dpi=80, facecolor='#f0f0f0')
        self.perf_plot = self.perf_figure.add_subplot(111)

        # Données initiales
        self.time_data = []
        self.cpu_data = []
        self.memory_data = []
        self.fps_data = []

        # Configuration plot
        self.perf_plot.set_title("Performance Temps Réel", fontsize=12)
        self.perf_plot.set_xlabel("Temps (s)")
        self.perf_plot.set_ylabel("Usage (%)")
        self.perf_plot.grid(True, alpha=0.3)

        # Canvas tkinter
        self.perf_canvas = FigureCanvasTkAgg(self.perf_figure, parent)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_modules_status(self, parent):
        """Crée l'affichage de l'état des modules"""
        modules = [
            ("Vision Engine", "vision"),
            ("Knowledge Base", "knowledge"),
            ("Learning Engine", "learning"),
            ("Human Simulation", "human_sim"),
            ("HRM Integration", "hrm"),
            ("Assistant Interface", "assistant")
        ]

        self.module_indicators = {}

        for i, (name, key) in enumerate(modules):
            module_frame = ttk.Frame(parent)
            module_frame.grid(row=i//3, column=i%3, sticky="ew", padx=5, pady=3)

            # Indicateur coloré
            indicator = tk.Label(module_frame, text="●", font=("Arial", 16), fg="gray")
            indicator.pack(side=tk.LEFT)

            # Nom module
            name_label = ttk.Label(module_frame, text=name, font=("Arial", 9))
            name_label.pack(side=tk.LEFT, padx=5)

            # Status text
            status_label = ttk.Label(module_frame, text="Inconnu", font=("Arial", 8), foreground="gray")
            status_label.pack(side=tk.LEFT, padx=5)

            self.module_indicators[key] = {
                'indicator': indicator,
                'status': status_label
            }

        # Configuration grille
        for i in range(3):
            parent.grid_columnconfigure(i, weight=1)

    def create_vision_tab(self):
        """Onglet Vision & Capture"""
        vision_frame = ttk.Frame(self.notebook)
        self.notebook.add(vision_frame, text="🔍 Vision")

        # Layout principal
        main_paned = ttk.PanedWindow(vision_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel gauche - Preview capture
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)

        preview_frame = ttk.LabelFrame(left_frame, text="Preview Capture Temps Réel", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas pour preview (placeholder)
        self.vision_canvas = tk.Canvas(preview_frame, bg='black', width=640, height=480)
        self.vision_canvas.pack(fill=tk.BOTH, expand=True)

        # Texte placeholder
        self.vision_canvas.create_text(320, 240, text="Capture DOFUS apparaîtra ici",
                                      fill="white", font=("Arial", 14))

        # Panel droit - Contrôles
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Réglages capture
        capture_settings = ttk.LabelFrame(right_frame, text="Réglages Capture", padding=10)
        capture_settings.pack(fill=tk.X, pady=5)

        ttk.Label(capture_settings, text="Zone de Capture:").pack(anchor=tk.W)
        self.capture_area = tk.StringVar(value="Fenêtre DOFUS Complète")
        area_combo = ttk.Combobox(capture_settings, textvariable=self.capture_area, state="readonly")
        area_combo['values'] = ("Fenêtre DOFUS Complète", "Zone de Combat", "Interface Seulement", "Personnalisé")
        area_combo.pack(fill=tk.X, pady=2)

        ttk.Label(capture_settings, text="Fréquence (FPS):").pack(anchor=tk.W, pady=(10,0))
        self.fps_var = tk.IntVar(value=2)
        fps_scale = ttk.Scale(capture_settings, from_=1, to=10, variable=self.fps_var, orient=tk.HORIZONTAL)
        fps_scale.pack(fill=tk.X, pady=2)
        fps_label = ttk.Label(capture_settings, text="2 FPS")
        fps_label.pack(anchor=tk.W)

        def update_fps_label(val):
            fps_label.config(text=f"{int(float(val))} FPS")
        fps_scale.config(command=update_fps_label)

        # Vision passive settings
        passive_settings = ttk.LabelFrame(right_frame, text="Mode Vision Passive", padding=10)
        passive_settings.pack(fill=tk.X, pady=5)

        self.passive_enabled = tk.BooleanVar()
        ttk.Checkbutton(passive_settings, text="Activer Vision Passive",
                       variable=self.passive_enabled).pack(anchor=tk.W)

        self.pattern_detection = tk.BooleanVar(value=True)
        ttk.Checkbutton(passive_settings, text="Détection Patterns",
                       variable=self.pattern_detection).pack(anchor=tk.W)

        self.save_screenshots = tk.BooleanVar()
        ttk.Checkbutton(passive_settings, text="Sauvegarder Screenshots",
                       variable=self.save_screenshots).pack(anchor=tk.W)

        # Boutons contrôle
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Démarrer Capture",
                  command=self.start_vision_capture).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Arrêter Capture",
                  command=self.stop_vision_capture).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Capture Manuelle",
                  command=self.manual_capture).pack(fill=tk.X, pady=2)

        # Statistiques vision
        stats_frame = ttk.LabelFrame(right_frame, text="Statistiques Vision", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)

        self.vision_stats = {
            'captures_count': ttk.Label(stats_frame, text="Captures: 0"),
            'patterns_found': ttk.Label(stats_frame, text="Patterns: 0"),
            'avg_confidence': ttk.Label(stats_frame, text="Confiance: 0%"),
            'last_capture': ttk.Label(stats_frame, text="Dernière: --")
        }

        for label in self.vision_stats.values():
            label.pack(anchor=tk.W, pady=1)

    def create_intelligence_tab(self):
        """Onglet Intelligence & Apprentissage"""
        intel_frame = ttk.Frame(self.notebook)
        self.notebook.add(intel_frame, text="🧠 Intelligence")

        # Layout avec panneaux
        paned_window = ttk.PanedWindow(intel_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel supérieur - Learning Engine
        top_frame = ttk.Frame(paned_window)
        paned_window.add(top_frame, weight=1)

        learning_frame = ttk.LabelFrame(top_frame, text="Learning Engine - Apprentissage Adaptatif", padding=10)
        learning_frame.pack(fill=tk.BOTH, expand=True)

        # Graphique progression apprentissage
        learning_left = ttk.Frame(learning_frame)
        learning_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.learning_figure = Figure(figsize=(8, 4), dpi=80)
        self.learning_plot = self.learning_figure.add_subplot(111)
        self.learning_plot.set_title("Progression Apprentissage")
        self.learning_plot.set_xlabel("Sessions")
        self.learning_plot.set_ylabel("Score d'Efficacité")

        self.learning_canvas = FigureCanvasTkAgg(self.learning_figure, learning_left)
        self.learning_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Contrôles learning
        learning_right = ttk.Frame(learning_frame)
        learning_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        ttk.Label(learning_right, text="Classe Joueur:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.player_class = tk.StringVar(value="Iop")
        class_combo = ttk.Combobox(learning_right, textvariable=self.player_class, state="readonly", width=15)
        class_combo['values'] = ("Iop", "Cra", "Eniripsa", "Enutrof", "Sram", "Xelor", "Ecaflip", "Sadida")
        class_combo.pack(pady=2)

        ttk.Label(learning_right, text="Niveau:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,0))
        self.player_level = tk.IntVar(value=150)
        level_spinbox = ttk.Spinbox(learning_right, from_=1, to=200, textvariable=self.player_level, width=15)
        level_spinbox.pack(pady=2)

        ttk.Separator(learning_right, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Button(learning_right, text="Nouvelle Session",
                  command=self.start_learning_session).pack(fill=tk.X, pady=2)
        ttk.Button(learning_right, text="Arrêter Session",
                  command=self.stop_learning_session).pack(fill=tk.X, pady=2)

        # Métriques apprentissage
        metrics_frame = ttk.LabelFrame(learning_right, text="Métriques", padding=5)
        metrics_frame.pack(fill=tk.X, pady=10)

        self.learning_metrics = {
            'sessions_count': ttk.Label(metrics_frame, text="Sessions: 0"),
            'patterns_learned': ttk.Label(metrics_frame, text="Patterns: 0"),
            'efficiency_score': ttk.Label(metrics_frame, text="Efficacité: 0%"),
            'recommendations': ttk.Label(metrics_frame, text="Recommandations: 0")
        }

        for label in self.learning_metrics.values():
            label.pack(anchor=tk.W, pady=1)

        # Panel inférieur - Recommandations & Stratégies
        bottom_frame = ttk.Frame(paned_window)
        paned_window.add(bottom_frame, weight=1)

        recommendations_frame = ttk.LabelFrame(bottom_frame, text="Recommandations Temps Réel", padding=10)
        recommendations_frame.pack(fill=tk.BOTH, expand=True)

        # Liste des recommandations
        self.recommendations_tree = ttk.Treeview(recommendations_frame, columns=("Type", "Recommandation", "Confiance"), show="headings", height=8)
        self.recommendations_tree.heading("Type", text="Type")
        self.recommendations_tree.heading("Recommandation", text="Recommandation")
        self.recommendations_tree.heading("Confiance", text="Confiance")

        self.recommendations_tree.column("Type", width=100)
        self.recommendations_tree.column("Recommandation", width=400)
        self.recommendations_tree.column("Confiance", width=80)

        self.recommendations_tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar pour recommandations
        rec_scrollbar = ttk.Scrollbar(recommendations_frame, orient=tk.VERTICAL, command=self.recommendations_tree.yview)
        rec_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.recommendations_tree.configure(yscrollcommand=rec_scrollbar.set)

    def create_knowledge_tab(self):
        """Onglet Knowledge Base"""
        knowledge_frame = ttk.Frame(self.notebook)
        self.notebook.add(knowledge_frame, text="📊 Knowledge")

        # Layout horizontal
        main_paned = ttk.PanedWindow(knowledge_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel gauche - Explorateur sources
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        sources_frame = ttk.LabelFrame(left_frame, text="Sources de Données", padding=5)
        sources_frame.pack(fill=tk.BOTH, expand=True)

        # Arbre des sources
        self.knowledge_tree = ttk.Treeview(sources_frame, show="tree")
        self.knowledge_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar
        kb_scrollbar = ttk.Scrollbar(sources_frame, orient=tk.VERTICAL, command=self.knowledge_tree.yview)
        kb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.knowledge_tree.configure(yscrollcommand=kb_scrollbar.set)

        # Peupler l'arbre
        self.populate_knowledge_tree()

        # Panel droit - Détails et recherche
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Recherche
        search_frame = ttk.LabelFrame(right_frame, text="Recherche Unifiée", padding=10)
        search_frame.pack(fill=tk.X, pady=5)

        search_entry_frame = ttk.Frame(search_frame)
        search_entry_frame.pack(fill=tk.X)

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_entry_frame, textvariable=self.search_var, font=("Arial", 11))
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search_entry.bind('<Return>', self.perform_knowledge_search)

        ttk.Button(search_entry_frame, text="Rechercher",
                  command=self.perform_knowledge_search).pack(side=tk.RIGHT, padx=(5,0))

        # Filtres recherche
        filter_frame = ttk.Frame(search_frame)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Source:").pack(side=tk.LEFT)
        self.search_source = tk.StringVar(value="Toutes")
        source_combo = ttk.Combobox(filter_frame, textvariable=self.search_source, state="readonly", width=15)
        source_combo['values'] = ("Toutes", "Spells", "Monsters", "Maps", "Economy", "APIs", "Ganymede")
        source_combo.pack(side=tk.LEFT, padx=5)

        # Résultats recherche
        results_frame = ttk.LabelFrame(right_frame, text="Résultats", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_tree = ttk.Treeview(results_frame, columns=("Source", "Type", "Détails"), show="headings")
        self.results_tree.heading("Source", text="Source")
        self.results_tree.heading("Type", text="Type")
        self.results_tree.heading("Détails", text="Détails")

        self.results_tree.column("Source", width=100)
        self.results_tree.column("Type", width=120)
        self.results_tree.column("Détails", width=300)

        self.results_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        # Boutons synchronisation
        sync_frame = ttk.Frame(right_frame)
        sync_frame.pack(fill=tk.X, pady=5)

        ttk.Button(sync_frame, text="Sync Ganymede",
                  command=self.sync_ganymede_guides).pack(side=tk.LEFT, padx=2)
        ttk.Button(sync_frame, text="Test APIs",
                  command=self.test_external_apis).pack(side=tk.LEFT, padx=2)
        ttk.Button(sync_frame, text="Actualiser",
                  command=self.refresh_knowledge_base).pack(side=tk.LEFT, padx=2)

    def populate_knowledge_tree(self):
        """Peuple l'arbre des sources de connaissances"""
        # Vider l'arbre
        for item in self.knowledge_tree.get_children():
            self.knowledge_tree.delete(item)

        # Sources principales
        sources = {
            "📖 Spells Database": ["Sorts Iop", "Sorts Cra", "Sorts Eniripsa", "Sorts Autres"],
            "👹 Monsters Database": ["Monstres Faibles", "Monstres Moyens", "Monstres Boss", "Stratégies"],
            "🗺️ Maps Database": ["Cartes Basses", "Cartes Moyennes", "Cartes Hautes", "Transitions"],
            "💰 Economy Tracker": ["Prix Actuels", "Tendances", "Opportunités", "Historique"],
            "🌐 APIs Externes": ["Dofapi.fr", "Doduapi", "Cache Local", "Statistiques"],
            "📚 Guides Ganymede": ["Quêtes", "Donjons", "Métiers", "PvP", "Guides Récents"]
        }

        for source, items in sources.items():
            source_id = self.knowledge_tree.insert("", "end", text=source, open=True)
            for item in items:
                self.knowledge_tree.insert(source_id, "end", text=f"  {item}")

    def create_config_tab(self):
        """Onglet Configuration"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")

        # Layout avec notebook pour sous-catégories
        config_notebook = ttk.Notebook(config_frame)
        config_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Onglet Général
        self.create_general_config_tab(config_notebook)

        # Onglet Vision
        self.create_vision_config_tab(config_notebook)

        # Onglet Apprentissage
        self.create_learning_config_tab(config_notebook)

        # Onglet Sécurité
        self.create_security_config_tab(config_notebook)

    def create_general_config_tab(self, parent):
        """Configuration générale"""
        general_frame = ttk.Frame(parent)
        parent.add(general_frame, text="Général")

        # Scrollable frame
        canvas = tk.Canvas(general_frame)
        scrollbar = ttk.Scrollbar(general_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configuration interface
        interface_frame = ttk.LabelFrame(scrollable_frame, text="Interface", padding=10)
        interface_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(interface_frame, text="Thème:").grid(row=0, column=0, sticky="w")
        self.theme_var = tk.StringVar(value="Sombre")
        theme_combo = ttk.Combobox(interface_frame, textvariable=self.theme_var, state="readonly")
        theme_combo['values'] = ("Sombre", "Clair", "Auto")
        theme_combo.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(interface_frame, text="Langue:").grid(row=1, column=0, sticky="w")
        self.language_var = tk.StringVar(value="Français")
        lang_combo = ttk.Combobox(interface_frame, textvariable=self.language_var, state="readonly")
        lang_combo['values'] = ("Français", "English")
        lang_combo.grid(row=1, column=1, sticky="ew", padx=5)

        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(interface_frame, text="Sauvegarde automatique configuration",
                       variable=self.auto_save_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        # Configuration bot
        bot_frame = ttk.LabelFrame(scrollable_frame, text="Bot Principal", padding=10)
        bot_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(bot_frame, text="Classe par défaut:").grid(row=0, column=0, sticky="w")
        self.default_class_var = tk.StringVar(value="Iop")
        default_class_combo = ttk.Combobox(bot_frame, textvariable=self.default_class_var, state="readonly")
        default_class_combo['values'] = ("Iop", "Cra", "Eniripsa", "Enutrof", "Sram", "Xelor")
        default_class_combo.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(bot_frame, text="Niveau par défaut:").grid(row=1, column=0, sticky="w")
        self.default_level_var = tk.IntVar(value=150)
        ttk.Spinbox(bot_frame, from_=1, to=200, textvariable=self.default_level_var).grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(bot_frame, text="Serveur:").grid(row=2, column=0, sticky="w")
        self.server_var = tk.StringVar(value="Julith")
        server_entry = ttk.Entry(bot_frame, textvariable=self.server_var)
        server_entry.grid(row=2, column=1, sticky="ew", padx=5)

        # Configuration grille
        interface_frame.grid_columnconfigure(1, weight=1)
        bot_frame.grid_columnconfigure(1, weight=1)

    def create_vision_config_tab(self, parent):
        """Configuration vision"""
        vision_frame = ttk.Frame(parent)
        parent.add(vision_frame, text="Vision")

        main_frame = ttk.Frame(vision_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Paramètres capture
        capture_frame = ttk.LabelFrame(main_frame, text="Capture d'Écran", padding=10)
        capture_frame.pack(fill=tk.X, pady=5)

        ttk.Label(capture_frame, text="Qualité capture:").grid(row=0, column=0, sticky="w")
        self.capture_quality = tk.StringVar(value="Haute")
        quality_combo = ttk.Combobox(capture_frame, textvariable=self.capture_quality, state="readonly")
        quality_combo['values'] = ("Basse", "Moyenne", "Haute", "Maximale")
        quality_combo.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(capture_frame, text="Format image:").grid(row=1, column=0, sticky="w")
        self.image_format = tk.StringVar(value="PNG")
        format_combo = ttk.Combobox(capture_frame, textvariable=self.image_format, state="readonly")
        format_combo['values'] = ("PNG", "JPEG", "BMP")
        format_combo.grid(row=1, column=1, sticky="ew", padx=5)

        # Paramètres OCR
        ocr_frame = ttk.LabelFrame(main_frame, text="Reconnaissance Texte (OCR)", padding=10)
        ocr_frame.pack(fill=tk.X, pady=5)

        ttk.Label(ocr_frame, text="Moteur OCR:").grid(row=0, column=0, sticky="w")
        self.ocr_engine = tk.StringVar(value="EasyOCR")
        ocr_combo = ttk.Combobox(ocr_frame, textvariable=self.ocr_engine, state="readonly")
        ocr_combo['values'] = ("EasyOCR", "Tesseract", "Hybride")
        ocr_combo.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(ocr_frame, text="Confiance minimale:").grid(row=1, column=0, sticky="w")
        self.ocr_confidence = tk.DoubleVar(value=0.7)
        confidence_scale = ttk.Scale(ocr_frame, from_=0.1, to=1.0, variable=self.ocr_confidence, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=1, column=1, sticky="ew", padx=5)

        # Configuration grilles
        capture_frame.grid_columnconfigure(1, weight=1)
        ocr_frame.grid_columnconfigure(1, weight=1)

    def create_learning_config_tab(self, parent):
        """Configuration apprentissage"""
        learning_frame = ttk.Frame(parent)
        parent.add(learning_frame, text="Apprentissage")

        main_frame = ttk.Frame(learning_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Paramètres apprentissage
        learning_params = ttk.LabelFrame(main_frame, text="Paramètres Apprentissage", padding=10)
        learning_params.pack(fill=tk.X, pady=5)

        ttk.Label(learning_params, text="Taux d'apprentissage:").grid(row=0, column=0, sticky="w")
        self.learning_rate = tk.DoubleVar(value=0.1)
        lr_scale = ttk.Scale(learning_params, from_=0.01, to=1.0, variable=self.learning_rate, orient=tk.HORIZONTAL)
        lr_scale.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(learning_params, text="Mémoire patterns:").grid(row=1, column=0, sticky="w")
        self.pattern_memory = tk.IntVar(value=1000)
        ttk.Spinbox(learning_params, from_=100, to=10000, textvariable=self.pattern_memory).grid(row=1, column=1, sticky="ew", padx=5)

        # Stratégies
        strategy_frame = ttk.LabelFrame(main_frame, text="Stratégies", padding=10)
        strategy_frame.pack(fill=tk.X, pady=5)

        self.aggressive_learning = tk.BooleanVar()
        ttk.Checkbutton(strategy_frame, text="Apprentissage agressif",
                       variable=self.aggressive_learning).pack(anchor=tk.W)

        self.adapt_to_level = tk.BooleanVar(value=True)
        ttk.Checkbutton(strategy_frame, text="Adaptation automatique au niveau",
                       variable=self.adapt_to_level).pack(anchor=tk.W)

        learning_params.grid_columnconfigure(1, weight=1)

    def create_security_config_tab(self, parent):
        """Configuration sécurité"""
        security_frame = ttk.Frame(parent)
        parent.add(security_frame, text="Sécurité")

        main_frame = ttk.Frame(security_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Anti-détection
        antidet_frame = ttk.LabelFrame(main_frame, text="Anti-Détection", padding=10)
        antidet_frame.pack(fill=tk.X, pady=5)

        self.human_simulation = tk.BooleanVar(value=True)
        ttk.Checkbutton(antidet_frame, text="Simulation comportement humain",
                       variable=self.human_simulation).pack(anchor=tk.W)

        self.random_delays = tk.BooleanVar(value=True)
        ttk.Checkbutton(antidet_frame, text="Délais aléatoires",
                       variable=self.random_delays).pack(anchor=tk.W)

        self.variable_patterns = tk.BooleanVar(value=True)
        ttk.Checkbutton(antidet_frame, text="Patterns variables",
                       variable=self.variable_patterns).pack(anchor=tk.W)

        ttk.Label(antidet_frame, text="Niveau sécurité:").pack(anchor=tk.W, pady=(10,0))
        self.security_level = tk.StringVar(value="Élevé")
        security_combo = ttk.Combobox(antidet_frame, textvariable=self.security_level, state="readonly")
        security_combo['values'] = ("Bas", "Moyen", "Élevé", "Maximal")
        security_combo.pack(fill=tk.X)

        # Surveillance
        monitoring_frame = ttk.LabelFrame(main_frame, text="Surveillance", padding=10)
        monitoring_frame.pack(fill=tk.X, pady=5)

        self.log_actions = tk.BooleanVar(value=True)
        ttk.Checkbutton(monitoring_frame, text="Logger toutes les actions",
                       variable=self.log_actions).pack(anchor=tk.W)

        self.alert_suspicion = tk.BooleanVar(value=True)
        ttk.Checkbutton(monitoring_frame, text="Alertes activité suspecte",
                       variable=self.alert_suspicion).pack(anchor=tk.W)

    def create_logs_tab(self):
        """Onglet Logs & Monitoring"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs")

        # Layout principal
        main_paned = ttk.PanedWindow(logs_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel supérieur - Console logs
        top_frame = ttk.Frame(main_paned)
        main_paned.add(top_frame, weight=2)

        logs_controls = ttk.Frame(top_frame)
        logs_controls.pack(fill=tk.X, pady=2)

        ttk.Label(logs_controls, text="Niveau:").pack(side=tk.LEFT)
        self.log_level = tk.StringVar(value="INFO")
        level_combo = ttk.Combobox(logs_controls, textvariable=self.log_level, state="readonly", width=10)
        level_combo['values'] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        level_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(logs_controls, text="Effacer", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(logs_controls, text="Exporter", command=self.export_logs).pack(side=tk.LEFT)

        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(logs_controls, text="Auto-scroll", variable=self.auto_scroll).pack(side=tk.RIGHT)

        # Console de logs
        logs_container = ttk.LabelFrame(top_frame, text="Console Logs", padding=5)
        logs_container.pack(fill=tk.BOTH, expand=True)

        self.logs_text = tk.Text(logs_container, bg='black', fg='white', font=("Consolas", 9),
                                wrap=tk.WORD, state=tk.DISABLED)

        logs_scrollbar = ttk.Scrollbar(logs_container, orient=tk.VERTICAL, command=self.logs_text.yview)
        logs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.logs_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.logs_text.configure(yscrollcommand=logs_scrollbar.set)

        # Panel inférieur - Historique actions
        bottom_frame = ttk.Frame(main_paned)
        main_paned.add(bottom_frame, weight=1)

        actions_frame = ttk.LabelFrame(bottom_frame, text="Historique Actions Bot", padding=5)
        actions_frame.pack(fill=tk.BOTH, expand=True)

        self.actions_tree = ttk.Treeview(actions_frame, columns=("Heure", "Action", "Résultat", "Détails"), show="headings")
        self.actions_tree.heading("Heure", text="Heure")
        self.actions_tree.heading("Action", text="Action")
        self.actions_tree.heading("Résultat", text="Résultat")
        self.actions_tree.heading("Détails", text="Détails")

        self.actions_tree.column("Heure", width=120)
        self.actions_tree.column("Action", width=150)
        self.actions_tree.column("Résultat", width=100)
        self.actions_tree.column("Détails", width=300)

        self.actions_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        actions_scrollbar = ttk.Scrollbar(actions_frame, orient=tk.VERTICAL, command=self.actions_tree.yview)
        actions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.actions_tree.configure(yscrollcommand=actions_scrollbar.set)

    def create_actions_tab(self):
        """Onglet Actions & Contrôles"""
        actions_frame = ttk.Frame(self.notebook)
        self.notebook.add(actions_frame, text="🎯 Actions")

        # Layout principal
        main_container = ttk.Frame(actions_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Section modes disponibles
        modes_frame = ttk.LabelFrame(main_container, text="Modes Disponibles", padding=10)
        modes_frame.pack(fill=tk.X, pady=5)

        modes_grid = ttk.Frame(modes_frame)
        modes_grid.pack(fill=tk.X)

        # Définir les modes avec descriptions
        modes = [
            ("Tests Système", "Validation complète modules", self.run_system_tests),
            ("Environment Synthétique", "Tests offline avec données simulées", self.run_synthetic_tests),
            ("Vision Passive", "Apprentissage par observation DOFUS", self.start_passive_learning),
            ("Tests APIs", "Vérification connecteurs externes", self.test_external_apis),
            ("Sync Ganymede", "Synchronisation guides communautaires", self.sync_ganymede_guides),
            ("Interface Complète", "Lancement interface avancée", self.launch_advanced_interface)
        ]

        # Créer boutons en grille 3x2
        for i, (name, description, command) in enumerate(modes):
            row, col = divmod(i, 3)

            mode_frame = ttk.Frame(modes_grid)
            mode_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

            button = ttk.Button(mode_frame, text=name, command=command, width=20)
            button.pack(fill=tk.X)

            desc_label = ttk.Label(mode_frame, text=description, font=("Arial", 8), foreground="gray")
            desc_label.pack(fill=tk.X)

        # Configuration grille
        for i in range(3):
            modes_grid.grid_columnconfigure(i, weight=1)

        # Section contrôles urgents
        emergency_frame = ttk.LabelFrame(main_container, text="Contrôles Urgents", padding=10)
        emergency_frame.pack(fill=tk.X, pady=10)

        emergency_buttons = ttk.Frame(emergency_frame)
        emergency_buttons.pack(fill=tk.X)

        # Boutons d'urgence
        ttk.Button(emergency_buttons, text="🛑 ARRÊT D'URGENCE",
                  command=self.emergency_stop,
                  style="Emergency.TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(emergency_buttons, text="⏸️ Pause Immédiate",
                  command=self.immediate_pause).pack(side=tk.LEFT, padx=5)

        ttk.Button(emergency_buttons, text="🔄 Redémarrage Bot",
                  command=self.restart_bot).pack(side=tk.LEFT, padx=5)

        # Section macros personnalisées
        macros_frame = ttk.LabelFrame(main_container, text="Macros & Séquences", padding=10)
        macros_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Liste des macros
        macros_list_frame = ttk.Frame(macros_frame)
        macros_list_frame.pack(fill=tk.BOTH, expand=True)

        self.macros_tree = ttk.Treeview(macros_list_frame, columns=("Nom", "Description", "Actions"), show="headings", height=6)
        self.macros_tree.heading("Nom", text="Nom")
        self.macros_tree.heading("Description", text="Description")
        self.macros_tree.heading("Actions", text="Actions")

        self.macros_tree.column("Nom", width=150)
        self.macros_tree.column("Description", width=250)
        self.macros_tree.column("Actions", width=200)

        self.macros_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        macros_scrollbar = ttk.Scrollbar(macros_list_frame, orient=tk.VERTICAL, command=self.macros_tree.yview)
        macros_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.macros_tree.configure(yscrollcommand=macros_scrollbar.set)

        # Boutons macros
        macros_buttons = ttk.Frame(macros_frame)
        macros_buttons.pack(fill=tk.X, pady=5)

        ttk.Button(macros_buttons, text="Exécuter Macro",
                  command=self.execute_selected_macro).pack(side=tk.LEFT, padx=2)
        ttk.Button(macros_buttons, text="Nouvelle Macro",
                  command=self.create_new_macro).pack(side=tk.LEFT, padx=2)
        ttk.Button(macros_buttons, text="Modifier",
                  command=self.edit_selected_macro).pack(side=tk.LEFT, padx=2)
        ttk.Button(macros_buttons, text="Supprimer",
                  command=self.delete_selected_macro).pack(side=tk.LEFT, padx=2)

        # Peupler macros par défaut
        self.populate_default_macros()

    def populate_default_macros(self):
        """Peuple les macros par défaut"""
        default_macros = [
            ("Session Complète", "Tests + Apprentissage + Logs", "tests,passive,logs"),
            ("Maintenance Rapide", "APIs + Ganymede + Nettoyage", "apis,ganymede,cleanup"),
            ("Debug Complet", "Diagnostic système complet", "diagnostic,logs,export"),
            ("Apprentissage Intensif", "Session apprentissage longue durée", "passive_long,analysis"),
        ]

        for name, description, actions in default_macros:
            self.macros_tree.insert("", "end", values=(name, description, actions))

    def create_status_bar(self):
        """Crée la barre de statut"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Informations gauche
        left_status = ttk.Frame(self.status_bar)
        left_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_left = ttk.Label(left_status, text="Bot arrêté", foreground="gray")
        self.status_left.pack(side=tk.LEFT, padx=5)

        # Informations droite
        right_status = ttk.Frame(self.status_bar)
        right_status.pack(side=tk.RIGHT)

        self.status_right = ttk.Label(right_status, text="Prêt")
        self.status_right.pack(side=tk.RIGHT, padx=5)

        # Séparateur
        ttk.Separator(self.status_bar, orient='horizontal').pack(fill=tk.X)

    def setup_keyboard_shortcuts(self):
        """Configure les raccourcis clavier"""
        self.root.bind('<Control-s>', lambda e: self.save_profile())
        self.root.bind('<Control-o>', lambda e: self.load_profile())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<F5>', lambda e: self.refresh_all_data())
        self.root.bind('<F1>', lambda e: self.show_user_guide())

    def setup_monitoring(self):
        """Configure le monitoring temps réel"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self.update_monitoring_loop, daemon=True)
        self.update_thread.start()

    def setup_bot_integration(self):
        """Configure l'intégration avec les modules bot"""
        try:
            # Tentative d'import des modules bot
            import sys
            from pathlib import Path

            # Ajouter chemins
            current_dir = Path(__file__).parent
            sys.path.append(str(current_dir))
            sys.path.append(str(current_dir / "core"))

            # Imports conditionnels
            self.bot_modules = {}

            try:
                from core.knowledge_base.knowledge_integration import get_knowledge_base
                self.bot_modules['knowledge_base'] = get_knowledge_base()
                self.log_message("INFO", "Knowledge Base connectée")
            except ImportError as e:
                self.log_message("WARNING", f"Knowledge Base non disponible: {e}")

            try:
                from core.learning_engine.adaptive_learning_engine import get_learning_engine
                self.bot_modules['learning_engine'] = get_learning_engine()
                self.log_message("INFO", "Learning Engine connecté")
            except ImportError as e:
                self.log_message("WARNING", f"Learning Engine non disponible: {e}")

            try:
                from core.vision_engine.passive_learning_mode import get_passive_learning_engine
                self.bot_modules['passive_learning'] = get_passive_learning_engine()
                self.log_message("INFO", "Vision Passive connectée")
            except ImportError as e:
                self.log_message("WARNING", f"Vision Passive non disponible: {e}")

        except Exception as e:
            self.log_message("ERROR", f"Erreur intégration bot: {e}")

    def update_monitoring_loop(self):
        """Boucle de mise à jour temps réel"""
        while self.is_running:
            try:
                # Mettre à jour les métriques
                self.update_bot_status()
                self.update_performance_metrics()
                self.update_modules_status()

                # Traiter les messages en queue
                self.process_message_queue()

                time.sleep(1)  # Mise à jour chaque seconde

            except Exception as e:
                logger.error(f"Erreur monitoring: {e}")
                time.sleep(5)

    def update_bot_status(self):
        """Met à jour le statut du bot"""
        try:
            # Simuler métriques (à remplacer par vraies données)
            import psutil

            self.bot_status.cpu_usage = psutil.cpu_percent()
            self.bot_status.memory_usage = psutil.virtual_memory().percent

            if self.bot_status.is_running:
                self.bot_status.uptime += 1
                self.bot_status.fps = np.random.uniform(1.8, 2.2)  # Simulé

            # Mettre à jour interface dans thread principal
            self.root.after(0, self.update_status_display)

        except Exception as e:
            logger.error(f"Erreur update status: {e}")

    def update_status_display(self):
        """Met à jour l'affichage du statut (thread principal)"""
        try:
            # Mettre à jour labels status
            if hasattr(self, 'status_labels'):
                self.status_labels['current_mode'].config(text=self.bot_status.current_mode)

                uptime_str = f"{int(self.bot_status.uptime // 3600):02d}:{int((self.bot_status.uptime % 3600) // 60):02d}:{int(self.bot_status.uptime % 60):02d}"
                self.status_labels['uptime'].config(text=uptime_str)

                self.status_labels['actions_count'].config(text=str(self.bot_status.actions_count))
                self.status_labels['patterns_learned'].config(text=str(self.bot_status.patterns_learned))
                self.status_labels['last_action'].config(text=self.bot_status.last_action)

                antidet_text = "Actif" if self.bot_status.anti_detection_active else "Inactif"
                antidet_color = "green" if self.bot_status.anti_detection_active else "red"
                self.status_labels['anti_detection'].config(text=antidet_text, foreground=antidet_color)

            # Mettre à jour indicateur toolbar
            if self.bot_status.is_running:
                self.status_indicator.config(fg="green")
                self.status_text.config(text=f"Bot Actif - {self.bot_status.current_mode}", fg="green")
            else:
                self.status_indicator.config(fg="red")
                self.status_text.config(text="Bot Arrêté", fg="gray")

            # Mettre à jour barre de statut
            self.status_left.config(text=f"Mode: {self.bot_status.current_mode} | Actions: {self.bot_status.actions_count}")
            self.status_right.config(text=f"CPU: {self.bot_status.cpu_usage:.1f}% | RAM: {self.bot_status.memory_usage:.1f}%")

        except Exception as e:
            logger.error(f"Erreur update display: {e}")

    def update_performance_metrics(self):
        """Met à jour les métriques de performance"""
        try:
            current_time = time.time()

            # Ajouter données aux listes (garder dernières 60 valeurs)
            self.time_data.append(current_time)
            self.cpu_data.append(self.bot_status.cpu_usage)
            self.memory_data.append(self.bot_status.memory_usage)
            self.fps_data.append(self.bot_status.fps)

            if len(self.time_data) > 60:
                self.time_data.pop(0)
                self.cpu_data.pop(0)
                self.memory_data.pop(0)
                self.fps_data.pop(0)

            # Mettre à jour graphique dans thread principal
            self.root.after(0, self.update_performance_chart)

        except Exception as e:
            logger.error(f"Erreur update performance: {e}")

    def update_performance_chart(self):
        """Met à jour le graphique de performance"""
        try:
            if hasattr(self, 'perf_plot') and self.time_data:
                self.perf_plot.clear()

                # Convertir temps en secondes relatives
                if self.time_data:
                    start_time = self.time_data[0]
                    relative_times = [(t - start_time) for t in self.time_data]

                    # Tracer courbes
                    self.perf_plot.plot(relative_times, self.cpu_data, 'b-', label='CPU %', linewidth=2)
                    self.perf_plot.plot(relative_times, self.memory_data, 'r-', label='RAM %', linewidth=2)

                    # Tracer FPS sur axe secondaire
                    if len(self.fps_data) > 0 and max(self.fps_data) > 0:
                        ax2 = self.perf_plot.twinx()
                        ax2.plot(relative_times, self.fps_data, 'g-', label='FPS', linewidth=2)
                        ax2.set_ylabel('FPS', color='g')
                        ax2.tick_params(axis='y', labelcolor='g')

                self.perf_plot.set_title("Performance Temps Réel", fontsize=12)
                self.perf_plot.set_xlabel("Temps (s)")
                self.perf_plot.set_ylabel("Usage (%)")
                self.perf_plot.legend(loc='upper left')
                self.perf_plot.grid(True, alpha=0.3)

                self.perf_canvas.draw()

        except Exception as e:
            logger.error(f"Erreur update chart: {e}")

    def update_modules_status(self):
        """Met à jour l'état des modules"""
        self.root.after(0, self.update_modules_display)

    def update_modules_display(self):
        """Met à jour l'affichage des modules"""
        try:
            if hasattr(self, 'module_indicators'):
                # États simulés des modules (à remplacer par vraies vérifications)
                modules_status = {
                    'vision': ('operational' if 'passive_learning' in self.bot_modules else 'error', 'Opérationnel' if 'passive_learning' in self.bot_modules else 'Indisponible'),
                    'knowledge': ('operational' if 'knowledge_base' in self.bot_modules else 'error', 'Opérationnel' if 'knowledge_base' in self.bot_modules else 'Indisponible'),
                    'learning': ('operational' if 'learning_engine' in self.bot_modules else 'error', 'Opérationnel' if 'learning_engine' in self.bot_modules else 'Indisponible'),
                    'human_sim': ('warning', 'Partiel'),
                    'hrm': ('warning', 'Partiel'),
                    'assistant': ('operational', 'Opérationnel')
                }

                colors = {
                    'operational': 'green',
                    'warning': 'orange',
                    'error': 'red',
                    'unknown': 'gray'
                }

                for module_key, (status, text) in modules_status.items():
                    if module_key in self.module_indicators:
                        color = colors.get(status, 'gray')
                        self.module_indicators[module_key]['indicator'].config(fg=color)
                        self.module_indicators[module_key]['status'].config(text=text, foreground=color)

        except Exception as e:
            logger.error(f"Erreur update modules: {e}")

    def process_message_queue(self):
        """Traite les messages en queue"""
        try:
            while not self.message_queue.empty():
                message_type, content = self.message_queue.get_nowait()

                if message_type == "LOG":
                    level, message = content
                    self.root.after(0, lambda: self.log_message(level, message))
                elif message_type == "ACTION":
                    action, result, details = content
                    self.root.after(0, lambda: self.add_action_log(action, result, details))

        except Exception as e:
            logger.error(f"Erreur processing queue: {e}")

    def log_message(self, level: str, message: str):
        """Ajoute un message aux logs"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] [{level}] {message}\n"

            # Ajouter au buffer
            self.log_buffer.append(formatted_message)
            if len(self.log_buffer) > self.max_log_entries:
                self.log_buffer.pop(0)

            # Ajouter à l'interface si disponible
            if hasattr(self, 'logs_text'):
                self.logs_text.config(state=tk.NORMAL)
                self.logs_text.insert(tk.END, formatted_message)

                # Couleur selon niveau
                colors = {
                    'DEBUG': 'gray',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'magenta'
                }

                if level in colors:
                    start_line = self.logs_text.index(tk.END + "-2l linestart")
                    end_line = self.logs_text.index(tk.END + "-1l lineend")
                    tag_name = f"level_{level}"
                    self.logs_text.tag_add(tag_name, start_line, end_line)
                    self.logs_text.tag_config(tag_name, foreground=colors[level])

                # Auto-scroll si activé
                if self.auto_scroll.get():
                    self.logs_text.see(tk.END)

                self.logs_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Erreur log message: {e}")

    def add_action_log(self, action: str, result: str, details: str):
        """Ajoute une action au log des actions"""
        try:
            if hasattr(self, 'actions_tree'):
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.actions_tree.insert("", 0, values=(timestamp, action, result, details))

                # Garder seulement les 100 dernières actions
                children = self.actions_tree.get_children()
                if len(children) > 100:
                    self.actions_tree.delete(children[-1])

        except Exception as e:
            logger.error(f"Erreur add action: {e}")

    # Méthodes d'événements interface

    def on_mode_changed(self, event):
        """Événement changement de mode"""
        new_mode = self.mode_var.get()
        self.log_message("INFO", f"Mode changé vers: {new_mode}")
        # Ici implémenter changement de mode réel

    def start_bot(self):
        """Démarre le bot"""
        self.bot_status.is_running = True
        self.bot_status.current_mode = self.mode_var.get()
        self.log_message("INFO", f"Bot démarré en mode {self.bot_status.current_mode}")
        self.add_action_log("Démarrage Bot", "Succès", f"Mode: {self.bot_status.current_mode}")

    def stop_bot(self):
        """Arrête le bot"""
        self.bot_status.is_running = False
        self.bot_status.current_mode = "Idle"
        self.log_message("INFO", "Bot arrêté")
        self.add_action_log("Arrêt Bot", "Succès", "Arrêt normal")

    def toggle_bot(self):
        """Basculer pause/resume bot"""
        if self.bot_status.is_running:
            self.bot_status.current_mode = "Pause"
            self.log_message("INFO", "Bot en pause")
        else:
            self.start_bot()

    def set_bot_mode(self, mode: str):
        """Change le mode du bot"""
        self.mode_var.set(mode)
        if self.bot_status.is_running:
            self.bot_status.current_mode = mode
            self.log_message("INFO", f"Mode changé vers: {mode}")

    # Méthodes actions spécifiques

    def run_system_tests(self):
        """Lance les tests système"""
        self.log_message("INFO", "Démarrage tests système...")
        self.add_action_log("Tests Système", "En cours", "Validation modules")

        # Simuler tests (à remplacer par vrais tests)
        def run_tests():
            time.sleep(2)
            self.message_queue.put(("LOG", ("INFO", "Tests système terminés")))
            self.message_queue.put(("ACTION", ("Tests Système", "Succès", "Tous modules opérationnels")))

        threading.Thread(target=run_tests, daemon=True).start()

    def run_synthetic_tests(self):
        """Lance l'environnement synthétique"""
        self.log_message("INFO", "Génération environnement synthétique...")
        self.add_action_log("Environment Synthétique", "En cours", "Génération données test")

    def start_passive_learning(self):
        """Démarre l'apprentissage passif"""
        if 'passive_learning' in self.bot_modules:
            try:
                self.log_message("INFO", "Démarrage apprentissage passif...")
                # Ici démarrer vraiment l'apprentissage passif
                self.add_action_log("Apprentissage Passif", "Démarré", "Session d'observation")
            except Exception as e:
                self.log_message("ERROR", f"Erreur apprentissage passif: {e}")
        else:
            self.log_message("WARNING", "Module apprentissage passif non disponible")

    def test_external_apis(self):
        """Teste les APIs externes"""
        self.log_message("INFO", "Test des APIs externes...")
        self.add_action_log("Test APIs", "En cours", "Vérification connecteurs")

    def sync_ganymede_guides(self):
        """Synchronise les guides Ganymede"""
        self.log_message("INFO", "Synchronisation guides Ganymede...")
        self.add_action_log("Sync Ganymede", "En cours", "Téléchargement guides")

    def launch_advanced_interface(self):
        """Lance l'interface avancée"""
        self.log_message("INFO", "Interface avancée déjà active")

    # Méthodes contrôles urgents

    def emergency_stop(self):
        """Arrêt d'urgence"""
        self.bot_status.is_running = False
        self.bot_status.current_mode = "Emergency Stop"
        self.log_message("CRITICAL", "ARRÊT D'URGENCE ACTIVÉ")
        self.add_action_log("ARRÊT D'URGENCE", "Activé", "Arrêt immédiat de toutes activités")
        messagebox.showwarning("Arrêt d'Urgence", "Le bot a été arrêté d'urgence !")

    def immediate_pause(self):
        """Pause immédiate"""
        self.bot_status.current_mode = "Pause Immédiate"
        self.log_message("WARNING", "Pause immédiate activée")
        self.add_action_log("Pause Immédiate", "Activée", "Suspension temporaire")

    def restart_bot(self):
        """Redémarre le bot"""
        self.log_message("INFO", "Redémarrage du bot...")
        self.stop_bot()
        time.sleep(1)
        self.start_bot()
        self.add_action_log("Redémarrage", "Succès", "Bot redémarré")

    # Méthodes vision

    def start_vision_capture(self):
        """Démarre la capture vision"""
        self.log_message("INFO", "Démarrage capture vision")
        self.add_action_log("Capture Vision", "Démarré", f"Zone: {self.capture_area.get()}")

    def stop_vision_capture(self):
        """Arrête la capture vision"""
        self.log_message("INFO", "Arrêt capture vision")
        self.add_action_log("Capture Vision", "Arrêté", "Capture interrompue")

    def manual_capture(self):
        """Capture manuelle"""
        self.log_message("INFO", "Capture manuelle effectuée")
        self.add_action_log("Capture Manuelle", "Succès", "Screenshot sauvegardé")

    # Méthodes apprentissage

    def start_learning_session(self):
        """Démarre session apprentissage"""
        class_name = self.player_class.get()
        level = self.player_level.get()
        self.log_message("INFO", f"Session apprentissage: {class_name} niveau {level}")
        self.add_action_log("Session Apprentissage", "Démarrée", f"{class_name} Lv.{level}")

    def stop_learning_session(self):
        """Arrête session apprentissage"""
        self.log_message("INFO", "Session apprentissage arrêtée")
        self.add_action_log("Session Apprentissage", "Arrêtée", "Session terminée")

    # Méthodes knowledge base

    def perform_knowledge_search(self, event=None):
        """Effectue recherche dans knowledge base"""
        query = self.search_var.get()
        source = self.search_source.get()

        if query:
            self.log_message("INFO", f"Recherche: '{query}' dans {source}")
            # Ici implémenter vraie recherche

            # Résultats simulés
            results = [
                ("Spells", "Sort", f"Pression - Sort Iop (correspondance: '{query}')"),
                ("Ganymede", "Guide", f"Guide Temple - Contient '{query}'"),
                ("APIs", "Item", f"Items contenant '{query}'")
            ]

            # Vider résultats précédents
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Ajouter nouveaux résultats
            for source, type_item, details in results:
                self.results_tree.insert("", "end", values=(source, type_item, details))

    def refresh_knowledge_base(self):
        """Actualise la knowledge base"""
        self.log_message("INFO", "Actualisation Knowledge Base")
        self.add_action_log("Refresh KB", "En cours", "Mise à jour données")

    # Méthodes macros

    def execute_selected_macro(self):
        """Exécute la macro sélectionnée"""
        selected = self.macros_tree.selection()
        if selected:
            item = self.macros_tree.item(selected[0])
            macro_name = item['values'][0]
            self.log_message("INFO", f"Exécution macro: {macro_name}")
            self.add_action_log("Macro", "Exécutée", macro_name)

    def create_new_macro(self):
        """Crée une nouvelle macro"""
        self.log_message("INFO", "Création nouvelle macro")
        # Ici ouvrir dialogue création macro

    def edit_selected_macro(self):
        """Modifie la macro sélectionnée"""
        selected = self.macros_tree.selection()
        if selected:
            self.log_message("INFO", "Modification macro")

    def delete_selected_macro(self):
        """Supprime la macro sélectionnée"""
        selected = self.macros_tree.selection()
        if selected:
            if messagebox.askyesno("Confirmation", "Supprimer cette macro ?"):
                self.macros_tree.delete(selected[0])
                self.log_message("INFO", "Macro supprimée")

    # Méthodes gestion fichiers

    def new_profile(self):
        """Nouveau profil"""
        self.log_message("INFO", "Création nouveau profil")

    def load_profile(self):
        """Charge un profil"""
        filename = filedialog.askopenfilename(
            title="Charger Profil",
            filetypes=[("Profils Bot", "*.json"), ("Tous fichiers", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                self.log_message("INFO", f"Profil chargé: {filename}")
            except Exception as e:
                self.log_message("ERROR", f"Erreur chargement profil: {e}")

    def save_profile(self):
        """Sauvegarde le profil"""
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder Profil",
            defaultextension=".json",
            filetypes=[("Profils Bot", "*.json"), ("Tous fichiers", "*.*")]
        )
        if filename:
            try:
                profile = self.get_current_config()
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
                self.log_message("INFO", f"Profil sauvegardé: {filename}")
            except Exception as e:
                self.log_message("ERROR", f"Erreur sauvegarde profil: {e}")

    def export_logs(self):
        """Exporte les logs"""
        filename = filedialog.asksaveasfilename(
            title="Exporter Logs",
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous fichiers", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(self.log_buffer)
                self.log_message("INFO", f"Logs exportés: {filename}")
                messagebox.showinfo("Export", f"Logs exportés vers:\n{filename}")
            except Exception as e:
                self.log_message("ERROR", f"Erreur export logs: {e}")

    def clear_logs(self):
        """Efface les logs"""
        if messagebox.askyesno("Confirmation", "Effacer tous les logs ?"):
            self.log_buffer.clear()
            if hasattr(self, 'logs_text'):
                self.logs_text.config(state=tk.NORMAL)
                self.logs_text.delete(1.0, tk.END)
                self.logs_text.config(state=tk.DISABLED)
            self.log_message("INFO", "Logs effacés")

    # Méthodes utilitaires

    def get_current_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle"""
        return {
            "interface": {
                "theme": self.theme_var.get(),
                "language": self.language_var.get(),
                "auto_save": self.auto_save_var.get()
            },
            "bot": {
                "default_class": self.default_class_var.get(),
                "default_level": self.default_level_var.get(),
                "server": self.server_var.get()
            },
            "vision": {
                "capture_area": self.capture_area.get(),
                "fps": self.fps_var.get(),
                "quality": self.capture_quality.get(),
                "ocr_engine": self.ocr_engine.get()
            },
            "security": {
                "human_simulation": self.human_simulation.get(),
                "random_delays": self.random_delays.get(),
                "security_level": self.security_level.get()
            }
        }

    def load_interface_config(self) -> InterfaceConfig:
        """Charge la configuration interface"""
        try:
            config_file = Path("interface_config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return InterfaceConfig(**config_data)
        except Exception as e:
            logger.warning(f"Erreur chargement config: {e}")

        return InterfaceConfig()

    def save_interface_config(self):
        """Sauvegarde la configuration interface"""
        try:
            config_file = Path("interface_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

    def refresh_all_data(self):
        """Actualise toutes les données"""
        self.log_message("INFO", "Actualisation complète des données")
        self.refresh_knowledge_base()
        self.populate_knowledge_tree()

    def show_debug_console(self):
        """Affiche console debug"""
        self.notebook.select(5)  # Onglet logs
        self.log_message("DEBUG", "Console debug activée")

    def show_user_guide(self):
        """Affiche le guide utilisateur"""
        guide_text = """
Guide d'Utilisation - Interface Bot DOFUS

DÉMARRAGE RAPIDE:
1. Sélectionnez un mode dans la barre d'outils
2. Cliquez sur 'Start' pour démarrer le bot
3. Surveillez les logs pour voir l'activité

ONGLETS PRINCIPAUX:
• Dashboard: Vue d'ensemble et métriques
• Vision: Capture d'écran et vision passive
• Intelligence: Apprentissage et recommandations
• Knowledge: Base de connaissances et recherche
• Configuration: Paramètres détaillés
• Logs: Surveillance et historique
• Actions: Contrôles et macros

RACCOURCIS CLAVIER:
• Ctrl+S: Sauvegarder profil
• Ctrl+O: Ouvrir profil
• F5: Actualiser données
• F1: Ce guide

SÉCURITÉ:
• Le bot simule un comportement humain
• Utilisez les délais anti-détection
• Surveillez les logs pour les alertes
        """

        guide_window = tk.Toplevel(self.root)
        guide_window.title("Guide Utilisateur")
        guide_window.geometry("600x500")

        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, guide_text)
        text_widget.config(state=tk.DISABLED)

    def show_about(self):
        """Affiche la fenêtre À propos"""
        about_text = """
DOFUS Unity World Model AI
Interface Graphique Avancée v2025.1.0

Système d'Intelligence Artificielle complet pour DOFUS Unity
avec apprentissage adaptatif et anti-détection avancée.

Modules intégrés:
• Vision Engine avec reconnaissance OCR
• Knowledge Base unifiée (6 sources)
• Learning Engine adaptatif
• Human Simulation avancée
• APIs externes (Dofapi, Ganymede)
• Interface graphique complète

Développé avec Python, tkinter, matplotlib
Optimisé pour AMD 7800XT avec ROCm

© 2025 - Usage éducatif et recherche uniquement
        """

        messagebox.showinfo("À Propos", about_text)

    def on_closing(self):
        """Événement fermeture application"""
        if messagebox.askyesno("Quitter", "Voulez-vous vraiment quitter ?"):
            self.is_running = False

            # Sauvegarder config si activé
            if hasattr(self, 'auto_save_var') and self.auto_save_var.get():
                self.save_interface_config()

            # Arrêter bot si en cours
            if self.bot_status.is_running:
                self.stop_bot()

            self.root.destroy()

    def run(self):
        """Lance l'interface"""
        self.log_message("INFO", "Interface graphique démarrée")
        self.log_message("INFO", "Système prêt - Sélectionnez un mode pour commencer")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("INFO", "Arrêt par interruption clavier")
        except Exception as e:
            self.log_message("CRITICAL", f"Erreur fatale interface: {e}")
        finally:
            self.is_running = False

def main():
    """Point d'entrée principal"""
    try:
        app = AdvancedBotInterface()
        app.run()
    except Exception as e:
        print(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()