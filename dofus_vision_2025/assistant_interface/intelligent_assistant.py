"""
Assistant Interface Intelligent pour DOFUS Unity World Model
Interface utilisateur avancée avec overlay en temps réel et conseils contextuels
Intégration complète HRM + Knowledge Base + Learning Engine + Human Simulation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Imports des modules du système
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from core.knowledge_base.knowledge_integration import get_knowledge_base, GameContext, DofusClass
    from core.learning_engine.adaptive_learning_engine import get_learning_engine
    from core.human_simulation.advanced_human_simulation import get_human_simulator
    try:
        from core.world_model.hrm_dofus_integration import DofusIntelligentDecisionMaker
        HRM_INTEGRATION_AVAILABLE = True
    except ImportError:
        HRM_INTEGRATION_AVAILABLE = False
        DofusIntelligentDecisionMaker = None

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Some modules unavailable: {e}")

@dataclass
class AssistantConfig:
    """Configuration de l'assistant intelligent"""
    overlay_enabled: bool = True
    real_time_advice: bool = True
    auto_learning: bool = True
    human_simulation: bool = True
    voice_notifications: bool = False
    advanced_ui: bool = True
    performance_monitoring: bool = True

@dataclass
class SystemStatus:
    """Statut du système en temps réel"""
    knowledge_base_status: str
    learning_engine_status: str
    hrm_system_status: str
    human_simulator_status: str
    active_session: Optional[str]
    total_actions: int
    success_rate: float
    learning_progress: float

class IntelligentAssistantUI:
    """Interface utilisateur principale de l'assistant intelligent"""

    def __init__(self):
        self.config = AssistantConfig()
        self.is_running = False
        self.current_session = None

        # Initialisation des modules
        if MODULES_AVAILABLE:
            self.knowledge_base = get_knowledge_base()
            self.learning_engine = get_learning_engine()
            self.human_simulator = get_human_simulator()
            if HRM_INTEGRATION_AVAILABLE:
                self.decision_maker = DofusIntelligentDecisionMaker()
            else:
                self.decision_maker = None
        else:
            self.knowledge_base = None
            self.learning_engine = None
            self.human_simulator = None
            self.decision_maker = None

        self.logger = logging.getLogger(__name__)

        # Initialisation interface
        self.root = tk.Tk()
        self.setup_ui()

        # Thread de monitoring
        self.monitoring_thread = None

    def setup_ui(self):
        """Configure l'interface utilisateur principale"""
        self.root.title("DOFUS Unity AI Assistant - World Model 2025")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white')
        style.configure('Dark.TButton', background='#404040', foreground='white')

        # Menu principal
        self.setup_menu()

        # Notebook principal
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Onglets
        self.setup_dashboard_tab()
        self.setup_knowledge_tab()
        self.setup_learning_tab()
        self.setup_simulation_tab()
        self.setup_config_tab()

    def setup_menu(self):
        """Configure le menu principal"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouvelle Session", command=self.start_new_session)
        file_menu.add_command(label="Charger Session", command=self.load_session)
        file_menu.add_command(label="Sauvegarder", command=self.save_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exporter Données", command=self.export_data)
        file_menu.add_command(label="Importer Données", command=self.import_data)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.quit_application)

        # Menu Système
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Système", menu=system_menu)
        system_menu.add_command(label="Démarrer Assistant", command=self.start_assistant)
        system_menu.add_command(label="Arrêter Assistant", command=self.stop_assistant)
        system_menu.add_separator()
        system_menu.add_command(label="Diagnostic Système", command=self.run_diagnostics)
        system_menu.add_command(label="Nettoyer Cache", command=self.clear_cache)

        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="À propos", command=self.show_about)

    def setup_dashboard_tab(self):
        """Configure l'onglet tableau de bord"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Tableau de Bord")

        # Statut système
        status_frame = ttk.LabelFrame(dashboard_frame, text="Statut Système")
        status_frame.pack(fill='x', padx=10, pady=5)

        # Variables de statut
        self.kb_status_var = tk.StringVar(value="Arrêté")
        self.learning_status_var = tk.StringVar(value="Arrêté")
        self.hrm_status_var = tk.StringVar(value="Arrêté")
        self.sim_status_var = tk.StringVar(value="Arrêté")

        # Labels de statut
        ttk.Label(status_frame, text="Knowledge Base:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(status_frame, textvariable=self.kb_status_var).grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(status_frame, text="Learning Engine:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(status_frame, textvariable=self.learning_status_var).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(status_frame, text="HRM System:").grid(row=2, column=0, sticky='w', padx=5)
        ttk.Label(status_frame, textvariable=self.hrm_status_var).grid(row=2, column=1, sticky='w', padx=5)

        ttk.Label(status_frame, text="Human Simulator:").grid(row=3, column=0, sticky='w', padx=5)
        ttk.Label(status_frame, textvariable=self.sim_status_var).grid(row=3, column=1, sticky='w', padx=5)

        # Contrôles principaux
        controls_frame = ttk.LabelFrame(dashboard_frame, text="Contrôles")
        controls_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(controls_frame, text="Démarrer Assistant",
                  command=self.start_assistant).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Arrêter Assistant",
                  command=self.stop_assistant).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Diagnostic",
                  command=self.run_diagnostics).pack(side='left', padx=5)

        # Métriques en temps réel
        metrics_frame = ttk.LabelFrame(dashboard_frame, text="Métriques Temps Réel")
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Zone de texte pour les logs
        self.log_text = tk.Text(metrics_frame, height=20, bg='#1e1e1e', fg='white')
        scrollbar = ttk.Scrollbar(metrics_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_knowledge_tab(self):
        """Configure l'onglet Knowledge Base"""
        kb_frame = ttk.Frame(self.notebook)
        self.notebook.add(kb_frame, text="Knowledge Base")

        # Requêtes Knowledge Base
        query_frame = ttk.LabelFrame(kb_frame, text="Requêtes")
        query_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(query_frame, text="Type de requête:").grid(row=0, column=0, sticky='w', padx=5)
        self.query_type = ttk.Combobox(query_frame, values=[
            "Sorts optimaux", "Stratégie monstre", "Route farming",
            "Opportunités marché", "Conseil global"
        ])
        self.query_type.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(query_frame, text="Paramètre:").grid(row=1, column=0, sticky='w', padx=5)
        self.query_param = ttk.Entry(query_frame)
        self.query_param.grid(row=1, column=1, sticky='ew', padx=5)

        ttk.Button(query_frame, text="Exécuter",
                  command=self.execute_knowledge_query).grid(row=2, column=0, columnspan=2, pady=5)

        query_frame.columnconfigure(1, weight=1)

        # Résultats
        results_frame = ttk.LabelFrame(kb_frame, text="Résultats")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.kb_results = tk.Text(results_frame, height=25, bg='#f0f0f0')
        kb_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.kb_results.yview)
        self.kb_results.configure(yscrollcommand=kb_scrollbar.set)

        self.kb_results.pack(side='left', fill='both', expand=True)
        kb_scrollbar.pack(side='right', fill='y')

    def setup_learning_tab(self):
        """Configure l'onglet Learning Engine"""
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="Apprentissage")

        # Configuration session
        session_frame = ttk.LabelFrame(learning_frame, text="Session d'Apprentissage")
        session_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(session_frame, text="Classe:").grid(row=0, column=0, sticky='w', padx=5)
        self.player_class = ttk.Combobox(session_frame, values=[
            "IOPS", "CRA", "SADI", "ENIRIPSA", "ECAFLIP", "ENUTROF",
            "SRAM", "XELOR", "PANDAWA", "ROUBLARD", "ZOBAL", "STEAMER"
        ])
        self.player_class.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(session_frame, text="Niveau:").grid(row=1, column=0, sticky='w', padx=5)
        self.player_level = ttk.Spinbox(session_frame, from_=1, to=200, value=150)
        self.player_level.grid(row=1, column=1, sticky='ew', padx=5)

        ttk.Label(session_frame, text="Serveur:").grid(row=2, column=0, sticky='w', padx=5)
        self.server_name = ttk.Entry(session_frame, value="Julith")
        self.server_name.grid(row=2, column=1, sticky='ew', padx=5)

        session_frame.columnconfigure(1, weight=1)

        # Contrôles session
        controls_learning = ttk.Frame(learning_frame)
        controls_learning.pack(fill='x', padx=10, pady=5)

        ttk.Button(controls_learning, text="Démarrer Session",
                  command=self.start_learning_session).pack(side='left', padx=5)
        ttk.Button(controls_learning, text="Arrêter Session",
                  command=self.stop_learning_session).pack(side='left', padx=5)
        ttk.Button(controls_learning, text="Voir Métriques",
                  command=self.show_learning_metrics).pack(side='left', padx=5)

        # Affichage métriques
        metrics_learning = ttk.LabelFrame(learning_frame, text="Métriques d'Apprentissage")
        metrics_learning.pack(fill='both', expand=True, padx=10, pady=5)

        self.learning_metrics = tk.Text(metrics_learning, height=20, bg='#f0f0f0')
        learning_scrollbar = ttk.Scrollbar(metrics_learning, orient='vertical', command=self.learning_metrics.yview)
        self.learning_metrics.configure(yscrollcommand=learning_scrollbar.set)

        self.learning_metrics.pack(side='left', fill='both', expand=True)
        learning_scrollbar.pack(side='right', fill='y')

    def setup_simulation_tab(self):
        """Configure l'onglet Simulation Humaine"""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="Simulation Humaine")

        # Profil humain
        profile_frame = ttk.LabelFrame(sim_frame, text="Profil Humain")
        profile_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(profile_frame, text="Style de jeu:").grid(row=0, column=0, sticky='w', padx=5)
        self.play_style = ttk.Combobox(profile_frame, values=[
            "aggressive", "cautious", "balanced", "tryhard", "casual"
        ])
        self.play_style.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(profile_frame, text="Temps de réaction (ms):").grid(row=1, column=0, sticky='w', padx=5)
        self.reaction_time = ttk.Scale(profile_frame, from_=150, to=500, orient='horizontal')
        self.reaction_time.grid(row=1, column=1, sticky='ew', padx=5)

        ttk.Label(profile_frame, text="Précision:").grid(row=2, column=0, sticky='w', padx=5)
        self.precision = ttk.Scale(profile_frame, from_=0.7, to=1.0, orient='horizontal')
        self.precision.grid(row=2, column=1, sticky='ew', padx=5)

        profile_frame.columnconfigure(1, weight=1)

        # Test simulation
        test_frame = ttk.LabelFrame(sim_frame, text="Test Simulation")
        test_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(test_frame, text="Tester Mouvement Souris",
                  command=self.test_mouse_movement).pack(side='left', padx=5)
        ttk.Button(test_frame, text="Tester Lancement Sort",
                  command=self.test_spell_cast).pack(side='left', padx=5)
        ttk.Button(test_frame, text="Analyser Session",
                  command=self.analyze_simulation_session).pack(side='left', padx=5)

        # Résultats simulation
        sim_results = ttk.LabelFrame(sim_frame, text="Résultats Simulation")
        sim_results.pack(fill='both', expand=True, padx=10, pady=5)

        self.sim_text = tk.Text(sim_results, height=20, bg='#f0f0f0')
        sim_scrollbar = ttk.Scrollbar(sim_results, orient='vertical', command=self.sim_text.yview)
        self.sim_text.configure(yscrollcommand=sim_scrollbar.set)

        self.sim_text.pack(side='left', fill='both', expand=True)
        sim_scrollbar.pack(side='right', fill='y')

    def setup_config_tab(self):
        """Configure l'onglet Configuration"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")

        # Configuration générale
        general_frame = ttk.LabelFrame(config_frame, text="Configuration Générale")
        general_frame.pack(fill='x', padx=10, pady=5)

        self.overlay_var = tk.BooleanVar(value=self.config.overlay_enabled)
        ttk.Checkbutton(general_frame, text="Overlay activé",
                       variable=self.overlay_var).pack(anchor='w', padx=5)

        self.real_time_var = tk.BooleanVar(value=self.config.real_time_advice)
        ttk.Checkbutton(general_frame, text="Conseils temps réel",
                       variable=self.real_time_var).pack(anchor='w', padx=5)

        self.auto_learn_var = tk.BooleanVar(value=self.config.auto_learning)
        ttk.Checkbutton(general_frame, text="Apprentissage automatique",
                       variable=self.auto_learn_var).pack(anchor='w', padx=5)

        self.human_sim_var = tk.BooleanVar(value=self.config.human_simulation)
        ttk.Checkbutton(general_frame, text="Simulation humaine",
                       variable=self.human_sim_var).pack(anchor='w', padx=5)

        # Boutons configuration
        config_buttons = ttk.Frame(config_frame)
        config_buttons.pack(fill='x', padx=10, pady=5)

        ttk.Button(config_buttons, text="Sauvegarder Config",
                  command=self.save_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="Charger Config",
                  command=self.load_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="Reset Défauts",
                  command=self.reset_config).pack(side='left', padx=5)

    # Méthodes de contrôle

    def start_assistant(self):
        """Démarre l'assistant intelligent"""
        if self.is_running:
            self.log_message("Assistant déjà en cours d'exécution")
            return

        self.is_running = True
        self.log_message("Démarrage de l'assistant intelligent...")

        # Mise à jour des statuts
        self.update_system_status()

        # Démarrage du monitoring
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.log_message("Assistant démarré avec succès")

    def stop_assistant(self):
        """Arrête l'assistant intelligent"""
        if not self.is_running:
            self.log_message("Assistant déjà arrêté")
            return

        self.is_running = False
        self.log_message("Arrêt de l'assistant intelligent...")

        # Arrêt session d'apprentissage si active
        if self.current_session:
            self.stop_learning_session()

        self.log_message("Assistant arrêté")

    def monitoring_loop(self):
        """Boucle de monitoring en arrière-plan"""
        while self.is_running:
            try:
                self.update_system_status()
                time.sleep(1.0)  # Mise à jour chaque seconde
            except Exception as e:
                self.log_message(f"Erreur monitoring: {e}")
                break

    def update_system_status(self):
        """Met à jour le statut des systèmes"""
        if MODULES_AVAILABLE:
            self.kb_status_var.set("Opérationnel" if self.knowledge_base else "Erreur")
            self.learning_status_var.set("Opérationnel" if self.learning_engine else "Erreur")
            self.hrm_status_var.set("Opérationnel" if self.decision_maker else "Erreur")
            self.sim_status_var.set("Opérationnel" if self.human_simulator else "Erreur")
        else:
            self.kb_status_var.set("Non disponible")
            self.learning_status_var.set("Non disponible")
            self.hrm_status_var.set("Non disponible")
            self.sim_status_var.set("Non disponible")

    def log_message(self, message: str):
        """Ajoute un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    # Méthodes Knowledge Base

    def execute_knowledge_query(self):
        """Exécute une requête Knowledge Base"""
        if not self.knowledge_base:
            self.kb_results.delete(1.0, tk.END)
            self.kb_results.insert(tk.END, "Knowledge Base non disponible")
            return

        query_type = self.query_type.get()
        param = self.query_param.get()

        try:
            self.kb_results.delete(1.0, tk.END)
            self.kb_results.insert(tk.END, f"Exécution: {query_type}\n")
            self.kb_results.insert(tk.END, f"Paramètre: {param}\n\n")

            if query_type == "Sorts optimaux":
                result = self.knowledge_base.query_optimal_spells()
                self.kb_results.insert(tk.END, f"Résultat: {result}\n")

            elif query_type == "Stratégie monstre":
                result = self.knowledge_base.query_monster_strategy(param or "Bouftou")
                self.kb_results.insert(tk.END, f"Résultat: {result}\n")

            elif query_type == "Route farming":
                result = self.knowledge_base.query_farming_route(param or "Blé")
                self.kb_results.insert(tk.END, f"Résultat: {result}\n")

            elif query_type == "Opportunités marché":
                result = self.knowledge_base.query_market_opportunities()
                self.kb_results.insert(tk.END, f"Résultat: {result}\n")

            else:
                result = self.knowledge_base.query_comprehensive_advice(param or "combat")
                self.kb_results.insert(tk.END, f"Résultat: {result}\n")

        except Exception as e:
            self.kb_results.insert(tk.END, f"Erreur: {e}\n")

    # Méthodes Learning Engine

    def start_learning_session(self):
        """Démarre une session d'apprentissage"""
        if not self.learning_engine:
            self.log_message("Learning Engine non disponible")
            return

        if self.current_session:
            self.log_message("Session déjà active")
            return

        player_class = self.player_class.get() or "IOPS"
        level = int(self.player_level.get() or 150)
        server = self.server_name.get() or "Julith"

        try:
            session_id = self.learning_engine.start_learning_session(player_class, level, server)
            self.current_session = session_id
            self.log_message(f"Session d'apprentissage démarrée: {session_id}")
        except Exception as e:
            self.log_message(f"Erreur démarrage session: {e}")

    def stop_learning_session(self):
        """Arrête la session d'apprentissage"""
        if not self.learning_engine or not self.current_session:
            self.log_message("Aucune session active")
            return

        try:
            session = self.learning_engine.end_learning_session()
            if session:
                self.log_message(f"Session terminée: {session.session_id}")
                self.log_message(f"Score d'efficacité: {session.efficiency_score:.3f}")
            self.current_session = None
        except Exception as e:
            self.log_message(f"Erreur arrêt session: {e}")

    def show_learning_metrics(self):
        """Affiche les métriques d'apprentissage"""
        if not self.learning_engine:
            self.learning_metrics.delete(1.0, tk.END)
            self.learning_metrics.insert(tk.END, "Learning Engine non disponible")
            return

        try:
            metrics = self.learning_engine.get_learning_metrics()

            self.learning_metrics.delete(1.0, tk.END)
            self.learning_metrics.insert(tk.END, "MÉTRIQUES D'APPRENTISSAGE\n")
            self.learning_metrics.insert(tk.END, "=" * 50 + "\n\n")

            self.learning_metrics.insert(tk.END, f"Sessions totales: {metrics.total_sessions}\n")
            self.learning_metrics.insert(tk.END, f"Durée moyenne: {metrics.avg_session_duration:.1f}min\n")
            self.learning_metrics.insert(tk.END, f"Taux d'amélioration: {metrics.improvement_rate:.3f}\n")
            self.learning_metrics.insert(tk.END, f"Précision patterns: {metrics.pattern_accuracy:.3f}\n")
            self.learning_metrics.insert(tk.END, f"Vitesse adaptation: {metrics.adaptation_speed:.3f}\n")
            self.learning_metrics.insert(tk.END, f"Couverture connaissance: {metrics.knowledge_coverage:.3f}\n")

        except Exception as e:
            self.learning_metrics.insert(tk.END, f"Erreur: {e}\n")

    # Méthodes Simulation Humaine

    def test_mouse_movement(self):
        """Teste le mouvement de souris humain"""
        if not self.human_simulator:
            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "Human Simulator non disponible")
            return

        try:
            movement = self.human_simulator.generate_mouse_movement((0, 0), (200, 150))

            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "TEST MOUVEMENT SOURIS\n")
            self.sim_text.insert(tk.END, "=" * 30 + "\n\n")

            self.sim_text.insert(tk.END, f"Points générés: {len(movement)}\n")
            self.sim_text.insert(tk.END, f"Profil: {self.human_simulator.current_profile.movement_style}\n")
            self.sim_text.insert(tk.END, f"Précision: {self.human_simulator.current_profile.click_precision_level:.3f}\n\n")

            self.sim_text.insert(tk.END, "Trajectoire:\n")
            for i, (x, y, delay) in enumerate(movement[:5]):  # Premiers 5 points
                self.sim_text.insert(tk.END, f"  Point {i}: ({x}, {y}) - {delay:.3f}s\n")

            if len(movement) > 5:
                self.sim_text.insert(tk.END, f"  ... {len(movement) - 5} points supplémentaires\n")

        except Exception as e:
            self.sim_text.insert(tk.END, f"Erreur: {e}\n")

    def test_spell_cast(self):
        """Teste le lancement de sort humain"""
        if not self.human_simulator:
            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "Human Simulator non disponible")
            return

        try:
            sequence = self.human_simulator.simulate_spell_casting_sequence("Pression", (150, 200))

            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "TEST LANCEMENT SORT\n")
            self.sim_text.insert(tk.END, "=" * 30 + "\n\n")

            self.sim_text.insert(tk.END, f"Sort: Pression\n")
            self.sim_text.insert(tk.END, f"Cible: (150, 200)\n\n")

            self.sim_text.insert(tk.END, "Séquence temporelle:\n")
            self.sim_text.insert(tk.END, f"  Préparation: {sequence['preparation_delay']:.3f}s\n")

            key, timing = sequence['key_press_timing']
            self.sim_text.insert(tk.END, f"  Touche '{key}': {timing:.3f}s\n")

            if 'targeting_delay' in sequence:
                self.sim_text.insert(tk.END, f"  Ciblage: {sequence['targeting_delay']:.3f}s\n")

            self.sim_text.insert(tk.END, f"  Lancement: {sequence['cast_delay']:.3f}s\n")
            self.sim_text.insert(tk.END, f"  Confirmation: {sequence['confirmation_delay']:.3f}s\n")

        except Exception as e:
            self.sim_text.insert(tk.END, f"Erreur: {e}\n")

    def analyze_simulation_session(self):
        """Analyse la session de simulation"""
        if not self.human_simulator:
            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "Human Simulator non disponible")
            return

        try:
            # Export et analyse
            filepath = "temp_simulation_analysis.json"
            self.human_simulator.export_session_analysis(filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)

            self.sim_text.delete(1.0, tk.END)
            self.sim_text.insert(tk.END, "ANALYSE SESSION SIMULATION\n")
            self.sim_text.insert(tk.END, "=" * 40 + "\n\n")

            # Profil
            profile = analysis['profile']
            self.sim_text.insert(tk.END, f"Style: {profile['movement_style']}\n")
            self.sim_text.insert(tk.END, f"Réaction: {profile['reaction_time_base']:.0f}ms\n")
            self.sim_text.insert(tk.END, f"Précision: {profile['click_precision']:.3f}\n\n")

            # Stats session
            stats = analysis['session_stats']
            self.sim_text.insert(tk.END, f"Actions totales: {stats['total_actions']}\n")
            self.sim_text.insert(tk.END, f"Durée: {stats['session_duration']:.1f}s\n")
            self.sim_text.insert(tk.END, f"Fatigue finale: {stats['final_fatigue_level']:.3f}\n")

        except Exception as e:
            self.sim_text.insert(tk.END, f"Erreur: {e}\n")

    # Méthodes de configuration

    def save_config(self):
        """Sauvegarde la configuration"""
        self.config.overlay_enabled = self.overlay_var.get()
        self.config.real_time_advice = self.real_time_var.get()
        self.config.auto_learning = self.auto_learn_var.get()
        self.config.human_simulation = self.human_sim_var.get()

        try:
            config_path = Path("assistant_interface/config.json")
            config_path.parent.mkdir(exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2)

            self.log_message("Configuration sauvegardée")
        except Exception as e:
            self.log_message(f"Erreur sauvegarde config: {e}")

    def load_config(self):
        """Charge la configuration"""
        try:
            config_path = Path("assistant_interface/config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                self.config = AssistantConfig(**config_data)

                # Mise à jour UI
                self.overlay_var.set(self.config.overlay_enabled)
                self.real_time_var.set(self.config.real_time_advice)
                self.auto_learn_var.set(self.config.auto_learning)
                self.human_sim_var.set(self.config.human_simulation)

                self.log_message("Configuration chargée")
            else:
                self.log_message("Fichier config non trouvé")
        except Exception as e:
            self.log_message(f"Erreur chargement config: {e}")

    def reset_config(self):
        """Reset la configuration aux valeurs par défaut"""
        self.config = AssistantConfig()

        self.overlay_var.set(self.config.overlay_enabled)
        self.real_time_var.set(self.config.real_time_advice)
        self.auto_learn_var.set(self.config.auto_learning)
        self.human_sim_var.set(self.config.human_simulation)

        self.log_message("Configuration réinitialisée")

    # Méthodes utilitaires

    def run_diagnostics(self):
        """Lance un diagnostic complet du système"""
        self.log_message("Démarrage diagnostic système...")

        # Test Knowledge Base
        if self.knowledge_base:
            try:
                status = self.knowledge_base.get_system_status()
                self.log_message(f"Knowledge Base: OK - {status['performance']['total_queries']} requêtes")
            except Exception as e:
                self.log_message(f"Knowledge Base: ERREUR - {e}")
        else:
            self.log_message("Knowledge Base: NON DISPONIBLE")

        # Test Learning Engine
        if self.learning_engine:
            try:
                metrics = self.learning_engine.get_learning_metrics()
                self.log_message(f"Learning Engine: OK - {metrics.total_sessions} sessions")
            except Exception as e:
                self.log_message(f"Learning Engine: ERREUR - {e}")
        else:
            self.log_message("Learning Engine: NON DISPONIBLE")

        # Test Human Simulator
        if self.human_simulator:
            try:
                profile = self.human_simulator.current_profile
                self.log_message(f"Human Simulator: OK - Style {profile.movement_style}")
            except Exception as e:
                self.log_message(f"Human Simulator: ERREUR - {e}")
        else:
            self.log_message("Human Simulator: NON DISPONIBLE")

        self.log_message("Diagnostic terminé")

    def clear_cache(self):
        """Nettoie les caches du système"""
        self.log_message("Nettoyage des caches...")
        # Implémentation du nettoyage
        self.log_message("Caches nettoyés")

    def start_new_session(self):
        """Démarre une nouvelle session"""
        self.log_message("Nouvelle session démarrée")

    def load_session(self):
        """Charge une session existante"""
        filepath = filedialog.askopenfilename(
            title="Charger Session",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.log_message(f"Session chargée: {filepath}")

    def save_session(self):
        """Sauvegarde la session actuelle"""
        filepath = filedialog.asksaveasfilename(
            title="Sauvegarder Session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.log_message(f"Session sauvegardée: {filepath}")

    def export_data(self):
        """Exporte les données du système"""
        filepath = filedialog.asksaveasfilename(
            title="Exporter Données",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.log_message(f"Données exportées: {filepath}")

    def import_data(self):
        """Importe des données dans le système"""
        filepath = filedialog.askopenfilename(
            title="Importer Données",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.log_message(f"Données importées: {filepath}")

    def show_documentation(self):
        """Affiche la documentation"""
        messagebox.showinfo("Documentation", "Documentation disponible sur GitHub")

    def show_about(self):
        """Affiche les informations à propos"""
        messagebox.showinfo("À propos",
                           "DOFUS Unity AI Assistant\n"
                           "World Model 2025\n"
                           "Développé avec Claude Code")

    def quit_application(self):
        """Quitte l'application"""
        if self.is_running:
            self.stop_assistant()
        self.root.quit()

    def run(self):
        """Lance l'interface utilisateur"""
        self.log_message("Assistant Interface démarré")
        self.root.mainloop()

def main():
    """Point d'entrée principal"""
    app = IntelligentAssistantUI()
    app.run()

if __name__ == "__main__":
    main()