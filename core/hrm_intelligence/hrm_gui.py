"""
Interface Graphique de Contrôle HRM Intelligence
Interface utilisateur pour surveiller et contrôler le bot intelligent

Auteur: Claude Code
Version: 1.0.0
"""

import sys
import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime, timedelta

# Ajouter le chemin du module HRM
sys.path.append(str(Path(__file__).parent))

try:
    from main_hrm_system import HRMIntelligenceSystem, HRMSystemConfig
    from hrm_core import GameState
except ImportError as e:
    print(f"Erreur d'import HRM: {e}")
    sys.exit(1)

class HRMControlInterface:
    """Interface de contrôle principale pour HRM Intelligence"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 HRM Intelligence - Contrôle TacticalBot")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')

        # Variables d'état
        self.hrm_system: Optional[HRMIntelligenceSystem] = None
        self.is_running = False
        self.update_thread = None
        self.last_update_time = time.time()

        # Données pour les graphiques
        self.performance_history = {
            'timestamps': [],
            'success_rate': [],
            'confidence': [],
            'decisions_per_minute': [],
            'reward_rate': []
        }

        # Configuration des couleurs
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'info': '#2196F3'
        }

        self.setup_ui()
        self.setup_monitoring()

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background=self.colors['bg'], foreground=self.colors['fg'])
        style.configure('Info.TLabel', font=('Arial', 10), background=self.colors['bg'], foreground=self.colors['fg'])

        # Notebook principal
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Onglets
        self.create_control_tab()
        self.create_monitoring_tab()
        self.create_learning_tab()
        self.create_quest_tab()
        self.create_logs_tab()
        self.create_config_tab()

    def create_control_tab(self):
        """Onglet de contrôle principal"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="🎮 Contrôle")

        # Section état du bot
        status_frame = ttk.LabelFrame(control_frame, text="État du Bot", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="🔴 Arrêté", style='Title.TLabel')
        self.status_label.pack(side='left')

        self.uptime_label = ttk.Label(status_frame, text="Temps d'activité: 00:00:00", style='Info.TLabel')
        self.uptime_label.pack(side='right')

        # Boutons de contrôle
        controls_frame = ttk.Frame(control_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)

        # Première rangée
        row1 = ttk.Frame(controls_frame)
        row1.pack(fill='x', pady=2)

        self.start_btn = ttk.Button(row1, text="▶️ Démarrer", command=self.start_bot, width=15)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(row1, text="⏹️ Arrêter", command=self.stop_bot, width=15, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.pause_btn = ttk.Button(row1, text="⏸️ Pause", command=self.pause_bot, width=15, state='disabled')
        self.pause_btn.pack(side='left', padx=5)

        # Deuxième rangée
        row2 = ttk.Frame(controls_frame)
        row2.pack(fill='x', pady=2)

        ttk.Button(row2, text="💾 Sauvegarder", command=self.save_system, width=15).pack(side='left', padx=5)
        ttk.Button(row2, text="📁 Charger", command=self.load_system, width=15).pack(side='left', padx=5)
        ttk.Button(row2, text="🔄 Reset Stats", command=self.reset_stats, width=15).pack(side='left', padx=5)

        # Section commandes manuelles
        manual_frame = ttk.LabelFrame(control_frame, text="Commandes Manuelles", padding=10)
        manual_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(manual_frame, text="Action à exécuter:").pack(anchor='w')

        self.action_var = tk.StringVar()
        action_combo = ttk.Combobox(manual_frame, textvariable=self.action_var, width=30)
        action_combo['values'] = [
            'move_up', 'move_down', 'move_left', 'move_right',
            'attack', 'defend', 'use_skill_1', 'use_skill_2',
            'use_potion', 'open_inventory', 'interact', 'cast_spell',
            'rest', 'explore', 'gather_resource', 'craft_item'
        ]
        action_combo.pack(fill='x', pady=5)

        ttk.Button(manual_frame, text="🎯 Exécuter Action", command=self.execute_manual_action).pack(pady=5)

        # Section objectifs/instructions
        objective_frame = ttk.LabelFrame(control_frame, text="Instructions/Objectif", padding=10)
        objective_frame.pack(fill='both', expand=True, padx=10, pady=5)

        ttk.Label(objective_frame, text="Objectif principal:").pack(anchor='w')
        self.objective_text = scrolledtext.ScrolledText(objective_frame, height=4, wrap='word')
        self.objective_text.pack(fill='x', pady=5)
        self.objective_text.insert('1.0', "Exemple: Compléter les quêtes de niveau 15-20 dans la forêt mystique")

        ttk.Button(objective_frame, text="📝 Définir Objectif", command=self.set_objective).pack(pady=5)

        # Section décision en cours
        decision_frame = ttk.LabelFrame(control_frame, text="Décision Actuelle", padding=10)
        decision_frame.pack(fill='x', padx=10, pady=5)

        self.current_action_label = ttk.Label(decision_frame, text="Action: Aucune", style='Info.TLabel')
        self.current_action_label.pack(anchor='w')

        self.confidence_label = ttk.Label(decision_frame, text="Confiance: 0%", style='Info.TLabel')
        self.confidence_label.pack(anchor='w')

        self.reasoning_text = scrolledtext.ScrolledText(decision_frame, height=3, wrap='word')
        self.reasoning_text.pack(fill='x', pady=5)

    def create_monitoring_tab(self):
        """Onglet de monitoring des performances"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="📊 Monitoring")

        # Statistiques en temps réel
        stats_frame = ttk.LabelFrame(monitor_frame, text="Statistiques Temps Réel", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=5)

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill='x')

        # Colonne 1
        col1 = ttk.Frame(stats_grid)
        col1.pack(side='left', fill='both', expand=True)

        self.decisions_label = ttk.Label(col1, text="Décisions: 0", style='Info.TLabel')
        self.decisions_label.pack(anchor='w')

        self.success_rate_label = ttk.Label(col1, text="Taux de succès: 0%", style='Info.TLabel')
        self.success_rate_label.pack(anchor='w')

        self.avg_confidence_label = ttk.Label(col1, text="Confiance moy.: 0%", style='Info.TLabel')
        self.avg_confidence_label.pack(anchor='w')

        # Colonne 2
        col2 = ttk.Frame(stats_grid)
        col2.pack(side='left', fill='both', expand=True)

        self.total_reward_label = ttk.Label(col2, text="Récompense totale: 0", style='Info.TLabel')
        self.total_reward_label.pack(anchor='w')

        self.learning_sessions_label = ttk.Label(col2, text="Sessions d'apprentissage: 0", style='Info.TLabel')
        self.learning_sessions_label.pack(anchor='w')

        self.dpm_label = ttk.Label(col2, text="Décisions/min: 0", style='Info.TLabel')
        self.dpm_label.pack(anchor='w')

        # Graphiques de performance
        charts_frame = ttk.LabelFrame(monitor_frame, text="Graphiques de Performance", padding=10)
        charts_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Figure matplotlib
        self.fig = Figure(figsize=(12, 6), facecolor='#2b2b2b')
        self.fig.suptitle('Performance HRM Intelligence', color='white')

        # Sous-graphiques
        self.ax1 = self.fig.add_subplot(221, facecolor='#3b3b3b')
        self.ax2 = self.fig.add_subplot(222, facecolor='#3b3b3b')
        self.ax3 = self.fig.add_subplot(223, facecolor='#3b3b3b')
        self.ax4 = self.fig.add_subplot(224, facecolor='#3b3b3b')

        # Configuration des axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

        self.ax1.set_title('Taux de Succès', color='white')
        self.ax2.set_title('Confiance Moyenne', color='white')
        self.ax3.set_title('Décisions/Minute', color='white')
        self.ax4.set_title('Récompenses/Minute', color='white')

        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_learning_tab(self):
        """Onglet d'apprentissage et adaptation"""
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="🧠 Apprentissage")

        # Configuration d'apprentissage
        config_frame = ttk.LabelFrame(learning_frame, text="Configuration Apprentissage", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        # Taux d'apprentissage
        ttk.Label(config_frame, text="Taux d'apprentissage:").pack(anchor='w')
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Scale(config_frame, from_=0.0001, to=0.01, variable=self.learning_rate_var,
                 orient='horizontal').pack(fill='x', pady=2)

        # Mode d'apprentissage
        ttk.Label(config_frame, text="Mode d'apprentissage:").pack(anchor='w')
        self.learning_mode_var = tk.StringVar(value="adaptatif")
        mode_frame = ttk.Frame(config_frame)
        mode_frame.pack(fill='x')

        ttk.Radiobutton(mode_frame, text="Conservateur", variable=self.learning_mode_var,
                       value="conservateur").pack(side='left')
        ttk.Radiobutton(mode_frame, text="Adaptatif", variable=self.learning_mode_var,
                       value="adaptatif").pack(side='left')
        ttk.Radiobutton(mode_frame, text="Agressif", variable=self.learning_mode_var,
                       value="agressif").pack(side='left')

        # Expériences récentes
        exp_frame = ttk.LabelFrame(learning_frame, text="Expériences Récentes", padding=10)
        exp_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Tableau des expériences
        columns = ('Temps', 'Action', 'Résultat', 'Récompense', 'Confiance')
        self.exp_tree = ttk.Treeview(exp_frame, columns=columns, show='headings', height=10)

        for col in columns:
            self.exp_tree.heading(col, text=col)
            self.exp_tree.column(col, width=100)

        scrollbar_exp = ttk.Scrollbar(exp_frame, orient='vertical', command=self.exp_tree.yview)
        self.exp_tree.configure(yscrollcommand=scrollbar_exp.set)

        self.exp_tree.pack(side='left', fill='both', expand=True)
        scrollbar_exp.pack(side='right', fill='y')

    def create_quest_tab(self):
        """Onglet de gestion des quêtes"""
        quest_frame = ttk.Frame(self.notebook)
        self.notebook.add(quest_frame, text="📋 Quêtes")

        # Quêtes actives
        active_frame = ttk.LabelFrame(quest_frame, text="Quêtes Actives", padding=10)
        active_frame.pack(fill='x', padx=10, pady=5)

        columns = ('Titre', 'Progression', 'Priorité', 'Temps Estimé')
        self.quest_tree = ttk.Treeview(active_frame, columns=columns, show='headings', height=6)

        for col in columns:
            self.quest_tree.heading(col, text=col)
            self.quest_tree.column(col, width=150)

        scrollbar_quest = ttk.Scrollbar(active_frame, orient='vertical', command=self.quest_tree.yview)
        self.quest_tree.configure(yscrollcommand=scrollbar_quest.set)

        self.quest_tree.pack(side='left', fill='both', expand=True)
        scrollbar_quest.pack(side='right', fill='y')

        # Contrôles de quêtes
        quest_controls = ttk.Frame(quest_frame)
        quest_controls.pack(fill='x', padx=10, pady=5)

        ttk.Button(quest_controls, text="🔍 Détecter Quêtes", command=self.detect_quests).pack(side='left', padx=5)
        ttk.Button(quest_controls, text="➕ Ajouter Quête", command=self.add_manual_quest).pack(side='left', padx=5)
        ttk.Button(quest_controls, text="❌ Abandonner", command=self.abandon_quest).pack(side='left', padx=5)

        # Recommandations
        rec_frame = ttk.LabelFrame(quest_frame, text="Recommandations", padding=10)
        rec_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.recommendations_text = scrolledtext.ScrolledText(rec_frame, height=8, wrap='word')
        self.recommendations_text.pack(fill='both', expand=True)

    def create_logs_tab(self):
        """Onglet des logs système"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📝 Logs")

        # Filtres de logs
        filter_frame = ttk.Frame(logs_frame)
        filter_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(filter_frame, text="Niveau:").pack(side='left')
        self.log_level_var = tk.StringVar(value="INFO")
        level_combo = ttk.Combobox(filter_frame, textvariable=self.log_level_var, width=10)
        level_combo['values'] = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        level_combo.pack(side='left', padx=5)

        ttk.Button(filter_frame, text="🔄 Actualiser", command=self.refresh_logs).pack(side='left', padx=5)
        ttk.Button(filter_frame, text="🗑️ Effacer", command=self.clear_logs).pack(side='left', padx=5)

        # Zone de logs
        self.logs_text = scrolledtext.ScrolledText(logs_frame, height=25, wrap='word',
                                                  bg='#1e1e1e', fg='#ffffff', insertbackground='white')
        self.logs_text.pack(fill='both', expand=True, padx=10, pady=5)

    def create_config_tab(self):
        """Onglet de configuration"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")

        # Configuration générale
        general_frame = ttk.LabelFrame(config_frame, text="Configuration Générale", padding=10)
        general_frame.pack(fill='x', padx=10, pady=5)

        # Player ID
        ttk.Label(general_frame, text="ID Joueur:").pack(anchor='w')
        self.player_id_var = tk.StringVar(value="tactical_bot")
        ttk.Entry(general_frame, textvariable=self.player_id_var, width=30).pack(fill='x', pady=2)

        # Délais humains
        self.human_delays_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Délais humains", variable=self.human_delays_var).pack(anchor='w')

        # Mode debug
        self.debug_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(general_frame, text="Mode debug", variable=self.debug_mode_var).pack(anchor='w')

        # Configuration avancée
        advanced_frame = ttk.LabelFrame(config_frame, text="Configuration Avancée", padding=10)
        advanced_frame.pack(fill='x', padx=10, pady=5)

        # Timeout décision
        ttk.Label(advanced_frame, text="Timeout décision (s):").pack(anchor='w')
        self.decision_timeout_var = tk.DoubleVar(value=1.0)
        ttk.Scale(advanced_frame, from_=0.1, to=5.0, variable=self.decision_timeout_var,
                 orient='horizontal').pack(fill='x', pady=2)

        # Interval sauvegarde
        ttk.Label(advanced_frame, text="Interval sauvegarde (s):").pack(anchor='w')
        self.save_interval_var = tk.IntVar(value=300)
        ttk.Scale(advanced_frame, from_=60, to=1800, variable=self.save_interval_var,
                 orient='horizontal').pack(fill='x', pady=2)

        # Boutons de configuration
        config_buttons = ttk.Frame(config_frame)
        config_buttons.pack(fill='x', padx=10, pady=5)

        ttk.Button(config_buttons, text="💾 Sauvegarder Config", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="📁 Charger Config", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="🔄 Défaut", command=self.reset_config).pack(side='left', padx=5)

    def setup_monitoring(self):
        """Configure le monitoring automatique"""
        self.update_ui_periodically()

    def update_ui_periodically(self):
        """Met à jour l'interface périodiquement"""
        if self.hrm_system and self.is_running:
            try:
                # Mettre à jour les statistiques
                self.update_statistics()
                self.update_graphs()
                self.update_current_decision()
                self.update_quest_status()

            except Exception as e:
                self.log_message(f"Erreur mise à jour UI: {e}", "ERROR")

        # Programmer la prochaine mise à jour
        self.root.after(1000, self.update_ui_periodically)  # Toutes les secondes

    def update_statistics(self):
        """Met à jour les statistiques affichées"""
        if not self.hrm_system:
            return

        status = self.hrm_system.get_system_status()
        stats = status['statistics']

        # État du bot
        if status['running'] and not status['paused']:
            self.status_label.config(text="🟢 En marche")
        elif status['paused']:
            self.status_label.config(text="🟡 En pause")
        else:
            self.status_label.config(text="🔴 Arrêté")

        # Temps d'activité
        self.uptime_label.config(text=f"Temps d'activité: {status['uptime_formatted']}")

        # Statistiques détaillées
        total_actions = stats['successful_actions'] + stats['failed_actions']
        success_rate = (stats['successful_actions'] / max(total_actions, 1)) * 100

        self.decisions_label.config(text=f"Décisions: {stats['decisions_made']}")
        self.success_rate_label.config(text=f"Taux de succès: {success_rate:.1f}%")

        performance = status.get('performance', {})
        avg_conf = performance.get('average_confidence', 0) * 100
        self.avg_confidence_label.config(text=f"Confiance moy.: {avg_conf:.1f}%")

        self.total_reward_label.config(text=f"Récompense totale: {stats['total_reward']:.1f}")
        self.learning_sessions_label.config(text=f"Sessions d'apprentissage: {stats['learning_sessions']}")

        # Calcul DPM
        uptime_minutes = max(status['uptime_seconds'] / 60, 1)
        dpm = stats['decisions_made'] / uptime_minutes
        self.dpm_label.config(text=f"Décisions/min: {dpm:.1f}")

        # Historique pour les graphiques
        current_time = time.time()
        self.performance_history['timestamps'].append(current_time)
        self.performance_history['success_rate'].append(success_rate)
        self.performance_history['confidence'].append(avg_conf)
        self.performance_history['decisions_per_minute'].append(dpm)
        self.performance_history['reward_rate'].append(stats['total_reward'] / uptime_minutes)

        # Garder seulement les 50 derniers points
        for key in self.performance_history:
            if len(self.performance_history[key]) > 50:
                self.performance_history[key] = self.performance_history[key][-50:]

    def update_graphs(self):
        """Met à jour les graphiques de performance"""
        if len(self.performance_history['timestamps']) < 2:
            return

        # Nettoyer les axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        timestamps = self.performance_history['timestamps']

        # Convertir en temps relatifs (minutes)
        if timestamps:
            start_time = timestamps[0]
            relative_times = [(t - start_time) / 60 for t in timestamps]

            # Graphique 1: Taux de succès
            self.ax1.plot(relative_times, self.performance_history['success_rate'],
                         color='#4CAF50', linewidth=2)
            self.ax1.set_title('Taux de Succès (%)', color='white')
            self.ax1.set_ylim(0, 100)

            # Graphique 2: Confiance
            self.ax2.plot(relative_times, self.performance_history['confidence'],
                         color='#2196F3', linewidth=2)
            self.ax2.set_title('Confiance Moyenne (%)', color='white')
            self.ax2.set_ylim(0, 100)

            # Graphique 3: Décisions/minute
            self.ax3.plot(relative_times, self.performance_history['decisions_per_minute'],
                         color='#FF9800', linewidth=2)
            self.ax3.set_title('Décisions/Minute', color='white')

            # Graphique 4: Récompenses/minute
            self.ax4.plot(relative_times, self.performance_history['reward_rate'],
                         color='#9C27B0', linewidth=2)
            self.ax4.set_title('Récompenses/Minute', color='white')

            # Configuration des axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.tick_params(colors='white')
                ax.set_xlabel('Temps (min)', color='white')
                for spine in ax.spines.values():
                    spine.set_color('white')

        self.fig.tight_layout()
        self.canvas.draw()

    def update_current_decision(self):
        """Met à jour l'affichage de la décision actuelle"""
        # Cette fonction serait connectée au système HRM pour obtenir la décision en cours
        # Pour l'instant, simulé
        pass

    def update_quest_status(self):
        """Met à jour le statut des quêtes"""
        # Effacer l'arbre existant
        for item in self.quest_tree.get_children():
            self.quest_tree.delete(item)

        # Cette fonction serait connectée au quest_tracker
        # Pour l'instant, ajout d'exemples
        if self.hrm_system:
            # Exemples de quêtes
            example_quests = [
                ("Vaincre 5 Gobelins", "3/5", "Haute", "10 min"),
                ("Collecter 20 Herbes", "15/20", "Moyenne", "5 min"),
                ("Parler au PNJ Marcus", "0/1", "Faible", "2 min")
            ]

            for quest in example_quests:
                self.quest_tree.insert('', 'end', values=quest)

    def start_bot(self):
        """Démarre le bot HRM"""
        try:
            if not self.hrm_system:
                config = self.create_config_from_ui()
                self.hrm_system = HRMIntelligenceSystem(
                    config=config,
                    player_id=self.player_id_var.get()
                )

            self.hrm_system.start()
            self.is_running = True

            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.pause_btn.config(state='normal')

            self.log_message("Bot HRM Intelligence démarré", "INFO")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de démarrer le bot: {e}")
            self.log_message(f"Erreur démarrage: {e}", "ERROR")

    def stop_bot(self):
        """Arrête le bot HRM"""
        try:
            if self.hrm_system:
                self.hrm_system.stop()
                self.is_running = False

            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.pause_btn.config(state='disabled')

            self.log_message("Bot HRM Intelligence arrêté", "INFO")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'arrêter le bot: {e}")
            self.log_message(f"Erreur arrêt: {e}", "ERROR")

    def pause_bot(self):
        """Met en pause/reprend le bot"""
        try:
            if self.hrm_system:
                if self.hrm_system.paused:
                    self.hrm_system.resume()
                    self.pause_btn.config(text="⏸️ Pause")
                    self.log_message("Bot repris", "INFO")
                else:
                    self.hrm_system.pause()
                    self.pause_btn.config(text="▶️ Reprendre")
                    self.log_message("Bot mis en pause", "INFO")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de modifier l'état: {e}")

    def execute_manual_action(self):
        """Exécute une action manuelle"""
        action = self.action_var.get()
        if not action:
            messagebox.showwarning("Attention", "Sélectionnez une action")
            return

        try:
            # TODO: Implémenter l'exécution d'action manuelle
            self.log_message(f"Action manuelle exécutée: {action}", "INFO")
            messagebox.showinfo("Succès", f"Action '{action}' exécutée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exécuter l'action: {e}")

    def set_objective(self):
        """Définit l'objectif du bot"""
        objective = self.objective_text.get('1.0', 'end-1c')
        if not objective.strip():
            messagebox.showwarning("Attention", "Entrez un objectif")
            return

        try:
            # TODO: Implémenter la définition d'objectif
            self.log_message(f"Nouvel objectif défini: {objective[:50]}...", "INFO")
            messagebox.showinfo("Succès", "Objectif mis à jour")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de définir l'objectif: {e}")

    def detect_quests(self):
        """Détecte automatiquement les quêtes"""
        try:
            if self.hrm_system and self.hrm_system.quest_tracker:
                detected = self.hrm_system.quest_tracker.detect_quests_from_screen()
                self.log_message(f"Détection de quêtes: {len(detected)} trouvées", "INFO")
                self.update_quest_status()
            else:
                messagebox.showwarning("Attention", "Système non initialisé")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur détection quêtes: {e}")

    def add_manual_quest(self):
        """Ajoute une quête manuellement"""
        # TODO: Implémenter dialogue d'ajout de quête
        messagebox.showinfo("Info", "Fonctionnalité à implémenter")

    def abandon_quest(self):
        """Abandonne la quête sélectionnée"""
        selection = self.quest_tree.selection()
        if not selection:
            messagebox.showwarning("Attention", "Sélectionnez une quête")
            return

        # TODO: Implémenter abandon de quête
        messagebox.showinfo("Info", "Quête abandonnée")

    def create_config_from_ui(self) -> HRMSystemConfig:
        """Crée une configuration à partir de l'interface"""
        config = HRMSystemConfig()
        config.debug_mode = self.debug_mode_var.get()
        config.human_like_delays = self.human_delays_var.get()
        config.decision_timeout = self.decision_timeout_var.get()
        config.auto_save_interval = self.save_interval_var.get()
        config.adaptive_learning_rate = self.learning_rate_var.get()
        return config

    def save_config(self):
        """Sauvegarde la configuration"""
        try:
            config_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Sauvegarder la configuration"
            )

            if config_path:
                config = self.create_config_from_ui()
                config_dict = {
                    'player_id': self.player_id_var.get(),
                    'learning_rate': self.learning_rate_var.get(),
                    'learning_mode': self.learning_mode_var.get(),
                    'human_delays': self.human_delays_var.get(),
                    'debug_mode': self.debug_mode_var.get(),
                    'decision_timeout': self.decision_timeout_var.get(),
                    'save_interval': self.save_interval_var.get()
                }

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)

                messagebox.showinfo("Succès", "Configuration sauvegardée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder: {e}")

    def load_config(self):
        """Charge une configuration"""
        try:
            config_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
                title="Charger une configuration"
            )

            if config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                # Mettre à jour l'interface
                self.player_id_var.set(config_dict.get('player_id', 'tactical_bot'))
                self.learning_rate_var.set(config_dict.get('learning_rate', 0.001))
                self.learning_mode_var.set(config_dict.get('learning_mode', 'adaptatif'))
                self.human_delays_var.set(config_dict.get('human_delays', True))
                self.debug_mode_var.set(config_dict.get('debug_mode', False))
                self.decision_timeout_var.set(config_dict.get('decision_timeout', 1.0))
                self.save_interval_var.set(config_dict.get('save_interval', 300))

                messagebox.showinfo("Succès", "Configuration chargée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger: {e}")

    def reset_config(self):
        """Remet la configuration par défaut"""
        self.player_id_var.set("tactical_bot")
        self.learning_rate_var.set(0.001)
        self.learning_mode_var.set("adaptatif")
        self.human_delays_var.set(True)
        self.debug_mode_var.set(False)
        self.decision_timeout_var.set(1.0)
        self.save_interval_var.set(300)
        messagebox.showinfo("Succès", "Configuration remise par défaut")

    def save_system(self):
        """Sauvegarde l'état du système"""
        try:
            if self.hrm_system:
                self.hrm_system.periodic_save()
                messagebox.showinfo("Succès", "Système sauvegardé")
            else:
                messagebox.showwarning("Attention", "Aucun système à sauvegarder")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder: {e}")

    def load_system(self):
        """Charge un état de système"""
        # TODO: Implémenter le chargement d'état
        messagebox.showinfo("Info", "Fonctionnalité à implémenter")

    def reset_stats(self):
        """Remet à zéro les statistiques"""
        if messagebox.askyesno("Confirmation", "Remettre à zéro toutes les statistiques ?"):
            # TODO: Implémenter reset stats
            self.performance_history = {
                'timestamps': [],
                'success_rate': [],
                'confidence': [],
                'decisions_per_minute': [],
                'reward_rate': []
            }
            messagebox.showinfo("Succès", "Statistiques remises à zéro")

    def refresh_logs(self):
        """Actualise l'affichage des logs"""
        # TODO: Implémenter lecture des logs
        pass

    def clear_logs(self):
        """Efface l'affichage des logs"""
        self.logs_text.delete('1.0', 'end')

    def log_message(self, message: str, level: str = "INFO"):
        """Ajoute un message aux logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colored_message = f"[{timestamp}] {level}: {message}\n"

        self.logs_text.insert('end', colored_message)
        self.logs_text.see('end')

    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()

    def on_closing(self):
        """Gestionnaire de fermeture"""
        if self.is_running and messagebox.askyesno("Confirmation", "Arrêter le bot avant de fermer ?"):
            self.stop_bot()

        self.root.destroy()

def main():
    """Point d'entrée principal"""
    try:
        app = HRMControlInterface()
        app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.run()

    except Exception as e:
        messagebox.showerror("Erreur Fatale", f"Impossible de lancer l'interface: {e}")

if __name__ == "__main__":
    main()