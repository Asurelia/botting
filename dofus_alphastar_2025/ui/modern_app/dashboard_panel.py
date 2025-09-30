#!/usr/bin/env python3
"""
DashboardPanel - Panneau principal avec monitoring temps réel
Vue d'ensemble complète de l'état du bot et performances
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, Callable
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .theme_manager import ThemeManager

@dataclass
class BotMetrics:
    """Métriques du bot"""
    uptime: float = 0.0
    quests_completed: int = 0
    experience_gained: int = 0
    kamas_earned: int = 0
    items_collected: int = 0
    deaths: int = 0
    fights_won: int = 0
    fights_lost: int = 0
    maps_explored: int = 0
    actions_per_minute: float = 0.0
    success_rate: float = 100.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    fps: float = 0.0

@dataclass
class CharacterInfo:
    """Informations du personnage"""
    name: str = "Unknown"
    level: int = 1
    class_name: str = "Unknown"
    experience: int = 0
    experience_next: int = 100
    kamas: int = 0
    energy: int = 10000
    position: tuple = (0, 0)
    map_name: str = "Unknown"
    current_quest: str = "None"

class RealTimeChart:
    """Widget graphique temps réel"""

    def __init__(self, parent, theme_manager: ThemeManager, title: str, max_points: int = 100):
        self.parent = parent
        self.theme = theme_manager
        self.title = title
        self.max_points = max_points
        self.data_points = []

        self.frame = self.theme.create_panel(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        # Titre
        title_label = self.theme.create_subtitle_label(
            self.frame,
            text=self.title
        )
        title_label.pack(pady=(10, 5))

        # Canvas pour le graphique
        self.canvas = tk.Canvas(
            self.frame,
            width=300,
            height=120,
            bg=self.theme.get_colors().bg_secondary,
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

    def add_data_point(self, value: float):
        """Ajoute un point de données"""
        self.data_points.append(value)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
        self.update_chart()

    def update_chart(self):
        """Met à jour le graphique"""
        self.canvas.delete("all")

        if len(self.data_points) < 2:
            return

        colors = self.theme.get_colors()

        # Dimensions du canvas
        width = self.canvas.winfo_width() or 300
        height = self.canvas.winfo_height() or 120
        margin = 20

        chart_width = width - 2 * margin
        chart_height = height - 2 * margin

        # Normalisation des données
        min_val = min(self.data_points) if self.data_points else 0
        max_val = max(self.data_points) if self.data_points else 1
        range_val = max_val - min_val if max_val != min_val else 1

        # Dessiner la grille
        for i in range(5):
            y = margin + (i * chart_height / 4)
            self.canvas.create_line(
                margin, y, width - margin, y,
                fill=colors.border_light, width=1
            )

        # Dessiner la courbe
        points = []
        for i, value in enumerate(self.data_points):
            x = margin + (i * chart_width / max(len(self.data_points) - 1, 1))
            y = height - margin - ((value - min_val) / range_val * chart_height)
            points.extend([x, y])

        if len(points) >= 4:
            self.canvas.create_line(
                points,
                fill=colors.primary,
                width=2,
                smooth=True
            )

        # Dessiner les points
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            self.canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2,
                fill=colors.primary,
                outline=colors.primary_dark
            )

class StatusWidget:
    """Widget de statut"""

    def __init__(self, parent, theme_manager: ThemeManager, title: str, icon: str = "●"):
        self.parent = parent
        self.theme = theme_manager
        self.title = title
        self.icon = icon

        self.frame = self.theme.create_panel(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame.configure(width=200, height=80)
        self.frame.pack_propagate(False)

        # Container principal
        main_container = self.theme.create_frame(self.frame, "primary")
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Titre avec icône
        title_frame = self.theme.create_frame(main_container, "primary")
        title_frame.pack(fill=tk.X, pady=(0, 5))

        self.icon_label = self.theme.create_body_label(
            title_frame,
            text=self.icon,
            fg=self.theme.get_colors().accent_success
        )
        self.icon_label.pack(side=tk.LEFT)

        title_label = self.theme.create_body_label(
            title_frame,
            text=self.title
        )
        title_label.pack(side=tk.LEFT, padx=(5, 0))

        # Valeur
        self.value_label = self.theme.create_subtitle_label(
            main_container,
            text="--"
        )
        self.value_label.pack()

        # Détail
        self.detail_label = self.theme.create_body_label(
            main_container,
            text="",
            fg=self.theme.get_colors().text_secondary
        )
        self.detail_label.pack()

    def update_status(self, value: str, detail: str = "", status: str = "success"):
        """Met à jour le statut"""
        colors = self.theme.get_colors()

        status_colors = {
            "success": colors.accent_success,
            "warning": colors.accent_warning,
            "error": colors.accent_error,
            "info": colors.accent_info
        }

        icon_color = status_colors.get(status, colors.accent_success)

        self.value_label.configure(text=value)
        self.detail_label.configure(text=detail)
        self.icon_label.configure(fg=icon_color)

class DashboardPanel:
    """Panneau principal du dashboard"""

    def __init__(self, parent, theme_manager: ThemeManager, app_controller=None):
        self.parent = parent
        self.theme = theme_manager
        self.app_controller = app_controller

        # Données
        self.metrics = BotMetrics()
        self.character_info = CharacterInfo()
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None

        # Interface
        self.frame = self.theme.create_frame(parent, "primary")
        self.setup_ui()

        # Démarrer les mises à jour
        self.start_monitoring()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme.create_title_label(
            header_frame,
            text="🤖 Dashboard AlphaStar DOFUS"
        )
        title_label.pack(side=tk.LEFT)

        # Statut global
        self.status_label = self.theme.create_body_label(
            header_frame,
            text="● IDLE",
            fg=self.theme.get_colors().accent_warning
        )
        self.status_label.pack(side=tk.RIGHT)

        # Container principal avec scroll
        self.setup_scrollable_content()

    def setup_scrollable_content(self):
        """Configure le contenu scrollable"""
        # Canvas et scrollbar
        canvas = tk.Canvas(
            self.frame,
            bg=self.theme.get_colors().bg_primary,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)

        self.scrollable_frame = self.theme.create_frame(canvas, "primary")

        # Configuration du scroll
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas et scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 20))
        scrollbar.pack(side="right", fill="y", padx=(0, 20), pady=(0, 20))

        # Contenu du dashboard
        self.create_status_section()
        self.create_metrics_section()
        self.create_character_section()
        self.create_activity_section()
        self.create_charts_section()

    def create_status_section(self):
        """Section de statut principal"""
        section_frame = self.theme.create_panel(self.scrollable_frame)
        section_frame.pack(fill=tk.X, pady=(0, 20))

        # Titre de section
        title = self.theme.create_subtitle_label(
            section_frame,
            text="📊 État du Bot"
        )
        title.pack(pady=(15, 10))

        # Grid de statuts
        status_grid = self.theme.create_frame(section_frame, "primary")
        status_grid.pack(padx=15, pady=(0, 15))

        # Widgets de statut
        self.status_widgets = {}

        statuses = [
            ("bot_status", "État", "🤖"),
            ("quest_status", "Quête", "📋"),
            ("health_status", "Santé", "❤️"),
            ("connection_status", "Connexion", "🌐")
        ]

        for i, (key, title, icon) in enumerate(statuses):
            widget = StatusWidget(status_grid, self.theme, title, icon)
            widget.frame.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="ew")
            self.status_widgets[key] = widget

        # Configuration des colonnes
        status_grid.grid_columnconfigure(0, weight=1)
        status_grid.grid_columnconfigure(1, weight=1)

    def create_metrics_section(self):
        """Section des métriques"""
        section_frame = self.theme.create_panel(self.scrollable_frame)
        section_frame.pack(fill=tk.X, pady=(0, 20))

        # Titre
        title = self.theme.create_subtitle_label(
            section_frame,
            text="📈 Métriques de Performance"
        )
        title.pack(pady=(15, 10))

        # Grid des métriques
        metrics_grid = self.theme.create_frame(section_frame, "primary")
        metrics_grid.pack(padx=15, pady=(0, 15))

        # Métriques principales
        self.metric_widgets = {}

        metrics = [
            ("uptime", "Temps d'activité", "⏱️"),
            ("quests", "Quêtes", "✅"),
            ("experience", "Expérience", "⭐"),
            ("kamas", "Kamas", "💰"),
            ("fights", "Combats", "⚔️"),
            ("success_rate", "Taux de succès", "🎯")
        ]

        for i, (key, title, icon) in enumerate(metrics):
            widget = StatusWidget(metrics_grid, self.theme, title, icon)
            widget.frame.grid(row=i//3, column=i%3, padx=10, pady=5, sticky="ew")
            self.metric_widgets[key] = widget

        # Configuration des colonnes
        for i in range(3):
            metrics_grid.grid_columnconfigure(i, weight=1)

    def create_character_section(self):
        """Section informations personnage"""
        section_frame = self.theme.create_panel(self.scrollable_frame)
        section_frame.pack(fill=tk.X, pady=(0, 20))

        # Titre
        title = self.theme.create_subtitle_label(
            section_frame,
            text="👤 Informations Personnage"
        )
        title.pack(pady=(15, 10))

        # Grid personnage
        char_grid = self.theme.create_frame(section_frame, "primary")
        char_grid.pack(padx=15, pady=(0, 15))

        # Informations du personnage
        self.character_widgets = {}

        char_info = [
            ("name", "Nom", "👤"),
            ("level", "Niveau", "🔢"),
            ("class", "Classe", "🏹"),
            ("position", "Position", "📍"),
            ("map", "Carte", "🗺️"),
            ("energy", "Énergie", "⚡")
        ]

        for i, (key, title, icon) in enumerate(char_info):
            widget = StatusWidget(char_grid, self.theme, title, icon)
            widget.frame.grid(row=i//3, column=i%3, padx=10, pady=5, sticky="ew")
            self.character_widgets[key] = widget

        # Configuration des colonnes
        for i in range(3):
            char_grid.grid_columnconfigure(i, weight=1)

    def create_activity_section(self):
        """Section activité récente"""
        section_frame = self.theme.create_panel(self.scrollable_frame)
        section_frame.pack(fill=tk.X, pady=(0, 20))

        # Titre
        title = self.theme.create_subtitle_label(
            section_frame,
            text="📝 Activité Récente"
        )
        title.pack(pady=(15, 10))

        # Zone de texte pour les logs
        logs_frame = self.theme.create_frame(section_frame, "secondary")
        logs_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Scrollable text widget
        self.activity_text = tk.Text(
            logs_frame,
            height=8,
            wrap=tk.WORD,
            bg=self.theme.get_colors().bg_secondary,
            fg=self.theme.get_colors().text_primary,
            font=self.theme.get_fonts()["body"],
            relief="flat",
            bd=0
        )

        activity_scroll = ttk.Scrollbar(logs_frame, command=self.activity_text.yview)
        self.activity_text.configure(yscrollcommand=activity_scroll.set)

        self.activity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        activity_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        # Messages initiaux
        self.add_activity_log("🚀 Dashboard AlphaStar initialisé")
        self.add_activity_log("📡 Connexion aux systèmes en cours...")

    def create_charts_section(self):
        """Section graphiques temps réel"""
        section_frame = self.theme.create_panel(self.scrollable_frame)
        section_frame.pack(fill=tk.X, pady=(0, 20))

        # Titre
        title = self.theme.create_subtitle_label(
            section_frame,
            text="📊 Graphiques Temps Réel"
        )
        title.pack(pady=(15, 10))

        # Grid des graphiques
        charts_grid = self.theme.create_frame(section_frame, "primary")
        charts_grid.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Graphiques
        self.charts = {}

        chart_configs = [
            ("cpu", "Utilisation CPU (%)"),
            ("memory", "Mémoire (MB)"),
            ("fps", "FPS"),
            ("actions", "Actions/min")
        ]

        for i, (key, title) in enumerate(chart_configs):
            chart = RealTimeChart(charts_grid, self.theme, title)
            chart.frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            self.charts[key] = chart

        # Configuration du grid
        charts_grid.grid_columnconfigure(0, weight=1)
        charts_grid.grid_columnconfigure(1, weight=1)
        charts_grid.grid_rowconfigure(0, weight=1)
        charts_grid.grid_rowconfigure(1, weight=1)

    def add_activity_log(self, message: str, level: str = "info"):
        """Ajoute un message au log d'activité"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        level_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "debug": "🔧"
        }

        icon = level_icons.get(level, "ℹ️")
        formatted_message = f"[{timestamp}] {icon} {message}\n"

        self.activity_text.insert(tk.END, formatted_message)
        self.activity_text.see(tk.END)

        # Limiter le nombre de lignes
        lines = int(self.activity_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.activity_text.delete(1.0, "2.0")

    def start_monitoring(self):
        """Démarre le monitoring temps réel"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.update_thread.start()

    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)

    def _monitoring_loop(self):
        """Boucle de monitoring"""
        while self.is_running:
            try:
                self.update_dashboard()
                time.sleep(1)  # Mise à jour chaque seconde
            except Exception as e:
                print(f"Erreur monitoring dashboard: {e}")
                time.sleep(5)

    def update_dashboard(self):
        """Met à jour tout le dashboard"""
        try:
            # Mettre à jour les métriques
            self.update_metrics()

            # Mettre à jour les statuts
            self.update_status_widgets()

            # Mettre à jour les informations personnage
            self.update_character_widgets()

            # Mettre à jour les graphiques
            self.update_charts()

        except Exception as e:
            print(f"Erreur mise à jour dashboard: {e}")

    def update_metrics(self):
        """Met à jour les métriques depuis le contrôleur"""
        if not self.app_controller:
            return

        try:
            # Récupérer les métriques du contrôleur
            # TODO: Implémenter l'interface avec app_controller

            # Simulation pour le moment
            import random
            self.metrics.uptime += 1
            self.metrics.cpu_usage = random.uniform(10, 80)
            self.metrics.memory_usage = random.uniform(200, 800)
            self.metrics.fps = random.uniform(25, 60)
            self.metrics.actions_per_minute = random.uniform(10, 50)

        except Exception as e:
            print(f"Erreur récupération métriques: {e}")

    def update_status_widgets(self):
        """Met à jour les widgets de statut"""
        try:
            # État du bot
            if self.app_controller and hasattr(self.app_controller, 'is_running') and self.app_controller.is_running:
                self.status_widgets["bot_status"].update_status("ACTIF", "En cours d'exécution", "success")
                self.status_label.configure(text="● ACTIF", fg=self.theme.get_colors().accent_success)
            else:
                self.status_widgets["bot_status"].update_status("IDLE", "En attente", "warning")
                self.status_label.configure(text="● IDLE", fg=self.theme.get_colors().accent_warning)

            # Quête actuelle
            quest = self.character_info.current_quest
            self.status_widgets["quest_status"].update_status(
                "ACTIVE" if quest != "None" else "AUCUNE",
                quest if quest != "None" else "Pas de quête",
                "success" if quest != "None" else "info"
            )

            # Santé (simulation)
            self.status_widgets["health_status"].update_status("100%", "Pleine santé", "success")

            # Connexion
            self.status_widgets["connection_status"].update_status("CONNECTÉ", "Stable", "success")

        except Exception as e:
            print(f"Erreur mise à jour statuts: {e}")

    def update_character_widgets(self):
        """Met à jour les widgets du personnage"""
        try:
            char = self.character_info

            self.character_widgets["name"].update_status(char.name, "", "info")
            self.character_widgets["level"].update_status(str(char.level), f"XP: {char.experience}", "info")
            self.character_widgets["class"].update_status(char.class_name, "", "info")
            self.character_widgets["position"].update_status(f"{char.position[0]}, {char.position[1]}", "", "info")
            self.character_widgets["map"].update_status(char.map_name, "", "info")
            self.character_widgets["energy"].update_status(str(char.energy), "/ 10000", "success" if char.energy > 5000 else "warning")

        except Exception as e:
            print(f"Erreur mise à jour personnage: {e}")

    def update_charts(self):
        """Met à jour les graphiques"""
        try:
            self.charts["cpu"].add_data_point(self.metrics.cpu_usage)
            self.charts["memory"].add_data_point(self.metrics.memory_usage)
            self.charts["fps"].add_data_point(self.metrics.fps)
            self.charts["actions"].add_data_point(self.metrics.actions_per_minute)

        except Exception as e:
            print(f"Erreur mise à jour graphiques: {e}")

    def update_metric_widgets(self):
        """Met à jour les widgets de métriques"""
        try:
            metrics = self.metrics

            # Temps d'activité
            uptime_str = str(timedelta(seconds=int(metrics.uptime)))
            self.metric_widgets["uptime"].update_status(uptime_str, "Depuis le démarrage", "info")

            # Quêtes
            self.metric_widgets["quests"].update_status(
                str(metrics.quests_completed),
                "Terminées",
                "success"
            )

            # Expérience
            self.metric_widgets["experience"].update_status(
                f"+{metrics.experience_gained:,}",
                "XP gagnée",
                "success"
            )

            # Kamas
            self.metric_widgets["kamas"].update_status(
                f"+{metrics.kamas_earned:,}",
                "Gagnés",
                "success"
            )

            # Combats
            total_fights = metrics.fights_won + metrics.fights_lost
            fight_text = f"{metrics.fights_won}/{total_fights}" if total_fights > 0 else "0/0"
            self.metric_widgets["fights"].update_status(
                fight_text,
                "Gagnés/Total",
                "success" if metrics.fights_won > metrics.fights_lost else "warning"
            )

            # Taux de succès
            self.metric_widgets["success_rate"].update_status(
                f"{metrics.success_rate:.1f}%",
                "Réussite",
                "success" if metrics.success_rate >= 90 else "warning" if metrics.success_rate >= 70 else "error"
            )

        except Exception as e:
            print(f"Erreur mise à jour métriques widgets: {e}")

def create_dashboard_panel(parent, theme_manager: ThemeManager, app_controller=None) -> DashboardPanel:
    """Factory function pour créer DashboardPanel"""
    return DashboardPanel(parent, theme_manager, app_controller)