#!/usr/bin/env python3
"""
AnalyticsPanel - Panneau d'analytics et visualisations temps réel
Analyses avancées, tendances et rapports détaillés
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional, List, Tuple
import threading
import time
import json
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import math

from .theme_manager import ThemeManager

@dataclass
class AnalyticsData:
    """Données d'analytics"""
    timestamp: float
    level: int
    experience: int
    kamas: int
    quests_completed: int
    fights_won: int
    fights_lost: int
    items_collected: int
    deaths: int
    maps_explored: int
    playtime_minutes: float
    efficiency_score: float

@dataclass
class StatisticsReport:
    """Rapport statistique"""
    total_sessions: int = 0
    total_playtime: float = 0.0
    avg_session_time: float = 0.0
    total_experience: int = 0
    total_kamas: int = 0
    total_quests: int = 0
    avg_exp_per_hour: float = 0.0
    avg_kamas_per_hour: float = 0.0
    success_rate: float = 0.0
    efficiency_trend: str = "stable"

class AdvancedChart:
    """Widget graphique avancé avec multiple séries"""

    def __init__(self, parent, theme_manager: ThemeManager, title: str,
                 width: int = 400, height: int = 250):
        self.parent = parent
        self.theme = theme_manager
        self.title = title
        self.width = width
        self.height = height

        # Données
        self.series: Dict[str, List[Tuple[float, float]]] = {}
        self.series_colors: Dict[str, str] = {}
        self.max_points = 100

        # Interface
        self.frame = self.theme.create_panel(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        # Header avec titre et contrôles
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        title_label = self.theme.create_subtitle_label(
            header_frame,
            text=self.title
        )
        title_label.pack(side=tk.LEFT)

        # Boutons de contrôle
        controls_frame = self.theme.create_frame(header_frame, "primary")
        controls_frame.pack(side=tk.RIGHT)

        self.zoom_var = tk.StringVar(value="1h")
        zoom_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.zoom_var,
            values=["5m", "15m", "1h", "6h", "24h", "7d"],
            width=6,
            state="readonly"
        )
        zoom_combo.pack(side=tk.LEFT, padx=(0, 5))
        zoom_combo.bind("<<ComboboxSelected>>", self.on_zoom_changed)

        export_btn = self.theme.create_secondary_button(
            controls_frame,
            text="[STATS]",
            command=self.export_data
        )
        export_btn.configure(width=3)
        export_btn.pack(side=tk.LEFT)

        # Canvas pour le graphique
        self.canvas = tk.Canvas(
            self.frame,
            width=self.width,
            height=self.height,
            bg=self.theme.get_colors().bg_secondary,
            highlightthickness=1,
            highlightcolor=self.theme.get_colors().border_light
        )
        self.canvas.pack(padx=10, pady=(0, 10))

        # Légende
        self.legend_frame = self.theme.create_frame(self.frame, "primary")
        self.legend_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Bind des événements
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

    def add_series(self, name: str, color: str):
        """Ajoute une série de données"""
        self.series[name] = []
        self.series_colors[name] = color
        self.update_legend()

    def add_data_point(self, series_name: str, timestamp: float, value: float):
        """Ajoute un point de données"""
        if series_name not in self.series:
            colors = [
                self.theme.get_colors().primary,
                self.theme.get_colors().accent_success,
                self.theme.get_colors().accent_warning,
                self.theme.get_colors().accent_info,
                self.theme.get_colors().secondary
            ]
            color = colors[len(self.series) % len(colors)]
            self.add_series(series_name, color)

        self.series[series_name].append((timestamp, value))

        # Limiter le nombre de points
        if len(self.series[series_name]) > self.max_points:
            self.series[series_name].pop(0)

        self.update_chart()

    def update_chart(self):
        """Met à jour le graphique"""
        self.canvas.delete("all")

        if not self.series or not any(self.series.values()):
            self.draw_empty_state()
            return

        # Calcul des dimensions
        margin = 40
        chart_width = self.width - 2 * margin
        chart_height = self.height - 2 * margin

        # Calcul des limites
        all_timestamps = []
        all_values = []

        for series_data in self.series.values():
            if series_data:
                all_timestamps.extend([point[0] for point in series_data])
                all_values.extend([point[1] for point in series_data])

        if not all_timestamps or not all_values:
            return

        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        min_val = min(all_values)
        max_val = max(all_values)

        # Éviter la division par zéro
        time_range = max_time - min_time if max_time != min_time else 1
        value_range = max_val - min_val if max_val != min_val else 1

        # Dessiner les axes
        self.draw_axes(margin, chart_width, chart_height, min_val, max_val, min_time, max_time)

        # Dessiner les séries
        for series_name, data in self.series.items():
            if len(data) < 2:
                continue

            color = self.series_colors[series_name]
            points = []

            for timestamp, value in data:
                x = margin + ((timestamp - min_time) / time_range) * chart_width
                y = margin + chart_height - ((value - min_val) / value_range) * chart_height
                points.extend([x, y])

            # Dessiner la ligne
            if len(points) >= 4:
                self.canvas.create_line(
                    points,
                    fill=color,
                    width=2,
                    smooth=True,
                    tags=f"series_{series_name}"
                )

            # Dessiner les points
            for i in range(0, len(points), 2):
                x, y = points[i], points[i + 1]
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3,
                    fill=color,
                    outline=self.theme.get_colors().bg_primary,
                    width=1,
                    tags=f"point_{series_name}"
                )

    def draw_axes(self, margin, chart_width, chart_height, min_val, max_val, min_time, max_time):
        """Dessine les axes et la grille"""
        colors = self.theme.get_colors()

        # Axe Y (valeurs)
        self.canvas.create_line(
            margin, margin,
            margin, margin + chart_height,
            fill=colors.border_medium,
            width=1
        )

        # Axe X (temps)
        self.canvas.create_line(
            margin, margin + chart_height,
            margin + chart_width, margin + chart_height,
            fill=colors.border_medium,
            width=1
        )

        # Grille horizontale
        for i in range(5):
            y = margin + (i * chart_height / 4)
            self.canvas.create_line(
                margin, y,
                margin + chart_width, y,
                fill=colors.border_light,
                width=1,
                dash=(2, 2)
            )

            # Labels Y
            if max_val != min_val:
                value = max_val - (i * (max_val - min_val) / 4)
                label_text = f"{value:.1f}" if value < 1000 else f"{value/1000:.1f}K"
                self.canvas.create_text(
                    margin - 5, y,
                    text=label_text,
                    fill=colors.text_secondary,
                    font=self.theme.get_fonts()["status"],
                    anchor="e"
                )

        # Grille verticale
        for i in range(5):
            x = margin + (i * chart_width / 4)
            self.canvas.create_line(
                x, margin,
                x, margin + chart_height,
                fill=colors.border_light,
                width=1,
                dash=(2, 2)
            )

            # Labels X (temps)
            if max_time != min_time:
                timestamp = min_time + (i * (max_time - min_time) / 4)
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M")
                self.canvas.create_text(
                    x, margin + chart_height + 15,
                    text=time_str,
                    fill=colors.text_secondary,
                    font=self.theme.get_fonts()["status"],
                    anchor="n"
                )

    def draw_empty_state(self):
        """Dessine l'état vide"""
        colors = self.theme.get_colors()

        self.canvas.create_text(
            self.width // 2, self.height // 2,
            text="Aucune donnée disponible",
            fill=colors.text_secondary,
            font=self.theme.get_fonts()["body"],
            anchor="center"
        )

    def update_legend(self):
        """Met à jour la légende"""
        # Effacer la légende existante
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        # Créer la nouvelle légende
        for i, (series_name, color) in enumerate(self.series_colors.items()):
            legend_item = self.theme.create_frame(self.legend_frame, "primary")
            legend_item.pack(side=tk.LEFT, padx=(0, 20))

            # Indicateur de couleur
            color_indicator = tk.Label(
                legend_item,
                text="●",
                fg=color,
                bg=self.theme.get_colors().bg_primary,
                font=("Arial", 12)
            )
            color_indicator.pack(side=tk.LEFT)

            # Nom de la série
            series_label = self.theme.create_body_label(
                legend_item,
                text=series_name
            )
            series_label.pack(side=tk.LEFT, padx=(5, 0))

    def on_zoom_changed(self, event=None):
        """Gestion du changement de zoom"""
        zoom_map = {
            "5m": 300,      # 5 minutes
            "15m": 900,     # 15 minutes
            "1h": 3600,     # 1 heure
            "6h": 21600,    # 6 heures
            "24h": 86400,   # 24 heures
            "7d": 604800    # 7 jours
        }

        zoom_seconds = zoom_map.get(self.zoom_var.get(), 3600)
        current_time = time.time()

        # Filtrer les données selon le zoom
        for series_name in self.series:
            original_data = self.series[series_name][:]
            filtered_data = [
                (timestamp, value) for timestamp, value in original_data
                if current_time - timestamp <= zoom_seconds
            ]
            self.series[series_name] = filtered_data

        self.update_chart()

    def on_click(self, event):
        """Gestion des clics"""
        # TODO: Ajouter interaction (zoom, sélection, etc.)
        pass

    def on_motion(self, event):
        """Gestion du mouvement de souris pour tooltip"""
        # TODO: Afficher tooltip avec valeurs
        pass

    def export_data(self):
        """Exporte les données du graphique"""
        if not self.series:
            messagebox.showwarning("Attention", "Aucune donnée à exporter")
            return

        filename = filedialog.asksaveasfilename(
            title="Exporter données",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")]
        )

        if filename:
            try:
                if filename.endswith('.csv'):
                    self.export_csv(filename)
                else:
                    self.export_json(filename)

                messagebox.showinfo("Succès", f"Données exportées: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export: {e}")

    def export_csv(self, filename):
        """Exporte en CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Series", "Value"])

            for series_name, data in self.series.items():
                for timestamp, value in data:
                    writer.writerow([timestamp, series_name, value])

    def export_json(self, filename):
        """Exporte en JSON"""
        export_data = {}
        for series_name, data in self.series.items():
            export_data[series_name] = [
                {"timestamp": timestamp, "value": value}
                for timestamp, value in data
            ]

        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)

class StatisticsWidget:
    """Widget de statistiques"""

    def __init__(self, parent, theme_manager: ThemeManager, title: str):
        self.parent = parent
        self.theme = theme_manager
        self.title = title

        self.frame = self.theme.create_panel(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        # Titre
        title_label = self.theme.create_subtitle_label(
            self.frame,
            text=self.title
        )
        title_label.pack(pady=(15, 10))

        # Container des stats
        self.stats_frame = self.theme.create_frame(self.frame, "primary")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Stats par défaut
        self.stat_labels = {}

    def add_stat(self, key: str, label: str, value: str = "--",
                 trend: str = "neutral", format_func=None):
        """Ajoute une statistique"""
        stat_frame = self.theme.create_frame(self.stats_frame, "primary")
        stat_frame.pack(fill=tk.X, pady=2)

        # Label
        label_widget = self.theme.create_body_label(
            stat_frame,
            text=f"{label}:"
        )
        label_widget.pack(side=tk.LEFT)

        # Valeur
        value_widget = self.theme.create_body_label(
            stat_frame,
            text=value,
            fg=self.get_trend_color(trend)
        )
        value_widget.pack(side=tk.RIGHT)

        self.stat_labels[key] = {
            'widget': value_widget,
            'format_func': format_func
        }

    def update_stat(self, key: str, value: Any, trend: str = "neutral"):
        """Met à jour une statistique"""
        if key in self.stat_labels:
            stat_info = self.stat_labels[key]
            widget = stat_info['widget']
            format_func = stat_info['format_func']

            display_value = format_func(value) if format_func else str(value)
            widget.configure(
                text=display_value,
                fg=self.get_trend_color(trend)
            )

    def get_trend_color(self, trend: str) -> str:
        """Retourne la couleur selon la tendance"""
        colors = self.theme.get_colors()

        trend_colors = {
            "positive": colors.accent_success,
            "negative": colors.accent_error,
            "neutral": colors.text_primary,
            "warning": colors.accent_warning
        }

        return trend_colors.get(trend, colors.text_primary)

class AnalyticsPanel:
    """Panneau principal d'analytics"""

    def __init__(self, parent, theme_manager: ThemeManager, app_controller=None):
        self.parent = parent
        self.theme = theme_manager
        self.app_controller = app_controller

        # Données
        self.analytics_data: List[AnalyticsData] = []
        self.current_session_start = time.time()
        self.statistics = StatisticsReport()

        # Threads
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Interface
        self.frame = self.theme.create_frame(parent, "primary")
        self.setup_ui()

        # Démarrer le monitoring
        self.start_monitoring()

    def setup_ui(self):
        """Configure l'interface principale"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme.create_title_label(
            header_frame,
            text="[STATS] Analytics et Rapports"
        )
        title_label.pack(side=tk.LEFT)

        # Contrôles globaux
        controls_frame = self.theme.create_frame(header_frame, "primary")
        controls_frame.pack(side=tk.RIGHT)

        refresh_btn = self.theme.create_secondary_button(
            controls_frame,
            text="[RELOAD] Actualiser",
            command=self.refresh_analytics
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        report_btn = self.theme.create_primary_button(
            controls_frame,
            text=" Rapport",
            command=self.generate_report
        )
        report_btn.pack(side=tk.LEFT)

        # Notebook pour les sections
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Sections
        self.create_performance_section()
        self.create_progression_section()
        self.create_efficiency_section()
        self.create_statistics_section()

    def create_performance_section(self):
        """Section performance temps réel"""
        perf_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(perf_frame, text=" Performance")

        # Grid des graphiques de performance
        charts_frame = self.theme.create_frame(perf_frame, "primary")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Graphique XP/heure
        self.exp_chart = AdvancedChart(
            charts_frame,
            self.theme,
            "Expérience par Heure",
            width=380,
            height=200
        )
        self.exp_chart.frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.exp_chart.add_series("XP/h", self.theme.get_colors().primary)

        # Graphique Kamas/heure
        self.kamas_chart = AdvancedChart(
            charts_frame,
            self.theme,
            "Kamas par Heure",
            width=380,
            height=200
        )
        self.kamas_chart.frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.kamas_chart.add_series("Kamas/h", self.theme.get_colors().accent_success)

        # Graphique Quêtes/heure
        self.quests_chart = AdvancedChart(
            charts_frame,
            self.theme,
            "Quêtes par Heure",
            width=380,
            height=200
        )
        self.quests_chart.frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.quests_chart.add_series("Quêtes/h", self.theme.get_colors().accent_warning)

        # Graphique Efficacité
        self.efficiency_chart = AdvancedChart(
            charts_frame,
            self.theme,
            "Score d'Efficacité",
            width=380,
            height=200
        )
        self.efficiency_chart.frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.efficiency_chart.add_series("Efficacité", self.theme.get_colors().accent_info)

        # Configuration du grid
        charts_frame.grid_columnconfigure(0, weight=1)
        charts_frame.grid_columnconfigure(1, weight=1)
        charts_frame.grid_rowconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(1, weight=1)

    def create_progression_section(self):
        """Section progression du personnage"""
        prog_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(prog_frame, text="[CHART] Progression")

        # Graphique de progression multi-séries
        self.progression_chart = AdvancedChart(
            prog_frame,
            self.theme,
            "Progression du Personnage",
            width=780,
            height=300
        )
        self.progression_chart.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Ajouter plusieurs séries
        colors = self.theme.get_colors()
        self.progression_chart.add_series("Niveau", colors.primary)
        self.progression_chart.add_series("Expérience", colors.accent_success)
        self.progression_chart.add_series("Kamas", colors.accent_warning)
        self.progression_chart.add_series("Quêtes", colors.accent_info)

        # Stats de progression
        stats_frame = self.theme.create_frame(prog_frame, "primary")
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.progression_stats = StatisticsWidget(
            stats_frame,
            self.theme,
            "[STATS] Statistiques de Progression"
        )
        self.progression_stats.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Ajouter les stats
        self.progression_stats.add_stat("level_gained", "Niveaux gagnés", "0")
        self.progression_stats.add_stat("exp_total", "Expérience totale", "0",
                                       format_func=lambda x: f"{x:,}")
        self.progression_stats.add_stat("kamas_total", "Kamas totaux", "0",
                                       format_func=lambda x: f"{x:,}")
        self.progression_stats.add_stat("quests_total", "Quêtes terminées", "0")
        self.progression_stats.add_stat("time_per_level", "Temps/niveau", "0m",
                                       format_func=lambda x: f"{x:.0f}m")

        # Objectifs
        objectives_widget = StatisticsWidget(
            stats_frame,
            self.theme,
            "[TARGET] Objectifs"
        )
        objectives_widget.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        objectives_widget.add_stat("next_level", "Prochain niveau", "1h 30m")
        objectives_widget.add_stat("daily_exp", "XP objectif quotidien", "75%")
        objectives_widget.add_stat("daily_kamas", "Kamas objectif", "120%")
        objectives_widget.add_stat("weekly_quests", "Quêtes semaine", "8/10")

    def create_efficiency_section(self):
        """Section analyse d'efficacité"""
        eff_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(eff_frame, text="[TARGET] Efficacité")

        # Container principal
        main_container = self.theme.create_frame(eff_frame, "primary")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Graphique d'efficacité temporel
        efficiency_container = self.theme.create_frame(main_container, "primary")
        efficiency_container.pack(fill=tk.X, pady=(0, 10))

        self.efficiency_temporal_chart = AdvancedChart(
            efficiency_container,
            self.theme,
            "Efficacité dans le Temps",
            width=780,
            height=200
        )
        self.efficiency_temporal_chart.frame.pack()

        colors = self.theme.get_colors()
        self.efficiency_temporal_chart.add_series("Score Global", colors.primary)
        self.efficiency_temporal_chart.add_series("Combat", colors.accent_error)
        self.efficiency_temporal_chart.add_series("Navigation", colors.accent_success)
        self.efficiency_temporal_chart.add_series("Quêtes", colors.accent_warning)

        # Analyses et recommandations
        analysis_frame = self.theme.create_frame(main_container, "primary")
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        # Stats d'efficacité
        efficiency_stats = StatisticsWidget(
            analysis_frame,
            self.theme,
            "[STATS] Métriques d'Efficacité"
        )
        efficiency_stats.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        efficiency_stats.add_stat("overall_score", "Score global", "85%")
        efficiency_stats.add_stat("combat_efficiency", "Efficacité combat", "92%")
        efficiency_stats.add_stat("nav_efficiency", "Efficacité navigation", "78%")
        efficiency_stats.add_stat("quest_efficiency", "Efficacité quêtes", "88%")
        efficiency_stats.add_stat("idle_time", "Temps d'inactivité", "12%")

        # Recommandations
        recommendations_frame = self.theme.create_panel(analysis_frame)
        recommendations_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        rec_title = self.theme.create_subtitle_label(
            recommendations_frame,
            text="[INFO] Recommandations"
        )
        rec_title.pack(pady=(15, 10))

        self.recommendations_text = tk.Text(
            recommendations_frame,
            height=8,
            wrap=tk.WORD,
            bg=self.theme.get_colors().bg_secondary,
            fg=self.theme.get_colors().text_primary,
            font=self.theme.get_fonts()["body"],
            relief="flat",
            bd=0
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Recommandations initiales
        recommendations = [
            "- Optimiser les temps de pause entre combats",
            "- Améliorer la précision du pathfinding",
            "- Réduire les délais d'interaction avec les NPCs",
            "- Ajuster la stratégie de combat pour certains mobs",
            "- Optimiser l'ordre des quêtes"
        ]

        for rec in recommendations:
            self.recommendations_text.insert(tk.END, rec + "\n")

        self.recommendations_text.configure(state="disabled")

    def create_statistics_section(self):
        """Section statistiques globales"""
        stats_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(stats_frame, text=" Statistiques")

        # Container avec scroll
        canvas = tk.Canvas(
            stats_frame,
            bg=self.theme.get_colors().bg_primary,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=canvas.yview)

        scrollable_frame = self.theme.create_frame(canvas, "primary")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)

        # Statistiques de session
        session_stats = StatisticsWidget(
            scrollable_frame,
            self.theme,
            " Session Actuelle"
        )
        session_stats.frame.pack(fill=tk.X, pady=(0, 20))

        session_stats.add_stat("session_time", "Durée session", "0h 0m")
        session_stats.add_stat("session_exp", "XP gagnée", "0")
        session_stats.add_stat("session_kamas", "Kamas gagnés", "0")
        session_stats.add_stat("session_quests", "Quêtes terminées", "0")
        session_stats.add_stat("session_fights", "Combats", "0")

        # Statistiques globales
        global_stats = StatisticsWidget(
            scrollable_frame,
            self.theme,
            " Statistiques Globales"
        )
        global_stats.frame.pack(fill=tk.X, pady=(0, 20))

        global_stats.add_stat("total_playtime", "Temps total", "0h 0m")
        global_stats.add_stat("total_sessions", "Sessions", "0")
        global_stats.add_stat("avg_session", "Durée moyenne", "0h 0m")
        global_stats.add_stat("total_exp", "XP totale", "0")
        global_stats.add_stat("total_kamas", "Kamas totaux", "0")

        # Comparaisons et records
        records_stats = StatisticsWidget(
            scrollable_frame,
            self.theme,
            " Records"
        )
        records_stats.frame.pack(fill=tk.X, pady=(0, 20))

        records_stats.add_stat("best_exp_hour", "Meilleure XP/h", "0")
        records_stats.add_stat("best_kamas_hour", "Meilleurs Kamas/h", "0")
        records_stats.add_stat("longest_session", "Plus longue session", "0h 0m")
        records_stats.add_stat("best_efficiency", "Meilleure efficacité", "0%")
        records_stats.add_stat("total_maps", "Cartes explorées", "0")

        # Stockage des widgets pour mise à jour
        self.session_stats_widget = session_stats
        self.global_stats_widget = global_stats
        self.records_stats_widget = records_stats

    def start_monitoring(self):
        """Démarre le monitoring analytics"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitoring_loop(self):
        """Boucle de monitoring analytics"""
        while self.is_monitoring:
            try:
                self.collect_analytics_data()
                self.update_charts()
                self.update_statistics()
                time.sleep(10)  # Collecte toutes les 10 secondes
            except Exception as e:
                print(f"Erreur monitoring analytics: {e}")
                time.sleep(30)

    def collect_analytics_data(self):
        """Collecte les données d'analytics"""
        try:
            # Simuler la collecte de données
            # TODO: Intégrer avec le contrôleur d'application
            current_time = time.time()

            # Données simulées
            import random
            data = AnalyticsData(
                timestamp=current_time,
                level=random.randint(1, 200),
                experience=random.randint(0, 1000000),
                kamas=random.randint(0, 100000),
                quests_completed=random.randint(0, 50),
                fights_won=random.randint(0, 100),
                fights_lost=random.randint(0, 10),
                items_collected=random.randint(0, 200),
                deaths=random.randint(0, 5),
                maps_explored=random.randint(0, 50),
                playtime_minutes=(current_time - self.current_session_start) / 60,
                efficiency_score=random.uniform(70, 95)
            )

            self.analytics_data.append(data)

            # Limiter les données stockées
            if len(self.analytics_data) > 1000:
                self.analytics_data = self.analytics_data[-1000:]

        except Exception as e:
            print(f"Erreur collecte données: {e}")

    def update_charts(self):
        """Met à jour tous les graphiques"""
        if not self.analytics_data:
            return

        try:
            current_time = time.time()
            recent_data = [
                d for d in self.analytics_data
                if current_time - d.timestamp <= 3600  # Dernière heure
            ]

            if len(recent_data) < 2:
                return

            # Calculer les métriques par heure
            for i, data in enumerate(recent_data[1:], 1):
                prev_data = recent_data[i-1]
                time_diff_hours = (data.timestamp - prev_data.timestamp) / 3600

                if time_diff_hours > 0:
                    exp_per_hour = (data.experience - prev_data.experience) / time_diff_hours
                    kamas_per_hour = (data.kamas - prev_data.kamas) / time_diff_hours
                    quests_per_hour = (data.quests_completed - prev_data.quests_completed) / time_diff_hours

                    # Mettre à jour les graphiques
                    self.exp_chart.add_data_point("XP/h", data.timestamp, exp_per_hour)
                    self.kamas_chart.add_data_point("Kamas/h", data.timestamp, kamas_per_hour)
                    self.quests_chart.add_data_point("Quêtes/h", data.timestamp, quests_per_hour)
                    self.efficiency_chart.add_data_point("Efficacité", data.timestamp, data.efficiency_score)

                    # Graphique de progression
                    self.progression_chart.add_data_point("Niveau", data.timestamp, data.level)
                    self.progression_chart.add_data_point("Expérience", data.timestamp, data.experience / 1000)  # En milliers
                    self.progression_chart.add_data_point("Kamas", data.timestamp, data.kamas / 100)  # En centaines
                    self.progression_chart.add_data_point("Quêtes", data.timestamp, data.quests_completed)

        except Exception as e:
            print(f"Erreur mise à jour graphiques: {e}")

    def update_statistics(self):
        """Met à jour les statistiques"""
        if not self.analytics_data:
            return

        try:
            current_data = self.analytics_data[-1]
            session_time_hours = current_data.playtime_minutes / 60

            # Mise à jour des stats de session
            self.session_stats_widget.update_stat(
                "session_time",
                f"{int(session_time_hours)}h {int(current_data.playtime_minutes % 60)}m"
            )
            self.session_stats_widget.update_stat("session_exp", current_data.experience)
            self.session_stats_widget.update_stat("session_kamas", current_data.kamas)
            self.session_stats_widget.update_stat("session_quests", current_data.quests_completed)
            self.session_stats_widget.update_stat("session_fights",
                                                  current_data.fights_won + current_data.fights_lost)

        except Exception as e:
            print(f"Erreur mise à jour statistiques: {e}")

    def refresh_analytics(self):
        """Actualise les analytics"""
        try:
            self.collect_analytics_data()
            self.update_charts()
            self.update_statistics()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'actualisation: {e}")

    def generate_report(self):
        """Génère un rapport complet"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Sauvegarder rapport",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("JSON files", "*.json")]
            )

            if filename:
                if filename.endswith('.html'):
                    self.generate_html_report(filename)
                else:
                    self.generate_json_report(filename)

                messagebox.showinfo("Succès", f"Rapport généré: {filename}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur génération rapport: {e}")

    def generate_html_report(self, filename):
        """Génère un rapport HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport Analytics DOFUS AlphaStar</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2563eb; color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px; }}
                .stat {{ display: flex; justify-content: space-between; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1> Rapport Analytics DOFUS AlphaStar</h1>
                <p>Généré le {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <div class="section">
                <h2>[STATS] Résumé de Session</h2>
                <div class="stat"><span>Durée totale:</span><span>{len(self.analytics_data)} points de données</span></div>
                <div class="stat"><span>Efficacité moyenne:</span><span>N/A</span></div>
            </div>

            <div class="section">
                <h2>[CHART] Données Collectées</h2>
                <p>Total de {len(self.analytics_data)} points de données analytics</p>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_json_report(self, filename):
        """Génère un rapport JSON"""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "session_start": self.current_session_start,
            "total_data_points": len(self.analytics_data),
            "analytics_data": [asdict(data) for data in self.analytics_data],
            "statistics": asdict(self.statistics)
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

def create_analytics_panel(parent, theme_manager: ThemeManager, app_controller=None) -> AnalyticsPanel:
    """Factory function pour créer AnalyticsPanel"""
    return AnalyticsPanel(parent, theme_manager, app_controller)