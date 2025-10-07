#!/usr/bin/env python3
"""
ConfigPanel - Panneau de configuration générale
Configuration globale de l'application et personnalisation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from typing import Dict, Any, Optional, List, Callable
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from .theme_manager import ThemeManager

@dataclass
class AppSettings:
    """Configuration de l'application"""
    # Interface
    theme: str = "dark"
    language: str = "fr"
    auto_save_interval: int = 300  # secondes
    show_tooltips: bool = True
    confirm_actions: bool = True

    # Fenêtre
    window_width: int = 1200
    window_height: int = 800
    window_maximized: bool = False
    remember_position: bool = True
    always_on_top: bool = False

    # Notifications
    enable_notifications: bool = True
    notification_sound: bool = True
    notification_position: str = "bottom_right"
    show_tray_icon: bool = True

    # Logs et debug
    log_to_file: bool = True
    log_level: str = "INFO"
    max_log_files: int = 10
    log_file_size_mb: int = 10

    # Performance
    update_interval_ms: int = 1000
    max_chart_points: int = 100
    enable_gpu_acceleration: bool = True
    low_performance_mode: bool = False

    # Sécurité
    auto_backup: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 5
    encrypt_configs: bool = False

class ThemeCustomizer:
    """Personnalisateur de thèmes"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager
        self.custom_colors = {}

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface de personnalisation"""
        self.frame = self.theme.create_panel(self.parent)

        # Titre
        title = self.theme.create_subtitle_label(
            self.frame,
            text=" Personnalisation Thème"
        )
        title.pack(pady=(15, 10))

        # Sélection de thème
        theme_frame = self.theme.create_frame(self.frame, "primary")
        theme_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        theme_label = self.theme.create_body_label(theme_frame, text="Thème:")
        theme_label.pack(side=tk.LEFT)

        self.theme_var = tk.StringVar(value=self.theme.current_theme_name)
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=list(self.theme.themes.keys()),
            state="readonly"
        )
        theme_combo.pack(side=tk.RIGHT)
        theme_combo.bind("<<ComboboxSelected>>", self.on_theme_changed)

        # Personnalisation des couleurs
        colors_frame = self.theme.create_frame(self.frame, "secondary")
        colors_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        colors_title = self.theme.create_body_label(colors_frame, text="Couleurs personnalisées:")
        colors_title.pack(anchor="w", padx=10, pady=(10, 5))

        # Grid des couleurs
        color_grid = self.theme.create_frame(colors_frame, "primary")
        color_grid.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Couleurs principales
        self.create_color_picker(color_grid, "Primaire", "primary", 0, 0)
        self.create_color_picker(color_grid, "Secondaire", "secondary", 0, 1)
        self.create_color_picker(color_grid, "Succès", "accent_success", 1, 0)
        self.create_color_picker(color_grid, "Erreur", "accent_error", 1, 1)
        self.create_color_picker(color_grid, "Avertissement", "accent_warning", 2, 0)
        self.create_color_picker(color_grid, "Info", "accent_info", 2, 1)

        # Boutons d'action
        buttons_frame = self.theme.create_frame(colors_frame, "primary")
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        reset_btn = self.theme.create_secondary_button(
            buttons_frame,
            text="[RELOAD] Reset",
            command=self.reset_colors
        )
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))

        apply_btn = self.theme.create_primary_button(
            buttons_frame,
            text="[OK] Appliquer",
            command=self.apply_custom_theme
        )
        apply_btn.pack(side=tk.LEFT)

        export_btn = self.theme.create_secondary_button(
            buttons_frame,
            text="[SAVE] Exporter",
            command=self.export_theme
        )
        export_btn.pack(side=tk.RIGHT)

    def create_color_picker(self, parent, label: str, color_key: str, row: int, col: int):
        """Crée un sélecteur de couleur"""
        frame = self.theme.create_frame(parent, "primary")
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

        # Label
        color_label = self.theme.create_body_label(frame, text=f"{label}:")
        color_label.pack(side=tk.LEFT)

        # Couleur actuelle
        current_color = getattr(self.theme.get_colors(), color_key)

        # Bouton de couleur
        color_btn = tk.Button(
            frame,
            text="●",
            bg=current_color,
            fg=current_color,
            width=3,
            command=lambda: self.pick_color(color_key, color_btn)
        )
        color_btn.pack(side=tk.RIGHT)

        # Stocker la référence
        self.custom_colors[color_key] = {"button": color_btn, "color": current_color}

        # Configuration du grid
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

    def pick_color(self, color_key: str, button: tk.Button):
        """Ouvre le sélecteur de couleur"""
        current_color = self.custom_colors[color_key]["color"]
        color = colorchooser.askcolor(
            color=current_color,
            title=f"Choisir couleur {color_key}"
        )

        if color[1]:  # Si une couleur a été sélectionnée
            new_color = color[1]
            self.custom_colors[color_key]["color"] = new_color
            button.configure(bg=new_color, fg=new_color)

    def on_theme_changed(self, event=None):
        """Gestion du changement de thème"""
        new_theme = self.theme_var.get()
        self.theme.set_theme(new_theme)

        # Mettre à jour les couleurs dans l'interface
        for color_key, color_info in self.custom_colors.items():
            current_color = getattr(self.theme.get_colors(), color_key)
            color_info["color"] = current_color
            color_info["button"].configure(bg=current_color, fg=current_color)

    def reset_colors(self):
        """Remet les couleurs par défaut"""
        if messagebox.askyesno("Reset", "Remettre les couleurs par défaut?"):
            # Remettre le thème original
            self.theme.set_theme(self.theme.current_theme_name)

            # Mettre à jour l'interface
            for color_key, color_info in self.custom_colors.items():
                current_color = getattr(self.theme.get_colors(), color_key)
                color_info["color"] = current_color
                color_info["button"].configure(bg=current_color, fg=current_color)

    def apply_custom_theme(self):
        """Applique le thème personnalisé"""
        # TODO: Implémenter l'application des couleurs personnalisées
        messagebox.showinfo("Info", "Thème personnalisé appliqué (fonctionnalité à implémenter)")

    def export_theme(self):
        """Exporte le thème personnalisé"""
        filename = filedialog.asksaveasfilename(
            title="Exporter thème",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if filename:
            theme_data = {
                "name": f"Custom_{self.theme.current_theme_name}",
                "colors": {key: info["color"] for key, info in self.custom_colors.items()}
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(theme_data, f, indent=2)

            messagebox.showinfo("Succès", f"Thème exporté: {filename}")

class NotificationSettings:
    """Configuration des notifications"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame = self.theme.create_panel(self.parent)

        # Titre
        title = self.theme.create_subtitle_label(
            self.frame,
            text=" Notifications"
        )
        title.pack(pady=(15, 10))

        # Options de notifications
        options_frame = self.theme.create_frame(self.frame, "primary")
        options_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Activer notifications
        self.notifications_var = tk.BooleanVar(value=True)
        notifications_check = tk.Checkbutton(
            options_frame,
            text="Activer les notifications",
            variable=self.notifications_var,
            **self.theme.get_style("body")
        )
        notifications_check.pack(anchor="w", pady=5)

        # Son
        self.sound_var = tk.BooleanVar(value=True)
        sound_check = tk.Checkbutton(
            options_frame,
            text="Son des notifications",
            variable=self.sound_var,
            **self.theme.get_style("body")
        )
        sound_check.pack(anchor="w", pady=5)

        # Icône dans la barre des tâches
        self.tray_var = tk.BooleanVar(value=True)
        tray_check = tk.Checkbutton(
            options_frame,
            text="Icône dans la barre des tâches",
            variable=self.tray_var,
            **self.theme.get_style("body")
        )
        tray_check.pack(anchor="w", pady=5)

        # Position des notifications
        position_frame = self.theme.create_frame(options_frame, "primary")
        position_frame.pack(fill=tk.X, pady=10)

        position_label = self.theme.create_body_label(position_frame, text="Position:")
        position_label.pack(side=tk.LEFT)

        self.position_var = tk.StringVar(value="bottom_right")
        position_combo = ttk.Combobox(
            position_frame,
            textvariable=self.position_var,
            values=["top_left", "top_right", "bottom_left", "bottom_right"],
            state="readonly"
        )
        position_combo.pack(side=tk.RIGHT)

        # Types de notifications
        types_frame = self.theme.create_frame(self.frame, "secondary")
        types_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        types_title = self.theme.create_body_label(types_frame, text="Types de notifications:")
        types_title.pack(anchor="w", padx=10, pady=(10, 5))

        # Checkboxes pour différents types
        notifications_types = [
            ("Quest complétée", "quest_completed"),
            ("Niveau gagné", "level_up"),
            ("Erreur critique", "critical_error"),
            ("Bot arrêté", "bot_stopped"),
            ("Combat perdu", "combat_lost"),
            ("Objectif atteint", "goal_reached")
        ]

        self.notification_types = {}
        for label, key in notifications_types:
            var = tk.BooleanVar(value=True)
            self.notification_types[key] = var

            check = tk.Checkbutton(
                types_frame,
                text=label,
                variable=var,
                **self.theme.get_style("body")
            )
            check.pack(anchor="w", padx=20, pady=2)

        # Test de notification
        test_btn = self.theme.create_secondary_button(
            types_frame,
            text=" Tester notification",
            command=self.test_notification
        )
        test_btn.pack(pady=10)

    def test_notification(self):
        """Teste une notification"""
        messagebox.showinfo("Test", "Ceci est une notification de test!")

class AdvancedSettings:
    """Paramètres avancés"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame = self.theme.create_panel(self.parent)

        # Titre
        title = self.theme.create_subtitle_label(
            self.frame,
            text="[SETTINGS] Paramètres Avancés"
        )
        title.pack(pady=(15, 10))

        # Container avec scroll
        canvas = tk.Canvas(
            self.frame,
            bg=self.theme.get_colors().bg_primary,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)

        scrollable_frame = self.theme.create_frame(canvas, "primary")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(15, 0), pady=(0, 15))
        scrollbar.pack(side="right", fill="y", padx=(0, 15), pady=(0, 15))

        # Performance
        self.create_performance_section(scrollable_frame)

        # Sécurité
        self.create_security_section(scrollable_frame)

        # Debug
        self.create_debug_section(scrollable_frame)

        # Réseau
        self.create_network_section(scrollable_frame)

    def create_performance_section(self, parent):
        """Section performance"""
        section = self.theme.create_panel(parent)
        section.pack(fill=tk.X, pady=(0, 20))

        section_title = self.theme.create_subtitle_label(section, text="[START] Performance")
        section_title.pack(pady=(15, 10))

        content = self.theme.create_frame(section, "primary")
        content.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Intervalle de mise à jour
        update_frame = self.theme.create_frame(content, "primary")
        update_frame.pack(fill=tk.X, pady=5)

        update_label = self.theme.create_body_label(update_frame, text="Intervalle mise à jour (ms):")
        update_label.pack(side=tk.LEFT)

        self.update_interval_var = tk.IntVar(value=1000)
        update_scale = tk.Scale(
            update_frame,
            from_=100,
            to=5000,
            resolution=100,
            orient=tk.HORIZONTAL,
            variable=self.update_interval_var,
            **self.theme.get_style("body")
        )
        update_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Points max graphiques
        points_frame = self.theme.create_frame(content, "primary")
        points_frame.pack(fill=tk.X, pady=5)

        points_label = self.theme.create_body_label(points_frame, text="Points max graphiques:")
        points_label.pack(side=tk.LEFT)

        self.max_points_var = tk.IntVar(value=100)
        points_scale = tk.Scale(
            points_frame,
            from_=50,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.max_points_var,
            **self.theme.get_style("body")
        )
        points_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Options booléennes
        self.gpu_acceleration_var = tk.BooleanVar(value=True)
        gpu_check = tk.Checkbutton(
            content,
            text="Accélération GPU",
            variable=self.gpu_acceleration_var,
            **self.theme.get_style("body")
        )
        gpu_check.pack(anchor="w", pady=2)

        self.low_perf_var = tk.BooleanVar(value=False)
        low_perf_check = tk.Checkbutton(
            content,
            text="Mode basse performance",
            variable=self.low_perf_var,
            **self.theme.get_style("body")
        )
        low_perf_check.pack(anchor="w", pady=2)

    def create_security_section(self, parent):
        """Section sécurité"""
        section = self.theme.create_panel(parent)
        section.pack(fill=tk.X, pady=(0, 20))

        section_title = self.theme.create_subtitle_label(section, text=" Sécurité")
        section_title.pack(pady=(15, 10))

        content = self.theme.create_frame(section, "primary")
        content.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Auto-backup
        self.auto_backup_var = tk.BooleanVar(value=True)
        backup_check = tk.Checkbutton(
            content,
            text="Sauvegarde automatique",
            variable=self.auto_backup_var,
            **self.theme.get_style("body")
        )
        backup_check.pack(anchor="w", pady=2)

        # Intervalle backup
        backup_frame = self.theme.create_frame(content, "primary")
        backup_frame.pack(fill=tk.X, pady=5)

        backup_label = self.theme.create_body_label(backup_frame, text="Intervalle backup (heures):")
        backup_label.pack(side=tk.LEFT)

        self.backup_interval_var = tk.IntVar(value=24)
        backup_scale = tk.Scale(
            backup_frame,
            from_=1,
            to=168,  # 1 semaine
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.backup_interval_var,
            **self.theme.get_style("body")
        )
        backup_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Chiffrement
        self.encrypt_var = tk.BooleanVar(value=False)
        encrypt_check = tk.Checkbutton(
            content,
            text="Chiffrer les configurations",
            variable=self.encrypt_var,
            **self.theme.get_style("body")
        )
        encrypt_check.pack(anchor="w", pady=2)

    def create_debug_section(self, parent):
        """Section debug"""
        section = self.theme.create_panel(parent)
        section.pack(fill=tk.X, pady=(0, 20))

        section_title = self.theme.create_subtitle_label(section, text="[CONFIG] Debug")
        section_title.pack(pady=(15, 10))

        content = self.theme.create_frame(section, "primary")
        content.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Niveau de log
        log_level_frame = self.theme.create_frame(content, "primary")
        log_level_frame.pack(fill=tk.X, pady=5)

        log_level_label = self.theme.create_body_label(log_level_frame, text="Niveau de log:")
        log_level_label.pack(side=tk.LEFT)

        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(
            log_level_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            state="readonly"
        )
        log_level_combo.pack(side=tk.RIGHT)

        # Log vers fichier
        self.log_to_file_var = tk.BooleanVar(value=True)
        log_file_check = tk.Checkbutton(
            content,
            text="Enregistrer logs dans fichier",
            variable=self.log_to_file_var,
            **self.theme.get_style("body")
        )
        log_file_check.pack(anchor="w", pady=2)

        # Taille max fichier log
        log_size_frame = self.theme.create_frame(content, "primary")
        log_size_frame.pack(fill=tk.X, pady=5)

        log_size_label = self.theme.create_body_label(log_size_frame, text="Taille max fichier log (MB):")
        log_size_label.pack(side=tk.LEFT)

        self.log_size_var = tk.IntVar(value=10)
        log_size_scale = tk.Scale(
            log_size_frame,
            from_=1,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.log_size_var,
            **self.theme.get_style("body")
        )
        log_size_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

    def create_network_section(self, parent):
        """Section réseau"""
        section = self.theme.create_panel(parent)
        section.pack(fill=tk.X, pady=(0, 20))

        section_title = self.theme.create_subtitle_label(section, text="[WEB] Réseau")
        section_title.pack(pady=(15, 10))

        content = self.theme.create_frame(section, "primary")
        content.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Timeout connexion
        timeout_frame = self.theme.create_frame(content, "primary")
        timeout_frame.pack(fill=tk.X, pady=5)

        timeout_label = self.theme.create_body_label(timeout_frame, text="Timeout connexion (sec):")
        timeout_label.pack(side=tk.LEFT)

        self.timeout_var = tk.IntVar(value=30)
        timeout_scale = tk.Scale(
            timeout_frame,
            from_=5,
            to=120,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.timeout_var,
            **self.theme.get_style("body")
        )
        timeout_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Retry attempts
        retry_frame = self.theme.create_frame(content, "primary")
        retry_frame.pack(fill=tk.X, pady=5)

        retry_label = self.theme.create_body_label(retry_frame, text="Tentatives de reconnexion:")
        retry_label.pack(side=tk.LEFT)

        self.retry_var = tk.IntVar(value=3)
        retry_scale = tk.Scale(
            retry_frame,
            from_=1,
            to=10,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.retry_var,
            **self.theme.get_style("body")
        )
        retry_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

class ConfigPanel:
    """Panneau principal de configuration"""

    def __init__(self, parent, theme_manager: ThemeManager, app_controller=None):
        self.parent = parent
        self.theme = theme_manager
        self.app_controller = app_controller

        # Configuration
        self.settings = AppSettings()
        self.config_file = Path("config/app_settings.json")

        # Interface
        self.frame = self.theme.create_frame(parent, "primary")
        self.setup_ui()

        # Charger la configuration
        self.load_settings()

    def setup_ui(self):
        """Configure l'interface principale"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme.create_title_label(
            header_frame,
            text="[SETTINGS] Configuration Générale"
        )
        title_label.pack(side=tk.LEFT)

        # Boutons d'action
        buttons_frame = self.theme.create_frame(header_frame, "primary")
        buttons_frame.pack(side=tk.RIGHT)

        import_btn = self.theme.create_secondary_button(
            buttons_frame,
            text=" Importer",
            command=self.import_settings
        )
        import_btn.pack(side=tk.LEFT, padx=(0, 10))

        export_btn = self.theme.create_secondary_button(
            buttons_frame,
            text="[SAVE] Exporter",
            command=self.export_settings
        )
        export_btn.pack(side=tk.LEFT, padx=(0, 10))

        reset_btn = self.theme.create_secondary_button(
            buttons_frame,
            text="[RELOAD] Reset",
            command=self.reset_settings
        )
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))

        apply_btn = self.theme.create_primary_button(
            buttons_frame,
            text="[OK] Appliquer",
            command=self.apply_settings
        )
        apply_btn.pack(side=tk.LEFT)

        # Notebook pour les sections
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Sections de configuration
        self.create_general_section()
        self.create_appearance_section()
        self.create_notifications_section()
        self.create_advanced_section()

    def create_general_section(self):
        """Section générale"""
        general_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(general_frame, text=" Général")

        # Language
        lang_frame = self.theme.create_frame(general_frame, "primary")
        lang_frame.pack(fill=tk.X, padx=20, pady=20)

        lang_label = self.theme.create_body_label(lang_frame, text="Langue:")
        lang_label.pack(side=tk.LEFT)

        self.language_var = tk.StringVar(value=self.settings.language)
        lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=["fr", "en", "es", "de"],
            state="readonly"
        )
        lang_combo.pack(side=tk.RIGHT)

        # Auto-save
        auto_save_frame = self.theme.create_frame(general_frame, "primary")
        auto_save_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        auto_save_label = self.theme.create_body_label(auto_save_frame, text="Auto-save (secondes):")
        auto_save_label.pack(side=tk.LEFT)

        self.auto_save_var = tk.IntVar(value=self.settings.auto_save_interval)
        auto_save_scale = tk.Scale(
            auto_save_frame,
            from_=60,
            to=3600,
            resolution=60,
            orient=tk.HORIZONTAL,
            variable=self.auto_save_var,
            **self.theme.get_style("body")
        )
        auto_save_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Options booléennes
        options_frame = self.theme.create_frame(general_frame, "primary")
        options_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.tooltips_var = tk.BooleanVar(value=self.settings.show_tooltips)
        tooltips_check = tk.Checkbutton(
            options_frame,
            text="Afficher les tooltips",
            variable=self.tooltips_var,
            **self.theme.get_style("body")
        )
        tooltips_check.pack(anchor="w", pady=2)

        self.confirm_var = tk.BooleanVar(value=self.settings.confirm_actions)
        confirm_check = tk.Checkbutton(
            options_frame,
            text="Confirmer les actions",
            variable=self.confirm_var,
            **self.theme.get_style("body")
        )
        confirm_check.pack(anchor="w", pady=2)

        self.remember_pos_var = tk.BooleanVar(value=self.settings.remember_position)
        remember_check = tk.Checkbutton(
            options_frame,
            text="Se souvenir de la position de la fenêtre",
            variable=self.remember_pos_var,
            **self.theme.get_style("body")
        )
        remember_check.pack(anchor="w", pady=2)

        self.always_top_var = tk.BooleanVar(value=self.settings.always_on_top)
        top_check = tk.Checkbutton(
            options_frame,
            text="Toujours au premier plan",
            variable=self.always_top_var,
            **self.theme.get_style("body")
        )
        top_check.pack(anchor="w", pady=2)

    def create_appearance_section(self):
        """Section apparence"""
        appearance_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(appearance_frame, text=" Apparence")

        # Intégrer le personnalisateur de thèmes
        self.theme_customizer = ThemeCustomizer(appearance_frame, self.theme)

    def create_notifications_section(self):
        """Section notifications"""
        notifications_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(notifications_frame, text=" Notifications")

        # Intégrer les paramètres de notifications
        self.notification_settings = NotificationSettings(notifications_frame, self.theme)

    def create_advanced_section(self):
        """Section avancée"""
        advanced_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(advanced_frame, text="[SETTINGS] Avancé")

        # Intégrer les paramètres avancés
        self.advanced_settings = AdvancedSettings(advanced_frame, self.theme)

    def apply_settings(self):
        """Applique les paramètres"""
        try:
            # Mettre à jour les paramètres
            self.settings.language = self.language_var.get()
            self.settings.auto_save_interval = self.auto_save_var.get()
            self.settings.show_tooltips = self.tooltips_var.get()
            self.settings.confirm_actions = self.confirm_var.get()
            self.settings.remember_position = self.remember_pos_var.get()
            self.settings.always_on_top = self.always_top_var.get()

            # Sauvegarder
            self.save_settings()

            # Appliquer les changements
            if self.app_controller:
                # TODO: Appliquer les paramètres au contrôleur
                pass

            messagebox.showinfo("Succès", "Paramètres appliqués avec succès")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'application: {e}")

    def save_settings(self):
        """Sauvegarde les paramètres"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.settings), f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erreur sauvegarde paramètres: {e}")

    def load_settings(self):
        """Charge les paramètres"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)

                # Mettre à jour les paramètres
                for key, value in settings_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)

                # Mettre à jour l'interface
                self.update_ui_from_settings()

        except Exception as e:
            print(f"Erreur chargement paramètres: {e}")

    def update_ui_from_settings(self):
        """Met à jour l'interface depuis les paramètres"""
        if hasattr(self, 'language_var'):
            self.language_var.set(self.settings.language)
        if hasattr(self, 'auto_save_var'):
            self.auto_save_var.set(self.settings.auto_save_interval)
        if hasattr(self, 'tooltips_var'):
            self.tooltips_var.set(self.settings.show_tooltips)
        if hasattr(self, 'confirm_var'):
            self.confirm_var.set(self.settings.confirm_actions)
        if hasattr(self, 'remember_pos_var'):
            self.remember_pos_var.set(self.settings.remember_position)
        if hasattr(self, 'always_top_var'):
            self.always_top_var.set(self.settings.always_on_top)

    def import_settings(self):
        """Importe des paramètres"""
        filename = filedialog.askopenfilename(
            title="Importer paramètres",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)

                for key, value in settings_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)

                self.update_ui_from_settings()
                self.save_settings()
                messagebox.showinfo("Succès", "Paramètres importés avec succès")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'import: {e}")

    def export_settings(self):
        """Exporte les paramètres"""
        filename = filedialog.asksaveasfilename(
            title="Exporter paramètres",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(asdict(self.settings), f, indent=2, ensure_ascii=False)

                messagebox.showinfo("Succès", f"Paramètres exportés: {filename}")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export: {e}")

    def reset_settings(self):
        """Remet les paramètres par défaut"""
        if messagebox.askyesno("Reset", "Remettre tous les paramètres par défaut?"):
            self.settings = AppSettings()
            self.update_ui_from_settings()
            self.save_settings()
            messagebox.showinfo("Reset", "Paramètres remis par défaut")

def create_config_panel(parent, theme_manager: ThemeManager, app_controller=None) -> ConfigPanel:
    """Factory function pour créer ConfigPanel"""
    return ConfigPanel(parent, theme_manager, app_controller)