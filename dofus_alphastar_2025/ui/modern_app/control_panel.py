#!/usr/bin/env python3
"""
ControlPanel - Panneau de contrôle et configuration
Interface pour gérer le bot, ses paramètres et configurations
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional, Callable, List
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from .theme_manager import ThemeManager

@dataclass
class BotConfig:
    """Configuration du bot"""
    # Contrôles de base
    auto_start: bool = False
    auto_reconnect: bool = True
    max_runtime_hours: float = 8.0

    # Quêtes
    quest_mode: str = "auto"  # auto, manual, specific
    selected_quests: List[str] = None
    quest_priority: str = "level"  # level, rewards, time

    # Navigation
    movement_speed: str = "normal"  # slow, normal, fast
    pathfinding_mode: str = "ganymede"  # basic, ganymede, advanced
    avoid_monsters: bool = True
    avoid_players: bool = False

    # Combat
    combat_mode: str = "auto"  # auto, manual, aggressive, defensive
    auto_healing: bool = True
    heal_threshold: int = 70
    use_consumables: bool = True

    # IA et Reasoning
    ai_model: str = "alphastar"  # alphastar, basic, hybrid
    reasoning_level: str = "full"  # basic, normal, full
    learning_enabled: bool = True
    adaptation_rate: float = 0.1

    # Performance
    fps_limit: int = 60
    cpu_limit: int = 80
    memory_limit: int = 2048
    gpu_acceleration: bool = True

    # Sécurité
    screenshot_interval: int = 30
    log_level: str = "info"  # debug, info, warning, error
    save_replays: bool = True

    def __post_init__(self):
        if self.selected_quests is None:
            self.selected_quests = []

class ConfigSection:
    """Section de configuration réutilisable"""

    def __init__(self, parent, theme_manager: ThemeManager, title: str, icon: str = "[SETTINGS]"):
        self.parent = parent
        self.theme = theme_manager
        self.title = title
        self.icon = icon

        self.frame = self.theme.create_panel(parent)
        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface de base"""
        # Header avec titre
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=15, pady=(15, 10))

        title_label = self.theme.create_subtitle_label(
            header_frame,
            text=f"{self.icon} {self.title}"
        )
        title_label.pack(side=tk.LEFT)

        # Container pour le contenu
        self.content_frame = self.theme.create_frame(self.frame, "primary")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

class ControlPanel:
    """Panneau principal de contrôle"""

    def __init__(self, parent, theme_manager: ThemeManager, app_controller=None):
        self.parent = parent
        self.theme = theme_manager
        self.app_controller = app_controller

        # Configuration
        self.config = BotConfig()
        self.config_file = Path("config/bot_config.json")

        # Variables tkinter
        self.tk_vars = {}

        # Interface
        self.frame = self.theme.create_frame(parent, "primary")
        self.setup_ui()

        # Charger la configuration
        self.load_config()

    def setup_ui(self):
        """Configure l'interface principale"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme.create_title_label(
            header_frame,
            text="[GAME] Contrôle et Configuration"
        )
        title_label.pack(side=tk.LEFT)

        # Boutons d'action rapide
        self.create_quick_actions(header_frame)

        # Notebook pour les sections
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Sections de configuration
        self.create_basic_controls()
        self.create_quest_config()
        self.create_navigation_config()
        self.create_combat_config()
        self.create_ai_config()
        self.create_performance_config()
        self.create_security_config()

    def create_quick_actions(self, parent):
        """Boutons d'action rapide"""
        actions_frame = self.theme.create_frame(parent, "primary")
        actions_frame.pack(side=tk.RIGHT)

        # Bouton Start/Stop
        self.start_stop_btn = self.theme.create_primary_button(
            actions_frame,
            text=">️ DÉMARRER",
            command=self.toggle_bot
        )
        self.start_stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Bouton Pause/Resume
        self.pause_btn = self.theme.create_secondary_button(
            actions_frame,
            text="||️ PAUSE",
            command=self.toggle_pause,
            state="disabled"
        )
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Bouton Emergency Stop
        emergency_btn = self.theme.create_secondary_button(
            actions_frame,
            text=" ARRÊT D'URGENCE",
            command=self.emergency_stop
        )
        emergency_btn.configure(bg=self.theme.get_colors().accent_error)
        emergency_btn.pack(side=tk.LEFT)

    def create_basic_controls(self):
        """Section contrôles de base"""
        section = ConfigSection(self.notebook, self.theme, "Contrôles de Base", "[TARGET]")
        self.notebook.add(section.frame, text="[TARGET] Contrôles")

        # Auto-démarrage
        auto_start_var = tk.BooleanVar(value=self.config.auto_start)
        self.tk_vars["auto_start"] = auto_start_var

        auto_start_check = tk.Checkbutton(
            section.content_frame,
            text="Démarrage automatique",
            variable=auto_start_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("auto_start", auto_start_var.get())
        )
        auto_start_check.pack(anchor="w", pady=5)

        # Auto-reconnexion
        auto_reconnect_var = tk.BooleanVar(value=self.config.auto_reconnect)
        self.tk_vars["auto_reconnect"] = auto_reconnect_var

        auto_reconnect_check = tk.Checkbutton(
            section.content_frame,
            text="Reconnexion automatique",
            variable=auto_reconnect_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("auto_reconnect", auto_reconnect_var.get())
        )
        auto_reconnect_check.pack(anchor="w", pady=5)

        # Durée maximale
        runtime_frame = self.theme.create_frame(section.content_frame, "primary")
        runtime_frame.pack(fill=tk.X, pady=10)

        runtime_label = self.theme.create_body_label(
            runtime_frame,
            text="Durée maximale (heures):"
        )
        runtime_label.pack(side=tk.LEFT)

        runtime_var = tk.DoubleVar(value=self.config.max_runtime_hours)
        self.tk_vars["max_runtime_hours"] = runtime_var

        runtime_scale = tk.Scale(
            runtime_frame,
            from_=0.5,
            to=24.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=runtime_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("max_runtime_hours", float(val))
        )
        runtime_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

    def create_quest_config(self):
        """Section configuration des quêtes"""
        section = ConfigSection(self.notebook, self.theme, "Configuration des Quêtes", "")
        self.notebook.add(section.frame, text=" Quêtes")

        # Mode de quête
        mode_frame = self.theme.create_frame(section.content_frame, "primary")
        mode_frame.pack(fill=tk.X, pady=10)

        mode_label = self.theme.create_body_label(
            mode_frame,
            text="Mode de quête:"
        )
        mode_label.pack(side=tk.LEFT)

        quest_mode_var = tk.StringVar(value=self.config.quest_mode)
        self.tk_vars["quest_mode"] = quest_mode_var

        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=quest_mode_var,
            values=["auto", "manual", "specific"],
            state="readonly"
        )
        mode_combo.pack(side=tk.RIGHT, padx=(20, 0))
        mode_combo.bind("<<ComboboxSelected>>",
                       lambda e: self.update_config("quest_mode", quest_mode_var.get()))

        # Priorité des quêtes
        priority_frame = self.theme.create_frame(section.content_frame, "primary")
        priority_frame.pack(fill=tk.X, pady=10)

        priority_label = self.theme.create_body_label(
            priority_frame,
            text="Priorité:"
        )
        priority_label.pack(side=tk.LEFT)

        quest_priority_var = tk.StringVar(value=self.config.quest_priority)
        self.tk_vars["quest_priority"] = quest_priority_var

        priority_combo = ttk.Combobox(
            priority_frame,
            textvariable=quest_priority_var,
            values=["level", "rewards", "time"],
            state="readonly"
        )
        priority_combo.pack(side=tk.RIGHT, padx=(20, 0))
        priority_combo.bind("<<ComboboxSelected>>",
                           lambda e: self.update_config("quest_priority", quest_priority_var.get()))

        # Liste des quêtes spécifiques
        quest_list_frame = self.theme.create_frame(section.content_frame, "secondary")
        quest_list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        quest_list_label = self.theme.create_body_label(
            quest_list_frame,
            text="Quêtes sélectionnées:"
        )
        quest_list_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Listbox avec scrollbar
        listbox_frame = self.theme.create_frame(quest_list_frame, "primary")
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.quest_listbox = tk.Listbox(
            listbox_frame,
            selectmode=tk.MULTIPLE,
            **self.theme.get_style("body")
        )

        quest_scrollbar = ttk.Scrollbar(listbox_frame, command=self.quest_listbox.yview)
        self.quest_listbox.configure(yscrollcommand=quest_scrollbar.set)

        self.quest_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        quest_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Boutons de gestion des quêtes
        quest_buttons_frame = self.theme.create_frame(quest_list_frame, "primary")
        quest_buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        add_quest_btn = self.theme.create_secondary_button(
            quest_buttons_frame,
            text=" Ajouter",
            command=self.add_quest
        )
        add_quest_btn.pack(side=tk.LEFT, padx=(0, 5))

        remove_quest_btn = self.theme.create_secondary_button(
            quest_buttons_frame,
            text=" Supprimer",
            command=self.remove_quest
        )
        remove_quest_btn.pack(side=tk.LEFT, padx=(0, 5))

        load_quest_btn = self.theme.create_secondary_button(
            quest_buttons_frame,
            text=" Charger liste",
            command=self.load_quest_list
        )
        load_quest_btn.pack(side=tk.LEFT)

    def create_navigation_config(self):
        """Section configuration navigation"""
        section = ConfigSection(self.notebook, self.theme, "Configuration Navigation", "[MAP]")
        self.notebook.add(section.frame, text="[MAP] Navigation")

        # Vitesse de déplacement
        speed_frame = self.theme.create_frame(section.content_frame, "primary")
        speed_frame.pack(fill=tk.X, pady=10)

        speed_label = self.theme.create_body_label(
            speed_frame,
            text="Vitesse de déplacement:"
        )
        speed_label.pack(side=tk.LEFT)

        movement_speed_var = tk.StringVar(value=self.config.movement_speed)
        self.tk_vars["movement_speed"] = movement_speed_var

        speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=movement_speed_var,
            values=["slow", "normal", "fast"],
            state="readonly"
        )
        speed_combo.pack(side=tk.RIGHT, padx=(20, 0))
        speed_combo.bind("<<ComboboxSelected>>",
                        lambda e: self.update_config("movement_speed", movement_speed_var.get()))

        # Mode pathfinding
        pathfinding_frame = self.theme.create_frame(section.content_frame, "primary")
        pathfinding_frame.pack(fill=tk.X, pady=10)

        pathfinding_label = self.theme.create_body_label(
            pathfinding_frame,
            text="Mode pathfinding:"
        )
        pathfinding_label.pack(side=tk.LEFT)

        pathfinding_mode_var = tk.StringVar(value=self.config.pathfinding_mode)
        self.tk_vars["pathfinding_mode"] = pathfinding_mode_var

        pathfinding_combo = ttk.Combobox(
            pathfinding_frame,
            textvariable=pathfinding_mode_var,
            values=["basic", "ganymede", "advanced"],
            state="readonly"
        )
        pathfinding_combo.pack(side=tk.RIGHT, padx=(20, 0))
        pathfinding_combo.bind("<<ComboboxSelected>>",
                              lambda e: self.update_config("pathfinding_mode", pathfinding_mode_var.get()))

        # Options d'évitement
        avoid_monsters_var = tk.BooleanVar(value=self.config.avoid_monsters)
        self.tk_vars["avoid_monsters"] = avoid_monsters_var

        avoid_monsters_check = tk.Checkbutton(
            section.content_frame,
            text="Éviter les monstres agressifs",
            variable=avoid_monsters_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("avoid_monsters", avoid_monsters_var.get())
        )
        avoid_monsters_check.pack(anchor="w", pady=5)

        avoid_players_var = tk.BooleanVar(value=self.config.avoid_players)
        self.tk_vars["avoid_players"] = avoid_players_var

        avoid_players_check = tk.Checkbutton(
            section.content_frame,
            text="Éviter les autres joueurs",
            variable=avoid_players_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("avoid_players", avoid_players_var.get())
        )
        avoid_players_check.pack(anchor="w", pady=5)

    def create_combat_config(self):
        """Section configuration combat"""
        section = ConfigSection(self.notebook, self.theme, "Configuration Combat", "[COMBAT]")
        self.notebook.add(section.frame, text="[COMBAT] Combat")

        # Mode de combat
        combat_mode_frame = self.theme.create_frame(section.content_frame, "primary")
        combat_mode_frame.pack(fill=tk.X, pady=10)

        combat_mode_label = self.theme.create_body_label(
            combat_mode_frame,
            text="Mode de combat:"
        )
        combat_mode_label.pack(side=tk.LEFT)

        combat_mode_var = tk.StringVar(value=self.config.combat_mode)
        self.tk_vars["combat_mode"] = combat_mode_var

        combat_mode_combo = ttk.Combobox(
            combat_mode_frame,
            textvariable=combat_mode_var,
            values=["auto", "manual", "aggressive", "defensive"],
            state="readonly"
        )
        combat_mode_combo.pack(side=tk.RIGHT, padx=(20, 0))
        combat_mode_combo.bind("<<ComboboxSelected>>",
                              lambda e: self.update_config("combat_mode", combat_mode_var.get()))

        # Auto-healing
        auto_healing_var = tk.BooleanVar(value=self.config.auto_healing)
        self.tk_vars["auto_healing"] = auto_healing_var

        auto_healing_check = tk.Checkbutton(
            section.content_frame,
            text="Soins automatiques",
            variable=auto_healing_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("auto_healing", auto_healing_var.get())
        )
        auto_healing_check.pack(anchor="w", pady=5)

        # Seuil de soins
        heal_frame = self.theme.create_frame(section.content_frame, "primary")
        heal_frame.pack(fill=tk.X, pady=10)

        heal_label = self.theme.create_body_label(
            heal_frame,
            text="Seuil de soins (% PV):"
        )
        heal_label.pack(side=tk.LEFT)

        heal_threshold_var = tk.IntVar(value=self.config.heal_threshold)
        self.tk_vars["heal_threshold"] = heal_threshold_var

        heal_scale = tk.Scale(
            heal_frame,
            from_=10,
            to=90,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=heal_threshold_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("heal_threshold", int(val))
        )
        heal_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Utiliser consommables
        use_consumables_var = tk.BooleanVar(value=self.config.use_consumables)
        self.tk_vars["use_consumables"] = use_consumables_var

        use_consumables_check = tk.Checkbutton(
            section.content_frame,
            text="Utiliser les consommables",
            variable=use_consumables_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("use_consumables", use_consumables_var.get())
        )
        use_consumables_check.pack(anchor="w", pady=5)

    def create_ai_config(self):
        """Section configuration IA"""
        section = ConfigSection(self.notebook, self.theme, "Configuration IA", "")
        self.notebook.add(section.frame, text=" IA")

        # Modèle d'IA
        ai_model_frame = self.theme.create_frame(section.content_frame, "primary")
        ai_model_frame.pack(fill=tk.X, pady=10)

        ai_model_label = self.theme.create_body_label(
            ai_model_frame,
            text="Modèle d'IA:"
        )
        ai_model_label.pack(side=tk.LEFT)

        ai_model_var = tk.StringVar(value=self.config.ai_model)
        self.tk_vars["ai_model"] = ai_model_var

        ai_model_combo = ttk.Combobox(
            ai_model_frame,
            textvariable=ai_model_var,
            values=["alphastar", "basic", "hybrid"],
            state="readonly"
        )
        ai_model_combo.pack(side=tk.RIGHT, padx=(20, 0))
        ai_model_combo.bind("<<ComboboxSelected>>",
                           lambda e: self.update_config("ai_model", ai_model_var.get()))

        # Niveau de reasoning
        reasoning_frame = self.theme.create_frame(section.content_frame, "primary")
        reasoning_frame.pack(fill=tk.X, pady=10)

        reasoning_label = self.theme.create_body_label(
            reasoning_frame,
            text="Niveau de reasoning:"
        )
        reasoning_label.pack(side=tk.LEFT)

        reasoning_level_var = tk.StringVar(value=self.config.reasoning_level)
        self.tk_vars["reasoning_level"] = reasoning_level_var

        reasoning_combo = ttk.Combobox(
            reasoning_frame,
            textvariable=reasoning_level_var,
            values=["basic", "normal", "full"],
            state="readonly"
        )
        reasoning_combo.pack(side=tk.RIGHT, padx=(20, 0))
        reasoning_combo.bind("<<ComboboxSelected>>",
                            lambda e: self.update_config("reasoning_level", reasoning_level_var.get()))

        # Apprentissage activé
        learning_enabled_var = tk.BooleanVar(value=self.config.learning_enabled)
        self.tk_vars["learning_enabled"] = learning_enabled_var

        learning_check = tk.Checkbutton(
            section.content_frame,
            text="Apprentissage activé",
            variable=learning_enabled_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("learning_enabled", learning_enabled_var.get())
        )
        learning_check.pack(anchor="w", pady=5)

        # Taux d'adaptation
        adaptation_frame = self.theme.create_frame(section.content_frame, "primary")
        adaptation_frame.pack(fill=tk.X, pady=10)

        adaptation_label = self.theme.create_body_label(
            adaptation_frame,
            text="Taux d'adaptation:"
        )
        adaptation_label.pack(side=tk.LEFT)

        adaptation_rate_var = tk.DoubleVar(value=self.config.adaptation_rate)
        self.tk_vars["adaptation_rate"] = adaptation_rate_var

        adaptation_scale = tk.Scale(
            adaptation_frame,
            from_=0.01,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=adaptation_rate_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("adaptation_rate", float(val))
        )
        adaptation_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

    def create_performance_config(self):
        """Section configuration performance"""
        section = ConfigSection(self.notebook, self.theme, "Configuration Performance", "")
        self.notebook.add(section.frame, text=" Performance")

        # Limite FPS
        fps_frame = self.theme.create_frame(section.content_frame, "primary")
        fps_frame.pack(fill=tk.X, pady=10)

        fps_label = self.theme.create_body_label(
            fps_frame,
            text="Limite FPS:"
        )
        fps_label.pack(side=tk.LEFT)

        fps_limit_var = tk.IntVar(value=self.config.fps_limit)
        self.tk_vars["fps_limit"] = fps_limit_var

        fps_scale = tk.Scale(
            fps_frame,
            from_=15,
            to=120,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=fps_limit_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("fps_limit", int(val))
        )
        fps_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Limite CPU
        cpu_frame = self.theme.create_frame(section.content_frame, "primary")
        cpu_frame.pack(fill=tk.X, pady=10)

        cpu_label = self.theme.create_body_label(
            cpu_frame,
            text="Limite CPU (%):"
        )
        cpu_label.pack(side=tk.LEFT)

        cpu_limit_var = tk.IntVar(value=self.config.cpu_limit)
        self.tk_vars["cpu_limit"] = cpu_limit_var

        cpu_scale = tk.Scale(
            cpu_frame,
            from_=10,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=cpu_limit_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("cpu_limit", int(val))
        )
        cpu_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Limite mémoire
        memory_frame = self.theme.create_frame(section.content_frame, "primary")
        memory_frame.pack(fill=tk.X, pady=10)

        memory_label = self.theme.create_body_label(
            memory_frame,
            text="Limite mémoire (MB):"
        )
        memory_label.pack(side=tk.LEFT)

        memory_limit_var = tk.IntVar(value=self.config.memory_limit)
        self.tk_vars["memory_limit"] = memory_limit_var

        memory_scale = tk.Scale(
            memory_frame,
            from_=512,
            to=8192,
            resolution=256,
            orient=tk.HORIZONTAL,
            variable=memory_limit_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("memory_limit", int(val))
        )
        memory_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Accélération GPU
        gpu_acceleration_var = tk.BooleanVar(value=self.config.gpu_acceleration)
        self.tk_vars["gpu_acceleration"] = gpu_acceleration_var

        gpu_check = tk.Checkbutton(
            section.content_frame,
            text="Accélération GPU (ROCm/DirectML)",
            variable=gpu_acceleration_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("gpu_acceleration", gpu_acceleration_var.get())
        )
        gpu_check.pack(anchor="w", pady=5)

    def create_security_config(self):
        """Section configuration sécurité"""
        section = ConfigSection(self.notebook, self.theme, "Configuration Sécurité", "")
        self.notebook.add(section.frame, text=" Sécurité")

        # Intervalle screenshots
        screenshot_frame = self.theme.create_frame(section.content_frame, "primary")
        screenshot_frame.pack(fill=tk.X, pady=10)

        screenshot_label = self.theme.create_body_label(
            screenshot_frame,
            text="Intervalle screenshots (sec):"
        )
        screenshot_label.pack(side=tk.LEFT)

        screenshot_interval_var = tk.IntVar(value=self.config.screenshot_interval)
        self.tk_vars["screenshot_interval"] = screenshot_interval_var

        screenshot_scale = tk.Scale(
            screenshot_frame,
            from_=5,
            to=300,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=screenshot_interval_var,
            **self.theme.get_style("body"),
            command=lambda val: self.update_config("screenshot_interval", int(val))
        )
        screenshot_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        # Niveau de log
        log_level_frame = self.theme.create_frame(section.content_frame, "primary")
        log_level_frame.pack(fill=tk.X, pady=10)

        log_level_label = self.theme.create_body_label(
            log_level_frame,
            text="Niveau de log:"
        )
        log_level_label.pack(side=tk.LEFT)

        log_level_var = tk.StringVar(value=self.config.log_level)
        self.tk_vars["log_level"] = log_level_var

        log_level_combo = ttk.Combobox(
            log_level_frame,
            textvariable=log_level_var,
            values=["debug", "info", "warning", "error"],
            state="readonly"
        )
        log_level_combo.pack(side=tk.RIGHT, padx=(20, 0))
        log_level_combo.bind("<<ComboboxSelected>>",
                            lambda e: self.update_config("log_level", log_level_var.get()))

        # Sauvegarder replays
        save_replays_var = tk.BooleanVar(value=self.config.save_replays)
        self.tk_vars["save_replays"] = save_replays_var

        save_replays_check = tk.Checkbutton(
            section.content_frame,
            text="Sauvegarder les replays",
            variable=save_replays_var,
            **self.theme.get_style("body"),
            command=lambda: self.update_config("save_replays", save_replays_var.get())
        )
        save_replays_check.pack(anchor="w", pady=5)

        # Boutons de gestion des fichiers
        files_frame = self.theme.create_frame(section.content_frame, "primary")
        files_frame.pack(fill=tk.X, pady=20)

        save_config_btn = self.theme.create_primary_button(
            files_frame,
            text="[SAVE] Sauvegarder Config",
            command=self.save_config
        )
        save_config_btn.pack(side=tk.LEFT, padx=(0, 10))

        load_config_btn = self.theme.create_secondary_button(
            files_frame,
            text=" Charger Config",
            command=self.load_config_file
        )
        load_config_btn.pack(side=tk.LEFT, padx=(0, 10))

        reset_config_btn = self.theme.create_secondary_button(
            files_frame,
            text="[RELOAD] Reset Config",
            command=self.reset_config
        )
        reset_config_btn.pack(side=tk.LEFT)

    def toggle_bot(self):
        """Démarre/arrête le bot"""
        try:
            if self.app_controller and hasattr(self.app_controller, 'is_running'):
                if self.app_controller.is_running:
                    self.app_controller.stop()
                    self.start_stop_btn.configure(text=">️ DÉMARRER")
                    self.pause_btn.configure(state="disabled")
                else:
                    self.app_controller.start()
                    self.start_stop_btn.configure(text="[]️ ARRÊTER")
                    self.pause_btn.configure(state="normal")
            else:
                messagebox.showwarning("Attention", "Contrôleur d'application non disponible")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du contrôle du bot: {e}")

    def toggle_pause(self):
        """Met en pause/reprend le bot"""
        try:
            if self.app_controller and hasattr(self.app_controller, 'is_paused'):
                if self.app_controller.is_paused:
                    self.app_controller.resume()
                    self.pause_btn.configure(text="||️ PAUSE")
                else:
                    self.app_controller.pause()
                    self.pause_btn.configure(text=">️ REPRENDRE")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la pause: {e}")

    def emergency_stop(self):
        """Arrêt d'urgence"""
        try:
            if messagebox.askyesno("Arrêt d'urgence", "Confirmer l'arrêt d'urgence du bot?"):
                if self.app_controller and hasattr(self.app_controller, 'emergency_stop'):
                    self.app_controller.emergency_stop()
                self.start_stop_btn.configure(text=">️ DÉMARRER")
                self.pause_btn.configure(state="disabled")
                messagebox.showinfo("Arrêt d'urgence", "Bot arrêté d'urgence")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'arrêt d'urgence: {e}")

    def add_quest(self):
        """Ajoute une quête à la liste"""
        quest_name = tk.simpledialog.askstring("Ajouter quête", "Nom de la quête:")
        if quest_name:
            self.quest_listbox.insert(tk.END, quest_name)
            self.config.selected_quests.append(quest_name)
            self.save_config()

    def remove_quest(self):
        """Supprime la quête sélectionnée"""
        selection = self.quest_listbox.curselection()
        if selection:
            index = selection[0]
            quest_name = self.quest_listbox.get(index)
            self.quest_listbox.delete(index)
            if quest_name in self.config.selected_quests:
                self.config.selected_quests.remove(quest_name)
            self.save_config()

    def load_quest_list(self):
        """Charge une liste de quêtes depuis un fichier"""
        filename = filedialog.askopenfilename(
            title="Charger liste de quêtes",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    quest_list = json.load(f)

                self.quest_listbox.delete(0, tk.END)
                self.config.selected_quests = quest_list

                for quest in quest_list:
                    self.quest_listbox.insert(tk.END, quest)

                self.save_config()
                messagebox.showinfo("Succès", f"Liste de quêtes chargée: {len(quest_list)} quêtes")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")

    def update_config(self, key: str, value: Any):
        """Met à jour une valeur de configuration"""
        setattr(self.config, key, value)
        self.save_config()

    def save_config(self):
        """Sauvegarde la configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erreur sauvegarde configuration: {e}")

    def load_config(self):
        """Charge la configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Mettre à jour la configuration
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                # Mettre à jour les variables tkinter
                self.update_ui_from_config()

        except Exception as e:
            print(f"Erreur chargement configuration: {e}")

    def load_config_file(self):
        """Charge une configuration depuis un fichier"""
        filename = filedialog.askopenfilename(
            title="Charger configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                self.update_ui_from_config()
                self.save_config()
                messagebox.showinfo("Succès", "Configuration chargée avec succès")

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")

    def reset_config(self):
        """Remet la configuration par défaut"""
        if messagebox.askyesno("Reset", "Remettre la configuration par défaut?"):
            self.config = BotConfig()
            self.update_ui_from_config()
            self.save_config()
            messagebox.showinfo("Reset", "Configuration remise par défaut")

    def update_ui_from_config(self):
        """Met à jour l'interface depuis la configuration"""
        for key, var in self.tk_vars.items():
            if hasattr(self.config, key):
                value = getattr(self.config, key)
                var.set(value)

        # Mettre à jour la liste des quêtes
        self.quest_listbox.delete(0, tk.END)
        for quest in self.config.selected_quests:
            self.quest_listbox.insert(tk.END, quest)

def create_control_panel(parent, theme_manager: ThemeManager, app_controller=None) -> ControlPanel:
    """Factory function pour créer ControlPanel"""
    return ControlPanel(parent, theme_manager, app_controller)