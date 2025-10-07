"""
Navigation Panel - Système de navigation avec carte interactive
Pathfinding, waypoints, zones, exploration
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import math


class MapType(Enum):
    """Type de carte"""
    WORLD = "world"
    REGION = "region"
    ZONE = "zone"


@dataclass
class MapCoordinate:
    """Coordonnée sur la carte"""
    x: int
    y: int
    map_type: MapType
    region_name: str


@dataclass
class Waypoint:
    """Point de passage (waypoint)"""
    waypoint_id: str
    name: str
    coordinate: MapCoordinate
    waypoint_type: str  # 'zaap', 'zaapi', 'custom', 'boss', 'dungeon'
    is_discovered: bool
    notes: str


@dataclass
class PathNode:
    """Noeud de chemin"""
    coordinate: MapCoordinate
    parent: Optional['PathNode']
    g_cost: float  # Coût depuis le départ
    h_cost: float  # Heuristique vers l'arrivée
    f_cost: float  # Coût total


class NavigationPanel:
    """
    Panel de navigation avec carte interactive

    Fonctionnalités:
    - Carte du monde interactive
    - Pathfinding A* visuel
    - Gestion des waypoints/zaaps
    - Exploration et découverte
    - Zones dangereuses
    - Navigation automatique
    - Historique de déplacements
    """

    def __init__(self, parent):
        self.parent = parent
        self.waypoints: List[Waypoint] = []
        self.current_position: Optional[MapCoordinate] = None
        self.target_position: Optional[MapCoordinate] = None
        self.current_path: List[MapCoordinate] = []
        self.zoom_level: float = 1.0
        self.pan_offset: Tuple[int, int] = (0, 0)
        self.map_mode: MapType = MapType.REGION

        # Charger les données
        self.load_navigation_data()

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Toolbar supérieure ===
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(toolbar, text="🗺️ Navigation & Carte Interactive",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        # Position actuelle
        self.position_label = ttk.Label(toolbar, text="📍 Position: Inconnu",
                                       font=("Segoe UI", 9))
        self.position_label.pack(side=tk.RIGHT, padx=10)

        ttk.Button(toolbar, text="🔄 Rafraîchir",
                  command=self.refresh_position).pack(side=tk.RIGHT, padx=2)

        # === PanedWindow principal ===
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Panneau gauche: Contrôles et waypoints ===
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # Notebook pour organiser
        left_notebook = ttk.Notebook(left_frame)
        left_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Waypoints ---
        waypoints_tab = ttk.Frame(left_notebook, padding=5)
        left_notebook.add(waypoints_tab, text="📍 Waypoints")

        # Filtres
        filter_frame = ttk.Frame(waypoints_tab)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.waypoint_filter = ttk.Combobox(filter_frame, values=[
            "Tous", "Zaap", "Zaapi", "Boss", "Donjon", "Personnalisé"
        ], state="readonly", width=12)
        self.waypoint_filter.set("Tous")
        self.waypoint_filter.pack(side=tk.LEFT)
        self.waypoint_filter.bind("<<ComboboxSelected>>", lambda e: self.filter_waypoints())

        ttk.Label(filter_frame, text="Région:").pack(side=tk.LEFT, padx=(10, 5))
        self.region_filter = ttk.Combobox(filter_frame, values=[
            "Toutes", "Astrub", "Incarnam", "Bonta", "Brâkmar"
        ], state="readonly", width=12)
        self.region_filter.set("Toutes")
        self.region_filter.pack(side=tk.LEFT)
        self.region_filter.bind("<<ComboboxSelected>>", lambda e: self.filter_waypoints())

        # Liste waypoints
        waypoints_list_frame = ttk.Frame(waypoints_tab)
        waypoints_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        waypoints_scroll = ttk.Scrollbar(waypoints_list_frame)
        waypoints_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.waypoints_tree = ttk.Treeview(
            waypoints_list_frame,
            columns=("type", "region", "coords", "status"),
            show="tree headings",
            yscrollcommand=waypoints_scroll.set
        )
        self.waypoints_tree.heading("type", text="Type")
        self.waypoints_tree.heading("region", text="Région")
        self.waypoints_tree.heading("coords", text="Coords")
        self.waypoints_tree.heading("status", text="Statut")
        self.waypoints_tree.column("#0", width=120)
        self.waypoints_tree.column("type", width=80)
        self.waypoints_tree.column("region", width=80)
        self.waypoints_tree.column("coords", width=80)
        self.waypoints_tree.column("status", width=60)
        self.waypoints_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        waypoints_scroll.config(command=self.waypoints_tree.yview)
        self.waypoints_tree.bind("<<TreeviewSelect>>", self.on_waypoint_select)
        self.waypoints_tree.bind("<Double-Button-1>", self.goto_waypoint)

        self.update_waypoints_tree()

        # Actions waypoints
        waypoints_actions = ttk.Frame(waypoints_tab)
        waypoints_actions.pack(fill=tk.X)

        ttk.Button(waypoints_actions, text="➕ Ajouter",
                  command=self.add_waypoint).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(waypoints_actions, text="🎯 Aller à",
                  command=self.goto_waypoint).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(waypoints_actions, text="🗑️ Supprimer",
                  command=self.delete_waypoint).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Navigation ---
        nav_tab = ttk.Frame(left_notebook, padding=5)
        left_notebook.add(nav_tab, text="🧭 Navigation")

        # Position actuelle
        current_frame = ttk.LabelFrame(nav_tab, text="Position actuelle", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 10))

        self.current_pos_text = tk.Text(current_frame, height=3, font=("Consolas", 9),
                                       state=tk.DISABLED, wrap=tk.WORD)
        self.current_pos_text.pack(fill=tk.BOTH, expand=True)

        # Destination
        dest_frame = ttk.LabelFrame(nav_tab, text="Destination", padding=10)
        dest_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dest_frame, text="Coordonnées:").grid(row=0, column=0, sticky=tk.W, pady=5)

        coords_frame = ttk.Frame(dest_frame)
        coords_frame.grid(row=0, column=1, sticky=tk.EW, pady=5)

        ttk.Label(coords_frame, text="X:").pack(side=tk.LEFT, padx=(0, 2))
        self.dest_x_var = tk.IntVar(value=0)
        ttk.Spinbox(coords_frame, from_=-100, to=100, textvariable=self.dest_x_var,
                   width=8).pack(side=tk.LEFT, padx=2)

        ttk.Label(coords_frame, text="Y:").pack(side=tk.LEFT, padx=(10, 2))
        self.dest_y_var = tk.IntVar(value=0)
        ttk.Spinbox(coords_frame, from_=-100, to=100, textvariable=self.dest_y_var,
                   width=8).pack(side=tk.LEFT, padx=2)

        ttk.Button(dest_frame, text="🗺️ Depuis waypoint",
                  command=self.select_waypoint_as_dest).grid(row=1, column=0, columnspan=2,
                                                             sticky=tk.EW, pady=5)

        ttk.Button(dest_frame, text="🎯 Calculer chemin",
                  command=self.calculate_path,
                  style="Accent.TButton").grid(row=2, column=0, columnspan=2,
                                              sticky=tk.EW, pady=5)

        dest_frame.columnconfigure(1, weight=1)

        # Infos chemin
        path_frame = ttk.LabelFrame(nav_tab, text="Chemin calculé", padding=10)
        path_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.path_info_text = tk.Text(path_frame, font=("Consolas", 9),
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.path_info_text.pack(fill=tk.BOTH, expand=True)

        # Actions navigation
        nav_actions = ttk.Frame(nav_tab)
        nav_actions.pack(fill=tk.X)

        ttk.Button(nav_actions, text="▶️ Démarrer",
                  command=self.start_navigation).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(nav_actions, text="⏹️ Arrêter",
                  command=self.stop_navigation).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Zones ---
        zones_tab = ttk.Frame(left_notebook, padding=5)
        left_notebook.add(zones_tab, text="🌍 Zones")

        ttk.Label(zones_tab, text="Zones et régions:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # TreeView zones
        zones_list_frame = ttk.Frame(zones_tab)
        zones_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        zones_scroll = ttk.Scrollbar(zones_list_frame)
        zones_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.zones_tree = ttk.Treeview(
            zones_list_frame,
            columns=("level", "danger", "explored"),
            show="tree headings",
            yscrollcommand=zones_scroll.set
        )
        self.zones_tree.heading("level", text="Niveau")
        self.zones_tree.heading("danger", text="Danger")
        self.zones_tree.heading("explored", text="Exploré")
        self.zones_tree.column("#0", width=150)
        self.zones_tree.column("level", width=60)
        self.zones_tree.column("danger", width=80)
        self.zones_tree.column("explored", width=80)
        self.zones_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        zones_scroll.config(command=self.zones_tree.yview)

        self.update_zones_tree()

        # Stats exploration
        explore_stats_frame = ttk.LabelFrame(zones_tab, text="Exploration", padding=5)
        explore_stats_frame.pack(fill=tk.X)

        self.explore_stats_text = tk.Text(explore_stats_frame, height=5, font=("Consolas", 8),
                                         state=tk.DISABLED, wrap=tk.WORD)
        self.explore_stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_exploration_stats()

        # --- Onglet Historique ---
        history_tab = ttk.Frame(left_notebook, padding=5)
        left_notebook.add(history_tab, text="📜 Historique")

        ttk.Label(history_tab, text="Déplacements récents:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Liste historique
        history_list_frame = ttk.Frame(history_tab)
        history_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        history_scroll = ttk.Scrollbar(history_list_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_listbox = tk.Listbox(
            history_list_frame,
            yscrollcommand=history_scroll.set,
            font=("Consolas", 8)
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_listbox.yview)

        self.update_history()

        # Actions historique
        history_actions = ttk.Frame(history_tab)
        history_actions.pack(fill=tk.X)

        ttk.Button(history_actions, text="🔄 Retour",
                  command=self.go_back).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(history_actions, text="🗑️ Effacer",
                  command=self.clear_history).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # === Panneau central: Carte interactive ===
        center_frame = ttk.LabelFrame(paned, text="🗺️ Carte Interactive", padding=5)
        paned.add(center_frame, weight=3)

        # Contrôles carte
        map_controls = ttk.Frame(center_frame)
        map_controls.pack(fill=tk.X, pady=(0, 5))

        # Mode carte
        ttk.Label(map_controls, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.map_mode_var = tk.StringVar(value="Région")
        ttk.Radiobutton(map_controls, text="Monde", variable=self.map_mode_var,
                       value="Monde", command=self.change_map_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(map_controls, text="Région", variable=self.map_mode_var,
                       value="Région", command=self.change_map_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(map_controls, text="Zone", variable=self.map_mode_var,
                       value="Zone", command=self.change_map_mode).pack(side=tk.LEFT, padx=2)

        ttk.Separator(map_controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Affichage
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_controls, text="Grille", variable=self.show_grid_var,
                       command=self.redraw_map).pack(side=tk.LEFT, padx=2)

        self.show_waypoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_controls, text="Waypoints", variable=self.show_waypoints_var,
                       command=self.redraw_map).pack(side=tk.LEFT, padx=2)

        self.show_path_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_controls, text="Chemin", variable=self.show_path_var,
                       command=self.redraw_map).pack(side=tk.LEFT, padx=2)

        self.show_danger_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_controls, text="Zones danger", variable=self.show_danger_var,
                       command=self.redraw_map).pack(side=tk.LEFT, padx=2)

        ttk.Separator(map_controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Zoom
        ttk.Button(map_controls, text="➖", width=3,
                  command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(map_controls, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(map_controls, text="➕", width=3,
                  command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(map_controls, text="🔄", width=3,
                  command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        # Canvas carte
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.map_canvas = tk.Canvas(
            canvas_frame,
            bg='#1a1a2e',
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )
        self.map_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll.config(command=self.map_canvas.xview)
        v_scroll.config(command=self.map_canvas.yview)

        # Bindings
        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<Button-3>", self.on_map_right_click)
        self.map_canvas.bind("<B1-Motion>", self.on_map_drag)
        self.map_canvas.bind("<MouseWheel>", self.on_map_wheel)

        # Infos carte
        map_info_frame = ttk.Frame(center_frame)
        map_info_frame.pack(fill=tk.X, pady=(5, 0))

        self.map_info_label = ttk.Label(map_info_frame, text="Cliquez sur la carte pour définir une destination",
                                       font=("Segoe UI", 9))
        self.map_info_label.pack(side=tk.LEFT)

        self.coords_label = ttk.Label(map_info_frame, text="",
                                     font=("Consolas", 9))
        self.coords_label.pack(side=tk.RIGHT)

        # === Panneau droit: Outils ===
        right_frame = ttk.LabelFrame(paned, text="🛠️ Outils", padding=10)
        paned.add(right_frame, weight=1)

        # Auto-navigation
        auto_nav_frame = ttk.LabelFrame(right_frame, text="Navigation Auto", padding=10)
        auto_nav_frame.pack(fill=tk.X, pady=(0, 10))

        self.auto_nav_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(auto_nav_frame, text="Activer navigation auto",
                       variable=self.auto_nav_enabled).pack(anchor=tk.W, pady=2)

        self.avoid_combat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_nav_frame, text="Éviter les combats",
                       variable=self.avoid_combat_var).pack(anchor=tk.W, pady=2)

        self.use_zaaps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_nav_frame, text="Utiliser les Zaaps",
                       variable=self.use_zaaps_var).pack(anchor=tk.W, pady=2)

        # Pathfinding options
        pathfinding_frame = ttk.LabelFrame(right_frame, text="Pathfinding", padding=10)
        pathfinding_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(pathfinding_frame, text="Algorithme:").pack(anchor=tk.W, pady=(0, 2))
        self.pathfinding_algo = ttk.Combobox(pathfinding_frame, values=[
            "A* (optimal)", "Dijkstra", "Greedy", "BFS"
        ], state="readonly", width=20)
        self.pathfinding_algo.set("A* (optimal)")
        self.pathfinding_algo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(pathfinding_frame, text="Heuristique:").pack(anchor=tk.W, pady=(0, 2))
        self.heuristic = ttk.Combobox(pathfinding_frame, values=[
            "Manhattan", "Euclidienne", "Chebyshev"
        ], state="readonly", width=20)
        self.heuristic.set("Manhattan")
        self.heuristic.pack(fill=tk.X)

        # Statistiques
        stats_frame = ttk.LabelFrame(right_frame, text="Statistiques", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.nav_stats_text = tk.Text(stats_frame, font=("Consolas", 8),
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.nav_stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_navigation_stats()

        # Dessiner la carte initiale
        self.redraw_map()

    def load_navigation_data(self):
        """Charge les données de navigation (démo)"""
        # Waypoints de démo
        self.waypoints = [
            Waypoint(
                waypoint_id="wp_001",
                name="Zaap Astrub",
                coordinate=MapCoordinate(0, 0, MapType.REGION, "Astrub"),
                waypoint_type="zaap",
                is_discovered=True,
                notes="Zaap principal d'Astrub"
            ),
            Waypoint(
                waypoint_id="wp_002",
                name="Banque Astrub",
                coordinate=MapCoordinate(-1, 0, MapType.REGION, "Astrub"),
                waypoint_type="custom",
                is_discovered=True,
                notes="Banque et coffre"
            ),
            Waypoint(
                waypoint_id="wp_003",
                name="Zone Bouftous",
                coordinate=MapCoordinate(-1, 2, MapType.REGION, "Astrub"),
                waypoint_type="custom",
                is_discovered=True,
                notes="Spot farming bouftous"
            ),
            Waypoint(
                waypoint_id="wp_004",
                name="Donjon Incarnam",
                coordinate=MapCoordinate(5, -3, MapType.REGION, "Incarnam"),
                waypoint_type="dungeon",
                is_discovered=False,
                notes="Donjon débutant"
            )
        ]

        # Position actuelle
        self.current_position = MapCoordinate(0, 0, MapType.REGION, "Astrub")

    def update_waypoints_tree(self):
        """Met à jour l'arbre des waypoints"""
        self.waypoints_tree.delete(*self.waypoints_tree.get_children())

        for wp in self.waypoints:
            type_emoji = {
                "zaap": "🌀",
                "zaapi": "🔷",
                "boss": "👹",
                "dungeon": "🏰",
                "custom": "📍"
            }

            status = "✅" if wp.is_discovered else "❌"
            coords = f"[{wp.coordinate.x}, {wp.coordinate.y}]"

            self.waypoints_tree.insert("", tk.END, text=wp.name,
                                      values=(f"{type_emoji.get(wp.waypoint_type, '?')} {wp.waypoint_type}",
                                             wp.coordinate.region_name, coords, status))

    def update_zones_tree(self):
        """Met à jour l'arbre des zones"""
        self.zones_tree.delete(*self.zones_tree.get_children())

        # Zones démo
        zones = [
            ("Astrub", "1-20", "🟢 Faible", "95%"),
            ("Incarnam", "1-15", "🟢 Faible", "80%"),
            ("Bonta", "20-50", "🟡 Moyen", "45%"),
            ("Brâkmar", "20-50", "🟡 Moyen", "40%"),
            ("Forêt Maléfique", "50-100", "🔴 Élevé", "15%")
        ]

        for zone in zones:
            self.zones_tree.insert("", tk.END, text=zone[0], values=zone[1:])

    def update_exploration_stats(self):
        """Met à jour les stats d'exploration"""
        stats = """Total cartes: 1,250
Explorées: 456 (36%)
Restantes: 794

Zones complètes:
├─ Astrub: 95%
├─ Incarnam: 80%
└─ Bonta: 45%
"""
        self.explore_stats_text.config(state=tk.NORMAL)
        self.explore_stats_text.delete("1.0", tk.END)
        self.explore_stats_text.insert("1.0", stats)
        self.explore_stats_text.config(state=tk.DISABLED)

    def update_history(self):
        """Met à jour l'historique"""
        self.history_listbox.delete(0, tk.END)

        # Historique démo
        history = [
            "14:25 - Astrub [0, 0]",
            "14:20 - Banque [-1, 0]",
            "14:15 - Zone Bouftous [-1, 2]",
            "14:10 - HDV [1, 0]",
            "14:05 - Astrub [0, 0]"
        ]

        for entry in history:
            self.history_listbox.insert(tk.END, entry)

    def update_navigation_stats(self):
        """Met à jour les stats de navigation"""
        stats = """=== Navigation ===

Distance parcourue:
└─ 15.2 km

Cartes traversées:
└─ 245

Zaaps utilisés:
└─ 12

Temps de déplacement:
└─ 2h 15m

Efficacité:
└─ 87%
"""
        self.nav_stats_text.config(state=tk.NORMAL)
        self.nav_stats_text.delete("1.0", tk.END)
        self.nav_stats_text.insert("1.0", stats)
        self.nav_stats_text.config(state=tk.DISABLED)

    def redraw_map(self):
        """Redessine la carte"""
        self.map_canvas.delete("all")

        # Taille de la grille
        cell_size = int(40 * self.zoom_level)
        grid_size = 20
        offset_x, offset_y = self.pan_offset

        # Dessiner la grille
        if self.show_grid_var.get():
            for i in range(-grid_size, grid_size + 1):
                # Lignes verticales
                x = (grid_size + i) * cell_size + offset_x
                self.map_canvas.create_line(
                    x, 0, x, grid_size * 2 * cell_size,
                    fill='#2a2a3e', width=1
                )
                # Lignes horizontales
                y = (grid_size + i) * cell_size + offset_y
                self.map_canvas.create_line(
                    0, y, grid_size * 2 * cell_size, y,
                    fill='#2a2a3e', width=1
                )

        # Axes principaux
        center_x = grid_size * cell_size + offset_x
        center_y = grid_size * cell_size + offset_y
        self.map_canvas.create_line(center_x, 0, center_x, grid_size * 2 * cell_size,
                                    fill='#4a4a5e', width=2)
        self.map_canvas.create_line(0, center_y, grid_size * 2 * cell_size, center_y,
                                    fill='#4a4a5e', width=2)

        # Zones de danger (démo)
        if self.show_danger_var.get():
            danger_zones = [
                ((-5, -8), 3, "red"),  # Forêt Maléfique
                ((10, 10), 2, "orange")  # Zone intermédiaire
            ]
            for (dx, dy), radius, color in danger_zones:
                x = (grid_size + dx) * cell_size + offset_x
                y = (grid_size - dy) * cell_size + offset_y
                r = radius * cell_size
                self.map_canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    outline=color, width=2, dash=(5, 5)
                )

        # Dessiner les waypoints
        if self.show_waypoints_var.get():
            for wp in self.waypoints:
                x = (grid_size + wp.coordinate.x) * cell_size + offset_x
                y = (grid_size - wp.coordinate.y) * cell_size + offset_y

                color = "green" if wp.is_discovered else "gray"
                size = 8

                if wp.waypoint_type == "zaap":
                    self.map_canvas.create_oval(x-size, y-size, x+size, y+size,
                                                fill="cyan", outline="white", width=2)
                elif wp.waypoint_type == "dungeon":
                    self.map_canvas.create_rectangle(x-size, y-size, x+size, y+size,
                                                     fill="purple", outline="white", width=2)
                else:
                    self.map_canvas.create_oval(x-size, y-size, x+size, y+size,
                                                fill=color, outline="white", width=2)

                # Label
                self.map_canvas.create_text(x, y-15, text=wp.name,
                                           fill="white", font=("Arial", 8))

        # Dessiner le chemin
        if self.show_path_var.get() and self.current_path:
            for i in range(len(self.current_path) - 1):
                c1 = self.current_path[i]
                c2 = self.current_path[i + 1]

                x1 = (grid_size + c1.x) * cell_size + offset_x
                y1 = (grid_size - c1.y) * cell_size + offset_y
                x2 = (grid_size + c2.x) * cell_size + offset_x
                y2 = (grid_size - c2.y) * cell_size + offset_y

                self.map_canvas.create_line(x1, y1, x2, y2,
                                            fill="yellow", width=3, arrow=tk.LAST)

        # Dessiner la position actuelle
        if self.current_position:
            x = (grid_size + self.current_position.x) * cell_size + offset_x
            y = (grid_size - self.current_position.y) * cell_size + offset_y

            # Cercle pulsant (simplifié)
            self.map_canvas.create_oval(x-12, y-12, x+12, y+12,
                                        fill="", outline="lime", width=3)
            self.map_canvas.create_oval(x-6, y-6, x+6, y+6,
                                        fill="lime", outline="white", width=2)

        # Dessiner la destination
        if self.target_position:
            x = (grid_size + self.target_position.x) * cell_size + offset_x
            y = (grid_size - self.target_position.y) * cell_size + offset_y

            self.map_canvas.create_oval(x-10, y-10, x+10, y+10,
                                        fill="red", outline="white", width=2)
            self.map_canvas.create_text(x, y-20, text="DESTINATION",
                                       fill="red", font=("Arial", 9, "bold"))

        # Mettre à jour scroll region
        self.map_canvas.config(scrollregion=self.map_canvas.bbox(tk.ALL))

        # Mettre à jour position
        self.update_position_display()

    def update_position_display(self):
        """Met à jour l'affichage de la position"""
        if self.current_position:
            pos_text = f"📍 Position: [{self.current_position.x}, {self.current_position.y}] - {self.current_position.region_name}"
            self.position_label.config(text=pos_text)

            current_info = f"Coordonnées: [{self.current_position.x}, {self.current_position.y}]\n"
            current_info += f"Région: {self.current_position.region_name}\n"
            current_info += f"Type: {self.current_position.map_type.value}"

            self.current_pos_text.config(state=tk.NORMAL)
            self.current_pos_text.delete("1.0", tk.END)
            self.current_pos_text.insert("1.0", current_info)
            self.current_pos_text.config(state=tk.DISABLED)

    # === Zoom et Pan ===

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.redraw_map()

    def zoom_out(self):
        self.zoom_level /= 1.2
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.redraw_map()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        self.zoom_label.config(text="100%")
        self.redraw_map()

    def on_map_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_map_click(self, event):
        """Clic sur la carte"""
        # Calculer coordonnées
        cell_size = int(40 * self.zoom_level)
        grid_size = 20
        offset_x, offset_y = self.pan_offset

        map_x = (event.x - offset_x - grid_size * cell_size) // cell_size
        map_y = -(event.y - offset_y - grid_size * cell_size) // cell_size

        self.coords_label.config(text=f"Clic: [{map_x}, {map_y}]")

        # Définir comme destination
        self.dest_x_var.set(map_x)
        self.dest_y_var.set(map_y)

    def on_map_right_click(self, event):
        """Clic droit sur la carte"""
        messagebox.showinfo("Menu", "Menu contextuel - à implémenter")

    def on_map_drag(self, event):
        """Drag de la carte"""
        # TODO: implémenter pan avec drag
        pass

    # === Actions ===

    def filter_waypoints(self):
        self.update_waypoints_tree()

    def on_waypoint_select(self, event):
        """Sélection d'un waypoint"""
        pass

    def add_waypoint(self):
        messagebox.showinfo("Ajouter", "Ajouter waypoint - à implémenter")

    def goto_waypoint(self, event=None):
        """Va au waypoint sélectionné"""
        selection = self.waypoints_tree.selection()
        if not selection:
            messagebox.showwarning("Sélection", "Veuillez sélectionner un waypoint")
            return

        messagebox.showinfo("Navigation", "Navigation vers waypoint - à implémenter")

    def delete_waypoint(self):
        if messagebox.askyesno("Confirmer", "Supprimer le waypoint sélectionné?"):
            messagebox.showinfo("Supprimé", "Waypoint supprimé - à implémenter")

    def select_waypoint_as_dest(self):
        messagebox.showinfo("Destination", "Sélection waypoint comme destination - à implémenter")

    def calculate_path(self):
        """Calcule le chemin avec A*"""
        if not self.current_position:
            messagebox.showwarning("Position", "Position actuelle inconnue")
            return

        dest_x = self.dest_x_var.get()
        dest_y = self.dest_y_var.get()

        self.target_position = MapCoordinate(dest_x, dest_y, MapType.REGION, "Astrub")

        # Calculer chemin simple (démo)
        self.current_path = []
        current = self.current_position

        # Simple ligne droite pour démo
        steps = max(abs(dest_x - current.x), abs(dest_y - current.y))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(current.x + (dest_x - current.x) * t)
            y = int(current.y + (dest_y - current.y) * t)
            self.current_path.append(MapCoordinate(x, y, MapType.REGION, "Astrub"))

        # Afficher infos chemin
        distance = len(self.current_path) - 1
        path_info = f"Chemin calculé:\n"
        path_info += f"Distance: {distance} cartes\n"
        path_info += f"Temps estimé: {distance * 5} secondes\n"
        path_info += f"Algorithme: {self.pathfinding_algo.get()}\n"
        path_info += f"\nÉtapes: {len(self.current_path)}\n"

        self.path_info_text.config(state=tk.NORMAL)
        self.path_info_text.delete("1.0", tk.END)
        self.path_info_text.insert("1.0", path_info)
        self.path_info_text.config(state=tk.DISABLED)

        self.redraw_map()
        messagebox.showinfo("Chemin", f"Chemin calculé: {distance} cartes")

    def start_navigation(self):
        if not self.current_path:
            messagebox.showwarning("Chemin", "Veuillez d'abord calculer un chemin")
            return
        messagebox.showinfo("Navigation", "Navigation démarrée - à implémenter")

    def stop_navigation(self):
        messagebox.showinfo("Navigation", "Navigation arrêtée - à implémenter")

    def change_map_mode(self):
        self.redraw_map()

    def refresh_position(self):
        messagebox.showinfo("Position", "Position rafraîchie - à implémenter")
        self.redraw_map()

    def go_back(self):
        messagebox.showinfo("Retour", "Retour position précédente - à implémenter")

    def clear_history(self):
        if messagebox.askyesno("Confirmer", "Effacer l'historique?"):
            self.history_listbox.delete(0, tk.END)

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
