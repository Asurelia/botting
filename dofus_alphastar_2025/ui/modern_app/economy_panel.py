"""
Economy Panel - Gestion économique avancée
HDV, craft, farming rentabilité, gestion inventaire
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json


class ItemCategory(Enum):
    """Catégorie d'objet"""
    RESOURCE = "resource"
    EQUIPMENT = "equipment"
    CONSUMABLE = "consumable"
    QUEST = "quest"
    RUNE = "rune"


@dataclass
class MarketItem:
    """Objet sur le marché"""
    item_id: str
    name: str
    category: ItemCategory
    level: int
    price_avg: int
    price_min: int
    price_max: int
    quantity_available: int
    last_update: datetime
    trend: str  # 'up', 'down', 'stable'
    price_history: List[Tuple[datetime, int]]


@dataclass
class CraftRecipe:
    """Recette de craft"""
    recipe_id: str
    result_item: str
    result_quantity: int
    ingredients: List[Dict[str, Any]]
    craft_cost: int
    success_rate: float
    profession: str
    level_required: int


@dataclass
class FarmingSpot:
    """Spot de farming"""
    spot_id: str
    name: str
    map_location: str
    monsters: List[str]
    loot_table: List[Dict[str, Any]]
    estimated_kamas_per_hour: int
    estimated_xp_per_hour: int
    difficulty: str  # 'easy', 'medium', 'hard'
    recommended_level: int


class EconomyPanel:
    """
    Panel de gestion économique avancée

    Fonctionnalités:
    - Monitoring HDV (prix, tendances)
    - Calculateur de rentabilité farming
    - Gestion inventaire intelligente
    - Optimisation craft
    - Opportunités d'achat/vente
    - Prédiction de prix (ML)
    - Suivi du patrimoine
    """

    def __init__(self, parent):
        self.parent = parent
        self.market_items: List[MarketItem] = []
        self.craft_recipes: List[CraftRecipe] = []
        self.farming_spots: List[FarmingSpot] = []
        self.inventory: Dict[str, int] = {}
        self.total_wealth: int = 0

        # Charger les données
        self.load_economy_data()

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Toolbar supérieure ===
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(toolbar, text="💰 Centre Économique & Gestion",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        # Stats rapides
        self.wealth_label = ttk.Label(toolbar, text="💰 Patrimoine: 0K",
                                     font=("Segoe UI", 10, "bold"))
        self.wealth_label.pack(side=tk.RIGHT, padx=10)

        ttk.Button(toolbar, text="🔄 Actualiser HDV",
                  command=self.refresh_market).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="📊 Rapport",
                  command=self.generate_report).pack(side=tk.RIGHT, padx=2)

        # === PanedWindow principal ===
        main_paned = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # === Section supérieure: Notebook principal ===
        top_frame = ttk.Frame(main_paned)
        main_paned.add(top_frame, weight=2)

        self.main_notebook = ttk.Notebook(top_frame)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet HDV & Marché ---
        market_tab = ttk.Frame(self.main_notebook, padding=5)
        self.main_notebook.add(market_tab, text="🏪 HDV & Marché")

        # Split horizontal
        market_paned = ttk.PanedWindow(market_tab, orient=tk.HORIZONTAL)
        market_paned.pack(fill=tk.BOTH, expand=True)

        # Liste des objets
        left_market = ttk.Frame(market_paned)
        market_paned.add(left_market, weight=1)

        # Filtres
        filter_frame = ttk.LabelFrame(left_market, text="Filtres", padding=5)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Catégorie:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.market_category_filter = ttk.Combobox(filter_frame, values=[
            "Toutes", "Ressources", "Équipements", "Consommables", "Quête", "Runes"
        ], state="readonly", width=15)
        self.market_category_filter.set("Toutes")
        self.market_category_filter.grid(row=0, column=1, padx=2)
        self.market_category_filter.bind("<<ComboboxSelected>>", lambda e: self.filter_market())

        ttk.Label(filter_frame, text="Tendance:").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.market_trend_filter = ttk.Combobox(filter_frame, values=[
            "Toutes", "↗️ Hausse", "↘️ Baisse", "➡️ Stable"
        ], state="readonly", width=12)
        self.market_trend_filter.set("Toutes")
        self.market_trend_filter.grid(row=0, column=3, padx=2)
        self.market_trend_filter.bind("<<ComboboxSelected>>", lambda e: self.filter_market())

        ttk.Label(filter_frame, text="Recherche:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=(5, 0))
        self.market_search_var = tk.StringVar()
        self.market_search_var.trace_add("write", lambda *args: self.filter_market())
        ttk.Entry(filter_frame, textvariable=self.market_search_var,
                 width=40).grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=2, pady=(5, 0))

        # TreeView marché
        market_list_frame = ttk.Frame(left_market)
        market_list_frame.pack(fill=tk.BOTH, expand=True)

        market_scroll = ttk.Scrollbar(market_list_frame)
        market_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.market_tree = ttk.Treeview(
            market_list_frame,
            columns=("category", "level", "price", "qty", "trend"),
            show="tree headings",
            yscrollcommand=market_scroll.set
        )
        self.market_tree.heading("category", text="Catégorie")
        self.market_tree.heading("level", text="Niv.")
        self.market_tree.heading("price", text="Prix")
        self.market_tree.heading("qty", text="Qté")
        self.market_tree.heading("trend", text="Tendance")
        self.market_tree.column("#0", width=150)
        self.market_tree.column("category", width=100)
        self.market_tree.column("level", width=50)
        self.market_tree.column("price", width=80)
        self.market_tree.column("qty", width=60)
        self.market_tree.column("trend", width=80)
        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        market_scroll.config(command=self.market_tree.yview)
        self.market_tree.bind("<<TreeviewSelect>>", self.on_market_item_select)

        self.update_market_tree()

        # Détails objet sélectionné
        right_market = ttk.LabelFrame(market_paned, text="Détails & Historique", padding=10)
        market_paned.add(right_market, weight=1)

        # Infos objet
        item_info_frame = ttk.LabelFrame(right_market, text="Informations", padding=5)
        item_info_frame.pack(fill=tk.X, pady=(0, 10))

        self.item_info_text = tk.Text(item_info_frame, height=6, font=("Consolas", 9),
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.item_info_text.pack(fill=tk.BOTH, expand=True)

        # Graphique prix
        graph_frame = ttk.LabelFrame(right_market, text="Historique des prix", padding=5)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.price_graph_canvas = tk.Canvas(graph_frame, bg='#2b2b2b', height=200)
        self.price_graph_canvas.pack(fill=tk.BOTH, expand=True)

        # Actions
        actions_frame = ttk.Frame(right_market)
        actions_frame.pack(fill=tk.X)

        ttk.Button(actions_frame, text="📊 Analyser tendance",
                  command=self.analyze_price_trend).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(actions_frame, text="🔔 Créer alerte",
                  command=self.create_price_alert).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Craft & Optimisation ---
        craft_tab = ttk.Frame(self.main_notebook, padding=5)
        self.main_notebook.add(craft_tab, text="🔨 Craft & Optimisation")

        craft_paned = ttk.PanedWindow(craft_tab, orient=tk.HORIZONTAL)
        craft_paned.pack(fill=tk.BOTH, expand=True)

        # Liste des recettes
        left_craft = ttk.Frame(craft_paned)
        craft_paned.add(left_craft, weight=1)

        ttk.Label(left_craft, text="Recettes de craft:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Filtres craft
        craft_filter_frame = ttk.Frame(left_craft)
        craft_filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(craft_filter_frame, text="Métier:").pack(side=tk.LEFT, padx=(0, 5))
        self.craft_profession_filter = ttk.Combobox(craft_filter_frame, values=[
            "Tous", "Forgeron", "Tailleur", "Bijoutier", "Cordonnier", "Alchimiste"
        ], state="readonly", width=15)
        self.craft_profession_filter.set("Tous")
        self.craft_profession_filter.pack(side=tk.LEFT)
        self.craft_profession_filter.bind("<<ComboboxSelected>>", lambda e: self.filter_crafts())

        # TreeView craft
        craft_list_frame = ttk.Frame(left_craft)
        craft_list_frame.pack(fill=tk.BOTH, expand=True)

        craft_scroll = ttk.Scrollbar(craft_list_frame)
        craft_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.craft_tree = ttk.Treeview(
            craft_list_frame,
            columns=("profession", "level", "cost", "profit"),
            show="tree headings",
            yscrollcommand=craft_scroll.set
        )
        self.craft_tree.heading("profession", text="Métier")
        self.craft_tree.heading("level", text="Niv.")
        self.craft_tree.heading("cost", text="Coût")
        self.craft_tree.heading("profit", text="Profit")
        self.craft_tree.column("#0", width=150)
        self.craft_tree.column("profession", width=100)
        self.craft_tree.column("level", width=50)
        self.craft_tree.column("cost", width=80)
        self.craft_tree.column("profit", width=80)
        self.craft_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        craft_scroll.config(command=self.craft_tree.yview)

        self.update_craft_tree()

        # Détails recette
        right_craft = ttk.LabelFrame(craft_paned, text="Détails Recette", padding=10)
        craft_paned.add(right_craft, weight=1)

        # Ingrédients
        ingredients_frame = ttk.LabelFrame(right_craft, text="Ingrédients", padding=5)
        ingredients_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.ingredients_text = tk.Text(ingredients_frame, height=10, font=("Consolas", 9),
                                       state=tk.DISABLED, wrap=tk.WORD)
        self.ingredients_text.pack(fill=tk.BOTH, expand=True)

        # Calcul rentabilité
        profitability_frame = ttk.LabelFrame(right_craft, text="Rentabilité", padding=5)
        profitability_frame.pack(fill=tk.X, pady=(0, 10))

        self.profitability_text = tk.Text(profitability_frame, height=6, font=("Consolas", 9),
                                         state=tk.DISABLED, wrap=tk.WORD)
        self.profitability_text.pack(fill=tk.BOTH, expand=True)

        # Actions craft
        craft_actions = ttk.Frame(right_craft)
        craft_actions.pack(fill=tk.X)

        ttk.Button(craft_actions, text="📊 Optimiser",
                  command=self.optimize_craft).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(craft_actions, text="🔨 Craft Auto",
                  command=self.auto_craft).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Farming & Rentabilité ---
        farming_tab = ttk.Frame(self.main_notebook, padding=5)
        self.main_notebook.add(farming_tab, text="🌾 Farming & Rentabilité")

        farming_paned = ttk.PanedWindow(farming_tab, orient=tk.HORIZONTAL)
        farming_paned.pack(fill=tk.BOTH, expand=True)

        # Liste spots
        left_farming = ttk.Frame(farming_paned)
        farming_paned.add(left_farming, weight=1)

        ttk.Label(left_farming, text="Spots de farming:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Filtres farming
        farming_filter_frame = ttk.Frame(left_farming)
        farming_filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(farming_filter_frame, text="Difficulté:").pack(side=tk.LEFT, padx=(0, 5))
        self.farming_difficulty_filter = ttk.Combobox(farming_filter_frame, values=[
            "Toutes", "Facile", "Moyen", "Difficile"
        ], state="readonly", width=12)
        self.farming_difficulty_filter.set("Toutes")
        self.farming_difficulty_filter.pack(side=tk.LEFT)

        ttk.Label(farming_filter_frame, text="Tri:").pack(side=tk.LEFT, padx=(10, 5))
        self.farming_sort = ttk.Combobox(farming_filter_frame, values=[
            "Kamas/h", "XP/h", "Niveau"
        ], state="readonly", width=12)
        self.farming_sort.set("Kamas/h")
        self.farming_sort.pack(side=tk.LEFT)
        self.farming_sort.bind("<<ComboboxSelected>>", lambda e: self.sort_farming_spots())

        # TreeView farming
        farming_list_frame = ttk.Frame(left_farming)
        farming_list_frame.pack(fill=tk.BOTH, expand=True)

        farming_scroll = ttk.Scrollbar(farming_list_frame)
        farming_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.farming_tree = ttk.Treeview(
            farming_list_frame,
            columns=("location", "level", "kamas_h", "xp_h", "difficulty"),
            show="tree headings",
            yscrollcommand=farming_scroll.set
        )
        self.farming_tree.heading("location", text="Localisation")
        self.farming_tree.heading("level", text="Niv.")
        self.farming_tree.heading("kamas_h", text="K/h")
        self.farming_tree.heading("xp_h", text="XP/h")
        self.farming_tree.heading("difficulty", text="Difficulté")
        self.farming_tree.column("#0", width=150)
        self.farming_tree.column("location", width=120)
        self.farming_tree.column("level", width=50)
        self.farming_tree.column("kamas_h", width=80)
        self.farming_tree.column("xp_h", width=80)
        self.farming_tree.column("difficulty", width=80)
        self.farming_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        farming_scroll.config(command=self.farming_tree.yview)

        self.update_farming_tree()

        # Détails spot
        right_farming = ttk.LabelFrame(farming_paned, text="Détails Spot", padding=10)
        farming_paned.add(right_farming, weight=1)

        # Infos spot
        spot_info_frame = ttk.LabelFrame(right_farming, text="Informations", padding=5)
        spot_info_frame.pack(fill=tk.X, pady=(0, 10))

        self.spot_info_text = tk.Text(spot_info_frame, height=5, font=("Consolas", 9),
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.spot_info_text.pack(fill=tk.BOTH, expand=True)

        # Loot table
        loot_frame = ttk.LabelFrame(right_farming, text="Table de loot", padding=5)
        loot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.loot_text = tk.Text(loot_frame, font=("Consolas", 8),
                                state=tk.DISABLED, wrap=tk.WORD)
        self.loot_text.pack(fill=tk.BOTH, expand=True)

        # Rentabilité estimée
        farming_profit_frame = ttk.LabelFrame(right_farming, text="Rentabilité", padding=5)
        farming_profit_frame.pack(fill=tk.X, pady=(0, 10))

        self.farming_profit_text = tk.Text(farming_profit_frame, height=4, font=("Consolas", 9),
                                          state=tk.DISABLED, wrap=tk.WORD)
        self.farming_profit_text.pack(fill=tk.BOTH, expand=True)

        # Actions
        farming_actions = ttk.Frame(right_farming)
        farming_actions.pack(fill=tk.X)

        ttk.Button(farming_actions, text="🎯 Aller au spot",
                  command=self.goto_farming_spot).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(farming_actions, text="▶️ Farm Auto",
                  command=self.start_auto_farm).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Inventaire ---
        inventory_tab = ttk.Frame(self.main_notebook, padding=5)
        self.main_notebook.add(inventory_tab, text="🎒 Inventaire")

        inventory_paned = ttk.PanedWindow(inventory_tab, orient=tk.HORIZONTAL)
        inventory_paned.pack(fill=tk.BOTH, expand=True)

        # Liste inventaire
        left_inventory = ttk.Frame(inventory_paned)
        inventory_paned.add(left_inventory, weight=1)

        # Stats inventaire
        inv_stats_frame = ttk.Frame(left_inventory)
        inv_stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.inv_stats_label = ttk.Label(inv_stats_frame, text="Inventaire: 0/100 | Poids: 0/1000",
                                        font=("Segoe UI", 9))
        self.inv_stats_label.pack(side=tk.LEFT)

        ttk.Button(inv_stats_frame, text="🗑️ Nettoyer",
                  command=self.clean_inventory).pack(side=tk.RIGHT, padx=2)
        ttk.Button(inv_stats_frame, text="💰 Vendre tout",
                  command=self.sell_all).pack(side=tk.RIGHT, padx=2)

        # TreeView inventaire
        inv_list_frame = ttk.Frame(left_inventory)
        inv_list_frame.pack(fill=tk.BOTH, expand=True)

        inv_scroll = ttk.Scrollbar(inv_list_frame)
        inv_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.inventory_tree = ttk.Treeview(
            inv_list_frame,
            columns=("quantity", "unit_price", "total_value"),
            show="tree headings",
            yscrollcommand=inv_scroll.set
        )
        self.inventory_tree.heading("quantity", text="Qté")
        self.inventory_tree.heading("unit_price", text="Prix unit.")
        self.inventory_tree.heading("total_value", text="Valeur")
        self.inventory_tree.column("#0", width=200)
        self.inventory_tree.column("quantity", width=80)
        self.inventory_tree.column("unit_price", width=100)
        self.inventory_tree.column("total_value", width=100)
        self.inventory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inv_scroll.config(command=self.inventory_tree.yview)

        self.update_inventory_tree()

        # Recommandations
        right_inventory = ttk.LabelFrame(inventory_paned, text="💡 Recommandations", padding=10)
        inventory_paned.add(right_inventory, weight=1)

        self.recommendations_text = tk.Text(right_inventory, font=("Segoe UI", 9),
                                           wrap=tk.WORD, state=tk.DISABLED)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)

        self.update_recommendations()

        # --- Onglet Opportunités ---
        opportunities_tab = ttk.Frame(self.main_notebook, padding=10)
        self.main_notebook.add(opportunities_tab, text="💡 Opportunités")

        ttk.Label(opportunities_tab, text="Opportunités d'achat/vente détectées",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Liste opportunités
        opp_frame = ttk.Frame(opportunities_tab)
        opp_frame.pack(fill=tk.BOTH, expand=True)

        opp_scroll = ttk.Scrollbar(opp_frame)
        opp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.opportunities_tree = ttk.Treeview(
            opp_frame,
            columns=("type", "item", "profit", "confidence"),
            show="tree headings",
            yscrollcommand=opp_scroll.set
        )
        self.opportunities_tree.heading("type", text="Type")
        self.opportunities_tree.heading("item", text="Objet")
        self.opportunities_tree.heading("profit", text="Profit potentiel")
        self.opportunities_tree.heading("confidence", text="Confiance")
        self.opportunities_tree.column("#0", width=30)
        self.opportunities_tree.column("type", width=100)
        self.opportunities_tree.column("item", width=200)
        self.opportunities_tree.column("profit", width=120)
        self.opportunities_tree.column("confidence", width=100)
        self.opportunities_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        opp_scroll.config(command=self.opportunities_tree.yview)

        self.update_opportunities()

        # === Section inférieure: Statistiques globales ===
        bottom_frame = ttk.LabelFrame(main_paned, text="📊 Vue d'ensemble", padding=10)
        main_paned.add(bottom_frame, weight=1)

        # Stats globales en colonnes
        stats_container = ttk.Frame(bottom_frame)
        stats_container.pack(fill=tk.BOTH, expand=True)

        # Colonne 1: Patrimoine
        wealth_frame = ttk.LabelFrame(stats_container, text="💰 Patrimoine", padding=10)
        wealth_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.wealth_text = tk.Text(wealth_frame, height=8, font=("Consolas", 9),
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.wealth_text.pack(fill=tk.BOTH, expand=True)

        # Colonne 2: Transactions récentes
        transactions_frame = ttk.LabelFrame(stats_container, text="📈 Transactions", padding=10)
        transactions_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.transactions_text = tk.Text(transactions_frame, height=8, font=("Consolas", 8),
                                        state=tk.DISABLED, wrap=tk.WORD)
        self.transactions_text.pack(fill=tk.BOTH, expand=True)

        # Colonne 3: Performance
        performance_frame = ttk.LabelFrame(stats_container, text="📊 Performance", padding=10)
        performance_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.performance_text = tk.Text(performance_frame, height=8, font=("Consolas", 9),
                                       state=tk.DISABLED, wrap=tk.WORD)
        self.performance_text.pack(fill=tk.BOTH, expand=True)

        self.update_global_stats()

    def load_economy_data(self):
        """Charge les données économiques (démo)"""
        now = datetime.now()

        # Market items (démo)
        self.market_items = [
            MarketItem(
                item_id="item_001",
                name="Poil de Bouftou",
                category=ItemCategory.RESOURCE,
                level=1,
                price_avg=50,
                price_min=30,
                price_max=80,
                quantity_available=1500,
                last_update=now,
                trend="up",
                price_history=[]
            ),
            MarketItem(
                item_id="item_002",
                name="Épée de Boisaille",
                category=ItemCategory.EQUIPMENT,
                level=10,
                price_avg=5000,
                price_min=4500,
                price_max=5500,
                quantity_available=25,
                last_update=now,
                trend="stable",
                price_history=[]
            ),
            MarketItem(
                item_id="item_003",
                name="Potion de Vie",
                category=ItemCategory.CONSUMABLE,
                level=1,
                price_avg=100,
                price_min=80,
                price_max=150,
                quantity_available=800,
                last_update=now,
                trend="down",
                price_history=[]
            )
        ]

        # Craft recipes (démo)
        self.craft_recipes = [
            CraftRecipe(
                recipe_id="recipe_001",
                result_item="Épée Forgée",
                result_quantity=1,
                ingredients=[
                    {"item": "Fer", "quantity": 10},
                    {"item": "Bois", "quantity": 5}
                ],
                craft_cost=500,
                success_rate=0.95,
                profession="Forgeron",
                level_required=20
            )
        ]

        # Farming spots (démo)
        self.farming_spots = [
            FarmingSpot(
                spot_id="spot_001",
                name="Zone des Bouftous",
                map_location="Astrub [-1, 2]",
                monsters=["Bouftou", "Bouftou Blanc"],
                loot_table=[
                    {"item": "Poil de Bouftou", "drop_rate": 0.8, "value": 50},
                    {"item": "Viande", "drop_rate": 0.5, "value": 30}
                ],
                estimated_kamas_per_hour=15000,
                estimated_xp_per_hour=5000,
                difficulty="easy",
                recommended_level=5
            ),
            FarmingSpot(
                spot_id="spot_002",
                name="Forêt des Larves",
                map_location="Astrub [0, 1]",
                monsters=["Larve Bleue", "Larve Verte"],
                loot_table=[
                    {"item": "Œuf de Larve", "drop_rate": 0.6, "value": 80}
                ],
                estimated_kamas_per_hour=20000,
                estimated_xp_per_hour=8000,
                difficulty="easy",
                recommended_level=3
            )
        ]

        # Inventaire (démo)
        self.inventory = {
            "Poil de Bouftou": 25,
            "Viande": 10,
            "Potion de Vie": 5
        }

        self.total_wealth = 125000

    def update_market_tree(self):
        """Met à jour l'arbre du marché"""
        self.market_tree.delete(*self.market_tree.get_children())

        for item in self.market_items:
            trend_emoji = {"up": "↗️", "down": "↘️", "stable": "➡️"}
            cat_str = item.category.value
            trend_str = f"{trend_emoji.get(item.trend, '?')} {item.trend}"

            self.market_tree.insert("", tk.END, text=item.name,
                                   values=(cat_str, item.level, f"{item.price_avg}K",
                                          item.quantity_available, trend_str))

    def update_craft_tree(self):
        """Met à jour l'arbre du craft"""
        self.craft_tree.delete(*self.craft_tree.get_children())

        for recipe in self.craft_recipes:
            # Calculer profit estimé (simplifié)
            profit = 5000  # TODO: calculer réel

            self.craft_tree.insert("", tk.END, text=recipe.result_item,
                                  values=(recipe.profession, recipe.level_required,
                                         f"{recipe.craft_cost}K", f"+{profit}K"))

    def update_farming_tree(self):
        """Met à jour l'arbre du farming"""
        self.farming_tree.delete(*self.farming_tree.get_children())

        for spot in self.farming_spots:
            diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

            self.farming_tree.insert("", tk.END, text=spot.name,
                                    values=(spot.map_location, spot.recommended_level,
                                           f"{spot.estimated_kamas_per_hour // 1000}K",
                                           f"{spot.estimated_xp_per_hour // 1000}K",
                                           f"{diff_emoji.get(spot.difficulty, '?')} {spot.difficulty}"))

    def update_inventory_tree(self):
        """Met à jour l'arbre de l'inventaire"""
        self.inventory_tree.delete(*self.inventory_tree.get_children())

        total_value = 0
        for item_name, quantity in self.inventory.items():
            # Trouver le prix de l'objet
            unit_price = 50  # TODO: récupérer prix réel
            total_value += quantity * unit_price

            self.inventory_tree.insert("", tk.END, text=item_name,
                                      values=(quantity, f"{unit_price}K", f"{quantity * unit_price}K"))

        # Mettre à jour stats
        self.inv_stats_label.config(text=f"Inventaire: {len(self.inventory)}/100 | Valeur: {total_value}K")

    def update_opportunities(self):
        """Met à jour les opportunités"""
        self.opportunities_tree.delete(*self.opportunities_tree.get_children())

        # Opportunités démo
        opportunities = [
            ("Achat", "Poil de Bouftou", "+5K par unité", "Haute"),
            ("Vente", "Épée de Boisaille", "+15K", "Moyenne"),
            ("Craft", "Épée Forgée", "+20K", "Haute")
        ]

        for i, opp in enumerate(opportunities):
            self.opportunities_tree.insert("", tk.END, text=str(i+1), values=opp)

    def update_recommendations(self):
        """Met à jour les recommandations"""
        reco = """💡 Recommandations intelligentes:

1. VENDRE: Poil de Bouftou
   ↗️ Prix en hausse (+15%)
   💰 Profit estimé: +1,250K

2. ACHETER: Potion de Vie
   ↘️ Prix en baisse (-20%)
   📈 Bonne opportunité

3. CRAFT: Épée Forgée
   💰 Rentabilité: +20K par craft
   ⚙️ Taux de succès: 95%

4. FARMING: Zone des Bouftous
   ⏱️ 15K kamas/h
   🎯 Adapté à ton niveau
"""
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete("1.0", tk.END)
        self.recommendations_text.insert("1.0", reco)
        self.recommendations_text.config(state=tk.DISABLED)

    def update_global_stats(self):
        """Met à jour les stats globales"""
        # Patrimoine
        wealth = f"""Total: {self.total_wealth:,} kamas

Répartition:
├─ Liquide: 50,000K
├─ Inventaire: 25,000K
├─ HDV: 30,000K
└─ Banque: 20,000K

Évolution:
└─ +15% (7 jours)
"""
        self.wealth_text.config(state=tk.NORMAL)
        self.wealth_text.delete("1.0", tk.END)
        self.wealth_text.insert("1.0", wealth)
        self.wealth_text.config(state=tk.DISABLED)

        # Transactions
        transactions = """Dernières 24h:

• Vendu Poil x50: +2,500K
• Acheté Potion x10: -1,000K
• Craft Épée: +20,000K
• Loot farming: +15,000K

Total: +36,500K
"""
        self.transactions_text.config(state=tk.NORMAL)
        self.transactions_text.delete("1.0", tk.END)
        self.transactions_text.insert("1.0", transactions)
        self.transactions_text.config(state=tk.DISABLED)

        # Performance
        performance = f"""Aujourd'hui:

Revenus: +45,000K
Dépenses: -8,500K
Net: +36,500K

Kamas/h: 15,200K
ROI: +23%
"""
        self.performance_text.config(state=tk.NORMAL)
        self.performance_text.delete("1.0", tk.END)
        self.performance_text.insert("1.0", performance)
        self.performance_text.config(state=tk.DISABLED)

        # Mettre à jour label toolbar
        self.wealth_label.config(text=f"💰 Patrimoine: {self.total_wealth // 1000}K")

    # === Actions ===

    def filter_market(self):
        """Filtre les objets du marché"""
        # TODO: implémenter filtrage réel
        self.update_market_tree()

    def on_market_item_select(self, event):
        """Gère la sélection d'un objet"""
        # TODO: afficher détails
        pass

    def refresh_market(self):
        messagebox.showinfo("HDV", "Actualisation du HDV - à implémenter")

    def generate_report(self):
        messagebox.showinfo("Rapport", "Génération rapport économique - à implémenter")

    def analyze_price_trend(self):
        messagebox.showinfo("Analyse", "Analyse de tendance - à implémenter")

    def create_price_alert(self):
        messagebox.showinfo("Alerte", "Création d'alerte prix - à implémenter")

    def filter_crafts(self):
        self.update_craft_tree()

    def optimize_craft(self):
        messagebox.showinfo("Optimisation", "Optimisation craft - à implémenter")

    def auto_craft(self):
        messagebox.showinfo("Auto-craft", "Craft automatique - à implémenter")

    def sort_farming_spots(self):
        self.update_farming_tree()

    def goto_farming_spot(self):
        messagebox.showinfo("Navigation", "Navigation vers spot - à implémenter")

    def start_auto_farm(self):
        messagebox.showinfo("Auto-farm", "Farming automatique - à implémenter")

    def clean_inventory(self):
        if messagebox.askyesno("Nettoyer", "Vendre les objets non essentiels?"):
            messagebox.showinfo("Nettoyé", "Inventaire nettoyé - à implémenter")

    def sell_all(self):
        if messagebox.askyesno("Vendre", "Vendre tout l'inventaire au HDV?"):
            messagebox.showinfo("Vendu", "Tout vendu - à implémenter")

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
