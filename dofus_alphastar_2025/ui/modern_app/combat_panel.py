"""
Combat Panel - Analyse de combat avec AAR (After Action Review)
Statistiques, historique des combats, analyse tactique
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json


class CombatResult(Enum):
    """Résultat d'un combat"""
    VICTORY = "victory"
    DEFEAT = "defeat"
    FLEE = "flee"
    DRAW = "draw"


@dataclass
class CombatAction:
    """Action effectuée durant un combat"""
    turn: int
    actor: str
    action_type: str  # 'spell', 'move', 'item', 'pass'
    target: Optional[str]
    damage: int
    healing: int
    ap_cost: int
    mp_cost: int
    success: bool
    critical: bool


@dataclass
class CombatLog:
    """Log complet d'un combat"""
    combat_id: str
    start_time: datetime
    end_time: datetime
    duration: timedelta
    result: CombatResult
    enemies: List[Dict[str, Any]]
    allies: List[Dict[str, Any]]
    total_turns: int
    actions: List[CombatAction]
    total_damage_dealt: int
    total_damage_taken: int
    total_healing: int
    xp_gained: int
    loot: List[Dict[str, Any]]
    kamas_gained: int
    map_location: str
    tactical_score: float  # 0-100
    efficiency_score: float  # 0-100
    notes: str


class CombatPanel:
    """
    Panel d'analyse de combat avec AAR (After Action Review)

    Fonctionnalités:
    - Historique des combats
    - Statistiques détaillées
    - Analyse tactique (AAR)
    - Replay visuel des combats
    - Graphiques de performance
    - Recommandations d'amélioration
    - Export de données
    """

    def __init__(self, parent):
        self.parent = parent
        self.combat_logs: List[CombatLog] = []
        self.current_combat: Optional[CombatLog] = None

        # Charger les données
        self.load_combat_data()

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Toolbar supérieure ===
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(toolbar, text="⚔️ Centre d'Analyse Combat & AAR",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        ttk.Button(toolbar, text="🔄 Rafraîchir",
                  command=self.refresh_data).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="📊 Rapport AAR",
                  command=self.generate_aar_report).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="💾 Exporter",
                  command=self.export_data).pack(side=tk.RIGHT, padx=2)

        # === PanedWindow pour layout flexible ===
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Panneau gauche: Historique des combats ===
        left_frame = ttk.LabelFrame(paned, text="📜 Historique des Combats", padding=5)
        paned.add(left_frame, weight=1)

        # Filtres
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Résultat:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.filter_result = ttk.Combobox(filter_frame, values=[
            "Tous", "Victoire", "Défaite", "Fuite", "Égalité"
        ], state="readonly", width=12)
        self.filter_result.set("Tous")
        self.filter_result.grid(row=0, column=1, sticky=tk.W)
        self.filter_result.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        ttk.Label(filter_frame, text="Période:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.filter_period = ttk.Combobox(filter_frame, values=[
            "Aujourd'hui", "7 derniers jours", "30 derniers jours", "Tout"
        ], state="readonly", width=15)
        self.filter_period.set("7 derniers jours")
        self.filter_period.grid(row=0, column=3, sticky=tk.W)
        self.filter_period.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # TreeView pour historique
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.combats_tree = ttk.Treeview(
            list_frame,
            columns=("time", "result", "enemies", "duration", "score"),
            show="tree headings",
            yscrollcommand=list_scroll.set
        )
        self.combats_tree.heading("time", text="Heure")
        self.combats_tree.heading("result", text="Résultat")
        self.combats_tree.heading("enemies", text="Ennemis")
        self.combats_tree.heading("duration", text="Durée")
        self.combats_tree.heading("score", text="Score")
        self.combats_tree.column("#0", width=30)
        self.combats_tree.column("time", width=80)
        self.combats_tree.column("result", width=80)
        self.combats_tree.column("enemies", width=120)
        self.combats_tree.column("duration", width=60)
        self.combats_tree.column("score", width=60)
        self.combats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.combats_tree.yview)
        self.combats_tree.bind("<<TreeviewSelect>>", self.on_combat_select)

        self.update_combats_tree()

        # Stats rapides
        stats_frame = ttk.LabelFrame(left_frame, text="📊 Stats Rapides", padding=5)
        stats_frame.pack(fill=tk.X)

        self.quick_stats_text = tk.Text(stats_frame, height=6, font=("Consolas", 8),
                                       state=tk.DISABLED, wrap=tk.WORD)
        self.quick_stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_quick_stats()

        # === Panneau central: Détails du combat ===
        center_frame = ttk.Frame(paned)
        paned.add(center_frame, weight=2)

        # Notebook pour organiser les informations
        self.details_notebook = ttk.Notebook(center_frame)
        self.details_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Résumé ---
        summary_tab = ttk.Frame(self.details_notebook, padding=10)
        self.details_notebook.add(summary_tab, text="📋 Résumé")

        # Infos générales
        info_frame = ttk.LabelFrame(summary_tab, text="Informations générales", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=6, font=("Consolas", 9),
                                state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Participants
        participants_frame = ttk.LabelFrame(summary_tab, text="Participants", padding=10)
        participants_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Split en deux colonnes
        parts_paned = ttk.PanedWindow(participants_frame, orient=tk.HORIZONTAL)
        parts_paned.pack(fill=tk.BOTH, expand=True)

        # Alliés
        allies_frame = ttk.LabelFrame(parts_paned, text="👥 Alliés", padding=5)
        parts_paned.add(allies_frame, weight=1)

        self.allies_text = tk.Text(allies_frame, height=8, font=("Consolas", 8),
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.allies_text.pack(fill=tk.BOTH, expand=True)

        # Ennemis
        enemies_frame = ttk.LabelFrame(parts_paned, text="👹 Ennemis", padding=5)
        parts_paned.add(enemies_frame, weight=1)

        self.enemies_text = tk.Text(enemies_frame, height=8, font=("Consolas", 8),
                                   state=tk.DISABLED, wrap=tk.WORD)
        self.enemies_text.pack(fill=tk.BOTH, expand=True)

        # Récompenses
        rewards_frame = ttk.LabelFrame(summary_tab, text="🎁 Récompenses", padding=10)
        rewards_frame.pack(fill=tk.X)

        self.rewards_text = tk.Text(rewards_frame, height=4, font=("Consolas", 9),
                                   state=tk.DISABLED, wrap=tk.WORD)
        self.rewards_text.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Timeline ---
        timeline_tab = ttk.Frame(self.details_notebook, padding=10)
        self.details_notebook.add(timeline_tab, text="⏱️ Timeline")

        ttk.Label(timeline_tab, text="Déroulement du combat (tour par tour)",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # TreeView pour actions
        timeline_frame = ttk.Frame(timeline_tab)
        timeline_frame.pack(fill=tk.BOTH, expand=True)

        timeline_scroll = ttk.Scrollbar(timeline_frame)
        timeline_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.timeline_tree = ttk.Treeview(
            timeline_frame,
            columns=("turn", "actor", "action", "target", "damage", "result"),
            show="tree headings",
            yscrollcommand=timeline_scroll.set
        )
        self.timeline_tree.heading("turn", text="Tour")
        self.timeline_tree.heading("actor", text="Acteur")
        self.timeline_tree.heading("action", text="Action")
        self.timeline_tree.heading("target", text="Cible")
        self.timeline_tree.heading("damage", text="Dégâts")
        self.timeline_tree.heading("result", text="Résultat")
        self.timeline_tree.column("#0", width=30)
        self.timeline_tree.column("turn", width=50)
        self.timeline_tree.column("actor", width=100)
        self.timeline_tree.column("action", width=120)
        self.timeline_tree.column("target", width=100)
        self.timeline_tree.column("damage", width=80)
        self.timeline_tree.column("result", width=80)
        self.timeline_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        timeline_scroll.config(command=self.timeline_tree.yview)

        # --- Onglet Statistiques ---
        stats_tab = ttk.Frame(self.details_notebook, padding=10)
        self.details_notebook.add(stats_tab, text="📊 Statistiques")

        # Métriques de combat
        metrics_frame = ttk.LabelFrame(stats_tab, text="Métriques de combat", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.metrics_text = tk.Text(metrics_frame, height=15, font=("Consolas", 9),
                                   state=tk.DISABLED, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Graphiques (simulé avec texte ASCII pour l'instant)
        graphs_frame = ttk.LabelFrame(stats_tab, text="Graphiques", padding=10)
        graphs_frame.pack(fill=tk.BOTH, expand=True)

        self.graphs_text = tk.Text(graphs_frame, height=10, font=("Consolas", 8),
                                  state=tk.DISABLED, wrap=tk.NONE)
        self.graphs_text.pack(fill=tk.BOTH, expand=True)

        # --- Onglet AAR (After Action Review) ---
        aar_tab = ttk.Frame(self.details_notebook, padding=10)
        self.details_notebook.add(aar_tab, text="🎯 AAR")

        ttk.Label(aar_tab, text="After Action Review - Analyse Tactique",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Scores
        scores_frame = ttk.LabelFrame(aar_tab, text="Scores de Performance", padding=10)
        scores_frame.pack(fill=tk.X, pady=(0, 10))

        # Score tactique
        ttk.Label(scores_frame, text="Score Tactique:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.tactical_score_var = tk.DoubleVar(value=0)
        tactical_progress = ttk.Progressbar(scores_frame, variable=self.tactical_score_var,
                                          maximum=100, mode='determinate', length=200)
        tactical_progress.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.tactical_score_label = ttk.Label(scores_frame, text="0/100")
        self.tactical_score_label.grid(row=0, column=2, sticky=tk.W)

        # Score efficacité
        ttk.Label(scores_frame, text="Score Efficacité:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.efficiency_score_var = tk.DoubleVar(value=0)
        efficiency_progress = ttk.Progressbar(scores_frame, variable=self.efficiency_score_var,
                                            maximum=100, mode='determinate', length=200)
        efficiency_progress.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.efficiency_score_label = ttk.Label(scores_frame, text="0/100")
        self.efficiency_score_label.grid(row=1, column=2, sticky=tk.W)

        scores_frame.columnconfigure(1, weight=1)

        # Analyse détaillée
        analysis_frame = ttk.LabelFrame(aar_tab, text="Analyse Détaillée", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.aar_text = tk.Text(analysis_frame, font=("Segoe UI", 9),
                               wrap=tk.WORD, state=tk.DISABLED)
        self.aar_text.pack(fill=tk.BOTH, expand=True)

        # Recommandations
        reco_frame = ttk.LabelFrame(aar_tab, text="💡 Recommandations", padding=10)
        reco_frame.pack(fill=tk.BOTH, expand=True)

        self.reco_text = tk.Text(reco_frame, font=("Segoe UI", 9),
                                wrap=tk.WORD, state=tk.DISABLED)
        self.reco_text.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Replay ---
        replay_tab = ttk.Frame(self.details_notebook, padding=10)
        self.details_notebook.add(replay_tab, text="▶️ Replay")

        ttk.Label(replay_tab, text="Replay Visuel du Combat",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Canvas pour visualisation (grille de combat)
        canvas_frame = ttk.LabelFrame(replay_tab, text="Plateau de combat", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.replay_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', width=600, height=400)
        self.replay_canvas.pack(fill=tk.BOTH, expand=True)

        # Contrôles replay
        controls_frame = ttk.Frame(replay_tab)
        controls_frame.pack(fill=tk.X)

        ttk.Button(controls_frame, text="⏮️ Début",
                  command=self.replay_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="◀️ Précédent",
                  command=self.replay_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="▶️ Suivant",
                  command=self.replay_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="⏭️ Fin",
                  command=self.replay_end).pack(side=tk.LEFT, padx=2)

        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(controls_frame, text="▶️ Play Auto",
                  command=self.replay_auto).pack(side=tk.LEFT, padx=2)

        self.replay_turn_var = tk.IntVar(value=0)
        ttk.Label(controls_frame, text="Tour:").pack(side=tk.LEFT, padx=(20, 5))
        self.replay_turn_label = ttk.Label(controls_frame, text="0/0")
        self.replay_turn_label.pack(side=tk.LEFT)

        # === Panneau droit: Statistiques globales ===
        right_frame = ttk.LabelFrame(paned, text="📊 Statistiques Globales", padding=10)
        paned.add(right_frame, weight=1)

        # Stats par période
        period_frame = ttk.LabelFrame(right_frame, text="Période", padding=5)
        period_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(period_frame, text="Afficher:").pack(side=tk.LEFT, padx=(0, 5))
        self.stats_period = ttk.Combobox(period_frame, values=[
            "Aujourd'hui", "7 jours", "30 jours", "Tout"
        ], state="readonly", width=12)
        self.stats_period.set("7 jours")
        self.stats_period.pack(side=tk.LEFT)
        self.stats_period.bind("<<ComboboxSelected>>", lambda e: self.update_global_stats())

        # Statistiques globales
        global_stats_frame = ttk.Frame(right_frame)
        global_stats_frame.pack(fill=tk.BOTH, expand=True)

        self.global_stats_text = tk.Text(global_stats_frame, font=("Consolas", 9),
                                        state=tk.DISABLED, wrap=tk.WORD)
        self.global_stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_global_stats()

        # Boutons d'action
        actions_frame = ttk.LabelFrame(right_frame, text="Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(actions_frame, text="📈 Tendances",
                  command=self.show_trends).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="🏆 Records",
                  command=self.show_records).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="📊 Comparer combats",
                  command=self.compare_combats).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="🧹 Nettoyer historique",
                  command=self.clean_history).pack(fill=tk.X, pady=2)

    def load_combat_data(self):
        """Charge les données de combat (démo)"""
        # TODO: Charger depuis fichiers JSON
        # Créer des données de démo
        now = datetime.now()

        self.combat_logs = [
            CombatLog(
                combat_id="combat_001",
                start_time=now - timedelta(hours=2),
                end_time=now - timedelta(hours=2) + timedelta(minutes=5),
                duration=timedelta(minutes=5),
                result=CombatResult.VICTORY,
                enemies=[
                    {"name": "Bouftou", "level": 5, "hp": 150},
                    {"name": "Bouftou", "level": 4, "hp": 120}
                ],
                allies=[{"name": "Joueur", "level": 10, "class": "Iop"}],
                total_turns=8,
                actions=[],
                total_damage_dealt=270,
                total_damage_taken=45,
                total_healing=0,
                xp_gained=250,
                loot=[{"item": "Poil de Bouftou", "quantity": 3}],
                kamas_gained=50,
                map_location="Astrub [-1, 2]",
                tactical_score=85.0,
                efficiency_score=92.0,
                notes=""
            ),
            CombatLog(
                combat_id="combat_002",
                start_time=now - timedelta(hours=1),
                end_time=now - timedelta(hours=1) + timedelta(minutes=3),
                duration=timedelta(minutes=3),
                result=CombatResult.VICTORY,
                enemies=[{"name": "Larve Bleue", "level": 3, "hp": 80}],
                allies=[{"name": "Joueur", "level": 10, "class": "Iop"}],
                total_turns=4,
                actions=[],
                total_damage_dealt=80,
                total_damage_taken=15,
                total_healing=0,
                xp_gained=100,
                loot=[],
                kamas_gained=20,
                map_location="Astrub [0, 1]",
                tactical_score=78.0,
                efficiency_score=88.0,
                notes=""
            ),
            CombatLog(
                combat_id="combat_003",
                start_time=now - timedelta(minutes=30),
                end_time=now - timedelta(minutes=30) + timedelta(minutes=8),
                duration=timedelta(minutes=8),
                result=CombatResult.DEFEAT,
                enemies=[
                    {"name": "Craqueleur", "level": 15, "hp": 500},
                    {"name": "Craqueleur", "level": 14, "hp": 450}
                ],
                allies=[{"name": "Joueur", "level": 10, "class": "Iop"}],
                total_turns=12,
                actions=[],
                total_damage_dealt=350,
                total_damage_taken=500,
                total_healing=50,
                xp_gained=0,
                loot=[],
                kamas_gained=0,
                map_location="Forêt Maléfique [-5, -8]",
                tactical_score=45.0,
                efficiency_score=52.0,
                notes="Combat trop difficile - ennemis trop forts"
            )
        ]

    def update_combats_tree(self):
        """Met à jour l'arbre des combats"""
        self.combats_tree.delete(*self.combats_tree.get_children())

        filtered = self._get_filtered_combats()

        for i, combat in enumerate(reversed(filtered)):
            time_str = combat.start_time.strftime("%H:%M")
            result_emoji = {"victory": "✅", "defeat": "❌", "flee": "🏃", "draw": "⚖️"}
            result_str = f"{result_emoji.get(combat.result.value, '?')} {combat.result.value}"
            enemies_str = f"{len(combat.enemies)} ennemis"
            duration_str = f"{int(combat.duration.total_seconds() // 60)}m"
            score_str = f"{int(combat.tactical_score)}/100"

            self.combats_tree.insert("", tk.END, text=str(i+1),
                                    values=(time_str, result_str, enemies_str,
                                           duration_str, score_str))

    def _get_filtered_combats(self) -> List[CombatLog]:
        """Retourne les combats filtrés"""
        filtered = self.combat_logs

        # Filtre par résultat
        result_filter = self.filter_result.get()
        if result_filter != "Tous":
            result_map = {
                "Victoire": CombatResult.VICTORY,
                "Défaite": CombatResult.DEFEAT,
                "Fuite": CombatResult.FLEE,
                "Égalité": CombatResult.DRAW
            }
            if result_filter in result_map:
                filtered = [c for c in filtered if c.result == result_map[result_filter]]

        # Filtre par période
        period_filter = self.filter_period.get()
        now = datetime.now()
        if period_filter == "Aujourd'hui":
            filtered = [c for c in filtered if c.start_time.date() == now.date()]
        elif period_filter == "7 derniers jours":
            cutoff = now - timedelta(days=7)
            filtered = [c for c in filtered if c.start_time >= cutoff]
        elif period_filter == "30 derniers jours":
            cutoff = now - timedelta(days=30)
            filtered = [c for c in filtered if c.start_time >= cutoff]

        return filtered

    def update_quick_stats(self):
        """Met à jour les stats rapides"""
        filtered = self._get_filtered_combats()
        total = len(filtered)

        if total == 0:
            stats = "Aucun combat dans la période sélectionnée"
        else:
            victories = sum(1 for c in filtered if c.result == CombatResult.VICTORY)
            defeats = sum(1 for c in filtered if c.result == CombatResult.DEFEAT)
            win_rate = (victories / total * 100) if total > 0 else 0

            avg_duration = sum((c.duration.total_seconds() for c in filtered), 0) / total
            total_xp = sum(c.xp_gained for c in filtered)
            total_kamas = sum(c.kamas_gained for c in filtered)

            stats = f"""Total: {total} combats
Victoires: {victories} ({win_rate:.1f}%)
Défaites: {defeats}
Durée moy.: {int(avg_duration // 60)}m {int(avg_duration % 60)}s
XP totale: {total_xp:,}
Kamas: {total_kamas:,}
"""

        self.quick_stats_text.config(state=tk.NORMAL)
        self.quick_stats_text.delete("1.0", tk.END)
        self.quick_stats_text.insert("1.0", stats)
        self.quick_stats_text.config(state=tk.DISABLED)

    def update_global_stats(self):
        """Met à jour les statistiques globales"""
        period = self.stats_period.get()

        # Filtrer par période
        now = datetime.now()
        if period == "Aujourd'hui":
            filtered = [c for c in self.combat_logs if c.start_time.date() == now.date()]
        elif period == "7 jours":
            cutoff = now - timedelta(days=7)
            filtered = [c for c in self.combat_logs if c.start_time >= cutoff]
        elif period == "30 jours":
            cutoff = now - timedelta(days=30)
            filtered = [c for c in self.combat_logs if c.start_time >= cutoff]
        else:
            filtered = self.combat_logs

        if not filtered:
            stats = "Aucune donnée pour cette période"
        else:
            total = len(filtered)
            victories = sum(1 for c in filtered if c.result == CombatResult.VICTORY)
            defeats = sum(1 for c in filtered if c.result == CombatResult.DEFEAT)

            avg_tactical = sum(c.tactical_score for c in filtered) / total
            avg_efficiency = sum(c.efficiency_score for c in filtered) / total

            total_damage_dealt = sum(c.total_damage_dealt for c in filtered)
            total_damage_taken = sum(c.total_damage_taken for c in filtered)

            stats = f"""=== Statistiques {period} ===

Combats: {total}
├─ Victoires: {victories} ({victories/total*100:.1f}%)
├─ Défaites: {defeats} ({defeats/total*100:.1f}%)
└─ Taux de victoire: {victories/total*100:.1f}%

Scores moyens:
├─ Tactique: {avg_tactical:.1f}/100
└─ Efficacité: {avg_efficiency:.1f}/100

Dégâts:
├─ Infligés: {total_damage_dealt:,}
├─ Reçus: {total_damage_taken:,}
└─ Ratio: {total_damage_dealt/total_damage_taken if total_damage_taken > 0 else 0:.2f}

XP & Kamas:
├─ XP totale: {sum(c.xp_gained for c in filtered):,}
└─ Kamas: {sum(c.kamas_gained for c in filtered):,}

Durée moyenne:
└─ {int(sum(c.duration.total_seconds() for c in filtered) / total // 60)}m par combat
"""

        self.global_stats_text.config(state=tk.NORMAL)
        self.global_stats_text.delete("1.0", tk.END)
        self.global_stats_text.insert("1.0", stats)
        self.global_stats_text.config(state=tk.DISABLED)

    def on_combat_select(self, event):
        """Gère la sélection d'un combat"""
        selection = self.combats_tree.selection()
        if not selection:
            return

        filtered = self._get_filtered_combats()
        reversed_filtered = list(reversed(filtered))

        item = self.combats_tree.item(selection[0])
        index = int(item['text']) - 1

        if index < len(reversed_filtered):
            self.current_combat = reversed_filtered[index]
            self.display_combat_details()

    def display_combat_details(self):
        """Affiche les détails du combat sélectionné"""
        if not self.current_combat:
            return

        combat = self.current_combat

        # Résumé
        info = f"""Combat ID: {combat.combat_id}
Début: {combat.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Durée: {int(combat.duration.total_seconds() // 60)}m {int(combat.duration.total_seconds() % 60)}s
Résultat: {combat.result.value.upper()}
Localisation: {combat.map_location}
Tours: {combat.total_turns}
"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", info)
        self.info_text.config(state=tk.DISABLED)

        # Alliés
        allies_info = ""
        for ally in combat.allies:
            allies_info += f"• {ally['name']} (Niv.{ally['level']} {ally['class']})\n"
        self.allies_text.config(state=tk.NORMAL)
        self.allies_text.delete("1.0", tk.END)
        self.allies_text.insert("1.0", allies_info)
        self.allies_text.config(state=tk.DISABLED)

        # Ennemis
        enemies_info = ""
        for enemy in combat.enemies:
            enemies_info += f"• {enemy['name']} (Niv.{enemy['level']}) - {enemy['hp']} HP\n"
        self.enemies_text.config(state=tk.NORMAL)
        self.enemies_text.delete("1.0", tk.END)
        self.enemies_text.insert("1.0", enemies_info)
        self.enemies_text.config(state=tk.DISABLED)

        # Récompenses
        rewards_info = f"""XP: {combat.xp_gained}
Kamas: {combat.kamas_gained}
Loot: {len(combat.loot)} objets
"""
        for item in combat.loot:
            rewards_info += f"  • {item['item']} x{item['quantity']}\n"

        self.rewards_text.config(state=tk.NORMAL)
        self.rewards_text.delete("1.0", tk.END)
        self.rewards_text.insert("1.0", rewards_info)
        self.rewards_text.config(state=tk.DISABLED)

        # Statistiques
        metrics = f"""=== Métriques de Combat ===

Dégâts:
├─ Infligés: {combat.total_damage_dealt}
├─ Reçus: {combat.total_damage_taken}
├─ Soins: {combat.total_healing}
└─ Ratio D/R: {combat.total_damage_dealt/combat.total_damage_taken if combat.total_damage_taken > 0 else "∞"}

Efficacité:
├─ DPS: {combat.total_damage_dealt / combat.duration.total_seconds():.1f}
├─ Dégâts/Tour: {combat.total_damage_dealt / combat.total_turns:.1f}
└─ Durée/Tour: {combat.duration.total_seconds() / combat.total_turns:.1f}s
"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", metrics)
        self.metrics_text.config(state=tk.DISABLED)

        # Scores AAR
        self.tactical_score_var.set(combat.tactical_score)
        self.tactical_score_label.config(text=f"{int(combat.tactical_score)}/100")
        self.efficiency_score_var.set(combat.efficiency_score)
        self.efficiency_score_label.config(text=f"{int(combat.efficiency_score)}/100")

        # Analyse AAR
        aar_analysis = self._generate_aar_analysis(combat)
        self.aar_text.config(state=tk.NORMAL)
        self.aar_text.delete("1.0", tk.END)
        self.aar_text.insert("1.0", aar_analysis)
        self.aar_text.config(state=tk.DISABLED)

        # Recommandations
        recommendations = self._generate_recommendations(combat)
        self.reco_text.config(state=tk.NORMAL)
        self.reco_text.delete("1.0", tk.END)
        self.reco_text.insert("1.0", recommendations)
        self.reco_text.config(state=tk.DISABLED)

    def _generate_aar_analysis(self, combat: CombatLog) -> str:
        """Génère l'analyse AAR"""
        analysis = f"""=== After Action Review ===

Résultat: {"VICTOIRE ✅" if combat.result == CombatResult.VICTORY else "DÉFAITE ❌"}

Points Forts:
"""
        if combat.result == CombatResult.VICTORY:
            analysis += f"• Combat remporté avec succès\n"
            if combat.total_damage_taken < 100:
                analysis += "• Dégâts subis minimaux - excellente défense\n"
            if combat.duration.total_seconds() < 180:
                analysis += "• Combat rapide - bonne efficacité\n"
            if combat.tactical_score >= 80:
                analysis += "• Excellentes décisions tactiques\n"

        analysis += "\nPoints Faibles:\n"
        if combat.total_damage_taken > 200:
            analysis += "• Trop de dégâts subis - améliorer le positionnement\n"
        if combat.duration.total_seconds() > 420:
            analysis += "• Combat trop long - optimiser la rotation de sorts\n"
        if combat.tactical_score < 70:
            analysis += "• Décisions tactiques sous-optimales\n"

        analysis += f"\nScore Global: {(combat.tactical_score + combat.efficiency_score) / 2:.1f}/100\n"

        return analysis

    def _generate_recommendations(self, combat: CombatLog) -> str:
        """Génère les recommandations"""
        reco = "💡 Recommandations pour améliorer:\n\n"

        if combat.total_damage_taken > 150:
            reco += "1. Améliorer le positionnement pour réduire les dégâts subis\n"
            reco += "   → Utiliser les obstacles et maintenir la distance\n\n"

        if combat.duration.total_seconds() > 300:
            reco += "2. Optimiser la rotation de sorts pour finir plus vite\n"
            reco += "   → Prioriser les sorts de dégâts élevés\n\n"

        if combat.tactical_score < 75:
            reco += "3. Revoir les décisions tactiques\n"
            reco += "   → Analyser les moments clés du combat\n\n"

        if combat.result == CombatResult.DEFEAT:
            reco += "4. Évaluer la difficulté des combats\n"
            reco += "   → Éviter les groupes trop difficiles\n"
            reco += "   → Améliorer l'équipement\n\n"

        reco += "💪 Continue tes efforts!"

        return reco

    # === Actions ===

    def apply_filters(self):
        """Applique les filtres"""
        self.update_combats_tree()
        self.update_quick_stats()

    def refresh_data(self):
        """Rafraîchit les données"""
        self.load_combat_data()
        self.update_combats_tree()
        self.update_quick_stats()
        self.update_global_stats()
        messagebox.showinfo("Rafraîchi", "Données rafraîchies")

    def generate_aar_report(self):
        """Génère un rapport AAR complet"""
        messagebox.showinfo("Rapport AAR", "Fonctionnalité à implémenter: génération rapport PDF")

    def export_data(self):
        """Exporte les données de combat"""
        messagebox.showinfo("Export", "Fonctionnalité à implémenter: export CSV/JSON")

    def show_trends(self):
        messagebox.showinfo("Tendances", "Fonctionnalité à implémenter: graphiques de tendances")

    def show_records(self):
        messagebox.showinfo("Records", "Fonctionnalité à implémenter: meilleurs records")

    def compare_combats(self):
        messagebox.showinfo("Comparaison", "Fonctionnalité à implémenter: comparaison de combats")

    def clean_history(self):
        if messagebox.askyesno("Confirmer", "Supprimer les combats de plus de 30 jours?"):
            messagebox.showinfo("Nettoyé", "Historique nettoyé - à implémenter")

    def replay_start(self):
        messagebox.showinfo("Replay", "Début du replay - à implémenter")

    def replay_prev(self):
        messagebox.showinfo("Replay", "Tour précédent - à implémenter")

    def replay_next(self):
        messagebox.showinfo("Replay", "Tour suivant - à implémenter")

    def replay_end(self):
        messagebox.showinfo("Replay", "Fin du replay - à implémenter")

    def replay_auto(self):
        messagebox.showinfo("Replay", "Replay automatique - à implémenter")

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
