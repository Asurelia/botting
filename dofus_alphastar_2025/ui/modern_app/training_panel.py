"""
Training Panel - Système d'entraînement et d'apprentissage avancé
Interface pour gérer les modèles, datasets et entraînements
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class TrainingSession:
    """Session d'entraînement d'un modèle"""
    session_id: str
    model_name: str
    start_time: datetime
    end_time: Optional[datetime]
    dataset_size: int
    epochs: int
    current_epoch: int
    loss: float
    accuracy: float
    status: str  # 'running', 'completed', 'failed', 'paused'
    metrics: Dict[str, Any]
    config: Dict[str, Any]


@dataclass
class Dataset:
    """Dataset pour l'entraînement"""
    name: str
    path: str
    size: int
    categories: List[str]
    created: datetime
    last_modified: datetime
    description: str


class TrainingPanel:
    """
    Panel d'entraînement et de machine learning

    Fonctionnalités:
    - Gestion des modèles (HRM, Vision, Combat, etc.)
    - Import/export de datasets
    - Configuration des hyperparamètres
    - Monitoring des entraînements en temps réel
    - Validation et benchmarking
    - Historique des sessions
    - Comparaison de modèles
    """

    def __init__(self, parent):
        self.parent = parent
        self.training_sessions: List[TrainingSession] = []
        self.datasets: List[Dataset] = []
        self.current_session: Optional[TrainingSession] = None

        # Charger les données existantes
        self.load_data()

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Toolbar supérieure ===
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(toolbar, text="🎓 Centre d'Entraînement & Machine Learning",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        ttk.Button(toolbar, text="🔄 Rafraîchir",
                  command=self.refresh_data).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="📊 Rapport complet",
                  command=self.generate_report).pack(side=tk.RIGHT, padx=2)

        # === PanedWindow pour layout flexible ===
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Panneau gauche: Navigation ===
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # Notebook pour différentes sections
        self.nav_notebook = ttk.Notebook(left_frame)
        self.nav_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Modèles ---
        models_tab = ttk.Frame(self.nav_notebook, padding=5)
        self.nav_notebook.add(models_tab, text="🧠 Modèles")

        ttk.Label(models_tab, text="Modèles disponibles:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Liste des modèles
        models_frame = ttk.Frame(models_tab)
        models_frame.pack(fill=tk.BOTH, expand=True)

        models_scroll = ttk.Scrollbar(models_frame)
        models_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.models_listbox = tk.Listbox(
            models_frame,
            yscrollcommand=models_scroll.set,
            font=("Consolas", 9)
        )
        self.models_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        models_scroll.config(command=self.models_listbox.yview)
        self.models_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        # Peupler la liste
        models = [
            "🧠 HRM (Hierarchical Reasoning Machine)",
            "👁️ Vision V2 (Object Detection)",
            "⚔️ Combat AI (Tactical)",
            "🗺️ Navigation (A* + Heuristics)",
            "💰 Economy (Prediction & Optimization)",
            "🎯 Action Predictor (Reinforcement Learning)",
            "📈 Pattern Recognition (ML)",
            "🔮 State Predictor (Temporal)"
        ]
        for model in models:
            self.models_listbox.insert(tk.END, model)

        # Actions sur les modèles
        models_actions = ttk.Frame(models_tab)
        models_actions.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(models_actions, text="🚀 Entraîner",
                  command=self.start_training).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(models_actions, text="📥 Charger",
                  command=self.load_model).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(models_actions, text="💾 Sauvegarder",
                  command=self.save_model).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Datasets ---
        datasets_tab = ttk.Frame(self.nav_notebook, padding=5)
        self.nav_notebook.add(datasets_tab, text="📦 Datasets")

        ttk.Label(datasets_tab, text="Datasets disponibles:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # TreeView pour datasets
        datasets_frame = ttk.Frame(datasets_tab)
        datasets_frame.pack(fill=tk.BOTH, expand=True)

        datasets_scroll = ttk.Scrollbar(datasets_frame)
        datasets_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.datasets_tree = ttk.Treeview(
            datasets_frame,
            columns=("size", "categories", "modified"),
            show="tree headings",
            yscrollcommand=datasets_scroll.set
        )
        self.datasets_tree.heading("size", text="Taille")
        self.datasets_tree.heading("categories", text="Catégories")
        self.datasets_tree.heading("modified", text="Modifié")
        self.datasets_tree.column("#0", width=150)
        self.datasets_tree.column("size", width=80)
        self.datasets_tree.column("categories", width=80)
        self.datasets_tree.column("modified", width=100)
        self.datasets_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        datasets_scroll.config(command=self.datasets_tree.yview)

        self.update_datasets_tree()

        # Actions sur datasets
        datasets_actions = ttk.Frame(datasets_tab)
        datasets_actions.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(datasets_actions, text="➕ Créer",
                  command=self.create_dataset).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(datasets_actions, text="📥 Importer",
                  command=self.import_dataset).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(datasets_actions, text="📤 Exporter",
                  command=self.export_dataset).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(datasets_actions, text="🗑️ Supprimer",
                  command=self.delete_dataset).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Onglet Historique ---
        history_tab = ttk.Frame(self.nav_notebook, padding=5)
        self.nav_notebook.add(history_tab, text="📜 Historique")

        ttk.Label(history_tab, text="Sessions d'entraînement:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # TreeView pour historique
        history_frame = ttk.Frame(history_tab)
        history_frame.pack(fill=tk.BOTH, expand=True)

        history_scroll = ttk.Scrollbar(history_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("model", "date", "epochs", "accuracy", "status"),
            show="tree headings",
            yscrollcommand=history_scroll.set
        )
        self.history_tree.heading("model", text="Modèle")
        self.history_tree.heading("date", text="Date")
        self.history_tree.heading("epochs", text="Epochs")
        self.history_tree.heading("accuracy", text="Accuracy")
        self.history_tree.heading("status", text="Statut")
        self.history_tree.column("#0", width=30)
        self.history_tree.column("model", width=150)
        self.history_tree.column("date", width=120)
        self.history_tree.column("epochs", width=60)
        self.history_tree.column("accuracy", width=80)
        self.history_tree.column("status", width=80)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_tree.yview)
        self.history_tree.bind("<<TreeviewSelect>>", self.on_session_select)

        self.update_history_tree()

        # Actions historique
        history_actions = ttk.Frame(history_tab)
        history_actions.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(history_actions, text="📊 Voir détails",
                  command=self.view_session_details).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(history_actions, text="🔄 Ré-entraîner",
                  command=self.retrain_session).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(history_actions, text="🗑️ Supprimer",
                  command=self.delete_session).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # === Panneau central: Configuration & Entraînement ===
        center_frame = ttk.LabelFrame(paned, text="⚙️ Configuration & Entraînement", padding=10)
        paned.add(center_frame, weight=2)

        # Notebook pour organiser les options
        self.config_notebook = ttk.Notebook(center_frame)
        self.config_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Configuration ---
        config_tab = ttk.Frame(self.config_notebook, padding=10)
        self.config_notebook.add(config_tab, text="⚙️ Configuration")

        # Modèle sélectionné
        model_frame = ttk.LabelFrame(config_tab, text="Modèle", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Modèle:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="HRM (Hierarchical Reasoning Machine)")
        ttk.Entry(model_frame, textvariable=self.model_var, state="readonly",
                 width=50).grid(row=0, column=1, sticky=tk.EW, pady=5)

        # Hyperparamètres
        hyper_frame = ttk.LabelFrame(config_tab, text="Hyperparamètres", padding=10)
        hyper_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(hyper_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(hyper_frame, from_=1, to=1000, textvariable=self.epochs_var,
                   width=20).grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(hyper_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(hyper_frame, from_=1, to=512, textvariable=self.batch_size_var,
                   width=20).grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(hyper_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(hyper_frame, textvariable=self.lr_var,
                 width=20).grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Label(hyper_frame, text="Optimizer:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.optimizer_var = tk.StringVar(value="Adam")
        ttk.Combobox(hyper_frame, textvariable=self.optimizer_var,
                    values=["Adam", "SGD", "RMSprop", "AdaGrad"],
                    state="readonly", width=18).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Dataset
        dataset_frame = ttk.LabelFrame(config_tab, text="Dataset", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar(value="dataset_default")
        dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                    state="readonly", width=40)
        dataset_combo.grid(row=0, column=1, sticky=tk.EW, pady=5)
        dataset_combo['values'] = self._get_dataset_names()

        ttk.Label(dataset_frame, text="Train/Val Split:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.split_var = tk.DoubleVar(value=0.8)
        ttk.Scale(dataset_frame, from_=0.5, to=0.95, variable=self.split_var,
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.EW, pady=5)

        split_label = ttk.Label(dataset_frame, text="80%")
        split_label.grid(row=1, column=2, padx=(5, 0))
        self.split_var.trace_add("write", lambda *args: split_label.config(
            text=f"{int(self.split_var.get() * 100)}%"))

        # Options avancées
        advanced_frame = ttk.LabelFrame(config_tab, text="Options avancées", padding=10)
        advanced_frame.pack(fill=tk.X, pady=(0, 10))

        self.use_augmentation = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Data Augmentation",
                       variable=self.use_augmentation).pack(anchor=tk.W, pady=2)

        self.use_early_stopping = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Early Stopping",
                       variable=self.use_early_stopping).pack(anchor=tk.W, pady=2)

        self.use_checkpoints = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Sauvegarder checkpoints",
                       variable=self.use_checkpoints).pack(anchor=tk.W, pady=2)

        self.use_tensorboard = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Activer TensorBoard",
                       variable=self.use_tensorboard).pack(anchor=tk.W, pady=2)

        # Boutons de contrôle
        control_frame = ttk.Frame(config_tab)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(control_frame, text="🚀 Démarrer Entraînement",
                  command=self.start_training_with_config,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(control_frame, text="💾 Sauvegarder Config",
                  command=self.save_config).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(control_frame, text="📥 Charger Config",
                  command=self.load_config).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # --- Onglet Entraînement en cours ---
        training_tab = ttk.Frame(self.config_notebook, padding=10)
        self.config_notebook.add(training_tab, text="🚀 Entraînement")

        # Infos session
        session_frame = ttk.LabelFrame(training_tab, text="Session en cours", padding=10)
        session_frame.pack(fill=tk.X, pady=(0, 10))

        self.session_info_text = tk.Text(session_frame, height=4, font=("Consolas", 9),
                                        state=tk.DISABLED, wrap=tk.WORD)
        self.session_info_text.pack(fill=tk.BOTH, expand=True)

        # Barre de progression
        progress_frame = ttk.LabelFrame(training_tab, text="Progression", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = ttk.Label(progress_frame, text="Aucun entraînement en cours")
        self.progress_label.pack()

        # Métriques en temps réel
        metrics_frame = ttk.LabelFrame(training_tab, text="Métriques temps réel", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Utiliser un Canvas pour les graphiques (simulé avec Text pour l'instant)
        self.metrics_text = tk.Text(metrics_frame, height=15, font=("Consolas", 9),
                                   state=tk.DISABLED, wrap=tk.NONE)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Contrôles entraînement
        train_control_frame = ttk.Frame(training_tab)
        train_control_frame.pack(fill=tk.X)

        ttk.Button(train_control_frame, text="⏸️ Pause",
                  command=self.pause_training).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(train_control_frame, text="▶️ Reprendre",
                  command=self.resume_training).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(train_control_frame, text="⏹️ Arrêter",
                  command=self.stop_training).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # --- Onglet Validation ---
        validation_tab = ttk.Frame(self.config_notebook, padding=10)
        self.config_notebook.add(validation_tab, text="✅ Validation")

        ttk.Label(validation_tab, text="Outils de validation et benchmarking",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Actions de validation
        ttk.Button(validation_tab, text="🧪 Validation croisée (K-Fold)",
                  command=self.run_cross_validation).pack(fill=tk.X, pady=5)
        ttk.Button(validation_tab, text="📊 Benchmark sur dataset test",
                  command=self.run_benchmark).pack(fill=tk.X, pady=5)
        ttk.Button(validation_tab, text="🔍 Analyse des erreurs",
                  command=self.analyze_errors).pack(fill=tk.X, pady=5)
        ttk.Button(validation_tab, text="📈 Matrice de confusion",
                  command=self.show_confusion_matrix).pack(fill=tk.X, pady=5)
        ttk.Button(validation_tab, text="🎯 Courbes ROC/AUC",
                  command=self.show_roc_curves).pack(fill=tk.X, pady=5)

        ttk.Separator(validation_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        ttk.Label(validation_tab, text="Résultats de validation:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.validation_text = tk.Text(validation_tab, height=15, font=("Consolas", 9),
                                      state=tk.DISABLED, wrap=tk.WORD)
        self.validation_text.pack(fill=tk.BOTH, expand=True)

        # === Panneau droit: Comparaison & Statistiques ===
        right_frame = ttk.LabelFrame(paned, text="📊 Comparaison & Stats", padding=10)
        paned.add(right_frame, weight=1)

        # Comparaison de modèles
        comparison_frame = ttk.LabelFrame(right_frame, text="Comparaison", padding=10)
        comparison_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(comparison_frame, text="Sélectionner 2+ modèles à comparer:",
                 font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(0, 5))

        # Liste de sélection multiple
        compare_list_frame = ttk.Frame(comparison_frame)
        compare_list_frame.pack(fill=tk.BOTH, expand=True)

        compare_scroll = ttk.Scrollbar(compare_list_frame)
        compare_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.compare_listbox = tk.Listbox(
            compare_list_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=compare_scroll.set,
            font=("Consolas", 8),
            height=8
        )
        self.compare_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        compare_scroll.config(command=self.compare_listbox.yview)

        ttk.Button(comparison_frame, text="🔄 Comparer sélection",
                  command=self.compare_models).pack(fill=tk.X, pady=(5, 0))

        # Statistiques globales
        stats_frame = ttk.LabelFrame(right_frame, text="Statistiques globales", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Text(stats_frame, height=15, font=("Consolas", 8),
                                 state=tk.DISABLED, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        self.update_global_stats()

    def load_data(self):
        """Charge les données existantes (datasets, sessions)"""
        # TODO: Charger depuis fichiers JSON
        # Pour l'instant, créer des données de démo
        self.datasets = [
            Dataset(
                name="dataset_combat_v1",
                path="data/training/combat_v1",
                size=5000,
                categories=["attack", "defend", "move", "spell"],
                created=datetime.now() - timedelta(days=30),
                last_modified=datetime.now() - timedelta(days=5),
                description="Dataset de situations de combat"
            ),
            Dataset(
                name="dataset_navigation_v2",
                path="data/training/navigation_v2",
                size=8000,
                categories=["move", "pathfinding", "obstacle"],
                created=datetime.now() - timedelta(days=60),
                last_modified=datetime.now() - timedelta(days=10),
                description="Dataset de navigation et pathfinding"
            ),
            Dataset(
                name="dataset_vision_objects",
                path="data/training/vision_objects",
                size=15000,
                categories=["monster", "npc", "resource", "ui_element"],
                created=datetime.now() - timedelta(days=90),
                last_modified=datetime.now() - timedelta(days=2),
                description="Dataset d'objets détectés par vision"
            )
        ]

        # Sessions d'entraînement démo
        self.training_sessions = [
            TrainingSession(
                session_id="sess_001",
                model_name="HRM",
                start_time=datetime.now() - timedelta(hours=5),
                end_time=datetime.now() - timedelta(hours=3),
                dataset_size=5000,
                epochs=100,
                current_epoch=100,
                loss=0.023,
                accuracy=0.967,
                status="completed",
                metrics={"val_loss": 0.031, "val_accuracy": 0.951},
                config={}
            ),
            TrainingSession(
                session_id="sess_002",
                model_name="Vision V2",
                start_time=datetime.now() - timedelta(hours=2),
                end_time=datetime.now() - timedelta(hours=1),
                dataset_size=15000,
                epochs=50,
                current_epoch=50,
                loss=0.089,
                accuracy=0.923,
                status="completed",
                metrics={"val_loss": 0.102, "val_accuracy": 0.911},
                config={}
            )
        ]

    def update_datasets_tree(self):
        """Met à jour l'arbre des datasets"""
        self.datasets_tree.delete(*self.datasets_tree.get_children())

        for dataset in self.datasets:
            size_str = f"{dataset.size} ex."
            cat_str = str(len(dataset.categories))
            modified_str = dataset.last_modified.strftime("%Y-%m-%d")

            self.datasets_tree.insert("", tk.END, text=dataset.name,
                                     values=(size_str, cat_str, modified_str))

    def update_history_tree(self):
        """Met à jour l'arbre d'historique"""
        self.history_tree.delete(*self.history_tree.get_children())

        for i, session in enumerate(reversed(self.training_sessions)):
            date_str = session.start_time.strftime("%Y-%m-%d %H:%M")
            accuracy_str = f"{session.accuracy:.2%}"
            status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄", "paused": "⏸️"}

            self.history_tree.insert("", tk.END, text=str(len(self.training_sessions) - i),
                                    values=(session.model_name, date_str,
                                           session.current_epoch, accuracy_str,
                                           f"{status_emoji.get(session.status, '?')} {session.status}"))

    def update_global_stats(self):
        """Met à jour les statistiques globales"""
        total_sessions = len(self.training_sessions)
        completed_sessions = sum(1 for s in self.training_sessions if s.status == "completed")
        total_datasets = len(self.datasets)
        total_examples = sum(d.size for d in self.datasets)

        if completed_sessions > 0:
            avg_accuracy = sum(s.accuracy for s in self.training_sessions
                             if s.status == "completed") / completed_sessions
        else:
            avg_accuracy = 0.0

        stats = f"""Statistiques globales
{"=" * 30}

Sessions d'entraînement:
  Total: {total_sessions}
  Terminées: {completed_sessions}
  Taux de succès: {completed_sessions/total_sessions*100:.1f}%

Datasets:
  Total: {total_datasets}
  Exemples totaux: {total_examples:,}

Performance moyenne:
  Accuracy: {avg_accuracy:.2%}

Meilleur modèle:
  {"HRM - 96.7% accuracy" if self.training_sessions else "Aucun"}
"""

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats)
        self.stats_text.config(state=tk.DISABLED)

    def _get_dataset_names(self) -> List[str]:
        """Retourne les noms des datasets"""
        return [d.name for d in self.datasets]

    # === Actions ===

    def on_model_select(self, event):
        """Gère la sélection d'un modèle"""
        selection = self.models_listbox.curselection()
        if selection:
            model = self.models_listbox.get(selection[0])
            # Retirer l'emoji
            model_name = model.split(" ", 1)[1] if " " in model else model
            self.model_var.set(model_name)

    def on_session_select(self, event):
        """Gère la sélection d'une session"""
        # TODO: Afficher les détails de la session
        pass

    def start_training(self):
        """Démarre un entraînement"""
        messagebox.showinfo("Entraînement", "Fonctionnalité à implémenter: démarrer entraînement")

    def start_training_with_config(self):
        """Démarre un entraînement avec la config actuelle"""
        config = {
            "model": self.model_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "learning_rate": self.lr_var.get(),
            "optimizer": self.optimizer_var.get(),
            "dataset": self.dataset_var.get(),
            "train_val_split": self.split_var.get(),
            "use_augmentation": self.use_augmentation.get(),
            "use_early_stopping": self.use_early_stopping.get(),
            "use_checkpoints": self.use_checkpoints.get(),
            "use_tensorboard": self.use_tensorboard.get()
        }

        message = f"Démarrer entraînement avec la configuration suivante?\n\n"
        message += f"Modèle: {config['model']}\n"
        message += f"Epochs: {config['epochs']}\n"
        message += f"Dataset: {config['dataset']}\n"

        if messagebox.askyesno("Confirmer", message):
            # TODO: Lancer l'entraînement réel
            messagebox.showinfo("Lancé", "Entraînement lancé! (à implémenter)")
            self.config_notebook.select(1)  # Basculer sur l'onglet Entraînement

    def pause_training(self):
        messagebox.showinfo("Pause", "Entraînement mis en pause - à implémenter")

    def resume_training(self):
        messagebox.showinfo("Reprise", "Entraînement repris - à implémenter")

    def stop_training(self):
        if messagebox.askyesno("Confirmer", "Arrêter l'entraînement en cours?"):
            messagebox.showinfo("Arrêté", "Entraînement arrêté - à implémenter")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Charger un modèle",
            filetypes=[("Model files", "*.pkl *.h5 *.pth"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("Chargé", f"Modèle chargé: {file_path}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder le modèle",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl"), ("HDF5", "*.h5"), ("PyTorch", "*.pth")]
        )
        if file_path:
            messagebox.showinfo("Sauvegardé", f"Modèle sauvegardé: {file_path}")

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder la configuration",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if file_path:
            config = {
                "model": self.model_var.get(),
                "epochs": self.epochs_var.get(),
                "batch_size": self.batch_size_var.get(),
                "learning_rate": self.lr_var.get(),
                "optimizer": self.optimizer_var.get(),
                "dataset": self.dataset_var.get(),
                "train_val_split": self.split_var.get()
            }
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Sauvegardé", f"Configuration sauvegardée: {file_path}")

    def load_config(self):
        file_path = filedialog.askopenfilename(
            title="Charger une configuration",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'r') as f:
                config = json.load(f)
            self.model_var.set(config.get("model", ""))
            self.epochs_var.set(config.get("epochs", 100))
            self.batch_size_var.set(config.get("batch_size", 32))
            self.lr_var.set(config.get("learning_rate", 0.001))
            self.optimizer_var.set(config.get("optimizer", "Adam"))
            self.dataset_var.set(config.get("dataset", ""))
            self.split_var.set(config.get("train_val_split", 0.8))
            messagebox.showinfo("Chargé", f"Configuration chargée: {file_path}")

    def create_dataset(self):
        messagebox.showinfo("Créer", "Fonctionnalité à implémenter: créer dataset")

    def import_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Importer un dataset",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            messagebox.showinfo("Importé", f"Dataset importé: {file_path}")
            self.update_datasets_tree()

    def export_dataset(self):
        messagebox.showinfo("Exporter", "Fonctionnalité à implémenter: exporter dataset")

    def delete_dataset(self):
        if messagebox.askyesno("Confirmer", "Supprimer le dataset sélectionné?"):
            messagebox.showinfo("Supprimé", "Dataset supprimé - à implémenter")
            self.update_datasets_tree()

    def view_session_details(self):
        messagebox.showinfo("Détails", "Fonctionnalité à implémenter: voir détails session")

    def retrain_session(self):
        messagebox.showinfo("Ré-entraîner", "Fonctionnalité à implémenter: ré-entraîner session")

    def delete_session(self):
        if messagebox.askyesno("Confirmer", "Supprimer la session sélectionnée?"):
            messagebox.showinfo("Supprimé", "Session supprimée - à implémenter")
            self.update_history_tree()

    def run_cross_validation(self):
        messagebox.showinfo("Validation croisée", "Fonctionnalité à implémenter: validation croisée K-Fold")

    def run_benchmark(self):
        messagebox.showinfo("Benchmark", "Fonctionnalité à implémenter: benchmark sur dataset test")

    def analyze_errors(self):
        messagebox.showinfo("Analyse", "Fonctionnalité à implémenter: analyse des erreurs")

    def show_confusion_matrix(self):
        messagebox.showinfo("Matrice", "Fonctionnalité à implémenter: matrice de confusion")

    def show_roc_curves(self):
        messagebox.showinfo("ROC", "Fonctionnalité à implémenter: courbes ROC/AUC")

    def compare_models(self):
        selection = self.compare_listbox.curselection()
        if len(selection) < 2:
            messagebox.showwarning("Sélection", "Veuillez sélectionner au moins 2 modèles")
            return
        messagebox.showinfo("Comparaison", f"Comparaison de {len(selection)} modèles - à implémenter")

    def refresh_data(self):
        """Rafraîchit toutes les données"""
        self.load_data()
        self.update_datasets_tree()
        self.update_history_tree()
        self.update_global_stats()
        messagebox.showinfo("Rafraîchi", "Données rafraîchies")

    def generate_report(self):
        """Génère un rapport complet"""
        messagebox.showinfo("Rapport", "Fonctionnalité à implémenter: générer rapport complet PDF")

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
