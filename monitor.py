#!/usr/bin/env python3
"""
Interface de monitoring et dashboard pour le bot DOFUS.

Ce module fournit une interface de surveillance en temps réel pour monitorer :
- Performances et statistiques du bot
- État des modules et processus
- Métriques système et ressources
- Logs et événements en temps réel
- Alertes et notifications

Usage:
    python monitor.py --dashboard
    python monitor.py --web-server --port 8080
    python monitor.py --export-report --format json
    python monitor.py --alerts --config alerts.json
"""

import sys
import os
import json
import argparse
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import sqlite3

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

# Imports GUI
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Interface graphique non disponible - mode CLI uniquement")

# Imports Web (optionnel)
try:
    from flask import Flask, render_template_string, jsonify, request
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("Serveur web non disponible")

# Imports système
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible - métriques système limitées")

from engine.event_bus import EventBus
from state.realtime_state import RealtimeState


@dataclass
class SystemMetrics:
    """Métriques système."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: float = 0.0  # MB
    memory_total: float = 0.0  # MB
    disk_usage: float = 0.0
    network_sent: int = 0  # bytes
    network_recv: int = 0  # bytes
    processes_count: int = 0
    bot_process_cpu: float = 0.0
    bot_process_memory: float = 0.0


@dataclass
class BotMetrics:
    """Métriques du bot."""
    timestamp: float = field(default_factory=time.time)
    status: str = "stopped"  # stopped, running, paused, error
    uptime: float = 0.0  # secondes
    actions_count: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    
    # Modules
    active_modules: List[str] = field(default_factory=list)
    module_performance: Dict[str, float] = field(default_factory=dict)
    
    # Performances
    actions_per_minute: float = 0.0
    success_rate: float = 100.0
    average_response_time: float = 0.0
    
    # Ressources
    character_level: int = 1
    experience_gained: int = 0
    items_collected: int = 0
    kamas_earned: int = 0


@dataclass
class AlertConfig:
    """Configuration d'alerte."""
    name: str
    condition: str  # Condition Python évaluable
    message: str
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    cooldown: int = 300  # secondes avant re-notification
    last_triggered: float = 0.0


class MetricsCollector:
    """Collecteur de métriques système et bot."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("logs/metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        self._init_database()
        
        # Cache des dernières métriques
        self.current_system_metrics = SystemMetrics()
        self.current_bot_metrics = BotMetrics()
        
        # Historique en mémoire (pour les graphiques temps réel)
        self.system_history = deque(maxlen=1000)  # ~16 minutes à 1 mesure/seconde
        self.bot_history = deque(maxlen=1000)
        
        # État de collecte
        self.collecting = False
        self.collection_interval = 1.0  # secondes
        
    def _init_database(self):
        """Initialiser la base de données des métriques."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        timestamp REAL PRIMARY KEY,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_used REAL,
                        memory_total REAL,
                        disk_usage REAL,
                        network_sent INTEGER,
                        network_recv INTEGER,
                        processes_count INTEGER,
                        bot_process_cpu REAL,
                        bot_process_memory REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bot_metrics (
                        timestamp REAL PRIMARY KEY,
                        status TEXT,
                        uptime REAL,
                        actions_count INTEGER,
                        errors_count INTEGER,
                        warnings_count INTEGER,
                        active_modules TEXT,
                        actions_per_minute REAL,
                        success_rate REAL,
                        average_response_time REAL,
                        character_level INTEGER,
                        experience_gained INTEGER,
                        items_collected INTEGER,
                        kamas_earned INTEGER
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        timestamp REAL,
                        type TEXT,
                        source TEXT,
                        message TEXT,
                        data TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la base : {e}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collecter les métriques système."""
        metrics = SystemMetrics()
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU et mémoire
                metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                metrics.memory_percent = memory.percent
                metrics.memory_used = memory.used / 1024 / 1024  # MB
                metrics.memory_total = memory.total / 1024 / 1024  # MB
                
                # Disque
                disk = psutil.disk_usage('/')
                metrics.disk_usage = disk.percent
                
                # Réseau
                network = psutil.net_io_counters()
                metrics.network_sent = network.bytes_sent
                metrics.network_recv = network.bytes_recv
                
                # Processus
                metrics.processes_count = len(psutil.pids())
                
                # Processus du bot (si détectable)
                current_process = psutil.Process()
                metrics.bot_process_cpu = current_process.cpu_percent()
                metrics.bot_process_memory = current_process.memory_info().rss / 1024 / 1024  # MB
                
            except Exception as e:
                self.logger.warning(f"Erreur lors de la collecte des métriques système : {e}")
        
        self.current_system_metrics = metrics
        return metrics
    
    def collect_bot_metrics(self, realtime_state: RealtimeState = None) -> BotMetrics:
        """Collecter les métriques du bot."""
        metrics = BotMetrics()
        
        if realtime_state:
            try:
                state = realtime_state.get_current_state()
                
                # État général
                metrics.status = state.get("bot_status", "unknown")
                metrics.uptime = state.get("uptime", 0.0)
                
                # Compteurs
                metrics.actions_count = state.get("total_actions", 0)
                metrics.errors_count = state.get("total_errors", 0)
                metrics.warnings_count = state.get("total_warnings", 0)
                
                # Modules actifs
                metrics.active_modules = state.get("active_modules", [])
                metrics.module_performance = state.get("module_performance", {})
                
                # Performances
                if metrics.uptime > 0:
                    metrics.actions_per_minute = (metrics.actions_count / metrics.uptime) * 60
                
                total_attempts = state.get("total_attempts", metrics.actions_count)
                if total_attempts > 0:
                    metrics.success_rate = ((total_attempts - metrics.errors_count) / total_attempts) * 100
                
                metrics.average_response_time = state.get("avg_response_time", 0.0)
                
                # Ressources du personnage
                character = state.get("character", {})
                metrics.character_level = character.get("level", 1)
                metrics.experience_gained = character.get("experience_gained", 0)
                metrics.items_collected = character.get("items_collected", 0)
                metrics.kamas_earned = character.get("kamas_earned", 0)
                
            except Exception as e:
                self.logger.warning(f"Erreur lors de la collecte des métriques bot : {e}")
        
        self.current_bot_metrics = metrics
        return metrics
    
    def store_metrics(self, system_metrics: SystemMetrics = None, bot_metrics: BotMetrics = None):
        """Stocker les métriques en base."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if system_metrics:
                    conn.execute("""
                        INSERT OR REPLACE INTO system_metrics VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        system_metrics.timestamp,
                        system_metrics.cpu_percent,
                        system_metrics.memory_percent,
                        system_metrics.memory_used,
                        system_metrics.memory_total,
                        system_metrics.disk_usage,
                        system_metrics.network_sent,
                        system_metrics.network_recv,
                        system_metrics.processes_count,
                        system_metrics.bot_process_cpu,
                        system_metrics.bot_process_memory
                    ))
                
                if bot_metrics:
                    conn.execute("""
                        INSERT OR REPLACE INTO bot_metrics VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        bot_metrics.timestamp,
                        bot_metrics.status,
                        bot_metrics.uptime,
                        bot_metrics.actions_count,
                        bot_metrics.errors_count,
                        bot_metrics.warnings_count,
                        json.dumps(bot_metrics.active_modules),
                        bot_metrics.actions_per_minute,
                        bot_metrics.success_rate,
                        bot_metrics.average_response_time,
                        bot_metrics.character_level,
                        bot_metrics.experience_gained,
                        bot_metrics.items_collected,
                        bot_metrics.kamas_earned
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage des métriques : {e}")
    
    def log_event(self, event_type: str, source: str, message: str, data: Dict = None):
        """Enregistrer un événement."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events VALUES (?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    event_type,
                    source,
                    message,
                    json.dumps(data) if data else None
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement d'événement : {e}")
    
    def get_historical_data(self, table: str, hours: int = 24) -> List[Dict]:
        """Récupérer les données historiques."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                since = time.time() - (hours * 3600)
                cursor.execute(f"""
                    SELECT * FROM {table} 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, (since,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données : {e}")
            return []
    
    async def start_collection(self, realtime_state: RealtimeState = None):
        """Démarrer la collecte de métriques."""
        self.collecting = True
        self.logger.info("Démarrage de la collecte de métriques")
        
        while self.collecting:
            try:
                # Collecter les métriques
                system_metrics = self.collect_system_metrics()
                bot_metrics = self.collect_bot_metrics(realtime_state)
                
                # Ajouter à l'historique en mémoire
                self.system_history.append(system_metrics)
                self.bot_history.append(bot_metrics)
                
                # Stocker en base (moins fréquemment)
                if len(self.system_history) % 10 == 0:  # Toutes les 10 secondes
                    self.store_metrics(system_metrics, bot_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de collecte : {e}")
                await asyncio.sleep(5)  # Attendre plus longtemps en cas d'erreur
    
    def stop_collection(self):
        """Arrêter la collecte de métriques."""
        self.collecting = False
        self.logger.info("Arrêt de la collecte de métriques")


class AlertManager:
    """Gestionnaire d'alertes."""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/alerts.json")
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        
        self.alerts: Dict[str, AlertConfig] = {}
        self.active_alerts: List[Tuple[str, str, datetime]] = []  # (name, message, timestamp)
        
        self.load_alerts_config()
        
    def load_alerts_config(self):
        """Charger la configuration des alertes."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for alert_name, alert_data in data.items():
                    self.alerts[alert_name] = AlertConfig(**alert_data)
                    
                self.logger.info(f"{len(self.alerts)} alertes chargées")
                
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement des alertes : {e}")
        else:
            # Créer des alertes par défaut
            self._create_default_alerts()
    
    def _create_default_alerts(self):
        """Créer des alertes par défaut."""
        default_alerts = {
            "high_cpu": AlertConfig(
                name="high_cpu",
                condition="system_metrics.cpu_percent > 80",
                message="CPU élevé: {cpu_percent:.1f}%",
                severity="warning"
            ),
            "high_memory": AlertConfig(
                name="high_memory", 
                condition="system_metrics.memory_percent > 90",
                message="Mémoire élevée: {memory_percent:.1f}%",
                severity="error"
            ),
            "bot_stopped": AlertConfig(
                name="bot_stopped",
                condition="bot_metrics.status == 'stopped' and bot_metrics.uptime > 0",
                message="Bot arrêté inopinément",
                severity="critical"
            ),
            "low_success_rate": AlertConfig(
                name="low_success_rate",
                condition="bot_metrics.success_rate < 50 and bot_metrics.actions_count > 100",
                message="Taux de succès faible: {success_rate:.1f}%",
                severity="warning"
            ),
            "no_actions": AlertConfig(
                name="no_actions",
                condition="bot_metrics.uptime > 300 and bot_metrics.actions_per_minute < 1",
                message="Aucune action détectée depuis 5 minutes",
                severity="error"
            )
        }
        
        self.alerts.update(default_alerts)
        self.save_alerts_config()
    
    def save_alerts_config(self):
        """Sauvegarder la configuration des alertes."""
        try:
            data = {}
            for name, alert in self.alerts.items():
                data[name] = asdict(alert)
                
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des alertes : {e}")
    
    def check_alerts(self, system_metrics: SystemMetrics, bot_metrics: BotMetrics):
        """Vérifier les conditions d'alerte."""
        current_time = time.time()
        
        for alert_name, alert_config in self.alerts.items():
            if not alert_config.enabled:
                continue
                
            # Vérifier le cooldown
            if current_time - alert_config.last_triggered < alert_config.cooldown:
                continue
            
            try:
                # Évaluer la condition
                context = {
                    "system_metrics": system_metrics,
                    "bot_metrics": bot_metrics,
                    "current_time": current_time
                }
                
                if eval(alert_config.condition, {"__builtins__": {}}, context):
                    # Alerte déclenchée
                    message = alert_config.message.format(
                        cpu_percent=system_metrics.cpu_percent,
                        memory_percent=system_metrics.memory_percent,
                        success_rate=bot_metrics.success_rate,
                        actions_per_minute=bot_metrics.actions_per_minute,
                        **asdict(system_metrics),
                        **asdict(bot_metrics)
                    )
                    
                    self.trigger_alert(alert_name, message, alert_config.severity)
                    alert_config.last_triggered = current_time
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de l'évaluation de l'alerte '{alert_name}' : {e}")
    
    def trigger_alert(self, alert_name: str, message: str, severity: str):
        """Déclencher une alerte."""
        self.logger.warning(f"ALERTE [{severity.upper()}] {alert_name}: {message}")
        
        # Ajouter à la liste des alertes actives
        self.active_alerts.append((alert_name, message, datetime.now()))
        
        # Limiter l'historique des alertes
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
    
    def get_active_alerts(self, last_minutes: int = 60) -> List[Tuple[str, str, datetime]]:
        """Récupérer les alertes récentes."""
        cutoff = datetime.now() - timedelta(minutes=last_minutes)
        return [(name, msg, ts) for name, msg, ts in self.active_alerts if ts > cutoff]


class MonitorDashboard:
    """Dashboard de monitoring avec interface graphique."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        if not GUI_AVAILABLE:
            raise ImportError("Interface graphique non disponible")
            
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        self.root = tk.Tk()
        self.root.title("Bot DOFUS - Monitoring Dashboard")
        self.root.geometry("1400x900")
        
        # Variables pour les graphiques
        self.update_interval = 2000  # ms
        self.animation_active = False
        
        self.create_interface()
        
    def create_interface(self):
        """Créer l'interface du dashboard."""
        # Notebook avec onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Onglet Vue d'ensemble
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Vue d'ensemble")
        
        # Onglet Métriques système
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="Système")
        
        # Onglet Métriques bot
        bot_frame = ttk.Frame(notebook)
        notebook.add(bot_frame, text="Bot")
        
        # Onglet Alertes
        alerts_frame = ttk.Frame(notebook)
        notebook.add(alerts_frame, text="Alertes")
        
        # Onglet Logs
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Créer le contenu de chaque onglet
        self.create_overview_tab(overview_frame)
        self.create_system_tab(system_frame)
        self.create_bot_tab(bot_frame)
        self.create_alerts_tab(alerts_frame)
        self.create_logs_tab(logs_frame)
        
        # Démarrer les mises à jour automatiques
        self.start_auto_update()
        
    def create_overview_tab(self, parent):
        """Créer l'onglet vue d'ensemble."""
        # Frame pour les indicateurs principaux
        indicators_frame = ttk.LabelFrame(parent, text="Indicateurs Principaux", padding="10")
        indicators_frame.pack(fill="x", pady=(0, 10))
        
        # Créer une grille d'indicateurs
        indicators = [
            ("Status Bot", "bot_status"),
            ("Uptime", "uptime"),
            ("Actions/min", "actions_per_minute"),
            ("Taux Succès", "success_rate"),
            ("CPU", "cpu_percent"),
            ("Mémoire", "memory_percent"),
            ("Niveau", "character_level"),
            ("Kamas", "kamas_earned")
        ]
        
        self.indicator_labels = {}
        
        for i, (name, key) in enumerate(indicators):
            row = i // 4
            col = (i % 4) * 2
            
            ttk.Label(indicators_frame, text=f"{name}:", 
                     font=("Arial", 10, "bold")).grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
            label = ttk.Label(indicators_frame, text="--", 
                            font=("Arial", 12), foreground="blue")
            label.grid(row=row, column=col+1, padx=5, pady=5, sticky="w")
            self.indicator_labels[key] = label
        
        # Frame pour les graphiques en temps réel
        charts_frame = ttk.LabelFrame(parent, text="Graphiques Temps Réel", padding="10")
        charts_frame.pack(fill="both", expand=True)
        
        # Créer les graphiques
        self.create_overview_charts(charts_frame)
        
    def create_overview_charts(self, parent):
        """Créer les graphiques de vue d'ensemble."""
        # Figure matplotlib
        self.overview_fig = Figure(figsize=(12, 6), dpi=100)
        
        # Sous-graphiques
        self.cpu_ax = self.overview_fig.add_subplot(2, 2, 1)
        self.memory_ax = self.overview_fig.add_subplot(2, 2, 2)
        self.actions_ax = self.overview_fig.add_subplot(2, 2, 3)
        self.success_ax = self.overview_fig.add_subplot(2, 2, 4)
        
        # Configuration des axes
        self.cpu_ax.set_title("CPU (%)")
        self.cpu_ax.set_ylim(0, 100)
        
        self.memory_ax.set_title("Mémoire (%)")
        self.memory_ax.set_ylim(0, 100)
        
        self.actions_ax.set_title("Actions/min")
        self.actions_ax.set_ylim(0, 50)
        
        self.success_ax.set_title("Taux de Succès (%)")
        self.success_ax.set_ylim(0, 100)
        
        # Canvas Tkinter
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, parent)
        self.overview_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def create_system_tab(self, parent):
        """Créer l'onglet métriques système."""
        # Informations système détaillées
        info_frame = ttk.LabelFrame(parent, text="Informations Système", padding="10")
        info_frame.pack(fill="x", pady=(0, 10))
        
        self.system_info_text = tk.Text(info_frame, height=8, state="disabled")
        self.system_info_text.pack(fill="both", expand=True)
        
        # Graphiques système détaillés
        system_charts_frame = ttk.LabelFrame(parent, text="Graphiques Système", padding="10")
        system_charts_frame.pack(fill="both", expand=True)
        
        self.system_fig = Figure(figsize=(12, 8), dpi=100)
        self.system_canvas = FigureCanvasTkAgg(self.system_fig, system_charts_frame)
        self.system_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def create_bot_tab(self, parent):
        """Créer l'onglet métriques bot."""
        # Statistiques du bot
        stats_frame = ttk.LabelFrame(parent, text="Statistiques du Bot", padding="10")
        stats_frame.pack(fill="x", pady=(0, 10))
        
        self.bot_stats_text = tk.Text(stats_frame, height=10, state="disabled")
        self.bot_stats_text.pack(fill="both", expand=True)
        
        # Modules actifs
        modules_frame = ttk.LabelFrame(parent, text="Modules Actifs", padding="10")
        modules_frame.pack(fill="both", expand=True)
        
        self.modules_tree = ttk.Treeview(modules_frame, columns=("Status", "Performance", "Actions"), show="tree headings")
        self.modules_tree.heading("#0", text="Module")
        self.modules_tree.heading("Status", text="Status")
        self.modules_tree.heading("Performance", text="Performance")
        self.modules_tree.heading("Actions", text="Actions")
        
        modules_scrollbar = ttk.Scrollbar(modules_frame, orient="vertical", command=self.modules_tree.yview)
        self.modules_tree.configure(yscrollcommand=modules_scrollbar.set)
        
        self.modules_tree.pack(side="left", fill="both", expand=True)
        modules_scrollbar.pack(side="right", fill="y")
        
    def create_alerts_tab(self, parent):
        """Créer l'onglet alertes."""
        # Contrôles d'alertes
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(controls_frame, text="Actualiser", 
                  command=self.refresh_alerts).pack(side="left", padx=(0, 5))
        ttk.Button(controls_frame, text="Effacer Alertes", 
                  command=self.clear_alerts).pack(side="left", padx=(0, 5))
        ttk.Button(controls_frame, text="Configuration", 
                  command=self.config_alerts).pack(side="left", padx=(0, 5))
        
        # Liste des alertes
        alerts_list_frame = ttk.LabelFrame(parent, text="Alertes Récentes", padding="5")
        alerts_list_frame.pack(fill="both", expand=True)
        
        self.alerts_tree = ttk.Treeview(alerts_list_frame, columns=("Time", "Severity", "Message"), show="headings")
        self.alerts_tree.heading("Time", text="Heure")
        self.alerts_tree.heading("Severity", text="Sévérité")
        self.alerts_tree.heading("Message", text="Message")
        
        alerts_scrollbar = ttk.Scrollbar(alerts_list_frame, orient="vertical", command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_tree.pack(side="left", fill="both", expand=True)
        alerts_scrollbar.pack(side="right", fill="y")
        
    def create_logs_tab(self, parent):
        """Créer l'onglet logs."""
        # Contrôles de logs
        log_controls_frame = ttk.Frame(parent)
        log_controls_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(log_controls_frame, text="Actualiser", 
                  command=self.refresh_logs).pack(side="left", padx=(0, 5))
        ttk.Button(log_controls_frame, text="Effacer", 
                  command=self.clear_logs).pack(side="left", padx=(0, 5))
        
        ttk.Label(log_controls_frame, text="Niveau:").pack(side="left", padx=(10, 5))
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(log_controls_frame, textvariable=self.log_level_var, 
                                     values=["DEBUG", "INFO", "WARNING", "ERROR"], width=10, state="readonly")
        log_level_combo.pack(side="left", padx=(0, 5))
        
        # Zone de logs
        logs_frame = ttk.LabelFrame(parent, text="Logs Temps Réel", padding="5")
        logs_frame.pack(fill="both", expand=True)
        
        self.logs_text = tk.Text(logs_frame, height=25, state="disabled")
        logs_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scrollbar.set)
        
        self.logs_text.pack(side="left", fill="both", expand=True)
        logs_scrollbar.pack(side="right", fill="y")
        
    def update_indicators(self):
        """Mettre à jour les indicateurs principaux."""
        system = self.metrics_collector.current_system_metrics
        bot = self.metrics_collector.current_bot_metrics
        
        updates = {
            "bot_status": bot.status.title(),
            "uptime": self._format_uptime(bot.uptime),
            "actions_per_minute": f"{bot.actions_per_minute:.1f}",
            "success_rate": f"{bot.success_rate:.1f}%",
            "cpu_percent": f"{system.cpu_percent:.1f}%",
            "memory_percent": f"{system.memory_percent:.1f}%", 
            "character_level": str(bot.character_level),
            "kamas_earned": f"{bot.kamas_earned:,}"
        }
        
        for key, value in updates.items():
            if key in self.indicator_labels:
                self.indicator_labels[key].config(text=value)
                
                # Colorier selon les valeurs
                if key in ["cpu_percent", "memory_percent"]:
                    color = "red" if float(value.rstrip('%')) > 80 else "orange" if float(value.rstrip('%')) > 60 else "green"
                    self.indicator_labels[key].config(foreground=color)
                elif key == "success_rate":
                    color = "green" if float(value.rstrip('%')) > 90 else "orange" if float(value.rstrip('%')) > 70 else "red"
                    self.indicator_labels[key].config(foreground=color)
    
    def update_overview_charts(self):
        """Mettre à jour les graphiques de vue d'ensemble."""
        if not self.metrics_collector.system_history or not self.metrics_collector.bot_history:
            return
            
        # Données récentes (dernières 60 mesures)
        recent_system = list(self.metrics_collector.system_history)[-60:]
        recent_bot = list(self.metrics_collector.bot_history)[-60:]
        
        # Temps relatif
        times = list(range(-len(recent_system), 0))
        
        # CPU
        self.cpu_ax.clear()
        self.cpu_ax.plot(times, [m.cpu_percent for m in recent_system], 'b-', linewidth=2)
        self.cpu_ax.plot(times, [m.bot_process_cpu for m in recent_system], 'r-', linewidth=1, alpha=0.7)
        self.cpu_ax.set_title("CPU (%) - Bleu: Système, Rouge: Bot")
        self.cpu_ax.set_ylim(0, 100)
        self.cpu_ax.grid(True, alpha=0.3)
        
        # Mémoire
        self.memory_ax.clear()
        self.memory_ax.plot(times, [m.memory_percent for m in recent_system], 'g-', linewidth=2)
        self.memory_ax.set_title("Mémoire (%)")
        self.memory_ax.set_ylim(0, 100)
        self.memory_ax.grid(True, alpha=0.3)
        
        # Actions
        self.actions_ax.clear()
        self.actions_ax.plot(times, [m.actions_per_minute for m in recent_bot], 'purple', linewidth=2)
        self.actions_ax.set_title("Actions/min")
        self.actions_ax.grid(True, alpha=0.3)
        
        # Succès
        self.success_ax.clear()
        self.success_ax.plot(times, [m.success_rate for m in recent_bot], 'orange', linewidth=2)
        self.success_ax.set_title("Taux de Succès (%)")
        self.success_ax.set_ylim(0, 100)
        self.success_ax.grid(True, alpha=0.3)
        
        # Mise à jour du canvas
        self.overview_canvas.draw()
    
    def update_system_info(self):
        """Mettre à jour les informations système."""
        system = self.metrics_collector.current_system_metrics
        
        info = f"""Informations Système (Temps: {datetime.fromtimestamp(system.timestamp).strftime('%H:%M:%S')})

CPU: {system.cpu_percent:.1f}%
Mémoire: {system.memory_percent:.1f}% ({system.memory_used:.0f} MB / {system.memory_total:.0f} MB)
Disque: {system.disk_usage:.1f}%
Processus: {system.processes_count}

Bot Process:
  CPU: {system.bot_process_cpu:.1f}%
  Mémoire: {system.bot_process_memory:.1f} MB

Réseau:
  Envoyé: {self._format_bytes(system.network_sent)}
  Reçu: {self._format_bytes(system.network_recv)}
"""
        
        self.system_info_text.config(state="normal")
        self.system_info_text.delete(1.0, tk.END)
        self.system_info_text.insert(1.0, info)
        self.system_info_text.config(state="disabled")
    
    def update_bot_stats(self):
        """Mettre à jour les statistiques du bot."""
        bot = self.metrics_collector.current_bot_metrics
        
        stats = f"""Statistiques du Bot (Temps: {datetime.fromtimestamp(bot.timestamp).strftime('%H:%M:%S')})

État: {bot.status.title()}
Temps de fonctionnement: {self._format_uptime(bot.uptime)}

Compteurs:
  Actions totales: {bot.actions_count:,}
  Erreurs: {bot.errors_count}
  Avertissements: {bot.warnings_count}

Performances:
  Actions/minute: {bot.actions_per_minute:.1f}
  Taux de succès: {bot.success_rate:.1f}%
  Temps de réponse moyen: {bot.average_response_time:.3f}s

Personnage:
  Niveau: {bot.character_level}
  Expérience gagnée: {bot.experience_gained:,}
  Objets collectés: {bot.items_collected:,}
  Kamas gagnés: {bot.kamas_earned:,}

Modules actifs: {len(bot.active_modules)}
{', '.join(bot.active_modules) if bot.active_modules else 'Aucun'}
"""
        
        self.bot_stats_text.config(state="normal")
        self.bot_stats_text.delete(1.0, tk.END)
        self.bot_stats_text.insert(1.0, stats)
        self.bot_stats_text.config(state="disabled")
    
    def refresh_alerts(self):
        """Actualiser la liste des alertes."""
        # Effacer la liste actuelle
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        
        # Ajouter les alertes récentes
        recent_alerts = self.alert_manager.get_active_alerts(60)  # Dernière heure
        
        for alert_name, message, timestamp in reversed(recent_alerts):
            self.alerts_tree.insert("", 0, values=(
                timestamp.strftime("%H:%M:%S"),
                alert_name.upper(),
                message
            ))
    
    def clear_alerts(self):
        """Effacer les alertes."""
        self.alert_manager.active_alerts.clear()
        self.refresh_alerts()
    
    def config_alerts(self):
        """Ouvrir la configuration des alertes."""
        messagebox.showinfo("Configuration", "Configuration des alertes non implémentée")
    
    def refresh_logs(self):
        """Actualiser les logs."""
        # Pour l'exemple, afficher des logs fictifs
        self.logs_text.config(state="normal")
        current_time = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{current_time}] INFO - Monitoring actif\n"
        self.logs_text.insert(tk.END, log_entry)
        self.logs_text.see(tk.END)
        self.logs_text.config(state="disabled")
    
    def clear_logs(self):
        """Effacer les logs."""
        self.logs_text.config(state="normal")
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state="disabled")
    
    def start_auto_update(self):
        """Démarrer les mises à jour automatiques."""
        def update_all():
            if self.animation_active:
                self.update_indicators()
                self.update_overview_charts()
                self.update_system_info()
                self.update_bot_stats()
                self.refresh_alerts()
            
            self.root.after(self.update_interval, update_all)
        
        self.animation_active = True
        update_all()
    
    def stop_auto_update(self):
        """Arrêter les mises à jour automatiques."""
        self.animation_active = False
    
    def _format_uptime(self, seconds: float) -> str:
        """Formater le temps de fonctionnement."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Formater les octets."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
    
    def run(self):
        """Lancer le dashboard."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        finally:
            self.stop_auto_update()
    
    def on_closing(self):
        """Gérer la fermeture de l'application."""
        self.stop_auto_update()
        self.root.destroy()


class WebDashboard:
    """Dashboard web (optionnel)."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager, port: int = 8080):
        if not WEB_AVAILABLE:
            raise ImportError("Flask non disponible")
            
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Configurer les routes web."""
        @self.app.route("/")
        def dashboard():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route("/api/metrics")
        def api_metrics():
            return jsonify({
                "system": asdict(self.metrics_collector.current_system_metrics),
                "bot": asdict(self.metrics_collector.current_bot_metrics),
                "timestamp": time.time()
            })
        
        @self.app.route("/api/alerts")
        def api_alerts():
            alerts = self.alert_manager.get_active_alerts(60)
            return jsonify([{
                "name": name,
                "message": message,
                "timestamp": timestamp.isoformat()
            } for name, message, timestamp in alerts])
    
    def get_dashboard_template(self) -> str:
        """Template HTML du dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Bot DOFUS - Monitoring</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; margin-top: 5px; }
        .status-running { color: green; }
        .status-stopped { color: red; }
        .alert { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
    <script>
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpu').textContent = data.system.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory').textContent = data.system.memory_percent.toFixed(1) + '%';
                    document.getElementById('status').textContent = data.bot.status;
                    document.getElementById('status').className = 'metric-value status-' + data.bot.status;
                    document.getElementById('actions').textContent = data.bot.actions_per_minute.toFixed(1);
                    document.getElementById('success').textContent = data.bot.success_rate.toFixed(1) + '%';
                });
        }
        setInterval(updateMetrics, 2000);
    </script>
</head>
<body onload="updateMetrics()">
    <div class="container">
        <div class="header">
            <h1>Bot DOFUS - Monitoring Dashboard</h1>
            <p>Surveillance en temps réel</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="status">--</div>
                <div class="metric-label">Status du Bot</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="cpu">--</div>
                <div class="metric-label">CPU Système</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="memory">--</div>
                <div class="metric-label">Mémoire Système</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="actions">--</div>
                <div class="metric-label">Actions/minute</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="success">--</div>
                <div class="metric-label">Taux de Succès</div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h2>Alertes Récentes</h2>
            <div id="alerts">Chargement des alertes...</div>
        </div>
    </div>
</body>
</html>
        """
    
    def run(self):
        """Lancer le serveur web."""
        print(f"Dashboard web disponible sur http://localhost:{self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=False)


async def main():
    """Fonction principale du monitoring."""
    parser = argparse.ArgumentParser(
        description="Interface de monitoring pour le bot DOFUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --dashboard                    # Interface graphique
  %(prog)s --web-server --port 8080     # Serveur web
  %(prog)s --cli                         # Mode ligne de commande
  %(prog)s --export-report --format json # Exporter un rapport
        """
    )
    
    # Modes d'affichage
    parser.add_argument("--dashboard", action="store_true", help="Interface graphique")
    parser.add_argument("--web-server", action="store_true", help="Serveur web")
    parser.add_argument("--cli", action="store_true", help="Mode ligne de commande")
    
    # Configuration serveur web
    parser.add_argument("--port", type=int, default=8080, help="Port du serveur web")
    parser.add_argument("--host", default="localhost", help="Adresse du serveur web")
    
    # Rapports et exports
    parser.add_argument("--export-report", action="store_true", help="Exporter un rapport")
    parser.add_argument("--format", choices=["json", "html", "txt"], default="json", help="Format d'export")
    
    # Configuration
    parser.add_argument("--config-dir", default="config", help="Répertoire de configuration")
    parser.add_argument("--db-path", help="Chemin de la base de données de métriques")
    parser.add_argument("--alerts-config", help="Fichier de configuration des alertes")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer les composants
    db_path = Path(args.db_path) if args.db_path else Path("logs/metrics.db")
    metrics_collector = MetricsCollector(db_path)
    
    alerts_config_path = Path(args.alerts_config) if args.alerts_config else Path("config/alerts.json")
    alert_manager = AlertManager(alerts_config_path)
    
    # État temps réel (simulé pour l'exemple)
    realtime_state = RealtimeState()
    
    try:
        if args.dashboard:
            if not GUI_AVAILABLE:
                print("Interface graphique non disponible")
                return 1
                
            # Démarrer la collecte de métriques en arrière-plan
            collection_task = asyncio.create_task(metrics_collector.start_collection(realtime_state))
            
            # Lancer le dashboard
            dashboard = MonitorDashboard(metrics_collector, alert_manager)
            
            def run_dashboard():
                dashboard.run()
            
            dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            dashboard_thread.start()
            
            # Boucle de vérification des alertes
            while dashboard_thread.is_alive():
                alert_manager.check_alerts(
                    metrics_collector.current_system_metrics,
                    metrics_collector.current_bot_metrics
                )
                await asyncio.sleep(5)
                
            collection_task.cancel()
            
        elif args.web_server:
            if not WEB_AVAILABLE:
                print("Serveur web non disponible")
                return 1
                
            # Démarrer la collecte
            collection_task = asyncio.create_task(metrics_collector.start_collection(realtime_state))
            
            # Lancer le serveur web
            web_dashboard = WebDashboard(metrics_collector, alert_manager, args.port)
            
            def run_web():
                web_dashboard.run()
            
            web_thread = threading.Thread(target=run_web, daemon=True)
            web_thread.start()
            
            print("Serveur web démarré. Appuyez sur Ctrl+C pour arrêter.")
            
            try:
                while True:
                    alert_manager.check_alerts(
                        metrics_collector.current_system_metrics,
                        metrics_collector.current_bot_metrics
                    )
                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                pass
                
            collection_task.cancel()
            
        elif args.cli:
            print("=== MONITORING BOT DOFUS - MODE CLI ===")
            
            # Démarrer la collecte
            collection_task = asyncio.create_task(metrics_collector.start_collection(realtime_state))
            
            try:
                while True:
                    # Afficher les métriques
                    system = metrics_collector.current_system_metrics
                    bot = metrics_collector.current_bot_metrics
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"Système - CPU: {system.cpu_percent:.1f}% | Mémoire: {system.memory_percent:.1f}%")
                    print(f"Bot - Status: {bot.status} | Actions/min: {bot.actions_per_minute:.1f} | Succès: {bot.success_rate:.1f}%")
                    
                    # Vérifier les alertes
                    alert_manager.check_alerts(system, bot)
                    
                    await asyncio.sleep(10)
                    
            except KeyboardInterrupt:
                print("\nArrêt du monitoring")
                
            collection_task.cancel()
            
        elif args.export_report:
            print("Export de rapport...")
            
            # Collecter les données récentes
            system_data = metrics_collector.get_historical_data("system_metrics", 24)
            bot_data = metrics_collector.get_historical_data("bot_metrics", 24)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_data,
                "bot_metrics": bot_data,
                "alerts": [asdict(alert) for alert in alert_manager.alerts.values()]
            }
            
            if args.format == "json":
                report_file = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Rapport JSON exporté : {report_file}")
                
            # Autres formats non implémentés pour cet exemple
            
        else:
            print("=== MONITORING BOT DOFUS ===")
            print("Utilisez --help pour voir les options disponibles")
            print("Modes recommandés:")
            print("  --dashboard  : Interface graphique complète")
            print("  --web-server : Interface web accessible depuis un navigateur")
            print("  --cli        : Monitoring en ligne de commande")
            
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
        return 0
    except Exception as e:
        logging.error(f"Erreur critique : {e}")
        return 1
    finally:
        metrics_collector.stop_collection()
        
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)