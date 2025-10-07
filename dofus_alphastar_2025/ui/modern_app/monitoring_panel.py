#!/usr/bin/env python3
"""
MonitoringPanel - Panneau de monitoring et debugging
Console de logs, debugging avancé et inspection système
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional, List, Callable
import threading
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
import sys
import psutil
import re

from .theme_manager import ThemeManager

@dataclass
class LogEntry:
    """Entrée de log"""
    timestamp: float
    level: str
    module: str
    message: str
    details: Optional[str] = None
    exception: Optional[str] = None

@dataclass
class SystemMetrics:
    """Métriques système"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int

class LogFilter:
    """Filtres pour les logs"""

    def __init__(self):
        self.levels: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.modules: List[str] = []
        self.text_filter: str = ""
        self.time_range: Optional[tuple] = None

    def matches(self, entry: LogEntry) -> bool:
        """Vérifie si l'entrée correspond aux filtres"""
        # Niveau
        if entry.level not in self.levels:
            return False

        # Module
        if self.modules and entry.module not in self.modules:
            return False

        # Texte
        if self.text_filter:
            search_text = f"{entry.message} {entry.details or ''}".lower()
            if self.text_filter.lower() not in search_text:
                return False

        # Plage temporelle
        if self.time_range:
            start, end = self.time_range
            if not (start <= entry.timestamp <= end):
                return False

        return True

class LogConsole:
    """Console de logs avancée"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager
        self.entries: List[LogEntry] = []
        self.filter = LogFilter()
        self.auto_scroll = True
        self.max_entries = 10000

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        # Frame principal
        self.frame = self.theme.create_frame(self.parent, "primary")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Toolbar
        self.create_toolbar()

        # Zone de logs
        self.create_log_area()

        # Filtres
        self.create_filters()

    def create_toolbar(self):
        """Crée la barre d'outils"""
        toolbar = self.theme.create_frame(self.frame, "secondary")
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        # Boutons d'action
        clear_btn = self.theme.create_secondary_button(
            toolbar,
            text="️ Effacer",
            command=self.clear_logs
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        export_btn = self.theme.create_secondary_button(
            toolbar,
            text="[SAVE] Exporter",
            command=self.export_logs
        )
        export_btn.pack(side=tk.LEFT, padx=(0, 5))

        pause_btn = self.theme.create_secondary_button(
            toolbar,
            text="||️ Pause",
            command=self.toggle_pause
        )
        pause_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Auto-scroll
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = tk.Checkbutton(
            toolbar,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            **self.theme.get_style("body"),
            command=self.toggle_auto_scroll
        )
        auto_scroll_check.pack(side=tk.LEFT, padx=(0, 10))

        # Compteur d'entrées
        self.entry_count_label = self.theme.create_body_label(
            toolbar,
            text="0 entrées"
        )
        self.entry_count_label.pack(side=tk.RIGHT)

    def create_log_area(self):
        """Crée la zone d'affichage des logs"""
        log_frame = self.theme.create_frame(self.frame, "primary")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Text widget avec scrollbars
        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            bg=self.theme.get_colors().bg_secondary,
            fg=self.theme.get_colors().text_primary,
            font=("Consolas", 9),
            relief="flat",
            bd=0,
            state="disabled"
        )

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        h_scrollbar = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)

        self.log_text.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )

        # Placement
        self.log_text.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Configuration des couleurs par niveau
        self.configure_log_colors()

        # Bind des événements
        self.log_text.bind("<Button-3>", self.show_context_menu)

    def create_filters(self):
        """Crée les filtres"""
        filters_frame = self.theme.create_frame(self.frame, "secondary")
        filters_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Filtre de niveau
        level_frame = self.theme.create_frame(filters_frame, "primary")
        level_frame.pack(side=tk.LEFT, padx=(0, 10))

        level_label = self.theme.create_body_label(level_frame, text="Niveau:")
        level_label.pack(side=tk.LEFT)

        self.level_vars = {}
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in levels:
            var = tk.BooleanVar(value=True)
            self.level_vars[level] = var

            check = tk.Checkbutton(
                level_frame,
                text=level[:3],
                variable=var,
                **self.theme.get_style("body"),
                command=self.update_filters
            )
            check.pack(side=tk.LEFT, padx=2)

        # Filtre de texte
        text_frame = self.theme.create_frame(filters_frame, "primary")
        text_frame.pack(side=tk.LEFT, padx=(0, 10))

        text_label = self.theme.create_body_label(text_frame, text="Recherche:")
        text_label.pack(side=tk.LEFT)

        self.text_filter_var = tk.StringVar()
        text_entry = self.theme.create_entry(text_frame, textvariable=self.text_filter_var)
        text_entry.pack(side=tk.LEFT, padx=(5, 0))
        text_entry.bind("<KeyRelease>", lambda e: self.update_filters())

        # Filtre de module
        module_frame = self.theme.create_frame(filters_frame, "primary")
        module_frame.pack(side=tk.RIGHT)

        module_label = self.theme.create_body_label(module_frame, text="Module:")
        module_label.pack(side=tk.LEFT)

        self.module_filter_var = tk.StringVar()
        self.module_combo = ttk.Combobox(
            module_frame,
            textvariable=self.module_filter_var,
            values=["Tous"],
            width=15,
            state="readonly"
        )
        self.module_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.module_combo.bind("<<ComboboxSelected>>", lambda e: self.update_filters())

    def configure_log_colors(self):
        """Configure les couleurs par niveau de log"""
        colors = self.theme.get_colors()

        # Tags pour les niveaux
        self.log_text.tag_configure("DEBUG", foreground=colors.text_secondary)
        self.log_text.tag_configure("INFO", foreground=colors.text_primary)
        self.log_text.tag_configure("WARNING", foreground=colors.accent_warning)
        self.log_text.tag_configure("ERROR", foreground=colors.accent_error)
        self.log_text.tag_configure("CRITICAL",
                                   foreground=colors.accent_error,
                                   background=colors.bg_tertiary)

        # Tags pour les composants
        self.log_text.tag_configure("timestamp", foreground=colors.text_secondary)
        self.log_text.tag_configure("module", foreground=colors.accent_info)

    def add_log_entry(self, entry: LogEntry):
        """Ajoute une entrée de log"""
        self.entries.append(entry)

        # Limiter le nombre d'entrées
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Mettre à jour les modules disponibles
        self.update_module_filter()

        # Afficher si correspond aux filtres
        if self.filter.matches(entry):
            self.display_log_entry(entry)

        # Mettre à jour le compteur
        self.update_entry_count()

    def display_log_entry(self, entry: LogEntry):
        """Affiche une entrée de log"""
        if not hasattr(self, 'paused') or not self.paused:
            self.log_text.configure(state="normal")

            # Timestamp
            timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
            self.log_text.insert(tk.END, f"[{timestamp_str}] ", "timestamp")

            # Niveau
            level_str = f"[{entry.level:>8}] "
            self.log_text.insert(tk.END, level_str, entry.level)

            # Module
            module_str = f"[{entry.module}] "
            self.log_text.insert(tk.END, module_str, "module")

            # Message
            self.log_text.insert(tk.END, f"{entry.message}\n")

            # Détails si présents
            if entry.details:
                self.log_text.insert(tk.END, f"    └─ {entry.details}\n", "details")

            # Exception si présente
            if entry.exception:
                for line in entry.exception.split('\n'):
                    if line.strip():
                        self.log_text.insert(tk.END, f"    │ {line}\n", "ERROR")

            self.log_text.configure(state="disabled")

            # Auto-scroll
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)

    def update_filters(self):
        """Met à jour les filtres et rafraîchit l'affichage"""
        # Mettre à jour les filtres
        self.filter.levels = [level for level, var in self.level_vars.items() if var.get()]
        self.filter.text_filter = self.text_filter_var.get()

        module = self.module_filter_var.get()
        if module and module != "Tous":
            self.filter.modules = [module]
        else:
            self.filter.modules = []

        # Rafraîchir l'affichage
        self.refresh_display()

    def refresh_display(self):
        """Rafraîchit l'affichage des logs"""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)

        for entry in self.entries:
            if self.filter.matches(entry):
                self.display_log_entry(entry)

        self.log_text.configure(state="disabled")

    def update_module_filter(self):
        """Met à jour la liste des modules"""
        modules = set(entry.module for entry in self.entries)
        current_values = ["Tous"] + sorted(modules)

        self.module_combo.configure(values=current_values)

    def update_entry_count(self):
        """Met à jour le compteur d'entrées"""
        filtered_count = sum(1 for entry in self.entries if self.filter.matches(entry))
        total_count = len(self.entries)

        self.entry_count_label.configure(
            text=f"{filtered_count}/{total_count} entrées"
        )

    def clear_logs(self):
        """Efface tous les logs"""
        if messagebox.askyesno("Confirmation", "Effacer tous les logs?"):
            self.entries.clear()
            self.log_text.configure(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.configure(state="disabled")
            self.update_entry_count()

    def export_logs(self):
        """Exporte les logs"""
        filename = filedialog.asksaveasfilename(
            title="Exporter logs",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv")
            ]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    self.export_json(filename)
                elif filename.endswith('.csv'):
                    self.export_csv(filename)
                else:
                    self.export_text(filename)

                messagebox.showinfo("Succès", f"Logs exportés: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export: {e}")

    def export_text(self, filename):
        """Exporte en format texte"""
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                if self.filter.matches(entry):
                    timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] [{entry.level}] [{entry.module}] {entry.message}\n")
                    if entry.details:
                        f.write(f"    Details: {entry.details}\n")
                    if entry.exception:
                        f.write(f"    Exception:\n{entry.exception}\n")
                    f.write("\n")

    def export_json(self, filename):
        """Exporte en format JSON"""
        filtered_entries = [
            asdict(entry) for entry in self.entries
            if self.filter.matches(entry)
        ]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_entries, f, indent=2, ensure_ascii=False)

    def export_csv(self, filename):
        """Exporte en format CSV"""
        import csv

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Level", "Module", "Message", "Details", "Exception"])

            for entry in self.entries:
                if self.filter.matches(entry):
                    writer.writerow([
                        entry.timestamp,
                        entry.level,
                        entry.module,
                        entry.message,
                        entry.details or "",
                        entry.exception or ""
                    ])

    def toggle_pause(self):
        """Basculer pause/reprise"""
        self.paused = getattr(self, 'paused', False)
        self.paused = not self.paused

        # Mettre à jour le bouton
        # TODO: Implémenter avec référence au bouton

    def toggle_auto_scroll(self):
        """Basculer auto-scroll"""
        self.auto_scroll = self.auto_scroll_var.get()

    def show_context_menu(self, event):
        """Affiche le menu contextuel"""
        context_menu = tk.Menu(self.log_text, tearoff=0)

        context_menu.add_command(label="Copier", command=self.copy_selection)
        context_menu.add_command(label="Sélectionner tout", command=self.select_all)
        context_menu.add_separator()
        context_menu.add_command(label="Rechercher", command=self.show_search_dialog)
        context_menu.add_separator()
        context_menu.add_command(label="Effacer", command=self.clear_logs)

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def copy_selection(self):
        """Copie la sélection"""
        try:
            selected_text = self.log_text.selection_get()
            self.log_text.clipboard_clear()
            self.log_text.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def select_all(self):
        """Sélectionne tout le texte"""
        self.log_text.tag_add("sel", "1.0", "end")

    def show_search_dialog(self):
        """Affiche la boîte de dialogue de recherche"""
        # TODO: Implémenter dialogue de recherche avancée
        pass

class SystemMonitor:
    """Moniteur système"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame = self.theme.create_panel(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre
        title = self.theme.create_subtitle_label(
            self.frame,
            text=" Monitoring Système"
        )
        title.pack(pady=(15, 10))

        # Métriques système
        metrics_frame = self.theme.create_frame(self.frame, "primary")
        metrics_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # CPU
        self.cpu_var = tk.StringVar(value="CPU: 0%")
        cpu_label = self.theme.create_body_label(metrics_frame, textvariable=self.cpu_var)
        cpu_label.grid(row=0, column=0, sticky="w", pady=2)

        # Mémoire
        self.memory_var = tk.StringVar(value="Mémoire: 0%")
        memory_label = self.theme.create_body_label(metrics_frame, textvariable=self.memory_var)
        memory_label.grid(row=0, column=1, sticky="w", padx=(20, 0), pady=2)

        # Disque
        self.disk_var = tk.StringVar(value="Disque: 0%")
        disk_label = self.theme.create_body_label(metrics_frame, textvariable=self.disk_var)
        disk_label.grid(row=1, column=0, sticky="w", pady=2)

        # Threads
        self.threads_var = tk.StringVar(value="Threads: 0")
        threads_label = self.theme.create_body_label(metrics_frame, textvariable=self.threads_var)
        threads_label.grid(row=1, column=1, sticky="w", padx=(20, 0), pady=2)

        # Démarrer le monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Démarre le monitoring système"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Arrête le monitoring système"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitoring_loop(self):
        """Boucle de monitoring"""
        while self.is_monitoring:
            try:
                self.update_metrics()
                time.sleep(2)
            except Exception as e:
                print(f"Erreur monitoring système: {e}")
                time.sleep(5)

    def update_metrics(self):
        """Met à jour les métriques"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent()
            self.cpu_var.set(f"CPU: {cpu_percent:.1f}%")

            # Mémoire
            memory = psutil.virtual_memory()
            self.memory_var.set(f"Mémoire: {memory.percent:.1f}% ({memory.used // (1024**2)}MB)")

            # Disque
            disk = psutil.disk_usage('/')
            self.disk_var.set(f"Disque: {disk.percent:.1f}%")

            # Threads
            thread_count = threading.active_count()
            self.threads_var.set(f"Threads: {thread_count}")

        except Exception as e:
            print(f"Erreur collecte métriques: {e}")

class DebugInspector:
    """Inspecteur de debugging"""

    def __init__(self, parent, theme_manager: ThemeManager):
        self.parent = parent
        self.theme = theme_manager

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface"""
        self.frame = self.theme.create_panel(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre
        title = self.theme.create_subtitle_label(
            self.frame,
            text="[SEARCH] Inspecteur Debug"
        )
        title.pack(pady=(15, 10))

        # Variables système
        vars_frame = self.theme.create_frame(self.frame, "secondary")
        vars_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Tree view pour les variables
        columns = ("Variable", "Type", "Valeur")
        self.vars_tree = ttk.Treeview(vars_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.vars_tree.heading(col, text=col)
            self.vars_tree.column(col, width=150)

        # Scrollbar
        scrollbar = ttk.Scrollbar(vars_frame, orient="vertical", command=self.vars_tree.yview)
        self.vars_tree.configure(yscrollcommand=scrollbar.set)

        self.vars_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Boutons de contrôle
        controls_frame = self.theme.create_frame(self.frame, "primary")
        controls_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        refresh_btn = self.theme.create_secondary_button(
            controls_frame,
            text="[RELOAD] Actualiser",
            command=self.refresh_variables
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        inspect_btn = self.theme.create_secondary_button(
            controls_frame,
            text="[SEARCH] Inspecter",
            command=self.inspect_selected
        )
        inspect_btn.pack(side=tk.LEFT)

        # Charger les variables initiales
        self.refresh_variables()

    def refresh_variables(self):
        """Actualise la liste des variables"""
        # Effacer le contenu existant
        for item in self.vars_tree.get_children():
            self.vars_tree.delete(item)

        # Variables globales
        self.vars_tree.insert("", "end", values=("Python Version", "str", sys.version))
        self.vars_tree.insert("", "end", values=("Platform", "str", sys.platform))

        # Variables système
        try:
            import os
            self.vars_tree.insert("", "end", values=("PID", "int", os.getpid()))
            self.vars_tree.insert("", "end", values=("CWD", "str", os.getcwd()))
        except:
            pass

        # Variables de l'application
        if hasattr(self.parent, 'app_controller') and self.parent.app_controller:
            controller = self.parent.app_controller
            self.vars_tree.insert("", "end", values=("Bot Running", "bool", getattr(controller, 'is_running', False)))
            self.vars_tree.insert("", "end", values=("Bot Paused", "bool", getattr(controller, 'is_paused', False)))

    def inspect_selected(self):
        """Inspecte la variable sélectionnée"""
        selection = self.vars_tree.selection()
        if selection:
            item = self.vars_tree.item(selection[0])
            var_name = item['values'][0]
            messagebox.showinfo("Inspection", f"Inspection de: {var_name}")

class MonitoringPanel:
    """Panneau principal de monitoring"""

    def __init__(self, parent, theme_manager: ThemeManager, app_controller=None):
        self.parent = parent
        self.theme = theme_manager
        self.app_controller = app_controller

        # Logger personnalisé
        self.setup_logging()

        # Interface
        self.frame = self.theme.create_frame(parent, "primary")
        self.setup_ui()

    def setup_logging(self):
        """Configure le système de logging"""
        # Créer un handler personnalisé pour capturer les logs
        self.log_handler = LogHandler(self)
        self.log_handler.setLevel(logging.DEBUG)

        # Format des logs
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        self.log_handler.setFormatter(formatter)

        # Ajouter aux loggers principaux
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.DEBUG)

    def setup_ui(self):
        """Configure l'interface principale"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme.create_frame(self.frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme.create_title_label(
            header_frame,
            text="[SEARCH] Monitoring et Debug"
        )
        title_label.pack(side=tk.LEFT)

        # Status général
        self.status_label = self.theme.create_body_label(
            header_frame,
            text="● Monitoring actif",
            fg=self.theme.get_colors().accent_success
        )
        self.status_label.pack(side=tk.RIGHT)

        # Notebook pour les sections
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Console de logs
        logs_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(logs_frame, text=" Logs")

        self.log_console = LogConsole(logs_frame, self.theme)

        # Monitoring système
        system_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(system_frame, text=" Système")

        self.system_monitor = SystemMonitor(system_frame, self.theme)

        # Debug inspector
        debug_frame = self.theme.create_frame(self.notebook, "primary")
        self.notebook.add(debug_frame, text="[SEARCH] Debug")

        self.debug_inspector = DebugInspector(debug_frame, self.theme)

        # Générer quelques logs de test
        self.generate_test_logs()

    def generate_test_logs(self):
        """Génère des logs de test"""
        test_entries = [
            LogEntry(
                timestamp=time.time(),
                level="INFO",
                module="bot_controller",
                message="Bot initialisé avec succès"
            ),
            LogEntry(
                timestamp=time.time() + 1,
                level="DEBUG",
                module="quest_manager",
                message="Chargement de la liste des quêtes",
                details="Trouvé 15 quêtes disponibles"
            ),
            LogEntry(
                timestamp=time.time() + 2,
                level="WARNING",
                module="navigation",
                message="Pathfinding: route sub-optimale détectée",
                details="Distance: 150m, optimale: 120m"
            ),
            LogEntry(
                timestamp=time.time() + 3,
                level="ERROR",
                module="vision_system",
                message="Erreur reconnaissance NPC",
                details="NPC non trouvé dans la région spécifiée",
                exception="NPCNotFoundException: NPC 'Forgeron' introuvable à (100, 200)"
            )
        ]

        for entry in test_entries:
            self.log_console.add_log_entry(entry)

    def add_log(self, level: str, module: str, message: str,
                details: Optional[str] = None, exception: Optional[str] = None):
        """Ajoute une entrée de log"""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            module=module,
            message=message,
            details=details,
            exception=exception
        )

        self.log_console.add_log_entry(entry)

    def stop_monitoring(self):
        """Arrête tous les systèmes de monitoring"""
        self.system_monitor.stop_monitoring()

class LogHandler(logging.Handler):
    """Handler de logging personnalisé"""

    def __init__(self, monitoring_panel):
        super().__init__()
        self.monitoring_panel = monitoring_panel

    def emit(self, record):
        """Émet un log"""
        try:
            # Extraire les informations du record
            level = record.levelname
            module = record.name
            message = record.getMessage()

            # Exception si présente
            exception = None
            if record.exc_info:
                exception = self.format_exception(record.exc_info)

            # Ajouter au panneau de monitoring
            if hasattr(self.monitoring_panel, 'log_console'):
                entry = LogEntry(
                    timestamp=record.created,
                    level=level,
                    module=module,
                    message=message,
                    exception=exception
                )
                self.monitoring_panel.log_console.add_log_entry(entry)

        except Exception:
            # Éviter les boucles d'erreur
            pass

    def format_exception(self, exc_info):
        """Formate une exception"""
        return ''.join(traceback.format_exception(*exc_info)).strip()

def create_monitoring_panel(parent, theme_manager: ThemeManager, app_controller=None) -> MonitoringPanel:
    """Factory function pour créer MonitoringPanel"""
    return MonitoringPanel(parent, theme_manager, app_controller)