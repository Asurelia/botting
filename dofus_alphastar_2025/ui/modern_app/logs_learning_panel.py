"""
Logs & Learning Panel - Panneau de logs temps réel et système d'apprentissage
Affiche les décisions du bot et permet le feedback utilisateur
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
import json
import queue
import threading
from pathlib import Path


@dataclass
class BotDecision:
    """Représente une décision du bot"""
    timestamp: datetime
    decision_id: str
    action_type: str
    reason: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    success: Optional[bool] = None
    user_feedback: Optional[str] = None  # 'correct', 'incorrect', 'improve'
    user_comment: Optional[str] = None


class LogEntry:
    """Entrée de log formatée"""
    def __init__(self, level: str, timestamp: datetime, message: str, category: str = "general"):
        self.level = level  # INFO, WARNING, ERROR, SUCCESS, DECISION
        self.timestamp = timestamp
        self.message = message
        self.category = category

    def format(self, show_timestamp: bool = True) -> str:
        """Formate le log pour affichage"""
        timestamp_str = self.timestamp.strftime("%H:%M:%S") if show_timestamp else ""
        level_prefix = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "DECISION": "🧠"
        }.get(self.level, "•")

        if show_timestamp:
            return f"[{timestamp_str}] {level_prefix} {self.message}"
        return f"{level_prefix} {self.message}"


class LogsLearningPanel:
    """
    Panneau des logs temps réel et système d'apprentissage

    Fonctionnalités:
    - Affichage en temps réel des logs du bot
    - Visualisation des décisions prises
    - Système de feedback utilisateur
    - Apprentissage par correction
    """

    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)

        # Queues pour communication thread-safe
        self.log_queue = queue.Queue()
        self.decision_queue = queue.Queue()

        # Historique
        self.logs_history: List[LogEntry] = []
        self.decisions_history: List[BotDecision] = []
        self.feedback_db = self._load_feedback_database()

        # Configuration
        self.max_logs_display = 1000
        self.max_decisions_display = 100
        self.auto_scroll = True

        # Filtres
        self.filter_level = "ALL"
        self.filter_category = "ALL"

        # Setup UI
        self._setup_ui()

        # Démarrer le processing des logs
        self._start_log_processing()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        self.frame.pack(fill=tk.BOTH, expand=True)

        # === TOOLBAR ===
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="🗑️ Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Export Logs", command=self.export_logs).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 View Feedback", command=self.show_feedback_stats).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Filtres
        ttk.Label(toolbar, text="Niveau:").pack(side=tk.LEFT, padx=2)
        self.filter_level_var = tk.StringVar(value="ALL")
        filter_combo = ttk.Combobox(toolbar, textvariable=self.filter_level_var,
                                   values=["ALL", "INFO", "WARNING", "ERROR", "SUCCESS", "DECISION"],
                                   state="readonly", width=10)
        filter_combo.pack(side=tk.LEFT, padx=2)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Auto-scroll toggle
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=10)

        # === PANED WINDOW PRINCIPAL ===
        main_paned = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === PANNEAU LOGS ===
        logs_frame = ttk.LabelFrame(main_paned, text="📋 Logs Temps Réel")
        main_paned.add(logs_frame, weight=3)

        # Zone de texte pour logs
        self.logs_text = scrolledtext.ScrolledText(
            logs_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tags pour coloration
        self.logs_text.tag_config("INFO", foreground="#58a6ff")
        self.logs_text.tag_config("WARNING", foreground="#f0ad4e")
        self.logs_text.tag_config("ERROR", foreground="#f85149")
        self.logs_text.tag_config("SUCCESS", foreground="#56d364")
        self.logs_text.tag_config("DECISION", foreground="#bc8cff", font=("Consolas", 9, "bold"))

        # === PANNEAU DÉCISIONS ===
        decisions_frame = ttk.LabelFrame(main_paned, text="🧠 Décisions du Bot & Apprentissage")
        main_paned.add(decisions_frame, weight=2)

        # Paned horizontal pour décisions + feedback
        decisions_paned = ttk.PanedWindow(decisions_frame, orient=tk.HORIZONTAL)
        decisions_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Liste des décisions
        decisions_list_frame = ttk.Frame(decisions_paned)
        decisions_paned.add(decisions_list_frame, weight=2)

        ttk.Label(decisions_list_frame, text="Dernières décisions:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=2)

        # Treeview pour décisions
        columns = ("Time", "Action", "Reason", "Status")
        self.decisions_tree = ttk.Treeview(decisions_list_frame, columns=columns, show="headings", height=8)

        self.decisions_tree.heading("Time", text="Heure")
        self.decisions_tree.heading("Action", text="Action")
        self.decisions_tree.heading("Reason", text="Raison")
        self.decisions_tree.heading("Status", text="Statut")

        self.decisions_tree.column("Time", width=80)
        self.decisions_tree.column("Action", width=120)
        self.decisions_tree.column("Reason", width=250)
        self.decisions_tree.column("Status", width=80)

        scrollbar = ttk.Scrollbar(decisions_list_frame, orient=tk.VERTICAL, command=self.decisions_tree.yview)
        self.decisions_tree.configure(yscrollcommand=scrollbar.set)

        self.decisions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.decisions_tree.bind("<<TreeviewSelect>>", self._on_decision_selected)

        # Panneau de feedback
        feedback_frame = ttk.Frame(decisions_paned)
        decisions_paned.add(feedback_frame, weight=1)

        ttk.Label(feedback_frame, text="Feedback sur la décision:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=2)

        # Détails de la décision sélectionnée
        self.decision_details_text = tk.Text(feedback_frame, height=6, wrap=tk.WORD, font=("Consolas", 8),
                                            bg="#2d2d2d", fg="#d4d4d4")
        self.decision_details_text.pack(fill=tk.X, pady=5)
        self.decision_details_text.config(state=tk.DISABLED)

        # Boutons de feedback
        feedback_buttons_frame = ttk.Frame(feedback_frame)
        feedback_buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(feedback_buttons_frame, text="✅ Correct",
                  command=lambda: self._submit_feedback("correct")).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(feedback_buttons_frame, text="❌ Incorrect",
                  command=lambda: self._submit_feedback("incorrect")).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(feedback_buttons_frame, text="🔄 À améliorer",
                  command=lambda: self._submit_feedback("improve")).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # Zone de commentaire
        ttk.Label(feedback_frame, text="Commentaire (optionnel):").pack(anchor=tk.W, pady=(5, 2))
        self.feedback_comment_text = tk.Text(feedback_frame, height=3, wrap=tk.WORD, font=("Arial", 9))
        self.feedback_comment_text.pack(fill=tk.X, pady=2)

        # Suggestions d'amélioration
        ttk.Label(feedback_frame, text="Suggestion d'action correcte:").pack(anchor=tk.W, pady=(5, 2))
        self.feedback_suggestion_text = tk.Text(feedback_frame, height=2, wrap=tk.WORD, font=("Arial", 9))
        self.feedback_suggestion_text.pack(fill=tk.X, pady=2)

        ttk.Button(feedback_frame, text="💾 Soumettre Feedback",
                  command=self._save_feedback).pack(fill=tk.X, pady=5)

        # Statistiques d'apprentissage
        stats_frame = ttk.LabelFrame(feedback_frame, text="📊 Statistiques")
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="Feedback soumis: 0\nCorrections: 0\nTaux de réussite: N/A",
                                     font=("Arial", 8))
        self.stats_label.pack(pady=5)

    def _start_log_processing(self):
        """Démarre le thread de processing des logs"""
        self.processing = True
        self.processing_thread = threading.Thread(target=self._process_queues, daemon=True)
        self.processing_thread.start()

    def _process_queues(self):
        """Process les queues de logs et décisions"""
        while self.processing:
            # Process logs
            try:
                while not self.log_queue.empty():
                    log_entry = self.log_queue.get_nowait()
                    self._add_log_to_display(log_entry)
            except queue.Empty:
                pass

            # Process decisions
            try:
                while not self.decision_queue.empty():
                    decision = self.decision_queue.get_nowait()
                    self._add_decision_to_display(decision)
            except queue.Empty:
                pass

            threading.Event().wait(0.1)  # Update every 100ms

    def add_log(self, level: str, message: str, category: str = "general"):
        """Ajoute un log (thread-safe)"""
        log_entry = LogEntry(level, datetime.now(), message, category)
        self.log_queue.put(log_entry)

    def add_decision(self, decision: BotDecision):
        """Ajoute une décision du bot (thread-safe)"""
        self.decision_queue.put(decision)

    def _add_log_to_display(self, log_entry: LogEntry):
        """Ajoute un log à l'affichage"""
        self.logs_history.append(log_entry)

        # Limiter historique
        if len(self.logs_history) > self.max_logs_display:
            self.logs_history = self.logs_history[-self.max_logs_display:]

        # Filtrage
        if not self._should_display_log(log_entry):
            return

        # Affichage
        self.logs_text.config(state=tk.NORMAL)
        formatted_log = log_entry.format() + "\n"
        self.logs_text.insert(tk.END, formatted_log, log_entry.level)

        # Auto-scroll
        if self.auto_scroll_var.get():
            self.logs_text.see(tk.END)

        self.logs_text.config(state=tk.DISABLED)

    def _add_decision_to_display(self, decision: BotDecision):
        """Ajoute une décision à l'affichage"""
        self.decisions_history.append(decision)

        # Limiter historique
        if len(self.decisions_history) > self.max_decisions_display:
            self.decisions_history = self.decisions_history[-self.max_decisions_display:]

        # Ajouter au treeview
        time_str = decision.timestamp.strftime("%H:%M:%S")
        status = "✅" if decision.success else "❌" if decision.success is False else "⏳"

        self.decisions_tree.insert("", 0, values=(
            time_str,
            decision.action_type,
            decision.reason[:50] + "..." if len(decision.reason) > 50 else decision.reason,
            status
        ), tags=(decision.decision_id,))

        # Log également
        self.add_log("DECISION", f"{decision.action_type}: {decision.reason}")

    def _should_display_log(self, log_entry: LogEntry) -> bool:
        """Détermine si un log doit être affiché selon les filtres"""
        if self.filter_level_var.get() != "ALL" and log_entry.level != self.filter_level_var.get():
            return False
        return True

    def _apply_filters(self):
        """Applique les filtres et rafraîchit l'affichage"""
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete(1.0, tk.END)

        for log_entry in self.logs_history:
            if self._should_display_log(log_entry):
                formatted_log = log_entry.format() + "\n"
                self.logs_text.insert(tk.END, formatted_log, log_entry.level)

        self.logs_text.config(state=tk.DISABLED)
        if self.auto_scroll_var.get():
            self.logs_text.see(tk.END)

    def _on_decision_selected(self, event):
        """Callback quand une décision est sélectionnée"""
        selection = self.decisions_tree.selection()
        if not selection:
            return

        item = selection[0]
        decision_id = self.decisions_tree.item(item, "tags")[0]

        # Trouver la décision
        decision = next((d for d in self.decisions_history if d.decision_id == decision_id), None)
        if not decision:
            return

        # Afficher les détails
        self.decision_details_text.config(state=tk.NORMAL)
        self.decision_details_text.delete(1.0, tk.END)

        details_text = f"""Action: {decision.action_type}
Raison: {decision.reason}
Détails: {json.dumps(decision.details, indent=2)}
Contexte: HP={decision.context.get('hp', 'N/A')} PA={decision.context.get('pa', 'N/A')} PM={decision.context.get('pm', 'N/A')}
Succès: {decision.success if decision.success is not None else 'En cours'}
"""

        if decision.user_feedback:
            details_text += f"\nFeedback: {decision.user_feedback}"
            if decision.user_comment:
                details_text += f"\nCommentaire: {decision.user_comment}"

        self.decision_details_text.insert(1.0, details_text)
        self.decision_details_text.config(state=tk.DISABLED)

    def _submit_feedback(self, feedback_type: str):
        """Soumet un feedback sur la décision"""
        selection = self.decisions_tree.selection()
        if not selection:
            messagebox.showwarning("Attention", "Veuillez sélectionner une décision")
            return

        item = selection[0]
        decision_id = self.decisions_tree.item(item, "tags")[0]

        decision = next((d for d in self.decisions_history if d.decision_id == decision_id), None)
        if not decision:
            return

        decision.user_feedback = feedback_type

        # Afficher message de confirmation
        feedback_messages = {
            "correct": "✅ Décision marquée comme correcte",
            "incorrect": "❌ Décision marquée comme incorrecte",
            "improve": "🔄 Décision marquée à améliorer"
        }

        self.add_log("INFO", feedback_messages[feedback_type])

    def _save_feedback(self):
        """Sauvegarde le feedback complet"""
        selection = self.decisions_tree.selection()
        if not selection:
            messagebox.showwarning("Attention", "Veuillez sélectionner une décision")
            return

        item = selection[0]
        decision_id = self.decisions_tree.item(item, "tags")[0]

        decision = next((d for d in self.decisions_history if d.decision_id == decision_id), None)
        if not decision:
            return

        # Récupérer commentaire et suggestion
        decision.user_comment = self.feedback_comment_text.get(1.0, tk.END).strip()
        suggested_action = self.feedback_suggestion_text.get(1.0, tk.END).strip()

        # Sauvegarder dans la DB
        feedback_entry = {
            "decision_id": decision.decision_id,
            "timestamp": decision.timestamp.isoformat(),
            "action_type": decision.action_type,
            "reason": decision.reason,
            "details": decision.details,
            "context": decision.context,
            "user_feedback": decision.user_feedback,
            "user_comment": decision.user_comment,
            "suggested_action": suggested_action
        }

        self.feedback_db.append(feedback_entry)
        self._persist_feedback_database()

        # Mettre à jour statistiques
        self._update_stats()

        # Clear les champs
        self.feedback_comment_text.delete(1.0, tk.END)
        self.feedback_suggestion_text.delete(1.0, tk.END)

        messagebox.showinfo("Succès", "Feedback sauvegardé avec succès!")
        self.add_log("SUCCESS", f"Feedback sauvegardé pour décision {decision_id}")

    def _load_feedback_database(self) -> List[Dict]:
        """Charge la base de données de feedback"""
        feedback_path = Path("data/feedback/decisions_feedback.json")
        if feedback_path.exists():
            try:
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _persist_feedback_database(self):
        """Sauvegarde la base de données de feedback"""
        feedback_path = Path("data/feedback")
        feedback_path.mkdir(parents=True, exist_ok=True)

        with open(feedback_path / "decisions_feedback.json", 'w', encoding='utf-8') as f:
            json.dump(self.feedback_db, f, indent=2, ensure_ascii=False, default=str)

    def _update_stats(self):
        """Met à jour les statistiques d'apprentissage"""
        total_feedback = len(self.feedback_db)
        correct_count = len([f for f in self.feedback_db if f.get("user_feedback") == "correct"])
        incorrect_count = len([f for f in self.feedback_db if f.get("user_feedback") == "incorrect"])

        success_rate = (correct_count / total_feedback * 100) if total_feedback > 0 else 0

        stats_text = f"""Feedback soumis: {total_feedback}
Correct: {correct_count} | Incorrect: {incorrect_count}
Taux de réussite: {success_rate:.1f}%"""

        self.stats_label.config(text=stats_text)

    def clear_logs(self):
        """Efface tous les logs"""
        if messagebox.askyesno("Confirmation", "Effacer tous les logs?"):
            self.logs_text.config(state=tk.NORMAL)
            self.logs_text.delete(1.0, tk.END)
            self.logs_text.config(state=tk.DISABLED)
            self.logs_history.clear()
            self.add_log("INFO", "Logs effacés")

    def export_logs(self):
        """Exporte les logs vers un fichier"""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            title="Exporter les logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                if filepath.endswith('.json'):
                    logs_data = [{
                        "timestamp": log.timestamp.isoformat(),
                        "level": log.level,
                        "message": log.message,
                        "category": log.category
                    } for log in self.logs_history]

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(logs_data, f, indent=2, ensure_ascii=False)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for log in self.logs_history:
                            f.write(log.format() + "\n")

                messagebox.showinfo("Succès", f"Logs exportés vers:\n{filepath}")
                self.add_log("SUCCESS", f"Logs exportés: {filepath}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{e}")

    def show_feedback_stats(self):
        """Affiche les statistiques détaillées de feedback"""
        stats_window = tk.Toplevel(self.parent)
        stats_window.title("Statistiques de Feedback")
        stats_window.geometry("600x400")

        # Analyse des feedbacks
        total = len(self.feedback_db)
        correct = len([f for f in self.feedback_db if f.get("user_feedback") == "correct"])
        incorrect = len([f for f in self.feedback_db if f.get("user_feedback") == "incorrect"])
        improve = len([f for f in self.feedback_db if f.get("user_feedback") == "improve"])

        # Groupement par type d'action
        action_stats = {}
        for feedback in self.feedback_db:
            action_type = feedback.get("action_type", "unknown")
            if action_type not in action_stats:
                action_stats[action_type] = {"correct": 0, "incorrect": 0, "improve": 0}

            feedback_type = feedback.get("user_feedback", "unknown")
            if feedback_type in action_stats[action_type]:
                action_stats[action_type][feedback_type] += 1

        # Affichage
        text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        stats_text = f"""📊 STATISTIQUES DE FEEDBACK DÉTAILLÉES
{'=' * 50}

Total de feedbacks: {total}
  ✅ Correct: {correct} ({correct/total*100:.1f}% si total > 0 else 0)
  ❌ Incorrect: {incorrect} ({incorrect/total*100:.1f}% si total > 0 else 0)
  🔄 À améliorer: {improve} ({improve/total*100:.1f}% si total > 0 else 0)

{'=' * 50}
PAR TYPE D'ACTION:
{'=' * 50}

"""

        for action_type, stats in action_stats.items():
            total_action = sum(stats.values())
            stats_text += f"\n{action_type}:\n"
            stats_text += f"  Total: {total_action}\n"
            stats_text += f"  ✅ Correct: {stats['correct']} ({stats['correct']/total_action*100:.1f}%)\n"
            stats_text += f"  ❌ Incorrect: {stats['incorrect']} ({stats['incorrect']/total_action*100:.1f}%)\n"
            stats_text += f"  🔄 À améliorer: {stats['improve']} ({stats['improve']/total_action*100:.1f}%)\n"

        text.insert(1.0, stats_text)
        text.config(state=tk.DISABLED)

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame du panneau"""
        return self.frame


def create_logs_learning_panel(parent) -> LogsLearningPanel:
    """Factory function"""
    return LogsLearningPanel(parent)
