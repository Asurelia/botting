"""
Notification System - Système de notifications avancé
Alertes, toasts, popups, historique
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import threading
import queue
import time


class NotificationType(Enum):
    """Type de notification"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationPriority(Enum):
    """Priorité de notification"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Notification:
    """Notification"""
    notification_id: str
    title: str
    message: str
    notification_type: NotificationType
    priority: NotificationPriority
    timestamp: datetime
    read: bool = False
    dismissed: bool = False
    action_callback: Optional[Callable] = None
    action_label: Optional[str] = None


class ToastNotification:
    """Widget de notification toast (temporaire)"""

    def __init__(self, parent, notification: Notification, on_close: Callable):
        self.parent = parent
        self.notification = notification
        self.on_close = on_close
        self.toplevel: Optional[tk.Toplevel] = None

        self.create_toast()

    def create_toast(self):
        """Crée le toast"""
        # Toplevel sans bordure
        self.toplevel = tk.Toplevel(self.parent)
        self.toplevel.overrideredirect(True)  # Pas de bordure
        self.toplevel.attributes('-topmost', True)  # Toujours au premier plan

        # Couleurs selon type
        colors = self._get_colors()

        # Frame principal
        main_frame = tk.Frame(
            self.toplevel,
            bg=colors['bg'],
            relief=tk.RAISED,
            borderwidth=2
        )
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header avec titre et bouton fermer
        header_frame = tk.Frame(main_frame, bg=colors['bg'])
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Icône
        icon = self._get_icon()
        icon_label = tk.Label(
            header_frame,
            text=icon,
            font=("Segoe UI", 12),
            bg=colors['bg'],
            fg=colors['fg']
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 5))

        # Titre
        title_label = tk.Label(
            header_frame,
            text=self.notification.title,
            font=("Segoe UI", 10, "bold"),
            bg=colors['bg'],
            fg=colors['fg']
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bouton fermer
        close_btn = tk.Label(
            header_frame,
            text="✕",
            font=("Segoe UI", 12),
            bg=colors['bg'],
            fg=colors['fg'],
            cursor="hand2"
        )
        close_btn.pack(side=tk.RIGHT)
        close_btn.bind("<Button-1>", lambda e: self.close())

        # Message
        message_label = tk.Label(
            main_frame,
            text=self.notification.message,
            font=("Segoe UI", 9),
            bg=colors['bg'],
            fg=colors['fg'],
            wraplength=250,
            justify=tk.LEFT
        )
        message_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Bouton d'action si présent
        if self.notification.action_callback and self.notification.action_label:
            action_btn = tk.Button(
                main_frame,
                text=self.notification.action_label,
                font=("Segoe UI", 9),
                bg=colors['action_bg'],
                fg="white",
                relief=tk.FLAT,
                cursor="hand2",
                command=self._execute_action
            )
            action_btn.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Position (coin inférieur droit)
        self.position_toast()

        # Auto-fermeture après délai
        duration = self._get_duration()
        self.toplevel.after(duration, self.close)

    def _get_colors(self) -> Dict[str, str]:
        """Retourne les couleurs selon le type"""
        colors_map = {
            NotificationType.INFO: {
                'bg': '#2196F3',
                'fg': 'white',
                'action_bg': '#1976D2'
            },
            NotificationType.SUCCESS: {
                'bg': '#4CAF50',
                'fg': 'white',
                'action_bg': '#388E3C'
            },
            NotificationType.WARNING: {
                'bg': '#FF9800',
                'fg': 'white',
                'action_bg': '#F57C00'
            },
            NotificationType.ERROR: {
                'bg': '#F44336',
                'fg': 'white',
                'action_bg': '#D32F2F'
            },
            NotificationType.CRITICAL: {
                'bg': '#9C27B0',
                'fg': 'white',
                'action_bg': '#7B1FA2'
            }
        }
        return colors_map.get(self.notification.notification_type, colors_map[NotificationType.INFO])

    def _get_icon(self) -> str:
        """Retourne l'icône selon le type"""
        icons = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.CRITICAL: "🔥"
        }
        return icons.get(self.notification.notification_type, "ℹ️")

    def _get_duration(self) -> int:
        """Retourne la durée d'affichage (ms)"""
        durations = {
            NotificationPriority.LOW: 3000,
            NotificationPriority.NORMAL: 5000,
            NotificationPriority.HIGH: 8000,
            NotificationPriority.URGENT: 0  # Ne se ferme pas auto
        }
        return durations.get(self.notification.priority, 5000)

    def position_toast(self):
        """Positionne le toast dans le coin inférieur droit"""
        self.toplevel.update_idletasks()
        width = self.toplevel.winfo_width()
        height = self.toplevel.winfo_height()
        screen_width = self.toplevel.winfo_screenwidth()
        screen_height = self.toplevel.winfo_screenheight()

        # Coin inférieur droit avec marge
        x = screen_width - width - 20
        y = screen_height - height - 60

        self.toplevel.geometry(f"+{x}+{y}")

    def _execute_action(self):
        """Exécute l'action de la notification"""
        if self.notification.action_callback:
            self.notification.action_callback()
        self.close()

    def close(self):
        """Ferme le toast"""
        if self.toplevel:
            self.toplevel.destroy()
            self.on_close(self.notification.notification_id)


class NotificationCenter:
    """Centre de notifications avec historique"""

    def __init__(self, parent):
        self.parent = parent
        self.notifications: List[Notification] = []
        self.active_toasts: Dict[str, ToastNotification] = {}
        self.notification_queue = queue.Queue()
        self.notification_id_counter = 0
        self.max_toasts = 3  # Nombre max de toasts affichés simultanément

        # Observer pattern
        self.observers: List[Callable] = []

        # Démarrer le thread de gestion des notifications
        self.running = True
        self.notification_thread = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        self.notification_thread.start()

    def _process_notifications(self):
        """Traite la queue de notifications"""
        while self.running:
            try:
                notification = self.notification_queue.get(timeout=0.1)

                # Ajouter à l'historique
                self.notifications.insert(0, notification)

                # Limiter l'historique à 100
                if len(self.notifications) > 100:
                    self.notifications.pop()

                # Afficher toast si pas trop de toasts actifs
                if len(self.active_toasts) < self.max_toasts:
                    self.parent.after(0, lambda n=notification: self._show_toast(n))

                # Notifier les observers
                for observer in self.observers:
                    self.parent.after(0, lambda o=observer, n=notification: o(n))

            except queue.Empty:
                continue

    def _show_toast(self, notification: Notification):
        """Affiche un toast"""
        if notification.notification_id not in self.active_toasts:
            toast = ToastNotification(
                self.parent,
                notification,
                self._on_toast_closed
            )
            self.active_toasts[notification.notification_id] = toast

    def _on_toast_closed(self, notification_id: str):
        """Callback quand un toast est fermé"""
        if notification_id in self.active_toasts:
            del self.active_toasts[notification_id]

    def notify(self,
              title: str,
              message: str,
              notification_type: NotificationType = NotificationType.INFO,
              priority: NotificationPriority = NotificationPriority.NORMAL,
              action_callback: Optional[Callable] = None,
              action_label: Optional[str] = None):
        """Envoie une notification"""
        self.notification_id_counter += 1
        notification = Notification(
            notification_id=f"notif_{self.notification_id_counter}",
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            timestamp=datetime.now(),
            action_callback=action_callback,
            action_label=action_label
        )

        self.notification_queue.put(notification)

    def add_observer(self, callback: Callable):
        """Ajoute un observer"""
        self.observers.append(callback)

    def remove_observer(self, callback: Callable):
        """Retire un observer"""
        if callback in self.observers:
            self.observers.remove(callback)

    def get_notifications(self,
                         unread_only: bool = False,
                         notification_type: Optional[NotificationType] = None) -> List[Notification]:
        """Retourne les notifications filtrées"""
        filtered = self.notifications

        if unread_only:
            filtered = [n for n in filtered if not n.read]

        if notification_type:
            filtered = [n for n in filtered if n.notification_type == notification_type]

        return filtered

    def mark_as_read(self, notification_id: str):
        """Marque une notification comme lue"""
        for notif in self.notifications:
            if notif.notification_id == notification_id:
                notif.read = True
                break

    def mark_all_as_read(self):
        """Marque toutes les notifications comme lues"""
        for notif in self.notifications:
            notif.read = False

    def dismiss(self, notification_id: str):
        """Rejette une notification"""
        for notif in self.notifications:
            if notif.notification_id == notification_id:
                notif.dismissed = True
                break

    def clear_all(self):
        """Efface toutes les notifications"""
        self.notifications.clear()

    def get_unread_count(self) -> int:
        """Retourne le nombre de notifications non lues"""
        return sum(1 for n in self.notifications if not n.read)

    def stop(self):
        """Arrête le système de notifications"""
        self.running = False


class NotificationPanel:
    """Panel d'affichage de l'historique des notifications"""

    def __init__(self, parent, notification_center: NotificationCenter):
        self.parent = parent
        self.notification_center = notification_center

        self._setup_ui()

        # S'abonner aux nouvelles notifications
        self.notification_center.add_observer(self.on_new_notification)

    def _setup_ui(self):
        """Configure l'interface"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="🔔 Centre de Notifications",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        # Badge non lues
        self.unread_badge = ttk.Label(header_frame, text="0",
                                      font=("Segoe UI", 9, "bold"))
        self.unread_badge.pack(side=tk.LEFT, padx=(10, 0))

        # Boutons
        ttk.Button(header_frame, text="✓ Tout lire",
                  command=self.mark_all_read).pack(side=tk.RIGHT, padx=2)
        ttk.Button(header_frame, text="🗑️ Tout effacer",
                  command=self.clear_all).pack(side=tk.RIGHT, padx=2)

        # Filtres
        filter_frame = ttk.Frame(self.main_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="Filtrer:").pack(side=tk.LEFT, padx=(0, 5))

        self.filter_var = tk.StringVar(value="Toutes")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                    values=["Toutes", "Non lues", "Info", "Succès", "Avertissement", "Erreur", "Critique"],
                                    state="readonly", width=15)
        filter_combo.pack(side=tk.LEFT)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self.update_list())

        # Liste des notifications
        list_frame = ttk.Frame(self.main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.notifications_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 9),
            selectmode=tk.SINGLE
        )
        self.notifications_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.notifications_listbox.yview)
        self.notifications_listbox.bind("<<ListboxSelect>>", self.on_select)

        # Détails notification
        details_frame = ttk.LabelFrame(self.main_frame, text="Détails", padding=10)
        details_frame.pack(fill=tk.X, pady=(10, 0))

        self.details_text = tk.Text(details_frame, height=8, font=("Segoe UI", 9),
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.details_text.pack(fill=tk.BOTH, expand=True)

        # Mettre à jour l'affichage
        self.update_list()
        self.update_badge()

    def update_list(self):
        """Met à jour la liste des notifications"""
        self.notifications_listbox.delete(0, tk.END)

        # Filtrer
        filter_value = self.filter_var.get()
        notifications = self.notification_center.notifications

        if filter_value == "Non lues":
            notifications = [n for n in notifications if not n.read]
        elif filter_value in ["Info", "Succès", "Avertissement", "Erreur", "Critique"]:
            type_map = {
                "Info": NotificationType.INFO,
                "Succès": NotificationType.SUCCESS,
                "Avertissement": NotificationType.WARNING,
                "Erreur": NotificationType.ERROR,
                "Critique": NotificationType.CRITICAL
            }
            notifications = [n for n in notifications if n.notification_type == type_map[filter_value]]

        # Afficher
        for notif in notifications:
            icon = self._get_icon(notif.notification_type)
            time_str = notif.timestamp.strftime("%H:%M")
            read_mark = " " if notif.read else "●"

            display = f"{read_mark} {icon} {time_str} - {notif.title}"
            self.notifications_listbox.insert(tk.END, display)

    def update_badge(self):
        """Met à jour le badge de notifications non lues"""
        count = self.notification_center.get_unread_count()
        self.unread_badge.config(text=f"{count} non lues")

    def _get_icon(self, notification_type: NotificationType) -> str:
        """Retourne l'icône selon le type"""
        icons = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.CRITICAL: "🔥"
        }
        return icons.get(notification_type, "ℹ️")

    def on_select(self, event):
        """Gère la sélection d'une notification"""
        selection = self.notifications_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        notifications = self._get_filtered_notifications()

        if index < len(notifications):
            notif = notifications[index]

            # Marquer comme lue
            self.notification_center.mark_as_read(notif.notification_id)
            self.update_list()
            self.update_badge()

            # Afficher détails
            details = f"Titre: {notif.title}\n"
            details += f"Type: {notif.notification_type.value}\n"
            details += f"Priorité: {notif.priority.name}\n"
            details += f"Heure: {notif.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            details += f"\nMessage:\n{notif.message}"

            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete("1.0", tk.END)
            self.details_text.insert("1.0", details)
            self.details_text.config(state=tk.DISABLED)

    def _get_filtered_notifications(self) -> List[Notification]:
        """Retourne les notifications filtrées"""
        filter_value = self.filter_var.get()
        notifications = self.notification_center.notifications

        if filter_value == "Non lues":
            notifications = [n for n in notifications if not n.read]
        elif filter_value in ["Info", "Succès", "Avertissement", "Erreur", "Critique"]:
            type_map = {
                "Info": NotificationType.INFO,
                "Succès": NotificationType.SUCCESS,
                "Avertissement": NotificationType.WARNING,
                "Erreur": NotificationType.ERROR,
                "Critique": NotificationType.CRITICAL
            }
            notifications = [n for n in notifications if n.notification_type == type_map[filter_value]]

        return notifications

    def on_new_notification(self, notification: Notification):
        """Callback pour nouvelle notification"""
        self.update_list()
        self.update_badge()

    def mark_all_read(self):
        """Marque toutes les notifications comme lues"""
        self.notification_center.mark_all_as_read()
        self.update_list()
        self.update_badge()

    def clear_all(self):
        """Efface toutes les notifications"""
        from tkinter import messagebox
        if messagebox.askyesno("Confirmer", "Effacer toutes les notifications?"):
            self.notification_center.clear_all()
            self.update_list()
            self.update_badge()

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame


# Fonction helper pour créer le système de notifications globalement
_global_notification_center: Optional[NotificationCenter] = None


def initialize_notifications(root):
    """Initialise le système de notifications global"""
    global _global_notification_center
    _global_notification_center = NotificationCenter(root)
    return _global_notification_center


def get_notification_center() -> Optional[NotificationCenter]:
    """Retourne le centre de notifications global"""
    return _global_notification_center


def notify(title: str, message: str,
          notification_type: NotificationType = NotificationType.INFO,
          priority: NotificationPriority = NotificationPriority.NORMAL,
          action_callback: Optional[Callable] = None,
          action_label: Optional[str] = None):
    """Fonction helper pour envoyer une notification"""
    if _global_notification_center:
        _global_notification_center.notify(
            title, message, notification_type, priority,
            action_callback, action_label
        )
