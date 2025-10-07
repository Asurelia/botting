"""
Keyboard Shortcuts Manager - Gestionnaire de raccourcis clavier
Configuration, détection, gestion des raccourcis globaux
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
import json
from pathlib import Path


class ShortcutCategory(Enum):
    """Catégorie de raccourci"""
    BOT_CONTROL = "Contrôle Bot"
    NAVIGATION = "Navigation"
    COMBAT = "Combat"
    ECONOMY = "Économie"
    INTERFACE = "Interface"
    VISION = "Vision"
    LEARNING = "Apprentissage"
    UTILITY = "Utilitaire"


@dataclass
class KeyboardShortcut:
    """Raccourci clavier"""
    shortcut_id: str
    name: str
    description: str
    category: ShortcutCategory
    keys: str  # Format: "Ctrl+Shift+A"
    callback: Optional[Callable]
    enabled: bool = True
    global_hotkey: bool = False  # Si True, fonctionne même hors focus


class KeySequenceRecorder:
    """Widget pour enregistrer une séquence de touches"""

    def __init__(self, parent, on_record: Callable):
        self.parent = parent
        self.on_record = on_record
        self.recording = False
        self.keys_pressed: Set[str] = set()

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface"""
        self.frame = ttk.Frame(self.parent)

        self.entry = ttk.Entry(self.frame, state="readonly", width=30)
        self.entry.pack(side=tk.LEFT, padx=(0, 5))

        self.record_button = ttk.Button(
            self.frame,
            text="🎤 Enregistrer",
            command=self.start_recording
        )
        self.record_button.pack(side=tk.LEFT)

        # Bind événements clavier
        self.entry.bind("<KeyPress>", self.on_key_press)
        self.entry.bind("<KeyRelease>", self.on_key_release)

    def start_recording(self):
        """Démarre l'enregistrement"""
        self.recording = True
        self.keys_pressed.clear()
        self.entry.config(state="normal")
        self.entry.delete(0, tk.END)
        self.entry.insert(0, "Appuyez sur les touches...")
        self.entry.focus_set()
        self.record_button.config(text="⏹️ Arrêter", command=self.stop_recording)

    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.recording = False
        self.entry.config(state="readonly")
        self.record_button.config(text="🎤 Enregistrer", command=self.start_recording)

        # Callback avec la séquence enregistrée
        if self.keys_pressed:
            sequence = self._format_keys()
            self.on_record(sequence)

    def on_key_press(self, event):
        """Touche pressée"""
        if not self.recording:
            return

        # Ajouter la touche
        key = self._normalize_key(event.keysym)
        self.keys_pressed.add(key)

        # Afficher
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._format_keys())

    def on_key_release(self, event):
        """Touche relâchée"""
        if not self.recording:
            return

        # Si toutes les touches sont relâchées, arrêter
        # (Simplifié - en réalité il faudrait tracker chaque touche)

    def _normalize_key(self, keysym: str) -> str:
        """Normalise une touche"""
        # Mapper les noms de touches
        key_map = {
            "Control_L": "Ctrl",
            "Control_R": "Ctrl",
            "Shift_L": "Shift",
            "Shift_R": "Shift",
            "Alt_L": "Alt",
            "Alt_R": "Alt",
            "Super_L": "Win",
            "Super_R": "Win"
        }
        return key_map.get(keysym, keysym)

    def _format_keys(self) -> str:
        """Formate la séquence de touches"""
        # Ordre: Ctrl, Shift, Alt, Win, puis lettre
        modifiers = []
        letters = []

        for key in self.keys_pressed:
            if key in ["Ctrl", "Shift", "Alt", "Win"]:
                modifiers.append(key)
            else:
                letters.append(key)

        # Trier les modifiers
        modifier_order = ["Ctrl", "Shift", "Alt", "Win"]
        sorted_modifiers = [m for m in modifier_order if m in modifiers]

        return "+".join(sorted_modifiers + letters)

    def get_frame(self) -> ttk.Frame:
        """Retourne le frame"""
        return self.frame


class KeyboardShortcutsManager:
    """Gestionnaire de raccourcis clavier"""

    def __init__(self, root):
        self.root = root
        self.shortcuts: Dict[str, KeyboardShortcut] = {}
        self.key_bindings: Dict[str, str] = {}  # keys -> shortcut_id
        self.config_file = Path("config/keyboard_shortcuts.json")

        # Charger la configuration
        self.load_config()

        # Créer les raccourcis par défaut si aucun
        if not self.shortcuts:
            self.create_default_shortcuts()

    def create_default_shortcuts(self):
        """Crée les raccourcis par défaut"""
        default_shortcuts = [
            # Contrôle Bot
            ("bot_start", "Démarrer bot", "Démarre l'exécution du bot",
             ShortcutCategory.BOT_CONTROL, "Ctrl+Shift+S", None),
            ("bot_stop", "Arrêter bot", "Arrête le bot",
             ShortcutCategory.BOT_CONTROL, "Ctrl+Shift+X", None),
            ("bot_pause", "Pause/Reprendre", "Met en pause ou reprend le bot",
             ShortcutCategory.BOT_CONTROL, "Ctrl+Shift+P", None),
            ("emergency_stop", "Arrêt d'urgence", "Arrête immédiatement tout",
             ShortcutCategory.BOT_CONTROL, "Ctrl+Shift+E", None),

            # Navigation
            ("nav_waypoint", "Aller au waypoint", "Navigation vers waypoint sélectionné",
             ShortcutCategory.NAVIGATION, "Ctrl+G", None),
            ("nav_back", "Retour position", "Retourne à la position précédente",
             ShortcutCategory.NAVIGATION, "Ctrl+B", None),
            ("nav_refresh", "Rafraîchir position", "Rafraîchit la position actuelle",
             ShortcutCategory.NAVIGATION, "F5", None),

            # Combat
            ("combat_flee", "Fuir combat", "Fuit le combat en cours",
             ShortcutCategory.COMBAT, "Escape", None),
            ("combat_analyze", "Analyser combat", "Ouvre l'analyse du dernier combat",
             ShortcutCategory.COMBAT, "Ctrl+A", None),

            # Économie
            ("economy_hdv", "Ouvrir HDV", "Ouvre le panel HDV",
             ShortcutCategory.ECONOMY, "Ctrl+H", None),
            ("economy_inventory", "Ouvrir inventaire", "Ouvre la gestion d'inventaire",
             ShortcutCategory.ECONOMY, "Ctrl+I", None),

            # Interface
            ("ui_fullscreen", "Plein écran", "Bascule en plein écran",
             ShortcutCategory.INTERFACE, "F11", None),
            ("ui_dashboard", "Dashboard", "Affiche le dashboard",
             ShortcutCategory.INTERFACE, "Ctrl+1", None),
            ("ui_vision", "Panel Vision", "Affiche le panel Vision",
             ShortcutCategory.INTERFACE, "Ctrl+2", None),
            ("ui_combat", "Panel Combat", "Affiche le panel Combat",
             ShortcutCategory.INTERFACE, "Ctrl+3", None),
            ("ui_economy", "Panel Économie", "Affiche le panel Économie",
             ShortcutCategory.INTERFACE, "Ctrl+4", None),
            ("ui_navigation", "Panel Navigation", "Affiche le panel Navigation",
             ShortcutCategory.INTERFACE, "Ctrl+5", None),

            # Vision
            ("vision_capture", "Capture écran", "Prend un screenshot",
             ShortcutCategory.VISION, "Ctrl+Shift+C", None),
            ("vision_toggle_debug", "Toggle debug vision", "Active/désactive le mode debug vision",
             ShortcutCategory.VISION, "Ctrl+D", None),

            # Apprentissage
            ("learning_correct", "Marquer correct", "Marque la décision comme correcte",
             ShortcutCategory.LEARNING, "Ctrl+Shift+Y", None),
            ("learning_incorrect", "Marquer incorrect", "Marque la décision comme incorrecte",
             ShortcutCategory.LEARNING, "Ctrl+Shift+N", None),

            # Utilitaire
            ("util_refresh", "Rafraîchir tout", "Rafraîchit toutes les données",
             ShortcutCategory.UTILITY, "F5", None),
            ("util_settings", "Paramètres", "Ouvre les paramètres",
             ShortcutCategory.UTILITY, "Ctrl+,", None),
            ("util_help", "Aide", "Affiche l'aide",
             ShortcutCategory.UTILITY, "F1", None),
        ]

        for shortcut_id, name, desc, category, keys, callback in default_shortcuts:
            self.register_shortcut(shortcut_id, name, desc, category, keys, callback)

        # Sauvegarder
        self.save_config()

    def register_shortcut(self,
                         shortcut_id: str,
                         name: str,
                         description: str,
                         category: ShortcutCategory,
                         keys: str,
                         callback: Optional[Callable],
                         enabled: bool = True,
                         global_hotkey: bool = False):
        """Enregistre un raccourci"""
        shortcut = KeyboardShortcut(
            shortcut_id=shortcut_id,
            name=name,
            description=description,
            category=category,
            keys=keys,
            callback=callback,
            enabled=enabled,
            global_hotkey=global_hotkey
        )

        self.shortcuts[shortcut_id] = shortcut

        # Bind si callback fourni
        if callback and enabled:
            self._bind_shortcut(shortcut)

    def _bind_shortcut(self, shortcut: KeyboardShortcut):
        """Bind un raccourci"""
        # Convertir format "Ctrl+Shift+A" en format Tkinter "<Control-Shift-A>"
        tk_binding = self._convert_to_tk_binding(shortcut.keys)

        # Unbind si déjà bindé
        if shortcut.keys in self.key_bindings:
            old_binding = self._convert_to_tk_binding(shortcut.keys)
            try:
                self.root.unbind(old_binding)
            except:
                pass

        # Bind
        try:
            self.root.bind(tk_binding, lambda e: self._execute_shortcut(shortcut.shortcut_id))
            self.key_bindings[shortcut.keys] = shortcut.shortcut_id
        except Exception as e:
            print(f"Erreur binding raccourci {shortcut.name}: {e}")

    def _convert_to_tk_binding(self, keys: str) -> str:
        """Convertit le format de touches en binding Tkinter"""
        # "Ctrl+Shift+A" -> "<Control-Shift-A>"
        parts = keys.split("+")
        tk_parts = []

        for part in parts:
            if part == "Ctrl":
                tk_parts.append("Control")
            elif part == "Win":
                tk_parts.append("Super")
            else:
                tk_parts.append(part)

        return f"<{'-'.join(tk_parts)}>"

    def _execute_shortcut(self, shortcut_id: str):
        """Exécute un raccourci"""
        if shortcut_id in self.shortcuts:
            shortcut = self.shortcuts[shortcut_id]
            if shortcut.enabled and shortcut.callback:
                try:
                    shortcut.callback()
                except Exception as e:
                    print(f"Erreur exécution raccourci {shortcut.name}: {e}")

    def update_shortcut_keys(self, shortcut_id: str, new_keys: str):
        """Met à jour les touches d'un raccourci"""
        if shortcut_id in self.shortcuts:
            shortcut = self.shortcuts[shortcut_id]

            # Unbind ancien
            if shortcut.keys in self.key_bindings:
                old_binding = self._convert_to_tk_binding(shortcut.keys)
                try:
                    self.root.unbind(old_binding)
                    del self.key_bindings[shortcut.keys]
                except:
                    pass

            # Mettre à jour
            shortcut.keys = new_keys

            # Re-bind
            if shortcut.callback and shortcut.enabled:
                self._bind_shortcut(shortcut)

            self.save_config()

    def enable_shortcut(self, shortcut_id: str):
        """Active un raccourci"""
        if shortcut_id in self.shortcuts:
            shortcut = self.shortcuts[shortcut_id]
            shortcut.enabled = True
            if shortcut.callback:
                self._bind_shortcut(shortcut)
            self.save_config()

    def disable_shortcut(self, shortcut_id: str):
        """Désactive un raccourci"""
        if shortcut_id in self.shortcuts:
            shortcut = self.shortcuts[shortcut_id]
            shortcut.enabled = False

            # Unbind
            if shortcut.keys in self.key_bindings:
                old_binding = self._convert_to_tk_binding(shortcut.keys)
                try:
                    self.root.unbind(old_binding)
                    del self.key_bindings[shortcut.keys]
                except:
                    pass

            self.save_config()

    def load_config(self):
        """Charge la configuration depuis JSON"""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for shortcut_data in data.get('shortcuts', []):
                # Recréer le shortcut (sans callback pour l'instant)
                self.register_shortcut(
                    shortcut_id=shortcut_data['shortcut_id'],
                    name=shortcut_data['name'],
                    description=shortcut_data['description'],
                    category=ShortcutCategory[shortcut_data['category']],
                    keys=shortcut_data['keys'],
                    callback=None,  # Les callbacks seront réassignés par l'app
                    enabled=shortcut_data['enabled'],
                    global_hotkey=shortcut_data.get('global_hotkey', False)
                )

        except Exception as e:
            print(f"Erreur chargement raccourcis: {e}")

    def save_config(self):
        """Sauvegarde la configuration en JSON"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'shortcuts': [
                {
                    'shortcut_id': s.shortcut_id,
                    'name': s.name,
                    'description': s.description,
                    'category': s.category.name,
                    'keys': s.keys,
                    'enabled': s.enabled,
                    'global_hotkey': s.global_hotkey
                }
                for s in self.shortcuts.values()
            ]
        }

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde raccourcis: {e}")

    def get_shortcuts_by_category(self, category: ShortcutCategory) -> List[KeyboardShortcut]:
        """Retourne les raccourcis d'une catégorie"""
        return [s for s in self.shortcuts.values() if s.category == category]

    def reset_to_defaults(self):
        """Réinitialise aux raccourcis par défaut"""
        # Unbind tous les raccourcis actuels
        for shortcut in self.shortcuts.values():
            if shortcut.keys in self.key_bindings:
                old_binding = self._convert_to_tk_binding(shortcut.keys)
                try:
                    self.root.unbind(old_binding)
                except:
                    pass

        self.shortcuts.clear()
        self.key_bindings.clear()
        self.create_default_shortcuts()


class ShortcutsConfigPanel:
    """Panel de configuration des raccourcis"""

    def __init__(self, parent, shortcuts_manager: KeyboardShortcutsManager):
        self.parent = parent
        self.shortcuts_manager = shortcuts_manager
        self.current_shortcut: Optional[KeyboardShortcut] = None

        self._setup_ui()

    def _setup_ui(self):
        """Configure l'interface"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="⌨️ Gestionnaire de Raccourcis Clavier",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        ttk.Button(header_frame, text="🔄 Réinitialiser",
                  command=self.reset_to_defaults).pack(side=tk.RIGHT, padx=2)
        ttk.Button(header_frame, text="💾 Sauvegarder",
                  command=self.save_config).pack(side=tk.RIGHT, padx=2)

        # PanedWindow
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Panneau gauche: Liste des raccourcis
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # Filtres par catégorie
        filter_frame = ttk.LabelFrame(left_frame, text="Catégories", padding=5)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        self.category_var = tk.StringVar(value="Toutes")
        categories = ["Toutes"] + [cat.value for cat in ShortcutCategory]

        for i, category in enumerate(categories):
            ttk.Radiobutton(
                filter_frame,
                text=category,
                variable=self.category_var,
                value=category,
                command=self.update_shortcuts_list
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)

        # Liste des raccourcis
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.shortcuts_tree = ttk.Treeview(
            list_frame,
            columns=("keys", "status"),
            show="tree headings",
            yscrollcommand=scrollbar.set
        )
        self.shortcuts_tree.heading("keys", text="Touches")
        self.shortcuts_tree.heading("status", text="Statut")
        self.shortcuts_tree.column("#0", width=200)
        self.shortcuts_tree.column("keys", width=150)
        self.shortcuts_tree.column("status", width=80)
        self.shortcuts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.shortcuts_tree.yview)
        self.shortcuts_tree.bind("<<TreeviewSelect>>", self.on_shortcut_select)

        self.update_shortcuts_list()

        # Panneau droit: Configuration
        right_frame = ttk.LabelFrame(paned, text="Configuration", padding=10)
        paned.add(right_frame, weight=1)

        # Infos raccourci
        info_frame = ttk.LabelFrame(right_frame, text="Informations", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_frame, text="Nom:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_label = ttk.Label(info_frame, text="-")
        self.name_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        ttk.Label(info_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.desc_label = ttk.Label(info_frame, text="-", wraplength=300)
        self.desc_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        ttk.Label(info_frame, text="Catégorie:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.category_label = ttk.Label(info_frame, text="-")
        self.category_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Configuration touches
        keys_frame = ttk.LabelFrame(right_frame, text="Touches", padding=10)
        keys_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(keys_frame, text="Combinaison actuelle:").pack(anchor=tk.W, pady=(0, 5))

        # Recorder
        self.key_recorder = KeySequenceRecorder(keys_frame, self.on_keys_recorded)
        self.key_recorder.get_frame().pack(fill=tk.X, pady=(0, 10))

        # Bouton appliquer
        ttk.Button(keys_frame, text="✓ Appliquer nouveau raccourci",
                  command=self.apply_new_keys).pack(fill=tk.X)

        # Options
        options_frame = ttk.LabelFrame(right_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Activé",
                       variable=self.enabled_var,
                       command=self.toggle_enabled).pack(anchor=tk.W, pady=2)

        self.global_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Raccourci global (hors focus)",
                       variable=self.global_var,
                       state="disabled").pack(anchor=tk.W, pady=2)

        # Aide
        help_frame = ttk.LabelFrame(right_frame, text="💡 Aide", padding=10)
        help_frame.pack(fill=tk.BOTH, expand=True)

        help_text = """Utilisation:

1. Sélectionnez un raccourci dans la liste
2. Cliquez sur "Enregistrer" pour définir une nouvelle combinaison
3. Appuyez sur les touches souhaitées
4. Cliquez sur "Appliquer"

Modificateurs disponibles:
• Ctrl, Shift, Alt, Win
• Lettres A-Z
• Touches F1-F12
• Escape, Tab, Space, etc.

Les raccourcis sont sauvegardés automatiquement.
"""
        help_label = ttk.Label(help_frame, text=help_text,
                              justify=tk.LEFT, wraplength=350)
        help_label.pack(fill=tk.BOTH, expand=True)

    def update_shortcuts_list(self):
        """Met à jour la liste des raccourcis"""
        self.shortcuts_tree.delete(*self.shortcuts_tree.get_children())

        category_filter = self.category_var.get()

        # Grouper par catégorie
        categories = {}
        for shortcut in self.shortcuts_manager.shortcuts.values():
            if category_filter != "Toutes" and shortcut.category.value != category_filter:
                continue

            cat_name = shortcut.category.value
            if cat_name not in categories:
                categories[cat_name] = []
            categories[cat_name].append(shortcut)

        # Afficher
        for cat_name, shortcuts in sorted(categories.items()):
            cat_node = self.shortcuts_tree.insert("", tk.END, text=f"📁 {cat_name}", open=True)

            for shortcut in sorted(shortcuts, key=lambda s: s.name):
                status = "✅ Actif" if shortcut.enabled else "❌ Inactif"
                self.shortcuts_tree.insert(cat_node, tk.END,
                                          text=shortcut.name,
                                          values=(shortcut.keys, status),
                                          tags=(shortcut.shortcut_id,))

    def on_shortcut_select(self, event):
        """Gère la sélection d'un raccourci"""
        selection = self.shortcuts_tree.selection()
        if not selection:
            return

        item = self.shortcuts_tree.item(selection[0])
        tags = item.get('tags', [])

        if not tags:
            return

        shortcut_id = tags[0]
        if shortcut_id in self.shortcuts_manager.shortcuts:
            self.current_shortcut = self.shortcuts_manager.shortcuts[shortcut_id]
            self.display_shortcut_info()

    def display_shortcut_info(self):
        """Affiche les infos du raccourci sélectionné"""
        if not self.current_shortcut:
            return

        self.name_label.config(text=self.current_shortcut.name)
        self.desc_label.config(text=self.current_shortcut.description)
        self.category_label.config(text=self.current_shortcut.category.value)
        self.enabled_var.set(self.current_shortcut.enabled)
        self.global_var.set(self.current_shortcut.global_hotkey)

        # Mettre à jour le recorder
        self.key_recorder.entry.config(state="normal")
        self.key_recorder.entry.delete(0, tk.END)
        self.key_recorder.entry.insert(0, self.current_shortcut.keys)
        self.key_recorder.entry.config(state="readonly")

    def on_keys_recorded(self, keys: str):
        """Callback quand de nouvelles touches sont enregistrées"""
        self.new_keys = keys
        self.key_recorder.entry.delete(0, tk.END)
        self.key_recorder.entry.insert(0, keys)

    def apply_new_keys(self):
        """Applique les nouvelles touches"""
        if not self.current_shortcut or not hasattr(self, 'new_keys'):
            messagebox.showwarning("Aucune touche", "Veuillez d'abord enregistrer une combinaison")
            return

        # Vérifier si déjà utilisé
        for s in self.shortcuts_manager.shortcuts.values():
            if s.keys == self.new_keys and s.shortcut_id != self.current_shortcut.shortcut_id:
                messagebox.showwarning("Conflit",
                                      f"Ce raccourci est déjà utilisé par:\n{s.name}")
                return

        # Appliquer
        self.shortcuts_manager.update_shortcut_keys(
            self.current_shortcut.shortcut_id,
            self.new_keys
        )

        self.update_shortcuts_list()
        messagebox.showinfo("Appliqué", "Raccourci modifié avec succès")

    def toggle_enabled(self):
        """Active/désactive le raccourci"""
        if not self.current_shortcut:
            return

        if self.enabled_var.get():
            self.shortcuts_manager.enable_shortcut(self.current_shortcut.shortcut_id)
        else:
            self.shortcuts_manager.disable_shortcut(self.current_shortcut.shortcut_id)

        self.update_shortcuts_list()

    def save_config(self):
        """Sauvegarde la configuration"""
        self.shortcuts_manager.save_config()
        messagebox.showinfo("Sauvegardé", "Configuration sauvegardée")

    def reset_to_defaults(self):
        """Réinitialise aux raccourcis par défaut"""
        if messagebox.askyesno("Confirmer", "Réinitialiser tous les raccourcis aux valeurs par défaut?"):
            self.shortcuts_manager.reset_to_defaults()
            self.update_shortcuts_list()
            messagebox.showinfo("Réinitialisé", "Raccourcis réinitialisés")

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
