#!/usr/bin/env python3
"""
Test direct des themes sans imports externes
Copie simplifiee du theme manager pour test independant
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod

@dataclass
class ColorScheme:
    """Schema de couleurs"""
    # Couleurs primaires
    primary: str
    primary_dark: str
    primary_light: str

    # Couleurs secondaires
    secondary: str
    secondary_dark: str
    secondary_light: str

    # Couleurs d'arriere-plan
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str

    # Couleurs de texte
    text_primary: str
    text_secondary: str
    text_disabled: str

    # Couleurs d'accent
    accent_success: str
    accent_warning: str
    accent_error: str
    accent_info: str

    # Couleurs de bordure
    border_light: str
    border_medium: str
    border_dark: str

class BaseTheme(ABC):
    """Classe de base pour les themes"""

    def __init__(self):
        self.colors = self.get_color_scheme()
        self.fonts = self.get_font_scheme()
        self.styles = self.get_style_definitions()

    @abstractmethod
    def get_color_scheme(self) -> ColorScheme:
        """Retourne le schema de couleurs"""
        pass

    @abstractmethod
    def get_font_scheme(self) -> Dict[str, tuple]:
        """Retourne le schema de polices"""
        pass

    def get_style_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les definitions de styles"""
        return {
            "title": {
                "font": self.fonts["title"],
                "fg": self.colors.text_primary,
                "bg": self.colors.bg_primary
            },
            "subtitle": {
                "font": self.fonts["subtitle"],
                "fg": self.colors.text_secondary,
                "bg": self.colors.bg_primary
            },
            "body": {
                "font": self.fonts["body"],
                "fg": self.colors.text_primary,
                "bg": self.colors.bg_primary
            },
            "button_primary": {
                "font": self.fonts["button"],
                "fg": "white",
                "bg": self.colors.primary,
                "activebackground": self.colors.primary_dark,
                "relief": "flat",
                "bd": 0,
                "padx": 20,
                "pady": 8
            },
            "button_secondary": {
                "font": self.fonts["button"],
                "fg": self.colors.text_primary,
                "bg": self.colors.bg_secondary,
                "activebackground": self.colors.bg_tertiary,
                "relief": "flat",
                "bd": 0,
                "padx": 20,
                "pady": 8
            },
            "entry": {
                "font": self.fonts["body"],
                "fg": self.colors.text_primary,
                "bg": self.colors.bg_secondary,
                "insertbackground": self.colors.text_primary,
                "relief": "flat",
                "bd": 1,
                "highlightthickness": 2,
                "highlightcolor": self.colors.primary,
                "highlightbackground": self.colors.border_light
            },
            "frame_primary": {
                "bg": self.colors.bg_primary,
                "relief": "flat",
                "bd": 0
            },
            "frame_secondary": {
                "bg": self.colors.bg_secondary,
                "relief": "flat",
                "bd": 0
            },
            "panel": {
                "bg": self.colors.bg_secondary,
                "relief": "flat",
                "bd": 1,
                "highlightthickness": 1,
                "highlightcolor": self.colors.border_light,
                "highlightbackground": self.colors.border_light
            }
        }

class DarkTheme(BaseTheme):
    """Theme sombre moderne"""

    def get_color_scheme(self) -> ColorScheme:
        return ColorScheme(
            # Primaires - Bleu AlphaStar
            primary="#2563eb",
            primary_dark="#1d4ed8",
            primary_light="#3b82f6",

            # Secondaires - Violet
            secondary="#7c3aed",
            secondary_dark="#6d28d9",
            secondary_light="#8b5cf6",

            # Arriere-plans sombres
            bg_primary="#0f172a",      # Tres sombre
            bg_secondary="#1e293b",    # Sombre moyen
            bg_tertiary="#334155",     # Moins sombre

            # Textes
            text_primary="#f8fafc",    # Blanc casse
            text_secondary="#cbd5e1",  # Gris clair
            text_disabled="#64748b",   # Gris moyen

            # Accents
            accent_success="#10b981",  # Vert
            accent_warning="#f59e0b",  # Orange
            accent_error="#ef4444",    # Rouge
            accent_info="#06b6d4",     # Cyan

            # Bordures
            border_light="#475569",
            border_medium="#334155",
            border_dark="#1e293b"
        )

    def get_font_scheme(self) -> Dict[str, tuple]:
        return {
            "title": ("Segoe UI", 18, "bold"),
            "subtitle": ("Segoe UI", 14, "bold"),
            "body": ("Segoe UI", 10, "normal"),
            "button": ("Segoe UI", 10, "bold"),
            "code": ("Consolas", 9, "normal"),
            "status": ("Segoe UI", 9, "normal")
        }

class LightTheme(BaseTheme):
    """Theme clair moderne"""

    def get_color_scheme(self) -> ColorScheme:
        return ColorScheme(
            # Primaires - Bleu
            primary="#2563eb",
            primary_dark="#1d4ed8",
            primary_light="#3b82f6",

            # Secondaires
            secondary="#7c3aed",
            secondary_dark="#6d28d9",
            secondary_light="#8b5cf6",

            # Arriere-plans clairs
            bg_primary="#ffffff",
            bg_secondary="#f8fafc",
            bg_tertiary="#e2e8f0",

            # Textes
            text_primary="#0f172a",
            text_secondary="#475569",
            text_disabled="#94a3b8",

            # Accents
            accent_success="#10b981",
            accent_warning="#f59e0b",
            accent_error="#ef4444",
            accent_info="#06b6d4",

            # Bordures
            border_light="#e2e8f0",
            border_medium="#cbd5e1",
            border_dark="#94a3b8"
        )

    def get_font_scheme(self) -> Dict[str, tuple]:
        return {
            "title": ("Segoe UI", 18, "bold"),
            "subtitle": ("Segoe UI", 14, "bold"),
            "body": ("Segoe UI", 10, "normal"),
            "button": ("Segoe UI", 10, "bold"),
            "code": ("Consolas", 9, "normal"),
            "status": ("Segoe UI", 9, "normal")
        }

class SimpleThemeManager:
    """Gestionnaire de themes simplifie"""

    def __init__(self):
        self.themes = {
            "dark": DarkTheme(),
            "light": LightTheme()
        }
        self.current_theme_name = "dark"
        self.current_theme = self.themes[self.current_theme_name]
        self.root = None

    def set_theme(self, theme_name: str):
        """Change le theme actuel"""
        if theme_name in self.themes:
            self.current_theme_name = theme_name
            self.current_theme = self.themes[theme_name]

            if self.root:
                self.apply_theme_to_root()

    def apply_theme_to_root(self):
        """Applique le theme a la fenetre racine"""
        if self.root:
            self.root.configure(bg=self.current_theme.colors.bg_primary)

    def register_root(self, root: tk.Tk):
        """Enregistre la fenetre racine"""
        self.root = root
        self.apply_theme_to_root()

    def get_style(self, style_name: str) -> Dict[str, Any]:
        """Retourne un style specifique"""
        return self.current_theme.styles.get(style_name, {})

    def create_styled_widget(self, widget_class, parent, style_name: str, **kwargs):
        """Cree un widget avec style applique"""
        style = self.get_style(style_name)

        # Fusionner style avec kwargs
        final_kwargs = {**style, **kwargs}

        return widget_class(parent, **final_kwargs)

    def create_title_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Cree un label de titre style"""
        return self.create_styled_widget(tk.Label, parent, "title", text=text, **kwargs)

    def create_subtitle_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Cree un label de sous-titre style"""
        return self.create_styled_widget(tk.Label, parent, "subtitle", text=text, **kwargs)

    def create_body_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Cree un label de corps style"""
        return self.create_styled_widget(tk.Label, parent, "body", text=text, **kwargs)

    def create_primary_button(self, parent, text: str, **kwargs) -> tk.Button:
        """Cree un bouton primaire style"""
        return self.create_styled_widget(tk.Button, parent, "button_primary", text=text, **kwargs)

    def create_secondary_button(self, parent, text: str, **kwargs) -> tk.Button:
        """Cree un bouton secondaire style"""
        return self.create_styled_widget(tk.Button, parent, "button_secondary", text=text, **kwargs)

    def create_entry(self, parent, **kwargs) -> tk.Entry:
        """Cree une entree stylee"""
        return self.create_styled_widget(tk.Entry, parent, "entry", **kwargs)

    def create_frame(self, parent, style_type: str = "primary", **kwargs) -> tk.Frame:
        """Cree un frame style"""
        style_name = f"frame_{style_type}"
        return self.create_styled_widget(tk.Frame, parent, style_name, **kwargs)

    def create_panel(self, parent, **kwargs) -> tk.Frame:
        """Cree un panel style"""
        return self.create_styled_widget(tk.Frame, parent, "panel", **kwargs)

    def get_colors(self) -> ColorScheme:
        """Retourne le schema de couleurs actuel"""
        return self.current_theme.colors

    def get_fonts(self) -> Dict[str, tuple]:
        """Retourne le schema de polices actuel"""
        return self.current_theme.fonts

class TestThemeApp:
    """Application de test pour les themes"""

    def __init__(self):
        print("[INIT] Creation de l'application de test...")

        self.root = tk.Tk()
        self.root.title("Test Themes DOFUS AlphaStar")
        self.root.geometry("900x700")

        # Centrer la fenetre
        self.center_window()

        # Initialiser le gestionnaire de themes
        self.theme_manager = SimpleThemeManager()
        self.theme_manager.register_root(self.root)

        self.setup_ui()

        print("[OK] Application initialisee")

    def center_window(self):
        """Centre la fenetre sur l'ecran"""
        self.root.update_idletasks()

        width = 900
        height = 700

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        print("[UI] Creation de l'interface...")

        # Frame principal
        main_frame = self.theme_manager.create_frame(self.root, "primary")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        self.create_header(main_frame)

        # Notebook avec differents tests
        self.create_notebook(main_frame)

        # Footer avec controles de theme
        self.create_footer(main_frame)

        print("[OK] Interface creee")

    def create_header(self, parent):
        """Cree l'en-tete"""
        header_frame = self.theme_manager.create_frame(parent, "primary")
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title = self.theme_manager.create_title_label(
            header_frame,
            text="Test Themes DOFUS AlphaStar 2025"
        )
        title.pack()

        subtitle = self.theme_manager.create_subtitle_label(
            header_frame,
            text="Demonstration des themes dark/light"
        )
        subtitle.pack(pady=(5, 0))

    def create_notebook(self, parent):
        """Cree le notebook avec differents tests"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Onglet widgets
        self.create_widgets_tab()

        # Onglet couleurs
        self.create_colors_tab()

        # Onglet demo dashboard
        self.create_dashboard_tab()

    def create_widgets_tab(self):
        """Onglet test des widgets"""
        widgets_frame = self.theme_manager.create_frame(self.notebook, "primary")
        self.notebook.add(widgets_frame, text="Widgets")

        # Panel principal
        main_panel = self.theme_manager.create_panel(widgets_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        panel_title = self.theme_manager.create_subtitle_label(
            main_panel,
            text="Test des Widgets Thematiques"
        )
        panel_title.pack(pady=(15, 20))

        content_frame = self.theme_manager.create_frame(main_panel, "primary")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Labels
        labels_frame = self.theme_manager.create_frame(content_frame, "secondary")
        labels_frame.pack(fill=tk.X, pady=(0, 15))

        title_test = self.theme_manager.create_title_label(
            labels_frame,
            text="Titre de Test"
        )
        title_test.pack(pady=5)

        subtitle_test = self.theme_manager.create_subtitle_label(
            labels_frame,
            text="Sous-titre de Test"
        )
        subtitle_test.pack(pady=5)

        body_test = self.theme_manager.create_body_label(
            labels_frame,
            text="Texte de corps pour le test des polices et couleurs"
        )
        body_test.pack(pady=5)

        # Boutons
        buttons_frame = self.theme_manager.create_frame(content_frame, "primary")
        buttons_frame.pack(fill=tk.X, pady=(0, 15))

        primary_btn = self.theme_manager.create_primary_button(
            buttons_frame,
            text="Bouton Primaire",
            command=lambda: self.show_message("Bouton primaire clique!")
        )
        primary_btn.pack(side=tk.LEFT, padx=(0, 10))

        secondary_btn = self.theme_manager.create_secondary_button(
            buttons_frame,
            text="Bouton Secondaire",
            command=lambda: self.show_message("Bouton secondaire clique!")
        )
        secondary_btn.pack(side=tk.LEFT)

        # Champs de saisie
        entry_frame = self.theme_manager.create_frame(content_frame, "primary")
        entry_frame.pack(fill=tk.X)

        entry_label = self.theme_manager.create_body_label(
            entry_frame,
            text="Champ de test:"
        )
        entry_label.pack(side=tk.LEFT)

        test_entry = self.theme_manager.create_entry(entry_frame)
        test_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        test_entry.insert(0, "Tapez ici...")

    def create_colors_tab(self):
        """Onglet demonstration des couleurs"""
        colors_frame = self.theme_manager.create_frame(self.notebook, "primary")
        self.notebook.add(colors_frame, text="Couleurs")

        # Panel des couleurs
        colors_panel = self.theme_manager.create_panel(colors_frame)
        colors_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        panel_title = self.theme_manager.create_subtitle_label(
            colors_panel,
            text="Palette de Couleurs"
        )
        panel_title.pack(pady=(15, 20))

        # Grille des couleurs
        self.create_color_grid(colors_panel)

    def create_color_grid(self, parent):
        """Cree une grille des couleurs du theme"""
        content_frame = self.theme_manager.create_frame(parent, "primary")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        colors = self.theme_manager.get_colors()

        # Couleurs principales
        main_colors = [
            ("Primaire", colors.primary),
            ("Primaire Sombre", colors.primary_dark),
            ("Primaire Clair", colors.primary_light),
            ("Secondaire", colors.secondary)
        ]

        # Couleurs d'accent
        accent_colors = [
            ("Succes", colors.accent_success),
            ("Avertissement", colors.accent_warning),
            ("Erreur", colors.accent_error),
            ("Info", colors.accent_info)
        ]

        # Afficher les couleurs principales
        main_frame = self.theme_manager.create_frame(content_frame, "secondary")
        main_frame.pack(fill=tk.X, pady=(0, 10))

        main_title = self.theme_manager.create_body_label(
            main_frame,
            text="Couleurs Principales:"
        )
        main_title.pack(anchor="w", padx=10, pady=(10, 5))

        main_grid = self.theme_manager.create_frame(main_frame, "primary")
        main_grid.pack(fill=tk.X, padx=10, pady=(0, 10))

        for i, (name, color) in enumerate(main_colors):
            self.create_color_swatch(main_grid, name, color, i // 2, i % 2)

        # Afficher les couleurs d'accent
        accent_frame = self.theme_manager.create_frame(content_frame, "secondary")
        accent_frame.pack(fill=tk.X)

        accent_title = self.theme_manager.create_body_label(
            accent_frame,
            text="Couleurs d'Accent:"
        )
        accent_title.pack(anchor="w", padx=10, pady=(10, 5))

        accent_grid = self.theme_manager.create_frame(accent_frame, "primary")
        accent_grid.pack(fill=tk.X, padx=10, pady=(0, 10))

        for i, (name, color) in enumerate(accent_colors):
            self.create_color_swatch(accent_grid, name, color, i // 2, i % 2)

    def create_color_swatch(self, parent, name, color, row, col):
        """Cree un echantillon de couleur"""
        swatch_frame = self.theme_manager.create_frame(parent, "primary")
        swatch_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")

        # Carre de couleur
        color_square = tk.Label(
            swatch_frame,
            text="",
            bg=color,
            width=6,
            height=2,
            relief="solid",
            bd=1
        )
        color_square.pack(side=tk.LEFT, padx=(0, 10))

        # Informations
        info_frame = self.theme_manager.create_frame(swatch_frame, "primary")
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        name_label = self.theme_manager.create_body_label(
            info_frame,
            text=name
        )
        name_label.pack(anchor="w")

        color_label = self.theme_manager.create_body_label(
            info_frame,
            text=color,
            fg=self.theme_manager.get_colors().text_secondary
        )
        color_label.pack(anchor="w")

        # Configuration du grid
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

    def create_dashboard_tab(self):
        """Onglet demo dashboard"""
        dashboard_frame = self.theme_manager.create_frame(self.notebook, "primary")
        self.notebook.add(dashboard_frame, text="Demo Dashboard")

        # Panel de demo
        demo_panel = self.theme_manager.create_panel(dashboard_frame)
        demo_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        panel_title = self.theme_manager.create_subtitle_label(
            demo_panel,
            text="Demo Interface Dashboard"
        )
        panel_title.pack(pady=(15, 20))

        # Contenu de demo
        content_frame = self.theme_manager.create_frame(demo_panel, "primary")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Widgets de statut
        status_frame = self.theme_manager.create_frame(content_frame, "secondary")
        status_frame.pack(fill=tk.X, pady=(0, 15))

        status_title = self.theme_manager.create_body_label(
            status_frame,
            text="Statut du Bot:"
        )
        status_title.pack(anchor="w", padx=10, pady=(10, 5))

        status_content = self.theme_manager.create_frame(status_frame, "primary")
        status_content.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Statuts simules
        statuses = [
            ("Bot", "ACTIF", self.theme_manager.get_colors().accent_success),
            ("Quete", "En cours", self.theme_manager.get_colors().accent_info),
            ("Niveau", "75", self.theme_manager.get_colors().text_primary),
            ("XP/h", "125,000", self.theme_manager.get_colors().accent_warning)
        ]

        for i, (label, value, color) in enumerate(statuses):
            status_item = self.theme_manager.create_frame(status_content, "primary")
            status_item.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="ew")

            item_label = self.theme_manager.create_body_label(
                status_item,
                text=f"{label}:"
            )
            item_label.pack(side=tk.LEFT)

            item_value = self.theme_manager.create_body_label(
                status_item,
                text=value,
                fg=color
            )
            item_value.pack(side=tk.RIGHT)

        status_content.grid_columnconfigure(0, weight=1)
        status_content.grid_columnconfigure(1, weight=1)

        # Zone de logs simulee
        logs_frame = self.theme_manager.create_frame(content_frame, "secondary")
        logs_frame.pack(fill=tk.BOTH, expand=True)

        logs_title = self.theme_manager.create_body_label(
            logs_frame,
            text="Logs Recents:"
        )
        logs_title.pack(anchor="w", padx=10, pady=(10, 5))

        logs_text = tk.Text(
            logs_frame,
            height=8,
            wrap=tk.WORD,
            bg=self.theme_manager.get_colors().bg_secondary,
            fg=self.theme_manager.get_colors().text_primary,
            font=self.theme_manager.get_fonts()["code"],
            relief="flat",
            bd=0
        )
        logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Logs simules
        sample_logs = [
            "[10:30:15] INFO - Bot demarre avec succes",
            "[10:30:16] DEBUG - Chargement de la configuration",
            "[10:30:17] INFO - Connexion au serveur etablie",
            "[10:30:18] INFO - Quete 'Collecter 10 bles' demarree",
            "[10:30:19] DEBUG - Navigation vers (150, 200)",
            "[10:30:20] INFO - Ressource collectee: Ble (1/10)",
            "[10:30:22] WARNING - Monstre aggressif detecte",
            "[10:30:23] INFO - Evitement reussi"
        ]

        for log in sample_logs:
            logs_text.insert(tk.END, log + "\n")

        logs_text.configure(state="disabled")

    def create_footer(self, parent):
        """Cree le footer avec controles de theme"""
        footer_frame = self.theme_manager.create_frame(parent, "secondary")
        footer_frame.pack(fill=tk.X)

        footer_title = self.theme_manager.create_subtitle_label(
            footer_frame,
            text="Controles de Theme"
        )
        footer_title.pack(pady=(15, 10))

        controls_frame = self.theme_manager.create_frame(footer_frame, "primary")
        controls_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Boutons de theme
        dark_btn = self.theme_manager.create_secondary_button(
            controls_frame,
            text="Theme Sombre",
            command=lambda: self.switch_theme("dark")
        )
        dark_btn.pack(side=tk.LEFT, padx=(0, 10))

        light_btn = self.theme_manager.create_secondary_button(
            controls_frame,
            text="Theme Clair",
            command=lambda: self.switch_theme("light")
        )
        light_btn.pack(side=tk.LEFT)

        # Indicateur de theme actuel
        self.theme_indicator = self.theme_manager.create_body_label(
            controls_frame,
            text=f"Theme actuel: {self.theme_manager.current_theme_name}",
            fg=self.theme_manager.get_colors().accent_info
        )
        self.theme_indicator.pack(side=tk.RIGHT)

    def switch_theme(self, theme_name):
        """Change le theme"""
        print(f"[THEME] Changement vers: {theme_name}")
        self.theme_manager.set_theme(theme_name)

        # Mettre a jour l'indicateur
        self.theme_indicator.configure(
            text=f"Theme actuel: {theme_name}",
            fg=self.theme_manager.get_colors().accent_info
        )

        # Rafraichir les couleurs dans l'onglet couleurs
        print(f"[THEME] Theme {theme_name} applique")

    def show_message(self, message):
        """Affiche un message"""
        print(f"[ACTION] {message}")

    def run(self):
        """Lance l'application"""
        print("[START] Lancement de l'interface de test")
        print("[INFO] Testez les differents onglets et le changement de theme")
        print("[INFO] Tous les widgets utilisent le systeme de themes")

        self.root.mainloop()

def main():
    """Point d'entree principal"""
    print("=" * 60)
    print("[TEST] Test Direct des Themes DOFUS AlphaStar")
    print("=" * 60)

    try:
        app = TestThemeApp()
        app.run()

    except Exception as e:
        print(f"[ERREUR] Erreur dans l'application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("[BYE] Test termine")

if __name__ == "__main__":
    main()