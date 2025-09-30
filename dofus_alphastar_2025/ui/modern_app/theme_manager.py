#!/usr/bin/env python3
"""
ThemeManager - Gestionnaire de thèmes pour interface moderne
Styles et couleurs pour une UI professionnelle
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod

@dataclass
class ColorScheme:
    """Schéma de couleurs"""
    # Couleurs primaires
    primary: str
    primary_dark: str
    primary_light: str

    # Couleurs secondaires
    secondary: str
    secondary_dark: str
    secondary_light: str

    # Couleurs d'arrière-plan
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
    """Classe de base pour les thèmes"""

    def __init__(self):
        self.colors = self.get_color_scheme()
        self.fonts = self.get_font_scheme()
        self.styles = self.get_style_definitions()

    @abstractmethod
    def get_color_scheme(self) -> ColorScheme:
        """Retourne le schéma de couleurs"""
        pass

    @abstractmethod
    def get_font_scheme(self) -> Dict[str, tuple]:
        """Retourne le schéma de polices"""
        pass

    def get_style_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les définitions de styles"""
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
    """Thème sombre moderne"""

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

            # Arrière-plans sombres
            bg_primary="#0f172a",      # Très sombre
            bg_secondary="#1e293b",    # Sombre moyen
            bg_tertiary="#334155",     # Moins sombre

            # Textes
            text_primary="#f8fafc",    # Blanc cassé
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
    """Thème clair moderne"""

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

            # Arrière-plans clairs
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

class ModernTheme(DarkTheme):
    """Thème moderne par défaut (hérite du thème sombre)"""
    pass

class ThemeManager:
    """Gestionnaire de thèmes"""

    def __init__(self):
        self.themes = {
            "dark": DarkTheme(),
            "light": LightTheme(),
            "modern": ModernTheme()
        }
        self.current_theme_name = "dark"
        self.current_theme = self.themes[self.current_theme_name]
        self.root: Optional[tk.Tk] = None

    def set_theme(self, theme_name: str):
        """Change le thème actuel"""
        if theme_name in self.themes:
            self.current_theme_name = theme_name
            self.current_theme = self.themes[theme_name]

            if self.root:
                self.apply_theme_to_root()

    def apply_theme_to_root(self):
        """Applique le thème à la fenêtre racine"""
        if self.root:
            self.root.configure(bg=self.current_theme.colors.bg_primary)

            # Configurer les styles ttk si disponible
            try:
                style = ttk.Style()
                self._configure_ttk_styles(style)
            except:
                pass

    def _configure_ttk_styles(self, style: ttk.Style):
        """Configure les styles TTK"""
        theme = self.current_theme

        # Notebook (onglets)
        style.configure("TNotebook",
                       background=theme.colors.bg_primary,
                       borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=theme.colors.bg_secondary,
                       foreground=theme.colors.text_secondary,
                       padding=[20, 8],
                       font=theme.fonts["body"])
        style.map("TNotebook.Tab",
                 background=[("selected", theme.colors.primary),
                           ("active", theme.colors.bg_tertiary)],
                 foreground=[("selected", "white"),
                           ("active", theme.colors.text_primary)])

        # Frame
        style.configure("TFrame",
                       background=theme.colors.bg_primary)

        # Label
        style.configure("TLabel",
                       background=theme.colors.bg_primary,
                       foreground=theme.colors.text_primary,
                       font=theme.fonts["body"])

        # Button
        style.configure("TButton",
                       background=theme.colors.primary,
                       foreground="white",
                       font=theme.fonts["button"],
                       relief="flat",
                       padding=[20, 8])
        style.map("TButton",
                 background=[("active", theme.colors.primary_dark),
                           ("pressed", theme.colors.primary_dark)])

        # Entry
        style.configure("TEntry",
                       fieldbackground=theme.colors.bg_secondary,
                       foreground=theme.colors.text_primary,
                       bordercolor=theme.colors.border_light,
                       lightcolor=theme.colors.primary,
                       darkcolor=theme.colors.primary,
                       font=theme.fonts["body"])

        # Combobox
        style.configure("TCombobox",
                       fieldbackground=theme.colors.bg_secondary,
                       foreground=theme.colors.text_primary,
                       bordercolor=theme.colors.border_light,
                       font=theme.fonts["body"])

        # Progressbar
        style.configure("TProgressbar",
                       background=theme.colors.primary,
                       troughcolor=theme.colors.bg_tertiary,
                       borderwidth=0,
                       lightcolor=theme.colors.primary,
                       darkcolor=theme.colors.primary)

        # Treeview
        style.configure("Treeview",
                       background=theme.colors.bg_secondary,
                       foreground=theme.colors.text_primary,
                       fieldbackground=theme.colors.bg_secondary,
                       font=theme.fonts["body"])
        style.configure("Treeview.Heading",
                       background=theme.colors.bg_tertiary,
                       foreground=theme.colors.text_primary,
                       font=theme.fonts["subtitle"])

    def register_root(self, root: tk.Tk):
        """Enregistre la fenêtre racine"""
        self.root = root
        self.apply_theme_to_root()

    def get_style(self, style_name: str) -> Dict[str, Any]:
        """Retourne un style spécifique"""
        return self.current_theme.styles.get(style_name, {})

    def create_styled_widget(self, widget_class, parent, style_name: str, **kwargs):
        """Crée un widget avec style appliqué"""
        style = self.get_style(style_name)

        # Fusionner style avec kwargs
        final_kwargs = {**style, **kwargs}

        return widget_class(parent, **final_kwargs)

    def create_title_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Crée un label de titre stylé"""
        return self.create_styled_widget(tk.Label, parent, "title", text=text, **kwargs)

    def create_subtitle_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Crée un label de sous-titre stylé"""
        return self.create_styled_widget(tk.Label, parent, "subtitle", text=text, **kwargs)

    def create_body_label(self, parent, text: str, **kwargs) -> tk.Label:
        """Crée un label de corps stylé"""
        return self.create_styled_widget(tk.Label, parent, "body", text=text, **kwargs)

    def create_primary_button(self, parent, text: str, **kwargs) -> tk.Button:
        """Crée un bouton primaire stylé"""
        return self.create_styled_widget(tk.Button, parent, "button_primary", text=text, **kwargs)

    def create_secondary_button(self, parent, text: str, **kwargs) -> tk.Button:
        """Crée un bouton secondaire stylé"""
        return self.create_styled_widget(tk.Button, parent, "button_secondary", text=text, **kwargs)

    def create_entry(self, parent, **kwargs) -> tk.Entry:
        """Crée une entrée stylée"""
        return self.create_styled_widget(tk.Entry, parent, "entry", **kwargs)

    def create_frame(self, parent, style_type: str = "primary", **kwargs) -> tk.Frame:
        """Crée un frame stylé"""
        style_name = f"frame_{style_type}"
        return self.create_styled_widget(tk.Frame, parent, style_name, **kwargs)

    def create_panel(self, parent, **kwargs) -> tk.Frame:
        """Crée un panel stylé"""
        return self.create_styled_widget(tk.Frame, parent, "panel", **kwargs)

    def get_colors(self) -> ColorScheme:
        """Retourne le schéma de couleurs actuel"""
        return self.current_theme.colors

    def get_fonts(self) -> Dict[str, tuple]:
        """Retourne le schéma de polices actuel"""
        return self.current_theme.fonts

def create_theme_manager() -> ThemeManager:
    """Factory function pour créer ThemeManager"""
    return ThemeManager()