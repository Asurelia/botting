#!/usr/bin/env python3
"""
Test minimal de l'interface moderne
Version ultra-simplifiee pour tester les composants UI de base
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Ajouter le repertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("[TEST] Interface Moderne - Test Minimal")
print("=" * 60)

try:
    # Import minimal - seulement le theme manager
    from ui.modern_app.theme_manager import ThemeManager, DarkTheme, LightTheme
    print("[OK] Theme manager importe avec succes")

except ImportError as e:
    print(f"[ERREUR] Impossible d'importer le theme manager: {e}")
    sys.exit(1)

class TestApp:
    """Application de test minimal"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Test Interface Moderne DOFUS AlphaStar")
        self.root.geometry("800x600")

        # Initialiser le gestionnaire de themes
        self.theme_manager = ThemeManager()
        self.theme_manager.register_root(self.root)

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface de test"""
        print("[UI] Creation de l'interface de test...")

        # Frame principal
        main_frame = self.theme_manager.create_frame(self.root, "primary")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Titre
        title = self.theme_manager.create_title_label(
            main_frame,
            text="Test Interface DOFUS AlphaStar 2025"
        )
        title.pack(pady=(0, 20))

        # Test des widgets thematiques
        self.create_widget_tests(main_frame)

        # Test des themes
        self.create_theme_switcher(main_frame)

        print("[OK] Interface de test creee")

    def create_widget_tests(self, parent):
        """Cree des tests de widgets"""
        # Panel de test
        test_panel = self.theme_manager.create_panel(parent)
        test_panel.pack(fill=tk.X, pady=(0, 20))

        panel_title = self.theme_manager.create_subtitle_label(
            test_panel,
            text="Test des Widgets Thematiques"
        )
        panel_title.pack(pady=(15, 10))

        content_frame = self.theme_manager.create_frame(test_panel, "primary")
        content_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Labels de test
        body_label = self.theme_manager.create_body_label(
            content_frame,
            text="Ceci est un label de corps"
        )
        body_label.pack(anchor="w", pady=2)

        # Boutons de test
        button_frame = self.theme_manager.create_frame(content_frame, "primary")
        button_frame.pack(fill=tk.X, pady=10)

        primary_btn = self.theme_manager.create_primary_button(
            button_frame,
            text="Bouton Primaire",
            command=lambda: print("[ACTION] Bouton primaire clique")
        )
        primary_btn.pack(side=tk.LEFT, padx=(0, 10))

        secondary_btn = self.theme_manager.create_secondary_button(
            button_frame,
            text="Bouton Secondaire",
            command=lambda: print("[ACTION] Bouton secondaire clique")
        )
        secondary_btn.pack(side=tk.LEFT)

        # Entry de test
        entry_frame = self.theme_manager.create_frame(content_frame, "primary")
        entry_frame.pack(fill=tk.X, pady=10)

        entry_label = self.theme_manager.create_body_label(
            entry_frame,
            text="Champ de saisie:"
        )
        entry_label.pack(side=tk.LEFT)

        test_entry = self.theme_manager.create_entry(entry_frame)
        test_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        test_entry.insert(0, "Texte de test")

    def create_theme_switcher(self, parent):
        """Cree un commutateur de themes"""
        theme_panel = self.theme_manager.create_panel(parent)
        theme_panel.pack(fill=tk.X)

        theme_title = self.theme_manager.create_subtitle_label(
            theme_panel,
            text="Commutateur de Themes"
        )
        theme_title.pack(pady=(15, 10))

        button_frame = self.theme_manager.create_frame(theme_panel, "primary")
        button_frame.pack(padx=15, pady=(0, 15))

        # Bouton theme sombre
        dark_btn = self.theme_manager.create_secondary_button(
            button_frame,
            text="Theme Sombre",
            command=lambda: self.switch_theme("dark")
        )
        dark_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Bouton theme clair
        light_btn = self.theme_manager.create_secondary_button(
            button_frame,
            text="Theme Clair",
            command=lambda: self.switch_theme("light")
        )
        light_btn.pack(side=tk.LEFT)

        # Indicateur de theme actuel
        self.theme_indicator = self.theme_manager.create_body_label(
            button_frame,
            text=f"Theme actuel: {self.theme_manager.current_theme_name}",
            fg=self.theme_manager.get_colors().accent_info
        )
        self.theme_indicator.pack(side=tk.RIGHT)

    def switch_theme(self, theme_name):
        """Change le theme"""
        print(f"[THEME] Passage au theme: {theme_name}")
        self.theme_manager.set_theme(theme_name)

        # Mettre a jour l'indicateur
        self.theme_indicator.configure(
            text=f"Theme actuel: {theme_name}",
            fg=self.theme_manager.get_colors().accent_info
        )

    def run(self):
        """Lance l'application de test"""
        print("[START] Lancement de l'application de test")
        print("[INFO] Interface basique avec themes fonctionnels")
        print("[INFO] Testez les boutons et le changement de theme")

        self.root.mainloop()

def main():
    """Point d'entree principal"""
    try:
        app = TestApp()
        app.run()

    except Exception as e:
        print(f"[ERREUR] Erreur dans l'application de test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("[BYE] Test termine")

if __name__ == "__main__":
    main()