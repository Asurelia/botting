#!/usr/bin/env python3
"""
Script de lancement pour l'interface moderne DOFUS AlphaStar
Lance l'application complète avec tous les panneaux
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback
from pathlib import Path

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Imports de l'application
    from ui.modern_app import (
        create_app_controller,
        create_main_window,
        create_theme_manager
    )

    print("[OK] Imports reussis")

except ImportError as e:
    print(f"[ERREUR] Erreur d'import: {e}")
    print(f"[DEBUG] Traceback complet:")
    traceback.print_exc()
    sys.exit(1)

class DofusAlphaStarApp:
    """Application principale DOFUS AlphaStar"""

    def __init__(self):
        """Initialise l'application"""
        self.root = None
        self.theme_manager = None
        self.app_controller = None
        self.main_window = None

    def setup_root_window(self):
        """Configure la fenêtre racine"""
        self.root = tk.Tk()
        self.root.title("DOFUS AlphaStar 2025 - Modern Interface")
        self.root.geometry("1200x800")

        # Icône et configuration
        try:
            # self.root.iconbitmap("assets/icon.ico")  # Si disponible
            pass
        except:
            pass

        # Centrer la fenêtre
        self.center_window()

        # Configuration de fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def center_window(self):
        """Centre la fenêtre sur l'écran"""
        self.root.update_idletasks()

        width = 1200
        height = 800

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def initialize_components(self):
        """Initialise tous les composants"""
        try:
            print("[INIT] Initialisation du gestionnaire de themes...")
            self.theme_manager = create_theme_manager()
            self.theme_manager.register_root(self.root)

            print("[INIT] Initialisation du controleur d'application...")
            self.app_controller = create_app_controller()

            print("[INIT] Initialisation de la fenetre principale...")
            self.main_window = create_main_window(
                self.root,
                self.theme_manager,
                self.app_controller
            )

            print("[OK] Tous les composants initialises avec succes")

        except Exception as e:
            print(f"[ERREUR] Erreur lors de l'initialisation: {e}")
            traceback.print_exc()
            raise

    def run(self):
        """Lance l'application"""
        try:
            print("[START] Demarrage de DOFUS AlphaStar 2025...")

            # Configuration de la fenêtre racine
            self.setup_root_window()

            # Initialisation des composants
            self.initialize_components()

            # Message de bienvenue
            self.show_welcome_message()

            print("[OK] Interface moderne lancee avec succes!")
            print("[INFO] Dashboard, analytics, controles et monitoring disponibles")
            print("[INFO] Configuration et debugging integres")

            # Démarrer la boucle principale
            self.root.mainloop()

        except Exception as e:
            print(f"[FATAL] Erreur fatale: {e}")
            traceback.print_exc()

            # Afficher une boîte de dialogue d'erreur
            if self.root:
                messagebox.showerror(
                    "Erreur fatale",
                    f"Une erreur fatale s'est produite:\n\n{str(e)}\n\nVeuillez consulter la console pour plus de details."
                )

            sys.exit(1)

    def show_welcome_message(self):
        """Affiche un message de bienvenue"""
        welcome_text = """
[BOT] DOFUS AlphaStar 2025 - Interface Moderne

[INFO] Fonctionnalites disponibles:

[DASH] Dashboard - Monitoring temps reel et metriques
[CTRL] Controle - Configuration complete du bot
[STATS] Analytics - Visualisations et rapports detailles
[DEBUG] Monitoring - Logs, debug et inspection systeme
[CONFIG] Configuration - Parametres globaux et themes

[READY] L'interface est maintenant prete a l'utilisation!

Navigation: Utilisez les onglets pour acceder aux differentes fonctionnalites.
        """

        # Optionnel: Afficher dans une messagebox ou dans les logs
        print(welcome_text)

    def on_closing(self):
        """Gestion de la fermeture de l'application"""
        try:
            # Arrêter les threads de monitoring
            if hasattr(self.main_window, 'dashboard_panel'):
                self.main_window.dashboard_panel.stop_monitoring()

            if hasattr(self.main_window, 'monitoring_panel'):
                self.main_window.monitoring_panel.stop_monitoring()

            if hasattr(self.main_window, 'analytics_panel'):
                self.main_window.analytics_panel.stop_monitoring()

            print("[STOP] Arret propre de l'application")

        except Exception as e:
            print(f"[WARN] Erreur lors de la fermeture: {e}")

        finally:
            self.root.destroy()

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    required_modules = [
        'tkinter',
        'threading',
        'json',
        'pathlib',
        'dataclasses',
        'typing',
        'datetime',
        'time'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"[ERREUR] Modules manquants: {', '.join(missing_modules)}")
        return False

    return True

def main():
    """Point d'entree principal"""
    print("=" * 60)
    print("[BOT] DOFUS AlphaStar 2025 - Modern Interface")
    print("=" * 60)

    # Verification des dependances
    print("[CHECK] Verification des dependances...")
    if not check_dependencies():
        print("[ERREUR] Dependances manquantes. Installation requise.")
        sys.exit(1)

    print("[OK] Toutes les dependances sont disponibles")

    # Verification de la structure du projet
    print("[CHECK] Verification de la structure du projet...")
    required_dirs = [
        "ui/modern_app",
        "core",
        "config"
    ]

    for directory in required_dirs:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"[CREATE] Creation du repertoire manquant: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)

    print("[OK] Structure du projet verifiee")

    # Lancement de l'application
    try:
        app = DofusAlphaStarApp()
        app.run()

    except KeyboardInterrupt:
        print("\n[STOP] Arret demande par l'utilisateur")

    except Exception as e:
        print(f"[FATAL] Erreur inattendue: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("[BYE] Au revoir!")

if __name__ == "__main__":
    main()