#!/usr/bin/env python3
"""
Script de lancement standalone pour l'interface moderne DOFUS AlphaStar
Version simplifiee sans dependances du systeme core
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback
from pathlib import Path

# Ajouter le repertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("[BOT] DOFUS AlphaStar 2025 - Modern Interface (Standalone)")
print("=" * 60)

try:
    # Import direct des modules UI
    from ui.modern_app.theme_manager import create_theme_manager
    from ui.modern_app.main_window import create_main_window
    from ui.modern_app.dashboard_panel import create_dashboard_panel
    from ui.modern_app.control_panel import create_control_panel
    from ui.modern_app.analytics_panel import create_analytics_panel
    from ui.modern_app.monitoring_panel import create_monitoring_panel
    from ui.modern_app.config_panel import create_config_panel

    print("[OK] Imports UI reussis")

except ImportError as e:
    print(f"[ERREUR] Erreur d'import UI: {e}")
    print(f"[DEBUG] Traceback complet:")
    traceback.print_exc()
    sys.exit(1)

class MockAppController:
    """Controleur d'application mock pour les tests"""

    def __init__(self):
        self.is_running = False
        self.is_paused = False

    def start(self):
        self.is_running = True
        print("[MOCK] Bot demarre (simulation)")

    def stop(self):
        self.is_running = False
        print("[MOCK] Bot arrete (simulation)")

    def pause(self):
        self.is_paused = True
        print("[MOCK] Bot en pause (simulation)")

    def resume(self):
        self.is_paused = False
        print("[MOCK] Bot repris (simulation)")

    def emergency_stop(self):
        self.is_running = False
        self.is_paused = False
        print("[MOCK] Arret d'urgence (simulation)")

class StandaloneDofusApp:
    """Application standalone pour tester l'interface"""

    def __init__(self):
        self.root = None
        self.theme_manager = None
        self.app_controller = None
        self.main_window = None

    def setup_root_window(self):
        """Configure la fenetre racine"""
        self.root = tk.Tk()
        self.root.title("DOFUS AlphaStar 2025 - Modern Interface (Standalone)")
        self.root.geometry("1200x800")

        # Centrer la fenetre
        self.center_window()

        # Configuration de fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def center_window(self):
        """Centre la fenetre sur l'ecran"""
        self.root.update_idletasks()

        width = 1200
        height = 800

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def initialize_components(self):
        """Initialise les composants UI"""
        try:
            print("[INIT] Initialisation du gestionnaire de themes...")
            self.theme_manager = create_theme_manager()
            self.theme_manager.register_root(self.root)

            print("[INIT] Creation du controleur mock...")
            self.app_controller = MockAppController()

            print("[INIT] Creation de l'interface principale...")
            self.create_main_interface()

            print("[OK] Interface initialisee avec succes")

        except Exception as e:
            print(f"[ERREUR] Erreur lors de l'initialisation: {e}")
            traceback.print_exc()
            raise

    def create_main_interface(self):
        """Cree l'interface principale manuellement"""
        # Frame principal
        main_frame = self.theme_manager.create_frame(self.root, "primary")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Titre principal
        header_frame = self.theme_manager.create_frame(main_frame, "primary")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        title_label = self.theme_manager.create_title_label(
            header_frame,
            text="DOFUS AlphaStar 2025 - Interface Moderne"
        )
        title_label.pack(side=tk.LEFT)

        # Status
        status_label = self.theme_manager.create_body_label(
            header_frame,
            text="Mode Standalone",
            fg=self.theme_manager.get_colors().accent_info
        )
        status_label.pack(side=tk.RIGHT)

        # Notebook pour les onglets
        from tkinter import ttk
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Creation des panels
        self.create_panels()

    def create_panels(self):
        """Cree tous les panels"""
        try:
            # Dashboard
            print("[PANEL] Creation du Dashboard...")
            dashboard_frame = self.theme_manager.create_frame(self.notebook, "primary")
            self.dashboard_panel = create_dashboard_panel(
                dashboard_frame,
                self.theme_manager,
                self.app_controller
            )
            self.notebook.add(dashboard_frame, text="Dashboard")

            # Controle
            print("[PANEL] Creation du panel Controle...")
            control_frame = self.theme_manager.create_frame(self.notebook, "primary")
            self.control_panel = create_control_panel(
                control_frame,
                self.theme_manager,
                self.app_controller
            )
            self.notebook.add(control_frame, text="Controle")

            # Analytics
            print("[PANEL] Creation du panel Analytics...")
            analytics_frame = self.theme_manager.create_frame(self.notebook, "primary")
            self.analytics_panel = create_analytics_panel(
                analytics_frame,
                self.theme_manager,
                self.app_controller
            )
            self.notebook.add(analytics_frame, text="Analytics")

            # Monitoring
            print("[PANEL] Creation du panel Monitoring...")
            monitoring_frame = self.theme_manager.create_frame(self.notebook, "primary")
            self.monitoring_panel = create_monitoring_panel(
                monitoring_frame,
                self.theme_manager,
                self.app_controller
            )
            self.notebook.add(monitoring_frame, text="Monitoring")

            # Configuration
            print("[PANEL] Creation du panel Configuration...")
            config_frame = self.theme_manager.create_frame(self.notebook, "primary")
            self.config_panel = create_config_panel(
                config_frame,
                self.theme_manager,
                self.app_controller
            )
            self.notebook.add(config_frame, text="Configuration")

            print("[OK] Tous les panels crees avec succes")

        except Exception as e:
            print(f"[ERREUR] Erreur creation panels: {e}")
            traceback.print_exc()
            # Continuer meme en cas d'erreur sur un panel

    def run(self):
        """Lance l'application"""
        try:
            print("[START] Demarrage de l'interface standalone...")

            # Configuration de la fenetre racine
            self.setup_root_window()

            # Initialisation des composants
            self.initialize_components()

            # Message de bienvenue
            self.show_welcome_message()

            print("[OK] Interface lancee avec succes!")
            print("[INFO] Mode standalone - Toutes les fonctions sont simulees")

            # Demarrer la boucle principale
            self.root.mainloop()

        except Exception as e:
            print(f"[FATAL] Erreur fatale: {e}")
            traceback.print_exc()

            if self.root:
                messagebox.showerror(
                    "Erreur fatale",
                    f"Une erreur fatale s'est produite:\n\n{str(e)}\n\nVeuillez consulter la console pour plus de details."
                )

            sys.exit(1)

    def show_welcome_message(self):
        """Affiche un message de bienvenue"""
        welcome_text = """
[BOT] DOFUS AlphaStar 2025 - Interface Moderne (Standalone)

[INFO] Mode de demonstration - Fonctionnalites disponibles:

[DASH] Dashboard - Monitoring temps reel (simule)
[CTRL] Controle - Configuration du bot (simule)
[STATS] Analytics - Visualisations et rapports
[DEBUG] Monitoring - Logs et inspection systeme
[CONFIG] Configuration - Parametres et themes

[READY] Interface prete! Navigation par onglets.
[NOTE] Mode standalone - Les donnees sont simulees
        """
        print(welcome_text)

    def on_closing(self):
        """Gestion de la fermeture"""
        try:
            # Arreter les threads de monitoring
            if hasattr(self, 'dashboard_panel'):
                self.dashboard_panel.stop_monitoring()

            if hasattr(self, 'monitoring_panel'):
                self.monitoring_panel.stop_monitoring()

            if hasattr(self, 'analytics_panel'):
                self.analytics_panel.stop_monitoring()

            print("[STOP] Arret propre de l'application")

        except Exception as e:
            print(f"[WARN] Erreur lors de la fermeture: {e}")

        finally:
            self.root.destroy()

def main():
    """Point d'entree principal"""
    print("[CHECK] Verification des dependances de base...")

    # Verification de tkinter
    try:
        import tkinter
        print("[OK] Tkinter disponible")
    except ImportError:
        print("[ERREUR] Tkinter non disponible")
        sys.exit(1)

    print("[CHECK] Creation des repertoires necessaires...")
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)

    print("[OK] Environnement prepare")

    # Lancement de l'application
    try:
        app = StandaloneDofusApp()
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