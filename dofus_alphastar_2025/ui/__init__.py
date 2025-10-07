"""
UI Module - Interface utilisateur moderne AlphaStar pour DOFUS
Application complète avec dashboard, analytics, contrôles et monitoring
"""

# Import du module d'interface moderne
from .modern_app import *

__all__ = [
    # Réexport depuis modern_app
    "AppController",
    "create_app_controller",
    "MainWindow",
    "create_main_window",
    "DashboardPanel",
    "create_dashboard_panel",
    "ControlPanel",
    "create_control_panel",
    "ConfigPanel",
    "create_config_panel",
    "AnalyticsPanel",
    "create_analytics_panel",
    "MonitoringPanel",
    "create_monitoring_panel",
    "ThemeManager",
    "ModernTheme",
    "DarkTheme",
    "create_theme_manager"
]