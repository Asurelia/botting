"""
Modern App - Interface utilisateur moderne pour DOFUS AlphaStar
Application GUI complète avec tableau de bord centralisé
"""

from .app_controller import (
    AppController,
    create_app_controller
)

from .main_window import (
    MainWindow,
    create_main_window
)

from .dashboard_panel import (
    DashboardPanel,
    create_dashboard_panel
)

from .control_panel import (
    ControlPanel,
    create_control_panel
)

from .config_panel import (
    ConfigPanel,
    create_config_panel
)

from .analytics_panel import (
    AnalyticsPanel,
    create_analytics_panel
)

from .monitoring_panel import (
    MonitoringPanel,
    create_monitoring_panel
)

from .theme_manager import (
    ThemeManager,
    ModernTheme,
    DarkTheme,
    create_theme_manager
)

__all__ = [
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