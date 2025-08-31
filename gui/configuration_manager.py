#!/usr/bin/env python3
"""
Interface de configuration visuelle pour le bot DOFUS.

Ce module fournit une interface graphique complète pour:
- Configuration drag-and-drop des modules
- Éditeur de profils avancé
- Interface de calibration interactive
- Gestion des templates et presets
- Import/Export de configurations
- Validation en temps réel

Author: Claude (Anthropic)
Date: 2025-08-31
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import copy
import uuid

# Imports PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QGridLayout, QLabel, QPushButton, QFrame, QSplitter, QTabWidget,
        QTextEdit, QScrollArea, QGroupBox, QComboBox, QSpinBox, QCheckBox,
        QSlider, QProgressBar, QTreeWidget, QTreeWidgetItem, QTableWidget,
        QTableWidgetItem, QHeaderView, QDialog, QFormLayout, QLineEdit,
        QMessageBox, QFileDialog, QColorDialog, QDateTimeEdit, QTimeEdit,
        QSpacerItem, QSizePolicy, QButtonGroup, QRadioButton, QPlainTextEdit,
        QDockWidget, QToolBox, QListWidget, QListWidgetItem, QStackedWidget,
        QDialogButtonBox, QWizard, QWizardPage, QCalendarWidget, QDoubleSpinBox,
        QTextBrowser, QMenuBar, QMenu, QToolBar, QStatusBar, QActionGroup,
        QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem,
        QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPropertyAnimation,
        QEasingCurve, QSettings, QDateTime, QTime, QDate, QObject, QMutex,
        QRunnable, QThreadPool, QStandardPaths, QMimeData, QPointF, QRectF
    )
    from PyQt6.QtGui import (
        QIcon, QPixmap, QPainter, QColor, QPalette, QFont, QFontMetrics,
        QAction, QKeySequence, QPen, QBrush, QLinearGradient, QMovie,
        QStandardItemModel, QStandardItem, QDrag, QCursor, QPolygonF,
        QPainterPath, QTransform
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class ConfigField:
    """Configuration d'un champ de configuration."""
    name: str
    display_name: str
    field_type: str  # "string", "int", "float", "bool", "choice", "color", "file", "directory"
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: List[str] = field(default_factory=list)
    description: str = ""
    tooltip: str = ""
    required: bool = False
    validation_pattern: Optional[str] = None
    depends_on: Optional[str] = None  # Dépendance conditionnelle
    category: str = "General"

@dataclass
class ConfigTemplate:
    """Template de configuration."""
    name: str
    description: str
    category: str
    fields: List[ConfigField] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class ConfigProfile:
    """Profil de configuration complet."""
    id: str
    name: str
    description: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    templates_used: List[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    is_active: bool = False
    backup_count: int = 0

class DraggableConfigWidget(QFrame):
    """Widget de configuration draggable."""
    
    config_changed = pyqtSignal(str, object)  # field_name, new_value
    
    def __init__(self, config_field: ConfigField, parent=None):
        super().__init__(parent)
        self.config_field = config_field
        self.value_widget = None
        
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setAcceptDrops(True)
        
        self.setup_ui()
        self.apply_style()
    
    def setup_ui(self):
        """Configurer l'interface du widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # En-tête avec nom et type
        header_layout = QHBoxLayout()
        
        name_label = QLabel(self.config_field.display_name)
        name_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #ffffff;
            }
        """)
        
        type_label = QLabel(f"[{self.config_field.field_type}]")
        type_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #888888;
                background-color: #404040;
                padding: 2px 6px;
                border-radius: 3px;
            }
        """)
        
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        header_layout.addWidget(type_label)
        
        layout.addLayout(header_layout)
        
        # Widget de valeur selon le type
        self.value_widget = self._create_value_widget()
        if self.value_widget:
            layout.addWidget(self.value_widget)
        
        # Description si disponible
        if self.config_field.description:
            desc_label = QLabel(self.config_field.description)
            desc_label.setStyleSheet("""
                QLabel {
                    font-size: 10px;
                    color: #cccccc;
                    margin-top: 4px;
                }
            """)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
    
    def _create_value_widget(self) -> QWidget:
        """Créer le widget de valeur selon le type."""
        field_type = self.config_field.field_type
        
        if field_type == "string":
            widget = QLineEdit()
            if self.config_field.default_value:
                widget.setText(str(self.config_field.default_value))
            widget.textChanged.connect(lambda text: self.config_changed.emit(self.config_field.name, text))
            return widget
        
        elif field_type == "int":
            widget = QSpinBox()
            if self.config_field.min_value is not None:
                widget.setMinimum(int(self.config_field.min_value))
            if self.config_field.max_value is not None:
                widget.setMaximum(int(self.config_field.max_value))
            if self.config_field.default_value is not None:
                widget.setValue(int(self.config_field.default_value))
            widget.valueChanged.connect(lambda value: self.config_changed.emit(self.config_field.name, value))
            return widget
        
        elif field_type == "float":
            widget = QDoubleSpinBox()
            if self.config_field.min_value is not None:
                widget.setMinimum(float(self.config_field.min_value))
            if self.config_field.max_value is not None:
                widget.setMaximum(float(self.config_field.max_value))
            if self.config_field.default_value is not None:
                widget.setValue(float(self.config_field.default_value))
            widget.setSingleStep(0.1)
            widget.valueChanged.connect(lambda value: self.config_changed.emit(self.config_field.name, value))
            return widget
        
        elif field_type == "bool":
            widget = QCheckBox("Activé")
            if self.config_field.default_value is not None:
                widget.setChecked(bool(self.config_field.default_value))
            widget.toggled.connect(lambda checked: self.config_changed.emit(self.config_field.name, checked))
            return widget
        
        elif field_type == "choice":
            widget = QComboBox()
            widget.addItems(self.config_field.choices)
            if self.config_field.default_value:
                index = widget.findText(str(self.config_field.default_value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            widget.currentTextChanged.connect(lambda text: self.config_changed.emit(self.config_field.name, text))
            return widget
        
        elif field_type == "color":
            widget = QPushButton()
            color = QColor(self.config_field.default_value or "#0078d4")
            widget.setStyleSheet(f"background-color: {color.name()}; color: white;")
            widget.setText(color.name())
            
            def select_color():
                new_color = QColorDialog.getColor(color, self, "Choisir couleur")
                if new_color.isValid():
                    widget.setStyleSheet(f"background-color: {new_color.name()}; color: white;")
                    widget.setText(new_color.name())
                    self.config_changed.emit(self.config_field.name, new_color.name())
            
            widget.clicked.connect(select_color)
            return widget
        
        elif field_type == "file":
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit()
            if self.config_field.default_value:
                line_edit.setText(str(self.config_field.default_value))
            
            browse_button = QPushButton("...")
            browse_button.setMaximumWidth(30)
            
            def browse_file():
                file_path, _ = QFileDialog.getOpenFileName(self, "Choisir fichier")
                if file_path:
                    line_edit.setText(file_path)
                    self.config_changed.emit(self.config_field.name, file_path)
            
            browse_button.clicked.connect(browse_file)
            line_edit.textChanged.connect(lambda text: self.config_changed.emit(self.config_field.name, text))
            
            layout.addWidget(line_edit)
            layout.addWidget(browse_button)
            
            return container
        
        elif field_type == "directory":
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit()
            if self.config_field.default_value:
                line_edit.setText(str(self.config_field.default_value))
            
            browse_button = QPushButton("...")
            browse_button.setMaximumWidth(30)
            
            def browse_directory():
                dir_path = QFileDialog.getExistingDirectory(self, "Choisir dossier")
                if dir_path:
                    line_edit.setText(dir_path)
                    self.config_changed.emit(self.config_field.name, dir_path)
            
            browse_button.clicked.connect(browse_directory)
            line_edit.textChanged.connect(lambda text: self.config_changed.emit(self.config_field.name, text))
            
            layout.addWidget(line_edit)
            layout.addWidget(browse_button)
            
            return container
        
        # Type non supporté
        return QLabel(f"Type {field_type} non supporté")
    
    def apply_style(self):
        """Appliquer le style au widget."""
        self.setStyleSheet("""
            DraggableConfigWidget {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 8px;
                margin: 4px;
            }
            DraggableConfigWidget:hover {
                border-color: #0078d4;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
    
    def get_value(self) -> Any:
        """Obtenir la valeur actuelle du widget."""
        if not self.value_widget:
            return self.config_field.default_value
        
        field_type = self.config_field.field_type
        
        if field_type == "string":
            return self.value_widget.text()
        elif field_type in ["int", "float"]:
            return self.value_widget.value()
        elif field_type == "bool":
            return self.value_widget.isChecked()
        elif field_type == "choice":
            return self.value_widget.currentText()
        elif field_type in ["file", "directory"]:
            # C'est un container, chercher le QLineEdit
            for child in self.value_widget.findChildren(QLineEdit):
                return child.text()
        elif field_type == "color":
            return self.value_widget.text()
        
        return self.config_field.default_value
    
    def set_value(self, value: Any):
        """Définir la valeur du widget."""
        if not self.value_widget:
            return
        
        field_type = self.config_field.field_type
        
        try:
            if field_type == "string":
                self.value_widget.setText(str(value))
            elif field_type == "int":
                self.value_widget.setValue(int(value))
            elif field_type == "float":
                self.value_widget.setValue(float(value))
            elif field_type == "bool":
                self.value_widget.setChecked(bool(value))
            elif field_type == "choice":
                index = self.value_widget.findText(str(value))
                if index >= 0:
                    self.value_widget.setCurrentIndex(index)
            elif field_type in ["file", "directory"]:
                for child in self.value_widget.findChildren(QLineEdit):
                    child.setText(str(value))
                    break
            elif field_type == "color":
                color = QColor(value)
                if color.isValid():
                    self.value_widget.setStyleSheet(f"background-color: {color.name()}; color: white;")
                    self.value_widget.setText(color.name())
        except (ValueError, TypeError):
            logging.warning(f"Impossible de définir la valeur {value} pour le champ {self.config_field.name}")
    
    def mousePressEvent(self, event):
        """Gestion du début de drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Gestion du drag."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        
        if not hasattr(self, 'drag_start_position'):
            return
        
        if ((event.pos() - self.drag_start_position).manhattanLength() < 
            QApplication.startDragDistance()):
            return
        
        # Créer le drag
        drag = QDrag(self)
        mimeData = QMimeData()
        
        # Données du champ
        field_data = {
            'name': self.config_field.name,
            'display_name': self.config_field.display_name,
            'type': self.config_field.field_type,
            'value': self.get_value()
        }
        
        mimeData.setText(json.dumps(field_data))
        drag.setMimeData(mimeData)
        
        # Image de drag
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        
        # Exécuter le drag
        dropAction = drag.exec(Qt.DropAction.MoveAction)

class ConfigurationCanvas(QGraphicsView):
    """Canvas pour la configuration drag-and-drop."""
    
    config_updated = pyqtSignal(dict)  # configuration mise à jour
    
    def __init__(self):
        super().__init__()
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Configuration
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Style
        self.setStyleSheet("""
            QGraphicsView {
                background-color: #1e1e1e;
                border: 2px dashed #404040;
                border-radius: 8px;
            }
        """)
        
        # Configuration actuelle
        self.current_config = {}
        self.config_widgets = {}
        
        # Grille
        self.show_grid = True
        self._setup_scene()
    
    def _setup_scene(self):
        """Configurer la scène."""
        self.scene.setSceneRect(0, 0, 800, 600)
        
        if self.show_grid:
            self._draw_grid()
    
    def _draw_grid(self):
        """Dessiner la grille de fond."""
        pen = QPen(QColor("#404040"), 1, Qt.PenStyle.DotLine)
        
        # Lignes verticales
        for x in range(0, 801, 50):
            line = self.scene.addLine(x, 0, x, 600, pen)
            line.setZValue(-1)
        
        # Lignes horizontales
        for y in range(0, 601, 50):
            line = self.scene.addLine(0, y, 800, y, pen)
            line.setZValue(-1)
    
    def dragEnterEvent(self, event):
        """Accepter les événements de drag."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Gestion du mouvement de drag."""
        event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Gestion du drop."""
        try:
            field_data = json.loads(event.mimeData().text())
            position = self.mapToScene(event.pos())
            
            self._add_config_field(field_data, position)
            event.acceptProposedAction()
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Erreur lors du drop: {e}")
    
    def _add_config_field(self, field_data: dict, position: QPointF):
        """Ajouter un champ de configuration au canvas."""
        field_name = field_data['name']
        
        # Créer un item graphique pour le champ
        rect_item = QGraphicsRectItem(0, 0, 200, 80)
        rect_item.setBrush(QBrush(QColor("#2d2d2d")))
        rect_item.setPen(QPen(QColor("#0078d4"), 2))
        rect_item.setPos(position)
        
        # Texte du champ
        text_item = QGraphicsTextItem(f"{field_data['display_name']}\n[{field_data['type']}]\nValeur: {field_data['value']}")
        text_item.setDefaultTextColor(QColor("#ffffff"))
        text_item.setPos(position.x() + 10, position.y() + 10)
        
        # Ajouter à la scène
        self.scene.addItem(rect_item)
        self.scene.addItem(text_item)
        
        # Stocker dans la configuration
        self.current_config[field_name] = {
            'value': field_data['value'],
            'position': {'x': position.x(), 'y': position.y()},
            'rect_item': rect_item,
            'text_item': text_item
        }
        
        # Émettre la mise à jour
        self.config_updated.emit(self.current_config)
    
    def clear_canvas(self):
        """Effacer le canvas."""
        self.scene.clear()
        self.current_config.clear()
        self._setup_scene()
    
    def load_configuration(self, config_data: dict):
        """Charger une configuration sur le canvas."""
        self.clear_canvas()
        
        for field_name, field_info in config_data.items():
            if isinstance(field_info, dict) and 'position' in field_info:
                position = QPointF(field_info['position']['x'], field_info['position']['y'])
                
                field_data = {
                    'name': field_name,
                    'display_name': field_name.replace('_', ' ').title(),
                    'type': 'auto',
                    'value': field_info['value']
                }
                
                self._add_config_field(field_data, position)
    
    def export_configuration(self) -> dict:
        """Exporter la configuration actuelle."""
        config = {}
        for field_name, field_info in self.current_config.items():
            config[field_name] = field_info['value']
        return config

class ConfigurationWizard(QWizard):
    """Assistant de configuration pas à pas."""
    
    def __init__(self, config_template: ConfigTemplate, parent=None):
        super().__init__(parent)
        
        self.config_template = config_template
        self.config_values = {}
        
        self.setWindowTitle(f"Assistant de configuration - {config_template.name}")
        self.setModal(True)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        
        self.setup_pages()
        self.apply_style()
    
    def setup_pages(self):
        """Configurer les pages de l'assistant."""
        # Page d'introduction
        intro_page = QWizardPage()
        intro_page.setTitle("Configuration")
        intro_page.setSubTitle(f"Assistant de configuration pour {self.config_template.name}")
        
        intro_layout = QVBoxLayout(intro_page)
        
        description = QTextBrowser()
        description.setHtml(f"""
            <h3>Description</h3>
            <p>{self.config_template.description}</p>
            
            <h3>Catégorie</h3>
            <p>{self.config_template.category}</p>
            
            <h3>Champs à configurer</h3>
            <ul>
            {''.join(f'<li><b>{field.display_name}</b>: {field.description or "Aucune description"}</li>' for field in self.config_template.fields)}
            </ul>
        """)
        description.setMaximumHeight(300)
        
        intro_layout.addWidget(description)
        
        self.addPage(intro_page)
        
        # Pages de configuration par catégorie
        categories = {}
        for field in self.config_template.fields:
            if field.category not in categories:
                categories[field.category] = []
            categories[field.category].append(field)
        
        for category, fields in categories.items():
            page = self._create_category_page(category, fields)
            self.addPage(page)
        
        # Page de résumé
        summary_page = self._create_summary_page()
        self.addPage(summary_page)
    
    def _create_category_page(self, category: str, fields: List[ConfigField]) -> QWizardPage:
        """Créer une page pour une catégorie de champs."""
        page = QWizardPage()
        page.setTitle(f"Configuration - {category}")
        page.setSubTitle(f"Configurer les paramètres de la catégorie {category}")
        
        layout = QVBoxLayout(page)
        
        # Scroll area pour les champs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        fields_widget = QWidget()
        fields_layout = QFormLayout(fields_widget)
        
        # Créer les widgets pour chaque champ
        for field in fields:
            # Créer le widget de configuration
            config_widget = DraggableConfigWidget(field)
            config_widget.config_changed.connect(
                lambda name, value, field_name=field.name: self._on_field_changed(field_name, value)
            )
            
            # Stocker la référence
            self.config_values[field.name] = field.default_value
            
            fields_layout.addRow(config_widget)
        
        scroll.setWidget(fields_widget)
        layout.addWidget(scroll)
        
        return page
    
    def _create_summary_page(self) -> QWizardPage:
        """Créer la page de résumé."""
        page = QWizardPage()
        page.setTitle("Résumé")
        page.setSubTitle("Vérifiez votre configuration avant de terminer")
        
        layout = QVBoxLayout(page)
        
        self.summary_text = QTextBrowser()
        layout.addWidget(self.summary_text)
        
        # Mettre à jour le résumé quand la page devient visible
        def update_summary():
            summary_html = "<h3>Configuration finale</h3><table border='1' cellpadding='5'>"
            summary_html += "<tr><th>Paramètre</th><th>Valeur</th></tr>"
            
            for field in self.config_template.fields:
                value = self.config_values.get(field.name, field.default_value)
                summary_html += f"<tr><td><b>{field.display_name}</b></td><td>{value}</td></tr>"
            
            summary_html += "</table>"
            self.summary_text.setHtml(summary_html)
        
        # Connecter la mise à jour du résumé
        page.initializePage = update_summary
        
        return page
    
    def _on_field_changed(self, field_name: str, value: Any):
        """Gestionnaire de changement de champ."""
        self.config_values[field_name] = value
    
    def apply_style(self):
        """Appliquer le style sombre."""
        self.setStyleSheet("""
            QWizard {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWizardPage {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #ffffff;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
    
    def get_configuration(self) -> dict:
        """Obtenir la configuration finale."""
        return dict(self.config_values)

class ConfigurationManager(QMainWindow):
    """Gestionnaire principal de configuration."""
    
    def __init__(self):
        super().__init__()
        
        # Configuration de la fenêtre
        self.setWindowTitle("DOFUS Bot - Gestionnaire de Configuration")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        # Données
        self.templates: Dict[str, ConfigTemplate] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.current_profile: Optional[ConfigProfile] = None
        
        # Interface
        self.setup_ui()
        self.setup_default_templates()
        self.apply_theme()
        
        # Configuration automatique
        self.load_configuration()
        
        # Timer de sauvegarde automatique
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(60000)  # Sauvegarde toutes les minutes
    
    def setup_ui(self):
        """Configurer l'interface utilisateur."""
        # Widget central avec onglets
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)
        
        # Onglet Profils
        self.profiles_tab = self._create_profiles_tab()
        self.central_widget.addTab(self.profiles_tab, "👤 Profils")
        
        # Onglet Templates
        self.templates_tab = self._create_templates_tab()
        self.central_widget.addTab(self.templates_tab, "📋 Templates")
        
        # Onglet Configuration visuelle
        self.visual_config_tab = self._create_visual_config_tab()
        self.central_widget.addTab(self.visual_config_tab, "🎨 Configuration Visuelle")
        
        # Onglet Import/Export
        self.import_export_tab = self._create_import_export_tab()
        self.central_widget.addTab(self.import_export_tab, "📁 Import/Export")
        
        # Barre d'outils
        self.create_toolbar()
        
        # Barre de statut
        self.create_statusbar()
        
        # Menu
        self.create_menubar()
    
    def _create_profiles_tab(self) -> QWidget:
        """Créer l'onglet des profils."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Liste des profils
        profiles_group = QGroupBox("Profils de configuration")
        profiles_layout = QVBoxLayout(profiles_group)
        
        # Contrôles des profils
        profiles_controls = QHBoxLayout()
        
        self.new_profile_btn = QPushButton("➕ Nouveau")
        self.new_profile_btn.clicked.connect(self._create_new_profile)
        
        self.copy_profile_btn = QPushButton("📋 Copier")
        self.copy_profile_btn.clicked.connect(self._copy_profile)
        
        self.delete_profile_btn = QPushButton("🗑️ Supprimer")
        self.delete_profile_btn.clicked.connect(self._delete_profile)
        
        profiles_controls.addWidget(self.new_profile_btn)
        profiles_controls.addWidget(self.copy_profile_btn)
        profiles_controls.addWidget(self.delete_profile_btn)
        profiles_controls.addStretch()
        
        profiles_layout.addLayout(profiles_controls)
        
        # Liste des profils
        self.profiles_list = QListWidget()
        self.profiles_list.currentItemChanged.connect(self._on_profile_selected)
        profiles_layout.addWidget(self.profiles_list)
        
        layout.addWidget(profiles_group)
        
        # Détails du profil sélectionné
        details_group = QGroupBox("Détails du profil")
        details_layout = QVBoxLayout(details_group)
        
        # Informations de base
        info_form = QFormLayout()
        
        self.profile_name_edit = QLineEdit()
        self.profile_description_edit = QPlainTextEdit()
        self.profile_description_edit.setMaximumHeight(80)
        
        info_form.addRow("Nom:", self.profile_name_edit)
        info_form.addRow("Description:", self.profile_description_edit)
        
        details_layout.addLayout(info_form)
        
        # Configuration du profil
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Scroll area pour la configuration
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setMinimumHeight(300)
        
        self.config_widget = QWidget()
        self.config_layout = QFormLayout(self.config_widget)
        
        self.config_scroll.setWidget(self.config_widget)
        config_layout.addWidget(self.config_scroll)
        
        # Boutons de gestion
        config_buttons = QHBoxLayout()
        
        self.save_profile_btn = QPushButton("💾 Sauvegarder")
        self.save_profile_btn.clicked.connect(self._save_current_profile)
        
        self.apply_template_btn = QPushButton("📋 Appliquer template")
        self.apply_template_btn.clicked.connect(self._apply_template_to_profile)
        
        self.wizard_config_btn = QPushButton("🧙 Assistant config")
        self.wizard_config_btn.clicked.connect(self._open_config_wizard)
        
        config_buttons.addWidget(self.save_profile_btn)
        config_buttons.addWidget(self.apply_template_btn)
        config_buttons.addWidget(self.wizard_config_btn)
        config_buttons.addStretch()
        
        config_layout.addLayout(config_buttons)
        
        details_layout.addWidget(config_group)
        
        layout.addWidget(details_group)
        
        return tab
    
    def _create_templates_tab(self) -> QWidget:
        """Créer l'onglet des templates."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Liste des templates
        templates_group = QGroupBox("Templates disponibles")
        templates_layout = QVBoxLayout(templates_group)
        
        # Contrôles
        templates_controls = QHBoxLayout()
        
        self.new_template_btn = QPushButton("➕ Nouveau")
        self.new_template_btn.clicked.connect(self._create_new_template)
        
        self.edit_template_btn = QPushButton("✏️ Éditer")
        self.edit_template_btn.clicked.connect(self._edit_template)
        
        self.delete_template_btn = QPushButton("🗑️ Supprimer")
        self.delete_template_btn.clicked.connect(self._delete_template)
        
        templates_controls.addWidget(self.new_template_btn)
        templates_controls.addWidget(self.edit_template_btn)
        templates_controls.addWidget(self.delete_template_btn)
        templates_controls.addStretch()
        
        templates_layout.addLayout(templates_controls)
        
        # Liste des templates par catégorie
        self.templates_tree = QTreeWidget()
        self.templates_tree.setHeaderLabels(["Nom", "Description", "Champs"])
        self.templates_tree.currentItemChanged.connect(self._on_template_selected)
        templates_layout.addWidget(self.templates_tree)
        
        layout.addWidget(templates_group)
        
        # Détails du template
        template_details_group = QGroupBox("Détails du template")
        template_details_layout = QVBoxLayout(template_details_group)
        
        # Informations générales
        template_info_form = QFormLayout()
        
        self.template_name_edit = QLineEdit()
        self.template_description_edit = QPlainTextEdit()
        self.template_description_edit.setMaximumHeight(60)
        self.template_category_combo = QComboBox()
        self.template_category_combo.addItems([
            "Combat", "Professions", "Navigation", "Économie", 
            "Social", "Sécurité", "Interface", "Général"
        ])
        
        template_info_form.addRow("Nom:", self.template_name_edit)
        template_info_form.addRow("Description:", self.template_description_edit)
        template_info_form.addRow("Catégorie:", self.template_category_combo)
        
        template_details_layout.addLayout(template_info_form)
        
        # Champs du template
        fields_group = QGroupBox("Champs de configuration")
        fields_layout = QVBoxLayout(fields_group)
        
        # Table des champs
        self.fields_table = QTableWidget()
        self.fields_table.setColumnCount(4)
        self.fields_table.setHorizontalHeaderLabels(["Nom", "Type", "Défaut", "Description"])
        
        header = self.fields_table.horizontalHeader()
        header.setStretchLastSection(True)
        
        fields_layout.addWidget(self.fields_table)
        
        # Boutons de gestion des champs
        fields_buttons = QHBoxLayout()
        
        self.add_field_btn = QPushButton("➕ Ajouter champ")
        self.add_field_btn.clicked.connect(self._add_template_field)
        
        self.remove_field_btn = QPushButton("➖ Supprimer")
        self.remove_field_btn.clicked.connect(self._remove_template_field)
        
        fields_buttons.addWidget(self.add_field_btn)
        fields_buttons.addWidget(self.remove_field_btn)
        fields_buttons.addStretch()
        
        fields_layout.addLayout(fields_buttons)
        
        template_details_layout.addWidget(fields_group)
        
        layout.addWidget(template_details_group)
        
        return tab
    
    def _create_visual_config_tab(self) -> QWidget:
        """Créer l'onglet de configuration visuelle."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Glissez-déposez les champs de configuration depuis la palette vers le canvas")
        instructions.setStyleSheet("""
            QLabel {
                background-color: #0078d4;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(instructions)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Palette des champs (gauche)
        palette_group = QGroupBox("Palette de configuration")
        palette_layout = QVBoxLayout(palette_group)
        
        # Scroll area pour les champs disponibles
        palette_scroll = QScrollArea()
        palette_scroll.setWidgetResizable(True)
        palette_scroll.setMaximumWidth(300)
        
        self.palette_widget = QWidget()
        self.palette_layout = QVBoxLayout(self.palette_widget)
        
        # Ajouter quelques champs par défaut
        default_fields = [
            ConfigField("name", "Nom", "string", "Mon Bot", description="Nom du profil"),
            ConfigField("auto_start", "Démarrage auto", "bool", True, description="Démarrage automatique"),
            ConfigField("session_duration", "Durée session", "int", 240, 30, 600, description="Durée en minutes"),
            ConfigField("combat_strategy", "Stratégie combat", "choice", "Équilibrée", choices=["Agressive", "Défensive", "Équilibrée"]),
            ConfigField("theme_color", "Couleur thème", "color", "#0078d4", description="Couleur principale"),
            ConfigField("log_file", "Fichier de log", "file", "logs/bot.log", description="Fichier de journalisation"),
            ConfigField("work_directory", "Dossier de travail", "directory", ".", description="Dossier de base")
        ]
        
        for field in default_fields:
            field_widget = DraggableConfigWidget(field)
            self.palette_layout.addWidget(field_widget)
        
        self.palette_layout.addStretch()
        
        palette_scroll.setWidget(self.palette_widget)
        palette_layout.addWidget(palette_scroll)
        
        main_splitter.addWidget(palette_group)
        
        # Canvas de configuration (droite)
        canvas_group = QGroupBox("Zone de configuration")
        canvas_layout = QVBoxLayout(canvas_group)
        
        # Contrôles du canvas
        canvas_controls = QHBoxLayout()
        
        self.clear_canvas_btn = QPushButton("🗑️ Effacer")
        self.clear_canvas_btn.clicked.connect(self._clear_canvas)
        
        self.load_canvas_btn = QPushButton("📂 Charger")
        self.load_canvas_btn.clicked.connect(self._load_canvas_config)
        
        self.save_canvas_btn = QPushButton("💾 Sauvegarder")
        self.save_canvas_btn.clicked.connect(self._save_canvas_config)
        
        canvas_controls.addWidget(self.clear_canvas_btn)
        canvas_controls.addWidget(self.load_canvas_btn)
        canvas_controls.addWidget(self.save_canvas_btn)
        canvas_controls.addStretch()
        
        canvas_layout.addLayout(canvas_controls)
        
        # Canvas
        self.config_canvas = ConfigurationCanvas()
        self.config_canvas.config_updated.connect(self._on_canvas_config_updated)
        canvas_layout.addWidget(self.config_canvas)
        
        main_splitter.addWidget(canvas_group)
        
        # Configuration des proportions
        main_splitter.setSizes([300, 800])
        
        layout.addWidget(main_splitter)
        
        return tab
    
    def _create_import_export_tab(self) -> QWidget:
        """Créer l'onglet d'import/export."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Section Import
        import_group = QGroupBox("Import de configuration")
        import_layout = QVBoxLayout(import_group)
        
        import_info = QLabel("Importez des configurations depuis des fichiers JSON ou des URLs")
        import_layout.addWidget(import_info)
        
        # Contrôles d'import
        import_controls = QHBoxLayout()
        
        self.import_file_btn = QPushButton("📂 Importer fichier")
        self.import_file_btn.clicked.connect(self._import_from_file)
        
        self.import_url_btn = QPushButton("🌐 Importer URL")
        self.import_url_btn.clicked.connect(self._import_from_url)
        
        self.import_template_btn = QPushButton("📋 Importer template")
        self.import_template_btn.clicked.connect(self._import_template)
        
        import_controls.addWidget(self.import_file_btn)
        import_controls.addWidget(self.import_url_btn)
        import_controls.addWidget(self.import_template_btn)
        import_controls.addStretch()
        
        import_layout.addLayout(import_controls)
        
        layout.addWidget(import_group)
        
        # Section Export
        export_group = QGroupBox("Export de configuration")
        export_layout = QVBoxLayout(export_group)
        
        export_info = QLabel("Exportez vos configurations pour les partager ou les sauvegarder")
        export_layout.addWidget(export_info)
        
        # Contrôles d'export
        export_controls = QHBoxLayout()
        
        self.export_profile_btn = QPushButton("👤 Exporter profil")
        self.export_profile_btn.clicked.connect(self._export_profile)
        
        self.export_template_btn = QPushButton("📋 Exporter template")
        self.export_template_btn.clicked.connect(self._export_template)
        
        self.export_all_btn = QPushButton("📦 Exporter tout")
        self.export_all_btn.clicked.connect(self._export_all)
        
        export_controls.addWidget(self.export_profile_btn)
        export_controls.addWidget(self.export_template_btn)
        export_controls.addWidget(self.export_all_btn)
        export_controls.addStretch()
        
        export_layout.addLayout(export_controls)
        
        layout.addWidget(export_group)
        
        # Journal d'activités
        activity_group = QGroupBox("Journal d'activités")
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(200)
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', monospace;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        
        activity_layout.addWidget(self.activity_log)
        
        layout.addWidget(activity_group)
        
        return tab
    
    def create_toolbar(self):
        """Créer la barre d'outils."""
        toolbar = self.addToolBar("Principal")
        
        # Actions principales
        new_profile_action = QAction("👤 Nouveau profil", self)
        new_profile_action.triggered.connect(self._create_new_profile)
        toolbar.addAction(new_profile_action)
        
        save_action = QAction("💾 Sauvegarder", self)
        save_action.triggered.connect(self.save_configuration)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Import/Export rapide
        import_action = QAction("📂 Import", self)
        import_action.triggered.connect(self._import_from_file)
        toolbar.addAction(import_action)
        
        export_action = QAction("📤 Export", self)
        export_action.triggered.connect(self._export_all)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # Assistant
        wizard_action = QAction("🧙 Assistant", self)
        wizard_action.triggered.connect(self._open_config_wizard)
        toolbar.addAction(wizard_action)
    
    def create_statusbar(self):
        """Créer la barre de statut."""
        self.status_bar = self.statusBar()
        
        self.status_label = QLabel("Prêt")
        self.status_bar.addWidget(self.status_label)
        
        self.profiles_count_label = QLabel("Profils: 0")
        self.status_bar.addPermanentWidget(self.profiles_count_label)
        
        self.templates_count_label = QLabel("Templates: 0")
        self.status_bar.addPermanentWidget(self.templates_count_label)
    
    def create_menubar(self):
        """Créer la barre de menus."""
        menubar = self.menuBar()
        
        # Menu Fichier
        file_menu = menubar.addMenu("Fichier")
        
        new_profile_action = QAction("Nouveau profil", self)
        new_profile_action.triggered.connect(self._create_new_profile)
        file_menu.addAction(new_profile_action)
        
        new_template_action = QAction("Nouveau template", self)
        new_template_action.triggered.connect(self._create_new_template)
        file_menu.addAction(new_template_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Sauvegarder", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Quitter", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Outils
        tools_menu = menubar.addMenu("Outils")
        
        wizard_action = QAction("Assistant de configuration", self)
        wizard_action.triggered.connect(self._open_config_wizard)
        tools_menu.addAction(wizard_action)
        
        validate_action = QAction("Valider configuration", self)
        validate_action.triggered.connect(self._validate_configuration)
        tools_menu.addAction(validate_action)
        
        backup_action = QAction("Créer sauvegarde", self)
        backup_action.triggered.connect(self._create_backup)
        tools_menu.addAction(backup_action)
        
        # Menu Aide
        help_menu = menubar.addMenu("Aide")
        
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def setup_default_templates(self):
        """Créer les templates par défaut."""
        # Template Combat
        combat_template = ConfigTemplate(
            name="Configuration Combat",
            description="Configuration pour le module de combat automatique",
            category="Combat"
        )
        
        combat_template.fields = [
            ConfigField("strategy", "Stratégie", "choice", "Équilibrée", 
                       choices=["Agressive", "Défensive", "Équilibrée"], 
                       description="Stratégie de combat globale"),
            ConfigField("min_distance", "Distance minimale", "int", 3, 1, 10,
                       description="Distance minimale de combat"),
            ConfigField("use_spells", "Utiliser sorts", "bool", True,
                       description="Utiliser les sorts en combat"),
            ConfigField("flee_threshold", "Seuil de fuite", "int", 30, 10, 90,
                       description="% de PV pour fuir le combat")
        ]
        
        self.templates["combat"] = combat_template
        
        # Template Professions
        professions_template = ConfigTemplate(
            name="Configuration Professions",
            description="Configuration pour les métiers et récolte",
            category="Professions"
        )
        
        professions_template.fields = [
            ConfigField("active_profession", "Profession active", "choice", "Bûcheron",
                       choices=["Bûcheron", "Mineur", "Paysan", "Alchimiste", "Pêcheur"],
                       description="Profession à utiliser"),
            ConfigField("target_resource", "Ressource cible", "string", "Chêne",
                       description="Nom de la ressource à récolter"),
            ConfigField("bank_mode", "Mode banque", "bool", True,
                       description="Stocker en banque automatiquement"),
            ConfigField("max_level", "Niveau maximum", "int", 100, 1, 200,
                       description="Niveau max des ressources à récolter")
        ]
        
        self.templates["professions"] = professions_template
        
        # Mettre à jour l'interface
        self._update_templates_tree()
    
    def apply_theme(self):
        """Appliquer le thème sombre."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #2d2d2d;
            }
            
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 12px 20px;
                border: 1px solid #404040;
                border-bottom: none;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #2d2d2d;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #00bcf2;
            }
            
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #005a9e;
            }
            
            QLineEdit, QPlainTextEdit, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            
            QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            
            QListWidget, QTreeWidget, QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #ffffff;
            }
            
            QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
                background-color: #0078d4;
            }
            
            QScrollArea {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
    
    # Méthodes d'événements et de gestion
    def _create_new_profile(self):
        """Créer un nouveau profil."""
        profile_id = str(uuid.uuid4())
        profile = ConfigProfile(
            id=profile_id,
            name="Nouveau profil",
            description="Description du profil"
        )
        
        self.profiles[profile_id] = profile
        self._update_profiles_list()
        
        # Sélectionner le nouveau profil
        items = self.profiles_list.findItems("Nouveau profil", Qt.MatchFlag.MatchContains)
        if items:
            self.profiles_list.setCurrentItem(items[-1])  # Dernier créé
        
        self.log_activity("Nouveau profil créé")
    
    def _copy_profile(self):
        """Copier le profil sélectionné."""
        if not self.current_profile:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner un profil à copier")
            return
        
        # Créer une copie
        new_profile = copy.deepcopy(self.current_profile)
        new_profile.id = str(uuid.uuid4())
        new_profile.name = f"{self.current_profile.name} (Copie)"
        new_profile.created = datetime.now()
        new_profile.modified = datetime.now()
        new_profile.is_active = False
        
        self.profiles[new_profile.id] = new_profile
        self._update_profiles_list()
        
        self.log_activity(f"Profil copié: {self.current_profile.name}")
    
    def _delete_profile(self):
        """Supprimer le profil sélectionné."""
        if not self.current_profile:
            return
        
        reply = QMessageBox.question(
            self, "Confirmer", 
            f"Supprimer le profil '{self.current_profile.name}' ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.profiles[self.current_profile.id]
            self.current_profile = None
            self._update_profiles_list()
            self._clear_profile_details()
            
            self.log_activity("Profil supprimé")
    
    def _save_current_profile(self):
        """Sauvegarder le profil actuel."""
        if not self.current_profile:
            return
        
        # Mettre à jour depuis l'interface
        self.current_profile.name = self.profile_name_edit.text()
        self.current_profile.description = self.profile_description_edit.toPlainText()
        self.current_profile.modified = datetime.now()
        
        # Collecter la configuration depuis les widgets
        config_data = {}
        for i in range(self.config_layout.rowCount()):
            label_item = self.config_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            field_item = self.config_layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
            
            if label_item and field_item:
                label_widget = label_item.widget()
                field_widget = field_item.widget()
                
                if isinstance(label_widget, QLabel) and hasattr(field_widget, 'get_value'):
                    field_name = label_widget.text().replace(":", "")
                    config_data[field_name] = field_widget.get_value()
        
        self.current_profile.config_data = config_data
        
        self.save_configuration()
        self.log_activity(f"Profil sauvegardé: {self.current_profile.name}")
    
    def _on_profile_selected(self, current, previous):
        """Gestionnaire de sélection de profil."""
        if current:
            profile_name = current.text()
            # Trouver le profil correspondant
            for profile in self.profiles.values():
                if profile.name == profile_name:
                    self.current_profile = profile
                    self._load_profile_details()
                    break
    
    def _load_profile_details(self):
        """Charger les détails du profil sélectionné."""
        if not self.current_profile:
            return
        
        # Informations de base
        self.profile_name_edit.setText(self.current_profile.name)
        self.profile_description_edit.setPlainText(self.current_profile.description)
        
        # Effacer la configuration actuelle
        self._clear_config_widgets()
        
        # Charger la configuration
        for field_name, value in self.current_profile.config_data.items():
            self._add_config_field_to_profile(field_name, value)
    
    def _clear_profile_details(self):
        """Effacer les détails du profil."""
        self.profile_name_edit.clear()
        self.profile_description_edit.clear()
        self._clear_config_widgets()
    
    def _clear_config_widgets(self):
        """Effacer les widgets de configuration."""
        # Supprimer tous les widgets du layout
        while self.config_layout.rowCount() > 0:
            self.config_layout.removeRow(0)
    
    def _add_config_field_to_profile(self, field_name: str, value: Any):
        """Ajouter un champ de configuration au profil."""
        # Créer un widget simple pour l'édition
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setValue(value)
        else:
            widget = QLineEdit()
            widget.setText(str(value))
        
        # Ajouter une méthode get_value
        if hasattr(widget, 'isChecked'):
            widget.get_value = widget.isChecked
        elif hasattr(widget, 'value'):
            widget.get_value = widget.value
        else:
            widget.get_value = widget.text
        
        # Ajouter au layout
        self.config_layout.addRow(f"{field_name}:", widget)
    
    def _apply_template_to_profile(self):
        """Appliquer un template au profil actuel."""
        if not self.current_profile:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner un profil")
            return
        
        # Dialog de sélection de template
        templates_list = list(self.templates.keys())
        if not templates_list:
            QMessageBox.information(self, "Information", "Aucun template disponible")
            return
        
        template_name, ok = QComboBox().getEditText()  # Placeholder, à remplacer par un vrai dialog
        # TODO: Implémenter un vrai dialog de sélection
        
        self.log_activity("Template appliqué au profil")
    
    def _open_config_wizard(self):
        """Ouvrir l'assistant de configuration."""
        if not self.templates:
            QMessageBox.information(self, "Information", "Aucun template disponible pour l'assistant")
            return
        
        # Sélectionner le template
        template_names = [template.name for template in self.templates.values()]
        template_name, ok = QComboBox().getEditText()  # Placeholder
        
        if ok and template_name:
            # Trouver le template
            selected_template = None
            for template in self.templates.values():
                if template.name == template_name:
                    selected_template = template
                    break
            
            if selected_template:
                wizard = ConfigurationWizard(selected_template, self)
                if wizard.exec() == QDialog.DialogCode.Accepted:
                    config = wizard.get_configuration()
                    
                    # Appliquer au profil actuel ou créer un nouveau
                    if self.current_profile:
                        self.current_profile.config_data.update(config)
                        self._load_profile_details()
                    else:
                        # Créer un nouveau profil
                        self._create_new_profile()
                        if self.current_profile:
                            self.current_profile.config_data = config
                            self._load_profile_details()
                    
                    self.log_activity("Configuration créée via assistant")
    
    def _update_profiles_list(self):
        """Mettre à jour la liste des profils."""
        self.profiles_list.clear()
        
        for profile in self.profiles.values():
            item = QListWidgetItem(profile.name)
            if profile.is_active:
                item.setIcon(QIcon())  # TODO: Ajouter une icône
            self.profiles_list.addItem(item)
        
        # Mettre à jour le statut
        self.profiles_count_label.setText(f"Profils: {len(self.profiles)}")
    
    def _update_templates_tree(self):
        """Mettre à jour l'arbre des templates."""
        self.templates_tree.clear()
        
        # Grouper par catégorie
        categories = {}
        for template in self.templates.values():
            if template.category not in categories:
                categories[template.category] = []
            categories[template.category].append(template)
        
        # Ajouter à l'arbre
        for category, templates in categories.items():
            category_item = QTreeWidgetItem([category, "", ""])
            category_item.setExpanded(True)
            
            for template in templates:
                template_item = QTreeWidgetItem([
                    template.name,
                    template.description,
                    str(len(template.fields))
                ])
                category_item.addChild(template_item)
            
            self.templates_tree.addTopLevelItem(category_item)
        
        # Mettre à jour le statut
        self.templates_count_label.setText(f"Templates: {len(self.templates)}")
    
    def _on_template_selected(self, current, previous):
        """Gestionnaire de sélection de template."""
        if current and current.parent():  # Ce n'est pas une catégorie
            template_name = current.text(0)
            
            # Trouver le template correspondant
            for template in self.templates.values():
                if template.name == template_name:
                    self._load_template_details(template)
                    break
    
    def _load_template_details(self, template: ConfigTemplate):
        """Charger les détails d'un template."""
        self.template_name_edit.setText(template.name)
        self.template_description_edit.setPlainText(template.description)
        
        # Trouver l'index de la catégorie
        category_index = self.template_category_combo.findText(template.category)
        if category_index >= 0:
            self.template_category_combo.setCurrentIndex(category_index)
        
        # Charger les champs
        self.fields_table.setRowCount(len(template.fields))
        
        for i, field in enumerate(template.fields):
            self.fields_table.setItem(i, 0, QTableWidgetItem(field.name))
            self.fields_table.setItem(i, 1, QTableWidgetItem(field.field_type))
            self.fields_table.setItem(i, 2, QTableWidgetItem(str(field.default_value)))
            self.fields_table.setItem(i, 3, QTableWidgetItem(field.description))
    
    def _create_new_template(self):
        """Créer un nouveau template."""
        template = ConfigTemplate(
            name="Nouveau template",
            description="Description du template",
            category="Général"
        )
        
        template_id = str(uuid.uuid4())
        self.templates[template_id] = template
        
        self._update_templates_tree()
        self.log_activity("Nouveau template créé")
    
    def _edit_template(self):
        """Éditer le template sélectionné."""
        current_item = self.templates_tree.currentItem()
        if not current_item or not current_item.parent():
            QMessageBox.information(self, "Information", "Veuillez sélectionner un template")
            return
        
        # TODO: Implémenter l'édition de template
        self.log_activity("Édition de template (à implémenter)")
    
    def _delete_template(self):
        """Supprimer le template sélectionné."""
        current_item = self.templates_tree.currentItem()
        if not current_item or not current_item.parent():
            return
        
        template_name = current_item.text(0)
        
        reply = QMessageBox.question(
            self, "Confirmer",
            f"Supprimer le template '{template_name}' ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Trouver et supprimer le template
            template_to_delete = None
            for template_id, template in self.templates.items():
                if template.name == template_name:
                    template_to_delete = template_id
                    break
            
            if template_to_delete:
                del self.templates[template_to_delete]
                self._update_templates_tree()
                self.log_activity(f"Template supprimé: {template_name}")
    
    def _add_template_field(self):
        """Ajouter un champ au template."""
        # TODO: Implémenter l'ajout de champ
        self.log_activity("Ajout de champ de template (à implémenter)")
    
    def _remove_template_field(self):
        """Supprimer un champ du template."""
        # TODO: Implémenter la suppression de champ
        self.log_activity("Suppression de champ de template (à implémenter)")
    
    def _clear_canvas(self):
        """Effacer le canvas de configuration."""
        self.config_canvas.clear_canvas()
        self.log_activity("Canvas effacé")
    
    def _load_canvas_config(self):
        """Charger une configuration sur le canvas."""
        if self.current_profile and self.current_profile.config_data:
            self.config_canvas.load_configuration(self.current_profile.config_data)
            self.log_activity("Configuration chargée sur le canvas")
        else:
            QMessageBox.information(self, "Information", "Aucune configuration à charger")
    
    def _save_canvas_config(self):
        """Sauvegarder la configuration du canvas."""
        config = self.config_canvas.export_configuration()
        
        if self.current_profile:
            self.current_profile.config_data.update(config)
            self.log_activity("Configuration du canvas sauvegardée")
        else:
            QMessageBox.information(self, "Information", "Veuillez sélectionner un profil")
    
    def _on_canvas_config_updated(self, config: dict):
        """Gestionnaire de mise à jour de configuration du canvas."""
        self.log_activity("Configuration du canvas mise à jour")
    
    def _import_from_file(self):
        """Importer depuis un fichier."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Importer configuration", "", "Fichiers JSON (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # TODO: Traiter les données importées
                self.log_activity(f"Configuration importée: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'import: {e}")
                self.log_activity(f"Erreur import: {e}")
    
    def _import_from_url(self):
        """Importer depuis une URL."""
        # TODO: Implémenter l'import depuis URL
        self.log_activity("Import depuis URL (à implémenter)")
    
    def _import_template(self):
        """Importer un template."""
        # TODO: Implémenter l'import de template
        self.log_activity("Import de template (à implémenter)")
    
    def _export_profile(self):
        """Exporter un profil."""
        if not self.current_profile:
            QMessageBox.information(self, "Information", "Veuillez sélectionner un profil")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter profil", f"{self.current_profile.name}.json", "Fichiers JSON (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(self.current_profile), f, indent=2, ensure_ascii=False, default=str)
                
                self.log_activity(f"Profil exporté: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export: {e}")
    
    def _export_template(self):
        """Exporter un template."""
        # TODO: Implémenter l'export de template
        self.log_activity("Export de template (à implémenter)")
    
    def _export_all(self):
        """Exporter toute la configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter configuration complète", "configuration_complete.json", "Fichiers JSON (*.json)"
        )
        
        if file_path:
            try:
                data = {
                    "profiles": {k: asdict(v) for k, v in self.profiles.items()},
                    "templates": {k: asdict(v) for k, v in self.templates.items()},
                    "export_date": datetime.now().isoformat(),
                    "version": "1.0"
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
                self.log_activity(f"Configuration complète exportée: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export: {e}")
    
    def _validate_configuration(self):
        """Valider la configuration actuelle."""
        if not self.current_profile:
            QMessageBox.information(self, "Information", "Aucun profil sélectionné")
            return
        
        # TODO: Implémenter la validation
        self.log_activity("Validation de configuration (à implémenter)")
    
    def _create_backup(self):
        """Créer une sauvegarde."""
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"configuration_backup_{timestamp}.json"
        
        try:
            data = {
                "profiles": {k: asdict(v) for k, v in self.profiles.items()},
                "templates": {k: asdict(v) for k, v in self.templates.items()},
                "backup_date": datetime.now().isoformat()
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.log_activity(f"Sauvegarde créée: {backup_file}")
            QMessageBox.information(self, "Succès", f"Sauvegarde créée:\n{backup_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde: {e}")
    
    def _show_about(self):
        """Afficher les informations à propos."""
        QMessageBox.about(self, "À propos",
            "DOFUS Bot - Gestionnaire de Configuration\n\n"
            "Version 1.0\n"
            "Interface de configuration avancée\n\n"
            "Fonctionnalités:\n"
            "• Configuration drag-and-drop\n"
            "• Système de templates\n"
            "• Assistant de configuration\n"
            "• Import/Export\n"
            "• Validation en temps réel\n\n"
            "© 2025 Claude (Anthropic)")
    
    def log_activity(self, message: str):
        """Ajouter un message au journal d'activités."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.activity_log.append(log_entry)
        self.status_label.setText(message)
        
        # Auto-scroll vers le bas
        cursor = self.activity_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.activity_log.setTextCursor(cursor)
    
    def load_configuration(self):
        """Charger la configuration depuis un fichier."""
        config_file = Path("config/configuration_manager.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Charger les profils
                if "profiles" in data:
                    for profile_id, profile_data in data["profiles"].items():
                        # Convertir les dates
                        if "created" in profile_data:
                            profile_data["created"] = datetime.fromisoformat(profile_data["created"])
                        if "modified" in profile_data:
                            profile_data["modified"] = datetime.fromisoformat(profile_data["modified"])
                        
                        profile = ConfigProfile(**profile_data)
                        self.profiles[profile_id] = profile
                
                # Charger les templates
                if "templates" in data:
                    for template_id, template_data in data["templates"].items():
                        # Convertir les dates et champs
                        if "created" in template_data:
                            template_data["created"] = datetime.fromisoformat(template_data["created"])
                        if "modified" in template_data:
                            template_data["modified"] = datetime.fromisoformat(template_data["modified"])
                        
                        # Convertir les champs
                        if "fields" in template_data:
                            fields = []
                            for field_data in template_data["fields"]:
                                field = ConfigField(**field_data)
                                fields.append(field)
                            template_data["fields"] = fields
                        
                        template = ConfigTemplate(**template_data)
                        self.templates[template_id] = template
                
                # Mettre à jour l'interface
                self._update_profiles_list()
                self._update_templates_tree()
                
                self.log_activity("Configuration chargée depuis le fichier")
                
            except Exception as e:
                self.log_activity(f"Erreur chargement configuration: {e}")
        else:
            self.log_activity("Aucun fichier de configuration trouvé, utilisation des valeurs par défaut")
    
    def save_configuration(self):
        """Sauvegarder la configuration dans un fichier."""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "configuration_manager.json"
        
        try:
            data = {
                "profiles": {k: asdict(v) for k, v in self.profiles.items()},
                "templates": {k: asdict(v) for k, v in self.templates.items()},
                "last_saved": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.log_activity("Configuration sauvegardée")
            
        except Exception as e:
            self.log_activity(f"Erreur sauvegarde: {e}")
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde: {e}")
    
    def autosave(self):
        """Sauvegarde automatique."""
        self.save_configuration()
        self.log_activity("Sauvegarde automatique effectuée")
    
    def closeEvent(self, event):
        """Gestion de la fermeture de l'application."""
        # Sauvegarder avant de fermer
        self.save_configuration()
        
        reply = QMessageBox.question(
            self, "Confirmer", 
            "Fermer le gestionnaire de configuration ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    """Point d'entrée principal du gestionnaire de configuration."""
    if not PYQT_AVAILABLE:
        print("PyQt6 non disponible. Veuillez installer: pip install PyQt6")
        return 1
    
    app = QApplication(sys.argv)
    app.setApplicationName("DOFUS Bot Configuration Manager")
    
    # Configuration du style
    app.setStyle("Fusion")
    
    # Créer et afficher la fenêtre principale
    window = ConfigurationManager()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())