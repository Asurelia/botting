# 👨‍💻 DEVELOPER GUIDE - DOFUS Unity World Model AI

**Version 2025.1.0** | **Guide Développeur Complet** | **Septembre 2025**

---

## 📋 Table des Matières

1. [Environment Setup](#-environment-setup)
2. [Architecture de Développement](#-architecture-de-développement)
3. [Conventions de Code](#-conventions-de-code)
4. [Développement de Modules](#-développement-de-modules)
5. [Testing Framework](#-testing-framework)
6. [Debugging et Profiling](#-debugging-et-profiling)
7. [Contribution Workflow](#-contribution-workflow)
8. [Ressources Développeur](#-ressources-développeur)

---

## 🛠️ Environment Setup

### Développement Local

#### **Installation Environnement Complet**
```bash
# Clone du repository avec historique complet
git clone --recursive <repository-url> dofus_vision_2025
cd dofus_vision_2025

# Environnement virtuel dédié développement
python -m venv venv_dev --prompt="DOFUS-DEV"
source venv_dev/bin/activate  # Linux/Mac
venv_dev\Scripts\activate     # Windows

# Installation dépendances développement
pip install -r requirements_advanced.txt
pip install -r requirements_dev.txt

# Pre-commit hooks
pre-commit install
```

#### **Dépendances Développement**
```bash
# Outils de développement
pip install flake8==6.0.0           # Linting
pip install black==23.7.0           # Formatage code
pip install mypy==1.5.1             # Type checking
pip install isort==5.12.0           # Import sorting

# Testing et coverage
pip install pytest==7.4.0           # Framework tests
pip install pytest-cov==4.1.0       # Coverage
pip install pytest-mock==3.11.1     # Mocking
pip install pytest-xvfb==3.0.0      # Headless testing

# Profiling et debugging
pip install memory-profiler==0.60.0 # Profiling mémoire
pip install line-profiler==4.1.1    # Profiling ligne par ligne
pip install pdb-attach==3.2.0       # Debugging attaché

# Documentation
pip install sphinx==7.1.2           # Documentation generator
pip install sphinx-rtd-theme==1.3.0 # Theme ReadTheDocs
pip install myst-parser==2.0.0      # Markdown support
```

### Configuration IDE

#### **VS Code Configuration**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv_dev/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.md": "markdown"
    },
    "editor.rulers": [88],
    "editor.formatOnSave": true
}
```

#### **Extensions Recommandées**
```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-toolsai.jupyter",
        "github.copilot",
        "github.vscode-github-actions"
    ]
}
```

#### **PyCharm Configuration**
```python
# Settings → Tools → External Tools
# Black Formatter
Program: $ProjectFileDir$/venv_dev/Scripts/black.exe
Arguments: $FilePath$
Working Directory: $ProjectFileDir$

# Flake8 Linter
Program: $ProjectFileDir$/venv_dev/Scripts/flake8.exe
Arguments: $FilePath$
Working Directory: $ProjectFileDir$
```

---

## 🏗️ Architecture de Développement

### Structure des Modules

#### **Anatomie d'un Module Core**
```
core/example_module/
├── __init__.py                 # Exports publics
├── example_class.py            # Classe principale
├── utils.py                    # Utilitaires module
├── exceptions.py               # Exceptions spécifiques
├── types.py                    # Types et dataclasses
├── config.py                   # Configuration module
└── tests/
    ├── __init__.py
    ├── test_example_class.py   # Tests classe principale
    ├── test_utils.py           # Tests utilitaires
    └── fixtures.py             # Fixtures de test
```

#### **Template Module**
```python
# core/new_module/__init__.py
"""
New Module - Description du module
Fonctionnalités principales du module
"""

from .main_class import MainClass, HelperClass
from .types import ModuleConfig, ModuleResult
from .exceptions import ModuleError, ModuleConfigError

__version__ = "1.0.0"
__author__ = "Developer Name"

__all__ = [
    "MainClass",
    "HelperClass",
    "ModuleConfig",
    "ModuleResult",
    "ModuleError",
    "ModuleConfigError"
]

# Factory function
def get_module_instance(config: ModuleConfig = None) -> MainClass:
    """Factory pour créer une instance du module"""
    return MainClass(config or ModuleConfig())
```

### Patterns de Conception

#### **Singleton Pattern**
```python
class DatabaseManager:
    """Manager singleton pour base de données"""
    _instance: Optional['DatabaseManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'DatabaseManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._setup_database()
```

#### **Factory Pattern**
```python
class VisionEngineFactory:
    """Factory pour créer différents types de vision engines"""

    @staticmethod
    def create_engine(engine_type: str, config: Dict[str, Any]) -> VisionEngine:
        engines = {
            'opencv': OpenCVVisionEngine,
            'tensorflow': TensorFlowVisionEngine,
            'custom': CustomVisionEngine
        }

        if engine_type not in engines:
            raise ValueError(f"Engine type {engine_type} not supported")

        return engines[engine_type](config)
```

#### **Observer Pattern**
```python
class EventManager:
    """Gestionnaire d'événements global"""

    def __init__(self):
        self._observers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._observers[event_type].append(callback)

    def notify(self, event_type: str, data: Any) -> None:
        for callback in self._observers[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in observer callback: {e}")
```

### Gestion de Configuration

#### **Configuration Hiérarchique**
```python
@dataclass
class BaseConfig:
    """Configuration de base pour tous les modules"""
    debug: bool = False
    log_level: str = "INFO"
    enable_profiling: bool = False

@dataclass
class VisionConfig(BaseConfig):
    """Configuration spécifique Vision Engine"""
    window_title: str = "Dofus"
    capture_fps: int = 30
    ocr_languages: List[str] = field(default_factory=lambda: ["fr", "en"])
    confidence_threshold: float = 0.85

class ConfigManager:
    """Manager centralisé de configuration"""

    def __init__(self):
        self._configs: Dict[str, Any] = {}

    def load_config(self, module_name: str, config_class: Type[BaseConfig]) -> BaseConfig:
        if module_name not in self._configs:
            # Chargement depuis fichier, env vars, etc.
            self._configs[module_name] = self._load_from_sources(config_class)
        return self._configs[module_name]
```

---

## 📜 Conventions de Code

### Style Guide Python

#### **Formatage avec Black**
```python
# Configuration dans pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

#### **Imports avec isort**
```python
# Configuration isort
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["core", "tests"]
known_third_party = ["numpy", "opencv", "pandas"]

# Ordre des imports dans un fichier
# 1. Standard library
import os
import sys
import time
from typing import Dict, List, Optional

# 2. Third party
import numpy as np
import cv2
import pandas as pd

# 3. Local application
from core.base import BaseClass
from core.utils import helper_function
from .types import CustomType
```

### Type Hints

#### **Annotations Complètes**
```python
from typing import Dict, List, Optional, Union, Callable, Any, TypeVar, Generic

T = TypeVar('T')

class DataProcessor(Generic[T]):
    """Processeur de données générique avec type hints complets"""

    def __init__(self, data_type: type[T]) -> None:
        self._data_type = data_type
        self._cache: Dict[str, T] = {}

    def process_data(
        self,
        data: List[T],
        filter_func: Optional[Callable[[T], bool]] = None,
        transform_func: Optional[Callable[[T], T]] = None
    ) -> List[T]:
        """
        Traite une liste de données avec filtrage et transformation optionnels.

        Args:
            data: Liste des données à traiter
            filter_func: Fonction de filtrage optionnelle
            transform_func: Fonction de transformation optionnelle

        Returns:
            Liste des données traitées

        Raises:
            ValueError: Si data est vide
            TypeError: Si les types ne correspondent pas
        """
        if not data:
            raise ValueError("Data list cannot be empty")

        result = data

        if filter_func:
            result = [item for item in result if filter_func(item)]

        if transform_func:
            result = [transform_func(item) for item in result]

        return result
```

### Documentation

#### **Docstrings Google Style**
```python
class KnowledgeBase:
    """Base de connaissances DOFUS avec gestion intelligente des requêtes.

    Cette classe fournit une interface unifiée pour accéder aux différentes
    bases de données DOFUS (sorts, monstres, cartes, économie).

    Attributes:
        databases: Dict des différentes bases de données chargées
        query_cache: Cache des requêtes récentes pour optimisation
        last_update: Timestamp de la dernière mise à jour

    Example:
        >>> kb = KnowledgeBase()
        >>> kb.load_databases()
        >>> result = kb.query_spells(class_name="Iops", level_range=(1, 50))
        >>> print(f"Found {len(result.data)} spells")
    """

    def query_optimal_spells(
        self,
        context: GameContext,
        target_type: str = "enemy",
        max_results: int = 5
    ) -> QueryResult:
        """Recherche les sorts optimaux selon le contexte de jeu.

        Analyse le contexte actuel du jeu et retourne les sorts les plus
        efficaces selon la situation tactique.

        Args:
            context: Contexte actuel du jeu (classe, niveau, PA disponibles)
            target_type: Type de cible ('enemy', 'ally', 'self')
            max_results: Nombre maximum de sorts à retourner

        Returns:
            QueryResult contenant:
                - success: Boolean indiquant le succès
                - data: Liste des sorts avec scores d'efficacité
                - confidence_score: Score de confiance de la recommandation
                - execution_time_ms: Temps d'exécution de la requête

        Raises:
            ValueError: Si context est invalide
            DatabaseError: Si problème d'accès aux données

        Example:
            >>> context = GameContext(player_class=DofusClass.IOPS, available_ap=6)
            >>> result = kb.query_optimal_spells(context, "enemy")
            >>> if result.success:
            ...     best_spell = result.data[0]
            ...     print(f"Meilleur sort: {best_spell['name']}")
        """
        # Implementation...
```

### Error Handling

#### **Hiérarchie d'Exceptions**
```python
class DofusVisionError(Exception):
    """Exception de base pour le système DOFUS Vision"""
    pass

class ConfigurationError(DofusVisionError):
    """Erreur de configuration"""
    pass

class DatabaseError(DofusVisionError):
    """Erreur d'accès base de données"""
    pass

class VisionEngineError(DofusVisionError):
    """Erreur du moteur de vision"""
    pass

class WindowNotFoundError(VisionEngineError):
    """Fenêtre DOFUS non trouvée"""
    pass

class OCRError(VisionEngineError):
    """Erreur de reconnaissance OCR"""
    pass

# Usage avec context managers
class DatabaseConnection:
    def __enter__(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            return self.connection
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
        if exc_type is not None:
            logger.error(f"Database error: {exc_val}")
```

### Logging

#### **Configuration Logging**
```python
import logging
import structlog
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Configure le système de logging structuré"""

    # Configuration structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configuration logger standard
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

# Usage dans les modules
logger = structlog.get_logger(__name__)

def process_screenshot(screenshot: np.ndarray) -> GameState:
    logger.info(
        "Processing screenshot",
        image_shape=screenshot.shape,
        image_dtype=str(screenshot.dtype)
    )

    try:
        game_state = extract_game_state(screenshot)
        logger.info(
            "Screenshot processed successfully",
            player_hp=game_state.player_hp,
            in_combat=game_state.in_combat
        )
        return game_state

    except Exception as e:
        logger.error(
            "Screenshot processing failed",
            error=str(e),
            exc_info=True
        )
        raise
```

---

## 🔧 Développement de Modules

### Création d'un Nouveau Module

#### **1. Structure Initial**
```bash
# Créer structure module
mkdir -p core/new_module/tests
touch core/new_module/__init__.py
touch core/new_module/main.py
touch core/new_module/types.py
touch core/new_module/exceptions.py
touch core/new_module/config.py
touch core/new_module/tests/__init__.py
touch core/new_module/tests/test_main.py
```

#### **2. Template de Base**
```python
# core/new_module/types.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

class ModuleStatus(Enum):
    """Status du module"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"

@dataclass
class ModuleConfig:
    """Configuration du module"""
    enabled: bool = True
    debug_mode: bool = False
    cache_enabled: bool = True
    max_memory_mb: int = 100

@dataclass
class ModuleResult:
    """Résultat d'opération du module"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
```

```python
# core/new_module/main.py
import time
import logging
from typing import Any, Dict, Optional

from .types import ModuleConfig, ModuleResult, ModuleStatus
from .exceptions import ModuleError, ModuleConfigError

logger = logging.getLogger(__name__)

class NewModule:
    """Template pour nouveau module du système DOFUS Vision"""

    def __init__(self, config: Optional[ModuleConfig] = None):
        self.config = config or ModuleConfig()
        self.status = ModuleStatus.INACTIVE
        self._validate_config()
        self._initialize()

    def _validate_config(self) -> None:
        """Valide la configuration du module"""
        if self.config.max_memory_mb <= 0:
            raise ModuleConfigError("max_memory_mb must be positive")

    def _initialize(self) -> None:
        """Initialise le module"""
        logger.info("Initializing module", module=self.__class__.__name__)
        self.status = ModuleStatus.INITIALIZING

        try:
            # Logique d'initialisation
            self._setup_resources()
            self.status = ModuleStatus.ACTIVE
            logger.info("Module initialized successfully")

        except Exception as e:
            self.status = ModuleStatus.ERROR
            logger.error("Module initialization failed", error=str(e))
            raise ModuleError(f"Initialization failed: {e}")

    def _setup_resources(self) -> None:
        """Configure les ressources du module"""
        # Implementation spécifique
        pass

    def process(self, input_data: Any) -> ModuleResult:
        """Traite les données d'entrée"""
        start_time = time.time()

        try:
            if self.status != ModuleStatus.ACTIVE:
                raise ModuleError(f"Module not active, status: {self.status}")

            # Logique de traitement principal
            result_data = self._process_data(input_data)

            execution_time = (time.time() - start_time) * 1000

            return ModuleResult(
                success=True,
                data=result_data,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Processing failed", error=str(e))

            return ModuleResult(
                success=False,
                data=None,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    def _process_data(self, input_data: Any) -> Any:
        """Logique de traitement spécifique (à override)"""
        raise NotImplementedError("Subclasses must implement _process_data")

    def cleanup(self) -> None:
        """Nettoie les ressources du module"""
        logger.info("Cleaning up module resources")
        self.status = ModuleStatus.INACTIVE

# Factory function
def get_new_module(config: Optional[ModuleConfig] = None) -> NewModule:
    """Factory pour créer une instance du module"""
    return NewModule(config)
```

### Intégration avec le Core

#### **Ajout au Core Init**
```python
# core/__init__.py - Ajouter le nouveau module
try:
    from .new_module.main import NewModule
    __all__.append('NewModule')
except ImportError as e:
    logger.warning(f"Failed to import NewModule: {e}")
```

#### **Tests d'Intégration**
```python
# tests/integration/test_new_module_integration.py
import pytest
from core.new_module import NewModule, ModuleConfig

class TestNewModuleIntegration:
    """Tests d'intégration pour le nouveau module"""

    def test_module_creation(self):
        """Test création du module"""
        config = ModuleConfig(enabled=True)
        module = NewModule(config)
        assert module.status.value == "active"

    def test_module_integration_with_core(self):
        """Test intégration avec les autres modules core"""
        from core import NewModule as CoreNewModule
        module = CoreNewModule()
        assert module is not None

    def test_module_performance_baseline(self):
        """Test performance de base du module"""
        module = NewModule()
        test_data = {"test": "data"}

        result = module.process(test_data)

        assert result.success
        assert result.execution_time_ms < 100  # Moins de 100ms
```

---

## 🧪 Testing Framework

### Structure des Tests

#### **Organisation des Tests**
```
tests/
├── __init__.py
├── conftest.py                 # Configuration pytest globale
├── unit/                       # Tests unitaires
│   ├── test_vision_engine.py
│   ├── test_knowledge_base.py
│   └── test_learning_engine.py
├── integration/                # Tests d'intégration
│   ├── test_full_pipeline.py
│   └── test_module_interactions.py
├── performance/                # Tests de performance
│   ├── test_benchmarks.py
│   └── test_memory_usage.py
├── fixtures/                   # Données de test
│   ├── sample_screenshots/
│   ├── mock_databases/
│   └── test_configs/
└── utils/                      # Utilitaires de test
    ├── mock_helpers.py
    └── test_data_generators.py
```

#### **Configuration Pytest**
```python
# conftest.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture(scope="session")
def test_data_dir():
    """Répertoire des données de test"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def temp_database():
    """Base de données temporaire pour tests"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Setup test database
    setup_test_database(db_path)

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)

@pytest.fixture
def mock_screenshot():
    """Screenshot mock pour tests vision"""
    import numpy as np
    return np.zeros((800, 600, 3), dtype=np.uint8)

@pytest.fixture
def mock_dofus_window():
    """Mock de fenêtre DOFUS"""
    mock = Mock()
    mock.get_window_info.return_value = {
        "x": 100, "y": 100, "width": 800, "height": 600, "pid": 1234
    }
    return mock
```

### Tests Unitaires

#### **Template Test Unitaire**
```python
# tests/unit/test_knowledge_base.py
import pytest
from unittest.mock import Mock, patch, MagicMock

from core.knowledge_base import DofusKnowledgeBase, GameContext, DofusClass
from core.knowledge_base.exceptions import DatabaseError

class TestDofusKnowledgeBase:
    """Tests unitaires pour DofusKnowledgeBase"""

    @pytest.fixture
    def knowledge_base(self, temp_database):
        """Instance de test de KnowledgeBase"""
        with patch('core.knowledge_base.knowledge_integration.DATABASE_PATH', temp_database):
            kb = DofusKnowledgeBase()
            yield kb
            kb.cleanup()

    @pytest.fixture
    def game_context(self):
        """Contexte de jeu pour tests"""
        return GameContext(
            player_class=DofusClass.IOPS,
            player_level=150,
            available_ap=6,
            available_mp=3
        )

    def test_initialization(self, knowledge_base):
        """Test initialisation correcte"""
        assert knowledge_base is not None
        assert hasattr(knowledge_base, 'databases')

    def test_query_optimal_spells_success(self, knowledge_base, game_context):
        """Test requête sorts réussie"""
        knowledge_base.update_game_context(game_context)
        result = knowledge_base.query_optimal_spells()

        assert result.success
        assert result.data is not None
        assert result.confidence_score > 0
        assert result.execution_time_ms >= 0

    def test_query_optimal_spells_no_context(self, knowledge_base):
        """Test requête sorts sans contexte"""
        result = knowledge_base.query_optimal_spells()
        # Devrait fonctionner avec contexte par défaut
        assert result.success

    @patch('core.knowledge_base.knowledge_integration.sqlite3.connect')
    def test_database_connection_error(self, mock_connect, knowledge_base):
        """Test gestion erreur connexion DB"""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(DatabaseError):
            knowledge_base._reconnect_database()

    def test_query_monster_strategy(self, knowledge_base):
        """Test requête stratégie monstre"""
        result = knowledge_base.query_monster_strategy("Bouftou")

        assert result.success
        assert 'strategy' in result.data
        assert 'resistances' in result.data['strategy']

    @pytest.mark.parametrize("monster_name,expected_in_data", [
        ("Bouftou", "strategy"),
        ("Tofu", "strategy"),
        ("MonstreInexistant", None)
    ])
    def test_query_various_monsters(self, knowledge_base, monster_name, expected_in_data):
        """Test requêtes divers monstres"""
        result = knowledge_base.query_monster_strategy(monster_name)

        if expected_in_data:
            assert result.success
            assert expected_in_data in result.data
        else:
            # Pour monstres inexistants
            assert not result.success or result.data is None
```

### Tests d'Intégration

#### **Pipeline Complet**
```python
# tests/integration/test_full_pipeline.py
import pytest
import time
from unittest.mock import patch

from core import (
    DofusWindowCapture, DofusUnityInterfaceReader,
    DofusKnowledgeBase, AdaptiveLearningEngine
)

class TestFullPipeline:
    """Tests d'intégration du pipeline complet"""

    @pytest.fixture
    def pipeline_components(self):
        """Composants du pipeline pour tests"""
        return {
            'capture': DofusWindowCapture(),
            'reader': DofusUnityInterfaceReader(),
            'knowledge_base': DofusKnowledgeBase(),
            'learning_engine': AdaptiveLearningEngine()
        }

    @pytest.mark.integration
    def test_vision_to_knowledge_pipeline(self, pipeline_components, mock_screenshot):
        """Test pipeline vision → knowledge base"""
        capture = pipeline_components['capture']
        reader = pipeline_components['reader']
        kb = pipeline_components['knowledge_base']

        # Mock capture
        with patch.object(capture, 'capture_screenshot', return_value=mock_screenshot):
            # Simulation pipeline
            screenshot = capture.capture_screenshot()
            assert screenshot is not None

            game_state = reader.extract_game_state(screenshot)
            assert game_state is not None

            # Update context avec état extrait
            context = GameContext(
                player_hp=game_state.player_hp,
                player_ap=game_state.player_ap
            )
            kb.update_game_context(context)

            # Requête avec nouveau contexte
            result = kb.query_optimal_spells()
            assert result.success

    @pytest.mark.integration
    def test_learning_integration(self, pipeline_components):
        """Test intégration avec learning engine"""
        learning_engine = pipeline_components['learning_engine']
        kb = pipeline_components['knowledge_base']

        # Démarrage session
        session_id = learning_engine.start_learning_session("IOPS", 150, "Julith")
        assert session_id is not None

        # Simulation d'actions et apprentissage
        for i in range(5):
            action = {"type": "spell_cast", "spell": f"Sort{i}"}
            outcome = {"success": True, "damage": 100 + i * 10}
            context = {"turn": i, "in_combat": True}

            learning_engine.record_action_outcome(action, outcome, context)

        # Vérification apprentissage
        metrics = learning_engine.get_learning_metrics()
        assert metrics['total_actions'] == 5

        # Fin session
        session = learning_engine.end_learning_session()
        assert session.total_actions == 5
```

### Tests de Performance

#### **Benchmarks**
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import psutil
import os
from memory_profiler import profile

from core import DofusKnowledgeBase

class TestPerformanceBenchmarks:
    """Tests de performance et benchmarks"""

    @pytest.fixture
    def performance_kb(self):
        """KB configurée pour tests performance"""
        kb = DofusKnowledgeBase()
        return kb

    def test_knowledge_base_query_speed(self, performance_kb):
        """Test vitesse requêtes KB"""
        kb = performance_kb

        # Warmup
        kb.query_optimal_spells()

        # Benchmark
        start_time = time.time()
        for _ in range(100):
            result = kb.query_optimal_spells()
            assert result.success

        total_time = time.time() - start_time
        avg_time_ms = (total_time / 100) * 1000

        # Assertion performance
        assert avg_time_ms < 50  # Moins de 50ms par requête

    def test_memory_usage(self, performance_kb):
        """Test utilisation mémoire"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        kb = performance_kb

        # Opérations intensives
        for i in range(1000):
            result = kb.query_optimal_spells()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Assertion mémoire
        assert memory_increase < 50  # Moins de 50MB d'augmentation

    @profile
    def test_memory_profile_detailed(self, performance_kb):
        """Profiling mémoire détaillé"""
        kb = performance_kb

        for i in range(100):
            result = kb.query_optimal_spells()
            result = kb.query_monster_strategy("Bouftou")
```

---

## 🐛 Debugging et Profiling

### Configuration Debug

#### **Debug Avancé**
```python
# debug_config.py
import logging
import sys
from pathlib import Path

def setup_debug_environment():
    """Configure environnement debug avancé"""

    # Logging très verbeux
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log')
        ]
    )

    # Debugging automatique sur erreurs
    def debug_hook(type_, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb
            traceback.print_exception(type_, value, tb)
            print("\n")
            pdb.post_mortem(tb)

    sys.excepthook = debug_hook

    # Variables debug globales
    import builtins
    builtins.DEBUG = True
```

#### **Debugging Conditionnel**
```python
import os
from functools import wraps

DEBUG = os.getenv('DOFUS_DEBUG', 'false').lower() == 'true'

def debug_trace(func):
    """Décorateur pour tracer l'exécution des fonctions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"TRACE: Entering {func.__name__}")
            print(f"TRACE: Args: {args}")
            print(f"TRACE: Kwargs: {kwargs}")

        try:
            result = func(*args, **kwargs)
            if DEBUG:
                print(f"TRACE: {func.__name__} returned: {type(result)}")
            return result
        except Exception as e:
            if DEBUG:
                print(f"TRACE: {func.__name__} raised: {e}")
                import traceback
                traceback.print_exc()
            raise

    return wrapper

# Usage
@debug_trace
def complex_function(data):
    # Implementation...
    pass
```

### Profiling Performance

#### **Line Profiler**
```python
# Installer: pip install line_profiler
# Usage: kernprof -l -v script.py

@profile
def performance_critical_function():
    """Fonction critique à profiler ligne par ligne"""
    # Chaque ligne sera chronométrée
    data = load_large_dataset()           # Ligne 1
    processed = process_data(data)        # Ligne 2
    result = analyze_results(processed)   # Ligne 3
    return result
```

#### **Memory Profiler**
```python
# memory_debug.py
from memory_profiler import profile
import tracemalloc

@profile
def memory_intensive_function():
    """Fonction à profiler pour mémoire"""
    large_list = [i for i in range(1000000)]
    processed = [x * 2 for x in large_list]
    return sum(processed)

def trace_memory_allocation():
    """Trace allocation mémoire détaillée"""
    tracemalloc.start()

    # Code à tracer
    result = memory_intensive_function()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()
    return result
```

#### **Performance Monitoring**
```python
# performance_monitor.py
import time
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    """Monitor performance en temps réel"""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.metrics = {
            'cpu_percent': deque(maxlen=100),
            'memory_mb': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }

    def start_monitoring(self):
        """Démarre monitoring en arrière-plan"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Boucle de monitoring"""
        process = psutil.Process()

        while self.running:
            timestamp = time.time()
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024

            self.metrics['timestamps'].append(timestamp)
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_mb'].append(memory_mb)

            time.sleep(self.interval)

    def stop_monitoring(self):
        """Arrête monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

    def get_stats(self):
        """Retourne statistiques"""
        if not self.metrics['cpu_percent']:
            return None

        return {
            'avg_cpu_percent': sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent']),
            'max_cpu_percent': max(self.metrics['cpu_percent']),
            'avg_memory_mb': sum(self.metrics['memory_mb']) / len(self.metrics['memory_mb']),
            'max_memory_mb': max(self.metrics['memory_mb']),
            'duration_seconds': self.metrics['timestamps'][-1] - self.metrics['timestamps'][0]
        }
```

---

## 🔄 Contribution Workflow

### Git Workflow

#### **Branch Strategy**
```bash
# Structure des branches
main                    # Production stable
├── develop            # Intégration continue
├── feature/xxx        # Nouvelles fonctionnalités
├── bugfix/xxx         # Corrections de bugs
├── hotfix/xxx         # Corrections urgentes
└── release/xxx        # Préparation releases

# Workflow standard
git checkout develop
git pull origin develop
git checkout -b feature/new-vision-algorithm
# ... développement ...
git commit -m "feat(vision): implement new detection algorithm"
git push -u origin feature/new-vision-algorithm
# ... créer Pull Request ...
```

#### **Conventional Commits**
```bash
# Format: type(scope): description
feat(vision): add new OCR engine support
fix(knowledge): correct spell damage calculation
docs(api): update API reference for v2.0
style(core): format code with black
refactor(learning): optimize memory usage
test(integration): add full pipeline tests
chore(deps): update dependencies to latest
```

### Code Review Process

#### **Pull Request Template**
```markdown
<!-- .github/pull_request_template.md -->
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work)
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Performance impact assessed

## Checklist:
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
```

#### **Review Guidelines**
```python
# review_checklist.py
REVIEW_CHECKLIST = {
    "code_quality": [
        "Code follows PEP 8 and project style guide",
        "Functions have clear docstrings",
        "Type hints are present and accurate",
        "Error handling is appropriate",
        "No obvious performance issues"
    ],
    "testing": [
        "Unit tests cover new functionality",
        "Tests are meaningful and not just for coverage",
        "Edge cases are tested",
        "Tests pass consistently"
    ],
    "documentation": [
        "Public APIs are documented",
        "Complex logic is explained",
        "README updated if needed",
        "CHANGELOG updated for user-facing changes"
    ],
    "security": [
        "No sensitive data in code",
        "Input validation where needed",
        "No obvious security vulnerabilities",
        "Dependencies are secure"
    ]
}
```

### CI/CD Pipeline

#### **GitHub Actions**
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_dev.txt

    - name: Lint with flake8
      run: |
        flake8 core tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 core tests --count --exit-zero --max-complexity=10 --max-line-length=88

    - name: Type check with mypy
      run: mypy core

    - name: Test with pytest
      run: |
        pytest tests/ --cov=core --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## 📚 Ressources Développeur

### Documentation Technique

#### **Génération Documentation**
```bash
# Sphinx documentation
cd docs/
make clean
make html

# Serveur local documentation
python -m http.server 8000 -d _build/html/
```

#### **Configuration Sphinx**
```python
# docs/conf.py
project = 'DOFUS Vision 2025'
copyright = '2025, Claude Code Team'
author = 'Claude Code Team'

extensions = [
    'sphinx.ext.autodoc',        # Auto-documentation depuis docstrings
    'sphinx.ext.viewcode',       # Liens vers code source
    'sphinx.ext.napoleon',       # Support Google/NumPy docstrings
    'sphinx.ext.intersphinx',    # Liens vers autres documentations
    'myst_parser'                # Support Markdown
]

html_theme = 'sphinx_rtd_theme'
```

### Outils de Développement

#### **Scripts Utilitaires**
```python
# scripts/dev_helpers.py
import subprocess
import sys
from pathlib import Path

def run_all_checks():
    """Lance tous les checks de qualité code"""
    checks = [
        ("Format check", ["black", "--check", "core", "tests"]),
        ("Import sort check", ["isort", "--check-only", "core", "tests"]),
        ("Lint check", ["flake8", "core", "tests"]),
        ("Type check", ["mypy", "core"]),
        ("Security check", ["bandit", "-r", "core"])
    ]

    results = {}
    for name, cmd in checks:
        print(f"Running {name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        results[name] = result.returncode == 0
        if not results[name]:
            print(f"❌ {name} failed:")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"✅ {name} passed")

    return all(results.values())

def setup_pre_commit():
    """Configure pre-commit hooks"""
    pre_commit_config = """
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
"""

    with open(".pre-commit-config.yaml", "w") as f:
        f.write(pre_commit_config)

    subprocess.run(["pre-commit", "install"])
    print("Pre-commit hooks configured")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_pre_commit()
    else:
        success = run_all_checks()
        sys.exit(0 if success else 1)
```

### Environnements de Test

#### **Docker Development**
```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libopencv-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements_dev.txt .
RUN pip install -r requirements_dev.txt

# Code source
COPY . .

# Environnement développement
ENV PYTHONPATH=/app
ENV DOFUS_DEBUG=true

CMD ["python", "-m", "pytest", "tests/", "-v"]
```

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  dofus-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    environment:
      - DISPLAY=${DISPLAY}
    network_mode: host
    stdin_open: true
    tty: true
```

### Community et Support

#### **Ressources Externes**
- **Stack Overflow** : Tag `dofus-vision-ai`
- **Reddit** : r/DofusAI (communauté)
- **Discord** : Serveur développeurs DOFUS AI
- **GitHub Discussions** : Questions techniques

#### **Contribution Guidelines**
1. **Fork** le repository
2. **Créer branch** feature/bugfix
3. **Développer** avec tests
4. **Documenter** changements
5. **Tester** localement
6. **Pull Request** avec description détaillée
7. **Review** par maintainers
8. **Merge** après approbation

---

*Guide Développeur maintenu par Claude Code - AI Development Specialist*
*Version 2025.1.0 - Septembre 2025*
*Contributions communautaires bienvenues*