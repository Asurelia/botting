"""
Pytest Configuration and Fixtures
Configuration commune pour tous les tests

Author: Claude Code
Date: 2025-10-06
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Retourne chemin vers test_data/"""
    path = Path(__file__).parent / "test_data"
    assert path.exists(), f"test_data directory not found: {path}"
    return path


@pytest.fixture(scope="session")
def screenshots_dir(test_data_dir):
    """Retourne chemin vers screenshots/"""
    path = test_data_dir / "screenshots"
    if not path.exists():
        pytest.skip("Screenshots directory not found - run tests/setup_test_data.py")
    return path


@pytest.fixture
def sample_screenshot(screenshots_dir):
    """Charge un screenshot de test"""
    # Essayer de charger screenshot avec combat
    combat_path = screenshots_dir / "combat_with_monsters.png"
    if combat_path.exists():
        img = cv2.imread(str(combat_path))
        if img is not None:
            return img

    # Fallback: chercher n'importe quelle image
    for img_path in screenshots_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        if img is not None:
            return img

    pytest.skip("No valid screenshots found")


@pytest.fixture
def sample_ui_screenshot(screenshots_dir):
    """Charge screenshot avec UI (HP/MP bars)"""
    ui_path = screenshots_dir / "ui_hp_mp_bars.png"
    if not ui_path.exists():
        pytest.skip("UI screenshot not found")

    img = cv2.imread(str(ui_path))
    if img is None:
        pytest.skip("Failed to load UI screenshot")

    return img


@pytest.fixture
def gpu_manager():
    """Initialize GPU manager (si disponible)"""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Import GPU manager
        from core.hrm_reasoning.hrm_amd_core import AMDGPUManager
        return AMDGPUManager()
    except ImportError as e:
        pytest.skip(f"GPU dependencies not available: {e}")


@pytest.fixture
def mock_ui_path(test_data_dir):
    """Retourne chemin vers mock UI HTML"""
    path = test_data_dir / "mock_dofus_ui.html"
    if not path.exists():
        pytest.skip("Mock UI not found - run tests/create_mock_ui.py")
    return path


@pytest.fixture(scope="session")
def annotations(test_data_dir):
    """Charge annotations des screenshots"""
    import json

    annotations_file = test_data_dir / "annotations" / "screenshots_annotations.json"
    if not annotations_file.exists():
        return {}

    with open(annotations_file, 'r') as f:
        return json.load(f)


# === Markers ===

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
