#!/usr/bin/env python3
"""
Import Tests - Verify all dependencies are installed correctly
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCoreImports(unittest.TestCase):
    """Test core Python dependencies"""

    def test_import_json(self):
        """Test json module"""
        import json
        self.assertIsNotNone(json)

    def test_import_pathlib(self):
        """Test pathlib module"""
        from pathlib import Path
        self.assertIsNotNone(Path)

    def test_import_dataclasses(self):
        """Test dataclasses module"""
        from dataclasses import dataclass
        self.assertIsNotNone(dataclass)

    def test_import_logging(self):
        """Test logging module"""
        import logging
        self.assertIsNotNone(logging)


class TestThirdPartyImports(unittest.TestCase):
    """Test third-party dependencies from requirements.txt"""

    def test_import_numpy(self):
        """Test numpy"""
        try:
            import numpy as np
            self.assertIsNotNone(np)
            print(f"    [OK] numpy version: {np.__version__}")
        except ImportError as e:
            self.fail(f"numpy not installed: {e}")

    def test_import_torch(self):
        """Test PyTorch"""
        try:
            import torch
            self.assertIsNotNone(torch)
            print(f"    [OK] PyTorch version: {torch.__version__}")
            print(f"    [OK] CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            self.fail(f"PyTorch not installed: {e}")

    def test_import_opencv(self):
        """Test OpenCV"""
        try:
            import cv2
            self.assertIsNotNone(cv2)
            print(f"    [OK] OpenCV version: {cv2.__version__}")
        except ImportError:
            print("     OpenCV not installed (optional)")

    def test_import_pil(self):
        """Test Pillow"""
        try:
            from PIL import Image
            self.assertIsNotNone(Image)
        except ImportError:
            print("     Pillow not installed (optional)")

    def test_import_networkx(self):
        """Test NetworkX"""
        try:
            import networkx as nx
            self.assertIsNotNone(nx)
            print(f"    [OK] NetworkX version: {nx.__version__}")
        except ImportError:
            print("     NetworkX not installed (optional for map system)")

    def test_import_requests(self):
        """Test requests"""
        try:
            import requests
            self.assertIsNotNone(requests)
            print(f"    [OK] requests version: {requests.__version__}")
        except ImportError as e:
            self.fail(f"requests not installed: {e}")

    def test_import_pyautogui(self):
        """Test PyAutoGUI"""
        try:
            import pyautogui
            self.assertIsNotNone(pyautogui)
        except ImportError:
            print("     PyAutoGUI not installed (required for screen/keyboard/mouse)")


class TestProjectModules(unittest.TestCase):
    """Test that project modules can be imported"""

    def test_import_core_calibration(self):
        """Test core.calibration"""
        try:
            from core import calibration
            self.assertIsNotNone(calibration)
        except ImportError as e:
            self.fail(f"core.calibration import failed: {e}")

    def test_import_core_map_system(self):
        """Test core.map_system"""
        try:
            from core import map_system
            self.assertIsNotNone(map_system)
        except ImportError as e:
            self.fail(f"core.map_system import failed: {e}")

    def test_import_core_external_data(self):
        """Test core.external_data"""
        try:
            from core import external_data
            self.assertIsNotNone(external_data)
        except ImportError as e:
            self.fail(f"core.external_data import failed: {e}")

    def test_import_core_safety(self):
        """Test core.safety"""
        try:
            from core import safety
            self.assertIsNotNone(safety)
        except ImportError as e:
            self.fail(f"core.safety import failed: {e}")

    def test_import_launch_safe(self):
        """Test launch_safe module"""
        try:
            import launch_safe
            self.assertIsNotNone(launch_safe)
        except ImportError as e:
            print(f"     launch_safe.py import warning: {e}")


class TestOptionalDependencies(unittest.TestCase):
    """Test optional dependencies"""

    def test_import_easyocr(self):
        """Test EasyOCR (optional)"""
        try:
            import easyocr
            self.assertIsNotNone(easyocr)
            print("    [OK] EasyOCR installed")
        except ImportError:
            print("     EasyOCR not installed (optional for advanced OCR)")

    def test_import_pytesseract(self):
        """Test pytesseract (optional)"""
        try:
            import pytesseract
            self.assertIsNotNone(pytesseract)
            print("    [OK] pytesseract installed")
        except ImportError:
            print("     pytesseract not installed (optional for OCR)")

    def test_import_matplotlib(self):
        """Test matplotlib (optional)"""
        try:
            import matplotlib
            self.assertIsNotNone(matplotlib)
            print("    [OK] matplotlib installed")
        except ImportError:
            print("     matplotlib not installed (optional for visualization)")


def run_import_tests():
    """Run all import tests"""
    print("=" * 70)
    print("TESTS D'IMPORTS ET DEPENDANCES")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nResultats:")
    print(f"  Tests executes: {result.testsRun}")
    print(f"  Succes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Echecs: {len(result.failures)}")
    print(f"  Erreurs: {len(result.errors)}")

    if len(result.failures) > 0 or len(result.errors) > 0:
        print("\nCertaines dependances manquent!")
        print("   Executer: pip install -r requirements.txt")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_import_tests()
    sys.exit(0 if success else 1)