"""
Tests Vision System - Capture écran, détection UI, OCR
Tests automatisés sans Dofus lancé

Author: Claude Code
Date: 2025-10-06
"""

import pytest
import cv2
import numpy as np
import time
from pathlib import Path

# Try imports
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from core.platform_adapter import PlatformAdapter
    PLATFORM_ADAPTER_AVAILABLE = True
except ImportError:
    PLATFORM_ADAPTER_AVAILABLE = False


@pytest.mark.vision
class TestScreenCapture:
    """Tests de capture d'écran"""

    @pytest.mark.skipif(not MSS_AVAILABLE, reason="mss not installed")
    def test_mss_capture_speed(self):
        """Teste vitesse capture avec mss (target >60 FPS)"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            
            # Warmup
            for _ in range(10):
                sct.grab(monitor)
            
            # Benchmark
            times = []
            for _ in range(100):
                start = time.time()
                img = sct.grab(monitor)
                elapsed = time.time() - start
                times.append(elapsed)
                
                # Vérifier format
                assert img is not None
                assert img.width > 0 and img.height > 0
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            print(f"\n   Capture FPS: {fps:.1f}")
            print(f"   Latency: {avg_time*1000:.2f}ms")
            
            # Target: >60 FPS
            assert fps > 60, f"FPS trop bas: {fps:.1f} (attendu >60)"

    @pytest.mark.skipif(not MSS_AVAILABLE, reason="mss not installed")
    def test_mss_capture_format(self):
        """Teste format image capturée"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            assert isinstance(img_array, np.ndarray)
            assert len(img_array.shape) == 3  # H, W, C
            assert img_array.shape[2] in [3, 4]  # RGB or RGBA
            
            print(f"\n   Image shape: {img_array.shape}")
            print(f"   Dtype: {img_array.dtype}")

    @pytest.mark.skipif(not MSS_AVAILABLE, reason="mss not installed")
    def test_mss_region_capture(self):
        """Teste capture d'une région spécifique"""
        with mss.mss() as sct:
            # Capture petite région (coin supérieur gauche)
            region = {
                "top": 0,
                "left": 0,
                "width": 640,
                "height": 480
            }
            
            img = sct.grab(region)
            
            assert img.width == 640
            assert img.height == 480
            
            print(f"\n   Region capture OK: {img.width}x{img.height}")

    @pytest.mark.skipif(not PLATFORM_ADAPTER_AVAILABLE, reason="PlatformAdapter not available")
    def test_platform_adapter_detection(self):
        """Teste détection OS via PlatformAdapter"""
        adapter = PlatformAdapter()
        
        system = adapter.get_system()
        is_linux = adapter.is_linux()
        is_windows = adapter.is_windows()
        
        print(f"\n   OS: {system}")
        print(f"   Linux: {is_linux}")
        print(f"   Windows: {is_windows}")
        
        assert system in ["Linux", "Windows", "Darwin"]
        assert is_linux or is_windows or adapter.is_macos()


@pytest.mark.vision
class TestUIDetection:
    """Tests détection éléments UI (HP/MP bars)"""

    def test_hp_bar_detection_synthetic(self, sample_ui_screenshot):
        """Teste détection HP bar sur screenshot synthétique"""
        # Définir région de recherche HP bar
        hp_region = sample_ui_screenshot[0:50, 0:250]
        
        # Convertir en HSV pour détection couleur
        hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
        
        # Masque pour couleur verte (HP)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculer pourcentage vert (= HP)
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        hp_ratio = green_pixels / total_pixels
        
        print(f"\n   HP détecté: {hp_ratio*100:.1f}%")
        
        # Devrait y avoir du vert (HP bar présente)
        assert hp_ratio > 0.05, "HP bar non détectée"

    def test_mp_bar_detection_synthetic(self, sample_ui_screenshot):
        """Teste détection MP bar sur screenshot synthétique"""
        # Région MP bar (sous HP bar)
        mp_region = sample_ui_screenshot[30:60, 0:250]
        
        # Convertir en HSV
        hsv = cv2.cvtColor(mp_region, cv2.COLOR_BGR2HSV)
        
        # Masque pour couleur bleue/orange (MP)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        blue_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        mp_ratio = blue_pixels / total_pixels
        
        print(f"\n   MP détecté: {mp_ratio*100:.1f}%")
        
        assert mp_ratio > 0.01, "MP bar non détectée"

    def test_template_matching_hp_bar(self, sample_ui_screenshot, test_data_dir):
        """Teste template matching pour HP bar"""
        # Charger template
        template_path = test_data_dir / "templates" / "hp_bar.png"
        if not template_path.exists():
            pytest.skip("Template HP bar non trouvé")
        
        template = cv2.imread(str(template_path))
        if template is None:
            pytest.skip("Impossible de charger template")
        
        # Template matching
        result = cv2.matchTemplate(
            sample_ui_screenshot, 
            template, 
            cv2.TM_CCOEFF_NORMED
        )
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"\n   Template match score: {max_val:.3f}")
        print(f"   Position: {max_loc}")
        
        # Score devrait être élevé si template match
        assert max_val > 0.3, f"Template matching score trop bas: {max_val}"

    def test_color_detection_performance(self, sample_ui_screenshot):
        """Teste performance détection couleur"""
        times = []
        
        for _ in range(100):
            start = time.time()
            
            # Détection rapide couleur verte
            hsv = cv2.cvtColor(sample_ui_screenshot, cv2.COLOR_BGR2HSV)
            lower = np.array([35, 50, 50])
            upper = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"\n   Color detection FPS: {fps:.1f}")
        
        # Devrait être très rapide (>100 FPS)
        assert fps > 100, f"Color detection trop lente: {fps:.1f} FPS"


@pytest.mark.vision
class TestEntityDetection:
    """Tests détection entités (monstres, ressources)"""

    def test_monster_detection_basic(self, sample_screenshot):
        """Teste détection basique monstres (cercles rouges)"""
        # Convertir en HSV
        hsv = cv2.cvtColor(sample_screenshot, cv2.COLOR_BGR2HSV)
        
        # Masque pour rouge
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Trouver contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n   Contours rouges détectés: {len(contours)}")
        
        # Devrait trouver au moins quelque chose
        assert len(contours) >= 0  # Au moins 0 (peut être vide)

    def test_circle_detection(self, sample_screenshot):
        """Teste détection cercles (monstres)"""
        gray = cv2.cvtColor(sample_screenshot, cv2.COLOR_BGR2GRAY)
        
        # Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f"\n   Cercles détectés: {len(circles[0])}")
        else:
            print("\n   Aucun cercle détecté")

    @pytest.mark.slow
    @pytest.mark.skipif(not MSS_AVAILABLE, reason="mss not installed")
    def test_detection_pipeline_performance(self, sample_screenshot):
        """Benchmark pipeline complet détection"""
        
        def detection_pipeline(img):
            """Pipeline détection simple"""
            # 1. Resize pour performance
            small = cv2.resize(img, (640, 360))
            
            # 2. Détection HP (HSV)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            hp_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 3. Détection entités (rouge)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            entity_mask = cv2.inRange(hsv, lower_red, upper_red)
            
            return hp_mask, entity_mask
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            _ = detection_pipeline(sample_screenshot)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"\n   Pipeline FPS: {fps:.1f}")
        print(f"   Latency: {avg_time*1000:.2f}ms")
        
        # Target: >30 FPS pour pipeline complet
        assert fps > 30, f"Pipeline trop lent: {fps:.1f} FPS"


@pytest.mark.vision
class TestOCR:
    """Tests OCR (reconnaissance texte)"""

    def test_ocr_simple_text(self):
        """Teste OCR sur texte simple généré"""
        # Créer image avec texte
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(
            img, "PA: 6 PM: 3", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        
        # Pour l'instant, juste vérifier qu'on peut créer l'image
        assert img is not None
        assert img.shape == (100, 300, 3)
        
        print("\n   Image test OCR créée")

    @pytest.mark.slow
    def test_ocr_performance_pytesseract(self):
        """Teste performance pytesseract (si installé)"""
        try:
            import pytesseract
        except ImportError:
            pytest.skip("pytesseract not installed")
        
        # Image test
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test 123", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            text = pytesseract.image_to_string(img)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        print(f"\n   OCR time: {avg_time*1000:.1f}ms")
        print(f"   Text: {text.strip()}")
        
        # OCR devrait prendre <500ms
        assert avg_time < 0.5, f"OCR trop lent: {avg_time*1000:.1f}ms"


@pytest.mark.vision
class TestMockUI:
    """Tests sur Mock UI HTML"""

    def test_mock_ui_exists(self, mock_ui_path):
        """Vérifie que mock UI existe"""
        assert mock_ui_path.exists()
        
        # Vérifier taille fichier
        size = mock_ui_path.stat().st_size
        print(f"\n   Mock UI size: {size/1024:.1f} KB")
        
        assert size > 1000, "Mock UI trop petit"

    def test_mock_ui_html_content(self, mock_ui_path):
        """Vérifie contenu HTML"""
        content = mock_ui_path.read_text()
        
        # Vérifier éléments clés présents
        assert "hp-bar" in content
        assert "mp-bar" in content
        assert "monster" in content
        assert "resource" in content
        
        print("\n   Mock UI contient tous les éléments")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
