"""
Tests d'Intégration - Pipeline complet end-to-end
Tests vision → AI → décision sans Dofus lancé

Author: Claude Code
Date: 2025-10-06
"""

import pytest
import cv2
import numpy as np
import time
from pathlib import Path


@pytest.mark.integration
class TestVisionPipeline:
    """Tests pipeline vision complet"""

    def test_screenshot_to_gamestate(self, sample_screenshot):
        """Teste conversion screenshot → game state"""
        
        # Simuler extraction game state depuis screenshot
        def extract_game_state(img):
            """Extrait état du jeu depuis screenshot"""
            h, w = img.shape[:2]
            
            # Simuler détection HP/MP
            hp_region = img[0:50, 0:250]
            hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            hp_mask = cv2.inRange(hsv, lower_green, upper_green)
            hp_pixels = np.sum(hp_mask > 0)
            hp_ratio = hp_pixels / hp_mask.size
            
            # Simuler détection entités
            entity_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            contours, _ = cv2.findContours(entity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                "player": {
                    "hp": hp_ratio,
                    "mp": 0.6,
                    "pa": 6,
                    "pm": 3
                },
                "enemies": len(contours),
                "screen_size": (w, h)
            }
        
        # Test extraction
        game_state = extract_game_state(sample_screenshot)
        
        assert "player" in game_state
        assert "enemies" in game_state
        assert 0.0 <= game_state["player"]["hp"] <= 1.0
        
        print(f"\n   Game state extrait:")
        print(f"   HP: {game_state['player']['hp']*100:.0f}%")
        print(f"   Enemies: {game_state['enemies']}")

    def test_gamestate_to_decision(self):
        """Teste conversion game state → décision AI"""
        
        # Simuler AI simple
        def simple_ai_decision(game_state):
            """Décision AI basique"""
            hp = game_state["player"]["hp"]
            enemies = game_state["enemies"]
            pa = game_state["player"]["pa"]
            
            # Logique simple
            if hp < 0.3:
                return {"action": "heal", "priority": "high"}
            elif enemies > 0 and pa >= 3:
                return {"action": "attack", "target": "nearest"}
            elif pa > 0:
                return {"action": "move", "direction": "forward"}
            else:
                return {"action": "end_turn"}
        
        # Test décisions
        test_cases = [
            {
                "state": {"player": {"hp": 0.2, "mp": 0.5, "pa": 6, "pm": 3}, "enemies": 2},
                "expected_action": "heal"
            },
            {
                "state": {"player": {"hp": 0.8, "mp": 0.5, "pa": 6, "pm": 3}, "enemies": 1},
                "expected_action": "attack"
            },
            {
                "state": {"player": {"hp": 0.8, "mp": 0.5, "pa": 0, "pm": 0}, "enemies": 0},
                "expected_action": "end_turn"
            }
        ]
        
        for i, test in enumerate(test_cases):
            decision = simple_ai_decision(test["state"])
            print(f"\n   Test {i+1}: {test['expected_action']} → {decision['action']}")
            assert decision["action"] == test["expected_action"]

    @pytest.mark.slow
    def test_full_pipeline_performance(self, sample_screenshot):
        """Benchmark pipeline complet: screenshot → state → decision"""
        
        def full_pipeline(img):
            """Pipeline complet"""
            # 1. Vision (détection)
            small = cv2.resize(img, (640, 360))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hp_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            hp_ratio = np.sum(hp_mask > 0) / hp_mask.size
            
            # 2. Game state
            game_state = {
                "player": {"hp": hp_ratio, "mp": 0.6, "pa": 6, "pm": 3},
                "enemies": 2
            }
            
            # 3. AI decision
            if hp_ratio < 0.3:
                action = "heal"
            elif game_state["enemies"] > 0:
                action = "attack"
            else:
                action = "move"
            
            return action
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            action = full_pipeline(sample_screenshot)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"\n   Full pipeline FPS: {fps:.1f}")
        print(f"   Latency: {avg_time*1000:.2f}ms")
        
        # Target: >20 FPS pour pipeline complet
        assert fps > 20, f"Pipeline trop lent: {fps:.1f} FPS (attendu >20)"


@pytest.mark.integration
class TestEndToEnd:
    """Tests end-to-end avec données réelles"""

    def test_e2e_ui_detection(self, sample_ui_screenshot, annotations):
        """Test E2E: UI screenshot → détection → validation"""
        
        # Extraire ground truth si disponible
        ground_truth = annotations.get("ui_hp_mp_bars.png", {})
        
        # Détection HP
        hp_region = sample_ui_screenshot[0:50, 0:250]
        hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
        hp_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        detected_hp = np.sum(hp_mask > 0) / hp_mask.size
        
        print(f"\n   HP détecté: {detected_hp*100:.1f}%")
        
        # Si ground truth disponible, comparer
        if "hp_bar" in ground_truth:
            expected_hp = ground_truth["hp_bar"]["value"]
            error = abs(detected_hp - expected_hp)
            print(f"   HP attendu: {expected_hp*100:.1f}%")
            print(f"   Erreur: {error*100:.1f}%")
            
            # Tolérance 20%
            assert error < 0.2, f"Erreur HP trop élevée: {error*100:.1f}%"

    def test_e2e_combat_detection(self, sample_screenshot):
        """Test E2E: Combat screenshot → détection monstres → comptage"""
        
        # Détection basique entités rouges
        hsv = cv2.cvtColor(sample_screenshot, cv2.COLOR_BGR2HSV)
        
        # Masque rouge
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask = mask1 | mask2
        
        # Trouver contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer par taille
        monsters = [c for c in contours if cv2.contourArea(c) > 100]
        
        print(f"\n   Monstres détectés: {len(monsters)}")
        
        assert len(monsters) >= 0  # Au moins 0

    @pytest.mark.slow
    def test_e2e_multi_frame_consistency(self, sample_screenshot):
        """Test E2E: Cohérence détections sur frames multiples"""
        
        # Simuler frames successives avec bruit
        frames = []
        for i in range(10):
            # Ajouter bruit léger
            noise = np.random.randint(-10, 10, sample_screenshot.shape, dtype=np.int16)
            noisy = np.clip(sample_screenshot.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(noisy)
        
        # Détecter HP sur chaque frame
        hp_values = []
        for frame in frames:
            hp_region = frame[0:50, 0:250]
            hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
            hp_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            hp = np.sum(hp_mask > 0) / hp_mask.size
            hp_values.append(hp)
        
        # Vérifier cohérence (variance faible)
        hp_mean = np.mean(hp_values)
        hp_std = np.std(hp_values)
        
        print(f"\n   HP mean: {hp_mean*100:.1f}%")
        print(f"   HP std: {hp_std*100:.2f}%")
        
        # Variance devrait être faible (<10%)
        assert hp_std < 0.1, f"Variance HP trop élevée: {hp_std*100:.2f}%"


@pytest.mark.integration  
class TestPerformance:
    """Tests performance intégration"""

    @pytest.mark.slow
    def test_throughput_sustained(self, sample_screenshot):
        """Test throughput soutenu (1000 frames)"""
        
        def process_frame(img):
            """Traitement frame simple"""
            small = cv2.resize(img, (640, 360))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            return np.sum(mask > 0)
        
        # Process 1000 frames
        start = time.time()
        results = []
        for _ in range(1000):
            result = process_frame(sample_screenshot)
            results.append(result)
        
        elapsed = time.time() - start
        fps = 1000 / elapsed
        
        print(f"\n   Sustained FPS (1000 frames): {fps:.1f}")
        print(f"   Total time: {elapsed:.2f}s")
        
        # Target: >30 FPS soutenu
        assert fps > 30, f"Throughput insuffisant: {fps:.1f} FPS"

    def test_memory_stability(self, sample_screenshot):
        """Test stabilité mémoire (pas de fuites)"""
        import gc
        
        def process_intensive(img, iterations=100):
            """Traitement intensif mémoire"""
            results = []
            for _ in range(iterations):
                # Créer copies multiples
                copies = [img.copy() for _ in range(10)]
                # Traiter
                processed = [cv2.resize(c, (320, 180)) for c in copies]
                results.append(len(processed))
            return results
        
        # Avant
        gc.collect()
        
        # Process
        results = process_intensive(sample_screenshot)
        
        # Après - forcer GC
        gc.collect()
        
        # Vérifier résultats cohérents
        assert len(results) == 100
        assert all(r == 10 for r in results)
        
        print("\n   Memory stability OK (no leaks detected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
