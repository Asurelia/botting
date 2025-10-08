"""
Test Threading Fix - Vérification fix mss threading bug
Tests que VisionCaptureAdapter fonctionne correctement avec threads multiples

Author: Claude Code
Date: 2025-10-07
"""

import pytest
import threading
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.vision_capture_adapter import VisionCaptureAdapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False


@pytest.mark.skipif(not ADAPTER_AVAILABLE, reason="VisionCaptureAdapter not available")
class TestThreadingFix:
    """Tests pour vérifier le fix du bug threading mss"""

    def test_single_thread_capture(self):
        """Test capture dans un seul thread (baseline)"""
        adapter = VisionCaptureAdapter()

        # Capture sans fenêtre spécifique (écran complet)
        img = adapter.capture()

        assert img is not None, "Capture should succeed"
        assert len(img.shape) == 3, "Image should be 3D array"
        assert img.shape[2] == 3, "Image should be BGR (3 channels)"

        print(f"\n   ✅ Single thread capture OK: {img.shape}")

    def test_multi_thread_capture_parallel(self):
        """
        Test capture dans threads multiples en parallèle

        C'est le test clé qui révèle le bug threading.
        Avant fix: 'thread._local' object has no attribute 'display'
        Après fix: Devrait fonctionner sans erreur
        """
        adapter = VisionCaptureAdapter()

        results = []
        errors = []

        def capture_worker(thread_id: int, num_captures: int = 5):
            """Worker thread qui fait plusieurs captures"""
            try:
                for i in range(num_captures):
                    img = adapter.capture()
                    if img is not None:
                        results.append({
                            'thread_id': thread_id,
                            'capture_id': i,
                            'shape': img.shape,
                            'success': True
                        })
                    else:
                        errors.append({
                            'thread_id': thread_id,
                            'capture_id': i,
                            'error': 'Capture returned None'
                        })

                    # Small delay to simulate real usage
                    time.sleep(0.01)

            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'type': type(e).__name__
                })

        # Lancer 4 threads en parallèle
        num_threads = 4
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=capture_worker, args=(i, 5))
            t.start()
            threads.append(t)

        # Attendre que tous les threads finissent
        for t in threads:
            t.join(timeout=10)

        # Vérifier résultats
        print(f"\n   Captures réussies: {len(results)}")
        print(f"   Erreurs: {len(errors)}")

        if errors:
            print("\n   ❌ ERREURS DÉTECTÉES:")
            for error in errors:
                print(f"      Thread {error.get('thread_id')}: {error.get('error')}")

        # Assertions
        assert len(errors) == 0, f"Threading errors detected: {errors}"
        assert len(results) == num_threads * 5, f"Expected {num_threads * 5} captures, got {len(results)}"

        print(f"\n   ✅ Multi-thread capture OK: {num_threads} threads x 5 captures")

    def test_thread_local_mss_instances(self):
        """
        Test que chaque thread obtient sa propre instance mss

        Vérifie que le mécanisme thread-local fonctionne
        """
        adapter = VisionCaptureAdapter()

        instances = {}
        lock = threading.Lock()

        def check_instance(thread_id: int):
            """Vérifie l'instance mss dans ce thread"""
            # Force la création de l'instance thread-local
            sct = adapter._get_sct()

            with lock:
                instances[thread_id] = id(sct)

        # Lancer plusieurs threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=check_instance, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Vérifier que chaque thread a une instance différente
        instance_ids = list(instances.values())
        unique_instances = len(set(instance_ids))

        print(f"\n   Threads: {len(instances)}")
        print(f"   Instances mss uniques: {unique_instances}")

        assert unique_instances == len(instances), \
            "Each thread should have its own mss instance"

        print(f"\n   ✅ Thread-local instances OK: {unique_instances} unique instances")

    @pytest.mark.slow
    def test_sustained_multi_thread_capture(self):
        """
        Test capture soutenue multi-thread (stress test)

        Simule usage réel du session_recorder avec 3 threads:
        - Thread 1: Video capture @ 30 FPS
        - Thread 2: State extraction @ 5 FPS
        - Thread 3: Periodic captures @ 10 FPS
        """
        adapter = VisionCaptureAdapter()

        results = {'video': [], 'state': [], 'periodic': []}
        errors = []
        stop_flag = threading.Event()

        def video_capture_thread():
            """Thread 1: Video @ 30 FPS"""
            fps_target = 30
            interval = 1.0 / fps_target

            try:
                while not stop_flag.is_set():
                    start = time.time()
                    img = adapter.capture()
                    if img is not None:
                        results['video'].append(time.time())

                    # Maintenir FPS
                    elapsed = time.time() - start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

            except Exception as e:
                errors.append(('video', str(e)))

        def state_extraction_thread():
            """Thread 2: State extraction @ 5 FPS"""
            fps_target = 5
            interval = 1.0 / fps_target

            try:
                while not stop_flag.is_set():
                    start = time.time()
                    img = adapter.capture()
                    if img is not None:
                        results['state'].append(time.time())

                    elapsed = time.time() - start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

            except Exception as e:
                errors.append(('state', str(e)))

        def periodic_capture_thread():
            """Thread 3: Periodic @ 10 FPS"""
            fps_target = 10
            interval = 1.0 / fps_target

            try:
                while not stop_flag.is_set():
                    start = time.time()
                    img = adapter.capture()
                    if img is not None:
                        results['periodic'].append(time.time())

                    elapsed = time.time() - start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

            except Exception as e:
                errors.append(('periodic', str(e)))

        # Lancer les 3 threads
        t1 = threading.Thread(target=video_capture_thread)
        t2 = threading.Thread(target=state_extraction_thread)
        t3 = threading.Thread(target=periodic_capture_thread)

        t1.start()
        t2.start()
        t3.start()

        # Laisser tourner 2 secondes
        time.sleep(2.0)

        # Stop
        stop_flag.set()

        t1.join(timeout=1)
        t2.join(timeout=1)
        t3.join(timeout=1)

        # Vérifier résultats
        video_fps = len(results['video']) / 2.0
        state_fps = len(results['state']) / 2.0
        periodic_fps = len(results['periodic']) / 2.0

        print(f"\n   Video thread: {len(results['video'])} frames (~{video_fps:.1f} FPS)")
        print(f"   State thread: {len(results['state'])} frames (~{state_fps:.1f} FPS)")
        print(f"   Periodic thread: {len(results['periodic'])} frames (~{periodic_fps:.1f} FPS)")
        print(f"   Erreurs: {len(errors)}")

        if errors:
            print("\n   ❌ ERREURS:")
            for thread_name, error in errors:
                print(f"      {thread_name}: {error}")

        # Assertions
        assert len(errors) == 0, f"Threading errors: {errors}"
        assert video_fps > 20, f"Video FPS trop bas: {video_fps:.1f}"
        assert state_fps > 3, f"State FPS trop bas: {state_fps:.1f}"
        assert periodic_fps > 7, f"Periodic FPS trop bas: {periodic_fps:.1f}"

        print(f"\n   ✅ Sustained multi-thread capture OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
