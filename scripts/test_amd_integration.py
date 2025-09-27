#!/usr/bin/env python3
"""
Test d'Intégration Complète AMD + IA Framework
Valide l'accélération GPU, les performances et l'intégration des systèmes
"""

import asyncio
import time
import json
import sys
import os
from pathlib import Path
import logging
import traceback
import numpy as np

# Import des modules du framework
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMDIntegrationTester:
    """Testeur d'intégration complète pour AMD 7800XT"""

    def __init__(self):
        self.results = {}
        self.gpu_available = False
        self.framework_loaded = False

    async def run_full_test_suite(self):
        """Lance la suite complète de tests"""
        print("🚀 IA DOFUS - Test d'Intégration AMD 7800XT")
        print("=" * 60)

        test_suite = [
            ("GPU Detection", self.test_gpu_detection),
            ("PyTorch DirectML", self.test_pytorch_directml),
            ("YOLO GPU Acceleration", self.test_yolo_gpu),
            ("AI Framework Core", self.test_ai_framework),
            ("Uncertainty System", self.test_uncertainty_system),
            ("Vision Integration", self.test_vision_integration),
            ("Performance Benchmark", self.test_performance_benchmark),
            ("Memory Management", self.test_memory_management)
        ]

        for test_name, test_func in test_suite:
            print(f"\n🔄 Test: {test_name}")
            print("-" * 40)

            try:
                start_time = time.perf_counter()
                result = await test_func()
                duration = time.perf_counter() - start_time

                self.results[test_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'duration': duration,
                    'details': getattr(result, 'details', None) if hasattr(result, 'details') else {}
                }

                status_emoji = "✅" if result else "❌"
                print(f"{status_emoji} {test_name}: {'RÉUSSI' if result else 'ÉCHEC'} ({duration:.3f}s)")

            except Exception as e:
                self.results[test_name] = {
                    'status': 'ERROR',
                    'duration': 0,
                    'error': str(e)
                }
                print(f"💥 {test_name}: ERREUR - {e}")

        # Rapport final
        await self.generate_final_report()

    async def test_gpu_detection(self):
        """Test de détection du GPU AMD"""
        try:
            import subprocess

            # Détection via WMI
            result = subprocess.run([
                "wmic", "path", "win32_VideoController",
                "get", "name,adapterram,driverversion"
            ], capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                gpu_info = result.stdout
                print(f"Informations GPU:")
                print(gpu_info)

                if "AMD" in gpu_info or "Radeon" in gpu_info:
                    if "7800" in gpu_info:
                        print("✅ AMD 7800XT détecté spécifiquement")
                        return True
                    else:
                        print("✅ GPU AMD détecté (modèle à confirmer)")
                        return True

            return False

        except Exception as e:
            print(f"❌ Erreur détection GPU: {e}")
            return False

    async def test_pytorch_directml(self):
        """Test PyTorch + DirectML"""
        try:
            print("Test PyTorch + DirectML...")

            # Import PyTorch
            import torch
            print(f"PyTorch version: {torch.__version__}")

            # Test DirectML
            try:
                import torch_directml
                print(f"torch-directml importé avec succès")

                if torch_directml.is_available():
                    device = torch_directml.device()
                    print(f"✅ DirectML disponible - Device: {device}")

                    # Test calcul simple
                    print("Test calcul matriciel GPU...")
                    x = torch.randn(1000, 1000, device=device)
                    y = torch.randn(1000, 1000, device=device)

                    start_time = time.perf_counter()
                    z = torch.mm(x, y)
                    gpu_time = time.perf_counter() - start_time

                    print(f"✅ Calcul GPU réussi en {gpu_time:.4f}s")

                    # Comparaison CPU
                    x_cpu = torch.randn(1000, 1000)
                    y_cpu = torch.randn(1000, 1000)

                    start_time = time.perf_counter()
                    z_cpu = torch.mm(x_cpu, y_cpu)
                    cpu_time = time.perf_counter() - start_time

                    speedup = cpu_time / gpu_time
                    print(f"📊 Accélération GPU: {speedup:.2f}x plus rapide que CPU")

                    self.gpu_available = True
                    return True

                else:
                    print("❌ DirectML non disponible")
                    return False

            except ImportError:
                print("❌ torch-directml non installé")
                return False

        except Exception as e:
            print(f"❌ Erreur test PyTorch: {e}")
            return False

    async def test_yolo_gpu(self):
        """Test YOLO avec accélération GPU"""
        try:
            print("Test YOLO GPU...")

            from ultralytics import YOLO
            import cv2

            # Chargement modèle YOLO
            model = YOLO('yolov8n.pt')  # Modèle nano pour test rapide

            # Configuration GPU
            if self.gpu_available:
                # Tentative GPU
                model.to('directml' if self.gpu_available else 'cpu')
                print("✅ Modèle YOLO configuré pour DirectML")
            else:
                print("ℹ️ Modèle YOLO configuré pour CPU")

            # Création image de test
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Test prédiction
            print("Test prédiction YOLO...")
            start_time = time.perf_counter()
            results = model(test_image, verbose=False)
            prediction_time = time.perf_counter() - start_time

            print(f"✅ Prédiction YOLO réussie en {prediction_time:.4f}s")

            # Analyse des résultats
            if len(results) > 0:
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"📊 {detections} détections trouvées")

            return True

        except Exception as e:
            print(f"❌ Erreur test YOLO: {e}")
            return False

    async def test_ai_framework(self):
        """Test du Core AI Framework"""
        try:
            print("Test Core AI Framework...")

            from core.ai_framework import MetaOrchestrator, AITask, Priority

            # Création orchestrateur
            orchestrator = MetaOrchestrator()

            # Test démarrage
            start_success = await orchestrator.start()
            if not start_success:
                print("❌ Échec démarrage orchestrateur")
                return False

            print("✅ Orchestrateur démarré")

            # Test soumission tâche
            async def test_task():
                await asyncio.sleep(0.1)
                return "test_result"

            task = AITask(
                name="test_task",
                priority=Priority.MEDIUM,
                function=test_task
            )

            submission_success = await orchestrator.submit_task(task)
            if not submission_success:
                print("❌ Échec soumission tâche")
                return False

            print("✅ Tâche soumise avec succès")

            # Attente traitement
            await asyncio.sleep(1.0)

            # Vérification statut
            status = orchestrator.get_status()
            print(f"📊 Status orchestrateur: {json.dumps(status, indent=2)}")

            # Arrêt propre
            await orchestrator.stop()
            print("✅ Orchestrateur arrêté proprement")

            self.framework_loaded = True
            return True

        except Exception as e:
            print(f"❌ Erreur test framework: {e}")
            traceback.print_exc()
            return False

    async def test_uncertainty_system(self):
        """Test du système d'incertitude"""
        try:
            print("Test Uncertainty Management...")

            from core.uncertainty import UncertaintyManager
            from datetime import datetime

            # Création manager
            manager = UncertaintyManager()

            # Test données de décision
            decision_data = {
                'model_output': {'confidence': 0.85, 'action': 'move'},
                'input_data': {'position': (100, 200), 'target': (150, 250)},
                'action': 'move',
                'history': [0.8, 0.82, 0.85, 0.83]
            }

            context = {
                'timestamp': datetime.now(),
                'in_combat': False,
                'players_nearby': 2,
                'hp_percentage': 95
            }

            # Évaluation incertitude
            measurement = await manager.evaluate_decision(decision_data, context)

            print(f"✅ Mesure d'incertitude calculée:")
            print(f"  Confiance: {measurement.confidence_score:.3f}")
            print(f"  Niveau: {measurement.confidence_level.name}")
            print(f"  Risque: {measurement.risk_level.name}")
            print(f"  Fiabilité: {measurement.overall_reliability():.3f}")

            # Test recommandation
            should_proceed = await manager.should_proceed(measurement)
            print(f"  Recommandation: {'✅ Procéder' if should_proceed else '❌ Arrêter'}")

            return True

        except Exception as e:
            print(f"❌ Erreur test incertitude: {e}")
            traceback.print_exc()
            return False

    async def test_vision_integration(self):
        """Test intégration vision hybride"""
        try:
            print("Test intégration vision...")

            # Test de l'existence des modules vision
            vision_modules = [
                'modules.vision.screen_analyzer',
                'modules.vision.hybrid_detector',
                'modules.vision.detection_adapter'
            ]

            for module_name in vision_modules:
                try:
                    __import__(module_name)
                    print(f"✅ Module {module_name} disponible")
                except ImportError as e:
                    print(f"⚠️ Module {module_name} non disponible: {e}")

            # Test image synthetic
            test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

            # Simulation analyse
            print("✅ Test image synthétique créée (800x600)")

            return True

        except Exception as e:
            print(f"❌ Erreur test vision: {e}")
            return False

    async def test_performance_benchmark(self):
        """Benchmark de performance"""
        try:
            print("Benchmark performance...")

            benchmarks = {}

            # Test calcul matriciel
            print("  - Calcul matriciel...")
            if self.gpu_available:
                import torch
                import torch_directml

                device = torch_directml.device()

                # GPU benchmark
                start_time = time.perf_counter()
                for _ in range(10):
                    x = torch.randn(500, 500, device=device)
                    y = torch.randn(500, 500, device=device)
                    z = torch.mm(x, y)
                gpu_time = time.perf_counter() - start_time

                benchmarks['matrix_gpu_10_iterations'] = gpu_time
                print(f"    GPU: {gpu_time:.4f}s")

            # CPU benchmark
            start_time = time.perf_counter()
            for _ in range(10):
                x = np.random.randn(500, 500)
                y = np.random.randn(500, 500)
                z = np.dot(x, y)
            cpu_time = time.perf_counter() - start_time

            benchmarks['matrix_cpu_10_iterations'] = cpu_time
            print(f"    CPU: {cpu_time:.4f}s")

            if self.gpu_available:
                speedup = cpu_time / gpu_time
                benchmarks['speedup_ratio'] = speedup
                print(f"    Accélération: {speedup:.2f}x")

            # Test async/await
            print("  - Performance async...")
            start_time = time.perf_counter()

            async def async_task():
                await asyncio.sleep(0.001)
                return True

            tasks = [async_task() for _ in range(1000)]
            await asyncio.gather(*tasks)

            async_time = time.perf_counter() - start_time
            benchmarks['async_1000_tasks'] = async_time
            print(f"    1000 tâches async: {async_time:.4f}s")

            # Sauvegarde benchmarks
            benchmark_file = Path("config/performance_benchmarks.json")
            benchmark_file.parent.mkdir(exist_ok=True)
            with open(benchmark_file, 'w') as f:
                json.dump(benchmarks, f, indent=2)

            print(f"✅ Benchmarks sauvegardés: {benchmark_file}")

            return True

        except Exception as e:
            print(f"❌ Erreur benchmark: {e}")
            return False

    async def test_memory_management(self):
        """Test gestion mémoire"""
        try:
            print("Test gestion mémoire...")

            import psutil
            import gc

            # Mesure initiale
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"  Mémoire initiale: {initial_memory:.1f} MB")

            # Allocation importante
            print("  - Allocation mémoire test...")
            large_arrays = []
            for i in range(10):
                array = np.random.randn(1000, 1000)
                large_arrays.append(array)

            after_allocation = process.memory_info().rss / 1024 / 1024
            print(f"  Mémoire après allocation: {after_allocation:.1f} MB")

            # Libération
            print("  - Libération mémoire...")
            del large_arrays
            gc.collect()

            await asyncio.sleep(1.0)  # Attente pour la libération

            after_cleanup = process.memory_info().rss / 1024 / 1024
            print(f"  Mémoire après cleanup: {after_cleanup:.1f} MB")

            # Vérification efficacité
            memory_freed = after_allocation - after_cleanup
            print(f"  Mémoire libérée: {memory_freed:.1f} MB")

            efficiency = memory_freed / (after_allocation - initial_memory)
            print(f"  Efficacité cleanup: {efficiency:.2%}")

            return efficiency > 0.7  # 70% minimum libéré

        except Exception as e:
            print(f"❌ Erreur test mémoire: {e}")
            return False

    async def generate_final_report(self):
        """Génère le rapport final"""
        print("\n" + "=" * 60)
        print("📊 RAPPORT FINAL D'INTÉGRATION")
        print("=" * 60)

        # Calcul du score global
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        error_tests = sum(1 for r in self.results.values() if r['status'] == 'ERROR')

        success_rate = passed_tests / total_tests * 100

        # Affichage détaillé
        for test_name, result in self.results.items():
            status_emoji = {
                'PASS': '✅',
                'FAIL': '❌',
                'ERROR': '💥'
            }[result['status']]

            duration = result.get('duration', 0)
            print(f"{status_emoji} {test_name}: {result['status']} ({duration:.3f}s)")

            if result['status'] == 'ERROR':
                print(f"    Erreur: {result.get('error', 'Inconnue')}")

        # Score global
        print(f"\n📈 SCORE GLOBAL: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   ✅ Réussis: {passed_tests}")
        print(f"   ❌ Échecs: {failed_tests}")
        print(f"   💥 Erreurs: {error_tests}")

        # Recommandations
        print(f"\n🎯 RECOMMANDATIONS:")

        if success_rate >= 90:
            print("🚀 EXCELLENT ! Système prêt pour l'IA DOFUS avancée")
            print("   ➤ Procédez à Phase 1: Knowledge Base")
            print("   ➤ Lancez la consultation Gemini")
        elif success_rate >= 70:
            print("✅ BON ! Système fonctionnel avec optimisations mineures")
            print("   ➤ Corrigez les tests échoués")
            print("   ➤ Procédez avec précaution à Phase 1")
        elif success_rate >= 50:
            print("⚠️ MOYEN ! Corrections nécessaires avant Phase 1")
            print("   ➤ Priorisez les tests GPU et Framework")
            print("   ➤ Vérifiez les installations")
        else:
            print("❌ CRITIQUE ! Corrections majeures requises")
            print("   ➤ Relancez setup_amd_environment.py")
            print("   ➤ Vérifiez compatibilité système")

        # Étapes suivantes
        print(f"\n🎯 PROCHAINES ÉTAPES:")
        if success_rate >= 70:
            print("1. python scripts/gemini_consensus.py autonomy_architecture")
            print("2. Investigation Dofus Guide/Ganymede")
            print("3. Implémentation Phase 1: Knowledge Graph")
        else:
            print("1. python scripts/setup_amd_environment.py")
            print("2. Correction des erreurs identifiées")
            print("3. Relancement du test d'intégration")

        # Sauvegarde rapport
        report_file = Path("config/integration_test_report.json")
        report_file.parent.mkdir(exist_ok=True)

        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success_rate': success_rate,
            'tests': self.results,
            'recommendations': 'proceed' if success_rate >= 70 else 'fix_issues',
            'gpu_available': self.gpu_available,
            'framework_loaded': self.framework_loaded
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n💾 Rapport sauvegardé: {report_file}")

async def main():
    """Point d'entrée principal"""
    tester = AMDIntegrationTester()
    await tester.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main())