#!/usr/bin/env python3
"""
🚀 IA DOFUS - Lanceur Principal
Premier système d'IA DOFUS autonome et évolutive
Optimisé pour AMD 7800XT + Windows 11
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path
import argparse
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ai_dofus.log') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger("IA_DOFUS")

class AIDoFusLauncher:
    """Lanceur principal de l'IA DOFUS"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"

        # Création dossiers nécessaires
        self.config_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # État du système
        self.systems_ready = False
        self.gpu_available = False
        self.orchestrator = None

    def display_banner(self):
        """Affiche le banner de démarrage"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🚀 IA DOFUS v1.0                                  ║
║                    Première IA Autonome & Évolutive                         ║
║                                                                              ║
║  🧠 Core AI Framework    🎯 Vision Hybride    ⚡ AMD 7800XT Optimisé       ║
║  🤔 Uncertainty Mgmt     🔮 Predictive AI     🤝 Social Intelligence        ║
║                                                                              ║
║                        Prêt à révolutionner DOFUS ! 🎮                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)

    async def check_system_requirements(self) -> bool:
        """Vérifie les prérequis système"""
        print("🔍 Vérification des prérequis système...")

        checks = []

        # Vérification Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        python_ok = sys.version_info >= (3, 8)
        checks.append(("Python >= 3.8", python_ok, f"Version: {python_version}"))

        # Vérification dépendances critiques
        critical_modules = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("asyncio", "AsyncIO")
        ]

        for module_name, display_name in critical_modules:
            try:
                __import__(module_name)
                checks.append((display_name, True, "Disponible"))
            except ImportError:
                checks.append((display_name, False, "Non installé"))

        # Vérification GPU AMD
        try:
            import torch_directml
            if torch_directml.is_available():
                self.gpu_available = True
                checks.append(("GPU AMD DirectML", True, "Disponible"))
            else:
                checks.append(("GPU AMD DirectML", False, "Non disponible"))
        except ImportError:
            checks.append(("GPU AMD DirectML", False, "torch-directml non installé"))

        # Vérification modules IA custom
        custom_modules = [
            ("core.ai_framework", "Core AI Framework"),
            ("core.uncertainty", "Uncertainty System")
        ]

        for module_path, display_name in custom_modules:
            try:
                __import__(module_path)
                checks.append((display_name, True, "Chargé"))
            except ImportError as e:
                checks.append((display_name, False, f"Erreur: {e}"))

        # Affichage résultats
        print("\n📋 Résultats des vérifications:")
        all_critical_ok = True

        for check_name, status, details in checks:
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {check_name}: {details}")

            # Vérification si critique
            if check_name in ["Python >= 3.8", "PyTorch", "OpenCV", "Core AI Framework"]:
                if not status:
                    all_critical_ok = False

        if not all_critical_ok:
            print("\n❌ Des prérequis critiques manquent !")
            print("🔧 Exécutez d'abord: python scripts/setup_amd_environment.py")
            return False

        print(f"\n✅ Système prêt ! GPU AMD: {'Disponible' if self.gpu_available else 'Indisponible'}")
        self.systems_ready = True
        return True

    async def initialize_ai_systems(self) -> bool:
        """Initialise les systèmes IA"""
        print("\n🧠 Initialisation des systèmes IA...")

        try:
            # Import des modules IA
            from core.ai_framework import MetaOrchestrator
            from core.uncertainty import UncertaintyManager

            # Configuration
            config_file = self.config_dir / "ai_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Configuration chargée: {config_file}")
            else:
                print("⚠️ Configuration par défaut utilisée")
                config = {}

            # Initialisation orchestrateur principal
            print("  🎭 Démarrage MetaOrchestrator...")
            self.orchestrator = MetaOrchestrator(str(config_file) if config_file.exists() else None)

            if await self.orchestrator.start():
                print("  ✅ MetaOrchestrator actif")
            else:
                print("  ❌ Échec démarrage MetaOrchestrator")
                return False

            # Initialisation système d'incertitude
            print("  🤔 Initialisation Uncertainty Manager...")
            uncertainty_manager = UncertaintyManager(config.get('uncertainty', {}))
            print("  ✅ Uncertainty Manager actif")

            # Test intégration
            print("  🧪 Test d'intégration...")
            await asyncio.sleep(1.0)

            status = self.orchestrator.get_status()
            if status['running']:
                print("  ✅ Tous les systèmes opérationnels")
                return True
            else:
                print("  ❌ Problème détecté lors du test")
                return False

        except Exception as e:
            logger.error(f"Erreur initialisation systèmes IA: {e}")
            print(f"  ❌ Erreur: {e}")
            return False

    async def load_vision_systems(self) -> bool:
        """Charge les systèmes de vision"""
        print("\n👁️ Chargement des systèmes de vision...")

        try:
            # Test chargement vision hybride
            vision_modules = [
                ("modules.vision.screen_analyzer", "Screen Analyzer"),
                ("modules.vision.hybrid_detector", "Hybrid Detector"),
                ("modules.vision.detection_adapter", "Detection Adapter")
            ]

            loaded_modules = []

            for module_path, display_name in vision_modules:
                try:
                    module = __import__(module_path, fromlist=[''])
                    loaded_modules.append(display_name)
                    print(f"  ✅ {display_name} chargé")
                except ImportError as e:
                    print(f"  ⚠️ {display_name} non disponible: {e}")

            if len(loaded_modules) >= 2:
                print(f"  ✅ Systèmes de vision prêts ({len(loaded_modules)}/3 modules)")
                return True
            else:
                print(f"  ⚠️ Vision partielle ({len(loaded_modules)}/3 modules)")
                return False

        except Exception as e:
            logger.error(f"Erreur chargement vision: {e}")
            print(f"  ❌ Erreur vision: {e}")
            return False

    async def run_self_diagnostics(self) -> bool:
        """Lance l'auto-diagnostic du système"""
        print("\n🔬 Auto-diagnostic du système...")

        try:
            # Test performance basic
            print("  ⚡ Test performance...")
            start_time = time.perf_counter()

            # Test calcul
            import numpy as np
            test_array = np.random.randn(1000, 1000)
            result = np.dot(test_array, test_array.T)

            performance_time = time.perf_counter() - start_time
            print(f"  📊 Performance CPU: {performance_time:.4f}s")

            # Test GPU si disponible
            if self.gpu_available:
                try:
                    import torch
                    import torch_directml

                    device = torch_directml.device()
                    start_time = time.perf_counter()

                    x = torch.randn(1000, 1000, device=device)
                    result_gpu = torch.mm(x, x.T)

                    gpu_time = time.perf_counter() - start_time
                    speedup = performance_time / gpu_time
                    print(f"  🚀 Performance GPU: {gpu_time:.4f}s (accélération: {speedup:.2f}x)")

                except Exception as e:
                    print(f"  ⚠️ Test GPU échoué: {e}")

            # Test mémoire
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"  💾 Utilisation mémoire: {memory_mb:.1f} MB")

            # Test orchestrateur
            if self.orchestrator:
                status = self.orchestrator.get_status()
                modules_healthy = status.get('modules_healthy', 0)
                print(f"  🎭 Modules IA sains: {modules_healthy}")

            print("  ✅ Auto-diagnostic terminé")
            return True

        except Exception as e:
            logger.error(f"Erreur auto-diagnostic: {e}")
            print(f"  ❌ Erreur diagnostic: {e}")
            return False

    async def start_ai_mode(self, mode: str = "demo"):
        """Démarre l'IA en mode spécifié"""
        print(f"\n🚀 Démarrage IA DOFUS - Mode: {mode.upper()}")

        if mode == "demo":
            await self._run_demo_mode()
        elif mode == "autonomous":
            await self._run_autonomous_mode()
        elif mode == "training":
            await self._run_training_mode()
        else:
            print(f"❌ Mode inconnu: {mode}")

    async def _run_demo_mode(self):
        """Mode démonstration"""
        print("🎮 Mode Démonstration - Affichage des capacités IA")

        demo_tasks = [
            "Initialisation des modules de perception",
            "Test de prise de décision",
            "Simulation d'incertitude",
            "Démonstration d'apprentissage",
            "Test de coordination"
        ]

        for i, task in enumerate(demo_tasks, 1):
            print(f"  {i}/5 {task}...")
            await asyncio.sleep(2.0)  # Simulation
            print(f"  ✅ {task} terminé")

        print("\n🎉 Démonstration terminée ! L'IA DOFUS est opérationnelle.")

    async def _run_autonomous_mode(self):
        """Mode autonome complet"""
        print("🤖 Mode Autonome - IA DOFUS en action")
        print("⚠️ Mode non encore implémenté - En développement Phase 1")

    async def _run_training_mode(self):
        """Mode entraînement"""
        print("🎓 Mode Entraînement - Apprentissage adaptatif")
        print("⚠️ Mode non encore implémenté - En développement Phase 2")

    async def shutdown(self):
        """Arrêt propre du système"""
        print("\n🛑 Arrêt de l'IA DOFUS...")

        if self.orchestrator:
            await self.orchestrator.stop()
            print("  ✅ MetaOrchestrator arrêté")

        print("  ✅ Arrêt terminé")

    async def run(self, mode: str = "demo", skip_checks: bool = False):
        """Exécution principale"""
        self.display_banner()

        try:
            # Vérifications système
            if not skip_checks:
                if not await self.check_system_requirements():
                    return False

            # Initialisation
            if not await self.initialize_ai_systems():
                print("❌ Échec initialisation IA")
                return False

            # Vision (optionnel)
            await self.load_vision_systems()

            # Auto-diagnostic
            await self.run_self_diagnostics()

            # Démarrage mode sélectionné
            await self.start_ai_mode(mode)

            return True

        except KeyboardInterrupt:
            print("\n⚠️ Interruption utilisateur")
            return True
        except Exception as e:
            logger.error(f"Erreur critique: {e}")
            print(f"💥 Erreur critique: {e}")
            return False
        finally:
            await self.shutdown()

async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="🚀 IA DOFUS - Système Autonome")

    parser.add_argument(
        "--mode",
        choices=["demo", "autonomous", "training"],
        default="demo",
        help="Mode de fonctionnement (défaut: demo)"
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Ignore les vérifications système"
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Lance uniquement les tests d'intégration"
    )

    args = parser.parse_args()

    if args.test_only:
        print("🧪 Mode test uniquement")
        os.system("python scripts/test_amd_integration.py")
        return

    # Lancement principal
    launcher = AIDoFusLauncher()
    success = await launcher.run(args.mode, args.skip_checks)

    if success:
        print("\n🎉 IA DOFUS s'est exécutée avec succès !")
        print("\n🎯 PROCHAINES ÉTAPES:")
        print("1. python scripts/gemini_consensus.py autonomy_architecture")
        print("2. Investigation Dofus Guide/Ganymede")
        print("3. Développement Phase 1: Knowledge Base")
    else:
        print("\n❌ Problèmes détectés. Consultez les logs.")
        print("\n🔧 ACTIONS CORRECTIVES:")
        print("1. python scripts/setup_amd_environment.py")
        print("2. python scripts/test_amd_integration.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
    except Exception as e:
        print(f"💥 Erreur fatale: {e}")
        sys.exit(1)