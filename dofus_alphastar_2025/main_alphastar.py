#!/usr/bin/env python3
"""
DOFUS AlphaStar 2025 - Main Entry Point
Lanceur principal pour le système AlphaStar complet
"""

import sys
import argparse
import logging
import signal
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, update_config
from ui.alphastar_dashboard import create_dashboard
from core.alphastar_engine import LeagueManager, create_league_system
from core.rl_training import RLlibTrainer, create_rllib_trainer

def setup_logging(level: str = "INFO"):
    """Configure le logging"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/alphastar.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_banner():
    """Affiche la bannière AlphaStar"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    🎮 DOFUS ALPHASTAR 2025 🤖                               ║
    ║                                                                              ║
    ║               Bot d'Intelligence Artificielle Avancé                        ║
    ║                    Inspiré d'AlphaStar + HRM                                ║
    ║                                                                              ║
    ║  🚀 Architecture: Transformer + LSTM + HRM + Multi-Agent League             ║
    ║  ⚡ GPU: AMD 7800XT avec ROCm/DirectML                                      ║
    ║  🧠 Vision: SAM 2 + TrOCR                                                   ║
    ║  🏆 Training: Ray RLlib + League System                                     ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def handle_interrupt(signum, frame):
    """Gestionnaire d'interruption propre"""
    print("\n⚠️ Interruption détectée - Arrêt en cours...")
    sys.exit(0)

def run_dashboard():
    """Lance le dashboard principal"""
    print("🖥️ Lancement du dashboard AlphaStar...")

    try:
        dashboard = create_dashboard()
        dashboard.run()
    except Exception as e:
        logging.error(f"Erreur dashboard: {e}")
        sys.exit(1)

def run_training_only(algorithm: str = "PPO", max_iterations: int = 1000):
    """Lance uniquement l'entraînement sans interface"""
    print(f"🚀 Lancement entraînement {algorithm}...")

    try:
        trainer = create_rllib_trainer(experiment_name=f"dofus_alphastar_{algorithm.lower()}")
        trainer.setup_algorithm(algorithm)

        results = trainer.train_single_agent(
            max_iterations=max_iterations,
            checkpoint_freq=100
        )

        print(f"✅ Entraînement terminé:")
        print(f"   - Itérations: {results['final_iteration']}")
        print(f"   - Reward final: {results['final_reward']:.2f}")
        print(f"   - Meilleur reward: {results['best_reward']:.2f}")
        print(f"   - Temps total: {results['training_time']:.1f}s")

    except Exception as e:
        logging.error(f"Erreur entraînement: {e}")
        sys.exit(1)

def run_league_only(league_size: int = 32, max_generations: int = 100):
    """Lance uniquement le système de league"""
    print(f"🏆 Lancement league (taille: {league_size})...")

    try:
        league_manager = create_league_system(league_size=league_size)

        for generation in range(max_generations):
            step_info = league_manager.step_league()

            if generation % 10 == 0:
                print(f"Generation {generation}:")
                print(f"   - Matches créés: {step_info['matches_created']}")
                print(f"   - Total parties: {step_info['total_games']}")
                print(f"   - Agents actifs: {len(league_manager.agent_pool.agents)}")

                # Afficher top 3
                ranking = league_manager.get_league_ranking()[:3]
                print("   - Top 3:")
                for i, agent in enumerate(ranking, 1):
                    print(f"     {i}. {agent['agent_id']} (ELO: {agent['elo']:.0f})")

        print("✅ League training terminé")

    except Exception as e:
        logging.error(f"Erreur league: {e}")
        sys.exit(1)

def run_test_mode():
    """Lance les tests de validation"""
    print("🧪 Lancement mode test...")

    try:
        # Test imports
        print("   - Test imports...")
        from core.networks import create_alphastar_model
        from core.hrm_reasoning import create_hrm_model
        from core.vision_engine_v2 import create_vision_engine

        # Test création modèles
        print("   - Test création modèles...")
        alphastar_model = create_alphastar_model(use_hrm=True)
        hrm_model = create_hrm_model()

        print("   - Test device AMD...")
        device = config.amd.device if hasattr(config.amd, 'device') else 'cpu'
        print(f"     Device: {device}")

        # Test configuration
        print("   - Test configuration...")
        print(f"     Hidden size: {config.hrm.hidden_size}")
        print(f"     League size: {config.alphastar.league_size}")
        print(f"     Mixed precision: {config.amd.use_mixed_precision}")

        print("✅ Tous les tests passés avec succès!")

    except Exception as e:
        logging.error(f"Erreur tests: {e}")
        print("❌ Tests échoués")
        sys.exit(1)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="DOFUS AlphaStar 2025 - Bot IA Avancé",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s                              # Lance le dashboard complet
  %(prog)s --mode training              # Entraînement uniquement
  %(prog)s --mode league                # League uniquement
  %(prog)s --mode test                  # Tests de validation
  %(prog)s --dashboard --league-size 16 # Dashboard avec petite league
        """
    )

    # Mode de lancement
    parser.add_argument("--mode", choices=["dashboard", "training", "league", "test"],
                       default="dashboard", help="Mode de lancement")

    # Options communes
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Niveau de logging")
    parser.add_argument("--config", type=str, help="Fichier de configuration personnalisé")

    # Options entraînement
    parser.add_argument("--algorithm", choices=["PPO", "IMPALA", "SAC"],
                       default="PPO", help="Algorithme d'entraînement")
    parser.add_argument("--max-iterations", type=int, default=1000,
                       help="Nombre max d'itérations d'entraînement")

    # Options league
    parser.add_argument("--league-size", type=int, default=32,
                       help="Taille de la league")
    parser.add_argument("--max-generations", type=int, default=100,
                       help="Nombre max de générations league")

    # Options avancées
    parser.add_argument("--no-amd-optimizations", action="store_true",
                       help="Désactive les optimisations AMD")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force l'utilisation du CPU uniquement")

    args = parser.parse_args()

    # Configuration du logging
    setup_logging(args.log_level)

    # Gestionnaire d'interruption
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Bannière
    print_banner()

    # Charger configuration personnalisée
    if args.config:
        print(f"📄 Chargement configuration: {args.config}")
        # TODO: Implémenter chargement config

    # Ajustements configuration selon arguments
    if args.no_amd_optimizations:
        update_config(amd={"use_mixed_precision": False, "use_directml": False})
        print("⚠️ Optimisations AMD désactivées")

    if args.cpu_only:
        update_config(amd={"use_directml": False})
        print("💻 Mode CPU forcé")

    # Configuration league size
    if args.league_size != 32:
        update_config(alphastar={"league_size": args.league_size})

    # Lancement selon le mode
    try:
        if args.mode == "dashboard":
            run_dashboard()

        elif args.mode == "training":
            run_training_only(args.algorithm, args.max_iterations)

        elif args.mode == "league":
            run_league_only(args.league_size, args.max_generations)

        elif args.mode == "test":
            run_test_mode()

    except KeyboardInterrupt:
        print("\n⚠️ Arrêt demandé par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Erreur fatale: {e}")
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Créer dossiers nécessaires
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # Lancement
    main()