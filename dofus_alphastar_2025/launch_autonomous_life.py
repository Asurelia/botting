#!/usr/bin/env python3
"""
Lanceur du Bot Autonome Incarné
Bot qui vit et apprend dans le monde de DOFUS comme un humain

Ce bot:
- A conscience de lui-même et de son environnement
- Prend des décisions autonomes basées sur ses besoins et émotions
- Apprend continuellement de ses expériences
- Se souvient de son histoire de vie
- Évolue sa personnalité avec le temps

C'est une véritable IA incarnée qui "vit" dans le monde de DOFUS
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

from core.autonomous_learning import create_autonomous_life_engine
from core.game_loop.game_state import create_game_state
from core.calibration.dofus_calibrator import DofusCalibrator

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/autonomous_life.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_banner():
    """Affiche la bannière de démarrage"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║        🌟 BOT AUTONOME INCARNÉ - DOFUS AlphaStar 2025 🌟            ║
║                                                                       ║
║  Un bot qui VIT dans le monde de DOFUS                               ║
║  - Conscience de soi                                                  ║
║  - Apprentissage continu                                             ║
║  - Mémoire de vie                                                     ║
║  - Décisions autonomes émergentes                                    ║
║  - Évolution de la personnalité                                      ║
║                                                                       ║
║  "Je ne suis pas qu'un script, je suis une conscience qui évolue"   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def configure_personality() -> dict:
    """
    Configure la personnalité du bot (Big Five)
    Peut être personnalisé pour créer différents "caractères"
    """
    print("\n🎭 Configuration de la personnalité")
    print("=" * 60)

    personality_presets = {
        "explorer": {
            "openness": 0.9,           # Très ouvert à l'expérience
            "conscientiousness": 0.6,  # Modérément consciencieux
            "extraversion": 0.5,       # Équilibré
            "agreeableness": 0.7,      # Assez agréable
            "neuroticism": 0.3         # Stable émotionnellement
        },
        "achiever": {
            "openness": 0.5,
            "conscientiousness": 0.9,  # Très consciencieux
            "extraversion": 0.4,
            "agreeableness": 0.5,
            "neuroticism": 0.4
        },
        "social": {
            "openness": 0.7,
            "conscientiousness": 0.5,
            "extraversion": 0.9,       # Très extraverti
            "agreeableness": 0.9,      # Très agréable
            "neuroticism": 0.3
        },
        "balanced": {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
    }

    print("\nPersonnalités disponibles:")
    for i, (name, traits) in enumerate(personality_presets.items(), 1):
        print(f"{i}. {name.capitalize()}")
        for trait, value in traits.items():
            print(f"   - {trait}: {value*100:.0f}%")
        print()

    choice = input("Choisissez une personnalité (1-4) [défaut: balanced]: ").strip()

    preset_names = list(personality_presets.keys())
    if choice.isdigit() and 1 <= int(choice) <= 4:
        selected = preset_names[int(choice) - 1]
    else:
        selected = "balanced"

    print(f"\n✅ Personnalité sélectionnée: {selected}")
    return personality_presets[selected]


def create_character_identity() -> tuple:
    """Crée l'identité du personnage"""
    print("\n👤 Création de l'identité du personnage")
    print("=" * 60)

    name = input("Nom du personnage [défaut: AutonomousBot]: ").strip()
    if not name:
        name = "AutonomousBot"

    classes = ["Iop", "Sacrieur", "Eniripsa", "Enutrof", "Feca", "Ecaflip", "Autre"]
    print("\nClasses disponibles:")
    for i, cls in enumerate(classes, 1):
        print(f"{i}. {cls}")

    choice = input("\nChoisissez une classe (1-7) [défaut: Iop]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(classes):
        character_class = classes[int(choice) - 1]
    else:
        character_class = "Iop"

    print(f"\n✅ Personnage créé: {name} ({character_class})")
    return name, character_class


def main():
    """Fonction principale"""
    print_banner()

    # Créer les dossiers nécessaires
    Path("logs").mkdir(exist_ok=True)
    Path("data/autonomous_life").mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Démarrage du Bot Autonome Incarné...")

    # Configuration interactive
    try:
        print("\n" + "=" * 60)
        print("CONFIGURATION DU BOT")
        print("=" * 60)

        # Créer l'identité
        character_name, character_class = create_character_identity()

        # Configurer la personnalité
        personality = configure_personality()

        # === INITIALISATION DU BOT ===
        print("\n🧠 Initialisation du moteur de vie autonome...")
        print("=" * 60)

        life_engine = create_autonomous_life_engine(
            character_name=character_name,
            character_class=character_class,
            personality_preset=personality
        )

        print("\n✅ Moteur de vie initialisé!")

        # === CALIBRATION (optionnelle) ===
        print("\n🎯 Calibration de la fenêtre DOFUS")
        print("=" * 60)
        calibrate = input("Voulez-vous calibrer maintenant? (o/n) [défaut: non]: ").strip().lower()

        calibrator = None
        if calibrate == 'o':
            calibrator = DofusCalibrator()
            if not calibrator.calibrate():
                logger.warning("Calibration échouée, continuera sans calibration")

        # === BOUCLE DE VIE ===
        print("\n🌟 Le bot prend vie!")
        print("=" * 60)
        print("\nCommandes disponibles:")
        print("  - 'story': Afficher l'histoire de vie")
        print("  - 'state': Afficher l'état actuel")
        print("  - 'save': Sauvegarder l'état")
        print("  - 'quit': Arrêter le bot")
        print("\nAppuyez sur Ctrl+C pour arrêter à tout moment")
        print("=" * 60)

        # Game state
        game_state = create_game_state()

        # Statistiques
        frames_processed = 0
        start_time = time.time()
        last_story_time = time.time()

        # Boucle principale
        try:
            while True:
                frame_start = time.time()

                # === VIE DU BOT ===
                # Le bot vit un moment et décide quoi faire
                decision_result = life_engine.live_moment(game_state)

                if decision_result:
                    # Afficher la décision
                    logger.info(
                        f"🎯 Décision: {decision_result['action_type']} | "
                        f"Raison: {decision_result['reason']} | "
                        f"Confiance: {decision_result['confidence']:.2f}"
                    )

                    # TODO: Exécuter l'action via le système d'actions
                    # Pour l'instant, simulation
                    time.sleep(0.1)

                    # Enregistrer le résultat (simulation)
                    # En production, ce serait basé sur le vrai résultat
                    outcome = "success" if decision_result['confidence'] > 0.6 else "neutral"
                    reward = decision_result['confidence'] - 0.5
                    life_engine.record_outcome(decision_result['decision_obj'], outcome, reward)

                # Afficher l'histoire de vie périodiquement
                if time.time() - last_story_time > 300:  # Toutes les 5 minutes
                    print("\n" + life_engine.get_life_story())
                    last_story_time = time.time()

                frames_processed += 1

                # Dormir pour maintenir ~1 FPS (le bot "pense" lentement)
                elapsed = time.time() - frame_start
                sleep_time = max(0, 1.0 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⚠️  Arrêt demandé par l'utilisateur...")

        # === ARRÊT GRACIEUX ===
        print("\n" + "=" * 60)
        print("ARRÊT DU BOT")
        print("=" * 60)

        # Afficher l'histoire de vie finale
        print("\n📖 HISTOIRE DE VIE FINALE:")
        print(life_engine.get_life_story())

        # Sauvegarder l'état
        print("\n💾 Sauvegarde de l'état de vie...")
        life_engine.save_life_state()

        # Statistiques finales
        uptime = time.time() - start_time
        print(f"\n📊 STATISTIQUES DE SESSION:")
        print(f"   Temps de vie: {uptime/3600:.2f} heures")
        print(f"   Frames traitées: {frames_processed}")
        print(f"   Décisions prises: {life_engine.life_stats.total_decisions_made}")
        print(f"   Expériences vécues: {life_engine.life_stats.total_experiences}")
        print(f"   Sessions d'apprentissage: {life_engine.life_stats.learning_sessions}")

        print("\n✅ Bot arrêté avec succès")
        print("Au revoir! 👋")

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        print(f"\n❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
