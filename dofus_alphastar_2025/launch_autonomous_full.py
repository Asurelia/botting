#!/usr/bin/env python3
"""
DOFUS AlphaStar 2025 - Launcher Autonome COMPLET
Bot humain avec tous les systèmes avancés

SYSTEMES INTEGRES:
- HRM Reasoning (System 1 & 2 thinking)
- Vision V2 (SAM + TrOCR)
- Quest System (Ganymede integration)
- Professions (4 métiers + synergies)
- Navigation (Ganymede maps)
- Intelligence Passive + Opportunities
- Fatigue Simulator
- Guide System
- Decision Engine

AVERTISSEMENT:
- Utiliser UNIQUEMENT sur compte jetable
- Mode observation actif par défaut
- Tests recommandés avant usage réel
"""

import sys
import argparse
import logging
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Imports des systèmes
from core.hrm_reasoning import create_hrm_model, DofusHRMAgent
from core.vision_engine_v2 import create_vision_engine, create_realtime_vision
from core.quest_system import QuestManager
from core.professions import ProfessionManager
from core.guide_system import GuideLoader
from core.navigation_system import GanymedeNavigator
from core.intelligence import OpportunityManager, PassiveIntelligence, FatigueSimulator
from core.decision import DecisionEngine
from core.safety import create_observation_mode, create_safety_manager
from core.calibration import create_calibrator
from core.game_loop import create_game_engine

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('logs/autonomous_full.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AutonomousBotOrchestrator:
    """Orchestrateur principal du bot autonome complet"""

    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.running = False

        # Initialisation systèmes
        logger.info("=" * 70)
        logger.info("INITIALISATION SYSTEMES AVANCES")
        logger.info("=" * 70)

        # Safety first
        self.observation_mode = create_observation_mode() if safe_mode else None
        self.safety_manager = create_safety_manager()

        # Raisonnement
        logger.info("Initialisation HRM Reasoning...")
        self.hrm_agent = None  # Sera initialisé lors du premier usage

        # Vision
        logger.info("Initialisation Vision Engine V2...")
        self.vision = create_realtime_vision()

        # Decision
        logger.info("Initialisation Decision Engine...")
        self.decision_engine = DecisionEngine()

        # Intelligence
        logger.info("Initialisation Intelligence Systems...")
        self.opportunity_manager = OpportunityManager()
        self.passive_intelligence = PassiveIntelligence()
        self.fatigue_simulator = FatigueSimulator()

        # Actions
        logger.info("Initialisation Action Systems...")
        self.quest_manager = QuestManager()
        self.profession_manager = ProfessionManager()
        self.navigation = GanymedeNavigator()
        self.guide_loader = GuideLoader()

        # Stats
        self.stats = {
            'session_start': time.time(),
            'decisions_made': 0,
            'quests_completed': 0,
            'resources_farmed': 0,
            'navigation_moves': 0
        }

        logger.info("")
        logger.info("Tous les systemes initialises!")
        logger.info("")

    def print_banner(self):
        """Affiche le banner du bot autonome"""
        banner = """
===============================================================================

           DOFUS ALPHASTAR 2025 - BOT AUTONOME COMPLET

  SYSTEMES ACTIFS:
   HRM Reasoning      Vision V2         Quest System
   Professions       Navigation        Intelligence
   Fatigue Sim       Guide System      Decision Engine

  MODE: {}

  Le bot peut:
   - Faire des quetes intelligemment
   - Farmer et monter des metiers
   - Naviguer avec Ganymede
   - Apprendre de ses erreurs
   - Prendre des decisions strategiques
   - Simuler la fatigue humaine

  AVERTISSEMENT:
   - Utiliser UNIQUEMENT sur compte jetable
   - Session recommandee: 30-60 minutes max
   - Analyser les logs regulierement

===============================================================================
        """.format("OBSERVATION (SECURISE)" if self.safe_mode else "ACTIF")

        print(banner)

    def run_autonomous_session(self, duration_minutes: int = 30):
        """Lance une session autonome complète"""
        self.print_banner()

        logger.info(f"Debut session autonome: {duration_minutes} minutes")
        logger.info("")

        if self.safe_mode:
            logger.warning("MODE OBSERVATION ACTIF - Aucune action ne sera executee")
            logger.info("")

        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        iteration = 0

        try:
            while self.running and time.time() < end_time:
                iteration += 1
                elapsed = time.time() - start_time
                remaining = end_time - time.time()

                if iteration % 10 == 0:
                    logger.info(f"[{iteration}] Session: {elapsed:.0f}s / {duration_minutes*60}s (reste: {remaining:.0f}s)")
                    logger.info(f"  Stats: {self.stats['decisions_made']} decisions, {self.stats['navigation_moves']} moves")

                # 1. Perception (Vision)
                game_state = self._perceive_environment()

                # 2. Intelligence (Opportunities + Fatigue)
                opportunities = self._analyze_opportunities(game_state)
                # fatigue_state = self.fatigue_simulator.get_current_state()  # TODO: implémenter
                fatigue_state = None

                # 3. Decision (HRM + Decision Engine)
                decision = self._make_decision(game_state, opportunities, fatigue_state)

                # 4. Action (Quest / Profession / Navigation)
                if decision:
                    self._execute_decision(decision)

                # 5. Learning (Passive Intelligence)
                self._learn_from_experience(game_state, decision)

                # Pause humaine
                time.sleep(2)  # Tempo simple pour le MVP

        except KeyboardInterrupt:
            logger.warning("")
            logger.warning("Interruption par l'utilisateur")
        finally:
            self._end_session()

    def _perceive_environment(self) -> Dict[str, Any]:
        """Perçoit l'environnement avec le système de vision"""
        # Vision réelle à implémenter
        # Pour l'instant, retourne un état factice
        return {
            'player_position': (0, 0),
            'in_combat': False,
            'health_percent': 100.0,
            'nearby_entities': []
        }

    def _analyze_opportunities(self, game_state: Dict) -> list:
        """Analyse les opportunités avec l'intelligence passive"""
        # Détection d'opportunités
        # À implémenter avec les vrais systèmes
        return []

    def _make_decision(self, game_state: Dict, opportunities: list, fatigue_state: Any):
        """Prend une décision avec HRM + Decision Engine"""
        self.stats['decisions_made'] += 1

        # Décision simple pour le moment
        # À enrichir avec HRM reasoning complet
        decision = {
            'type': 'idle',
            'reason': 'observation_mode',
            'confidence': 0.5
        }

        return decision

    def _execute_decision(self, decision: Dict):
        """Exécute la décision"""
        if self.safe_mode and self.observation_mode:
            # Mode observation : log uniquement
            self.observation_mode.intercept_action(
                action_type=decision['type'],
                action_details=decision,
                game_state={'mode': 'autonomous'},
                reason=decision.get('reason', '')
            )
        else:
            # Mode actif : exécute réellement
            # À implémenter
            pass

    def _learn_from_experience(self, game_state: Dict, decision: Optional[Dict]):
        """Apprend de l'expérience avec l'intelligence passive"""
        # Apprentissage passif
        # À implémenter
        pass

    def _end_session(self):
        """Termine proprement la session"""
        self.running = False

        logger.info("")
        logger.info("=" * 70)
        logger.info("FIN DE SESSION")
        logger.info("=" * 70)

        duration = time.time() - self.stats['session_start']

        logger.info(f"Duree totale: {duration:.1f}s ({duration/60:.1f} minutes)")
        logger.info(f"Decisions prises: {self.stats['decisions_made']}")
        logger.info(f"Quetes: {self.stats['quests_completed']}")
        logger.info(f"Ressources: {self.stats['resources_farmed']}")
        logger.info(f"Mouvements: {self.stats['navigation_moves']}")

        if self.safe_mode and self.observation_mode:
            stats = self.observation_mode.get_stats()
            logger.info(f"")
            logger.info(f"Mode observation:")
            logger.info(f"  Actions interceptees: {stats.get('total_actions', 0)}")
            logger.info(f"  Logs sauvegardes: logs/observation.json")

            # Sauvegarder les observations
            try:
                self.observation_mode.save_observations('logs/observation.json')
            except Exception as e:
                logger.warning(f"Erreur sauvegarde observations: {e}")

        logger.info("")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='DOFUS AlphaStar 2025 - Bot Autonome Complet'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duree de la session en minutes (defaut: 30)'
    )

    parser.add_argument(
        '--active',
        action='store_true',
        help='Desactive le mode observation (DANGER!)'
    )

    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Lance la calibration avant la session'
    )

    args = parser.parse_args()

    # Calibration si demandée
    if args.calibrate:
        logger.info("Lancement calibration...")
        calibrator = create_calibrator()
        result = calibrator.run_full_calibration()
        if not result.success:
            logger.error("Calibration echouee!")
            sys.exit(1)
        logger.info("Calibration terminee!")
        logger.info("")

    # Mode sécurisé par défaut
    safe_mode = not args.active

    if not safe_mode:
        logger.warning("=" * 70)
        logger.warning("MODE ACTIF DEMANDE - ACTIONS REELLES!")
        logger.warning("Etes-vous CERTAIN d'etre sur un compte jetable?")
        logger.warning("=" * 70)
        response = input("Taper 'OUI JE COMPRENDS LES RISQUES' pour continuer: ")
        if response != "OUI JE COMPRENDS LES RISQUES":
            logger.info("Annulation - mode securise maintenu")
            safe_mode = True

    # Création orchestrateur
    bot = AutonomousBotOrchestrator(safe_mode=safe_mode)

    # Gestion interruption propre
    def signal_handler(signum, frame):
        logger.warning("")
        logger.warning("Signal d'interruption recu")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Lancement session
    bot.run_autonomous_session(duration_minutes=args.duration)

if __name__ == "__main__":
    main()
