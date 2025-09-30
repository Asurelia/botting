#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Launcher - Lanceur amélioré avec modules IA avancés
Intègre les nouveaux modules de vision, apprentissage et overlay au framework existant

Usage:
    python enhanced_ai_launcher.py --mode advisor    # Mode conseil avec overlay
    python enhanced_ai_launcher.py --mode learning   # Mode apprentissage pur
    python enhanced_ai_launcher.py --mode hybrid     # Mode hybride (recommandé)
    python enhanced_ai_launcher.py --test            # Tests système
"""

import asyncio
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Ajouter le répertoire au path Python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import du framework IA existant
from core.ai_framework import MetaOrchestrator, AIModule

# Import des nouveaux modules développés
from modules.vision.ai_vision_module import create_vision_module
from modules.learning.ai_learning_module import create_learning_module
from modules.overlay.ai_overlay_module import create_overlay_module
from modules.worldmodel.dofus_world_model import create_world_model

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(current_dir / "data" / "logs" / "enhanced_ai.log")
    ]
)

logger = logging.getLogger(__name__)

class EnhancedAIOrchestrator:
    """Orchestrateur IA amélioré avec nouveaux modules"""

    def __init__(self, mode: str = "hybrid", data_dir: Path = None):
        self.mode = mode
        self.data_dir = data_dir or (current_dir / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Orchestrateur principal
        self.orchestrator: Optional[MetaOrchestrator] = None

        # Nouveaux modules
        self.vision_module: Optional[AIModule] = None
        self.learning_module: Optional[AIModule] = None
        self.overlay_module: Optional[AIModule] = None
        self.world_model: Optional[AIModule] = None

        # État
        self.running = False
        self.session_start_time = 0.0

        logger.info(f"EnhancedAIOrchestrator initialisé en mode {mode}")

    async def initialize(self) -> bool:
        """Initialise l'orchestrateur amélioré"""
        try:
            logger.info("Initialisation Enhanced AI Orchestrator...")

            # Créer configuration améliorée
            config = self._create_enhanced_config()

            # Initialiser orchestrateur principal
            self.orchestrator = MetaOrchestrator(config)
            if not await self.orchestrator.initialize():
                logger.error("Échec initialisation MetaOrchestrator")
                return False

            # Créer et ajouter nouveaux modules
            await self._initialize_enhanced_modules(config)

            logger.info("Enhanced AI Orchestrator initialisé avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            return False

    def _create_enhanced_config(self) -> Dict[str, Any]:
        """Crée configuration améliorée"""
        config = {
            "mode": self.mode,
            "data_directory": str(self.data_dir),

            # Configuration vision
            "vision": {
                "enabled": True,
                "capture_fps": 30 if self.mode == "advisor" else 15,
                "quality": "high" if self.mode != "learning" else "medium",
                "analysis_interval": 1.0,
                "ocr_languages": ["fra", "eng"]
            },

            # Configuration apprentissage
            "learning": {
                "enabled": True,
                "observation_interval": 0.5,
                "pattern_analysis_interval": 30.0,
                "save_sessions": True,
                "max_patterns": 1000
            },

            # Configuration overlay
            "overlay": {
                "enabled": self.mode in ["advisor", "hybrid"],
                "transparency": 0.8,
                "max_elements": 10,
                "default_duration": 5.0,
                "anti_detection": True,
                "update_interval": 0.1
            },

            # Configuration World Model
            "world_model": {
                "enabled": True,
                "mapping_enabled": True,
                "update_interval": 1.0,
                "prediction_interval": 30.0,
                "spatial_memory_size": 10000,
                "temporal_analysis": True
            },

            # Configuration orchestrateur
            "orchestrator": {
                "coordination_interval": 0.1,
                "monitor_interval": 1.0,
                "performance_monitoring": True,
                "auto_optimization": True
            },

            # Modules existants
            "knowledge_graph": {"enabled": True},
            "predictive_engine": {"enabled": True},
            "decision_engine": {"enabled": True},
            "emotional_state": {"enabled": True},
            "social_intelligence": {"enabled": False},

            # Performance
            "performance": {
                "optimization_enabled": True,
                "max_cpu_percent": 80,
                "max_memory_percent": 70,
                "gc_interval": 300
            }
        }

        return config

    async def _initialize_enhanced_modules(self, config: Dict[str, Any]):
        """Initialise les nouveaux modules améliorés"""
        try:
            # Module Vision
            logger.info("Création module Vision...")
            self.vision_module = create_vision_module()
            if await self.vision_module.initialize(config):
                await self.orchestrator.add_module("enhanced_vision", self.vision_module)
                logger.info("Module Vision ajouté à l'orchestrateur")
            else:
                logger.error("Échec initialisation module Vision")

            # Module Learning
            logger.info("Création module Learning...")
            self.learning_module = create_learning_module(self.data_dir / "learning")
            if await self.learning_module.initialize(config):
                await self.orchestrator.add_module("enhanced_learning", self.learning_module)
                logger.info("Module Learning ajouté à l'orchestrateur")
            else:
                logger.error("Échec initialisation module Learning")

            # Module Overlay (seulement si activé)
            if config.get("overlay", {}).get("enabled", False):
                logger.info("Création module Overlay...")
                self.overlay_module = create_overlay_module()
                if await self.overlay_module.initialize(config):
                    await self.orchestrator.add_module("enhanced_overlay", self.overlay_module)
                    logger.info("Module Overlay ajouté à l'orchestrateur")
                else:
                    logger.error("Échec initialisation module Overlay")

            # Module World Model
            if config.get("world_model", {}).get("enabled", True):
                logger.info("Création module World Model...")
                self.world_model = create_world_model(self.data_dir / "world_model")
                if await self.world_model.initialize(config):
                    await self.orchestrator.add_module("world_model", self.world_model)
                    logger.info("Module World Model ajouté à l'orchestrateur")
                else:
                    logger.error("Échec initialisation module World Model")

        except Exception as e:
            logger.error(f"Erreur initialisation modules améliorés: {e}")

    async def start(self) -> bool:
        """Démarre l'orchestrateur amélioré"""
        try:
            if self.running:
                logger.warning("Orchestrateur déjà en cours")
                return False

            logger.info("Démarrage Enhanced AI Orchestrator...")

            # Démarrer orchestrateur principal
            if not await self.orchestrator.start():
                logger.error("Échec démarrage orchestrateur")
                return False

            self.running = True
            self.session_start_time = time.time()

            # Démarrer session d'apprentissage si module activé
            if self.learning_module:
                await self.learning_module.start_learning_session(f"session_{self.mode}_{int(time.time())}")

            logger.info(f"Enhanced AI démarré en mode {self.mode}")
            return True

        except Exception as e:
            logger.error(f"Erreur démarrage: {e}")
            return False

    async def stop(self):
        """Arrête l'orchestrateur amélioré"""
        try:
            logger.info("Arrêt Enhanced AI Orchestrator...")

            if not self.running:
                return

            # Terminer session d'apprentissage
            if self.learning_module:
                await self.learning_module.end_learning_session()

            # Arrêter orchestrateur
            if self.orchestrator:
                await self.orchestrator.stop()

            self.running = False

            # Sauvegarder statistiques de session
            await self._save_session_stats()

            logger.info("Enhanced AI arrêté")

        except Exception as e:
            logger.error(f"Erreur arrêt: {e}")

    async def _save_session_stats(self):
        """Sauvegarde statistiques de session"""
        try:
            session_duration = time.time() - self.session_start_time
            stats = await self.get_comprehensive_stats()

            session_data = {
                "mode": self.mode,
                "start_time": self.session_start_time,
                "duration_minutes": round(session_duration / 60, 2),
                "end_time": time.time(),
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }

            # Sauvegarder dans fichier
            sessions_dir = self.data_dir / "sessions"
            sessions_dir.mkdir(exist_ok=True)

            session_file = sessions_dir / f"enhanced_session_{int(self.session_start_time)}.json"

            import json
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Statistiques session sauvegardées: {session_file}")

        except Exception as e:
            logger.error(f"Erreur sauvegarde session: {e}")

    async def observe_user_action(self, action_type: str, coordinates: tuple,
                                success: bool = True, target_info: dict = None):
        """Observe une action utilisateur pour apprentissage"""
        try:
            if self.learning_module:
                await self.learning_module.observe_user_action(
                    action_type, coordinates, success, target_info
                )

        except Exception as e:
            logger.error(f"Erreur observation action: {e}")

    async def get_action_recommendations(self, context: str = None) -> List[Dict[str, Any]]:
        """Obtient recommandations d'actions"""
        try:
            if not self.learning_module:
                return []

            return await self.learning_module.get_contextual_recommendations(context)

        except Exception as e:
            logger.error(f"Erreur recommandations: {e}")
            return []

    async def display_manual_overlay(self, overlay_type: str, position: tuple,
                                   text: str, priority: int = 5):
        """Affiche overlay manuel"""
        try:
            if self.overlay_module:
                await self.overlay_module.display_manual_recommendation(
                    overlay_type, position, text, priority
                )

        except Exception as e:
            logger.error(f"Erreur overlay manuel: {e}")

    async def get_current_game_state(self) -> Dict[str, Any]:
        """Obtient état actuel du jeu"""
        try:
            game_state = {
                "timestamp": time.time(),
                "running": self.running,
                "mode": self.mode
            }

            # État vision
            if self.vision_module:
                vision_stats = self.vision_module.get_module_stats()
                game_state["vision"] = {
                    "has_screenshot": vision_stats.get("has_screenshot", False),
                    "game_state": vision_stats.get("game_state", "unknown"),
                    "fps": vision_stats.get("current_fps", 0),
                    "window_detected": vision_stats.get("window_detected", False)
                }

                # Éléments de jeu détectés
                if hasattr(self.vision_module, 'get_game_elements'):
                    game_state["game_elements"] = self.vision_module.get_game_elements()

            # État apprentissage
            if self.learning_module:
                learning_stats = self.learning_module.get_module_stats()
                game_state["learning"] = {
                    "current_situation": learning_stats.get("current_situation"),
                    "accuracy_score": learning_stats.get("accuracy_score", 0.0),
                    "patterns_learned": learning_stats.get("patterns_discovered", 0),
                    "actions_observed": learning_stats.get("actions_observed", 0)
                }

            return game_state

        except Exception as e:
            logger.error(f"Erreur état jeu: {e}")
            return {"error": str(e)}

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtient statistiques complètes"""
        try:
            stats = {
                "session": {
                    "mode": self.mode,
                    "uptime_minutes": round((time.time() - self.session_start_time) / 60, 2),
                    "running": self.running
                }
            }

            # Stats orchestrateur
            if self.orchestrator:
                orchestrator_stats = self.orchestrator.get_orchestrator_stats()
                stats["orchestrator"] = orchestrator_stats

            # Stats modules améliorés
            if self.vision_module:
                stats["vision"] = self.vision_module.get_module_stats()

            if self.learning_module:
                stats["learning"] = self.learning_module.get_module_stats()

            if self.overlay_module:
                stats["overlay"] = self.overlay_module.get_module_stats()

            return stats

        except Exception as e:
            logger.error(f"Erreur statistiques: {e}")
            return {"error": str(e)}

class EnhancedAIInterface:
    """Interface pour l'IA améliorée"""

    def __init__(self, orchestrator: EnhancedAIOrchestrator):
        self.orchestrator = orchestrator

    async def run_interactive_mode(self):
        """Mode interactif avec commandes"""
        print("\n=== Assistant IA DOFUS Ultime 2025 ===")
        print(f"Mode: {self.orchestrator.mode}")
        print("\nCommandes disponibles:")
        print("  start       - Démarrer l'assistant")
        print("  stop        - Arrêter l'assistant")
        print("  status      - Afficher statut")
        print("  stats       - Statistiques complètes")
        print("  gamestate   - État du jeu actuel")
        print("  recommend   - Obtenir recommandations")
        print("  observe     - Observer action utilisateur")
        print("  overlay     - Afficher overlay manuel")
        print("  worldmap    - Afficher cartographie découverte")
        print("  predict     - Prédictions événements")
        print("  path        - Calculer chemin optimal")
        print("  danger      - Évaluer danger zone")
        print("  help        - Aide")
        print("  quit        - Quitter")
        print("-" * 50)

        while True:
            try:
                command = input("\nai> ").strip().lower()

                if command == "quit":
                    break
                elif command == "start":
                    await self._cmd_start()
                elif command == "stop":
                    await self._cmd_stop()
                elif command == "status":
                    await self._cmd_status()
                elif command == "stats":
                    await self._cmd_stats()
                elif command == "gamestate":
                    await self._cmd_gamestate()
                elif command == "recommend":
                    await self._cmd_recommend()
                elif command == "observe":
                    await self._cmd_observe()
                elif command == "overlay":
                    await self._cmd_overlay()
                elif command == "worldmap":
                    await self._cmd_worldmap()
                elif command == "predict":
                    await self._cmd_predict()
                elif command == "path":
                    await self._cmd_path()
                elif command == "danger":
                    await self._cmd_danger()
                elif command == "help":
                    self._cmd_help()
                elif command == "":
                    continue
                else:
                    print(f"Commande inconnue: {command}")

            except KeyboardInterrupt:
                print("\nArrêt demandé...")
                break
            except Exception as e:
                print(f"Erreur: {e}")

        await self.orchestrator.stop()

    async def _cmd_start(self):
        """Commande start"""
        if await self.orchestrator.start():
            print("✓ Assistant démarré")
        else:
            print("✗ Échec démarrage")

    async def _cmd_stop(self):
        """Commande stop"""
        await self.orchestrator.stop()
        print("✓ Assistant arrêté")

    async def _cmd_status(self):
        """Commande status"""
        print(f"État: {'En cours' if self.orchestrator.running else 'Arrêté'}")
        print(f"Mode: {self.orchestrator.mode}")

        if self.orchestrator.running:
            uptime = time.time() - self.orchestrator.session_start_time
            print(f"Durée: {uptime/60:.1f} minutes")

    async def _cmd_stats(self):
        """Commande stats"""
        stats = await self.orchestrator.get_comprehensive_stats()
        print("\nStatistiques complètes:")
        self._print_stats(stats, indent=0)

    async def _cmd_gamestate(self):
        """Commande gamestate"""
        state = await self.orchestrator.get_current_game_state()
        print("\nÉtat du jeu:")
        self._print_stats(state, indent=0)

    async def _cmd_recommend(self):
        """Commande recommend"""
        context = input("Contexte (combat/quest/exploration/auto): ").strip()
        if context == "auto":
            context = None

        recommendations = await self.orchestrator.get_action_recommendations(context)

        if recommendations:
            print(f"\nRecommandations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                confidence = rec.get('confidence', 0.5)
                actions = rec.get('actions', rec.get('action_sequence', ['Action inconnue']))
                print(f"  {i}. {actions[0]} (confiance: {confidence:.1%})")
        else:
            print("Aucune recommandation disponible")

    async def _cmd_observe(self):
        """Commande observe"""
        try:
            action_type = input("Type d'action (spell/click/movement): ").strip()
            x = int(input("Position X: "))
            y = int(input("Position Y: "))
            success = input("Succès (o/N): ").strip().lower() == 'o'

            await self.orchestrator.observe_user_action(action_type, (x, y), success)
            print("✓ Action observée")

        except ValueError:
            print("✗ Format invalide")

    async def _cmd_overlay(self):
        """Commande overlay"""
        try:
            overlay_type = input("Type (spell/movement/target): ").strip()
            x = int(input("Position X: "))
            y = int(input("Position Y: "))
            text = input("Texte: ").strip()
            priority = int(input("Priorité (1-10): ") or "5")

            await self.orchestrator.display_manual_overlay(overlay_type, (x, y), text, priority)
            print("✓ Overlay affiché")

        except ValueError:
            print("✗ Format invalide")

    async def _cmd_worldmap(self):
        """Commande worldmap"""
        if not self.orchestrator.world_model:
            print("World Model non disponible")
            return

        try:
            world_state = self.orchestrator._shared_data.get('world_state', {})
            print("\nCarte du monde découverte:")
            print(f"  Carte actuelle: {world_state.get('current_map_id', 'Inconnue')}")
            print(f"  Position: {world_state.get('current_position', 'Inconnue')}")
            print(f"  Cartes découvertes: {world_state.get('maps_discovered', 0)}")
            print(f"  Entités trackées: {world_state.get('entities_tracked', 0)}")

            # Afficher entités à proximité
            if hasattr(self.orchestrator.world_model, 'get_nearby_opportunities'):
                opportunities = await self.orchestrator.world_model.get_nearby_opportunities()
                if opportunities:
                    print("\nOpportunités à proximité:")
                    for opp in opportunities[:5]:
                        print(f"  - {opp['type']}: {opp['name']} à ({opp['position']['x']}, {opp['position']['y']})")

        except Exception as e:
            print(f"Erreur worldmap: {e}")

    async def _cmd_predict(self):
        """Commande predict"""
        try:
            predictions = self.orchestrator._shared_data.get('world_predictions', [])

            if predictions:
                print(f"\nPrédictions événements ({len(predictions)}):")
                for i, pred in enumerate(predictions[:5], 1):
                    event_type = pred.get('event_type', 'Inconnu')
                    time_until = pred.get('time_until_event', 0)
                    confidence = pred.get('confidence', 0)
                    pos = pred.get('predicted_position', (0, 0))

                    print(f"  {i}. {event_type} dans {time_until:.1f}s à ({pos[0]}, {pos[1]}) - {confidence:.1%}")
            else:
                print("Aucune prédiction disponible")
                print("Le système a besoin de plus de données pour établir des patterns")

        except Exception as e:
            print(f"Erreur prédictions: {e}")

    async def _cmd_path(self):
        """Commande path"""
        if not self.orchestrator.world_model:
            print("World Model non disponible")
            return

        try:
            target_x = int(input("Position cible X: "))
            target_y = int(input("Position cible Y: "))

            if hasattr(self.orchestrator.world_model, 'get_optimal_path'):
                path = await self.orchestrator.world_model.get_optimal_path(target_x, target_y)

                if path:
                    print(f"\nChemin optimal ({len(path)} étapes):")
                    for i, step in enumerate(path):
                        print(f"  {i+1}. ({step['x']}, {step['y']})")
                else:
                    print("Aucun chemin trouvé")
            else:
                print("Calcul de chemin non disponible")

        except ValueError:
            print("Format invalide")
        except Exception as e:
            print(f"Erreur calcul chemin: {e}")

    async def _cmd_danger(self):
        """Commande danger"""
        try:
            danger_assessment = self.orchestrator._shared_data.get('current_danger', {})

            if danger_assessment:
                print("\nÉvaluation du danger:")
                print(f"  Niveau: {danger_assessment.get('danger_level', 'Inconnu')}")
                print(f"  Score: {danger_assessment.get('danger_score', 0):.2f}/1.0")
                print(f"  Recommandation: {danger_assessment.get('recommendation', 'Prudence')}")

                factors = danger_assessment.get('factors', [])
                if factors:
                    print("\nFacteurs de risque:")
                    for factor in factors:
                        print(f"  - {factor}")
            else:
                print("Évaluation du danger non disponible")

        except Exception as e:
            print(f"Erreur évaluation danger: {e}")

    def _cmd_help(self):
        """Commande help"""
        print("\nCommandes détaillées:")
        print("  start       - Démarre tous les modules (vision, apprentissage, overlay, world model)")
        print("  stop        - Arrête proprement l'assistant")
        print("  status      - État actuel et durée de session")
        print("  stats       - Statistiques complètes de tous les modules")
        print("  gamestate   - État détecté du jeu (combat, exploration, etc.)")
        print("  recommend   - Obtenir recommandations IA contextuelles")
        print("  observe     - Signaler une action pour apprentissage")
        print("  overlay     - Afficher overlay manuel à l'écran")
        print("  worldmap    - Afficher cartographie et entités découvertes")
        print("  predict     - Voir prédictions événements futurs")
        print("  path        - Calculer chemin optimal vers position")
        print("  danger      - Évaluer danger de la zone actuelle")

    def _print_stats(self, data: dict, indent: int = 0):
        """Affiche statistiques de manière formatée"""
        spaces = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                self._print_stats(value, indent + 1)
            else:
                print(f"{spaces}{key}: {value}")

async def run_tests():
    """Tests système complets"""
    print("=== Tests Enhanced AI System ===")

    try:
        # Test 1: Création orchestrateur
        print("1. Test création orchestrateur...")
        orchestrator = EnhancedAIOrchestrator("learning", Path("data/test"))
        print("   ✓ Orchestrateur créé")

        # Test 2: Initialisation
        print("2. Test initialisation...")
        if await orchestrator.initialize():
            print("   ✓ Initialisation réussie")
        else:
            print("   ✗ Échec initialisation")
            return False

        # Test 3: Démarrage
        print("3. Test démarrage...")
        if await orchestrator.start():
            print("   ✓ Démarrage réussi")
        else:
            print("   ✗ Échec démarrage")
            return False

        # Test 4: Fonctionnement pendant 5 secondes
        print("4. Test fonctionnement (5s)...")
        await asyncio.sleep(5)

        # Test 5: État du jeu
        print("5. Test état du jeu...")
        game_state = await orchestrator.get_current_game_state()
        print(f"   ✓ État obtenu: {game_state.get('vision', {}).get('game_state', 'unknown')}")

        # Test 6: Statistiques
        print("6. Test statistiques...")
        stats = await orchestrator.get_comprehensive_stats()
        print(f"   ✓ Modules actifs: {len([k for k in stats.keys() if k != 'session'])}")

        # Test 7: Arrêt
        print("7. Test arrêt...")
        await orchestrator.stop()
        print("   ✓ Arrêt réussi")

        print("\n✓ Tous les tests passés avec succès!")
        return True

    except Exception as e:
        print(f"\n✗ Erreur test: {e}")
        return False

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Enhanced AI DOFUS Launcher")
    parser.add_argument("--mode", choices=["advisor", "learning", "hybrid"],
                       default="hybrid", help="Mode de fonctionnement")
    parser.add_argument("--test", action="store_true", help="Exécuter tests")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    parser.add_argument("--data-dir", type=str, help="Répertoire de données")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Créer répertoires
    data_dir = Path(args.data_dir) if args.data_dir else (current_dir / "data")
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)

    if args.test:
        return 0 if await run_tests() else 1

    try:
        # Créer orchestrateur amélioré
        orchestrator = EnhancedAIOrchestrator(args.mode, data_dir)

        # Initialiser
        if not await orchestrator.initialize():
            logger.error("Échec initialisation")
            return 1

        # Créer interface
        interface = EnhancedAIInterface(orchestrator)

        # Démarrer en mode interactif
        await interface.run_interactive_mode()

    except KeyboardInterrupt:
        print("\nArrêt demandé par utilisateur")
    except Exception as e:
        logger.error(f"Erreur: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))