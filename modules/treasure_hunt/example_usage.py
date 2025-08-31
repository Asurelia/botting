"""
Exemple d'utilisation du système d'automatisation des chasses aux trésors DOFUS
Démontre toutes les fonctionnalités principales du module
"""

import logging
import time
import numpy as np
from pathlib import Path

# Import du module de chasse aux trésors
from . import (
    create_treasure_hunt_system,
    create_treasure_hunt_gui,
    TreasureHuntType,
    HintType,
    HintDifficulty,
    DEFAULT_CONFIG
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('treasure_hunt_example.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MockDofusInterface:
    """
    Interface simulée pour DOFUS - remplace les vraies interactions
    En production, ceci serait remplacé par de vraies fonctions d'interaction
    """
    
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.current_position = (0, 0)
        self.simulation_mode = True
        
        logger.info("Interface DOFUS simulée initialisée")
    
    def click_at_position(self, x: int, y: int):
        """Simule un clic à une position donnée"""
        logger.debug(f"Clic simulé à la position ({x}, {y})")
        
        # En production, utiliser pyautogui, win32api, etc.
        # pyautogui.click(x, y)
        
        # Simulation d'un déplacement de position
        if 100 <= x <= 200 and 100 <= y <= 200:  # Zone de carte simulée
            self.current_position = (x - 150, y - 150)
            logger.debug(f"Position simulée mise à jour: {self.current_position}")
    
    def capture_screen(self) -> np.ndarray:
        """Simule la capture d'écran"""
        # En production, utiliser PIL, win32gui, etc.
        # screenshot = pyautogui.screenshot()
        # return np.array(screenshot)
        
        # Simulation avec une image noire
        simulated_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Ajouter quelques éléments simulés
        # Zone de mini-carte
        simulated_screen[50:250, 50:250] = [40, 40, 60]
        
        # Zone d'interface de chasse (simulation)
        simulated_screen[300:400, 400:800] = [60, 40, 40]
        
        return simulated_screen


def example_basic_usage():
    """Exemple d'utilisation basique du système"""
    logger.info("=== Exemple d'utilisation basique ===")
    
    # 1. Créer l'interface simulée
    dofus_interface = MockDofusInterface()
    
    # 2. Créer le système d'automatisation
    treasure_system = create_treasure_hunt_system(
        click_handler=dofus_interface.click_at_position,
        screen_capture_handler=dofus_interface.capture_screen
    )
    
    # 3. Démarrer une chasse aux trésors
    success = treasure_system.start_treasure_hunt(
        hunt_type=TreasureHuntType.CLASSIC,
        character_name="ExampleCharacter"
    )
    
    if success:
        logger.info("Chasse aux trésors démarrée avec succès!")
        
        # 4. Surveiller l'état pendant quelques secondes
        for i in range(10):
            status = treasure_system.get_current_status()
            logger.info(f"État actuel: {status['state']} - Étape: {status.get('current_session', {}).get('current_step', 'N/A')}")
            time.sleep(1)
    else:
        logger.error("Impossible de démarrer la chasse aux trésors")
    
    # 5. Arrêter le système
    treasure_system.stop_automation()
    treasure_system.close()
    
    logger.info("Exemple basique terminé")


def example_with_gui():
    """Exemple d'utilisation avec l'interface graphique"""
    logger.info("=== Exemple avec interface graphique ===")
    
    try:
        # 1. Créer l'interface simulée
        dofus_interface = MockDofusInterface()
        
        # 2. Créer le système d'automatisation
        treasure_system = create_treasure_hunt_system(
            click_handler=dofus_interface.click_at_position,
            screen_capture_handler=dofus_interface.capture_screen
        )
        
        # 3. Créer et lancer l'interface graphique
        gui = create_treasure_hunt_gui(treasure_system)
        
        logger.info("Lancement de l'interface graphique...")
        logger.info("Fermez la fenêtre pour terminer l'exemple")
        
        # L'interface GUI prend le contrôle jusqu'à fermeture
        gui.run()
        
    except Exception as e:
        logger.error(f"Erreur dans l'exemple GUI: {e}")
    
    logger.info("Exemple GUI terminé")


def example_database_management():
    """Exemple de gestion de la base de données d'indices"""
    logger.info("=== Exemple de gestion de base de données ===")
    
    from .hint_database import HintDatabase, HintData
    from datetime import datetime
    
    # 1. Créer/ouvrir la base de données
    db = HintDatabase("example_hints.db")
    
    # 2. Ajouter quelques indices d'exemple
    example_hints = [
        {
            "text": "Cherchez près de la statue du Dofus Emeraude",
            "type": HintType.ELEMENT,
            "difficulty": HintDifficulty.HARD,
            "area": "Cimetière des Torturés",
            "description": "Statue distinctive dans le cimetière"
        },
        {
            "text": "Rendez-vous à l'auberge de Bonta",
            "type": HintType.BUILDING,
            "difficulty": HintDifficulty.EASY,
            "area": "Bonta",
            "description": "Auberge principale de la ville"
        },
        {
            "text": "Dirigez-vous vers l'ouest de la carte",
            "type": HintType.DIRECTION,
            "difficulty": HintDifficulty.EASY,
            "area": "Toutes zones",
            "description": "Direction ouest générique"
        }
    ]
    
    for hint_info in example_hints:
        hint = HintData(
            id=f"example_{len(example_hints)}",
            text=hint_info["text"],
            hint_type=hint_info["type"],
            difficulty=hint_info["difficulty"],
            map_coordinates=None,
            area_name=hint_info["area"],
            sub_area_name="",
            cell_id=None,
            description=hint_info["description"],
            keywords=hint_info["text"].lower().split(),
            image_hash=None,
            image_data=None,
            success_rate=0.0,
            usage_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            validated=True,
            community_rating=0.0
        )
        
        success = db.add_hint(hint)
        if success:
            logger.info(f"Indice ajouté: {hint.text[:50]}...")
    
    # 3. Tester la recherche
    search_results = db.find_hint_by_text("auberge")
    logger.info(f"Résultats de recherche pour 'auberge': {len(search_results)} trouvés")
    
    for hint in search_results:
        logger.info(f"  - {hint.text}")
    
    # 4. Afficher les statistiques
    stats = db.get_database_stats()
    logger.info(f"Statistiques de la base:")
    logger.info(f"  Total d'indices: {stats['total_hints']}")
    logger.info(f"  Répartition par type: {stats['by_type']}")
    
    # 5. Exporter la base
    export_path = "example_hints_export.json"
    if db.export_hints(export_path):
        logger.info(f"Base exportée vers: {export_path}")
    
    # 6. Fermer la base
    db.close()
    
    logger.info("Exemple de gestion de base terminé")


def example_solver_testing():
    """Exemple de test du solveur d'indices"""
    logger.info("=== Exemple de test du solveur ===")
    
    from .hint_database import HintDatabase
    from .treasure_solver import TreasureSolver
    
    # 1. Créer la base et le solveur
    db = HintDatabase("example_hints.db")
    solver = TreasureSolver(db)
    
    # 2. Tester avec différents indices
    test_hints = [
        "Allez vers le nord de la taverne",
        "Cherchez près du grand arbre",
        "Dirigez-vous vers l'atelier du forgeron",
        "Localisez la pierre mystérieuse",
        "Rendez-vous au temple de la ville"
    ]
    
    for hint in test_hints:
        logger.info(f"\n--- Test de résolution: {hint} ---")
        
        # Résoudre l'indice
        solutions = solver.solve_hint(hint)
        
        if solutions:
            logger.info(f"Nombre de solutions trouvées: {len(solutions)}")
            
            # Afficher les 3 meilleures solutions
            for i, solution in enumerate(solutions[:3]):
                logger.info(f"Solution {i+1}:")
                logger.info(f"  Type: {solution.solution_type.value}")
                logger.info(f"  Confiance: {solution.confidence:.2f}")
                logger.info(f"  Raisonnement: {solution.reasoning}")
                logger.info(f"  Temps estimé: {solution.estimated_time:.1f}s")
                
                if solution.estimated_coordinates:
                    logger.info(f"  Coordonnées: {solution.estimated_coordinates}")
        else:
            logger.warning("Aucune solution trouvée")
    
    # 3. Fermer
    db.close()
    
    logger.info("Exemple de test du solveur terminé")


def example_configuration():
    """Exemple de configuration avancée"""
    logger.info("=== Exemple de configuration avancée ===")
    
    # 1. Configuration personnalisée
    custom_config = DEFAULT_CONFIG.copy()
    custom_config.update({
        'max_attempts_per_step': 5,
        'step_timeout': 600,  # 10 minutes
        'auto_fight': False,  # Pas de combat automatique
        'debug_mode': True,
        'screenshots_path': Path('custom_screenshots/')
    })
    
    logger.info("Configuration personnalisée:")
    for key, value in custom_config.items():
        logger.info(f"  {key}: {value}")
    
    # 2. Créer le système avec la configuration
    dofus_interface = MockDofusInterface()
    treasure_system = create_treasure_hunt_system(
        click_handler=dofus_interface.click_at_position,
        screen_capture_handler=dofus_interface.capture_screen
    )
    
    # 3. Appliquer la configuration
    treasure_system.config.update(custom_config)
    
    # 4. Enregistrer des callbacks personnalisés
    def on_hunt_started(session):
        logger.info(f"🎯 Callback: Chasse démarrée - {session.session_id}")
    
    def on_step_completed(step):
        logger.info(f"✅ Callback: Étape {step.step_number} complétée")
    
    def on_error(error_msg):
        logger.error(f"❌ Callback: Erreur détectée - {error_msg}")
    
    treasure_system.register_callback('on_hunt_started', on_hunt_started)
    treasure_system.register_callback('on_step_completed', on_step_completed)
    treasure_system.register_callback('on_error', on_error)
    
    logger.info("Callbacks enregistrés")
    
    # 5. Test avec la configuration personnalisée
    logger.info("Test avec configuration personnalisée...")
    # (Le test serait similaire aux autres exemples)
    
    treasure_system.close()
    
    logger.info("Exemple de configuration terminé")


def main():
    """Point d'entrée principal des exemples"""
    logger.info("🏴‍☠️ Démarrage des exemples d'utilisation du système de chasse aux trésors DOFUS")
    logger.info(f"Version du module: 1.0.0")
    logger.info("-" * 60)
    
    try:
        # Lancer les différents exemples
        print("\n1. Exemple d'utilisation basique")
        print("2. Exemple avec interface graphique") 
        print("3. Exemple de gestion de base de données")
        print("4. Exemple de test du solveur")
        print("5. Exemple de configuration avancée")
        print("6. Tous les exemples (sauf GUI)")
        
        choice = input("\nChoisissez un exemple (1-6): ").strip()
        
        if choice == "1":
            example_basic_usage()
        elif choice == "2":
            example_with_gui()
        elif choice == "3":
            example_database_management()
        elif choice == "4":
            example_solver_testing()
        elif choice == "5":
            example_configuration()
        elif choice == "6":
            example_database_management()
            example_solver_testing()
            example_configuration()
            example_basic_usage()
        else:
            logger.info("Choix invalide, lancement de l'interface graphique par défaut")
            example_with_gui()
            
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur dans les exemples: {e}")
        raise
    
    logger.info("-" * 60)
    logger.info("🎉 Exemples d'utilisation terminés")


if __name__ == "__main__":
    main()