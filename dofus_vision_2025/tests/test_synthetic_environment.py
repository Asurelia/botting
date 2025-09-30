"""
Module de Test Synthétique - DOFUS Unity World Model AI
Génération d'environnements de test offline avec données simulées
"""

import os
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import cv2

# Import des modules core
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.vision_engine.unity_interface_reader import GameState
    from core.knowledge_base.knowledge_integration import DofusClass
    from core.world_model.hrm_dofus_integration import DofusGameState, DofusAction, ActionType
except ImportError:
    # Fallback pour tests
    print("⚠️ Modules core non disponibles - utilisation de mocks")

    from enum import Enum

    class DofusClass(Enum):
        IOPS = "Iop"
        CRA = "Cra"
        ENIRIPSA = "Eniripsa"
        ENUTROF = "Enutrof"

    class GameState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DofusGameState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ActionType(Enum):
        SPELL_CAST = "spell_cast"
        MOVEMENT = "movement"

    class DofusAction:
        def __init__(self, action_type, **kwargs):
            self.action_type = action_type
            self.__dict__.update(kwargs)

class SyntheticDofusScreenshotGenerator:
    """Générateur de captures d'écran DOFUS synthétiques pour tests offline"""

    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dimensions standard DOFUS Unity
        self.screen_width = 1920
        self.screen_height = 1080

        # Couleurs DOFUS typiques
        self.colors = {
            "ui_background": (45, 45, 45),
            "health_bar_green": (76, 175, 80),
            "health_bar_red": (244, 67, 54),
            "mana_bar_blue": (33, 150, 243),
            "spell_button": (121, 85, 72),
            "chat_background": (33, 33, 33),
            "gold_text": (255, 193, 7)
        }

        # Templates de sorts et interface
        self.spell_names = [
            "Pression", "Intimidation", "Compulsion", "Puissance", "Concentration",
            "Mutilation", "Uppercut", "Vitupération", "Wrath", "Sentence"
        ]

        self.monster_names = [
            "Bouftou", "Larve Bleue", "Moskito", "Tofu", "Prespic",
            "Sanglier", "Arakne", "Chafer", "Dragodinde", "Gelée"
        ]

    def generate_combat_interface(self, player_class: DofusClass = DofusClass.IOPS,
                                player_level: int = 150, in_combat: bool = True) -> np.ndarray:
        """Génère une interface de combat synthétique"""

        # Créer image base
        img = Image.new('RGB', (self.screen_width, self.screen_height), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Zone de jeu principale (grille tactique)
        if in_combat:
            self._draw_combat_grid(draw, img)

        # Interface utilisateur
        self._draw_player_stats(draw, player_class, player_level)
        self._draw_spell_bar(draw)
        self._draw_chat_window(draw, font_small)

        # Convertir en numpy array pour OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _draw_combat_grid(self, draw: ImageDraw, img: Image):
        """Dessine la grille tactique de combat"""
        grid_start_x, grid_start_y = 400, 200
        cell_size = 40

        # Grille 15x17 (approximation DOFUS)
        for row in range(17):
            for col in range(15):
                x = grid_start_x + col * cell_size
                y = grid_start_y + row * cell_size

                # Cellule de grille
                if (row + col) % 2 == 0:
                    color = (60, 60, 60)
                else:
                    color = (70, 70, 70)

                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color, outline=(80, 80, 80))

        # Positions joueurs/ennemis simulées
        player_pos = (grid_start_x + 7 * cell_size, grid_start_y + 10 * cell_size)
        enemy_pos = (grid_start_x + 9 * cell_size, grid_start_y + 6 * cell_size)

        # Joueur (bleu)
        draw.ellipse([player_pos[0] + 5, player_pos[1] + 5,
                     player_pos[0] + cell_size - 5, player_pos[1] + cell_size - 5],
                    fill=(33, 150, 243))

        # Ennemi (rouge)
        draw.ellipse([enemy_pos[0] + 5, enemy_pos[1] + 5,
                     enemy_pos[0] + cell_size - 5, enemy_pos[1] + cell_size - 5],
                    fill=(244, 67, 54))

    def _draw_player_stats(self, draw: ImageDraw, player_class: DofusClass, level: int):
        """Dessine les statistiques du joueur"""
        # Zone stats (coin supérieur gauche)
        stats_x, stats_y = 20, 20

        # Nom et niveau
        draw.text((stats_x, stats_y), f"Iop Level {level}", fill=(255, 255, 255))

        # Barres de vie et mana
        bar_width, bar_height = 200, 20

        # Barre de vie
        hp_current, hp_max = random.randint(80, 100), 100
        life_bar_y = stats_y + 30
        draw.rectangle([stats_x, life_bar_y, stats_x + bar_width, life_bar_y + bar_height],
                      fill=(40, 40, 40), outline=(80, 80, 80))
        hp_width = int((hp_current / hp_max) * bar_width)
        draw.rectangle([stats_x + 2, life_bar_y + 2, stats_x + hp_width - 2, life_bar_y + bar_height - 2],
                      fill=self.colors["health_bar_green"])
        draw.text((stats_x + 5, life_bar_y + 2), f"{hp_current}/{hp_max} HP", fill=(255, 255, 255))

        # Barre de mana
        mp_current, mp_max = random.randint(60, 100), 100
        mana_bar_y = stats_y + 60
        draw.rectangle([stats_x, mana_bar_y, stats_x + bar_width, mana_bar_y + bar_height],
                      fill=(40, 40, 40), outline=(80, 80, 80))
        mp_width = int((mp_current / mp_max) * bar_width)
        draw.rectangle([stats_x + 2, mana_bar_y + 2, stats_x + mp_width - 2, mana_bar_y + bar_height - 2],
                      fill=self.colors["mana_bar_blue"])
        draw.text((stats_x + 5, mana_bar_y + 2), f"{mp_current}/{mp_max} MP", fill=(255, 255, 255))

        # AP et MP de combat
        ap_text = f"AP: {random.randint(3, 6)}"
        mp_text = f"MP: {random.randint(2, 4)}"
        draw.text((stats_x, stats_y + 100), ap_text, fill=(255, 193, 7))
        draw.text((stats_x + 80, stats_y + 100), mp_text, fill=(33, 150, 243))

    def _draw_spell_bar(self, draw: ImageDraw):
        """Dessine la barre de sorts"""
        spell_bar_y = self.screen_height - 100
        spell_size = 60
        spell_spacing = 70
        start_x = (self.screen_width - len(self.spell_names) * spell_spacing) // 2

        for i, spell in enumerate(self.spell_names[:8]):  # Limiter à 8 sorts
            x = start_x + i * spell_spacing

            # Bouton de sort
            draw.rectangle([x, spell_bar_y, x + spell_size, spell_bar_y + spell_size],
                          fill=self.colors["spell_button"], outline=(100, 100, 100))

            # Raccourci clavier
            draw.text((x + 5, spell_bar_y + 5), str(i + 1), fill=(255, 255, 255))

            # Nom du sort (raccourci)
            spell_short = spell[:3]
            draw.text((x + 10, spell_bar_y + 35), spell_short, fill=(200, 200, 200))

    def _draw_chat_window(self, draw: ImageDraw, font):
        """Dessine la fenêtre de chat"""
        chat_x, chat_y = 20, self.screen_height - 250
        chat_width, chat_height = 600, 140

        # Fond du chat
        draw.rectangle([chat_x, chat_y, chat_x + chat_width, chat_y + chat_height],
                      fill=self.colors["chat_background"], outline=(80, 80, 80))

        # Messages de combat simulés
        messages = [
            "[Combat] Vous lancez Pression sur Bouftou",
            "[Combat] Bouftou perd 45 points de vie",
            "[Info] C'est votre tour !",
            "[System] Combat en cours..."
        ]

        for i, msg in enumerate(messages):
            y_pos = chat_y + 10 + i * 25
            if i < 2:  # Messages de combat en vert
                color = (76, 175, 80)
            elif i == 2:  # Message d'info en jaune
                color = (255, 193, 7)
            else:  # Messages système en blanc
                color = (255, 255, 255)

            draw.text((chat_x + 10, y_pos), msg, fill=color, font=font)

    def generate_dataset(self, num_screenshots: int = 50) -> List[Dict[str, Any]]:
        """Génère un dataset complet de captures d'écran synthétiques"""
        dataset = []

        print(f"Génération de {num_screenshots} captures d'écran synthétiques...")

        for i in range(num_screenshots):
            # Paramètres aléatoires
            player_class = random.choice(list(DofusClass))
            player_level = random.randint(50, 200)
            in_combat = random.choice([True, False])

            # Générer screenshot
            screenshot = self.generate_combat_interface(player_class, player_level, in_combat)

            # Sauvegarder image
            filename = f"synthetic_screenshot_{i:03d}.png"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), screenshot)

            # Métadonnées
            metadata = {
                "filename": filename,
                "filepath": str(filepath),
                "player_class": player_class.value,
                "player_level": player_level,
                "in_combat": in_combat,
                "timestamp": time.time(),
                "screen_resolution": f"{self.screen_width}x{self.screen_height}",
                "synthetic": True
            }

            dataset.append(metadata)

            if (i + 1) % 10 == 0:
                print(f"  Généré {i + 1}/{num_screenshots} captures...")

        # Sauvegarder métadonnées
        dataset_file = self.output_dir / "synthetic_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Dataset synthétique sauvegardé : {dataset_file}")
        return dataset

class MockGameStateGenerator:
    """Générateur d'états de jeu mockés pour tests"""

    def __init__(self):
        self.monster_types = [
            "Bouftou", "Larve Bleue", "Moskito", "Tofu", "Prespic",
            "Sanglier", "Arakne", "Chafer", "Dragodinde", "Gelée"
        ]

        self.spell_list = [
            "Pression", "Intimidation", "Compulsion", "Puissance", "Concentration",
            "Mutilation", "Uppercut", "Vitupération", "Wrath", "Sentence"
        ]

    def generate_combat_state(self) -> DofusGameState:
        """Génère un état de combat mocké"""
        return DofusGameState(
            player_class=random.choice(list(DofusClass)),
            player_level=random.randint(50, 200),
            current_server="Julith",
            current_map_id=random.randint(10000, 99999),
            in_combat=True,
            available_ap=random.randint(3, 6),
            available_mp=random.randint(2, 4),
            current_health=random.randint(70, 100),
            max_health=100,
            player_position=(random.randint(0, 14), random.randint(0, 16)),
            enemies_positions=[(random.randint(0, 14), random.randint(0, 16)) for _ in range(random.randint(1, 3))],
            allies_positions=[],
            interface_elements_visible=["spells", "stats", "chat"],
            spell_cooldowns={spell: random.randint(0, 3) for spell in random.sample(self.spell_list, 3)},
            inventory_items={f"Item_{i}": random.randint(1, 10) for i in range(5)},
            current_kamas=random.randint(10000, 100000),
            market_opportunities=[],
            timestamp=time.time(),
            screenshot_path=None
        )

    def generate_exploration_state(self) -> DofusGameState:
        """Génère un état d'exploration mocké"""
        return DofusGameState(
            player_class=random.choice(list(DofusClass)),
            player_level=random.randint(50, 200),
            current_server="Julith",
            current_map_id=random.randint(10000, 99999),
            in_combat=False,
            available_ap=6,
            available_mp=3,
            current_health=100,
            max_health=100,
            player_position=(random.randint(0, 14), random.randint(0, 16)),
            enemies_positions=[],
            allies_positions=[],
            interface_elements_visible=["map", "inventory", "chat"],
            spell_cooldowns={},
            inventory_items={f"Resource_{i}": random.randint(1, 50) for i in range(10)},
            current_kamas=random.randint(10000, 100000),
            market_opportunities=[
                {"item": f"Item_{i}", "price": random.randint(100, 1000)}
                for i in range(random.randint(0, 5))
            ],
            timestamp=time.time(),
            screenshot_path=None
        )

    def generate_state_sequence(self, length: int = 10) -> List[DofusGameState]:
        """Génère une séquence d'états cohérente"""
        sequence = []

        # Commencer par un état d'exploration
        current_state = self.generate_exploration_state()
        sequence.append(current_state)

        for i in range(length - 1):
            # 30% de chance d'entrer en combat
            if not current_state.in_combat and random.random() < 0.3:
                current_state = self.generate_combat_state()
            # 20% de chance de sortir du combat
            elif current_state.in_combat and random.random() < 0.2:
                current_state = self.generate_exploration_state()
            # Sinon, maintenir le type d'état actuel avec variations
            elif current_state.in_combat:
                current_state = self.generate_combat_state()
            else:
                current_state = self.generate_exploration_state()

            sequence.append(current_state)

        return sequence

class SyntheticTestEnvironment:
    """Environnement de test synthétique complet"""

    def __init__(self, data_dir: str = "synthetic_test_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.screenshot_generator = SyntheticDofusScreenshotGenerator(
            str(self.data_dir / "screenshots")
        )
        self.state_generator = MockGameStateGenerator()

        # Métriques de test
        self.test_results = {
            "screenshots_generated": 0,
            "states_generated": 0,
            "test_sessions": 0,
            "performance_metrics": {}
        }

    def setup_complete_test_environment(self, num_screenshots: int = 100,
                                       num_state_sequences: int = 20) -> Dict[str, Any]:
        """Configure un environnement de test complet"""

        print("[CONFIG] Configuration de l'environnement de test synthetique...")

        # Générer screenshots synthétiques
        print("\n[SCREENSHOTS] Generation des captures d'ecran...")
        screenshot_dataset = self.screenshot_generator.generate_dataset(num_screenshots)
        self.test_results["screenshots_generated"] = len(screenshot_dataset)

        # Générer séquences d'états
        print("\n[STATES] Generation des sequences d'etats...")
        state_sequences = []
        for i in range(num_state_sequences):
            sequence = self.state_generator.generate_state_sequence(random.randint(5, 15))
            state_sequences.append({
                "sequence_id": i,
                "length": len(sequence),
                "states": sequence
            })

            if (i + 1) % 5 == 0:
                print(f"  Generate {i + 1}/{num_state_sequences} sequences...")

        self.test_results["states_generated"] = sum(len(seq["states"]) for seq in state_sequences)

        # Sauvegarder séquences d'états
        sequences_file = self.data_dir / "state_sequences.json"
        with open(sequences_file, 'w', encoding='utf-8') as f:
            # Sérialiser les DofusGameState
            serializable_sequences = []
            for seq in state_sequences:
                serializable_states = []
                for state in seq["states"]:
                    try:
                        state_dict = {
                            "player_class": state.player_class.value if hasattr(state.player_class, 'value') else str(state.player_class),
                            "player_level": state.player_level,
                            "current_server": state.current_server,
                            "current_map_id": state.current_map_id,
                            "in_combat": state.in_combat,
                            "available_ap": state.available_ap,
                            "available_mp": state.available_mp,
                            "current_health": state.current_health,
                            "max_health": state.max_health,
                            "player_position": state.player_position,
                            "enemies_positions": state.enemies_positions,
                            "allies_positions": state.allies_positions,
                            "interface_elements_visible": state.interface_elements_visible,
                            "spell_cooldowns": state.spell_cooldowns,
                            "inventory_items": state.inventory_items,
                            "current_kamas": state.current_kamas,
                            "market_opportunities": state.market_opportunities,
                            "timestamp": state.timestamp
                        }
                    except AttributeError:
                        # Mock state simple
                        state_dict = {
                            "player_class": "Iop",
                            "player_level": 150,
                            "in_combat": True,
                            "timestamp": time.time()
                        }
                    serializable_states.append(state_dict)

                serializable_sequences.append({
                    "sequence_id": seq["sequence_id"],
                    "length": seq["length"],
                    "states": serializable_states
                })

            json.dump(serializable_sequences, f, indent=2, ensure_ascii=False)

        print(f"Sequences d'etats sauvegardees : {sequences_file}")

        # Sauvegarder configuration de test
        config = {
            "environment_type": "synthetic",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "screenshots": {
                "count": len(screenshot_dataset),
                "directory": str(self.screenshot_generator.output_dir)
            },
            "state_sequences": {
                "count": len(state_sequences),
                "total_states": self.test_results["states_generated"],
                "file": str(sequences_file)
            },
            "test_results": self.test_results
        }

        config_file = self.data_dir / "test_environment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Environnement de test configure !")
        print(f"[STATS] Statistiques :")
        print(f"  - Screenshots generes : {self.test_results['screenshots_generated']}")
        print(f"  - Etats generes : {self.test_results['states_generated']}")
        print(f"  - Sequences creees : {len(state_sequences)}")
        print(f"  - Configuration : {config_file}")

        return config

def run_synthetic_test_suite():
    """Lance la suite de tests synthétiques complète"""

    print("[TEST] DOFUS Unity AI - Suite de Tests Synthetiques")
    print("=" * 60)

    # Créer environnement
    test_env = SyntheticTestEnvironment("tests/synthetic_test_data")

    # Configuration
    config = test_env.setup_complete_test_environment(
        num_screenshots=50,  # Réduire pour les tests
        num_state_sequences=10
    )

    print("\n[READY] Environnement de test pret pour utilisation !")
    print("\nPour utiliser cet environnement :")
    print("1. Charger les screenshots depuis : tests/synthetic_test_data/screenshots/")
    print("2. Charger les etats depuis : tests/synthetic_test_data/state_sequences.json")
    print("3. Lancer les tests avec : python tests/test_complete_system.py")

    return config

if __name__ == "__main__":
    run_synthetic_test_suite()