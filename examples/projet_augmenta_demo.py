"""
Démonstration du Projet Augmenta
Montre l'utilisation des modules d'intelligence passive, opportunités, fatigue, combos et analyse post-combat
"""

import time
import logging
from typing import Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Imports des modules (avec fallback pour les tests)
try:
    from modules.intelligence.passive_intelligence import PassiveIntelligence
    from modules.intelligence.opportunity_manager import OpportunityManager
    from modules.intelligence.fatigue_simulation import FatigueSimulation
    from modules.combat.combo_library import ComboLibrary
    from modules.combat.post_combat_analysis import PostCombatAnalysis
    from core.hrm_intelligence.amd_gpu_optimizer import AMDGPUOptimizer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Certains modules ne sont pas disponibles: {e}")
    MODULES_AVAILABLE = False


class AugmentaDemo:
    """Démonstration du Projet Augmenta"""

    def __init__(self):
        self.modules = {}
        self.demo_running = False

    def initialize_modules(self) -> bool:
        """Initialise tous les modules du Projet Augmenta"""
        if not MODULES_AVAILABLE:
            print("❌ Modules non disponibles - démonstration limitée")
            return False

        print("🚀 Initialisation du Projet Augmenta...")

        # Configuration commune
        base_config = {
            "scan_interval": 30.0,
            "enable_learning": True,
            "auto_generate_combos": True,
            "enable_effects": True
        }

        # Initialisation des modules
        modules_to_init = [
            ("passive_intelligence", PassiveIntelligence, base_config),
            ("opportunity_manager", OpportunityManager, base_config),
            ("fatigue_simulation", FatigueSimulation, base_config),
            ("combo_library", ComboLibrary, base_config),
            ("post_combat_analysis", PostCombatAnalysis, base_config),
            ("gpu_optimizer", AMDGPUOptimizer, {})
        ]

        for module_name, module_class, config in modules_to_init:
            try:
                module = module_class()
                if module.initialize(config):
                    self.modules[module_name] = module
                    print(f"✅ {module_name} initialisé")
                else:
                    print(f"❌ Échec initialisation {module_name}")
            except Exception as e:
                print(f"❌ Erreur {module_name}: {e}")

        print(f"📊 {len(self.modules)} modules initialisés sur {len(modules_to_init)}")
        return len(self.modules) > 0

    def run_demo(self):
        """Exécute la démonstration"""
        if not self.modules:
            print("❌ Aucun module initialisé")
            return

        print("\n🎮 Démonstration du Projet Augmenta")
        print("=" * 50)

        self.demo_running = True

        try:
            # Phase 1: Intelligence Passive
            self._demo_passive_intelligence()

            # Phase 2: Gestion des Opportunités
            self._demo_opportunity_management()

            # Phase 3: Simulation de Fatigue
            self._demo_fatigue_simulation()

            # Phase 4: Bibliothèque de Combos
            self._demo_combo_library()

            # Phase 5: Analyse Post-Combat
            self._demo_post_combat_analysis()

            # Phase 6: Optimisation GPU
            self._demo_gpu_optimization()

            # Rapport final
            self._generate_final_report()

        except KeyboardInterrupt:
            print("\n⏹️ Démonstration interrompue par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur durant la démonstration: {e}")
        finally:
            self.cleanup()

    def _demo_passive_intelligence(self):
        """Démontre l'intelligence passive"""
        print("\n🔍 Phase 1: Intelligence Passive")
        print("-" * 30)

        if "passive_intelligence" not in self.modules:
            print("❌ Module d'intelligence passive non disponible")
            return

        module = self.modules["passive_intelligence"]

        # Simulation d'observations
        print("📊 Collecte d'observations...")

        # Création d'observations de test
        test_observations = [
            {"type": "enemy_spawn", "location": (100, 150), "confidence": 0.8},
            {"type": "resource_availability", "location": (120, 140), "confidence": 0.9},
            {"type": "player_behavior", "location": (110, 160), "confidence": 0.7}
        ]

        for obs in test_observations:
            # Simulation d'ajout d'observation
            print(f"  📍 Observation: {obs['type']} à {obs['location']} (confiance: {obs['confidence']})")

        # Mise à jour du module
        mock_game_state = self._create_mock_game_state()
        result = module.update(mock_game_state)

        if result:
            data = result.get("passive_data", {})
            print(f"  📈 Données collectées: {data['observations_count']} observations")

    def _demo_opportunity_management(self):
        """Démontre la gestion des opportunités"""
        print("\n🎯 Phase 2: Gestion des Opportunités")
        print("-" * 30)

        if "opportunity_manager" not in self.modules:
            print("❌ Module de gestion d'opportunités non disponible")
            return

        module = self.modules["opportunity_manager"]

        # Simulation de détection d'opportunités
        print("🔍 Recherche d'opportunités...")

        mock_game_state = self._create_mock_game_state()
        opportunities = module.detect_opportunities(mock_game_state)

        print(f"  🎯 {len(opportunities)} opportunités détectées")

        for opp in opportunities[:3]:  # Affichage des 3 premières
            print(f"    💎 {opp.opportunity_type}: valeur {opp.value_estimate:.1f}, risque {opp.risk_level:.2f}")

        # Mise à jour du module
        result = module.update(mock_game_state)
        if result:
            opp_data = result.get("opportunities", {})
            print(f"  📊 Total opportunités: {opp_data['total']}")

    def _demo_fatigue_simulation(self):
        """Démontre la simulation de fatigue"""
        print("\n😴 Phase 3: Simulation de Fatigue")
        print("-" * 30)

        if "fatigue_simulation" not in self.modules:
            print("❌ Module de simulation de fatigue non disponible")
            return

        module = self.modules["fatigue_simulation"]

        # Simulation de session longue
        print("⏰ Simulation de session de 2 heures...")

        for hour in range(2):
            # Mise à jour de la fatigue
            mock_game_state = self._create_mock_game_state()
            result = module.update(mock_game_state)

            if result and "fatigue_state" in result:
                fatigue_level = result["fatigue_state"]["level"]
                print(f"  🕐 Heure {hour+1}: Fatigue = {fatigue_level:.2f}")

                if fatigue_level > 0.6:
                    print("    ⚠️ Performances dégradées détectées")
                    break

        # Test d'application d'effets
        base_accuracy = 0.95
        adjusted_accuracy, adjusted_speed = module.apply_fatigue_effects(base_accuracy, 1.0)
        print(f"  🎯 Précision ajustée: {base_accuracy:.2f} → {adjusted_accuracy:.2f}")

    def _demo_combo_library(self):
        """Démontre la bibliothèque de combos"""
        print("\n⚔️ Phase 4: Bibliothèque de Combos")
        print("-" * 30)

        if "combo_library" not in self.modules:
            print("❌ Module de bibliothèque de combos non disponible")
            return

        module = self.modules["combo_library"]

        # Génération de combos
        print("🔧 Génération de combos...")

        mock_game_state = self._create_mock_game_state()
        combo = module.generate_combo_for_situation(mock_game_state)

        if combo:
            print(f"  ✨ Combo généré: {combo.name}")
            print(f"    📜 Sorts: {', '.join(combo.spell_sequence)}")
            print(f"    💪 Effets: {combo.effects}")
        else:
            print("  ⚠️ Aucun combo généré")

        # Mise à jour du module
        result = module.update(mock_game_state)
        if result:
            combo_data = result.get("combo_data", {})
            print(f"  📚 Bibliothèque: {combo_data['library_size']} combos")

    def _demo_post_combat_analysis(self):
        """Démontre l'analyse post-combat"""
        print("\n📊 Phase 5: Analyse Post-Combat")
        print("-" * 30)

        if "post_combat_analysis" not in self.modules:
            print("❌ Module d'analyse post-combat non disponible")
            return

        module = self.modules["post_combat_analysis"]

        # Simulation d'un combat
        print("⚔️ Simulation d'un combat...")

        # États avant et après
        before_state = self._create_mock_game_state()
        after_state = self._create_mock_game_state()
        after_state.character._hp_percentage = 60  # HP réduits après combat

        # Événements de combat
        combat_events = [
            {"type": "spell_cast", "damage_dealt": 150, "success": True},
            {"type": "spell_cast", "damage_dealt": 200, "success": True},
            {"type": "damage_received", "damage": 80}
        ]

        # Analyse
        report = module.analyzer.analyze_combat(before_state, after_state, combat_events)

        print(f"  📈 Combat analysé: Victoire = {report.victory}")
        print(f"    💥 Dégâts infligés: {report.damage_dealt}")
        print(f"    🛡️ Dégâts reçus: {report.damage_received}")
        print(f"    ⭐ Score de performance: {report.performance_score:.1f}")

        if report.recommendations:
            print("  💡 Recommandations:")
            for rec in report.recommendations:
                print(f"    • {rec}")

    def _demo_gpu_optimization(self):
        """Démontre l'optimisation GPU"""
        print("\n🚀 Phase 6: Optimisation GPU AMD")
        print("-" * 30)

        if "gpu_optimizer" not in self.modules:
            print("❌ Module d'optimisation GPU non disponible")
            return

        optimizer = self.modules["gpu_optimizer"]

        # Initialisation
        success = optimizer.initialize()
        print(f"  🔧 Initialisation: {'✅ Réussie' if success else '❌ Échouée'}")

        # Rapport d'optimisation
        report = optimizer.get_optimization_report()
        print(f"  📊 GPU détecté: {report['gpu_capabilities']['model']}")
        print(f"    💾 Mémoire: {report['gpu_capabilities']['memory_gb']}GB")
        print(f"    🔄 ROCm: {'✅ Supporté' if report['gpu_capabilities']['rocm_supported'] else '❌ Non supporté'}")
        print(f"    🎯 DirectML: {'✅ Supporté' if report['gpu_capabilities']['directml_supported'] else '❌ Non supporté'}")

    def _generate_final_report(self):
        """Génère un rapport final"""
        print("\n📋 Rapport Final du Projet Augmenta")
        print("=" * 50)

        total_modules = len(self.modules)
        print(f"✅ Modules initialisés: {total_modules}")

        # Résumé par module
        for name, module in self.modules.items():
            state = module.get_state()
            status = "🟢 Actif" if state.get("status") == "active" else "🟡 Inactif"
            print(f"  • {name}: {status}")

        print("\n🎉 Démonstration terminée avec succès!")
        print("Le Projet Augmenta est maintenant opérationnel avec:")
        print("  🔍 Intelligence passive pour l'observation")
        print("  🎯 Gestion d'opportunités pour l'optimisation")
        print("  😴 Simulation de fatigue pour le réalisme")
        print("  ⚔️ Bibliothèque de combos pour les combats")
        print("  📊 Analyse post-combat pour l'apprentissage")
        print("  🚀 Optimisation GPU AMD pour les performances")

    def _create_mock_game_state(self) -> Any:
        """Crée un état de jeu mock pour les tests"""
        mock_state = type('MockGameState', (), {})()

        # Character mock
        mock_state.character = type('MockCharacter', (), {})()
        mock_state.character.position = type('MockPosition', (), {'x': 100, 'y': 150})()
        mock_state.character.current_pa = 8
        mock_state.character.current_pm = 4
        mock_state.character.spells = {
            "sword_celestial": type('MockSpell', (), {'spell_id': 1, 'name': 'Épée Céleste'})(),
            "colere_iop": type('MockSpell', (), {'spell_id': 2, 'name': 'Colère de Iop'})()
        }
        mock_state.character.character_class = type('MockClass', (), {'value': 'iop'})()
        mock_state.character.is_dead = False

        # HP percentage mock
        mock_state.character._hp_percentage = 80.0
        mock_state.character.hp_percentage = lambda: mock_state.character._hp_percentage

        # Combat mock
        mock_state.combat = type('MockCombat', (), {})()
        mock_state.combat.enemies = []
        mock_state.combat.allies = [type('MockAlly', (), {'entity_id': 1, 'name': 'Ally1'})()]
        mock_state.combat.state = "active"

        # Map mock
        mock_state.current_map = "Astrub"
        mock_state.timestamp = time.time()

        return mock_state

    def cleanup(self):
        """Nettoie la démonstration"""
        print("\n🧹 Nettoyage...")

        for name, module in self.modules.items():
            try:
                module.cleanup()
                print(f"  ✅ {name} nettoyé")
            except Exception as e:
                print(f"  ❌ Erreur nettoyage {name}: {e}")

        self.demo_running = False
        print("✅ Démonstration terminée")


def main():
    """Fonction principale"""
    print("🤖 TacticalBot - Projet Augmenta - Démonstration")
    print("=" * 60)

    demo = AugmentaDemo()

    if demo.initialize_modules():
        demo.run_demo()
    else:
        print("❌ Impossible d'initialiser les modules")
        print("Assurez-vous que tous les modules sont correctement installés")


if __name__ == "__main__":
    main()