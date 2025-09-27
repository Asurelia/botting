"""
Point d'entrée simplifié pour usage personnel
Version streamlinée sans complexité inutile
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Ajout du dossier au path pour les imports
sys.path.append(str(Path(__file__).parent))

from core.simple_security import SimpleSecurity
from modules.vision.screen_analyzer import ScreenAnalyzer
from modules.combat.classes.iop import IopClass
from modules.professions.farmer import Farmer
from engine.core import BotEngine, EngineConfig
from state.realtime_state import GameState


class SimpleDofusBot:
    """
    Bot DOFUS simplifié pour usage personnel
    Juste l'essentiel : combat, farming, sécurité de base
    """
    
    def __init__(self, max_session_hours: float = 3.0):
        """
        Initialise le bot simple
        
        Args:
            max_session_hours: Durée max de session (défaut 3h)
        """
        print("[BOT] Initialisation du bot DOFUS simplifié...")
        
        # Sécurité simplifiée
        self.security = SimpleSecurity(max_session_hours)
        
        # État du jeu
        self.game_state = GameState()
        
        # Vision de base
        self.vision = ScreenAnalyzer()
        
        # Modules de gameplay
        self.character_class = None  # Sera détecté automatiquement
        self.farmer = Farmer()
        
        # État du bot
        self.is_running = False
        self.current_mode = "idle"  # idle, combat, farming
        
        print("[OK] Bot initialisé avec succès !")
    
    def start(self, mode: str = "auto"):
        """
        Démarre le bot
        
        Args:
            mode: Mode de fonctionnement (auto, combat, farming)
        """
        print(f"[START] Démarrage du bot en mode '{mode}'...")
        
        try:
            # Initialisation des modules
            if not self._initialize_modules():
                print("[ERROR] Échec de l'initialisation des modules")
                return False
            
            self.is_running = True
            self.current_mode = mode
            
            # Boucle principale simplifiée
            self._main_loop()
            
        except KeyboardInterrupt:
            print("\n[STOP] Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"[ERROR] Erreur critique : {e}")
            self.security.logger.log_error("Erreur critique", e)
        finally:
            self._cleanup()
    
    def _initialize_modules(self) -> bool:
        """Initialise les modules nécessaires"""
        try:
            # Vision
            if not self.vision.initialize({}):
                print("[ERROR] Impossible d'initialiser la vision")
                return False
            
            # Test de capture d'écran
            screenshot = self.vision.get_current_screenshot()
            if screenshot is None:
                print("[WARNING] Capture d'écran non disponible pour le moment")
                # On continue quand même
            
            print("[OK] Vision initialisée")
            
            # Farmer
            if not self.farmer.initialize({}):
                print("[ERROR] Impossible d'initialiser le farmer")
                return False
            
            print("[OK] Farmer initialisé")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Erreur lors de l'initialisation : {e}")
            return False
    
    def _main_loop(self):
        """
        Boucle principale simplifiée
        Plus simple que le moteur complexe, juste l'essentiel
        """
        print("[LOOP] Démarrage de la boucle principale...")
        
        loop_count = 0
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                # Vérification sécurité
                if self.security.should_stop():
                    print("[STOP] Session trop longue, arrêt du bot")
                    break
                
                # Mise à jour état du jeu
                self._update_game_state()
                
                # Action selon le mode
                if self.current_mode == "auto":
                    self._auto_mode()
                elif self.current_mode == "farming":
                    self._farming_mode()
                elif self.current_mode == "combat":
                    self._combat_mode()
                
                # Statistiques toutes les 100 boucles (environ 10 secondes)
                loop_count += 1
                if loop_count % 100 == 0:
                    self._print_status()
                
                # Attente pour maintenir ~10 FPS (plus économe que 30 FPS)
                loop_time = time.time() - loop_start
                target_time = 0.1  # 100ms = 10 FPS
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
                
            except Exception as e:
                print(f"[WARNING] Erreur dans la boucle : {e}")
                self.security.logger.log_error("Erreur boucle principale", e)
                
                # Attente avant de continuer
                time.sleep(1.0)
    
    def _update_game_state(self):
        """Met à jour l'état du jeu via la vision"""
        try:
            screenshot = self.vision.get_current_screenshot()
            if screenshot is not None:
                # Analyse basique de l'écran
                analysis = self.vision.analyze(screenshot)
                
                # Mise à jour de l'état
                if analysis:
                    self.game_state.update_from_screen(analysis)
        
        except Exception as e:
            # Pas critique, on continue
            pass
    
    def _auto_mode(self):
        """
        Mode automatique : décide quoi faire selon la situation
        """
        # Simple logique de décision
        context = self.game_state.get_context_for_decision()
        
        if context["combat"]["in_combat"]:
            self._handle_combat()
        elif context["opportunities"]["harvestable_resources"] > 0:
            self._handle_farming()
        else:
            # Mode exploration/attente
            self._handle_idle()
    
    def _farming_mode(self):
        """Mode farming pur"""
        self._handle_farming()
    
    def _combat_mode(self):
        """Mode combat pur"""
        self._handle_combat()
    
    def _handle_combat(self):
        """Gestion simple du combat"""
        try:
            # Logique de combat basique
            context = self.game_state.get_context_for_decision()
            
            if context["combat"]["is_my_turn"]:
                print("[COMBAT] Mon tour de combat")
                
                # Action de combat simple (à améliorer)
                # Ici on pourrait utiliser l'IA de combat, mais gardons simple
                self._simple_combat_action()
                
                self.security.logger.log_action("combat_turn")
        
        except Exception as e:
            print(f"[WARNING] Erreur combat : {e}")
    
    def _simple_combat_action(self):
        """Action de combat ultra simple"""
        # Pour l'instant, juste une attente
        # Dans la vraie implémentation, on utiliserait les sorts de la classe
        print("[ACTION] Action de combat basique")
        self.security.safe_wait(1.0, 0.5)
    
    def _handle_farming(self):
        """Gestion simple du farming"""
        try:
            print("[FARMING] Mode farming")
            
            # Logique de farming basique
            # Ici on utiliserait le système de métiers complet
            self._simple_farming_action()
            
            self.security.logger.log_action("farming_action")
        
        except Exception as e:
            print(f"[WARNING] Erreur farming : {e}")
    
    def _simple_farming_action(self):
        """Action de farming ultra simple"""
        # Pour l'instant, juste une simulation
        # Dans la vraie implémentation, on utiliserait le farmer complet
        print("[ACTION] Action de farming basique")
        self.security.safe_wait(2.0, 0.5)
    
    def _handle_idle(self):
        """Mode attente/exploration"""
        # Juste une attente
        self.security.safe_wait(1.0, 0.3)
    
    def _print_status(self):
        """Affiche le statut du bot"""
        status = self.security.get_status()
        context = self.game_state.get_context_for_decision()
        
        print(f"\n[STATUS] Mode: {self.current_mode} | "
              f"Actions: {status['actions_count']} | "
              f"Session: {status['session']['duration_minutes']:.1f}min | "
              f"HP: {context['character']['hp_percentage']:.1f}%")
        
        # Avertissement si session longue
        warning = self.security.session.suggest_break()
        if warning:
            print(f"[WARNING] {warning}")
    
    def _cleanup(self):
        """Nettoyage à l'arrêt"""
        print("[CLEANUP] Nettoyage en cours...")
        
        self.is_running = False
        
        # Arrêt des modules
        try:
            self.vision.cleanup()
            print("[OK] Vision arrêtée")
        except:
            pass
        
        # Statistiques finales
        print("\n[STATS] STATISTIQUES FINALES:")
        self.security.print_status()
        
        print("[OK] Bot arrêté proprement")


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Bot DOFUS simplifié")
    parser.add_argument("--mode", choices=["auto", "combat", "farming"], 
                       default="auto", help="Mode de fonctionnement")
    parser.add_argument("--max-hours", type=float, default=3.0,
                       help="Durée max de session en heures")
    parser.add_argument("--test", action="store_true",
                       help="Mode test (simulation)")
    
    args = parser.parse_args()
    
    if args.test:
        # Mode test sans DOFUS
        print("[TEST] Mode test - simulation sans DOFUS")
        from core.simple_security import SimpleSecurity
        
        security = SimpleSecurity(0.05)  # 3 minutes pour test
        
        for i in range(20):
            pos = security.safe_click(100 + i*5, 200 + i*3, f"test_{i}")
            print(f"Test {i+1}: clic en {pos}")
            
            if i % 5 == 0:
                security.print_status()
                
            if security.should_stop():
                print("[STOP] Test arrêté - session simulée trop longue")
                break
        
        return
    
    # Mode normal
    print("="*60)
    print("[BOT] BOT DOFUS SIMPLIFIÉ - Version Streamlined")
    print("="*60)
    print("[INFO] Conseils:")
    print("   - Assure-toi que DOFUS est ouvert et visible")
    print("   - Le bot s'arrêtera automatiquement après la durée max")
    print("   - Utilise Ctrl+C pour arrêter manuellement")
    print("="*60)
    
    # Création et lancement du bot
    bot = SimpleDofusBot(max_session_hours=args.max_hours)
    bot.start(mode=args.mode)


if __name__ == "__main__":
    main()