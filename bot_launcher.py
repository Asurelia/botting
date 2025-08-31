#!/usr/bin/env python3
"""
Launcher intelligent du bot DOFUS avec gestion avancée des profils.

Ce module fournit un launcher intelligent qui peut :
- Gérer plusieurs profils utilisateur
- Détecter automatiquement les conditions optimales de lancement
- Planifier des sessions de bot
- Gérer les redémarrages automatiques
- Monitorer les performances et ajuster les paramètres

Usage:
    python bot_launcher.py --create-profile "Farmer"
    python bot_launcher.py --launch-profile "Farmer" --schedule "14:00"
    python bot_launcher.py --auto-detect
    python bot_launcher.py --batch-profiles "Farmer,Miner" --interval 30
"""

import sys
import os
import json
import argparse
import logging
import asyncio
import schedule
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible - certaines fonctionnalités seront limitées")


class ProfileType(Enum):
    """Types de profils disponibles."""
    FARMER = "farmer"
    MINER = "miner"
    LUMBERJACK = "lumberjack"
    ALCHEMIST = "alchemist"
    COMBAT = "combat"
    DUNGEON = "dungeon"
    LEVELING = "leveling"
    ECONOMY = "economy"
    CUSTOM = "custom"


class LaunchCondition(Enum):
    """Conditions de lancement."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    SYSTEM_IDLE = "system_idle"
    LOW_CPU = "low_cpu"
    NETWORK_AVAILABLE = "network_available"
    GAME_DETECTED = "game_detected"


@dataclass
class ProfileConfiguration:
    """Configuration d'un profil utilisateur."""
    name: str
    type: ProfileType
    description: str
    created: str
    last_used: Optional[str] = None
    
    # Configuration du personnage
    character_name: str = "BotCharacter"
    character_class: str = "iop"
    character_level: int = 1
    server: str = "Ily"
    
    # Modules activés
    modules: Dict[str, Dict[str, Any]] = None
    
    # Paramètres de sécurité
    safety_settings: Dict[str, Any] = None
    
    # Paramètres d'automatisation
    automation_settings: Dict[str, Any] = None
    
    # Statistiques
    total_runtime: int = 0  # en secondes
    sessions_count: int = 0
    last_performance: Dict[str, Any] = None
    
    # Planification
    schedule_enabled: bool = False
    schedule_times: List[str] = None
    max_session_duration: int = 14400  # 4 heures par défaut
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = self._get_default_modules()
        if self.safety_settings is None:
            self.safety_settings = self._get_default_safety()
        if self.automation_settings is None:
            self.automation_settings = self._get_default_automation()
        if self.schedule_times is None:
            self.schedule_times = []
            
    def _get_default_modules(self) -> Dict[str, Dict[str, Any]]:
        """Obtenir la configuration par défaut des modules."""
        base_modules = {
            "combat": {"enabled": False, "priority": 1},
            "professions": {"enabled": False, "priority": 2},
            "navigation": {"enabled": True, "priority": 3},
            "economy": {"enabled": False, "priority": 4},
            "social": {"enabled": False, "priority": 5},
            "safety": {"enabled": True, "priority": 0},
            "automation": {"enabled": True, "priority": 6}
        }
        
        # Configuration spécifique selon le type de profil
        if self.type == ProfileType.FARMER:
            base_modules["professions"]["enabled"] = True
            base_modules["professions"]["focus"] = ["farmer"]
        elif self.type == ProfileType.MINER:
            base_modules["professions"]["enabled"] = True
            base_modules["professions"]["focus"] = ["miner"]
        elif self.type == ProfileType.COMBAT:
            base_modules["combat"]["enabled"] = True
        elif self.type == ProfileType.ECONOMY:
            base_modules["economy"]["enabled"] = True
            
        return base_modules
    
    def _get_default_safety(self) -> Dict[str, Any]:
        """Obtenir la configuration par défaut de sécurité."""
        return {
            "detection_avoidance": True,
            "human_behavior": True,
            "anti_detection_delay_min": 500,
            "anti_detection_delay_max": 2000,
            "max_session_time": self.max_session_duration,
            "break_intervals": [3600, 7200],  # Pauses toutes les heures
            "randomize_breaks": True,
            "emergency_stop_conditions": [
                "high_cpu_usage",
                "network_disconnection", 
                "game_crash_detection",
                "suspicious_activity"
            ]
        }
    
    def _get_default_automation(self) -> Dict[str, Any]:
        """Obtenir la configuration par défaut d'automatisation."""
        return {
            "auto_start": False,
            "auto_shutdown": True,
            "auto_restart_on_crash": True,
            "daily_routine": True,
            "smart_scheduling": True,
            "performance_monitoring": True,
            "adaptive_settings": True
        }


class SystemMonitor:
    """Moniteur système pour détecter les conditions optimales de lancement."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Obtenir le statut système complet."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_usage": 0,
            "network_active": False,
            "game_running": False,
            "system_idle": False
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU et mémoire
                status["cpu_percent"] = psutil.cpu_percent(interval=1)
                status["memory_percent"] = psutil.virtual_memory().percent
                status["disk_usage"] = psutil.disk_usage('/').percent
                
                # Activité réseau
                net_io = psutil.net_io_counters()
                if hasattr(net_io, 'bytes_sent') and hasattr(net_io, 'bytes_recv'):
                    status["network_active"] = True
                    
                # Détection du jeu DOFUS
                status["game_running"] = self._is_game_running()
                
                # Système inactif (CPU < 20% pendant 5 minutes)
                status["system_idle"] = status["cpu_percent"] < 20
                
            except Exception as e:
                self.logger.warning(f"Erreur lors de la récupération du statut système : {e}")
                
        return status
    
    def _is_game_running(self) -> bool:
        """Détecter si le jeu DOFUS est en cours d'exécution."""
        if not PSUTIL_AVAILABLE:
            return False
            
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'dofus' in proc.info['name'].lower():
                    return True
        except Exception as e:
            self.logger.warning(f"Erreur lors de la détection du jeu : {e}")
            
        return False
    
    def check_launch_conditions(self, conditions: List[LaunchCondition]) -> Tuple[bool, List[str]]:
        """Vérifier si les conditions de lancement sont remplies."""
        status = self.get_system_status()
        can_launch = True
        reasons = []
        
        for condition in conditions:
            if condition == LaunchCondition.IMMEDIATE:
                continue
            elif condition == LaunchCondition.SYSTEM_IDLE:
                if not status["system_idle"]:
                    can_launch = False
                    reasons.append(f"Système non inactif (CPU: {status['cpu_percent']:.1f}%)")
            elif condition == LaunchCondition.LOW_CPU:
                if status["cpu_percent"] > 50:
                    can_launch = False
                    reasons.append(f"CPU trop élevé : {status['cpu_percent']:.1f}%")
            elif condition == LaunchCondition.NETWORK_AVAILABLE:
                if not status["network_active"]:
                    can_launch = False
                    reasons.append("Réseau non disponible")
            elif condition == LaunchCondition.GAME_DETECTED:
                if not status["game_running"]:
                    can_launch = False
                    reasons.append("Jeu DOFUS non détecté")
                    
        return can_launch, reasons


class ProfileManager:
    """Gestionnaire de profils utilisateur."""
    
    def __init__(self, profiles_dir: Path = None):
        self.profiles_dir = profiles_dir or Path("config/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ProfileManager")
        
    def create_profile(self, name: str, profile_type: ProfileType, **kwargs) -> ProfileConfiguration:
        """Créer un nouveau profil."""
        if self.profile_exists(name):
            raise ValueError(f"Le profil '{name}' existe déjà")
            
        profile = ProfileConfiguration(
            name=name,
            type=profile_type,
            description=kwargs.get('description', f'Profil {profile_type.value}'),
            created=datetime.now().isoformat(),
            **{k: v for k, v in kwargs.items() if k != 'description'}
        )
        
        self.save_profile(profile)
        self.logger.info(f"Profil '{name}' créé avec succès")
        return profile
    
    def load_profile(self, name: str) -> ProfileConfiguration:
        """Charger un profil existant."""
        profile_path = self.profiles_dir / f"{name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Le profil '{name}' n'existe pas")
            
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convertir le type de profil
            if 'type' in data and isinstance(data['type'], str):
                data['type'] = ProfileType(data['type'])
                
            return ProfileConfiguration(**data)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du profil '{name}' : {e}")
            raise
    
    def save_profile(self, profile: ProfileConfiguration):
        """Sauvegarder un profil."""
        profile_path = self.profiles_dir / f"{profile.name}.json"
        
        try:
            data = asdict(profile)
            # Convertir l'enum en string
            if 'type' in data:
                data['type'] = data['type'].value if isinstance(data['type'], ProfileType) else data['type']
                
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Profil '{profile.name}' sauvegardé")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du profil '{profile.name}' : {e}")
            raise
    
    def list_profiles(self) -> List[str]:
        """Lister tous les profils disponibles."""
        return [f.stem for f in self.profiles_dir.glob("*.json")]
    
    def profile_exists(self, name: str) -> bool:
        """Vérifier si un profil existe."""
        return (self.profiles_dir / f"{name}.json").exists()
    
    def delete_profile(self, name: str):
        """Supprimer un profil."""
        profile_path = self.profiles_dir / f"{name}.json"
        
        if profile_path.exists():
            profile_path.unlink()
            self.logger.info(f"Profil '{name}' supprimé")
        else:
            raise FileNotFoundError(f"Le profil '{name}' n'existe pas")
    
    def update_profile_stats(self, name: str, runtime: int, performance_data: Dict[str, Any]):
        """Mettre à jour les statistiques d'un profil."""
        try:
            profile = self.load_profile(name)
            profile.total_runtime += runtime
            profile.sessions_count += 1
            profile.last_used = datetime.now().isoformat()
            profile.last_performance = performance_data
            self.save_profile(profile)
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des stats du profil '{name}' : {e}")


class BotLauncher:
    """Launcher intelligent du bot."""
    
    def __init__(self, profiles_dir: Path = None):
        self.profile_manager = ProfileManager(profiles_dir)
        self.system_monitor = SystemMonitor()
        self.logger = logging.getLogger(f"{__name__}.BotLauncher")
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
    async def launch_profile(self, profile_name: str, conditions: List[LaunchCondition] = None) -> bool:
        """Lancer un profil avec les conditions spécifiées."""
        if conditions is None:
            conditions = [LaunchCondition.IMMEDIATE]
            
        try:
            # Vérifier les conditions de lancement
            if LaunchCondition.IMMEDIATE not in conditions:
                can_launch, reasons = self.system_monitor.check_launch_conditions(conditions)
                if not can_launch:
                    self.logger.warning(f"Conditions de lancement non remplies : {', '.join(reasons)}")
                    return False
            
            # Charger le profil
            profile = self.profile_manager.load_profile(profile_name)
            
            # Vérifier si le profil n'est pas déjà en cours
            if profile_name in self.active_processes:
                if self.active_processes[profile_name].poll() is None:
                    self.logger.warning(f"Le profil '{profile_name}' est déjà en cours d'exécution")
                    return False
                else:
                    # Processus terminé, nettoyer
                    del self.active_processes[profile_name]
            
            # Lancer le bot avec le profil
            success = await self._start_bot_process(profile)
            
            if success:
                self.logger.info(f"Profil '{profile_name}' lancé avec succès")
                return True
            else:
                self.logger.error(f"Échec du lancement du profil '{profile_name}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors du lancement du profil '{profile_name}' : {e}")
            return False
    
    async def _start_bot_process(self, profile: ProfileConfiguration) -> bool:
        """Démarrer le processus du bot."""
        try:
            # Commande de lancement
            cmd = [
                sys.executable,
                "main.py",
                "--mode", "service",
                "--profile", profile.name,
                "--daemon"
            ]
            
            # Ajouter le debug si nécessaire
            if profile.automation_settings.get("debug_mode", False):
                cmd.append("--debug")
            
            # Démarrer le processus
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Attendre un peu pour vérifier que le processus démarre correctement
            await asyncio.sleep(2)
            
            if process.poll() is None:
                # Processus démarré avec succès
                self.active_processes[profile.name] = process
                return True
            else:
                # Processus a échoué
                stdout, stderr = process.communicate()
                self.logger.error(f"Erreur de démarrage : {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors du démarrage du processus : {e}")
            return False
    
    def stop_profile(self, profile_name: str) -> bool:
        """Arrêter un profil en cours d'exécution."""
        if profile_name not in self.active_processes:
            self.logger.warning(f"Le profil '{profile_name}' n'est pas en cours d'exécution")
            return False
            
        try:
            process = self.active_processes[profile_name]
            process.terminate()
            
            # Attendre jusqu'à 10 secondes pour l'arrêt propre
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill si nécessaire
                process.kill()
                process.wait()
            
            del self.active_processes[profile_name]
            self.logger.info(f"Profil '{profile_name}' arrêté")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'arrêt du profil '{profile_name}' : {e}")
            return False
    
    def get_active_profiles(self) -> List[str]:
        """Obtenir la liste des profils actifs."""
        active = []
        for name, process in list(self.active_processes.items()):
            if process.poll() is None:
                active.append(name)
            else:
                # Processus terminé, nettoyer
                del self.active_processes[name]
        return active
    
    def setup_schedule(self, profile_name: str, schedule_times: List[str]):
        """Configurer la planification pour un profil."""
        try:
            profile = self.profile_manager.load_profile(profile_name)
            profile.schedule_enabled = True
            profile.schedule_times = schedule_times
            self.profile_manager.save_profile(profile)
            
            # Configurer les tâches planifiées
            for time_str in schedule_times:
                schedule.every().day.at(time_str).do(
                    lambda p=profile_name: asyncio.run(self.launch_profile(p))
                )
            
            self.logger.info(f"Planification configurée pour '{profile_name}' : {schedule_times}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de la planification : {e}")
    
    async def batch_launch(self, profile_names: List[str], interval: int = 30):
        """Lancer plusieurs profils en série avec un intervalle."""
        self.logger.info(f"Lancement en série de {len(profile_names)} profils")
        
        for i, profile_name in enumerate(profile_names):
            self.logger.info(f"Lancement du profil {i+1}/{len(profile_names)}: {profile_name}")
            
            success = await self.launch_profile(profile_name)
            
            if success:
                self.logger.info(f"Profil '{profile_name}' lancé, attente de {interval}s")
                if i < len(profile_names) - 1:  # Pas d'attente après le dernier
                    await asyncio.sleep(interval)
            else:
                self.logger.error(f"Échec du lancement de '{profile_name}', continuation...")
    
    def auto_detect_and_launch(self) -> Optional[str]:
        """Détecter automatiquement le meilleur profil à lancer."""
        try:
            profiles = self.profile_manager.list_profiles()
            if not profiles:
                self.logger.warning("Aucun profil disponible pour la détection automatique")
                return None
            
            system_status = self.system_monitor.get_system_status()
            current_hour = datetime.now().hour
            
            # Logique de sélection intelligente
            best_profile = None
            best_score = 0
            
            for profile_name in profiles:
                try:
                    profile = self.profile_manager.load_profile(profile_name)
                    score = self._calculate_profile_score(profile, system_status, current_hour)
                    
                    if score > best_score:
                        best_score = score
                        best_profile = profile_name
                        
                except Exception as e:
                    self.logger.warning(f"Erreur lors de l'évaluation du profil '{profile_name}' : {e}")
                    continue
            
            if best_profile:
                self.logger.info(f"Profil sélectionné automatiquement : {best_profile} (score: {best_score})")
                return best_profile
            else:
                self.logger.warning("Aucun profil approprié trouvé")
                return None
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection automatique : {e}")
            return None
    
    def _calculate_profile_score(self, profile: ProfileConfiguration, system_status: Dict, current_hour: int) -> float:
        """Calculer un score pour un profil basé sur les conditions actuelles."""
        score = 0.0
        
        # Score basé sur les performances passées
        if profile.last_performance:
            efficiency = profile.last_performance.get("efficiency", 0.5)
            score += efficiency * 30
        
        # Score basé sur le type de profil et l'heure
        if profile.type in [ProfileType.FARMER, ProfileType.MINER] and 6 <= current_hour <= 22:
            score += 20
        elif profile.type == ProfileType.COMBAT and (20 <= current_hour or current_hour <= 2):
            score += 25
        
        # Score basé sur la charge système
        cpu_percent = system_status.get("cpu_percent", 100)
        if cpu_percent < 30:
            score += 15
        elif cpu_percent < 50:
            score += 10
        
        # Score basé sur la disponibilité du jeu
        if system_status.get("game_running", False):
            score += 20
        
        # Pénalité si le profil a été utilisé récemment
        if profile.last_used:
            last_used = datetime.fromisoformat(profile.last_used)
            hours_since = (datetime.now() - last_used).total_seconds() / 3600
            if hours_since < 4:
                score -= 15
                
        return max(0, score)


async def main():
    """Fonction principale du launcher."""
    parser = argparse.ArgumentParser(
        description="Launcher intelligent du bot DOFUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --create-profile "Farmer" --type farmer
  %(prog)s --launch-profile "Farmer"
  %(prog)s --launch-profile "Miner" --schedule "14:00"
  %(prog)s --auto-detect
  %(prog)s --batch-profiles "Farmer,Miner" --interval 30
  %(prog)s --list-profiles
        """
    )
    
    # Actions principales
    parser.add_argument("--create-profile", help="Créer un nouveau profil")
    parser.add_argument("--launch-profile", help="Lancer un profil spécifique")
    parser.add_argument("--stop-profile", help="Arrêter un profil en cours")
    parser.add_argument("--delete-profile", help="Supprimer un profil")
    parser.add_argument("--list-profiles", action="store_true", help="Lister tous les profils")
    parser.add_argument("--batch-profiles", help="Lancer plusieurs profils (séparés par des virgules)")
    
    # Options de profil
    parser.add_argument("--type", choices=[t.value for t in ProfileType], help="Type de profil à créer")
    parser.add_argument("--description", help="Description du profil")
    parser.add_argument("--character-name", help="Nom du personnage")
    parser.add_argument("--character-class", help="Classe du personnage")
    parser.add_argument("--server", default="Ily", help="Serveur de jeu")
    
    # Options de lancement
    parser.add_argument("--schedule", help="Heure de lancement programmé (HH:MM)")
    parser.add_argument("--conditions", nargs="+", choices=[c.value for c in LaunchCondition], 
                       help="Conditions de lancement")
    parser.add_argument("--interval", type=int, default=30, help="Intervalle entre lancements batch (secondes)")
    
    # Détection automatique
    parser.add_argument("--auto-detect", action="store_true", help="Détection automatique du meilleur profil")
    
    # Configuration
    parser.add_argument("--profiles-dir", help="Répertoire des profils")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer le launcher
    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else None
    launcher = BotLauncher(profiles_dir)
    
    try:
        # Actions de gestion des profils
        if args.create_profile:
            if not args.type:
                print("Erreur: --type requis pour créer un profil")
                return 1
                
            profile_type = ProfileType(args.type)
            profile = launcher.profile_manager.create_profile(
                name=args.create_profile,
                profile_type=profile_type,
                description=args.description or f"Profil {args.type}",
                character_name=args.character_name or "BotCharacter",
                character_class=args.character_class or "iop",
                server=args.server
            )
            print(f"Profil '{args.create_profile}' créé avec succès")
            
        elif args.list_profiles:
            profiles = launcher.profile_manager.list_profiles()
            if profiles:
                print("Profils disponibles:")
                for profile_name in profiles:
                    try:
                        profile = launcher.profile_manager.load_profile(profile_name)
                        status = "actif" if profile_name in launcher.get_active_profiles() else "inactif"
                        print(f"  - {profile_name} ({profile.type.value}) - {status}")
                    except:
                        print(f"  - {profile_name} (erreur de chargement)")
            else:
                print("Aucun profil disponible")
                
        elif args.delete_profile:
            launcher.profile_manager.delete_profile(args.delete_profile)
            print(f"Profil '{args.delete_profile}' supprimé")
            
        # Actions de lancement
        elif args.launch_profile:
            conditions = []
            if args.conditions:
                conditions = [LaunchCondition(c) for c in args.conditions]
            if args.schedule:
                conditions.append(LaunchCondition.SCHEDULED)
                launcher.setup_schedule(args.launch_profile, [args.schedule])
                
            success = await launcher.launch_profile(args.launch_profile, conditions)
            if success:
                print(f"Profil '{args.launch_profile}' lancé avec succès")
            else:
                print(f"Échec du lancement du profil '{args.launch_profile}'")
                return 1
                
        elif args.stop_profile:
            success = launcher.stop_profile(args.stop_profile)
            if success:
                print(f"Profil '{args.stop_profile}' arrêté")
            else:
                print(f"Impossible d'arrêter le profil '{args.stop_profile}'")
                return 1
                
        elif args.batch_profiles:
            profile_names = [name.strip() for name in args.batch_profiles.split(",")]
            await launcher.batch_launch(profile_names, args.interval)
            print(f"Lancement en série terminé pour {len(profile_names)} profils")
            
        elif args.auto_detect:
            best_profile = launcher.auto_detect_and_launch()
            if best_profile:
                success = await launcher.launch_profile(best_profile)
                if success:
                    print(f"Profil '{best_profile}' lancé automatiquement")
                else:
                    print(f"Échec du lancement automatique de '{best_profile}'")
                    return 1
            else:
                print("Aucun profil approprié trouvé pour le lancement automatique")
                return 1
        else:
            # Mode interactif par défaut
            print("=== LAUNCHER INTELLIGENT DU BOT DOFUS ===")
            print("Utilisez --help pour voir les options disponibles")
            
            # Afficher les profils actifs
            active = launcher.get_active_profiles()
            if active:
                print(f"Profils actifs : {', '.join(active)}")
            else:
                print("Aucun profil actif")
                
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
        return 0
    except Exception as e:
        logging.error(f"Erreur critique : {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)