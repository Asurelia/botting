#!/usr/bin/env python3
"""
Exemples d'utilisation des points d'entrée du bot DOFUS.

Ce script démontre comment utiliser les différents points d'entrée
et leurs intégrations pour des scénarios courants.
"""

import sys
import asyncio
import subprocess
import time
from pathlib import Path

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

from bot_launcher import BotLauncher, ProfileType
from calibrate import CalibrationTool
from monitor import MetricsCollector, AlertManager


async def exemple_creation_profil():
    """Exemple : Créer et configurer un profil complet."""
    print("=== Exemple 1 : Création de Profil ===")
    
    launcher = BotLauncher()
    
    # Créer un profil farmer
    try:
        profile = launcher.profile_manager.create_profile(
            name="ExempleFarmer",
            profile_type=ProfileType.FARMER,
            description="Profil d'exemple pour farmer",
            character_name="BotFarmeur",
            character_class="osamodas",
            server="Ily"
        )
        
        print(f"✓ Profil '{profile.name}' créé avec succès")
        print(f"  Type: {profile.type.value}")
        print(f"  Personnage: {profile.character_name}")
        print(f"  Modules actifs: {len(profile.modules)}")
        
        # Configurer des horaires de planification
        launcher.setup_schedule("ExempleFarmer", ["09:00", "14:00", "20:00"])
        print("✓ Planification configurée")
        
    except ValueError as e:
        print(f"✗ Erreur : {e}")
    
    print()


async def exemple_calibration_automatique():
    """Exemple : Calibration automatique de l'interface."""
    print("=== Exemple 2 : Calibration Automatique ===")
    
    calibration_tool = CalibrationTool()
    
    # Tentative de calibration automatique
    result = calibration_tool.auto_calibrate_interface()
    
    if "error" in result:
        print(f"✗ Erreur de calibration : {result['error']}")
    else:
        print(f"✓ Zones calibrées : {len(result['zones_calibrated'])}")
        print(f"✗ Zones échouées : {len(result['zones_failed'])}")
        
        if result["zones_calibrated"]:
            print("  Zones réussies :")
            for zone in result["zones_calibrated"]:
                print(f"    - {zone}")
        
        if result.get("screenshot_saved"):
            print(f"  Capture sauvegardée : {result['screenshot_path']}")
    
    # Tester une zone spécifique
    if calibration_tool.config and "chat_zone" in calibration_tool.config.zones:
        test_result = calibration_tool.test_zone_detection("chat_zone")
        if test_result["success"]:
            print(f"✓ Test zone chat réussi (complexité: {test_result['complexity_score']:.2f})")
        else:
            print(f"✗ Test zone chat échoué : {test_result['error']}")
    
    print()


async def exemple_monitoring_integration():
    """Exemple : Intégration du système de monitoring."""
    print("=== Exemple 3 : Monitoring Intégré ===")
    
    # Créer les composants de monitoring
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    
    print("✓ Composants de monitoring créés")
    
    # Simuler quelques métriques
    for i in range(5):
        system_metrics = metrics_collector.collect_system_metrics()
        bot_metrics = metrics_collector.collect_bot_metrics()
        
        print(f"  Mesure {i+1} - CPU: {system_metrics.cpu_percent:.1f}%, "
              f"Mémoire: {system_metrics.memory_percent:.1f}%")
        
        # Vérifier les alertes
        alert_manager.check_alerts(system_metrics, bot_metrics)
        
        # Stocker en base
        metrics_collector.store_metrics(system_metrics, bot_metrics)
        
        await asyncio.sleep(1)
    
    # Récupérer les alertes récentes
    recent_alerts = alert_manager.get_active_alerts(10)
    if recent_alerts:
        print(f"⚠️  Alertes détectées : {len(recent_alerts)}")
        for name, message, timestamp in recent_alerts:
            print(f"    - {name}: {message}")
    else:
        print("✓ Aucune alerte détectée")
    
    print()


async def exemple_lancement_intelligent():
    """Exemple : Lancement intelligent avec conditions."""
    print("=== Exemple 4 : Lancement Intelligent ===")
    
    launcher = BotLauncher()
    
    # Vérifier les profils disponibles
    profiles = launcher.profile_manager.list_profiles()
    print(f"Profils disponibles : {len(profiles)}")
    
    if profiles:
        # Détection automatique du meilleur profil
        best_profile = launcher.auto_detect_and_launch()
        
        if best_profile:
            print(f"✓ Profil optimal détecté : {best_profile}")
            
            # Simuler un lancement (sans vraiment démarrer le bot)
            print("  Simulation du lancement...")
            print("  ✓ Conditions vérifiées")
            print("  ✓ Profil chargé")
            print("  ✓ Bot prêt à démarrer")
            
        else:
            print("✗ Aucun profil approprié trouvé")
    else:
        print("ℹ️  Aucun profil configuré - utilisez bot_launcher.py pour en créer")
    
    print()


async def exemple_workflow_complet():
    """Exemple : Workflow complet d'utilisation."""
    print("=== Exemple 5 : Workflow Complet ===")
    
    print("1. Vérification des prérequis...")
    
    # Vérifier les dossiers
    required_dirs = [Path("config"), Path("config/profiles"), Path("logs")]
    for directory in required_dirs:
        if directory.exists():
            print(f"  ✓ {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  📁 {directory} créé")
    
    print("\n2. Configuration initiale...")
    
    # Créer un profil de test si nécessaire
    launcher = BotLauncher()
    if not launcher.profile_manager.list_profiles():
        try:
            profile = launcher.profile_manager.create_profile(
                name="TestBot",
                profile_type=ProfileType.FARMER,
                description="Profil de test automatique"
            )
            print("  ✓ Profil de test créé")
        except:
            print("  ⚠️  Profil de test déjà existant")
    
    print("\n3. Test des composants...")
    
    # Test calibration
    calibration_tool = CalibrationTool()
    if calibration_tool.config:
        print("  ✓ Calibration chargée")
    else:
        print("  ⚠️  Calibration non configurée")
    
    # Test monitoring
    metrics_collector = MetricsCollector()
    system_metrics = metrics_collector.collect_system_metrics()
    print(f"  ✓ Monitoring actif (CPU: {system_metrics.cpu_percent:.1f}%)")
    
    print("\n4. Simulation d'une session complète...")
    
    steps = [
        "Chargement du profil",
        "Vérification de la calibration",
        "Initialisation des modules", 
        "Démarrage du monitoring",
        "Lancement du bot",
        "Surveillance active",
        "Arrêt propre"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"  {i}/7 {step}...")
        await asyncio.sleep(0.5)  # Simulation
        print(f"      ✓ Terminé")
    
    print("\n✅ Workflow complet simulé avec succès !")
    print()


def exemple_scripts_utilitaires():
    """Exemples de scripts utilitaires."""
    print("=== Exemple 6 : Scripts Utilitaires ===")
    
    scripts = {
        "Démarrage rapide CLI": "python main.py --mode cli --auto-start",
        "Interface graphique": "python main.py --mode gui",
        "Monitoring dashboard": "python monitor.py --dashboard", 
        "Calibration GUI": "python calibrate.py --gui",
        "Créer profil farmer": "python bot_launcher.py --create-profile Farmer --type farmer",
        "Lancement automatique": "python bot_launcher.py --auto-detect",
        "Serveur web monitoring": "python monitor.py --web-server --port 8080"
    }
    
    print("Scripts prêts à utiliser :")
    for name, command in scripts.items():
        print(f"  {name}:")
        print(f"    {command}")
        print()


async def demonstration_complete():
    """Démonstration complète de tous les exemples."""
    print("🤖 DÉMONSTRATION DES POINTS D'ENTRÉE BOT DOFUS")
    print("=" * 60)
    print()
    
    # Exécuter tous les exemples
    await exemple_creation_profil()
    await exemple_calibration_automatique()
    await exemple_monitoring_integration()
    await exemple_lancement_intelligent()
    await exemple_workflow_complet()
    exemple_scripts_utilitaires()
    
    print("🎉 Démonstration terminée avec succès !")
    print()
    print("Pour utiliser le bot en production :")
    print("1. Créez vos profils : python bot_launcher.py --create-profile MonBot --type farmer")
    print("2. Calibrez l'interface : python calibrate.py --gui")
    print("3. Lancez le bot : python main.py --mode gui --profile MonBot")
    print("4. Surveillez : python monitor.py --dashboard")


if __name__ == "__main__":
    try:
        print("Lancement de la démonstration...")
        print("(Appuyez sur Ctrl+C pour interrompre)")
        print()
        
        asyncio.run(demonstration_complete())
        
    except KeyboardInterrupt:
        print("\n⏹️  Démonstration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration : {e}")
        sys.exit(1)