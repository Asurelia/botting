"""
Script de test rapide pour l'extraction de données Dofus
Vérifie que tout fonctionne correctement
"""

import sys
from pathlib import Path

# Ajout du chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from tools.dofus_data_extractor import DofusDataManager

def main():
    print("=" * 70)
    print("🧪 TEST RAPIDE - EXTRACTION DONNÉES DOFUS")
    print("=" * 70)
    print()
    
    # Création du manager
    manager = DofusDataManager()
    
    # Test 1 : Recherche installation
    print("📍 Test 1 : Recherche installation Dofus Unity...")
    has_local = manager.setup()
    
    if has_local:
        print(f"   ✅ {len(manager.installations)} installation(s) trouvée(s)")
        for install in manager.installations:
            print(f"      • {install.path}")
            print(f"        Type: {install.install_type}")
            print(f"        Version: {install.version}")
            print(f"        Dossiers data: {len(install.data_folders)}")
    else:
        print("   ⚠️ Aucune installation locale trouvée")
        print("   💡 Le bot utilisera les fansites pour récupérer les données")
    
    print()
    
    # Test 2 : Extraction (si installation trouvée)
    if has_local:
        print("📊 Test 2 : Extraction des données...")
        print("   (Cela peut prendre 1-2 minutes)")
        print()
        
        all_data = manager.extract_all_data()
        
        print()
        print("✅ RÉSULTATS DE L'EXTRACTION:")
        print("-" * 70)
        
        total_entries = 0
        for category, data in all_data.items():
            count = len(data)
            total_entries += count
            if count > 0:
                status = "✅"
                print(f"   {status} {category.capitalize():15} : {count:5} entrées")
            else:
                status = "⚠️"
                print(f"   {status} {category.capitalize():15} : {count:5} entrées (aucune donnée)")
        
        print("-" * 70)
        print(f"   📊 TOTAL : {total_entries} entrées extraites")
        print()
        
        # Test 3 : Vérification qualité des données
        print("🔍 Test 3 : Vérification qualité des données...")
        
        if all_data["monsters"]:
            sample_monster = list(all_data["monsters"].values())[0]
            print(f"   ✅ Exemple monstre : {sample_monster.get('name', 'Unknown')}")
            print(f"      • Niveau : {sample_monster.get('level', 'N/A')}")
            print(f"      • HP : {sample_monster.get('health', 'N/A')}")
            
            required_fields = ["level", "health", "resistances"]
            has_all_fields = all(field in sample_monster for field in required_fields)
            
            if has_all_fields:
                print("   ✅ Structure des données : OK")
            else:
                print("   ⚠️ Structure des données : Incomplète")
        
        print()
        
        # Test 4 : Sauvegarde
        print("💾 Test 4 : Vérification fichiers sauvegardés...")
        
        extracted_dir = Path("data/extracted")
        if extracted_dir.exists():
            files = list(extracted_dir.glob("*.json"))
            print(f"   ✅ {len(files)} fichier(s) trouvé(s) dans data/extracted/")
            for file in files[:5]:  # Affiche les 5 premiers
                size_kb = file.stat().st_size / 1024
                print(f"      • {file.name} ({size_kb:.1f} KB)")
        else:
            print("   ⚠️ Dossier data/extracted/ non trouvé")
    
    else:
        print("📡 Test 2 : Mode fansite...")
        print("   💡 Exemple de récupération depuis fansite:")
        print("      monster_data = manager.get_monster_info('Bouftou')")
        print("   ⚠️ Non testé dans ce script (nécessite connexion internet)")
    
    print()
    print("=" * 70)
    print("✅ TESTS TERMINÉS")
    print("=" * 70)
    print()
    
    # Recommandations
    if has_local and total_entries > 0:
        print("🎉 SUCCÈS ! Votre bot peut maintenant utiliser les données extraites.")
        print()
        print("📝 PROCHAINES ÉTAPES:")
        print("   1. Intégrer les données dans knowledge_graph.py")
        print("   2. Utiliser dans decision_engine.py pour décisions intelligentes")
        print("   3. Référencer dans combat_advisor.py pour stratégies optimales")
        print()
        print("💡 ASTUCE:")
        print("   Planifiez une extraction hebdomadaire pour rester à jour avec")
        print("   les mises à jour du jeu Dofus Unity.")
    
    elif has_local and total_entries == 0:
        print("⚠️ ATTENTION : Installation trouvée mais aucune donnée extraite.")
        print()
        print("🔧 SOLUTIONS POSSIBLES:")
        print("   1. Vérifier que Dofus Unity est bien installé")
        print("   2. Lancer le jeu une fois pour générer les fichiers de données")
        print("   3. Vérifier les permissions d'accès aux dossiers")
        print("   4. Utiliser le mode fansite en attendant")
    
    else:
        print("💡 MODE FANSITE ACTIVÉ")
        print()
        print("   Le bot récupérera les données en ligne à la demande.")
        print("   Cela fonctionne très bien, mais peut être plus lent.")
        print()
        print("   Pour activer l'extraction locale:")
        print("   1. Installer Dofus Unity")
        print("   2. Relancer ce script")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
