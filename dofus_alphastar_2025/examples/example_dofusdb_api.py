#!/usr/bin/env python3
"""
Exemple 3: Utilisation de l'API DofusDB
Démontre recherche d'items, sorts, monstres
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.external_data import create_dofusdb_client

def main():
    print("=" * 70)
    print("EXEMPLE 3: DofusDB API")
    print("=" * 70)

    # 1. Créer le client
    print("\n[1] Création DofusDB Client...")
    client = create_dofusdb_client(cache_dir="cache/example3")
    print("[OK] Client créé")

    # 2. Recherche d'items
    print("\n[2] Recherche d'items...")

    query = "Dofus"
    print(f"\nRecherche: '{query}'")
    items = client.search_items(query, limit=5)

    if items:
        print(f"[OK] {len(items)} résultats trouvés:")
        for item in items:
            print(f"  - {item.name} (ID: {item.id}, Level: {item.level})")
            if item.description:
                print(f"    Description: {item.description[:50]}...")
    else:
        print(" Aucun résultat (API peut être offline)")

    # 3. Get item détaillé
    print("\n[3] Récupération item détaillé...")

    # Simulons avec un ID fictif ou utilisez un vrai ID si API disponible
    item_id = 1234
    print(f"\nRécupération item ID: {item_id}")

    item = client.get_item(item_id)

    if item:
        print(f"[OK] Item trouvé:")
        print(f"  Nom: {item.name}")
        print(f"  Type: {item.type}")
        print(f"  Level: {item.level}")
        if item.effects:
            print(f"  Effets: {len(item.effects)}")
    else:
        print("[FAIL] Item non trouvé (normal si ID fictif)")

    # 4. Statistiques du cache
    print("\n[4] Statistiques du cache...")
    stats = client.get_stats()
    print(f"Requêtes API: {stats['requests']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Erreurs: {stats['errors']}")
    print(f"Cache ratio: {stats['cache_ratio']}")

    # 5. Test résistances & dégâts
    print("\n[5] Calcul dégâts avec résistances...")

    # Note: Nécessite un spell valide de l'API
    # Cet exemple est simulé
    print("\nExemple théorique:")
    print("  Sort: Flamme (element: fire, damage: 100)")
    print("  Cible: 50% résistance feu")
    print("  Dégâts effectifs: 100 * (1 - 50/100) = 50")

    print("\n" + "=" * 70)
    print("[OK] Exemple terminé!")
    print("Note: API DofusDB peut être offline, utilise cache si disponible")
    print("=" * 70)


if __name__ == "__main__":
    main()