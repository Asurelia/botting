#!/usr/bin/env python3
"""
Exemple 2: Utilisation du Map System
Démontre pathfinding et découverte de maps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.map_system import create_map_graph, MapCoords

def main():
    print("=" * 70)
    print("EXEMPLE 2: Map System - Pathfinding")
    print("=" * 70)

    # 1. Créer le graph
    print("\n[1] Création du Map Graph...")
    graph = create_map_graph()
    print(f"[OK] Map Graph créé")

    # 2. Ajouter des maps manuellement (pour l'exemple)
    print("\n[2] Ajout de maps...")

    # Astrub
    astrub = MapCoords(5, -18)
    graph.add_map(
        coords=astrub,
        name="Astrub Centre",
        area="Astrub",
        has_zaap=True,
        is_pvp=False,
        exits={
            'north': MapCoords(5, -17),
            'south': MapCoords(5, -19),
            'east': MapCoords(6, -18),
            'west': MapCoords(4, -18)
        }
    )
    print(f"[OK] Ajouté: {astrub} - Astrub Centre")

    # Voisins
    north = MapCoords(5, -17)
    graph.add_map(
        coords=north,
        name="Route Astrub Nord",
        area="Astrub",
        has_zaap=False,
        is_pvp=False,
        exits={
            'south': astrub,
            'north': MapCoords(5, -16)
        }
    )
    print(f"[OK] Ajouté: {north} - Route Astrub Nord")

    south = MapCoords(5, -19)
    graph.add_map(
        coords=south,
        name="Route Astrub Sud",
        area="Astrub",
        has_zaap=False,
        is_pvp=False,
        exits={
            'north': astrub,
            'south': MapCoords(5, -20)
        }
    )
    print(f"[OK] Ajouté: {south} - Route Astrub Sud")

    # 3. Pathfinding
    print("\n[3] Pathfinding...")

    print(f"\nRecherche chemin: {astrub} -> {south}")
    path = graph.find_path(astrub, south)

    if path:
        print(f"[OK] Chemin trouvé ({len(path)} étapes):")
        for i, coords in enumerate(path):
            map_info = graph.get_map_info(coords)
            print(f"  {i+1}. {coords} - {map_info.get('name', 'Unknown')}")
    else:
        print("[FAIL] Aucun chemin trouvé")

    # 4. Stats
    print("\n[4] Statistiques du Graph...")
    print(f"Maps totales: {len(graph.maps)}")
    print(f"Maps découvertes: {len(graph.discovered_maps)}")

    # 5. Marquer comme découvert
    print("\n[5] Marquage découverte...")
    graph.mark_discovered(astrub)
    graph.mark_discovered(north)
    print(f"[OK] Maps découvertes: {len(graph.discovered_maps)}")

    print("\n" + "=" * 70)
    print("[OK] Exemple terminé avec succès!")
    print("=" * 70)


if __name__ == "__main__":
    main()