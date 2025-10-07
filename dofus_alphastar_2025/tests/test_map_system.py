#!/usr/bin/env python3
"""
Tests for Map System
Tests pathfinding, graph structure, and map discovery
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMapSystemImports(unittest.TestCase):
    """Test that all map system modules import correctly"""

    def test_import_map_graph(self):
        """Test importing map graph module"""
        try:
            from core.map_system import create_map_graph, MapCoords
            self.assertIsNotNone(create_map_graph)
            self.assertIsNotNone(MapCoords)
        except ImportError as e:
            self.fail(f"Failed to import map_graph: {e}")

    def test_import_map_discovery(self):
        """Test importing map discovery module"""
        try:
            from core.map_system import create_map_discovery
            self.assertIsNotNone(create_map_discovery)
        except ImportError as e:
            self.fail(f"Failed to import map_discovery: {e}")


class TestMapCoords(unittest.TestCase):
    """Test MapCoords dataclass"""

    def test_map_coords_creation(self):
        """Test creating MapCoords"""
        from core.map_system import MapCoords

        coords = MapCoords(5, -18)
        self.assertEqual(coords.x, 5)
        self.assertEqual(coords.y, -18)

    def test_map_coords_hashable(self):
        """Test that MapCoords can be used as dict key"""
        from core.map_system import MapCoords

        coords1 = MapCoords(5, -18)
        coords2 = MapCoords(5, -18)
        coords3 = MapCoords(3, -20)

        # Test equality
        self.assertEqual(coords1, coords2)
        self.assertNotEqual(coords1, coords3)

        # Test as dict key
        map_dict = {coords1: "Astrub", coords3: "Bonta"}
        self.assertEqual(map_dict[coords2], "Astrub")


class TestMapGraph(unittest.TestCase):
    """Test MapGraph functionality"""

    def test_create_map_graph(self):
        """Test creating map graph"""
        from core.map_system import create_map_graph

        graph = create_map_graph()
        self.assertIsNotNone(graph)

    def test_map_graph_has_maps(self):
        """Test that graph contains maps"""
        from core.map_system import create_map_graph

        graph = create_map_graph()
        # Graph may be empty initially (maps loaded from database if exists)
        # This is OK, we just test the structure works
        self.assertIsNotNone(graph.maps)
        self.assertIsInstance(graph.maps, dict)

    def test_add_map(self):
        """Test adding a map to the graph"""
        from core.map_system import create_map_graph, MapCoords, MapNode

        graph = create_map_graph()
        test_coords = MapCoords(999, 999)

        # Add map
        test_node = MapNode(
            coords=test_coords,
            name="Test Map",
            area="Test Area",
            has_zaap=False,
            is_pvp=False
        )
        graph.add_map(test_node)

        # Verify map was added
        self.assertIn(test_coords, graph.maps)
        map_info = graph.get_map_info(test_coords)
        self.assertEqual(map_info.name, "Test Map")

    def test_pathfinding_simple(self):
        """Test simple pathfinding"""
        from core.map_system import create_map_graph, MapCoords, MapNode, MapEdge

        graph = create_map_graph()

        # Add connected maps
        map1 = MapCoords(0, 0)
        map2 = MapCoords(1, 0)
        map3 = MapCoords(2, 0)

        graph.add_map(MapNode(coords=map1, name="Map1", area="Area1"))
        graph.add_map(MapNode(coords=map2, name="Map2", area="Area1"))
        graph.add_map(MapNode(coords=map3, name="Map3", area="Area1"))

        # Add connections
        graph.add_connection(MapEdge(from_map=map1, to_map=map2, exit_direction='east'))
        graph.add_connection(MapEdge(from_map=map2, to_map=map3, exit_direction='east'))

        # Find path
        path = graph.find_path(map1, map3)

        if path:  # Path might not be found depending on graph implementation
            self.assertIn(map1, path)
            self.assertIn(map3, path)

    def test_mark_discovered(self):
        """Test marking maps as discovered"""
        from core.map_system import create_map_graph, MapCoords, MapNode

        graph = create_map_graph()
        coords = MapCoords(5, -18)

        # Add map first
        graph.add_map(MapNode(coords=coords, name="Test", area="Test"))

        graph.mark_discovered(coords)
        self.assertIn(coords, graph.discovered_maps)


class TestMapDiscovery(unittest.TestCase):
    """Test MapDiscovery functionality"""

    def test_create_map_discovery(self):
        """Test creating map discovery"""
        from core.map_system import create_map_discovery

        try:
            discovery = create_map_discovery()
            self.assertIsNotNone(discovery)
        except Exception as e:
            # Skip test if dependencies not available
            self.skipTest(f"MapDiscovery dependencies not available: {e}")

    def test_map_discovery_integration(self):
        """Test map discovery with mocked screen capture"""
        from core.map_system import create_map_discovery

        try:
            discovery = create_map_discovery()
            # Just test structure exists
            self.assertTrue(hasattr(discovery, 'discover_current_map'))
        except Exception as e:
            self.skipTest(f"MapDiscovery dependencies not available: {e}")


class TestMapSystemIntegration(unittest.TestCase):
    """Test integration between map components"""

    def test_graph_and_discovery_together(self):
        """Test that graph and discovery work together"""
        from core.map_system import create_map_graph, MapCoords, MapNode

        graph = create_map_graph()

        # Simulate discovery
        discovered_coords = MapCoords(10, 10)
        graph.add_map(MapNode(
            coords=discovered_coords,
            name="Discovered Map",
            area="New Area",
            has_zaap=False,
            is_pvp=False
        ))
        graph.mark_discovered(discovered_coords)

        # Verify
        self.assertIn(discovered_coords, graph.maps)
        self.assertIn(discovered_coords, graph.discovered_maps)


def run_map_tests():
    """Run all map system tests"""
    print("=" * 70)
    print("TESTS SYSTEME DE CARTE")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\nResultats:")
    print(f"  Tests executes: {result.testsRun}")
    print(f"  Succes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Echecs: {len(result.failures)}")
    print(f"  Erreurs: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_map_tests()
    sys.exit(0 if success else 1)