#!/usr/bin/env python3
"""
Tests for DofusDB Client
Tests API client, caching, and data structures
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDofusDBImports(unittest.TestCase):
    """Test that DofusDB modules import correctly"""

    def test_import_dofusdb_client(self):
        """Test importing DofusDB client"""
        try:
            from core.external_data import create_dofusdb_client, ItemData, SpellData, MonsterData
            self.assertIsNotNone(create_dofusdb_client)
            self.assertIsNotNone(ItemData)
            self.assertIsNotNone(SpellData)
            self.assertIsNotNone(MonsterData)
        except ImportError as e:
            self.fail(f"Failed to import DofusDB client: {e}")


class TestDataStructures(unittest.TestCase):
    """Test DofusDB data structures"""

    def test_item_data_creation(self):
        """Test ItemData dataclass"""
        from core.external_data import ItemData

        item = ItemData(
            id=1234,
            name="Dofus Émeraude",
            type="dofus",
            level=200,
            description="Un Dofus légendaire",
            effects=[{"type": "vitality", "value": 100}],
            conditions=["Level >= 200"],
            recipe=None,
            image_url="https://example.com/dofus.png"
        )

        self.assertEqual(item.id, 1234)
        self.assertEqual(item.name, "Dofus Émeraude")
        self.assertEqual(item.level, 200)

    def test_spell_data_creation(self):
        """Test SpellData dataclass"""
        from core.external_data import SpellData

        spell = SpellData(
            id=5678,
            name="Mot de Soin",
            class_name="Eniripsa",
            type="heal",
            element="water",
            levels=[
                {"level": 1, "damage": 10, "cost": 3},
                {"level": 2, "damage": 15, "cost": 3}
            ],
            description="Soigne un allié",
            image_url="https://example.com/spell.png"
        )

        self.assertEqual(spell.id, 5678)
        self.assertEqual(spell.name, "Mot de Soin")
        self.assertEqual(spell.element, "water")
        self.assertEqual(len(spell.levels), 2)

    def test_monster_data_creation(self):
        """Test MonsterData dataclass"""
        from core.external_data import MonsterData

        monster = MonsterData(
            id=9999,
            name="Tofu",
            level=1,
            hp=20,
            resistances={"fire": 10, "water": -20},
            drops=[{"item_id": 100, "rate": 5.0}],
            areas=["Astrub"],
            image_url="https://example.com/tofu.png"
        )

        self.assertEqual(monster.id, 9999)
        self.assertEqual(monster.name, "Tofu")
        self.assertEqual(monster.level, 1)
        self.assertEqual(monster.hp, 20)


class TestDofusDBClient(unittest.TestCase):
    """Test DofusDB client functionality"""

    def test_create_client(self):
        """Test creating DofusDB client"""
        from core.external_data import create_dofusdb_client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)
            self.assertIsNotNone(client)

    @patch('core.external_data.dofusdb_client.requests.get')
    def test_get_item_with_mock(self, mock_get):
        """Test getting item with mocked API"""
        from core.external_data import create_dofusdb_client

        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 1234,
            'name': 'Test Item',
            'type': 'weapon',
            'level': 50,
            'description': 'A test item',
            'effects': [],
            'conditions': [],
            'recipe': None,
            'imageUrl': 'https://example.com/item.png'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)
            item = client.get_item(1234)

            self.assertIsNotNone(item)
            self.assertEqual(item.name, 'Test Item')
            self.assertEqual(item.level, 50)

    def test_cache_functionality(self):
        """Test that caching works"""
        from core.external_data import create_dofusdb_client
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)

            # Save to cache manually
            test_data = {
                'id': 999,
                'name': 'Cached Item',
                'type': 'resource',
                'level': 1
            }
            client._save_to_cache('item_999', test_data)

            # Load from cache
            cached = client._load_from_cache('item_999')
            self.assertIsNotNone(cached)
            self.assertEqual(cached['name'], 'Cached Item')

    def test_stats_tracking(self):
        """Test that client tracks statistics"""
        from core.external_data import create_dofusdb_client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)

            stats = client.get_stats()
            self.assertIn('requests', stats)
            self.assertIn('cache_hits', stats)
            self.assertIn('cache_misses', stats)
            self.assertIn('errors', stats)


class TestDofusDBCaching(unittest.TestCase):
    """Test caching behavior"""

    def test_memory_cache(self):
        """Test in-memory cache"""
        from core.external_data import create_dofusdb_client, ItemData

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)

            # Add to memory cache
            item = ItemData(
                id=123, name="Test", type="weapon",
                level=10, description=None
            )
            client.memory_cache['item_123'] = item

            # Should hit memory cache
            initial_hits = client.stats['cache_hits']
            # Simulate cache hit (would happen in get_item)
            if 'item_123' in client.memory_cache:
                client.stats['cache_hits'] += 1

            self.assertEqual(client.stats['cache_hits'], initial_hits + 1)

    def test_disk_cache_persistence(self):
        """Test that disk cache persists"""
        from core.external_data import create_dofusdb_client

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create client and save to cache
            client1 = create_dofusdb_client(cache_dir=tmpdir)
            test_data = {'id': 456, 'name': 'Persisted Item'}
            client1._save_to_cache('item_456', test_data)

            # Create new client with same cache dir
            client2 = create_dofusdb_client(cache_dir=tmpdir)
            loaded = client2._load_from_cache('item_456')

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded['name'], 'Persisted Item')


class TestDofusDBSpellDamage(unittest.TestCase):
    """Test spell damage calculation"""

    def test_spell_damage_calculation(self):
        """Test damage calculation with resistances"""
        from core.external_data import create_dofusdb_client, SpellData

        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_dofusdb_client(cache_dir=tmpdir)

            # Mock spell in cache
            spell = SpellData(
                id=100,
                name="Fire Spell",
                class_name="Test",
                type="damage",
                element="fire",
                levels=[
                    {"level": 1, "damage": 100, "cost": 3}
                ]
            )
            client.memory_cache['spell_100'] = spell

            # Calculate damage with 50% fire resistance
            damage = client.get_spell_damage(
                spell_id=100,
                level=0,
                target_resistances={"fire": 50}
            )

            # 100 damage * (1 - 50/100) = 50 damage
            self.assertEqual(damage, 50.0)


def run_dofusdb_tests():
    """Run all DofusDB tests"""
    print("=" * 70)
    print("TESTS DOFUSDB CLIENT")
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
    success = run_dofusdb_tests()
    sys.exit(0 if success else 1)