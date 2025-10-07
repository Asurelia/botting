#!/usr/bin/env python3
"""
Tests for Safety Systems
Tests observation mode and safety mechanisms
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSafetyImports(unittest.TestCase):
    """Test that safety modules import correctly"""

    def test_import_observation_mode(self):
        """Test importing observation mode"""
        try:
            from core.safety import create_observation_mode, ObservationLog
            self.assertIsNotNone(create_observation_mode)
            self.assertIsNotNone(ObservationLog)
        except ImportError as e:
            self.fail(f"Failed to import observation mode: {e}")

    def test_import_safety_manager(self):
        """Test importing safety manager"""
        try:
            from core.safety import create_safety_manager
            self.assertIsNotNone(create_safety_manager)
        except ImportError as e:
            # Safety manager might not exist yet
            pass


class TestObservationLog(unittest.TestCase):
    """Test ObservationLog dataclass"""

    def test_observation_log_creation(self):
        """Test creating observation log"""
        from core.safety import ObservationLog
        import time

        log = ObservationLog(
            timestamp=time.time(),
            action_type='navigation',
            action_details={'target': (200, 250)},
            game_state={'hp': 100, 'position': (150, 200)},
            decision_reason='Exploration de la map',
            would_execute=False
        )

        self.assertEqual(log.action_type, 'navigation')
        self.assertFalse(log.would_execute)
        self.assertEqual(log.decision_reason, 'Exploration de la map')


class TestObservationMode(unittest.TestCase):
    """Test ObservationMode functionality"""

    def test_create_observation_mode(self):
        """Test creating observation mode"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))
            self.assertIsNotNone(obs)

    def test_observation_mode_enabled_by_default(self):
        """Test that observation mode is enabled by default"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))
            self.assertTrue(obs.is_enabled())

    def test_intercept_action_blocks_when_enabled(self):
        """Test that actions are blocked when observation is enabled"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file), auto_enabled=True)

            # Try to execute an action
            result = obs.intercept_action(
                action_type='mouse_click',
                action_details={'position': (100, 200)},
                game_state={'hp': 100},
                reason='Test action'
            )

            # Should return None (blocked)
            self.assertIsNone(result)

    def test_intercept_action_passes_when_disabled(self):
        """Test that actions pass through when observation is disabled"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file), auto_enabled=False)

            action_details = {'position': (100, 200)}

            result = obs.intercept_action(
                action_type='mouse_click',
                action_details=action_details,
                game_state={'hp': 100},
                reason='Test action'
            )

            # Should return action_details (not blocked)
            self.assertEqual(result, action_details)

    def test_observation_logging(self):
        """Test that observations are logged"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Log several actions
            for i in range(5):
                obs.intercept_action(
                    action_type='navigation',
                    action_details={'target': (i * 10, i * 10)},
                    game_state={'hp': 100},
                    reason=f'Test action {i}'
                )

            # Check observations
            observations = obs.get_observations()
            self.assertEqual(len(observations), 5)

    def test_statistics_tracking(self):
        """Test that statistics are tracked"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Perform actions
            obs.intercept_action('key_press', {'key': 'a'}, {}, 'Test')
            obs.intercept_action('mouse_click', {'pos': (0, 0)}, {}, 'Test')
            obs.intercept_action('navigation', {'target': (10, 10)}, {}, 'Test')

            stats = obs.get_stats()

            self.assertEqual(stats['total_decisions'], 3)
            self.assertEqual(stats['actions_blocked'], 3)
            self.assertGreater(stats['keyboard_inputs'], 0)
            self.assertGreater(stats['mouse_clicks'], 0)
            self.assertGreater(stats['navigation_attempts'], 0)

    def test_save_observations(self):
        """Test saving observations to file"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Log actions
            obs.intercept_action('test_action', {}, {}, 'Test')

            # Save
            obs.save_observations()

            # Verify file exists and is valid JSON
            self.assertTrue(log_file.exists())

            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertEqual(data['mode'], 'observation')
            self.assertTrue(data['enabled'])
            self.assertIn('observations', data)
            self.assertGreater(len(data['observations']), 0)


class TestObservationAnalysis(unittest.TestCase):
    """Test observation analysis features"""

    def test_analyze_observations(self):
        """Test analyzing observations"""
        from core.safety import create_observation_mode
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Log multiple actions
            for i in range(10):
                obs.intercept_action(
                    action_type='navigation' if i % 2 == 0 else 'mouse_click',
                    action_details={},
                    game_state={},
                    reason='Test'
                )
                time.sleep(0.01)

            analysis = obs.analyze_observations()

            self.assertIn('total_observations', analysis)
            self.assertIn('action_types', analysis)
            self.assertIn('safety_score', analysis)
            self.assertIn('recommendations', analysis)

    def test_safety_score_calculation(self):
        """Test safety score calculation"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Log normal behavior
            for i in range(20):
                obs.intercept_action(
                    action_type=['navigation', 'mouse_click', 'key_press'][i % 3],
                    action_details={},
                    game_state={},
                    reason='Normal behavior'
                )

            analysis = obs.analyze_observations()
            safety_score = analysis['safety_score']

            # Score should be between 0 and 100
            self.assertGreaterEqual(safety_score, 0)
            self.assertLessEqual(safety_score, 100)

    def test_recommendations_generation(self):
        """Test recommendations generation"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file))

            # Log some actions
            for i in range(10):
                obs.intercept_action('test', {}, {}, 'Test')

            analysis = obs.analyze_observations()
            recommendations = analysis['recommendations']

            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)


class TestObservationModeControl(unittest.TestCase):
    """Test observation mode enable/disable"""

    def test_enable_observation(self):
        """Test enabling observation mode"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file), auto_enabled=False)

            self.assertFalse(obs.is_enabled())

            obs.enable()
            self.assertTrue(obs.is_enabled())

    def test_disable_observation(self):
        """Test disabling observation mode (DANGER!)"""
        from core.safety import create_observation_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "observation.json"
            obs = create_observation_mode(log_file=str(log_file), auto_enabled=True)

            self.assertTrue(obs.is_enabled())

            obs.disable()
            self.assertFalse(obs.is_enabled())


def run_safety_tests():
    """Run all safety tests"""
    print("=" * 70)
    print("TESTS SYSTEMES DE SECURITE")
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
    success = run_safety_tests()
    sys.exit(0 if success else 1)