#!/usr/bin/env python3
"""
Verify Migration - Check all migrated files are importable
Quick validation script to check import resolution without running full code
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_path: str, description: str) -> bool:
    """Test if a module can be imported"""
    try:
        # For py files, convert path to module import
        if module_path.endswith('.py'):
            # Remove .py and convert / to .
            module_path = module_path[:-3].replace('/', '.')

        __import__(module_path)
        print(f"✅ {description}: {module_path}")
        return True
    except ModuleNotFoundError as e:
        # Check if error is about the target module or a dependency
        error_msg = str(e)
        if "No module named 'cv2'" in error_msg or "No module named 'pytest'" in error_msg or "No module named 'torch'" in error_msg:
            # Missing dependency - import path is correct
            print(f"✅ {description}: {module_path} (dependencies needed)")
            return True
        else:
            # Module itself not found - import path wrong
            print(f"❌ {description}: {module_path} - {e}")
            return False
    except ImportError as e:
        print(f"❌ {description}: {module_path} - {e}")
        return False
    except Exception as e:
        # Other errors (like syntax errors) should still be reported
        print(f"⚠️  {description}: {module_path} - {e.__class__.__name__}")
        return False

def main():
    """Run all import tests"""
    print("=" * 60)
    print("MIGRATION VERIFICATION - Import Resolution Tests")
    print("=" * 60)
    print()

    tests = [
        # Core adapters
        ("core.platform_adapter", "Platform Adapter"),
        ("core.vision_capture_adapter", "Vision Capture Adapter"),
        ("core.vision.screenshot_capture_unified", "Screenshot Capture Unified"),

        # Tools
        ("tools.session_recorder", "Session Recorder"),
        ("tools.annotation_tool", "Annotation Tool"),

        # Tests
        ("tests.conftest", "Pytest Config"),
        ("tests.test_gpu", "GPU Tests"),
        ("tests.test_vision", "Vision Tests"),
        ("tests.test_integration", "Integration Tests"),
    ]

    results = []
    for module_path, description in tests:
        results.append(test_import(module_path, description))

    print()
    print("=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} imports verified")
    print("=" * 60)

    if all(results):
        print("\n🎉 All imports verified successfully!")
        print("Migration structure is correct.")
        return 0
    else:
        print("\n⚠️  Some imports failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
