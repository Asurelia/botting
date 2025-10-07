#!/bin/bash
#
# Script pour lancer tous les tests
# Usage: bash run_all_tests.sh
#

set -e

echo "🧪 LANCEMENT TESTS - Bot Dofus Autonome"
echo "======================================"
echo

# Check if venv activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  WARNING: venv non activé"
    echo "   Exécutez: source venv_rocm/bin/activate"
    echo
fi

# Check dependencies
echo "📦 Vérification dépendances..."
python3 -c "import pytest" 2>/dev/null && echo "   ✅ pytest" || echo "   ❌ pytest manquant"
python3 -c "import cv2" 2>/dev/null && echo "   ✅ opencv-python" || echo "   ❌ opencv-python manquant"
python3 -c "import numpy" 2>/dev/null && echo "   ✅ numpy" || echo "   ❌ numpy manquant"
python3 -c "import mss" 2>/dev/null && echo "   ✅ mss" || echo "   ❌ mss manquant"

echo

# Générer test data si nécessaire
if [ ! -d "tests/test_data/screenshots" ] || [ -z "$(ls -A tests/test_data/screenshots 2>/dev/null)" ]; then
    echo "📊 Génération test data..."
    python3 tests/setup_test_data.py || echo "⚠️  setup_test_data.py échoué (opencv manquant?)"
    echo
fi

# Tests GPU (si disponible)
echo "🎮 Tests GPU AMD..."
echo "-------------------"
pytest tests/test_gpu.py -v -m gpu --tb=short 2>&1 | head -50 || echo "⚠️  Tests GPU skipped/failed"
echo

# Tests Vision  
echo "👁️  Tests Vision..."
echo "-------------------"
pytest tests/test_vision.py -v -m vision --tb=short 2>&1 | head -50 || echo "⚠️  Tests Vision skipped/failed"
echo

# Tests Integration
echo "🔗 Tests Integration..."
echo "----------------------"
pytest tests/test_integration.py -v -m integration --tb=short 2>&1 | head -50 || echo "⚠️  Tests Integration skipped/failed"
echo

# Summary
echo "📊 RÉSUMÉ"
echo "=========="
echo
echo "Pour tests complets avec coverage:"
echo "  pytest tests/ -v --cov=. --cov-report=html"
echo
echo "Pour voir rapport coverage:"
echo "  firefox htmlcov/index.html"
echo
echo "✅ Script terminé"
