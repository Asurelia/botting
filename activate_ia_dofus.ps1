# Script d'activation IA DOFUS
Write-Host "🚀 Activation environnement IA DOFUS..." -ForegroundColor Green
conda activate ia-dofus

Write-Host "✅ Environnement activé !" -ForegroundColor Green
Write-Host "🎯 Commandes disponibles:" -ForegroundColor Yellow
Write-Host "  python launch_ai_dofus.py --mode demo" -ForegroundColor Cyan
Write-Host "  python scripts/test_amd_integration.py" -ForegroundColor Cyan
Write-Host "  python scripts/gemini_consensus.py autonomy_architecture" -ForegroundColor Cyan
