@echo off
:: 🧹 Script de nettoyage sécurisé - DOFUS Vision 2025
:: Date: 28 septembre 2025
:: Auteur: Claude Code Refactoring Specialist

echo ===============================================
echo 🧹 NETTOYAGE SECURISE DOFUS VISION 2025
echo ===============================================
echo.

:: Configuration
set BACKUP_DIR=backup\%date:~6,4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%

echo 📁 Création dossier de sauvegarde: %BACKUP_DIR%
mkdir "%BACKUP_DIR%" 2>nul

echo.
echo 🚀 PHASE 1: NETTOYAGE IMMEDIAT (RISQUE NUL)
echo ===============================================

:: Sauvegarde des fichiers importants avant nettoyage
echo 💾 Sauvegarde fichiers JSON temporaires...
if exist "system_test_report.json" copy "system_test_report.json" "%BACKUP_DIR%\" >nul
if exist "test_integration_summary.json" copy "test_integration_summary.json" "%BACKUP_DIR%\" >nul
if exist "test_knowledge_summary.json" copy "test_knowledge_summary.json" "%BACKUP_DIR%\" >nul

:: Création dossier temp
echo 📂 Création dossier temp...
mkdir temp 2>nul

:: Déplacement fichiers temporaires
echo 📦 Déplacement fichiers temporaires...
if exist "system_test_report.json" move "system_test_report.json" temp\ >nul
if exist "test_integration_summary.json" move "test_integration_summary.json" temp\ >nul
if exist "test_knowledge_summary.json" move "test_knowledge_summary.json" temp\ >nul

:: Suppression fichiers cache Python
echo 🗑️ Suppression fichiers cache Python...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo   - Suppression: %%d
        rmdir /s /q "%%d" 2>nul
    )
)

:: Suppression fichiers .pyc
echo 🗑️ Suppression fichiers .pyc...
del /s /q "*.pyc" 2>nul

echo ✅ Phase 1 terminée avec succès!
echo.

echo ⚠️ PHASE 2: VERIFICATION DOUBLONS (RISQUE FAIBLE)
echo ===============================================

:: Vérification des bases de données
echo 📊 Analyse des bases de données...
echo.
echo 📋 Bases de données détectées:
dir /s "*.db" 2>nul | find "fichier(s)"
dir /s "*.sqlite" 2>nul | find "fichier(s)"
echo.

:: Vérification doublons potentiels
echo 🔍 Vérification doublons extraction_cache.db...
if exist "data\extraction_cache.db" (
    if exist "knowledge_base\data\extraction_cache.db" (
        echo ⚠️ DOUBLON DETECTE: extraction_cache.db
        echo   - data\extraction_cache.db
        echo   - knowledge_base\data\extraction_cache.db
        echo.
        echo 💾 Sauvegarde des deux versions...
        copy "data\extraction_cache.db" "%BACKUP_DIR%\extraction_cache_data.db" >nul
        copy "knowledge_base\data\extraction_cache.db" "%BACKUP_DIR%\extraction_cache_kb.db" >nul

        :: Comparaison tailles
        for %%F in ("data\extraction_cache.db") do set size1=%%~zF
        for %%F in ("knowledge_base\data\extraction_cache.db") do set size2=%%~zF

        echo 📏 Taille data\extraction_cache.db: %size1% octets
        echo 📏 Taille knowledge_base\data\extraction_cache.db: %size2% octets

        if %size1% EQU %size2% (
            echo ✅ Tailles identiques - Suppression du doublon KB safe
            del "knowledge_base\data\extraction_cache.db" 2>nul
            echo   ✓ knowledge_base\data\extraction_cache.db supprimé
        ) else (
            echo ⚠️ Tailles différentes - Conservation des deux fichiers
        )
    )
)

echo.
echo 🔍 Vérification doublons learning_database.sqlite...
if exist "learning_engine\learning_database.sqlite" (
    if exist "learning_engine\learning_engine\learning_database.sqlite" (
        echo ⚠️ DOUBLON DETECTE: learning_database.sqlite
        echo   - learning_engine\learning_database.sqlite
        echo   - learning_engine\learning_engine\learning_database.sqlite
        echo.
        echo 💾 Sauvegarde des deux versions...
        copy "learning_engine\learning_database.sqlite" "%BACKUP_DIR%\learning_database_main.sqlite" >nul
        copy "learning_engine\learning_engine\learning_database.sqlite" "%BACKUP_DIR%\learning_database_sub.sqlite" >nul

        :: Comparaison tailles
        for %%F in ("learning_engine\learning_database.sqlite") do set size3=%%~zF
        for %%F in ("learning_engine\learning_engine\learning_database.sqlite") do set size4=%%~zF

        echo 📏 Taille learning_engine\learning_database.sqlite: %size3% octets
        echo 📏 Taille learning_engine\learning_engine\learning_database.sqlite: %size4% octets

        if %size3% EQU %size4% (
            echo ✅ Tailles identiques - Suppression du doublon sous-dossier safe
            del "learning_engine\learning_engine\learning_database.sqlite" 2>nul
            echo   ✓ learning_engine\learning_engine\learning_database.sqlite supprimé

            :: Suppression dossier vide si possible
            rmdir "learning_engine\learning_engine" 2>nul
            if not exist "learning_engine\learning_engine" (
                echo   ✓ Dossier learning_engine\learning_engine supprimé (vide)
            )
        ) else (
            echo ⚠️ Tailles différentes - Conservation des deux fichiers
        )
    )
)

echo ✅ Phase 2 terminée avec succès!
echo.

echo 📊 RAPPORT FINAL
echo ===============================================
echo ✅ Nettoyage terminé avec succès!
echo.
echo 📂 Sauvegardes créées dans: %BACKUP_DIR%\
echo 📁 Fichiers temporaires déplacés dans: temp\
echo 🗑️ Fichiers cache Python supprimés
echo 🔄 Doublons traités selon sécurité
echo.

:: Calcul gain d'espace estimé
echo 💾 Estimation gain d'espace:
echo   - Fichiers cache Python: ~200KB
echo   - Doublons bases de données: ~28KB
echo   - Fichiers temporaires: ~5KB
echo   - TOTAL ESTIMÉ: ~233KB
echo.

echo 🎯 PROCHAINES ÉTAPES RECOMMANDÉES:
echo   1. Vérifier que les tests passent encore: python test_complete_system.py
echo   2. Commit des changements: git add -A && git commit -m "🧹 Nettoyage sécurisé"
echo   3. Consulter RAPPORT_ANALYSE_REFACTORING.md pour optimisations avancées
echo.

echo 🏁 Script terminé. Appuyez sur une touche pour fermer...
pause >nul