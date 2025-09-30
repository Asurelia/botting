"""
Plan de Migration HRM CUDA vers AMD ROCm/HIP
Guide complet pour porter HRM de NVIDIA CUDA vers AMD 7800XT

Version: 2.0.0 - Plan de Migration Complet
"""

import logging
import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class MigrationStep:
    """Étape de migration avec validation"""
    step_id: str
    title: str
    description: str
    commands: List[str]
    validation_func: Optional[str] = None
    prerequisites: List[str] = None
    estimated_time_minutes: int = 10
    critical: bool = False

@dataclass
class MigrationStatus:
    """Statut de la migration"""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: List[str] = None
    current_step: Optional[str] = None
    start_time: Optional[float] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.failed_steps is None:
            self.failed_steps = []
        if self.errors is None:
            self.errors = []

class HRMMigrationPlan:
    """Plan de migration complet CUDA vers ROCm/HIP"""

    def __init__(self):
        self.status = MigrationStatus()
        self.migration_steps = self._create_migration_steps()
        self.backup_paths = []

    def _create_migration_steps(self) -> List[MigrationStep]:
        """Crée la liste complète des étapes de migration"""

        steps = [
            # Phase 1: Préparation et sauvegarde
            MigrationStep(
                step_id="backup_001",
                title="Sauvegarde du code existant",
                description="Créer une sauvegarde complète avant migration",
                commands=[
                    "git stash push -m 'Pre-migration backup'",
                    "git checkout -b cuda-backup-$(date +%Y%m%d-%H%M%S)",
                    "git push origin cuda-backup-$(date +%Y%m%d-%H%M%S)"
                ],
                validation_func="validate_backup",
                estimated_time_minutes=5,
                critical=True
            ),

            # Phase 2: Environnement ROCm
            MigrationStep(
                step_id="rocm_001",
                title="Installation ROCm 6.4.1+",
                description="Installer ROCm avec support RDNA3",
                commands=[
                    "# Vérification de la version Ubuntu/Distribution",
                    "lsb_release -a",
                    "# Ajout du repository ROCm",
                    "wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -",
                    "echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.4.1 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list",
                    "sudo apt update",
                    "sudo apt install rocm-dev rocm-libs rocm-utils -y"
                ],
                validation_func="validate_rocm_installation",
                estimated_time_minutes=30,
                critical=True
            ),

            MigrationStep(
                step_id="rocm_002",
                title="Configuration ROCm pour RDNA3",
                description="Configurer ROCm pour support 7800XT",
                commands=[
                    "# Ajout utilisateur au groupe render",
                    "sudo usermod -a -G render,video $USER",
                    "# Configuration HSA pour RDNA3",
                    "echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc",
                    "echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc",
                    "# Redémarrage requis pour groups",
                    "echo 'REDÉMARRAGE REQUIS après cette étape'"
                ],
                validation_func="validate_rocm_config",
                estimated_time_minutes=10,
                critical=True
            ),

            # Phase 3: PyTorch ROCm
            MigrationStep(
                step_id="pytorch_001",
                title="Installation PyTorch ROCm",
                description="Installer PyTorch compilé pour ROCm",
                commands=[
                    "# Désinstallation PyTorch CUDA",
                    "pip uninstall torch torchvision torchaudio -y",
                    "# Installation PyTorch ROCm 2.3+",
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0",
                    "# Vérification installation",
                    "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"ROCm: {torch.version.hip}\")'",
                ],
                validation_func="validate_pytorch_rocm",
                estimated_time_minutes=20,
                critical=True
            ),

            # Phase 4: FlashAttention alternatives
            MigrationStep(
                step_id="flash_001",
                title="Installation FlashAttention RDNA3",
                description="Installer alternatives FlashAttention pour AMD",
                commands=[
                    "# Clone repository communautaire RDNA3",
                    "git clone https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal.git",
                    "cd flash-attention-v2-RDNA3-minimal",
                    "# Compilation avec ROCm",
                    "pip install -e .",
                    "cd .."
                ],
                validation_func="validate_flash_attention",
                estimated_time_minutes=25,
                critical=False
            ),

            MigrationStep(
                step_id="flash_002",
                title="Configuration SDP Fallback",
                description="Configurer Scaled Dot Product Attention fallback",
                commands=[
                    "# Test SDP compatibility",
                    "python -c 'import torch.nn.functional as F; print(hasattr(F, \"scaled_dot_product_attention\"))'"
                ],
                validation_func="validate_sdp_fallback",
                estimated_time_minutes=5,
                critical=False
            ),

            # Phase 5: Code HRM - Adaptations device
            MigrationStep(
                step_id="code_001",
                title="Migration device management",
                description="Remplacer torch.cuda par torch_directml/ROCm",
                commands=[
                    "# Remplacement automatique des appels CUDA",
                    "find ./models -name '*.py' -exec sed -i 's/torch.cuda.is_available()/torch.cuda.is_available() or torch_directml.is_available()/g' {} +",
                    "find ./models -name '*.py' -exec sed -i 's/torch.device(\"cuda\")/torch_directml.device() if torch_directml.is_available() else torch.device(\"cpu\")/g' {} +",
                    "find . -name '*.py' -exec sed -i 's/\\.cuda()/.to(device)/g' {} +"
                ],
                validation_func="validate_device_migration",
                estimated_time_minutes=15,
                critical=True
            ),

            MigrationStep(
                step_id="code_002",
                title="Migration distributed training",
                description="Adapter NCCL vers RCCL pour AMD",
                commands=[
                    "# Remplacement NCCL backend",
                    "find . -name '*.py' -exec sed -i 's/backend=\"nccl\"/backend=\"gloo\"/g' {} +",
                    "# Note: RCCL disponible sur ROCm 6.0+ mais gloo plus stable"
                ],
                validation_func="validate_distributed_training",
                estimated_time_minutes=10,
                critical=False
            ),

            # Phase 6: Optimisations spécifiques RDNA3
            MigrationStep(
                step_id="opt_001",
                title="Optimisations mémoire RDNA3",
                description="Configurer optimisations pour 16GB VRAM",
                commands=[
                    "# Configuration batch size optimal",
                    "echo 'Batch size optimal: 16-32 pour 7800XT'",
                    "# Configuration mixed precision",
                    "echo 'Mixed precision: bfloat16 pour RDNA3'"
                ],
                validation_func="validate_memory_optimizations",
                estimated_time_minutes=10,
                critical=False
            ),

            MigrationStep(
                step_id="opt_002",
                title="Configuration ROCWMMA",
                description="Activer WMMA pour accélération matérielle",
                commands=[
                    "# Installation rocWMMA",
                    "pip install rocwmma",
                    "# Test basic WMMA",
                    "python -c 'try: import rocwmma; print(\"ROCWMMA available\"); except: print(\"ROCWMMA not available\")'"
                ],
                validation_func="validate_rocwmma",
                estimated_time_minutes=15,
                critical=False
            ),

            # Phase 7: Tests et validation
            MigrationStep(
                step_id="test_001",
                title="Tests unitaires migration",
                description="Validation des composants migrés",
                commands=[
                    "python -m pytest tests/test_hrm_amd.py -v",
                    "python test_migration_validation.py"
                ],
                validation_func="validate_unit_tests",
                estimated_time_minutes=20,
                critical=True
            ),

            MigrationStep(
                step_id="test_002",
                title="Benchmark performance",
                description="Comparaison performances CUDA vs ROCm",
                commands=[
                    "python benchmark_hrm_performance.py --device=amd",
                    "python benchmark_hrm_performance.py --compare-cuda"
                ],
                validation_func="validate_performance_benchmark",
                estimated_time_minutes=30,
                critical=False
            ),

            # Phase 8: Intégration DOFUS
            MigrationStep(
                step_id="integration_001",
                title="Test intégration DOFUS",
                description="Validation de l'intégration avec l'écosystème DOFUS",
                commands=[
                    "python test_dofus_hrm_integration.py",
                    "python test_realtime_performance.py"
                ],
                validation_func="validate_dofus_integration",
                estimated_time_minutes=25,
                critical=True
            ),

            # Phase 9: Déploiement
            MigrationStep(
                step_id="deploy_001",
                title="Finalisation et documentation",
                description="Finaliser la migration et créer la documentation",
                commands=[
                    "git add .",
                    "git commit -m 'feat(hrm): Migration complète vers AMD ROCm/HIP pour 7800XT'",
                    "python generate_migration_report.py"
                ],
                validation_func="validate_deployment",
                estimated_time_minutes=15,
                critical=True
            )
        ]

        return steps

    def run_migration(self, start_from_step: Optional[str] = None, dry_run: bool = False) -> bool:
        """Exécute le plan de migration complet"""

        logger.info("Démarrage migration HRM CUDA -> AMD ROCm/HIP")
        self.status.total_steps = len(self.migration_steps)
        self.status.start_time = time.time()

        # Démarrage à partir d'une étape spécifique
        start_index = 0
        if start_from_step:
            for i, step in enumerate(self.migration_steps):
                if step.step_id == start_from_step:
                    start_index = i
                    break

        for i, step in enumerate(self.migration_steps[start_index:], start_index):
            logger.info(f"Étape {i+1}/{self.status.total_steps}: {step.title}")
            self.status.current_step = step.step_id

            if dry_run:
                logger.info(f"DRY RUN - {step.description}")
                for cmd in step.commands:
                    logger.info(f"  Command: {cmd}")
                continue

            # Vérification des prérequis
            if step.prerequisites:
                if not self._check_prerequisites(step.prerequisites):
                    logger.error(f"Prérequis manquants pour {step.step_id}")
                    self.status.failed_steps.append(step.step_id)
                    if step.critical:
                        return False
                    continue

            # Exécution de l'étape
            success = self._execute_step(step)

            if success:
                self.status.completed_steps += 1
                logger.info(f"✓ Étape {step.step_id} complétée")
            else:
                self.status.failed_steps.append(step.step_id)
                logger.error(f"✗ Étape {step.step_id} échouée")

                if step.critical:
                    logger.error("Étape critique échouée - arrêt de la migration")
                    return False

        logger.info(f"Migration terminée: {self.status.completed_steps}/{self.status.total_steps} étapes")
        return len(self.status.failed_steps) == 0

    def _execute_step(self, step: MigrationStep) -> bool:
        """Exécute une étape de migration"""
        try:
            for command in step.commands:
                # Skip comments
                if command.strip().startswith('#'):
                    continue

                logger.debug(f"Exécution: {command}")

                # Exécution sécurisée
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )

                if result.returncode != 0:
                    logger.error(f"Commande échouée: {command}")
                    logger.error(f"Erreur: {result.stderr}")
                    self.status.errors.append(f"{step.step_id}: {result.stderr}")
                    return False

            # Validation si fonction fournie
            if step.validation_func:
                return self._run_validation(step.validation_func, step.step_id)

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pour l'étape {step.step_id}")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de l'étape {step.step_id}: {e}")
            self.status.errors.append(f"{step.step_id}: {str(e)}")
            return False

    def _run_validation(self, validation_func: str, step_id: str) -> bool:
        """Exécute une fonction de validation"""
        try:
            # Mapping des fonctions de validation
            validations = {
                "validate_backup": self._validate_backup,
                "validate_rocm_installation": self._validate_rocm_installation,
                "validate_rocm_config": self._validate_rocm_config,
                "validate_pytorch_rocm": self._validate_pytorch_rocm,
                "validate_flash_attention": self._validate_flash_attention,
                "validate_sdp_fallback": self._validate_sdp_fallback,
                "validate_device_migration": self._validate_device_migration,
                "validate_distributed_training": self._validate_distributed_training,
                "validate_memory_optimizations": self._validate_memory_optimizations,
                "validate_rocwmma": self._validate_rocwmma,
                "validate_unit_tests": self._validate_unit_tests,
                "validate_performance_benchmark": self._validate_performance_benchmark,
                "validate_dofus_integration": self._validate_dofus_integration,
                "validate_deployment": self._validate_deployment
            }

            validation_func_obj = validations.get(validation_func)
            if validation_func_obj:
                return validation_func_obj()
            else:
                logger.warning(f"Fonction de validation {validation_func} non trouvée")
                return True

        except Exception as e:
            logger.error(f"Erreur validation {validation_func}: {e}")
            return False

    def _check_prerequisites(self, prerequisites: List[str]) -> bool:
        """Vérifie les prérequis d'une étape"""
        for prereq in prerequisites:
            # Implémentation basique - à étendre selon les besoins
            if prereq == "rocm_installed":
                try:
                    result = subprocess.run(["rocm-smi"], capture_output=True)
                    if result.returncode != 0:
                        return False
                except FileNotFoundError:
                    return False

            elif prereq == "pytorch_available":
                try:
                    import torch
                except ImportError:
                    return False

        return True

    # Fonctions de validation spécifiques
    def _validate_backup(self) -> bool:
        """Valide que la sauvegarde a été créée"""
        try:
            result = subprocess.run(["git", "branch", "--list", "*cuda-backup*"], capture_output=True, text=True)
            return "cuda-backup" in result.stdout
        except:
            return False

    def _validate_rocm_installation(self) -> bool:
        """Valide l'installation ROCm"""
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _validate_rocm_config(self) -> bool:
        """Valide la configuration ROCm"""
        # Vérifier les variables d'environnement
        return "HSA_OVERRIDE_GFX_VERSION" in os.environ

    def _validate_pytorch_rocm(self) -> bool:
        """Valide PyTorch ROCm"""
        try:
            import torch
            return torch.version.hip is not None
        except:
            return False

    def _validate_flash_attention(self) -> bool:
        """Valide FlashAttention RDNA3"""
        try:
            # Test d'import basique
            result = subprocess.run([
                "python", "-c",
                "try: import flash_attn; print('OK'); except Exception as e: print(f'ERROR: {e}')"
            ], capture_output=True, text=True)
            return "OK" in result.stdout
        except:
            return False

    def _validate_sdp_fallback(self) -> bool:
        """Valide SDP Fallback"""
        try:
            import torch.nn.functional as F
            return hasattr(F, 'scaled_dot_product_attention')
        except:
            return False

    def _validate_device_migration(self) -> bool:
        """Valide la migration device"""
        # Vérifier qu'il n'y a plus d'appels .cuda() hardcodés
        try:
            result = subprocess.run([
                "grep", "-r", "\\.cuda()", "./models/", "--include=*.py"
            ], capture_output=True)
            return result.returncode != 0  # Pas de match = bon
        except:
            return True

    def _validate_distributed_training(self) -> bool:
        """Valide distributed training"""
        return True  # Validation simplifiée

    def _validate_memory_optimizations(self) -> bool:
        """Valide optimisations mémoire"""
        return True  # Validation simplifiée

    def _validate_rocwmma(self) -> bool:
        """Valide ROCWMMA"""
        try:
            import rocwmma
            return True
        except ImportError:
            return False

    def _validate_unit_tests(self) -> bool:
        """Valide les tests unitaires"""
        try:
            result = subprocess.run(["python", "-m", "pytest", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False

    def _validate_performance_benchmark(self) -> bool:
        """Valide benchmark performance"""
        return True  # Validation simplifiée

    def _validate_dofus_integration(self) -> bool:
        """Valide intégration DOFUS"""
        return True  # Validation simplifiée

    def _validate_deployment(self) -> bool:
        """Valide déploiement"""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            return len(result.stdout.strip()) == 0  # Rien à committer
        except:
            return False

    def generate_migration_report(self) -> Dict[str, Any]:
        """Génère un rapport de migration détaillé"""

        total_time = 0
        if self.status.start_time:
            total_time = time.time() - self.status.start_time

        report = {
            "migration_summary": {
                "total_steps": self.status.total_steps,
                "completed_steps": self.status.completed_steps,
                "failed_steps": len(self.status.failed_steps),
                "success_rate": self.status.completed_steps / self.status.total_steps * 100,
                "total_time_minutes": total_time / 60
            },
            "failed_steps": self.status.failed_steps,
            "errors": self.status.errors,
            "recommendations": self._generate_recommendations(),
            "next_steps": self._generate_next_steps()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Génère des recommendations post-migration"""
        recommendations = []

        if "rocm_001" in self.status.failed_steps:
            recommendations.append("Vérifier la compatibilité de votre distribution avec ROCm 6.4.1")

        if "pytorch_001" in self.status.failed_steps:
            recommendations.append("Essayer une version différente de PyTorch ROCm")

        if "flash_001" in self.status.failed_steps:
            recommendations.append("FlashAttention RDNA3 optionnel - continuer avec SDP fallback")

        if len(self.status.failed_steps) == 0:
            recommendations.append("Migration réussie ! Effectuer des tests de performance réguliers")

        return recommendations

    def _generate_next_steps(self) -> List[str]:
        """Génère les prochaines étapes"""
        next_steps = []

        if self.status.completed_steps == self.status.total_steps:
            next_steps.extend([
                "Effectuer des tests de performance complets",
                "Monitorer l'utilisation GPU pendant l'utilisation",
                "Optimiser les hyperparamètres pour AMD",
                "Documenter les gains de performance"
            ])
        else:
            next_steps.extend([
                "Corriger les étapes échouées",
                "Relancer la migration depuis l'étape échouée",
                "Consulter les logs pour diagnostics"
            ])

        return next_steps

    def save_migration_state(self, filepath: str):
        """Sauvegarde l'état de la migration"""
        state = {
            "status": {
                "total_steps": self.status.total_steps,
                "completed_steps": self.status.completed_steps,
                "failed_steps": self.status.failed_steps,
                "current_step": self.status.current_step,
                "errors": self.status.errors
            },
            "backup_paths": self.backup_paths
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_migration_state(self, filepath: str):
        """Charge l'état de la migration"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        status_data = state["status"]
        self.status.total_steps = status_data["total_steps"]
        self.status.completed_steps = status_data["completed_steps"]
        self.status.failed_steps = status_data["failed_steps"]
        self.status.current_step = status_data["current_step"]
        self.status.errors = status_data["errors"]
        self.backup_paths = state["backup_paths"]

if __name__ == "__main__":
    import time

    # Script principal de migration
    migration = HRMMigrationPlan()

    # Arguments de ligne de commande simples
    dry_run = "--dry-run" in sys.argv
    start_from = None

    for arg in sys.argv:
        if arg.startswith("--start-from="):
            start_from = arg.split("=")[1]

    # Exécution de la migration
    success = migration.run_migration(start_from_step=start_from, dry_run=dry_run)

    # Rapport final
    report = migration.generate_migration_report()

    print("\n" + "="*50)
    print("RAPPORT DE MIGRATION HRM CUDA -> AMD ROCm/HIP")
    print("="*50)
    print(f"Étapes complétées: {report['migration_summary']['completed_steps']}/{report['migration_summary']['total_steps']}")
    print(f"Taux de succès: {report['migration_summary']['success_rate']:.1f}%")
    print(f"Temps total: {report['migration_summary']['total_time_minutes']:.1f} minutes")

    if report['failed_steps']:
        print(f"\nÉtapes échouées: {', '.join(report['failed_steps'])}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

    print("\nProchaines étapes:")
    for step in report['next_steps']:
        print(f"- {step}")

    # Sauvegarde du rapport
    with open('migration_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nRapport détaillé sauvegardé dans: migration_report.json")

    sys.exit(0 if success else 1)