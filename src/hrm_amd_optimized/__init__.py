"""
HRM AMD Optimized - Hierarchical Reasoning Model pour AMD 7800XT
Migration complète de sapientinc/HRM vers ROCm/HIP avec optimisations RDNA3

Version: 2.0.0
Auteur: Claude Code
Licence: MIT

Modules principaux:
- hrm_amd_core: Architecture HRM optimisée AMD
- dofus_integration: Intégration gaming DOFUS temps réel
- migration_plan: Plan de migration automatisé CUDA -> ROCm
"""

__version__ = "2.0.0"
__author__ = "Claude Code"

# Imports principaux
try:
    from .hrm_amd_core import (
        HRMAMDModel,
        AMDOptimizationConfig,
        AMDDeviceManager,
        HRMSystemOne,
        HRMSystemTwo,
        AMDOptimizedAttention,
        OptimizedRotaryEmbedding,
        AMDOptimizedMLP,
        HRMReasoningBlock,
        RMSNorm
    )

    from .dofus_integration import (
        DofusHRMIntegration,
        DofusGameState,
        DofusAction,
        DofusStateEncoder,
        DofusActionDecoder
    )

    from .migration_plan import (
        HRMMigrationPlan,
        MigrationStep,
        MigrationStatus
    )

    # Flag de disponibilité
    HRM_AMD_AVAILABLE = True

except ImportError as e:
    # Imports de fallback si dépendances manquantes
    import warnings
    warnings.warn(f"HRM AMD modules non disponibles: {e}")

    HRM_AMD_AVAILABLE = False

# Fonctions utilitaires publiques
def check_amd_compatibility():
    """Vérifie la compatibilité AMD GPU"""
    try:
        import torch

        # Vérification DirectML
        try:
            import torch_directml
            if torch_directml.is_available():
                return {
                    "compatible": True,
                    "backend": "DirectML",
                    "device": str(torch_directml.device())
                }
        except ImportError:
            pass

        # Vérification ROCm
        if torch.cuda.is_available():
            # Note: torch.cuda peut détecter les GPU AMD avec ROCm
            device_name = torch.cuda.get_device_name(0)
            if "AMD" in device_name or "Radeon" in device_name:
                return {
                    "compatible": True,
                    "backend": "ROCm",
                    "device": device_name
                }

        return {
            "compatible": False,
            "backend": "CPU",
            "device": "CPU fallback"
        }

    except Exception as e:
        return {
            "compatible": False,
            "backend": "Error",
            "device": f"Error: {e}"
        }

def get_recommended_config():
    """Retourne la configuration recommandée selon le hardware"""
    compatibility = check_amd_compatibility()

    if not HRM_AMD_AVAILABLE:
        return None

    if compatibility["compatible"]:
        if "7800XT" in compatibility["device"] or "RDNA3" in compatibility["device"]:
            # Configuration optimale 7800XT
            return AMDOptimizationConfig(
                compute_units=60,
                memory_bandwidth_gbps=624.0,
                vram_gb=16,
                use_rocwmma=True,
                use_mixed_precision=True,
                memory_fraction=0.9,
                preferred_dtype=torch.bfloat16
            )
        else:
            # Configuration générique AMD
            return AMDOptimizationConfig(
                use_mixed_precision=True,
                memory_fraction=0.8
            )
    else:
        # Configuration CPU fallback
        return AMDOptimizationConfig(
            use_rocwmma=False,
            use_mixed_precision=False,
            memory_fraction=0.6
        )

def create_optimized_model(config=None):
    """Crée un modèle HRM optimisé avec configuration automatique"""
    if not HRM_AMD_AVAILABLE:
        raise ImportError("HRM AMD modules non disponibles")

    if config is None:
        config = get_recommended_config()

    model = HRMAMDModel(config)
    return model.to_device()

def create_dofus_integration(model_path=None, config=None):
    """Crée une intégration DOFUS complète"""
    if not HRM_AMD_AVAILABLE:
        raise ImportError("HRM AMD modules non disponibles")

    if config is None:
        config = get_recommended_config()

    return DofusHRMIntegration(model_path=model_path, config=config)

# Informations de diagnostic
def get_system_info():
    """Retourne les informations système pour diagnostic"""
    import sys
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hrm_amd_available": HRM_AMD_AVAILABLE,
        "compatibility": check_amd_compatibility()
    }

    # Informations PyTorch
    try:
        import torch
        info["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "hip_version": getattr(torch.version, 'hip', None)
        }
    except ImportError:
        info["pytorch"] = "Non disponible"

    # Informations DirectML
    try:
        import torch_directml
        info["directml"] = {
            "available": torch_directml.is_available(),
            "device_count": torch_directml.device_count() if torch_directml.is_available() else 0
        }
    except ImportError:
        info["directml"] = "Non disponible"

    return info

# Exports publics
__all__ = [
    # Classes principales
    "HRMAMDModel",
    "AMDOptimizationConfig",
    "AMDDeviceManager",
    "DofusHRMIntegration",
    "DofusGameState",
    "DofusAction",
    "HRMMigrationPlan",

    # Fonctions utilitaires
    "check_amd_compatibility",
    "get_recommended_config",
    "create_optimized_model",
    "create_dofus_integration",
    "get_system_info",

    # Flags et métadonnées
    "HRM_AMD_AVAILABLE",
    "__version__",
    "__author__"
]

# Message de démarrage
if __name__ != "__main__":
    import logging
    logger = logging.getLogger(__name__)

    if HRM_AMD_AVAILABLE:
        compatibility = check_amd_compatibility()
        if compatibility["compatible"]:
            logger.info(f"HRM AMD Optimized v{__version__} chargé - Device: {compatibility['device']}")
        else:
            logger.warning(f"HRM AMD Optimized v{__version__} - Mode CPU fallback")
    else:
        logger.error("HRM AMD Optimized - Modules non disponibles, vérifier les dépendances")