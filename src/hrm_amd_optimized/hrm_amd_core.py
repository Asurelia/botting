"""
HRM AMD Optimized Core - Implémentation HRM pour AMD 7800XT avec ROCm/HIP
Architecture hiérarchique adaptée pour RDNA3 avec optimisations spécifiques

Basé sur: sapientinc/HRM avec adaptations pour AMD GPU
Version: 2.0.0 - Optimisé RDNA3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time

# Imports spécifiques AMD
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

try:
    # ROCm WMMA pour RDNA3
    import rocwmma
    HAS_ROCWMMA = True
except ImportError:
    HAS_ROCWMMA = False

logger = logging.getLogger(__name__)

@dataclass
class AMDOptimizationConfig:
    """Configuration d'optimisation pour AMD 7800XT"""
    # Paramètres GPU RDNA3
    compute_units: int = 60  # 7800XT a 60 CU
    memory_bandwidth_gbps: float = 624.0  # 624 GB/s
    vram_gb: int = 16

    # Optimisations spécifiques
    use_rocwmma: bool = True
    use_mixed_precision: bool = True
    enable_memory_optimization: bool = True
    batch_size_optimization: bool = True

    # Paramètres de performance
    preferred_dtype: torch.dtype = torch.bfloat16
    memory_fraction: float = 0.9  # Utiliser 90% de la VRAM
    optimal_sequence_length: int = 2048

    # FlashAttention alternatives
    use_sdp_fallback: bool = True  # Scaled Dot Product Attention fallback
    enable_gradient_checkpointing: bool = True

class AMDDeviceManager:
    """Gestionnaire de device optimisé pour AMD GPU"""

    def __init__(self, config: AMDOptimizationConfig):
        self.config = config
        self.device = self._setup_device()
        self.device_props = self._get_device_properties()

    def _setup_device(self) -> torch.device:
        """Configure le device optimal pour AMD"""
        if HAS_DIRECTML:
            try:
                device = torch_directml.device()
                logger.info(f"DirectML device configuré: {device}")
                return device
            except Exception as e:
                logger.warning(f"Échec DirectML: {e}")

        # Fallback vers CPU
        device = torch.device("cpu")
        logger.warning("Utilisation du CPU - performance dégradée")
        return device

    def _get_device_properties(self) -> Dict[str, Any]:
        """Récupère les propriétés du device"""
        props = {
            "name": "AMD Radeon RX 7800 XT",
            "memory_gb": self.config.vram_gb,
            "compute_units": self.config.compute_units,
            "has_directml": HAS_DIRECTML,
            "has_rocwmma": HAS_ROCWMMA,
            "device_type": str(self.device)
        }

        if self.device.type != "cpu":
            try:
                # Informations supplémentaires si GPU disponible
                props["memory_allocated"] = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
                props["memory_cached"] = torch.cuda.memory_reserved(self.device) if torch.cuda.is_available() else 0
            except:
                pass

        return props

class OptimizedRotaryEmbedding(nn.Module):
    """Rotary Embedding optimisé pour RDNA3"""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pré-calcul optimisé pour AMD
        self.register_buffer("inv_freq", self._compute_inv_freq())

    def _compute_inv_freq(self) -> torch.Tensor:
        """Calcul optimisé des fréquences inverses"""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return inv_freq

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward optimisé pour RDNA3"""
        if seq_len is None:
            seq_len = x.shape[-2]

        # Optimisation: utiliser bfloat16 pour réduire la bande passante mémoire
        dtype_orig = x.dtype
        if self.training and dtype_orig == torch.float32:
            x = x.to(torch.bfloat16)

        # Génération des embeddings avec optimisation mémoire
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Optimisation: éviter les copies inutiles
        cos = freqs.cos().to(x.dtype)
        sin = freqs.sin().to(x.dtype)

        return cos, sin

class AMDOptimizedAttention(nn.Module):
    """Attention optimisée pour AMD avec alternatives FlashAttention"""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 head_dim: Optional[int] = None,
                 config: Optional[AMDOptimizationConfig] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.config = config or AMDOptimizationConfig()

        # Projections linéaires optimisées
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = OptimizedRotaryEmbedding(self.head_dim)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _apply_rotary_pos_emb(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Application optimisée des embeddings rotatoires"""
        # Split en parties réelle et imaginaire
        x1, x2 = tensor.chunk(2, dim=-1)

        # Rotation optimisée pour RDNA3
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.shape

        # Projections avec optimisation mémoire
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Reshape pour multi-head
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Application des embeddings rotatoires
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states = self._apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self._apply_rotary_pos_emb(key_states, cos, sin)

        # Gestion du cache KV
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Attention optimisée pour AMD
        attn_output = self._compute_attention(query_states, key_states, value_states, attention_mask)

        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _compute_attention(self,
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calcul d'attention optimisé pour RDNA3"""

        # Utilisation de Scaled Dot Product Attention si disponible (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention') and self.config.use_sdp_fallback:
            try:
                return F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=True if attention_mask is None else False
                )
            except Exception as e:
                logger.warning(f"SDP Attention failed, fallback to manual: {e}")

        # Implémentation manuelle optimisée
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Utilisation de bfloat16 pour économiser la bande passante
        if self.config.use_mixed_precision:
            attn_weights = attn_weights.to(torch.bfloat16)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output

class AMDOptimizedMLP(nn.Module):
    """MLP optimisé pour RDNA3 avec SwiGLU"""

    def __init__(self, hidden_size: int, intermediate_size: int, config: Optional[AMDOptimizationConfig] = None):
        super().__init__()

        self.config = config or AMDOptimizationConfig()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU implementation optimisée
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward optimisé avec SwiGLU"""
        # Optimisation: calculs parallèles pour RDNA3
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            gate = self.gate_proj(x)
            up = self.up_proj(x)

            # SwiGLU activation
            intermediate = F.silu(gate) * up

            # Projection finale
            output = self.down_proj(intermediate)

        return output

class HRMReasoningBlock(nn.Module):
    """Bloc de raisonnement HRM optimisé pour AMD"""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 config: Optional[AMDOptimizationConfig] = None):
        super().__init__()

        self.config = config or AMDOptimizationConfig()

        # Couches principales
        self.self_attn = AMDOptimizedAttention(hidden_size, num_heads, config=config)
        self.mlp = AMDOptimizedMLP(hidden_size, intermediate_size, config=config)

        # Normalisation (RMSNorm pour efficacité)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        residual = hidden_states

        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )

        # Connexion résiduelle
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Connexion résiduelle
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class RMSNorm(nn.Module):
    """RMS Normalization optimisée pour RDNA3"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class HRMSystemOne(nn.Module):
    """System 1 - Raisonnement rapide et intuitif (L-level dans HRM original)"""

    def __init__(self, config: AMDOptimizationConfig):
        super().__init__()

        self.config = config
        self.hidden_size = 512
        self.num_heads = 8
        self.intermediate_size = 2048
        self.num_layers = 6

        # Couches de raisonnement rapide
        self.layers = nn.ModuleList([
            HRMReasoningBlock(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size,
                config
            ) for _ in range(self.num_layers)
        ])

        self.norm = RMSNorm(self.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:

        past_key_values = [] if use_cache else None

        # Passage à travers les couches
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                past_key_values.append(layer_outputs[1])

        # Normalisation finale
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, past_key_values
        return hidden_states

class HRMSystemTwo(nn.Module):
    """System 2 - Raisonnement délibéré et analytique (H-level dans HRM original)"""

    def __init__(self, config: AMDOptimizationConfig):
        super().__init__()

        self.config = config
        self.hidden_size = 512
        self.num_heads = 8
        self.intermediate_size = 2048
        self.num_layers = 12  # Plus de couches pour raisonnement profond

        # Couches de raisonnement délibéré
        self.layers = nn.ModuleList([
            HRMReasoningBlock(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size,
                config
            ) for _ in range(self.num_layers)
        ])

        self.norm = RMSNorm(self.hidden_size)

        # Mécanisme de halting adaptatif
        self.halting_linear = nn.Linear(self.hidden_size, 1)
        self.threshold = 0.5

    def forward(self,
                hidden_states: torch.Tensor,
                system_one_output: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_steps: int = 8) -> Tuple[torch.Tensor, int]:

        # Intégration avec System 1
        hidden_states = hidden_states + system_one_output

        halting_probs = []
        steps_taken = 0

        for step in range(max_steps):
            # Calcul de la probabilité de halting
            halting_prob = torch.sigmoid(self.halting_linear(hidden_states.mean(dim=1)))
            halting_probs.append(halting_prob)

            # Décision de halting
            if halting_prob.mean() > self.threshold and step > 0:
                break

            # Passage à travers une couche
            layer_idx = step % self.num_layers
            layer_outputs = self.layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]
            steps_taken += 1

        # Normalisation finale
        hidden_states = self.norm(hidden_states)

        return hidden_states, steps_taken

class HRMAMDModel(nn.Module):
    """Modèle HRM complet optimisé pour AMD 7800XT"""

    def __init__(self, config: Optional[AMDOptimizationConfig] = None):
        super().__init__()

        self.config = config or AMDOptimizationConfig()
        self.device_manager = AMDDeviceManager(self.config)

        # Paramètres du modèle
        self.vocab_size = 32000
        self.hidden_size = 512
        self.max_position_embeddings = 4096

        # Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Systèmes de raisonnement
        self.system_one = HRMSystemOne(self.config)
        self.system_two = HRMSystemTwo(self.config)

        # Têtes de sortie
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Initialisation des poids
        self.apply(self._init_weights)

        logger.info(f"HRM AMD Model initialisé avec {self.count_parameters():,} paramètres")

    def _init_weights(self, module):
        """Initialisation des poids optimisée pour AMD"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Compte le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                max_reasoning_steps: int = 8) -> Dict[str, torch.Tensor]:

        # Embeddings d'entrée
        hidden_states = self.embed_tokens(input_ids)

        # System 1: Raisonnement rapide
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            system_one_output = self.system_one(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache
            )

            if use_cache:
                system_one_output, past_key_values = system_one_output

            # System 2: Raisonnement délibéré
            final_output, reasoning_steps = self.system_two(
                hidden_states,
                system_one_output,
                attention_mask=attention_mask,
                max_steps=max_reasoning_steps
            )

        # Prédiction finale
        logits = self.lm_head(final_output)

        outputs = {
            "logits": logits,
            "reasoning_steps": reasoning_steps,
            "system_one_output": system_one_output,
            "hidden_states": final_output
        }

        # Calcul de la loss si labels fournis
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            outputs["loss"] = loss

        if use_cache:
            outputs["past_key_values"] = past_key_values

        return outputs

    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 do_sample: bool = True) -> torch.Tensor:
        """Génération de texte optimisée pour AMD"""

        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        generated = input_ids.clone()
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    use_cache=True
                )

                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs.get("past_key_values")

                # Sampling
                if do_sample:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)

                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            probs[i][indices_to_remove] = 0

                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)

                # Stop si EOS
                if next_token.item() == 2:  # Assuming EOS token is 2
                    break

        return generated

    def to_device(self) -> 'HRMAMDModel':
        """Déplace le modèle vers le device AMD optimal"""
        return self.to(self.device_manager.device)

    def optimize_for_inference(self) -> 'HRMAMDModel':
        """Optimise le modèle pour l'inférence sur AMD"""
        self.eval()

        # Optimisations spécifiques AMD
        if self.config.use_mixed_precision:
            self.half()  # Conversion vers FP16

        # Fusion des opérations si possible
        try:
            # torch.jit.script pour optimisation
            self = torch.jit.script(self)
            logger.info("Modèle optimisé avec TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")

        return self

    def get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire du modèle"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())

        return {
            "parameters_mb": param_size / 1024 / 1024,
            "buffers_mb": buffer_size / 1024 / 1024,
            "total_mb": (param_size + buffer_size) / 1024 / 1024,
            "estimated_inference_mb": (param_size + buffer_size) * 1.5 / 1024 / 1024  # Estimation avec activations
        }