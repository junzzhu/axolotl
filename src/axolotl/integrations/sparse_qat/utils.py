"""
Utility functions for Sparse + QAT training.

This module provides a hybrid approach:
1. Use TorchAO's native QAT (quantize_ with QATConfig) for fake quantization
2. Add sparsity mask extraction and enforcement on top

Key Features:
- Gradient masking via hooks (FSDP safe)
- Parameter caching for performance
- TorchAO integration
"""

import re
from typing import Any

import torch
from torch import nn
from torchao.quantization.qat import QATConfig
from torchao.quantization import quantize_
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding

from axolotl.utils.logging import get_logger
from axolotl.utils.quantization import get_quantization_config
from axolotl.utils.schemas.enums import TorchAOQuantDType

LOG = get_logger(__name__)


def extract_sparsity_masks(
    model: nn.Module,
    target_patterns: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract sparsity masks from a pre-sparsified model.

    Args:
        model: The model to extract masks from.
        target_patterns: List of regex patterns matching layer names.

    Returns:
        Dict mapping layer names to boolean mask tensors (cpu).
    """
    masks = {}

    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue

        # Check if this parameter matches any target pattern
        for pattern in target_patterns:
            if _match_pattern(name, pattern):
                # Create mask: True where weight is non-zero
                # Use a small epsilon for floating point comparison safety
                mask = param.data.abs() > 1e-6
                masks[name] = mask.cpu()  # Keep on CPU until needed to save VRAM

                # Log sparsity stats
                sparsity = 1.0 - mask.float().mean().item()
                LOG.debug(f"Extracted mask for {name}: sparsity={sparsity:.2%}")
                break

    LOG.info(f"Extracted {len(masks)} sparsity masks from model")
    return masks


def _match_pattern(name: str, pattern: str) -> bool:
    """Match a parameter name against a pattern."""
    if pattern.startswith("re:"):
        regex = pattern[3:]
        return re.match(regex, name) is not None
    return name == pattern


def prepare_model_for_sparse_qat(
    model: nn.Module,
    target_patterns: list[str],
    weight_dtype: TorchAOQuantDType,
    activation_dtype: TorchAOQuantDType | None = None,
    group_size: int = 32,
    quantize_embedding: bool = False,
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """
    Prepare a model for combined sparse + QAT training.
    """
    LOG.info("Preparing model for Sparse + QAT training")

    # Step 1: Extract sparsity masks
    sparsity_masks = extract_sparsity_masks(model, target_patterns)

    if not sparsity_masks:
        LOG.warning("No sparsity masks extracted.")

    # Step 2: Apply TorchAO QAT
    base_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )
    qat_config = QATConfig(base_config)

    LOG.info(f"Applying TorchAO QAT with config: {base_config}")
    quantize_(model, qat_config)

    if quantize_embedding:
        embedding_base_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,
            group_size=group_size,
        )
        embedding_qat_config = QATConfig(embedding_base_config)
        quantize_(
            model,
            embedding_qat_config,
            filter_fn=lambda m, _: isinstance(m, nn.Embedding),
        )

    # Step 3: Register gradient masking hooks
    # This is critical for FSDP/DeepSpeed compatibility:
    # We mask gradients BEFORE they reach the optimizer.
    _register_gradient_hooks(model, sparsity_masks)

    # Step 4: Re-apply masks to weights (ensure starting valid state)
    _apply_sparsity_masks_to_model(model, sparsity_masks)

    return model, sparsity_masks


def _register_gradient_hooks(
    model: nn.Module,
    sparsity_masks: dict[str, torch.Tensor],
) -> None:
    """
    Register backward hooks to mask gradients.
    """
    hook_count = 0
    for name, param in model.named_parameters():
        if name in sparsity_masks:
            # We must move mask to the same device as param for the hook
            mask = sparsity_masks[name].to(param.device)

            def hook_fn(grad, mask=mask):
                if grad is None:
                    return None
                return grad * mask

            param.register_hook(hook_fn)
            hook_count += 1

    LOG.info(f"Registered gradient masking hooks for {hook_count} parameters")


def _apply_sparsity_masks_to_model(
    model: nn.Module,
    sparsity_masks: dict[str, torch.Tensor],
) -> None:
    """Apply sparsity masks to model weights."""
    for name, param in model.named_parameters():
        if name in sparsity_masks:
            mask = sparsity_masks[name].to(param.device)
            with torch.no_grad():
                param.data *= mask
            LOG.debug(f"Applied sparsity mask to {name}")


def enforce_sparsity_on_model(
    model: nn.Module,
    sparsity_masks: dict[str, torch.Tensor],
) -> None:
    """
    Enforce sparsity masks on model weights.
    """
    for name, param in model.named_parameters():
        if name in sparsity_masks:
            mask = sparsity_masks[name].to(param.device)
            with torch.no_grad():
                param.data *= mask
