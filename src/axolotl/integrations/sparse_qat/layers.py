"""
Layer utilities for Sparse + QAT training.

Note: With the TorchAO-native approach, we use TorchAO's built-in
FakeQuantizedLinear and FakeQuantizedEmbedding layers instead of
custom implementations.

This module provides helper functions for working with these layers.
"""

from torch import nn
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding


def is_fake_quantized_layer(module: nn.Module) -> bool:
    """Check if a module is a TorchAO fake-quantized layer."""
    return isinstance(module, (FakeQuantizedLinear, FakeQuantizedEmbedding))


def count_fake_quantized_layers(model: nn.Module) -> int:
    """Count the number of fake-quantized layers in a model."""
    count = 0
    for module in model.modules():
        if is_fake_quantized_layer(module):
            count += 1
    return count


def get_layer_sparsity(module: nn.Module) -> float | None:
    """
    Get the sparsity of a layer's weights.

    Returns:
        Sparsity as a float (0.0 to 1.0), or None if not applicable.
    """
    if hasattr(module, "weight") and module.weight is not None:
        weight = module.weight.data
        sparsity = (weight == 0).float().mean().item()
        return sparsity
    return None
