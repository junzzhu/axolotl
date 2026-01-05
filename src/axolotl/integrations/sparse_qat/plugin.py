"""
Sparse + QAT Plugin for Axolotl.

This plugin enables combined sparse fine-tuning and quantization-aware training.
It uses TorchAO's native QAT implementation and adds FSDP-safe sparsity enforcement.
"""

from typing import Any

import torch
from transformers.trainer import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.sparse_qat.callback import SparseQATCallback
from axolotl.integrations.sparse_qat.utils import prepare_model_for_sparse_qat
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class SparseQATPlugin(BasePlugin):
    """
    Plugin for combined Sparse + QAT training.

    This plugin:
    1. Prepares model with TorchAO QAT layers & registers gradient hooks (post_model_load)
    2. Stores sparsity masks for callback
    3. Adds optimized callback for sparsity enforcement
    """

    def __init__(self):
        super().__init__()
        self.sparsity_masks: dict[str, torch.Tensor] = {}

    def get_input_args(self) -> str:
        return "axolotl.integrations.sparse_qat.args.SparseQATArgs"

    def post_model_load(self, cfg: Any, model: Any) -> Any:
        """
        Prepare model for sparse QAT.
        
        Applies TorchAO QAT replacement and registers critical gradient hooks
        in utils.prepare_model_for_sparse_qat.
        """
        if not hasattr(cfg, "sparse_qat") or cfg.sparse_qat is None:
            return model

        LOG.info("Preparing model for Sparse + QAT training")

        # Prepare model and extract sparsity masks
        # Note: Gradient hooks are registered inside this function
        prepared_model, self.sparsity_masks = prepare_model_for_sparse_qat(
            model=model,
            target_patterns=cfg.sparse_qat.sparsity_targets,
            weight_dtype=cfg.sparse_qat.weight_dtype,
            activation_dtype=cfg.sparse_qat.activation_dtype,
            group_size=cfg.sparse_qat.group_size,
            quantize_embedding=cfg.sparse_qat.quantize_embedding,
        )

        return prepared_model

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Trainer) -> list:
        """
        Add Sparse QAT callback.
        
        Callback handles delayed QAT toggle and double-checks sparsity
        (though gradient hooks defined in post_model_load do the heavy lifting).
        """
        if not hasattr(cfg, "sparse_qat") or cfg.sparse_qat is None:
            return []

        LOG.info("Adding Sparse QAT callback to trainer")
        callback = SparseQATCallback(
            cfg=cfg,
            sparsity_masks=self.sparsity_masks,
            trainer=trainer,
        )
        return [callback]
