"""
Trainer callback for combined Sparse + QAT training.

Handles:
- Delayed fake quantization toggle (using TorchAO's FakeQuantizedLinear)
- Post-optimizer sparsity enforcement using stored masks

Note on FSDP/Zero Compatibility:
- Gradient masking is handled by hooks registered in utils.py
- This callback handles post-step re-zeroing to catch any drift
  (e.g. from weight decay if not properly masked)
"""

from functools import partial
from typing import Any

import torch
from torch import nn
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from transformers import TrainerCallback
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def toggle_fake_quant(mod: nn.Module, enable: bool) -> None:
    """Toggle fake quantization for TorchAO fake-quantized layers."""
    if isinstance(mod, (FakeQuantizedLinear, FakeQuantizedEmbedding)):
        if mod.weight_fake_quantizer is not None:
            # Handle both simple enablement and potentially more complex state
            mod.weight_fake_quantizer.enabled = enable
        
        # FakeQuantizedLinear has activation_fake_quantizer
        if (
            isinstance(mod, FakeQuantizedLinear)
            and hasattr(mod, "activation_fake_quantizer")
            and mod.activation_fake_quantizer is not None
        ):
            mod.activation_fake_quantizer.enabled = enable


class SparseQATCallback(TrainerCallback):
    """
    Trainer callback for combined Sparse + QAT training.

    Optimization:
    - We cache list of (param, mask) tuples to avoid iterating named_parameters
      and doing string lookups every step.
    - We ensure masks are on the correct device.
    """

    def __init__(
        self,
        cfg: Any,
        sparsity_masks: dict[str, torch.Tensor] | None = None,
        trainer: Trainer | None = None,
    ):
        self.cfg = cfg
        self.sparsity_masks = sparsity_masks or {}
        self.trainer = trainer
        self.fake_quant_enabled = (
            cfg.sparse_qat.fake_quant_after_n_steps is None
            or cfg.sparse_qat.fake_quant_after_n_steps == 0
        )
        self._logged_enable = False
        self._logged_disable = False
        
        # Cache for performance: list of (param, mask)
        self._param_mask_cache: list[tuple[nn.Parameter, torch.Tensor]] | None = None

    def _build_param_mask_cache(self, model: nn.Module) -> None:
        """
        Build cache of (parameter, mask) pairs on correct device.
        This runs once to avoid overhead in the training loop.
        """
        if self._param_mask_cache is not None:
            return

        cache = []
        for name, param in model.named_parameters():
            if name in self.sparsity_masks:
                # Ensure mask is on same device as param
                mask = self.sparsity_masks[name].to(param.device)
                cache.append((param, mask))
        
        self._param_mask_cache = cache
        LOG.info(f"Built sparsity enforcement cache for {len(cache)} parameters")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        """Handles delayed fake quantization toggle."""
        # Initialize cache on first step if needed
        if self._param_mask_cache is None:
            self._build_param_mask_cache(model)

        if self.cfg.sparse_qat.fake_quant_after_n_steps is not None:
            if state.global_step == 0 and not self.fake_quant_enabled:
                if not self._logged_disable:
                    LOG.info(
                        f"Disabling fake quantization for first "
                        f"{self.cfg.sparse_qat.fake_quant_after_n_steps} steps"
                    )
                    self._logged_disable = True
                model.apply(partial(toggle_fake_quant, enable=False))

            elif state.global_step == self.cfg.sparse_qat.fake_quant_after_n_steps:
                if not self._logged_enable:
                    LOG.info(
                        f"Enabling fake quantization at step {state.global_step}"
                    )
                    self._logged_enable = True
                model.apply(partial(toggle_fake_quant, enable=True))
                self.fake_quant_enabled = True

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        """
        Enforce sparsity using cached (param, mask) pairs.
        
        This loop is now highly optimized: no string matching, no lookups,
        no device transfers.
        """
        if self._param_mask_cache:
            for param, mask in self._param_mask_cache:
                with torch.no_grad():
                    param.data *= mask
