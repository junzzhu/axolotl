"""
Pydantic schema for Sparse + QAT configuration.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import validate_ao_dtype


class SparseQATArgs(BaseModel):
    """
    Configuration arguments for combined Sparse + QAT training.

    This schema is used by the SparseQATPlugin to configure both
    sparsity maintenance and fake quantization during training.
    """

    # Sparsity configuration - targets to keep sparse
    sparsity_targets: list[str] = Field(
        default_factory=lambda: [
            "re:.*q_proj.weight",
            "re:.*k_proj.weight",
            "re:.*v_proj.weight",
            "re:.*o_proj.weight",
            "re:.*gate_proj.weight",
            "re:.*up_proj.weight",
            "re:.*down_proj.weight",
        ],
        description=(
            "List of regex patterns matching layers to keep sparse. "
            "Use 're:' prefix for regex patterns."
        ),
    )

    # QAT configuration
    weight_dtype: TorchAOQuantDType = Field(
        default=TorchAOQuantDType.int4,
        description="Target dtype for weight quantization (int4, int8, float8, etc.).",
    )

    activation_dtype: TorchAOQuantDType | None = Field(
        default=None,
        description="Target dtype for activation quantization (optional).",
    )

    group_size: int = Field(
        default=32,
        description="Group size for per-group weight quantization.",
    )

    quantize_embedding: bool = Field(
        default=False,
        description="Whether to also apply sparse QAT to embedding layers.",
    )

    # Training configuration
    fake_quant_after_n_steps: int | None = Field(
        default=None,
        description=(
            "Number of training steps before enabling fake quantization. "
            "If None, fake quantization is enabled from the start."
        ),
    )

    save_compressed: bool = Field(
        default=True,
        description="Whether to save the model in compressed sparse format.",
    )

    clear_optimizer_state: bool = Field(
        default=False,
        description=(
            "Whether to explicitly clear optimizer state for pruned weights. "
            "Usually not needed as gradients are masked, but can prevent drift."
        ),
    )

    @field_validator("activation_dtype", "weight_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: Any) -> TorchAOQuantDType | None:
        return validate_ao_dtype(v)
