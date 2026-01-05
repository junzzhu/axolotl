"""
Sparse + QAT Combined Training Integration.

This integration enables training models that are both structurally sparse
(maintaining pre-existing zero weight patterns) and quantization-ready
(trained with fake quantization for deployment efficiency).
"""

from axolotl.integrations.sparse_qat.plugin import SparseQATPlugin

__all__ = ["SparseQATPlugin"]
